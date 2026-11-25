#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''The CPHF response by chunks of dms and H1 in a temporary file: the same numbers'''

import unittest
import weakref
from unittest import mock
import numpy
from pyscf import gto, dft, lib
from pyscf.dft import numint
from pyscf.scf import _response_functions
from pyscf.hessian import rhf as rhf_hess
from pyscf.hessian import rks as rks_hess


def setUpModule():
    global mol, mf
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='wb97m-v')
    mf.nlcgrids.level = 1
    mf.conv_tol = 1e-11
    mf.kernel()

def tearDownModule():
    global mol, mf
    del mol, mf

def _response(mf):
    nocc = (mf.mo_occ > 0).sum()
    rng = numpy.random.default_rng(5)
    x = rng.random((7, mol.nao, nocc)) - .5
    return rhf_hess.gen_vind(mf, mf.mo_coeff, mf.mo_occ)(x)

class KnownValues(unittest.TestCase):
    def test_vind_mo_exists(self):
        vresp = mf.gen_response(mf.mo_coeff, mf.mo_occ, hermi=1)
        self.assertTrue(hasattr(vresp, 'vind_mo'))

    def test_chunks(self):
        # J + K + long-range K + VV10; J + K; J only; J + long-range K alone; J + short-range K alone
        # and VV10 without any K
        for xc in ('wb97m-v', 'b3lyp', 'pbe', 'lrc-wpbe', 'hse06', 'b97m-v'):
            mf1 = dft.RKS(mol, xc=xc)
            mf1.conv_tol = 1e-11
            if xc in ('wb97m-v', 'b97m-v'):
                mf1.nlcgrids.level = 1
            mf1.kernel()
            ref = _response(mf1)
            for cheaper in (False, True):
                with mock.patch.object(numint._FusedXC, 'lowrank_cheaper', lambda *a: cheaper):
                    whole = _response(mf1)
                    with mock.patch.object(_response_functions, '_cphf_chunk', lambda *a: 2):
                        chunks = _response(mf1)
                self.assertLess(abs(chunks - whole).max(), 1e-10, (xc, cheaper))
                self.assertLess(abs(whole - ref).max(), 1e-10, (xc, cheaper))

    def test_vnlc_from_factors(self):
        # the VV10 response from the factors of the dms, with groups and batches down to one dm,
        # against the MO block of the AO matrices
        mocc = mf.mo_coeff[:,mf.mo_occ>0]
        x = numpy.random.default_rng(11).random((9, mf.mo_coeff.shape[1], mocc.shape[1])) - .5
        a = lib.einsum('pa,kai->kpi', mf.mo_coeff, x)
        dm = numpy.matmul(a, mocc.T)
        dm = dm + dm.transpose(0,2,1)
        ref = rks_hess.get_vnlc_resp(mf, mol, mf.mo_coeff, mf.mo_occ, dm, 2000)
        ref = numpy.matmul(mf.mo_coeff.T, numpy.matmul(ref, mocc))
        self.assertGreater(abs(ref).max(), 1e-4)
        for max_memory in (2000, 1):
            out = rks_hess.get_vnlc_resp_mo(mf, mol, mf.mo_coeff, mf.mo_occ, a, max_memory,
                                            numpy.zeros_like(ref))
            self.assertLess(abs(out - ref).max(), 1e-11 * abs(ref).max(), max_memory)

    def test_vind_mo_vnlc_path(self):
        # vind_mo takes VV10 from the factors, and through the chunks' AO matrices where that path
        # does not apply
        with mock.patch.object(rks_hess, 'get_vnlc_resp_mo', wraps=rks_hess.get_vnlc_resp_mo) as m:
            v = _response(mf)
        self.assertEqual(m.call_count, 1)
        with mock.patch.object(rks_hess, 'get_vnlc_resp_mo', lambda *a: None):
            self.assertLess(abs(_response(mf) - v).max(), 1e-10)

    def test_second_nlcgrids(self):
        # the VV10 part of the response on mf.second_nlcgrids: the factored path and the AO path take
        # that grid, the same grid gives the same response, and a coarser one a slightly different one
        ref = _response(mf)
        try:
            grids = dft.gen_grid.Grids(mol)
            grids.level = 1
            mf.set_second_nlcgrids(grids)
            self.assertIs(mf.second_nlcgrids, grids)
            self.assertIs(rks_hess._vv10_response_grids(mf), grids)
            self.assertAlmostEqual(abs(_response(mf) - ref).max(), 0, 11)
            self.assertIsNotNone(getattr(grids, '_vv10_hess_data', None))
            mf.set_second_nlcgrids(0)
            self.assertEqual(mf.second_nlcgrids.level, 0)
            v0 = _response(mf)
            d0 = abs(v0 - ref).max()
            self.assertLess(d0, 2e-4)
            self.assertGreater(d0, 1e-9)
            # the path through the AO matrices of the dms takes the same grid
            with mock.patch.object(rks_hess, 'get_vnlc_resp_mo', return_value=None):
                self.assertAlmostEqual(abs(_response(mf) - v0).max(), 0, 10)
            # the response without VV10 differs by several times what the coarse grid changes
            nocc = (mf.mo_occ > 0).sum()
            x = numpy.random.default_rng(5).random((7, mol.nao, nocc)) - .5
            vresp = mf.gen_response(mf.mo_coeff, mf.mo_occ, hermi=1, with_nlc=False)
            self.assertGreater(abs(vresp.vind_mo(None, x*2) - ref).max(), 4 * d0)
            self.assertRaises(ValueError, mf.set_second_nlcgrids, 'sg2')
        finally:
            mf.second_nlcgrids = None
        self.assertIs(rks_hess._vv10_response_grids(mf), mf.nlcgrids)
        # the Hessian's own VV10 terms keep mf.nlcgrids
        mf1 = dft.RKS(mol, xc='wb97m-v')
        mf1.nlcgrids.level = 1
        mf1.conv_tol = 1e-11
        mf1.kernel()
        hobj = mf1.Hessian()
        hobj.cphf_residual_tol = 1e-6
        h_ref = hobj.kernel()
        mf1.set_second_nlcgrids(0)
        hobj = mf1.Hessian()
        hobj.cphf_residual_tol = 1e-6
        d = abs(hobj.kernel() - h_ref).max()
        self.assertLess(d, 5e-5)
        self.assertGreater(d, 1e-10)

    def test_h1_split(self):
        # the XC and VV10 kernel potentials of make_h1 as products with the occupied orbitals
        # (rks._make_h1_split): the same H1 mocc, and the same Hessian
        mocc = mf.mo_coeff[:,mf.mo_occ>0]
        for xc in ('lda,vwn', 'pbe', 'r2scan', 'wb97m-v'):
            mf1 = dft.RKS(mol, xc=xc)
            mf1.nlcgrids.level = 0
            mf1.conv_tol = 1e-11
            mf1.kernel()
            mocc = mf1.mo_coeff[:,mf1.mo_occ>0]
            hobj = mf1.Hessian()
            h1ao = hobj.make_h1(mf1.mo_coeff, mf1.mo_occ)
            h1ao_s, h1occ = rks_hess._make_h1_split(hobj, mf1.mo_coeff, mf1.mo_occ)
            self.assertEqual(h1occ.shape, (mol.natm, 3, mol.nao, mocc.shape[1]))
            self.assertGreater(abs(h1occ).max(), 1e-3)
            ref = lib.einsum('axpq,qi->axpi', numpy.asarray(h1ao), mocc)
            new = lib.einsum('axpq,qi->axpi', numpy.asarray(h1ao_s), mocc) + h1occ
            self.assertAlmostEqual(abs(new - ref).max(), 0, 11)
            with lib.temporary_env(rks_hess, H1_OCC=False):
                hobj = mf1.Hessian()
                hobj.cphf_residual_tol = 1e-7
                h_ref = hobj.kernel()
                self.assertIsNone(rks_hess._make_h1_split(hobj, mf1.mo_coeff, mf1.mo_occ)[1])
            hobj = mf1.Hessian()
            hobj.cphf_residual_tol = 1e-7
            self.assertAlmostEqual(abs(hobj.kernel() - h_ref).max(), 0, 10)
        # hess_elec takes the split form of the RKS make_h1 and of the solvent's, not of others
        self.assertIs(rhf_hess._MAKE_H1_SPLIT[rks_hess.make_h1], rks_hess._make_h1_split)
        mfs = dft.RKS(mol, xc='pbe').PCM()
        mfs.conv_tol = 1e-11
        mfs.kernel()
        hs = mfs.Hessian()
        self.assertIn(hs.make_h1.__func__, rhf_hess._MAKE_H1_SPLIT)
        with lib.temporary_env(rks_hess, H1_OCC=False):
            h_ref = mfs.Hessian().kernel()
        self.assertAlmostEqual(abs(hs.kernel() - h_ref).max(), 0, 9)
        self.assertNotIn(dft.RKS(mol, xc='pbe').density_fit().Hessian().make_h1.__func__,
                         rhf_hess._MAKE_H1_SPLIT)

    def test_chunk_size(self):
        chunk = _response_functions._cphf_chunk
        with mock.patch.object(_response_functions._vhf, '_direct_jk_batch',
                               lambda n_dm, nao, with_j, with_k, sym_k, rss_added=0: 82 if not with_j else 41):
            # no room under max_memory: one batch of the direct J/K call
            mf1 = mock.Mock(max_memory=1, with_df=None)
            self.assertEqual(chunk(mf1, 186, 872, True, True), 41)
            self.assertEqual(chunk(mf1, 186, 872, False, True), 82)
            # all fit
            mf1 = mock.Mock(max_memory=1e6, with_df=None)
            self.assertEqual(chunk(mf1, 186, 872, True, True), 186)
        with mock.patch.object(_response_functions._vhf, '_direct_jk_batch', lambda *a: 5):
            # the many-dm kernels of the direct J/K need 13 dms or more
            mf1 = mock.Mock(max_memory=1, with_df=None)
            self.assertEqual(chunk(mf1, 186, 2000, True, True), 16)
            self.assertEqual(chunk(mf1, 7, 2000, True, True), 7)
            # no such passes in density-fitted and SGX J/K
            mf1 = mock.Mock(max_memory=1, with_df=object())
            self.assertEqual(chunk(mf1, 186, 2000, True, True), 186)

    def test_chunk_is_one_pass(self):
        # the direct J/K sizes its batch with the chunk's dms in RSS, at n-C20H42's size 41 where
        # the chunk was sized as 46 before them: no second pass for the last 5 dms of each chunk
        chunk = _response_functions._cphf_chunk
        batch = _response_functions._vhf._direct_jk_batch
        nao = 872
        mf1 = mock.Mock(max_memory=4000, with_df=None)
        with mock.patch.object(lib.param, 'MAX_MEMORY', 4000):
            for rss in range(100, 4300, 40):
                for with_j, with_k in ((True, True), (True, False), (False, True)):
                    with mock.patch.object(lib, 'current_memory', lambda: (rss, 0)):
                        n = chunk(mf1, 186, nao, with_j, with_k)
                    rss1 = rss + n * 8 * nao**2 / 1e6
                    with mock.patch.object(lib, 'current_memory', lambda: (rss1, 0)):
                        self.assertGreaterEqual(batch(n, nao, with_j, with_k, True), n, (rss, with_j, with_k))

    def test_chunk_freed(self):
        # the J/K of a chunk is gone when the next chunk's is made
        mf1 = dft.RKS(mol, xc='pbe')
        mf1.conv_tol = 1e-9
        mf1.kernel()
        refs = []
        get_j = mf1.get_j
        def spy(*args, **kwargs):
            self.assertTrue(all(r() is None for r in refs))
            v = get_j(*args, **kwargs)
            # the array that owns the memory: v may be a view of it
            refs.append(weakref.ref(v if v.base is None else v.base))
            return v
        with mock.patch.object(mf1, 'get_j', spy), mock.patch.object(_response_functions, '_cphf_chunk', lambda *a: 2):
            _response(mf1)
        self.assertGreater(len(refs), 2)

    def test_h1ao_file(self):
        h1ao = [None, numpy.arange(12.).reshape(3,2,2)]
        f = rhf_hess._H1aoFile(h1ao)
        self.assertEqual(abs(f[1] - h1ao[1]).max(), 0)
        log = mock.Mock()
        self.assertIs(rhf_hess._h1ao_out_of_core(h1ao, 1e9, log), h1ao)
        self.assertIsInstance(rhf_hess._h1ao_out_of_core(h1ao, 1, log), rhf_hess._H1aoFile)

    def test_hessian_out_of_core(self):
        ref = mf.Hessian().kernel()
        mf1 = mf.copy()
        mf1.max_memory = 1
        with mock.patch.object(_response_functions, '_cphf_chunk', lambda *a: 2):
            h = mf1.Hessian().kernel()
        self.assertLess(abs(h - ref).max(), 1e-8)


    def test_vnlc_deriv1_kept_kernel_terms(self):
        # the kernel terms of the density responses that the VV10 energy's second derivative hands to
        # the VV10 Fock matrix derivative with the grid response (_VV10HessData._f_t_dA), against the
        # derivative that makes its own
        mf1 = dft.RKS(mol, xc='wb97m-v')
        mf1.nlcgrids.level = 0
        mf1.conv_tol = 1e-11
        mf1.kernel()
        hobj = mf1.Hessian()
        vv = rks_hess._vv10_hess_data(mf1, mol, mf1.make_rdm1(), 2000)
        ref = rks_hess._get_vnlc_deriv1(hobj, mf1.mo_coeff, mf1.mo_occ, 2000)
        # without the grid response the energy term's density responses are not those of the derivative
        rks_hess._get_enlc_deriv2(hobj, mf1.mo_coeff, mf1.mo_occ, 2000)
        self.assertFalse(hasattr(vv, '_f_t_dA'))
        hobj.grid_response = True
        rks_hess._get_enlc_deriv2(hobj, mf1.mo_coeff, mf1.mo_occ, 2000)
        self.assertIs(rks_hess._vv10_hess_data(mf1, mol, mf1.make_rdm1(), 2000), vv)
        self.assertEqual(vv._f_t_dA[0].shape, (mol.natm*3, vv.ngrids))
        with mock.patch.object(rks_hess._VV10HessData, 'f_t', side_effect=AssertionError):
            v = rks_hess._get_vnlc_deriv1(hobj, mf1.mo_coeff, mf1.mo_occ, 2000)
        self.assertFalse(hasattr(vv, '_f_t_dA'))
        self.assertAlmostEqual(abs(v - ref).max(), 0, 11)

class UKSKnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                        basis='def2-svp', charge=1, spin=1, verbose=0)

    def _mf(self, xc):
        mf1 = dft.UKS(self.mol, xc=xc)
        mf1.nlcgrids.level = 0
        mf1.conv_tol = 1e-11
        return mf1.run()

    def test_h1_split(self):
        # as test_h1_split for the orbitals of each spin (uks._make_h1_split), with and without the
        # grid response: the same H1 mocc, and the same Hessian
        from pyscf.hessian import uks as uks_hess
        mol = self.mol
        for xc, grid_response in (('lda,vwn', True), ('pbe', False), ('r2scan', True),
                                  ('wb97m-v', True), ('wb97m-v', False)):
            mf1 = self._mf(xc)
            mocc = [mf1.mo_coeff[s][:,mf1.mo_occ[s]>0] for s in range(2)]
            hobj = mf1.Hessian()
            hobj.grid_response = grid_response
            h1ao = hobj.make_h1(mf1.mo_coeff, mf1.mo_occ)
            h1ao_s, h1occ = uks_hess._make_h1_split(hobj, mf1.mo_coeff, mf1.mo_occ)
            for s in range(2):
                self.assertEqual(h1occ[s].shape, (mol.natm, 3, mol.nao, mocc[s].shape[1]))
                self.assertGreater(abs(h1occ[s]).max(), 1e-3)
                ref = lib.einsum('axpq,qi->axpi', numpy.asarray(h1ao[s]), mocc[s])
                new = lib.einsum('axpq,qi->axpi', numpy.asarray(h1ao_s[s]), mocc[s]) + h1occ[s]
                self.assertAlmostEqual(abs(new - ref).max(), 0, 10)
            with lib.temporary_env(rks_hess, H1_OCC=False):
                hobj = mf1.Hessian()
                hobj.grid_response = grid_response
                hobj.cphf_residual_tol = 1e-7
                h_ref = hobj.kernel()
                self.assertIsNone(uks_hess._make_h1_split(hobj, mf1.mo_coeff, mf1.mo_occ)[1])
            hobj = mf1.Hessian()
            hobj.grid_response = grid_response
            hobj.cphf_residual_tol = 1e-7
            self.assertAlmostEqual(abs(hobj.kernel() - h_ref).max(), 0, 9)
        self.assertIs(rhf_hess._MAKE_H1_SPLIT[uks_hess.make_h1], uks_hess._make_h1_split)

    def test_vind_mo_chunks(self):
        # the UKS response in the MO basis from the factors of the dms and by chunks of their AO
        # matrices, without the dms (dm1=None), against the AO matrices of all of them
        rng = numpy.random.default_rng(3)
        for xc in ('pbe', 'b3lyp', 'wb97m-v'):
            mf1 = self._mf(xc)
            mo, occ = mf1.mo_coeff, mf1.mo_occ
            mocc = [mo[s][:,occ[s]>0] for s in range(2)]
            x = [rng.random((9, mo[s].shape[1], mocc[s].shape[1])) - .5 for s in range(2)]
            dm1 = numpy.array([[mo[s] @ xk @ mocc[s].T for xk in x[s]] for s in range(2)])
            dm1 = dm1 + dm1.transpose(0,1,3,2)
            for mfr in (mf1, rhf_hess._sgx_response_scf(mf1, 1)):
                vresp = mfr.gen_response(mo, occ, hermi=1)
                v = vresp(dm1)
                ref = [numpy.array([mo[s].T @ vk @ mocc[s] for vk in v[s]]) for s in range(2)]
                for dm in (None, dm1):
                    new = vresp.vind_mo(dm, x)
                    for s in range(2):
                        self.assertAlmostEqual(abs(new[s] - ref[s]).max(), 0, 8)
                # a chunk at a time: no AO array of all dms
                with mock.patch.object(_response_functions, '_cphf_chunk', return_value=2), \
                        lib.temporary_env(mfr, max_memory=lib.current_memory()[0] + 1):
                    new = vresp.vind_mo(None, x)
                for s in range(2):
                    self.assertAlmostEqual(abs(new[s] - ref[s]).max(), 0, 8)

    def test_hcore_deriv2_blocks(self):
        # the second derivatives of the core Hamiltonian with the density, from the blocks of each
        # pair of atoms, against the matrices of hcore_generator
        mol = self.mol
        mf1 = self._mf('pbe')
        dm0 = mf1.make_rdm1().sum(axis=0)
        hobj = mf1.Hessian()
        natm = mol.natm
        e1 = numpy.zeros((natm, natm, 3, 3))
        self.assertTrue(rhf_hess._hcore_deriv2_dm_(e1, hobj, mol, dm0, range(natm)))
        hcore_deriv = hobj.hcore_generator(mol)
        for ia in range(natm):
            for ja in range(ia+1):
                ref = numpy.einsum('xypq,pq->xy', hcore_deriv(ia, ja), dm0)
                self.assertAlmostEqual(abs(e1[ia,ja] - ref).max(), 0, 10)
        # a subset of the atoms, in another order
        e1 = numpy.zeros((2, 2, 3, 3))
        rhf_hess._hcore_deriv2_dm_(e1, hobj, mol, dm0, [2, 0])
        self.assertAlmostEqual(abs(e1[1,0] - numpy.einsum('xypq,pq->xy', hcore_deriv(0, 2), dm0)).max(),
                               0, 10)
        # not with ECPs
        mol1 = gto.M(atom='Zn 0 0 0; H 0 0 1.6', basis='lanl2dz', ecp='lanl2dz', spin=1, verbose=0)
        mf2 = dft.UKS(mol1, xc='pbe')
        self.assertFalse(rhf_hess._hcore_deriv2_dm_(numpy.zeros((2,2,3,3)), mf2.Hessian(), mol1,
                                                    numpy.eye(mol1.nao), range(2)))

    def test_grid_motion_owner_mo(self):
        # the gradient-type matrices of the points of each atom times the orbitals, for all atoms in
        # one pass (numint._FusedXC.dot_terms_owner_mo), against dot_terms for each atom's points
        mol = self.mol
        mf1 = self._mf('pbe')
        grids = mf1.grids
        nao = mol.nao
        rng = numpy.random.default_rng(2)
        q = rng.random((nao, 4)) - .5
        for xctype, nwv in (('LDA', 1), ('GGA', 4), ('MGGA', 5)):
            fused = numint._fused_xc(mf1._numint, mol, grids, nao, xctype, 2000, (),
                                     match_block_loop=False)
            self.assertIsNotNone(fused)
            wv = rng.random((nwv, grids.weights.size)) * grids.weights
            terms = numint._xc_grad_terms(xctype)
            out = numpy.zeros((mol.natm, 3, nao, 4))
            fused.dot_terms_owner_mo(wv, terms, grids.atm_idx, mol.natm, q, out)
            for ia in range(mol.natm):
                v = numpy.zeros((1, 3, nao, nao))
                fused.dot_terms(0, grids.weights.size, ((grids.atm_idx == ia) * wv)[None], terms, v)
                ref = lib.einsum('xpq,qi->xpi', v[0] + v[0].transpose(0,2,1), q)
                self.assertAlmostEqual(abs(out[ia] - ref).max() / abs(ref).max(), 0, 12)


if __name__ == '__main__':
    print('Full Tests for the memory of the CPHF response')
    unittest.main()
