#!/usr/bin/env python
# Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import ctypes
import numpy
import unittest
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf import lib
from pyscf.scf import _vhf


def setUpModule():
    global mol, mf, nao, nmo
    mol = gto.Mole()
    mol.build(
        verbose = 5,
        output = '/dev/null',
        atom = '''
    O     0    0        0
    H     0    -0.757   0.587
    H     0    0.757    0.587''',
        basis = '631g',
    )

    mf = scf.RHF(mol)
    mf.scf()
    nao, nmo = mf.mo_coeff.shape

def tearDownModule():
    global mol, mf
    mol.stdout.close()
    del mol, mf


class KnownValues(unittest.TestCase):
    def test_incore_s4(self):
        eri4 = ao2mo.restore(4, mf._eri, nmo)
        dm = mf.make_rdm1()
        vj0, vk0 = _vhf.incore(eri4, dm, hermi=1)
        vj1, vk1 = scf.hf.get_jk(mol, dm, hermi=1)
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_incore_s8_batch(self):
        numpy.random.seed(2)
        eri1 = ao2mo.restore(1, mf._eri, nao)
        a = numpy.random.random((3,nao,nao)) - .5
        # the last case: hermi=1 for non-symmetric dms gives the lower triangle of K
        for hermi, dms in ((0, a), (1, a + a.transpose(0,2,1)),
                           (2, a - a.transpose(0,2,1)), (1, a)):
            vj0 = numpy.einsum('ijkl,nlk->nij', eri1, dms)
            vk0 = numpy.einsum('ijkl,njk->nil', eri1, dms)
            if hermi != 0:
                for v in vk0:
                    lib.hermi_triu(v, hermi, inplace=True)
            vj1, vk1 = _vhf.incore(mf._eri, dms, hermi)
            self.assertAlmostEqual(abs(vj1 - vj0).max(), 0, 12)
            self.assertAlmostEqual(abs(vk1 - vk0).max(), 0, 12)
            # a single dm
            vj3, vk3 = _vhf.incore(mf._eri, dms[1], hermi)
            self.assertAlmostEqual(abs(vj3 - vj0[1]).max(), 0, 12)
            self.assertAlmostEqual(abs(vk3 - vk0[1]).max(), 0, 12)
            vj3 = _vhf.incore(mf._eri, dms[1], hermi, with_k=False)[0]
            vk3 = _vhf.incore(mf._eri, dms[1], hermi, with_j=False)[1]
            self.assertAlmostEqual(abs(vj3 - vj0[1]).max(), 0, 12)
            self.assertAlmostEqual(abs(vk3 - vk0[1]).max(), 0, 12)
            # the per-dm kernels
            with lib.temporary_env(_vhf, INCORE_BATCH_MIN=len(dms)+1):
                vj2, vk2 = _vhf.incore(mf._eri, dms, hermi)
            self.assertAlmostEqual(abs(vj1 - vj2).max(), 0, 12)
            self.assertAlmostEqual(abs(vk1 - vk2).max(), 0, 12)
            vj1 = _vhf.incore(mf._eri, dms, hermi, with_k=False)[0]
            vk1 = _vhf.incore(mf._eri, dms, hermi, with_j=False)[1]
            self.assertAlmostEqual(abs(vj1 - vj0).max(), 0, 12)
            self.assertAlmostEqual(abs(vk1 - vk0).max(), 0, 12)
            # BLAS-3 instead of the one-pass kernel for a few symmetric dms
            with lib.temporary_env(_vhf, INCORE_FUSED_MAX=0, INCORE_FUSED_J_MAX=0):
                for with_j, with_k in ((True, True), (True, False), (False, True)):
                    vj2, vk2 = _vhf.incore(mf._eri, dms, hermi, with_j, with_k)
                    if with_j:
                        self.assertAlmostEqual(abs(vj2 - vj0).max(), 0, 12)
                    if with_k:
                        self.assertAlmostEqual(abs(vk2 - vk0).max(), 0, 12)

    def test_incore_s8_fused(self):
        # The one-pass kernel takes the dms JK_NDM_CHUNK = 4 at a time; each dm's
        # results do not depend on the number of dms
        numpy.random.seed(3)
        eri1 = ao2mo.restore(1, mf._eri, nao)
        a = numpy.random.random((6,nao,nao)) - .5
        dms = a + a.transpose(0,2,1)
        vj0 = numpy.einsum('ijkl,nlk->nij', eri1, dms)
        vk0 = numpy.einsum('ijkl,njk->nil', eri1, dms)
        vj1, vk1 = _vhf.incore(mf._eri, dms, 1)
        self.assertAlmostEqual(abs(vj1 - vj0).max(), 0, 12)
        self.assertAlmostEqual(abs(vk1 - vk0).max(), 0, 12)
        for n in (1, 2, 3, 5):
            vj2, vk2 = _vhf.incore(mf._eri, dms[:n], 1)
            self.assertTrue(numpy.array_equal(vj2, vj1[:n]))
            self.assertTrue(numpy.array_equal(vk2, vk1[:n]))
        vk2 = _vhf.incore(mf._eri, dms[:2], 1, with_j=False)[1]
        self.assertTrue(numpy.array_equal(vk2, vk1[:2]))
        vj2 = _vhf.incore(mf._eri, dms[:2], 1, with_k=False)[0]
        self.assertTrue(numpy.array_equal(vj2, vj1[:2]))

    def test_direct_s8_batch(self):
        numpy.random.seed(2)
        eri1 = ao2mo.restore(1, mf._eri, nao)
        a = numpy.random.random((5,nao,nao)) - .5
        vhfopt = scf.RHF(mol).init_direct_scf()
        for hermi, dms in ((0, a), (1, a + a.transpose(0,2,1)),
                           (2, a - a.transpose(0,2,1)), (1, a)):
            vj0 = numpy.einsum('ijkl,nlk->nij', eri1, dms)
            vk0 = numpy.einsum('ijkl,njk->nil', eri1, dms)
            if hermi != 0:
                for v in vk0:
                    lib.hermi_triu(v, hermi, inplace=True)
            vj1, vk1 = _vhf.direct(dms, mol._atm, mol._bas, mol._env, None, hermi)
            self.assertAlmostEqual(abs(vj1 - vj0).max(), 0, 12)
            self.assertAlmostEqual(abs(vk1 - vk0).max(), 0, 12)
            # a single dm
            vj1, vk1 = _vhf.direct(dms[1], mol._atm, mol._bas, mol._env, None, hermi)
            self.assertAlmostEqual(abs(vj1 - vj0[1]).max(), 0, 12)
            self.assertAlmostEqual(abs(vk1 - vk0[1]).max(), 0, 12)
            vj1 = _vhf.direct(dms, mol._atm, mol._bas, mol._env, None, hermi,
                              with_k=False)[0]
            vk1 = _vhf.direct(dms, mol._atm, mol._bas, mol._env, None, hermi,
                              with_j=False)[1]
            self.assertAlmostEqual(abs(vj1 - vj0).max(), 0, 12)
            self.assertAlmostEqual(abs(vk1 - vk0).max(), 0, 12)
            # with the screening of SCF, against the per-dm kernels
            vj1, vk1 = scf.hf.get_jk(mol, dms, hermi, vhfopt)
            with lib.temporary_env(_vhf, DIRECT_BATCH_MIN=len(dms)+1):
                vj2, vk2 = scf.hf.get_jk(mol, dms, hermi, vhfopt)
            self.assertAlmostEqual(abs(vj1 - vj2).max(), 0, 12)
            self.assertAlmostEqual(abs(vk1 - vk2).max(), 0, 12)

    def test_semi_direct(self):
        numpy.random.seed(3)
        a = numpy.random.random((3,nao,nao)) - .5
        dms = a + a.transpose(0,2,1)
        ref = scf.RHF(mol).init_direct_scf()
        full = _vhf._semi_direct_size(ref.q_cond, mol.ao_loc_nr(), ref.direct_scf_tol)
        # all quartets that pass the Schwarz test, then those with the largest bounds
        for mem in (full, full * .4):
            vhfopt = scf.RHF(mol).init_direct_scf()
            sd = _vhf.semi_direct_eri(vhfopt, mol._atm, mol._bas, mol._env, mem / 1e6)
            self.assertEqual(sd.thr == ref.direct_scf_tol, mem == full)
            self.assertTrue(sd.nbytes <= mem)
            vhfopt.semi_direct_eri = sd
            # the first call stores the integrals, the others read them
            for d, hermi in ((dms * 1e-6, 1), (dms, 1), (dms[0], 1), (a, 0)):
                vj0, vk0 = scf.hf.get_jk(mol, d, hermi, ref)
                vj1, vk1 = scf.hf.get_jk(mol, d, hermi, vhfopt)
                self.assertAlmostEqual(abs(vj1 - vj0).max(), 0, 12)
                self.assertAlmostEqual(abs(vk1 - vk0).max(), 0, 12)
                for prescreen, k in (('CVHFnrs8_vj_prescreen', 0), ('CVHFnrs8_vk_prescreen', 1)):
                    with lib.temporary_env(vhfopt, prescreen=prescreen):
                        v1 = scf.hf.get_jk(mol, d, hermi, vhfopt, k == 0, k == 1)[k]
                    self.assertAlmostEqual(abs(v1 - (vj0, vk0)[k]).max(), 0, 12)
            self.assertTrue(sd.used)
            self.assertTrue((sd.state != 0).all())
            # integrals with other parameters do not take them
            with mol.with_range_coulomb(.3):
                vj0, vk0 = scf.hf.get_jk(mol, dms, 1, ref)
                vj1, vk1 = scf.hf.get_jk(mol, dms, 1, vhfopt)
            self.assertAlmostEqual(abs(vj1 - vj0).max(), 0, 12)
            self.assertAlmostEqual(abs(vk1 - vk0).max(), 0, 12)

        # SCF without the ERIs in memory
        mf1 = scf.RHF(mol)
        mf1._is_mem_enough = lambda: False
        e1 = mf1.kernel()
        self.assertTrue(mf1._opt[None].semi_direct_eri.used)
        self.assertAlmostEqual(e1, mf.e_tot, 10)
        # switched off: the integrals are released
        mf1.semi_direct = False
        mf1.get_jk()
        self.assertFalse(hasattr(mf1._opt[None], 'semi_direct_eri'))
        mf1 = scf.RHF(mol)
        mf1._is_mem_enough = lambda: False
        mf1.semi_direct = False
        mf1.kernel()
        self.assertFalse(hasattr(mf1._opt[None], 'semi_direct_eri'))
        self.assertAlmostEqual(mf1.e_tot, e1, 10)

    def test_direct_mapdm(self):
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))
        eri0 = mol.intor('int2e_ip1_sph', comp=3).reshape(3,nao,nao,nao,nao)
        vj0 = numpy.einsum('nijkl,lk->nij', eri0, dm)
        vk0 = numpy.einsum('nijkl,jk->nil', eri0, dm)
        vj1, vk1 = _vhf.direct_mapdm('int2e_ip1_sph', 's2kl',
                                     ('lk->s1ij', 'jk->s1il'),
                                     dm, 3, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_direct_mapdm1(self):
        numpy.random.seed(1)
        nao = mol.nao_nr(cart=True)
        dm = numpy.random.random((nao,nao))
        vhfopt = _vhf.VHFOpt(mol, 'int2e_cart', 'CVHFnrs8_prescreen',
                             'CVHFsetnr_direct_scf',
                             'CVHFsetnr_direct_scf_dm')
        vj0, vk0 = _vhf.direct(dm, mol._atm, mol._bas, mol._env,
                               vhfopt=vhfopt, hermi=0, cart=True)
        vj = _vhf.direct_mapdm('int2e_cart', 's1', 'kl->s1ij', dm, 1,
                               mol._atm, mol._bas, mol._env, vhfopt)
        self.assertTrue(numpy.allclose(vj0, vj))

        vk = _vhf.direct_mapdm('int2e_cart', 's1', 'jk->s1il', [dm]*2, 1,
                               mol._atm, mol._bas, mol._env, vhfopt)
        self.assertTrue(numpy.allclose(vk0, vk[0]))
        self.assertTrue(numpy.allclose(vk0, vk[1]))

    def test_direct_bindm(self):
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))
        vj0, vk0 = _vhf.direct_mapdm('int2e_ip1_sph', 's2kl',
                                     ('lk->s1ij', 'jk->s1il'),
                                     dm, 3, mol._atm, mol._bas, mol._env)
        dms = (dm,dm)
        vj1, vk1 = _vhf.direct_bindm('int2e_ip1_sph', 's2kl',
                                     ('lk->s1ij', 'jk->s1il'),
                                     dms, 3, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vj0,vj1))
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_direct_bindm1(self):
        numpy.random.seed(1)
        nao = mol.nao_nr(cart=True)
        dm = numpy.random.random((nao,nao))
        vhfopt = _vhf.VHFOpt(mol, 'int2e_cart', 'CVHFnrs8_prescreen',
                             'CVHFsetnr_direct_scf',
                             'CVHFsetnr_direct_scf_dm')
        vj0, vk0 = _vhf.direct(dm, mol._atm, mol._bas, mol._env,
                               vhfopt=vhfopt, hermi=0, cart=True)
        vj = _vhf.direct_bindm('int2e_cart', 's1', 'kl->s1ij', dm, 1,
                               mol._atm, mol._bas, mol._env, vhfopt)
        self.assertTrue(numpy.allclose(vj0,vj))

    def test_rdirect_mapdm(self):
        numpy.random.seed(1)
        n2c = nao*2
        dm = (numpy.random.random((n2c,n2c)) +
              numpy.random.random((n2c,n2c)) * 1j)
        eri0 = mol.intor('int2e_g1_spinor', comp=3).reshape(3,n2c,n2c,n2c,n2c)
        vk0 = numpy.einsum('nijkl,jk->nil', eri0, dm)
        vj1, vk1 = _vhf.rdirect_mapdm('int2e_g1_spinor', 'a4ij',
                                      ('lk->s2ij', 'jk->s1il'),
                                      dm, 3, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vk0,vk1))

        vj1 = _vhf.rdirect_mapdm('int2e_g1_spinor', 's1', 'lk->s1ij',
                                 dm, 3, mol._atm, mol._bas, mol._env)
        vj0 = numpy.einsum('nijkl,lk->nij', eri0, dm)
        self.assertTrue(numpy.allclose(vj0,vj1))

    def test_rdirect_bindm(self):
        n2c = nao*2
        numpy.random.seed(1)
        dm = (numpy.random.random((n2c,n2c)) +
              numpy.random.random((n2c,n2c)) * 1j)
        dm = dm + dm.conj().T

        eri0 = mol.intor('int2e_spsp1_spinor').reshape(n2c,n2c,n2c,n2c)
        vk0 = numpy.einsum('ijkl,jk->il', eri0, dm)
        vk1 = _vhf.rdirect_bindm('int2e_spsp1_spinor', 's4', 'jk->s1il',
                                 dm, 1, mol._atm, mol._bas, mol._env)
        self.assertTrue(numpy.allclose(vk0,vk1))

        opt_llll = _vhf.VHFOpt(mol, 'int2e_spinor',
                               'CVHFrkbllll_prescreen',
                               'CVHFrkbllll_direct_scf',
                               'CVHFrkbllll_direct_scf_dm')
        opt_llll._this.contents.r_vkscreen = _vhf._fpointer('CVHFrkbllll_vkscreen')
        eri0 = mol.intor('int2e_spinor').reshape(n2c,n2c,n2c,n2c)
        vk0 = numpy.einsum('ijkl,jk->il', eri0, dm)
        vk1 = _vhf.rdirect_bindm('int2e_spinor', 's1', 'jk->s1il',
                                 dm, 1, mol._atm, mol._bas, mol._env, opt_llll)
        self.assertTrue(numpy.allclose(vk0,vk1))

    def test_nr_direct_ex_drv(self):
        numpy.random.seed(1)
        dm = numpy.random.random((nao, nao))
        dm = dm + dm.conj().T
        shls_excludes = [0, 4] * 4
        vref = _vhf.direct_bindm('int2e_sph', 's8', 'jk->s1il', dm, 1, mol._atm, mol._bas, mol._env)
        self.assertAlmostEqual(lib.fp(vref), 5.0067176755619975, 12)

        v0 = _vhf.direct_bindm('int2e_sph', 's8', 'jk->s1il', dm[:6,:6], 1,
                               mol._atm, mol._bas, mol._env, shls_slice=shls_excludes)
        v1 = _vhf.direct_bindm('int2e_sph', 's8', 'jk->s1il', dm, 1,
                               mol._atm, mol._bas, mol._env, shls_excludes=shls_excludes)
        v1[:6,:6] += v0
        self.assertAlmostEqual(abs(v1 - vref).max(), 0, 12)

        vref = _vhf.direct_mapdm('int2e_sph', 's8', 'ji->s1kl', dm, 1, mol._atm, mol._bas, mol._env)
        self.assertAlmostEqual(lib.fp(vref), 48.61070262547175, 12)

        v0 = _vhf.direct_mapdm('int2e_sph', 's8', 'ji->s1kl', dm[:6,:6], 1,
                               mol._atm, mol._bas, mol._env, shls_slice=shls_excludes)
        v1 = _vhf.direct_mapdm('int2e_sph', 's8', 'ji->s1kl', dm, 1,
                               mol._atm, mol._bas, mol._env, shls_excludes=shls_excludes)
        v1[:6,:6] += v0
        self.assertAlmostEqual(abs(v1 - vref).max(), 0, 12)

    def test_direct_sr_vhf(self):
        numpy.random.seed(1)
        dm = numpy.random.random((nao,nao))
        omega = 0.15
        with mol.with_short_range_coulomb(omega):
            ref = _vhf.direct(dm, mol._atm, mol._bas, mol._env, optimize_sr=False)
            ref = numpy.array(ref)

        vhfopt = _VHFOpt(mol, 'int2e', omega=omega)
        vhfopt.init_cvhf_direct(mol)
        with mol.with_short_range_coulomb(omega):
            vjk = _vhf.direct(dm, mol._atm, mol._bas, mol._env, optimize_sr=True)
            vjk = numpy.array(vjk)

        self.assertAlmostEqual(abs(ref - vjk).max(), 0, 12)
        self.assertAlmostEqual(lib.fp(vjk), 25.317344717490613, 12)

MIN_CUTOFF = 1e-44
libcvhf = _vhf.libcvhf

class _VHFOpt(_vhf._VHFOpt):
    def __init__(self, mol, intor=None, prescreen='CVHFnoscreen',
                 qcondname=None, dmcondname=None, direct_scf_tol=1e-14,
                 omega=None):
        assert omega is not None
        with mol.with_short_range_coulomb(omega):
            _vhf._VHFOpt.__init__(self, mol, intor, prescreen, qcondname, dmcondname)
        self.omega = omega
        self._this.direct_scf_tol = numpy.log(direct_scf_tol)

    def init_cvhf_direct(self, mol, intor=None, qcondname=None):
        nbas = mol.nbas
        q_cond = numpy.empty((6,nbas,nbas), dtype=numpy.float32)
        ao_loc = mol.ao_loc
        cintopt = self._cintopt
        with mol.with_short_range_coulomb(self.omega):
            libcvhf.CVHFsetnr_sr_direct_scf(
                libcvhf.int2e_sph, cintopt,
                q_cond.ctypes.data_as(ctypes.c_void_p),
                ao_loc.ctypes.data_as(ctypes.c_void_p),
                mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                mol._env.ctypes.data_as(ctypes.c_void_p))

        self._q_cond = q_cond
        logq_cond = q_cond.ctypes.data_as(ctypes.c_void_p)
        self._this.q_cond = logq_cond

    def set_dm(self, dm, atm=None, bas=None, env=None):
        assert dm[0].ndim == 2
        ao_loc = self.mol.ao_loc_nr()
        dm_cond = [lib.condense('NP_absmax', d, ao_loc, ao_loc) for d in dm]
        dm_cond = numpy.max(dm_cond, axis=0)
        dm_cond += MIN_CUTOFF  # to remove divide-by-zero error
        self._dm_cond = numpy.asarray(dm_cond, order='C', dtype=np.float32)
        self._this.dm_cond = self._dm_cond.ctypes.data_as(ctypes.c_void_p)


if __name__ == "__main__":
    print("Full Tests for _vhf")
    unittest.main()
