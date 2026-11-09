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

'''Hessian.sgx_response and Hessian.cphf_residual_tol, and the solver behind them'''

import unittest
from unittest import mock
import numpy
from pyscf import gto, scf, dft, lib
from pyscf.scf import cphf, _response_functions
from pyscf.sgx import sgx, sgx_jk


def setUpModule():
    global mol, mf, h_ref
    mol = gto.M(atom='O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692',
                basis='def2-svp', verbose=0)
    mf = dft.RKS(mol, xc='b3lyp')
    mf.conv_tol = 1e-11
    mf.kernel()
    hobj = mf.Hessian()
    hobj.cphf_residual_tol = 1e-9
    h_ref = hobj.kernel()

def tearDownModule():
    global mol, mf, h_ref
    del mol, mf, h_ref

def _toy_problem(nset=3, nocc=3, nvir=5, seed=7):
    '''The equations of a small random system whose exact solution is a dense solve'''
    rng = numpy.random.default_rng(seed)
    nmo = nocc + nvir
    ndim = nmo * nocc
    a = rng.random((ndim, ndim))
    k = (a + a.T) * .03
    mo_energy = numpy.append(numpy.sort(-1 - rng.random(nocc)), numpy.sort(.5 + 3*rng.random(nvir)))
    mo_occ = numpy.array([2]*nocc + [0]*nvir)
    h1 = rng.random((nset, nmo, nocc)) - .5
    s1 = rng.random((nset, nmo, nocc)) - .5
    s1[:,:nocc] = s1[:,:nocc] + s1[:,:nocc].transpose(0,2,1)
    def fvind(x):
        n = len(x.reshape(-1, ndim))
        return numpy.dot(x.reshape(n, ndim), k).reshape(n, nmo, nocc)
    e_i = mo_energy[:nocc]
    e_ai = (mo_energy[nocc:,None] - e_i).ravel()
    oo = numpy.arange(nocc*nocc)
    vo = numpy.arange(nocc*nocc, ndim)
    # the occupied rows come first in the flattened (nmo,nocc)
    xoo = -s1[:,:nocc].reshape(nset, nocc*nocc) * .5
    hs = h1 - s1 * e_i
    rhs = -hs[:,nocc:].reshape(nset, nvir*nocc) - numpy.dot(xoo, k[oo[0]:oo[-1]+1,vo[0]:])
    xvo = numpy.linalg.solve(numpy.diag(e_ai) + k[vo[0]:,vo[0]:], rhs.T).T
    x = numpy.hstack((xoo, xvo)).reshape(nset, nmo, nocc)
    v = fvind(x)
    e1 = (hs + v)[:,:nocc] + x[:,:nocc] * (e_i[:,None] - e_i)
    return fvind, mo_energy, mo_occ, h1, s1, x, e1

class KnownValues(unittest.TestCase):
    def test_solve_residual_against_dense(self):
        fvind, mo_energy, mo_occ, h1, s1, x, e1 = _toy_problem()
        mo1, mo_e1 = cphf.solve_residual(fvind, mo_energy, mo_occ, h1, s1, tol=1e-10)
        self.assertLess(abs(mo1 - x).max(), 1e-9)
        self.assertLess(abs(mo_e1 - e1).max(), 1e-9)

    def test_solve_residual_fewer_vectors_when_loose(self):
        fvind, mo_energy, mo_occ, h1, s1, x, e1 = _toy_problem(nset=2)
        nvec = []
        for tol in (1e-1, 1e-8):
            count = [0]
            def f(m):
                count[0] += len(m)
                return fvind(m)
            cphf.solve_residual(f, mo_energy, mo_occ, h1, s1, tol=tol)
            nvec.append(count[0])
        self.assertLess(nvec[0], nvec[1])

    def test_solve_residual_max_cycle(self):
        fvind, mo_energy, mo_occ, h1, s1, x, e1 = _toy_problem()
        with self.assertRaises(RuntimeError):
            cphf.solve_residual(fvind, mo_energy, mo_occ, h1, s1, tol=1e-14, max_cycle=1)

    def test_solve_residual_level_shift(self):
        # only the preconditioner changes
        fvind, mo_energy, mo_occ, h1, s1, x, e1 = _toy_problem()
        mo1 = cphf.solve_residual(fvind, mo_energy, mo_occ, h1, s1, tol=1e-10, level_shift=.1)[0]
        self.assertLess(abs(mo1 - x).max(), 1e-9)

    def test_solve_residual_scale_trials(self):
        # fvind is linear: the same solution, and fvind sees smaller vectors after the first cycle
        fvind, mo_energy, mo_occ, h1, s1, x, e1 = _toy_problem()
        norms = []
        def f(m):
            norms.append(numpy.linalg.norm(m.reshape(len(m), -1), axis=1).max())
            return fvind(m)
        mo1, mo_e1 = cphf.solve_residual(f, mo_energy, mo_occ, h1, s1, tol=1e-10, scale_trials=True)
        self.assertLess(abs(mo1 - x).max(), 1e-9)
        self.assertLess(abs(mo_e1 - e1).max(), 1e-9)
        self.assertLess(norms[-1], 1e-2)

    def test_scale_trials_exact_response_only(self):
        used = []
        solver = cphf.solve_residual
        def record(*args, **kwargs):
            used.append(kwargs['scale_trials'])
            return solver(*args, **kwargs)
        cphf.solve_residual = record
        try:
            hobj = mf.Hessian()
            hobj.cphf_residual_tol = 1e-4
            h = hobj.kernel()
            hobj = mf.Hessian()
            hobj.sgx_response = True
            hobj.kernel()
        finally:
            cphf.solve_residual = solver
        self.assertEqual(used, [True, False])
        self.assertLess(abs(h - h_ref).max(), 1e-6)

    def test_residual_tol_against_stock(self):
        # cphf.solve stops short for DFT, here by 7.5e-6 from the converged Hessian
        self.assertLess(abs(mf.Hessian().kernel() - h_ref).max(), 3e-5)
        hobj = mf.Hessian()
        hobj.cphf_residual_tol = 1e-4
        self.assertLess(abs(hobj.kernel() - h_ref).max(), 1e-6)

    def test_sgx_response(self):
        hobj = mf.Hessian()
        hobj.sgx_response = True
        h0 = hobj.kernel()
        d0 = abs(h0 - h_ref).max()
        hobj = mf.Hessian()
        hobj.sgx_response = True
        hobj.sgx_response_level = 1
        d1 = abs(hobj.kernel() - h_ref).max()
        # the approximation shows, and shrinks with the grid
        self.assertLess(d0, 1e-3)
        self.assertGreater(d0, 1e-7)
        self.assertLess(d1, d0)

    def test_sgx_response_k_from_factors(self):
        # the K of the response from the factors of the dms in the MO basis, and direct J by
        # chunks, against K and J of the dms: plain, range-separated (long- and short-range K
        # as well), and only long-range or short-range exchange
        for xc in ('b3lyp', 'wb97m-v', 'lrc-wpbe', 'hse06'):
            mf1 = dft.RKS(mol, xc=xc)
            mf1.conv_tol = 1e-11
            if xc == 'wb97m-v':
                mf1.nlcgrids.level = 1
            mf1.kernel()
            hobj = mf1.Hessian()
            hobj.sgx_response = True
            used = []
            get_k_mo = sgx.SGX.get_k_mo
            def count(self, *args, **kwargs):
                v = get_k_mo(self, *args, **kwargs)
                used.append(v is not None)
                return v
            with lib.temporary_env(sgx_jk, SGX_MANY_DM_MIN=1):
                with mock.patch.object(sgx.SGX, 'get_k_mo', count), \
                        mock.patch.object(_response_functions, '_cphf_chunk', lambda *a: 2):
                    h = hobj.kernel()
                with mock.patch.object(sgx.SGX, 'get_k_mo', lambda *a, **k: None):
                    ref = hobj.kernel()
            self.assertTrue(used and all(used), xc)
            self.assertLess(abs(h - ref).max(), 1e-10, xc)

    def test_sgx_response_rhf(self):
        mf1 = scf.RHF(mol)
        mf1.conv_tol = 1e-11
        mf1.kernel()
        hobj = mf1.Hessian()
        hobj.cphf_residual_tol = 1e-8
        ref = hobj.kernel()
        hobj = mf1.Hessian()
        hobj.sgx_response = True
        self.assertLess(abs(hobj.kernel() - ref).max(), 1e-3)

    def test_not_implemented(self):
        molu = mol.copy()
        molu.charge = 1
        molu.spin = 1
        molu.build(0, 0)
        mfu = scf.UHF(molu).run()
        for attr, val in (('sgx_response', True), ('cphf_residual_tol', 1e-4)):
            hobj = mfu.Hessian()
            setattr(hobj, attr, val)
            with self.assertRaises(NotImplementedError):
                hobj.solve_mo1(mfu.mo_energy, mfu.mo_coeff, mfu.mo_occ, None)
        mfd = scf.RHF(mol).density_fit().run()
        hobj = mfd.Hessian()
        hobj.sgx_response = True
        with self.assertRaises(NotImplementedError):
            hobj.solve_mo1(mfd.mo_energy, mfd.mo_coeff, mfd.mo_occ, None)


if __name__ == '__main__':
    print('Full Tests for sgx_response and cphf_residual_tol')
    unittest.main()
