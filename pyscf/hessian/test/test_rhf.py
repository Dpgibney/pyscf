#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import io
import unittest
import numpy
from pyscf import gto, scf, lib
from pyscf import grad, hessian
try:
    from pyscf.dispersion import dftd3, dftd4
except (ImportError, OSError):
    dftd3 = dftd4 = None

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = 'ccpvdz'
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_cphf_iterations_are_logged(self):
        # solve_mo1 passes the Hessian's logger to the CPHF solver, so its Krylov cycles reach the Hessian's stdout
        mf = scf.RHF(mol).run()
        hobj = mf.Hessian()
        hobj.verbose = 5
        hobj.stdout = io.StringIO()
        hobj.kernel()
        self.assertIn('krylov cycle', hobj.stdout.getvalue())

    def test_rhf_hess(self):
        mf = scf.RHF(mol)
        e0 = mf.kernel()
        hess = hessian.RHF(mf).kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7816352153153946, 4)

        hobj = hessian.RHF(mf)
        hobj.max_cycle = 10
        hobj.level_shift = .1
        hess = hobj.kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7816352153153946, 4)

    def test_rhf_hess_atmlst(self):
        mf = scf.RHF(mol)
        e0 = mf.kernel()

        atmlst = [0, 1]
        hess_1 = mf.Hessian().kernel()[atmlst][:, atmlst]
        hess_2 = mf.Hessian().kernel(atmlst=atmlst)
        self.assertAlmostEqual(abs(hess_1-hess_2).max(), 0.0, 4)

    def test_partial_hess_2e_translational_invariance(self):
        # With every atom, _partial_hess_2e takes the diagonal blocks from translational
        # invariance; with one atom, from int2e_ipip1
        from pyscf.hessian import rhf as rhf_hess
        dm = scf.RHF(mol).run().make_rdm1()
        natm = mol.natm
        for dm_j, dms_k, k_fac, omega in ((dm, [dm], 1, None),
                                          (dm, [dm*.6, dm*.4], 2, None),
                                          (None, [dm], 1, .4)):
            with_j = dm_j is not None
            ej, ek = rhf_hess._partial_hess_2e(mol, dm_j, dms_k, k_fac, range(natm),
                                               with_j=with_j, omega=omega)
            for ia in range(natm):
                ej1, ek1 = rhf_hess._partial_hess_2e(mol, dm_j, dms_k, k_fac, [ia],
                                                     with_j=with_j, omega=omega)
                self.assertAlmostEqual(abs(ej1[0,0] - ej[ia,ia]).max(), 0, 9)
                self.assertAlmostEqual(abs(ek1[0,0] - ek[ia,ia]).max(), 0, 9)

    def test_finite_diff_x2c_rhf_hess(self):
        mf = scf.RHF(mol).x2c()
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        hess = hessian.RHF(mf).kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7800532318291435, 4)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)

#        e1 = g_scanner(pmol.set_geom_('O  0. 0.0001 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
#        e2 = g_scanner(pmol.set_geom_('O  0. -.0001 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
#        self.assertAlmostEqual(abs(hess[0,:,1] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)
#
#        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7571 0.587; 1  0. 0.757 0.587'))[1]
#        e2 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7569 0.587; 1  0. 0.757 0.587'))[1]
#        self.assertAlmostEqual(abs(hess[1,:,1] - (e2-e1)/2e-4*lib.param.BOHR).max(), 0, 4)

        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5871; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5869; 1  0. 0.757 0.587'))[1]
        self.assertAlmostEqual(abs(hess[1,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)


    def test_finite_diff_rhf_hess(self):
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        hess = hessian.RHF(mf).kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7816353049729151, 6)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)

    def test_finite_diff_rhf_hess1(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.output = None
        mol.atom = [
            [1 , (1. ,  0.     , 0.000)],
            [1 , (0. ,  1.     , 0.000)],
            [1 , (0. , -1.517  , 1.177)],
            [1 , (0. ,  1.517  , 1.177)] ]
        mol.basis = '631g'
        mol.unit = 'B'
        mol.build()
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        mf.scf()
        n3 = mol.natm * 3
        hobj = mf.Hessian()
        e2 = hobj.kernel().transpose(0,2,1,3).reshape(n3,n3)
        self.assertAlmostEqual(lib.fp(e2), -0.50693144355876429, 6)
        #from hessian import rhf_o0
        #e2ref = rhf_o0.Hessian(mf).kernel().transpose(0,2,1,3).reshape(n3,n3)
        #print numpy.linalg.norm(e2-e2ref)
        #print numpy.allclose(e2,e2ref)

        def grad_full(ia, inc):
            coord = mol.atom_coord(ia).copy()
            ptr = mol._atm[ia,gto.PTR_COORD]
            de = []
            for i in range(3):
                mol._env[ptr+i] = coord[i] + inc
                mf = scf.RHF(mol).run(conv_tol=1e-14)
                e1a = mf.nuc_grad_method().kernel()
                mol._env[ptr+i] = coord[i] - inc
                mf = scf.RHF(mol).run(conv_tol=1e-14)
                e1b = mf.nuc_grad_method().kernel()
                mol._env[ptr+i] = coord[i]
                de.append((e1a-e1b)/(2*inc))
            return de
        e2ref = [grad_full(ia, .5e-4) for ia in range(mol.natm)]
        e2ref = numpy.asarray(e2ref).reshape(n3,n3)
        self.assertAlmostEqual(abs(e2-e2ref).max(), 0, 6)

    # \partial^2 E / \partial R \partial R'
        e2 = hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
        e2 += hobj.hess_nuc(mol)
        e2 = e2.transpose(0,2,1,3).reshape(n3,n3)
        def grad_partial_R(ia, inc):
            coord = mol.atom_coord(ia).copy()
            ptr = mol._atm[ia,gto.PTR_COORD]
            de = []
            for i in range(3):
                mol._env[ptr+i] = coord[i] + inc
                e1a = mf.nuc_grad_method().kernel()
                mol._env[ptr+i] = coord[i] - inc
                e1b = mf.nuc_grad_method().kernel()
                mol._env[ptr+i] = coord[i]
                de.append((e1a-e1b)/(2*inc))
            return de
        e2ref = [grad_partial_R(ia, .5e-4) for ia in range(mol.natm)]
        e2ref = numpy.asarray(e2ref).reshape(n3,n3)
        self.assertAlmostEqual(abs(e2-e2ref).max(), 0, 8)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_finite_diff_rhf_d3_hess(self):
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        mf.disp = 'd3bj'
        e0 = mf.kernel()
        hess = hessian.RHF(mf).kernel()

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)

    @unittest.skipIf(dftd4 is None, "requires the dftd4 library")
    def test_finite_diff_rhf_d4_hess_high_cost(self):
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        mf.disp = 'd4'
        e0 = mf.kernel()
        hess = hessian.RHF(mf).kernel()

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)

#        e1 = g_scanner(pmol.set_geom_('O  0. 0.0001 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
#        e2 = g_scanner(pmol.set_geom_('O  0. -.0001 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
#        self.assertAlmostEqual(abs(hess[0,:,1] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)
#
#        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7571 0.587; 1  0. 0.757 0.587'))[1]
#        e2 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7569 0.587; 1  0. 0.757 0.587'))[1]
#        self.assertAlmostEqual(abs(hess[1,:,1] - (e2-e1)/2e-4*lib.param.BOHR).max(), 0, 4)
#
#        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5871; 1  0. 0.757 0.587'))[1]
#        e2 = g_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5869; 1  0. 0.757 0.587'))[1]
#        self.assertAlmostEqual(abs(hess[1,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 4)

    def test_ecp_hess(self):
        mol = gto.M(atom='Cu 0 0 0; H 0 0 1.5', basis='lanl2dz',
                    ecp={'Cu':'lanl2dz'}, verbose=0)
        mf = scf.RHF(mol).run(conv_tol=1e-14)
        hess = hessian.RHF(mf).kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.20927804440983355, 6)

        mfs = mf.nuc_grad_method().as_scanner()
        e1 = mfs(mol.set_geom_('Cu 0 0  0.001; H 0 0 1.5'))[1]
        e2 = mfs(mol.set_geom_('Cu 0 0 -0.001; H 0 0 1.5'))[1]
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/0.002*lib.param.BOHR).max(), 0, 5)

#        mfs = mf.nuc_grad_method().as_scanner()
#        e1 = mfs(mol.set_geom_('Cu 0 0 0; H 0 0 1.5001'))[1]
#        e2 = mfs(mol.set_geom_('Cu 0 0 0; H 0 0 1.4999'))[1]
#        self.assertAlmostEqual(abs(hess[1,:,2] - (e1-e2)/0.0002*lib.param.BOHR).max(), 0, 5)

    def test_hess_2e_int2e_batch(self):
        # qcint's batched Hessian integrals (int2e_ipip1, ipvip1, ip1ip2) and int2e_ip1 (make_h1)
        # against them quartet by quartet, with d and f shells, every atom and a part of them; with
        # libcint both take the same path
        from pyscf.scf import _vhf
        from pyscf.hessian import rhf as rhf_hess
        mol1 = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='def2-tzvp', verbose=0)
        mf1 = scf.RHF(mol1).run()
        dm0 = mf1.make_rdm1()
        res = []
        for batch in (False, True):
            with lib.temporary_env(_vhf, INT2E_BATCH=batch):
                ej, ek = rhf_hess._partial_hess_2e(mol1, dm0, [dm0], 1, range(mol1.natm))
                # not every atom: the diagonal blocks from int2e_ipip1
                ej1, ek1 = rhf_hess._partial_hess_2e(mol1, dm0, [dm0], 1, [2, 0])
                h1 = rhf_hess.make_h1(hessian.RHF(mf1), mf1.mo_coeff, mf1.mo_occ)
            res.append((ej, ek, ej1, ek1, numpy.asarray(h1)))
        for a, b in zip(*res):
            self.assertTrue(abs(a).max() > 1e-2)
            self.assertAlmostEqual(abs(a - b).max(), 0, 9)


if __name__ == "__main__":
    print("Full Tests for RHF Hessian")
    unittest.main()
