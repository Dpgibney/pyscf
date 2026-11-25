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

import unittest
import numpy
from pyscf import gto, dft, lib
from pyscf import grad, hessian
try:
    from pyscf.dispersion import dftd3, dftd4
except (ImportError, OSError):
    dftd3 = dftd4 = None

def setUpModule():
    global mol, h4
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = '/dev/null'
    mol.atom.extend([
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = '6-31g'
    mol.build()

    h4 = gto.Mole()
    h4.verbose = 0
    h4.output = '/dev/null'
    h4.atom = [
        [1 , (1. ,  0.     , 0.000)],
        [1 , (0. ,  1.     , 0.000)],
        [1 , (0. , -1.517  , 1.177)],
        [1 , (0. ,  1.517  , 1.177)]]
    h4.basis = '631g'
    h4.unit = 'B'
    h4.build()

def tearDownModule():
    global mol, h4
    mol.stdout.close()
    h4.stdout.close()
    del mol, h4

def finite_diff(mf):
    mol = mf.mol
    mfs = mf.Gradients().set(grid_response=True).as_scanner()
    def grad_full(ia, inc):
        coord = mol.atom_coord(ia).copy()
        ptr = mol._atm[ia,gto.PTR_COORD]
        de = []
        for i in range(3):
            mol._env[ptr+i] = coord[i] + inc
            e1a = mfs(mol.copy())[1]
            mol._env[ptr+i] = coord[i] - inc
            e1b = mfs(mol.copy())[1]
            mol._env[ptr+i] = coord[i]
            de.append((e1a-e1b)/(2*inc))
        return de
    natm = mol.natm
    e2ref = [grad_full(ia, .5e-3) for ia in range(mol.natm)]
    e2ref = numpy.asarray(e2ref).reshape(natm,3,natm,3).transpose(0,2,1,3)
    return e2ref

def finite_diff_rows(mf, ia, steps=(5e-4, 1e-3)):
    '''The rows [natm,3,3] of atom ia of the Hessian: central differences of the gradient with grid_response,
    extrapolated over the two steps'''
    mol = mf.mol
    g = mf.Gradients().set(grid_response=True)
    if hasattr(g, 'auxbasis_response'):
        g.auxbasis_response = True
    gs = g.as_scanner()
    x0 = mol.atom_coords()
    rows = []
    for h in steps:
        r = numpy.empty((mol.natm,3,3))
        for x in range(3):
            de = []
            for s in (h, -h):
                xs = x0.copy()
                xs[ia,x] += s
                de.append(gs(mol.set_geom_(xs, unit='Bohr', inplace=False))[1])
            r[:,x] = (de[0] - de[1]) / (2*h)
        rows.append(r)
    return (4 * rows[0] - rows[1]) / 3

def finite_partial_diff(mf):
    # \partial^2 E / \partial R \partial R'
    mol = mf.mol
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
    natm = mol.natm
    e2ref = [grad_partial_R(ia, .5e-3) for ia in range(mol.natm)]
    e2ref = numpy.asarray(e2ref).reshape(natm,3,natm,3).transpose(0,2,1,3)
    return e2ref

class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_grid_response(self):
        # With grid_response the Hessian is the derivative of the gradient with grid_response: the rows of the
        # Cl atom against central differences of that gradient (Richardson extrapolation of two steps), on
        # coarse grids where the terms of the grids are large (up to 30 here without them); LDA, GGA, meta-GGA,
        # VV10, and density fitting
        mol1 = gto.M(atom='O 0 0 0.1; H 0 .76 .58; Cl 0.3 -1.2 -0.4', basis='def2-svp', verbose=0)
        for xc, df in (('lda,vwn', False), ('pbe', True), ('tpss', False), ('wb97m-v', False)):
            with self.subTest(xc=xc, df=df):
                mf = dft.RKS(mol1, xc=xc)
                if df:
                    mf = mf.density_fit()
                mf.grids.atom_grid = (30, 110)
                mf.nlcgrids.atom_grid = (20, 50)
                mf.conv_tol = 1e-13
                mf.conv_tol_grad = 1e-10
                mf.kernel()
                hobj = mf.Hessian().set(grid_response=True, cphf_residual_tol=1e-8)
                if df:
                    hobj.auxbasis_response = 2
                hess = hobj.kernel()
                ref = finite_diff_rows(mf, 2)
                self.assertAlmostEqual(abs(hess[2] - ref).max(), 0, 6)
                self.assertAlmostEqual(abs(hess.sum(axis=0)).max(), 0, 8)
                hess0 = mf.Hessian().set(cphf_residual_tol=1e-8).kernel()
                self.assertTrue(abs(hess0[2] - ref).max() > 1e-2)

    def test_becke_weight_derivs(self):
        # The first and second derivatives of the Becke weights against central differences of the weights
        # of the points, which move with their atoms
        from pyscf.hessian import rks as rks_hess
        mol1 = gto.M(atom='O 0 0 0; H 0 .76 .58; H 0 -.8 .5; Cl 1.9 0.2 0.1', basis='sto-3g', charge=1,
                     verbose=0)
        def build(m):
            g = dft.gen_grid.Grids(m)
            g.atom_grid = (20, 50)
            g.alignment = 0
            return g.build(sort_grids=False)
        grids = build(mol1)
        eps = numpy.random.default_rng(1).standard_normal(grids.weights.size)
        dw, d2 = rks_hess._becke_weight_derivs(mol1, grids, eps=eps)
        x0 = mol1.atom_coords()
        h = 1e-4
        for ia in (0, 3):
            for x in range(3):
                w, d = [], []
                for s in (h, -h):
                    xs = x0.copy()
                    xs[ia,x] += s
                    m = mol1.set_geom_(xs, unit='Bohr', inplace=False)
                    g = build(m)
                    w.append(g.weights)
                    d.append(numpy.einsum('Byg,g->By', rks_hess._becke_weight_derivs(m, g)[0], eps))
                fd1 = (w[0] - w[1]) / (2*h)
                fd2 = (d[0] - d[1]) / (2*h)
                self.assertAlmostEqual(abs(dw[ia,x] - fd1).max() / abs(fd1).max(), 0, 7)
                self.assertAlmostEqual(abs(d2[ia,:,x] - fd2).max() / abs(fd2).max(), 0, 6)
        self.assertAlmostEqual(abs(d2 - d2.transpose(1,0,3,2)).max(), 0, 10)
        self.assertAlmostEqual(abs(d2.sum(axis=0)).max(), 0, 9)

    def test_cphf_xc_lowrank(self):
        # The CPHF operator with the XC response from the factors of the dms
        # (vind_mo of the response function, either path) against the one from
        # the dms
        from pyscf.dft import numint
        from pyscf.hessian import rhf as rhf_hess
        numpy.random.seed(2)
        for xc in ('lda,vwn', 'b3lyp', 'r2scan'):
            mf = dft.RKS(mol, xc=xc).run(conv_tol=1e-8)
            mo, occ = mf.mo_coeff, mf.mo_occ
            mo1 = numpy.random.random((6, mo.shape[1], (occ > 0).sum())) - .5
            self.assertTrue(hasattr(mf.gen_response(mo, occ, hermi=1), 'vind_mo'))
            with lib.temporary_env(numint, FUSED_XC=False):
                ref = rhf_hess.gen_vind(mf, mo, occ)(mo1)
            for cheaper in (True, False):
                with lib.temporary_env(numint._FusedXC,
                                       lowrank_cheaper=lambda *args: cheaper):
                    v = rhf_hess.gen_vind(mf, mo, occ)(mo1)
                self.assertAlmostEqual(abs(v - ref).max() / abs(ref).max(), 0, 13)

    def test_rks_hess_atmlst(self):
        mf = dft.RKS(mol)
        mf.xc = 'pbe0'
        mf.conv_tol = 1e-14
        e0 = mf.kernel()

        atmlst = [0, 1]
        hess_1 = mf.Hessian().kernel()[atmlst][:, atmlst]
        hess_2 = mf.Hessian().kernel(atmlst=atmlst)
        self.assertAlmostEqual(abs(hess_1-hess_2).max(), 0.0, 4)

    def test_finite_diff_lda_hess(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7828771346902333, 4)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        #FIXME: errors seems too big
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 3)

    def test_second_grids_hess(self):
        mf = dft.RKS(mol)
        mf.xc = 'b3lyp5'
        mf.conv_tol = 1e-14
        mf.kernel()
        hess = mf.Hessian().kernel()

        # second_grids == grids reproduces the default hessian
        mf.second_grids = mf.grids
        hess1 = mf.Hessian().kernel()
        self.assertAlmostEqual(abs(hess - hess1).max(), 0, 7)
        mf.second_grids = None

        # A coarse (level-1) secondary grid changes the CPHF response only
        mf.set_second_grids(1)
        hess2 = mf.Hessian().kernel()
        self.assertAlmostEqual(abs(hess - hess2).max(), 0, 4)
        mf.second_grids = None

    def test_finite_diff_b3lyp_hess(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        mf.xc = 'b3lyp5'
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7590878171493624, 4)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        #FIXME: errors seems too big
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 3)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_finite_diff_b3lyp_d3_hess_high_cost(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        mf.xc = 'b3lyp'
        mf.disp = 'd3bj'
        mf.kernel()
        hess = mf.Hessian().kernel()

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        #FIXME: errors seems too big
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 3)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_consistency_b3lyp_d3_hess(self):
        mf1 = dft.RKS(mol)
        mf1.conv_tol = 1e-14
        mf1.xc = 'b3lyp'
        mf1.disp = 'd3bj'
        mf1.kernel()
        hess1 = mf1.Hessian().kernel()

        mf2 = dft.RKS(mol)
        mf2.conv_tol = 1e-14
        mf2.xc = 'b3lyp-d3bj'
        mf2.kernel()
        hess2 = mf2.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(hess1), lib.fp(hess2), 4)

    @unittest.skipIf(dftd4 is None, "requires the dftd4 library")
    def test_finite_diff_b3lyp_d4_hess(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        mf.xc = 'b3lyp'
        mf.disp = 'd4'
        mf.kernel()
        hess = mf.Hessian().kernel()

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        #FIXME: errors seems too big
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 3)

    def test_finite_diff_wb97x_hess(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        mf.xc = 'wb97x'
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7637876979690904, 4)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        #FIXME: errors seems too big
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 2)

    def test_finite_diff_m06l_hess_high_cost(self):
        mf = dft.RKS(mol)
        mf.conv_tol = 1e-14
        mf.xc = 'm06l'
        # Note MGGA hessian is sensitive to grids level
        mf.grids.level = 4
        e0 = mf.kernel()
        hess = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(hess), -0.7671113269563008, 6)

        g_scanner = mf.nuc_grad_method().as_scanner()
        pmol = mol.copy()
        e1 = g_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        e2 = g_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))[1]
        #FIXME: errors seems too big
        self.assertAlmostEqual(abs(hess[0,:,2] - (e1-e2)/2e-4*lib.param.BOHR).max(), 0, 2)

    def test_finite_diff_lda_hess_high_cost(self):
        mf = dft.RKS(h4)
        mf.grids.level = 4
        mf.xc = 'lda,vwn'
        mf.conv_tol = 1e-14
        mf.kernel()
        e2 = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(e2), 0.5586931702431067, 6)
        e2ref = finite_diff(mf)
        self.assertAlmostEqual(abs(e2-e2ref).max(), 0, 3)

    def test_finite_diff_b3lyp_hess_high_cost(self):
        mf = dft.RKS(h4)
        mf.grids.level = 4
        mf.xc = 'b3lyp5'
        mf.conv_tol = 1e-14
        mf.kernel()
        e2 = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(e2), 0.5755856712112237, 6)
        e2ref = finite_diff(mf)
        self.assertAlmostEqual(abs(e2-e2ref).max(), 0, 4)

    def test_finite_diff_m06l_hess_high_cost(self):
        mf = dft.RKS(h4)
        mf.grids.level = 4
        mf.xc = 'm06l'
        mf.conv_tol = 1e-14
        mf.kernel()
        e2 = mf.Hessian().kernel()
        self.assertAlmostEqual(lib.fp(e2), 0.5479869865170132, 6)
        e2ref = finite_diff(mf)
        self.assertAlmostEqual(abs(e2-e2ref).max(), 0, 3)

    def test_finite_diff_lda_partial_hess_high_cost(self):
        mf = dft.RKS(h4)
        mf.grids.level = 4
        mf.xc = 'lda,vwn'
        mf.conv_tol = 1e-14
        mf.kernel()
        hobj = mf.Hessian()
        e2 = hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
        e2 += hobj.hess_nuc(h4)
        e2ref = finite_partial_diff(mf)
        self.assertAlmostEqual(abs(e2-e2ref).max(), 0, 6)

    def test_finite_diff_b3lyp_partial_hess_high_cost(self):
        mf = dft.RKS(h4)
        mf.grids.level = 4
        mf.xc = 'b3lyp'
        mf.conv_tol = 1e-14
        mf.kernel()
        hobj = mf.Hessian()
        e2 = hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
        e2 += hobj.hess_nuc(h4)
        e2ref = finite_partial_diff(mf)
        self.assertAlmostEqual(abs(e2-e2ref).max(), 0, 6)

    def test_finite_diff_m06l_partial_hess_high_cost(self):
        mf = dft.RKS(h4)
        mf.grids.level = 4
        mf.xc = 'm06l'
        mf.conv_tol = 1e-14
        mf.kernel()
        hobj = mf.Hessian()
        e2 = hobj.partial_hess_elec(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
        e2 += hobj.hess_nuc(h4)
        e2ref = finite_partial_diff(mf)
        self.assertAlmostEqual(abs(e2-e2ref).max(), 0, 6)


    def test_progress_lines(self):
        # The progress lines of the steps that take long (lib.logger.Progress, the CPHF cycles): with
        # Logger.info, once a step has run for PROGRESS_AFTER seconds, so none for a small molecule
        # unless forced; the Hessian is the same either way
        import io
        hess = {}
        text = {}
        for after in (lib.logger.PROGRESS_AFTER, 0.):
            mol1 = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='6-31g', verbose=0)
            mol1.stdout = io.StringIO()
            mol1.verbose = 4
            mf = dft.RKS(mol1, xc='b3lyp').run(conv_tol=1e-12)
            mol1.stdout.truncate(0)
            with lib.temporary_env(lib.logger, PROGRESS_AFTER=after):
                hess[after] = mf.Hessian().kernel()
            text[after] = mol1.stdout.getvalue()
        self.assertNotIn('% of the', text[lib.logger.PROGRESS_AFTER])
        self.assertNotIn('CPHF cycle', text[lib.logger.PROGRESS_AFTER])
        self.assertIn('    J and K Fock matrix derivatives: 33% of the atoms after ', text[0.])
        self.assertIn('    J and K Fock matrix derivatives: done in ', text[0.])
        self.assertIn('    CPHF cycle 0 after ', text[0.])
        self.assertAlmostEqual(abs(hess[0.] - hess[lib.logger.PROGRESS_AFTER]).max(), 0, 12)

if __name__ == "__main__":
    print("Full Tests for RKS Hessian")
    unittest.main()
