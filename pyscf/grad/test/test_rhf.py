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
from pyscf import gto, scf, lib
from pyscf import grad

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
    mol.basis = '6-31g'
    mol.build()

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

class KnownValues(unittest.TestCase):
    def test_rhf_grad_one_atom(self):
        mol = gto.Mole()
        mol.verbose = 0
        mol.atom = [['He', (0.,0.,0.)], ]
        mol.basis = {'He': 'ccpvdz'}
        mol.build()
        method = scf.RHF(mol)
        method.scf()
        g1 = method.Gradients().grad()
        self.assertAlmostEqual(lib.fp(g1), 0, 9)

    def test_rhf_grad(self):
        g_scan = scf.RHF(mol).nuc_grad_method().as_scanner()
        g = g_scan(mol)[1]
        self.assertAlmostEqual(lib.fp(g), 0.0055116240804341972, 7)

        gobj = g_scan.undo_scanner()
        g = gobj.kernel()
        self.assertAlmostEqual(lib.fp(g), 0.0055116240804341972, 7)

        mfs = g_scan.base.as_scanner()
        e1 = mfs('O  0.  0. -0.001; H  0.  -0.757  0.587; H  0.  0.757   0.587')
        e2 = mfs('O  0.  0.  0.001; H  0.  -0.757  0.587; H  0.  0.757   0.587')
        self.assertAlmostEqual(g[0,2], (e2-e1)/0.002*lib.param.BOHR, 5)

    def test_df_rhf_grad(self):
        g_scan = scf.RHF(mol).density_fit ().nuc_grad_method().as_scanner()
        g = g_scan(mol)[1]
        self.assertAlmostEqual(lib.fp(g), 0.005516638190188906, 7)

        mfs = g_scan.base.as_scanner()
        e1 = mfs('O  0.  0. -0.001; H  0.  -0.757  0.587; H  0.  0.757   0.587')
        e2 = mfs('O  0.  0.  0.001; H  0.  -0.757  0.587; H  0.  0.757   0.587')
        self.assertAlmostEqual(g[0,2], (e2-e1)/0.002*lib.param.BOHR, 5)

    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_rhf_d3_grad(self):
        mf = scf.RHF(mol)
        mf.disp = 'd3bj'
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(mol)[1]

        mf_scan = mf.as_scanner()
        e1 = mf_scan('O  0.  0. -0.001; H  0.  -0.757  0.587; H  0.  0.757   0.587')
        e2 = mf_scan('O  0.  0.  0.001; H  0.  -0.757  0.587; H  0.  0.757   0.587')
        self.assertAlmostEqual((e2-e1)/0.002*lib.param.BOHR, g[0,2], 5)

    @unittest.skipIf(dftd4 is None, "requires the dftd4 library")
    def test_rhf_d4_grad(self):
        mf = scf.RHF(mol)
        mf.disp = 'd4'
        g_scan = mf.nuc_grad_method().as_scanner()
        g = g_scan(mol)[1]

        mf_scan = mf.as_scanner()
        e1 = mf_scan('O  0.  0. -0.001; H  0.  -0.757  0.587; H  0.  0.757   0.587')
        e2 = mf_scan('O  0.  0.  0.001; H  0.  -0.757  0.587; H  0.  0.757   0.587')
        self.assertAlmostEqual((e2-e1)/0.002*lib.param.BOHR, g[0,2], 5)

    def test_x2c_rhf_grad(self):
        h2o = gto.Mole()
        h2o.verbose = 0
        h2o.atom = [
            ['O' , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ]
        h2o.basis = {'H': '631g',
                     'O': '631g',}
        h2o.symmetry = True
        h2o.build()
        mf = scf.RHF(h2o).x2c()
        mf.conv_tol = 1e-14
        e0 = mf.scf()
        g1 = mf.Gradients().grad()
#[[ 0   0               -2.40286232e-02]
# [ 0   4.27908498e-03   1.20143116e-02]
# [ 0  -4.27908498e-03   1.20143116e-02]]
        self.assertAlmostEqual(lib.fp(g1), 0.0056364301689991346, 6)

    def test_finite_diff_x2c_rhf_grad(self):
        mf = scf.RHF(mol).x2c()
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        g = grad.RHF(mf).kernel()
        self.assertAlmostEqual(lib.fp(g), 0.0056363502746766807, 6)

        mf_scanner = mf.as_scanner()
        pmol = mol.copy()
        e1 = mf_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

        #e1 = mf_scanner(pmol.set_geom_('O  0.  1e-5 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        #e2 = mf_scanner(pmol.set_geom_('O  0. -1e-5 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        #self.assertAlmostEqual(g[0,1], (e1-e2)/2e-5*lib.param.BOHR, 5)

        e1 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7571 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7569 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[1,1], (e2-e1)/2e-4*lib.param.BOHR, 5)

        #e1 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5871; 1  0. 0.757 0.587'))
        #e2 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5869; 1  0. 0.757 0.587'))
        #self.assertAlmostEqual(g[1,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    def test_finite_diff_rhf_grad(self):
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        g = grad.RHF(mf).kernel(atmlst=range(mol.natm))
        self.assertAlmostEqual(lib.fp(g), 0.0055115512502467556, 6)

        mf_scanner = mf.as_scanner()
        pmol = mol.copy()
        e1 = mf_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

        #e1 = mf_scanner(pmol.set_geom_('O  0.  1e-5 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        #e2 = mf_scanner(pmol.set_geom_('O  0. -1e-5 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        #self.assertAlmostEqual(g[0,1], (e1-e2)/2e-5*lib.param.BOHR, 5)

        #e1 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7571 0.587; 1  0. 0.757 0.587'))
        #e2 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7569 0.587; 1  0. 0.757 0.587'))
        #self.assertAlmostEqual(g[1,1], (e2-e1)/2e-4*lib.param.BOHR, 5)

        #e1 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5871; 1  0. 0.757 0.587'))
        #e2 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5869; 1  0. 0.757 0.587'))
        #self.assertAlmostEqual(g[1,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    def test_finite_diff_df_rhf_grad(self):
        mf = scf.RHF(mol).density_fit ()
        mf.conv_tol = 1e-14
        e0 = mf.kernel()
        g = mf.nuc_grad_method ().kernel(atmlst=range(mol.natm))
        self.assertAlmostEqual(lib.fp(g), 0.005516675903099752, 6)

        mf_scanner = mf.as_scanner()
        pmol = mol.copy()
        e1 = mf_scanner(pmol.set_geom_('O  0. 0. 0.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        e2 = mf_scanner(pmol.set_geom_('O  0. 0. -.0001; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        self.assertAlmostEqual(g[0,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

        #e1 = mf_scanner(pmol.set_geom_('O  0.  1e-5 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        #e2 = mf_scanner(pmol.set_geom_('O  0. -1e-5 0.; 1  0. -0.757 0.587; 1  0. 0.757 0.587'))
        #self.assertAlmostEqual(g[0,1], (e1-e2)/2e-5*lib.param.BOHR, 5)

        #e1 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7571 0.587; 1  0. 0.757 0.587'))
        #e2 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.7569 0.587; 1  0. 0.757 0.587'))
        #self.assertAlmostEqual(g[1,1], (e2-e1)/2e-4*lib.param.BOHR, 5)

        #e1 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5871; 1  0. 0.757 0.587'))
        #e2 = mf_scanner(pmol.set_geom_('O  0. 0. 0.; 1  0. -0.757 0.5869; 1  0. 0.757 0.587'))
        #self.assertAlmostEqual(g[1,2], (e1-e2)/2e-4*lib.param.BOHR, 5)

    def test_ecp_grad(self):
        mol = gto.M(atom='Cu 0 0 0; H 0 0 1.5', basis='lanl2dz',
                    ecp='lanl2dz', symmetry=True, verbose=0)
        mf = scf.RHF(mol)
        g_scan = mf.nuc_grad_method().as_scanner().as_scanner()
        g = g_scan(mol.atom)[1]
        self.assertAlmostEqual(lib.fp(g), -0.012310573162997052, 7)

        mfs = mf.as_scanner()
        e1 = mfs(mol.set_geom_('Cu 0 0 -0.001; H 0 0 1.5'))
        e2 = mfs(mol.set_geom_('Cu 0 0  0.001; H 0 0 1.5'))
        self.assertAlmostEqual(g[0,2], (e2-e1)/0.002*lib.param.BOHR, 6)

    def test_grad_with_symmetry(self):
        h2o = gto.Mole()
        h2o.verbose = 0
        h2o.atom = [
            ['O' , (0. , 0.     , 0.)],
            [1   , (0. , -0.757 , 0.587)],
            [1   , (0. , 0.757  , 0.587)] ]
        h2o.basis = {'H': '631g',
                     'O': '631g',}
        h2o.symmetry = True
        h2o.build()
        mf = scf.RHF(h2o)
        mf.conv_tol = 1e-14
        e0 = mf.scf()
        g1 = mf.Gradients().grad()
#[[ 0   0               -2.41134256e-02]
# [ 0   4.39690522e-03   1.20567128e-02]
# [ 0  -4.39690522e-03   1.20567128e-02]]
        self.assertAlmostEqual(lib.fp(g1), 0.0055115512502467556, 6)

        mol = gto.M (atom = 'H 0 0 0; H 0 0 1', basis='sto-3g', symmetry=True, verbose=0)
        ref = scf.RHF (mol).run ().nuc_grad_method ().kernel ()
        mol = gto.M (atom = 'H 0 0 0; H 1 0 0', basis='sto-3g', symmetry=True, verbose=0)
        g_x = scf.RHF (mol).run ().nuc_grad_method ().kernel ()
        self.assertAlmostEqual(abs(ref[:,2] - g_x[:,0]).max(), 0, 9)

    def test_grad_nuc(self):
        mol = gto.M(atom='He 0 0 0; He 0 1 2; H 1 2 1; H 1 0 0')
        gs = grad.rhf.grad_nuc(mol)
        ref = grad_nuc(mol)
        self.assertAlmostEqual(abs(gs - ref).max(), 0, 9)

    # issue #3198
    def test_gth_pp_grad(self):
        mol = gto.M(
            atom    = 'F 0 0 0; H 0 0 1.76',
            basis   = 'gth-dzv',
            pseudo  = 'gth-pbe',
            unit    = 'B',)
        mf = mol.RKS(xc='pbe').run()
        g  = mf.nuc_grad_method().kernel()

        mf_scan = mf.as_scanner()
        e1 = mf_scan('F 0 0 0; H 0 0 1.761')
        e2 = mf_scan('F 0 0 0; H 0 0 1.759')
        self.assertAlmostEqual(g[1, 2], (e1 - e2)/2e-3, delta=3e-6)

    def test_gth_pp_hcore_grad(self):
        from pyscf.df import incore
        from pyscf.gto import pp_int
        from pyscf.pbc.gto.pseudo.pp_int import fake_cell_vloc
        mol = gto.M(
            atom    = 'Al 0 0 0; Al 0 0 1.9',
            basis   = 'gth-dzvp',
            pseudo  = 'gth-pbe',
            unit    = 'B',)
        numpy.random.seed(1)
        nao = mol.nao
        dm = numpy.random.rand(nao,nao) - .5
        dm = dm.dot(dm.T)
        de = pp_int.vpploc_nuc_grad(mol, dm)
        de += pp_int.vppnl_nuc_grad(mol, dm)

        p0, p1 = mol.aoslice_by_atom()[1,2:]
        fakemol = fake_cell_vloc(mol, 0)
        Z = fakemol.atom_charges()
        v = incore.aux_e2(mol, fakemol, 'int3c2e_ip1', comp=3)
        v1 = numpy.einsum('xpqi,i->xpq', v, Z)
        vrinv = v[:,:,:,1] * -Z[1]
        vrinv[:,p0:p1] += v1[:,p0:p1]
        vrinv = vrinv + vrinv.transpose(0,2,1)
        de[1] += numpy.einsum('xpq,pq->x', vrinv, dm)

        mol = mol.set_geom_('Al 0 0 0; Al 0 0 1.901')
        e1 = numpy.einsum('ij,ij', dm, pp_int.get_gth_pp(mol))
        mol = mol.set_geom_('Al 0 0 0; Al 0 0 1.899')
        e2 = numpy.einsum('ij,ij', dm, pp_int.get_gth_pp(mol))
        self.assertAlmostEqual(de[1, 2], (e1 - e2) / 2e-3, 6)

    def test_jk_energy_per_atom(self):
        mol = gto.M(atom='''O 0 0 0.12; H 0 0.76 -0.47; H 0 -0.76 -0.47; H 1.3 0 1.2''',
                    basis={'O': 'cc-pvdz', 'H': '6-31g*'}, charge=1, verbose=0)
        nao = mol.nao
        aoslices = mol.aoslice_by_atom()
        numpy.random.seed(2)
        a = numpy.random.rand(nao, nao) - .5
        dm = a.dot(a.T) * .2
        b = numpy.random.rand(nao, nao) - .5
        dms = numpy.asarray([dm, b.dot(b.T) * .1])

        def contract(v, d):
            de = numpy.zeros((mol.natm, 3))
            v = v.reshape(-1, 3, nao, nao)
            d = d.reshape(-1, nao, nao)
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia,2:]
                de[ia] = numpy.einsum('sxij,sij->x', v[:,:,p0:p1], d[:,p0:p1]) * 2
            return de

        vj, vk = grad.rhf.get_jk(mol, dm)
        ref = contract(vj * .7 - vk * .5 * .3, dm)
        self.assertAlmostEqual(abs(grad.rhf.jk_energy_per_atom(mol, dm, .7, .3) - ref).max(), 0, 12)
        ref = contract(vj, dm)
        self.assertAlmostEqual(abs(grad.rhf.jk_energy_per_atom(mol, dm, 1, 0) - ref).max(), 0, 12)
        vj, vk = grad.rhf.get_jk(mol, dms)
        ref = contract(vj[0] + vj[1] - vk, dms)
        self.assertAlmostEqual(abs(grad.rhf.jk_energy_per_atom(mol, dms) - ref).max(), 0, 12)
        with mol.with_range_coulomb(.4):
            vk = grad.rhf.get_k(mol, dms)
        ref = contract(-vk * .6, dms)
        self.assertAlmostEqual(abs(grad.rhf.jk_energy_per_atom(mol, dms, 0, .6, omega=.4) - ref).max(),
                               0, 12)
        with mol.with_range_coulomb(-.4):
            vk = grad.rhf.get_k(mol, dm)
        ref = contract(-vk * .5, dm)
        self.assertAlmostEqual(abs(grad.rhf.jk_energy_per_atom(mol, dm, 0, 1, omega=-.4) - ref).max(),
                               0, 12)

    def test_jk_energy_per_atom_int2e_batch(self):
        # qcint's int2e_ip1_sph_batch (several orbits in the SIMD lanes), contracted in its _dot kernels
        # and in grad_jk_energy.c, against int2e_ip1_sph orbit by orbit, with d and f shells; with
        # libcint, which has none, all calls take the same path
        from pyscf.scf import _vhf
        mol = gto.M(atom='O 0 0 0.12; H 0 0.76 -0.47; H 0 -0.76 -0.47', basis='def2-tzvp', verbose=0)
        nao = mol.nao
        numpy.random.seed(3)
        a = numpy.random.rand(nao, nao) - .5
        dm = a.dot(a.T) * .2
        b = numpy.random.rand(nao, nao) - .5
        dms = numpy.asarray([dm, b.dot(b.T) * .1])
        for d, jf, kf, omega in ((dm, .7, .3, None), (dm, 1, 0, None), (dms, 1, 1, None),
                                 (dms, 0, .6, .4)):
            with lib.temporary_env(_vhf, INT2E_BATCH=False):
                ref = grad.rhf.jk_energy_per_atom(mol, d, jf, kf, omega=omega)
            for dot in (True, False):
                with lib.temporary_env(_vhf, INT2E_BATCH=True), \
                        lib.temporary_env(grad.rhf, JK_ENERGY_DOT=dot):
                    de = grad.rhf.jk_energy_per_atom(mol, d, jf, kf, omega=omega)
                self.assertAlmostEqual(abs(de - ref).max(), 0, 12)

    def test_grad_elec_jk_energy(self):
        from pyscf import dft
        mol = gto.M(atom='''O 0 0 0.12; H 0 0.76 -0.47; H 0 -0.76 -0.47''',
                    basis='6-31g*', verbose=0)
        molp = mol.copy()
        molp.charge = 1
        molp.spin = 1
        molp.build()
        for mf in (scf.RHF(mol), scf.UHF(molp), scf.ROHF(molp), dft.RKS(mol, xc='PBE'),
                   dft.RKS(mol, xc='CAM-B3LYP'), dft.UKS(molp, xc='wB97X'),
                   dft.ROKS(molp, xc='B3LYP')):
            mf.conv_tol = 1e-10
            g = mf.run().nuc_grad_method()
            self.assertIsNotNone(grad.rhf._VEFF_SPLIT.get(g.get_veff.__func__))
            with lib.temporary_env(grad.rhf, JK_ENERGY=False):
                ref = g.kernel()
            self.assertAlmostEqual(abs(g.kernel() - ref).max(), 0, 12)
            self.assertAlmostEqual(abs(g.kernel(atmlst=[2, 0]) - ref[[2, 0]]).max(), 0, 12)

def grad_nuc(mol):
    gs = numpy.zeros((mol.natm,3))
    for j in range(mol.natm):
        q2 = mol.atom_charge(j)
        r2 = mol.atom_coord(j)
        for i in range(mol.natm):
            if i != j:
                q1 = mol.atom_charge(i)
                r1 = mol.atom_coord(i)
                r = numpy.sqrt(numpy.dot(r1-r2,r1-r2))
                gs[j] -= q1 * q2 * (r2-r1) / r**3
    return gs

if __name__ == "__main__":
    print("Full Tests for RHF Gradients")
    unittest.main()
