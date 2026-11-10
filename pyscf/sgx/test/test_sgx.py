# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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

import unittest
from pyscf import gto, lib
from pyscf import scf, dft
from pyscf.sgx import sgx


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_grids = dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = False

    @classmethod
    def tearDownClass(cls):
        dft.radi.ATOM_SPECIFIC_TREUTLER_GRIDS = cls.original_grids

    def test_reset(self):
        mol = gto.M(atom='He')
        mol1 = gto.M(atom='C')
        mf = scf.RHF(mol).COSX()
        mf.reset(mol1)
        self.assertTrue(mf.mol is mol1)
        self.assertTrue(mf.with_df.mol is mol1)
        self.assertEqual(mf.undo_sgx().__class__.__name__, 'RHF')

    def test_sgx_scf(self):
        mol = gto.Mole()
        mol.build(
            atom = [["O" , (0. , 0.     , 0.)],
                    [1   , (0. , -0.757 , 0.587)],
                    [1   , (0. , 0.757  , 0.587)] ],
            basis='ccpvdz',
            verbose=0,
        )
        mf = sgx.sgx_fit(scf.RHF(mol), 'weigend')
        mf.with_df.dfj = True
        mf.with_df.use_opt_grids = False
        mf.with_df.grids_level_f = 1
        mf.with_df._symm_ovlp_fit = False
        energy = mf.kernel()
        self.assertAlmostEqual(energy, -76.02686422219752, 9)

        mf = sgx.sgx_fit(scf.RHF(mol))
        mf.with_df.use_opt_grids = False
        mf.with_df.grids_level_f = 1
        mf.with_df.optk = False
        mf.with_df.dfj = False
        energy = mf.kernel()
        self.assertAlmostEqual(energy, -76.02673747035047, 8)

    def test_sgx_pjs(self):
        mol = gto.Mole()
        atom0 = [["O" , (0. , 0.     , 0.)],
                 [1   , (0. , -0.757 , 0.587)],
                 [1   , (0. , 0.757  , 0.587)],]
        atom = []
        for i in range(3):
            atom = atom + [[z, (c[0] + i * 5, c[1], c[2])] for z, c in atom0]
        mol.build(
            atom=atom,
            basis='sto-3g',
            verbose=0,
        )
        mf = sgx.sgx_fit(mol.RHF())
        mf.with_df.dfj = True
        mf.with_df.use_opt_grids = False
        mf.with_df.grids_level_f = 1
        mf.with_df.sgx_tol_energy = None
        mf.with_df.sgx_tol_potential = None
        energy = mf.kernel()
        mf = sgx.sgx_fit(mol.RHF())
        mf.with_df.dfj = True
        mf.with_df.use_opt_grids = False
        mf.with_df.grids_level_f = 1
        mf.with_df.sgx_tol_energy = None
        mf.with_df.sgx_tol_potential = None
        mf.with_df.sgx_tol_energy = 1e-9
        mf.with_df.sgx_tol_potential = "auto"
        energy2 = mf.kernel()
        self.assertAlmostEqual(energy, energy2, 9)

    def test_sgx_tol_pair(self):
        # the screening of shell pairs by batch (sgx_tol_pair): the energy of the unscreened K
        # within about the threshold, in about as many cycles
        mol = gto.M(atom='''O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587
                            O 3 0 0; H 3 -0.757 0.587; H 3 0.757 0.587''',
                    basis='def2-svp', verbose=0)
        mf0 = sgx.sgx_fit(dft.RKS(mol, xc='b3lyp'), pjs=True)
        mf0.with_df.sgx_tol_energy = None
        mf0.with_df.sgx_tol_potential = None
        e0 = mf0.kernel()
        mf1 = sgx.sgx_fit(dft.RKS(mol, xc='b3lyp'), pjs=True)
        mf1.with_df.sgx_tol_energy = None
        mf1.with_df.sgx_tol_potential = None
        mf1.with_df.sgx_tol_pair = 1e-8
        e1 = mf1.kernel()
        self.assertAlmostEqual(e1, e0, 7)
        self.assertLessEqual(mf1.cycles, mf0.cycles + 2)

    def test_rijcosx(self):
        mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='def2-svp', verbose=0)
        mf = dft.RKS(mol, xc='b3lyp').RIJCOSX()
        self.assertEqual(mf.with_df.auxbasis, 'def2-universal-jfit')
        self.assertTrue(mf.with_df.dfj and mf.with_df.optk)
        self.assertEqual((mf.with_df.grids_level_i, mf.with_df.grids_level_f), (0, 1))
        self.assertEqual(mf.with_df.sgx_tol_pair, 1e-9)
        e = mf.kernel()
        # the energy of the converged density with K on level 2, the SCF's grid kept
        self.assertEqual(mf.with_df.grids_level_f, 1)
        dm = mf.make_rdm1()
        mf.with_df.build(level=2)
        mf._nsteps_direct = 0
        e2 = mf.energy_tot(dm, vhf=mf.get_veff(mol, dm))
        self.assertAlmostEqual(e, e2, 9)
        mf.with_df.grids_level_energy = None
        mf.with_df.build(level=0)
        e1 = mf.kernel()
        self.assertTrue(abs(e - e1) > 1e-7)
        # the exact SCF gives -76.3581494902
        self.assertAlmostEqual(e, -76.3582451667, 7)

    def test_rijcosx_energy_k_only(self):
        # the energy on the larger grid from the exchange alone (no XC, VV10 or J again)
        # against the whole get_veff on that grid
        from pyscf.dft import numint
        mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='def2-svp', verbose=0)
        mol1 = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='def2-svp',
                     charge=1, spin=1, verbose=0)
        for make in (lambda: dft.RKS(mol, xc='b3lyp'), lambda: dft.RKS(mol, xc='wb97m-v'),
                     lambda: dft.RKS(mol, xc='hse06'), lambda: dft.RKS(mol, xc='lrc-wpbe'),
                     lambda: dft.UKS(mol1, xc='wb97x'), lambda: dft.ROKS(mol1, xc='b3lyp')):
            mf = make().RIJCOSX()
            mf.conv_tol = 1e-10
            mf._energy_on_grid_k_only = False
            e_full = mf.kernel()
            mf = make().RIJCOSX()
            mf.conv_tol = 1e-10
            calls = []
            nr_rks, nr_uks = numint.NumInt.nr_rks, numint.NumInt.nr_uks
            def spy(fn):
                def f(*args, **kwargs):
                    calls.append(mf._sgx_last is not None)
                    return fn(*args, **kwargs)
                return f
            with lib.temporary_env(numint.NumInt, nr_rks=spy(nr_rks), nr_uks=spy(nr_uks)):
                e = mf.kernel()
            self.assertAlmostEqual(e, e_full, 10)
            # no XC integration after the SCF has ended
            self.assertTrue(len(calls) > 0 and not any(calls))
            self.assertEqual(mf.with_df.grids_level_f, 1)
        # a functional without exact exchange has nothing on the SGX grid
        mf = dft.RKS(mol, xc='pbe').RIJCOSX()
        e = mf.kernel()
        mf.with_df.grids_level_energy = None
        self.assertAlmostEqual(e, mf.kernel(), 10)

    def test_mixed_range(self):
        # hyb K + (alpha - hyb) K_LR of a range-separated hybrid from one evaluation of the
        # mixed-range integrals (qcint) against the two K builds
        from pyscf.sgx import sgx_jk
        if not sgx_jk.mixed_range_supported():
            self.skipTest('int1e_grids of this integral library has no mixed-range operator')
        mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='def2-svp', verbose=0)
        mol1 = gto.M(atom=mol.atom, basis='def2-svp', charge=1, spin=1, verbose=0)
        mf = dft.RKS(mol, xc='wb97x').RIJCOSX()
        sdf = mf.with_df
        sdf.sgx_tol_pair = None
        sdf.build(level=1)
        self.assertTrue(sdf.mixed_range_ok())
        dm = mf.get_init_guess()
        vk = sdf.get_jk(dm, 1, None, False, True)[1]
        vklr = sdf.get_jk(dm, 1, None, False, True, omega=.3)[1]
        vj, vkm = sdf.get_jk(dm, 1, None, True, True, omega=.3, mix=(.2, .7))
        self.assertAlmostEqual(abs(vkm - (.2 * vk + .7 * vklr)).max(), 0, 11)
        self.assertAlmostEqual(abs(vj - sdf.get_jk(dm, 1, None, True, False)[0]).max(), 0, 12)
        # the range parameter and the coefficients are back to those of the molecule
        self.assertEqual(mol.omega, 0)
        self.assertEqual(abs(mol._env[[sgx_jk.PTR_RANGE_COEFF_FULL, sgx_jk.PTR_RANGE_COEFF_LR]]).max(), 0)
        for make in (lambda: dft.RKS(mol, xc='wb97x'), lambda: dft.RKS(mol, xc='cam-b3lyp'),
                     lambda: dft.UKS(mol1, xc='wb97x')):
            with lib.temporary_env(sgx_jk, MIXED_RANGE=False):
                mf0 = make().RIJCOSX()
                mf0.conv_tol = 1e-11
                e0 = mf0.kernel()
                g0 = mf0.nuc_grad_method().kernel()
            mf1 = make().RIJCOSX()
            mf1.conv_tol = 1e-11
            kcalls = []
            get_k_only = sgx_jk.get_k_only
            def spy(sgx, *args, **kwargs):
                kcalls.append(sgx.mol.omega)
                return get_k_only(sgx, *args, **kwargs)
            with lib.temporary_env(sgx_jk, get_k_only=spy):
                e1 = mf1.kernel()
            # every K build is the mixed-range one
            self.assertTrue(len(kcalls) > 0 and all(w == .3 or abs(w - .33) < 1e-12 for w in kcalls))
            # one per Fock build (the first, each cycle, the last) and the energy on the larger grid
            self.assertEqual(len(kcalls), mf1.cycles + 3)
            self.assertAlmostEqual(e1, e0, 8)
            self.assertAlmostEqual(abs(mf1.nuc_grad_method().kernel() - g0).max(), 0, 6)
        # J on the SGX grid keeps the two passes
        mf = sgx.sgx_fit(dft.RKS(mol, xc='wb97x'))
        self.assertEqual(mf.get_jk_mixed_range(mol, dm, 1, .3, .2, .7), (None, None))

    def test_rijcosx_grad(self):
        # the step on the larger grid leaves the SCF's grid and data to the gradients
        mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='def2-svp', verbose=0)
        mf = dft.RKS(mol, xc='b3lyp').RIJCOSX()
        mf.kernel()
        g = mf.nuc_grad_method().kernel()
        mf1 = dft.RKS(mol, xc='b3lyp').RIJCOSX()
        mf1.with_df.grids_level_energy = None
        mf1.kernel()
        g1 = mf1.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(g - g1).max(), 0, 8)

    def test_rijcosx_rsh(self):
        # a range-separated hybrid: the long-range K on the RSH copy of the SGX object, with the
        # build's state (full density, pair threshold), and no long-range J
        from pyscf.df import df_jk
        mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='def2-svp', verbose=0)
        omegas = []
        get_j = df_jk.get_j
        def get_j_spy(dfobj, dm, *args, **kwargs):
            omegas.append(dfobj.mol.omega)
            return get_j(dfobj, dm, *args, **kwargs)
        from pyscf.sgx import sgx_jk
        # the two K builds of a range-separated hybrid (the mixed-range operator, which
        # replaces them where the integral library has it, is in test_mixed_range)
        mixed_range = sgx_jk.MIXED_RANGE
        sgx_jk.MIXED_RANGE = False
        self.addCleanup(setattr, sgx_jk, 'MIXED_RANGE', mixed_range)
        with lib.temporary_env(df_jk, get_j=get_j_spy):
            mf = dft.RKS(mol, xc='wb97x').RIJCOSX()
            e = mf.kernel()
        self.assertTrue(mf.converged)
        self.assertTrue(len(omegas) > 0 and all(w == 0 for w in omegas))
        # the long-range copy takes the full-range object's grid and the grid's data
        sdf = mf.with_df
        (rsh,) = sdf._rsh_df.values()
        self.assertTrue(rsh.grids is sdf.grids)
        self.assertTrue(rsh._pjs_data._share is sdf._pjs_data)
        self.assertTrue(rsh._pjs_data._overlap_correction_matrix
                        is sdf._pjs_data._overlap_correction_matrix)
        self.assertTrue(getattr(rsh._pjs_data, '_ao_on_grids', None) is None)
        self.assertTrue(getattr(sdf._pjs_data, '_ao_on_grids', None) is not None)
        mf1 = dft.RKS(mol, xc='wb97x').RIJCOSX()
        mf1.with_df.sgx_tol_pair = None
        e1 = mf1.kernel()
        self.assertAlmostEqual(e, e1, 7)

    def test_sgx_jk_blocks(self):
        # SGX J and K in several blocks of the grid (a small max_memory) as in one
        mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='ccpvdz', verbose=0)
        mf = sgx.sgx_fit(scf.RHF(mol))
        mf.with_df.grids_level_f = 1
        e0 = mf.kernel()
        mf = sgx.sgx_fit(scf.RHF(mol))
        mf.with_df.grids_level_f = 1
        mf.with_df.max_memory = 1
        e1 = mf.kernel()
        self.assertAlmostEqual(e1, e0, 10)

if __name__ == "__main__":
    print("Full Tests for SGX")
    unittest.main()
