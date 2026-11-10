# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import gto
from pyscf import scf
from pyscf import df
from pyscf.df import df_jk

def setUpModule():
    global mol, symol
    mol = gto.M(
        verbose = 5,
        output = '/dev/null',
        atom = '''
            O     0    0        0
            H     0    -0.757   0.587
            H     0    0.757    0.587''',
        basis = 'cc-pvdz',
    )
    symol = gto.M(
        verbose = 5,
        output = '/dev/null',
        atom = '''
            O     0    0        0
            H     0    -0.757   0.587
            H     0    0.757    0.587''',
        basis = 'cc-pvdz',
        symmetry = 1,
    )

def tearDownModule():
    global mol, symol
    mol.stdout.close()
    symol.stdout.close()
    del mol, symol


class KnownValues(unittest.TestCase):
    def test_rhf(self):
        mf = scf.density_fit(scf.RHF(mol), auxbasis='weigend')
        self.assertAlmostEqual(mf.scf(), -76.025936299702536, 8)
        self.assertTrue(mf._eri is None)

    def test_uhf(self):
        mf = scf.density_fit(scf.UHF(mol), auxbasis='weigend')
        self.assertAlmostEqual(mf.scf(), -76.025936299702536, 8)

    def test_uhf_cart(self):
        pmol = mol.copy()
        pmol.cart = True
        mf = scf.density_fit(scf.UHF(pmol), auxbasis='weigend')
        self.assertAlmostEqual(mf.scf(), -76.026760700636046, 8)

    def test_rohf(self):
        pmol = mol.copy()
        pmol.charge = 1
        pmol.spin = 1
        pmol.build(False, False)
        mf = scf.density_fit(scf.ROHF(pmol), auxbasis='weigend')
        self.assertAlmostEqual(mf.scf(), -75.626515724371814, 8)

    def test_dhf(self):
        pmol = mol.copy()
        pmol.build(False, False)
        mf = scf.density_fit(scf.DHF(pmol), auxbasis='weigend')
        mf.conv_tol = 1e-10
        self.assertAlmostEqual(mf.scf(), -76.080738677021458, 8)

    def test_rhf_symm(self):
        mf = scf.density_fit(scf.RHF(symol), auxbasis='weigend')
        self.assertAlmostEqual(mf.scf(), -76.025936299702536, 8)

    def test_uhf_symm(self):
        mf = scf.density_fit(scf.UHF(symol), auxbasis='weigend')
        self.assertAlmostEqual(mf.scf(), -76.025936299702536, 8)

    def test_rohf_symm(self):
        pmol = mol.copy()
        pmol.charge = 1
        pmol.spin = 1
        pmol.symmetry = 1
        pmol.build(False, False)
        mf = scf.density_fit(scf.ROHF(pmol), auxbasis='weigend')
        self.assertAlmostEqual(mf.scf(), -75.626515724371814, 8)

    def test_rhf_veff(self):
        nao = mol.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((2,nao,nao))
        mf = scf.density_fit(scf.RHF(mol), auxbasis='weigend')
        vhf1 = mf.get_veff(mol, dm, hermi=0)
        naux = mf._cderi.shape[0]
        cderi = numpy.empty((naux,nao,nao))
        for i in range(naux):
            cderi[i] = lib.unpack_tril(mf._cderi[i])
        vj0 = []
        vk0 = []
        for dmi in dm:
            v1 = numpy.einsum('kij,ij->k', cderi, dmi)
            vj0.append(numpy.einsum('kij,k->ij', cderi, v1))
            v1 = numpy.einsum('pij,jk->pki', cderi, dmi.T)
            vk0.append(numpy.einsum('pki,pkj->ij', cderi, v1))
        vj1, vk1 = df_jk.get_jk(mf.with_df, dm, 0)
        self.assertTrue(numpy.allclose(vj0, vj1))
        self.assertTrue(numpy.allclose(numpy.array(vk0), vk1))
        vhf0 = vj1 - vk1 * .5
        self.assertTrue(numpy.allclose(vhf0, vhf1))

    def test_uhf_veff(self):
        mf = scf.density_fit(scf.UHF(mol), auxbasis='weigend')
        nao = mol.nao_nr()
        numpy.random.seed(1)
        dm = numpy.random.random((2,4,nao,nao))
        vhf = mf.get_veff(mol, dm, hermi=0)
        self.assertAlmostEqual(numpy.linalg.norm(vhf), 413.82341595365853, 9)

    def test_assign_cderi(self):
        nao = mol.nao_nr()
        w, u = scipy.linalg.eigh(mol.intor('int2e_sph', aosym='s4'))
        idx = w > 1e-9

        mf = scf.density_fit(scf.UHF(mol), auxbasis='weigend')
        mf._cderi = (u[:,idx] * numpy.sqrt(w[idx])).T.copy()
        self.assertAlmostEqual(mf.kernel(), -76.026765673110447, 8)

    def test_nr_get_jk(self):
        numpy.random.seed(1)
        mf = scf.RHF(mol).density_fit(auxbasis='weigend')
        nao = mol.nao_nr()
        dms = numpy.random.random((2,nao,nao))

        vj, vk = mf.get_jk(mol, dms, hermi=0)
        self.assertAlmostEqual(lib.fp(vj), -194.15910890730066, 9)
        self.assertAlmostEqual(lib.fp(vk), -46.365071587653517, 9)
        vj = mf.get_j(mol, dms, hermi=0)
        self.assertAlmostEqual(lib.fp(vj), -194.15910890730066, 9)
        vk = mf.get_k(mol, dms, hermi=0)
        self.assertAlmostEqual(lib.fp(vk), -46.365071587653517, 9)

        mf.with_df = None
        vj, vk = mf.get_jk(mol, dms, hermi=0)
        self.assertAlmostEqual(lib.fp(vj), -194.08878302990749, 9)
        self.assertAlmostEqual(lib.fp(vk), -46.530782983591152, 9)
        vj = mf.get_j(mol, dms, hermi=0)
        self.assertAlmostEqual(lib.fp(vj), -194.08878302990749, 9)
        vk = mf.get_k(mol, dms, hermi=0)
        self.assertAlmostEqual(lib.fp(vk), -46.530782983591152, 9)

    def test_r_get_jk(self):
        numpy.random.seed(1)
        dfobj = df.df.DF4C(mol)
        n2c = mol.nao_2c()
        n4c = n2c * 2
        dms = numpy.random.random((2,n4c,n4c))
        vj, vk = dfobj.get_jk(dms, hermi=0)
        self.assertAlmostEqual(lib.fp(vj), 12.961687328405461+55.686811159338134j, 9)
        self.assertAlmostEqual(lib.fp(vk), 41.984238099875462+12.870888901217896j, 9)

    def test_df_jk_density_fit(self):
        mf = scf.RHF(mol).density_fit()
        mf.with_df = None
        mf = mf.density_fit()
        self.assertTrue(mf.with_df is not None)

        mf = mf.newton().density_fit(auxbasis='sto3g')
        self.assertEqual(mf.with_df.auxbasis, 'sto3g')

    def test_get_j(self):
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dms = numpy.random.random((2,nao,nao))

        mf = scf.RHF(mol).density_fit(auxbasis='weigend')
        vj0 = mf.get_j(mol, dms)
        vj1 = mf.get_jk(mol, dms)[0]
        self.assertAlmostEqual(abs(vj0-vj1).max(), 0, 12)
        self.assertAlmostEqual(lib.fp(vj0), -194.15910890730052, 9)

    def test_get_j_engine(self):
        # the J-engine (rij_engine.c) against the RI-J through int3c2e: general contractions
        # (cc-pVDZ), g functions (def2-QZVP), densities that are not symmetric
        numpy.random.seed(1)
        for basis, auxbasis in [('cc-pvdz', 'weigend'), ('def2-qzvp', 'def2-universal-jfit')]:
            m = gto.M(atom=mol.atom, basis=basis, verbose=0)
            dms = numpy.random.random((2, m.nao, m.nao))
            with_df = df.DF(m, auxbasis=auxbasis)
            with lib.temporary_env(df_jk, RIJ_ENGINE=False):
                vj0 = df_jk.get_j(with_df, dms)
            with_df = df.DF(m, auxbasis=auxbasis)
            vj1 = df_jk.get_j(with_df, dms)
            self.assertTrue(getattr(with_df, '_rij_engine', None) is not None)
            self.assertAlmostEqual(abs(vj1-vj0).max(), 0, 11)

    def test_get_j_engine_many(self):
        # the densities of a response together in each pass of the J-engine (RIJdrv_many)
        # against one after the other, in chunks of 2 with a last chunk of one, and for
        # densities of very different sizes, whose screening the chunk shares
        numpy.random.seed(2)
        for basis, auxbasis in [('cc-pvdz', 'weigend'), ('def2-qzvp', 'def2-universal-jfit')]:
            m = gto.M(atom=mol.atom, basis=basis, verbose=0)
            dms = numpy.random.random((7, m.nao, m.nao)) - .5
            dms = dms + dms.transpose(0,2,1)
            dms *= numpy.array([1, 1e-2, 1e-4, 1, 3, 1e-6, 1])[:,None,None]
            with_df = df.DF(m, auxbasis=auxbasis)
            with lib.temporary_env(df_jk, RIJ_MANY_MIN=100):
                vj0 = df_jk.get_j(with_df, dms)
            calls = []
            drv_many = df_jk._RIJEngineData.drv_many
            def spy(self, npass, out, *args):
                calls.append((npass, len(out)))
                return drv_many(self, npass, out, *args)
            with lib.temporary_env(df_jk._RIJEngineData, drv_many=spy):
                vj1 = df_jk.get_j(with_df, dms)
                self.assertEqual(calls, [(1, 7), (2, 7)])
                del calls[:]
                with lib.temporary_env(df_jk, RIJ_MANY_MEMORY=0):
                    vj2 = df_jk.get_j(with_df, dms)
                self.assertEqual(calls, [(1, 2), (2, 2)] * 3 + [(1, 1), (2, 1)])
            for k in range(7):
                scale = abs(vj0[k]).max()
                self.assertAlmostEqual(abs(vj1[k]-vj0[k]).max() / scale, 0, 9)
                self.assertAlmostEqual(abs(vj2[k]-vj0[k]).max() / scale, 0, 9)
            self.assertAlmostEqual(abs(vj1[0]-vj0[0]).max(), 0, 11)
            # a single density keeps the one-density driver
            with lib.temporary_env(df_jk._RIJEngineData, drv_many=None):
                self.assertAlmostEqual(abs(df_jk.get_j(with_df, dms[0])-vj0[0]).max(), 0, 12)

    def test_get_j_cderi(self):
        # J only: the 3-center tensor is built when it fits in memory, otherwise
        # get_j evaluates the integrals. Both fit the same way.
        numpy.random.seed(1)
        nao = mol.nao_nr()
        dms = numpy.random.random((2,nao,nao))
        with_df = df.DF(mol, auxbasis='weigend')
        vj0 = df_jk.get_j(with_df, dms)
        self.assertTrue(with_df._cderi is None)
        vj1 = df_jk.get_jk(with_df, dms, with_k=False)[0]
        self.assertTrue(with_df._cderi is not None)
        self.assertAlmostEqual(abs(vj1-vj0).max(), 0, 11)

        with_df = df.DF(mol, auxbasis='weigend')
        with_df.max_memory = 0
        vj2 = df_jk.get_jk(with_df, dms, with_k=False)[0]
        self.assertTrue(with_df._cderi is None)
        self.assertAlmostEqual(abs(vj2-vj0).max(), 0, 12)

    def test_df_jk_complex_dm(self):
        mol = gto.M(atom='H 0 0 0; H 0 0 1')
        mf = mol.RHF().run()
        dm = mf.make_rdm1() + 0j
        dm[0,:] += .1j
        dm[:,0] -= .1j
        mf.kernel(dm)
        self.assertTrue(mf.mo_coeff.dtype == numpy.complex128)
        dfmf = mf.density_fit()
        self.assertAlmostEqual(dfmf.energy_tot(), -1.0661355663696201, 9)
        self.assertAlmostEqual(dfmf.energy_tot(), mf.e_tot, 3)


if __name__ == "__main__":
    print("Full Tests for df")
    unittest.main()
