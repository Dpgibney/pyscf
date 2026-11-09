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


if __name__ == '__main__':
    print('Full Tests for the memory of the CPHF response')
    unittest.main()
