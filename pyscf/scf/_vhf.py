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

import sys
import ctypes
import numpy
from pyscf import lib
from pyscf import gto
from pyscf.gto.moleintor import make_cintopt, make_loc, ascint3

libcvhf = lib.load_library('libcvhf')

def _fpointer(name):
    return ctypes.cast(getattr(libcvhf, name), ctypes.c_void_p)

class VHFOpt:
    def __init__(self, mol, intor=None,
                 prescreen='CVHFnoscreen', qcondname=None, dmcondname=None):
        '''If function "qcondname" is presented, the qcond (sqrt(integrals))
        and will be initialized in __init__.

        prescreen, qcondname, dmcondname can be either function pointers or
        names of C functions defined in libcvhf module
        '''
        self._this = ctypes.POINTER(_CVHFOpt)()

        if intor is None:
            self._intor = intor
            self._cintopt = lib.c_null_ptr()
        else:
            self._intor = mol._add_suffix(intor)
            self._cintopt = make_cintopt(mol._atm, mol._bas, mol._env, intor)

        self._dmcondname = dmcondname
        self._qcondname = qcondname
        natm = ctypes.c_int(mol.natm)
        nbas = ctypes.c_int(mol.nbas)
        libcvhf.CVHFinit_optimizer(ctypes.byref(self._this),
                                   mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                                   mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                                   mol._env.ctypes.data_as(ctypes.c_void_p))
        self.prescreen = prescreen
        if qcondname is not None and intor is not None:
            self.init_cvhf_direct(mol, intor, qcondname)

    def init_cvhf_direct(self, mol, intor, qcondname):
        '''qcondname can be the function pointer or the name of a C function
        defined in libcvhf module
        '''
        intor = mol._add_suffix(intor)
        if intor == self._intor:
            cintopt = self._cintopt
        else:
            cintopt = lib.c_null_ptr()
        ao_loc = make_loc(mol._bas, intor)
        if isinstance(qcondname, ctypes._CFuncPtr):
            fsetqcond = qcondname
        else:
            fsetqcond = getattr(libcvhf, qcondname)
        natm = ctypes.c_int(mol.natm)
        nbas = ctypes.c_int(mol.nbas)
        fsetqcond(self._this, getattr(libcvhf, intor), cintopt,
                  ao_loc.ctypes.data_as(ctypes.c_void_p),
                  mol._atm.ctypes.data_as(ctypes.c_void_p), natm,
                  mol._bas.ctypes.data_as(ctypes.c_void_p), nbas,
                  mol._env.ctypes.data_as(ctypes.c_void_p))

    @property
    def direct_scf_tol(self):
        return self._this.contents.direct_scf_tol
    @direct_scf_tol.setter
    def direct_scf_tol(self, v):
        self._this.contents.direct_scf_tol = v

    @property
    def prescreen(self):
        return self._this.contents.fprescreen
    @prescreen.setter
    def prescreen(self, v):
        if isinstance(v, str):
            v = _fpointer(v)
        self._this.contents.fprescreen = v

    def set_dm(self, dm, atm, bas, env):
        if self._dmcondname is not None:
            c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
            c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
            c_env = numpy.asarray(env, dtype=numpy.double, order='C')
            natm = ctypes.c_int(c_atm.shape[0])
            nbas = ctypes.c_int(c_bas.shape[0])
            if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
                n_dm = 1
            else:
                n_dm = len(dm)
            dm = numpy.asarray(dm, order='C')
            ao_loc = make_loc(c_bas, self._intor)
            if isinstance(self._dmcondname, ctypes._CFuncPtr):
                fsetdm = self._dmcondname
            else:
                fsetdm = getattr(libcvhf, self._dmcondname)
            fsetdm(self._this,
                   dm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n_dm),
                   ao_loc.ctypes.data_as(ctypes.c_void_p),
                   c_atm.ctypes.data_as(ctypes.c_void_p), natm,
                   c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
                   c_env.ctypes.data_as(ctypes.c_void_p))

    def __del__(self):
        try:
            libcvhf.CVHFdel_optimizer(ctypes.byref(self._this))
        except AttributeError:
            pass

    def get_q_cond(self, shape=None):
        '''Return an array associated to q_cond. Contents of q_cond can be
        modified through this array
        '''
        if shape is None:
            nbas = self._this.contents.nbas
            shape = (nbas, nbas)
        data = ctypes.cast(self._this.contents.q_cond,
                           ctypes.POINTER(ctypes.c_double))
        return numpy.ctypeslib.as_array(data, shape=shape)
    q_cond = property(get_q_cond)

    def get_dm_cond(self, shape=None):
        '''Return an array associated to dm_cond. Contents of dm_cond can be
        modified through this array
        '''
        if shape is None:
            nbas = self._this.contents.nbas
            shape = (nbas, nbas)
        data = ctypes.cast(self._this.contents.dm_cond,
                           ctypes.POINTER(ctypes.c_double))
        return numpy.ctypeslib.as_array(data, shape=shape)
    dm_cond = property(get_dm_cond)

# TODO: replace VHFOpt in future release
class _VHFOpt:
    def __init__(self, mol, intor=None, prescreen='CVHFnoscreen',
                 qcondname=None, dmcondname=None, direct_scf_tol=1e-14):
        '''New version of VHFOpt (under development).

        If function "qcondname" is presented, the qcond (sqrt(integrals))
        and will be initialized in __init__.

        prescreen, qcondname, dmcondname can be either function pointers or
        names of C functions defined in libcvhf module
        '''
        self.mol = mol
        self._this = cvhfopt = _CVHFOpt()
        cvhfopt.nbas = mol.nbas
        cvhfopt.direct_scf_tol = direct_scf_tol
        cvhfopt.fprescreen = _fpointer(prescreen)
        cvhfopt.r_vkscreen = _fpointer('CVHFr_vknoscreen')
        self._q_cond = None
        self._dm_cond = None

        if intor is None:
            self._intor = intor
            self._cintopt = lib.c_null_ptr()
        else:
            intor = mol._add_suffix(intor)
            self._intor = intor
            self._cintopt = make_cintopt(mol._atm, mol._bas, mol._env, intor)

        self._dmcondname = dmcondname
        self._qcondname = qcondname
        if qcondname is not None and intor is not None:
            self.init_cvhf_direct(mol, intor, qcondname)

    def init_cvhf_direct(self, mol, intor, qcondname):
        '''qcondname can be the function pointer or the name of a C function
        defined in libcvhf module
        '''
        intor = mol._add_suffix(intor)
        if intor == self._intor:
            cintopt = self._cintopt
        else:
            cintopt = lib.c_null_ptr()
        ao_loc = make_loc(mol._bas, intor)
        if isinstance(qcondname, ctypes._CFuncPtr):
            fqcond = qcondname
        else:
            fqcond = getattr(libcvhf, qcondname)
        nbas = mol.nbas
        q_cond = numpy.empty((nbas, nbas))
        with mol.with_integral_screen(self.direct_scf_tol**2):
            fqcond(getattr(libcvhf, intor), cintopt, q_cond.ctypes,
                   ao_loc.ctypes, mol._atm.ctypes, ctypes.c_int(mol.natm),
                   mol._bas.ctypes, ctypes.c_int(nbas), mol._env.ctypes)

        self.q_cond = q_cond
        self._qcondname = qcondname

    @property
    def direct_scf_tol(self):
        return self._this.direct_scf_tol
    @direct_scf_tol.setter
    def direct_scf_tol(self, v):
        self._this.direct_scf_tol = v

    @property
    def prescreen(self):
        return self._this.fprescreen
    @prescreen.setter
    def prescreen(self, v):
        if isinstance(v, str):
            v = _fpointer(v)
        self._this.fprescreen = v

    def set_dm(self, dm, atm, bas, env):
        if self._dmcondname is None:
            return

        mol = self.mol
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            n_dm = 1
        else:
            n_dm = len(dm)
        dm = numpy.asarray(dm, order='C')
        ao_loc = make_loc(mol._bas, self._intor)
        if isinstance(self._dmcondname, ctypes._CFuncPtr):
            fdmcond = self._dmcondname
        else:
            fdmcond = getattr(libcvhf, self._dmcondname)
        nbas = mol.nbas
        dm_cond = numpy.empty((nbas, nbas))
        fdmcond(dm_cond.ctypes, dm.ctypes, ctypes.c_int(n_dm),
                ao_loc.ctypes, mol._atm.ctypes, ctypes.c_int(mol.natm),
                mol._bas.ctypes, ctypes.c_int(nbas), mol._env.ctypes)
        self.dm_cond = dm_cond

    def get_q_cond(self):
        return self._q_cond
    q_cond = property(get_q_cond)

    @q_cond.setter
    def q_cond(self, q_cond):
        self._q_cond = q_cond
        if q_cond is not None:
            self._this.q_cond = q_cond.ctypes.data_as(ctypes.c_void_p)

    def get_dm_cond(self):
        return self._dm_cond
    dm_cond = property(get_dm_cond)

    @dm_cond.setter
    def dm_cond(self, dm_cond):
        self._dm_cond = dm_cond
        if dm_cond is not None:
            self._this.dm_cond = dm_cond.ctypes.data_as(ctypes.c_void_p)


class _CVHFOpt(ctypes.Structure):
    __slots__ = []
    _fields_ = [('nbas', ctypes.c_int),
                ('ngrids', ctypes.c_int),
                ('direct_scf_tol', ctypes.c_double),
                ('q_cond', ctypes.c_void_p),
                ('dm_cond', ctypes.c_void_p),
                ('fprescreen', ctypes.c_void_p),
                ('r_vkscreen', ctypes.c_void_p)]

################################################
# for general DM
# hermi = 0 : arbitrary
# hermi = 1 : hermitian
# hermi = 2 : anti-hermitian
################################################
def incore(eri, dms, hermi=0, with_j=True, with_k=True):
    assert (eri.dtype == numpy.double)
    eri = numpy.asarray(eri, order='C')
    dms = numpy.asarray(dms, order='C', dtype=numpy.double)
    dms_shape = dms.shape
    nao = dms_shape[-1]

    dms = dms.reshape(-1,nao,nao)
    n_dm = dms.shape[0]

    vj = vk = None
    if with_j:
        vj = numpy.zeros((n_dm,nao,nao))
    if with_k:
        vk = numpy.zeros((n_dm,nao,nao))

    dmsptr = []
    vjkptr = []
    fjkptr = []

    npair = nao*(nao+1)//2
    if eri.ndim == 2 and npair*npair == eri.size: # 4-fold symmetry eri
        fdrv = getattr(libcvhf, 'CVHFnrs4_incore_drv')
        if with_j:
            # 'ijkl,kl->ij'
            fvj = _fpointer('CVHFics4_kl_s2ij')
            # or
            ## 'ijkl,ij->kl'
            #fvj = _fpointer('CVHFics4_ij_s2kl')
            for i, dm in enumerate(dms):
                dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
                vjkptr.append(vj[i].ctypes.data_as(ctypes.c_void_p))
                fjkptr.append(fvj)
        if with_k:
            # 'ijkl,il->jk'
            fvk = _fpointer('CVHFics4_il_s1jk')
            # or
            ## 'ijkl,jk->il'
            #fvk = _fpointer('CVHFics4_jk_s1il')
            for i, dm in enumerate(dms):
                dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
                vjkptr.append(vk[i].ctypes.data_as(ctypes.c_void_p))
                fjkptr.append(fvk)

    elif eri.ndim == 1 and npair*(npair+1)//2 == eri.size: # 8-fold symmetry eri
        if n_dm >= INCORE_BATCH_MIN:
            vj, vk = _incore_s8_batch(eri, dms, hermi, with_j, with_k)
            if with_j:
                vj = vj.reshape(dms_shape)
            if with_k:
                vk = vk.reshape(dms_shape)
            return vj, vk
        fdrv = getattr(libcvhf, 'CVHFnrs8_incore_drv')
        if with_j:
            fvj = _fpointer('CVHFics8_tridm_vj')
            tridms = lib.pack_tril(lib.hermi_sum(dms, axes=(0,2,1)))
            idx = numpy.arange(nao)
            tridms[:,idx*(idx+1)//2+idx] *= .5
            for i, tridm in enumerate(tridms):
                dmsptr.append(tridm.ctypes.data_as(ctypes.c_void_p))
                vjkptr.append(vj[i].ctypes.data_as(ctypes.c_void_p))
                fjkptr.append(fvj)
        if with_k:
            if hermi == 1:
                fvk = _fpointer('CVHFics8_jk_s2il')
            else:
                fvk = _fpointer('CVHFics8_jk_s1il')
            for i, dm in enumerate(dms):
                dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
                vjkptr.append(vk[i].ctypes.data_as(ctypes.c_void_p))
                fjkptr.append(fvk)
    else:
        raise RuntimeError('Array shape not consistent: DM %s, eri %s'
                           % (dms_shape, eri.shape))

    n_ops = len(dmsptr)
    fdrv(eri.ctypes.data_as(ctypes.c_void_p),
         (ctypes.c_void_p*n_ops)(*dmsptr), (ctypes.c_void_p*n_ops)(*vjkptr),
         ctypes.c_int(n_ops), ctypes.c_int(nao),
         (ctypes.c_void_p*n_ops)(*fjkptr))

    if with_j:
        for i in range(n_dm):
            lib.hermi_triu(vj[i], 1, inplace=True)
        vj = vj.reshape(dms_shape)
    if with_k:
        if hermi != 0:
            for i in range(n_dm):
                lib.hermi_triu(vk[i], hermi, inplace=True)
        vk = vk.reshape(dms_shape)
    return vj, vk

# Number of density matrices from which incore() contracts s8 ERIs with the
# kernels of nr_incore_batch.c instead of one kernel per dm: BLAS-3 for
# several dms, and one fused J/K pass for a single symmetric dm.
INCORE_BATCH_MIN = 1

def _incore_s8_batch(eri, dms, hermi, with_j, with_k):
    '''J and K of all dms from s8 ERIs, see nr_incore_batch.c'''
    n_dm, nao = dms.shape[:2]
    vj = vk = None
    # With hermi=1 the per-dm kernel returns the lower triangle of K for any
    # dm. The symmetric formulations equal it only for symmetric dms, so other
    # dms take the general one and its lower triangle.
    sym_k = with_k and hermi == 1 and (
        abs(dms - dms.transpose(0,2,1)).max() <= 1e-13 * max(abs(dms).max(), 1))
    if n_dm == 1 and (sym_k or not with_k):
        null = lib.c_null_ptr()
        dm = numpy.asarray(dms[0], order='C')
        dmtri = lib.pack_tril(dm + dm.T)
        idx = numpy.arange(nao)
        dmtri[idx*(idx+1)//2+idx] *= .5
        vjtri = numpy.empty_like(dmtri) if with_j else None
        z = numpy.empty_like(dm) if with_k else None
        libcvhf.CVHFnrs8_incore_jk_1dm(
            null if vjtri is None else vjtri.ctypes.data_as(ctypes.c_void_p),
            null if z is None else z.ctypes.data_as(ctypes.c_void_p),
            eri.ctypes.data_as(ctypes.c_void_p), dmtri.ctypes.data_as(ctypes.c_void_p),
            dm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nao))
        if with_j:
            vj = lib.unpack_tril(vjtri)[None]
        if with_k:
            vk = (z + z.T)[None]
        return vj, vk

    if with_j:
        dmtri = lib.pack_tril(dms + dms.transpose(0,2,1))
        idx = numpy.arange(nao)
        dmtri[:,idx*(idx+1)//2+idx] *= .5
        dmtri = numpy.asarray(dmtri.T, order='C')
        vjtri = numpy.empty_like(dmtri)
        libcvhf.CVHFnrs8_incore_vj_batch(
            vjtri.ctypes.data_as(ctypes.c_void_p), eri.ctypes.data_as(ctypes.c_void_p),
            dmtri.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n_dm), ctypes.c_int(nao))
        vj = lib.unpack_tril(vjtri.T)
    if with_k:
        # Thread-private accumulators of nao*nao*w doubles each: take the dms in
        # batches so that they stay within ~2 GB in total
        w = 1 if sym_k else 2
        batch = max(1, int(2e9 / (8 * w * nao**2 * lib.num_threads())))
        vk = numpy.empty((n_dm, nao, nao))
        for p0 in range(0, n_dm, batch):
            p1 = min(p0 + batch, n_dm)
            dm = dms[p0:p1]
            if sym_k:
                y = dm.transpose(1,0,2)
            else:
                y = numpy.vstack((dm, dm.transpose(0,2,1))).transpose(1,0,2)
            y = numpy.asarray(y, order='C')
            z = numpy.empty_like(y)
            libcvhf.CVHFnrs8_incore_vk_batch(
                z.ctypes.data_as(ctypes.c_void_p), eri.ctypes.data_as(ctypes.c_void_p),
                y.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(p1-p0),
                ctypes.c_int(nao), ctypes.c_int(1 if sym_k else 0))
            z = z.transpose(1,0,2)
            if sym_k:
                vk[p0:p1] = z + z.transpose(0,2,1)
            else:
                vk[p0:p1] = z[:p1-p0] + z[p1-p0:].transpose(0,2,1)
        if hermi != 0 and not sym_k:
            for i in range(n_dm):
                lib.hermi_triu(vk[i], hermi, inplace=True)
    return vj, vk

# Number of density matrices from which direct() contracts each shell quartet
# with all dms in one pass (nr_direct_batch.c) instead of one kernel call per
# dm and J/K script (CVHFnr_direct_drv). The batched kernels are also faster
# for a single dm.
DIRECT_BATCH_MIN = 1

def _direct_s8_batch(intor, dms, hermi, with_j, with_k, atm, bas, env,
                     cvhfopt=None, cintopt=None):
    '''J and K of all dms from s8 ERIs computed on the fly, see nr_direct_batch.c'''
    n_dm, nao = dms.shape[:2]
    c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    c_env = numpy.asarray(env, dtype=numpy.double, order='C')
    ao_loc = make_loc(c_bas, intor)
    if cintopt is None:
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)
    if cvhfopt is None:
        cvhfopt = lib.c_null_ptr()
    elif isinstance(cvhfopt, ctypes.Structure):
        assert cvhfopt.dm_cond and cvhfopt.q_cond
        cvhfopt = ctypes.byref(cvhfopt)
    # As for incore(): with hermi=1 the per-dm kernels return the lower
    # triangle of K for any dm; the symmetric formulation needs symmetric dms.
    sym_k = with_k and hermi == 1 and (
        abs(dms - dms.transpose(0,2,1)).max() <= 1e-13 * max(abs(dms).max(), 1))

    # Each thread accumulates nao*nao*n doubles for J and for K. Batches of dms
    # beyond the memory available recompute the integrals.
    mem_avail = max(lib.param.MAX_MEMORY - lib.current_memory()[0], 2000) * .5e6
    batch = max(1, int(mem_avail / (8 * nao**2 * (with_j + with_k) * lib.num_threads())))
    vj = numpy.empty((n_dm, nao, nao)) if with_j else None
    vk = numpy.empty((n_dm, nao, nao)) if with_k else None
    null = lib.c_null_ptr()
    for p0 in range(0, n_dm, batch):
        p1 = min(p0 + batch, n_dm)
        # the kernels take the dms in chunks of 4, the last one padded with
        # zeros, or a single dm
        npad = 1 if p1 - p0 == 1 else (p1 - p0 + 3) // 4 * 4
        dm = numpy.zeros((nao, nao, npad))
        dm[:,:,:p1-p0] = dms[p0:p1].transpose(1,2,0)
        dm_sum = dm + dm.transpose(1,0,2) if with_j else None
        vjb = numpy.empty_like(dm) if with_j else None
        vkb = numpy.empty_like(dm) if with_k else None
        libcvhf.CVHFnr_direct_jk_batch(
            getattr(libcvhf, intor),
            null if vjb is None else vjb.ctypes.data_as(ctypes.c_void_p),
            null if vkb is None else vkb.ctypes.data_as(ctypes.c_void_p),
            dm.ctypes.data_as(ctypes.c_void_p),
            null if dm_sum is None else dm_sum.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(npad), ctypes.c_int(1 if sym_k else 0),
            ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cvhfopt,
            c_atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(c_atm.shape[0]),
            c_bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(c_bas.shape[0]),
            c_env.ctypes.data_as(ctypes.c_void_p))
        if with_j:
            vj[p0:p1] = (vjb + vjb.transpose(1,0,2)).transpose(2,0,1)[:p1-p0]
        if with_k:
            if sym_k:
                vkb = vkb + vkb.transpose(1,0,2)
            vk[p0:p1] = vkb.transpose(2,0,1)[:p1-p0]
    if with_k and hermi != 0 and not sym_k:
        for i in range(n_dm):
            lib.hermi_triu(vk[i], hermi, inplace=True)
    return vj, vk

# use int2e_sph as cintor, CVHFnrs8_ij_s2kl, CVHFnrs8_jk_s2il as fjk to call
# direct_mapdm
def direct(dms, atm, bas, env, vhfopt=None, hermi=0, cart=False,
           with_j=True, with_k=True, out=None, optimize_sr=False):
    dms = numpy.asarray(dms, order='C', dtype=numpy.double)
    dms_shape = dms.shape
    nao = dms_shape[-1]
    dms = dms.reshape(-1,nao,nao)
    n_dm = dms.shape[0]

    if vhfopt is None:
        cvhfopt = None
        cintopt = None
        if cart:
            intor = 'int2e_cart'
        else:
            intor = 'int2e_sph'
    else:
        vhfopt.set_dm(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        intor = vhfopt._intor

    if (n_dm >= DIRECT_BATCH_MIN and (with_j or with_k) and
            not (optimize_sr and env[gto.PTR_RANGE_OMEGA] < 0)):
        vj, vk = _direct_s8_batch(intor, dms, hermi, with_j, with_k,
                                  atm, bas, env, cvhfopt, cintopt)
        if out is not None:  # [vj, vk] or the one requested
            out = out.reshape(-1, n_dm, nao, nao)
            if with_j:
                out[0] = vj
                vj = out[0]
            if with_k:
                out[int(with_j)] = vk
                vk = out[int(with_j)]
        if with_j:
            vj = vj.reshape(dms_shape)
        if with_k:
            vk = vk.reshape(dms_shape)
        return vj, vk

    vj = vk = None
    jkscripts = []
    n_jk = 0
    if with_j:
        jkscripts.extend(['ji->s2kl']*n_dm)
        n_jk += 1
    if with_k:
        if hermi == 1:
            jkscripts.extend(['li->s2kj']*n_dm)
        else:
            jkscripts.extend(['li->s1kj']*n_dm)
        n_jk += 1
    if n_jk == 0:
        return vj, vk

    dms = list(dms) * n_jk # make n_jk copies of dms
    if out is None:
        out = numpy.empty((n_jk*n_dm, nao, nao))
    nr_direct_drv(intor, 's8', jkscripts, dms, 1, atm, bas, env,
                  cvhfopt, cintopt, out=out, optimize_sr=optimize_sr)
    if with_j and with_k:
        vj = out[:n_dm]
        vk = out[n_dm:]
    elif with_j:
        vj = out
    else:
        vk = out

    if with_j:
        # vj must be symmetric
        for i in range(n_dm):
            lib.hermi_triu(vj[i], 1, inplace=True)
        vj = vj.reshape(dms_shape)
    if with_k:
        if hermi != 0:
            for i in range(n_dm):
                lib.hermi_triu(vk[i], hermi, inplace=True)
        vk = vk.reshape(dms_shape)
    return vj, vk

# call all fjk for each dm. The return has the shape
# [len(jkdescript),len(dms),ncomp,nao,nao]
# jkdescript: 'ij->s1kl', 'kl->s2ij', ...
def direct_mapdm(intor, aosym, jkdescript,
                 dms, ncomp, atm, bas, env, vhfopt=None, cintopt=None,
                 shls_slice=None, shls_excludes=None, out=None,
                 optimize_sr=False):
    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        dms = dms[numpy.newaxis,:,:]
        single_dm = True
    else:
        single_dm = False
    intor = ascint3(intor)
    if vhfopt is None:
        cvhfopt = lib.c_null_ptr()
        cintopt = None
    else:
        vhfopt.set_dm(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt

    n_dm = len(dms)
    dms = [numpy.asarray(dm, order='C', dtype=numpy.double) for dm in dms]
    if isinstance(jkdescript, str):
        jkscripts = (jkdescript,)
    else:
        jkscripts = jkdescript
    n_jk = len(jkscripts)

    # make n_jk copies of dms
    dms = dms * n_jk
    # make n_dm copies for each jk script
    jkscripts = numpy.repeat(jkscripts, n_dm)

    vjk = nr_direct_drv(intor, aosym, jkscripts, dms, ncomp, atm, bas, env,
                        cvhfopt, cintopt, shls_slice, shls_excludes, out,
                        optimize_sr=optimize_sr)
    if ncomp == 1:
        vjk = [v[0] for v in vjk]

    if single_dm:
        if isinstance(jkdescript, str):
            vjk = vjk[0]
    elif isinstance(jkdescript, str):
        vjk = numpy.asarray(vjk)
    else: # n_jk > 1
        vjk = [numpy.asarray(vjk[i*n_dm:(i+1)*n_dm]) for i in range(n_jk)]
    return vjk

# for density matrices in dms, bind each dm to a jk operator
# jkdescript: 'ij->s1kl', 'kl->s2ij', ...
def direct_bindm(intor, aosym, jkdescript,
                 dms, ncomp, atm, bas, env, vhfopt=None, cintopt=None,
                 shls_slice=None, shls_excludes=None, out=None,
                 optimize_sr=False):
    intor = ascint3(intor)
    if vhfopt is None:
        cvhfopt = lib.c_null_ptr()
        cintopt = None
    else:
        vhfopt.set_dm(dms, atm, bas, env)
        cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt

    vjk = nr_direct_drv(intor, aosym, jkdescript, dms, ncomp, atm, bas, env,
                        cvhfopt, cintopt, shls_slice, shls_excludes, out,
                        optimize_sr=optimize_sr)
    if ncomp == 1:
        if isinstance(jkdescript, str):
            vjk = vjk[0]
        else:
            vjk = [v[0] for v in vjk]
    return vjk

def nr_direct_drv(intor, aosym, jkscript,
                  dms, ncomp, atm, bas, env, cvhfopt=None, cintopt=None,
                  shls_slice=None, shls_excludes=None, out=None,
                  optimize_sr=True):
    if cvhfopt is None:
        optimize_sr = False

    if optimize_sr:
        assert aosym in ('s8', 's4', 's2ij', 's2kl', 's1')
    else:
        assert aosym in ('s8', 's4', 's2ij', 's2kl', 's1',
                         'aa4', 'a4ij', 'a4kl', 'a2ij', 'a2kl')
    c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    c_env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        dms = dms[numpy.newaxis,:,:]
    assert dms[0].ndim == 2
    n_dm = len(dms)
    dms = [numpy.asarray(dm, order='C', dtype=numpy.double) for dm in dms]

    if isinstance(jkscript, str):
        jkscripts = (jkscript,)
    else:
        jkscripts = jkscript
    assert len(jkscripts) == n_dm

    if cvhfopt is None:
        cvhfopt = lib.c_null_ptr()
    elif isinstance(cvhfopt, ctypes.Structure):
        # To make cvhfopt _VHFOpt comparable
        assert cvhfopt.dm_cond and cvhfopt.q_cond
        cvhfopt = ctypes.byref(cvhfopt)
    if cintopt is None:
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)

    if shls_slice is None:
        shls_slice = [0, c_bas.shape[0]] * 4
    if shls_excludes is not None:
        shls_excludes = _check_shls_excludes(shls_slice, shls_excludes)

    ao_loc = make_loc(bas, intor)

    vjk = []
    descr_sym = [x.split('->') for x in jkscripts]
    fjk = (ctypes.c_void_p*(n_dm))()
    dmsptr = (ctypes.c_void_p*(n_dm))()
    vjkptr = (ctypes.c_void_p*(n_dm))()
    for i, (dmsym, vsym) in enumerate(descr_sym):
        f1 = _fpointer('CVHFnr%s_%s_%s'%(aosym, dmsym, vsym))

        if dms[i].shape != get_dims(dmsym, shls_slice, ao_loc):
            raise RuntimeError('dm[%d] shape %s is inconsistent with the '
                               'shls_slice shape %s' %
                               (i, dms[i].shape, get_dims(dmsym, shls_slice, ao_loc)))
        vshape = (ncomp,) + get_dims(vsym[-2:], shls_slice, ao_loc)
        if out is None:
            buf = numpy.empty(vshape)
        else:
            buf = out[i]
            assert buf.size == numpy.prod(vshape)
            assert buf.dtype == numpy.double
            assert buf.flags.c_contiguous
        vjk.append(buf)
        dmsptr[i] = dms[i].ctypes.data_as(ctypes.c_void_p)
        vjkptr[i] = vjk[i].ctypes.data_as(ctypes.c_void_p)
        fjk[i] = f1

    omega = env[gto.PTR_RANGE_OMEGA]
    if omega < 0 and optimize_sr:
        assert shls_excludes is None
        drv = 'CVHFnr_sr_direct_drv'
        shls_slice = (ctypes.c_int*8)(*shls_slice)
    elif shls_excludes is None:
        drv = 'CVHFnr_direct_drv'
        shls_slice = (ctypes.c_int*8)(*shls_slice)
    else:
        drv = 'CVHFnr_direct_ex_drv'
        shls_slice = (ctypes.c_int*16)(*shls_slice, *shls_excludes)
    fdrv = getattr(libcvhf, drv)
    dotsym = _INTSYMAP[aosym]
    if omega < 0 and optimize_sr:
        fdot = getattr(libcvhf, 'CVHFdot_sr_nr'+dotsym)
    else:
        fdot = getattr(libcvhf, 'CVHFdot_nr'+dotsym)
    cintor = getattr(libcvhf, intor)

    fdrv(cintor, fdot, fjk, dmsptr, vjkptr,
         ctypes.c_int(n_dm), ctypes.c_int(ncomp), shls_slice,
         ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cvhfopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if isinstance(jkscript, str):
        vjk = vjk[0]
    return vjk


# 8-fold permutation symmetry
def int2e_sph(atm, bas, env, cart=False):  # pragma: no cover
    if cart:
        intor = 'int2e_cart'
    else:
        intor = 'int2e_sph'
    return gto.moleintor.getints4c(intor, atm, bas, env, aosym='s8')


################################################################
# relativistic
def rdirect_mapdm(intor, aosym, jkdescript,
                  dms, ncomp, atm, bas, env, vhfopt=None, cintopt=None,
                  shls_slice=None):
    assert (aosym in ('s8', 's4', 's2ij', 's2kl', 's1',
                     'a4ij', 'a4kl', 'a2ij', 'a2kl'))
    intor = ascint3(intor)
    c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    c_env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        dms = dms[numpy.newaxis,:,:]
    n_dm = len(dms)
    nao = dms[0].shape[0]
    dms = numpy.asarray(dms, order='C', dtype=numpy.complex128)
    if isinstance(jkdescript, str):
        jkdescript = (jkdescript,)
    njk = len(jkdescript)

    if vhfopt is None:
        cintor = _fpointer(intor)
        cvhfopt = lib.c_null_ptr()
    else:
        vhfopt.set_dm(dms, atm, bas, env)
        if isinstance(vhfopt, _VHFOpt):
            cvhfopt = ctypes.byref(vhfopt._this)
        else:
            cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        cintor = getattr(libcvhf, vhfopt._intor)
    if cintopt is None:
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)

    fdrv = getattr(libcvhf, 'CVHFr_direct_drv')
    dotsym = _INTSYMAP[aosym]
    fdot = _fpointer('CVHFdot_r'+dotsym)

    if shls_slice is None:
        shls_slice = (0, c_bas.shape[0])*4
    else:
        raise NotImplementedError
    ao_loc = make_loc(bas, intor)

    unpackas = _INTUNPACKMAP_R[aosym]
    descr_sym = [x.split('->') for x in jkdescript]
    fjk = (ctypes.c_void_p*(njk*n_dm))()
    dm1 = (ctypes.c_void_p*(njk*n_dm))()
    for i, (dmsym, vsym) in enumerate(descr_sym):
        f1 = _fpointer('CVHFr%s_%s_%s'%(unpackas, dmsym, vsym))
        for j in range(n_dm):
            dm1[i*n_dm+j] = dms[j].ctypes.data_as(ctypes.c_void_p)
            fjk[i*n_dm+j] = f1
    vjk = numpy.empty((njk,n_dm*ncomp,nao,nao), dtype=numpy.complex128)

    fdrv(cintor, fdot, fjk, dm1,
         vjk.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(njk*n_dm), ctypes.c_int(ncomp),
         (ctypes.c_int*8)(*shls_slice),
         ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cvhfopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if n_dm * ncomp == 1:
        vjk = vjk.reshape(njk,nao,nao)
    if njk == 1:
        vjk = vjk.reshape(vjk.shape[1:])
    return vjk

# for density matrices in dms, bind each dm to a jk operator
def rdirect_bindm(intor, aosym, jkdescript,
                  dms, ncomp, atm, bas, env, vhfopt=None, cintopt=None,
                  shls_slice=None):
    assert (aosym in ('s8', 's4', 's2ij', 's2kl', 's1',
                     'a4ij', 'a4kl', 'a2ij', 'a2kl'))
    intor = ascint3(intor)
    c_atm = numpy.asarray(atm, dtype=numpy.int32, order='C')
    c_bas = numpy.asarray(bas, dtype=numpy.int32, order='C')
    c_env = numpy.asarray(env, dtype=numpy.double, order='C')
    natm = ctypes.c_int(c_atm.shape[0])
    nbas = ctypes.c_int(c_bas.shape[0])

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        dms = dms[numpy.newaxis]
    n_dm = len(dms)
    nao = dms[0].shape[0]
    dms = numpy.asarray(dms, order='C', dtype=numpy.complex128)
    if isinstance(jkdescript, str):
        jkdescript = (jkdescript,)
    njk = len(jkdescript)
    # for SSLL integrals, njk can be less than n_dm, extra dms are required by
    # set_dm function to get dm_cond
    assert njk == n_dm or njk*4 == n_dm*3

    if vhfopt is None:
        cintor = _fpointer(intor)
        cvhfopt = lib.c_null_ptr()
    else:
        vhfopt.set_dm(dms, atm, bas, env)
        if isinstance(vhfopt, _VHFOpt):
            cvhfopt = ctypes.byref(vhfopt._this)
        else:
            cvhfopt = vhfopt._this
        cintopt = vhfopt._cintopt
        cintor = getattr(libcvhf, vhfopt._intor)
    if cintopt is None:
        cintopt = make_cintopt(c_atm, c_bas, c_env, intor)

    fdrv = getattr(libcvhf, 'CVHFr_direct_drv')
    dotsym = _INTSYMAP[aosym]
    fdot = _fpointer('CVHFdot_r'+dotsym)

    if shls_slice is None:
        shls_slice = (0, c_bas.shape[0])*4
    else:
        raise NotImplementedError
    ao_loc = make_loc(bas, intor)

    unpackas = _INTUNPACKMAP_R[aosym]
    descr_sym = [x.split('->') for x in jkdescript]
    fjk = (ctypes.c_void_p*(njk))()
    dm1 = (ctypes.c_void_p*(njk))()
    for i, (dmsym, vsym) in enumerate(descr_sym):
        f1 = _fpointer('CVHFr%s_%s_%s'%(unpackas, dmsym, vsym))
        dm1[i] = dms[i].ctypes.data_as(ctypes.c_void_p)
        fjk[i] = f1
    vjk = numpy.empty((njk,ncomp,nao,nao), dtype=numpy.complex128)

    fdrv(cintor, fdot, fjk, dm1,
         vjk.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(njk), ctypes.c_int(ncomp),
         (ctypes.c_int*8)(*shls_slice),
         ao_loc.ctypes.data_as(ctypes.c_void_p), cintopt, cvhfopt,
         c_atm.ctypes.data_as(ctypes.c_void_p), natm,
         c_bas.ctypes.data_as(ctypes.c_void_p), nbas,
         c_env.ctypes.data_as(ctypes.c_void_p))

    if ncomp == 1:
        vjk = vjk.reshape(njk,nao,nao)
    if njk == 1:
        vjk = vjk.reshape(vjk.shape[1:])
    return vjk

# 'a4ij': anti-symm between ij, symm between kl
# 'a4kl': anti-symm between kl, symm between ij
# 'a2ij': anti-symm between ij,
# 'a2kl': anti-symm between kl,
_INTSYMAP= {
    's8'  : 's8'  ,
    's4'  : 's4'  ,
    's2ij': 's2ij',
    's2kl': 's2kl',
    's1'  : 's1'  ,
    'aa4' : 's4'  ,
    'a4ij': 's4'  ,
    'a4kl': 's4'  ,
    'a2ij': 's2ij',
    'a2kl': 's2kl',
}

_INTUNPACKMAP_R = {
    's8'  : 's8'  ,
    's4'  : 's4'  ,
    's2ij': 's2ij',
    's2kl': 's2kl',
    's1'  : 's1'  ,
    'a4ij': 'ah4'  ,
    'a4kl': 'ha4'  ,
    'a2ij': 'ah2ij',
    'a2kl': 'ha2kl',
}

_SHLINDEX = {'i': 0, 'j': 2, 'k': 4, 'l': 6}
def get_dims(descr_sym, shls_slice, ao_loc):
    i = _SHLINDEX[descr_sym[0]]
    j = _SHLINDEX[descr_sym[1]]
    di = ao_loc[shls_slice[i+1]] - ao_loc[shls_slice[i]]
    dj = ao_loc[shls_slice[j+1]] - ao_loc[shls_slice[j]]
    return (di,dj)

def _check_shls_excludes(shls_slice, shls_excludes):
    '''shls_excludes must be inside shls_slice. Check this'''
    _slice = numpy.array(shls_slice)
    _excludes = numpy.array(shls_excludes)
    if ((_excludes < _slice)[[0, 2, 4, 6]].any() or  # i0, j0, k0, l0
        (_excludes > _slice)[[1, 3, 5, 7]].any()):   # i1, j1, k1, l1
        shls_excludes = numpy.max([_excludes, _slice], axis=0)
        shls_excludes[[1, 3, 5, 7]] = numpy.min(
            [_excludes[[1, 3, 5, 7]], _slice[[1, 3, 5, 7]]], axis=0)
        shls_excludes = shls_excludes.tolist()
        sys.stderr.write(f'shls_excludes {_excludes} conflicts to shls_slice {_slice}\n'
                         f'Set shls_excludes to {shls_excludes}')
    return shls_excludes
