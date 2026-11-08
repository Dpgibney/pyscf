#!/usr/bin/env python
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

'''
Non-relativistic UHF analytical Hessian
'''

from functools import reduce

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import ucphf
from pyscf.hessian import rhf as rhf_hess
_get_jk = rhf_hess._get_jk
_make_vhfopt = rhf_hess._make_vhfopt

# import _response_functions to load gen_response methods in SCF class
from pyscf.scf import _response_functions
# import pyscf.grad.uhf to activate nuc_grad_method method
from pyscf.grad import uhf


def hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              mo1=None, mo_e1=None, h1ao=None,
              atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    de2 = hessobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                    max_memory, log)

    if h1ao is None:
        h1ao = hessobj.make_h1(mo_coeff, mo_occ, None, atmlst, log)
        t1 = log.timer_debug1('making H1', *time0)
    if mo1 is None or mo_e1 is None:
        mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao,
                                       None, atmlst, max_memory, log)
        t1 = log.timer_debug1('solving MO1', *t1)

    h1aoa, h1aob = h1ao
    mo1a, mo1b = mo1
    mo_e1a, mo_e1b = mo_e1

    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    mo_ea = mo_energy[0][mo_occ[0]>0]
    mo_eb = mo_energy[1][mo_occ[1]>0]
    s1a = -mol.intor('int1e_ipovlp', comp=3)

    de2_mo1 = rhf_hess._hess_elec_mo1(mol, atmlst, h1aoa, mo1a, mo_e1a, mocca, mo_ea, s1a,
                                      max_memory)
    de2_mo1 += rhf_hess._hess_elec_mo1(mol, atmlst, h1aob, mo1b, mo_e1b, moccb, mo_eb, s1a,
                                       max_memory)
    de2_mo1 *= 2
    for i0 in range(len(atmlst)):
        de2[i0,:i0+1] += de2_mo1[i0,:i0+1]
        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    log.timer('UHF hessian', *time0)
    return de2

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    e1, ej, ek = _partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                   atmlst, max_memory, verbose, True)
    return e1 + ej - ek  # (A,B,dR_A,dR_B)

def _partial_hess_ejk(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None, with_k=True):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = numpy.dot(mocca, mocca.T)
    dm0b = numpy.dot(moccb, moccb.T)
    dm0 = dm0a + dm0b
    # Energy weighted density matrix
    mo_ea = mo_energy[0][mo_occ[0]>0]
    mo_eb = mo_energy[1][mo_occ[1]>0]
    dme0 = numpy.einsum('pi,qi,i->pq', mocca, mocca, mo_ea)
    dme0+= numpy.einsum('pi,qi,i->pq', moccb, moccb, mo_eb)

    hcore_deriv = hessobj.hcore_generator(mol)
    s1aa, s1ab, s1a = rhf_hess.get_ovlp(mol)

    # Skip the K-type contractions when no exact exchange is needed (with_k=False)
    ej, ek = rhf_hess._partial_hess_2e(mol, dm0, [dm0a, dm0b], 2, atmlst, with_k=with_k,
                                       verbose=log)
    t1 = log.timer_debug1('two-electron terms', *t1)

    aoslices = mol.aoslice_by_atom()
    natm = len(atmlst)
    e1 = numpy.zeros((natm, natm, 3, 3))  # (A,B,dR_A,dR_B)
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        e1[i0,i0] -= numpy.einsum('xypq,pq->xy', s1aa[:,:,p0:p1], dme0[p0:p1])*2

        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            e1[i0,j0] -= numpy.einsum('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], dme0[p0:p1,q0:q1])*2

            h1ao = hcore_deriv(ia, ja)
            e1[i0,j0] += numpy.einsum('xypq,pq->xy', h1ao, dm0)

        for j0 in range(i0):
            e1[j0,i0] = e1[i0,j0].T

    log.timer('UHF partial hessian', *time0)
    return e1, ej, ek

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    dm0a = numpy.dot(mocca, mocca.T)
    dm0b = numpy.dot(moccb, moccb.T)
    # J depends on the total density only
    dm0 = dm0a + dm0b
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)

    aoslices = mol.aoslice_by_atom()
    vhfopt = rhf_hess._make_h1_vhfopt(mol, [dm0, dm0a, dm0b], 'jk')
    h1aoa = [None] * mol.natm
    h1aob = [None] * mol.natm
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        vj1, vj2, vk1a, vk1b, vk2a, vk2b = \
                _get_jk(mol, 'int2e_ip1', 3, 's2kl',
                        ['ji->s2kl', -dm0[:,p0:p1],
                         'lk->s1ij', -dm0         ,
                         'li->s1kj', -dm0a[:,p0:p1], 'li->s1kj', -dm0b[:,p0:p1],
                         'jk->s1il', -dm0a         , 'jk->s1il', -dm0b         ],
                        shls_slice=shls_slice, vhfopt=vhfopt)
        vhfa = vj1 - vk1a
        vhfb = vj1 - vk1b
        vhfa[:,p0:p1] += vj2 - vk2a
        vhfb[:,p0:p1] += vj2 - vk2b
        h1 = hcore_deriv(ia)
        h1aoa[ia] = h1 + vhfa + vhfa.transpose(0,2,1)
        h1aob[ia] = h1 + vhfb + vhfb.transpose(0,2,1)
    return (h1aoa,h1aob)

def solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1ao,
              fx=None, atmlst=None, max_memory=4000, verbose=None,
              max_cycle=50, level_shift=0):
    mol = mf.mol
    log = logger.new_logger(mf, verbose)
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]

    if fx is None:
        fx = gen_vind(mf, mo_coeff, mo_occ)
    s1a = -mol.intor('int1e_ipovlp', comp=3)
    h1aoa, h1aob = h1ao

    def _ao2mo(mat, mo_coeff, mocc):
        return numpy.asarray([reduce(numpy.dot, (mo_coeff.T, x, mocc)) for x in mat])

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, max_memory*.9-mem_now)
    blksize = max(2, int(max_memory*1e6/8 / (nao*(nocca+noccb)*3*6)))
    mo1sa = [None] * mol.natm
    mo1sb = [None] * mol.natm
    e1sa = [None] * mol.natm
    e1sb = [None] * mol.natm
    aoslices = mol.aoslice_by_atom()
    for ia0, ia1 in lib.prange(0, len(atmlst), blksize):
        s1voa = []
        s1vob = []
        h1voa = []
        h1vob = []
        for i0 in range(ia0, ia1):
            ia = atmlst[i0]
            shl0, shl1, p0, p1 = aoslices[ia]
            s1ao = numpy.zeros((3,nao,nao))
            s1ao[:,p0:p1] += s1a[:,p0:p1]
            s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1voa.append(_ao2mo(s1ao, mo_coeff[0], mocca))
            s1vob.append(_ao2mo(s1ao, mo_coeff[1], moccb))
            h1voa.append(_ao2mo(h1aoa[ia], mo_coeff[0], mocca))
            h1vob.append(_ao2mo(h1aob[ia], mo_coeff[1], moccb))

        h1vo = (numpy.vstack(h1voa), numpy.vstack(h1vob))
        s1vo = (numpy.vstack(s1voa), numpy.vstack(s1vob))
        tol = mf.conv_tol_cpscf * (ia1 - ia0)
        mo1, e1 = ucphf.solve(fx, mo_energy, mo_occ, h1vo, s1vo,
                              max_cycle=max_cycle, level_shift=level_shift, tol=tol,
                              verbose=log)
        mo1a = numpy.einsum('pq,xqi->xpi', mo_coeff[0], mo1[0]).reshape(-1,3,nao,nocca)
        mo1b = numpy.einsum('pq,xqi->xpi', mo_coeff[1], mo1[1]).reshape(-1,3,nao,noccb)
        e1a = e1[0].reshape(-1,3,nocca,nocca)
        e1b = e1[1].reshape(-1,3,noccb,noccb)

        for k in range(ia1-ia0):
            ia = atmlst[k+ia0]
            mo1sa[ia] = mo1a[k]
            mo1sb[ia] = mo1b[k]
            e1sa[ia] = e1a[k].reshape(3,nocca,nocca)
            e1sb[ia] = e1b[k].reshape(3,noccb,noccb)
        mo1 = e1 = mo1a = mo1b = e1a = e1b = None

    return (mo1sa,mo1sb), (e1sa,e1sb)

def gen_vind(mf, mo_coeff, mo_occ):
    nao, nmoa = mo_coeff[0].shape
    nmob = mo_coeff[1].shape[1]
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]

    vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)
    # The XC response from the factors of dm1 where the response function offers it
    vind_mo = getattr(vresp, 'vind_mo', None)
    def fx(mo1):
        mo1 = mo1.reshape(-1,nmoa*nocca+nmob*noccb)
        nset = len(mo1)
        dm1 = numpy.empty((2,nset,nao,nao))
        for i, x in enumerate(mo1):
            xa = x[:nmoa*nocca].reshape(nmoa,nocca)
            xb = x[nmoa*nocca:].reshape(nmob,noccb)
            dma = reduce(numpy.dot, (mo_coeff[0], xa, mocca.T))
            dmb = reduce(numpy.dot, (mo_coeff[1], xb, moccb.T))
            dm1[0,i] = dma + dma.T
            dm1[1,i] = dmb + dmb.T
        if vind_mo is not None:
            v1a, v1b = vind_mo(dm1, (mo1[:,:nmoa*nocca], mo1[:,nmoa*nocca:]))
            v1vo = numpy.empty_like(mo1)
            v1vo[:,:nmoa*nocca] = v1a.reshape(nset,-1)
            v1vo[:,nmoa*nocca:] = v1b.reshape(nset,-1)
            return v1vo
        v1 = vresp(dm1)
        v1vo = numpy.empty_like(mo1)
        for i in range(nset):
            v1vo[i,:nmoa*nocca] = reduce(numpy.dot, (mo_coeff[0].T, v1[0,i], mocca)).ravel()
            v1vo[i,nmoa*nocca:] = reduce(numpy.dot, (mo_coeff[1].T, v1[1,i], moccb)).ravel()
        return v1vo
    return fx


def gen_hop(hobj, mo_energy=None, mo_coeff=None, mo_occ=None, verbose=None):
    log = logger.new_logger(hobj, verbose)
    mol = hobj.mol
    mf = hobj.base

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    natm = mol.natm
    nao, nmo = mo_coeff[0].shape
    mocca = mo_coeff[0][:,mo_occ[0]>0]
    moccb = mo_coeff[1][:,mo_occ[1]>0]
    mo_ea = mo_energy[0][mo_occ[0]>0]
    mo_eb = mo_energy[1][mo_occ[1]>0]
    nocca = mocca.shape[1]
    noccb = moccb.shape[1]

    atmlst = range(natm)
    max_memory = max(2000, hobj.max_memory - lib.current_memory()[0])
    de2 = hobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                 max_memory, log)
    de2 += hobj.hess_nuc()

    h1ao_cache = hobj.make_h1(mo_coeff, mo_occ, None, atmlst, log)
    h1aoa_cache, h1aob_cache = h1ao_cache

    aoslices = mol.aoslice_by_atom()
    s1a = -mol.intor('int1e_ipovlp', comp=3)

    fvind = gen_vind(mf, mo_coeff, mo_occ)
    def h_op(x):
        x = x.reshape(natm,3)
        hx = numpy.einsum('abxy,ax->by', de2, x)
        h1aoa = 0
        h1aob = 0
        s1ao = 0
        for ia in range(natm):
            shl0, shl1, p0, p1 = aoslices[ia]
            h1aoa += numpy.einsum('x,xij->ij', x[ia], h1aoa_cache[ia])
            h1aob += numpy.einsum('x,xij->ij', x[ia], h1aob_cache[ia])
            s1ao_i = numpy.zeros((3,nao,nao))
            s1ao_i[:,p0:p1] += s1a[:,p0:p1]
            s1ao_i[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1ao += numpy.einsum('x,xij->ij', x[ia], s1ao_i)

        s1voa = reduce(numpy.dot, (mo_coeff[0].T, s1ao, mocca))
        s1vob = reduce(numpy.dot, (mo_coeff[1].T, s1ao, moccb))
        h1voa = reduce(numpy.dot, (mo_coeff[0].T, h1aoa, mocca))
        h1vob = reduce(numpy.dot, (mo_coeff[1].T, h1aob, moccb))
        mo1, mo_e1 = ucphf.solve(fvind, mo_energy, mo_occ,
                                 (h1voa,h1vob), (s1voa,s1vob), verbose=log)
        mo1a = numpy.dot(mo_coeff[0], mo1[0])
        mo1b = numpy.dot(mo_coeff[1], mo1[1])
        mo_e1a = mo_e1[0].reshape(nocca,nocca)
        mo_e1b = mo_e1[1].reshape(noccb,noccb)
        dm1a = numpy.einsum('pi,qi->pq', mo1a, mocca)
        dm1b = numpy.einsum('pi,qi->pq', mo1b, moccb)
        dme1a = numpy.einsum('pi,qi,i->pq', mo1a, mocca, mo_ea)
        dme1a = dme1a + dme1a.T + reduce(numpy.dot, (mocca, mo_e1a, mocca.T))
        dme1b = numpy.einsum('pi,qi,i->pq', mo1b, moccb, mo_eb)
        dme1b = dme1b + dme1b.T + reduce(numpy.dot, (moccb, mo_e1b, moccb.T))
        dme1 = dme1a + dme1b

        for ja in range(natm):
            q0, q1 = aoslices[ja][2:]
            hx[ja] += numpy.einsum('xpq,pq->x', h1aoa_cache[ja], dm1a) * 2
            hx[ja] += numpy.einsum('xpq,pq->x', h1aob_cache[ja], dm1b) * 2
            hx[ja] -= numpy.einsum('xpq,pq->x', s1a[:,q0:q1], dme1[q0:q1])
            hx[ja] -= numpy.einsum('xpq,qp->x', s1a[:,q0:q1], dme1[:,q0:q1])
        return hx.ravel()

    hdiag = numpy.einsum('aaxx->ax', de2).ravel()
    return h_op, hdiag


class Hessian(rhf_hess.HessianBase):
    '''Non-relativistic UHF hessian'''

    partial_hess_elec = partial_hess_elec
    hess_elec = hess_elec
    make_h1 = make_h1
    gen_hop = gen_hop

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
        if self.sgx_response or self.cphf_residual_tol is not None:
            raise NotImplementedError('sgx_response and cphf_residual_tol for UHF and UKS')
        return solve_mo1(self.base, mo_energy, mo_coeff, mo_occ, h1ao,
                         fx, atmlst, max_memory, verbose,
                         max_cycle=self.max_cycle, level_shift=self.level_shift)

from pyscf import scf
scf.uhf.UHF.Hessian = lib.class_as_method(Hessian)
