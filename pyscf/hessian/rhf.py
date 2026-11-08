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
Non-relativistic RHF analytical Hessian
'''

import contextlib
from functools import reduce
import ctypes

import numpy
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf.scf import _vhf, hf
from pyscf.scf import cphf
from pyscf import __config__

# import _response_functions to load gen_response methods in SCF class
from pyscf.scf import _response_functions
# import pyscf.grad.rhf to activate nuc_grad_method method
from pyscf.grad import rhf

# _partial_hess_2e takes the two-electron blocks from hess_jk_energy when atmlst holds every atom
JK_ENERGY = getattr(__config__, 'hessian_rhf_jk_energy', True)
# The screening threshold of hess_jk_energy (that of the _VHFOpt of _make_vhfopt). 0 computes every shell
# quartet whose densities are not zero, as the loops of _partial_hess_2e do: _make_vhfopt finds no int2e_ip1ip2
# etc. in libcvhf (they are int2e_ip1ip2_sph etc.), so their integrals are not screened.
JK_ENERGY_TOL = getattr(__config__, 'hessian_rhf_jk_energy_tol', 1e-14)
# The memory (MB) hess_jk_energy's chunk buffers may take; fixed, so that the results do not depend on the memory
# available
JK_ENERGY_BUF = 128
# hess_jk_energy's batched integrals contracted with the densities inside qcint's kernels where it offers that
# (int2e_*_sph_batch_dot); False keeps the contraction in hess_jk_energy
JK_ENERGY_DOT = getattr(__config__, 'hessian_rhf_jk_energy_dot', True)
# The screening threshold of make_h1's int2e_ip1 J/K (_make_h1_vhfopt); 0 computes every quartet, as without a
# vhfopt
MAKE_H1_TOL = getattr(__config__, 'hessian_rhf_make_h1_tol', 1e-14)


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

    mocc = mo_coeff[:,mo_occ>0]
    # *2 for double occupancy, *2 for +c.c.
    de2_mo1 = _hess_elec_mo1(mol, atmlst, h1ao, mo1, mo_e1, mocc,
                             mo_energy[mo_occ>0], max_memory=max_memory) * 4
    for i0 in range(len(atmlst)):
        de2[i0,:i0+1] += de2_mo1[i0,:i0+1]
        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    log.timer('RHF hessian', *time0)
    return de2

def _hess_elec_mo1(mol, atmlst, h1ao, mo1, mo_e1, mocc, mo_eocc, s1a=None,
                   max_memory=4000):
    '''tr(h1ao[A] dm1[B]) - tr(s1ao[A] dme1[B]) - tr(s1oo[A] mo_e1[B])/2 for all pairs of atoms
    A, B of atmlst, as [natm,natm,3,3]. Here dm1[B] = mo1[B] C^T and dme1[B] = mo1[B] e C^T, where
    C holds the occupied orbitals and e their energies; s1ao[A] is the overlap derivative for
    atom A and s1oo[A] = C^T s1ao[A] C. The factors of A and of B are formed once per atom, and
    the pair sums are GEMMs.
    '''
    if s1a is None:
        s1a = -mol.intor('int1e_ipovlp', comp=3)
    nao, nocc = mocc.shape
    natm = len(atmlst)
    aoslices = mol.aoslice_by_atom()
    mocc_e = mocc * mo_eocc
    mo1 = numpy.asarray([mo1[ja] for ja in atmlst]).reshape(natm*3, nao*nocc)
    mo_e1 = numpy.asarray([mo_e1[ja] for ja in atmlst]).reshape(natm*3, nocc*nocc)
    out = numpy.empty((natm*3, natm*3))
    blksize = max(1, int(max_memory*.2e6/8 / (3*nao*nocc + 3*nocc**2)))
    for a0, a1 in lib.prange(0, natm, blksize):
        # h1ao[A] C - s1ao[A] C e and s1oo[A], where s1ao[A] is s1a[:,p0:p1] in the rows of A
        # plus its transpose
        h1mo = numpy.empty((a1-a0, 3, nao, nocc))
        s1oo = numpy.empty((a1-a0, 3, nocc, nocc))
        for k, ia in enumerate(atmlst[a0:a1]):
            p0, p1 = aoslices[ia][2:]
            s1c = lib.einsum('xpq,qi->xpi', s1a[:,p0:p1], mocc)
            s1oo_k = lib.einsum('pi,xpj->xij', mocc[p0:p1], s1c)
            s1oo[k] = s1oo_k + s1oo_k.transpose(0,2,1)
            h1mo[k] = lib.einsum('xpq,qi->xpi', h1ao[ia], mocc)
            h1mo[k,:,p0:p1] -= lib.einsum('xpq,qi->xpi', s1a[:,p0:p1], mocc_e)
            h1mo[k] -= lib.einsum('xqp,qi->xpi', s1a[:,p0:p1], mocc_e[p0:p1])
        out[a0*3:a1*3] = lib.dot(h1mo.reshape(-1,nao*nocc), mo1.T)
        out[a0*3:a1*3] -= lib.dot(s1oo.reshape(-1,nocc*nocc), mo_e1.T, .5)
    return out.reshape(natm,3,natm,3).transpose(0,2,1,3)

def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    '''Partial derivative
    '''
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

    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    # Energy weighted density matrix
    dme0 = numpy.einsum('pi,qi,i->pq', mocc, mocc, mo_energy[mo_occ>0]) * 2

    hcore_deriv = hessobj.hcore_generator(mol)
    s1aa, s1ab, s1a = get_ovlp(mol)

    # Skip the K-type contractions when no exact exchange is needed (with_k=False)
    ej, ek = _partial_hess_2e(mol, dm0, [dm0], 1, atmlst, with_k=with_k, verbose=log)
    t1 = log.timer_debug1('two-electron terms', *t1)

    aoslices = mol.aoslice_by_atom()
    natm = len(atmlst)
    e1 = numpy.zeros((natm, natm, 3, 3))
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        e1[i0, i0] -= numpy.einsum('xypq,pq->xy', s1aa[:,:,p0:p1], dme0[p0:p1])*2

        for j0, ja in enumerate(atmlst[:i0+1]):
            q0, q1 = aoslices[ja][2:]
            e1[i0, j0] -= numpy.einsum('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], dme0[p0:p1,q0:q1])*2

            h1ao = hcore_deriv(ia, ja)
            e1[i0, j0] += numpy.einsum('xypq,pq->xy', h1ao, dm0)

        for j0 in range(i0):
            e1[j0, i0] = e1[i0, j0].T

    log.timer('RHF partial hessian', *time0)
    return e1, ej, ek

def _partial_hess_2e(mol, dm_j, dms_k, k_fac, atmlst, with_j=True, with_k=True,
                     omega=None, verbose=None):
    '''The two-electron blocks ej, ek of _partial_hess_ejk: the second derivatives of the
    Coulomb energy of dm_j and of the exchange energies of dms_k at fixed densities, each exchange
    term weighted by k_fac (1 for the RHF dm0, 2 for the UHF spin densities). omega selects the
    range-separated Coulomb operator.

    The atom-pair loop reads, for atom A of atmlst, only the rows of the int2e_ip1ip2 and
    int2e_ipvip1 contractions that belong to the atoms before A in atmlst, so only those rows are
    computed. When atmlst holds every atom, the diagonal blocks follow from translational
    invariance, sum_B d2E/dR_A dR_B = 0, without int2e_ipip1. Otherwise ipip1 gives them and the
    rows of A itself are computed as well.
    '''
    log = logger.new_logger(mol, verbose)
    t1 = (logger.process_clock(), logger.perf_counter())
    atmlst = list(atmlst)
    natm = len(atmlst)
    nao = mol.nao
    nbas = mol.nbas
    ej = numpy.zeros((natm, natm, 3, 3))
    ek = numpy.zeros((natm, natm, 3, 3))
    if not with_k:
        dms_k = []
    nset = len(dms_k)
    if not with_j and nset == 0:
        return ej, ek
    by_ti = sorted(atmlst) == list(range(mol.natm))
    if JK_ENERGY and by_ti:
        with mol.with_range_coulomb(omega):
            ej, ek = hess_jk_energy(mol, dm_j if with_j else None, dms_k, k_fac, verbose=log)
        log.timer_debug1('two-electron blocks (hess_jk_energy)', *t1)
        return ej[atmlst][:,atmlst], ek[atmlst][:,atmlst]
    nj = int(with_j)
    if omega is None:
        ipip1_opt = _make_vhfopt(mol, dm_j, 'ipip1', 'int2e_ipip1ipip2')
        ip1ip2_opt = _make_vhfopt(mol, dm_j, 'ip1ip2', 'int2e_ip1ip2')
        ipvip1_opt = _make_vhfopt(mol, dm_j, 'ipvip1', 'int2e_ipvip1ipvip2')
    else:
        ipip1_opt = ip1ip2_opt = ipvip1_opt = None
    aoslices = mol.aoslice_by_atom()

    with mol.with_range_coulomb(omega):
        if not by_ti:
            jk_diag = []
            if with_j:
                jk_diag += ['lk->s1ij', dm_j]     # vj1
            for dm in dms_k:
                jk_diag += ['jk->s1il', dm]       # vk1
            vjk1_diag = _get_jk(mol, 'int2e_ipip1', 9, 's2kl', jk_diag, vhfopt=ipip1_opt)
            vjk1_diag = [v.reshape(3,3,nao,nao) for v in vjk1_diag]
            t1 = log.timer_debug1('contracting int2e_ipip1', *t1)

        for i0, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = aoslices[ia]
            if by_ti:
                atms_b = atmlst[:i0]
            else:
                if with_j:
                    ej[i0, i0] += numpy.einsum('xypq,pq->xy', vjk1_diag[0][:,:,p0:p1], dm_j[p0:p1])*2
                for v, dm in zip(vjk1_diag[nj:], dms_k):
                    ek[i0, i0] += numpy.einsum('xypq,pq->xy', v[:,:,p0:p1], dm[p0:p1]) * k_fac
                atms_b = atmlst[:i0+1]
            if not atms_b:
                continue
            # Rows of the shells up to the last of atms_b (more than atms_b if atmlst is not ordered)
            ksh1 = max(aoslices[ja][1] for ja in atms_b)
            nk = max(aoslices[ja][3] for ja in atms_b)

            jk1 = []
            if with_j:
                jk1 += ['ji->s1kl', dm_j[:,p0:p1]]    # vj1
            for dm in dms_k:
                jk1 += ['li->s1kj', dm[:,p0:p1]]      # vk1
            for dm in dms_k:
                jk1 += ['lj->s1ki', dm]               # vk2
            vjk1 = _get_jk(mol, 'int2e_ip1ip2', 9, 's1', jk1, vhfopt=ip1ip2_opt,
                           shls_slice=(shl0, shl1, 0, nbas, 0, ksh1, 0, nbas))
            t1 = log.timer_debug1('contracting int2e_ip1ip2 for atom %d'%ia, *t1)
            jk2 = []
            if with_j:
                jk2 += ['lk->s1ij', dm_j]             # vj1
            for dm in dms_k:
                jk2 += ['li->s1kj', dm[:,p0:p1]]      # vk1
            vjk2 = _get_jk(mol, 'int2e_ipvip1', 9, 's2kl', jk2, vhfopt=ipvip1_opt,
                           shls_slice=(shl0, shl1, 0, ksh1, 0, nbas, 0, nbas))
            t1 = log.timer_debug1('contracting int2e_ipvip1 for atom %d'%ia, *t1)

            if with_j:
                vj1 = vjk1[0]
                vj1[:,:,p0:p1] += vjk2[0].transpose(0,2,1) * .5
                vj1 = vj1.reshape(3,3,nk,nao)
            vk1 = []
            for s in range(nset):
                vk = vjk1[nj+s]
                vk[:,:,p0:p1] += vjk1[nj+nset+s]
                vk += vjk2[nj+s].transpose(0,2,1)
                vk1.append(vk.reshape(3,3,nk,nao))

            for j0, ja in enumerate(atms_b):
                q0, q1 = aoslices[ja][2:]
                # *2 for +c.c.
                if with_j:
                    ej[i0, j0] += numpy.einsum('xypq,pq->xy', vj1[:,:,q0:q1], dm_j[q0:q1])*4
                for vk, dm in zip(vk1, dms_k):
                    ek[i0, j0] += numpy.einsum('xypq,pq->xy', vk[:,:,q0:q1], dm[q0:q1]) * k_fac

            for j0 in range(i0):
                ej[j0, i0] = ej[i0, j0].T
                ek[j0, i0] = ek[i0, j0].T

    if by_ti:
        for i0 in range(natm):
            ej[i0, i0] = -ej[i0].sum(axis=0)
            ek[i0, i0] = -ek[i0].sum(axis=0)
    return ej, ek

def hess_jk_energy(mol, dm_j, dms_k, k_fac, tol=None, stats=None, verbose=None):
    '''The ej, ek of _partial_hess_2e for every atom [natm,natm,3,3]: the second derivatives of
        E_J = 1/2 sum_pqrs (pq|rs) Dj_pq Dj_rs,   E_K = k_fac/4 sum_n sum_pqrs (pq|rs) Dk[n]_pr Dk[n]_qs
    at fixed densities, contracted with the densities inside the integral loop: no J/K matrices are
    built, each s8 orbit of shell quartets is visited once, and translational invariance spares
    integrals of the orbits whose centres share atoms (lib/vhf/hess_jk_energy.c). dm_j None: no E_J;
    dms_k empty: no E_K. The densities must be symmetric. The operator is the current one of mol
    (mol.with_range_coulomb).

    tol: the screening threshold (JK_ENERGY_TOL); every shell quartet of the integrals of
    CVHFip1ip2_prescreen and CVHFipvip1_prescreen at this threshold is computed. stats, a list, receives
    the counts of the orbits computed, of their integral evaluations, of the cross-atom centre pairs the
    loops of _partial_hess_2e would evaluate for them, and of the orbits through qcint's batched
    integrals.
    '''
    log = logger.new_logger(mol, verbose)
    if tol is None:
        tol = JK_ENERGY_TOL
    libcvhf = _vhf.libcvhf
    natm = mol.natm
    nbas = mol.nbas
    ej = numpy.zeros((natm, natm, 3, 3))
    ek = numpy.zeros((natm, natm, 3, 3))
    dms = [] if dm_j is None else [dm_j]
    dms = numpy.asarray(dms + list(dms_k), order='C', dtype=numpy.double)
    if len(dms) == 0:
        return ej, ek
    dmj = dms[0] if dm_j is not None else None
    dmk = dms[len(dms)-len(dms_k):]
    nk = len(dmk)

    ao_loc = mol.ao_loc_nr()
    c_atm = mol._atm.ctypes
    c_bas = mol._bas.ctypes
    c_env = mol._env.ctypes
    q_cond = numpy.empty((3, nbas, nbas))
    with mol.with_integral_screen(max(tol, 1e-150)**2):
        for q, fqcond, intor in ((q_cond[0], 'CVHFnr_int2e_pp_q_cond', 'int2e_ip1ip2'),
                                 (q_cond[1], 'CVHFnr_int2e_pppp_q_cond', 'int2e_ipvip1ipvip2'),
                                 (q_cond[2], 'CVHFnr_int2e_q_cond', 'int2e')):
            getattr(libcvhf, fqcond)(
                getattr(libcvhf, mol._add_suffix(intor)), lib.c_null_ptr(), q.ctypes,
                ao_loc.ctypes, c_atm, ctypes.c_int(natm), c_bas, ctypes.c_int(nbas), c_env)
    dm_cond = numpy.empty((nbas, nbas))
    libcvhf.CVHFnr_dm_cond1(dm_cond.ctypes, dms.ctypes, ctypes.c_int(len(dms)), ao_loc.ctypes,
                            c_atm, ctypes.c_int(natm), c_bas, ctypes.c_int(nbas), c_env)

    intors = [mol._add_suffix('int2e_'+key) for key in ('ipip1', 'ipvip1', 'ip1ip2')]
    cintopts = [_vhf.make_cintopt(mol._atm, mol._bas, mol._env, intor) for intor in intors]
    c_intors = (ctypes.c_void_p*3)(*[ctypes.cast(getattr(libcvhf, intor), ctypes.c_void_p).value
                                     for intor in intors])
    c_cintopts = (ctypes.c_void_p*3)(*[opt.value for opt in cintopts])
    c_stats = numpy.zeros(4)
    libcvhf.CVHFhess_jk_energy(
        ej.ctypes if dmj is not None else lib.c_null_ptr(), ek.ctypes if nk else lib.c_null_ptr(),
        c_intors, c_cintopts, q_cond.ctypes, dm_cond.ctypes, ctypes.c_double(tol),
        lib.c_null_ptr() if dmj is None else dmj.ctypes, dmk.ctypes, ctypes.c_int(nk),
        ctypes.c_double(k_fac), ctypes.c_size_t(int(JK_ENERGY_BUF * 1e6 / 8)), c_stats.ctypes,
        ctypes.c_int((1 if JK_ENERGY_DOT else 2) if _vhf.INT2E_BATCH else 0), ao_loc.ctypes,
        c_atm, ctypes.c_int(natm), c_bas, ctypes.c_int(nbas), c_env)
    log.debug1('hess_jk_energy: %d shell quartet orbits, %d integral evaluations '
               '(%d cross-atom centre pairs), %d orbits batched', *c_stats)
    if stats is not None:
        stats.extend(c_stats)
    return ej, ek

def _make_vhfopt(mol, dms, key, vhf_intor):
    libcvhf = _vhf.libcvhf
    if not hasattr(libcvhf, vhf_intor):
        return None

    vhfopt = _vhf._VHFOpt(mol, 'int2e_'+key, 'CVHF'+key+'_prescreen',
                          dmcondname='CVHFnr_dm_cond1')
    ao_loc = mol.ao_loc_nr()
    nbas = mol.nbas
    q_cond = numpy.empty((2, nbas, nbas))
    with mol.with_integral_screen(vhfopt.direct_scf_tol**2):
        if vhf_intor == 'int2e_ip1ip2':
            fqcond = libcvhf.CVHFnr_int2e_pp_q_cond
        elif vhf_intor in ('int2e_ipip1ipip2', 'int2e_ipvip1ipvip2'):
            fqcond = libcvhf.CVHFnr_int2e_pppp_q_cond
        else:
            raise NotImplementedError(vhf_intor)
        fqcond(
            getattr(libcvhf, mol._add_suffix(vhf_intor)),
            lib.c_null_ptr(), q_cond[0].ctypes,
            ao_loc.ctypes, mol._atm.ctypes, ctypes.c_int(mol.natm),
            mol._bas.ctypes, ctypes.c_int(nbas), mol._env.ctypes)
        libcvhf.CVHFnr_int2e_q_cond(
            getattr(libcvhf, mol._add_suffix('int2e')),
            lib.c_null_ptr(), q_cond[1].ctypes,
            ao_loc.ctypes, mol._atm.ctypes, ctypes.c_int(mol.natm),
            mol._bas.ctypes, ctypes.c_int(nbas), mol._env.ctypes)
    vhfopt.q_cond = q_cond
    return vhfopt


def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)
    vhfopt = _make_h1_vhfopt(mol, dm0, 'jk')

    aoslices = mol.aoslice_by_atom()
    h1ao = [None] * mol.natm
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        vj1, vj2, vk1, vk2 = _get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                     ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                      'lk->s1ij', -dm0         ,  # vj2
                                      'li->s1kj', -dm0[:,p0:p1],  # vk1
                                      'jk->s1il', -dm0         ], # vk2
                                     shls_slice=shls_slice, vhfopt=vhfopt)
        vhf = vj1 - vk1*.5
        vhf[:,p0:p1] += vj2 - vk2*.5
        h1 = vhf + vhf.transpose(0,2,1)
        h1 += hcore_deriv(ia)
        h1ao[ia] = h1
    return h1ao

def _make_h1_vhfopt(mol, dms, screen='jk'):
    '''The screening of make_h1's int2e_ip1 J/K (_get_jk over s2kl, the scripts ji->s2kl and lk->s1ij
    for J, li->s1kj and jk->s1il for K): CVHFip1_h1_<screen>_prescreen at MAKE_H1_TOL, with the pair
    bounds of the gradients (grad.rhf._calc_q_cond) and the largest elements of dms, the full densities
    the scripts take or take slices of. Within mol.with_range_coulomb for range-separated K. None
    (no screening) when MAKE_H1_TOL is 0.
    '''
    if MAKE_H1_TOL == 0:
        return None
    vhfopt = _vhf._VHFOpt(mol, 'int2e_ip1', 'CVHFip1_h1_%s_prescreen' % screen,
                          direct_scf_tol=MAKE_H1_TOL)
    vhfopt.q_cond = rhf._calc_q_cond(mol, vhfopt)
    nao = mol.nao
    nbas = mol.nbas
    dms = numpy.asarray(dms, order='C', dtype=numpy.double).reshape(-1,nao,nao)
    dm_cond = numpy.empty((nbas, nbas))
    ao_loc = mol.ao_loc_nr()
    _vhf.libcvhf.CVHFnr_dm_cond1(
        dm_cond.ctypes, dms.ctypes, ctypes.c_int(len(dms)), ao_loc.ctypes, mol._atm.ctypes,
        ctypes.c_int(mol.natm), mol._bas.ctypes, ctypes.c_int(nbas), mol._env.ctypes)
    # Set here: direct_bindm's set_dm does nothing without a dmcondname (the scripts' slices cannot be
    # stacked)
    vhfopt.dm_cond = dm_cond
    return vhfopt

def get_hcore(mol):
    '''Part of the second derivatives of core Hamiltonian'''
    h1aa = mol.intor('int1e_ipipkin', comp=9)
    h1ab = mol.intor('int1e_ipkinip', comp=9)
    if mol._pseudo:
        raise NotImplementedError('Nuclear hessian for GTH PP')
    else:
        h1aa+= mol.intor('int1e_ipipnuc', comp=9)
        h1ab+= mol.intor('int1e_ipnucip', comp=9)
    if mol.has_ecp():
        h1aa += mol.intor('ECPscalar_ipipnuc', comp=9)
        h1ab += mol.intor('ECPscalar_ipnucip', comp=9)
    nao = h1aa.shape[-1]
    return h1aa.reshape(3,3,nao,nao), h1ab.reshape(3,3,nao,nao)

def get_ovlp(mol):
    s1a =-mol.intor('int1e_ipovlp', comp=3)
    nao = s1a.shape[-1]
    s1aa = mol.intor('int1e_ipipovlp', comp=9).reshape(3,3,nao,nao)
    s1ab = mol.intor('int1e_ipovlpip', comp=9).reshape(3,3,nao,nao)
    return s1aa, s1ab, s1a

def _get_jk(mol, intor, comp, aosym, script_dms,
            shls_slice=None, cintopt=None, vhfopt=None):
    intor = mol._add_suffix(intor)
    scripts = script_dms[::2]
    dms = script_dms[1::2]
    vs = _vhf.direct_bindm(intor, aosym, scripts, dms, comp,
                           mol._atm, mol._bas, mol._env, vhfopt=vhfopt,
                           cintopt=cintopt, shls_slice=shls_slice)
    for k, script in enumerate(scripts):
        if 's2' in script:
            hermi = 1
        elif 'a2' in script:
            hermi = 2
        else:
            continue

        shape = vs[k].shape
        if shape[-2] == shape[-1]:
            if comp > 1:
                for i in range(comp):
                    lib.hermi_triu(vs[k][i], hermi=hermi, inplace=True)
            else:
                lib.hermi_triu(vs[k], hermi=hermi, inplace=True)
    return vs

def solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1ao,
              fx=None, atmlst=None, max_memory=4000, verbose=None,
              max_cycle=50, level_shift=0):
    '''Solve the first order equation

    Kwargs:
        fx : function(dm_mo) => v1_mo
            A function to generate the induced potential.
            See also the function gen_vind.
    '''
    mol = mf.mol
    log = logger.new_logger(mf, verbose)
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]

    if fx is None:
        fx = gen_vind(mf, mo_coeff, mo_occ)
    s1a = -mol.intor('int1e_ipovlp', comp=3)

    def _ao2mo(mat):
        return numpy.asarray([reduce(numpy.dot, (mo_coeff.T, x, mocc)) for x in mat])

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, max_memory*.9-mem_now)
    blksize = max(2, int(max_memory*1e6/8 / (nmo*nocc*3*6)))
    mo1s = [None] * mol.natm
    e1s = [None] * mol.natm
    aoslices = mol.aoslice_by_atom()
    for ia0, ia1 in lib.prange(0, len(atmlst), blksize):
        s1vo = []
        h1vo = []
        for i0 in range(ia0, ia1):
            ia = atmlst[i0]
            shl0, shl1, p0, p1 = aoslices[ia]
            s1ao = numpy.zeros((3,nao,nao))
            s1ao[:,p0:p1] += s1a[:,p0:p1]
            s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1vo.append(_ao2mo(s1ao))
            h1vo.append(_ao2mo(h1ao[ia]))

        h1vo = numpy.vstack(h1vo)
        s1vo = numpy.vstack(s1vo)
        tol = mf.conv_tol_cpscf * (ia1 - ia0)
        mo1, e1 = cphf.solve(fx, mo_energy, mo_occ, h1vo, s1vo,
                             max_cycle=max_cycle, level_shift=level_shift, tol=tol,
                             verbose=log)
        mo1 = numpy.einsum('pq,xqi->xpi', mo_coeff, mo1).reshape(-1,3,nao,nocc)
        e1 = e1.reshape(-1,3,nocc,nocc)

        for k in range(ia1-ia0):
            ia = atmlst[k+ia0]
            mo1s[ia] = mo1[k]
            e1s[ia] = e1[k].reshape(3,nocc,nocc)
        mo1 = e1 = None
    return mo1s, e1s

def gen_vind(mf, mo_coeff, mo_occ):
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]
    vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)
    # The XC response from the factors of dm1 where the response function offers it
    vind_mo = getattr(vresp, 'vind_mo', None)
    def fx(mo1):
        mo1 = mo1.reshape(-1,nmo,nocc)
        nset = len(mo1)
        dm1 = numpy.empty((nset,nao,nao))
        for i, x in enumerate(mo1):
            dm = reduce(numpy.dot, (mo_coeff, x*2, mocc.T)) # *2 for double occupancy
            dm1[i] = dm + dm.T
        if vind_mo is not None:
            return vind_mo(dm1, mo1*2)
        v1 = vresp(dm1)
        v1vo = numpy.empty_like(mo1)
        for i, x in enumerate(v1):
            v1vo[i] = reduce(numpy.dot, (mo_coeff.T, x, mocc))
        return v1vo
    return fx

def hess_nuc(mol, atmlst=None):
    h = numpy.zeros((mol.natm,mol.natm,3,3))
    qs = numpy.asarray([mol.atom_charge(i) for i in range(mol.natm)])
    rs = numpy.asarray([mol.atom_coord(i) for i in range(mol.natm)])
    for i in range(mol.natm):
        r12 = rs[i] - rs
        s12 = numpy.sqrt(numpy.einsum('ki,ki->k', r12, r12))
        s12[i] = 1e60
        tmp1 = qs[i] * qs / s12**3
        tmp2 = numpy.einsum('k, ki,kj->kij',-3*qs[i]*qs/s12**5, r12, r12)

        h[i,i,0,0] = h[i,i,1,1] = h[i,i,2,2] = -tmp1.sum()
        h[i,i] -= numpy.einsum('kij->ij', tmp2)

        h[i,:,0,0] += tmp1
        h[i,:,1,1] += tmp1
        h[i,:,2,2] += tmp1
        h[i,:] += tmp2

    if atmlst is not None:
        h = h[atmlst][:,atmlst]
    return h


def gen_hop(hobj, mo_energy=None, mo_coeff=None, mo_occ=None, verbose=None):
    log = logger.new_logger(hobj, verbose)
    mol = hobj.mol
    mf = hobj.base

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    natm = mol.natm
    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    nocc = mocc.shape[1]

    atmlst = range(natm)
    max_memory = max(2000, hobj.max_memory - lib.current_memory()[0])
    de2 = hobj.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                 max_memory, log)
    de2 += hobj.hess_nuc()

    h1ao_cache = hobj.make_h1(mo_coeff, mo_occ, None, atmlst, log)

    aoslices = mol.aoslice_by_atom()
    s1a = -mol.intor('int1e_ipovlp', comp=3)

    fvind = gen_vind(mf, mo_coeff, mo_occ)
    def h_op(x):
        x = x.reshape(natm,3)
        hx = numpy.einsum('abxy,ax->by', de2, x)
        h1ao = 0
        s1ao = 0
        for ia in range(natm):
            shl0, shl1, p0, p1 = aoslices[ia]
            h1ao += numpy.einsum('x,xij->ij', x[ia], h1ao_cache[ia])
            s1ao_i = numpy.zeros((3,nao,nao))
            s1ao_i[:,p0:p1] += s1a[:,p0:p1]
            s1ao_i[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1ao += numpy.einsum('x,xij->ij', x[ia], s1ao_i)

        s1vo = reduce(numpy.dot, (mo_coeff.T, s1ao, mocc))
        h1vo = reduce(numpy.dot, (mo_coeff.T, h1ao, mocc))
        mo1, mo_e1 = cphf.solve(fvind, mo_energy, mo_occ, h1vo, s1vo, verbose=log)
        mo1 = numpy.dot(mo_coeff, mo1)
        mo_e1 = mo_e1.reshape(nocc,nocc)
        dm1 = numpy.einsum('pi,qi->pq', mo1, mocc)
        dme1 = numpy.einsum('pi,qi,i->pq', mo1, mocc, mo_energy[mo_occ>0])
        dme1 = dme1 + dme1.T + reduce(numpy.dot, (mocc, mo_e1.T, mocc.T))

        for ja in range(natm):
            q0, q1 = aoslices[ja][2:]
            hx[ja] += numpy.einsum('xpq,pq->x', h1ao_cache[ja], dm1) * 4
            hx[ja] -= numpy.einsum('xpq,pq->x', s1a[:,q0:q1], dme1[q0:q1]) * 2
            hx[ja] -= numpy.einsum('xpq,qp->x', s1a[:,q0:q1], dme1[:,q0:q1]) * 2
        return hx.ravel()

    hdiag = numpy.einsum('aaxx->ax', de2).ravel()
    return h_op, hdiag


@contextlib.contextmanager
def _without_semi_direct(mf, nset):
    '''Release the semi-direct ERIs of the SCF's J/K (SCF.semi_direct), and
    keep get_jk from storing them again for the duration, if the arrays of nset
    perturbations that the Hessian keeps (H1 and the CPHF's density matrices and
    responses, about five matrices each) do not fit under mf.max_memory beside
    them. Its derivative integrals are other ones, and the many-density J/K of
    the CPHF (_vhf._direct_s8_batch) does not read them; only its calls for few
    dms do. The next SCF stores them again.'''
    nao = mf.mol.nao
    need = 5 * nset * nao**2 * 8e-6
    if (not getattr(mf, 'semi_direct', False) or
            lib.current_memory()[0] + need <= mf.max_memory):
        yield
        return
    for vhfopt in getattr(mf, '_opt', {}).values():
        if vhfopt is not None:
            vhfopt.__dict__.pop('semi_direct_eri', None)
    with lib.temporary_env(mf, semi_direct=False):
        yield

class HessianBase(lib.StreamObject):
    '''Non-relativistic restricted Hartree-Fock hessian'''

    # Max. number of iterations for Krylov solver
    max_cycle = 50
    # Shift virtual orbitals to slightly improve the convergence speed of Krylov solver
    # A small level_shift ~ 0.1 is often helpful to decrease 2 - 3 iterations
    # while the error of cphf solver may be increased by one magnitude.
    level_shift = 0

    _keys = {
        'mol', 'base', 'atmlst', 'de', 'max_cycle', 'level_shift'
    }

    def __init__(self, scf_method):
        self.verbose = scf_method.verbose
        self.stdout = scf_method.stdout
        self.mol = scf_method.mol
        self.base = scf_method
        self.max_memory = self.mol.max_memory
        self.atmlst = range(self.mol.natm)
        self.de = numpy.zeros((0,0,3,3))  # (A,B,dR_A,dR_B)

    def hess_elec(self, *args, **kwargs):
        raise NotImplementedError

    def make_h1(self, *args, **kwargs):
        raise NotImplementedError

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        return get_hcore(mol)

    def hcore_generator(self, mol=None):
        if mol is None: mol = self.mol
        with_x2c = getattr(self.base, 'with_x2c', None)
        if with_x2c:
            return with_x2c.hcore_deriv_generator(deriv=2)

        with_ecp = mol.has_ecp()
        if with_ecp:
            ecp_atoms = set(mol._ecpbas[:,gto.ATOM_OF])
        else:
            ecp_atoms = ()
        aoslices = mol.aoslice_by_atom()
        nbas = mol.nbas
        nao = mol.nao_nr()
        h1aa, h1ab = self.get_hcore(mol)
        def get_hcore(iatm, jatm):
            ish0, ish1, i0, i1 = aoslices[iatm]
            jsh0, jsh1, j0, j1 = aoslices[jatm]
            zi = mol.atom_charge(iatm)
            zj = mol.atom_charge(jatm)
            if iatm == jatm:
                with mol.with_rinv_at_nucleus(iatm):
                    rinv2aa = mol.intor('int1e_ipiprinv', comp=9)
                    rinv2ab = mol.intor('int1e_iprinvip', comp=9)
                    rinv2aa *= zi
                    rinv2ab *= zi
                    if with_ecp and iatm in ecp_atoms:
                        rinv2aa -= mol.intor('ECPscalar_ipiprinv', comp=9)
                        rinv2ab -= mol.intor('ECPscalar_iprinvip', comp=9)
                rinv2aa = rinv2aa.reshape(3,3,nao,nao)
                rinv2ab = rinv2ab.reshape(3,3,nao,nao)
                hcore = -rinv2aa - rinv2ab
                hcore[:,:,i0:i1] += h1aa[:,:,i0:i1]
                hcore[:,:,i0:i1] += rinv2aa[:,:,i0:i1]
                hcore[:,:,i0:i1] += rinv2ab[:,:,i0:i1]
                hcore[:,:,:,i0:i1] += rinv2aa[:,:,i0:i1].transpose(0,1,3,2)
                hcore[:,:,:,i0:i1] += rinv2ab[:,:,:,i0:i1]
                hcore[:,:,i0:i1,i0:i1] += h1ab[:,:,i0:i1,i0:i1]

            else:
                hcore = numpy.zeros((3,3,nao,nao))
                hcore[:,:,i0:i1,j0:j1] += h1ab[:,:,i0:i1,j0:j1]
                with mol.with_rinv_at_nucleus(iatm):
                    shls_slice = (jsh0, jsh1, 0, nbas)
                    rinv2aa = mol.intor('int1e_ipiprinv', comp=9, shls_slice=shls_slice)
                    rinv2ab = mol.intor('int1e_iprinvip', comp=9, shls_slice=shls_slice)
                    rinv2aa *= zi
                    rinv2ab *= zi
                    if with_ecp and iatm in ecp_atoms:
                        rinv2aa -= mol.intor('ECPscalar_ipiprinv', comp=9, shls_slice=shls_slice)
                        rinv2ab -= mol.intor('ECPscalar_iprinvip', comp=9, shls_slice=shls_slice)
                    hcore[:,:,j0:j1] += rinv2aa.reshape(3,3,j1-j0,nao)
                    hcore[:,:,j0:j1] += rinv2ab.reshape(3,3,j1-j0,nao).transpose(1,0,2,3)

                with mol.with_rinv_at_nucleus(jatm):
                    shls_slice = (ish0, ish1, 0, nbas)
                    rinv2aa = mol.intor('int1e_ipiprinv', comp=9, shls_slice=shls_slice)
                    rinv2ab = mol.intor('int1e_iprinvip', comp=9, shls_slice=shls_slice)
                    rinv2aa *= zj
                    rinv2ab *= zj
                    if with_ecp and jatm in ecp_atoms:
                        rinv2aa -= mol.intor('ECPscalar_ipiprinv', comp=9, shls_slice=shls_slice)
                        rinv2ab -= mol.intor('ECPscalar_iprinvip', comp=9, shls_slice=shls_slice)
                    hcore[:,:,i0:i1] += rinv2aa.reshape(3,3,i1-i0,nao)
                    hcore[:,:,i0:i1] += rinv2ab.reshape(3,3,i1-i0,nao)
            return hcore + hcore.conj().transpose(0,1,3,2)
        return get_hcore

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
        return solve_mo1(self.base, mo_energy, mo_coeff, mo_occ, h1ao,
                         fx, atmlst, max_memory, verbose,
                         max_cycle=self.max_cycle, level_shift=self.level_shift)

    def hess_nuc(self, mol=None, atmlst=None):
        if mol is None: mol = self.mol
        return hess_nuc(mol, atmlst)

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        if mo_energy is None:
            if self.base.mo_energy is None:
                self.base.run()
            mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        # 3 perturbations of each atom, for each spin of UHF
        nset = 3 * len(atmlst) * (2 if numpy.ndim(mo_occ) == 2 else 1)
        with _without_semi_direct(self.base, nset):
            de = self.hess_elec(mo_energy, mo_coeff, mo_occ, atmlst=atmlst)
        self.de = de + self.hess_nuc(self.mol, atmlst=atmlst)
        if self.base.do_disp():
            self.de += self.get_dispersion()
        return self.de
    hess = kernel

    gen_hop = gen_hop

    # to_gpu can be reused only when __init__ still takes mf
    to_gpu = lib.to_gpu

class Hessian(HessianBase):

    partial_hess_elec = partial_hess_elec
    hess_elec = hess_elec
    make_h1 = make_h1

# Inject to RHF class
from pyscf import scf
scf.hf.RHF.Hessian = lib.class_as_method(Hessian)
scf.rohf.ROHF.Hessian = lib.invalid_method('Hessian')
