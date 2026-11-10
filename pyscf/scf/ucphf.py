#!/usr/bin/env python
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

'''
Unrestricted coupled perturbed Hartree-Fock solver
'''


import numpy
from pyscf import lib
from pyscf.lib import logger


def solve(fvind, mo_energy, mo_occ, h1, s1=None,
          max_cycle=50, tol=1e-9, hermi=False, verbose=logger.WARN,
          level_shift=0):
    '''
    Args:
        fvind : function
            Given density matrix, compute (ij|kl)D_{lk}*2 - (ij|kl)D_{jk}
    '''
    if s1 is None:
        return solve_nos1(fvind, mo_energy, mo_occ, h1,
                          max_cycle, tol, hermi, verbose, level_shift)
    else:
        return solve_withs1(fvind, mo_energy, mo_occ, h1, s1,
                            max_cycle, tol, hermi, verbose, level_shift)
kernel = solve

# h1 shape is (:,nvir,nocc)
def solve_nos1(fvind, mo_energy, mo_occ, h1,
               max_cycle=20, tol=1e-9, hermi=False, verbose=logger.WARN,
               level_shift=0):
    '''For field independent basis. First order overlap matrix is zero'''
    assert not hermi
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb
    nocca = numpy.count_nonzero(occidxa)
    noccb = numpy.count_nonzero(occidxb)
    nvira = mo_occ[0].size - nocca
    nvirb = mo_occ[1].size - noccb
    mo_ea, mo_eb = mo_energy
    e_ai = numpy.hstack(
        ((mo_ea[viridxa,None]+level_shift - mo_ea[occidxa]).ravel(),
         (mo_eb[viridxb,None]+level_shift - mo_eb[occidxb]).ravel()))
    e_ai = 1 / e_ai
    nova = nvira * nocca
    novb = nvirb * noccb
    nseta = 1 if h1[0].ndim == 2 else h1[0].shape[0]
    nsetb = 1 if h1[1].ndim == 2 else h1[1].shape[0]
    nset = max(nseta, nsetb)
    h1a = numpy.zeros((nset, nova))
    h1b = numpy.zeros((nset, novb))
    if nova > 0:
        h1a[:nseta] = h1[0].reshape(nseta,nova)
    if novb > 0:
        h1b[:nsetb] = h1[1].reshape(nsetb,novb)
    mo1base = numpy.hstack((h1a, h1b))
    mo1base *= -e_ai
    nov = e_ai.size
    if nov == 0:
        mo1_a = numpy.zeros((nset,nvira,nocca))
        mo1_b = numpy.zeros((nset,nvirb,noccb))
        if isinstance(h1[0], numpy.ndarray) and h1[0].ndim == 2:
            mo1 = (mo1_a[0], mo1_b[0])
        else:
            assert h1[0].ndim == 3
            mo1 = (mo1_a, mo1_b)
        return mo1, None

    def vind_vo(mo1):
        nd = mo1.shape[0]
        v = fvind(mo1).reshape(nd, nov)
        if level_shift != 0:
            v -= mo1 * level_shift
        v *= e_ai
        return v.reshape(-1, nov)
    lindep = max(tol**2, numpy.finfo(mo1base.real.dtype).eps)
    mo1 = lib.krylov(vind_vo, mo1base.reshape(-1, nov),
                     tol=tol, max_cycle=max_cycle, lindep=lindep,
                     hermi=hermi, verbose=log)
    log.timer('krylov solver in CPHF', *t0)

    mo1 = mo1.reshape(mo1base.shape)
    mo1_a = mo1[:,:nova].reshape(nset,nvira,nocca)
    mo1_b = mo1[:,nova:].reshape(nset,nvirb,noccb)
    if isinstance(h1[0], numpy.ndarray) and h1[0].ndim == 2:
        mo1 = (mo1_a[0], mo1_b[0])
    else:
        assert h1[0].ndim == 3
        mo1 = (mo1_a, mo1_b)
    return mo1, None

# h1 shape is (:,nvir+nocc,nocc)
def solve_withs1(fvind, mo_energy, mo_occ, h1, s1,
                 max_cycle=50, tol=1e-9, hermi=False, verbose=logger.WARN,
                 level_shift=0):
    '''For field dependent basis. First order overlap matrix is non-zero.
    The first order orbitals are set to
    C^1_{ij} = -1/2 S1
    e1 = h1 - s1*e0 + (e0_j-e0_i)*c1 + vhf[c1]
    '''
    assert not hermi
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    occidxa = mo_occ[0] > 0
    occidxb = mo_occ[1] > 0
    viridxa = ~occidxa
    viridxb = ~occidxb
    nocca = numpy.count_nonzero(occidxa)
    noccb = numpy.count_nonzero(occidxb)
    nmoa, nmob = mo_occ[0].size, mo_occ[1].size
    ei_a = mo_energy[0][occidxa]
    ei_b = mo_energy[1][occidxb]
    ea_a = mo_energy[0][viridxa]
    ea_b = mo_energy[1][viridxb]
    eai_a = 1. / (ea_a[:,None] + level_shift - ei_a)
    eai_b = 1. / (ea_b[:,None] + level_shift - ei_b)
    s1_a = s1[0].reshape(-1,nmoa,nocca)
    nset = s1_a.shape[0]
    s1_b = s1[1].reshape(nset,nmob,noccb)
    hs_a = h1[0].reshape(nset,nmoa,nocca) - s1_a * ei_a
    hs_b = h1[1].reshape(nset,nmob,noccb) - s1_b * ei_b

    mo1base_a = hs_a.copy()
    mo1base_b = hs_b.copy()
    mo1base_a[:,viridxa] *= -eai_a
    mo1base_b[:,viridxb] *= -eai_b
    mo1base_a[:,occidxa] = -s1_a[:,occidxa] * .5
    mo1base_b[:,occidxb] = -s1_b[:,occidxb] * .5
    mo1base = numpy.hstack((mo1base_a.reshape(nset,-1), mo1base_b.reshape(nset,-1)))
    nov = nocca * nmoa + noccb * nmob

    def vind_vo(mo1):
        nd = mo1.shape[0]
        mo1 = mo1.reshape(nd, nov)
        v = fvind(mo1).reshape(nd, nov)
        if level_shift != 0:
            v -= mo1 * level_shift
        v1a = v[:,:nmoa*nocca].reshape(nd,nmoa,nocca)
        v1b = v[:,nmoa*nocca:].reshape(nd,nmob,noccb)
        v1a[:,viridxa] *= eai_a
        v1b[:,viridxb] *= eai_b
        v1a[:,occidxa] = 0
        v1b[:,occidxb] = 0
        return v.reshape(nd, nov)
    lindep = max(tol**2, numpy.finfo(mo1base.real.dtype).eps)
    mo1 = lib.krylov(vind_vo, mo1base.reshape(-1, nov),
                     tol=tol, max_cycle=max_cycle, lindep=lindep,
                     hermi=hermi, verbose=log)
    mo1 = mo1.reshape(mo1base.shape)
    mo1_a = mo1[:,:nmoa*nocca].reshape(nset,nmoa,nocca)
    mo1_b = mo1[:,nmoa*nocca:].reshape(nset,nmob,noccb)
    mo1_a[:,occidxa] = mo1base_a[:,occidxa]
    mo1_b[:,occidxb] = mo1base_b[:,occidxb]
    log.timer('krylov solver in CPHF', *t0)

    v1mo = fvind(mo1).reshape(mo1base.shape)
    hs_a += v1mo[:,:nmoa*nocca].reshape(nset,nmoa,nocca)
    hs_b += v1mo[:,nmoa*nocca:].reshape(nset,nmob,noccb)
    mo1_a[:,viridxa] = hs_a[:,viridxa] / (ei_a - ea_a[:,None])
    mo1_b[:,viridxb] = hs_b[:,viridxb] / (ei_b - ea_b[:,None])

    mo_e1_a = hs_a[:,occidxa]
    mo_e1_b = hs_b[:,occidxb]
    mo_e1_a += mo1_a[:,occidxa] * (ei_a[:,None] - ei_a)
    mo_e1_b += mo1_b[:,occidxb] * (ei_b[:,None] - ei_b)

    if isinstance(h1[0], numpy.ndarray) and h1[0].ndim == 2:
        mo1_a, mo1_b = mo1_a[0], mo1_b[0]
        mo_e1_a, mo_e1_b = mo_e1_a[0], mo_e1_b[0]
    else:
        assert h1[0].ndim == 3
    return (mo1_a, mo1_b), (mo_e1_a, mo_e1_b)


def solve_residual(fvind, mo_energy, mo_occ, h1, s1, max_cycle=50, tol=1e-4,
                   verbose=logger.WARN, level_shift=0, scale_trials=False):
    '''cphf.solve_residual for unrestricted orbitals: the equations of
    solve_withs1 for real orbitals and perturbations, solved until the true
    residual of every perturbation is below tol. The trial vectors are those of
    the alpha and beta virtual blocks together, on which the operator
    (e_a - e_i) + fvind is symmetric. See cphf.solve_residual for the method
    and the arguments.

    Returns:
        First order orbital coefficients (in MO basis) and first order orbital
        energy matrix, each for alpha and beta
    '''
    from pyscf.scf import cphf
    assert not any(numpy.iscomplexobj(x) for x in (h1[0], h1[1], s1[0], s1[1]))
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    occidx = [mo_occ[0] > 0, mo_occ[1] > 0]
    viridx = [~occidx[0], ~occidx[1]]
    e_i = [mo_energy[k][occidx[k]] for k in range(2)]
    e_a = [mo_energy[k][viridx[k]] for k in range(2)]
    e_ai = [e_a[k][:,None] - e_i[k] for k in range(2)]
    nocc = [e_ai[k].shape[1] for k in range(2)]
    nvir = [e_ai[k].shape[0] for k in range(2)]
    nmo = [mo_occ[0].size, mo_occ[1].size]
    ndim = [nvir[k] * nocc[k] for k in range(2)]
    diag = numpy.hstack([e_ai[0].ravel(), e_ai[1].ravel()])
    precond = diag + level_shift

    one_set = isinstance(h1[0], numpy.ndarray) and h1[0].ndim == 2
    s1 = [s1[k].reshape(-1,nmo[k],nocc[k]) for k in range(2)]
    nset = s1[0].shape[0]
    hs = [h1[k].reshape(nset,nmo[k],nocc[k]) - s1[k]*e_i[k] for k in range(2)]

    def pack(xa, xb):
        return numpy.hstack((xa.reshape(len(xa),-1), xb.reshape(len(xb),-1)))
    def unpack(x):
        n = len(x)
        na = nmo[0] * nocc[0]
        return [x[:,:na].reshape(n,nmo[0],nocc[0]), x[:,na:].reshape(n,nmo[1],nocc[1])]
    def vir_rows(x):
        return pack(x[0][:,viridx[0]], x[1][:,viridx[1]])
    def occ_rows(x):
        return pack(x[0][:,occidx[0]], x[1][:,occidx[1]])
    def add_rows(x, idx, rows):
        # rows of the alpha and beta blocks together, added to x[k][:,idx[k]]
        p0 = 0
        for k in range(2):
            shape = x[k][:,idx[k]].shape
            p1 = p0 + shape[1] * shape[2]
            x[k][:,idx[k]] += rows[:,p0:p1].reshape(shape)
            p0 = p1

    mo1 = [numpy.empty_like(hs[k]) for k in range(2)]
    for k in range(2):
        mo1[k][:,viridx[k]] = -hs[k][:,viridx[k]] / e_ai[k]
        mo1[k][:,occidx[k]] = -s1[k][:,occidx[k]] * .5
    v1 = unpack(numpy.array(fvind(pack(*mo1)).reshape(nset,-1)))
    # -(e_ai x0 + fvind(x0) + hs) on the virtual blocks
    r0 = -vir_rows(v1)

    vs = []  # the trial vectors V, a block per cycle
    mvs = []  # M V
    voo = []  # the occupied rows of fvind(V), for the first order energies
    vtmv = numpy.zeros((0,0))
    vtr0 = numpy.zeros((0,nset))
    c = vtr0
    r = r0
    drop = max(tol*1e-2, 1e-13)
    for cycle in range(max_cycle+1):
        rnorm = numpy.linalg.norm(r / precond, axis=1)
        active = rnorm > tol
        log.debug('CPHF cycle %d: max residual %g, %d of %d perturbations left',
                  cycle, rnorm.max(initial=0), active.sum(), nset)
        if not active.any():
            break
        if cycle == max_cycle:
            raise RuntimeError('CPHF solver failed to converge.')

        x1, sizes = cphf._orthonormal_rows(r[active] / precond, vs, drop)
        nq = len(x1)
        if nq == 0:
            log.warn('CPHF stopped with no new trial vector; residual %g '
                     'above tol %g', rnorm.max(), tol)
            break
        scale = numpy.maximum(sizes, 1e-6).reshape(nq,1) if scale_trials else 1.
        trial = [numpy.zeros((nq,nmo[k],nocc[k])) for k in range(2)]
        add_rows(trial, viridx, x1 * scale)
        w = unpack(numpy.array(fvind(pack(*trial)).reshape(nq,-1)) / scale)
        trial = None
        mx1 = vir_rows(w) + x1 * diag
        voo.append(occ_rows(w))
        w = None
        vs.append(x1)
        mvs.append(mx1)

        nd = vtmv.shape[0]
        vtmv1 = numpy.empty((nd+nq,nd+nq))
        vtmv1[:nd,:nd] = vtmv
        p0 = 0
        for v, mv in zip(vs, mvs):
            p1 = p0 + len(v)
            vtmv1[p0:p1,nd:] = lib.dot(v, mx1.T)
            vtmv1[nd:,p0:p1] = lib.dot(x1, mv.T)
            p0 = p1
        vtmv = vtmv1
        vtr0 = numpy.vstack((vtr0, lib.dot(x1, r0.T)))
        c = numpy.linalg.solve(vtmv, vtr0)
        r = r0 - cphf._combine_blocks(c, mvs)

    nvec = vtmv.shape[0]
    log.info('CPHF: %d cycles, %d trial vectors, max residual %.3g',
             cycle, nvec, numpy.linalg.norm(r / precond, axis=1).max(initial=0))
    if vs:
        xv = cphf._combine_blocks(c, vs)
        add_rows(mo1, viridx, xv)
        # fvind(mo1), from the responses of x0 and of the trial vectors
        add_rows(v1, viridx, cphf._combine_blocks(c, mvs) - xv * diag)
        add_rows(v1, occidx, cphf._combine_blocks(c, voo))
    log.timer('CPHF solver', *t0)

    mo_e1 = []
    for k in range(2):
        hs[k] += v1[k]
        mo1[k][:,viridx[k]] = hs[k][:,viridx[k]] / (e_i[k] - e_a[k][:,None])
        e1 = hs[k][:,occidx[k]]
        e1 += mo1[k][:,occidx[k]] * (e_i[k][:,None] - e_i[k])
        mo_e1.append(e1)

    if one_set:
        return (mo1[0][0], mo1[1][0]), (mo_e1[0][0], mo_e1[1][0])
    return (mo1[0], mo1[1]), (mo_e1[0], mo_e1[1])

