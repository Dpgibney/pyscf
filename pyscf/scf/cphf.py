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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Restricted coupled perturbed Hartree-Fock solver
'''


import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger


def solve(fvind, mo_energy, mo_occ, h1, s1=None,
          max_cycle=50, tol=1e-9, hermi=False, verbose=logger.WARN,
          level_shift=0, x0=None):
    '''
    Args:
        fvind : function
            Given density matrix, compute (ij|kl)D_{lk}*2 - (ij|kl)D_{jk}

    Kwargs:
        hermi : boolean
            Whether the matrix defined by fvind is Hermitian or not.
        level_shift : float
            Add to diagonal terms to slightly improve the convergence speed of
            Krylov solver
        x0 : array
            With s1, an initial guess of the first order orbitals, e.g. the
            solution for an approximate fvind (see solve_withs1)
    '''
    if s1 is None:
        return solve_nos1(fvind, mo_energy, mo_occ, h1,
                          max_cycle, tol, hermi, verbose, level_shift)
    else:
        return solve_withs1(fvind, mo_energy, mo_occ, h1, s1,
                            max_cycle, tol, hermi, verbose, level_shift, x0)
kernel = solve

def _krylov_coefficients(xs, x, blksize=64):
    '''c[k,r] with x[r] = sum_k c[k,r] xs[k], for the vectors xs that
    lib.krylov spans its solution x[nset,ndim] with (a list, or lib's _Xlist)'''
    nd = len(xs)
    dtype = numpy.result_type(x, numpy.asarray(xs[0]))
    gram = numpy.empty((nd, nd), dtype=dtype)
    rhs = numpy.empty((nd, len(x)), dtype=dtype)
    def rows(k0, k1):
        return numpy.asarray([numpy.asarray(xs[k]) for k in range(k0, k1)])
    for i0, i1 in lib.prange(0, nd, blksize):
        xi = rows(i0, i1).conj()
        rhs[i0:i1] = lib.dot(xi, x.T)
        for j0, j1 in lib.prange(0, i1, blksize):
            xj = rows(j0, j1)
            gram[i0:i1,j0:j1] = lib.dot(xi, xj.T)
            gram[j0:j1,i0:i1] = gram[i0:i1,j0:j1].T.conj()
    return numpy.linalg.solve(gram, rhs)

def _krylov_combination(c, ax, blksize=64):
    '''sum_k c[k,r] ax[k] for the vectors ax (a list, or lib's _Xlist)'''
    out = 0
    for k0, k1 in lib.prange(0, len(ax), blksize):
        a = numpy.asarray([numpy.asarray(ax[k]) for k in range(k0, k1)])
        out = out + lib.dot(c[k0:k1].T, a)
    return out

# h1 shape is (:,nvir,nocc)
def solve_nos1(fvind, mo_energy, mo_occ, h1,
               max_cycle=50, tol=1e-9, hermi=False, verbose=logger.WARN,
               level_shift=0):
    '''For field independent basis. First order overlap matrix is zero

    Kwargs:
        level_shift : float
            Add to diagonal terms to slightly improve the convergence speed of
            Krylov solver
    '''
    assert not hermi
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    e_a = mo_energy[mo_occ==0]
    e_i = mo_energy[mo_occ>0]
    e_ai = 1 / (e_a[:,None] + level_shift - e_i)
    mo1base = h1 * -e_ai
    nvir, nocc = e_ai.shape

    def vind_vo(mo1):
        mo1 = mo1.reshape(-1, nvir, nocc)
        v = fvind(mo1).reshape(-1, nvir, nocc)
        if level_shift != 0:
            v -= mo1 * level_shift
        v *= e_ai
        return v.reshape(-1, nvir*nocc)
    lindep = max(tol**2, numpy.finfo(mo1base.real.dtype).eps)
    mo1 = lib.krylov(vind_vo, mo1base.reshape(-1, nvir*nocc),
                     tol=tol, max_cycle=max_cycle, lindep=lindep,
                     hermi=hermi, verbose=log)
    log.timer('krylov solver in CPHF', *t0)
    return mo1.reshape(h1.shape), None

# h1 shape is (:,nocc+nvir,nocc)
def solve_withs1(fvind, mo_energy, mo_occ, h1, s1,
                 max_cycle=50, tol=1e-9, hermi=False, verbose=logger.WARN,
                 level_shift=0, x0=None):
    '''For field dependent basis. First order overlap matrix is non-zero.
    The first order orbitals are set to
    C^1_{ij} = -1/2 S1
    e1 = h1 - s1*e0 + (e0_j-e0_i)*c1 + vhf[c1]

    The Krylov solver starts from x0 (by default the uncoupled solution), whose
    occupied block is set to -1/2 S1: the residual and so every Krylov vector
    then have no occupied rows, and the solution keeps that block exactly. The
    response fvind of the solution, which e1 and the last update of the virtual
    block take, is the same linear combination of the responses of x0 and of
    the Krylov vectors, so fvind is not called for it again.

    Kwargs:
        level_shift : float
            Add to diagonal terms to slightly improve the convergence speed of
            Krylov solver
        x0 : array
            An initial guess of the first order orbitals (:,nocc+nvir,nocc),
            e.g. the solution for an approximate fvind

    Returns:
        First order orbital coefficients (in MO basis) and first order orbital
        energy matrix
    '''
    assert not hermi
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    occidx = mo_occ > 0
    viridx = mo_occ == 0
    e_a = mo_energy[viridx]
    e_i = mo_energy[occidx]
    e_ai = 1 / (e_a[:,None] + level_shift - e_i)
    nvir, nocc = e_ai.shape
    nmo = nocc + nvir

    s1 = s1.reshape(-1,nmo,nocc)
    hs = mo1base = h1.reshape(-1,nmo,nocc) - s1*e_i

    mo1base = hs.copy()
    mo1base[:,viridx] *= -e_ai
    mo1base[:,occidx] = -s1[:,occidx] * .5
    nset = mo1base.shape[0]

    v_occ = []  # the occupied rows of fvind of the vectors in the order passed
    def vind_vo(mo1):
        mo1 = mo1.reshape(-1, nmo, nocc)
        v = fvind(mo1).reshape(-1, nmo, nocc)
        v_occ.append(v[:,occidx])
        if level_shift != 0:
            v -= mo1 * level_shift
        v[:,viridx,:] *= e_ai
        v[:,occidx,:] = 0
        return v.reshape(-1, nmo*nocc)

    if x0 is None:
        mo1_0 = mo1base.copy()
    else:
        mo1_0 = numpy.array(x0, dtype=mo1base.dtype).reshape(nset, nmo, nocc)
        mo1_0[:,occidx] = mo1base[:,occidx]
    mo1_0 = mo1_0.reshape(nset, nmo*nocc)
    av0 = vind_vo(mo1_0)
    v_occ0 = v_occ.pop()
    space = {}
    def keep_space(cycle, xs, ax):
        space['xs'] = xs
        space['ax'] = ax
    lindep = max(tol**2, numpy.finfo(mo1base.real.dtype).eps)
    mo1 = lib.krylov(vind_vo, mo1base.reshape(nset, nmo*nocc) - (mo1_0 + av0),
                     tol=tol, max_cycle=max_cycle, lindep=lindep,
                     hermi=hermi, callback=keep_space, verbose=log)
    mo1 = mo1.reshape(nset, nmo*nocc)
    if space:
        c = _krylov_coefficients(space['xs'], mo1)
        av0 += _krylov_combination(c, space['ax'])
        v_occ0 += numpy.einsum('kr,kij->rij', c, numpy.vstack(v_occ))
    mo1 += mo1_0
    mo1 = mo1.reshape(nset, nmo, nocc)
    log.timer('krylov solver in CPHF', *t0)

    # fvind(mo1): its virtual rows from the solver's products, which are
    # e_ai * (fvind - level_shift * mo1)
    v1 = numpy.empty_like(mo1)
    v1[:,viridx] = av0.reshape(nset, nmo, nocc)[:,viridx] / e_ai
    if level_shift != 0:
        v1[:,viridx] += mo1[:,viridx] * level_shift
    v1[:,occidx] = v_occ0
    hs += v1
    mo1[:,viridx] = hs[:,viridx] / (e_i - e_a[:,None])

    # mo_e1 has the same symmetry as the first order Fock matrix (hermitian or
    # anti-hermitian). mo_e1 = v1mo - s1*lib.direct_sum('i+j->ij',e_i,e_i)
    mo_e1 = hs[:,occidx,:]
    mo_e1 += mo1[:,occidx] * (e_i[:,None] - e_i)

    if h1.ndim == 3:
        return mo1, mo_e1
    else:
        assert h1.ndim == 2
        return mo1[0], mo_e1[0]

def _orthonormal_rows(t, vs, drop):
    '''Orthonormal rows spanning the part of the rows of t orthogonal to the
    rows of the blocks vs, without the directions whose norm is below drop,
    and the size of each row in t (the diagonal of the pivoted QR).
    t is overwritten.'''
    for _ in range(2):
        for v in vs:
            t -= lib.dot(lib.dot(t, v.T), v)
    q, r = scipy.linalg.qr(t.T, mode='economic', pivoting=True)[:2]
    sizes = abs(r.diagonal())
    keep = sizes > drop
    q = q[:,keep].T
    # the rows are combinations of t's, by large coefficients for near-dependent t
    for v in vs:
        q -= lib.dot(lib.dot(q, v.T), v)
    return q, sizes[keep]

def _combine_blocks(c, blocks):
    '''sum_k c[k,r] vectors[k] of the vectors in the list of blocks [nk,...]'''
    out = 0
    p0 = 0
    for b in blocks:
        p1 = p0 + len(b)
        out = out + lib.dot(c[p0:p1].T, b)
        p0 = p1
    return out

def solve_residual(fvind, mo_energy, mo_occ, h1, s1, max_cycle=50, tol=1e-4,
                   verbose=logger.WARN, level_shift=0, scale_trials=False):
    '''The equations of solve_withs1 for real orbitals and perturbations, solved
    until the true residual of every perturbation is below tol.

    The virtual block of the solution is x0 + V c, x0 the uncoupled solution
    and V one space of trial vectors for all perturbations. With the operator
    M = (e_a - e_i) + fvind on the virtual block, which is symmetric, the
    coefficients solve V^T M V c = V^T r0 for r0 the residual of x0, in the
    plain dot product, and the virtual block then gets the last update of
    solve_withs1, from the response of the solution (a combination of the
    responses already computed). The error of a Hessian built from this
    solution goes about as the square of the residuals (measured: tol 1e-3
    about 1e-6, tol 1e-4 about 1e-8), so a loose tol is enough for it; quantities
    linear in the solution would need a tighter one.

    A cycle adds the preconditioned residuals of the perturbations that are
    not converged, made orthonormal against V and among themselves. Unlike
    lib.krylov, which stops on the sizes of Krylov vectors that shrink by orders
    of magnitude, the stop and the choice of vectors look at the residuals.
    By default fvind gets densities of about unit size: the thresholds of its
    integrals (direct SCF, the seminumerical K of SGX) are absolute, and
    would otherwise mean less and less in the later cycles. That is what the
    approximate K of SGX needs; with scale_trials it is given up for the speed
    of exact integrals.

    Kwargs:
        tol : float
            Upper limit of the norm of the preconditioned residual,
            r / (e_a - e_i), of each perturbation
        max_cycle : int
            Max. number of cycles of trial vectors
        level_shift : float
            Added to (e_a - e_i) in the preconditioner
        scale_trials : bool
            Give fvind each trial vector at its size in the residuals (not
            below 1e-6), and divide the response by that size. fvind is
            linear, so only what the density screening of its integrals can
            skip changes: the later cycles cost less. Only for a fvind whose
            absolute thresholds are far below 1e-6, such as direct SCF's

    Returns:
        First order orbital coefficients (in MO basis) and first order orbital
        energy matrix
    '''
    assert not (numpy.iscomplexobj(h1) or numpy.iscomplexobj(s1))
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    occidx = mo_occ > 0
    viridx = mo_occ == 0
    e_i = mo_energy[occidx]
    e_a = mo_energy[viridx]
    e_ai = e_a[:,None] - e_i
    nvir, nocc = e_ai.shape
    nmo = nocc + nvir
    ndim = nvir * nocc
    diag = e_ai.ravel()
    precond = diag + level_shift

    s1 = s1.reshape(-1,nmo,nocc)
    hs = h1.reshape(-1,nmo,nocc) - s1*e_i
    nset = hs.shape[0]

    mo1 = numpy.empty_like(hs)
    mo1[:,viridx] = -hs[:,viridx] / e_ai
    mo1[:,occidx] = -s1[:,occidx] * .5
    v1 = fvind(mo1).reshape(nset,nmo,nocc)
    # -(e_ai x0 + fvind(x0) + hs) on the virtual block
    r0 = -v1[:,viridx].reshape(nset,ndim)

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

        x1, sizes = _orthonormal_rows(r[active] / precond, vs, drop)
        nq = len(x1)
        if nq == 0:
            log.warn('CPHF stopped with no new trial vector; residual %g '
                     'above tol %g', rnorm.max(), tol)
            break
        scale = numpy.maximum(sizes, 1e-6).reshape(nq,1,1) if scale_trials else 1.
        trial = numpy.zeros((nq,nmo,nocc))
        trial[:,viridx] = x1.reshape(nq,nvir,nocc) * scale
        w = fvind(trial).reshape(nq,nmo,nocc) / scale
        trial = None
        mx1 = w[:,viridx].reshape(nq,ndim) + x1 * diag
        voo.append(w[:,occidx].reshape(nq,nocc*nocc))
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
        r = r0 - _combine_blocks(c, mvs)

    nvec = vtmv.shape[0]
    log.info('CPHF: %d cycles, %d trial vectors, max residual %.3g',
             cycle, nvec, numpy.linalg.norm(r / precond, axis=1).max(initial=0))
    if vs:
        xv = _combine_blocks(c, vs)
        mo1[:,viridx] += xv.reshape(nset,nvir,nocc)
        # fvind(mo1), from the responses of x0 and of the trial vectors
        v1[:,viridx] += (_combine_blocks(c, mvs) - xv * diag).reshape(nset,nvir,nocc)
        v1[:,occidx] += _combine_blocks(c, voo).reshape(nset,nocc,nocc)
    log.timer('CPHF solver', *t0)

    hs += v1
    mo1[:,viridx] = hs[:,viridx] / (e_i - e_a[:,None])
    mo_e1 = hs[:,occidx,:]
    mo_e1 += mo1[:,occidx] * (e_i[:,None] - e_i)

    if h1.ndim == 3:
        return mo1, mo_e1
    else:
        assert h1.ndim == 2
        return mo1[0], mo_e1[0]
