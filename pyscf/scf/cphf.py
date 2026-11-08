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
