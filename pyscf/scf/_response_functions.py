#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
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
Generate SCF response functions
'''

import warnings
import numpy
from functools import reduce
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, rohf, uhf, ghf, dhf, _vhf

def _cphf_chunk(mf, nset, nao, with_j, with_k, direct=None):
    '''The dms that a CPHF response holds at a time for a J/K call (with_j,
    with_k) of direct SCF: as many whole batches of that call as their dms and
    AO matrices (4 per dm) fit in half of the memory left under max_memory, at
    least one, so that a chunk costs no extra pass over the integrals, and 16
    dms for its many-dm kernels. The call sizes its batch with the chunk's dms
    in RSS. direct: whether the call is made by such passes, by default if mf
    is neither density-fitted nor SGX. Otherwise all dms at once.'''
    if direct is None:
        direct = not getattr(mf, 'with_df', None)
    if not direct:
        return nset
    left = (mf.max_memory - lib.current_memory()[0]) * .5e6
    n = int(left / (4 * 8 * nao**2))
    batch = _vhf._direct_jk_batch(nset, nao, with_j, with_k, True)
    nbatch = max(1, n // batch)
    batch = _vhf._direct_jk_batch(nset, nao, with_j, with_k, True,
                                  nbatch * batch * 8 * nao**2 / 1e6)
    return min(nset, max(nbatch * batch, 16))

def _gen_rhf_response(mf, mo_coeff=None, mo_occ=None, *, dm0=None,
                      singlet=None, hermi=0, max_memory=None, with_nlc=True,
                      grids=None):
    '''Generate a function to compute the product of RHF response function and
    RHF density matrices.

    Kwargs:
        singlet (None or boolean) : If singlet is None, response function for
            orbital hessian or CPHF will be generated. If singlet is boolean,
            it is used in TDDFT response kernel.
        with_nlc (boolean) : NLC contribution is typically very small. This flag
        allows to skip NLC contribution.
        grids (Grids object) : Grids for the XC response kernel of DFT
            functionals. If not specified, the secondary grid
            ``mf.second_grids`` is used if available, otherwise ``mf.grids``.
    '''
    assert isinstance(mf, hf.RHF) and not isinstance(mf, (uhf.UHF, rohf.ROHF))

    if dm0 is None:
        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if isinstance(mf, hf.KohnShamDFT):
        ni = mf._numint
        if grids is None:
            grids = getattr(mf, 'second_grids', None)
        if grids is None:
            grids = mf.grids
        if grids.coords is None:
            grids.build(with_non0tab=True)
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        if singlet is None: # for ground state orbital hessian
            spin = 0
        else:
            spin = 1
        if mo_coeff is not None and mo_occ is not None:
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, mf.xc,
                                                mo_coeff, mo_occ, spin)
        else:
            rho0, vxc, fxc = ni.cache_xc_kernel1(mol, grids, mf.xc,
                                                 dm0, spin)

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        if not with_nlc and mf.do_nlc():
            logger.warn(mf, "NLC contribution in gen_response is NOT included")

        triplet = singlet is not None and not singlet

        def acc(v1, v, own=True):
            '''v1 + v, in place in v1; for v1=None, v itself if own (no other
            result shares its memory), else a copy'''
            if v1 is None:
                return v if own else v.copy()
            v1 += v
            return v1

        if not triplet:
            # full-range J and K in one call (and long-range K in another one)
            jk_together = hybrid and (omega == 0 or (alpha != 0 and hyb != 0))
            # J/K calls of the K alone
            k_alone = hybrid and (not jk_together or
                                  (omega != 0 and alpha != 0 and hyb != 0))

            def vind_j(dm1, v1=None):
                '''The response of the J/K call with J, the exchange of that call
                included, added to v1, or in the memory of J for v1=None'''
                if jk_together:
                    vj, vk = mf.get_jk(mol, dm1, hermi)
                    vk *= -.5 * hyb
                    own = not numpy.may_share_memory(vj, vk)
                    if hermi != 2:
                        v1 = acc(v1, vj, own)
                    vj = None
                    v1 = acc(v1, vk, own)
                elif hermi != 2:
                    v1 = acc(v1, mf.get_j(mol, dm1, hermi=hermi))
                return v1

            def vind_k(dm1, v1=None):
                '''The response of the J/K calls of the K alone added to v1'''
                if hybrid and not jk_together:
                    if alpha == 0: # LR=0, only SR exchange
                        vk = mf.get_k(mol, dm1, hermi, omega=-omega)
                        vk *= -.5 * hyb
                    else: # SR=0, only LR exchange
                        vk = mf.get_k(mol, dm1, hermi, omega=omega)
                        vk *= -.5 * alpha
                    v1 = acc(v1, vk)
                    vk = None
                if hybrid and omega != 0 and alpha != 0 and hyb != 0:
                    # SR and LR exchange with different ratios
                    vk = mf.get_k(mol, dm1, hermi, omega=omega)
                    vk *= -.5 * (alpha-hyb)
                    v1 = acc(v1, vk)
                return v1

            def vind_jk(dm1, v1=None):
                '''The Coulomb and exchange response of dm1 added to v1, or in
                the memory of J for v1=None. Each J or K stack is scaled in place
                and added before the next one is made, so that at most two are
                alive beside dm1 and v1.'''
                v1 = vind_k(dm1, vind_j(dm1, v1))
                if v1 is None:
                    v1 = numpy.zeros_like(dm1)
                return v1

            def k_mo_from_factors(a, mocc):
                '''The exchange response in the MO basis, mo_coeff^T v_k mocc, of the
                dms a[x] mocc^T + mocc a[x]^T, from SGX's K of many densities
                (SGX.get_k_mo), which needs no AO matrix of any of them; None
                if there is no such path for these dms.'''
                with_df = getattr(mf, 'with_df', None)
                if not hybrid or hermi != 1 or not hasattr(with_df, 'get_k_mo'):
                    return None
                # as vind_j and vind_k: (range parameter, coefficient)
                if omega == 0:
                    terms = ((None, -.5 * hyb),)
                elif alpha == 0: # LR=0, only SR exchange
                    terms = ((-omega, -.5 * hyb),)
                elif hyb == 0: # SR=0, only LR exchange
                    terms = ((omega, -.5 * alpha),)
                else: # SR and LR exchange with different ratios
                    terms = ((None, -.5 * hyb), (omega, -.5 * (alpha-hyb)))
                vk = 0
                for om, coeff in terms:
                    v = with_df.get_k_mo(a, mocc, mo_coeff, mf.direct_scf_tol, om)
                    if v is None:
                        return None
                    v *= coeff
                    vk = vk + v
                return vk
        else:
            def vind_jk(dm1, v1=None):
                '''The exchange response of dm1 (triplets have no Coulomb
                response) added to v1, or in the memory of K for v1=None, each K
                stack added before the next one is made'''
                if hybrid:
                    if omega == 0:
                        vk = mf.get_k(mol, dm1, hermi)
                        vk *= -.5 * hyb
                    elif alpha == 0: # LR=0, only SR exchange
                        vk = mf.get_k(mol, dm1, hermi, omega=-omega)
                        vk *= -.5 * hyb
                    elif hyb == 0: # SR=0, only LR exchange
                        vk = mf.get_k(mol, dm1, hermi, omega=omega)
                        vk *= -.5 * alpha
                    else: # SR and LR exchange with different ratios
                        vk = mf.get_k(mol, dm1, hermi)
                        vk *= -.5 * hyb
                    v1 = acc(v1, vk)
                    vk = None
                    if omega != 0 and alpha != 0 and hyb != 0:
                        vk = mf.get_k(mol, dm1, hermi, omega=omega)
                        vk *= -.5 * (alpha-hyb)
                        v1 = acc(v1, vk)
                if v1 is None:
                    v1 = numpy.zeros_like(dm1)
                return v1

        def vind_nlc(dm1, v1):
            # The NLC fxc of triplets is 0
            if with_nlc and mf.do_nlc() and not triplet:
                # Cannot import at top due to circular dependency
                from pyscf.hessian.rks import get_vnlc_resp, get_vnlc_resp1
                if mo_coeff is not None and mo_occ is not None:
                    v1 = get_vnlc_resp(mf, mol, mo_coeff, mo_occ, dm1, max_memory, out=v1)
                else:
                    v1 = get_vnlc_resp1(mf, mol, dm0, dm1, max_memory, out=v1)
            return v1

        if singlet is None:
            # Without specify singlet, used in ground state orbital hessian
            def vind_xc(dm1):
                return ni.nr_rks_fxc(mol, grids, mf.xc, dm0, dm1, 0, hermi,
                                     rho0, vxc, fxc, max_memory=max_memory)
        else:
            fxc *= .5
            def vind_xc(dm1):
                # nr_rks_fxc_st requires alpha of dm1, dm1*.5 should be scaled
                return ni.nr_rks_fxc_st(mol, grids, mf.xc, dm0, dm1, hermi, not triplet,
                                        rho0, vxc, fxc, max_memory=max_memory)

        def vind(dm1):
            if hermi == 2:
                return vind_jk(dm1)
            # J and K first, in memory of their own that the XC and VV10
            # responses are then added to
            v1 = vind_nlc(dm1, vind_jk(dm1))
            v1 += vind_xc(dm1)
            return v1

        # The XC response of dms whose symmetric parts are a[k] Q^T + Q a[k]^T,
        # as in CPHF (a = mo_coeff x, Q = mocc) and TDDFT (TDA: a = orbv z^T,
        # Q = orbo), can take the factors instead of the dms, with fewer flops
        # where Q has few columns next to the AOs of a grid group.
        # vind.vind_lowrank(dm1, a, P, Q) returns (v1, w) with vind(dm1) = v1 +
        # v_xc, the XC response, and w[k] = P^T v_xc[k] Q where that path is
        # cheaper for len(a) dms; otherwise (vind(dm1), 0).
        fused = None
        if hermi != 2:
            from pyscf.dft import numint
            xctype = ni._xc_type(mf.xc)
            fused = numint._fused_xc_mo(ni, mol, grids, xctype, max_memory)
        if fused is not None:
            fxc_mo = None
            def xc_lowrank(a, P, Q):
                '''w[k] = P^T v_xc[k] Q for the XC responses of the dms whose
                symmetric parts are a[k] Q^T + Q a[k]^T, None where this path
                is not the cheaper one'''
                nonlocal fxc_mo
                if not (a.dtype == P.dtype == Q.dtype == numpy.double and
                        fused.lowrank_cheaper(xctype, a.shape[2], Q.shape[1], len(a))):
                    return None
                if fxc_mo is None:
                    # the kernel that vind_xc contracts with
                    if singlet is None:
                        fxc_mo = fxc
                    elif not triplet:
                        fxc_mo = fxc[0,:,0] + fxc[0,:,1]
                    else:
                        fxc_mo = fxc[0,:,0] - fxc[0,:,1]
                w = numpy.zeros((len(a), P.shape[1], Q.shape[1]))
                fused.rks_fxc_mo(ni, xctype, fxc_mo, a, Q, P, Q, w)
                return w

            def vind_lowrank(dm1, a, P, Q):
                '''(v1, w): vind(dm1) = v1 + v_xc, w[k] = P^T v_xc[k] Q or 0, for
                dms dm1 whose symmetric parts are a[k] Q^T + Q a[k]^T'''
                a = numpy.asarray(a)
                w = xc_lowrank(a, P, Q)
                if w is None:
                    return vind(dm1), numpy.zeros((len(a), P.shape[1], Q.shape[1]))
                return vind_nlc(dm1, vind_jk(dm1)), w
            vind.vind_lowrank = vind_lowrank

            # vind.vind_mo(dm1, x) returns mo_coeff^T vind(dm1) mocc for CPHF's
            # dms dm1 = mo_coeff x mocc^T + h.c.
            if singlet is None and mo_coeff is not None and mo_occ is not None:
                mocc = mo_coeff[:,mo_occ>0]
                nmo, nocc = mo_coeff.shape[1], mocc.shape[1]
                def vind_mo(dm1, x):
                    '''mo_coeff^T vind(dm1) mocc for the symmetric dms
                    dm1 = mo_coeff x mocc^T + h.c., x[nset,nmo,nocc], made from x
                    for dm1=None. The AO matrices of the dms and of their J, K
                    and VV10 responses exist for a chunk of the dms at a time:
                    the J/K call with J by chunks of its own, then the K-only
                    calls and VV10 by chunks of the larger K-only batches, so
                    that no J/K call makes more passes over the integrals. The
                    XC response from the factors needs none of them, nor does
                    SGX's K of many densities, whose integrals would be
                    repeated for each chunk of its densities.'''
                    x = numpy.asarray(x).reshape(-1,nmo,nocc)
                    nset = len(x)
                    nao = mo_coeff.shape[0]
                    a = lib.einsum('pa,kai->kpi', mo_coeff, x)
                    v1mo = xc_lowrank(a, mo_coeff, mocc)
                    with_xc = v1mo is None
                    if with_xc:
                        v1mo = numpy.zeros((nset,nmo,nocc))

                    def by_chunks(chunk, response):
                        for p0, p1 in lib.prange(0, nset, chunk):
                            if dm1 is None:
                                d = numpy.matmul(a[p0:p1], mocc.T)
                                d = d + d.transpose(0,2,1)
                            else:
                                d = dm1[p0:p1]
                            v1 = response(d)
                            d = None
                            if v1 is not None:
                                v1mo[p0:p1] += numpy.matmul(mo_coeff.T, numpy.matmul(v1, mocc))
                            v1 = None

                    vk_mo = k_mo_from_factors(a, mocc)
                    k_done = vk_mo is not None
                    if k_done:
                        v1mo += vk_mo
                        vk_mo = None

                    def k_nlc_xc(d):
                        v1 = vind_nlc(d, None if k_done else vind_k(d))
                        if with_xc:
                            vx = vind_xc(d)
                            v1 = vx if v1 is None else v1 + vx
                        return v1

                    if k_done:
                        sgx = mf.with_df
                        by_chunks(_cphf_chunk(mf, nset, nao, True, False,
                                              sgx.direct_j and not sgx.dfj),
                                  lambda d: mf.get_j(mol, d, hermi=hermi))
                    else:
                        by_chunks(_cphf_chunk(mf, nset, nao, True, jk_together), vind_j)
                    if (k_alone and not k_done) or with_xc or (with_nlc and mf.do_nlc()):
                        by_chunks(_cphf_chunk(mf, nset, nao, False, True, k_done or None),
                                  k_nlc_xc)
                    return v1mo
                vind.vind_mo = vind_mo

    else:  # HF
        if (singlet is None or singlet) and hermi != 2:
            def vind(dm1):
                vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
                if numpy.may_share_memory(vj, vk):
                    return vj - .5 * vk
                # vj - .5 * vk in the memory of vj
                vk *= -.5
                vj += vk
                return vj
        else:
            def vind(dm1):
                vk = mf.get_k(mol, dm1, hermi=hermi)
                vk *= -.5
                return vk

    return vind


def _gen_uhf_response(mf, mo_coeff=None, mo_occ=None, *, dm0=None,
                      with_j=True, hermi=0, max_memory=None, with_nlc=True,
                      grids=None):
    '''Generate a function to compute the product of UHF response function and
    UHF density matrices.

    Kwargs:
        grids (Grids object) : Grids for the XC response kernel of DFT
            functionals. If not specified, the secondary grid
            ``mf.second_grids`` is used if available, otherwise ``mf.grids``.
    '''
    assert isinstance(mf, (uhf.UHF, rohf.ROHF))
    if dm0 is None:
        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if isinstance(mf, hf.KohnShamDFT):
        ni = mf._numint
        if grids is None:
            grids = getattr(mf, 'second_grids', None)
        if grids is None:
            grids = mf.grids
        if grids.coords is None:
            grids.build(with_non0tab=True)
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        if mo_coeff is not None and mo_occ is not None:
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, mf.xc, mo_coeff, mo_occ, 1)
        else:
            rho0, vxc, fxc = ni.cache_xc_kernel1(mol, grids, mf.xc, dm0, 1)

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        if not with_nlc and mf.do_nlc():
            logger.warn(mf, "NLC contribution in gen_response is NOT included")

        def vind_jk(dm1, v1):
            '''v1 += the Coulomb and exchange response of dm1'''
            if not hybrid:
                if with_j:
                    # J depends on the total density only
                    v1 += mf.get_j(mol, dm1[0] + dm1[1], hermi=hermi)
            else:
                if omega == 0:
                    vj, vk = mf.get_jk(mol, dm1, hermi, with_j=with_j)
                    vk *= hyb
                elif alpha == 0: # LR=0, only SR exchange
                    if with_j:
                        vj = mf.get_j(mol, dm1, hermi)
                    vk = mf.get_k(mol, dm1, hermi, omega=-omega)
                    vk *= hyb
                elif hyb == 0: # SR=0, only LR exchange
                    if with_j:
                        vj = mf.get_j(mol, dm1, hermi)
                    vk = mf.get_k(mol, dm1, hermi, omega=omega)
                    vk *= alpha
                else: # SR and LR exchange with different ratios
                    vj, vk = mf.get_jk(mol, dm1, hermi, with_j=with_j)
                    vk *= hyb
                    vk += mf.get_k(mol, dm1, hermi, omega=omega) * (alpha-hyb)
                if with_j:
                    v1 += vj[0] + vj[1] - vk
                else:
                    v1 -= vk
            return v1

        def vind_nlc(dm1, v1):
            if with_nlc and mf.do_nlc():
                # Cannot import at top due to circular dependency
                from pyscf.hessian.rks import get_vnlc_resp, get_vnlc_resp1
                if mo_coeff is not None and mo_occ is not None:
                    v1 += get_vnlc_resp(mf, mol, mo_coeff, mo_occ, dm1[0] + dm1[1], max_memory)
                else:
                    v1 += get_vnlc_resp1(mf, mol, dm0[0] + dm0[1], dm1[0] + dm1[1], max_memory)
            return v1

        def vind(dm1):
            if hermi == 2:
                v1 = numpy.zeros_like(dm1)
            else:
                v1 = ni.nr_uks_fxc(mol, grids, mf.xc, dm0, dm1, 0, hermi,
                                   rho0, vxc, fxc, max_memory=max_memory)
                v1 = vind_nlc(dm1, v1)
            return vind_jk(dm1, v1)

        # As in _gen_rhf_response, the XC response of spin dms whose symmetric
        # parts are a[s][k] Q[s]^T + Q[s] a[s][k]^T (s = alpha, beta), as in
        # CPHF (a = mo_coeff x, Q = mocc) and TDDFT (TDA: a = orbv z^T / 2,
        # Q = orbo), can take the factors instead of the dms.
        # vind.vind_lowrank(dm1, a, P, Q) returns (v1, w) with vind(dm1) = v1 +
        # v_xc, the XC response, and w[s][k] = P[s]^T v_xc[s][k] Q[s] where that
        # path is cheaper for len(a[0]) dms of each spin; otherwise
        # (vind(dm1), 0).
        fused = None
        if hermi != 2:
            from pyscf.dft import numint
            xctype = ni._xc_type(mf.xc)
            fused = numint._fused_xc_mo(ni, mol, grids, xctype, max_memory, spin=1)
        if fused is not None:
            def vind_lowrank(dm1, a, P, Q):
                '''(v1, w): vind(dm1) = v1 + v_xc, w[s][k] = P[s]^T v_xc[s][k] Q[s]
                or 0, for spin dms dm1 whose symmetric parts are
                a[s][k] Q[s]^T + Q[s] a[s][k]^T'''
                a = [numpy.asarray(x) for x in a]
                nset = len(a[0])
                w = [numpy.zeros((nset, P[s].shape[1], Q[s].shape[1])) for s in range(2)]
                if (all(x.dtype == numpy.double for x in (*a, *P, *Q)) and
                        fused.lowrank_cheaper(xctype, a[0].shape[2] + a[1].shape[2],
                                              Q[0].shape[1] + Q[1].shape[1], nset, nspin=2)):
                    fused.uks_fxc_mo(ni, xctype, fxc, a, Q, P, Q, w)
                    v1 = vind_jk(dm1, vind_nlc(dm1, numpy.zeros_like(dm1)))
                else:
                    v1 = vind(dm1)
                return v1, w
            vind.vind_lowrank = vind_lowrank

            # vind.vind_mo(dm1, x) returns (mo_coeff[s]^T vind(dm1)[s] mocc[s])
            # for CPHF's dms dm1[s] = mo_coeff[s] x[s] mocc[s]^T + h.c., given
            # the orbitals of each spin (not ROHF's mo_coeff and mo_occ)
            if (mo_coeff is not None and mo_occ is not None and
                    numpy.ndim(mo_occ[0]) == 1):
                mocc = [mo_coeff[s][:,mo_occ[s]>0] for s in range(2)]
                shapes = [(mo_coeff[s].shape[1], mocc[s].shape[1]) for s in range(2)]
                # the spin whose x tells the number of sets (the other may have
                # no occupied orbitals)
                s0 = int(numpy.prod(shapes[1]) > numpy.prod(shapes[0]))
                def vind_mo(dm1, x):
                    '''(mo_coeff[s]^T vind(dm1)[s] mocc[s]) for s = alpha, beta, for
                    the symmetric spin dms dm1[s] = mo_coeff[s] x[s] mocc[s]^T + h.c.,
                    x[s][nset,nmo_s,nocc_s]'''
                    x = [numpy.asarray(xs) for xs in x]
                    nset = x[s0].size // numpy.prod(shapes[s0])
                    a = [lib.einsum('pa,kai->kpi', mo_coeff[s], x[s].reshape(nset, *shapes[s]))
                         for s in range(2)]
                    v1, v1mo = vind_lowrank(dm1, a, mo_coeff, mocc)
                    for s in range(2):
                        for i, v in enumerate(v1[s]):
                            v1mo[s][i] += reduce(numpy.dot, (mo_coeff[s].T, v, mocc[s]))
                    return v1mo
                vind.vind_mo = vind_mo

    elif with_j:
        def vind(dm1):
            vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
            v1 = vj[0] + vj[1] - vk
            return v1

    else:
        def vind(dm1):
            return -mf.get_k(mol, dm1, hermi=hermi)

    return vind


def _gen_ghf_response(mf, mo_coeff=None, mo_occ=None, *, dm0=None,
                      with_j=True, hermi=0, max_memory=None, with_nlc=True,
                      grids=None):
    '''Generate a function to compute the product of GHF response function and
    GHF density matrices.

    Kwargs:
        grids (Grids object) : Grids for the XC response kernel of DFT
            functionals. If not specified, the secondary grid
            ``mf.second_grids`` is used if available, otherwise ``mf.grids``.
    '''
    if dm0 is None:
        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if mo_occ is None: mo_occ = mf.mo_occ
    mol = mf.mol
    if isinstance(mf, hf.KohnShamDFT):
        from pyscf.dft import numint2c, r_numint
        ni = mf._numint
        assert isinstance(ni, (numint2c.NumInt2C, r_numint.RNumInt))
        if grids is None:
            grids = getattr(mf, 'second_grids', None)
        if grids is None:
            grids = mf.grids
        if grids.coords is None:
            grids.build(with_non0tab=True)
        ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, mol.spin)
        hybrid = ni.libxc.is_hybrid_xc(mf.xc)

        if mo_coeff is not None and mo_occ is not None:
            rho0, vxc, fxc = ni.cache_xc_kernel(mol, grids, mf.xc, mo_coeff, mo_occ, 1)
        else:
            rho0, vxc, fxc = ni.cache_xc_kernel1(mol, grids, mf.xc, dm0, 1)

        if max_memory is None:
            mem_now = lib.current_memory()[0]
            max_memory = max(2000, mf.max_memory*.8-mem_now)

        if not with_nlc and mf.do_nlc():
            logger.warn(mf, "NLC contribution in gen_response is NOT included")

        def vind(dm1):
            if hermi == 2:
                v1 = numpy.zeros_like(dm1)
            else:
                v1 = ni.get_fxc(mol, grids, mf.xc, dm0, dm1, 0, 0, hermi,
                                rho0, vxc, fxc, max_memory=max_memory)
                if with_nlc and mf.do_nlc():
                    from pyscf.hessian.rks import get_vnlc_resp, get_vnlc_resp1
                    # The get_vnlc_resp/get_vnlc_resp1 functions use mf._numint
                    # and explicitly call the NumInt functions for mf._numint
                    ni1c = mf._numint._to_numint1c()
                    if mo_coeff is not None and mo_occ is not None:
                        nao = mo_coeff.shape[0] // 2
                        dm1_sf = dm1[...,:nao,:nao] + dm1[...,nao:,nao:]
                        mo_uks = [mo_coeff[:nao], mo_coeff[nao:]]
                        mo_occ_uks = [mo_occ, mo_occ]
                        with lib.temporary_env(mf, _numint=ni1c):
                            vxc_nlc = get_vnlc_resp(mf, mol, mo_uks, mo_occ_uks,
                                                    dm1_sf.real, max_memory)
                    else:
                        nao = dm0.shape[0] // 2
                        dm1_sf = dm1[...,:nao,:nao] + dm1[...,nao:,nao:]
                        dm0_sf = dm0[:nao,:nao] + dm0[nao:,nao:]
                        with lib.temporary_env(mf, _numint=ni1c):
                            vxc_nlc = get_vnlc_resp1(mf, mol, dm0_sf,
                                                     dm1_sf.real, max_memory)
                    v1[...,:nao,:nao] += vxc_nlc
                    v1[...,nao:,nao:] += vxc_nlc
            if not hybrid:
                if with_j:
                    vj = mf.get_j(mol, dm1, hermi=hermi)
                    v1 += vj
            else:
                if omega == 0:
                    vj, vk = mf.get_jk(mol, dm1, hermi, with_j=with_j)
                    vk *= hyb
                elif alpha == 0: # LR=0, only SR exchange
                    if with_j:
                        vj = mf.get_j(mol, dm1, hermi)
                    vk = mf.get_k(mol, dm1, hermi, omega=-omega)
                    vk *= hyb
                elif hyb == 0: # SR=0, only LR exchange
                    if with_j:
                        vj = mf.get_j(mol, dm1, hermi)
                    vk = mf.get_k(mol, dm1, hermi, omega=omega)
                    vk *= alpha
                else: # SR and LR exchange with different ratios
                    vj, vk = mf.get_jk(mol, dm1, hermi, with_j=with_j)
                    vk *= hyb
                    vk += mf.get_k(mol, dm1, hermi, omega=omega) * (alpha-hyb)
                if with_j:
                    v1 += vj - vk
                else:
                    v1 -= vk
            return v1

    elif with_j:
        def vind(dm1):
            vj, vk = mf.get_jk(mol, dm1, hermi=hermi)
            return vj - vk

    else:
        def vind(dm1):
            return -mf.get_k(mol, dm1, hermi=hermi)

    return vind


def _gen_dhf_response(mf, mo_coeff=None, mo_occ=None, *, dm0=None,
                      with_j=True, hermi=0, max_memory=None, with_nlc=True,
                      grids=None):
    '''Generate a function to compute the product of DHF response function and
    DHF density matrices.
    '''
    return _gen_ghf_response(mf, mo_coeff, mo_occ, dm0=dm0, with_j=with_j,
                             hermi=hermi, max_memory=max_memory,
                             with_nlc=with_nlc, grids=grids)


hf.RHF.gen_response = _gen_rhf_response
uhf.UHF.gen_response = _gen_uhf_response
ghf.GHF.gen_response = _gen_ghf_response
# Use UHF response function for ROHF because in second order solver uhf
# response function is called to compute ROHF orbital hessian
rohf.ROHF.gen_response = _gen_uhf_response
dhf.DHF.gen_response = _gen_dhf_response
