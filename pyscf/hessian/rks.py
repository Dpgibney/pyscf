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
Non-relativistic RKS analytical Hessian
'''


import numpy
import ctypes
from pyscf import lib
from pyscf import gto
from pyscf.lib import logger
from pyscf import __config__
from pyscf.hessian import rhf as rhf_hess
from pyscf.grad import rks as rks_grad
from pyscf.dft import numint, gen_grid


# import pyscf.grad.rks to activate nuc_grad_method method
from pyscf.grad import rks


min_grid_blksize = 128*128
NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD = 1e-8

libdft = lib.load_library('libdft')
contract = numpy.einsum


def partial_hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
                      atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2

    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    de2, ej, ek = rhf_hess._partial_hess_ejk(hessobj, mo_energy, mo_coeff, mo_occ,
                                             atmlst, max_memory, verbose,
                                             with_k=hybrid)
    de2 += ej - hyb * ek  # (A,B,dR_A,dR_B)
    if hybrid and omega != 0:
        ek_lr = rhf_hess._partial_hess_2e(mol, None, [dm0], 1, atmlst, with_j=False,
                                          omega=omega, verbose=log)[1]
        de2 -= (alpha - hyb) * ek_lr
        t1 = log.timer_debug1('range-separated exchange', *t1)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    veff_diag = _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory)

    aoslices = mol.aoslice_by_atom()
    vxc_dm = _get_vxc_deriv2_dm(hessobj, mo_coeff, mo_occ, max_memory)
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        de2[i0,i0] += numpy.einsum('xypq,pq->xy', veff_diag[:,:,p0:p1], dm0[p0:p1])*2
        for j0, ja in enumerate(atmlst[:i0+1]):
            de2[i0,j0] += vxc_dm[ia,ja] * 2

        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    if hessobj.grid_response:
        de2 += _get_exc_deriv2_grid_response(hessobj, mo_coeff, mo_occ, max_memory)[numpy.ix_(atmlst, atmlst)]
        t1 = log.timer_debug1('grid response of the XC energy', *t1)

    if mf.do_nlc():
        de2 += _get_enlc_deriv2(hessobj, mo_coeff, mo_occ, max_memory)

    log.timer('RKS partial hessian', *time0)
    return de2

# The XC and VV10 kernel potentials of make_h1 as products with the occupied orbitals where
# hess_elec asks for them that way (_make_h1_split); False: always the AO matrices
H1_OCC = getattr(__config__, 'hessian_rks_h1_occ', True)

def _make_h1_split(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    '''make_h1 for hess_elec: (h1ao, h1occ) with h1ao[ia] mocc + h1occ[ia] = the Fock derivative of
    make_h1 times the occupied orbitals, all that hess_elec uses of it. h1occ [natm,3,nao,nocc]
    holds the potential matrices of the 3*natm density derivatives of the XC and VV10 kernels,
    made as such products (nao x nocc instead of nao x nao per perturbation and grid point).'''
    if not H1_OCC:
        return make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose), None
    return make_h1(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose, _split=True)

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None, _split=False):
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    nao, nmo = mo_coeff.shape
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2
    hcore_deriv = hessobj.base.nuc_grad_method().hcore_generator(mol)

    mf = hessobj.base
    ni = mf._numint
    ni.libxc.test_deriv_order(mf.xc, 2, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    hybrid = ni.libxc.is_hybrid_xc(mf.xc)

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, mf.max_memory*.9-mem_now)
    h1occ = None
    if _split:
        h1occ = numpy.zeros((mol.natm, 3, nao, mocc.shape[1]))
    h1ao = _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory, occ_out=h1occ)
    if mf.do_nlc():
        _get_vnlc_deriv1(hessobj, mo_coeff, mo_occ, max_memory, out=h1ao, occ_out=h1occ)

    aoslices = mol.aoslice_by_atom()
    vhfopt = rhf_hess._make_h1_vhfopt(mol, dm0, 'jk' if hybrid else 'j')
    if hybrid and omega != 0:
        with mol.with_range_coulomb(omega):
            vhfopt_lr = rhf_hess._make_h1_vhfopt(mol, dm0, 'k')
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        shls_slice = (shl0, shl1) + (0, mol.nbas)*3
        if hybrid:
            vj1, vj2, vk1, vk2 = \
                    rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                     ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                      'lk->s1ij', -dm0         ,  # vj2
                                      'li->s1kj', -dm0[:,p0:p1],  # vk1
                                      'jk->s1il', -dm0         ], # vk2
                                     shls_slice=shls_slice, vhfopt=vhfopt)
            veff = vj1 - hyb * .5 * vk1
            veff[:,p0:p1] += vj2 - hyb * .5 * vk2
            if omega != 0:
                with mol.with_range_coulomb(omega):
                    vk1, vk2 = \
                        rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                         ['li->s1kj', -dm0[:,p0:p1],  # vk1
                                          'jk->s1il', -dm0         ], # vk2
                                         shls_slice=shls_slice, vhfopt=vhfopt_lr)
                veff -= (alpha-hyb) * .5 * vk1
                veff[:,p0:p1] -= (alpha-hyb) * .5 * vk2
        else:
            vj1, vj2 = rhf_hess._get_jk(mol, 'int2e_ip1', 3, 's2kl',
                                        ['ji->s2kl', -dm0[:,p0:p1],  # vj1
                                         'lk->s1ij', -dm0         ], # vj2
                                        shls_slice=shls_slice, vhfopt=vhfopt)
            veff = vj1
            veff[:,p0:p1] += vj2

        h1ao[ia] += veff + veff.transpose(0,2,1)
        h1ao[ia] += hcore_deriv(ia)

    if _split:
        return h1ao, h1occ
    return h1ao

rhf_hess._MAKE_H1_SPLIT[make_h1] = _make_h1_split

XX, XY, XZ = 4, 5, 6
YX, YY, YZ = 5, 7, 8
ZX, ZY, ZZ = 6, 8, 9
XXX, XXY, XXZ, XYY, XYZ, XZZ = 10, 11, 12, 13, 14, 15
YYY, YYZ, YZZ, ZZZ = 16, 17, 18, 19

def _xc_block_memory(mf, max_memory, ao_deriv, extra_per_point, held=0):
    '''The max_memory of ni.block_loop for the XC terms of the Hessian: what is left of mf.max_memory (at
    least 500 MB, at most max_memory), less held MB not yet in the RSS, for the AO values to ao_deriv that
    block_loop sizes and the extra_per_point doubles per grid point of the other arrays of a block'''
    nao = mf.mol.nao
    free = max(500, mf.max_memory - lib.current_memory()[0] - held)
    comp = (ao_deriv+1)*(ao_deriv+2)*(ao_deriv+3)//6
    return min(max_memory, free) * (comp+1)*nao / ((comp+1)*nao + extra_per_point)

def _get_vxc_diag(hessobj, mo_coeff, mo_occ, max_memory):
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((6,nao,nao))
    fused = numint._fused_xc(ni, mol, grids, nao, xctype, max_memory, (mo_coeff,))
    make_rho = None
    if fused is not None:
        make_rho = fused.rho_mo_evaluator([mo_coeff], [mo_occ], xctype)
    if make_rho is not None:
        # The AO values of the products computed in the kernels (numint._FusedXC)
        terms = numint._xc_hess_diag_terms(xctype)
        for ip0, ip1 in fused.blocks:
            weight = grids.weights[ip0:ip1]
            rho = make_rho(ip0, ip1)[0]
            wv = weight * ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
            if xctype == 'MGGA':
                wv[4] *= .5  # for the factor 1/2 in tau
            fused.dot_terms(ip0, ip1, wv[None], terms, vmat[None])

    elif xctype == 'LDA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
            wv = weight * vxc[0]
            aow = numint._scale_ao(ao[0], wv)
            for i in range(6):
                vmat[i] += numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)
            aow = None

    elif xctype == 'GGA':
        def contract_(mat, ao, aoidx, wv, mask):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1])
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2])
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3])
            mat += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
            wv = weight * vxc
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
            aow = numint._scale_ao(ao[:4], wv[:4])
            for i in range(6):
                vmat[i] += numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)

            contract_(vmat[0], ao, [XXX,XXY,XXZ], wv, mask)
            contract_(vmat[1], ao, [XXY,XYY,XYZ], wv, mask)
            contract_(vmat[2], ao, [XXZ,XYZ,XZZ], wv, mask)
            contract_(vmat[3], ao, [XYY,YYY,YYZ], wv, mask)
            contract_(vmat[4], ao, [XYZ,YYZ,YZZ], wv, mask)
            contract_(vmat[5], ao, [XZZ,YZZ,ZZZ], wv, mask)
            rho = vxc = wv = aow = None

    elif xctype == 'MGGA':
        def contract_(mat, ao, aoidx, wv, mask):
            aow = numint._scale_ao(ao[aoidx[0]], wv[1])
            aow+= numint._scale_ao(ao[aoidx[1]], wv[2])
            aow+= numint._scale_ao(ao[aoidx[2]], wv[3])
            mat += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)

        ao_deriv = 3
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:10], mo_coeff, mo_occ, mask, xctype)
            vxc = ni.eval_xc_eff(mf.xc, rho, 1, xctype=xctype)[1]
            wv = weight * vxc
            wv[4] *= .5  # for the factor 1/2 in tau
            #:aow = numpy.einsum('npi,np->pi', ao[:4], wv[:4])
            aow = numint._scale_ao(ao[:4], wv[:4])
            for i in range(6):
                vmat[i] += numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)

            contract_(vmat[0], ao, [XXX,XXY,XXZ], wv, mask)
            contract_(vmat[1], ao, [XXY,XYY,XYZ], wv, mask)
            contract_(vmat[2], ao, [XXZ,XYZ,XZZ], wv, mask)
            contract_(vmat[3], ao, [XYY,YYY,YYZ], wv, mask)
            contract_(vmat[4], ao, [XYZ,YYZ,YZZ], wv, mask)
            contract_(vmat[5], ao, [XZZ,YZZ,ZZZ], wv, mask)

            aow = [numint._scale_ao(ao[i], wv[4]) for i in range(1, 4)]
            for i, j in enumerate([XXX, XXY, XXZ, XYY, XYZ, XZZ]):
                vmat[i] += numint._dot_ao_ao(mol, ao[j], aow[0], mask, shls_slice, ao_loc)
            for i, j in enumerate([XXY, XYY, XYZ, YYY, YYZ, YZZ]):
                vmat[i] += numint._dot_ao_ao(mol, ao[j], aow[1], mask, shls_slice, ao_loc)
            for i, j in enumerate([XXZ, XYZ, XZZ, YYZ, YZZ, ZZZ]):
                vmat[i] += numint._dot_ao_ao(mol, ao[j], aow[2], mask, shls_slice, ao_loc)

    vmat = vmat[[0,1,2,
                 1,3,4,
                 2,4,5]]
    return vmat.reshape(3,3,nao,nao)

def _make_dR_rho1(ao, ao_dm0, atm_id, aoslices, xctype):
    p0, p1 = aoslices[atm_id][2:]
    ngrids = ao[0].shape[0]
    if xctype == 'GGA':
        rho1 = numpy.zeros((3,4,ngrids))
    elif xctype == 'MGGA':
        rho1 = numpy.zeros((3,5,ngrids))
        ao_dm0_x = ao_dm0[1][:,p0:p1]
        ao_dm0_y = ao_dm0[2][:,p0:p1]
        ao_dm0_z = ao_dm0[3][:,p0:p1]
        # (d_X \nabla mu) dot \nalba nu DM_{mu,nu}
        rho1[0,4] += numpy.einsum('pi,pi->p', ao[XX,:,p0:p1], ao_dm0_x)
        rho1[0,4] += numpy.einsum('pi,pi->p', ao[XY,:,p0:p1], ao_dm0_y)
        rho1[0,4] += numpy.einsum('pi,pi->p', ao[XZ,:,p0:p1], ao_dm0_z)
        rho1[1,4] += numpy.einsum('pi,pi->p', ao[YX,:,p0:p1], ao_dm0_x)
        rho1[1,4] += numpy.einsum('pi,pi->p', ao[YY,:,p0:p1], ao_dm0_y)
        rho1[1,4] += numpy.einsum('pi,pi->p', ao[YZ,:,p0:p1], ao_dm0_z)
        rho1[2,4] += numpy.einsum('pi,pi->p', ao[ZX,:,p0:p1], ao_dm0_x)
        rho1[2,4] += numpy.einsum('pi,pi->p', ao[ZY,:,p0:p1], ao_dm0_y)
        rho1[2,4] += numpy.einsum('pi,pi->p', ao[ZZ,:,p0:p1], ao_dm0_z)
        rho1[:,4] *= .5
    else:
        raise RuntimeError

    ao_dm0_0 = ao_dm0[0][:,p0:p1]
    # (d_X \nabla_x mu) nu DM_{mu,nu}
    rho1[:,0] = numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0_0)
    rho1[0,1]+= numpy.einsum('pi,pi->p', ao[XX,:,p0:p1], ao_dm0_0)
    rho1[0,2]+= numpy.einsum('pi,pi->p', ao[XY,:,p0:p1], ao_dm0_0)
    rho1[0,3]+= numpy.einsum('pi,pi->p', ao[XZ,:,p0:p1], ao_dm0_0)
    rho1[1,1]+= numpy.einsum('pi,pi->p', ao[YX,:,p0:p1], ao_dm0_0)
    rho1[1,2]+= numpy.einsum('pi,pi->p', ao[YY,:,p0:p1], ao_dm0_0)
    rho1[1,3]+= numpy.einsum('pi,pi->p', ao[YZ,:,p0:p1], ao_dm0_0)
    rho1[2,1]+= numpy.einsum('pi,pi->p', ao[ZX,:,p0:p1], ao_dm0_0)
    rho1[2,2]+= numpy.einsum('pi,pi->p', ao[ZY,:,p0:p1], ao_dm0_0)
    rho1[2,3]+= numpy.einsum('pi,pi->p', ao[ZZ,:,p0:p1], ao_dm0_0)
    # (d_X mu) (\nabla_x nu) DM_{mu,nu}
    rho1[:,1] += numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0[1][:,p0:p1])
    rho1[:,2] += numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0[2][:,p0:p1])
    rho1[:,3] += numpy.einsum('xpi,pi->xp', ao[1:4,:,p0:p1], ao_dm0[3][:,p0:p1])

    # *2 for |mu> DM <d_X nu|
    return rho1 * 2

def _d1d2_dot_(vmat, mol, ao1, ao2, mask, ao_loc, dR1_on_bra=True):
    shls_slice = (0, mol.nbas)
    if dR1_on_bra:  # (d/dR1 bra) * (d/dR2 ket)
        for d1 in range(3):
            for d2 in range(3):
                vmat[d1,d2] += numint._dot_ao_ao(mol, ao1[d1], ao2[d2], mask,
                                                 shls_slice, ao_loc)
    else:  # (d/dR2 bra) * (d/dR1 ket)
        for d1 in range(3):
            for d2 in range(3):
                vmat[d1,d2] += numint._dot_ao_ao(mol, ao1[d2], ao2[d1], mask,
                                                 shls_slice, ao_loc)

def _get_vxc_deriv2(hessobj, mo_coeff, mo_occ, max_memory):
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    vmat = numpy.zeros((mol.natm,3,3,nao,nao))
    ipip = numpy.zeros((3,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc[0]
            aow = [numint._scale_ao(ao[i], wv) for i in range(1, 4)]
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            wf = weight * fxc[0,0]
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
                # *2 for \nabla|ket> in rho1
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1]) * 2
                # aow ~ rho1 ~ d/dR1
                wv = wf * rho1
                aow = [numint._scale_ao(ao[0], wv[i]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, ao[1:4], aow, mask, ao_loc, False)
            ao_dm0 = aow = None

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,:,p0:p1]

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[0] *= .5
            aow = rks_grad._make_dR_dao_w(ao, wv)
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            for ia in range(mol.natm):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
                wv[:,0] *= .5
                for i in range(3):
                    aow = rks_grad._make_dR_dao_w(ao, wv[i])
                    rks_grad._d1_dot_(vmat[ia,i], mol, aow, ao[0], mask, ao_loc, True)

                aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, ao[1:4], aow, mask, ao_loc, False)
            ao_dm0 = aow = None

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,:,p0:p1]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,p0:p1].transpose(1,0,3,2)

    elif xctype == 'MGGA':
        XX, XY, XZ = 4, 5, 6
        YX, YY, YZ = 5, 7, 8
        ZX, ZY, ZZ = 6, 8, 9
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:10], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[0] *= .5
            wv[4] *= .25
            aow = rks_grad._make_dR_dao_w(ao, wv)
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            aow = [numint._scale_ao(ao[i], wv[4]) for i in range(4, 10)]
            _d1d2_dot_(ipip, mol, [aow[0], aow[1], aow[2]], [ao[XX], ao[XY], ao[XZ]], mask, ao_loc, False)
            _d1d2_dot_(ipip, mol, [aow[1], aow[3], aow[4]], [ao[YX], ao[YY], ao[YZ]], mask, ao_loc, False)
            _d1d2_dot_(ipip, mol, [aow[2], aow[4], aow[5]], [ao[ZX], ao[ZY], ao[ZZ]], mask, ao_loc, False)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            for ia in range(mol.natm):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
                wv[:,0] *= .5
                wv[:,4] *= .5  # for the factor 1/2 in tau
                for i in range(3):
                    aow = rks_grad._make_dR_dao_w(ao, wv[i])
                    rks_grad._d1_dot_(vmat[ia,i], mol, aow, ao[0], mask, ao_loc, True)

                aow = [numint._scale_ao(ao[:4], wv[i,:4]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, ao[1:4], aow, mask, ao_loc, False)

                aow = [numint._scale_ao(ao[1], wv[i,4]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, [ao[XX], ao[XY], ao[XZ]], aow, mask, ao_loc, False)
                aow = [numint._scale_ao(ao[2], wv[i,4]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, [ao[YX], ao[YY], ao[YZ]], aow, mask, ao_loc, False)
                aow = [numint._scale_ao(ao[3], wv[i,4]) for i in range(3)]
                _d1d2_dot_(vmat[ia], mol, [ao[ZX], ao[ZY], ao[ZZ]], aow, mask, ao_loc, False)

        for ia in range(mol.natm):
            p0, p1 = aoslices[ia][2:]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,:,p0:p1]
            vmat[ia,:,:,:,p0:p1] += ipip[:,:,p0:p1].transpose(1,0,3,2)

    return vmat

def _atom_pair_sum(aoslices, x, y):
    '''[A,g] = sum_{p in atom A} x[g,p] * y[g,p] for x, y of shape (ngrids,nao)'''
    xt, yt = x.T, y.T
    return numpy.array([numpy.einsum('pg,pg->g', xt[p0:p1], yt[p0:p1])
                        for p0, p1 in aoslices[:,2:]])

def _atom_pair_dot(wv, t):
    '''[A,B,s,t] = einsum('scg,btcg->bst', wv[A], t) for all atoms A, by one GEMM
    over the density components c and grid points g'''
    natm = wv.shape[0]
    r = lib.dot(wv.reshape(natm*3,-1), t.reshape(natm*3,-1).T)
    return r.reshape(natm,3,natm,3).transpose(0,2,1,3)

# _D2[x][y]: index of the AO component d_x d_y ao
_D2 = ((XX, XY, XZ), (YX, YY, YZ), (ZX, ZY, ZZ))

def _get_vxc_deriv2_dm(hessobj, mo_coeff, mo_occ, max_memory):
    '''The contraction of _get_vxc_deriv2 with dm0 that partial_hess_elec needs,
    e[A,B,x,y] = sum_{p in B} sum_q vxc[A,x,y,p,q] dm0[p,q], without building vxc.

    The atom-dependent terms of vxc[A] are AO products weighted on the grid. Their
    contraction with dm0 reduces to grid sums of the atom-resolved pair densities
    (ao_k * ao_dm0)[g, p in B], which are evaluated once per grid block for all
    atoms. This replaces 18 (ngrids x nao x nao) GEMMs per atom and grid block.
    '''
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    natm = mol.natm
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    vxc_dm = numpy.zeros((natm,natm,3,3))
    ipip = numpy.zeros((3,3,nao,nao))
    # Per grid point besides the AO values: the scaled AO products (aow), ao_dm0 and t and wv of all atoms
    if xctype == 'LDA':
        max_memory = _xc_block_memory(mf, max_memory, 1, 4*nao + 6*natm, ipip.nbytes/1e6)
    elif xctype == 'GGA':
        max_memory = _xc_block_memory(mf, max_memory, 2, 7*nao + 24*natm, ipip.nbytes/1e6)
    else:
        max_memory = _xc_block_memory(mf, max_memory, 2, 13*nao + 30*natm, ipip.nbytes/1e6)
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc[0]
            aow = [numint._scale_ao(ao[i], wv) for i in range(1, 4)]
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)

            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            wf = weight * fxc[0,0]
            # t[B,d] = sum_{p in B} (d_d ao * ao_dm0)[:,p]
            t = numpy.array([_atom_pair_sum(aoslices, ao[1+d], ao_dm0)
                             for d in range(3)]).transpose(1,0,2)
            wv = numpy.empty((natm,3,weight.size))
            for ia in range(natm):
                p0, p1 = aoslices[ia][2:]
                # *2 for \nabla|ket> in rho1
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1]) * 2
                wv[ia] = wf * rho1
            vxc_dm += _atom_pair_dot(wv, t)
            ao_dm0 = aow = t = wv = None

    elif xctype in ('GGA', 'MGGA'):
        ao_deriv = 2
        if xctype == 'GGA':
            ncomp, nrho = 4, 4
        else:
            ncomp, nrho = 5, 10
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho = ni.eval_rho2(mol, ao[:nrho], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[0] *= .5
            if xctype == 'MGGA':
                wv[4] *= .25
            aow = rks_grad._make_dR_dao_w(ao, wv)
            _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)
            if xctype == 'MGGA':
                aow = [numint._scale_ao(ao[i], wv[4]) for i in range(4, 10)]
                _d1d2_dot_(ipip, mol, [aow[0], aow[1], aow[2]], [ao[XX], ao[XY], ao[XZ]], mask, ao_loc, False)
                _d1d2_dot_(ipip, mol, [aow[1], aow[3], aow[4]], [ao[YX], ao[YY], ao[YZ]], mask, ao_loc, False)
                _d1d2_dot_(ipip, mol, [aow[2], aow[4], aow[5]], [ao[ZX], ao[ZY], ao[ZZ]], mask, ao_loc, False)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            # t[B,d,c] is the dm0 contraction of the c-th density component of the
            # atom-dependent terms: sum_{p in B} (D[d,c] * ao_dm0[0] + d_d ao * ao_dm0[c])
            # with D[d,0] = d_d ao and D[d,c] = d_d d_(c-1) ao. For meta-GGA, c=4 is
            # the tau term sum_x sum_{p in B} (d_x d_d ao * ao_dm0[1+x]).
            t = numpy.empty((natm,3,ncomp,weight.size))
            for d in range(3):
                t[:,d,0] = _atom_pair_sum(aoslices, ao[1+d], ao_dm0[0]) * 2
                for c in range(1, 4):
                    t[:,d,c] = _atom_pair_sum(aoslices, ao[_D2[d][c-1]], ao_dm0[0])
                    t[:,d,c] += _atom_pair_sum(aoslices, ao[1+d], ao_dm0[c])
                if xctype == 'MGGA':
                    t[:,d,4] = sum(_atom_pair_sum(aoslices, ao[_D2[x][d]], ao_dm0[1+x])
                                   for x in range(3))
            wv = numpy.empty((natm,3,ncomp,weight.size))
            for ia in range(natm):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv[ia] = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
            wv[:,:,0] *= .5
            if xctype == 'MGGA':
                wv[:,:,4] *= .5  # for the factor 1/2 in tau
            vxc_dm += _atom_pair_dot(wv, t)
            ao_dm0 = aow = t = wv = None

    # The atom-independent ipip terms enter vxc[A] through the AO columns of atom A
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        for ja in range(natm):
            q0, q1 = aoslices[ja][2:]
            vxc_dm[ia,ja] += numpy.einsum('xypq,pq->xy', ipip[:,:,q0:q1,p0:p1], dm0[q0:q1,p0:p1])
            if xctype != 'LDA':
                vxc_dm[ia,ja] += numpy.einsum('yxqp,pq->xy', ipip[:,:,p0:p1,q0:q1], dm0[q0:q1,p0:p1])
    return vxc_dm

def _get_enlc_deriv2_numerical(hessobj, mo_coeff, mo_occ, max_memory):
    """
        Attention: Numerical nlc energy 2nd derivative includes grid response.
    """
    mol = hessobj.mol
    mf = hessobj.base
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2

    de2 = numpy.empty([mol.natm, mol.natm, 3, 3])

    def get_nlc_de(grad_obj, dm):
        from pyscf.grad.rks import _initialize_grids, get_nlc_vxc_full_response
        mol = grad_obj.mol

        mf = grad_obj.base
        ni = mf._numint
        _, nlcgrids = _initialize_grids(grad_obj)

        if ni.libxc.is_nlc(mf.xc):
            xc = mf.xc
        else:
            xc = mf.nlc
        enlc, vnlc = get_nlc_vxc_full_response(
            ni, mol, nlcgrids, xc, dm,
            max_memory=max_memory, verbose=grad_obj.verbose)

        aoslices = mol.aoslice_by_atom()
        de = numpy.zeros((mol.natm,3))
        for i_atom in range(mol.natm):
            p0, p1 = aoslices[i_atom, 2:]
            de[i_atom] += numpy.einsum('xij,ij->x', vnlc[:,p0:p1], dm[p0:p1]) * 2

        assert enlc is not None
        de += enlc
        return de

    dx = 1e-3
    mol_copy = mol.copy()
    mol_copy.verbose = 0
    grad_obj = mf.Gradients()
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            xyz_p = mol.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            mol_copy.set_geom_(xyz_p, unit='Bohr')
            grad_obj.reset(mol_copy)
            de_p = get_nlc_de(grad_obj, dm0)

            xyz_m = mol.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            mol_copy.set_geom_(xyz_m, unit='Bohr')
            mol_copy.build()
            grad_obj.reset(mol_copy)
            de_m = get_nlc_de(grad_obj, dm0)

            de2[i_atom, :, i_xyz, :] = (de_p - de_m) / (2 * dx)
    grad_obj.reset(mol)

    return de2

def get_d2mu_dr2(ao):
    assert ao.ndim == 3
    nao = ao.shape[1]
    ngrids = ao.shape[2]

    d2mu_dr2 = numpy.empty([3, 3, nao, ngrids])
    d2mu_dr2[0,0,:,:] = ao[XX, :, :]
    d2mu_dr2[0,1,:,:] = ao[XY, :, :]
    d2mu_dr2[1,0,:,:] = ao[XY, :, :]
    d2mu_dr2[0,2,:,:] = ao[XZ, :, :]
    d2mu_dr2[2,0,:,:] = ao[XZ, :, :]
    d2mu_dr2[1,1,:,:] = ao[YY, :, :]
    d2mu_dr2[1,2,:,:] = ao[YZ, :, :]
    d2mu_dr2[2,1,:,:] = ao[YZ, :, :]
    d2mu_dr2[2,2,:,:] = ao[ZZ, :, :]
    return d2mu_dr2

def get_d3mu_dr3(ao):
    assert ao.ndim == 3
    nao = ao.shape[1]
    ngrids = ao.shape[2]

    d3mu_dr3 = numpy.empty([3, 3, 3, nao, ngrids])
    d3mu_dr3[0,0,0,:,:] = ao[XXX,:,:]
    d3mu_dr3[0,0,1,:,:] = ao[XXY,:,:]
    d3mu_dr3[0,1,0,:,:] = ao[XXY,:,:]
    d3mu_dr3[1,0,0,:,:] = ao[XXY,:,:]
    d3mu_dr3[0,0,2,:,:] = ao[XXZ,:,:]
    d3mu_dr3[0,2,0,:,:] = ao[XXZ,:,:]
    d3mu_dr3[2,0,0,:,:] = ao[XXZ,:,:]
    d3mu_dr3[0,1,1,:,:] = ao[XYY,:,:]
    d3mu_dr3[1,0,1,:,:] = ao[XYY,:,:]
    d3mu_dr3[1,1,0,:,:] = ao[XYY,:,:]
    d3mu_dr3[0,1,2,:,:] = ao[XYZ,:,:]
    d3mu_dr3[1,0,2,:,:] = ao[XYZ,:,:]
    d3mu_dr3[1,2,0,:,:] = ao[XYZ,:,:]
    d3mu_dr3[0,2,1,:,:] = ao[XYZ,:,:]
    d3mu_dr3[2,0,1,:,:] = ao[XYZ,:,:]
    d3mu_dr3[2,1,0,:,:] = ao[XYZ,:,:]
    d3mu_dr3[0,2,2,:,:] = ao[XZZ,:,:]
    d3mu_dr3[2,0,2,:,:] = ao[XZZ,:,:]
    d3mu_dr3[2,2,0,:,:] = ao[XZZ,:,:]
    d3mu_dr3[1,1,1,:,:] = ao[YYY,:,:]
    d3mu_dr3[1,1,2,:,:] = ao[YYZ,:,:]
    d3mu_dr3[1,2,1,:,:] = ao[YYZ,:,:]
    d3mu_dr3[2,1,1,:,:] = ao[YYZ,:,:]
    d3mu_dr3[1,2,2,:,:] = ao[YZZ,:,:]
    d3mu_dr3[2,1,2,:,:] = ao[YZZ,:,:]
    d3mu_dr3[2,2,1,:,:] = ao[YZZ,:,:]
    d3mu_dr3[2,2,2,:,:] = ao[ZZZ,:,:]

    return d3mu_dr3

def get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslices):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    natm = len(aoslices)
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert dm0.shape == (nao, nao)

    d2rho_dAdr = numpy.zeros([natm, 3, 3, ngrids])
    for i_atom in range(natm):
        p0, p1 = aoslices[i_atom][2:]
        # d2rho_dAdr[i_atom, :, :, :] += numpy.einsum('dDig,jg,ij->dDg', -d2mu_dr2[:, :, p0:p1, :], mu, dm0[p0:p1, :])
        # d2rho_dAdr[i_atom, :, :, :] += numpy.einsum('dDig,jg,ij->dDg', -d2mu_dr2[:, :, p0:p1, :], mu, dm0[:, p0:p1].T)
        # d2rho_dAdr[i_atom, :, :, :] += numpy.einsum('dig,Djg,ij->dDg', -dmu_dr[:, p0:p1, :], dmu_dr, dm0[p0:p1, :])
        # d2rho_dAdr[i_atom, :, :, :] += numpy.einsum('dig,Djg,ij->dDg', -dmu_dr[:, p0:p1, :], dmu_dr, dm0[:, p0:p1].T)
        nu_dot_dm = dm0[p0:p1, :] @ mu
        d2rho_dAdr[i_atom, :, :, :] += contract('dDig,ig->dDg', -d2mu_dr2[:, :, p0:p1, :], nu_dot_dm)
        nu_dot_dm = None
        mu_dot_dm = dm0[:, p0:p1].T @ mu
        d2rho_dAdr[i_atom, :, :, :] += contract('dDig,ig->dDg', -d2mu_dr2[:, :, p0:p1, :], mu_dot_dm)
        mu_dot_dm = None
        dnudr_dot_dm = contract('djg,ij->dig', dmu_dr, dm0[p0:p1, :])
        d2rho_dAdr[i_atom, :, :, :] += contract('dig,Dig->dDg', -dmu_dr[:, p0:p1, :], dnudr_dot_dm)
        dnudr_dot_dm = None
        dmudr_dot_dm = contract('djg,ij->dig', dmu_dr, dm0[:, p0:p1].T)
        d2rho_dAdr[i_atom, :, :, :] += contract('dig,Dig->dDg', -dmu_dr[:, p0:p1, :], dmudr_dot_dm)
        dmudr_dot_dm = None
    return d2rho_dAdr

def get_d2rho_dAdr_grid_response(d2mu_dr2, dmu_dr, mu, dm0, atom_to_grid_index_map = None, i_atom = None):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert dm0.shape == (nao, nao)

    if i_atom is None:
        assert atom_to_grid_index_map is not None
        natm = len(atom_to_grid_index_map)

        d2rho_dAdr_grid_response = numpy.zeros([natm, 3, 3, ngrids])
        for i_atom in range(natm):
            associated_grid_index = atom_to_grid_index_map[i_atom]
            # d2rho_dAdr_response  = numpy.einsum('dDig,jg,ij->dDg',
            #     d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index], dm0)
            # d2rho_dAdr_response += numpy.einsum('dDig,jg,ij->dDg',
            #     d2mu_dr2[:, :, :, associated_grid_index], mu[:, associated_grid_index], dm0.T)
            # d2rho_dAdr_response += numpy.einsum('dig,Djg,ij->dDg',
            #     dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], dm0)
            # d2rho_dAdr_response += numpy.einsum('dig,Djg,ij->dDg',
            #     dmu_dr[:, :, associated_grid_index], dmu_dr[:, :, associated_grid_index], dm0.T)
            dm_dot_mu_and_nu = (dm0 + dm0.T) @ mu[:, associated_grid_index]
            d2rho_dAdr_response  = contract('dDig,ig->dDg', d2mu_dr2[:, :, :, associated_grid_index], dm_dot_mu_and_nu)
            dm_dot_mu_and_nu = None
            dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr[:, :, associated_grid_index], dm0 + dm0.T)
            d2rho_dAdr_response += contract('dig,Dig->dDg', dmu_dr[:, :, associated_grid_index], dm_dot_dmu_and_dnu)
            dm_dot_dmu_and_dnu = None

            d2rho_dAdr_grid_response[i_atom][:, :, associated_grid_index] = d2rho_dAdr_response
    else:
        assert atom_to_grid_index_map is None

        # Here we assume all grids belong to atom i
        dm_dot_mu_and_nu = (dm0 + dm0.T) @ mu
        d2rho_dAdr_grid_response  = contract('dDig,ig->dDg', d2mu_dr2, dm_dot_mu_and_nu)
        dm_dot_mu_and_nu = None
        dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr, dm0 + dm0.T)
        d2rho_dAdr_grid_response += contract('dig,Dig->dDg', dmu_dr, dm_dot_dmu_and_dnu)
        dm_dot_dmu_and_dnu = None

    return d2rho_dAdr_grid_response

def get_drhodA_dgammadA_orbital_response(d2mu_dr2, dmu_dr, mu, drho_dr, dm0, aoslices):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    natm = len(aoslices)
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert drho_dr.shape == (3, ngrids)
    assert dm0.shape == (nao, nao)

    drhodr_dot_dmudr = contract('Djg,Dg->jg', dmu_dr, drho_dr)

    drho_dA = numpy.zeros([natm, 3, ngrids])
    dgamma_dA = numpy.zeros([natm, 3, ngrids])
    for i_atom in range(natm):
        p0, p1 = aoslices[i_atom][2:]

        # drho_dA[i_atom, :, :] += numpy.einsum('dig,jg,ij->dg', -dmu_dr[:, p0:p1, :], mu, dm0[p0:p1, :])
        # drho_dA[i_atom, :, :] += numpy.einsum('dig,jg,ij->dg', -dmu_dr[:, p0:p1, :], mu, dm0[:, p0:p1].T)
        nu_dot_dm = dm0[p0:p1, :] @ mu
        drho_dA[i_atom, :, :] += contract('dig,ig->dg', -dmu_dr[:, p0:p1, :], nu_dot_dm)
        mu_dot_dm = dm0[:, p0:p1].T @ mu
        drho_dA[i_atom, :, :] += contract('dig,ig->dg', -dmu_dr[:, p0:p1, :], mu_dot_dm)

        # dgamma_dA[i_atom, :, :] += numpy.einsum('dDig,jg,Dg,ij->dg',
        #     -d2mu_dr2[:, :, p0:p1, :], mu, drho_dr, dm0[p0:p1, :])
        # dgamma_dA[i_atom, :, :] += numpy.einsum('dDig,jg,Dg,ij->dg',
        #     -d2mu_dr2[:, :, p0:p1, :], mu, drho_dr, dm0[:, p0:p1].T)
        # dgamma_dA[i_atom, :, :] += numpy.einsum('dig,Djg,Dg,ij->dg',
        #     -dmu_dr[:, p0:p1, :], dmu_dr, drho_dr, dm0[p0:p1, :])
        # dgamma_dA[i_atom, :, :] += numpy.einsum('dig,Djg,Dg,ij->dg',
        #     -dmu_dr[:, p0:p1, :], dmu_dr, drho_dr, dm0[:, p0:p1].T)
        d2mudAdr_dot_drhodr = contract('dDig,Dg->dig', -d2mu_dr2[:, :, p0:p1, :], drho_dr)
        dgamma_dA[i_atom, :, :] += contract('dig,ig->dg', d2mudAdr_dot_drhodr, nu_dot_dm)
        dgamma_dA[i_atom, :, :] += contract('dig,ig->dg', d2mudAdr_dot_drhodr, mu_dot_dm)
        d2mudAdr_dot_drhodr = None
        nu_dot_dm = None
        mu_dot_dm = None
        drhodr_dot_dnudr_dot_dm = dm0[p0:p1, :] @ drhodr_dot_dmudr
        dgamma_dA[i_atom, :, :] += contract('dig,ig->dg', -dmu_dr[:, p0:p1, :], drhodr_dot_dnudr_dot_dm)
        drhodr_dot_dnudr_dot_dm = None
        drhodr_dot_dmudr_dot_dm = dm0[:, p0:p1].T @ drhodr_dot_dmudr
        dgamma_dA[i_atom, :, :] += contract('dig,ig->dg', -dmu_dr[:, p0:p1, :], drhodr_dot_dmudr_dot_dm)
        drhodr_dot_dmudr_dot_dm = None
    dgamma_dA *= 2

    return drho_dA, dgamma_dA

def get_drhodA_dgammadA_grid_response(d2mu_dr2, dmu_dr, mu, drho_dr, dm0, atom_to_grid_index_map = None, i_atom = None):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert drho_dr.shape == (3, ngrids)
    assert dm0.shape == (nao, nao)

    if i_atom is None:
        assert atom_to_grid_index_map is not None

        natm = len(atom_to_grid_index_map)
        drho_dA_grid_response   = numpy.zeros([natm, 3, ngrids])
        dgamma_dA_grid_response = numpy.zeros([natm, 3, ngrids])
        for i_atom in range(natm):
            associated_grid_index = atom_to_grid_index_map[i_atom]
            # rho_response  = numpy.einsum('dig,jg,ij->dg',
            #     dmu_dr[:, :, associated_grid_index], mu[:, associated_grid_index], dm0)
            # rho_response += numpy.einsum('dig,jg,ij->dg',
            #     dmu_dr[:, :, associated_grid_index], mu[:, associated_grid_index], dm0.T)
            dm_dot_mu_and_nu = (dm0 + dm0.T) @ mu[:, associated_grid_index]
            rho_response = contract('dig,ig->dg', dmu_dr[:, :, associated_grid_index], dm_dot_mu_and_nu)
            drho_dA_grid_response[i_atom][:, associated_grid_index] = rho_response
            rho_response = None

            # gamma_response  = numpy.einsum('dDig,jg,Dg,ij->dg',
            #     d2mu_dr2[:, :, :, associated_grid_index],
            #     mu[:, associated_grid_index], drho_dr[:, associated_grid_index], dm0)
            # gamma_response += numpy.einsum('dDig,jg,Dg,ij->dg',
            #     d2mu_dr2[:, :, :, associated_grid_index],
            #     mu[:, associated_grid_index], drho_dr[:, associated_grid_index], dm0.T)
            # gamma_response += numpy.einsum('dig,Djg,Dg,ij->dg',
            #     dmu_dr[:, :, associated_grid_index],
            #     dmu_dr[:, :, associated_grid_index], drho_dr[:, associated_grid_index], dm0)
            # gamma_response += numpy.einsum('dig,Djg,Dg,ij->dg',
            #     dmu_dr[:, :, associated_grid_index],
            #     dmu_dr[:, :, associated_grid_index], drho_dr[:, associated_grid_index], dm0.T)
            d2mudr2_dot_drhodr = contract('dDig,Dg->dig',
                d2mu_dr2[:, :, :, associated_grid_index], drho_dr[:, associated_grid_index])
            gamma_response  = contract('dig,ig->dg', d2mudr2_dot_drhodr, dm_dot_mu_and_nu)
            d2mudr2_dot_drhodr = None
            dm_dot_mu_and_nu = None
            dm_dot_dmu_and_dnu = contract('djg,ij->dig',
                dmu_dr[:, :, associated_grid_index], dm0 + dm0.T)
            dmudr_dot_drhodr = contract('dig,dg->ig',
                dmu_dr[:, :, associated_grid_index], drho_dr[:, associated_grid_index])
            gamma_response += contract('dig,ig->dg', dm_dot_dmu_and_dnu, dmudr_dot_drhodr)
            dmudr_dot_drhodr = None
            dm_dot_dmu_and_dnu = None
            dgamma_dA_grid_response[i_atom][:, associated_grid_index] = gamma_response
            gamma_response = None
    else:
        assert atom_to_grid_index_map is None

        # Here we assume all grids belong to atom i
        dm_dot_mu_and_nu = (dm0 + dm0.T) @ mu
        drho_dA_grid_response = contract('dig,ig->dg', dmu_dr, dm_dot_mu_and_nu)

        d2mudr2_dot_drhodr = contract('dDig,Dg->dig', d2mu_dr2, drho_dr)
        dgamma_dA_grid_response = contract('dig,ig->dg', d2mudr2_dot_drhodr, dm_dot_mu_and_nu)
        d2mudr2_dot_drhodr = None
        dm_dot_mu_and_nu = None
        dm_dot_dmu_and_dnu = contract('djg,ij->dig', dmu_dr, dm0 + dm0.T)
        dmudr_dot_drhodr = contract('dig,dg->ig', dmu_dr, drho_dr)
        dgamma_dA_grid_response += contract('dig,ig->dg', dm_dot_dmu_and_dnu, dmudr_dot_drhodr)
        dmudr_dot_drhodr = None
        dm_dot_dmu_and_dnu = None

    dgamma_dA_grid_response *= 2

    return drho_dA_grid_response, dgamma_dA_grid_response

def get_d2rhodAdB_d2gammadAdB(mol, grids_coords, dm0):
    """
        This function should never be used in practice. It requires crazy amount of memory,
        and it's left for debug purpose only. Use the contract function instead.
    """
    natm = mol.natm
    ngrids = grids_coords.shape[0]

    ao = numint.eval_ao(mol, grids_coords, deriv = 3)
    rho_drho = numint.eval_rho(mol, ao[:4, :], dm0, xctype = "GGA", hermi = 1, with_lapl = False)
    ao = ao.transpose(0,2,1) # order: component, ao, grid
    drho = rho_drho[1:4, :]
    mu = ao[0, :, :]
    dmu_dr = ao[1:4, :, :]
    d2mu_dr2 = get_d2mu_dr2(ao)
    d3mu_dr3 = get_d3mu_dr3(ao)

    aoslices = mol.aoslice_by_atom()
    d2rho_dAdB = numpy.zeros([natm, natm, 3, 3, ngrids])
    d2gamma_dAdB = numpy.zeros([natm, natm, 3, 3, ngrids])
    for i_atom in range(natm):
        pi0, pi1 = aoslices[i_atom][2:]
        d2rho_dAdB[i_atom, i_atom, :, :, :] += numpy.einsum('dDig,jg,ij->dDg',
            d2mu_dr2[:, :, pi0:pi1, :], mu, dm0[pi0:pi1, :])
        d2rho_dAdB[i_atom, i_atom, :, :, :] += numpy.einsum('dDig,jg,ij->dDg',
            d2mu_dr2[:, :, pi0:pi1, :], mu, dm0[:, pi0:pi1].T)
        d2gamma_dAdB[i_atom, i_atom, :, :, :] += numpy.einsum('dDPig,jg,Pg,ij->dDg',
            d3mu_dr3[:, :, :, pi0:pi1, :], mu, drho, dm0[pi0:pi1, :])
        d2gamma_dAdB[i_atom, i_atom, :, :, :] += numpy.einsum('dDPig,jg,Pg,ij->dDg',
            d3mu_dr3[:, :, :, pi0:pi1, :], mu, drho, dm0[:, pi0:pi1].T)
        d2gamma_dAdB[i_atom, i_atom, :, :, :] += numpy.einsum('dDig,Pjg,Pg,ij->dDg',
            d2mu_dr2[:, :, pi0:pi1, :], dmu_dr, drho, dm0[pi0:pi1, :])
        d2gamma_dAdB[i_atom, i_atom, :, :, :] += numpy.einsum('dDig,Pjg,Pg,ij->dDg',
            d2mu_dr2[:, :, pi0:pi1, :], dmu_dr, drho, dm0[:, pi0:pi1].T)
        for j_atom in range(natm):
            pj0, pj1 = aoslices[j_atom][2:]
            d2rho_dAdB[i_atom, j_atom, :, :, :] += numpy.einsum('dig,Djg,ij->dDg',
                dmu_dr[:, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], dm0[pi0:pi1, pj0:pj1])
            d2rho_dAdB[i_atom, j_atom, :, :, :] += numpy.einsum('dig,Djg,ij->dDg',
                dmu_dr[:, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], dm0[pj0:pj1, pi0:pi1].T)
            d2gamma_dAdB[i_atom, j_atom, :, :, :] += numpy.einsum('dPig,Djg,Pg,ij->dDg',
                d2mu_dr2[:, :, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], drho, dm0[pi0:pi1, pj0:pj1])
            d2gamma_dAdB[i_atom, j_atom, :, :, :] += numpy.einsum('dPig,Djg,Pg,ij->dDg',
                d2mu_dr2[:, :, pi0:pi1, :], dmu_dr[:, pj0:pj1, :], drho, dm0[pj0:pj1, pi0:pi1].T)
            d2gamma_dAdB[i_atom, j_atom, :, :, :] += numpy.einsum('dig,DPjg,Pg,ij->dDg',
                dmu_dr[:, pi0:pi1, :], d2mu_dr2[:, :, pj0:pj1, :], drho, dm0[pi0:pi1, pj0:pj1])
            d2gamma_dAdB[i_atom, j_atom, :, :, :] += numpy.einsum('dig,DPjg,Pg,ij->dDg',
                dmu_dr[:, pi0:pi1, :], d2mu_dr2[:, :, pj0:pj1, :], drho, dm0[pj0:pj1, pi0:pi1].T)

    d2rho_dAdr = get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslices)
    d2gamma_dAdB += numpy.einsum('AdPg,BDPg->ABdDg', d2rho_dAdr, d2rho_dAdr)
    d2gamma_dAdB *= 2
    return d2rho_dAdB, d2gamma_dAdB

def contract_d2rhodAdB_d2gammadAdB(d3mu_dr3, d2mu_dr2, dmu_dr, mu, drho_dr, dm0, aoslices, fw_rho, fw_gamma):
    assert mu.ndim == 2
    nao = mu.shape[0]
    ngrids = mu.shape[1]
    natm = len(aoslices)
    assert d3mu_dr3.shape == (3, 3, 3, nao, ngrids)
    assert d2mu_dr2.shape == (3, 3, nao, ngrids)
    assert dmu_dr.shape == (3, nao, ngrids)
    assert drho_dr.shape == (3, ngrids)
    assert dm0.shape == (nao, nao)

    drhodr_dot_dmudr = contract('djg,dg->jg', dmu_dr, drho_dr)

    d2e_rho_dAdB = numpy.zeros([natm, natm, 3, 3])
    d2e_gamma_dAdB = numpy.zeros([natm, natm, 3, 3])
    for i_atom in range(natm):
        pi0, pi1 = aoslices[i_atom][2:]

        nu_dot_dm = dm0[pi0:pi1, :] @ mu
        d2rho_dA2  = contract('dDig,ig->dDg', d2mu_dr2[:, :, pi0:pi1, :], nu_dot_dm)
        mu_dot_dm = dm0[:, pi0:pi1].T @ mu
        d2rho_dA2 += contract('dDig,ig->dDg', d2mu_dr2[:, :, pi0:pi1, :], mu_dot_dm)
        d2e_rho_dAdB[i_atom, i_atom, :, :] += contract('dDg,g->dD', d2rho_dA2, fw_rho)
        d2rho_dA2 = None

        d3mudA2dr_dot_drhodr = contract('dDPig,Pg->dDig', d3mu_dr3[:, :, :, pi0:pi1, :], drho_dr)
        d2gamma_dA2  = contract('dDig,ig->dDg', d3mudA2dr_dot_drhodr, nu_dot_dm)
        d2gamma_dA2 += contract('dDig,ig->dDg', d3mudA2dr_dot_drhodr, mu_dot_dm)
        d3mudA2dr_dot_drhodr = None
        nu_dot_dm = None
        mu_dot_dm = None
        drhodr_dot_dmudr_dot_dm = dm0[pi0:pi1, :] @ drhodr_dot_dmudr
        d2gamma_dA2 += contract('dDig,ig->dDg', d2mu_dr2[:, :, pi0:pi1, :], drhodr_dot_dmudr_dot_dm)
        drhodr_dot_dmudr_dot_dm = None
        drhodr_dot_dnudr_dot_dm = dm0[:, pi0:pi1].T @ drhodr_dot_dmudr
        d2gamma_dA2 += contract('dDig,ig->dDg', d2mu_dr2[:, :, pi0:pi1, :], drhodr_dot_dnudr_dot_dm)
        drhodr_dot_dnudr_dot_dm = None
        d2e_gamma_dAdB[i_atom, i_atom, :, :] += contract('dDg,g->dD', d2gamma_dA2, fw_gamma)
        d2gamma_dA2 = None

        for j_atom in range(natm):
            pj0, pj1 = aoslices[j_atom][2:]
            dnudr_dot_dm = contract('djg,ij->dig', dmu_dr[:, pj0:pj1, :], dm0[pi0:pi1, pj0:pj1])
            d2rho_dAdB  = contract('dig,Dig->dDg', dmu_dr[:, pi0:pi1, :], dnudr_dot_dm)
            dmudr_dot_dm = contract('djg,ij->dig', dmu_dr[:, pj0:pj1, :], dm0[pj0:pj1, pi0:pi1].T)
            d2rho_dAdB += contract('dig,Dig->dDg', dmu_dr[:, pi0:pi1, :], dmudr_dot_dm)
            d2e_rho_dAdB[i_atom, j_atom, :, :] += contract('dDg,g->dD', d2rho_dAdB, fw_rho)
            d2rho_dAdB = None

            drhodr_dot_d2mudAdr = contract('dDig,Dg->dig', d2mu_dr2[:, :, pi0:pi1, :], drho_dr)
            d2gamma_dAdB  = contract('dig,Dig->dDg', drhodr_dot_d2mudAdr, dnudr_dot_dm)
            dnudr_dot_dm = None
            d2gamma_dAdB += contract('dig,Dig->dDg', drhodr_dot_d2mudAdr, dmudr_dot_dm)
            dmudr_dot_dm = None
            drhodr_dot_d2mudAdr = None
            d2gamma_dAdB = contract('dDg,g->dD', d2gamma_dAdB, fw_gamma)
            d2e_gamma_dAdB[i_atom, j_atom, :, :] += d2gamma_dAdB
            d2e_gamma_dAdB[j_atom, i_atom, :, :] += d2gamma_dAdB.T
            d2gamma_dAdB = None

    d2rho_dAdr = get_d2rho_dAdr_orbital_response(d2mu_dr2, dmu_dr, mu, dm0, aoslices)
    d2e_gamma_dAdB += contract('AdPg,BDPg->ABdD', d2rho_dAdr, d2rho_dAdr * fw_gamma)

    return d2e_rho_dAdB + 2 * d2e_gamma_dAdB

class _VV10HessData:
    '''The ground-state VV10 quantities on mf.nlcgrids that the analytical Hessian terms and the response
    use (Liang, Feng, Liu, Head-Gordon, JCP 158, 204109 (2023)), on the points where
    rho >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD (mask): their coordinates, weights, atoms, rho, nabla rho,
    omega, kappa, U, W, A, B, C, E, the derivatives of omega and kappa, and f_rho, f_gamma, the first
    derivatives of the VV10 energy density (unweighted). _vv10_hess_data keeps it on the grids for the
    calls with the same density and functional, the CPHF iterations of a Hessian among them.
    '''
    def __init__(self, mf, mol, dm0, nlc_pars, max_memory, grids=None):
        if grids is None:
            grids = mf.nlcgrids
        ni = mf._numint
        self.coords_full = grids.coords
        self.mol_data = (mol._atm.copy(), mol._bas.copy(), mol._env.copy())
        self.dm0 = dm0
        self.nlc_pars = nlc_pars
        kappa_prefactor = nlc_pars[0] * 1.5 * numpy.pi * (9 * numpy.pi)**(-1.0/6.0)
        C_in_omega = nlc_pars[1]
        self.beta = 0.03125 * (3.0 / nlc_pars[0]**2)**0.75

        self.ngrids_full = grids.weights.size
        rho_drho = _vv10_rho(ni, mol, grids, dm0[numpy.newaxis], 1, max_memory)[0]
        mask = self.mask = rho_drho[0] >= NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD
        self.coords = numpy.ascontiguousarray(grids.coords[mask])
        self.weights = numpy.ascontiguousarray(grids.weights[mask])
        self.atm_idx = None
        if grids.atm_idx is not None:
            self.atm_idx = numpy.asarray(grids.atm_idx[mask], dtype=numpy.int32, order='C')
        rho = self.rho = numpy.ascontiguousarray(rho_drho[0, mask])
        nabla_rho = self.nabla_rho = numpy.ascontiguousarray(rho_drho[1:4, mask])
        rho_drho = None
        ngrids = self.ngrids = rho.size

        gamma = nabla_rho[0]**2 + nabla_rho[1]**2 + nabla_rho[2]**2
        self.omega = numpy.sqrt(C_in_omega * gamma**2 / rho**4 + (4.0/3.0*numpy.pi) * rho)
        self.kappa = kappa_prefactor * rho**(1.0/6.0)

        U, W, A, B, C, E = [numpy.empty(ngrids) for i in range(6)]
        libdft.VXC_vv10nlc_hessian_eval_UWABCE(
            U.ctypes.data_as(ctypes.c_void_p), W.ctypes.data_as(ctypes.c_void_p),
            A.ctypes.data_as(ctypes.c_void_p), B.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p), E.ctypes.data_as(ctypes.c_void_p),
            self.coords.ctypes.data_as(ctypes.c_void_p), self.weights.ctypes.data_as(ctypes.c_void_p),
            rho.ctypes.data_as(ctypes.c_void_p), self.omega.ctypes.data_as(ctypes.c_void_p),
            self.kappa.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(ngrids))
        self.U, self.W, self.A, self.B, self.C, self.E = U, W, A, B, C, E

        (self.domega_drho, self.domega_dgamma, self.d2omega_drho2, self.d2omega_dgamma2,
         self.d2omega_drho_dgamma) = d = [numpy.empty(ngrids) for i in range(5)]
        libdft.VXC_vv10nlc_hessian_eval_omega_derivative(
            *[x.ctypes.data_as(ctypes.c_void_p) for x in d],
            rho.ctypes.data_as(ctypes.c_void_p), gamma.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(C_in_omega), ctypes.c_int(ngrids))
        self.dkappa_drho   = kappa_prefactor * (1.0/6.0) * rho**(-5.0/6.0)
        self.d2kappa_drho2 = kappa_prefactor * (-5.0/36.0) * rho**(-11.0/6.0)

        self.f_rho = self.beta + E + rho * (self.dkappa_drho * U + self.domega_drho * W)
        self.f_gamma = rho * self.domega_dgamma * W

    def matches(self, mol, grids, dm0, nlc_pars):
        return (grids.coords is self.coords_full and self.nlc_pars == nlc_pars and
                all(numpy.array_equal(a, b) for a, b in
                    zip((mol._atm, mol._bas, mol._env), self.mol_data)) and
                numpy.array_equal(dm0, self.dm0))

    def grid_response_pair(self, natm, with_d2e=False, keep=False):
        '''E, U, W [natm*3,ngrids] of the motion of the other point of each pair (as
        VXC_vv10nlc_hessian_eval_EUW_grid_response), and with_d2e the term of the motion of both points, d2e
        [natm,natm,3,3], in the same pass over the pairs (points sorted by atom). keep: hand E, U, W to the next
        call without with_d2e (the Fock matrix derivative after the energy terms of a Hessian).'''
        euw = self.__dict__.pop('_euw_grid_response', None)
        if euw is not None and not with_d2e:
            return euw
        valid = numpy.nonzero(self.atm_idx >= 0)[0]
        idx = valid[numpy.argsort(self.atm_idx[valid], kind='stable')]
        offsets = numpy.zeros(natm+1, dtype=numpy.int32)
        offsets[1:] = numpy.cumsum(numpy.bincount(self.atm_idx[idx], minlength=natm))
        n = idx.size
        sorted_ = [numpy.ascontiguousarray(x[idx]) for x in
                   (self.coords, self.weights, self.rho, self.omega, self.kappa)]
        E, U, W = [numpy.empty((natm*3,n)) for i in range(3)]
        d2e = numpy.zeros((natm,natm,3,3)) if with_d2e else None
        libdft.VXC_vv10nlc_hessian_grid_response_sorted(
            *[x.ctypes.data_as(ctypes.c_void_p) for x in (E, U, W)],
            lib.c_null_ptr() if d2e is None else d2e.ctypes.data_as(ctypes.c_void_p),
            *[x.ctypes.data_as(ctypes.c_void_p) for x in sorted_],
            offsets.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(n), ctypes.c_int(natm))
        euw = []
        for x in (E, U, W):
            y = numpy.zeros((natm*3,self.ngrids))
            y[:,idx] = x
            euw.append(y)
        if keep:
            self._euw_grid_response = [x.copy() for x in euw]
        if with_d2e:
            return euw + [d2e]
        return euw

    def weight1_pair(self, dw, keep=False):
        '''Ew, Uw, Ww [nderivative,ngrids] of the weight derivatives dw [nderivative,ngrids] of the other point
        of each pair (VXC_vv10nlc_hessian_eval_EUW_with_weight1); keep: hand them to the next call'''
        euw = self.__dict__.pop('_euw_weight1', None)
        if euw is not None and euw[0].shape == dw.shape:
            return euw
        dw = numpy.ascontiguousarray(dw)
        nd = dw.shape[0]
        euw = [numpy.empty((nd,self.ngrids)) for i in range(3)]
        libdft.VXC_vv10nlc_hessian_eval_EUW_with_weight1(
            *[x.ctypes.data_as(ctypes.c_void_p) for x in euw],
            self.coords.ctypes.data_as(ctypes.c_void_p), dw.ctypes.data_as(ctypes.c_void_p),
            self.rho.ctypes.data_as(ctypes.c_void_p), self.omega.ctypes.data_as(ctypes.c_void_p),
            self.kappa.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(self.ngrids), ctypes.c_int(nd))
        if keep:
            self._euw_weight1 = [x.copy() for x in euw]
        return euw

    def full(self, x):
        '''x on the points of the mask, zero elsewhere: x[...,ngrids] -> [...,ngrids_full]'''
        out = numpy.zeros(x.shape[:-1] + (self.ngrids_full,))
        out[..., self.mask] = x
        return out

    def f_t(self, rho_t, gamma_t):
        '''f_rho_t, f_gamma_t [ntrial,ngrids] of the trial densities rho_t, gamma_t [ntrial,ngrids]
        (VXC_vv10nlc_hessian_eval_f_t)'''
        rho_t = numpy.ascontiguousarray(rho_t, dtype=numpy.double)
        gamma_t = numpy.ascontiguousarray(gamma_t, dtype=numpy.double)
        ntrial = rho_t.shape[0]
        f_rho_t = numpy.empty((ntrial, self.ngrids))
        f_gamma_t = numpy.empty((ntrial, self.ngrids))
        args = (self.coords, self.weights, self.rho, self.omega, self.kappa,
                self.U, self.W, self.A, self.B, self.C,
                self.domega_drho, self.domega_dgamma, self.dkappa_drho,
                self.d2omega_drho2, self.d2omega_dgamma2, self.d2omega_drho_dgamma,
                self.d2kappa_drho2, rho_t, gamma_t)
        libdft.VXC_vv10nlc_hessian_eval_f_t(
            f_rho_t.ctypes.data_as(ctypes.c_void_p), f_gamma_t.ctypes.data_as(ctypes.c_void_p),
            *[x.ctypes.data_as(ctypes.c_void_p) for x in args],
            ctypes.c_int(self.ngrids), ctypes.c_int(ntrial))
        return f_rho_t, f_gamma_t

def _vv10_response_grids(mf):
    '''The grids of the VV10 part of the response functions: mf.second_nlcgrids if set (see
    KohnShamDFT.set_second_nlcgrids), otherwise mf.nlcgrids'''
    grids = getattr(mf, 'second_nlcgrids', None)
    if grids is None:
        grids = mf.nlcgrids
    return grids

def _vv10_hess_data(mf, mol, dm0, max_memory, grids=None):
    '''The _VV10HessData of mf for the density dm0 on grids (mf.nlcgrids by default), from
    grids._vv10_hess_data when it matches'''
    if grids is None:
        grids = mf.nlcgrids
    if grids.coords is None:
        grids.build(with_non0tab=True)
    if numint.libxc.is_nlc(mf.xc):
        xc_code = mf.xc
    else:
        xc_code = mf.nlc
    nlc_coefs = mf._numint.nlc_coeff(xc_code)
    if len(nlc_coefs) != 1:
        raise NotImplementedError('Additive NLC')
    nlc_pars = tuple(nlc_coefs[0][0])
    dm0 = numpy.asarray(numpy.asarray(dm0).real, dtype=numpy.double, order='C')
    data = getattr(grids, '_vv10_hess_data', None)
    if data is None or not data.matches(mol, grids, dm0, nlc_pars):
        grids._vv10_hess_data = None  # releases the old arrays first
        data = grids._vv10_hess_data = _VV10HessData(mf, mol, dm0, nlc_pars, max_memory, grids)
    return data

def _vv10_rho(ni, mol, grids, dms, hermi, max_memory, fused=None):
    '''rho and nabla rho [len(dms),4,ngrids] of the density matrices dms on the grids, all of them per
    block of points through the fused kernels where those apply'''
    dms = numpy.asarray(dms)
    ngrids = grids.weights.size
    rho = numpy.empty((len(dms), 4, ngrids))
    if fused is None:
        fused = numint._fused_xc(ni, mol, grids, mol.nao, 'GGA', max_memory, (dms,),
                                 radial_cache=False)
    if fused is not None:
        make_rho = fused.rho_evaluator(ni, [dms], hermi, 'GGA')
        for ip0, ip1 in fused.blocks:
            rho[:, :, ip0:ip1] = make_rho(ip0, ip1)[0]
    else:
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, mol.nao, 1, max_memory):
            p0, p1 = p1, p1 + weight.size
            for k, dm in enumerate(dms):
                rho[k, :, p0:p1] = numint.eval_rho(mol, ao, dm, mask, 'GGA', hermi)
    return rho

def _vv10_vmat_(ni, mol, grids, wv, vmat, max_memory, fused=None):
    '''vmat[k] += half of the GGA potential matrix of wv[k] [4,ngrids] (the matrix is vmat + vmat^T),
    for all k per block of points (_FusedXC._dot_wv). wv[:,0] is halved in place.'''
    if fused is not None:
        for ip0, ip1 in fused.blocks:
            fused._dot_wv(ip0, ip1, wv[:, :, ip0:ip1], 'GGA', 1, vmat, None)
    else:
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()
        wv[:, 0] *= .5
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, mol.nao, 1, max_memory):
            p0, p1 = p1, p1 + weight.size
            for k in range(len(wv)):
                aow = numint._scale_ao(ao[:4], wv[k, :, p0:p1])
                vmat[k] += numint._dot_ao_ao(mol, ao[0], aow, mask, shls_slice, ao_loc)
    return vmat

def _vv10_drho_dA(mol, ao, dm0, mask, aoslices):
    '''The orbital response of rho and nabla rho to the nuclei on a block of points, from the AO values
    to second derivatives: drho[A,x,g] = d rho/dA_x, d2rho[A,x,D,g] = d (d_D rho)/dA_x'''
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    natm = len(aoslices)
    ngrids = ao.shape[1]
    c0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
    c1 = [numint._dot_ao_dm(mol, ao[1+i], dm0, mask, shls_slice, ao_loc) for i in range(3)]
    drho = numpy.empty((natm, 3, ngrids))
    d2rho = numpy.empty((natm, 3, 3, ngrids))
    for x in range(3):
        drho[:,x] = _atom_pair_sum(aoslices, ao[1+x], c0) * -2
        for i in range(3):
            d2rho[:,x,i] = (_atom_pair_sum(aoslices, ao[_D2[x][i]], c0) +
                            _atom_pair_sum(aoslices, ao[1+x], c1[i])) * -2
    return drho, d2rho

def _vv10_add_grid_response_(drho, d2rho, atm_idx):
    '''Adds to drho and d2rho of _vv10_drho_dA the response of the points of each atom, which move with
    it: nabla rho and nabla nabla rho, which are minus the sums over the atoms of the orbital response'''
    g = numpy.nonzero(atm_idx >= 0)[0]
    a = atm_idx[g]
    drho_dr = -drho.sum(axis=0)
    d2rho_dr2 = -d2rho.sum(axis=0)
    drho[a,:,g] += drho_dr[:,g].T
    d2rho[a,:,:,g] += d2rho_dr2[:,:,g].transpose(2,0,1)

def _vv10_dot_terms(ni, mol, grids, wv, kind, out, max_memory, fused=None):
    '''out[k] += the AO products of the GGA XC gradient (kind 'grad', numint._xc_grad_terms, out[k]
    [3,nao,nao], wv[k,0] halved as in get_vxc_grad) or of the AO-diagonal XC Hessian ('hess_diag',
    numint._xc_hess_diag_terms, out[k] [6,nao,nao]) weighted by the potentials wv[k] [4,ngrids]'''
    if fused is not None:
        if kind == 'grad':
            terms = numint._xc_grad_terms('GGA')
        else:
            terms = numint._xc_hess_diag_terms('GGA')
        for ip0, ip1 in fused.blocks:
            fused.dot_terms(ip0, ip1, wv[:, :, ip0:ip1], terms, out)
        return out
    ao_deriv = 2 if kind == 'grad' else 3
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, mol.nao, ao_deriv, max_memory):
        p0, p1 = p1, p1 + weight.size
        for k in range(len(wv)):
            w = wv[k, :, p0:p1]
            if ao_deriv == 2:
                rks_grad._gga_grad_sum_(out[k], mol, ao, w, mask, ao_loc)
            else:
                aow = numint._scale_ao(ao[:4], w)
                for i in range(6):
                    out[k,i] += numint._dot_ao_ao(mol, ao[i+4], aow, mask, shls_slice, ao_loc)
                for i, idx in enumerate(((XXX,XXY,XXZ), (XXY,XYY,XYZ), (XXZ,XYZ,XZZ),
                                         (XYY,YYY,YYZ), (XYZ,YYZ,YZZ), (XZZ,YZZ,ZZZ))):
                    aow = numint._scale_ao(ao[idx[0]], w[1])
                    aow += numint._scale_ao(ao[idx[1]], w[2])
                    aow += numint._scale_ao(ao[idx[2]], w[3])
                    out[k,i] += numint._dot_ao_ao(mol, aow, ao[0], mask, shls_slice, ao_loc)
    return out

def _vv10_blksize(mol, max_memory):
    '''The points of a block for the VV10 Fock derivative: its AO values to second derivatives, their
    products with dm0, the derivatives of rho and nabla rho of all atoms and the potentials of the 3*natm
    perturbations in a fifth of max_memory'''
    per_point = (10 + 4 + 3) * mol.nao + (12 + 12) * mol.natm
    blksize = int(max(max_memory, 500) * .2e6 / (8 * per_point)) // gen_grid.BLKSIZE
    return max(4, min(blksize, 1200)) * gen_grid.BLKSIZE

def _vv10_ao_blksize(max_memory):
    '''The max_memory of ni.block_loop for the AO values to second derivatives in the VV10 terms: with
    the arrays the terms make of each block, about a third of max_memory'''
    return max(min(max_memory * .2, 4000), 100)

def _get_enlc_deriv2(hessobj, mo_coeff, mo_occ, max_memory):
    """
        Equation notation follows:
        Liang J, Feng X, Liu X, Head-Gordon M. Analytical harmonic vibrational frequencies with
        VV10-containing density functionals: Theory, efficient implementation, and
        benchmark assessments. J Chem Phys. 2023 May 28;158(20):204109. doi: 10.1063/5.0152838.

        The VV10 energy second derivative (with hessobj.grid_response, the response of the points and the
        weights of the grids is added: the density responses below include the motion of the points, and
        _vv10_grid_response_terms adds the rest). Without it: f_rho d2rho/dAdB + f_gamma d2gamma/dAdB,
        whose AO-diagonal terms (A = B) and the pair terms take the GGA kernels of the XC Hessian
        (_get_vxc_diag, _get_vxc_deriv2_dm), and the kernel terms of drho/dA and dgamma/dA.
    """
    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint
    natm = mol.natm
    nao = mol.nao
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    mocc = mo_coeff[:,mo_occ>0]
    dm0 = 2 * mocc @ mocc.T
    vv = _vv10_hess_data(mf, mol, dm0, max_memory)
    grids = mf.nlcgrids
    grid_response = getattr(hessobj, 'grid_response', False)
    fused = numint._fused_xc(ni, mol, grids, nao, 'GGA', max_memory, (), radial_cache=False)

    # The ground-state potential in the GGA convention (d/d rho, 2 d/d gamma nabla rho), weighted
    fw_gamma = vv.full(vv.f_gamma * vv.weights)
    nabla_rho = vv.full(vv.nabla_rho)
    wv = numpy.empty((1, 4, vv.ngrids_full))
    wv[0,0] = vv.full(vv.f_rho * vv.weights)
    wv[0,1:4] = 2 * fw_gamma * nabla_rho

    d2e = numpy.zeros((natm,natm,3,3))
    # The AO-diagonal terms of rho and gamma
    vdiag = numpy.zeros((1,6,nao,nao))
    _vv10_dot_terms(ni, mol, grids, wv, 'hess_diag', vdiag, max_memory, fused)
    vdiag = vdiag[0,[0,1,2,1,3,4,2,4,5]].reshape(3,3,nao,nao)
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        d2e[ia,ia] += numpy.einsum('xypq,pq->xy', vdiag[:,:,p0:p1], dm0[p0:p1]) * 2
    vdiag = None

    # The pair terms (ipip, as _get_vxc_deriv2_dm), the term 2 f_gamma nabla drho/dA . nabla drho/dB
    # and drho/dA, dgamma/dA, in one pass over the AO values to second derivatives
    wvh = wv[0].copy()
    wvh[0] *= .5
    ipip = numpy.zeros((3,3,nao,nao))
    d2e_gg = numpy.zeros((natm*3,natm*3))
    drho_dA = numpy.empty((natm,3,vv.ngrids_full))
    dgamma_dA = numpy.empty((natm,3,vv.ngrids_full))
    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, 2, _vv10_ao_blksize(max_memory)):
        p0, p1 = p1, p1 + weight.size
        aow = rks_grad._make_dR_dao_w(ao, wvh[:,p0:p1])
        _d1d2_dot_(ipip, mol, aow, ao[1:4], mask, ao_loc, False)
        aow = None
        drho, d2rho = _vv10_drho_dA(mol, ao, dm0, mask, aoslices)
        if grid_response:
            _vv10_add_grid_response_(drho, d2rho, grids.atm_idx[p0:p1])
        drho_dA[:,:,p0:p1] = drho
        dgamma_dA[:,:,p0:p1] = 2 * numpy.einsum('axig,ig->axg', d2rho, nabla_rho[:,p0:p1])
        t = d2rho.reshape(natm*3, -1)
        d2e_gg += lib.dot((d2rho * fw_gamma[p0:p1]).reshape(natm*3, -1), t.T)
        drho = d2rho = t = None
    d2e += 2 * d2e_gg.reshape(natm,3,natm,3).transpose(0,2,1,3)
    d2e_gg = None

    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        for ja in range(natm):
            q0, q1 = aoslices[ja][2:]
            d2e[ia,ja] += numpy.einsum('xypq,pq->xy', ipip[:,:,q0:q1,p0:p1], dm0[q0:q1,p0:p1]) * 2
            d2e[ia,ja] += numpy.einsum('yxqp,pq->xy', ipip[:,:,p0:p1,q0:q1], dm0[q0:q1,p0:p1]) * 2
    ipip = None

    # The kernel terms of the 3*natm orbital responses
    drho_dA = drho_dA[:,:,vv.mask].reshape(natm*3, -1)
    dgamma_dA = dgamma_dA[:,:,vv.mask].reshape(natm*3, -1)
    f_rho_A, f_gamma_A = vv.f_t(drho_dA, dgamma_dA)
    d2e_k = lib.dot(drho_dA, (f_rho_A * vv.weights).T)
    d2e_k += lib.dot(dgamma_dA, (f_gamma_A * vv.weights).T)
    d2e += d2e_k.reshape(natm,3,natm,3).transpose(0,2,1,3)
    f_rho_A = f_gamma_A = d2e_k = None
    if grid_response:
        d2e += _vv10_grid_response_terms(mf, mol, grids, vv, dm0, drho_dA, dgamma_dA, wv[0], max_memory)
    return d2e

def _vv10_grid_response_terms(mf, mol, grids, vv, dm0, drho_dA, dgamma_dA, u, max_memory):
    '''The terms of the VV10 energy's second derivative that the Becke weights and the motion of the points add
    to those of _get_enlc_deriv2 with the full density responses drho_dA, dgamma_dA [natm*3,ngrids of the
    mask] (the orbital response and the motion of the points): Liang, Feng, Liu, Head-Gordon, JCP 158, 204109
    (2023), Eqs. 32-37, as gpu4pyscf's _get_enlc_deriv2 with grid_response. u [4,ngrids_full]: the weighted
    ground-state potential (f_rho, 2 f_gamma nabla rho).'''
    natm = mol.natm
    nao = mol.nao
    ng = vv.ngrids
    def pair(a, b):
        '''sum_g a[A,d,g] b[B,D,g] + the transpose, [natm,natm,3,3]'''
        m = lib.dot(a, b.T).reshape(natm,3,natm,3).transpose(0,2,1,3)
        return m + m.transpose(1,0,3,2)

    # E_ww: the second derivatives of the weights, and the first derivatives of the weights of both points
    table = _becke_radii_table(mol, grids)
    dw, d2e = _becke_weight_derivs(mol, grids, 0, None, vv.full(vv.rho * (vv.beta + vv.E)), True, table)
    dw = numpy.ascontiguousarray(dw[:,:,vv.mask]).reshape(natm*3,ng)
    # the pair terms of the weights and of the motion of the points are those of the VV10 Fock matrix
    # derivative too (_get_vnlc_deriv1, after this in hess_elec): kept for it if they fit in memory
    keep = 6 * dw.nbytes / 1e6 < .5 * max(0, mf.max_memory - lib.current_memory()[0])
    logger.debug1(mf, 'VV10 pair terms of the grid response kept for the Fock matrix derivative: %s', keep)
    Ew, Uw, Ww = vv.weight1_pair(dw, keep)
    d2e += lib.dot(dw, (Ew * vv.rho).T).reshape(natm,3,natm,3).transpose(0,2,1,3)
    # E_Gw: the weights with the density responses
    d2e += pair(dw, drho_dA * vv.f_rho + dgamma_dA * vv.f_gamma)
    d2e += pair(drho_dA, (Ew + (Uw * vv.dkappa_drho + Ww * vv.domega_drho) * vv.rho) * vv.weights)
    d2e += pair(dgamma_dA, Ww * (vv.domega_dgamma * vv.rho * vv.weights))
    Ew = Uw = Ww = None

    # E_wgr, E_Ggr: the motion of the other point of a pair with the weights and the density responses
    Eg, Ug, Wg, d2gr = vv.grid_response_pair(natm, with_d2e=True, keep=keep)
    d2e += pair(dw, Eg * vv.rho)
    d2e += pair(drho_dA, (Eg + (Ug * vv.dkappa_drho + Wg * vv.domega_drho) * vv.rho) * vv.weights)
    d2e += pair(dgamma_dA, Wg * (vv.domega_dgamma * vv.rho * vv.weights))
    Eg = Ug = Wg = dw = None

    # E_grgr: the motion of both points of a pair
    d2e += d2gr

    # the potential u with the motion of the points applied to the orbital response (d2rho, d2 nabla rho)
    ni = mf._numint
    ao_atm = numpy.empty(nao, dtype=int)
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:,2:]):
        ao_atm[p0:p1] = ia
    block_memory = _xc_block_memory(mf, max_memory, 3, 36*nao + natm*9)
    p1 = 0
    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, 3, block_memory):
        p0, p1 = p1, p1 + weight.size
        idx = _nonzero_aos(mol, mask, nao)
        own = grids.atm_idx[p0:p1]
        if idx.size == 0 or not numpy.any(u[0,p0:p1]):
            continue
        ao_s = ao[:,:,idx]
        seg = _AtomSegments(ao_atm[idx], natm)
        dm_s = numpy.asarray(dm0[idx[:,None],idx], order='C')
        X = numpy.array([lib.dot(ao_s[c], dm_s) for c in range(10)])
        for G in numpy.unique(own[own >= 0]):
            sel = own == G
            S = _grid_d2rho_dot(ao_s[:,sel], X[:,sel], u[:,p0:p1][:,sel], 'GGA', seg)
            d2e[G] += S.transpose(0,2,1)
            d2e[:,G] += S
            d2e[G,G] -= S.sum(axis=0).T
    return d2e

def _get_vxc_deriv1(hessobj, mo_coeff, mo_occ, max_memory, occ_out=None):
    '''The XC part of the Fock matrix derivatives, [natm,3,nao,nao].

    With occ_out [natm,3,nao,nocc]: the potential matrices of the 3*natm
    density derivatives (the kernel term), which the Hessian uses only as
    products with the occupied orbitals, are added to occ_out as such products
    (numint._FusedXC.dot_ao_mo: nao x nocc per perturbation and grid point
    instead of nao x nao) and left out of the matrices returned; the two
    together are the Fock derivative times the occupied orbitals. Where the
    fused XC kernels do not apply, the matrices hold everything and occ_out is
    not touched.'''
    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.grids is not None:
        grids = hessobj.grids
    else:
        grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    nao, nmo = mo_coeff.shape
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    aoslices = mol.aoslice_by_atom()
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    v_ip = numpy.zeros((3,nao,nao))
    vmat = numpy.zeros((mol.natm,3,nao,nao))
    # the grid response (see _grid_response_wv_, _grid_motion_vmat_)
    grid_response = getattr(hessobj, 'grid_response', False)
    if grid_response:
        table = _becke_radii_table(mol, grids)
        def grid_terms(wv, vxc, wv0, ao, mask, ip0, ip1, fac):
            own = grids.atm_idx[ip0:ip1]
            dw = _becke_weight_derivs(mol, grids, ip0, ip1, None, True, table)[0]
            _grid_response_wv_(wv, vxc, own, dw, fac)
            _grid_motion_vmat_(vmat, mol, ao, wv0, xctype, own, _nonzero_aos(mol, mask, nao))
    fused = mocc = None
    if occ_out is not None:
        fused = numint._fused_xc(ni, mol, grids, nao, xctype, max_memory, (),
                                 match_block_loop=False)
        mocc = numpy.asarray(mo_coeff[:,mo_occ>0], order='C')
        occ_sets = occ_out.reshape(mol.natm*3, nao, mocc.shape[1])

    def add_potentials(wv, p0, p1):
        '''The potential matrices of wv[natm,3,ncomp,p1-p0] (wv[:,:,0] halved, a meta-GGA's
        wv[:,:,4] times .25: half of each matrix, as vmat holds them): into vmat, or times the
        occupied orbitals into occ_out with the sign vmat gets at the end'''
        nset = wv.shape[0] * wv.shape[1]
        wv = wv.reshape(nset, wv.shape[2], p1-p0)
        if fused is None:
            return False
        if wv.shape[1] == 5:
            wv[:,4] *= 2  # dot_ao_mo takes the factor 1/2 of tau alone
        chunk = fused.mo_chunk(nset, mocc.shape[1])
        for k0, k1 in lib.prange(0, nset, chunk):
            vq = numpy.zeros((k1-k0, nao, mocc.shape[1]))
            fused.dot_ao_mo(p0, p1, wv[k0:k1], xctype, mocc, vq)
            occ_sets[k0:k1] -= vq
        return True
    # Per grid point besides the AO values: ao_dm0 and the scaled AO products of the gradient terms, and
    # the potentials of the 3*natm perturbations (with their temporaries)
    ncomp = {'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype]
    if xctype == 'LDA':
        max_memory = _xc_block_memory(mf, max_memory, 1, 3*nao + 6*mol.natm, vmat.nbytes/1e6)
    else:
        max_memory = _xc_block_memory(mf, max_memory, 2, 9*nao + 6*ncomp*mol.natm,
                                      vmat.nbytes/1e6)
    ip1 = 0
    if xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ip0, ip1 = ip1, ip1 + weight.size
            rho = ni.eval_rho2(mol, ao[0], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc[0]
            aow = numint._scale_ao(ao[0], wv)
            rks_grad._d1_dot_(v_ip, mol, ao[1:4], aow, mask, ao_loc, True)

            ao_dm0 = numint._dot_ao_dm(mol, ao[0], dm0, mask, shls_slice, ao_loc)
            wf = weight * fxc[0,0]
            wv = numpy.empty((mol.natm,3,1,weight.size))
            for ia in range(mol.natm):
                p0, p1 = aoslices[ia][2:]
# First order density = rho1 * 2.  *2 is not applied because + c.c. in the end
                rho1 = numpy.einsum('xpi,pi->xp', ao[1:,:,p0:p1], ao_dm0[:,p0:p1])
                wv[ia,:,0] = wf * rho1
            if grid_response:
                grid_terms(wv, vxc, (weight * vxc[0])[None], ao, mask, ip0, ip1, .5)
            # vmat[ia,i] += (ao[0]*wv[ia,i]).T.dot(ao[0]) for all atoms at once
            if fused is not None:
                # LDA: vmat holds half of the matrix, dot_ao_mo gives the whole one
                wv *= 2
            if not add_potentials(wv, ip0, ip1):
                numint._dot_sets_aow_ao(vmat.reshape(-1,nao,nao), ao, wv.reshape(-1,1,weight.size),
                                        mask, ao_loc)
            ao_dm0 = wv = None

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ip0, ip1 = ip1, ip1 + weight.size
            rho = ni.eval_rho2(mol, ao[:4], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[0] *= .5
            rks_grad._gga_grad_sum_(v_ip, mol, ao, wv, mask, ao_loc)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc)
                      for i in range(4)]
            wf = weight * fxc
            wv0 = wv
            wv = numpy.empty((mol.natm,3,4,weight.size))
            for ia in range(mol.natm):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv[ia] = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
            if grid_response:
                grid_terms(wv, vxc, wv0, ao, mask, ip0, ip1, 1.)
            wv[:,:,0] *= .5
            # vmat[ia,i] += einsum('xgp,xg->gp', ao[:4], wv[ia,i]).T.dot(ao[0])
            if not add_potentials(wv, ip0, ip1):
                numint._dot_sets_aow_ao(vmat.reshape(-1,nao,nao), ao, wv.reshape(-1,4,weight.size),
                                        mask, ao_loc)
            ao_dm0 = wv = None

    elif xctype == 'MGGA':
        _check_mgga_grids(grids)
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            ip0, ip1 = ip1, ip1 + weight.size
            rho = ni.eval_rho2(mol, ao[:10], mo_coeff, mo_occ, mask, xctype)
            vxc, fxc = ni.eval_xc_eff(mf.xc, rho, 2, xctype=xctype)[1:3]
            wv = weight * vxc
            wv[0] *= .5
            wv[4] *= .5  # for the factor 1/2 in tau
            rks_grad._gga_grad_sum_(v_ip, mol, ao, wv, mask, ao_loc)
            rks_grad._tau_grad_dot_(v_ip, mol, ao, wv[4], mask, ao_loc, True)

            ao_dm0 = [numint._dot_ao_dm(mol, ao[i], dm0, mask, shls_slice, ao_loc) for i in range(4)]
            wf = weight * fxc
            wv0 = wv
            wv = numpy.empty((mol.natm,3,5,weight.size))
            for ia in range(mol.natm):
                dR_rho1 = _make_dR_rho1(ao, ao_dm0, ia, aoslices, xctype)
                wv[ia] = numpy.einsum('xyg,sxg->syg', wf, dR_rho1)
            if grid_response:
                grid_terms(wv, vxc, wv0, ao, mask, ip0, ip1, 1.)
            wv[:,:,0] *= .5
            wv[:,:,4] *= .25
            # vmat[ia,i] += einsum('xgp,xg->gp', ao[:4], wv[ia,i,:4]).T.dot(ao[0])
            #             + sum_j (ao[j]*wv[ia,i,4]).T.dot(ao[j])
            if not add_potentials(wv, ip0, ip1):
                numint._dot_sets_aow_ao(vmat.reshape(-1,nao,nao), ao, wv.reshape(-1,5,weight.size),
                                        mask, ao_loc)
            ao_dm0 = wv = None

    for ia in range(mol.natm):
        p0, p1 = aoslices[ia][2:]
        vmat[ia,:,p0:p1] += v_ip[:,p0:p1]
        vmat[ia] = -vmat[ia] - vmat[ia].transpose(0,2,1)

    return vmat

def _get_vnlc_deriv1_numerical(hessobj, mo_coeff, mo_occ, max_memory):
    """
        Attention: Numerical nlc Fock matrix 1st derivative includes grid response.
    """
    mol = hessobj.mol
    mf = hessobj.base
    mocc = mo_coeff[:,mo_occ>0]
    dm0 = numpy.dot(mocc, mocc.T) * 2

    nao = mol.nao
    vmat = numpy.empty([mol.natm, 3, nao, nao])

    def get_nlc_vmat(mol, mf, dm):
        ni = mf._numint
        if ni.libxc.is_nlc(mf.xc):
            xc = mf.xc
        else:
            assert ni.libxc.is_nlc(mf.nlc)
            xc = mf.nlc
        mf.nlcgrids.build()
        _, _, vnlc = ni.nr_nlc_vxc(mol, mf.nlcgrids, xc, dm)
        return vnlc

    dx = 1e-3
    mol_copy = mol.copy()
    mol_copy.verbose = 0
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            xyz_p = mol.atom_coords()
            xyz_p[i_atom, i_xyz] += dx
            mol_copy.set_geom_(xyz_p, unit='Bohr')
            mol_copy.build()
            mf.reset(mol_copy)
            vmat_p = get_nlc_vmat(mol_copy, mf, dm0)

            xyz_m = mol.atom_coords()
            xyz_m[i_atom, i_xyz] -= dx
            mol_copy.set_geom_(xyz_m, unit='Bohr')
            mol_copy.build()
            mf.reset(mol_copy)
            vmat_m = get_nlc_vmat(mol_copy, mf, dm0)

            vmat[i_atom, i_xyz, :, :] = (vmat_p - vmat_m) / (2 * dx)
    mf.reset(mol)

    return vmat

def get_dweight_dA(mol, grids):
    '''The derivatives of the Becke weights of all points of grids with respect to the atom coordinates,
    [natm,3,ngrids], the points moving with their atoms (_becke_weight_derivs)'''
    return _becke_weight_derivs(mol, grids)[0]

def _get_vnlc_deriv1(hessobj, mo_coeff, mo_occ, max_memory, out=None, occ_out=None):
    """
        Equation notation follows:
        Liang J, Feng X, Liu X, Head-Gordon M. Analytical harmonic vibrational frequencies with
        VV10-containing density functionals: Theory, efficient implementation, and
        benchmark assessments. J Chem Phys. 2023 May 28;158(20):204109. doi: 10.1063/5.0152838.

        The VV10 Fock matrix derivative with grid response, h1[A,x]: the GGA potential matrices of the
        3*natm perturbed potentials (the kernel terms of drho/dA and dgamma/dA, the derivatives of the
        weights and the grid response of E, U and W) in one pass over the grid, and the derivatives of the
        AO products with the ground-state potential, for the AOs of atom A and for the points of atom A.
        With out (natm,3,nao,nao) of symmetric matrices, h1 is added to it and it is returned, without an
        array of h1's own. With occ_out (natm,3,nao,nocc), the potential matrices of the 3*natm perturbed
        potentials are added to it as products with the occupied orbitals instead (see _get_vxc_deriv1),
        where the fused XC kernels apply.
    """
    # Note (Henry Wang 20250428):
    # We observed that in several very simple systems, for example H2O2, H2CO, C2H4,
    # if we do not include the grid response term, the analytical and numerical Fock matrix
    # derivative, although only diff by else than 1e-7 (norm 1), can cause a 1e-3 error in hessian,
    # likely because the CPHF converged to a different solution.

    mol = hessobj.mol
    mf = hessobj.base
    ni = mf._numint
    natm = mol.natm
    nao = mol.nao
    aoslices = mol.aoslice_by_atom()

    mocc = mo_coeff[:,mo_occ>0]
    mocc_c = numpy.asarray(mocc, order='C')
    dm0 = 2 * mocc @ mocc.T
    vv = _vv10_hess_data(mf, mol, dm0, max_memory)
    grids = mf.nlcgrids
    assert grids.atm_idx.shape[0] == grids.coords.shape[0]
    atm_idx = grids.atm_idx
    mask = vv.mask
    ngrids_full = vv.ngrids_full
    weights = vv.full(vv.weights)
    f_gamma = vv.full(vv.f_gamma)
    nabla_rho = vv.full(vv.nabla_rho)

    # drho/dA and dgamma/dA with the grid response, on the points of the mask; the blocks of points (as many
    # as the AO values to second derivatives and the arrays of a block allow in a fifth of max_memory) are
    # those of the pass for the potentials below too
    blksize = _vv10_blksize(mol, max_memory)
    fused = numint._fused_xc(ni, mol, grids, nao, 'GGA', max_memory, (), blksize=blksize,
                             radial_cache=False)
    ngrids = vv.ngrids
    drho_dA = numpy.empty((natm*3,ngrids))
    dgamma_dA = numpy.empty((natm*3,ngrids))
    p1 = q1 = 0
    for ao, mask_ao, weight, coords \
            in ni.block_loop(mol, grids, nao, 2, blksize=blksize):
        p0, p1 = p1, p1 + weight.size
        sel = mask[p0:p1]
        q0, q1 = q1, q1 + numpy.count_nonzero(sel)
        drho, d2rho = _vv10_drho_dA(mol, ao, dm0, mask_ao, aoslices)
        _vv10_add_grid_response_(drho, d2rho, atm_idx[p0:p1])
        drho_dA[:,q0:q1] = drho.reshape(natm*3,-1)[:,sel]
        dgamma = 2 * numpy.einsum('axig,ig->axg', d2rho, nabla_rho[:,p0:p1])
        dgamma_dA[:,q0:q1] = dgamma.reshape(natm*3,-1)[:,sel]
        drho = d2rho = dgamma = None
    f_rho_A, f_gamma_A = vv.f_t(drho_dA, dgamma_dA)
    drho_dA = dgamma_dA = None
    f_rho_A = f_rho_A.reshape(natm,3,-1)
    f_gamma_A = f_gamma_A.reshape(natm,3,-1)

    # The grid response of E, U and W, of the points, which move with their atoms, and of their weights,
    # each added to the kernel terms in turn
    E, U, W = [x.reshape(natm,3,ngrids) for x in vv.grid_response_pair(natm)]
    for step in range(2):
        f_gamma_A += W * (vv.domega_dgamma * vv.rho)
        U *= vv.dkappa_drho * vv.rho
        W *= vv.domega_drho * vv.rho
        f_rho_A += E
        f_rho_A += U
        f_rho_A += W
        if step == 0:
            dw = numpy.ascontiguousarray(get_dweight_dA(mol, grids)[:,:,mask])
            E, U, W = [x.reshape(natm,3,ngrids) for x in vv.weight1_pair(dw.reshape(natm*3,ngrids))]
    E = U = W = None

    # The potentials of the 3*natm perturbations: a0 is the rho part, a1 the coefficient of nabla rho
    a0 = vv.full(vv.weights * f_rho_A + dw * vv.f_rho)
    a1 = vv.full(vv.weights * f_gamma_A + dw * vv.f_gamma)
    f_rho_A = f_gamma_A = dw = None

    # h1 holds half of each matrix: the result is h1 + h1^T. The potentials of a block take d(nabla rho)/dA,
    # evaluated again from the AO values of the block rather than kept for all points.
    if out is not None and out.flags.c_contiguous and out.dtype == numpy.double:
        # half of out, to which the symmetrization at the end adds h1 + h1^T
        h1 = out
        h1 *= .5
    else:
        h1 = numpy.zeros((natm,3,nao,nao))
    h1_sets = h1.reshape(natm*3,nao,nao)
    wf_gamma = weights * f_gamma
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    p1 = 0
    for ao, mask_ao, weight, coords \
            in ni.block_loop(mol, grids, nao, 2, blksize=blksize):
        p0, p1 = p1, p1 + weight.size
        drho, d2rho = _vv10_drho_dA(mol, ao, dm0, mask_ao, aoslices)
        _vv10_add_grid_response_(drho, d2rho, atm_idx[p0:p1])
        drho = None
        wv = numpy.empty((natm,3,4,p1-p0))
        wv[:,:,0] = a0[:,:,p0:p1]
        wv[:,:,1:4] = 2 * (a1[:,:,numpy.newaxis,p0:p1] * nabla_rho[:,p0:p1]
                           + d2rho * wf_gamma[p0:p1])
        d2rho = None
        wv = wv.reshape(natm*3,4,p1-p0)
        if fused is not None and occ_out is not None:
            wv[:,0] *= .5  # as _dot_wv: the matrix is v + v^T
            nocc = occ_out.shape[-1]
            occ_sets = occ_out.reshape(natm*3, nao, nocc)
            chunk = fused.mo_chunk(natm*3, nocc)
            for k0, k1 in lib.prange(0, natm*3, chunk):
                vq = numpy.zeros((k1-k0, nao, nocc))
                fused.dot_ao_mo(p0, p1, wv[k0:k1], 'GGA', mocc_c, vq)
                occ_sets[k0:k1] += vq
        elif fused is not None:
            fused._dot_wv(p0, p1, wv, 'GGA', 1, h1_sets, None)
        else:
            wv[:,0] *= .5
            for k in range(natm*3):
                aow = numint._scale_ao(ao[:4], wv[k])
                h1_sets[k] += numint._dot_ao_ao(mol, ao[0], aow, mask_ao, shls_slice, ao_loc)
        wv = None
    a0 = a1 = None

    # The derivatives of the AO products with the ground-state potential: of the AOs of atom A on all
    # points (minus), and of all AOs on the points of atom A
    wv = numpy.empty((1,4,ngrids_full))
    wv[0,0] = vv.full(vv.f_rho * vv.weights) * .5
    wv[0,1:4] = 2 * weights * f_gamma * nabla_rho
    g = numpy.zeros((1,3,nao,nao))
    _vv10_dot_terms(ni, mol, grids, wv, 'grad', g, max_memory, fused)
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        h1[ia,:,p0:p1] -= g[0,:,p0:p1]
    g = None
    nbatch = max(1, int(max_memory*.1e6 / (8*ngrids_full*4)))
    for ia0, ia1 in lib.prange(0, natm, nbatch):
        on_atom = numpy.arange(ia0, ia1)[:,numpy.newaxis,numpy.newaxis] == atm_idx
        _vv10_dot_terms(ni, mol, grids, on_atom * wv, 'grad', h1[ia0:ia1], max_memory, fused)
    lib.hermi_sum(h1_sets, axes=(0,2,1), inplace=True)
    if out is not None and h1 is not out:
        out += h1
        return out
    return h1

def get_vnlc_resp(mf, mol, mo_coeff, mo_occ, dm1s, max_memory, out=None):
    if isinstance(mo_coeff, numpy.ndarray) and mo_coeff.ndim == 2:
        mocc = mo_coeff[:,mo_occ>0]
        mo_occ = mo_occ[mo_occ > 0]
        dm0 = (mocc * mo_occ) @ mocc.T
    else:
        assert mo_coeff[0].ndim == 2 # unrestricted case
        assert len(mo_coeff) == 2
        assert len(mo_occ) == 2
        mocc_a = mo_coeff[0][:, mo_occ[0] > 0]
        mocc_b = mo_coeff[1][:, mo_occ[1] > 0]
        mo_occ_a = mo_occ[0][mo_occ[0] > 0]
        mo_occ_b = mo_occ[1][mo_occ[1] > 0]
        dm0 = (mocc_a * mo_occ_a) @ mocc_a.T + (mocc_b * mo_occ_b) @ mocc_b.T

    return get_vnlc_resp1(mf, mol, dm0, dm1s, max_memory, out)

def get_vnlc_resp1(mf, mol, dm0, dm1s, max_memory, out=None):
    """
        Equation notation follows:
        Liang J, Feng X, Liu X, Head-Gordon M. Analytical harmonic vibrational frequencies with
        VV10-containing density functionals: Theory, efficient implementation, and
        benchmark assessments. J Chem Phys. 2023 May 28;158(20):204109. doi: 10.1063/5.0152838.

        mo_coeff, mo_occ are 0-th order
        dm1s is first order

        The densities of the dm1s and their potential matrices go through the fused kernels for all
        dm1s of a batch per block of points; the ground-state quantities (_VV10HessData) are kept for
        the calls with the same dm0. With out (an array of the shape of dm1s), the matrices are added
        to it and it is returned: a batch of them is held at a time, not all.

        On mf.second_nlcgrids if set (KohnShamDFT.set_second_nlcgrids), otherwise mf.nlcgrids.
    """
    output_in_2d = False
    if dm1s.ndim == 2:
        assert dm1s.shape == (mol.nao, mol.nao)
        dm1s = dm1s.reshape((1, mol.nao, mol.nao))
        output_in_2d = True
    assert dm1s.ndim == 3

    grids = _vv10_response_grids(mf)
    vv = _vv10_hess_data(mf, mol, dm0, max_memory, grids)
    ni = mf._numint
    nao = mol.nao
    n_dm1 = dm1s.shape[0]
    mask = vv.mask
    fused = numint._fused_xc(ni, mol, grids, nao, 'GGA', max_memory, (dm1s,), radial_cache=False)

    # The pair kernel (f_t) evaluates every pair of points once per call, so a call takes as many dm1s
    # as their trial densities, kernels and nabla rho on the points of the mask (7 ngrids per dm1)
    # allow; their rho and nabla rho, then their potentials, on all points (4 ngrids_full per dm1) go
    # through the fused kernels in batches of at least 32 dm1s
    nbytes_kept = 7 * vv.ngrids * 8
    nbytes_full = 4 * vv.ngrids_full * 8
    mem = max_memory * .6e6
    batch = max(1, min(n_dm1, 32, int(mem / (nbytes_kept + nbytes_full))))
    group = max(batch, min(n_dm1, int((mem - batch * nbytes_full) / nbytes_kept)))
    if out is not None and out.flags.c_contiguous and out.dtype == numpy.double:
        vmat = out.reshape(n_dm1, nao, nao)
    else:
        vmat = numpy.zeros((n_dm1, nao, nao))
    for g0, g1 in lib.prange(0, n_dm1, group):
        nabla_rho_t = numpy.empty((g1-g0, 3, vv.ngrids))
        rho_t = numpy.empty((g1-g0, vv.ngrids))
        for k0, k1 in lib.prange(g0, g1, batch):
            rho1 = _vv10_rho(ni, mol, grids, dm1s[k0:k1], 0, max_memory, fused)
            rho_t[k0-g0:k1-g0] = rho1[:,0,mask]
            nabla_rho_t[k0-g0:k1-g0] = rho1[:,1:4,mask]
            rho1 = None
        # The factor of 2 before the gamma_j^t term in equation (22)
        gamma_t = 2 * numpy.einsum('ig,tig->tg', vv.nabla_rho, nabla_rho_t)
        f_rho_t, f_gamma_t = vv.f_t(rho_t, gamma_t)
        rho_t = gamma_t = None

        for k0, k1 in lib.prange(0, g1-g0, batch):
            wv = numpy.zeros((k1-k0, 4, vv.ngrids_full))
            wv[:,0,mask] = f_rho_t[k0:k1] * vv.weights
            wv[:,1:4,mask] = (vv.nabla_rho * f_gamma_t[k0:k1,numpy.newaxis]
                              + nabla_rho_t[k0:k1] * vv.f_gamma) * (2 * vv.weights)
            v = numpy.zeros((k1-k0, nao, nao))
            _vv10_vmat_(ni, mol, grids, wv, v, max_memory, fused)
            wv = None
            vmat[g0+k0:g0+k1] += lib.hermi_sum(v, axes=(0,2,1), inplace=True)
            v = None
        f_rho_t = f_gamma_t = nabla_rho_t = None

    if out is not None:
        if not numpy.shares_memory(vmat, out):
            out += vmat.reshape(out.shape)
        return out
    if output_in_2d:
        vmat = vmat.reshape((mol.nao, mol.nao))
    return vmat

def _even_batch(n, batch):
    '''The size of the fewest batches of at most batch items that cover n items, evened out'''
    return -(-n // -(-n // batch))

def get_vnlc_resp_mo(mf, mol, mo_coeff, mo_occ, a, max_memory, out):
    """
        out[k] += P^T v[k] Q (P = mo_coeff, Q the occupied orbitals) for the VV10 responses v[k] that
        get_vnlc_resp gives for the symmetric dms a[k] Q^T + Q a[k]^T, a[nset,nao,nocc], from these
        factors. Returns out; None, out untouched, if the fused kernels do not apply to the grids of
        the VV10 response (mf.second_nlcgrids if set, otherwise mf.nlcgrids).

        Neither the dms nor the v[k] exist in the AO basis. The pair kernel goes through all pairs of
        points once per call, so a group of dms is as large as its data on the points of the mask allow
        (7 ngrids per dm, all of it alive only around the kernel). The densities and the potentials go
        by blocks of points, and the matrices by batches of dms that keep the partial sums of the fused
        kernels within a quarter of max_memory.
    """
    grids = _vv10_response_grids(mf)
    if grids.coords is None:
        grids.build(with_non0tab=True)
    nao = mol.nao
    fused = numint._fused_xc(mf._numint, mol, grids, nao, 'GGA', max_memory, (),
                             match_block_loop=False, radial_cache=False)
    if fused is None:
        return None
    mocc = mo_coeff[:,mo_occ>0]
    vv = _vv10_hess_data(mf, mol, (mocc * mo_occ[mo_occ>0]) @ mocc.T, max_memory, grids)
    mocc = numpy.asarray(mocc, order='C')
    a = numpy.asarray(a, order='C')
    nset, nq = len(a), mocc.shape[1]
    mask = vv.mask

    group = _even_batch(nset, max(1, int(max_memory*.6e6 / (7 * vv.ngrids * 8))))
    # rho and nabla rho, or the potentials, of a group on a block of points in a tenth of max_memory
    ngemm = numint.BLKSIZE * numint.GEMM_NBLK
    blksize = max(1, min(int(max_memory*.1e6 / (8 * group * 4 * ngemm)), 300)) * ngemm
    def blocks():
        '''The blocks of points: (ip0, ip1) on the grids, the mask on them and (p0, p1) in the arrays on the mask'''
        p1 = 0
        for ip0, ip1 in lib.prange(0, vv.ngrids_full, blksize):
            on = mask[ip0:ip1]
            p0, p1 = p1, p1 + numpy.count_nonzero(on)
            yield ip0, ip1, on, p0, p1

    nparts = numint.STREAMS_PER_THREAD * lib.num_threads()
    max_batch = max(1, int(max_memory*.25e6 / (8 * (nparts+1) * nao * nq)))
    for g0, g1 in lib.prange(0, nset, group):
        n = g1 - g0
        rho_t = numpy.empty((n, vv.ngrids))
        nabla_rho_t = numpy.empty((n, 3, vv.ngrids))
        for ip0, ip1, on, p0, p1 in blocks():
            rho1 = fused.rho_lowrank(ip0, ip1, a[g0:g1], mocc, 'GGA')
            rho_t[:,p0:p1] = rho1[:,0,on]
            nabla_rho_t[:,:,p0:p1] = rho1[:,1:4,on]
        rho1 = None
        # The factor of 2 before the gamma_j^t term in equation (22)
        gamma_t = 2 * numpy.einsum('ig,tig->tg', vv.nabla_rho, nabla_rho_t)
        f_rho_t, f_gamma_t = vv.f_t(rho_t, gamma_t)
        rho_t = gamma_t = None

        for k0, k1 in lib.prange(0, n, _even_batch(n, max_batch)):
            vq = numpy.zeros((k1-k0, nao, nq))
            for ip0, ip1, on, p0, p1 in blocks():
                w = vv.weights[p0:p1]
                wv = numpy.zeros((k1-k0, 4, ip1-ip0))
                wv[:,0,on] = f_rho_t[k0:k1,p0:p1] * w
                wv[:,1:4,on] = ((vv.nabla_rho[:,p0:p1] * f_gamma_t[k0:k1,numpy.newaxis,p0:p1]
                                 + nabla_rho_t[k0:k1,:,p0:p1] * vv.f_gamma[p0:p1]) * (2 * w))
                wv[:,0] *= .5  # as _dot_wv: the matrix is v + v^T
                fused.dot_ao_mo(ip0, ip1, wv, 'GGA', mocc, vq)
            out[g0+k0:g0+k1] += lib.einsum('pa,kpi->kai', mo_coeff, vq)
            vq = None
        f_rho_t = f_gamma_t = nabla_rho_t = None
    return out

def _check_mgga_grids(grids):
    mol = grids.mol
    atom_grid = grids.atom_grid
    if atom_grid:
        if isinstance(atom_grid, (tuple, list)):
            n_rad = atom_grid[0]
            if n_rad < 150 and any(mol.atom_charges() > 10):
                logger.warn(mol, 'MGGA Hessian is sensitive to dft grids. '
                            f'{atom_grid} may not be dense enough.')
        else:
            symbols = [mol.atom_symbol(ia) for ia in range(mol.natm)]
            problematic = []
            for symb in symbols:
                chg = gto.charge(symb)
                if symb in atom_grid:
                    n_rad = atom_grid[symb][0]
                else:
                    n_rad = gen_grid._default_rad(chg, grids.level)
                if n_rad < 150 and chg > 10:
                    problematic.append((symb, n_rad))
            if problematic:
                problematic = [f'{symb}: {r}' for symb, r in problematic]
                logger.warn(mol, 'MGGA Hessian is sensitive to dft grids. '
                            f'Radial grids {",".join(problematic)} '
                            'may not be dense enough.')
    elif grids.level < 5:
        logger.warn(mol, 'MGGA Hessian is sensitive to dft grids. '
                    f'grids.level {grids.level} may not be dense enough.')


# The grid response of the XC terms (Hessian.grid_response = True). The points of an atom's grid move with
# it, and the Becke weights depend on all atom positions. These terms are added to those of the AO
# derivatives (the orbital response) that the functions above compute: in the energy's second derivative at
# fixed orbitals by _get_exc_deriv2_grid_response, in the Fock matrix derivative within _get_vxc_deriv1.
# Translational invariance gives the response of a point's own atom: the motion of the point changes the
# density there by nabla rho = minus the sum over the atoms of the orbital response, and the derivatives of a
# weight (and of each of its first derivatives) sum to zero over the atoms.

def _d3_index(x, y, z):
    return {(0,0,0): XXX, (0,0,1): XXY, (0,0,2): XXZ, (0,1,1): XYY, (0,1,2): XYZ,
            (0,2,2): XZZ, (1,1,1): YYY, (1,1,2): YYZ, (1,2,2): YZZ, (2,2,2): ZZZ}[tuple(sorted((x, y, z)))]
# _D3[x][y][z]: index of the AO component d_x d_y d_z ao
_D3 = [[[_d3_index(x, y, z) for z in range(3)] for y in range(3)] for x in range(3)]

def _becke_radii_table(mol, grids):
    '''The a_ij of the Becke partition of grids (VXCgen_grid's radii_table), or None without radii adjustment'''
    from pyscf.dft import radi
    if grids.becke_scheme is not gen_grid.original_becke:
        raise NotImplementedError(f'Hessian grid response with becke_scheme {grids.becke_scheme}')
    radii_adjust = grids.radii_adjust
    if not callable(radii_adjust) or grids.atomic_radii is None:
        return None
    if radii_adjust not in (radi.treutler_atomic_radii_adjust, radi.becke_atomic_radii_adjust):
        raise NotImplementedError(f'Hessian grid response with radii_adjust {radii_adjust}')
    f = radii_adjust(mol, grids.atomic_radii)
    return numpy.asarray([f(i, j, 0) for i in range(mol.natm) for j in range(mol.natm)])

def _becke_weight_derivs(mol, grids, p0=0, p1=None, eps=None, with_dw=True, table=False):
    '''Of the points p0:p1 of grids, which move with their atoms (grids.atm_idx): dw [natm,3,p1-p0], the
    derivatives of the Becke weights with respect to the atom coordinates (if with_dw), and
    d2e [natm,natm,3,3] = sum_g eps[g] d2w_g/dAdB (if eps is given). table: _becke_radii_table(mol, grids)'''
    if p1 is None:
        p1 = grids.weights.size
    if table is False:
        table = _becke_radii_table(mol, grids)
    natm = mol.natm
    n = p1 - p0
    coords = numpy.asarray(grids.coords[p0:p1], order='C')
    qweights = numpy.asarray(grids.quadrature_weights[p0:p1], order='C')
    atm_idx = numpy.asarray(grids.atm_idx[p0:p1], dtype=numpy.int32, order='C')
    atm_coords = numpy.asarray(mol.atom_coords(), order='C')
    dw = d2e = None
    null = lib.c_null_ptr()
    if with_dw:
        dw = numpy.empty((natm,3,n))
    if eps is not None:
        eps = numpy.asarray(eps, dtype=numpy.double, order='C')
        d2e = numpy.zeros((natm,3,natm,3))
    libdft.VXCbecke_weight_derivs(
        null if dw is None else dw.ctypes.data_as(ctypes.c_void_p),
        null if d2e is None else d2e.ctypes.data_as(ctypes.c_void_p),
        null if eps is None else eps.ctypes.data_as(ctypes.c_void_p),
        coords.ctypes.data_as(ctypes.c_void_p), qweights.ctypes.data_as(ctypes.c_void_p),
        atm_idx.ctypes.data_as(ctypes.c_void_p), atm_coords.ctypes.data_as(ctypes.c_void_p),
        null if table is None else table.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(n), ctypes.c_int(natm))
    if d2e is not None:
        d2e = d2e.transpose(0,2,1,3)
    return dw, d2e

def _nonzero_aos(mol, mask, nao):
    '''The AOs of the shells that mask (a block's non0tab) keeps on some point of the block'''
    if mask is None:
        return numpy.arange(nao)
    shls = numpy.nonzero(numpy.asarray(mask).any(axis=0))[0]
    ao_loc = mol.ao_loc_nr()
    if shls.size == 0:
        return numpy.zeros(0, dtype=int)
    return numpy.hstack([numpy.arange(ao_loc[i], ao_loc[i+1]) for i in shls])

class _AtomSegments:
    '''Sums over the AOs of each atom, for AOs idx sorted by atom'''
    def __init__(self, ao_atm, natm):
        self.natm = natm
        self.starts = numpy.hstack(([0], numpy.nonzero(numpy.diff(ao_atm))[0] + 1))
        self.atoms = ao_atm[self.starts]

    def rows(self, x):
        '''(ng, nao) -> (natm, ng)'''
        out = numpy.zeros((self.natm, x.shape[0]))
        out[self.atoms] = numpy.add.reduceat(x, self.starts, axis=1).T
        return out

    def vec(self, x):
        '''(nao,) -> (natm,)'''
        out = numpy.zeros(self.natm)
        out[self.atoms] = numpy.add.reduceat(x, self.starts)
        return out

def _orbital_response_rho(ao, X, xctype, seg):
    '''drho/dR [natm,3,ncomp,ng] of the AO derivatives (the atoms' AOs move) for X[c] = ao[c] dm'''
    ncomp = {'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype]
    ng = ao.shape[1]
    out = numpy.empty((seg.natm,3,ncomp,ng))
    for D in range(3):
        out[:,D,0] = seg.rows(ao[1+D] * X[0]) * -2
        if ncomp > 1:
            for x in range(3):
                out[:,D,1+x] = seg.rows(ao[_D2[D][x]] * X[0] + ao[1+D] * X[1+x]) * -2
        if xctype == 'MGGA':
            out[:,D,4] = -seg.rows(sum(ao[_D2[D][x]] * X[1+x] for x in range(3)))
    return out

def _grid_d2rho_dot(ao, X, u, xctype, seg):
    '''Q[B,D,d] = sum_g sum_k u[k,g] d/dr_d (drho_k/dR_B,D)(g): the potential u times the motion of the points
    (d) applied to the orbital response of atom B (D), for X[c] = ao[c] dm on the points of ao
    (VXCgrid_d2rho_dot)'''
    ao = numpy.asarray(ao, dtype=numpy.double, order='C')
    X = numpy.asarray(X, dtype=numpy.double, order='C')
    u = numpy.asarray(u, dtype=numpy.double, order='C')
    ng, nao = ao.shape[1:]
    q = numpy.zeros((nao,3,3))
    libdft.VXCgrid_d2rho_dot(
        q.ctypes.data_as(ctypes.c_void_p), ao.ctypes.data_as(ctypes.c_void_p),
        X.ctypes.data_as(ctypes.c_void_p), u.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int({'LDA': 0, 'GGA': 1, 'MGGA': 2}[xctype]), ctypes.c_int(ng), ctypes.c_int(nao))
    Q = numpy.zeros((seg.natm,3,3))
    Q[seg.atoms] = -numpy.add.reduceat(q, seg.starts, axis=0)
    return Q

def _get_exc_deriv2_grid_response(hessobj, mo_coeff, mo_occ, max_memory, dms=None):
    '''The grid response of the XC energy's second derivative at fixed orbitals, [natm,natm,3,3], for RKS (the
    density of mo_coeff, mo_occ) or, with dms [2,nao,nao], for the spin densities of UKS. With
    E = sum_g w_g eps(rho(r_g)), the terms

        sum_g d2w eps + (dw x vxc drho + transpose) + w fxc (drho drho - drho_AO drho_AO)
        + w vxc (d2rho - d2rho_AO)

    where drho = drho_AO + the motion of the point (nabla rho, on its atom), and the terms with drho_AO and
    d2rho_AO alone are those of _get_vxc_diag and _get_vxc_deriv2_dm.'''
    mol = hessobj.mol
    mf = hessobj.base
    grids = hessobj.grids if hessobj.grids is not None else mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)
    ni = mf._numint
    xctype = ni._xc_type(mf.xc)
    natm = mol.natm
    nao = mol.nao
    de2 = numpy.zeros((natm,natm,3,3))
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        return de2
    if dms is None:
        dms = mf.make_rdm1(mo_coeff, mo_occ)[None]
        spin = 0
    else:
        spin = 1
    nspin = len(dms)
    ncomp = {'LDA': 1, 'GGA': 4, 'MGGA': 5}[xctype]
    ao_deriv = 2 if xctype == 'LDA' else 3
    nX = 4 if xctype == 'LDA' else 10
    table = _becke_radii_table(mol, grids)
    ao_atm = numpy.empty(nao, dtype=int)
    for ia, (p0, p1) in enumerate(mol.aoslice_by_atom()[:,2:]):
        ao_atm[p0:p1] = ia
    comp = (ao_deriv+1)*(ao_deriv+2)*(ao_deriv+3)//6
    max_memory = _xc_block_memory(mf, max_memory, ao_deriv, (comp + nX*nspin + 6) * nao
                                  + natm*3*(ncomp*nspin*2 + 4))
    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
        p0, p1 = p1, p1 + weight.size
        idx = _nonzero_aos(mol, mask, nao)
        if idx.size == 0:
            continue
        ng = weight.size
        ao_s = ao[:,:,idx]
        seg = _AtomSegments(ao_atm[idx], natm)
        X = []
        rho = numpy.empty((nspin,ncomp,ng))
        for s, dm in enumerate(dms):
            dm_s = numpy.asarray(dm[idx[:,None],idx], order='C')
            Xs = numpy.empty((nX,ng,idx.size))
            for c in range(nX):
                Xs[c] = lib.dot(ao_s[c], dm_s)
            rho[s,0] = numpy.einsum('gm,gm->g', ao_s[0], Xs[0])
            if ncomp > 1:
                for x in range(3):
                    rho[s,1+x] = numpy.einsum('gm,gm->g', ao_s[1+x], Xs[0]) * 2
            if xctype == 'MGGA':
                rho[s,4] = sum(numpy.einsum('gm,gm->g', ao_s[1+x], Xs[1+x]) for x in range(3)) * .5
            X.append(Xs)
        rho_in = rho[0] if spin == 0 else rho
        if xctype == 'LDA':
            rho_in = rho_in[...,0,:]
        exc, vxc, fxc = ni.eval_xc_eff(mf.xc, rho_in, 2, xctype=xctype, spin=spin)[:3]
        vxc = vxc.reshape(nspin,ncomp,ng)
        fxc = fxc.reshape(nspin,ncomp,nspin,ncomp,ng)
        eps = exc.ravel() * rho[:,0].sum(axis=0)

        # sum_g d2w eps, and the derivatives of the weights
        dw, d2w = _becke_weight_derivs(mol, grids, p0, p1, eps, True, table)
        de2 += d2w

        # the orbital response and the motion of the points
        O = numpy.array([_orbital_response_rho(ao_s, X[s], xctype, seg) for s in range(nspin)])
        T = -O.sum(axis=1)
        own = grids.atm_idx[p0:p1]
        owners = [(G, own == G) for G in numpy.unique(own[own >= 0])]

        # dw x vxc drho + transpose
        F = numpy.einsum('sBDkg,skg->BDg', O, vxc)
        FT = numpy.einsum('sDkg,skg->Dg', T, vxc)
        for G, sel in owners:
            F[G][:,sel] += FT[:,sel]
        M = lib.dot(dw.reshape(natm*3,ng), F.reshape(natm*3,ng).T).reshape(natm,3,natm,3)
        de2 += M.transpose(0,2,1,3) + M.transpose(2,0,3,1)
        F = FT = M = dw = None

        # w fxc: the motion of the points of G with the orbital response of B, of G (and the transposes);
        # w vxc: the motion of the points of G applied to the orbital response of B
        u = vxc * weight
        Y = numpy.einsum('tdkg,tkslg->sdlg', T, fxc * weight)
        for G, sel in owners:
            S = numpy.einsum('sdlg,sBDlg->BDd', Y[:,:,:,sel], O[:,:,:,:,sel])
            ao_g = ao_s[:,sel]
            for s in range(nspin):
                S += _grid_d2rho_dot(ao_g, X[s][:,sel], u[s][:,sel], xctype, seg)
            de2[G] += S.transpose(0,2,1)
            de2[:,G] += S
            de2[G,G] -= S.sum(axis=0).T
        O = T = Y = X = None
    return de2

def _grid_response_wv_(wv, vxc, own, dw, fac):
    '''Adds to wv [natm,3,ncomp,ng], fac times minus the potentials of the orbital response (as _get_vxc_deriv1
    holds them: fac .5 for LDA, 1 otherwise), those of the grid response: of the motion of the points (on their
    atoms, minus the sum over the atoms of the orbital response) and of the weights, dw [natm,3,ng] times vxc
    [ncomp,ng]'''
    s = wv.sum(axis=0)
    for G in numpy.unique(own[own >= 0]):
        sel = own == G
        wv[G][:,:,sel] -= s[:,:,sel]
    wv -= fac * dw[:,:,None,:] * vxc
    return wv

def _grid_motion_vmat_(vmat, mol, ao, wv, xctype, own, idx):
    '''vmat[G] -= the AO-derivative matrix of the gradient (v_ip of _get_vxc_deriv1) of the potential wv on the
    points of atom G alone, for each atom G of the points of ao (own), on the AOs idx: the derivative of the AO
    products with the ground-state potential as the points move'''
    for G in numpy.unique(own[own >= 0]):
        sel = own == G
        ao_g = numpy.asarray(ao[:,sel][:,:,idx], order='C')
        wv_g = numpy.asarray(wv[:,sel], order='C')
        v = numpy.zeros((3,idx.size,idx.size))
        if xctype == 'LDA':
            aow = numint._scale_ao(ao_g[0], wv_g[0])
            rks_grad._d1_dot_(v, mol, ao_g[1:4], aow, None, None, True)
        else:
            rks_grad._gga_grad_sum_(v, mol, ao_g, wv_g, None, None)
            if xctype == 'MGGA':
                rks_grad._tau_grad_dot_(v, mol, ao_g, wv_g[4], None, None, True)
        vmat[G][:,idx[:,None],idx] -= v


class Hessian(rhf_hess.HessianBase):
    '''Non-relativistic RKS hessian.

    grid_response = True adds the response of the XC (and VV10) grids: their points
    move with their atoms and their Becke weights depend on all atom positions. The
    Hessian is then the second derivative of the energy as computed (that of the
    gradient with grid_response = True); without it, the terms of the grids are left
    out, which for meta-GGAs on heavier atoms can be several percent of the force
    constants at production grids.
    '''

    _keys = {'grids', 'grid_response'}

    def __init__(self, mf):
        rhf_hess.Hessian.__init__(self, mf)
        self.grids = None
        self.grid_response = False

    partial_hess_elec = partial_hess_elec
    hess_elec = rhf_hess.hess_elec
    make_h1 = make_h1

from pyscf import dft
dft.rks.RKS.Hessian = dft.rks_symm.RKS.Hessian = lib.class_as_method(Hessian)
dft.roks.ROKS.Hessian = dft.rks_symm.ROKS.Hessian = lib.invalid_method('Hessian')
