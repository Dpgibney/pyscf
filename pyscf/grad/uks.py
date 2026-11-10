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

'''Non-relativistic UKS analytical nuclear gradients'''


import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from pyscf.grad import rks as rks_grad
from pyscf.grad import uhf as uhf_grad
from pyscf.dft import numint, gen_grid
from pyscf import __config__


def get_veff(ks_grad, mol=None, dm=None):
    '''
    First order derivative of DFT effective potential matrix (wrt electron coordinates)

    Args:
        ks_grad : grad.uhf.Gradients or grad.uks.Gradients object
    '''
    if mol is None: mol = ks_grad.mol
    if dm is None: dm = ks_grad.base.make_rdm1()
    exc, vxc = _get_vxc_veff(ks_grad, mol, dm)

    mf = ks_grad.base
    ni = mf._numint
    if not ni.libxc.is_hybrid_xc(mf.xc):
        vj = ks_grad.get_j(mol, dm)
        vxc += vj[0] + vj[1]
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        vj, vk = ks_grad.get_jk(mol, dm)
        vk *= hyb
        if omega != 0:
            vk += ks_grad.get_k(mol, dm, omega=omega) * (alpha - hyb)
        vxc += vj[0] + vj[1] - vk
    return lib.tag_array(vxc, exc1_grid=exc)

def _get_veff_split(ks_grad, mol, dm):
    '''get_veff for grad_elec: the XC potential, and the energy gradient of J/K per atom. Where
    the XC term comes contracted with the density (_get_vxc_de), there is no potential (None)
    and the energy gradient holds the XC term as well.'''
    de_xc = _get_vxc_de(ks_grad, mol, dm)
    if de_xc is None:
        exc, vxc = _get_vxc_veff(ks_grad, mol, dm)
        vxc = lib.tag_array(vxc, exc1_grid=exc)
    else:
        vxc = None
    mf = ks_grad.base
    ni = mf._numint
    if not ni.libxc.is_hybrid_xc(mf.xc):
        de = rhf_grad.jk_energy_per_atom(mol, dm, 1, 0)
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
        de = rhf_grad.jk_energy_per_atom(mol, dm, 1, hyb, screen='jk')
        if omega != 0:
            de += rhf_grad.jk_energy_per_atom(mol, dm, 0, alpha - hyb, omega=omega)
    if de_xc is not None:
        de += de_xc
    return vxc, de

def _get_vxc_de(ks_grad, mol, dm, max_memory=None):
    '''The XC term of the gradient (with VV10) per atom [natm,3], what grad_elec makes of the
    potential of _get_vxc_veff, from the alpha and beta orbitals on the grid without the
    [2,3,nao,nao] matrices: grad.rks._get_vxc_de for UKS. None where this path does not apply.'''
    if not rks_grad.XC_DM or ks_grad.grid_response:
        return None
    mo_coeff = getattr(dm, 'mo_coeff', None)
    mo_occ = getattr(dm, 'mo_occ', None)
    if (mo_coeff is None or mo_occ is None or numpy.ndim(dm) != 3 or len(mo_coeff) != 2 or
            numpy.ndim(mo_coeff[0]) != 2 or numpy.iscomplexobj(mo_coeff[0]) or
            numpy.iscomplexobj(mo_coeff[1])):
        return None
    t0 = (logger.process_clock(), logger.perf_counter())
    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = rks_grad._initialize_grids(ks_grad)
    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    xctype = ni._xc_type(mf.xc)
    if xctype not in ('LDA', 'GGA', 'MGGA') or (xctype == 'MGGA' and numint.MGGA_DENSITY_LAPL):
        return None
    fused = numint._fused_xc(ni, mol, grids, mol.nao, xctype, max_memory,
                             (mo_coeff[0], mo_coeff[1]), match_block_loop=False)
    if fused is None:
        return None
    g = fused.get_vxc_grad_q_uks(ni, mf.xc, xctype, mo_coeff, mo_occ)
    if g is None:
        return None
    # - sign because nabla_X = -nabla_x
    g = -(g[0] + g[1])
    if mf.do_nlc():
        if ni.libxc.is_nlc(mf.xc):
            xc = mf.xc
        else:
            xc = mf.nlc
        # the VV10 potential is that of the total density, whose orbitals are
        # the alpha and the beta ones together
        dm_sf = numpy.asarray(dm[0] + dm[1])
        mo_sf = numpy.hstack([mo_coeff[0], mo_coeff[1]])
        occ_sf = numpy.hstack([mo_occ[0], mo_occ[1]])
        gnlc = rks_grad.get_nlc_vxc_dm(ni, mol, nlcgrids, xc, dm_sf, mo_sf, occ_sf, max_memory)
        if gnlc is None:
            vnlc = rks_grad.get_nlc_vxc(ni, mol, nlcgrids, xc, dm_sf, max_memory=max_memory,
                                        verbose=ks_grad.verbose)[1]
            gnlc = numpy.einsum('xij,ij->xi', vnlc, dm_sf)
        g += gnlc
    aoslices = mol.aoslice_by_atom()
    de = numpy.array([g[:,p0:p1].sum(axis=1) for p0, p1 in aoslices[:,2:]]) * 2
    logger.timer(ks_grad, 'vxc', *t0)
    return de

def _get_vxc_veff(ks_grad, mol, dm):
    '''The XC part of get_veff, and the grid response exc1_grid'''
    t0 = (logger.process_clock(), logger.perf_counter())

    mf = ks_grad.base
    ni = mf._numint
    grids, nlcgrids = rks_grad._initialize_grids(ks_grad)

    ni = mf._numint
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, ks_grad.max_memory*.9-mem_now)
    if ks_grad.grid_response:
        exc, vxc = get_vxc_full_response(ni, mol, grids, mf.xc, dm,
                                         max_memory=max_memory,
                                         verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc_full_response(
                ni, mol, nlcgrids, xc, dm[0]+dm[1],
                max_memory=max_memory, verbose=ks_grad.verbose)
            exc += enlc
            vxc += vnlc
        logger.debug1(ks_grad, 'sum(grids response) %s', exc.sum(axis=0))
    else:
        exc, vxc = get_vxc(ni, mol, grids, mf.xc, dm,
                           max_memory=max_memory, verbose=ks_grad.verbose)
        if mf.do_nlc():
            if ni.libxc.is_nlc(mf.xc):
                xc = mf.xc
            else:
                xc = mf.nlc
            enlc, vnlc = rks_grad.get_nlc_vxc(
                ni, mol, nlcgrids, xc, dm[0]+dm[1],
                max_memory=max_memory, verbose=ks_grad.verbose)
            vxc += vnlc
    t0 = logger.timer(ks_grad, 'vxc', *t0)
    return exc, vxc


def get_vxc(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()

    vmat = numpy.zeros((2,3,nao,nao))
    fused = numint._fused_xc(ni, mol, grids, nao, xctype, max_memory, (dms,))
    if fused is not None:
        fused.get_vxc_grad(ni, xc_code, xctype, dms, hermi, 1, vmat)

    elif xctype == 'LDA':
        ao_deriv = 1
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho_a = make_rho(0, ao[0], mask, xctype)
            rho_b = make_rho(1, ao[0], mask, xctype)
            vxc = ni.eval_xc_eff(xc_code, (rho_a,rho_b), 1, xctype=xctype)[1]
            wv = weight * vxc[:,0]
            aow = numint._scale_ao(ao[0], wv[0])
            rks_grad._d1_dot_(vmat[0], mol, ao[1:4], aow, mask, ao_loc, True)
            aow = numint._scale_ao(ao[0], wv[1])
            rks_grad._d1_dot_(vmat[1], mol, ao[1:4], aow, mask, ao_loc, True)

    elif xctype == 'GGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho_a = make_rho(0, ao[:4], mask, xctype)
            rho_b = make_rho(1, ao[:4], mask, xctype)
            vxc = ni.eval_xc_eff(xc_code, (rho_a,rho_b), 1, xctype=xctype)[1]
            wv = weight * vxc
            wv[:,0] *= .5
            rks_grad._gga_grad_sum_(vmat[0], mol, ao, wv[0], mask, ao_loc)
            rks_grad._gga_grad_sum_(vmat[1], mol, ao, wv[1], mask, ao_loc)

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')

    elif xctype == 'MGGA':
        ao_deriv = 2
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rho_a = make_rho(0, ao[:10], mask, xctype)
            rho_b = make_rho(1, ao[:10], mask, xctype)
            vxc = ni.eval_xc_eff(xc_code, (rho_a,rho_b), 1, xctype=xctype)[1]
            wv = weight * vxc
            wv[:,0] *= .5
            wv[:,4] *= .5
            rks_grad._gga_grad_sum_(vmat[0], mol, ao, wv[0], mask, ao_loc)
            rks_grad._gga_grad_sum_(vmat[1], mol, ao, wv[1], mask, ao_loc)
            rks_grad._tau_grad_dot_(vmat[0], mol, ao, wv[0,4], mask, ao_loc, True)
            rks_grad._tau_grad_dot_(vmat[1], mol, ao, wv[1,4], mask, ao_loc, True)

    exc = numpy.zeros((mol.natm,3))
    # - sign because nabla_X = -nabla_x
    return exc, -vmat


def get_vxc_full_response(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
                          max_memory=2000, verbose=None):
    '''Full response including the response of the grids'''
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()
    aoslices = mol.aoslice_by_atom()

    excsum = 0
    vmat = numpy.zeros((2,3,nao,nao))
    if xctype == 'LDA':
        ao_deriv = 1
        for atm_id, (coords, weight, weight1) \
                in enumerate(rks_grad.grids_response_cc(grids)):
            sh0, sh1 = aoslices[atm_id][:2]
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                            cutoff=grids.cutoff)
            rho_a = make_rho(0, ao[0], mask, xctype)
            rho_b = make_rho(1, ao[0], mask, xctype)
            exc, vxc = ni.eval_xc_eff(xc_code, (rho_a,rho_b), 1, xctype=xctype)[:2]
            wv = weight * vxc[:,0]

            vtmp = numpy.zeros((3,nao,nao))
            aow = numint._scale_ao(ao[0], wv[0])
            rks_grad._d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
            vmat[0] += vtmp
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho_a+rho_b, weight1)
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms[0]) * 2

            vtmp = numpy.zeros((3,nao,nao))
            aow = numint._scale_ao(ao[0], wv[1])
            rks_grad._d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
            vmat[1] += vtmp
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms[1]) * 2

    elif xctype == 'GGA':
        ao_deriv = 2
        for atm_id, (coords, weight, weight1) \
                in enumerate(rks_grad.grids_response_cc(grids)):
            sh0, sh1 = aoslices[atm_id][:2]
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                            cutoff=grids.cutoff)
            rho_a = make_rho(0, ao[:4], mask, xctype)
            rho_b = make_rho(1, ao[:4], mask, xctype)
            exc, vxc = ni.eval_xc_eff(xc_code, (rho_a,rho_b), 1, xctype=xctype)[:2]
            wv = weight * vxc
            wv[:,0] *= .5

            vtmp = numpy.zeros((3,nao,nao))
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wv[0], mask, ao_loc)
            vmat[0] += vtmp
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho_a[0]+rho_b[0], weight1)
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms[0]) * 2

            vtmp = numpy.zeros((3,nao,nao))
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wv[1], mask, ao_loc)
            vmat[1] += vtmp
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms[1]) * 2

    elif xctype == 'NLC':
        raise NotImplementedError('NLC')

    elif xctype == 'MGGA':
        ao_deriv = 2
        for atm_id, (coords, weight, weight1) \
                in enumerate(rks_grad.grids_response_cc(grids)):
            sh0, sh1 = aoslices[atm_id][:2]
            mask = gen_grid.make_mask(mol, coords)
            ao = ni.eval_ao(mol, coords, deriv=ao_deriv, non0tab=mask,
                            cutoff=grids.cutoff)
            rho_a = make_rho(0, ao[:10], mask, xctype)
            rho_b = make_rho(1, ao[:10], mask, xctype)
            exc, vxc = ni.eval_xc_eff(xc_code, (rho_a,rho_b), 1, xctype=xctype)[:2]
            wv = weight * vxc
            wv[:,0] *= .5
            wv[:,4] *= .5

            vtmp = numpy.zeros((3,nao,nao))
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wv[0], mask, ao_loc)
            rks_grad._tau_grad_dot_(vtmp, mol, ao, wv[0,4], mask, ao_loc, True)
            vmat[0] += vtmp
            excsum += numpy.einsum('r,r,nxr->nx', exc, rho_a[0]+rho_b[0], weight1)
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms[0]) * 2

            vtmp = numpy.zeros((3,nao,nao))
            rks_grad._gga_grad_sum_(vtmp, mol, ao, wv[1], mask, ao_loc)
            rks_grad._tau_grad_dot_(vtmp, mol, ao, wv[1,4], mask, ao_loc, True)
            vmat[1] += vtmp
            excsum[atm_id] += numpy.einsum('xij,ji->x', vtmp, dms[1]) * 2

    # - sign because nabla_X = -nabla_x
    return excsum, -vmat


class Gradients(uhf_grad.Gradients):

    grid_response = getattr(__config__, 'grad_uks_Gradients_grid_response', False)

    _keys = {'grid_response', 'grids', 'nlcgrids'}

    def __init__(self, mf):
        uhf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.nlcgrids = None

    def dump_flags(self, verbose=None):
        uhf_grad.Gradients.dump_flags(self, verbose)
        logger.info(self, 'grid_response = %s', self.grid_response)
        return self

    get_veff = get_veff

    def extra_force(self, atom_id, envs):
        '''Hook for extra contributions in analytical gradients.

        Contributions like the response of auxiliary basis in density fitting
        method, the grid response in DFT numerical integration can be put in
        this function.
        '''
        e1 = 0
        vhf = envs['vhf']
        # the XC term contracted with the density on the grid, where get_veff
        # left it out of the potential (_get_vxc_de)
        de_xc = getattr(vhf, 'de_xc', None)
        if de_xc is not None:
            e1 = e1 + de_xc[atom_id]
        if self.grid_response:
            log = envs['log']
            log.debug('grids response for atom %d %s',
                      atom_id, vhf.exc1_grid[atom_id])
            e1 = e1 + vhf.exc1_grid[atom_id]
        return e1

Grad = Gradients

rhf_grad._VEFF_SPLIT[get_veff] = _get_veff_split

from pyscf import dft
dft.uks.UKS.Gradients = dft.uks_symm.UKS.Gradients = lib.class_as_method(Gradients)
