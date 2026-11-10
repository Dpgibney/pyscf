#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
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
# Author: Xiaojie Wu <wxj6000@gmail.com>
#

'''
PCM family solvent models
'''

import ctypes
import numpy
import scipy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf.dft import gen_grid
from pyscf.data import radii
from pyscf.solvent import ddcosmo
from pyscf.solvent import _attach_solvent
from pyscf.solvent import _solvent_data
from scipy.special import erf

def _as_solvent_obj(solvent_obj, mol):
    '''Allows the solvent model to be specified by the name of the solvent,
    e.g. mf.PCM('toluene')
    '''
    if isinstance(solvent_obj, str):
        solvent_obj = PCM(mol, solvent_obj)
    return solvent_obj

@lib.with_doc(_attach_solvent._for_scf.__doc__)
def pcm_for_scf(mf, solvent_obj=None, dm=None):
    solvent_obj = _as_solvent_obj(solvent_obj, mf.mol)
    if solvent_obj is None:
        solvent_obj = PCM(mf.mol)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_casscf.__doc__)
def pcm_for_casscf(mc, solvent_obj=None, dm=None):
    solvent_obj = _as_solvent_obj(solvent_obj, mc.mol)
    if solvent_obj is None:
        if isinstance(getattr(mc._scf, 'with_solvent', None), PCM):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = PCM(mc.mol)
    return _attach_solvent._for_casscf(mc, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_casci.__doc__)
def pcm_for_casci(mc, solvent_obj=None, dm=None):
    solvent_obj = _as_solvent_obj(solvent_obj, mc.mol)
    if solvent_obj is None:
        if isinstance(getattr(mc._scf, 'with_solvent', None), PCM):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = PCM(mc.mol)
    return _attach_solvent._for_casci(mc, solvent_obj, dm)

@lib.with_doc(_attach_solvent._for_post_scf.__doc__)
def pcm_for_post_scf(method, solvent_obj=None, dm=None):
    solvent_obj = _as_solvent_obj(solvent_obj, method.mol)
    if solvent_obj is None:
        if isinstance(getattr(method._scf, 'with_solvent', None), PCM):
            solvent_obj = method._scf.with_solvent
        else:
            solvent_obj = PCM(method.mol)
    return _attach_solvent._for_post_scf(method, solvent_obj, dm)

pcm_for_tdscf = _attach_solvent._for_tdscf


# Inject PCM to other methods
from pyscf import scf
from pyscf import mcscf
from pyscf import mp, ci, cc
from pyscf import tdscf
scf.hf.SCF.PCM    = scf.hf.SCF.PCM    = pcm_for_scf
mp.mp2.MP2.PCM    = mp.mp2.MP2.PCM    = pcm_for_post_scf
ci.cisd.CISD.PCM  = ci.cisd.CISD.PCM  = pcm_for_post_scf
cc.ccsd.CCSD.PCM  = cc.ccsd.CCSD.PCM  = pcm_for_post_scf
tdscf.rhf.TDBase.PCM = tdscf.rhf.TDBase.PCM = pcm_for_tdscf
mcscf.casci.CASCI.PCM = mcscf.casci.CASCI.PCM = pcm_for_casci
mcscf.mc1step.CASSCF.PCM = mcscf.mc1step.CASSCF.PCM = pcm_for_casscf

# TABLE II,  J. Chem. Phys. 122, 194110 (2005)
XI = {
    6: 4.84566077868,
    14: 4.86458714334,
    26: 4.85478226219,
    38: 4.90105812685,
    50: 4.89250673295,
    86: 4.89741372580,
    110: 4.90101060987,
    146: 4.89825187392,
    170: 4.90685517725,
    194: 4.90337644248,
    302: 4.90498088169,
    350: 4.86879474832,
    434: 4.90567349080,
    590: 4.90624071359,
    770: 4.90656435779,
    974: 4.90685167998,
    1202: 4.90704098216,
    1454: 4.90721023869,
    1730: 4.90733270691,
    2030: 4.90744499142,
    2354: 4.90753082825,
    2702: 4.90760972766,
    3074: 4.90767282394,
    3470: 4.90773141371,
    3890: 4.90777965981,
    4334: 4.90782469526,
    4802: 4.90749125553,
    5294: 4.90762073452,
    5810: 4.90792902522,
}

modified_Bondi = radii.VDW.copy()
modified_Bondi[1] = 1.1/radii.BOHR      # modified version
PI = numpy.pi

def switch_h(x):
    '''
    switching function (eq. 3.19)
    J. Chem. Phys. 133, 244111 (2010)
    notice the typo in the paper
    '''
    y = x**3 * (10.0 - 15.0*x + 6.0*x**2)
    y[x<0] = 0.0
    y[x>1] = 1.0
    return y

def gen_surface(mol, ng=302, rad=modified_Bondi, surface_discretization_method="SWIG"):
    '''J. Phys. Chem. A 1999, 103, 11060-11079'''
    unit_sphere = gen_grid.MakeAngularGrid(ng)
    atom_coords = mol.atom_coords(unit='B')
    N_J = ng * numpy.ones(mol.natm)
    from pyscf.data.elements import charge as charge_of_element
    element_index = [charge_of_element(e) for e in mol.elements]
    R_J = numpy.asarray([rad[chg] for chg in element_index])
    if surface_discretization_method.upper() == "SWIG":
        R_sw_J = R_J * (14.0 / N_J)**0.5
        alpha_J = 1.0/2.0 + R_J/R_sw_J - ((R_J/R_sw_J)**2 - 1.0/28)**0.5
        R_in_J = R_J - alpha_J * R_sw_J

    grid_coords = []
    weights = []
    charge_exp = []
    switch_fun = []
    R_vdw = []
    norm_vec = []
    area = []
    gslice_by_atom = []
    p0 = p1 = 0
    for ia in range(mol.natm):
        r_vdw = R_J[ia]

        atom_grid = r_vdw * unit_sphere[:,:3] + atom_coords[ia,:]
        riJ = scipy.spatial.distance.cdist(atom_grid[:,:3], atom_coords)

        w = unit_sphere[:,3] * 4.0 * PI
        xi = XI[ng] / (r_vdw * w**0.5)

        if surface_discretization_method.upper() == "SWIG":
            diJ = (riJ - R_in_J) / R_sw_J
            diJ[:,ia] = 1.0
            diJ[diJ < 1e-8] = 0.0
            fiJ = switch_h(diJ)
        elif surface_discretization_method.upper() == "ISWIG":
            fiJ = 1 - 0.5 * (erf(xi[:, None] * (R_J[None, :] - riJ)) + erf(xi[:, None] * (R_J[None, :] + riJ)))
            fiJ[:,ia] = 1.0
            fiJ[fiJ < 1e-8] = 0
        else:
            raise NotImplementedError(f"surface_discretization_method = {surface_discretization_method} not recognized")

        swf = numpy.prod(fiJ, axis=1)

        idx = w*swf > 1e-16

        p0, p1 = p1, p1+sum(idx)
        gslice_by_atom.append([p0,p1])
        grid_coords.append(atom_grid[idx,:3])
        weights.append(w[idx])
        switch_fun.append(swf[idx])
        norm_vec.append(unit_sphere[idx,:3])
        charge_exp.append(xi[idx])
        R_vdw.append(numpy.ones(sum(idx)) * r_vdw)
        area.append(w[idx]*r_vdw**2*swf[idx])

    grid_coords = numpy.vstack(grid_coords)
    norm_vec = numpy.vstack(norm_vec)
    weights = numpy.concatenate(weights)
    charge_exp = numpy.concatenate(charge_exp)
    switch_fun = numpy.concatenate(switch_fun)
    area = numpy.concatenate(area)
    R_vdw = numpy.concatenate(R_vdw)

    surface = {
        'ng': ng,
        'gslice_by_atom': gslice_by_atom,
        'grid_coords': grid_coords,
        'weights': weights,
        'charge_exp': charge_exp,
        'switch_fun': switch_fun,
        'R_vdw': R_vdw,
        'norm_vec': norm_vec,
        'area': area,
        'atom_coords': atom_coords
    }
    if surface_discretization_method.upper() == "SWIG":
        surface.update({
            'R_in_J': R_in_J,
            'R_sw_J': R_sw_J,
        })
    elif surface_discretization_method.upper() == "ISWIG":
        surface.update({
            'R_J': R_J,
        })
    return surface

def get_F_A(surface):
    '''
    generate F and A matrix in  J. Chem. Phys. 133, 244111 (2010)
    '''
    R_vdw = surface['R_vdw']
    switch_fun = surface['switch_fun']
    weights = surface['weights']
    A = weights*R_vdw**2*switch_fun
    return switch_fun, A


def get_D_S(surface, with_S=True, with_D=False):
    '''
    generate D and S matrix in  J. Chem. Phys. 133, 244111 (2010)
    The diagonal entries of S is not filled

    The matrices are made a block of rows at a time, so that the temporaries are
    a few (rows, ngrids) arrays rather than a dozen (ngrids, ngrids) ones; every
    element is computed as before.
    '''
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    norm_vec    = surface['norm_vec']
    R_vdw       = surface['R_vdw']
    ngrids = grid_coords.shape[0]

    S = numpy.empty((ngrids, ngrids))
    D = numpy.empty((ngrids, ngrids)) if with_D else None
    # about 128 MB for each of the (rows, ngrids, 3) temporaries
    rows = max(1, int(128e6 / 8 / (ngrids * 3)))
    xi_j = charge_exp[None,:]
    for i0, i1 in lib.prange(0, ngrids, rows):
        xi_i = charge_exp[i0:i1,None]
        xi_ij = xi_i * xi_j / (xi_i**2 + xi_j**2)**0.5
        rij = scipy.spatial.distance.cdist(grid_coords[i0:i1], grid_coords)
        xi_r_ij = xi_ij * rij
        rij[numpy.arange(i1-i0), numpy.arange(i0,i1)] = 1
        S[i0:i1] = erf(xi_r_ij) / rij
        if with_D:
            # The diagonal of S is not set yet here; that of D is overwritten below
            drij = numpy.expand_dims(grid_coords[i0:i1], axis=1) - grid_coords
            nrij = numpy.sum(drij * norm_vec, axis=-1)
            D[i0:i1] = S[i0:i1]*nrij/rij**2 -2.0*xi_r_ij/PI**0.5*numpy.exp(-xi_r_ij**2)*nrij/rij**3
    numpy.fill_diagonal(S, charge_exp * (2.0 / PI)**0.5 / switch_fun)
    if with_D:
        numpy.fill_diagonal(D, -charge_exp * (2.0 / PI)**0.5 / (2.0 * R_vdw))

    return D, S

# Shell pairs whose overlap (the largest |S_ij| of the pair) is below this are
# left out of the surface integrals; that changes no result beyond rounding.
INT1E_GRIDS_CUTOFF = 1e-15

def _int1e_grids_dot(mol, surface, dms=None, charges=None):
    r'''
    The integrals of the AO pairs with the Gaussian charges of the surface,
    (ij|g) = \int \phi_i \phi_j erf(\xi_g |r-R_g|) / |r-R_g| dr, contracted as
    they are made: sum_ij D_ij (ij|g) for each density matrix of dms, or
    sum_g q_g (ij|g) for each set of charges.

    int1e_grids takes the Gaussian charge model (PTR_RINV_ZETA) with one
    exponent at a time, and the charge exponents take one value per Lebedev
    weight and atomic radius. So the points are sorted by exponent and cut into
    batches that keep to one exponent, and the SGX driver runs all the batches,
    each with its exponent, over the threads in one call. It evaluates the
    integrals of each shell pair (i >= j) over a batch and contracts them at
    once; no (nao, nao, ngrids) tensor is kept.
    '''
    from pyscf.sgx import sgx
    from pyscf.scf import _vhf
    xi, group = numpy.unique(surface['charge_exp'], return_inverse=True)
    order = numpy.argsort(group, kind='stable')
    ngrids = order.size
    blksize = _vhf.libcvhf.SGXreturn_blksize()
    group_loc = numpy.append(0, numpy.cumsum(numpy.bincount(group)))
    batch_loc = numpy.append(
        numpy.hstack([numpy.arange(p0, p1, blksize)
                      for p0, p1 in zip(group_loc[:-1], group_loc[1:])]),
        ngrids).astype(numpy.int32)
    batch_zeta = numpy.ascontiguousarray(xi[group[order[batch_loc[:-1]]]]**2)
    coords = numpy.asarray(surface['grid_coords'][order], order='C')
    env = numpy.append(mol._env, coords.ravel())
    env[gto.NGRIDS] = ngrids
    env[gto.PTR_GRIDS] = mol._env.size
    sgxopt = sgx._make_opt(mol, False, INT1E_GRIDS_CUTOFF)
    ao_loc = gto.moleintor.make_loc(mol._bas, sgxopt._intor)
    if dms is not None:
        # (ij|g) is symmetric in ij, so only the symmetric part of D contributes
        ins = [numpy.ascontiguousarray(dm + dm.T) * .5 for dm in dms]
        out = numpy.zeros((len(ins), ngrids))
        fjk = 'SGXnrs2_ijg_ji_g'
    else:
        ins = [numpy.ascontiguousarray(c[order]) for c in charges]
        out = numpy.zeros((len(ins), mol.nao, mol.nao))
        fjk = 'SGXnrs2_ijg_g_ij'
    n = len(ins)
    def ptrs(arrays):
        return (ctypes.c_void_p*n)(*[a.ctypes.data_as(ctypes.c_void_p) for a in arrays])
    if dms is None:
        # The shell pairs over the threads, summed into the matrices directly:
        # the driver below would keep a matrix per set of charges in each thread
        _vhf.libcvhf.SGXnr_g_ij_drv_batches(
            _vhf._fpointer(sgxopt._intor), ptrs(ins), ptrs(out), ctypes.c_int(n),
            (ctypes.c_int*4)(0, mol.nbas, 0, mol.nbas),
            ao_loc.ctypes.data_as(ctypes.c_void_p),
            sgxopt._cintopt, ctypes.byref(sgxopt._this),
            mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
            env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(2),
            ctypes.c_int(batch_loc.size - 1), batch_loc.ctypes.data_as(ctypes.c_void_p),
            batch_zeta.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))
        for v in out:
            # the driver fills the lower triangle
            lib.hermi_triu(v, inplace=True)
        return out
    _vhf.libcvhf.SGXnr_direct_drv_batches(
        _vhf._fpointer(sgxopt._intor), (ctypes.c_void_p*n)(*[_vhf._fpointer(fjk)]*n),
        ptrs(ins), ptrs(out), ctypes.c_int(n), ctypes.c_int(1),
        (ctypes.c_int*4)(0, mol.nbas, 0, mol.nbas),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        sgxopt._cintopt, ctypes.byref(sgxopt._this),
        mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
        mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(2),
        ctypes.c_int(batch_loc.size - 1), batch_loc.ctypes.data_as(ctypes.c_void_p),
        batch_zeta.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))
    if dms is not None:
        v = numpy.empty_like(out)
        v[:,order] = out
        return v
    for v in out:
        # the driver fills the lower triangle
        lib.hermi_triu(v, inplace=True)
    return out

def _int1e_grids_deriv(mol, surface, intor, comp, dms=None, charges=None,
                       dms_i=None, grids=None):
    r'''
    The derivative integrals of the AO pairs with the Gaussian charges of the
    surface, intor = 'int1e_grids_ip' (\nabla i j|g), 'int1e_grids_ipip'
    (\nabla\nabla i j|g) or 'int1e_grids_ipvip' (\nabla i \nabla j|g), with
    comp components, contracted as they are made in one pass (see
    _int1e_grids_dot). The pairs are not symmetric. Any of

    dms     : sum_ij D_ij (ij|g)  -> [len(dms),comp,ngrids]
    dms_i   : sum_j D_ij (ij|g)   -> [len(dms_i),comp,nao,ngrids]
    charges : sum_g q_g (ij|g)    -> [len(charges),comp,nao,nao]

    in this order in the list returned. grids = (g0, g1) takes these surface
    points only (the charges and the outputs are of g1-g0 points then).

    The derivatives with respect to the position of a charge follow from
    these: (ij|\nabla g) = -(\nabla i j|g) - (i \nabla j|g).
    '''
    from pyscf.sgx import sgx
    from pyscf.scf import _vhf
    g0, g1 = (0, len(surface['charge_exp'])) if grids is None else grids
    xi, group = numpy.unique(surface['charge_exp'][g0:g1], return_inverse=True)
    order = numpy.argsort(group, kind='stable')
    ngrids = order.size
    nao = mol.nao
    blksize = _vhf.libcvhf.SGXreturn_blksize()
    group_loc = numpy.append(0, numpy.cumsum(numpy.bincount(group)))
    batch_loc = numpy.append(
        numpy.hstack([numpy.arange(p0, p1, blksize)
                      for p0, p1 in zip(group_loc[:-1], group_loc[1:])]),
        ngrids).astype(numpy.int32)
    batch_zeta = numpy.ascontiguousarray(xi[group[order[batch_loc[:-1]]]]**2)
    coords = numpy.asarray(surface['grid_coords'][g0:g1][order], order='C')
    env = numpy.append(mol._env, coords.ravel())
    env[gto.NGRIDS] = ngrids
    env[gto.PTR_GRIDS] = mol._env.size
    sgxopt = sgx._make_opt(mol, False, INT1E_GRIDS_CUTOFF)
    intor = mol._add_suffix(intor)
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, intor)
    ao_loc = gto.moleintor.make_loc(mol._bas, intor)
    ins, outs, ops, kinds = [], [], [], []
    # the operators index the density as D[j,i]
    for dm in (() if dms is None else dms):
        ins.append(numpy.ascontiguousarray(dm.T))
        outs.append(numpy.zeros((comp, ngrids)))
        ops.append('SGXnrs1_ijg_ji_g')
        kinds.append('g')
    for dm in (() if dms_i is None else dms_i):
        ins.append(numpy.ascontiguousarray(dm.T))
        outs.append(numpy.zeros((comp, nao, ngrids)))
        ops.append('SGXnrs1_ijg_ji_gi')
        kinds.append('g')
    for c in (() if charges is None else charges):
        ins.append(numpy.ascontiguousarray(c[order]))
        outs.append(numpy.zeros((comp, nao, nao)))
        ops.append('SGXnrs1_ijg_g_ij')
        kinds.append('ij')
    n = len(ins)
    def ptrs(arrays):
        return (ctypes.c_void_p*n)(*[a.ctypes.data_as(ctypes.c_void_p) for a in arrays])
    _vhf.libcvhf.SGXnr_direct_drv_batches(
        _vhf._fpointer(intor), (ctypes.c_void_p*n)(*[_vhf._fpointer(op) for op in ops]),
        ptrs(ins), ptrs(outs), ctypes.c_int(n), ctypes.c_int(comp),
        (ctypes.c_int*4)(0, mol.nbas, 0, mol.nbas),
        ao_loc.ctypes.data_as(ctypes.c_void_p),
        cintopt, ctypes.byref(sgxopt._this),
        mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
        mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
        env.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(1),
        ctypes.c_int(batch_loc.size - 1), batch_loc.ctypes.data_as(ctypes.c_void_p),
        batch_zeta.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(env.size))
    for k, kind in enumerate(kinds):
        if kind == 'g':
            v = numpy.empty_like(outs[k])
            v[...,order] = outs[k]
            outs[k] = v
    return outs

def _R_dot(pcmobj, x, trans=False):
    '''R x (R^T x with trans) over the last axis of x, the surface points.
    C-PCM and COSMO keep no R: it is -f_epsilon times the identity there.'''
    R = pcmobj._intermediates['R']
    if R is None:
        return -pcmobj._intermediates['f_epsilon'] * x
    # one matrix product: numpy.dot of an array of more than two indices with a
    # matrix is a loop of dot products
    x2 = x.reshape(-1, x.shape[-1])
    return numpy.dot(x2, R if trans else R.T).reshape(x.shape)

def _K_solve(pcmobj, b, trans=False):
    '''K^-1 b (K^-T b with trans) over the last axis of b, the surface points,
    with a factorization of K made at the first call and kept in
    _intermediates: Cholesky for C-PCM and COSMO, whose K = S is symmetric
    positive definite, LU otherwise.'''
    intermediates = pcmobj._intermediates
    if pcmobj.method.upper() in ['C-PCM', 'CPCM', 'COSMO']:
        if 'K_cho' not in intermediates:
            intermediates['K_cho'] = scipy.linalg.cho_factor(intermediates['K'])
        return scipy.linalg.cho_solve(intermediates['K_cho'], b.T).T
    if 'K_lu' not in intermediates:
        intermediates['K_lu'] = scipy.linalg.lu_factor(intermediates['K'])
    return scipy.linalg.lu_solve(intermediates['K_lu'], b.T, trans=int(trans)).T


class PCM(lib.StreamObject):
    '''
    PCM Solvent Model

    This class implements the Polarizable Continuum Model (PCM) for solvent effects.

    Examples:

    >>> mf = mol.RHF().PCM()
    >>> mf.with_solvent.solvent = 'toluene'
    >>> mf.kernel()

    Input Attributes:
    -----------------
    method : str
        The PCM model. Options include 'C-PCM', 'IEF-PCM', 'COSMO', and 'SS(V)PE'.
        Default is 'C-PCM'.

    vdw_scale : float
        A scaling factor for van der Waals radii. Default is 1.2, consistent with Q-Chem settings.

    r_probe : float
        An additional radius (in Angstrom) added to the van der Waals radii.
        Default is 0.0.

    radii_table : dict
        Custom van der Waals radii for each element. By default, scaled van der Waals radii
        from `vdw_scale` and `r_probe` are used.

    lebedev_order : int
        The order of the Lebedev mesh used for the cavity sphere. Default is 29 (302 grids).

    solvent : str
        The name of the solvent. Assigning this attribute applies the dielectric constant
        and the refractive index of the solvent to `eps` and `eps_optical`, which can be
        overwritten afterwards. Supported solvents can be accessed via the variable
        `pyscf.solvent.smd.solvent_db`. Their names are matched case-insensitively,
        ignoring spaces and hyphens, and common abbreviations (such as "DMSO") are
        recognized as well. Default is '' (the parameters of water are then applied).

    eps : float
        The dielectric constant of the solvent. Default is 78.3553, the dielectric constant
        for water. It is overwritten by `solvent` if a solvent name is specified.

    eps_optical : float
        The optical (high-frequency) dielectric constant of the solvent, i.e. the square of
        its refractive index. It is only used by the non-equilibrium solvation of excited
        states (see `equilibrium_solvation`). It is assigned by `solvent` if a solvent name
        is specified. Otherwise, the value of water (eps_optical=1.78) is applied and a
        warning is issued whenever `eps` indicates a solvent other than water.
        Default is None.

    frozen : bool
        Whether to freeze the potential produced by the solvent during SCF iterations or
        other convergence processes. When frozen=True is set, the solvent is
        assumed to respond slowly, while the electron density relaxes quickly.
        Default is False.

    max_cycle : int
        The maximum number of iterations to relax the solvent.

    conv_tol : float
        The convergence tolerance for total energy during solvent relaxation.

    equilibrium_solvation : bool
        Affects TDDFT and other excited state computations. Controls whether the solvent
        relaxes rapidly with respect to the electron density of the excited state.
        For vertical excitations, it is recommended to set this to False, as the solvent
        typically does not fully relax. The non-equilibrium solvation is then applied with
        the optical dielectric constant `eps_optical`. Default is False.

    state_id : int
        Specifies the target state in excited state calculations.
        `state_id=0` corresponds to the ground state, while `state_id=1` corresponds
        to the first excited state. Default is 0.

    surface_discretization_method : str
        Specifies the algorithm for the switching function, i.e. how each grid is partitioned
        among the atoms.
        Available options are "SWIG" (switching/Gaussian method) and "ISWIG" (improved SWIG).
        Please refer to the following paper for the definition of both algorithms:
        Lange, A. W.; Herbert, J. M. A smooth, nonsingular, and faithful discretization scheme
        for polarizable continuum models: The switching/Gaussian approach. The Journal of
        Chemical Physics 2010, 133. https://doi.org/10.1063/1.3511297

    Saved Results:
    --------------
    e_tot : float
        The energy contribution from the solvent.

    v : ndarray
        The potential matrix generated by the solvent.

    Intermediate Attributes:
    ------------------------
    These attributes are generated during calculations and should not be modified.
    Additionally, they may not be compatible between GPU and CPU implementations.

    - surface
    - _intermediates
    - v_grids_n
    '''

    _keys = {
        'method', 'vdw_scale', 'surface', 'r_probe',
        'mol', 'radii_table', 'lebedev_order', 'solvent',
        'eps', 'eps_optical', 'max_cycle', 'conv_tol', 'state_id', 'frozen',
        'equilibrium_solvation', 'e', 'v', 'v_grids_n',
        'surface_discretization_method',
    }

    kernel = ddcosmo.DDCOSMO.kernel
    get_eps_optical = ddcosmo.DDCOSMO.get_eps_optical

    def __init__(self, mol, solvent=None):
        self.mol = mol
        self.stdout = mol.stdout
        self.verbose = mol.verbose
        self.max_memory = mol.max_memory
        self.method = 'C-PCM'

        self.vdw_scale = 1.2 # default value in qchem
        self.r_probe = 0.0
        self.radii_table = None
        self.lebedev_order = 29
        self.eps = ddcosmo.EPS_WATER
        self.eps_optical = None
        # Assigning .solvent overwrites .eps and .eps_optical with the
        # parameters of the solvent (see .__setattr__)
        self.solvent = solvent or ''
        self.surface_discretization_method = "SWIG"

        self.max_cycle = 20
        self.conv_tol = 1e-7
        self.state_id = 0

        self.frozen = False
        self.equilibrium_solvation = False

        self.surface = {}
        self._intermediates = {}
        self.v_grids_n = None # nuclear potential on grids

        self.e = None
        self.v = None
        self._dm = None

    def __setattr__(self, key, val):
        if key == 'solvent' and val:
            val = self._set_solvent(val)
        super().__setattr__(key, val)

    def _set_solvent(self, solvent):
        '''Applies the dielectric constant and the refractive index of the
        solvent found in the solvent database. Returns the name of the solvent
        as it is stored in the database.
        '''
        name = _solvent_data.resolve_solvent_name(solvent)
        n, _, _, _, _, eps = _solvent_data.solvent_db[name][:6]
        self.eps = eps
        self.eps_optical = n**2
        if getattr(self, '_intermediates', None):
            # .eps was modified. The cached intermediates are outdated.
            self.reset()
        return name

    def dump_flags(self, verbose=None):
        logger.info(self, '******** %s (In testing) ********', self.__class__)
        logger.warn(self, 'PCM is an experimental feature. It is '
                    'still in testing.\nFeatures and APIs may be changed '
                    'in the future.')
        logger.info(self, 'lebedev_order = %s (%d grids per sphere)',
                    self.lebedev_order, gen_grid.LEBEDEV_ORDER[self.lebedev_order])
        solvent = getattr(self, 'solvent', '')
        if solvent:
            logger.info(self, 'solvent = %s'  , solvent)
        logger.info(self, 'eps = %s'          , self.eps)
        logger.info(self, 'eps_optical = %s'  , self.eps_optical)
        logger.info(self, 'frozen = %s'       , self.frozen)
        #logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
        return self

    def to_gpu(self):
        from pyscf.lib import to_gpu
        obj = to_gpu(self)
        return obj.reset()

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._intermediates = None
        self.surface = None
        self.v_grids_n = None
        return self

    def build(self, ng=None):
        if self.radii_table is None:
            vdw_scale = self.vdw_scale
            radii_table = vdw_scale * modified_Bondi + self.r_probe/radii.BOHR
        else:
            radii_table = self.radii_table
        logger.debug2(self, 'radii_table %s', radii_table)
        mol = self.mol
        if ng is None:
            ng = gen_grid.LEBEDEV_ORDER[self.lebedev_order]

        self.surface = gen_surface(mol, rad=radii_table, ng=ng,
                                   surface_discretization_method = self.surface_discretization_method)
        self._intermediates = {}
        F, A = get_F_A(self.surface)
        # C-PCM and COSMO need no D
        with_D = self.method.upper() not in ['C-PCM', 'CPCM', 'COSMO']
        D, S = get_D_S(self.surface, with_S=True, with_D=with_D)

        epsilon = self.eps
        # R is None for C-PCM and COSMO, where it is -f_epsilon times the
        # identity (see _R_dot)
        if self.method.upper() in ['C-PCM', 'CPCM']:
            f_epsilon = (epsilon-1.)/epsilon if epsilon != float('inf') else 1.0
            K = S
            R = None
        elif self.method.upper() == 'COSMO':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0/2.0) if epsilon != float('inf') else 1.0
            K = S
            R = None
        elif self.method.upper() in ['IEF-PCM', 'IEFPCM']:
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0) if epsilon != float('inf') else 1.0
            DA = D*A
            DAS = numpy.dot(DA, S)
            K = S - f_epsilon/(2.0*PI) * DAS
            R = -f_epsilon * (numpy.eye(K.shape[0]) - 1.0/(2.0*PI)*DA)
        elif self.method.upper() == 'SS(V)PE':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0) if epsilon != float('inf') else 1.0
            DA = D*A
            DAS = numpy.dot(DA, S)
            K = S - f_epsilon/(4.0*PI) * (DAS + DAS.T)
            R = -f_epsilon * (numpy.eye(K.shape[0]) - 1.0/(2.0*PI)*DA)
        else:
            raise RuntimeError(f"Unknown implicit solvent model: {self.method}")

        intermediates = {
            'S': S,
            'D': D,
            'A': A,
            'K': K,
            'R': R,
            'f_epsilon': f_epsilon
        }
        self._intermediates.update(intermediates)

        charge_exp  = self.surface['charge_exp']
        grid_coords = self.surface['grid_coords']
        atom_coords = mol.atom_coords(unit='B')
        atom_charges = mol.atom_charges()

        int2c2e = mol._add_suffix('int2c2e')
        fakemol = gto.fakemol_for_charges(grid_coords, expnt=charge_exp**2)
        fakemol_nuc = gto.fakemol_for_charges(atom_coords)
        v_ng = gto.mole.intor_cross(int2c2e, fakemol_nuc, fakemol)
        self.v_grids_n = numpy.dot(atom_charges, v_ng)

    def _get_vind(self, dms):
        if not self._intermediates:
            self.build()

        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)
        if dms.shape[0] == 2:
            dms = (dms[0] + dms[1]).reshape(-1,nao,nao)

        v_grids_e = self._get_v(dms)
        v_grids = self.v_grids_n - v_grids_e

        q, q_sym = self._get_q(v_grids)

        vmat = self._get_vmat(q_sym)
        epcm = 0.5 * numpy.dot(q_sym[0], v_grids[0])

        self._intermediates['q'] = q[0]
        self._intermediates['q_sym'] = q_sym[0]
        self._intermediates['v_grids'] = v_grids[0]
        self._intermediates['dm'] = dms
        return epcm, vmat[0]

    def _get_q(self, v_grids):
        '''The surface charges q = K^-1 R v of the potentials v_grids (one set
        a row), and their symmetrized form (q + R^T K^-T v) / 2'''
        q = _K_solve(self, _R_dot(self, v_grids))
        if self._intermediates['R'] is None:
            # C-PCM and COSMO: K is symmetric and R = -f_epsilon I
            return q, q
        qt = _R_dot(self, _K_solve(self, v_grids, trans=True), trans=True)
        return q, (q + qt)/2.0

    def _get_v(self, dms):
        '''
        return electrostatic potential on surface
        '''
        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)
        return _int1e_grids_dot(self.mol, self.surface, dms=dms)

    def _get_vmat(self, q):
        ngrids = self.surface['grid_coords'].shape[0]
        q = q.reshape([-1,ngrids])
        return -_int1e_grids_dot(self.mol, self.surface, charges=q)

    def nuc_grad_method(self, grad_method):
        raise DeprecationWarning('Use the make_grad_object function from '
                                 'pyscf.solvent.grad.pcm or '
                                 'pyscf.solvent._ddcosmo_tdscf_grad instead.')

    def grad(self, dm):
        '''Computes the Jacobian for the energy associated with the solvent,
        including the derivatives of the solvent itsself and the interactions
        between the solvent and the charge density of the solute.
        '''
        from pyscf.solvent.grad.pcm import grad_qv, grad_nuc, grad_solver
        de_solvent = grad_qv(self, dm)
        de_solvent+= grad_nuc(self, dm)
        de_solvent+= grad_solver(self, dm)
        return de_solvent

    def Hessian(self, hess_method):
        raise DeprecationWarning('Use the make_hessian_object function from '
                                 'pyscf.solvent.hessian.pcm instead.')

    def hess(self, dm):
        '''Computes the Hessian for the energy associated with the solvent,
        including the derivatives of the solvent itsself and the interactions
        between the solvent and the charge density of the solute.
        '''
        from pyscf.solvent.hessian.pcm import (
            analytical_hess_nuc, analytical_hess_qv, analytical_hess_solver)
        de_solvent  =    analytical_hess_nuc(self, dm, verbose=self.verbose)
        de_solvent +=     analytical_hess_qv(self, dm, verbose=self.verbose)
        de_solvent += analytical_hess_solver(self, dm, verbose=self.verbose)
        return de_solvent

    def _B_dot_x(self, dms):
        if not self._intermediates:
            self.build()
        out_shape = dms.shape
        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)

        v_grids = -self._get_v(dms)
        q_sym = self._get_q(v_grids)[1]

        vmat = self._get_vmat(q_sym)
        return vmat.reshape(out_shape)
