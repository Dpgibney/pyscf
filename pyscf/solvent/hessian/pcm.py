# Copyright 2023 The GPU4PySCF Authors. All Rights Reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
Gradient of PCM family solvent model, copied from GPU4PyCF with modifications
'''
# pylint: disable=C0103

import numpy
from scipy.special import erf
from pyscf import lib, gto
from pyscf import scf
from pyscf.solvent.pcm import (PI, switch_h, _R_dot, _K_solve, _int1e_grids_dot,
                               _int1e_grids_deriv)
from pyscf.solvent.grad.pcm import get_dD_dS, get_dF_dA, grad_switch_h, _d1_block_rows, _map_row_blocks
from pyscf.lib import logger

# The largest block of 3-center integrals (MB) the surface terms hold at a time
AUX_BLOCK_MEMORY = 800
# The memory (MB) of a block of rows of the second derivatives of S and D
D2_BLOCK_MEMORY = 200

def _aux_blksize(pcmobj, nao, ncomp):
    '''The surface points per block of (ncomp, nao, nao, points) integrals'''
    max_memory = min((pcmobj.max_memory - lib.current_memory()[0]) * .4, AUX_BLOCK_MEMORY)
    return int(max(max_memory*1e6/8/nao**2/ncomp, 8))

def gradgrad_switch_h(x):
    ''' 2nd derivative of h(x) '''
    ddy = 60.0*x - 180.0*x**2 + 120.0*x**3
    ddy[x<0] = 0.0
    ddy[x>1] = 0.0
    return ddy

def get_d2F_d2A(surface, surface_discretization_method = "SWIG", grid_atom=None):
    '''
    Notations adopted from
    J. Chem. Phys. 133, 244111 (2010), Appendix C

    With grid_atom, for the surface points of that atom only.
    '''
    atom_coords = surface['atom_coords']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    area        = surface['area']
    if surface_discretization_method.upper() == "SWIG":
        R_in_J = surface['R_in_J']
        R_sw_J = surface['R_sw_J']
    elif surface_discretization_method.upper() == "ISWIG":
        charge_exp = surface['charge_exp']
        R_J = surface['R_J']

    ngrids = grid_coords.shape[0]
    natom = atom_coords.shape[0]
    off = 0
    grid_atoms = range(natom)
    if grid_atom is not None:
        off, p1 = surface['gslice_by_atom'][grid_atom]
        ngrids = p1 - off
        grid_atoms = [grid_atom]
    d2F = numpy.zeros([ngrids, natom, natom, 3, 3])
    d2A = numpy.zeros([ngrids, natom, natom, 3, 3])

    for i_grid_atom in grid_atoms:
        p0,p1 = surface['gslice_by_atom'][i_grid_atom]
        coords = grid_coords[p0:p1]
        si_rJ = numpy.expand_dims(coords, axis=1) - atom_coords
        norm_si_rJ = numpy.linalg.norm(si_rJ, axis=-1)
        si_rJ[:,i_grid_atom,:] = 0.0

        if surface_discretization_method.upper() == "SWIG":
            diJ = (norm_si_rJ - R_in_J) / R_sw_J
            diJ[:,i_grid_atom] = 1.0
            diJ[diJ < 1e-8] = 0.0
            si_rJ[diJ < 1e-8] = 0.0

            fiJ = switch_h(diJ)
            dfiJ = grad_switch_h(diJ)

            fiJK = fiJ[:, :, numpy.newaxis] * fiJ[:, numpy.newaxis, :]
            dfiJK = dfiJ[:, :, numpy.newaxis] * dfiJ[:, numpy.newaxis, :]
            R_sw_JK = R_sw_J[:, numpy.newaxis] * R_sw_J[numpy.newaxis, :]
            norm_si_rJK = norm_si_rJ[:, :, numpy.newaxis] * norm_si_rJ[:, numpy.newaxis, :]
            terms_size_ngrids_natm_natm = dfiJK / (fiJK * norm_si_rJK * R_sw_JK)
            si_rJK = si_rJ[:, :, numpy.newaxis, :, numpy.newaxis] * si_rJ[:, numpy.newaxis, :, numpy.newaxis, :]
            d2fiJK_offdiagonal = terms_size_ngrids_natm_natm[:, :, :, numpy.newaxis, numpy.newaxis] * si_rJK

            d2fiJ = gradgrad_switch_h(diJ)
            terms_size_ngrids_natm = d2fiJ / (norm_si_rJ**2 * R_sw_J) - dfiJ / (norm_si_rJ**3)
            si_rJJ = si_rJ[:, :, :, numpy.newaxis] * si_rJ[:, :, numpy.newaxis, :]
            d2fiJK_diagonal = numpy.einsum('qA,qAdD->qAdD', terms_size_ngrids_natm, si_rJJ)
            d2fiJK_diagonal += numpy.einsum('qA,dD->qAdD', dfiJ / norm_si_rJ, numpy.eye(3))
            d2fiJK_diagonal /= (fiJ * R_sw_J)[:, :, numpy.newaxis, numpy.newaxis]

            d2fiJK = d2fiJK_offdiagonal
            for i_atom in range(natom):
                d2fiJK[:, i_atom, i_atom, :, :] = d2fiJK_diagonal[:, i_atom, :, :]
        elif surface_discretization_method.upper() == "ISWIG":
            xi = charge_exp[p0:p1]
            erf_input_p = xi[:, None] * (R_J[None, :] + norm_si_rJ)
            erf_input_m = xi[:, None] * (R_J[None, :] - norm_si_rJ)
            fiJ = 1 - 0.5 * (erf(erf_input_p) + erf(erf_input_m))
            # fiJ[:,i_grid_atom] = 1.0
            dfiJ = 1/numpy.sqrt(numpy.pi) * xi[:, None] * (numpy.exp(-erf_input_m**2) - numpy.exp(-erf_input_p**2))
            ### It is necessary to zero out i \in I term in dfiJ, because the second term of d2fiJK_diagonal
            ### is not zeroed out otherwise.
            dfiJ[:,i_grid_atom] = 0

            fiJK = fiJ[:, :, numpy.newaxis] * fiJ[:, numpy.newaxis, :]
            dfiJK = dfiJ[:, :, numpy.newaxis] * dfiJ[:, numpy.newaxis, :]
            norm_si_rJK = norm_si_rJ[:, :, numpy.newaxis] * norm_si_rJ[:, numpy.newaxis, :]
            terms_size_ngrids_natm_natm = dfiJK / (fiJK * norm_si_rJK)
            si_rJK = si_rJ[:, :, numpy.newaxis, :, numpy.newaxis] * si_rJ[:, numpy.newaxis, :, numpy.newaxis, :]
            d2fiJK_offdiagonal = terms_size_ngrids_natm_natm[:, :, :, numpy.newaxis, numpy.newaxis] * si_rJK

            d2fiJ = 2.0/numpy.sqrt(numpy.pi) * (xi**2)[:, None] * (
                + erf_input_m * numpy.exp(-erf_input_m**2)
                + erf_input_p * numpy.exp(-erf_input_p**2)
            )
            # d2fiJ[:,i_grid_atom] = 0.0

            terms_size_ngrids_natm = d2fiJ / (norm_si_rJ**2) - dfiJ / (norm_si_rJ**3)
            si_rJJ = si_rJ[:, :, :, numpy.newaxis] * si_rJ[:, :, numpy.newaxis, :]
            d2fiJK_diagonal = numpy.einsum('qA,qAdD->qAdD', terms_size_ngrids_natm, si_rJJ)
            d2fiJK_diagonal += numpy.einsum('qA,dD->qAdD', dfiJ / norm_si_rJ, numpy.eye(3))
            d2fiJK_diagonal /= fiJ[:, :, numpy.newaxis, numpy.newaxis]

            d2fiJK = d2fiJK_offdiagonal
            for i_atom in range(natom):
                d2fiJK[:, i_atom, i_atom, :, :] = d2fiJK_diagonal[:, i_atom, :, :]
        else:
            raise NotImplementedError(f"surface_discretization_method = {surface_discretization_method} not recognized")

        Fi = switch_fun[p0:p1]
        Ai = area[p0:p1]

        d2F[p0-off:p1-off, :, :, :, :] += numpy.einsum('q,qABdD->qABdD', Fi, d2fiJK)
        d2A[p0-off:p1-off, :, :, :, :] += numpy.einsum('q,qABdD->qABdD', Ai, d2fiJK)

        d2fiJK_grid_atom_offdiagonal = -numpy.einsum('qABdD->qAdD', d2fiJK)
        d2F[p0-off:p1-off, i_grid_atom, :, :, :] = \
            numpy.einsum('q,qAdD->qAdD', Fi, d2fiJK_grid_atom_offdiagonal.transpose(0,1,3,2))
        d2F[p0-off:p1-off, :, i_grid_atom, :, :] = \
            numpy.einsum('q,qAdD->qAdD', Fi, d2fiJK_grid_atom_offdiagonal)
        d2A[p0-off:p1-off, i_grid_atom, :, :, :] = \
            numpy.einsum('q,qAdD->qAdD', Ai, d2fiJK_grid_atom_offdiagonal.transpose(0,1,3,2))
        d2A[p0-off:p1-off, :, i_grid_atom, :, :] = \
            numpy.einsum('q,qAdD->qAdD', Ai, d2fiJK_grid_atom_offdiagonal)

        d2fiJK_grid_atom_diagonal = -numpy.einsum('qAdD->qdD', d2fiJK_grid_atom_offdiagonal)
        d2F[p0-off:p1-off, i_grid_atom, i_grid_atom, :, :] = numpy.einsum('q,qdD->qdD', Fi, d2fiJK_grid_atom_diagonal)
        d2A[p0-off:p1-off, i_grid_atom, i_grid_atom, :, :] = numpy.einsum('q,qdD->qdD', Ai, d2fiJK_grid_atom_diagonal)

    d2F = d2F.transpose(1,2,3,4,0)
    d2A = d2A.transpose(1,2,3,4,0)
    return d2F, d2A

def get_d2Sii(surface, dF, d2F, stream=None, rows=None):
    ''' Second derivative of S matrix (diagonal only); with rows = (g0, g1)
    for those surface points, d2F being theirs
    '''
    charge_exp  = surface['charge_exp']
    switch_fun  = surface['switch_fun']
    if rows is not None:
        charge_exp = charge_exp[rows[0]:rows[1]]
        switch_fun = switch_fun[rows[0]:rows[1]]
        dF = dF[rows[0]:rows[1]]
    dF = dF.transpose(1,2,0)

    dF_dF = dF[:, numpy.newaxis, :, numpy.newaxis, :] * dF[numpy.newaxis, :, numpy.newaxis, :, :]
    dF_dF_over_F3 = dF_dF * (1.0/(switch_fun**3))
    d2F_over_F2 = d2F * (1.0/(switch_fun**2))
    d2Sii = 2 * dF_dF_over_F3 - d2F_over_F2
    d2Sii = (2.0/PI)**0.5 * (d2Sii * charge_exp)

    return d2Sii

def get_d2D_d2S(surface, with_S=True, with_D=False, stream=None, rows=None):
    ''' Second derivatives of D matrix and S matrix (offdiagonals only);
    with rows = (i0, i1), those rows of the matrices only: [3,3,i1-i0,n]
    '''
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    norm_vec    = surface['norm_vec']
    n = charge_exp.shape[0]
    i0, i1 = (0, n) if rows is None else rows
    diag = (numpy.arange(i1-i0), numpy.arange(i0, i1))

    ei, ej = numpy.meshgrid(charge_exp[i0:i1], charge_exp, indexing='ij')
    eij = ei * ej / (ei**2 + ej**2)**0.5
    ri_rj = grid_coords[numpy.newaxis, :, :] - grid_coords[i0:i1, numpy.newaxis, :]
    rij = numpy.linalg.norm(ri_rj, axis=-1)
    rij[diag] = 1 # Suppress warning
    rij_1 = 1.0/rij
    rij_1[diag] = 0

    erf_eij_rij = erf(eij * rij)
    two_eij_over_sqrt_pi_exp_minus_eij2_rij2 = 2.0 / numpy.sqrt(PI) * eij * numpy.exp(-(eij * rij)**2)

    S_direct_product_prefactor = -two_eij_over_sqrt_pi_exp_minus_eij2_rij2 \
                                  * (3 * rij_1**4 + 2 * eij**2 * rij_1**2) \
                                  + 3 * rij_1**5 * erf_eij_rij
    d2S = ri_rj[:, :, numpy.newaxis, :] * ri_rj[:, :, :, numpy.newaxis]
    d2S = d2S * S_direct_product_prefactor[:, :, numpy.newaxis, numpy.newaxis]

    S_xyz_diagonal_prefactor = two_eij_over_sqrt_pi_exp_minus_eij2_rij2 * rij_1**2 \
                               - rij_1**3 * erf_eij_rij
    d2S[:,:,0,0] += S_xyz_diagonal_prefactor
    d2S[:,:,1,1] += S_xyz_diagonal_prefactor
    d2S[:,:,2,2] += S_xyz_diagonal_prefactor

    d2S = d2S.transpose(2,3,0,1)

    if not with_D:
        return None, d2S

    nj_rij = numpy.einsum('ijd,jd->ij', ri_rj, norm_vec)
    D_direct_product_prefactor = (-two_eij_over_sqrt_pi_exp_minus_eij2_rij2
                                  * (15 * rij_1**6 + 10 * eij**2 * rij_1**4 + 4 * eij**4 * rij_1**2)
                                  + 15 * rij_1**7 * erf_eij_rij) \
                                 * nj_rij

    d2D = ri_rj[:, :, numpy.newaxis, :] * ri_rj[:, :, :, numpy.newaxis]
    d2D = d2D * D_direct_product_prefactor[:, :, numpy.newaxis, numpy.newaxis]

    nj_rij_direct_product = ri_rj[:, :, numpy.newaxis, :] * norm_vec[numpy.newaxis, :, :, numpy.newaxis]
    rij_nj_direct_product = nj_rij_direct_product.transpose(0,1,3,2)
    d2D -= (nj_rij_direct_product + rij_nj_direct_product) \
            * S_direct_product_prefactor[:, :, numpy.newaxis, numpy.newaxis]
    d2D[:,:,0,0] -= S_direct_product_prefactor * nj_rij
    d2D[:,:,1,1] -= S_direct_product_prefactor * nj_rij
    d2D[:,:,2,2] -= S_direct_product_prefactor * nj_rij

    d2D = -d2D.transpose(2,3,0,1)

    return d2D, d2S

def _d2_contract(surface, natom, gridslice, S_pairs=(), D_pairs=()):
    '''The second derivatives of S and D (get_d2D_d2S) contracted with pairs of
    vectors (v, q) of the surface points, by blocks of rows, without the
    [3,3,n,n] arrays: for each pair the contribution to the Hessian,
    _atom_pair_sums of G[A,B,d,D] = sum_{i in A, j in B} v[i] d2X[d,D,i,j] q[j].
    Returns the lists for S_pairs and for D_pairs.'''
    n = surface['charge_exp'].shape[0]
    P = numpy.zeros([n, natom])
    for i_atom, (g0, g1) in enumerate(gridslice):
        P[g0:g1, i_atom] = 1
    pairs = [(v[:, None] * P, q[:, None] * P) for v, q in list(S_pairs) + list(D_pairs)]
    nS = len(S_pairs)
    G = numpy.zeros([len(pairs), natom, natom, 9])
    # a row of the 9 components and the temporaries of get_d2D_d2S
    blk = max(1, int(D2_BLOCK_MEMORY * 1e6 / (8 * n * 9 * 8)))
    def block(i0, i1):
        d2D, d2S = get_d2D_d2S(surface, with_S=True, with_D=len(D_pairs) > 0, rows=(i0, i1))
        g = numpy.empty_like(G)
        for k, (vP, qP) in enumerate(pairs):
            d2X = d2S if k < nS else d2D
            tmp = numpy.dot(d2X.reshape(-1, n), qP).reshape(9, i1-i0, -1)
            g[k] = numpy.einsum('iA,ciB->ABc', vP[i0:i1], tmp, optimize=True)
        return g
    G = sum(_map_row_blocks(n, blk, block))
    out = [_atom_pair_sums(g.reshape(natom, natom, 3, 3)) for g in G]
    return out[:nS], out[nS:]

def _atom_pair_sums(G):
    '''The second derivatives with respect to the atoms from the sums G[A,B]
    over the points of the pairs of atoms: a matrix element of two points
    depends on their difference'''
    output = -G - G.transpose(1,0,2,3)
    idx = numpy.arange(G.shape[0])
    output[idx, idx] += G.sum(axis=1) + G.sum(axis=0)
    return output

def analytical_hess_nuc(pcmobj, dm, verbose=None):
    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get('dm', None)
    if dm_cache is not None and numpy.linalg.norm(dm_cache - dm) < 1e-10:
        pass
    else:
        pcmobj._get_vind(dm)
    mol = pcmobj.mol
    log = logger.new_logger(mol, verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    q_sym        = pcmobj._intermediates['q_sym']
    gridslice    = pcmobj.surface['gslice_by_atom']
    grid_coords  = pcmobj.surface['grid_coords']
    exponents    = pcmobj.surface['charge_exp']

    ngrids = q_sym.shape[0]

    atom_coords = mol.atom_coords(unit='B')
    atom_charges = numpy.asarray(mol.atom_charges(), dtype=numpy.float64)
    fakemol_nuc = gto.fakemol_for_charges(atom_coords)
    fakemol = gto.fakemol_for_charges(grid_coords, expnt=exponents**2)

    d2e_from_d2I = numpy.zeros([mol.natm, mol.natm, 3, 3])

    int2c2e_ip1ip2 = mol._add_suffix('int2c2e_ip1ip2')
    d2I_dAdC = gto.mole.intor_cross(int2c2e_ip1ip2, fakemol_nuc, fakemol)
    d2I_dAdC = d2I_dAdC.reshape(3, 3, mol.natm, ngrids)
    for i_atom in range(mol.natm):
        g0,g1 = gridslice[i_atom]
        d2e_from_d2I[:, i_atom, :, :] += \
            numpy.einsum('A,dDAq,q->AdD', atom_charges, d2I_dAdC[:, :, :, g0:g1], q_sym[g0:g1])
        d2e_from_d2I[i_atom, :, :, :] += \
            numpy.einsum('A,dDAq,q->AdD', atom_charges, d2I_dAdC[:, :, :, g0:g1], q_sym[g0:g1])

    int2c2e_ipip1 = mol._add_suffix('int2c2e_ipip1')
    # # Some explanations here:
    # # Why can we use the ip1ip2 here? Because of the translational invariance
    # # $\frac{\partial^2 I_{AC}}{\partial A^2} + \frac{\partial^2 I_{AC}}{\partial A \partial C} = 0$
    # # Why not using the ipip1 here? Because the nuclei, a point charge,
    # # is handled as a Gaussian charge with exponent = 1e16.
    # # This causes severe numerical problem in function int2c2e_ip1ip2,
    # # and make the main diagonal of hessian garbage.
    # d2I_dA2 = gto.mole.intor_cross(int2c2e_ipip1, fakemol_nuc, fakemol)
    d2I_dA2 = -gto.mole.intor_cross(int2c2e_ip1ip2, fakemol_nuc, fakemol)
    d2I_dA2 = d2I_dA2 @ q_sym
    d2I_dA2 = d2I_dA2.reshape(3, 3, mol.natm)
    for i_atom in range(mol.natm):
        d2e_from_d2I[i_atom, i_atom, :, :] += atom_charges[i_atom] * d2I_dA2[:, :, i_atom]

    d2I_dC2 = gto.mole.intor_cross(int2c2e_ipip1, fakemol, fakemol_nuc)
    d2I_dC2 = d2I_dC2 @ atom_charges
    d2I_dC2 = d2I_dC2.reshape(3, 3, ngrids)
    for i_atom in range(mol.natm):
        g0,g1 = gridslice[i_atom]
        d2e_from_d2I[i_atom, i_atom, :, :] += d2I_dC2[:, :, g0:g1] @ q_sym[g0:g1]

    # grad_nuc is linear in the charges: grad_nuc(q)[B,D] = -sum_g q_g dVnuc[B,D,g]
    dqdx = get_dqsym_dx(pcmobj, dm, range(mol.natm))
    dvnuc = _get_dv_parts(pcmobj, dm)[0]
    d2e_from_dIdq = -lib.einsum('Adg,BDg->ABdD', dqdx, dvnuc)

    d2e = d2e_from_d2I - d2e_from_dIdq

    t1 = log.timer_debug1('solvent hessian d(dVnuc/dx * q)/dx contribution', *t1)
    return d2e

def analytical_hess_qv(pcmobj, dm, verbose=None):
    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get('dm', None)
    if dm_cache is not None and numpy.linalg.norm(dm_cache - dm) < 1e-10:
        pass
    else:
        pcmobj._get_vind(dm)
    mol = pcmobj.mol
    nao = mol.nao
    log = logger.new_logger(mol, mol.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    gridslice   = pcmobj.surface['gslice_by_atom']
    charge_exp  = pcmobj.surface['charge_exp']
    grid_coords = pcmobj.surface['grid_coords']
    q_sym       = pcmobj._intermediates['q_sym']

    aoslice = mol.aoslice_by_atom()
    aoslice = numpy.array(aoslice)

    natm = mol.natm
    ngrids = q_sym.shape[0]
    dm2 = dm + dm.T
    # Three kinds of integrals, (dd i j|g), (d i d j|g) and those with derivatives
    # of the charge's position, which are sums of the first two (translational
    # invariance). Contracted with the density over the second AO, and summed
    # over the AOs of each atom: W[A,dD,g].
    def atom_sums(w):
        return numpy.array([w[:, p0:p1].sum(axis=1) for p0, p1 in aoslice[:, 2:]])
    w, = _int1e_grids_deriv(mol, pcmobj.surface, 'int1e_grids_ipip', 9, dms_i=[dm2])
    w_pp = atom_sums(w)
    w, d2I_dAdB = _int1e_grids_deriv(mol, pcmobj.surface, 'int1e_grids_ipvip', 9,
                                     dms_i=[dm2], charges=[q_sym])
    w_pv = atom_sums(w)
    w = None

    d2e_from_d2I = numpy.zeros([natm, natm, 3, 3])
    idx = numpy.arange(natm)
    # (dd i j|g): both derivatives on the AOs of one atom
    d2e_from_d2I[idx, idx] += numpy.dot(w_pp, q_sym).reshape(natm, 3, 3)

    # (d i d j|g): the AOs of two atoms
    d2I_dAdB = d2I_dAdB.reshape(3, 3, nao, nao)
    for i_atom in range(natm):
        pi0,pi1 = aoslice[i_atom, 2:]
        for j_atom in range(natm):
            pj0,pj1 = aoslice[j_atom, 2:]
            d2e_from_d2I[i_atom, j_atom, :, :] += \
                numpy.einsum('ij,dDij->dD', dm2[pi0:pi1, pj0:pj1], d2I_dAdB[:, :, pi0:pi1, pj0:pj1])
    d2I_dAdB = None

    # (d i j|d g) = -(dd i j|g) - (d i d j|g): an AO and the charges of an atom
    w_pp += w_pv
    w_pv = None
    for j_atom in range(natm):
        g0,g1 = gridslice[j_atom]
        x = -numpy.dot(w_pp[:, :, g0:g1], q_sym[g0:g1]).reshape(natm, 3, 3)
        d2e_from_d2I[:, j_atom] += x
        d2e_from_d2I[j_atom, :] += x.transpose(0, 2, 1)

    # (i j|dd g) = (dd i j|g) + (d i d j|g) + (d j d i|g) + (i dd j|g)
    d2I_dC2 = w_pp.sum(axis=0).reshape(3, 3, ngrids)
    w_pp = None
    for i_atom in range(natm):
        g0,g1 = gridslice[i_atom]
        d2e_from_d2I[i_atom, i_atom, :, :] += d2I_dC2[:, :, g0:g1] @ q_sym[g0:g1]
    d2I_dC2 = None

    # grad_qv is linear in the charges: the integrals of get_dvgrids, contracted with
    # the density, times the charges
    dqdx = get_dqsym_dx(pcmobj, dm, range(mol.natm))
    _, dIdA, dIdC = _get_dv_parts(pcmobj, dm)
    d2e_from_dIdq = lib.einsum('Adg,BDg->ABdD', dqdx, dIdA)
    for j_atom in range(mol.natm):
        g0,g1 = gridslice[j_atom]
        d2e_from_dIdq[:, j_atom] += lib.einsum('Adg,Dg->AdD', dqdx[:, :, g0:g1], dIdC[:, g0:g1])

    d2e = d2e_from_d2I + d2e_from_dIdq
    d2e *= -1

    t1 = log.timer_debug1('solvent hessian d(dI/dx * q)/dx contribution', *t1)
    return d2e

def _d2A_d2Sii_dot(pcmobj, dF, wA=(), wS=()):
    '''d2A @ w for the vectors of wA and d2Sii @ w for those of wS (the arrays of
    get_d2F_d2A and get_d2Sii, [natm,natm,3,3,points]), an atom's surface points
    at a time'''
    surface = pcmobj.surface
    natom = len(surface['gslice_by_atom'])
    outA = numpy.zeros([len(wA), natom, natom, 3, 3])
    outS = numpy.zeros([len(wS), natom, natom, 3, 3])
    for i_atom, (g0, g1) in enumerate(surface['gslice_by_atom']):
        if g1 == g0:
            continue
        d2F, d2A = get_d2F_d2A(surface, pcmobj.surface_discretization_method, grid_atom=i_atom)
        for k, w in enumerate(wA):
            outA[k] += d2A @ w[g0:g1]
        d2A = None
        if len(wS) > 0:
            d2Sii = get_d2Sii(surface, dF, d2F, rows=(g0, g1))
            for k, w in enumerate(wS):
                outS[k] += d2Sii @ w[g0:g1]
    return outA, outS

def _mdot(x, m):
    '''x[A,d,i] m[i,j] as one matrix product (numpy.dot of a 3-index array with
    a matrix is a loop of dot products)'''
    return numpy.dot(x.reshape(-1, x.shape[-1]), m).reshape(x.shape[:-1] + (m.shape[1],))

def _K_solve_3d(pcmobj, x, trans=False):
    '''K^-1 x (K^-T x with trans) over the last axis of x[A,d,points], with the
    factorization of K the SCF made (pcm._K_solve)'''
    return _K_solve(pcmobj, x.reshape(-1, x.shape[-1]), trans=trans).reshape(x.shape)

def einsum_ij_Adj_Adi_inverseK(K, Adj_term):
    nA, nd, nj = Adj_term.shape
    # return numpy.einsum('ij,Adj->Adi', numpy.linalg.inv(K), Adj_term)
    return numpy.linalg.solve(K, Adj_term.reshape(nA * nd, nj).T).T.reshape(nA, nd, nj)
def einsum_Adi_ij_Adj_inverseK(Adi_term, K):
    nA, nd, nj = Adi_term.shape
    # return numpy.einsum('Adi,ij->Adj', Adi_term, numpy.linalg.inv(K))
    return numpy.linalg.solve(K.T, Adi_term.reshape(nA * nd, nj).T).T.reshape(nA, nd, nj)

def _get_dSii(surface, dF):
    '''The derivative of the diagonal of S (the dSii of get_dD_dS)'''
    dSii_dF = -surface['charge_exp'] * (2.0/PI)**0.5 / surface['switch_fun']**2
    return numpy.expand_dims(dSii_dF, axis=(1,2)) * dF

class _LazyD1:
    '''dS or dD of get_dD_dS (kind 'S' or 'D') as an operator: the products
    with a vector the Hessian needs, by blocks of rows, without the [n,n,3]
    array'''
    def __init__(self, surface, kind):
        self.surface = surface
        self.kind = kind

    def dot_q(self, q, atmlst, gridslice, trans=False):
        '''output[A,x,i] = sum_j dX[x,i,j] q[j] for the points i of atom A, minus
        sum_{j of A} dX[x,i,j] q[j] for all i. With trans, for -dX^T.'''
        natom = len(gridslice)
        n = q.shape[0]
        P = numpy.zeros([n, natom])
        for i_atom, (g0, g1) in enumerate(gridslice):
            P[g0:g1, i_atom] = 1
        qP = q[:, None] * P
        T = numpy.zeros([3, n, natom])
        with_D = self.kind == 'D'
        def block(i0, i1):
            dD, dS, _ = get_dD_dS(self.surface, None, with_D=with_D, rows=(i0, i1))
            dX = (dD if with_D else dS).transpose(2,0,1)
            if trans:
                return numpy.array([numpy.dot(dX[x].T, qP[i0:i1]) for x in range(3)])
            T[:, i0:i1] = numpy.matmul(dX, qP)
        parts = _map_row_blocks(n, _d1_block_rows(n), block)
        if trans:
            T -= sum(parts)
        output = -T.transpose(2,0,1)
        Tsum = T.sum(axis=2)
        for i_atom, (g0, g1) in enumerate(gridslice):
            output[i_atom, :, g0:g1] += Tsum[:, g0:g1]
        return output[list(atmlst)]

def get_dS_dot_q(dS, dSii, q, atmlst, gridslice):
    output = numpy.einsum('iAd,i->Adi', dSii[:,atmlst,:], q)
    if isinstance(dS, _LazyD1):
        return output + dS.dot_q(q, atmlst, gridslice)
    dS = dS.transpose(2,0,1)
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        output[i_atom, :, g0:g1] += dS[:,g0:g1,:] @ q
        output[i_atom, :, :] -= dS[:,:,g0:g1] @ q[g0:g1]
    return output
def get_dST_dot_q(dS, dSii, q, atmlst, gridslice):
    # S is symmetric
    return get_dS_dot_q(dS, dSii, q, atmlst, gridslice)

def get_dA_dot_q(dA, q, atmlst):
    return numpy.einsum('iAd,i->Adi', dA[:,atmlst,:], q)

def get_dD_dot_q(dD, q, atmlst, gridslice, ngrids):
    if isinstance(dD, _LazyD1):
        return dD.dot_q(q, atmlst, gridslice)
    dD = dD.transpose(2,0,1)
    output = numpy.zeros([len(atmlst), 3, ngrids])
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        output[i_atom, :, g0:g1] += dD[:,g0:g1,:] @ q
        output[i_atom, :, :] -= dD[:,:,g0:g1] @ q[g0:g1]
    return output
def get_dDT_dot_q(dD, q, atmlst, gridslice, ngrids):
    if isinstance(dD, _LazyD1):
        return dD.dot_q(q, atmlst, gridslice, trans=True)
    return get_dD_dot_q(-dD.transpose(1,0,2), q, atmlst, gridslice, ngrids)

def get_v_dot_d2S_dot_q(d2S, d2Sii, v_left, q_right, natom, gridslice):
    output = d2Sii @ (v_left * q_right)
    for i_atom in range(natom):
        gi0,gi1 = gridslice[i_atom]
        for j_atom in range(natom):
            gj0,gj1 = gridslice[j_atom]
            d2S_atom_ij = numpy.einsum('q,dDq->dD', v_left[gi0:gi1], d2S[:,:,gi0:gi1,gj0:gj1] @ q_right[gj0:gj1])
            output[i_atom, i_atom, :, :] += d2S_atom_ij
            output[j_atom, j_atom, :, :] += d2S_atom_ij
            output[i_atom, j_atom, :, :] -= d2S_atom_ij
            output[j_atom, i_atom, :, :] -= d2S_atom_ij
    return output
def get_v_dot_d2ST_dot_q(d2S, d2Sii, v_left, q_right, natom, gridslice):
    # S is symmetric
    return get_v_dot_d2S_dot_q(d2S, d2Sii, v_left, q_right, natom, gridslice)

def get_v_dot_d2A_dot_q(d2A, v_left, q_right):
    return d2A @ (v_left * q_right)

def get_v_dot_d2D_dot_q(d2D, v_left, q_right, natom, gridslice):
    output = numpy.zeros([natom, natom, 3, 3])
    for i_atom in range(natom):
        gi0,gi1 = gridslice[i_atom]
        for j_atom in range(natom):
            gj0,gj1 = gridslice[j_atom]
            d2D_atom_ij = numpy.einsum('q,dDq->dD', v_left[gi0:gi1], d2D[:,:,gi0:gi1,gj0:gj1] @ q_right[gj0:gj1])
            output[i_atom, i_atom, :, :] += d2D_atom_ij
            output[j_atom, j_atom, :, :] += d2D_atom_ij
            output[i_atom, j_atom, :, :] -= d2D_atom_ij
            output[j_atom, i_atom, :, :] -= d2D_atom_ij
    return output
def get_v_dot_d2DT_dot_q(d2D, v_left, q_right, natom, gridslice):
    return get_v_dot_d2D_dot_q(d2D.transpose(0,1,3,2), v_left, q_right, natom, gridslice)

def analytical_hess_solver(pcmobj, dm, verbose=None):
    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get('dm', None)
    if dm_cache is not None and numpy.linalg.norm(dm_cache - dm) < 1e-10:
        pass
    else:
        pcmobj._get_vind(dm)
    mol = pcmobj.mol
    log = logger.new_logger(mol, mol.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    natom = mol.natm
    atmlst = range(natom) # Attention: This cannot be split

    gridslice    = pcmobj.surface['gslice_by_atom']
    v_grids      = pcmobj._intermediates['v_grids']
    A            = pcmobj._intermediates['A']
    D            = pcmobj._intermediates['D']
    S            = pcmobj._intermediates['S']
    K            = pcmobj._intermediates['K']
    R            = pcmobj._intermediates['R']
    q            = pcmobj._intermediates['q']
    f_epsilon    = pcmobj._intermediates['f_epsilon']

    ngrids = q.shape[0]

    vK_1 = _K_solve(pcmobj, v_grids, trans=True)

    if pcmobj.method.upper() in ['C-PCM', 'CPCM', 'COSMO']:
        dF, _ = get_dF_dA(pcmobj.surface, surface_discretization_method = pcmobj.surface_discretization_method)
        dS, dSii = _LazyD1(pcmobj.surface, 'S'), _get_dSii(pcmobj.surface, dF)

        # dR = 0, dK = dS
        # d(S-1 R) = - S-1 dS S-1 R
        # d2(S-1 R) = (S-1 dS S-1 dS S-1 R) + (S-1 dS S-1 dS S-1 R) - (S-1 d2S S-1 R)
        dSdx_dot_q = get_dS_dot_q(dS, dSii, q, atmlst, gridslice)
        S_1_dSdx_dot_q = _K_solve_3d(pcmobj, dSdx_dot_q)
        dSdx_dot_q = None
        VS_1_dot_dSdx = get_dST_dot_q(dS, dSii, vK_1, atmlst, gridslice)
        dS = None
        dSii = None
        d2e_from_d2KR = numpy.einsum('Adi,BDi->ABdD', VS_1_dot_dSdx, S_1_dSdx_dot_q) * 2

        (gS,), _ = _d2_contract(pcmobj.surface, natom, gridslice, S_pairs=[(vK_1, q)])
        _, (d2Sii_vq,) = _d2A_d2Sii_dot(pcmobj, dF, wS=[vK_1 * q])
        dF = None
        d2F = None
        d2e_from_d2KR -= d2Sii_vq + gS
        d2S = None
        d2Sii = None

        dK_1Rv = -S_1_dSdx_dot_q
        dvK_1R = -_R_dot(pcmobj, _K_solve_3d(pcmobj, VS_1_dot_dSdx, trans=True), trans=True)

    elif pcmobj.method.upper() in ['IEF-PCM', 'IEFPCM', 'SMD']:
        dF, dA = get_dF_dA(pcmobj.surface, surface_discretization_method = pcmobj.surface_discretization_method)
        dD, dS, dSii = _LazyD1(pcmobj.surface, 'D'), _LazyD1(pcmobj.surface, 'S'), _get_dSii(pcmobj.surface, dF)

        # dR = f_eps/(2*pi) * (dD*A + D*dA)
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)

        # d2R = f_eps/(2*pi) * (d2D*A + dD*dA + dD*dA + D*d2A)
        # d2K = d2S - f_eps/(2*pi) * (d2D*A*S + D*d2A*S + D*A*d2S
        #                             + dD*dA*S + dD*dA*S + dD*A*dS + dD*A*dS + D*dA*dS + D*dA*dS)
        # The terms showing up twice on equation above (dD*dA + dD*dA for example)
        # refer to dD/dx * dA/dy + dD/dy * dA/dx,
        # since D is not symmetric, they are not the same.

        # d(K-1 R) = - K-1 dK K-1 R + K-1 dR
        # d2(K-1 R) = (K-1 dK K-1 dK K-1 R) + (K-1 dK K-1 dK K-1 R) - (K-1 d2K K-1 R) - (K-1 dK K-1 dR)
        #             - (K-1 dK K-1 dR) + (K-1 d2R)
        f_eps_over_2pi = f_epsilon/(2.0*PI)

        dSdx_dot_q = get_dS_dot_q(dS, dSii, q, atmlst, gridslice)
        DA = D*A
        dKdx_dot_q = dSdx_dot_q - f_eps_over_2pi * _mdot(dSdx_dot_q, DA.T)
        dAdx_dot_Sq = get_dA_dot_q(dA, S @ q, atmlst)
        dKdx_dot_q -= f_eps_over_2pi * _mdot(dAdx_dot_Sq, D.T)
        AS = (A * S.T).T # It's just diag(A) @ S
        ASq = AS @ q
        dDdx_dot_ASq = get_dD_dot_q(dD, ASq, atmlst, gridslice, ngrids)
        dKdx_dot_q -= f_eps_over_2pi * dDdx_dot_ASq
        dDdx_dot_ASq = None

        K_1_dot_dKdx_dot_q = _K_solve_3d(pcmobj, dKdx_dot_q)
        dKdx_dot_q = None

        vK_1_dot_dSdx = get_dST_dot_q(dS, dSii, vK_1, atmlst, gridslice)
        vK_1_dot_dKdx = vK_1_dot_dSdx
        vK_1_dot_dSdx = None
        vK_1_dot_dDdx = get_dDT_dot_q(dD, vK_1, atmlst, gridslice, ngrids)
        vK_1_dot_dKdx -= f_eps_over_2pi * _mdot(vK_1_dot_dDdx, AS)
        AS = None
        vK_1D = D.T @ vK_1
        vK_1D_dot_dAdx = get_dA_dot_q(dA, vK_1D, atmlst)
        vK_1_dot_dKdx -= f_eps_over_2pi * _mdot(vK_1D_dot_dAdx, S)
        vK_1DA = DA.T @ vK_1
        DA = None
        vK_1DA_dot_dSdx = get_dST_dot_q(dS, dSii, vK_1DA, atmlst, gridslice)
        dS = None
        dSii = None
        vK_1_dot_dKdx -= f_eps_over_2pi * vK_1DA_dot_dSdx
        vK_1DA_dot_dSdx = None

        d2e_from_d2KR  = numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)
        d2e_from_d2KR += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)

        d2A_w, d2Sii_w = _d2A_d2Sii_dot(pcmobj, dF, wA=[vK_1D * (S @ q), vK_1D * v_grids],
                                        wS=[vK_1DA * q, vK_1 * q])
        vK_1_d2K_q  = d2A_w[0]
        vK_1_d2R_V  = d2A_w[1]
        d2A = None
        dF = None
        d2F = None
        gS, gD = _d2_contract(pcmobj.surface, natom, gridslice,
                              S_pairs=[(vK_1DA, q), (vK_1, q)],
                              D_pairs=[(vK_1, ASq), (vK_1, A * v_grids)])
        vK_1_d2K_q += gD[0]
        vK_1_d2R_V += gD[1]
        vK_1_d2K_q += d2Sii_w[0] + gS[0]
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q *= -f_eps_over_2pi
        vK_1_d2K_q += d2Sii_w[1] + gS[1]
        d2Sii = None

        d2e_from_d2KR -= vK_1_d2K_q

        dAdx_dot_V = get_dA_dot_q(dA, v_grids, atmlst)
        dDdx_dot_AV = get_dD_dot_q(dD, A * v_grids, atmlst, gridslice, ngrids)
        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + _mdot(dAdx_dot_V, D.T))
        dDdx_dot_AV = None

        K_1_dot_dRdx_dot_V = _K_solve_3d(pcmobj, dRdx_dot_V)
        dRdx_dot_V = None

        d2e_from_d2KR -= numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)
        d2e_from_d2KR -= numpy.einsum('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)

        vK_1_d2R_V += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V *= f_eps_over_2pi

        d2e_from_d2KR += vK_1_d2R_V

        dK_1Rv = -K_1_dot_dKdx_dot_q + K_1_dot_dRdx_dot_V

        VK_1D_dot_dAdx = get_dA_dot_q(dA, (D.T @ vK_1).T, atmlst)
        VK_1_dot_dDdx = get_dDT_dot_q(dD, vK_1, atmlst, gridslice, ngrids)
        VK_1_dot_dRdx = f_eps_over_2pi * (VK_1D_dot_dAdx + VK_1_dot_dDdx * A)

        dvK_1R = -_K_solve_3d(pcmobj, vK_1_dot_dKdx, trans=True) @ R + VK_1_dot_dRdx

    elif pcmobj.method.upper() in ['SS(V)PE']:
        dF, dA = get_dF_dA(pcmobj.surface, surface_discretization_method = pcmobj.surface_discretization_method)
        dD, dS, dSii = _LazyD1(pcmobj.surface, 'D'), _LazyD1(pcmobj.surface, 'S'), _get_dSii(pcmobj.surface, dF)

        # dR = f_eps/(2*pi) * (dD*A + D*dA)
        # dK = dS - f_eps/(4*pi) * (dD*A*S + D*dA*S + D*A*dS + dST*AT*DT + ST*dAT*DT + ST*AT*dDT)

        # d2R = f_eps/(2*pi) * (d2D*A + dD*dA + dD*dA + D*d2A)
        # d2K = d2S - f_eps/(4*pi)
        #             * (d2D*A*S + D*d2A*S + D*A*d2S
        #                + dD*dA*S + dD*dA*S + dD*A*dS + dD*A*dS + D*dA*dS + D*dA*dS
        #                + d2ST*AT*DT + ST*d2AT*DT + ST*AT*d2DT
        #                + dST*dAT*DT + dST*dAT*DT + dST*AT*dDT + dST*AT*dDT + ST*dAT*dDT + ST*dAT*dDT)
        f_eps_over_2pi = f_epsilon/(2.0*PI)
        f_eps_over_4pi = f_epsilon/(4.0*PI)

        dSdx_dot_q = get_dS_dot_q(dS, dSii, q, atmlst, gridslice)
        DA = D*A
        dKdx_dot_q = dSdx_dot_q - f_eps_over_4pi * _mdot(dSdx_dot_q, DA.T)
        dAdx_dot_Sq = get_dA_dot_q(dA, S @ q, atmlst)
        dKdx_dot_q -= f_eps_over_4pi * _mdot(dAdx_dot_Sq, D.T)
        AS = (A * S.T).T # It's just diag(A) @ S
        ASq = AS @ q
        dDdx_dot_ASq = get_dD_dot_q(dD, ASq, atmlst, gridslice, ngrids)
        dKdx_dot_q -= f_eps_over_4pi * dDdx_dot_ASq
        dDdx_dot_ASq = None
        dDdxT_dot_q = get_dDT_dot_q(dD, q, atmlst, gridslice, ngrids)
        dKdx_dot_q -= f_eps_over_4pi * _mdot(dDdxT_dot_q, AS)
        dAdxT_dot_DT_q = get_dA_dot_q(dA, D.T @ q, atmlst)
        dKdx_dot_q -= f_eps_over_4pi * _mdot(dAdxT_dot_DT_q, S)
        AT_DT_q = DA.T @ q
        dSdxT_dot_AT_DT_q = get_dS_dot_q(dS, dSii, AT_DT_q, atmlst, gridslice)
        dKdx_dot_q -= f_eps_over_4pi * dSdxT_dot_AT_DT_q
        dSdxT_dot_AT_DT_q = None

        K_1_dot_dKdx_dot_q = _K_solve_3d(pcmobj, dKdx_dot_q)
        dKdx_dot_q = None

        vK_1_dot_dSdx = get_dST_dot_q(dS, dSii, vK_1, atmlst, gridslice)
        vK_1_dot_dKdx = vK_1_dot_dSdx
        vK_1_dot_dSdx = None
        vK_1_dot_dDdx = get_dDT_dot_q(dD, vK_1, atmlst, gridslice, ngrids)
        vK_1_dot_dKdx -= f_eps_over_4pi * _mdot(vK_1_dot_dDdx, AS)
        vK_1D_dot_dAdx = get_dA_dot_q(dA, D.T @ vK_1, atmlst)
        vK_1_dot_dKdx -= f_eps_over_4pi * _mdot(vK_1D_dot_dAdx, S)
        vK_1DA = DA.T @ vK_1
        vK_1DA_dot_dSdx = get_dST_dot_q(dS, dSii, vK_1DA, atmlst, gridslice)
        vK_1_dot_dKdx -= f_eps_over_4pi * vK_1DA_dot_dSdx
        vK_1DA_dot_dSdx = None
        vK_1_dot_dSdxT = get_dS_dot_q(dS, dSii, vK_1, atmlst, gridslice)
        dS = None
        dSii = None
        vK_1_dot_dKdx -= f_eps_over_4pi * _mdot(vK_1_dot_dSdxT, DA.T)
        DA = None
        vK_1_ST_dot_dAdxT = get_dA_dot_q(dA, (S @ vK_1).T, atmlst)
        vK_1_dot_dKdx -= f_eps_over_4pi * _mdot(vK_1_ST_dot_dAdxT, D.T)
        vK_1_ST_AT = AS @ vK_1
        AS = None
        vK_1_ST_AT_dot_dDdxT = get_dD_dot_q(dD, vK_1_ST_AT, atmlst, gridslice, ngrids)
        vK_1_dot_dKdx -= f_eps_over_4pi * vK_1_ST_AT_dot_dDdxT
        vK_1_ST_AT_dot_dDdxT = None

        d2e_from_d2KR  = numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)
        d2e_from_d2KR += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)

        d2A_w, d2Sii_w = _d2A_d2Sii_dot(
            pcmobj, dF, wA=[(D.T @ vK_1) * (S @ q), (S @ vK_1) * (D.T @ q), (D.T @ vK_1) * v_grids],
            wS=[vK_1DA * q, vK_1 * q, vK_1 * AT_DT_q])
        vK_1_d2K_q  = d2A_w[0]
        vK_1_d2K_q += d2A_w[1]
        vK_1_d2R_V  = d2A_w[2]
        d2A = None
        dF = None
        d2F = None
        # the transpose of d2D: the two vectors change places
        gS, gD = _d2_contract(pcmobj.surface, natom, gridslice,
                              S_pairs=[(vK_1DA, q), (vK_1, q), (vK_1, AT_DT_q)],
                              D_pairs=[(vK_1, ASq), (vK_1, A * v_grids), (q, vK_1_ST_AT)])
        vK_1_d2K_q += gD[0]
        vK_1_d2K_q += gD[2].transpose(1,0,2,3)
        vK_1_d2R_V += gD[1]
        vK_1_d2K_q += d2Sii_w[0] + gS[0]
        vK_1_d2K_q += d2Sii_w[2] + gS[2]
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dSdxT, dAdxT_dot_DT_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dSdxT * A, dDdxT_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1_ST_dot_dAdxT, dDdxT_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dSdxT, dAdxT_dot_DT_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dSdxT * A, dDdxT_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1_ST_dot_dAdxT, dDdxT_dot_q)
        vK_1_d2K_q *= -f_eps_over_4pi
        vK_1_d2K_q += d2Sii_w[1] + gS[1]
        d2Sii = None

        d2e_from_d2KR -= vK_1_d2K_q

        dAdx_dot_V = get_dA_dot_q(dA, v_grids, atmlst)
        dDdx_dot_AV = get_dD_dot_q(dD, A * v_grids, atmlst, gridslice, ngrids)
        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + _mdot(dAdx_dot_V, D.T))
        dDdx_dot_AV = None

        K_1_dot_dRdx_dot_V = _K_solve_3d(pcmobj, dRdx_dot_V)

        d2e_from_d2KR -= numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)
        d2e_from_d2KR -= numpy.einsum('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)

        vK_1_d2R_V += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V *= f_eps_over_2pi

        d2e_from_d2KR += vK_1_d2R_V

        dK_1Rv = -K_1_dot_dKdx_dot_q + K_1_dot_dRdx_dot_V

        VK_1D_dot_dAdx = get_dA_dot_q(dA, (D.T @ vK_1).T, atmlst)
        VK_1_dot_dDdx = get_dDT_dot_q(dD, vK_1, atmlst, gridslice, ngrids)
        VK_1_dot_dRdx = f_eps_over_2pi * (VK_1D_dot_dAdx + VK_1_dot_dDdx * A)

        dvK_1R = -_K_solve_3d(pcmobj, vK_1_dot_dKdx, trans=True) @ R + VK_1_dot_dRdx

    else:
        raise RuntimeError(f"Unknown implicit solvent model: {pcmobj.method}")

    d2e = d2e_from_d2KR

    dVdx = get_dvgrids(pcmobj, dm, range(mol.natm))
    d2e -= numpy.einsum('Adi,BDi->BADd', dvK_1R, dVdx)
    d2e -= numpy.einsum('Adi,BDi->ABdD', dVdx, dK_1Rv)

    d2e *= 0.5
    t1 = log.timer_debug1('solvent hessian d(V * dK-1R/dx * V)/dx contribution', *t1)
    return d2e

def get_dqsym_dx_fix_vgrids(pcmobj, atmlst):
    assert pcmobj._intermediates is not None

    gridslice    = pcmobj.surface['gslice_by_atom']
    v_grids      = pcmobj._intermediates['v_grids']
    A            = pcmobj._intermediates['A']
    D            = pcmobj._intermediates['D']
    S            = pcmobj._intermediates['S']
    K            = pcmobj._intermediates['K']
    R            = pcmobj._intermediates['R']
    q            = pcmobj._intermediates['q']
    q_sym        = pcmobj._intermediates['q_sym']
    f_epsilon    = pcmobj._intermediates['f_epsilon']

    ngrids = q_sym.shape[0]

    if pcmobj.method.upper() in ['C-PCM', 'CPCM', 'COSMO']:
        dF, _ = get_dF_dA(pcmobj.surface, surface_discretization_method = pcmobj.surface_discretization_method)
        dS, dSii = _LazyD1(pcmobj.surface, 'S'), _get_dSii(pcmobj.surface, dF)
        dF = None

        # dR = 0, dK = dS
        dSdx_dot_q = get_dS_dot_q(dS, dSii, q_sym, atmlst, gridslice)

        dqdx_fix_Vq = _K_solve_3d(pcmobj, dSdx_dot_q)

    elif pcmobj.method.upper() in ['IEF-PCM', 'IEFPCM', 'SMD']:
        dF, dA = get_dF_dA(pcmobj.surface, surface_discretization_method = pcmobj.surface_discretization_method)
        dD, dS, dSii = _LazyD1(pcmobj.surface, 'D'), _LazyD1(pcmobj.surface, 'S'), _get_dSii(pcmobj.surface, dF)
        dF = None

        # dR = f_eps/(2*pi) * (dD*A + D*dA)
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)
        f_eps_over_2pi = f_epsilon/(2.0*PI)

        dSdx_dot_q = get_dS_dot_q(dS, dSii, q, atmlst, gridslice)

        DA = D*A
        dKdx_dot_q = dSdx_dot_q - f_eps_over_2pi * _mdot(dSdx_dot_q, DA.T)

        dAdx_dot_Sq = get_dA_dot_q(dA, S @ q, atmlst)
        dKdx_dot_q -= f_eps_over_2pi * _mdot(dAdx_dot_Sq, D.T)

        AS = (A * S.T).T # It's just diag(A) @ S
        dDdx_dot_ASq = get_dD_dot_q(dD, AS @ q, atmlst, gridslice, ngrids)
        dKdx_dot_q -= f_eps_over_2pi * dDdx_dot_ASq

        dqdx_fix_Vq = -_K_solve_3d(pcmobj, dKdx_dot_q)

        dAdx_dot_V = get_dA_dot_q(dA, v_grids, atmlst)

        dDdx_dot_AV = get_dD_dot_q(dD, A * v_grids, atmlst, gridslice, ngrids)

        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + _mdot(dAdx_dot_V, D.T))
        dqdx_fix_Vq += _K_solve_3d(pcmobj, dRdx_dot_V)

        invKT_V = _K_solve(pcmobj, v_grids, trans=True)
        dDdxT_dot_invKT_V = get_dDT_dot_q(dD, invKT_V, atmlst, gridslice, ngrids)

        DT_invKT_V = D.T @ invKT_V
        dAdxT_dot_DT_invKT_V = get_dA_dot_q(dA, DT_invKT_V, atmlst)
        dqdx_fix_Vq += f_eps_over_2pi * (numpy.einsum('i,Adi->Adi', A, dDdxT_dot_invKT_V) + dAdxT_dot_DT_invKT_V)

        dSdxT_dot_invKT_V = get_dST_dot_q(dS, dSii, invKT_V, atmlst, gridslice)
        dKdxT_dot_invKT_V = dSdxT_dot_invKT_V

        dKdxT_dot_invKT_V -= f_eps_over_2pi * _mdot(dDdxT_dot_invKT_V, AS)
        dKdxT_dot_invKT_V -= f_eps_over_2pi * _mdot(dAdxT_dot_DT_invKT_V, S)

        dSdxT_dot_AT_DT_invKT_V = get_dST_dot_q(dS, dSii, DA.T @ invKT_V, atmlst, gridslice)
        dKdxT_dot_invKT_V -= f_eps_over_2pi * dSdxT_dot_AT_DT_invKT_V
        invKT_dKdxT_dot_invKT_V = _K_solve_3d(pcmobj, dKdxT_dot_invKT_V, trans=True)

        dqdx_fix_Vq += -_mdot(invKT_dKdxT_dot_invKT_V, R)

        dqdx_fix_Vq *= -0.5

    elif pcmobj.method.upper() in ['SS(V)PE']:
        dF, dA = get_dF_dA(pcmobj.surface, surface_discretization_method = pcmobj.surface_discretization_method)
        dD, dS, dSii = _LazyD1(pcmobj.surface, 'D'), _LazyD1(pcmobj.surface, 'S'), _get_dSii(pcmobj.surface, dF)
        dF = None

        f_eps_over_4pi = f_epsilon/(4.0*PI)

        def dK_dot_q(q):
            dSdx_dot_q = get_dS_dot_q(dS, dSii, q, atmlst, gridslice)

            DA = D*A
            dKdx_dot_q = dSdx_dot_q - f_eps_over_4pi * _mdot(dSdx_dot_q, DA.T)

            dAdx_dot_Sq = get_dA_dot_q(dA, S @ q, atmlst)
            dKdx_dot_q -= f_eps_over_4pi * _mdot(dAdx_dot_Sq, D.T)

            AS = (A * S.T).T # It's just diag(A) @ S
            dDdx_dot_ASq = get_dD_dot_q(dD, AS @ q, atmlst, gridslice, ngrids)
            dKdx_dot_q -= f_eps_over_4pi * dDdx_dot_ASq

            dDdxT_dot_q = get_dDT_dot_q(dD, q, atmlst, gridslice, ngrids)
            dKdx_dot_q -= f_eps_over_4pi * _mdot(dDdxT_dot_q, AS)

            dAdxT_dot_DT_q = get_dA_dot_q(dA, D.T @ q, atmlst)
            dKdx_dot_q -= f_eps_over_4pi * _mdot(dAdxT_dot_DT_q, S)

            dSdxT_dot_AT_DT_q = get_dST_dot_q(dS, dSii, DA.T @ q, atmlst, gridslice)
            dKdx_dot_q -= f_eps_over_4pi * dSdxT_dot_AT_DT_q

            return dKdx_dot_q

        f_eps_over_2pi = f_epsilon/(2.0*PI)

        dKdx_dot_q = dK_dot_q(q)
        dqdx_fix_Vq = -_K_solve_3d(pcmobj, dKdx_dot_q)

        dAdx_dot_V = get_dA_dot_q(dA, v_grids, atmlst)

        dDdx_dot_AV = get_dD_dot_q(dD, A * v_grids, atmlst, gridslice, ngrids)

        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + _mdot(dAdx_dot_V, D.T))
        dqdx_fix_Vq += _K_solve_3d(pcmobj, dRdx_dot_V)

        invKT_V = _K_solve(pcmobj, v_grids, trans=True)
        dDdxT_dot_invKT_V = get_dDT_dot_q(dD, invKT_V, atmlst, gridslice, ngrids)

        DT_invKT_V = D.T @ invKT_V
        dAdxT_dot_DT_invKT_V = get_dA_dot_q(dA, DT_invKT_V, atmlst)
        dqdx_fix_Vq += f_eps_over_2pi * (numpy.einsum('i,Adi->Adi', A, dDdxT_dot_invKT_V) + dAdxT_dot_DT_invKT_V)

        dKdx_dot_invKT_V = dK_dot_q(invKT_V)
        invKT_dKdx_dot_invKT_V = _K_solve_3d(pcmobj, dKdx_dot_invKT_V, trans=True)

        dqdx_fix_Vq += -_mdot(invKT_dKdx_dot_invKT_V, R)

        dqdx_fix_Vq *= -0.5

    else:
        raise RuntimeError(f"Unknown implicit solvent model: {pcmobj.method}")

    return dqdx_fix_Vq

def _hess_intermediate(pcmobj, dm, key, fn):
    '''fn() for the density dm, kept in pcmobj._intermediates so that the terms
    of the Hessian and of H1 that need it compute it once'''
    dm = numpy.asarray(dm)
    cache = pcmobj._intermediates.get('hess_cache')
    if (cache is None or cache['dm'].shape != dm.shape or
        abs(cache['dm'] - dm).max() > 1e-12):
        cache = pcmobj._intermediates['hess_cache'] = {'dm': dm.copy()}
    if key not in cache:
        cache[key] = fn()
    return cache[key]

def _get_dv_parts(pcmobj, dm):
    '''The derivatives of the potential on the surface, for all atoms, in three
    parts: of the nuclei's potential, dvnuc[atom,x,g] (the nucleus and the
    points of the atom's own spheres move); the integrals contracted with the
    density for the AOs of an atom, dIdA[atom,x,g]; and for the point itself,
    dIdC[x,g]. dV/dx = dvnuc - dIdA - dIdC on the atom's own points.'''
    return _hess_intermediate(pcmobj, dm, 'dv_parts', lambda: _make_dv_parts(pcmobj, dm))

def _make_dv_parts(pcmobj, dm):
    assert pcmobj._intermediates is not None

    mol = pcmobj.mol
    nao = mol.nao
    natm = mol.natm
    gridslice    = pcmobj.surface['gslice_by_atom']
    charge_exp   = pcmobj.surface['charge_exp']
    grid_coords  = pcmobj.surface['grid_coords']
    ngrids = grid_coords.shape[0]

    atom_coords = mol.atom_coords(unit='B')
    atom_charges = numpy.asarray(mol.atom_charges(), dtype=numpy.float64)
    fakemol_nuc = gto.fakemol_for_charges(atom_coords)
    fakemol = gto.fakemol_for_charges(grid_coords, expnt=charge_exp**2)
    int2c2e_ip1 = mol._add_suffix('int2c2e_ip1')
    v_ng_ip1 = gto.mole.intor_cross(int2c2e_ip1, fakemol_nuc, fakemol)
    v_ng_ip1 = numpy.array(v_ng_ip1)
    dvnuc = numpy.einsum('dAq,A->Adq', v_ng_ip1, atom_charges)

    v_ng_ip2 = gto.mole.intor_cross(int2c2e_ip1, fakemol, fakemol_nuc)
    v_ng_ip2 = numpy.array(v_ng_ip2)
    for i_atom in range(natm):
        g0,g1 = gridslice[i_atom]
        dvnuc[i_atom,:,g0:g1] += numpy.einsum('dqA,A->dq', v_ng_ip2[:,g0:g1,:], atom_charges)

    # (d i j|g) with the density over the second AO, summed over the AOs of
    # each atom; (i j|d g) = -(d i j|g) - (i d j|g)
    aoslice = mol.aoslice_by_atom()
    w, = _int1e_grids_deriv(mol, pcmobj.surface, 'int1e_grids_ip', 3, dms_i=[dm + dm.T])
    dIdA = numpy.array([w[:, p0:p1].sum(axis=1) for p0, p1 in aoslice[:, 2:]])
    dIdC = -w.sum(axis=1)
    return dvnuc, dIdA, dIdC

def get_dvgrids(pcmobj, dm, atmlst):
    dvnuc, dIdA, dIdC = _get_dv_parts(pcmobj, dm)
    gridslice = pcmobj.surface['gslice_by_atom']
    dV_on_charge_dx = dvnuc - dIdA
    for i_atom in range(pcmobj.mol.natm):
        g0,g1 = gridslice[i_atom]
        dV_on_charge_dx[i_atom,:,g0:g1] -= dIdC[:,g0:g1]
    return dV_on_charge_dx[list(atmlst)]

def get_dqsym_dx_fix_K_R(pcmobj, dm, atmlst):
    dV_on_charge_dx = get_dvgrids(pcmobj, dm, atmlst)
    K = pcmobj._intermediates['K']
    R_dVdx = _R_dot(pcmobj, dV_on_charge_dx)
    K_1_R_dVdx = _K_solve_3d(pcmobj, R_dVdx)
    K_1T_dVdx = _K_solve_3d(pcmobj, dV_on_charge_dx, trans=True)
    RT_K_1T_dVdx = _R_dot(pcmobj, K_1T_dVdx, trans=True)
    dqdx_fix_K_R = 0.5 * (K_1_R_dVdx + RT_K_1T_dVdx)

    return dqdx_fix_K_R

def get_dqsym_dx(pcmobj, dm, atmlst):
    atmlst = tuple(atmlst)
    return _hess_intermediate(
        pcmobj, dm, ('dqsym_dx', atmlst),
        lambda: get_dqsym_dx_fix_vgrids(pcmobj, atmlst) + get_dqsym_dx_fix_K_R(pcmobj, dm, atmlst))

def analytical_grad_vmat(pcmobj, dm, atmlst=None, verbose=None):
    '''
    dv_solv / da
    '''
    if not pcmobj._intermediates:
        pcmobj.build()
    dm_cache = pcmobj._intermediates.get('dm', None)
    if dm_cache is not None and numpy.linalg.norm(dm_cache - dm) < 1e-10:
        pass
    else:
        pcmobj._get_vind(dm)
    mol = pcmobj.mol
    nao = mol.nao
    log = logger.new_logger(mol, mol.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    if atmlst is None:
        atmlst = range(mol.natm)

    gridslice    = pcmobj.surface['gslice_by_atom']
    charge_exp   = pcmobj.surface['charge_exp']
    grid_coords  = pcmobj.surface['grid_coords']
    q_sym        = pcmobj._intermediates['q_sym']
    ngrids = q_sym.shape[0]

    aoslice = mol.aoslice_by_atom()
    aoslice = numpy.array(aoslice)

    # M_B = sum_{g of atom B} q_g (d i j|g) for each atom. The derivative of the
    # potential with the charges of B moving is -(M_B + M_B^T); with the AOs of
    # an atom moving, the rows of sum_B M_B of that atom and their transpose.
    dIdA = numpy.zeros([3, nao, nao])
    dIdC_all = {}
    for j_atom in range(mol.natm):
        g0,g1 = gridslice[j_atom]
        if g1 == g0:
            continue
        m, = _int1e_grids_deriv(mol, pcmobj.surface, 'int1e_grids_ip', 3,
                                charges=[q_sym[g0:g1]], grids=(g0, g1))
        dIdA += m
        if j_atom in atmlst:
            dIdC_all[j_atom] = m
    dIdx = numpy.zeros([len(atmlst), 3, nao, nao])
    for i0, i_atom in enumerate(atmlst):
        p0,p1 = aoslice[i_atom, 2:]
        dIdx[i0, :, p0:p1, :] += dIdA[:, p0:p1, :]
        dIdx[i0, :, :, p0:p1] += dIdA[:, p0:p1, :].transpose(0,2,1)
        m = dIdC_all.pop(i_atom, None)
        if m is not None:
            dIdx[i0] -= m
            dIdx[i0] -= m.transpose(0,2,1)
    dIdA = m = None

    dV_on_molecule_dx = dIdx

    # the potentials of the charge derivatives of all perturbations from one
    # pass over the integrals
    dqdx = get_dqsym_dx(pcmobj, dm, atmlst)
    dqdx = dqdx.reshape(-1, ngrids)
    dv = dV_on_molecule_dx.reshape(-1, nao, nao)
    nset = max(3, int(4e8 / 8 / nao**2))
    for i0, i1 in lib.prange(0, len(dqdx), nset):
        dv[i0:i1] += _int1e_grids_dot(mol, pcmobj.surface, charges=dqdx[i0:i1])

    t1 = log.timer_debug1('computing solvent grad veff', *t1)
    return dV_on_molecule_dx

def make_hess_object(base_method):
    from pyscf.solvent._attach_solvent import _Solvation
    from pyscf.hessian.rhf import HessianBase
    if isinstance(base_method, HessianBase):
        # For backward compatibility. The input argument is a gradient object in
        # previous implementations.
        base_method = base_method.base

    # Must be a solvent-attached method
    assert isinstance(base_method, _Solvation)
    with_solvent = base_method.with_solvent
    if with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy hessian')

    vac_hess = base_method.undo_solvent().Hessian()
    vac_hess.base = base_method
    name = with_solvent.__class__.__name__ + vac_hess.__class__.__name__
    return lib.set_class(WithSolventHess(vac_hess),
                         (WithSolventHess, vac_hess.__class__), name)

class WithSolventHess:
    _keys = {'de_solvent', 'de_solute'}

    def __init__(self, hess_method):
        self.__dict__.update(hess_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventHess, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def to_gpu(self):
        from pyscf.lib.misc import to_gpu
        from pyscf.tdscf.rhf import TDBase
        if isinstance(self, TDBase):
            raise NotImplementedError('.to_gpu() for PCM-TDDFT')
        return to_gpu(self, self.base.to_gpu().Hessian())

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        with lib.temporary_env(self.base.with_solvent, equilibrium_solvation=True):
            logger.debug(self, 'Compute Hessian from solutes')
            self.de_solute = super().kernel(*args, **kwargs)

        logger.debug(self, 'Compute Hessian from solvents')
        if dm is None:
            dm = self.base.make_rdm1(ao_repr=True)
        if dm.ndim == 3:
            dm = dm[0] + dm[1]
        self.de_solvent = self.base.with_solvent.hess(dm)
        self.de = self.de_solute + self.de_solvent
        return self.de

    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        if atmlst is None:
            atmlst = range(self.mol.natm)
        h1ao = super().make_h1(mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
        if isinstance(self.base, scf.hf.RHF):
            dm = self.base.make_rdm1(ao_repr=True)
            dv = analytical_grad_vmat(self.base.with_solvent, dm, atmlst=atmlst, verbose=verbose)
            for i0, ia in enumerate(atmlst):
                h1ao[i0] += dv[i0]
            return h1ao
        elif isinstance(self.base, scf.uhf.UHF):
            h1aoa, h1aob = h1ao
            solvent = self.base.with_solvent
            dm = self.base.make_rdm1(ao_repr=True)
            dm = dm[0] + dm[1]
            dv = analytical_grad_vmat(solvent, dm, atmlst=atmlst, verbose=verbose)
            for i0, ia in enumerate(atmlst):
                h1aoa[i0] += dv[i0]
                h1aob[i0] += dv[i0]
            return h1aoa, h1aob
        else:
            raise NotImplementedError('Base object is not supported')

    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass

def _make_h1_split(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    '''WithSolventHess.make_h1 for hess_elec (hessian.rhf._MAKE_H1_SPLIT): the split form of
    the solute's make_h1 where it has one, with the solvent term in the AO matrices'''
    from pyscf.hessian import rhf as rhf_hess
    base_make = super(WithSolventHess, hessobj).make_h1
    split = rhf_hess._MAKE_H1_SPLIT.get(getattr(base_make, '__func__', None))
    if split is None or not isinstance(hessobj.base, scf.hf.RHF):
        return hessobj.make_h1(mo_coeff, mo_occ, chkfile, atmlst, verbose), None
    if atmlst is None:
        atmlst = range(hessobj.mol.natm)
    h1ao, h1occ = split(hessobj, mo_coeff, mo_occ, chkfile, atmlst, verbose)
    dm = hessobj.base.make_rdm1(ao_repr=True)
    dv = analytical_grad_vmat(hessobj.base.with_solvent, dm, atmlst=atmlst, verbose=verbose)
    for i0, ia in enumerate(atmlst):
        h1ao[i0] += dv[i0]
    return h1ao, h1occ

from pyscf.hessian import rhf as _rhf_hess
_rhf_hess._MAKE_H1_SPLIT[WithSolventHess.make_h1] = _make_h1_split


