#!/usr/bin/env python
# Copyright 2022 The PySCF Developers. All Rights Reserved.
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
Numerical integration functions for RKS and UKS with real AO basis
'''

import warnings
import ctypes
import inspect
import numpy
from pyscf import lib
try:
    from pyscf.dft import libxc
except (ImportError, OSError):
    try:
        from pyscf.dft import xcfun
        libxc = xcfun
    except (ImportError, OSError):
        warnings.warn('XC functional libraries (libxc or XCfun) are not available.')
        raise

from pyscf.dft.gen_grid import BLKSIZE, NBINS, CUTOFF, ALIGNMENT_UNIT, make_mask
from pyscf.gto.mole import NCTR_OF
from pyscf.dft import xc_deriv
from pyscf import __config__

libdft = lib.load_library('libdft')
OCCDROP = getattr(__config__, 'dft_numint_occdrop', 1e-12)
# The system size above which to consider the sparsity of the density matrix.
# If the number of AOs in the system is less than this value, all tensors are
# treated as dense quantities and contracted by dgemm directly.
SWITCH_SIZE = getattr(__config__, 'dft_numint_switch_size', 800)
# Grid blocks take the screened (sparse) kernels while less than this fraction of the
# shell x block screen index is nonzero. The screened kernels multiply gathered AO blocks
# with BLAS, so they are faster than the dense products until the index is nearly full.
SPARSE_THRESHOLD = getattr(__config__, 'dft_numint_sparse_threshold', 0.9)
# When every grid block takes the sparse kernels, nr_rks, nr_uks, the fxc contractions
# and cache_xc_kernel use kernels that compute the AO values in place instead of reading
# the AO blocks of block_loop (see _fused_xc). The results are the same bit for bit.
FUSED_XC = getattr(__config__, 'dft_numint_fused_xc', True)
# Those kernels can read the contracted radial parts of the AO values from a cache kept on
# the grids, built when it takes less than this fraction of the memory left (max_memory).
RADIAL_CACHE_FRACTION = getattr(__config__, 'dft_numint_radial_cache_fraction', .25)

# Whether to compute density laplacian for meta-GGA functionals
MGGA_DENSITY_LAPL = False

def eval_ao(mol, coords, deriv=0, shls_slice=None,
            non0tab=None, cutoff=None, out=None, verbose=None, zero_screened=True):
    '''Evaluate AO function value on the given grids.

    Args:
        mol : an instance of :class:`Mole`

        coords : 2D array, shape (N,3)
            The coordinates of the grids.

    Kwargs:
        deriv : int
            AO derivative order.  It affects the shape of the return array.
            If deriv=0, the returned AO values are stored in a (N,nao) array.
            Otherwise the AO values are stored in an array of shape (M,N,nao).
            Here N is the number of grids, nao is the number of AO functions,
            M is the size associated to the derivative deriv.
        relativity : bool
            No effects.
        shls_slice : 2-element list
            (shl_start, shl_end).
            If given, only part of AOs (shl_start <= shell_id < shl_end) are
            evaluated.  By default, all shells defined in mol will be evaluated.
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        cutoff : float
            AO values smaller than cutoff will be set to zero. The default
            cutoff threshold is ~1e-22 (defined in gto/grid_ao_drv.h)
        out : ndarray
            If provided, results are written into this array.
        verbose : int or object of :class:`Logger`
            No effects.
        zero_screened : bool
            If False, the AO values of the shells that non0tab screens on a
            block of grids are left unwritten instead of set to zero, for
            callers that only read the values where non0tab is nonzero.

    Returns:
        2D array of shape (N,nao) for AO values if deriv = 0.
        Or 3D array of shape (:,N,nao) for AO values and AO derivatives if deriv > 0.
        In the 3D array, the first (N,nao) elements are the AO values,
        followed by (3,N,nao) for x,y,z components;
        Then 2nd derivatives (6,N,nao) for xx, xy, xz, yy, yz, zz;
        Then 3rd derivatives (10,N,nao) for xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz;
        ...

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    >>> coords = numpy.random.random((100,3))  # 100 random points
    >>> ao_value = eval_ao(mol, coords)
    >>> print(ao_value.shape)
    (100, 24)
    >>> ao_value = eval_ao(mol, coords, deriv=1, shls_slice=(1,4))
    >>> print(ao_value.shape)
    (4, 100, 7)
    >>> ao_value = eval_ao(mol, coords, deriv=2, shls_slice=(1,4))
    >>> print(ao_value.shape)
    (10, 100, 7)
    '''
    comp = (deriv+1)*(deriv+2)*(deriv+3)//6
    if mol.cart:
        feval = 'GTOval_cart_deriv%d' % deriv
    else:
        feval = 'GTOval_sph_deriv%d' % deriv
    return mol.eval_gto(feval, coords, comp, shls_slice, non0tab,
                        cutoff=cutoff, out=out, zero_screened=zero_screened)

def eval_rho(mol, ao, dm, non0tab=None, xctype='LDA', hermi=0,
             with_lapl=True, verbose=None):
    r'''Calculate the electron density for LDA functional, and the density
    derivatives for GGA and MGGA functionals.

    Args:
        mol : an instance of :class:`Mole`

        ao : 2D array of shape (N,nao) for LDA, 3D array of shape (4,N,nao) for GGA
            or meta-GGA.  N is the number of grids, nao is the
            number of AO functions.  If xctype is GGA (MGGA), ao[0] is AO value
            and ao[1:3] are the AO gradients. ao[4:10] are second derivatives of
            ao values if applicable.
        dm : 2D array
            Density matrix

    Kwargs:
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        xctype : str
            LDA/GGA/mGGA.  It affects the shape of the return density.
        hermi : bool
            dm is hermitian or not
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        1D array of size N to store electron density if xctype = LDA;  2D array
        of (4,N) to store density and "density derivatives" for x,y,z components
        if xctype = GGA; For meta-GGA, returns can be a (6,N) (with_lapl=True)
        array where last two rows are \nabla^2 rho and tau = 1/2(\nabla f)^2
        or (5,N) (with_lapl=False) where the last row is tau = 1/2(\nabla f)^2

    Examples:

    >>> mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0', basis='ccpvdz')
    >>> coords = numpy.random.random((100,3))  # 100 random points
    >>> ao_value = eval_ao(mol, coords, deriv=0)
    >>> dm = numpy.random.random((mol.nao_nr(),mol.nao_nr()))
    >>> dm = dm + dm.T
    >>> rho, dx_rho, dy_rho, dz_rho = eval_rho(mol, ao, dm, xctype='LDA')
    '''
    xctype = xctype.upper()
    ngrids, nao = ao.shape[-2:]

    if not hermi and xctype in ('GGA', 'NLC', 'MGGA') and dm.dtype == numpy.double:
        # rho and its derivatives only depend on the symmetric part of a real dm.
        # Symmetrizing lets the hermi=1 branches skip the dm.T contractions.
        dm = lib.hermi_sum(numpy.asarray(dm, order='C')) * .5
        hermi = 1

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    if xctype == 'LDA' or xctype == 'HF':
        c0 = _dot_ao_dm(mol, ao, dm, non0tab, shls_slice, ao_loc)
        #:rho = numpy.einsum('pi,pi->p', ao, c0)
        rho = _contract_rho(ao, c0)
    elif xctype in ('GGA', 'NLC'):
        rho = numpy.empty((4,ngrids))
        c0 = _dot_ao_dm(mol, ao[0], dm, non0tab, shls_slice, ao_loc)
        #:rho[0] = numpy.einsum('pi,pi->p', c0, ao[0])
        rho[0] = _contract_rho(ao[0], c0)
        for i in range(1, 4):
            #:rho[i] = numpy.einsum('pi,pi->p', c0, ao[i])
            rho[i] = _contract_rho(ao[i], c0)
        if hermi:
            rho[1:4] *= 2  # *2 for + einsum('pi,ij,pj->p', ao[i], dm, ao[0])
        else:
            c1 = _dot_ao_dm(mol, ao[0], dm.conj().T, non0tab, shls_slice, ao_loc)
            for i in range(1, 4):
                rho[i] += _contract_rho(c1, ao[i])
    else: # meta-GGA
        if with_lapl:
            # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
            rho = numpy.empty((6,ngrids))
            tau_idx = 5
        else:
            rho = numpy.empty((5,ngrids))
            tau_idx = 4
        c0 = _dot_ao_dm(mol, ao[0], dm, non0tab, shls_slice, ao_loc)
        #:rho[0] = numpy.einsum('pi,pi->p', ao[0], c0)
        rho[0] = _contract_rho(ao[0], c0)

        rho[tau_idx] = 0
        for i in range(1, 4):
            c1 = _dot_ao_dm(mol, ao[i], dm, non0tab, shls_slice, ao_loc)
            #:rho[tau_idx] += numpy.einsum('pi,pi->p', c1, ao[i])
            rho[tau_idx] += _contract_rho(ao[i], c1)

            #:rho[i] = numpy.einsum('pi,pi->p', c0, ao[i])
            rho[i] = _contract_rho(ao[i], c0)
            if hermi:
                rho[i] *= 2
            else:
                rho[i] += _contract_rho(c1, ao[0])

        if with_lapl:
            if ao.shape[0] > 4:
                XX, YY, ZZ = 4, 7, 9
                ao2 = ao[XX] + ao[YY] + ao[ZZ]
                # \nabla^2 rho
                #:rho[4] = numpy.einsum('pi,pi->p', c0, ao2)
                rho[4] = _contract_rho(ao2, c0)
                rho[4] += rho[5]
                if hermi:
                    rho[4] *= 2
                else:
                    c2 = _dot_ao_dm(mol, ao2, dm, non0tab, shls_slice, ao_loc)
                    rho[4] += _contract_rho(ao[0], c2)
                    rho[4] += rho[5]
            elif MGGA_DENSITY_LAPL:
                raise ValueError('Not enough derivatives in ao')
        # tau = 1/2 (\nabla f)^2
        rho[tau_idx] *= .5
    return rho

def eval_rho1(mol, ao, dm, screen_index=None, xctype='LDA', hermi=0,
              with_lapl=True, cutoff=None, ao_cutoff=CUTOFF, pair_mask=None,
              verbose=None):
    r'''Calculate the electron density for LDA and the density derivatives for
    GGA and MGGA with sparsity information.

    Args:
        mol : an instance of :class:`Mole`

        ao : 2D array of shape (N,nao) for LDA, 3D array of shape (4,N,nao) for GGA
            or meta-GGA.  N is the number of grids, nao is the
            number of AO functions.  If xctype is GGA (MGGA), ao[0] is AO value
            and ao[1:3] are the AO gradients. ao[4:10] are second derivatives of
            ao values if applicable.
        dm : 2D array
            Density matrix

    Kwargs:
        screen_index : 2D uint8 array
            How likely the AO values on grids are negligible. This array can be
            obtained by calling :func:`gen_grid.make_screen_index`
        xctype : str
            LDA/GGA/mGGA.  It affects the shape of the return density.
        hermi : bool
            dm is hermitian or not
        cutoff : float
            cutoff for density value
        ao_cutoff : float
            cutoff for AO value. Needs to be the same to the cutoff when
            generating screen_index
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        1D array of size N to store electron density if xctype = LDA;  2D array
        of (4,N) to store density and "density derivatives" for x,y,z components
        if xctype = GGA; For meta-GGA, returns can be a (6,N) (with_lapl=True)
        array where last two rows are \nabla^2 rho and tau = 1/2(\nabla f)^2
        or (5,N) (with_lapl=False) where the last row is tau = 1/2(\nabla f)^2
    '''
    if not (dm.dtype == ao.dtype == numpy.double):
        lib.logger.warn(mol, 'eval_rho1 does not support complex density, '
                        'eval_rho is called instead')
        return eval_rho(mol, ao, dm, screen_index, xctype, hermi, with_lapl, verbose)

    xctype = xctype.upper()
    ngrids = ao.shape[-2]

    ao_loc = mol.ao_loc_nr()
    if (hermi and screen_index is not None and ngrids % ALIGNMENT_UNIT == 0 and
            not (xctype == 'MGGA' and with_lapl) and
            (ao.ndim == 2 and ao.flags.f_contiguous if xctype in ('LDA', 'HF') else
             _ao_comps_contiguous(ao) and ao.shape[0] >= 4)):
        return _eval_rho_sparse(ao, dm, screen_index, ao_loc, xctype)

    if cutoff is None:
        cutoff = CUTOFF
    cutoff = min(cutoff, .1)
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(ao_cutoff))

    if pair_mask is None:
        ovlp_cond = mol.get_overlap_cond()
        pair_mask = numpy.asarray(ovlp_cond < -numpy.log(cutoff), dtype=numpy.uint8)

    if xctype == 'LDA' or xctype == 'HF':
        c0 = _dot_ao_dm_sparse(ao, dm, nbins, screen_index, pair_mask, ao_loc)
        rho = _contract_rho_sparse(ao, c0, screen_index, ao_loc)
    elif xctype in ('GGA', 'NLC'):
        rho = numpy.empty((4,ngrids))
        c0 = _dot_ao_dm_sparse(ao[0], dm, nbins, screen_index, pair_mask, ao_loc)
        rho[0] = _contract_rho_sparse(ao[0], c0, screen_index, ao_loc)
        for i in range(1, 4):
            rho[i] = _contract_rho_sparse(ao[i], c0, screen_index, ao_loc)
        if hermi:
            rho[1:4] *= 2  # *2 for + einsum('pi,ij,pj->p', ao[i], dm, ao[0])
        else:
            dm = lib.transpose(dm)
            c0 = _dot_ao_dm_sparse(ao[0], dm, nbins, screen_index, pair_mask, ao_loc)
            for i in range(1, 4):
                rho[i] += _contract_rho_sparse(c0, ao[i], screen_index, ao_loc)
    else: # meta-GGA
        if with_lapl:
            if MGGA_DENSITY_LAPL:
                raise NotImplementedError('density laplacian not supported')
            rho = numpy.empty((6,ngrids))
            tau_idx = 5
        else:
            rho = numpy.empty((5,ngrids))
            tau_idx = 4
        c0 = _dot_ao_dm_sparse(ao[0], dm, nbins, screen_index, pair_mask, ao_loc)
        rho[0] = _contract_rho_sparse(ao[0], c0, screen_index, ao_loc)

        rho[tau_idx] = 0
        for i in range(1, 4):
            c1 = _dot_ao_dm_sparse(ao[i], dm, nbins, screen_index, pair_mask, ao_loc)
            rho[tau_idx] += _contract_rho_sparse(ao[i], c1, screen_index, ao_loc)

            rho[i] = _contract_rho_sparse(ao[i], c0, screen_index, ao_loc)
            if hermi:
                rho[i] *= 2
            else:
                rho[i] += _contract_rho_sparse(ao[0], c1, screen_index, ao_loc)

        # tau = 1/2 (\nabla f)^2
        rho[tau_idx] *= .5
    return rho

def eval_rho2(mol, ao, mo_coeff, mo_occ, non0tab=None, xctype='LDA',
              with_lapl=True, verbose=None):
    r'''Calculate the electron density for LDA functional, and the density
    derivatives for GGA functional.  This function has the same functionality
    as :func:`eval_rho` except that the density are evaluated based on orbital
    coefficients and orbital occupancy.  It is more efficient than
    :func:`eval_rho` in most scenario.

    Args:
        mol : an instance of :class:`Mole`

        ao : 2D array of shape (N,nao) for LDA, 3D array of shape (4,N,nao) for GGA
            or meta-GGA.  N is the number of grids, nao is the
            number of AO functions.  If xctype is GGA (MGGA), ao[0] is AO value
            and ao[1:3] are the AO gradients. ao[4:10] are second derivatives of
            ao values if applicable.
        dm : 2D array
            Density matrix

    Kwargs:
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        xctype : str
            LDA/GGA/mGGA.  It affects the shape of the return density.
        with_lapl: bool
            Whether to compute laplacian. It affects the shape of returns.
        verbose : int or object of :class:`Logger`
            No effects.

    Returns:
        1D array of size N to store electron density if xctype = LDA;  2D array
        of (4,N) to store density and "density derivatives" for x,y,z components
        if xctype = GGA; For meta-GGA, returns can be a (6,N) (with_lapl=True)
        array where last two rows are \nabla^2 rho and tau = 1/2(\nabla f)^2
        or (5,N) (with_lapl=False) where the last row is tau = 1/2(\nabla f)^2
    '''
    xctype = xctype.upper()
    ngrids, nao = ao.shape[-2:]

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    pos = mo_occ > OCCDROP
    if numpy.any(pos):
        cpos = numpy.einsum('ij,j->ij', mo_coeff[:,pos], numpy.sqrt(mo_occ[pos]))
        if xctype == 'LDA' or xctype == 'HF':
            c0 = _dot_ao_dm(mol, ao, cpos, non0tab, shls_slice, ao_loc)
            #:rho = numpy.einsum('pi,pi->p', c0, c0)
            rho = _contract_rho(c0, c0)
        elif xctype in ('GGA', 'NLC'):
            rho = numpy.empty((4,ngrids))
            c0 = _dot_ao_dm(mol, ao[0], cpos, non0tab, shls_slice, ao_loc)
            #:rho[0] = numpy.einsum('pi,pi->p', c0, c0)
            rho[0] = _contract_rho(c0, c0)
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cpos, non0tab, shls_slice, ao_loc)
                #:rho[i] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
                rho[i] = _contract_rho(c0, c1) * 2
        else: # meta-GGA
            if with_lapl:
                # rho[4] = \nabla^2 rho, rho[5] = 1/2 |nabla f|^2
                rho = numpy.empty((6,ngrids))
                tau_idx = 5
            else:
                rho = numpy.empty((5,ngrids))
                tau_idx = 4
            c0 = _dot_ao_dm(mol, ao[0], cpos, non0tab, shls_slice, ao_loc)
            #:rho[0] = numpy.einsum('pi,pi->p', c0, c0)
            rho[0] = _contract_rho(c0, c0)

            rho[tau_idx] = 0
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cpos, non0tab, shls_slice, ao_loc)
                #:rho[i] = numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
                #:rho[5] += numpy.einsum('pi,pi->p', c1, c1)
                rho[i] = _contract_rho(c0, c1) * 2
                rho[tau_idx] += _contract_rho(c1, c1)

            if with_lapl:
                if ao.shape[0] > 4:
                    XX, YY, ZZ = 4, 7, 9
                    ao2 = ao[XX] + ao[YY] + ao[ZZ]
                    c1 = _dot_ao_dm(mol, ao2, cpos, non0tab, shls_slice, ao_loc)
                    #:rho[4] = numpy.einsum('pi,pi->p', c0, c1)
                    rho[4] = _contract_rho(c0, c1)
                    rho[4] += rho[5]
                    rho[4] *= 2
                else:
                    rho[4] = 0
            rho[tau_idx] *= .5
    else:
        if xctype == 'LDA' or xctype == 'HF':
            rho = numpy.zeros(ngrids)
        elif xctype in ('GGA', 'NLC'):
            rho = numpy.zeros((4,ngrids))
        else:  # meta-GGA
            if with_lapl:
                rho = numpy.zeros((6, ngrids))
            else:
                rho = numpy.zeros((5, ngrids))

    neg = mo_occ < -OCCDROP
    if numpy.any(neg):
        cneg = numpy.einsum('ij,j->ij', mo_coeff[:,neg], numpy.sqrt(-mo_occ[neg]))
        if xctype == 'LDA' or xctype == 'HF':
            c0 = _dot_ao_dm(mol, ao, cneg, non0tab, shls_slice, ao_loc)
            #:rho -= numpy.einsum('pi,pi->p', c0, c0)
            rho -= _contract_rho(c0, c0)
        elif xctype == 'GGA':
            c0 = _dot_ao_dm(mol, ao[0], cneg, non0tab, shls_slice, ao_loc)
            #:rho[0] -= numpy.einsum('pi,pi->p', c0, c0)
            rho[0] -= _contract_rho(c0, c0)
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cneg, non0tab, shls_slice, ao_loc)
                #:rho[i] -= numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
                rho[i] -= _contract_rho(c0, c1) * 2 # *2 for +c.c.
        else:
            c0 = _dot_ao_dm(mol, ao[0], cneg, non0tab, shls_slice, ao_loc)
            #:rho[0] -= numpy.einsum('pi,pi->p', c0, c0)
            rho[0] -= _contract_rho(c0, c0)

            rho5 = 0
            for i in range(1, 4):
                c1 = _dot_ao_dm(mol, ao[i], cneg, non0tab, shls_slice, ao_loc)
                #:rho[i] -= numpy.einsum('pi,pi->p', c0, c1) * 2 # *2 for +c.c.
                #:rho5 += numpy.einsum('pi,pi->p', c1, c1)
                rho[i] -= _contract_rho(c0, c1) * 2 # *2 for +c.c.
                rho5 += _contract_rho(c1, c1)

            if with_lapl:
                if ao.shape[0] > 4:
                    XX, YY, ZZ = 4, 7, 9
                    ao2 = ao[XX] + ao[YY] + ao[ZZ]
                    c1 = _dot_ao_dm(mol, ao2, cneg, non0tab, shls_slice, ao_loc)
                    #:rho[4] -= numpy.einsum('pi,pi->p', c0, c1) * 2
                    rho[4] -= _contract_rho(c0, c1) * 2
                    rho[4] -= rho5 * 2
                else:
                    rho[4] = 0

            rho[tau_idx] -= rho5 * .5
    return rho

def _vv10nlc(rho, coords, vvrho, vvweight, vvcoords, nlc_pars):
    thresh=1e-8

    #output
    exc=numpy.zeros(rho[0,:].size)
    vxc=numpy.zeros([2,rho[0,:].size])

    #outer grid needs threshing
    threshind=rho[0,:]>=thresh
    coords=coords[threshind]
    R=rho[0,:][threshind]
    Gx=rho[1,:][threshind]
    Gy=rho[2,:][threshind]
    Gz=rho[3,:][threshind]
    G=Gx**2.+Gy**2.+Gz**2.

    #inner grid needs threshing
    innerthreshind=vvrho[0,:]>=thresh
    vvcoords=vvcoords[innerthreshind]
    vvweight=vvweight[innerthreshind]
    Rp=vvrho[0,:][innerthreshind]
    RpW=Rp*vvweight
    Gxp=vvrho[1,:][innerthreshind]
    Gyp=vvrho[2,:][innerthreshind]
    Gzp=vvrho[3,:][innerthreshind]
    Gp=Gxp**2.+Gyp**2.+Gzp**2.

    #constants and parameters
    Pi=numpy.pi
    Pi43=4.*Pi/3.
    Bvv, Cvv = nlc_pars
    Kvv=Bvv*1.5*Pi*((9.*Pi)**(-1./6.))
    Beta=((3./(Bvv*Bvv))**(0.75))/32.

    #inner grid
    W0p=Gp/(Rp*Rp)
    W0p=Cvv*W0p*W0p
    W0p=(W0p+Pi43*Rp)**0.5
    Kp=Kvv*(Rp**(1./6.))

    #outer grid
    W0tmp=G/(R**2)
    W0tmp=Cvv*W0tmp*W0tmp
    W0=(W0tmp+Pi43*R)**0.5
    dW0dR=(0.5*Pi43*R-2.*W0tmp)/W0
    dW0dG=W0tmp*R/(G*W0)
    K=Kvv*(R**(1./6.))
    dKdR=(1./6.)*K

    vvcoords = numpy.asarray(vvcoords, order='C')
    coords = numpy.asarray(coords, order='C')
    F = numpy.empty_like(R)
    U = numpy.empty_like(R)
    W = numpy.empty_like(R)
    #for i in range(R.size):
    #    DX=vvcoords[:,0]-coords[i,0]
    #    DY=vvcoords[:,1]-coords[i,1]
    #    DZ=vvcoords[:,2]-coords[i,2]
    #    R2=DX*DX+DY*DY+DZ*DZ
    #    gp=R2*W0p+Kp
    #    g=R2*W0[i]+K[i]
    #    gt=g+gp
    #    T=RpW/(g*gp*gt)
    #    F=numpy.sum(T)
    #    T*=(1./g+1./gt)
    #    U=numpy.sum(T)
    #    W=numpy.sum(T*R2)
    #    F*=-1.5
    libdft.VXC_vv10nlc(F.ctypes.data_as(ctypes.c_void_p),
                       U.ctypes.data_as(ctypes.c_void_p),
                       W.ctypes.data_as(ctypes.c_void_p),
                       vvcoords.ctypes.data_as(ctypes.c_void_p),
                       coords.ctypes.data_as(ctypes.c_void_p),
                       W0p.ctypes.data_as(ctypes.c_void_p),
                       W0.ctypes.data_as(ctypes.c_void_p),
                       K.ctypes.data_as(ctypes.c_void_p),
                       Kp.ctypes.data_as(ctypes.c_void_p),
                       RpW.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_int(vvcoords.shape[0]),
                       ctypes.c_int(coords.shape[0]))
    #exc is multiplied by Rho later
    exc[threshind] = Beta+0.5*F
    vxc[0,threshind] = Beta+F+1.5*(U*dKdR+W*dW0dR)
    vxc[1,threshind] = 1.5*W*dW0dG
    return exc,vxc

def eval_mat(mol, ao, weight, rho, vxc,
             non0tab=None, xctype='LDA', spin=0, verbose=None):
    r'''Calculate XC potential matrix.

    Args:
        mol : an instance of :class:`Mole`

        ao : ([4/10,] ngrids, nao) ndarray
            2D array of shape (N,nao) for LDA,
            3D array of shape (4,N,nao) for GGA
            or (10,N,nao) for meta-GGA.
            N is the number of grids, nao is the number of AO functions.
            If xctype is GGA (MGGA), ao[0] is AO value and ao[1:3] are the real space
            gradients. ao[4:10] are second derivatives of ao values if applicable.
        weight : 1D array
            Integral weights on grids.
        rho : ([4/6,] ngrids) ndarray
            Shape of ((*,N)) for electron density (and derivatives) if spin = 0;
            Shape of ((*,N),(*,N)) for alpha/beta electron density (and derivatives) if spin > 0;
            where N is number of grids.
            rho (*,N) are ordered as (den,grad_x,grad_y,grad_z,laplacian,tau)
            where grad_x = d/dx den, laplacian = \nabla^2 den, tau = 1/2(\nabla f)^2
            In spin unrestricted case,
            rho is ((den_u,grad_xu,grad_yu,grad_zu,laplacian_u,tau_u)
                    (den_d,grad_xd,grad_yd,grad_zd,laplacian_d,tau_d))
        vxc : ([4,] ngrids) ndarray
            XC potential value on each grid = (vrho, vsigma, vlapl, vtau)
            vsigma is GGA potential value on each grid.
            If the kwarg spin != 0, a list [vsigma_uu,vsigma_ud] is required.

    Kwargs:
        xctype : str
            LDA/GGA/mGGA.  It affects the shape of `ao` and `rho`
        non0tab : 2D bool array
            mask array to indicate whether the AO values are zero.  The mask
            array can be obtained by calling :func:`make_mask`
        spin : int
            If not 0, the returned matrix is the Vxc matrix of alpha-spin.  It
            is computed with the spin non-degenerated UKS formula.

    Returns:
        XC potential matrix in 2D array of shape (nao,nao) where nao is the
        number of AO functions.
    '''
    xctype = xctype.upper()
    ngrids, nao = ao.shape[-2:]

    if non0tab is None:
        non0tab = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                             dtype=numpy.uint8)
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    transpose_for_uks = False
    if xctype == 'LDA' or xctype == 'HF':
        if not isinstance(vxc, numpy.ndarray) or vxc.ndim == 2:
            vrho = vxc[0]
        else:
            vrho = vxc
        # *.5 because return mat + mat.T
        #:aow = numpy.einsum('pi,p->pi', ao, .5*weight*vrho)
        aow = _scale_ao(ao, .5*weight*vrho)
        mat = _dot_ao_ao(mol, ao, aow, non0tab, shls_slice, ao_loc)
    else:
        #wv = weight * vsigma * 2
        #aow  = numpy.einsum('pi,p->pi', ao[1], rho[1]*wv)
        #aow += numpy.einsum('pi,p->pi', ao[2], rho[2]*wv)
        #aow += numpy.einsum('pi,p->pi', ao[3], rho[3]*wv)
        #aow += numpy.einsum('pi,p->pi', ao[0], .5*weight*vrho)
        vrho, vsigma = vxc[:2]
        if spin == 0:
            assert (vsigma is not None and rho.ndim==2)
            wv = _rks_gga_wv0(rho, vxc, weight)
        else:
            rho_a, rho_b = rho
            wv = numpy.empty((4,ngrids))
            wv[0] = weight * vrho * .5
            try:
                wv[1:4] = rho_a[1:4] * (weight * vsigma[0] * 2)  # sigma_uu
                wv[1:4]+= rho_b[1:4] * (weight * vsigma[1])      # sigma_ud
            except ValueError:
                warnings.warn('Note the output of libxc.eval_xc cannot be '
                              'directly used in eval_mat.\nvsigma from eval_xc '
                              'should be restructured as '
                              '(vsigma[:,0],vsigma[:,1])\n')
                transpose_for_uks = True
                vsigma = vsigma.T
                wv[1:4] = rho_a[1:4] * (weight * vsigma[0] * 2)  # sigma_uu
                wv[1:4]+= rho_b[1:4] * (weight * vsigma[1])      # sigma_ud
        #:aow = numpy.einsum('npi,np->pi', ao[:4], wv)
        aow = _scale_ao(ao[:4], wv)
        mat = _dot_ao_ao(mol, ao[0], aow, non0tab, shls_slice, ao_loc)

# JCP 138, 244108 (2013); DOI:10.1063/1.4811270
# JCP 112, 7002 (2000); DOI:10.1063/1.481298
    if xctype == 'MGGA':
        vlapl, vtau = vxc[2:]

        if vlapl is None:
            if spin != 0:
                if transpose_for_uks:
                    vtau = vtau.T
                vtau = vtau[0]
            wv = weight * .25 * vtau
            mat += _tau_dot(mol, ao, ao, wv, non0tab, shls_slice, ao_loc)
        else:
            if spin != 0:
                if transpose_for_uks:
                    vlapl = vlapl.T
                vlapl = vlapl[0]
            if ao.shape[0] > 4:
                XX, YY, ZZ = 4, 7, 9
                ao2 = ao[XX] + ao[YY] + ao[ZZ]
                #:aow = numpy.einsum('pi,p->pi', ao2, .5 * weight * vlapl, out=aow)
                aow = _scale_ao(ao2, .5 * weight * vlapl, out=aow)
                mat += _dot_ao_ao(mol, ao[0], aow, non0tab, shls_slice, ao_loc)
            else:
                raise ValueError('Not enough derivatives in ao')

    return mat + mat.T.conj()

def _screened_gemm(mol, ao, non0tab, shls_slice, ao_loc):
    '''Whether the products of real ao[ngrids,nao] over all shells of mol use the
    sparse kernels for the screen index non0tab (zero where eval_ao zeroed the
    AO values).'''
    return (non0tab is not None and shls_slice is not None and ao_loc is not None
            and ao.dtype == numpy.double and non0tab.ndim == 2
            and tuple(shls_slice) == (0, mol.nbas) and non0tab.shape[1] == mol.nbas
            and ao.shape[1] == ao_loc[mol.nbas]
            and non0tab.shape[0] * BLKSIZE >= ao.shape[0]
            and (ao.shape[1] > SWITCH_SIZE or _sparse_enough(non0tab)))

def _dot_ao_ao(mol, ao1, ao2, non0tab, shls_slice, ao_loc, hermi=0):
    '''return numpy.dot(ao1.T, ao2)'''
    ngrids, nao = ao1.shape
    if (_screened_gemm(mol, ao1, non0tab, shls_slice, ao_loc) and
            ao2.dtype == numpy.double):
        if not ao1.flags.f_contiguous:
            ao1 = lib.transpose(ao1)
        if not ao2.flags.f_contiguous:
            ao2 = lib.transpose(ao2)
        vv = numpy.zeros((nao,nao))
        libdft.VXCdot_aow_ao_sparse_gemm(
            vv.ctypes.data_as(ctypes.c_void_p),
            ao1.ctypes.data_as(ctypes.c_void_p),
            ao2.ctypes.data_as(ctypes.c_void_p), None,
            ctypes.c_int(nao), ctypes.c_int(ngrids),
            ctypes.c_int(mol.nbas), ctypes.c_int(hermi),
            non0tab.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p))
        return vv
    if (nao < SWITCH_SIZE or
        non0tab is None or shls_slice is None or ao_loc is None):
        return lib.dot(ao1.conj().T, ao2)

    if not ao1.flags.f_contiguous:
        ao1 = lib.transpose(ao1)
    if not ao2.flags.f_contiguous:
        ao2 = lib.transpose(ao2)
    if ao1.dtype == ao2.dtype == numpy.double:
        fn = libdft.VXCdot_ao_ao
    else:
        fn = libdft.VXCzdot_ao_ao
        ao1 = numpy.asarray(ao1, numpy.complex128)
        ao2 = numpy.asarray(ao2, numpy.complex128)

    vv = numpy.empty((nao,nao), dtype=ao1.dtype)
    fn(vv.ctypes.data_as(ctypes.c_void_p),
       ao1.ctypes.data_as(ctypes.c_void_p),
       ao2.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(nao), ctypes.c_int(ngrids),
       ctypes.c_int(mol.nbas), ctypes.c_int(hermi),
       non0tab.ctypes.data_as(ctypes.c_void_p),
       (ctypes.c_int*2)(*shls_slice),
       ao_loc.ctypes.data_as(ctypes.c_void_p))
    return vv

def _dot_ao_dm(mol, ao, dm, non0tab, shls_slice, ao_loc, out=None):
    '''return numpy.dot(ao, dm)'''
    ngrids, nao = ao.shape
    if (_screened_gemm(mol, ao, non0tab, shls_slice, ao_loc) and
            dm.dtype == numpy.double):
        if not ao.flags.f_contiguous:
            ao = lib.transpose(ao)
        dm = numpy.asarray(dm, order='C')
        vm = numpy.ndarray((ngrids,dm.shape[1]), order='F', buffer=out)
        libdft.VXCdot_ao_mo_sparse_gemm(
            vm.ctypes.data_as(ctypes.c_void_p),
            ao.ctypes.data_as(ctypes.c_void_p),
            dm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(dm.shape[1]),
            ctypes.c_int(nao), ctypes.c_int(ngrids), ctypes.c_int(mol.nbas),
            non0tab.ctypes.data_as(ctypes.c_void_p),
            ao_loc.ctypes.data_as(ctypes.c_void_p))
        return vm
    if (nao < SWITCH_SIZE or
        non0tab is None or shls_slice is None or ao_loc is None):
        return lib.dot(dm.T, ao.T).T

    if not ao.flags.f_contiguous:
        ao = lib.transpose(ao)
    if ao.dtype == dm.dtype == numpy.double:
        fn = libdft.VXCdot_ao_dm
    else:
        fn = libdft.VXCzdot_ao_dm
        ao = numpy.asarray(ao, numpy.complex128)
        dm = numpy.asarray(dm, numpy.complex128)

    vm = numpy.ndarray((ngrids,dm.shape[1]), dtype=ao.dtype, order='F', buffer=out)
    dm = numpy.asarray(dm, order='C')
    fn(vm.ctypes.data_as(ctypes.c_void_p),
       ao.ctypes.data_as(ctypes.c_void_p),
       dm.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(nao), ctypes.c_int(dm.shape[1]),
       ctypes.c_int(ngrids), ctypes.c_int(mol.nbas),
       non0tab.ctypes.data_as(ctypes.c_void_p),
       (ctypes.c_int*2)(*shls_slice),
       ao_loc.ctypes.data_as(ctypes.c_void_p))
    return vm

def _scale_ao(ao, wv, out=None):
    #:aow = numpy.einsum('npi,np->pi', ao[:4], wv)
    if wv.ndim == 2:
        ao = ao.transpose(0,2,1)
    else:
        ngrids, nao = ao.shape
        ao = ao.T.reshape(1,nao,ngrids)
        wv = wv.reshape(1,ngrids)

    if not ao.flags.c_contiguous:
        return numpy.einsum('nip,np->pi', ao, wv)

    if ao.dtype == numpy.double:
        if wv.dtype == numpy.double:
            fn = libdft.VXC_dscale_ao
            dtype = numpy.double
        elif wv.dtype == numpy.complex128:
            fn = libdft.VXC_dzscale_ao
            dtype = numpy.complex128
        else:
            return numpy.einsum('nip,np->pi', ao, wv)
    elif ao.dtype == numpy.complex128:
        if wv.dtype == numpy.double:
            fn = libdft.VXC_zscale_ao
            dtype = numpy.complex128
        elif wv.dtype == numpy.complex128:
            fn = libdft.VXC_zzscale_ao
            dtype = numpy.complex128
        else:
            return numpy.einsum('nip,np->pi', ao, wv)
    else:
        return numpy.einsum('nip,np->pi', ao, wv)

    wv = numpy.asarray(wv, order='C')
    comp, nao, ngrids = ao.shape
    assert wv.shape[0] == comp
    aow = numpy.ndarray((nao,ngrids), dtype=dtype, buffer=out).T
    fn(aow.ctypes.data_as(ctypes.c_void_p),
       ao.ctypes.data_as(ctypes.c_void_p),
       wv.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(comp), ctypes.c_int(nao),
       ctypes.c_int(ngrids))
    return aow

def _contract_rho(bra, ket):
    '''Real part of rho for rho=einsum('pi,pi->p', bra.conj(), ket)'''
    bra = bra.T
    ket = ket.T
    nao, ngrids = bra.shape
    rho = numpy.empty(ngrids)

    if not (bra.flags.c_contiguous and ket.flags.c_contiguous):
        rho  = numpy.einsum('ip,ip->p', bra.real, ket.real)
        rho += numpy.einsum('ip,ip->p', bra.imag, ket.imag)
    elif bra.dtype == numpy.double and ket.dtype == numpy.double:
        libdft.VXC_dcontract_rho(rho.ctypes.data_as(ctypes.c_void_p),
                                 bra.ctypes.data_as(ctypes.c_void_p),
                                 ket.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_int(nao), ctypes.c_int(ngrids))
    elif bra.dtype == numpy.complex128 and ket.dtype == numpy.complex128:
        libdft.VXC_zcontract_rho(rho.ctypes.data_as(ctypes.c_void_p),
                                 bra.ctypes.data_as(ctypes.c_void_p),
                                 ket.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_int(nao), ctypes.c_int(ngrids))
    else:
        rho  = numpy.einsum('ip,ip->p', bra.real, ket.real)
        rho += numpy.einsum('ip,ip->p', bra.imag, ket.imag)
    return rho

def _contract_rho_batch(ao, bra):
    '''rho[k,c,g] = einsum('cgp,kpg->kcg', ao, bra) for real ao[ncomp,ngrids,nao]
    and bra[nset,nao,ngrids]. Each AO value is read once for all nset sets.
    '''
    ncomp, ngrids, nao = ao.shape
    nset = bra.shape[0]
    aot = ao.transpose(0,2,1)
    assert aot.flags.c_contiguous and bra.flags.c_contiguous
    rho = numpy.empty((nset,ncomp,ngrids))
    libdft.VXC_dcontract_rho_batch(
        rho.ctypes.data_as(ctypes.c_void_p),
        aot.ctypes.data_as(ctypes.c_void_p),
        bra.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ncomp), ctypes.c_int(nset), ctypes.c_int(nao),
        ctypes.c_int(ngrids))
    return rho

def _scale_ao_batch(ao, wv):
    '''aow[k,p,g] = einsum('cgp,kcg->kpg', ao, wv) for real ao[ncomp,ngrids,nao]
    and wv[nset,ncomp,ngrids]. Each AO value is read once for all nset sets.
    '''
    ncomp, ngrids, nao = ao.shape
    nset = wv.shape[0]
    aot = ao.transpose(0,2,1)
    assert aot.flags.c_contiguous
    wv = numpy.asarray(wv, order='C')
    aow = numpy.empty((nset,nao,ngrids))
    libdft.VXC_dscale_ao_batch(
        aow.ctypes.data_as(ctypes.c_void_p),
        aot.ctypes.data_as(ctypes.c_void_p),
        wv.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ncomp), ctypes.c_int(nset), ctypes.c_int(nao),
        ctypes.c_int(ngrids))
    return aow

def _tau_dot(mol, bra, ket, wv, mask, shls_slice, ao_loc):
    '''nabla_ao dot nabla_ao
    numpy.einsum('p,xpi,xpj->ij', wv, bra[1:4].conj(), ket[1:4])
    '''
    aow = _scale_ao(ket[1], wv)
    mat = _dot_ao_ao(mol, bra[1], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ket[2], wv, aow)
    mat += _dot_ao_ao(mol, bra[2], aow, mask, shls_slice, ao_loc)
    aow = _scale_ao(ket[3], wv, aow)
    mat += _dot_ao_ao(mol, bra[3], aow, mask, shls_slice, ao_loc)
    return mat

def _sparse_enough(screen_index, threshold=None):
    if threshold is None:
        threshold = SPARSE_THRESHOLD
    return numpy.count_nonzero(screen_index) < screen_index.size * threshold

def _sparse_reader_kw(ni, *dms):
    '''Keyword arguments of ni.block_loop for nr_rks, nr_uks and the fxc
    contractions, which read the AO blocks yielded with a mask only with the
    sparse kernels (_eval_rho_sparse, _dot_ao_dm and the *_sparse functions).
    These read the AO values where the mask is nonzero, so the others need not be
    set to zero, provided that ni evaluates AOs and densities with NumInt's own
    functions and the density matrices and orbitals are real (complex ones take
    dense paths).
    '''
    for key in ('block_loop', 'eval_ao', 'eval_rho1', 'eval_rho2',
                '_gen_rho_evaluator'):
        if inspect.getattr_static(ni, key, None) is not inspect.getattr_static(NumInt, key):
            return {}
    for dm in dms:
        if (numpy.iscomplexobj(dm) or
                numpy.iscomplexobj(getattr(dm, 'mo_coeff', None))):
            return {}
    return {'zero_screened': False}

_FUSED_XCTYPE = {'LDA': 0, 'GGA': 1, 'MGGA': 2}
_FUSED_NRHO = {'LDA': 1, 'GGA': 4, 'MGGA': 5}

class _GridAOData:
    '''What the fused XC kernels (lib/dft/nr_numint_fused.c) use of a molecule's
    grids, kept on the grids (grids._fused_ao_data) for the XC integrations that
    follow: the coordinates as [3,ngrids], and when built, the contracted radial
    parts of the AO values of every block of BLKSIZE grid points and every shell
    unscreened on it (for deriv 1, also those of their first derivatives).
    '''
    def __init__(self, mol, grids):
        self.coords = numpy.asarray(grids.coords.T, order='C')
        self.non0tab = grids.non0tab
        self.mol_data = (mol._atm.copy(), mol._bas.copy(), mol._env.copy())
        self.deriv = -1  # of the radial parts, -1 if not built
        self.radial = self.offsets = None

    def matches(self, mol, grids):
        return (grids.non0tab is self.non0tab and
                numpy.array_equal(grids.coords, self.coords.T) and
                all(numpy.array_equal(a, b) for a, b in
                    zip((mol._atm, mol._bas, mol._env), self.mol_data)))

    def group_naos(self, mol):
        '''The number of AOs of the shells unscreened on each group of GEMM_NBLK
        (4) blocks, as the sparse kernels gather them'''
        if getattr(self, '_group_naos', None) is None:
            m = self.non0tab != 0
            nblk = m.shape[0]
            ngroup = (nblk + 3) // 4
            m = numpy.vstack([m, numpy.zeros((ngroup*4-nblk, m.shape[1]), dtype=bool)])
            m = m.reshape(ngroup, 4, -1).any(axis=1)
            self._group_naos = (m * numpy.diff(mol.ao_loc_nr())).sum(axis=1).astype(float)
        return self._group_naos

    def radial_sizes(self, mol, deriv):
        nctr = mol._bas[:,NCTR_OF]
        return ((self.non0tab != 0) * ((deriv+1) * BLKSIZE * nctr)).ravel().astype(numpy.int64)

    def build_radial(self, mol, deriv):
        self.radial = self.offsets = None
        self.deriv = -1
        sizes = self.radial_sizes(mol, deriv)
        offsets = numpy.cumsum(sizes) - sizes
        radial = numpy.empty(sizes.sum())
        libdft.VXCfused_radial_fill(
            radial.ctypes.data_as(ctypes.c_void_p),
            offsets.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(deriv),
            self.coords.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(self.coords.shape[1]),
            self.non0tab.ctypes.data_as(ctypes.c_void_p),
            mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
            mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
            mol._env.ctypes.data_as(ctypes.c_void_p))
        self.radial = radial
        self.offsets = offsets.reshape(self.non0tab.shape)
        self.deriv = deriv

def _grid_ao_data(mol, grids, deriv, max_memory):
    '''The _GridAOData of grids for mol, with the radial parts for deriv if they
    fit in RADIAL_CACHE_FRACTION of max_memory (MB)'''
    data = getattr(grids, '_fused_ao_data', None)
    if data is None or not data.matches(mol, grids):
        grids._fused_ao_data = None  # releases the old arrays first
        data = grids._fused_ao_data = _GridAOData(mol, grids)
    if data.deriv < deriv:
        nbytes = data.radial_sizes(mol, deriv).sum() * 8
        nbytes_old = 0 if data.radial is None else data.radial.nbytes
        if nbytes < max_memory * 1e6 * RADIAL_CACHE_FRACTION + nbytes_old:
            data.build_radial(mol, deriv)
    return data

def _rho1_preferred(rho1_rho2_ratio, xctype):
    '''Whether make_rho of NumInt._gen_rho_evaluator evaluates the density of
    orbitals from their density matrix (eval_rho1) on screened blocks'''
    if xctype == "GGA":
        # GGA has to do more contractions using rho2 compared to rho1,
        # so the threshold for switching to rho1 is less strict.
        return rho1_rho2_ratio < 4
    else:
        return rho1_rho2_ratio < 1

class _FusedXC:
    '''The XC integrations of nr_rks, nr_uks, the fxc contractions and
    cache_xc_kernel over the blocks of NumInt.block_loop, with kernels that
    compute the AO values in place (lib/dft/nr_numint_fused.c), for all density
    matrices or potentials of a block at once. They are the sparse kernels
    _eval_rho_sparse, eval_rho2's _dot_ao_dm, _dot_ao_ao_sparse,
    _dot_ao_aow_sparse and _tau_dot_sparse without the AO arrays, and the
    results equal those of the block_loop path on the same blocks bit for bit.
    The fxc contractions take blocks of their own (_fused_fxc_blksize), and
    rks_fxc_mo the factors of CPHF-type dms instead of the dms.
    '''
    def __init__(self, mol, grids, nao, data, blocks, max_memory):
        self.mol = mol
        self.grids = grids
        self.nao = nao
        self.data = data
        self.blocks = blocks
        self.max_memory = max_memory
        self.ao_loc = numpy.asarray(mol.ao_loc_nr(), dtype=numpy.int32)

    def _args(self, ip0, ip1, deriv):
        mol, data = self.mol, self.data
        b0 = ip0 // BLKSIZE
        if data.deriv >= deriv:
            radial = data.radial.ctypes.data_as(ctypes.c_void_p)
            offsets = data.offsets[b0:].ctypes.data_as(ctypes.c_void_p)
        else:
            radial = offsets = None
        return (ctypes.c_int(self.nao), ctypes.c_int(ip1-ip0),
                ctypes.c_void_p(data.coords.ctypes.data + ip0 * 8),
                ctypes.c_int(data.coords.shape[1]),
                data.non0tab[b0:].ctypes.data_as(ctypes.c_void_p),
                self.ao_loc.ctypes.data_as(ctypes.c_void_p),
                mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                mol._env.ctypes.data_as(ctypes.c_void_p), radial, offsets)

    @staticmethod
    def _occupied(mo_coeff, mo_occ):
        '''The orbitals of eval_rho2 for each set, cpos = mo_coeff[:,occ] *
        sqrt(mo_occ[occ]), as consecutive [nao,nocc[i]] arrays, and nocc; or None
        if any occupation is negative'''
        if any(numpy.any(occ < -OCCDROP) for occ in mo_occ):
            return None
        cpos = []
        for c, occ in zip(mo_coeff, mo_occ):
            pos = occ > OCCDROP
            cpos.append(numpy.einsum('ij,j->ij', c[:,pos], numpy.sqrt(occ[pos])))
        nocc = numpy.asarray([c.shape[1] for c in cpos], dtype=numpy.int32)
        return numpy.hstack([c.ravel() for c in cpos]), nocc

    def _rho_kernel(self, ip0, ip1, xctype, cmo=None, nocc=None, dms=None):
        nderiv = _FUSED_XCTYPE[xctype]
        args = self._args(ip0, ip1, min(nderiv, 1))
        if dms is None:
            rho = numpy.empty((len(nocc), _FUSED_NRHO[xctype], ip1-ip0))
            libdft.VXCfused_eval_rho_mo(
                rho.ctypes.data_as(ctypes.c_void_p),
                cmo.ctypes.data_as(ctypes.c_void_p),
                nocc.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(nocc)),
                ctypes.c_int(nderiv), *args)
        else:
            rho = numpy.empty((len(dms), _FUSED_NRHO[xctype], ip1-ip0))
            libdft.VXCfused_eval_rho(
                rho.ctypes.data_as(ctypes.c_void_p),
                dms.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(len(dms)),
                ctypes.c_int(nderiv), *args)
        if xctype == 'LDA':
            rho = rho[:,0]
        return rho

    def rho_mo_evaluator(self, mo_coeff, mo_occ, xctype):
        '''make_rho(ip0, ip1) -> the densities of eval_rho2 for each set of
        orbitals on the block, rho[nset,nrho,ngrids] (LDA: [nset,ngrids]), or
        None if any occupation is negative'''
        occupied = self._occupied(mo_coeff, mo_occ)
        if occupied is None:
            return None
        cmo, nocc = occupied
        cmo = numpy.hstack((cmo, numpy.zeros(1)))  # not empty without occupied orbitals
        return lambda ip0, ip1: self._rho_kernel(ip0, ip1, xctype, cmo, nocc)

    def rho_evaluator(self, ni, dms_list, hermi, xctype):
        '''make_rho(ip0, ip1) -> [rho of dms for dms in dms_list] on the block,
        rho[nset,nrho,ngrids] (LDA: [nset,ngrids]), the densities that make_rho of
        ni._gen_rho_evaluator(dms) gives. The dms of all lists that take the same
        density path go through one kernel call.'''
        mo_paths = []  # (index in dms_list, cpos, nocc) taking eval_rho2's path
        dm_paths = []  # (index in dms_list, dms) taking eval_rho1's
        for k, dms in enumerate(dms_list):
            dms, hermi1, mo_coeff, mo_occ, pair_mask, rho1_rho2_ratio = \
                    _rho_setup(ni, self.mol, dms, hermi, self.grids)
            if mo_coeff is not None and not _rho1_preferred(rho1_rho2_ratio, xctype):
                occupied = self._occupied(mo_coeff, mo_occ)
                # Otherwise from the dms: eval_rho2's sums with negative occupations
                # are not implemented, the results differ by the rounding
                if occupied is not None:
                    mo_paths.append((k,) + occupied)
                    continue
            # eval_rho1 on screened blocks: _eval_rho_sparse for real dms (hermi=1)
            dm_paths.append((k, numpy.asarray(dms, order='C')))
        kernels = []  # (kernel arguments, [(index in dms_list, number of sets)])
        if mo_paths:
            # not empty without occupied orbitals
            all_cmo = numpy.hstack([p[1] for p in mo_paths] + [numpy.zeros(1)])
            all_nocc = numpy.hstack([p[2] for p in mo_paths])
            kernels.append(({'cmo': all_cmo, 'nocc': all_nocc},
                            [(p[0], len(p[2])) for p in mo_paths]))
        if dm_paths:
            all_dms = numpy.vstack([p[1] for p in dm_paths])
            kernels.append(({'dms': all_dms}, [(p[0], len(p[1])) for p in dm_paths]))

        def make_rho(ip0, ip1):
            rhos = [None] * len(dms_list)
            for kw, sets in kernels:
                rho = self._rho_kernel(ip0, ip1, xctype, **kw)
                p0 = 0
                for k, n in sets:
                    rhos[k] = rho[p0:p0+n]
                    p0 += n
            return rhos
        return make_rho

    def dot(self, ip0, ip1, wv, xctype, hermi, vmat, vtau=None):
        '''vmat[i] += the XC matrix of the potential wv[i,:,ip0:ip1], as
        _dot_ao_ao_sparse (LDA, with hermi) and _dot_ao_aow_sparse add it; for
        meta-GGA also vtau[i] += its tau part, as _tau_dot_sparse adds it'''
        wv = numpy.asarray(wv, order='C')
        assert vmat.flags.c_contiguous and vmat.dtype == numpy.double
        nset = wv.shape[0]
        nao = self.nao
        # Each thread keeps one (meta-GGA: two) [nao,nao] matrices per set
        npriv = 2 if xctype == 'MGGA' else 1
        blk = int(self.max_memory*.5e6 / (lib.num_threads()*npriv*nao**2*8))
        for i0, i1 in lib.prange(0, nset, max(1, blk)):
            libdft.VXCfused_dot_wv(
                vmat[i0:i1].ctypes.data_as(ctypes.c_void_p),
                None if vtau is None else vtau[i0:i1].ctypes.data_as(ctypes.c_void_p),
                wv[i0:i1].ctypes.data_as(ctypes.c_void_p), ctypes.c_int(i1-i0),
                ctypes.c_int(_FUSED_XCTYPE[xctype]), ctypes.c_int(hermi),
                *self._args(ip0, ip1, 0 if xctype == 'LDA' else 1))

    def dot_terms(self, ip0, ip1, wv, terms, out):
        '''out[k,o] += the sums of products of weighted AO values given by terms
        (see _xc_grad_terms) for the weights wv[k,:,ip0:ip1] on the block'''
        kets, bras, deriv = terms
        wv = numpy.asarray(wv, order='C')
        assert out.flags.c_contiguous and out.dtype == numpy.double
        nset, nwv = wv.shape[:2]
        nout, nseg = bras.shape[:2]
        # Each thread keeps nout [nao,nao] matrices per set
        blk = int(self.max_memory*.5e6 / (lib.num_threads()*nout*self.nao**2*8))
        for i0, i1 in lib.prange(0, nset, max(1, blk)):
            libdft.VXCfused_dot_terms(
                out[i0:i1].ctypes.data_as(ctypes.c_void_p),
                wv[i0:i1].ctypes.data_as(ctypes.c_void_p), ctypes.c_int(i1-i0),
                ctypes.c_int(nwv), ctypes.c_int(nout), ctypes.c_int(nseg),
                kets.ctypes.data_as(ctypes.c_void_p),
                bras.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(deriv),
                *self._args(ip0, ip1, deriv))

    def get_vxc_grad(self, ni, xc_code, xctype, dms, hermi, spin, vmat):
        '''The loop of grad.rks.get_vxc (spin=0) and grad.uks.get_vxc (spin=1,
        dms of alpha and beta); adds to vmat[nset,3,nao,nao] (spin=1: nset=2).'''
        terms = _xc_grad_terms(xctype)
        make_rho = self.rho_evaluator(ni, [dms], hermi, xctype)
        for ip0, ip1 in self.blocks:
            weight = self.grids.weights[ip0:ip1]
            rho = make_rho(ip0, ip1)[0]
            if spin == 0:
                wv = numpy.empty((len(rho), _FUSED_NRHO[xctype], ip1-ip0))
                for i in range(len(rho)):
                    wv[i] = weight * ni.eval_xc_eff(xc_code, rho[i], 1, xctype=xctype)[1]
            else:
                wv = weight * ni.eval_xc_eff(xc_code, (rho[0], rho[1]), 1, xctype=xctype)[1]
            if xctype != 'LDA':
                wv[:,0] *= .5
            if xctype == 'MGGA':
                wv[:,4] *= .5
            self.dot_terms(ip0, ip1, wv, terms, vmat)
        return vmat

    def rho_lowrank(self, ip0, ip1, A, B, xctype):
        '''rho[nset,nrho,ngrids] (LDA: [nset,ngrids]) of the symmetric density
        matrices A[k] B^T + B A[k]^T on the block, A[nset,nao,r] and B[nao,r]'''
        nderiv = _FUSED_XCTYPE[xctype]
        A = numpy.asarray(A, order='C')
        B = numpy.asarray(B, order='C')
        nset, nao, r = A.shape
        rho = numpy.empty((nset, _FUSED_NRHO[xctype], ip1-ip0))
        libdft.VXCfused_eval_rho_lowrank(
            rho.ctypes.data_as(ctypes.c_void_p), A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nset), ctypes.c_int(r),
            ctypes.c_int(nderiv), *self._args(ip0, ip1, min(nderiv, 1)))
        if xctype == 'LDA':
            rho = rho[:,0]
        return rho

    def dot_ao_mo(self, ip0, ip1, wv, xctype, Q, out):
        '''out[k] += v[k] Q, out[nset,nao,nq], for the XC matrices v[k] that
        dot() would add for the potentials wv[k] (halved as _dot_wv does)'''
        wv = numpy.asarray(wv, order='C')
        Q = numpy.asarray(Q, order='C')
        assert out.flags.c_contiguous and out.dtype == numpy.double
        libdft.VXCfused_dot_ao_mo(
            out.ctypes.data_as(ctypes.c_void_p), wv.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(len(wv)), ctypes.c_int(_FUSED_XCTYPE[xctype]),
            Q.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(Q.shape[1]),
            *self._args(ip0, ip1, 0 if xctype == 'LDA' else 1))

    def mo_chunk(self, nset, nq):
        '''The number of sets rks_fxc_mo takes per pass over the grids: their
        matrices v[k] Q, one copy per thread in dot_ao_mo, fit in half of
        max_memory'''
        n = self.max_memory*.5e6 / (8 * (lib.num_threads()+1) * self.nao * nq)
        return max(1, min(nset, int(n)))

    def lowrank_cheaper(self, xctype, r, nq, nset):
        '''Whether rks_fxc_mo (rho_lowrank, dot_ao_mo) takes fewer flops than
        rks_fxc (the dm kernels) for nset symmetric dms of rank 2*r and the
        products of their XC responses with nq orbitals'''
        ns = self.data.group_naos(self.mol)
        # flops per grid point of a group of ns AOs, per set: the dm path
        # c_dm*ns^2 (2*ns^2 per GEMM, ns^2 per component of a dsyrk: LDA the
        # density and a dsyrk, GGA two GEMMs, meta-GGA five GEMMs and a dsyrk
        # of 3), the factored path 2*ns*r per GEMM of the density (LDA 1, GGA 2,
        # meta-GGA 4) and 2*ns*nq per segment of the matrix (LDA 1, GGA 2,
        # meta-GGA 5); and once per pass of rks_fxc_mo 2*ns*(r+nq) per AO
        # component (LDA 1, GGA and meta-GGA 4) for the products with B and Q
        c_dm, c_rho, c_v, c_pass = {'LDA': (3, 2, 2, 2), 'GGA': (4, 4, 4, 8),
                                    'MGGA': (13, 8, 10, 8)}[xctype]
        npass = -(-nset // self.mo_chunk(nset, nq))
        cost_dm = nset * c_dm * (ns**2).sum()
        cost_lr = (nset * (c_rho*r + c_v*nq) + npass * c_pass*(r+nq)) * ns.sum()
        return cost_lr < .8 * cost_dm

    def rks_fxc_mo(self, ni, xctype, fxc, A, B, P, Q, out):
        '''out[k] += P^T v1[k] Q for the XC responses v1[k] that nr_rks_fxc
        computes with the kernel fxc for the symmetric dms A[k] B^T + B A[k]^T'''
        nset = len(A)
        nq = Q.shape[1]
        nrho = _FUSED_NRHO[xctype]
        ngrids = self.grids.weights.size
        ngemm = BLKSIZE * 4  # the GEMM groups (GEMM_NGRIDS) of the kernels
        nchunk = self.mo_chunk(nset, nq)
        for k0, k1 in lib.prange(0, nset, nchunk):
            n = k1 - k0
            # rho1 and wv of a block in a quarter of max_memory
            blksize = int(self.max_memory*.25e6 / (16*n*nrho*ngemm))
            blksize = max(1, min(blksize, 300)) * ngemm
            vq = numpy.zeros((n, self.nao, nq))
            for ip0, ip1 in lib.prange(0, ngrids, blksize):
                weight = self.grids.weights[ip0:ip1]
                _fxc = fxc[:,:,ip0:ip1]
                rho1 = self.rho_lowrank(ip0, ip1, A[k0:k1], B, xctype)
                wv = numpy.empty((n, nrho, ip1-ip0))
                for i in range(n):
                    if xctype == 'LDA':
                        wv[i] = weight * rho1[i] * _fxc[0]
                    else:
                        wv[i] = numpy.einsum('yg,xyg,g->xg', rho1[i], _fxc, weight)
                if xctype != 'LDA':
                    wv[:,0] *= .5
                if xctype == 'MGGA':
                    wv[:,4] *= .5
                self.dot_ao_mo(ip0, ip1, wv, xctype, Q, vq)
            out[k0:k1] += lib.einsum('pa,kpi->kai', P, vq)
        return out

    def nr_rks(self, ni, xc_code, xctype, dms, hermi, nelec, excsum, vmat):
        '''The loop of nr_rks; adds to nelec, excsum and vmat[nset,nao,nao],
        and returns the XC matrices as nr_rks does.'''
        make_rho = self.rho_evaluator(ni, [dms], hermi, xctype)
        nset = vmat.shape[0]
        vtau = numpy.zeros_like(vmat) if xctype == 'MGGA' else None
        for ip0, ip1 in self.blocks:
            weight = self.grids.weights[ip0:ip1]
            rho = make_rho(ip0, ip1)[0]
            wv = numpy.empty((nset, _FUSED_NRHO[xctype], ip1-ip0))
            for i in range(nset):
                exc, vxc = ni.eval_xc_eff(xc_code, rho[i], deriv=1, xctype=xctype, spin=0)[:2]
                if xctype == 'LDA':
                    den = rho[i] * weight
                else:
                    den = rho[i][0] * weight
                nelec[i] += den.sum()
                excsum[i] += numpy.dot(den, exc)
                wv[i] = weight * vxc
            self._dot_wv(ip0, ip1, wv, xctype, hermi, vmat, vtau)
        return self._vmat(xctype, vmat, vtau)

    def nr_uks(self, ni, xc_code, xctype, dma, dmb, hermi, nelec, excsum, vmat):
        '''The loop of nr_uks; adds to nelec, excsum and vmat[2,nset,nao,nao],
        and returns the XC matrices as nr_uks does.'''
        make_rho = self.rho_evaluator(ni, [dma, dmb], hermi, xctype)
        nset, nao = vmat.shape[1:3]
        vmat = vmat.reshape(2*nset, nao, nao)
        vtau = numpy.zeros_like(vmat) if xctype == 'MGGA' else None
        for ip0, ip1 in self.blocks:
            weight = self.grids.weights[ip0:ip1]
            rho_a, rho_b = make_rho(ip0, ip1)
            wv = numpy.empty((2, nset, _FUSED_NRHO[xctype], ip1-ip0))
            for i in range(nset):
                rho = (rho_a[i], rho_b[i])
                exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=1)[:2]
                if xctype == 'LDA':
                    den_a = rho_a[i] * weight
                    den_b = rho_b[i] * weight
                else:
                    den_a = rho_a[i][0] * weight
                    den_b = rho_b[i][0] * weight
                nelec[0,i] += den_a.sum()
                nelec[1,i] += den_b.sum()
                excsum[i] += numpy.dot(den_a, exc)
                excsum[i] += numpy.dot(den_b, exc)
                wv[:,i] = weight * vxc
            self._dot_wv(ip0, ip1, wv.reshape(2*nset, *wv.shape[2:]), xctype, hermi,
                         vmat, vtau)
        return self._vmat(xctype, vmat, vtau).reshape(2, nset, nao, nao)

    def rks_fxc(self, ni, xctype, fxc, dms, hermi, vmat):
        '''The loop of nr_rks_fxc (_rks_fxc_batch for LDA and GGA, on the blocks
        of _fused_fxc_blksize) with the kernel fxc; adds to vmat[nset,nao,nao]
        and returns the XC response matrices as nr_rks_fxc does.'''
        make_rho1 = self.rho_evaluator(ni, [dms], hermi, xctype)
        nset = vmat.shape[0]
        vtau = numpy.zeros_like(vmat) if xctype == 'MGGA' else None
        for ip0, ip1 in self.blocks:
            weight = self.grids.weights[ip0:ip1]
            _fxc = fxc[:,:,ip0:ip1]
            rho1 = make_rho1(ip0, ip1)[0]
            wv = numpy.empty((nset, _FUSED_NRHO[xctype], ip1-ip0))
            for i in range(nset):
                if xctype == 'LDA':
                    wv[i] = weight * rho1[i] * _fxc[0]
                else:
                    wv[i] = numpy.einsum('yg,xyg,g->xg', rho1[i], _fxc, weight)
            self._dot_wv(ip0, ip1, wv, xctype, hermi, vmat, vtau)
        return self._vmat(xctype, vmat, vtau)

    def uks_fxc(self, ni, xctype, fxc, dma, dmb, hermi, vmat):
        '''The loop of nr_uks_fxc with the kernel fxc; adds to
        vmat[2,nset,nao,nao] and returns the XC response matrices as nr_uks_fxc
        does.'''
        make_rho1 = self.rho_evaluator(ni, [dma, dmb], hermi, xctype)
        nset, nao = vmat.shape[1:3]
        vmat = vmat.reshape(2*nset, nao, nao)
        vtau = numpy.zeros_like(vmat) if xctype == 'MGGA' else None
        for ip0, ip1 in self.blocks:
            weight = self.grids.weights[ip0:ip1]
            _fxc = fxc[:,:,:,:,ip0:ip1]
            rho1a, rho1b = make_rho1(ip0, ip1)
            wv = numpy.empty((2, nset, _FUSED_NRHO[xctype], ip1-ip0))
            for i in range(nset):
                if xctype == 'LDA':
                    w = rho1a[i] * _fxc[0,0] + rho1b[i] * _fxc[1,0]
                else:
                    w  = numpy.einsum('xg,xbyg->byg', rho1a[i], _fxc[0])
                    w += numpy.einsum('xg,xbyg->byg', rho1b[i], _fxc[1])
                w *= weight
                wv[:,i] = w
            self._dot_wv(ip0, ip1, wv.reshape(2*nset, *wv.shape[2:]), xctype, hermi,
                         vmat, vtau)
        return self._vmat(xctype, vmat, vtau).reshape(2, nset, nao, nao)

    def _dot_wv(self, ip0, ip1, wv, xctype, hermi, vmat, vtau):
        if xctype != 'LDA':
            wv[:,0] *= .5  # *.5 because vmat + vmat.T at the end
        if xctype == 'MGGA':
            wv[:,4] *= .5  # *.5 for 1/2 in tau
        self.dot(ip0, ip1, wv, xctype, hermi, vmat, vtau)

    @staticmethod
    def _vmat(xctype, vmat, vtau):
        if xctype != 'LDA':
            vmat = lib.hermi_sum(vmat, axes=(0,2,1))
        if xctype == 'MGGA':
            vmat += vtau
        return vmat

# The second (4..9: xx xy xz yy yz zz) and third (10..19: xxx xxy xxz xyy xyz xzz yyy
# yyz yzz zzz) derivatives of eval_ao: _AO_D2[x][y] is d_x d_y, and _AO_D3[i][z] is d_z of
# the i-th second derivative
_AO_D2 = ((4, 5, 6), (5, 7, 8), (6, 8, 9))
_AO_D3 = ((10, 11, 12), (11, 13, 14), (12, 14, 15), (13, 16, 17), (14, 17, 18), (15, 18, 19))

def _pack_terms(kets, bras, deriv):
    '''The terms of VXCfused_dot_terms from lists of (AO component, weight)
    pairs (weight -1: none), with the AO derivative order they need'''
    def pack(term):
        t = numpy.zeros(9, dtype=numpy.int32)
        t[0] = len(term)
        t[1:1+2*len(term)] = numpy.ravel(term)
        return t
    return (numpy.asarray([pack(t) for t in kets]),
            numpy.asarray([[pack(t) for t in b] for b in bras]), deriv)

def _xc_grad_terms(xctype):
    '''The terms of the XC part of nuclear gradients (grad.rks.get_vxc), out[x]:
    LDA: ao[1+x]^T (wv[0] ao[0]);
    GGA: ao[1+x]^T sum_c wv[c] ao[c] + (wv[0] ao[1+x] + sum_y wv[1+y] d_y ao[1+x])^T ao[0];
    meta-GGA: also sum_k (d_k ao[1+x])^T wv[4] ao[1+k].'''
    if xctype == 'LDA':
        kets = [[(0, 0)]]
        bras = [[[(1+x, -1)]] for x in range(3)]
        return _pack_terms(kets, bras, 1)
    kets = [[(c, c) for c in range(4)], [(0, -1)]]
    bras = [[[(1+x, -1)], [(1+x, 0)] + [(_AO_D2[x][y], 1+y) for y in range(3)]]
            for x in range(3)]
    if xctype == 'MGGA':
        kets += [[(1+k, 4)] for k in range(3)]
        for x in range(3):
            bras[x] += [[(_AO_D2[k][x], -1)] for k in range(3)]
    return _pack_terms(kets, bras, 2)

def _xc_hess_diag_terms(xctype):
    '''The terms of the AO-diagonal XC part of Hessians (hessian.rks._get_vxc_diag),
    out[i] for the second derivatives i = xx, xy, xz, yy, yz, zz:
    LDA: (d_i ao)^T (wv[0] ao[0]);
    GGA: (d_i ao)^T sum_c wv[c] ao[c] + (sum_y wv[1+y] d_y d_i ao)^T ao[0];
    meta-GGA: also sum_k (d_k d_i ao)^T wv[4] ao[1+k].'''
    if xctype == 'LDA':
        kets = [[(0, 0)]]
        bras = [[[(4+i, -1)]] for i in range(6)]
        return _pack_terms(kets, bras, 2)
    kets = [[(c, c) for c in range(4)], [(0, -1)]]
    bras = [[[(4+i, -1)], [(_AO_D3[i][y], 1+y) for y in range(3)]] for i in range(6)]
    if xctype == 'MGGA':
        kets += [[(1+k, 4)] for k in range(3)]
        for i in range(6):
            bras[i] += [[(_AO_D3[i][k], -1)] for k in range(3)]
    return _pack_terms(kets, bras, 3)

def _fused_xc(ni, mol, grids, nao, xctype, max_memory, dms_list, blksize=None,
              match_block_loop=True):
    '''A _FusedXC for the blocks NumInt.block_loop would give the callers of
    _sparse_reader_kw, if the fused kernels can take them all: FUSED_XC,
    NumInt's own functions and real density matrices (_sparse_reader_kw),
    spherical GTOs, and with match_block_loop the sparse path on every block
    (the kernels give the results of the sparse ones). Otherwise None.
    '''
    if (not FUSED_XC or xctype not in _FUSED_XCTYPE or mol.cart or
            not _sparse_reader_kw(ni, *dms_list)):
        return None
    if grids.coords is None:
        grids.build(with_non0tab=True)
    screen_index = grids.non0tab
    ngrids = grids.weights.size
    if (mol is not grids.mol or screen_index is None or
            ngrids % ALIGNMENT_UNIT != 0 or nao != mol.nao_nr()):
        return None
    deriv = 0 if xctype == 'LDA' else 1
    if blksize is None:  # as block_loop
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
        blksize = int(max_memory*1e6/((comp+1)*nao*8*BLKSIZE))
        blksize = max(4, min(blksize, ngrids//BLKSIZE+1, 1200)) * BLKSIZE
    blocks = list(lib.prange(0, ngrids, blksize))
    if (match_block_loop and nao <= SWITCH_SIZE and
            not all(_sparse_enough(screen_index[ip0//BLKSIZE:]) for ip0, ip1 in blocks)):
        return None
    data = _grid_ao_data(mol, grids, deriv, max_memory)
    return _FusedXC(mol, grids, nao, data, blocks, max_memory)

def _fused_rho_mo(ni, mol, grids, nao, xctype, max_memory, mo_coeff, mo_occ):
    '''make_rho(ip0, ip1) of _FusedXC.rho_mo_evaluator for the densities of
    eval_rho2 of each set of orbitals, with make_rho.blocks the grid blocks; or
    None where the fused kernels do not apply'''
    if MGGA_DENSITY_LAPL:
        return None
    fused = _fused_xc(ni, mol, grids, nao, xctype, max_memory, mo_coeff)
    if fused is None:
        return None
    make_rho = fused.rho_mo_evaluator(mo_coeff, mo_occ, xctype)
    if make_rho is not None:
        make_rho.blocks = fused.blocks
    return make_rho

def _dot_ao_ao_dense(ao1, ao2, wv, out=None):
    '''Returns (bra*wv).T.dot(ket)
    '''
    assert ao1.flags.f_contiguous
    assert ao2.flags.f_contiguous
    assert ao1.dtype == ao2.dtype == numpy.double
    ngrids, nao = ao1.shape
    if out is None:
        out = numpy.zeros((nao, nao), dtype=ao1.dtype)

    if wv is None:
        return lib.ddot(ao1.T, ao2, 1, out, 1)
    else:
        assert wv.dtype == numpy.double
        if _same_ao(ao1, ao2) and out.flags.c_contiguous:
            return _dot_ao_wv_ao_hermi(ao1[None], wv.ravel(), out)
        ao1 = _scale_ao(ao1, wv.ravel())
        return lib.ddot(ao1.T, ao2, 1, out, 1)

def _same_ao(ao1, ao2):
    return (ao1.ctypes.data == ao2.ctypes.data and ao1.shape == ao2.shape and
            ao1.strides == ao2.strides)

def _dot_ao_wv_ao_hermi(ao, wv, out):
    '''out += einsum('xgi,g,xgj->ij', ao, wv, ao) for real ao[ncomp,ngrids,nao].
    The product is symmetric, so it is evaluated with dsyrk on one triangle.
    '''
    ncomp, ngrids, nao = ao.shape
    aot = ao.transpose(0,2,1)
    assert aot.flags.c_contiguous and out.flags.c_contiguous
    wv = numpy.asarray(wv, order='C')
    libdft.VXCdot_ao_wv_ao_hermi(
        out.ctypes.data_as(ctypes.c_void_p),
        aot.ctypes.data_as(ctypes.c_void_p),
        wv.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(ncomp), ctypes.c_int(nao), ctypes.c_int(ngrids))
    return out

def _dot_ao_ao_sparse(ao1, ao2, wv, nbins, screen_index, pair_mask, ao_loc,
                      hermi=0, out=None):
    '''Returns (bra*wv).T.dot(ket) while sparsity is explicitly considered.
    The AO values on the grid blocks where screen_index is zero are taken as
    zero. nbins and pair_mask are not used: the return equals _dot_ao_ao up to
    the order of summation.
    '''
    ngrids, nao = ao1.shape
    if screen_index is None or pair_mask is None or ngrids % ALIGNMENT_UNIT != 0:
        return _dot_ao_ao_dense(ao1, ao2, wv, out)

    assert ao1.flags.f_contiguous
    assert ao2.flags.f_contiguous
    assert ao1.dtype == ao2.dtype == numpy.double
    nbas = screen_index.shape[1]
    if out is None:
        out = numpy.zeros((nao, nao), dtype=ao1.dtype)
    if wv is None:
        wv_ptr = None
    else:
        assert wv.dtype == numpy.double
        wv = numpy.asarray(wv, order='C')
        wv_ptr = wv.ctypes.data_as(ctypes.c_void_p)

    libdft.VXCdot_aow_ao_sparse_gemm(
        out.ctypes.data_as(ctypes.c_void_p),
        ao1.ctypes.data_as(ctypes.c_void_p),
        ao2.ctypes.data_as(ctypes.c_void_p), wv_ptr,
        ctypes.c_int(nao), ctypes.c_int(ngrids),
        ctypes.c_int(nbas), ctypes.c_int(hermi),
        screen_index.ctypes.data_as(ctypes.c_void_p),
        ao_loc.ctypes.data_as(ctypes.c_void_p))
    return out

def _dot_ao_dm_sparse(ao, dm, nbins, screen_index, pair_mask, ao_loc):
    '''Returns numpy.dot(ao, dm) while sparsity is explicitly considered.
    The AO values on the grid blocks where screen_index is zero are taken as
    zero, and the return is only defined on the blocks where the AO's shell is
    unscreened. After contracting to another ao matrix,
    (numpy.dot(ao, dm)*ao).sum(axis=1), it matches _dot_ao_dm up to the order of
    summation.
    '''
    ngrids, nao = ao.shape
    if screen_index is None or pair_mask is None or ngrids % ALIGNMENT_UNIT != 0:
        return lib.dot(dm.T, ao.T).T

    assert ao.flags.f_contiguous
    assert ao.dtype == dm.dtype == numpy.double
    nbas = screen_index.shape[1]
    dm = numpy.asarray(dm, order='C')
    out = _empty_aligned((nao, ngrids)).T

    libdft.VXCdot_ao_dm_sparse_gemm(
        out.ctypes.data_as(ctypes.c_void_p),
        ao.ctypes.data_as(ctypes.c_void_p),
        dm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao), ctypes.c_int(ngrids), ctypes.c_int(nbas),
        screen_index.ctypes.data_as(ctypes.c_void_p),
        ao_loc.ctypes.data_as(ctypes.c_void_p))
    return out

def _ao_comps_contiguous(ao):
    '''Whether ao[ncomp,ngrids,nao] is stored as consecutive [nao,ngrids] blocks,
    the layout of eval_ao'''
    return (ao.ndim == 3 and ao.dtype == numpy.double and ao[0].flags.f_contiguous
            and ao.strides[0] == ao[0].nbytes)

def _eval_rho_sparse(ao, dm, screen_index, ao_loc, xctype):
    '''eval_rho1 with hermi=1 and with_lapl=False for a real dm: rho for LDA,
    rho and nabla rho for GGA, and also tau for meta-GGA. The products of
    _dot_ao_dm_sparse and _contract_rho_sparse, without storing ao.dm.
    '''
    if xctype in ('LDA', 'HF'):
        nderiv, nrho, ngrids, nao = 0, 1, *ao.shape
    elif xctype in ('GGA', 'NLC'):
        nderiv, nrho, ngrids, nao = 1, 4, *ao[0].shape
    else:
        nderiv, nrho, ngrids, nao = 2, 5, *ao[0].shape
    dm = numpy.asarray(dm, order='C')
    rho = numpy.empty((nrho, ngrids))
    libdft.VXCeval_rho_sparse_gemm(
        rho.ctypes.data_as(ctypes.c_void_p),
        ao.ctypes.data_as(ctypes.c_void_p),
        dm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nderiv), ctypes.c_int(nao), ctypes.c_int(ngrids),
        ctypes.c_int(screen_index.shape[1]),
        screen_index.ctypes.data_as(ctypes.c_void_p),
        ao_loc.ctypes.data_as(ctypes.c_void_p))
    if nderiv == 0:
        rho = rho[0]
    return rho

def _dot_ao_aow_sparse(ao, wv, nbins, screen_index, pair_mask, ao_loc, out,
                       aow=None):
    '''out += ao[0].T.dot(einsum('xgi,xg->gi', ao, wv)) for real ao[ncomp,ngrids,nao]
    and wv[ncomp,ngrids]: _scale_ao_sparse followed by _dot_ao_ao_sparse,
    without storing the scaled AOs on the sparse path. Returns the buffer of the
    scaled AOs, if one was used, for the next call.
    '''
    ngrids, nao = ao[0].shape
    if (screen_index is None or pair_mask is None or ngrids % ALIGNMENT_UNIT != 0
            or not _ao_comps_contiguous(ao) or not out.flags.c_contiguous):
        aow = _scale_ao_sparse(ao, wv, screen_index, ao_loc, out=aow)
        _dot_ao_ao_sparse(ao[0], aow, None, nbins, screen_index, pair_mask, ao_loc,
                          hermi=0, out=out)
        return aow
    assert wv.dtype == numpy.double
    wv = numpy.asarray(wv, order='C')
    libdft.VXCdot_ao_aow_sparse_gemm(
        out.ctypes.data_as(ctypes.c_void_p),
        ao.ctypes.data_as(ctypes.c_void_p),
        wv.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(wv.shape[0]),
        ctypes.c_int(nao), ctypes.c_int(ngrids),
        ctypes.c_int(screen_index.shape[1]),
        screen_index.ctypes.data_as(ctypes.c_void_p),
        ao_loc.ctypes.data_as(ctypes.c_void_p))
    return aow

def _dot_sets_aow_ao(out, ao, wv, screen_index, ao_loc):
    '''out[k] += einsum('xgi,xg,gj->ij', ao[:nw], wv[k,:nw], ao[0]) for each set k,
    nw = min(nwv, 4), and for nwv = 5 also the meta-GGA tau term
    einsum('xgi,g,xgj->ij', ao[1:4], wv[k,4], ao[1:4]), for real ao[ncomp,ngrids,nao]
    of eval_ao, wv[nset,nwv,ngrids] and out[nset,nao,nao] (C order). The AO values
    are read once for all sets, and no weighted AO array is stored.
    screen_index=None for blocks without screening.
    '''
    ngrids, nao = ao[0].shape
    nbas = len(ao_loc) - 1
    nset, nwv = wv.shape[:2]
    if screen_index is None:
        screen_index = numpy.ones(((ngrids+BLKSIZE-1)//BLKSIZE, nbas), dtype=numpy.uint8)
    assert _ao_comps_contiguous(ao) and ao.shape[0] >= (1 if nwv == 1 else 4)
    assert out.flags.c_contiguous and out.dtype == numpy.double and len(out) == nset
    wv = numpy.asarray(wv, order='C')
    ao_loc = numpy.asarray(ao_loc, dtype=numpy.int32)
    libdft.VXCdot_sets_aow_ao_sparse(
        out.ctypes.data_as(ctypes.c_void_p), ao.ctypes.data_as(ctypes.c_void_p),
        wv.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(nset), ctypes.c_int(nwv),
        ctypes.c_int(nao), ctypes.c_int(ngrids), ctypes.c_int(nbas),
        screen_index.ctypes.data_as(ctypes.c_void_p),
        ao_loc.ctypes.data_as(ctypes.c_void_p))
    return out

def _scale_ao_sparse(ao, wv, screen_index, ao_loc, out=None):
    '''Returns einsum('xgi,xg->gi', ao, wv) while sparsity is explicitly considered.
    Note the return may be different to _scale_ao. After contracting to another
    ao matrix, scale_ao.T.dot(ao), their value can be matched up to ~1e-13.
    '''
    if screen_index is None:
        return _scale_ao(ao, wv, out=out)

    assert ao.dtype == wv.dtype == numpy.double
    if ao.ndim == 3:
        assert ao[0].flags.f_contiguous
        ngrids, nao = ao[0].shape
        comp = wv.shape[0]
    else:
        assert ao.flags.f_contiguous
        ngrids, nao = ao.shape
        comp = 1
    nbas = screen_index.shape[1]
    if ngrids % ALIGNMENT_UNIT != 0:
        return _scale_ao(ao, wv, out=out)

    if out is None:
        out = _empty_aligned((nao, ngrids)).T
    else:
        out = numpy.ndarray((ngrids, nao), buffer=out, order='F')
    libdft.VXCdscale_ao_sparse(
        out.ctypes.data_as(ctypes.c_void_p),
        ao.ctypes.data_as(ctypes.c_void_p),
        wv.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(comp), ctypes.c_int(nao),
        ctypes.c_int(ngrids), ctypes.c_int(nbas),
        screen_index.ctypes.data_as(ctypes.c_void_p),
        ao_loc.ctypes.data_as(ctypes.c_void_p))
    return out

def _contract_rho_sparse(bra, ket, screen_index, ao_loc):
    '''Returns numpy.einsum('gi,gi->g', bra, ket) while sparsity is explicitly
    considered. Note the return may have ~1e-13 difference to _contract_rho.
    '''
    ngrids, nao = bra.shape
    if screen_index is None or ngrids % ALIGNMENT_UNIT != 0:
        return _contract_rho(bra, ket)

    assert bra.flags.f_contiguous
    assert ket.flags.f_contiguous
    assert bra.dtype == ket.dtype == numpy.double
    nbas = screen_index.shape[1]
    rho = numpy.empty(ngrids)
    libdft.VXCdcontract_rho_sparse(
        rho.ctypes.data_as(ctypes.c_void_p),
        bra.ctypes.data_as(ctypes.c_void_p),
        ket.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nao), ctypes.c_int(ngrids), ctypes.c_int(nbas),
        screen_index.ctypes.data_as(ctypes.c_void_p),
        ao_loc.ctypes.data_as(ctypes.c_void_p))
    return rho

def _tau_dot_sparse(bra, ket, wv, nbins, screen_index, pair_mask, ao_loc, out=None):
    '''Similar to _tau_dot, while sparsity is explicitly considered. Note the
    return may have ~1e-13 difference to _tau_dot.
    '''
    nao = bra.shape[1]
    if out is None:
        out = numpy.zeros((nao, nao), dtype=bra.dtype)
    dense = (screen_index is None or pair_mask is None or
             bra.shape[1] % ALIGNMENT_UNIT != 0)
    if _same_ao(bra, ket) and bra.dtype == numpy.double and out.flags.c_contiguous:
        # The three products below as one symmetric product
        if dense:
            return _dot_ao_wv_ao_hermi(ket[1:4], wv, out)
        if _ao_comps_contiguous(ket):
            wv = numpy.asarray(wv, order='C')
            ngrids, nao = ket[0].shape
            libdft.VXCdot_ao_wv_ao_sparse_gemm(
                out.ctypes.data_as(ctypes.c_void_p),
                ket[1:4].ctypes.data_as(ctypes.c_void_p),
                wv.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(3),
                ctypes.c_int(nao), ctypes.c_int(ngrids),
                ctypes.c_int(screen_index.shape[1]),
                screen_index.ctypes.data_as(ctypes.c_void_p),
                ao_loc.ctypes.data_as(ctypes.c_void_p))
            return out
    hermi = 1
    _dot_ao_ao_sparse(bra[1], ket[1], wv, nbins, screen_index, pair_mask,
                      ao_loc, hermi, out)
    _dot_ao_ao_sparse(bra[2], ket[2], wv, nbins, screen_index, pair_mask,
                      ao_loc, hermi, out)
    _dot_ao_ao_sparse(bra[3], ket[3], wv, nbins, screen_index, pair_mask,
                      ao_loc, hermi, out)
    return out

def nr_vxc(mol, grids, xc_code, dms, spin=0, relativity=0, hermi=0,
           max_memory=2000, verbose=None):
    '''
    Evaluate RKS/UKS XC functional and potential matrix on given meshgrids
    for a set of density matrices.  See :func:`nr_rks` and :func:`nr_uks`
    for more details.

    Args:
        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : 2D array or a list of 2D arrays
            Density matrix or multiple density matrices

    Kwargs:
        hermi : int
            Input density matrices symmetric or not
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.

    Examples:

    >>> from pyscf import gto, dft
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> grids = dft.gen_grid.Grids(mol)
    >>> grids.coords = numpy.random.random((100,3))  # 100 random points
    >>> grids.weights = numpy.random.random(100)
    >>> nao = mol.nao_nr()
    >>> dm = numpy.random.random((2,nao,nao))
    >>> nelec, exc, vxc = dft.numint.nr_vxc(mol, grids, 'lda,vwn', dm, spin=1)
    '''
    ni = NumInt()
    return ni.nr_vxc(mol, grids, xc_code, dms, spin, relativity,
                     hermi, max_memory, verbose)

def nr_sap_vxc(ni, mol, grids, max_memory=2000, verbose=None):
    '''Calculate superposition of atomic potentials matrix on given meshgrids.

    Args:
        ni : an instance of :class:`NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.

    Kwargs:
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        vmat is the XC potential matrix in 2D array of shape (nao,nao)
        where nao is the number of AO functions.

    Examples:
    >>> import numpy
    >>> from pyscf import gto, dft
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> grids = dft.gen_grid.Grids(mol)
    >>> ni = dft.numint.NumInt()
    >>> vsap = ni.nr_sap(mol, grids)
    '''
    from pyscf.dft.sap import sap_effective_charge
    assert not mol.has_ecp(), 'ECP or PP not supported'
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    nao = mol.nao
    vmat = numpy.zeros((nao,nao))
    aow = None
    ao_deriv = 0

    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()

    for ao, mask, weight, coords in ni.block_loop(mol, grids, nao, ao_deriv,
                                                  max_memory=max_memory):
        vxc = numpy.zeros(weight.size)
        # Form potential
        for ia, z in enumerate(atom_charges):
            rnuc = numpy.linalg.norm(atom_coords[ia] - coords, axis=1)
            Zeff = sap_effective_charge(z, rnuc)
            vxc -= Zeff/rnuc

        aow = _scale_ao(ao, weight*vxc, out=aow)
        vmat += _dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
        vxc = None

    return vmat

def nr_rks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    '''Calculate RKS XC functional and potential matrix on given meshgrids
    for a set of density matrices

    Args:
        ni : an instance of :class:`NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : 2D array or a list of 2D arrays
            Density matrix or multiple density matrices

    Kwargs:
        hermi : int
            Input density matrices symmetric or not. It also indicates whether
            the potential matrices in return are symmetric or not.
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.

    Examples:

    >>> from pyscf import gto, dft
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> grids = dft.gen_grid.Grids(mol)
    >>> grids.coords = numpy.random.random((100,3))  # 100 random points
    >>> grids.weights = numpy.random.random(100)
    >>> nao = mol.nao_nr()
    >>> dm = numpy.random.random((nao,nao))
    >>> ni = dft.numint.NumInt()
    >>> nelec, exc, vxc = ni.nr_rks(mol, grids, 'lda,vwn', dm)
    '''
    xctype = ni._xc_type(xc_code)
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))

    nelec = numpy.zeros(nset)
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((nset,nao,nao))

    loop_kw = _sparse_reader_kw(ni, dms)
    def block_loop(ao_deriv):
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory,
                                 **loop_kw):
            for i in range(nset):
                rho = make_rho(i, ao, mask, xctype)
                exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=0)[:2]
                if xctype == 'LDA':
                    den = rho * weight
                else:
                    den = rho[0] * weight
                nelec[i] += den.sum()
                excsum[i] += numpy.dot(den, exc)
                wv = weight * vxc
                yield i, ao, mask, wv

    aow = None
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    if xctype == 'MGGA' and any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00')):
        raise NotImplementedError('laplacian in meta-GGA method')
    fused = _fused_xc(ni, mol, grids, nao, xctype, max_memory, (dms,))
    if fused is not None:
        vmat = fused.nr_rks(ni, xc_code, xctype, dms, hermi, nelec, excsum, vmat)

    elif xctype == 'LDA':
        ao_deriv = 0
        for i, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv, nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[i])

    elif xctype == 'GGA':
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5  # *.5 because vmat + vmat.T at the end
            aow = _dot_ao_aow_sparse(ao[:4], wv[:4], nbins, mask, pair_mask,
                                     ao_loc, vmat[i], aow)
        vmat = lib.hermi_sum(vmat, axes=(0,2,1))

    elif xctype == 'MGGA':
        ao_deriv = 1
        v1 = numpy.zeros_like(vmat)
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5  # *.5 for v+v.conj().T
            wv[4] *= .5  # *.5 for 1/2 in tau
            aow = _dot_ao_aow_sparse(ao[:4], wv[:4], nbins, mask, pair_mask,
                                     ao_loc, vmat[i], aow)
            _tau_dot_sparse(ao, ao, wv[4], nbins, mask, pair_mask, ao_loc, out=v1[i])
        vmat = lib.hermi_sum(vmat, axes=(0,2,1))
        vmat += v1

    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint.nr_rks for functional {xc_code}')

    if nset == 1:
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]

    if isinstance(dms, numpy.ndarray):
        dtype = dms.dtype
    else:
        dtype = numpy.result_type(*dms)
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return nelec, excsum, vmat

def nr_uks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    '''Calculate UKS XC functional and potential matrix on given meshgrids
    for a set of density matrices

    Args:
        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dms : a list of 2D arrays
            A list of density matrices, stored as (alpha,alpha,...,beta,beta,...)

    Kwargs:
        hermi : int
            Input density matrices symmetric or not. It also indicates whether
            the potential matrices in return are symmetric or not.
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        nelec, excsum, vmat.
        nelec is the number of (alpha,beta) electrons generated by numerical integration.
        excsum is the XC functional value.
        vmat is the XC potential matrix for (alpha,beta) spin.

    Examples:

    >>> from pyscf import gto, dft
    >>> mol = gto.M(atom='H 0 0 0; H 0 0 1.1')
    >>> grids = dft.gen_grid.Grids(mol)
    >>> grids.coords = numpy.random.random((100,3))  # 100 random points
    >>> grids.weights = numpy.random.random(100)
    >>> nao = mol.nao_nr()
    >>> dm = numpy.random.random((2,nao,nao))
    >>> ni = dft.numint.NumInt()
    >>> nelec, exc, vxc = ni.nr_uks(mol, grids, 'lda,vwn', dm)
    '''
    xctype = ni._xc_type(xc_code)
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))

    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]

    nelec = numpy.zeros((2,nset))
    excsum = numpy.zeros(nset)
    vmat = numpy.zeros((2,nset,nao,nao))

    loop_kw = _sparse_reader_kw(ni, dma, dmb)
    def block_loop(ao_deriv):
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory,
                                 **loop_kw):
            for i in range(nset):
                rho_a = make_rhoa(i, ao, mask, xctype)
                rho_b = make_rhob(i, ao, mask, xctype)
                rho = (rho_a, rho_b)
                exc, vxc = ni.eval_xc_eff(xc_code, rho, deriv=1, xctype=xctype, spin=1)[:2]
                if xctype == 'LDA':
                    den_a = rho_a * weight
                    den_b = rho_b * weight
                else:
                    den_a = rho_a[0] * weight
                    den_b = rho_b[0] * weight
                nelec[0,i] += den_a.sum()
                nelec[1,i] += den_b.sum()
                excsum[i] += numpy.dot(den_a, exc)
                excsum[i] += numpy.dot(den_b, exc)
                wv = weight * vxc
                yield i, ao, mask, wv

    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    aow = None
    if xctype == 'MGGA' and any(x in xc_code.upper() for x in ('CC06', 'CS', 'BR89', 'MK00')):
        raise NotImplementedError('laplacian in meta-GGA method')
    fused = _fused_xc(ni, mol, grids, nao, xctype, max_memory, (dma, dmb))
    if fused is not None:
        vmat = fused.nr_uks(ni, xc_code, xctype, dma, dmb, hermi, nelec, excsum, vmat)

    elif xctype == 'LDA':
        ao_deriv = 0
        for i, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv[0,0], nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[0,i])
            _dot_ao_ao_sparse(ao, ao, wv[1,0], nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[1,i])

    elif xctype == 'GGA':
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[:,0] *= .5
            wva, wvb = wv
            aow = _dot_ao_aow_sparse(ao, wva, nbins, mask, pair_mask,
                                     ao_loc, vmat[0,i], aow)
            aow = _dot_ao_aow_sparse(ao, wvb, nbins, mask, pair_mask,
                                     ao_loc, vmat[1,i], aow)
        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(2,nset,nao,nao)

    elif xctype == 'MGGA':
        assert not MGGA_DENSITY_LAPL
        ao_deriv = 1
        v1 = numpy.zeros_like(vmat)
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[:,0] *= .5
            wv[:,4] *= .5
            wva, wvb = wv
            aow = _dot_ao_aow_sparse(ao[:4], wva[:4], nbins, mask, pair_mask,
                                     ao_loc, vmat[0,i], aow)
            _tau_dot_sparse(ao, ao, wva[4], nbins, mask, pair_mask, ao_loc, out=v1[0,i])
            aow = _dot_ao_aow_sparse(ao[:4], wvb[:4], nbins, mask, pair_mask,
                                     ao_loc, vmat[1,i], aow)
            _tau_dot_sparse(ao, ao, wvb[4], nbins, mask, pair_mask, ao_loc, out=v1[1,i])
        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(2,nset,nao,nao)
        vmat += v1
    elif xctype == 'HF':
        pass
    else:
        raise NotImplementedError(f'numint.nr_uks for functional {xc_code}')

    if isinstance(dma, numpy.ndarray) and dma.ndim == 2:
        vmat = vmat[:,0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]

    dtype = numpy.result_type(dma, dmb)
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return nelec, excsum, vmat

def _format_uks_dm(dms):
    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:  # RHF DM
        dma = dmb = dms * .5
    else:
        dma, dmb = dms
    if getattr(dms, 'mo_coeff', None) is not None:
        mo_coeff = dms.mo_coeff
        mo_occ = dms.mo_occ
        if mo_coeff[0].ndim < dma.ndim: # handle ROKS
            mo_occa = numpy.array(mo_occ> 0, dtype=numpy.double)
            mo_occb = numpy.array(mo_occ==2, dtype=numpy.double)
            dma = lib.tag_array(dma, mo_coeff=mo_coeff, mo_occ=mo_occa)
            dmb = lib.tag_array(dmb, mo_coeff=mo_coeff, mo_occ=mo_occb)
        else:
            dma = lib.tag_array(dma, mo_coeff=mo_coeff[0], mo_occ=mo_occ[0])
            dmb = lib.tag_array(dmb, mo_coeff=mo_coeff[1], mo_occ=mo_occ[1])
    return dma, dmb

nr_rks_vxc = nr_rks
nr_uks_vxc = nr_uks

def nr_nlc_vxc(ni, mol, grids, xc_code, dm, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
    '''Calculate NLC functional and potential matrix on given grids

    Args:
        ni : an instance of :class:`NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dm : 2D array
            Density matrix or multiple density matrices

    Kwargs:
        hermi : int
            Input density matrices symmetric or not. It also indicates whether
            the potential matrices in return are symmetric or not.
        max_memory : int or float
            The maximum size of cache to use (in MB).

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.
    '''
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi, False, grids)
    assert nset == 1
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))

    ao_deriv = 1
    vvrho = []
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
        vvrho.append(make_rho(0, ao, mask, 'GGA'))
    rho = numpy.hstack(vvrho)

    exc = 0
    vxc = 0
    nlc_coefs = ni.nlc_coeff(xc_code)
    for nlc_pars, fac in nlc_coefs:
        e, v = _vv10nlc(rho, grids.coords, rho, grids.weights,
                        grids.coords, nlc_pars)
        exc += e * fac
        vxc += v * fac
    den = rho[0] * grids.weights
    nelec = den.sum()
    excsum = numpy.dot(den, exc)
    vv_vxc = xc_deriv.transform_vxc(rho, vxc, 'GGA', spin=0)

    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    aow = None
    vmat = numpy.zeros((nao,nao))
    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
        p0, p1 = p1, p1 + weight.size
        wv = vv_vxc[:,p0:p1] * weight
        wv[0] *= .5
        aow = _dot_ao_aow_sparse(ao[:4], wv[:4], nbins, mask, pair_mask,
                                 ao_loc, vmat, aow)
    vmat = vmat + vmat.T
    return nelec, excsum, vmat

def _rks_fxc_blksize(xctype, nset, nao, ngrids, max_memory):
    '''The grid block size of _rks_fxc_batch: comp AO arrays plus ao.dm and the
    scaled AOs of every dm in each block'''
    ao_deriv = 0 if xctype == 'LDA' else 1
    comp = (ao_deriv+1)*(ao_deriv+2)*(ao_deriv+3)//6
    blksize = int(max_memory*1e6/((comp+1+2*nset)*nao*8*BLKSIZE))
    return max(4, min(blksize, ngrids//BLKSIZE+1, 1200)) * BLKSIZE

def _fused_fxc_blksize(xctype, nset, nao, ngrids, max_memory):
    '''The grid block size of _FusedXC.rks_fxc and uks_fxc for nset dms:
    block_loop's, with fewer points where rho1 and wv of the dms would take
    more than a quarter of max_memory. (The kernels hold no AO arrays. The
    blocks of _rks_fxc_batch, which shrink with nset for its AO arrays, would
    multiply the private XC matrices that the kernels sum per block.)'''
    ao_deriv = 0 if xctype == 'LDA' else 1
    comp = (ao_deriv+1)*(ao_deriv+2)*(ao_deriv+3)//6
    blksize = min(max_memory*1e6/((comp+1)*nao*8*BLKSIZE),
                  max_memory*.25e6/(16*nset*_FUSED_NRHO[xctype]*BLKSIZE))
    return max(4, min(int(blksize), ngrids//BLKSIZE+1, 1200)) * BLKSIZE

def _rks_fxc_batch(ni, mol, grids, xctype, fxc, dms, hermi, make_rho1, nbins,
                   pair_mask, ao_loc, max_memory):
    '''nr_rks_fxc for real dms and LDA/GGA kernels. On dense grid blocks the
    response densities and potentials of all dms are evaluated together: one GEMM
    gives ao.dm for every dm, and each AO block is read once for all dms by the
    batched contraction and scaling kernels. Sparse blocks use the per-dm kernels.
    '''
    dms = numpy.asarray(dms, order='C')
    nao = dms.shape[-1]
    dms = dms.reshape(-1,nao,nao)
    nset = dms.shape[0]
    if hermi != 1:
        # rho1 only depends on the symmetric part of a real dm
        dms = lib.hermi_sum(dms, axes=(0,2,1)) * .5
    if xctype == 'LDA':
        ao_deriv, ncomp = 0, 1
    else:
        ao_deriv, ncomp = 1, 4
    blksize = _rks_fxc_blksize(xctype, nset, nao, grids.weights.size, max_memory)

    vmat = numpy.zeros((nset,nao,nao))
    aow = None
    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory,
                             blksize=blksize, **_sparse_reader_kw(ni, dms)):
        p0, p1 = p1, p1 + weight.size
        _fxc = fxc[:,:,p0:p1]
        if mask is not None:
            for i in range(nset):
                rho1 = make_rho1(i, ao, mask, xctype)
                if xctype == 'LDA':
                    wv = weight * rho1 * _fxc[0]
                    _dot_ao_ao_sparse(ao, ao, wv[0], nbins, mask, pair_mask, ao_loc,
                                      hermi, vmat[i])
                else:
                    wv = numpy.einsum('yg,xyg,g->xg', rho1, _fxc, weight)
                    wv[0] *= .5  # *.5 for v+v.conj().T
                    aow = _dot_ao_aow_sparse(ao, wv, nbins, mask, pair_mask,
                                             ao_loc, vmat[i], aow)
            continue

        if xctype == 'LDA':
            ao = ao[numpy.newaxis]
        ng = weight.size
        # (ao.dm)^T = dm.ao^T for symmetric dm, in the layout of ao.T
        ao_dm = lib.dot(dms.reshape(nset*nao,nao), ao[0].T).reshape(nset,nao,ng)
        rho1 = _contract_rho_batch(ao[:ncomp], ao_dm)
        ao_dm = None
        if xctype == 'LDA':
            wv = weight * rho1[:,0] * _fxc[0,0]
            for i in range(nset):
                _dot_ao_wv_ao_hermi(ao, wv[i], vmat[i])
        else:
            rho1[:,1:4] *= 2  # *2 for |nabla mu> dm <nu| + |mu> dm <nabla nu|
            wv = numpy.einsum('kyg,xyg,g->kxg', rho1, _fxc, weight)
            wv[:,0] *= .5  # *.5 for v+v.conj().T
            aow_batch = _scale_ao_batch(ao[:4], wv)
            # vmat[k] += aow_k^T.ao0, the transpose of the per-dm ao0^T.aow_k; the
            # hermi_sum below makes them equal
            lib.dot(aow_batch.reshape(nset*nao,ng), ao[0], 1,
                    vmat.reshape(nset*nao,nao), 1)
            aow_batch = None

    if xctype == 'GGA':
        # For real orbitals, K_{ia,bj} = K_{ia,jb}. It simplifies real fxc_jb
        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        vmat = lib.hermi_sum(vmat, axes=(0,2,1))
    return vmat

def nr_rks_fxc(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    '''Contract RKS XC (singlet hessian) kernel matrix with given density matrices

    Args:
        ni : an instance of :class:`NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dm0 : 2D array
            Zeroth order density matrix
        dms : 2D array a list of 2D arrays
            First order density matrix or density matrices

    Kwargs:
        hermi : int
            First order density matrix symmetric or not. It also indicates
            whether the matrices in return are symmetric or not.
        max_memory : int or float
            The maximum size of cache to use (in MB).
        rho0 : float array
            Zero-order density (and density derivative for GGA).  Giving kwargs rho0,
            vxc and fxc to improve better performance.
        vxc : float array
            First order XC derivatives
        fxc : float array
            Second order XC derivatives

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.

    Examples:

    '''
    if isinstance(dms, numpy.ndarray):
        dtype = dms.dtype
    else:
        dtype = numpy.result_type(*dms)
    if hermi != 1 and dtype != numpy.double:
        raise NotImplementedError('complex density matrix')

    xctype = ni._xc_type(xc_code)
    if fxc is None and xctype in ('LDA', 'GGA', 'MGGA'):
        fxc = ni.cache_xc_kernel1(mol, grids, xc_code, dm0, spin=0,
                                  max_memory=max_memory)[2]

    make_rho1, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)

    loop_kw = _sparse_reader_kw(ni, dms)
    def block_loop(ao_deriv):
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory,
                                 **loop_kw):
            p0, p1 = p1, p1 + weight.size
            _fxc = fxc[:,:,p0:p1]
            for i in range(nset):
                rho1 = make_rho1(i, ao, mask, xctype)
                if xctype == 'LDA':
                    wv = weight * rho1 * _fxc[0]
                else:
                    wv = numpy.einsum('yg,xyg,g->xg', rho1, _fxc, weight)
                yield i, ao, mask, wv

    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    vmat = numpy.zeros((nset,nao,nao))
    aow = None
    fused = None
    if xctype in _FUSED_XCTYPE and dtype == numpy.double:
        if grids.coords is None:
            grids.build(with_non0tab=True)
        blksize = _fused_fxc_blksize(xctype, nset, nao, grids.weights.size, max_memory)
        fused = _fused_xc(ni, mol, grids, nao, xctype, max_memory, (dms,), blksize)
    if fused is not None:
        vmat = fused.rks_fxc(ni, xctype, fxc, dms, hermi, vmat)

    elif xctype in ('LDA', 'GGA') and dtype == numpy.double:
        vmat = _rks_fxc_batch(ni, mol, grids, xctype, fxc, dms, hermi, make_rho1,
                              nbins, pair_mask, ao_loc, max_memory)

    elif xctype == 'LDA':
        ao_deriv = 0
        for i, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv[0], nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[i])

    elif xctype == 'GGA':
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5  # *.5 for v+v.conj().T
            aow = _dot_ao_aow_sparse(ao, wv, nbins, mask, pair_mask,
                                     ao_loc, vmat[i], aow)

        # For real orbitals, K_{ia,bj} = K_{ia,jb}. It simplifies real fxc_jb
        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        vmat = lib.hermi_sum(vmat, axes=(0,2,1))

    elif xctype == 'MGGA':
        assert not MGGA_DENSITY_LAPL
        ao_deriv = 2 if MGGA_DENSITY_LAPL else 1
        v1 = numpy.zeros_like(vmat)
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[0] *= .5  # *.5 for v+v.conj().T
            wv[4] *= .5  # *.5 for 1/2 in tau
            aow = _dot_ao_aow_sparse(ao, wv[:4], nbins, mask, pair_mask,
                                     ao_loc, vmat[i], aow)
            _tau_dot_sparse(ao, ao, wv[4], nbins, mask, pair_mask, ao_loc, out=v1[i])
        vmat = lib.hermi_sum(vmat, axes=(0,2,1))
        vmat += v1

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        vmat = vmat[0]
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return vmat

def nr_rks_fxc_st(ni, mol, grids, xc_code, dm0, dms_alpha, hermi=0, singlet=True,
                  rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    '''Associated to singlet or triplet Hessian
    Note the difference to nr_rks_fxc, dms_alpha is the response density
    matrices of alpha spin, alpha+/-beta DM is applied due to singlet/triplet
    coupling

    Ref. CPL, 256, 454
    '''
    if fxc is None:
        fxc = ni.cache_xc_kernel1(mol, grids, xc_code, dm0, spin=1,
                                  max_memory=max_memory)[2]
    if singlet:
        fxc = fxc[0,:,0] + fxc[0,:,1]
    else:
        fxc = fxc[0,:,0] - fxc[0,:,1]
    return ni.nr_rks_fxc(mol, grids, xc_code, dm0, dms_alpha, hermi=hermi, fxc=fxc,
                         max_memory=max_memory)

def _rks_gga_wv0(rho, vxc, weight):
    vrho, vgamma = vxc[:2]
    ngrid = vrho.size
    wv = numpy.empty((4,ngrid))
    wv[0]  = vrho * .5  # v+v.T should be applied in the caller
    wv[1:] = 2 * vgamma * rho[1:4]
    wv[:] *= weight
    return wv

def _rks_gga_wv1(rho0, rho1, vxc, fxc, weight):
    vgamma = vxc[1]
    frho, frhogamma, fgg = fxc[:3]
    # sigma1 ~ \nabla(\rho_\alpha+\rho_\beta) dot \nabla(|b><j|) z_{bj}
    sigma1 = numpy.einsum('xi,xi->i', rho0[1:4], rho1[1:4])
    ngrid = sigma1.size
    wv = numpy.empty((4,ngrid))
    wv[0]  = frho * rho1[0]
    wv[0] += frhogamma * sigma1 * 2
    wv[1:] = (fgg * sigma1 * 4 + frhogamma * rho1[0] * 2) * rho0[1:4]
    wv[1:]+= vgamma * rho1[1:4] * 2
    wv *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1.dtype == numpy.double
    wv[0] *= .5
    return wv

def _rks_gga_wv2(rho0, rho1, fxc, kxc, weight):
    frr, frg, fgg = fxc[:3]
    frrr, frrg, frgg, fggg = kxc[:4]
    sigma1 = numpy.einsum('xi,xi->i', rho0[1:4], rho1[1:4])
    r1r1 = rho1[0]**2
    s1s1 = sigma1**2
    r1s1 = rho1[0] * sigma1
    sigma2 = numpy.einsum('xi,xi->i', rho1[1:4], rho1[1:4])
    ngrid = sigma1.size
    wv = numpy.empty((4,ngrid))
    wv[0]  = frrr * r1r1
    wv[0] += 4 * frrg * r1s1
    wv[0] += 4 * frgg * s1s1
    wv[0] += 2 * frg * sigma2
    wv[1:4]  = 2 * frrg * r1r1 * rho0[1:4]
    wv[1:4] += 8 * frgg * r1s1 * rho0[1:4]
    wv[1:4] += 4 * frg * rho1[0] * rho1[1:4]
    wv[1:4] += 4 * fgg * sigma2 * rho0[1:4]
    wv[1:4] += 8 * fgg * sigma1 * rho1[1:4]
    wv[1:4] += 8 * fggg * s1s1 * rho0[1:4]
    wv *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1.dtype == numpy.double
    wv[0]*=.5
    return wv

def _rks_mgga_wv0(rho, vxc, weight):
    vrho, vgamma, vlapl, vtau = vxc[:4]
    ngrid = vrho.size
    wv = numpy.zeros((6,ngrid))
    wv[0] = weight * vrho
    wv[1:4] = (weight * vgamma * 2) * rho[1:4]
    # *0.5 is for tau = 1/2 \nabla\phi\dot\nabla\phi
    wv[5] = weight * vtau * .5
    # *0.5 because v+v.T should be applied in the caller
    wv[0] *= .5
    wv[5] *= .5
    return wv

def _rks_mgga_wv1(rho0, rho1, vxc, fxc, weight):
    vsigma = vxc[1]
    frr, frg, fgg, fll, ftt, frl, frt, flt, fgl, fgt = fxc
    sigma1 = numpy.einsum('xi,xi->i', rho0[1:4], rho1[1:4])
    ngrids = sigma1.size
    wv = numpy.zeros((6, ngrids))
    wv[0]  = frr * rho1[0]
    wv[0] += frt * rho1[5]
    wv[0] += frg * sigma1 * 2
    wv[1:4] = (fgg * sigma1 * 4 + frg * rho1[0] * 2 + fgt * rho1[5] * 2) * rho0[1:4]
    wv[1:4]+= vsigma * rho1[1:4] * 2
    wv[5]  = ftt * rho1[5] * .5
    wv[5] += frt * rho1[0] * .5
    wv[5] += fgt * sigma1
    wv *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1.dtype == numpy.double
    wv[0] *= .5
    wv[5] *= .5
    return wv

def _rks_mgga_wv2(rho0, rho1, fxc, kxc, weight):
    frr, frg, fgg, fll, ftt, frl, frt, flt, fgl, fgt = fxc
    frrr, frrg, frgg, fggg = kxc[:4]
    frrt = kxc[5]
    frgt = kxc[7]
    frtt = kxc[10]
    fggt = kxc[12]
    fgtt = kxc[15]
    fttt = kxc[19]
    sigma1 = numpy.einsum('xi,xi->i', rho0[1:4], rho1[1:4])
    r1r1 = rho1[0]**2
    t1t1 = rho1[5]**2
    r1t1 = rho1[0] * rho1[5]
    s1s1 = sigma1**2
    r1s1 = rho1[0] * sigma1
    s1t1 = sigma1 * rho1[5]
    sigma2 = numpy.einsum('xi,xi->i', rho1[1:4], rho1[1:4])

    ngrid = sigma1.size
    wv = numpy.zeros((6, ngrid))
    wv[0]  = frrr * r1r1
    wv[0] += 4 * frrg * r1s1
    wv[0] += 4 * frgg * s1s1
    wv[0] += 2 * frg * sigma2
    wv[0] += frtt * t1t1
    wv[0] += 2 * frrt * r1t1
    wv[0] += 4 * frgt * s1t1
    wv[1:4] += 2 * frrg * r1r1 * rho0[1:4]
    wv[1:4] += 8 * frgg * r1s1 * rho0[1:4]
    wv[1:4] += 4 * fgg * sigma2 * rho0[1:4]
    wv[1:4] += 8 * fggg * s1s1 * rho0[1:4]
    wv[1:4] += 2 * fgtt * t1t1 * rho0[1:4]
    wv[1:4] += 8 * fggt * s1t1 * rho0[1:4]
    wv[1:4] += 4 * frgt * r1t1 * rho0[1:4]
    wv[1:4] += 8 * fgg * sigma1 * rho1[1:4]
    wv[1:4] += 4 * frg * rho1[0] * rho1[1:4]
    wv[1:4] += 4 * fgt * rho1[5] * rho1[1:4]
    wv[5] += fttt * t1t1 * .5
    wv[5] += frtt * r1t1
    wv[5] += frrt * r1r1 * .5
    wv[5] += fgtt * s1t1 * 2
    wv[5] += fggt * s1s1 * 2
    wv[5] += frgt * r1s1 * 2
    wv[5] += fgt * sigma2
    wv *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1.dtype == numpy.double
    wv[0] *= .5
    wv[5] *= .5
    return wv

def nr_uks_fxc(ni, mol, grids, xc_code, dm0, dms, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    '''Contract UKS XC kernel matrix with given density matrices

    Args:
        ni : an instance of :class:`NumInt`

        mol : an instance of :class:`Mole`

        grids : an instance of :class:`Grids`
            grids.coords and grids.weights are needed for coordinates and weights of meshgrids.
        xc_code : str
            XC functional description.
            See :func:`parse_xc` of pyscf/dft/libxc.py for more details.
        dm0 : (2, N, N) array
            Zeroth order density matrices
        dms : 2D array a list of 2D arrays
            First order density matrices

    Kwargs:
        hermi : int
            First order density matrix symmetric or not. It also indicates
            whether the matrices in return are symmetric or not.
        max_memory : int or float
            The maximum size of cache to use (in MB).
        rho0 : float array
            Zero-order density (and density derivative for GGA).  Giving kwargs rho0,
            vxc and fxc to improve better performance.
        vxc : float array
            First order XC derivatives
        fxc : float array
            Second order XC derivatives

    Returns:
        nelec, excsum, vmat.
        nelec is the number of electrons generated by numerical integration.
        excsum is the XC functional value.  vmat is the XC potential matrix in
        2D array of shape (nao,nao) where nao is the number of AO functions.

    Examples:

    '''
    if isinstance(dms, numpy.ndarray):
        dtype = dms.dtype
    else:
        dtype = numpy.result_type(*dms)
    if hermi != 1 and dtype != numpy.double:
        raise NotImplementedError('complex density matrix')

    xctype = ni._xc_type(xc_code)
    if fxc is None and xctype in ('LDA', 'GGA', 'MGGA'):
        fxc = ni.cache_xc_kernel1(mol, grids, xc_code, dm0, spin=1,
                                  max_memory=max_memory)[2]

    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob       = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]

    loop_kw = _sparse_reader_kw(ni, dma, dmb)
    def block_loop(ao_deriv):
        p1 = 0
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory,
                                 **loop_kw):
            p0, p1 = p1, p1 + weight.size
            _fxc = fxc[:,:,:,:,p0:p1]
            for i in range(nset):
                rho1a = make_rhoa(i, ao, mask, xctype)
                rho1b = make_rhob(i, ao, mask, xctype)
                if xctype == 'LDA':
                    wv = rho1a * _fxc[0,0] + rho1b * _fxc[1,0]
                else:
                    wv  = numpy.einsum('xg,xbyg->byg', rho1a, _fxc[0])
                    wv += numpy.einsum('xg,xbyg->byg', rho1b, _fxc[1])
                wv *= weight
                yield i, ao, mask, wv

    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
    pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
    vmat = numpy.zeros((2,nset,nao,nao))
    aow = None
    fused = None
    if xctype in _FUSED_XCTYPE:
        if grids.coords is None:
            grids.build(with_non0tab=True)
        blksize = _fused_fxc_blksize(xctype, 2*nset, nao, grids.weights.size, max_memory)
        fused = _fused_xc(ni, mol, grids, nao, xctype, max_memory, (dma, dmb), blksize)
    if fused is not None:
        vmat = fused.uks_fxc(ni, xctype, fxc, dma, dmb, hermi, vmat)

    elif xctype == 'LDA':
        ao_deriv = 0
        for i, ao, mask, wv in block_loop(ao_deriv):
            _dot_ao_ao_sparse(ao, ao, wv[0,0], nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[0,i])
            _dot_ao_ao_sparse(ao, ao, wv[1,0], nbins, mask, pair_mask, ao_loc,
                              hermi, vmat[1,i])

    elif xctype == 'GGA':
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[:,0] *= .5
            wva, wvb = wv
            aow = _dot_ao_aow_sparse(ao, wva, nbins, mask, pair_mask,
                                     ao_loc, vmat[0,i], aow)
            aow = _dot_ao_aow_sparse(ao, wvb, nbins, mask, pair_mask,
                                     ao_loc, vmat[1,i], aow)

        # For real orbitals, K_{ia,bj} = K_{ia,jb}. It simplifies real fxc_jb
        # [(\nabla mu) nu + mu (\nabla nu)] * fxc_jb = ((\nabla mu) nu f_jb) + h.c.
        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(2,nset,nao,nao)

    elif xctype == 'MGGA':
        assert not MGGA_DENSITY_LAPL
        ao_deriv = 1
        v1 = numpy.zeros_like(vmat)
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[:,0] *= .5
            wv[:,4] *= .5
            wva, wvb = wv
            aow = _dot_ao_aow_sparse(ao[:4], wva[:4], nbins, mask, pair_mask,
                                     ao_loc, vmat[0,i], aow)
            _tau_dot_sparse(ao, ao, wva[4], nbins, mask, pair_mask, ao_loc, out=v1[0,i])
            aow = _dot_ao_aow_sparse(ao[:4], wvb[:4], nbins, mask, pair_mask,
                                     ao_loc, vmat[1,i], aow)
            _tau_dot_sparse(ao, ao, wvb[4], nbins, mask, pair_mask, ao_loc, out=v1[1,i])
        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(2,nset,nao,nao)
        vmat += v1

    if isinstance(dma, numpy.ndarray) and dma.ndim == 2:
        vmat = vmat[:,0]
    if vmat.dtype != dtype:
        vmat = numpy.asarray(vmat, dtype=dtype)
    return vmat

def _uks_gga_wv0(rho, vxc, weight):
    rhoa, rhob = rho
    vrho, vsigma = vxc[:2]
    ngrids = vrho.shape[0]
    wva, wvb = numpy.empty((2, 4, ngrids))
    wva[0]  = vrho[:,0] * .5  # v+v.T should be applied in the caller
    wva[1:] = rhoa[1:4] * vsigma[:,0] * 2  # sigma_uu
    wva[1:]+= rhob[1:4] * vsigma[:,1]      # sigma_ud
    wva[:] *= weight
    wvb[0]  = vrho[:,1] * .5  # v+v.T should be applied in the caller
    wvb[1:] = rhob[1:4] * vsigma[:,2] * 2  # sigma_dd
    wvb[1:]+= rhoa[1:4] * vsigma[:,1]      # sigma_ud
    wvb[:] *= weight
    return wva, wvb

def _uks_gga_wv1(rho0, rho1, vxc, fxc, weight):
    uu, ud, dd = vxc[1].T
    u_u, u_d, d_d = fxc[0].T
    u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1].T
    uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2].T
    ngrid = uu.size

    rho0a, rho0b = rho0
    rho1a, rho1b = rho1
    a0a1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1a[1:4]) * 2
    a0b1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1b[1:4])
    b0a1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1a[1:4])
    b0b1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1b[1:4]) * 2
    ab_1 = a0b1 + b0a1

    wva, wvb = numpy.empty((2,4,ngrid))
    # alpha = alpha-alpha * alpha
    wva[0]  = u_u * rho1a[0]
    wva[0] += u_uu * a0a1
    wva[0] += u_ud * ab_1
    wva[1:] = uu * rho1a[1:4] * 2
    wva[1:]+= u_uu * rho1a[0] * rho0a[1:4] * 2
    wva[1:]+= u_ud * rho1a[0] * rho0b[1:4]
    wva[1:]+= uu_uu * a0a1 * rho0a[1:4] * 2
    wva[1:]+= uu_ud * a0a1 * rho0b[1:4]
    wva[1:]+= uu_ud * ab_1 * rho0a[1:4] * 2
    wva[1:]+= ud_ud * ab_1 * rho0b[1:4]
    # alpha = alpha-beta  * beta
    wva[0] += u_d * rho1b[0]
    wva[0] += u_dd * b0b1
    wva[1:]+= ud * rho1b[1:4]
    wva[1:]+= d_uu * rho1b[0] * rho0a[1:4] * 2
    wva[1:]+= d_ud * rho1b[0] * rho0b[1:4]
    wva[1:]+= uu_dd * b0b1 * rho0a[1:4] * 2
    wva[1:]+= ud_dd * b0b1 * rho0b[1:4]
    wva *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1a.dtype == numpy.double
    wva[0] *= .5

    # beta = beta-alpha * alpha
    wvb[0]  = u_d * rho1a[0]
    wvb[0] += d_ud * ab_1
    wvb[0] += d_uu * a0a1
    wvb[1:] = ud * rho1a[1:4]
    wvb[1:]+= u_dd * rho1a[0] * rho0b[1:4] * 2
    wvb[1:]+= u_ud * rho1a[0] * rho0a[1:4]
    wvb[1:]+= ud_dd * ab_1 * rho0b[1:4] * 2
    wvb[1:]+= ud_ud * ab_1 * rho0a[1:4]
    wvb[1:]+= uu_dd * a0a1 * rho0b[1:4] * 2
    wvb[1:]+= uu_ud * a0a1 * rho0a[1:4]
    # beta = beta-beta  * beta
    wvb[0] += d_d * rho1b[0]
    wvb[0] += d_dd * b0b1
    wvb[1:]+= dd * rho1b[1:4] * 2
    wvb[1:]+= d_dd * rho1b[0] * rho0b[1:4] * 2
    wvb[1:]+= d_ud * rho1b[0] * rho0a[1:4]
    wvb[1:]+= dd_dd * b0b1 * rho0b[1:4] * 2
    wvb[1:]+= ud_dd * b0b1 * rho0a[1:4]
    wvb *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1b.dtype == numpy.double
    wvb[0] *= .5
    return wva, wvb

def _uks_gga_wv2(rho0, rho1, fxc, kxc, weight):
    u_u, u_d, d_d = fxc[0].T
    u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1].T
    uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2].T
    u_u_u, u_u_d, u_d_d, d_d_d = kxc[0].T
    u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, \
            d_d_ud, d_d_dd = kxc[1].T
    u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, \
            d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd = kxc[2].T
    uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, \
            ud_ud_dd, ud_dd_dd, dd_dd_dd = kxc[3].T
    ngrid = u_u.size

    rho0a, rho0b = rho0
    rho1a, rho1b = rho1
    a0a1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1a[1:4]) * 2
    a0b1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1b[1:4])
    b0a1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1a[1:4])
    b0b1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1b[1:4]) * 2
    a1a1 = numpy.einsum('xi,xi->i', rho1a[1:4], rho1a[1:4]) * 2
    a1b1 = numpy.einsum('xi,xi->i', rho1a[1:4], rho1b[1:4]) * 2
    b1b1 = numpy.einsum('xi,xi->i', rho1b[1:4], rho1b[1:4]) * 2
    rara = rho1a[0] * rho1a[0]
    rarb = rho1a[0] * rho1b[0]
    rbrb = rho1b[0] * rho1b[0]
    ab_1 = a0b1 + b0a1

    wva, wvb = numpy.zeros((2, 4, ngrid))
    wva[0] += u_u_u * rho1a[0] * rho1a[0]
    wva[0] += u_u_d * rho1a[0] * rho1b[0] * 2
    wva[0] += u_d_d * rho1b[0] * rho1b[0]
    wva[0] += u_uu * a1a1
    wva[0] += u_ud * a1b1
    wva[0] += u_dd * b1b1
    wva[1:4] += u_uu * rho1a[0] * rho1a[1:4] * 4
    wva[1:4] += u_ud * rho1a[0] * rho1b[1:4] * 2
    wva[1:4] += d_uu * rho1b[0] * rho1a[1:4] * 4
    wva[1:4] += d_ud * rho1b[0] * rho1b[1:4] * 2
    wva[1:4] += uu_uu * a1a1 * rho0a[1:4] * 2
    wva[1:4] += uu_uu * a0a1 * rho1a[1:4] * 4
    wva[1:4] += uu_ud * ab_1 * rho1a[1:4] * 4
    wva[1:4] += uu_ud * a1b1 * rho0a[1:4] * 2
    wva[1:4] += uu_ud * a1a1 * rho0b[1:4]
    wva[1:4] += uu_ud * a0a1 * rho1b[1:4] * 2
    wva[1:4] += uu_dd * b1b1 * rho0a[1:4] * 2
    wva[1:4] += uu_dd * b0b1 * rho1a[1:4] * 4
    wva[1:4] += ud_ud * ab_1 * rho1b[1:4] * 2
    wva[1:4] += ud_ud * a1b1 * rho0b[1:4]
    wva[1:4] += ud_dd * b1b1 * rho0b[1:4]
    wva[1:4] += ud_dd * b0b1 * rho1b[1:4] * 2
    wva[0] += u_u_uu * rho1a[0] * a0a1 * 2
    wva[0] += u_d_uu * rho1b[0] * a0a1 * 2
    wva[0] += u_u_ud * rho1a[0] * ab_1 * 2
    wva[0] += u_d_ud * rho1b[0] * ab_1 * 2
    wva[0] += u_u_dd * rho1a[0] * b0b1 * 2
    wva[0] += u_d_dd * rho1b[0] * b0b1 * 2
    wva[1:4] += u_u_uu * rara * rho0a[1:4] * 2
    wva[1:4] += u_u_ud * rara * rho0b[1:4]
    wva[1:4] += u_d_uu * rarb * rho0a[1:4] * 4
    wva[1:4] += u_d_ud * rarb * rho0b[1:4] * 2
    wva[1:4] += d_d_uu * rbrb * rho0a[1:4] * 2
    wva[1:4] += d_d_ud * rbrb * rho0b[1:4]
    wva[1:4] += u_uu_uu * rho1a[0] * a0a1 * rho0a[1:4] * 4
    wva[1:4] += d_uu_uu * rho1b[0] * a0a1 * rho0a[1:4] * 4
    wva[1:4] += u_uu_ud * rho1a[0] * ab_1 * rho0a[1:4] * 4
    wva[1:4] += u_uu_ud * rho1a[0] * a0a1 * rho0b[1:4] * 2
    wva[1:4] += u_uu_dd * rho1a[0] * b0b1 * rho0a[1:4] * 4
    wva[1:4] += d_uu_dd * rho1b[0] * b0b1 * rho0a[1:4] * 4
    wva[1:4] += d_uu_ud * rho1b[0] * ab_1 * rho0a[1:4] * 4
    wva[1:4] += d_uu_ud * rho1b[0] * a0a1 * rho0b[1:4] * 2
    wva[1:4] += u_ud_ud * rho1a[0] * ab_1 * rho0b[1:4] * 2
    wva[1:4] += d_ud_ud * rho1b[0] * ab_1 * rho0b[1:4] * 2
    wva[1:4] += u_ud_dd * rho1a[0] * b0b1 * rho0b[1:4] * 2
    wva[1:4] += d_ud_dd * rho1b[0] * b0b1 * rho0b[1:4] * 2
    wva[0] += u_uu_uu * a0a1 * a0a1
    wva[0] += u_uu_ud * a0a1 * ab_1 * 2
    wva[0] += u_uu_dd * a0a1 * b0b1 * 2
    wva[0] += u_ud_ud * ab_1**2
    wva[0] += u_ud_dd * ab_1 * b0b1 * 2
    wva[0] += u_dd_dd * b0b1 * b0b1
    wva[1:4] += uu_uu_uu * a0a1 * a0a1 * rho0a[1:4] * 2
    wva[1:4] += uu_uu_ud * a0a1 * ab_1 * rho0a[1:4] * 4
    wva[1:4] += uu_uu_ud * a0a1 * a0a1 * rho0b[1:4]
    wva[1:4] += uu_uu_dd * a0a1 * b0b1 * rho0a[1:4] * 4
    wva[1:4] += uu_ud_ud * ab_1**2 * rho0a[1:4] * 2
    wva[1:4] += uu_ud_ud * a0a1 * ab_1 * rho0b[1:4] * 2
    wva[1:4] += uu_ud_dd * ab_1 * b0b1 * rho0a[1:4] * 4
    wva[1:4] += uu_ud_dd * a0a1 * b0b1 * rho0b[1:4] * 2
    wva[1:4] += uu_dd_dd * b0b1 * b0b1 * rho0a[1:4] * 2
    wva[1:4] += ud_ud_ud * ab_1**2 * rho0b[1:4]
    wva[1:4] += ud_ud_dd * ab_1 * b0b1 * rho0b[1:4] * 2
    wva[1:4] += ud_dd_dd * b0b1 * b0b1 * rho0b[1:4]
    wva *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1a.dtype == numpy.double
    wva[0]*=.5

    wvb[0] += d_d_d * rho1b[0] * rho1b[0]
    wvb[0] += u_d_d * rho1b[0] * rho1a[0] * 2
    wvb[0] += u_u_d * rho1a[0] * rho1a[0]
    wvb[0] += d_dd * b1b1
    wvb[0] += d_ud * a1b1
    wvb[0] += d_uu * a1a1
    wvb[1:4] += u_ud * rho1a[0] * rho1a[1:4] * 2
    wvb[1:4] += u_dd * rho1a[0] * rho1b[1:4] * 4
    wvb[1:4] += d_ud * rho1b[0] * rho1a[1:4] * 2
    wvb[1:4] += d_dd * rho1b[0] * rho1b[1:4] * 4
    wvb[1:4] += dd_dd * b0b1 * rho1b[1:4] * 4
    wvb[1:4] += ud_dd * b0b1 * rho1a[1:4] * 2
    wvb[1:4] += ud_dd * ab_1 * rho1b[1:4] * 4
    wvb[1:4] += ud_ud * ab_1 * rho1a[1:4] * 2
    wvb[1:4] += uu_dd * a0a1 * rho1b[1:4] * 4
    wvb[1:4] += uu_ud * a0a1 * rho1a[1:4] * 2
    wvb[1:4] += dd_dd * b1b1 * rho0b[1:4] * 2
    wvb[1:4] += ud_dd * a1b1 * rho0b[1:4] * 2
    wvb[1:4] += uu_dd * a1a1 * rho0b[1:4] * 2
    wvb[1:4] += ud_dd * b1b1 * rho0a[1:4]
    wvb[1:4] += ud_ud * a1b1 * rho0a[1:4]
    wvb[1:4] += uu_ud * a1a1 * rho0a[1:4]
    wvb[0] += d_d_dd * rho1b[0] * b0b1 * 2
    wvb[0] += u_d_dd * rho1a[0] * b0b1 * 2
    wvb[0] += d_d_ud * rho1b[0] * ab_1 * 2
    wvb[0] += u_d_ud * rho1a[0] * ab_1 * 2
    wvb[0] += d_d_uu * rho1b[0] * a0a1 * 2
    wvb[0] += u_d_uu * rho1a[0] * a0a1 * 2
    wvb[1:4] += u_u_ud * rara * rho0a[1:4]
    wvb[1:4] += u_u_dd * rara * rho0b[1:4] * 2
    wvb[1:4] += u_d_ud * rarb * rho0a[1:4] * 2
    wvb[1:4] += u_d_dd * rarb * rho0b[1:4] * 4
    wvb[1:4] += d_d_ud * rbrb * rho0a[1:4]
    wvb[1:4] += d_d_dd * rbrb * rho0b[1:4] * 2
    wvb[1:4] += d_dd_dd * rho1b[0] * b0b1 * rho0b[1:4] * 4
    wvb[1:4] += u_dd_dd * rho1a[0] * b0b1 * rho0b[1:4] * 4
    wvb[1:4] += d_ud_dd * rho1b[0] * ab_1 * rho0b[1:4] * 4
    wvb[1:4] += u_ud_dd * rho1a[0] * ab_1 * rho0b[1:4] * 4
    wvb[1:4] += d_uu_dd * rho1b[0] * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += u_uu_dd * rho1a[0] * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += d_ud_dd * rho1b[0] * b0b1 * rho0a[1:4] * 2
    wvb[1:4] += u_ud_dd * rho1a[0] * b0b1 * rho0a[1:4] * 2
    wvb[1:4] += d_ud_ud * rho1b[0] * ab_1 * rho0a[1:4] * 2
    wvb[1:4] += u_ud_ud * rho1a[0] * ab_1 * rho0a[1:4] * 2
    wvb[1:4] += d_uu_ud * rho1b[0] * a0a1 * rho0a[1:4] * 2
    wvb[1:4] += u_uu_ud * rho1a[0] * a0a1 * rho0a[1:4] * 2
    wvb[0] += d_dd_dd * b0b1 * b0b1
    wvb[0] += d_ud_dd * ab_1 * b0b1 * 2
    wvb[0] += d_ud_ud * ab_1**2
    wvb[0] += d_uu_dd * b0b1 * a0a1 * 2
    wvb[0] += d_uu_ud * ab_1 * a0a1 * 2
    wvb[0] += d_uu_uu * a0a1 * a0a1
    wvb[1:4] += uu_uu_ud * a0a1 * a0a1 * rho0a[1:4]
    wvb[1:4] += uu_uu_dd * a0a1 * a0a1 * rho0b[1:4] * 2
    wvb[1:4] += uu_ud_ud * ab_1 * a0a1 * rho0a[1:4] * 2
    wvb[1:4] += uu_ud_dd * b0b1 * a0a1 * rho0a[1:4] * 2
    wvb[1:4] += uu_ud_dd * ab_1 * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += uu_dd_dd * b0b1 * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += ud_ud_ud * ab_1**2 * rho0a[1:4]
    wvb[1:4] += ud_ud_dd * ab_1 * b0b1 * rho0a[1:4] * 2
    wvb[1:4] += ud_ud_dd * ab_1**2 * rho0b[1:4] * 2
    wvb[1:4] += ud_dd_dd * b0b1 * b0b1 * rho0a[1:4]
    wvb[1:4] += ud_dd_dd * ab_1 * b0b1 * rho0b[1:4] * 4
    wvb[1:4] += dd_dd_dd * b0b1 * b0b1 * rho0b[1:4] * 2
    wvb *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1b.dtype == numpy.double
    wvb[0]*=.5
    return wva, wvb

def _uks_mgga_wv0(rho, vxc, weight):
    rhoa, rhob = rho
    vrho, vsigma, vlapl, vtau = vxc
    ngrid = vrho.shape[0]
    wva, wvb = numpy.zeros((2,6,ngrid))
    wva[0] = vrho[:,0] * .5  # v+v.T should be applied in the caller
    wva[1:4] = rhoa[1:4] * vsigma[:,0] * 2  # sigma_uu
    wva[1:4]+= rhob[1:4] * vsigma[:,1]      # sigma_ud
    wva[5] = vtau[:,0] * .25
    wva *= weight
    wvb[0] = vrho[:,1] * .5  # v+v.T should be applied in the caller
    wvb[1:4] = rhob[1:4] * vsigma[:,2] * 2  # sigma_dd
    wvb[1:4]+= rhoa[1:4] * vsigma[:,1]      # sigma_ud
    wvb[5] = vtau[:,1] * .25
    wvb *= weight
    return wva, wvb

def _uks_mgga_wv1(rho0, rho1, vxc, fxc, weight):
    uu, ud, dd = vxc[1].T
    u_u, u_d, d_d = fxc[0].T
    u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1].T
    uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2].T
    ftt = fxc[4].T
    frt = fxc[6].T
    fgt = fxc[9].T
    ngrids = uu.size

    rho0a, rho0b = rho0
    rho1a, rho1b = rho1
    a0a1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1a[1:4]) * 2
    a0b1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1b[1:4])
    b0a1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1a[1:4])
    b0b1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1b[1:4]) * 2
    ab_1 = a0b1 + b0a1

    wva, wvb = numpy.zeros((2, 6, ngrids))

    # alpha = alpha-alpha * alpha
    wva[0] += u_u * rho1a[0]
    wva[0] += u_uu * a0a1
    wva[0] += u_ud * ab_1
    wva[0] += frt[0] * rho1a[5]
    wva[1:4]+= uu * rho1a[1:4] * 2
    wva[1:4]+= u_uu * rho1a[0] * rho0a[1:4] * 2
    wva[1:4]+= u_ud * rho1a[0] * rho0b[1:4]
    wva[1:4]+= uu_uu * a0a1 * rho0a[1:4] * 2
    wva[1:4]+= uu_ud * ab_1 * rho0a[1:4] * 2
    wva[1:4]+= uu_ud * a0a1 * rho0b[1:4]
    wva[1:4]+= ud_ud * ab_1 * rho0b[1:4]
    wva[1:4]+= fgt[0] * rho1a[5] * rho0a[1:4] * 2
    wva[1:4]+= fgt[2] * rho1a[5] * rho0b[1:4]
    wva[5] += ftt[0] * rho1a[5] * .5
    wva[5] += frt[0] * rho1a[0] * .5
    wva[5] += fgt[0] * a0a1 * .5
    wva[5] += fgt[2] * ab_1 * .5
    # alpha = alpha-beta  * beta
    wva[0] += u_d * rho1b[0]
    wva[0] += u_dd * b0b1
    wva[0] += frt[1] * rho1b[5]
    wva[1:4]+= ud * rho1b[1:4]
    wva[1:4]+= d_uu * rho1b[0] * rho0a[1:4] * 2
    wva[1:4]+= d_ud * rho1b[0] * rho0b[1:4]
    wva[1:4]+= uu_dd * b0b1 * rho0a[1:4] * 2
    wva[1:4]+= ud_dd * b0b1 * rho0b[1:4]
    # uu_d * rho1b[5] * rho0a[1:4]
    wva[1:4]+= fgt[1] * rho1b[5] * rho0a[1:4] * 2
    wva[1:4]+= fgt[3] * rho1b[5] * rho0b[1:4]
    wva[5] += ftt[1] * rho1b[5] * .5
    wva[5] += frt[2] * rho1b[0] * .5
    wva[5] += fgt[4] * b0b1 * .5
    wva *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1a.dtype == numpy.double
    wva[0] *= .5
    wva[5] *= .5

    # beta = beta-alpha * alpha
    wvb[0] += u_d * rho1a[0]
    wvb[0] += d_ud * ab_1
    wvb[0] += d_uu * a0a1
    wvb[0] += frt[2] * rho1a[5]
    wvb[1:4]+= ud * rho1a[1:4]
    wvb[1:4]+= u_dd * rho1a[0] * rho0b[1:4] * 2
    wvb[1:4]+= u_ud * rho1a[0] * rho0a[1:4]
    wvb[1:4]+= ud_dd * ab_1 * rho0b[1:4] * 2
    wvb[1:4]+= ud_ud * ab_1 * rho0a[1:4]
    wvb[1:4]+= uu_dd * a0a1 * rho0b[1:4] * 2
    wvb[1:4]+= uu_ud * a0a1 * rho0a[1:4]
    # dd_u * rho1a[5] * rho0b[1:4]
    wvb[1:4]+= fgt[4] * rho1a[5] * rho0b[1:4] * 2
    wvb[1:4]+= fgt[2] * rho1a[5] * rho0a[1:4]
    wvb[5] += ftt[1] * rho1a[5] * .5
    wvb[5] += frt[1] * rho1a[0] * .5
    wvb[5] += fgt[3] * ab_1 * .5
    wvb[5] += fgt[1] * a0a1 * .5
    # beta = beta-beta  * beta
    wvb[0] += d_d * rho1b[0]
    wvb[0] += d_dd * b0b1
    wvb[0] += frt[3] * rho1b[5]
    wvb[1:4]+= dd * rho1b[1:4] * 2
    wvb[1:4]+= d_dd * rho1b[0] * rho0b[1:4] * 2
    wvb[1:4]+= d_ud * rho1b[0] * rho0a[1:4]
    wvb[1:4]+= dd_dd * b0b1 * rho0b[1:4] * 2
    wvb[1:4]+= ud_dd * b0b1 * rho0a[1:4]
    wvb[1:4]+= fgt[5] * rho1b[5] * rho0b[1:4] * 2
    wvb[1:4]+= fgt[3] * rho1b[5] * rho0a[1:4]
    wvb[5] += ftt[2] * rho1b[5] * .5
    wvb[5] += frt[3] * rho1b[0] * .5
    wvb[5] += fgt[5] * b0b1 * .5
    wvb *= weight
    # Apply v+v.T in the caller, only if all quantities are real
    assert rho1b.dtype == numpy.double
    wvb[0] *= .5
    wvb[5] *= .5
    return wva, wvb

def _uks_mgga_wv2(rho0, rho1, fxc, kxc, weight):
    u_u, u_d, d_d = fxc[0].T
    u_uu, u_ud, u_dd, d_uu, d_ud, d_dd = fxc[1].T
    uu_uu, uu_ud, uu_dd, ud_ud, ud_dd, dd_dd = fxc[2].T
    u_u_u, u_u_d, u_d_d, d_d_d = kxc[0].T
    u_u_uu, u_u_ud, u_u_dd, u_d_uu, u_d_ud, u_d_dd, d_d_uu, \
            d_d_ud, d_d_dd = kxc[1].T
    u_uu_uu, u_uu_ud, u_uu_dd, u_ud_ud, u_ud_dd, u_dd_dd, d_uu_uu, d_uu_ud, \
            d_uu_dd, d_ud_ud, d_ud_dd, d_dd_dd = kxc[2].T
    uu_uu_uu, uu_uu_ud, uu_uu_dd, uu_ud_ud, uu_ud_dd, uu_dd_dd, ud_ud_ud, \
            ud_ud_dd, ud_dd_dd, dd_dd_dd = kxc[3].T
    fgt = fxc[9].T
    frrt = kxc[5].T
    frgt = kxc[7].T
    frtt = kxc[10].T
    fggt = kxc[12].T
    fgtt = kxc[15].T
    fttt = kxc[19].T
    ngrid = u_u.size

    rho0a, rho0b = rho0
    rho1a, rho1b = rho1
    a0a1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1a[1:4]) * 2
    a0b1 = numpy.einsum('xi,xi->i', rho0a[1:4], rho1b[1:4])
    b0a1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1a[1:4])
    b0b1 = numpy.einsum('xi,xi->i', rho0b[1:4], rho1b[1:4]) * 2
    a1a1 = numpy.einsum('xi,xi->i', rho1a[1:4], rho1a[1:4]) * 2
    a1b1 = numpy.einsum('xi,xi->i', rho1a[1:4], rho1b[1:4]) * 2
    b1b1 = numpy.einsum('xi,xi->i', rho1b[1:4], rho1b[1:4]) * 2
    ab_1 = a0b1 + b0a1
    rara = rho1a[0] * rho1a[0]
    rarb = rho1a[0] * rho1b[0]
    rbrb = rho1b[0] * rho1b[0]
    rata = rho1a[0] * rho1a[5]
    ratb = rho1a[0] * rho1b[5]
    rbta = rho1b[0] * rho1a[5]
    rbtb = rho1b[0] * rho1b[5]
    tata = rho1a[5] * rho1a[5]
    tatb = rho1a[5] * rho1b[5]
    tbtb = rho1b[5] * rho1b[5]

    wva, wvb = numpy.zeros((2, 6, ngrid))

    wva[0] += u_u_u * rara
    wva[0] += u_u_d * rarb * 2
    wva[0] += u_d_d * rbrb
    wva[0] += u_uu * a1a1
    wva[0] += u_ud * a1b1
    wva[0] += u_dd * b1b1
    wva[0] += u_u_uu * rho1a[0] * a0a1 * 2
    wva[0] += u_u_ud * rho1a[0] * ab_1 * 2
    wva[0] += u_u_dd * rho1a[0] * b0b1 * 2
    wva[0] += u_d_uu * rho1b[0] * a0a1 * 2
    wva[0] += u_d_ud * rho1b[0] * ab_1 * 2
    wva[0] += u_d_dd * rho1b[0] * b0b1 * 2
    wva[0] += u_uu_uu * a0a1 * a0a1
    wva[0] += u_uu_ud * a0a1 * ab_1 * 2
    wva[0] += u_uu_dd * a0a1 * b0b1 * 2
    wva[0] += u_ud_ud * ab_1**2
    wva[0] += u_ud_dd * b0b1 * ab_1 * 2
    wva[0] += u_dd_dd * b0b1 * b0b1
    wva[0] += frgt[0] * rho1a[5] * a0a1 * 2                     # u_uu_u
    wva[0] += frgt[1] * rho1b[5] * a0a1 * 2                     # u_uu_d
    wva[0] += frgt[2] * rho1a[5] * ab_1 * 2                     # u_ud_u
    wva[0] += frgt[3] * rho1b[5] * ab_1 * 2                     # u_ud_d
    wva[0] += frgt[4] * rho1a[5] * b0b1 * 2                     # u_dd_u
    wva[0] += frgt[5] * rho1b[5] * b0b1 * 2                     # u_dd_d
    wva[0] += frrt[0] * rata * 2                                # u_u_u
    wva[0] += frrt[1] * ratb * 2                                # u_u_d
    wva[0] += frrt[2] * rbta * 2                                # u_d_u
    wva[0] += frrt[3] * rbtb * 2                                # u_d_d
    wva[0] += frtt[0] * tata                                    # u_u_u
    wva[0] += frtt[1] * tatb * 2                                # u_u_d
    wva[0] += frtt[2] * tbtb                                    # u_d_d
    wva[1:4] += u_uu * rho1a[0] * rho1a[1:4] * 4
    wva[1:4] += u_ud * rho1a[0] * rho1b[1:4] * 2
    wva[1:4] += d_uu * rho1b[0] * rho1a[1:4] * 4
    wva[1:4] += d_ud * rho1b[0] * rho1b[1:4] * 2
    wva[1:4] += uu_uu * a1a1 * rho0a[1:4] * 2
    wva[1:4] += uu_uu * a0a1 * rho1a[1:4] * 4
    wva[1:4] += uu_ud * ab_1 * rho1a[1:4] * 4
    wva[1:4] += uu_ud * a1b1 * rho0a[1:4] * 2
    wva[1:4] += uu_ud * a1a1 * rho0b[1:4]
    wva[1:4] += uu_ud * a0a1 * rho1b[1:4] * 2
    wva[1:4] += uu_dd * b1b1 * rho0a[1:4] * 2
    wva[1:4] += uu_dd * b0b1 * rho1a[1:4] * 4
    wva[1:4] += ud_ud * ab_1 * rho1b[1:4] * 2
    wva[1:4] += ud_ud * a1b1 * rho0b[1:4]
    wva[1:4] += ud_dd * b1b1 * rho0b[1:4]
    wva[1:4] += ud_dd * b0b1 * rho1b[1:4] * 2
    wva[1:4] += u_u_uu * rara * rho0a[1:4] * 2
    wva[1:4] += u_u_ud * rara * rho0b[1:4]
    wva[1:4] += u_d_uu * rarb * rho0a[1:4] * 4
    wva[1:4] += u_d_ud * rarb * rho0b[1:4] * 2
    wva[1:4] += d_d_uu * rbrb * rho0a[1:4] * 2
    wva[1:4] += d_d_ud * rbrb * rho0b[1:4]
    wva[1:4] += u_uu_uu * rho1a[0] * a0a1 * rho0a[1:4] * 4
    wva[1:4] += u_uu_ud * rho1a[0] * ab_1 * rho0a[1:4] * 4
    wva[1:4] += u_uu_ud * rho1a[0] * a0a1 * rho0b[1:4] * 2
    wva[1:4] += u_uu_dd * rho1a[0] * b0b1 * rho0a[1:4] * 4
    wva[1:4] += u_ud_ud * rho1a[0] * ab_1 * rho0b[1:4] * 2
    wva[1:4] += u_ud_dd * rho1a[0] * b0b1 * rho0b[1:4] * 2
    wva[1:4] += d_uu_uu * rho1b[0] * a0a1 * rho0a[1:4] * 4
    wva[1:4] += d_uu_ud * rho1b[0] * ab_1 * rho0a[1:4] * 4
    wva[1:4] += d_uu_ud * rho1b[0] * a0a1 * rho0b[1:4] * 2
    wva[1:4] += d_uu_dd * rho1b[0] * b0b1 * rho0a[1:4] * 4
    wva[1:4] += d_ud_ud * rho1b[0] * ab_1 * rho0b[1:4] * 2
    wva[1:4] += d_ud_dd * rho1b[0] * b0b1 * rho0b[1:4] * 2
    wva[1:4] += uu_uu_uu * a0a1 * a0a1 * rho0a[1:4] * 2
    wva[1:4] += uu_uu_ud * a0a1 * ab_1 * rho0a[1:4] * 4
    wva[1:4] += uu_uu_ud * a0a1 * a0a1 * rho0b[1:4]
    wva[1:4] += uu_uu_dd * a0a1 * b0b1 * rho0a[1:4] * 4
    wva[1:4] += uu_ud_ud * ab_1**2 * rho0a[1:4] * 2
    wva[1:4] += uu_ud_ud * a0a1 * ab_1 * rho0b[1:4] * 2
    wva[1:4] += uu_ud_dd * ab_1 * b0b1 * rho0a[1:4] * 4
    wva[1:4] += uu_ud_dd * a0a1 * b0b1 * rho0b[1:4] * 2
    wva[1:4] += uu_dd_dd * b0b1 * b0b1 * rho0a[1:4] * 2
    wva[1:4] += ud_ud_ud * ab_1**2 * rho0b[1:4]
    wva[1:4] += ud_ud_dd * ab_1 * b0b1 * rho0b[1:4] * 2
    wva[1:4] += ud_dd_dd * b0b1 * b0b1 * rho0b[1:4]
    wva[1:4] += frgt[0] * rata * rho0a[1:4] * 4                 # u_uu_u
    wva[1:4] += frgt[1] * ratb * rho0a[1:4] * 4                 # u_uu_d
    wva[1:4] += frgt[2] * rata * rho0b[1:4] * 2                 # u_ud_u
    wva[1:4] += frgt[3] * ratb * rho0b[1:4] * 2                 # u_ud_d
    wva[1:4] += frgt[6] * rbta * rho0a[1:4] * 4                 # d_uu_u
    wva[1:4] += frgt[7] * rbtb * rho0a[1:4] * 4                 # d_uu_d
    wva[1:4] += frgt[8] * rbta * rho0b[1:4] * 2                 # d_ud_u
    wva[1:4] += frgt[9] * rbtb * rho0b[1:4] * 2                 # d_ud_d
    wva[1:4] += fgt[0] * rho1a[5] * rho1a[1:4] * 4              # uu_u
    wva[1:4] += fgt[1] * rho1b[5] * rho1a[1:4] * 4              # uu_d
    wva[1:4] += fgt[2] * rho1a[5] * rho1b[1:4] * 2              # ud_u
    wva[1:4] += fgt[3] * rho1b[5] * rho1b[1:4] * 2              # ud_d
    wva[1:4] += fggt[0] * rho1a[5] * a0a1 * rho0a[1:4] * 4      # uu_uu_u
    wva[1:4] += fggt[1] * rho1b[5] * a0a1 * rho0a[1:4] * 4      # uu_uu_d
    wva[1:4] += fggt[2] * rho1a[5] * a0a1 * rho0b[1:4] * 2      # uu_ud_u
    wva[1:4] += fggt[2] * rho1a[5] * ab_1 * rho0a[1:4] * 4      # uu_ud_u
    wva[1:4] += fggt[3] * rho1b[5] * a0a1 * rho0b[1:4] * 2      # uu_ud_d
    wva[1:4] += fggt[3] * rho1b[5] * ab_1 * rho0a[1:4] * 4      # uu_ud_d
    wva[1:4] += fggt[4] * rho1a[5] * b0b1 * rho0a[1:4] * 4      # uu_dd_u
    wva[1:4] += fggt[5] * rho1b[5] * b0b1 * rho0a[1:4] * 4      # uu_dd_d
    wva[1:4] += fggt[6] * rho1a[5] * ab_1 * rho0b[1:4] * 2      # ud_ud_u
    wva[1:4] += fggt[7] * rho1b[5] * ab_1 * rho0b[1:4] * 2      # ud_ud_d
    wva[1:4] += fggt[8] * rho1a[5] * b0b1 * rho0b[1:4] * 2      # ud_dd_u
    wva[1:4] += fggt[9] * rho1b[5] * b0b1 * rho0b[1:4] * 2      # ud_dd_d
    wva[1:4] += fgtt[0] * tata * rho0a[1:4] * 2                 # uu_u_u
    wva[1:4] += fgtt[1] * tatb * rho0a[1:4] * 4                 # uu_u_d
    wva[1:4] += fgtt[2] * tbtb * rho0a[1:4] * 2                 # uu_d_d
    wva[1:4] += fgtt[3] * tata * rho0b[1:4]                     # ud_u_u
    wva[1:4] += fgtt[4] * tatb * rho0b[1:4] * 2                 # ud_u_d
    wva[1:4] += fgtt[5] * tbtb * rho0b[1:4]                     # ud_d_d
    wva[5] += frgt[0 ] * rho1a[0] * a0a1                        # u_uu_u
    wva[5] += frgt[2 ] * rho1a[0] * ab_1                        # u_ud_u
    wva[5] += frgt[4 ] * rho1a[0] * b0b1                        # u_dd_u
    wva[5] += frgt[6 ] * rho1b[0] * a0a1                        # d_uu_u
    wva[5] += frgt[8 ] * rho1b[0] * ab_1                        # d_ud_u
    wva[5] += frgt[10] * rho1b[0] * b0b1                        # d_dd_u
    wva[5] += fggt[0 ] * a0a1 * a0a1 * .5                       # uu_uu_u
    wva[5] += fggt[2 ] * a0a1 * ab_1                            # uu_ud_u
    wva[5] += fggt[4 ] * a0a1 * b0b1                            # uu_dd_u
    wva[5] += fggt[6 ] * ab_1**2 * .5                           # ud_ud_u
    wva[5] += fggt[8 ] * ab_1 * b0b1                            # ud_dd_u
    wva[5] += fggt[10] * b0b1 * b0b1 * .5                       # dd_dd_u
    wva[5] += fgtt[0] * a0a1 * rho1a[5]                         # uu_u_u
    wva[5] += fgtt[1] * a0a1 * rho1b[5]                         # uu_u_d
    wva[5] += fgtt[3] * ab_1 * rho1a[5]                         # ud_u_u
    wva[5] += fgtt[4] * ab_1 * rho1b[5]                         # ud_u_d
    wva[5] += fgtt[6] * b0b1 * rho1a[5]                         # dd_u_u
    wva[5] += fgtt[7] * b0b1 * rho1b[5]                         # dd_u_d
    wva[5] += fgt[0] * a1a1 * .5                                # uu_u
    wva[5] += fgt[2] * a1b1 * .5                                # ud_u
    wva[5] += fgt[4] * b1b1 * .5                                # dd_u
    wva[5] += frrt[0] * rara * .5                               # u_u_u
    wva[5] += frrt[2] * rarb                                    # u_d_u
    wva[5] += frrt[4] * rbrb * .5                               # d_d_u
    wva[5] += frtt[0] * rata                                    # u_u_u
    wva[5] += frtt[1] * ratb                                    # u_u_d
    wva[5] += frtt[3] * rbta                                    # d_u_u
    wva[5] += frtt[4] * rbtb                                    # d_u_d
    wva[5] += fttt[0] * tata * .5                               # u_u_u
    wva[5] += fttt[1] * tatb                                    # u_u_d
    wva[5] += fttt[2] * tbtb * .5                               # u_d_d
    wva *= weight
    wva[0] *= .5
    wva[5] *= .5

    wvb[0] += u_u_d * rara
    wvb[0] += u_d_d * rarb * 2
    wvb[0] += d_d_d * rbrb
    wvb[0] += d_uu * a1a1
    wvb[0] += d_ud * a1b1
    wvb[0] += d_dd * b1b1
    wvb[0] += u_d_uu * rho1a[0] * a0a1 * 2
    wvb[0] += u_d_ud * rho1a[0] * ab_1 * 2
    wvb[0] += u_d_dd * rho1a[0] * b0b1 * 2
    wvb[0] += d_d_uu * rho1b[0] * a0a1 * 2
    wvb[0] += d_d_ud * rho1b[0] * ab_1 * 2
    wvb[0] += d_d_dd * rho1b[0] * b0b1 * 2
    wvb[0] += d_uu_uu * a0a1 * a0a1
    wvb[0] += d_uu_ud * a0a1 * ab_1 * 2
    wvb[0] += d_uu_dd * b0b1 * a0a1 * 2
    wvb[0] += d_ud_ud * ab_1**2
    wvb[0] += d_ud_dd * b0b1 * ab_1 * 2
    wvb[0] += d_dd_dd * b0b1 * b0b1
    wvb[0] += frgt[6 ] * rho1a[5] * a0a1 * 2                    # d_uu_u
    wvb[0] += frgt[7 ] * rho1b[5] * a0a1 * 2                    # d_uu_d
    wvb[0] += frgt[8 ] * rho1a[5] * ab_1 * 2                    # d_ud_u
    wvb[0] += frgt[9 ] * rho1b[5] * ab_1 * 2                    # d_ud_d
    wvb[0] += frgt[10] * rho1a[5] * b0b1 * 2                    # d_dd_u
    wvb[0] += frgt[11] * rho1b[5] * b0b1 * 2                    # d_dd_d
    wvb[0] += frrt[2] * rata * 2                                # u_d_u
    wvb[0] += frrt[3] * ratb * 2                                # u_d_d
    wvb[0] += frrt[4] * rbta * 2                                # d_d_u
    wvb[0] += frrt[5] * rbtb * 2                                # d_d_d
    wvb[0] += frtt[3] * tata                                    # d_u_u
    wvb[0] += frtt[4] * tatb * 2                                # d_u_d
    wvb[0] += frtt[5] * tbtb                                    # d_d_d
    wvb[1:4] += u_ud * rho1a[0] * rho1a[1:4] * 2
    wvb[1:4] += u_dd * rho1a[0] * rho1b[1:4] * 4
    wvb[1:4] += d_ud * rho1b[0] * rho1a[1:4] * 2
    wvb[1:4] += d_dd * rho1b[0] * rho1b[1:4] * 4
    wvb[1:4] += uu_ud * a1a1 * rho0a[1:4]
    wvb[1:4] += uu_ud * a0a1 * rho1a[1:4] * 2
    wvb[1:4] += uu_dd * a1a1 * rho0b[1:4] * 2
    wvb[1:4] += uu_dd * a0a1 * rho1b[1:4] * 4
    wvb[1:4] += ud_ud * a1b1 * rho0a[1:4]
    wvb[1:4] += ud_ud * ab_1 * rho1a[1:4] * 2
    wvb[1:4] += ud_dd * b1b1 * rho0a[1:4]
    wvb[1:4] += ud_dd * a1b1 * rho0b[1:4] * 2
    wvb[1:4] += ud_dd * b0b1 * rho1a[1:4] * 2
    wvb[1:4] += ud_dd * ab_1 * rho1b[1:4] * 4
    wvb[1:4] += dd_dd * b1b1 * rho0b[1:4] * 2
    wvb[1:4] += dd_dd * b0b1 * rho1b[1:4] * 4
    wvb[1:4] += u_u_ud * rara * rho0a[1:4]
    wvb[1:4] += u_u_dd * rara * rho0b[1:4] * 2
    wvb[1:4] += u_d_ud * rarb * rho0a[1:4] * 2
    wvb[1:4] += u_d_dd * rarb * rho0b[1:4] * 4
    wvb[1:4] += d_d_ud * rbrb * rho0a[1:4]
    wvb[1:4] += d_d_dd * rbrb * rho0b[1:4] * 2
    wvb[1:4] += u_uu_ud * rho1a[0] * a0a1 * rho0a[1:4] * 2
    wvb[1:4] += u_uu_dd * rho1a[0] * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += u_ud_ud * rho1a[0] * ab_1 * rho0a[1:4] * 2
    wvb[1:4] += u_ud_dd * rho1a[0] * b0b1 * rho0a[1:4] * 2
    wvb[1:4] += u_ud_dd * rho1a[0] * ab_1 * rho0b[1:4] * 4
    wvb[1:4] += u_dd_dd * rho1a[0] * b0b1 * rho0b[1:4] * 4
    wvb[1:4] += d_uu_ud * rho1b[0] * a0a1 * rho0a[1:4] * 2
    wvb[1:4] += d_uu_dd * rho1b[0] * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += d_ud_ud * rho1b[0] * ab_1 * rho0a[1:4] * 2
    wvb[1:4] += d_ud_dd * rho1b[0] * b0b1 * rho0a[1:4] * 2
    wvb[1:4] += d_ud_dd * rho1b[0] * ab_1 * rho0b[1:4] * 4
    wvb[1:4] += d_dd_dd * rho1b[0] * b0b1 * rho0b[1:4] * 4
    wvb[1:4] += uu_uu_ud * a0a1 * a0a1 * rho0a[1:4]
    wvb[1:4] += uu_uu_dd * a0a1 * a0a1 * rho0b[1:4] * 2
    wvb[1:4] += uu_ud_ud * ab_1 * a0a1 * rho0a[1:4] * 2
    wvb[1:4] += uu_ud_dd * b0b1 * a0a1 * rho0a[1:4] * 2
    wvb[1:4] += uu_ud_dd * ab_1 * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += uu_dd_dd * b0b1 * a0a1 * rho0b[1:4] * 4
    wvb[1:4] += ud_ud_ud * ab_1**2 * rho0a[1:4]
    wvb[1:4] += ud_ud_dd * b0b1 * ab_1 * rho0a[1:4] * 2
    wvb[1:4] += ud_ud_dd * ab_1**2 * rho0b[1:4] * 2
    wvb[1:4] += ud_dd_dd * b0b1 * b0b1 * rho0a[1:4]
    wvb[1:4] += ud_dd_dd * b0b1 * ab_1 * rho0b[1:4] * 4
    wvb[1:4] += dd_dd_dd * b0b1 * b0b1 * rho0b[1:4] * 2
    wvb[1:4] += frgt[2 ] * rata * rho0a[1:4] * 2                # u_ud_u
    wvb[1:4] += frgt[3 ] * ratb * rho0a[1:4] * 2                # u_ud_d
    wvb[1:4] += frgt[4 ] * rata * rho0b[1:4] * 4                # u_dd_u
    wvb[1:4] += frgt[5 ] * ratb * rho0b[1:4] * 4                # u_dd_d
    wvb[1:4] += frgt[8 ] * rbta * rho0a[1:4] * 2                # d_ud_u
    wvb[1:4] += frgt[9 ] * rbtb * rho0a[1:4] * 2                # d_ud_d
    wvb[1:4] += frgt[10] * rbta * rho0b[1:4] * 4                # d_dd_u
    wvb[1:4] += frgt[11] * rbtb * rho0b[1:4] * 4                # d_dd_d
    wvb[1:4] += fgt[2] * rho1a[5] * rho1a[1:4] * 2              # ud_u
    wvb[1:4] += fgt[3] * rho1b[5] * rho1a[1:4] * 2              # ud_d
    wvb[1:4] += fgt[4] * rho1a[5] * rho1b[1:4] * 4              # dd_u
    wvb[1:4] += fgt[5] * rho1b[5] * rho1b[1:4] * 4              # dd_d
    wvb[1:4] += fggt[2 ] * rho1a[5] * a0a1 * rho0a[1:4] * 2     # uu_ud_u
    wvb[1:4] += fggt[3 ] * rho1b[5] * a0a1 * rho0a[1:4] * 2     # uu_ud_d
    wvb[1:4] += fggt[4 ] * rho1a[5] * a0a1 * rho0b[1:4] * 4     # uu_dd_u
    wvb[1:4] += fggt[5 ] * rho1b[5] * a0a1 * rho0b[1:4] * 4     # uu_dd_d
    wvb[1:4] += fggt[6 ] * rho1a[5] * ab_1 * rho0a[1:4] * 2     # ud_ud_u
    wvb[1:4] += fggt[7 ] * rho1b[5] * ab_1 * rho0a[1:4] * 2     # ud_ud_d
    wvb[1:4] += fggt[8 ] * rho1a[5] * ab_1 * rho0b[1:4] * 4     # ud_dd_u
    wvb[1:4] += fggt[8 ] * rho1a[5] * b0b1 * rho0a[1:4] * 2     # ud_dd_u
    wvb[1:4] += fggt[9 ] * rho1b[5] * ab_1 * rho0b[1:4] * 4     # ud_dd_d
    wvb[1:4] += fggt[9 ] * rho1b[5] * b0b1 * rho0a[1:4] * 2     # ud_dd_d
    wvb[1:4] += fggt[10] * rho1a[5] * b0b1 * rho0b[1:4] * 4     # dd_dd_u
    wvb[1:4] += fggt[11] * rho1b[5] * b0b1 * rho0b[1:4] * 4     # dd_dd_d
    wvb[1:4] += fgtt[3] * tata * rho0a[1:4]                     # ud_u_u
    wvb[1:4] += fgtt[4] * tatb * rho0a[1:4] * 2                 # ud_u_d
    wvb[1:4] += fgtt[5] * tbtb * rho0a[1:4]                     # ud_d_d
    wvb[1:4] += fgtt[6] * tata * rho0b[1:4] * 2                 # dd_u_u
    wvb[1:4] += fgtt[7] * tatb * rho0b[1:4] * 4                 # dd_u_d
    wvb[1:4] += fgtt[8] * tbtb * rho0b[1:4] * 2                 # dd_d_d
    wvb[5] += frgt[1 ] * rho1a[0] * a0a1                        # u_uu_d
    wvb[5] += frgt[3 ] * rho1a[0] * ab_1                        # u_ud_d
    wvb[5] += frgt[5 ] * rho1a[0] * b0b1                        # u_dd_d
    wvb[5] += frgt[7 ] * rho1b[0] * a0a1                        # d_uu_d
    wvb[5] += frgt[9 ] * rho1b[0] * ab_1                        # d_ud_d
    wvb[5] += frgt[11] * rho1b[0] * b0b1                        # d_dd_d
    wvb[5] += fggt[1 ] * a0a1 * a0a1 * .5                       # uu_uu_d
    wvb[5] += fggt[3 ] * ab_1 * a0a1                            # uu_ud_d
    wvb[5] += fggt[5 ] * b0b1 * a0a1                            # uu_dd_d
    wvb[5] += fggt[7 ] * ab_1**2 * .5                           # ud_ud_d
    wvb[5] += fggt[9 ] * b0b1 * ab_1                            # ud_dd_d
    wvb[5] += fggt[11] * b0b1 * b0b1 * .5                       # dd_dd_d
    wvb[5] += fgt[1] * a1a1 * .5                                # uu_d
    wvb[5] += fgt[3] * a1b1 * .5                                # ud_d
    wvb[5] += fgt[5] * b1b1 * .5                                # dd_d
    wvb[5] += fgtt[1] * a0a1 * rho1a[5]                         # uu_u_d
    wvb[5] += fgtt[2] * a0a1 * rho1b[5]                         # uu_d_d
    wvb[5] += fgtt[4] * ab_1 * rho1a[5]                         # ud_u_d
    wvb[5] += fgtt[5] * ab_1 * rho1b[5]                         # ud_d_d
    wvb[5] += fgtt[7] * b0b1 * rho1a[5]                         # dd_u_d
    wvb[5] += fgtt[8] * b0b1 * rho1b[5]                         # dd_d_d
    wvb[5] += frrt[1] * rara * .5                               # u_u_d
    wvb[5] += frrt[3] * rarb                                    # u_d_d
    wvb[5] += frrt[5] * rbrb * .5                               # d_d_d
    wvb[5] += frtt[1] * rata                                    # u_u_d
    wvb[5] += frtt[2] * ratb                                    # u_d_d
    wvb[5] += frtt[4] * rbta                                    # d_u_d
    wvb[5] += frtt[5] * rbtb                                    # d_d_d
    wvb[5] += fttt[1] * tata * .5                               # u_u_d
    wvb[5] += fttt[2] * tatb                                    # u_d_d
    wvb[5] += fttt[3] * tbtb * .5                               # d_d_d
    wvb *= weight
    wvb[0] *= .5
    wvb[5] *= .5

    return wva, wvb

def _empty_aligned(shape, alignment=8):
    if alignment <= 1:
        return numpy.empty(shape)

    size = numpy.prod(shape)
    buf = numpy.empty(size + alignment - 1)
    align8 = alignment * 8
    offset = buf.ctypes.data % align8
    if offset != 0:
        offset = (align8 - offset) // 8
    return numpy.ndarray(size, buffer=buf[offset:offset+size]).reshape(shape)


def nr_fxc(mol, grids, xc_code, dm0, dms, spin=0, relativity=0, hermi=0,
           rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
    r'''Contract XC kernel matrix with given density matrices

    ... math::

            a_{pq} = f_{pq,rs} * x_{rs}

    '''
    ni = NumInt()
    return ni.nr_fxc(mol, grids, xc_code, dm0, dms, spin, relativity,
                     hermi, rho0, vxc, fxc, max_memory, verbose)


def cache_xc_kernel(ni, mol, grids, xc_code, mo_coeff, mo_occ, spin=0,
                    max_memory=2000):
    '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
    DFT hessian module etc.
    '''
    xctype = ni._xc_type(xc_code)
    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 2 if MGGA_DENSITY_LAPL else 1
    else:
        ao_deriv = 0
    with_lapl = MGGA_DENSITY_LAPL

    if mo_coeff[0].ndim == 1:  # RKS
        nao = mo_coeff.shape[0]
        make_rho = _fused_rho_mo(ni, mol, grids, nao, xctype, max_memory,
                                 [mo_coeff], [mo_occ])
        if make_rho is not None:
            rho = numpy.hstack([make_rho(ip0, ip1)[0] for ip0, ip1 in make_rho.blocks])
        else:
            rho = []
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
                rho.append(ni.eval_rho2(mol, ao, mo_coeff, mo_occ, mask, xctype, with_lapl))
            rho = numpy.hstack(rho)
        if spin == 1:  # RKS with nr_rks_fxc_st
            rho *= .5
            rho = numpy.repeat(rho[numpy.newaxis], 2, axis=0)
    else:  # UKS
        assert mo_coeff[0].ndim == 2
        assert spin == 1
        nao = mo_coeff[0].shape[0]
        make_rho = _fused_rho_mo(ni, mol, grids, nao, xctype, max_memory,
                                 mo_coeff, mo_occ)
        if make_rho is not None:
            rhos = [make_rho(ip0, ip1) for ip0, ip1 in make_rho.blocks]
            rho = (numpy.hstack([r[0] for r in rhos]), numpy.hstack([r[1] for r in rhos]))
        else:
            rhoa = []
            rhob = []
            for ao, mask, weight, coords \
                    in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
                rhoa.append(ni.eval_rho2(mol, ao, mo_coeff[0], mo_occ[0], mask, xctype, with_lapl))
                rhob.append(ni.eval_rho2(mol, ao, mo_coeff[1], mo_occ[1], mask, xctype, with_lapl))
            rho = (numpy.hstack(rhoa), numpy.hstack(rhob))
    vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype, spin=spin)[1:3]
    return rho, vxc, fxc

def cache_xc_kernel1(ni, mol, grids, xc_code, dm, spin=0, max_memory=2000):
    '''Compute the 0th order density, Vxc and fxc.  They can be used in TDDFT,
    DFT hessian module etc. Note dm the zeroth order density matrix must be a
    hermitian matrix.
    '''
    xctype = ni._xc_type(xc_code)
    if xctype == 'GGA':
        ao_deriv = 1
    elif xctype == 'MGGA':
        ao_deriv = 2 if MGGA_DENSITY_LAPL else 1
    else:
        ao_deriv = 0

    hermi = 1
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, hermi, False, grids)
    fused = None
    if not MGGA_DENSITY_LAPL:
        fused = _fused_xc(ni, mol, grids, nao, xctype, max_memory, (dm,))
    if fused is not None:
        make_rho = fused.rho_evaluator(ni, [dm], hermi, xctype)
        rhos = [make_rho(ip0, ip1)[0] for ip0, ip1 in fused.blocks]
        if dm[0].ndim == 1:  # RKS
            rho = numpy.hstack([r[0] for r in rhos])
            if spin == 1:  # RKS with nr_rks_fxc_st
                rho *= .5
                rho = numpy.repeat(rho[numpy.newaxis], 2, axis=0)
        else:  # UKS
            assert spin == 1
            rho = (numpy.hstack([r[0] for r in rhos]), numpy.hstack([r[1] for r in rhos]))
    elif dm[0].ndim == 1:  # RKS
        rho = []
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            rho.append(make_rho(0, ao, mask, xctype))
        rho = numpy.hstack(rho)
        if spin == 1:  # RKS with nr_rks_fxc_st
            rho *= .5
            rho = numpy.repeat(rho[numpy.newaxis], 2, axis=0)
    else:  # UKS
        assert dm[0].ndim == 2
        assert spin == 1
        rhoa = []
        rhob = []
        for ao, mask, weight, coords \
                in ni.block_loop(mol, grids, nao, ao_deriv, max_memory):
            rhoa.append(make_rho(0, ao, mask, xctype))
            rhob.append(make_rho(1, ao, mask, xctype))
        rho = (numpy.hstack(rhoa), numpy.hstack(rhob))
    vxc, fxc = ni.eval_xc_eff(xc_code, rho, deriv=2, xctype=xctype, spin=spin)[1:3]
    return rho, vxc, fxc

def get_rho(ni, mol, dm, grids, max_memory=2000):
    '''Density in real space
    '''
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, 1, False, grids)
    assert nset == 1
    rho = numpy.empty(grids.weights.size)
    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, 0, max_memory=max_memory):
        p0, p1 = p1, p1 + weight.size
        rho[p0:p1] = make_rho(0, ao, mask, 'LDA')
    return rho

def get_rho_naive(mol, dm, grids):
    # No cache, no sparsity, no reordering, just use the most naive way, to get a correct rho result
    ni = NumInt()

    if dm.ndim == 2:
        dm = dm[None, :, :]
    nset = dm.shape[0]
    assert nset in (1, 2)
    assert dm.shape == (nset, mol.nao, mol.nao)
    dm = dm.copy() # Remove all attached fields like mo_coeff

    grids_coords = grids.coords
    assert grids_coords is not None
    ngrids = grids_coords.shape[0]

    rho_tot = numpy.zeros([nset, ngrids])

    ngrids_per_batch = 4096
    for g0 in range(0, ngrids, ngrids_per_batch):
        g1 = min(g0 + ngrids_per_batch, ngrids)
        ao = ni.eval_ao(mol, grids_coords[g0:g1, :], deriv = 0)
        for i_dm in range(nset):
            rho_tot[i_dm, g0:g1] = numpy.einsum("gi,gj,ij->g", ao, ao, dm[i_dm])

    rho_tot = numpy.sum(rho_tot, axis = 0)
    return rho_tot

def get_rho_with_derivatives(ni, mol, dm, grids, xc = "r2scan", max_memory=2000, verbose=None):
    xctype = ni._xc_type(xc)
    assert xctype in ['LDA', 'GGA', 'MGGA']

    if xctype == 'LDA':
        ao_deriv = 0
    else:
        ao_deriv = 1

    if xctype == 'LDA':
        rho_dim = 1
    elif xctype == 'GGA':
        rho_dim = 4
    else:
        rho_dim = 5

    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm, 1, False, grids)

    ngrids = grids.coords.shape[0]
    rho_tot = numpy.empty([nset, rho_dim, ngrids])

    p1 = 0
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
        p0, p1 = p1, p1 + weight.size
        for i_dm in range(nset):
            rho_tot[i_dm, :, p0:p1] = make_rho(i_dm, ao, mask, xctype)

    return rho_tot

def get_rho_with_derivatives_naive(mol, dm, grids, xc = "r2scan"):
    # No cache, no sparsity, no reordering, just use the most naive way, to get a correct rho result
    ni = NumInt()
    xctype = ni._xc_type(xc)
    assert xctype in ['LDA', 'GGA', 'MGGA']

    if dm.ndim == 2:
        dm = dm[None, :, :]
    nset = dm.shape[0]
    assert nset in (1, 2)
    dm = dm.copy() # Remove all attached fields like mo_coeff

    grids_coords = grids.coords
    assert grids_coords is not None
    ngrids = grids_coords.shape[0]

    if xctype == 'LDA':
        ao_deriv = 0
    else:
        ao_deriv = 1

    if xctype == 'LDA':
        rho_dim = 1
    elif xctype == 'GGA':
        rho_dim = 4
    else:
        rho_dim = 5
    rho_tot = numpy.empty([nset, rho_dim, ngrids])

    ngrids_per_batch = 4096
    for g0 in range(0, ngrids, ngrids_per_batch):
        g1 = min(g0 + ngrids_per_batch, ngrids)
        ao = ni.eval_ao(mol, grids_coords[g0:g1, :], deriv = ao_deriv)
        for i_dm in range(nset):
            rho = ni.eval_rho(mol, ao, dm[i_dm], xctype = xctype, hermi = 1, with_lapl = False)
            rho_tot[i_dm, :, g0:g1] = rho

    return rho_tot


def _rho_setup(ni, mol, dms, hermi, grids):
    '''The density matrices and orbitals of NumInt._gen_rho_evaluator, with the
    pair mask and the ratio by which it chooses eval_rho1 or eval_rho2'''
    if getattr(dms, 'mo_coeff', None) is not None:
        #TODO: test whether dm.mo_coeff matching dm
        mo_coeff = dms.mo_coeff
        mo_occ = dms.mo_occ
        if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
            mo_coeff = [mo_coeff]
            mo_occ = [mo_occ]
    else:
        mo_coeff = mo_occ = None

    if isinstance(dms, numpy.ndarray) and dms.ndim == 2:
        dms = dms[numpy.newaxis]

    if hermi != 1 and dms[0].dtype == numpy.double:
        # (D + D.T)/2 because eval_rho computes 2*(|\nabla i> D_ij <j|) instead of
        # |\nabla i> D_ij <j| + |i> D_ij <\nabla j| for efficiency when dm is real
        dms = lib.hermi_sum(numpy.asarray(dms, order='C'), axes=(0,2,1)) * .5
        hermi = 1

    nao = dms[0].shape[0]

    pair_mask = None
    if grids is not None:
        ovlp_cond = mol.get_overlap_cond()
        if dms[0].dtype == numpy.double:
            dm_cond = [mol.condense_to_shell(dm, 'absmax') for dm in dms]
            dm_cond = numpy.max(dm_cond, axis=0)
            pair_mask = numpy.exp(-ovlp_cond) * dm_cond > ni.cutoff
        else:
            pair_mask = ovlp_cond < -numpy.log(ni.cutoff)
        pair_mask = numpy.asarray(pair_mask, dtype=numpy.uint8)

    if (mo_occ is not None) and (grids is not None):
        # eval_rho2 is more efficient unless we have a very large system
        # for which the pair_mask is significantly sparser than the
        # ratio of occupied to total molecular orbitals. So we use this ratio
        # to switch between eval_rho1 and eval_rho2.
        mo_ao_sparsity = max(0.5 * numpy.sum(mo_occ) / nao, 1e-8)
        wts = mol.ao_loc_nr()
        wts = (wts[1:] - wts[:-1]) / wts[-1]
        rho1_rho2_ratio = numpy.dot(wts, pair_mask).dot(wts) / mo_ao_sparsity
    else:
        rho1_rho2_ratio = 0.0
    return dms, hermi, mo_coeff, mo_occ, pair_mask, rho1_rho2_ratio


class LibXCMixin:
    libxc = libxc

    omega = None  # RSH parameter

####################
# Overwrite following functions to use custom XC functional

    def hybrid_coeff(self, xc_code, spin=0):
        return self.libxc.hybrid_coeff(xc_code, spin)

    def nlc_coeff(self, xc_code):
        return self.libxc.nlc_coeff(xc_code)

    def rsh_coeff(self, xc_code):
        return self.libxc.rsh_coeff(xc_code)

    @lib.with_doc(libxc.eval_xc.__doc__)
    def eval_xc(self, xc_code, rho, spin=0, relativity=0, deriv=1, omega=None,
                verbose=None):
        if omega is None: omega = self.omega
        return self.libxc.eval_xc(xc_code, rho, spin, relativity, deriv,
                                  omega, verbose)

    def eval_xc1(self, xc_code, rho, spin=0, deriv=1, omega=None):
        if omega is None: omega = self.omega
        return self.libxc.eval_xc1(xc_code, rho, spin, deriv, omega)

    def eval_xc_eff(self, xc_code, rho, deriv=1, omega=None, xctype=None,
                    verbose=None, spin=None):
        r'''Returns the derivative tensor against the density parameters

        [density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a]

        or spin-polarized density parameters

        [[density_a, (nabla_x)_a, (nabla_y)_a, (nabla_z)_a, tau_a],
         [density_b, (nabla_x)_b, (nabla_y)_b, (nabla_z)_b, tau_b]].

        It differs from the eval_xc method in the derivatives of non-local part.
        The eval_xc method returns the XC functional derivatives to sigma
        (|\nabla \rho|^2)

        Args:
            rho: 2-dimensional or 3-dimensional array
                Total density or (spin-up, spin-down) densities (and their
                derivatives if GGA or MGGA functionals) on grids

        Kwargs:
            deriv: int
                derivative orders
            omega: float
                define the exponent in the attenuated Coulomb for RSH functional
            spin : int
                spin polarized if spin > 0
        '''
        if omega is None: omega = self.omega
        if xctype is None: xctype = self._xc_type(xc_code)

        rho = numpy.asarray(rho, order='C', dtype=numpy.double)
        if xctype == 'MGGA' and rho.shape[-2] == 6:
            rho = numpy.asarray(rho[...,[0,1,2,3,5],:], order='C')

        if spin is None:
            spin_polarized = rho.ndim >= 2 and rho.shape[0] == 2
            if spin_polarized:
                spin = 1
            else:
                spin = 0

        out = self.eval_xc1(xc_code, rho, spin, deriv, omega)
        evfk = [out[0]]
        for order in range(1, deriv+1):
            evfk.append(xc_deriv.transform_xc(rho, out, xctype, spin, order))
        if deriv < 3:
            # Returns at least [e, v, f, k] terms
            evfk.extend([None] * (3 - deriv))
        return evfk

    def _xc_type(self, xc_code):
        return self.libxc.xc_type(xc_code)

    def rsh_and_hybrid_coeff(self, xc_code, spin=0):
        '''Range-separated parameter and HF exchange components: omega, alpha, beta

        Exc_RSH = c_SR * SR_HFX + c_LR * LR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec
                = alpha * HFX + beta * SR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec
                = alpha * LR_HFX + hyb * SR_HFX + (1-c_SR) * Ex_SR + (1-c_LR) * Ex_LR + Ec

        SR_HFX = < pi | (1-erf(-omega r_{12}))/r_{12} | iq >
        LR_HFX = < pi | erf(-omega r_{12})/r_{12} | iq >
        alpha = c_LR
        beta = c_SR - c_LR
        '''
        omega, alpha, beta = self.rsh_coeff(xc_code)
        if self.omega is not None:
            if omega == 0 and self.omega != 0:
                raise RuntimeError(f'Not support assigning omega={self.omega}. '
                                   f'{xc_code} is not a RSH functional')
            omega = self.omega

        if omega != 0:
            hyb = alpha + beta
        else:
            hyb = self.hybrid_coeff(xc_code, spin)
        return omega, alpha, hyb

# Export the symbol _NumIntMixin for backward compatibility.
# _NumIntMixin should be dropped in the future.
_NumIntMixin = LibXCMixin


class NumInt(lib.StreamObject, LibXCMixin):
    '''Numerical integration methods for non-relativistic RKS and UKS

    Input Attributes:
        omega :
            The Coulomb attenuation parameter for range-separated functionals.
            If specified, this value will replace the default setting in libxc
            when evaluating the libxc RSH functional.
    '''

    cutoff = CUTOFF * 1e2  # cutoff for small AO product

    @lib.with_doc(nr_vxc.__doc__)
    def nr_vxc(self, mol, grids, xc_code, dms, spin=0, relativity=0, hermi=0,
               max_memory=2000, verbose=None):
        if spin == 0:
            return self.nr_rks(mol, grids, xc_code, dms, relativity, hermi,
                               max_memory, verbose)
        else:
            return self.nr_uks(mol, grids, xc_code, dms, relativity, hermi,
                               max_memory, verbose)
    get_vxc = nr_vxc

    @lib.with_doc(nr_fxc.__doc__)
    def nr_fxc(self, mol, grids, xc_code, dm0, dms, spin=0, relativity=0, hermi=0,
               rho0=None, vxc=None, fxc=None, max_memory=2000, verbose=None):
        if spin == 0:
            return self.nr_rks_fxc(mol, grids, xc_code, dm0, dms, relativity,
                                   hermi, rho0, vxc, fxc, max_memory, verbose)
        else:
            return self.nr_uks_fxc(mol, grids, xc_code, dm0, dms, relativity,
                                   hermi, rho0, vxc, fxc, max_memory, verbose)
    get_fxc = nr_fxc

    nr_rks = nr_rks
    nr_uks = nr_uks
    nr_nlc_vxc = nr_nlc_vxc
    nr_sap = nr_sap_vxc = nr_sap_vxc
    nr_rks_fxc = nr_rks_fxc
    nr_uks_fxc = nr_uks_fxc
    nr_rks_fxc_st = nr_rks_fxc_st
    cache_xc_kernel  = cache_xc_kernel
    cache_xc_kernel1 = cache_xc_kernel1

    make_mask = staticmethod(make_mask)
    eval_ao = staticmethod(eval_ao)
    eval_rho  = staticmethod(eval_rho)
    eval_rho1 = lib.module_method(eval_rho1, absences=['cutoff'])
    eval_rho2 = staticmethod(eval_rho2)
    get_rho = get_rho
    get_rho_with_derivatives = get_rho_with_derivatives

    def block_loop(self, mol, grids, nao=None, deriv=0, max_memory=2000,
                   non0tab=None, blksize=None, buf=None, zero_screened=True):
        '''Define this macro to loop over grids by blocks.

        With zero_screened=False, the AO values of the shells that the mask
        screens are left unwritten on the blocks yielded with a mask, which is
        for callers that read those blocks only with the sparse kernels (see
        _sparse_reader_kw).
        '''
        if grids.coords is None:
            grids.build(with_non0tab=True)
        if nao is None:
            nao = mol.nao
        ngrids = grids.coords.shape[0]
        comp = (deriv+1)*(deriv+2)*(deriv+3)//6
        # NOTE to index grids.non0tab, the blksize needs to be an integer
        # multiplier of BLKSIZE
        if blksize is None:
            blksize = int(max_memory*1e6/((comp+1)*nao*8*BLKSIZE))
            blksize = max(4, min(blksize, ngrids//BLKSIZE+1, 1200)) * BLKSIZE
        assert blksize % BLKSIZE == 0

        if non0tab is None and mol is grids.mol:
            non0tab = grids.non0tab
        if non0tab is None:
            non0tab = numpy.empty(((ngrids+BLKSIZE-1)//BLKSIZE,mol.nbas),
                                  dtype=numpy.uint8)
            non0tab[:] = NBINS + 1  # Corresponding to AO value ~= 1
        screen_index = non0tab

        # the xxx_sparse() functions require ngrids 8-byte aligned
        allow_sparse = ngrids % ALIGNMENT_UNIT == 0 and nao > SWITCH_SIZE

        if buf is None:
            buf = _empty_aligned(comp * blksize * nao)
        for ip0, ip1 in lib.prange(0, ngrids, blksize):
            coords = grids.coords[ip0:ip1]
            weight = grids.weights[ip0:ip1]
            mask = screen_index[ip0//BLKSIZE:]
            sparse = allow_sparse or _sparse_enough(mask)
            # TODO: pass grids.cutoff to eval_ao
            if sparse and not zero_screened and (ip1 - ip0) % ALIGNMENT_UNIT == 0:
                # Unaligned blocks may fall back to the dense kernels
                ao = self.eval_ao(mol, coords, deriv=deriv, non0tab=mask,
                                  cutoff=grids.cutoff, out=buf, zero_screened=False)
            else:
                ao = self.eval_ao(mol, coords, deriv=deriv, non0tab=mask,
                                  cutoff=grids.cutoff, out=buf)
            if not sparse:
                # Unset mask for dense AO tensor. It determines which eval_rho
                # to be called in make_rho
                mask = None
            yield ao, mask, weight, coords

    def _gen_rho_evaluator(self, mol, dms, hermi=0, with_lapl=True, grids=None):
        dms, hermi, mo_coeff, mo_occ, pair_mask, rho1_rho2_ratio = \
                _rho_setup(self, mol, dms, hermi, grids)
        nao = dms[0].shape[0]
        ndms = len(dms)

        def make_rho(idm, ao, sindex, xctype):
            has_screening = sindex is not None and grids is not None
            has_mo = mo_coeff is not None
            is_sparse = _rho1_preferred(rho1_rho2_ratio, xctype)
            if has_screening and (not has_mo or is_sparse):
                return self.eval_rho1(mol, ao, dms[idm], sindex, xctype, hermi,
                                      with_lapl, cutoff=self.cutoff,
                                      ao_cutoff=grids.cutoff, pair_mask=pair_mask)
            elif has_mo:
                return self.eval_rho2(mol, ao, mo_coeff[idm], mo_occ[idm],
                                      sindex, xctype, with_lapl)
            else:
                return self.eval_rho(mol, ao, dms[idm], sindex, xctype, hermi,
                                     with_lapl)
        return make_rho, ndms, nao

    def to_gpu(self):
        try:
            from gpu4pyscf.dft import numint # type: ignore
            return numint.NumInt()
        except ImportError:
            raise ImportError('Cannot find GPU4PySCF')
_NumInt = NumInt


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import dft

    mol = gto.M(atom=[
        ["O" , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ],
        basis='6311g**')
    mf = dft.RKS(mol)
    mf.grids.atom_grid = {"H": (30, 194), "O": (30, 194),}
    mf.grids.prune = None
    mf.grids.build()
    dm = mf.get_init_guess(key='minao')

    numpy.random.seed(1)
    dm1 = numpy.random.random(dm.shape)
    dm1 = lib.hermi_triu(dm1)
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, dm1, spin=0)
    print(res[1] - -37.084047825971282)
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, (dm1,dm1), spin=1)
    print(res[1] - -92.436362308687094)
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, dm, spin=0)
    print(res[1] - -8.6313329288394947)
    res = mf._numint.nr_vxc(mol, mf.grids, mf.xc, (dm,dm), spin=1)
    print(res[1] - -21.520301399504582)
