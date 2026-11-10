#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
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

'''
Two-level SCF: the SCF solution of a model from a few of its own Fock builds.

The Fock matrix of an expensive model (exact J and K, fine XC and VV10 grids)
and that of a cheap one of the same Hamiltonian (RI-J, seminumerical K, coarse
grids) differ by a matrix that changes little with the density: J and K are
linear in it, and the grid errors of the XC terms are smooth. With the
expensive potential V[D_ref] of a reference density, the SCF iterates on

    F[D] = h + V[D_ref] + (V_cheap[D] - V_cheap[D_ref])

whose builds are the cheap model's; its error is the difference of the two
models applied to D - D_ref, second order. The reference is the converged
density of the cheap model first, then that of each such iteration; every
reference costs one expensive build, which also gives the energy and the
orbital gradient of the expensive model there, the tests of its convergence.
The solution is the expensive model's own, to its convergence thresholds.

    >>> mf = dft.RKS(mol, xc='wb97m-v')
    >>> mf.two_level = True
    >>> mf.kernel()

Caffeine, wB97M-V/def2-TZVPD with XC grid (99,590) and VV10 grid (50,194): the
energy of the plain SCF to 7e-11 Eh with 2 Fock builds of the model instead of
15, 204 s against 451 s.
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, chkfile
from pyscf.scf import diis as scf_diis

# The micro cycles stop when the predicted energy changes by less than this
# fraction of conv_tol and the orbital gradient is below this fraction of
# conv_tol_grad
MICRO_ETOL = .1
MICRO_GTOL = .3


def cheap_scf(mf):
    '''The cheap model of mf for the two-level SCF: a copy with the grids of
    levels two_level_grids_level and two_level_nlcgrids_level (where mf's are
    finer, see hf._pre_scf_grids) and, unless mf fits its integrals already,
    RI-J in two_level_auxbasis with seminumerical K (sgx.rijcosx, without its
    step on the larger grid). None if nothing of mf can be made cheaper this
    way: it is fitted already and has no finer grids, or is not a real
    restricted or unrestricted model.
    '''
    from pyscf.scf import ghf, dhf
    if isinstance(mf, (ghf.GHF, dhf.DHF)):
        return None
    mol = mf.mol
    cheap = mf.copy()
    cheap._eri = None
    cheap._opt = {None: None}
    cheap.diis = True
    cheap.diis_file = None
    cheap.chkfile = None
    cheap.callback = None
    cheap.scf_summary = {}
    cheap.conv_check = False
    cheap.two_level = False
    cheaper = False
    for key, level in (('grids', mf.two_level_grids_level),
                       ('nlcgrids', mf.two_level_nlcgrids_level)):
        grids = getattr(mf, key, None)
        if grids is None:
            continue
        g = hf._pre_scf_grids(grids, mol, level)
        setattr(cheap, key, g)
        if g.level != grids.level or g.atom_grid != grids.atom_grid:
            cheaper = True
    if getattr(cheap, 'with_df', None) is None:
        from pyscf.sgx import sgx
        auxbasis = mf.two_level_auxbasis
        if isinstance(auxbasis, str) and auxbasis.lower().replace('-', '_') == 'aug_etb':
            from pyscf.df import addons
            auxbasis = addons.aug_etb(mol)
        cheap = sgx.rijcosx(cheap, auxbasis=auxbasis)
        cheap.with_df.grids_level_energy = None
        cheaper = True
    if not cheaper:
        return None
    return cheap


def _veff(mf, mol, dm):
    '''The potential of dm from a full build of mf'''
    if hasattr(mf, '_nsteps_direct'):
        # an SGX object: no incremental state, the fixed pair threshold
        mf._nsteps_direct = 0
    return mf.get_veff(mol, dm)


def _tagged(v, like):
    '''The array v with the solvent terms of the potential like, which
    get_fock and energy_elec of a solvated model read from the potential'''
    tags = {key: getattr(like, key) for key in ('e_solvent', 'v_solvent')
            if getattr(like, key, None) is not None}
    if tags:
        return lib.tag_array(v, **tags)
    return v


def kernel(mf, conv_tol=1e-10, conv_tol_grad=None, dump_chk=True, dm0=None,
           callback=None, conv_check=True, **kwargs):
    '''The SCF of mf by the two-level scheme, with the arguments and the
    return values of hf.kernel: (converged, e_tot, mo_energy, mo_coeff, mo_occ).
    mf.cycles counts the Fock builds of mf itself.

    Converged when, at a reference density, the orbital gradient of mf is below
    conv_tol_grad and the energy of mf is within conv_tol of the one predicted
    for that density from the previous reference: its energy there, the change
    of the cheap model's energy, and the first-order term of the difference of
    the two potentials. With conv_check the orbitals returned are those of the
    last Fock matrix of mf, without DIIS; unlike hf.kernel, no further build
    follows them, and e_tot is the energy of the density that Fock matrix was
    built from.
    '''
    if mf.max_cycle <= 0:
        return hf.kernel(mf, conv_tol, conv_tol_grad, dump_chk, dm0, callback,
                         conv_check, **kwargs)
    cheap = mf.two_level_scf
    if cheap is None:
        cheap = cheap_scf(mf)
    if cheap is None:
        logger.info(mf, 'two-level SCF: no cheaper model of this one, the plain SCF instead')
        return hf.kernel(mf, conv_tol, conv_tol_grad, dump_chk, dm0, callback,
                         conv_check, **kwargs)

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mf)
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        log.info('Set gradient conv threshold to %g', conv_tol_grad)
    mol = mf.mol

    # the cheap model's solution
    cheap.conv_tol = conv_tol * mf.two_level_conv_factor
    cheap.conv_tol_grad = None
    cheap.kernel(dm0=dm0)
    if not cheap.converged:
        log.warn('two-level SCF: the cheap model did not converge')
    dm = cheap.make_rdm1()
    mo_energy, mo_coeff, mo_occ = cheap.mo_energy, cheap.mo_coeff, cheap.mo_occ
    cput1 = log.timer('two-level SCF: the cheap model, %d cycles' % cheap.cycles, *cput0)

    s1e = mf.get_ovlp(mol)
    h1e = mf.get_hcore(mol)
    x_orth = mf.check_linear_dependency(s1e, log)
    if dump_chk and mf.chkfile:
        chkfile.save_mol(mol, mf.chkfile)
    mf.pre_kernel(locals())

    def gnorm(fock):
        return numpy.linalg.norm(mf.get_grad(mo_coeff, mo_occ, fock))

    scf_conv = False
    e_pred = None
    e_tot = 0
    ncheap = 0
    mf.cycles = 0
    for cycle in range(mf.max_cycle):
        # a reference: the potential, energy and orbital gradient of mf itself
        vref = _veff(mf, mol, dm)
        last_hf_e = e_tot
        e_tot = mf.energy_tot(dm, h1e, vref)
        fock = mf.get_fock(h1e, s1e, vref, dm)  # = h1e + vref, no DIIS
        norm_gorb = gnorm(fock)
        mf.cycles = cycle + 1
        if e_pred is None:
            log.info('reference %d  E= %.15g  |g|= %4.3g', cycle+1, e_tot, norm_gorb)
        else:
            log.info('reference %d  E= %.15g  E - predicted= %4.3g  |g|= %4.3g  '
                     '(%d cheap builds)', cycle+1, e_tot, e_tot-e_pred, norm_gorb, ncheap)
            if abs(e_tot-e_pred) < conv_tol and norm_gorb < conv_tol_grad:
                scf_conv = True
        if dump_chk and mf.chkfile:
            mf.dump_chk(locals())
        if callable(callback):
            callback(locals())
        cput1 = log.timer('reference %d' % (cycle+1), *cput1)
        if scf_conv:
            break

        # micro cycles on h + V[D_ref] + (V_cheap[D] - V_cheap[D_ref])
        vx = _veff(cheap, mol, dm)
        ncheap += 1
        ex_ref = cheap.energy_tot(dm, h1e, vx)
        dm_ref = dm
        vref_a = numpy.asarray(vref)
        vx_ref = numpy.asarray(vx)
        dv = vref_a - vx_ref
        mf_diis = scf_diis.SCF_DIIS(mf, None)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
        mf_diis.damp = mf.diis_damp
        mf_diis.Corth = x_orth
        e_micro = e_tot
        vhf = _tagged(vref_a, vx)
        for micro in range(mf.two_level_micro_cycles):
            f = mf.get_fock(h1e, s1e, vhf, dm, micro, mf_diis)
            mo_energy, mo_coeff = mf.eig(f, s1e, x=x_orth)
            mo_occ = mf.get_occ(mo_energy, mo_coeff)
            dm = mf.make_rdm1(mo_coeff, mo_occ)
            vx = _veff(cheap, mol, dm)
            ncheap += 1
            vhf = _tagged(vref_a + (numpy.asarray(vx) - vx_ref), vx)
            # the energy of mf predicted for dm: that of the reference, the change of the cheap
            # model's, and the first order of the difference of the two models
            e_last = e_micro
            e_micro = (e_tot + cheap.energy_tot(dm, h1e, vx) - ex_ref
                       + (dv * (numpy.asarray(dm) - dm_ref).swapaxes(-1, -2)).sum().real)
            g = gnorm(mf.get_fock(h1e, s1e, vhf, dm))
            log.debug('  micro %d  E(predicted)= %.15g  delta_E= %4.3g  |g|= %4.3g',
                      micro+1, e_micro, e_micro-e_last, g)
            if abs(e_micro-e_last) < conv_tol * MICRO_ETOL and g < conv_tol_grad * MICRO_GTOL:
                break
        e_pred = e_micro
        cput1 = log.timer('micro cycles of reference %d' % (cycle+1), *cput1)

    if scf_conv and conv_check:
        # the orbitals of mf's own Fock matrix at the converged density, without DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e, x=x_orth)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        if dump_chk and mf.chkfile:
            mf.dump_chk(locals())
    log.info('two-level SCF: %d Fock builds of the model, %d of the cheap one after its SCF '
             'of %d cycles', mf.cycles, ncheap, cheap.cycles)
    log.timer('two-level scf_cycle', *cput0)
    # for the hooks that read the potential of the final density (SGX's energy on its larger grid)
    vhf = vref if scf_conv else None
    mf.post_kernel(locals())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ
