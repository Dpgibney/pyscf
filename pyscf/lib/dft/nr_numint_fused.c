/* Copyright 2014-2020 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * The BLAS level 3 sparse XC kernels of nr_numint_sparse.c with the AO values
 * computed in place.
 *
 * VXCeval_rho_sparse_gemm, VXCdot_ao_mo_sparse_gemm, VXCdot_aow_ao_sparse_gemm,
 * VXCdot_ao_aow_sparse_gemm and VXCdot_ao_wv_ao_sparse_gemm gather, for each
 * group of GEMM_NBLK blocks of BLKSIZE grid points, the AO values of the shells
 * unscreened on any of the blocks from an AO array that eval_ao wrote to
 * memory. eval_ao is bound by writing that array, and the kernels read it back.
 * The kernels below compute the AO values of each group directly in the gather
 * buffers, with the functions eval_ao calls (GTOcontract_exp0/1,
 * GTOshell_eval_grid_cart(_deriv1), CINTc2s_ket_sph1), and pass them through
 * the same dgemm/dsyrk calls and sums. Their results equal those of the kernels
 * above bit for bit, for each density matrix or potential of a batch, while the
 * AO values of a group are computed once for the whole batch.
 *
 * The contracted radial parts of the AO values of each block and unscreened
 * shell, sum_p c_p exp(-a_p r^2), and for first derivatives also
 * sum_p -2 a_p c_p exp(-a_p r^2), can be kept in a cache (VXCfused_radial_fill)
 * and read instead of computed. A cache for first derivatives also serves
 * kernels that need no derivatives.
 *
 * The kernels take one block of block_loop: coords[3,ldc] are the coordinates
 * of its grid points, screen_index[nblk,nbas] its rows of the screen index and
 * roffs[nblk,nbas] the positions of its radial parts in rcache (rcache = NULL:
 * computed).
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "config.h"
#include "gto/grid_ao_drv.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"
#include "nr_numint_gemm.h"

// GEMM_NBLK for the Python side (numint._GridAOData.group_naos)
int VXCfused_gemm_nblk()
{
        return GEMM_NBLK;
}

// The work buffer of _eval_group for ncomp AO components
#define EVAL_WORK(ncomp)        (BLKSIZE*(3 + NPRIMAX*2 + NCTR_CART*(ncomp)))
#define FUSED_LDA       0
#define FUSED_GGA       1
#define FUSED_MGGA      2

void GTOshell_eval_grid_cart(double *gto, double *ri, double *exps,
                             double *coord, double *alpha, double *coeff,
                             double *env, int l, int np, int nc,
                             size_t nao, size_t ngrids, size_t blksize);
void GTOshell_eval_grid_cart_deriv1(double *gto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    double *env, int l, int np, int nc,
                                    size_t nao, size_t ngrids, size_t blksize);
void GTOshell_eval_grid_cart_deriv2(double *gto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    double *env, int l, int np, int nc,
                                    size_t nao, size_t ngrids, size_t blksize);
void GTOshell_eval_grid_cart_deriv3(double *gto, double *ri, double *exps,
                                    double *coord, double *alpha, double *coeff,
                                    double *env, int l, int np, int nc,
                                    size_t nao, size_t ngrids, size_t blksize);

// The functions of GTOval_sph_deriv0..3: contracted (deriv 0, 1) or primitive
// (deriv 2, 3) exponentials, and the polynomial parts
static const FPtr_exp _eval_exp[] = {
        GTOcontract_exp0, GTOcontract_exp1, GTOprim_exp, GTOprim_exp};
static const FPtr_eval _eval_poly[] = {
        GTOshell_eval_grid_cart, GTOshell_eval_grid_cart_deriv1,
        GTOshell_eval_grid_cart_deriv2, GTOshell_eval_grid_cart_deriv3};
double *CINTc2s_ket_sph1(double *sph, double *cart, int lds, int ldc, int l);
double CINTcommon_fac_sp(int l);

/*
 * aobuf[c,n,k] for the components c < (deriv+1)(deriv+2)(deriv+3)/6, deriv <= 3,
 * n < g->nao and k < ng: the AO value (c = 0) or its derivative c of the n-th AO
 * of group g at grid point blk0*BLKSIZE+k of the block, as eval_ao
 * (GTOval_sph_deriv0..3) computes it, and zero on the blocks where the shell of
 * the AO is screened. The radial parts rcache are for deriv <= 1.
 */
static void _eval_group(double *aobuf, int deriv, GemmGroup *g, int ng,
                        double *coords, size_t ldc, int blk0,
                        double *rcache, int64_t *roffs, int nbas,
                        int *atm, int *bas, double *env, double *work)
{
        const int ncomp = (deriv+1)*(deriv+2)*(deriv+3)/6;
        const size_t comp_size = (size_t)g->nao * ng;
        const size_t ig0 = (size_t)blk0 * BLKSIZE;
        FPtr_exp fexp = _eval_exp[deriv];
        FPtr_eval feval = _eval_poly[deriv];
        double *grid2atm = work;
        double *eprim = grid2atm + 3*BLKSIZE;
        double *cart = eprim + NPRIMAX*BLKSIZE*2;
        int s, b, k0, k, r, c, ic, ish, l, np, nc, deg, dcart, bgrids;
        size_t n = 0;
        double *ri, *p_exp, *pcoeff, *pao, *pcart, *pc, *px;
        for (s = 0; s < g->nshl; s++) {
                ish = g->shl[s];
                l  = bas[ish*BAS_SLOTS+ANG_OF];
                np = bas[ish*BAS_SLOTS+NPRIM_OF];
                nc = bas[ish*BAS_SLOTS+NCTR_OF];
                deg = 2 * l + 1;
                dcart = (l+1)*(l+2)/2;
                ri = env + atm[bas[ish*BAS_SLOTS+ATOM_OF]*ATM_SLOTS+PTR_COORD];
                p_exp  = env + bas[ish*BAS_SLOTS+PTR_EXP];
                pcoeff = env + bas[ish*BAS_SLOTS+PTR_COEFF];
                for (b = 0, k0 = 0; k0 < ng; b++, k0 += BLKSIZE) {
                        bgrids = MIN(BLKSIZE, ng - k0);
                        pao = aobuf + n * ng + k0;
                        if (!(g->blkmask[s] & (1 << b))) {
                                for (c = 0; c < ncomp; c++) {
                                for (r = 0; r < nc*deg; r++) {
                                        px = pao + c * comp_size + r * (size_t)ng;
                                        for (k = 0; k < bgrids; k++) {
                                                px[k] = 0;
                                        }
                                } }
                                continue;
                        }
                        // as _fill_grid2atm of grid_ao_drv.c
                        px = coords + ig0 + k0;
                        for (k = 0; k < bgrids; k++) {
                                grid2atm[0*BLKSIZE+k] = px[0*ldc+k] - ri[0];
                                grid2atm[1*BLKSIZE+k] = px[1*ldc+k] - ri[1];
                                grid2atm[2*BLKSIZE+k] = px[2*ldc+k] - ri[2];
                        }
                        if (rcache != NULL) {
                                pc = rcache + roffs[(size_t)(blk0+b)*nbas+ish];
                                memcpy(eprim, pc, sizeof(double)*nc*BLKSIZE);
                                if (deriv == 1) {
                                        memcpy(eprim+NPRIMAX*BLKSIZE, pc+nc*BLKSIZE,
                                               sizeof(double)*nc*BLKSIZE);
                                }
                        } else {
                                (*fexp)(eprim, grid2atm, p_exp, pcoeff, l, np, nc,
                                        bgrids, CINTcommon_fac_sp(l));
                        }
                        // as GTOeval_sph_iter of grid_ao_drv.c
                        if (l <= 1) {
                                (*feval)(pao, ri, eprim, grid2atm, p_exp, pcoeff,
                                         env, l, np, nc, g->nao, ng, bgrids);
                        } else {
                                (*feval)(cart, ri, eprim, grid2atm, p_exp, pcoeff,
                                         env, l, np, nc, nc*dcart, bgrids, bgrids);
                                pcart = cart;
                                for (c = 0; c < ncomp; c++) {
                                for (ic = 0; ic < nc; ic++) {
                                        CINTc2s_ket_sph1(pao + c * comp_size + ic * deg * (size_t)ng,
                                                         pcart, ng, bgrids, l);
                                        pcart += dcart * bgrids;
                                } }
                        }
                }
                n += nc * deg;
        }
}

/*
 * The radial parts of the AO values of every block of BLKSIZE grid points and
 * every shell unscreened on it: nctr doubles per grid point for deriv 0,
 * 2*nctr for deriv 1, at rcache + roffs[ib*nbas+ish]. coords[3,ngrids].
 */
void VXCfused_radial_fill(double *rcache, int64_t *roffs, int deriv,
                          double *coords, int ngrids, uint8_t *screen_index,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t Ngrids = ngrids;
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        FPtr_exp fexp = deriv == 0 ? GTOcontract_exp0 : GTOcontract_exp1;
#pragma omp parallel
{
        double *grid2atm = malloc(sizeof(double) * BLKSIZE*(3 + NPRIMAX*2));
        double *eprim = grid2atm + 3*BLKSIZE;
        int ib, ish, k, l, nc, bgrids;
        double *ri, *pc, *px;
#pragma omp for schedule(dynamic, 4)
        for (ib = 0; ib < nblk; ib++) {
                bgrids = MIN(BLKSIZE, ngrids - ib*BLKSIZE);
                px = coords + (size_t)ib * BLKSIZE;
                for (ish = 0; ish < nbas; ish++) {
                        if (!screen_index[(size_t)ib*nbas+ish]) {
                                continue;
                        }
                        l  = bas[ish*BAS_SLOTS+ANG_OF];
                        nc = bas[ish*BAS_SLOTS+NCTR_OF];
                        ri = env + atm[bas[ish*BAS_SLOTS+ATOM_OF]*ATM_SLOTS+PTR_COORD];
                        for (k = 0; k < bgrids; k++) {
                                grid2atm[0*BLKSIZE+k] = px[0*Ngrids+k] - ri[0];
                                grid2atm[1*BLKSIZE+k] = px[1*Ngrids+k] - ri[1];
                                grid2atm[2*BLKSIZE+k] = px[2*Ngrids+k] - ri[2];
                        }
                        (*fexp)(eprim, grid2atm, env + bas[ish*BAS_SLOTS+PTR_EXP],
                                env + bas[ish*BAS_SLOTS+PTR_COEFF], l,
                                bas[ish*BAS_SLOTS+NPRIM_OF], nc, bgrids,
                                CINTcommon_fac_sp(l));
                        pc = rcache + roffs[(size_t)ib*nbas+ish];
                        memcpy(pc, eprim, sizeof(double)*nc*BLKSIZE);
                        if (deriv == 1) {
                                memcpy(pc+nc*BLKSIZE, eprim+NPRIMAX*BLKSIZE,
                                       sizeof(double)*nc*BLKSIZE);
                        }
                }
        }
        free(grid2atm);
}
}

// rho[k] = sum_n a[n,k] * c[n,k] over the blocks where the shell of the n-th AO
// is unscreened: the sums of _gemm_contract_rho (nr_numint_sparse.c) on the
// buffer a of _eval_group
static void _contract_group(double *rho, double *a, double *c, GemmGroup *g,
                            int *ao_loc, int ng)
{
        int s, i, b, k, k0, k1;
        int n = 0;
        double *pa, *pc;
        for (k = 0; k < ng; k++) {
                rho[k] = 0;
        }
        for (s = 0; s < g->nshl; s++) {
        for (i = ao_loc[g->shl[s]]; i < ao_loc[g->shl[s]+1]; i++, n++) {
                pa = a + n * (size_t)ng;
                pc = c + n * (size_t)ng;
                for (b = 0, k0 = 0; k0 < ng; b++, k0 += BLKSIZE) {
                        if (g->blkmask[s] & (1 << b)) {
                                k1 = MIN(k0 + BLKSIZE, ng);
                                for (k = k0; k < k1; k++) {
                                        rho[k] += pa[k] * pc[k];
                                }
                        }
                }
        } }
}

static void _zero_rho(double *rho, int nset, int nrho, size_t Ngrids, int ig0, int ng)
{
        int i, k;
        double *prho;
        for (i = 0; i < nset * nrho; i++) {
                prho = rho + i * Ngrids + ig0;
                for (k = 0; k < ng; k++) {
                        prho[k] = 0;
                }
        }
}

/*
 * VXCeval_rho_sparse_gemm for each of the real symmetric dms[nset,nao,nao]:
 * rho[i,0], for nderiv >= 1 also rho[i,1:4] (the gradient) and for nderiv == 2
 * also rho[i,4] (tau) of rho[nset,nrho,ngrids].
 */
void VXCfused_eval_rho(double *rho, double *dms, int nset, int nderiv,
                       int nao, int ngrids, double *coords, int ldc,
                       uint8_t *screen_index, int *ao_loc,
                       int *atm, int natm, int *bas, int nbas, double *env,
                       double *rcache, int64_t *roffs)
{
        const size_t Nao = nao;
        const size_t Ngrids = ngrids;
        const int deriv = nderiv == 0 ? 0 : 1;
        const int ncomp = deriv == 0 ? 1 : 4;
        const int nrho = nderiv == 0 ? 1 : (nderiv == 1 ? 4 : 5);
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        const int ngroup = (nblk + GEMM_NBLK - 1) / GEMM_NBLK;
        const int max_nao = MAX(1, _gemm_max_group_aos(screen_index, ngrids, nbas, ao_loc));
#pragma omp parallel
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        GemmGroup g;
        _gemm_group_init(&g, nbas);
        double *aobuf = malloc(sizeof(double) * max_nao * (GEMM_NGRIDS*(ncomp+1) + max_nao));
        double *cbuf = aobuf + max_nao * GEMM_NGRIDS * ncomp;
        double *dmbuf = cbuf + max_nao * GEMM_NGRIDS;
        double *work = malloc(sizeof(double) * EVAL_WORK(4));
        double rho1[GEMM_NGRIDS];
        double *dm, *rhoi, *prho, *pin, *pout, *pa;
        int igroup, blk0, blk1, ig0, ng, ns, i, s, s1, j, c, k, n;
// each group writes its own grid points of rho, so the result does not depend
// on the schedule
#pragma omp for schedule(dynamic, 4)
        for (igroup = 0; igroup < ngroup; igroup++) {
                blk0 = igroup * GEMM_NBLK;
                blk1 = MIN(blk0 + GEMM_NBLK, nblk);
                ig0 = blk0 * BLKSIZE;
                ng = MIN(blk1 * BLKSIZE, ngrids) - ig0;
                _gemm_group_shells(&g, screen_index, blk0, blk1, nbas, ao_loc);
                ns = g.nao;
                if (ns == 0) {
                        _zero_rho(rho, nset, nrho, Ngrids, ig0, ng);
                        continue;
                }
                _eval_group(aobuf, deriv, &g, ng, coords, ldc, blk0, rcache, roffs,
                            nbas, atm, bas, env, work);
                for (i = 0; i < nset; i++) {
                        dm = dms + i * Nao * Nao;
                        rhoi = rho + i * nrho * Ngrids;
                        // dmbuf[m,n] = dm[j,j1] for the m-th AO j and the n-th AO j1
                        n = 0;
                        for (s = 0; s < g.nseg; s++) {
                        for (j = g.seg_ao[s]; j < g.seg_ao[s] + g.seg_len[s]; j++, n++) {
                                pin = dm + j * Nao;
                                pout = dmbuf + n * (size_t)ns;
                                for (s1 = 0; s1 < g.nseg; s1++) {
                                        for (k = 0; k < g.seg_len[s1]; k++) {
                                                pout[g.seg_off[s1]+k] = pin[g.seg_ao[s1]+k];
                                        }
                                }
                        } }
                        // cbuf[n,k] = sum_m dmbuf[m,n] * aobuf[0,m,k]
                        dgemm_(&TRANS_N, &TRANS_T, &ng, &ns, &ns, &D1, aobuf, &ng,
                               dmbuf, &ns, &D0, cbuf, &ng);
                        _gemm_contract_buf(rhoi + ig0, aobuf, cbuf, ns, ng);
                        if (nderiv == 0) {
                                continue;
                        }
                        for (c = 1; c < 4; c++) {
                                _contract_group(rho1, aobuf + c * (size_t)ns * ng, cbuf,
                                                &g, ao_loc, ng);
                                prho = rhoi + c * Ngrids + ig0;
                                for (k = 0; k < ng; k++) {
                                        prho[k] = rho1[k] * 2;
                                }
                        }
                        if (nderiv == 1) {
                                continue;
                        }
                        prho = rhoi + 4 * Ngrids + ig0;
                        for (k = 0; k < ng; k++) {
                                prho[k] = 0;
                        }
                        for (c = 1; c < 4; c++) {
                                pa = aobuf + c * (size_t)ns * ng;
                                dgemm_(&TRANS_N, &TRANS_T, &ng, &ns, &ns, &D1, pa, &ng,
                                       dmbuf, &ns, &D0, cbuf, &ng);
                                _gemm_contract_buf(rho1, pa, cbuf, ns, ng);
                                for (k = 0; k < ng; k++) {
                                        prho[k] += rho1[k] * .5;
                                }
                        }
                }
        }
        _gemm_group_free(&g);
        free(aobuf);
        free(work);
}
}

// rho[k] = sum_o a[o,k] * b[o,k], in the order of VXC_dcontract_rho
static void _contract_mo(double *rho, double *a, double *b, int nocc, int ng)
{
        int o, k;
        for (k = 0; k < ng; k++) {
                rho[k] = a[k] * b[k];
        }
        for (o = 1; o < nocc; o++) {
                for (k = 0; k < ng; k++) {
                        rho[k] += a[o*(size_t)ng+k] * b[o*(size_t)ng+k];
                }
        }
}

/*
 * The densities of eval_rho2 from the orbitals cmo[i] = mo_coeff[:,occ] *
 * sqrt(mo_occ[occ]) of each set i, given as consecutive [nao,nocc[i]] arrays
 * in cmo: rho[i,0], for nderiv >= 1 also rho[i,1:4] and for nderiv == 2 also
 * rho[i,4] (tau) of rho[nset,nrho,ngrids]. The products of the AO values with
 * the orbitals are those of VXCdot_ao_mo_sparse_gemm, and the sums those of
 * _contract_rho (VXC_dcontract_rho) in eval_rho2, except for the GGA gradient:
 * 2 sum_n ao[c,n] Z[n] with Z = cmo (ao[0] cmo)^T, one GEMM for the three
 * components instead of three. It equals eval_rho2's to rounding.
 */
void VXCfused_eval_rho_mo(double *rho, double *cmo, int *nocc, int nset, int nderiv,
                          int nao, int ngrids, double *coords, int ldc,
                          uint8_t *screen_index, int *ao_loc,
                          int *atm, int natm, int *bas, int nbas, double *env,
                          double *rcache, int64_t *roffs)
{
        const size_t Ngrids = ngrids;
        const int deriv = nderiv == 0 ? 0 : 1;
        const int ncomp = deriv == 0 ? 1 : 4;
        const int nrho = nderiv == 0 ? 1 : (nderiv == 1 ? 4 : 5);
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        const int ngroup = (nblk + GEMM_NBLK - 1) / GEMM_NBLK;
        const int max_nao = MAX(1, _gemm_max_group_aos(screen_index, ngrids, nbas, ao_loc));
        size_t *cmo_off = malloc(sizeof(size_t) * (nset + 1));
        int iset, max_nocc = 1;
        cmo_off[0] = 0;
        for (iset = 0; iset < nset; iset++) {
                cmo_off[iset+1] = cmo_off[iset] + (size_t)nao * nocc[iset];
                max_nocc = MAX(max_nocc, nocc[iset]);
        }
#pragma omp parallel
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        GemmGroup g;
        _gemm_group_init(&g, nbas);
        // v1: Z[ns,ng] for GGA, ao[c]*cbuf for meta-GGA
        const int nv1 = nderiv == 1 ? max_nao : max_nocc;
        double *aobuf = malloc(sizeof(double) * (max_nao * (GEMM_NGRIDS*ncomp + (size_t)max_nocc)
                                                 + (max_nocc + (size_t)nv1) * GEMM_NGRIDS));
        double *cbuf = aobuf + max_nao * GEMM_NGRIDS * ncomp;
        double *v0 = cbuf + max_nao * (size_t)max_nocc;
        double *v1 = v0 + max_nocc * GEMM_NGRIDS;
        double *work = malloc(sizeof(double) * EVAL_WORK(4));
        double rho1[GEMM_NGRIDS];
        double *rhoi, *prho, *pin, *pout, *ptau;
        int igroup, blk0, blk1, ig0, ng, ns, nc, i, s, c, k;
        size_t nseg_c;
#pragma omp for schedule(dynamic, 4)
        for (igroup = 0; igroup < ngroup; igroup++) {
                blk0 = igroup * GEMM_NBLK;
                blk1 = MIN(blk0 + GEMM_NBLK, nblk);
                ig0 = blk0 * BLKSIZE;
                ng = MIN(blk1 * BLKSIZE, ngrids) - ig0;
                _gemm_group_shells(&g, screen_index, blk0, blk1, nbas, ao_loc);
                ns = g.nao;
                if (ns == 0) {
                        _zero_rho(rho, nset, nrho, Ngrids, ig0, ng);
                        continue;
                }
                _eval_group(aobuf, deriv, &g, ng, coords, ldc, blk0, rcache, roffs,
                            nbas, atm, bas, env, work);
                for (i = 0; i < nset; i++) {
                        nc = nocc[i];
                        rhoi = rho + i * nrho * Ngrids;
                        if (nc == 0) {
                                _zero_rho(rhoi, 1, nrho, Ngrids, ig0, ng);
                                continue;
                        }
                        // cbuf[n,o] = cmo[i][j,o] for the n-th AO j of the group
                        pout = cbuf;
                        for (s = 0; s < g.nseg; s++) {
                                pin = cmo + cmo_off[i] + g.seg_ao[s] * (size_t)nc;
                                nseg_c = g.seg_len[s] * (size_t)nc;
                                for (k = 0; k < nseg_c; k++) {
                                        pout[k] = pin[k];
                                }
                                pout += nseg_c;
                        }
                        // v0[o,k] = sum_n aobuf[0,n,k] * cbuf[n,o]
                        dgemm_(&TRANS_N, &TRANS_T, &ng, &nc, &ns, &D1, aobuf, &ng,
                               cbuf, &nc, &D0, v0, &ng);
                        _contract_mo(rhoi + ig0, v0, v0, nc, ng);
                        if (nderiv == 0) {
                                continue;
                        }
                        if (nderiv == 1) {
                                // sum_o v0[o,k] (ao[c] cbuf)[o,k] = sum_n aobuf[c,n,k] Z[n,k]
                                // with Z[n,k] = sum_o cbuf[n,o] v0[o,k]: one GEMM for the
                                // three gradient components (the AO values are zero on the
                                // screened blocks _contract_group skips)
                                dgemm_(&TRANS_N, &TRANS_N, &ng, &ns, &nc, &D1, v0, &ng,
                                       cbuf, &nc, &D0, v1, &ng);
                                for (c = 1; c < 4; c++) {
                                        _contract_group(rho1, aobuf + c * (size_t)ns * ng, v1,
                                                        &g, ao_loc, ng);
                                        prho = rhoi + c * Ngrids + ig0;
                                        for (k = 0; k < ng; k++) {
                                                prho[k] = rho1[k] * 2;
                                        }
                                }
                                continue;
                        }
                        ptau = rhoi + 4 * Ngrids + ig0;
                        for (k = 0; k < ng; k++) {
                                ptau[k] = 0;
                        }
                        for (c = 1; c < 4; c++) {
                                dgemm_(&TRANS_N, &TRANS_T, &ng, &nc, &ns, &D1,
                                       aobuf + c * (size_t)ns * ng, &ng,
                                       cbuf, &nc, &D0, v1, &ng);
                                _contract_mo(rho1, v0, v1, nc, ng);
                                prho = rhoi + c * Ngrids + ig0;
                                for (k = 0; k < ng; k++) {
                                        prho[k] = rho1[k] * 2;
                                }
                                _contract_mo(rho1, v1, v1, nc, ng);
                                for (k = 0; k < ng; k++) {
                                        ptau[k] += rho1[k];
                                }
                        }
                        for (k = 0; k < ng; k++) {
                                ptau[k] *= .5;
                        }
                }
        }
        _gemm_group_free(&g);
        free(aobuf);
        free(work);
}
        free(cmo_off);
}

/*
 * The XC matrices of the potentials wv[nset,nwv,ngrids], added to
 * vmat[nset,nao,nao] (and vtau[nset,nao,nao]), as the *_sparse_gemm kernels
 * add them for each set:
 *   FUSED_LDA  (nwv = 1): VXCdot_aow_ao_sparse_gemm(vmat[i], ao, ao, wv[i], hermi),
 *              vmat[i] += ao[0]^T wv[i] ao[0]; with hermi (symmetric), by dsyrk on
 *              the lower triangle, which is then copied to the upper one.
 *   FUSED_GGA  (nwv = 4): VXCdot_ao_aow_sparse_gemm(vmat[i], ao, wv[i], 4),
 *              vmat[i] += ao[0]^T sum_c wv[i,c] ao[c].
 *   FUSED_MGGA (nwv = 5): as FUSED_GGA with wv[i,0:4], and
 *              VXCdot_ao_wv_ao_sparse_gemm(vtau[i], ao[1:4], wv[i,4], 3),
 *              vtau[i] += sum_c ao[c]^T wv[i,4] ao[c] for c = 1,2,3, symmetric.
 * Each thread keeps nset (FUSED_MGGA: 2*nset) private [nao,nao] matrices.
 */
void VXCfused_dot_wv(double *vmat, double *vtau, double *wv, int nset, int xctype,
                     int hermi, int nao, int ngrids, double *coords, int ldc,
                     uint8_t *screen_index, int *ao_loc,
                     int *atm, int natm, int *bas, int nbas, double *env,
                     double *rcache, int64_t *roffs)
{
        const size_t Nao = nao;
        const size_t Ngrids = ngrids;
        const size_t nn = Nao * Nao;
        const int nwv = xctype == FUSED_LDA ? 1 : (xctype == FUSED_GGA ? 4 : 5);
        const int deriv = xctype == FUSED_LDA ? 0 : 1;
        const int ncomp = deriv == 0 ? 1 : 4;
        // the products for vmat by dsyrk (LDA with a symmetric hermi) or by dgemm;
        // with vhermi, the lower triangle of vmat is summed and copied to the upper
        const int symmetric = xctype == FUSED_LDA && (hermi == HERMITIAN || hermi == SYMMETRIC);
        const int vhermi = xctype == FUSED_LDA ? hermi : 0;
        const int npriv = xctype == FUSED_MGGA ? 2 * nset : nset;
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        const int ngroup = (nblk + GEMM_NBLK - 1) / GEMM_NBLK;
        const int max_nao = _gemm_max_group_aos(screen_index, ngrids, nbas, ao_loc);
        double **vprivs = malloc(sizeof(double *) * omp_get_max_threads());
        int iset;
        if (max_nao > 0 && nset > 0) {
#pragma omp parallel
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const char UPLO = 'U';  // cbuf[n,m] for m <= n in C order
        const double D0 = 0;
        const double D1 = 1;
        const double DM1 = -1;
        GemmGroup g;
        _gemm_group_init(&g, nbas);
        double *vpriv = calloc(npriv * nn, sizeof(double));
        // Published here so the barrier in NPomp_dsum_reduce_inplace makes it
        // visible to the other threads.
        vprivs[omp_get_thread_num()] = vpriv;
        // AO values, then the gathered operands (3 components at most), then the product
        double *aobuf = malloc(sizeof(double) * max_nao * (GEMM_NGRIDS*(ncomp+3) + (size_t)max_nao));
        double *opbuf = aobuf + max_nao * GEMM_NGRIDS * ncomp;
        double *cbuf = opbuf + max_nao * GEMM_NGRIDS * 3;
        double *work = malloc(sizeof(double) * EVAL_WORK(4));
        double sqrtw[GEMM_NGRIDS];
        int pidx[GEMM_NGRIDS];
        int igroup, blk0, blk1, ig0, ng, ns, np, nk, sign, i, j, k, k0, k1;
        int b, s, c, p, n, mask, comp0, ncsym, set;
        size_t comp_size;
        double w, beta;
        double *wvi, *wsym, *pbuf, *pa, *pw, *vout;
// static: each thread sums a fixed set of groups, so vmat is reproducible for a
// given number of threads, and equals the *_sparse_gemm kernels' result
#pragma omp for schedule(static, 1)
        for (igroup = 0; igroup < ngroup; igroup++) {
                blk0 = igroup * GEMM_NBLK;
                blk1 = MIN(blk0 + GEMM_NBLK, nblk);
                ig0 = blk0 * BLKSIZE;
                ng = MIN(blk1 * BLKSIZE, ngrids) - ig0;
                _gemm_group_shells(&g, screen_index, blk0, blk1, nbas, ao_loc);
                ns = g.nao;
                if (ns == 0) {
                        continue;
                }
                comp_size = (size_t)ns * ng;
                _eval_group(aobuf, deriv, &g, ng, coords, ldc, blk0, rcache, roffs,
                            nbas, atm, bas, env, work);
                for (set = 0; set < npriv; set++) {
                        i = xctype == FUSED_MGGA ? set / 2 : set;
                        wvi = wv + (size_t)i * nwv * Ngrids;
                        vout = vpriv + set * nn;
                        if (xctype == FUSED_MGGA && set % 2 == 1) {
                                // the tau term: dsyrk of ao[1:4] scaled by sqrt|wv[i,4]|
                                wsym = wvi + 4 * Ngrids;
                                comp0 = 1;
                                ncsym = 3;
                        } else if (symmetric) {
                                wsym = wvi;
                                comp0 = 0;
                                ncsym = 1;
                        } else {
                                // opbuf[n,k] = sum_c wv[i,c,ig0+k] * ao[c,n,k] for
                                // c < nwv (LDA: 1, GGA and MGGA: 4), zero where
                                // the shell is screened (_gemm_gather with weights)
                                pbuf = opbuf;
                                n = 0;
                                for (s = 0; s < g.nshl; s++) {
                                for (j = ao_loc[g.shl[s]]; j < ao_loc[g.shl[s]+1]; j++, n++) {
                                        pa = aobuf + n * (size_t)ng;
                                        for (b = 0, k0 = 0; k0 < ng; b++, k0 += BLKSIZE) {
                                                k1 = MIN(k0 + BLKSIZE, ng);
                                                if (!(g.blkmask[s] & (1 << b))) {
                                                        for (k = k0; k < k1; k++) {
                                                                pbuf[k] = 0;
                                                        }
                                                        continue;
                                                }
                                                pw = wvi + ig0;
                                                for (k = k0; k < k1; k++) {
                                                        pbuf[k] = pa[k] * pw[k];
                                                }
                                                for (c = 1; c < MIN(nwv, 4); c++) {
                                                        pw = wvi + c * Ngrids + ig0;
                                                        for (k = k0; k < k1; k++) {
                                                                pbuf[k] += pa[c*comp_size+k] * pw[k];
                                                        }
                                                }
                                        }
                                        pbuf += ng;
                                } }
                                if (xctype == FUSED_LDA) {
                                        // cbuf[n,m] = sum_k opbuf[n,k] * ao[0,m,k]
                                        dgemm_(&TRANS_T, &TRANS_N, &ns, &ns, &ng, &D1,
                                               aobuf, &ng, opbuf, &ng, &D0, cbuf, &ns);
                                } else {
                                        // cbuf[n,m] = sum_k ao[0,n,k] * opbuf[m,k]
                                        dgemm_(&TRANS_T, &TRANS_N, &ns, &ns, &ng, &D1,
                                               opbuf, &ng, aobuf, &ng, &D0, cbuf, &ns);
                                }
                                _gemm_scatter_add(vout, cbuf, &g, Nao, vhermi);
                                continue;
                        }
                        // dsyrk needs the form A^T A, so the grid points are split by
                        // the sign of the weight and the AO values scaled by
                        // sqrt(|w|) (_dot_ao_ao_gemm, symmetric branch)
                        beta = 0;
                        for (sign = 1; sign >= -1; sign -= 2) {
                                np = 0;
                                for (k = 0; k < ng; k++) {
                                        w = wsym[ig0+k] * sign;
                                        // NaN goes to the positive set so that it propagates
                                        if (w > 0 || (sign > 0 && w != w)) {
                                                pidx[np] = k;
                                                sqrtw[np] = sqrt(w);
                                                np++;
                                        }
                                }
                                if (np == 0) {
                                        continue;
                                }
                                // opbuf[n,c*np+p] = ao[comp0+c,n,pidx[p]] * sqrtw[p],
                                // zero where the shell is screened (_gemm_gather_points)
                                pbuf = opbuf;
                                n = 0;
                                for (s = 0; s < g.nshl; s++) {
                                        mask = g.blkmask[s];
                                        for (j = ao_loc[g.shl[s]]; j < ao_loc[g.shl[s]+1]; j++, n++) {
                                        for (c = 0; c < ncsym; c++) {
                                                pa = aobuf + (comp0+c) * comp_size + n * (size_t)ng;
                                                for (p = 0; p < np; p++) {
                                                        if (mask & (1 << (pidx[p] / BLKSIZE))) {
                                                                pbuf[p] = pa[pidx[p]] * sqrtw[p];
                                                        } else {
                                                                pbuf[p] = 0;
                                                        }
                                                }
                                                pbuf += np;
                                        } }
                                }
                                nk = np * ncsym;
                                dsyrk_(&UPLO, &TRANS_T, &ns, &nk, (sign > 0 ? &D1 : &DM1),
                                       opbuf, &nk, &beta, cbuf, &ns);
                                beta = 1;
                        }
                        if (beta == 0) {  // the weight is zero on all points of the group
                                continue;
                        }
                        _gemm_scatter_add(vout, cbuf, &g, Nao, 1);
                }
        }
        NPomp_dsum_reduce_inplace(vprivs, npriv * nn);
        double *vsum = vprivs[0];
        int tril;
#pragma omp for schedule(static)
        for (k = 0; k < npriv * nao; k++) {
                set = k / nao;
                j = k % nao;
                i = xctype == FUSED_MGGA ? set / 2 : set;
                if (xctype == FUSED_MGGA && set % 2 == 1) {
                        vout = vtau + i * nn;
                        tril = 1;
                } else {
                        vout = vmat + i * nn;
                        tril = vhermi;
                }
                pa = vsum + set * nn + j * Nao;
                for (n = 0; n < (tril ? j+1 : nao); n++) {
                        vout[j*Nao+n] += pa[n];
                }
        }
        _gemm_group_free(&g);
        free(vpriv);
        free(aobuf);
        free(work);
}
        }
        free(vprivs);
        if (max_nao > 0) {
                for (iset = 0; iset < nset; iset++) {
                        if (vhermi != 0) {
                                NPdsymm_triu(nao, vmat + iset * nn, vhermi);
                        }
                        if (xctype == FUSED_MGGA) {
                                NPdsymm_triu(nao, vtau + iset * nn, HERMITIAN);
                        }
                }
        }
}

// dst[n*ld+k] = sum_t aobuf[c_t,n,k] * wv[u_t,ig0+k] (u_t < 0: weight 1) for the
// nt = term[0] pairs (c_t, u_t) = (term[1+2t], term[2+2t]), and zero on the
// blocks where the shell of the n-th AO is screened
static void _combine(double *dst, size_t ld, double *aobuf, size_t comp_size,
                     int *term, double *wv, size_t Ngrids, int ig0,
                     GemmGroup *g, int *ao_loc, int ng)
{
        const int nt = term[0];
        int s, i, n, b, k0, k1, k, t;
        double *pd, *pa, *pw;
        n = 0;
        for (s = 0; s < g->nshl; s++) {
        for (i = ao_loc[g->shl[s]]; i < ao_loc[g->shl[s]+1]; i++, n++) {
                pd = dst + n * ld;
                for (b = 0, k0 = 0; k0 < ng; b++, k0 += BLKSIZE) {
                        k1 = MIN(k0 + BLKSIZE, ng);
                        if (!(g->blkmask[s] & (1 << b))) {
                                for (k = k0; k < k1; k++) {
                                        pd[k] = 0;
                                }
                                continue;
                        }
                        for (t = 0; t < nt; t++) {
                                pa = aobuf + term[1+2*t] * comp_size + n * (size_t)ng;
                                if (term[2+2*t] < 0) {
                                        if (t == 0) {
                                                for (k = k0; k < k1; k++) {
                                                        pd[k] = pa[k];
                                                }
                                        } else {
                                                for (k = k0; k < k1; k++) {
                                                        pd[k] += pa[k];
                                                }
                                        }
                                } else {
                                        pw = wv + term[2+2*t] * Ngrids + ig0;
                                        if (t == 0) {
                                                for (k = k0; k < k1; k++) {
                                                        pd[k] = pa[k] * pw[k];
                                                }
                                        } else {
                                                for (k = k0; k < k1; k++) {
                                                        pd[k] += pa[k] * pw[k];
                                                }
                                        }
                                }
                        }
                }
        } }
}

/*
 * Sums of products of weighted AO values (up to third derivatives, computed in
 * place), for the XC terms of nuclear gradients and Hessians: for each set k of
 * weights wv[nset,nwv,ngrids] and each output o,
 *   out[k,o,i,j] += sum_s sum_g bra[k,o,s,i,g] * ket[k,s,j,g]
 * where every factor is a combination sum_t ao[c_t,:,g] * wv[k,u_t,g] given by
 * a term {nt, c_0, u_0, ..., c_3, u_3} (u_t < 0: weight 1; TERM_SLOTS ints):
 * ket_terms[nseg] and bra_terms[nout,nseg]. The segments s are stacked along
 * the grid points, so each (k,o) takes one GEMM per group, and the kets are
 * shared by all outputs. out[nset,nout,nao,nao]; deriv <= 3 is the derivative
 * order of the AO components used; rcache only for deriv <= 1.
 */
#define TERM_SLOTS      9
void VXCfused_dot_terms(double *out, double *wv, int nset, int nwv,
                        int nout, int nseg, int *ket_terms, int *bra_terms, int deriv,
                        int nao, int ngrids, double *coords, int ldc,
                        uint8_t *screen_index, int *ao_loc,
                        int *atm, int natm, int *bas, int nbas, double *env,
                        double *rcache, int64_t *roffs)
{
        const size_t Nao = nao;
        const size_t Ngrids = ngrids;
        const size_t nn = Nao * Nao;
        const int ncomp = (deriv+1)*(deriv+2)*(deriv+3)/6;
        const int npriv = nset * nout;
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        const int ngroup = (nblk + GEMM_NBLK - 1) / GEMM_NBLK;
        const int max_nao = _gemm_max_group_aos(screen_index, ngrids, nbas, ao_loc);
        double **vprivs = malloc(sizeof(double *) * omp_get_max_threads());
        if (deriv > 1) {
                rcache = NULL;
        }
        if (max_nao > 0 && npriv > 0) {
#pragma omp parallel
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        GemmGroup g;
        _gemm_group_init(&g, nbas);
        double *vpriv = calloc(npriv * nn, sizeof(double));
        // Published here so the barrier in NPomp_dsum_reduce_inplace makes it
        // visible to the other threads.
        vprivs[omp_get_thread_num()] = vpriv;
        double *aobuf = malloc(sizeof(double) * max_nao * GEMM_NGRIDS * ncomp);
        double *ketbuf = malloc(sizeof(double) * max_nao * GEMM_NGRIDS * nseg * 2);
        double *brabuf = ketbuf + max_nao * GEMM_NGRIDS * nseg;
        double *cbuf = malloc(sizeof(double) * max_nao * max_nao);
        double *work = malloc(sizeof(double) * EVAL_WORK(ncomp));
        int igroup, blk0, blk1, ig0, ng, ns, nk, k, o, s, i, j;
        size_t comp_size;
        double *wvk;
// static: each thread sums a fixed set of groups, so out is reproducible for a
// given number of threads
#pragma omp for schedule(static, 1)
        for (igroup = 0; igroup < ngroup; igroup++) {
                blk0 = igroup * GEMM_NBLK;
                blk1 = MIN(blk0 + GEMM_NBLK, nblk);
                ig0 = blk0 * BLKSIZE;
                ng = MIN(blk1 * BLKSIZE, ngrids) - ig0;
                _gemm_group_shells(&g, screen_index, blk0, blk1, nbas, ao_loc);
                ns = g.nao;
                if (ns == 0) {
                        continue;
                }
                comp_size = (size_t)ns * ng;
                nk = nseg * ng;
                _eval_group(aobuf, deriv, &g, ng, coords, ldc, blk0, rcache, roffs,
                            nbas, atm, bas, env, work);
                for (k = 0; k < nset; k++) {
                        wvk = wv + (size_t)k * nwv * Ngrids;
                        for (s = 0; s < nseg; s++) {
                                _combine(ketbuf + s * ng, nk, aobuf, comp_size,
                                         ket_terms + s * TERM_SLOTS, wvk, Ngrids, ig0,
                                         &g, ao_loc, ng);
                        }
                        for (o = 0; o < nout; o++) {
                                for (s = 0; s < nseg; s++) {
                                        _combine(brabuf + s * ng, nk, aobuf, comp_size,
                                                 bra_terms + (o * nseg + s) * TERM_SLOTS,
                                                 wvk, Ngrids, ig0, &g, ao_loc, ng);
                                }
                                // cbuf[n,m] = sum_q brabuf[n,q] * ketbuf[m,q]
                                dgemm_(&TRANS_T, &TRANS_N, &ns, &ns, &nk, &D1, ketbuf, &nk,
                                       brabuf, &nk, &D0, cbuf, &ns);
                                _gemm_scatter_add(vpriv + (k * nout + o) * nn, cbuf, &g, Nao, 0);
                        }
                }
        }
        NPomp_dsum_reduce_inplace(vprivs, npriv * nn);
        double *vsum = vprivs[0];
#pragma omp for schedule(static)
        for (i = 0; i < npriv * nao; i++) {
                for (j = 0; j < nao; j++) {
                        out[i*Nao+j] += vsum[i*Nao+j];
                }
        }
        _gemm_group_free(&g);
        free(vpriv);
        free(aobuf);
        free(ketbuf);
        free(cbuf);
        free(work);
}
        }
        free(vprivs);
}

/*
 * The densities of the symmetric matrices D[k] = A[k] B^T + B A[k]^T for real
 * A[nset,nao,r] and B[nao,r] (A[k] and B in C order): rho[k,0], for nderiv >= 1
 * also rho[k,1:4] (the gradient) and for nderiv == 2 also rho[k,4] (tau) of
 * rho[nset,nrho,ngrids]. With TA[c] = ao[c] A[k] and TB[c] = ao[c] B,
 *   rho[0]   = 2 sum_l TA[0]_l TB[0]_l
 *   rho[1+x] = 2 sum_l TA[1+x]_l TB[0]_l + TA[0]_l TB[1+x]_l
 *   rho[4]   = sum_x sum_l TA[1+x]_l TB[1+x]_l
 * TB is shared by the sets. GGA takes sum_l TA[1+x]_l TB[0]_l as
 * sum_i ao[1+x]_i Z_i with Z = A TB[0]^T: per set two GEMMs of the size of TA
 * (TA[0] and Z) instead of four.
 */
void VXCfused_eval_rho_lowrank(double *rho, double *A, double *B, int nset, int r,
                               int nderiv, int nao, int ngrids, double *coords, int ldc,
                               uint8_t *screen_index, int *ao_loc,
                               int *atm, int natm, int *bas, int nbas, double *env,
                               double *rcache, int64_t *roffs)
{
        const size_t Ngrids = ngrids;
        const size_t Nr = r;
        const int deriv = nderiv == 0 ? 0 : 1;
        const int ncomp = deriv == 0 ? 1 : 4;
        // the components of TB (GGA and MGGA: 4) and of TA (MGGA: 4)
        const int ntb = nderiv == 0 ? 1 : 4;
        const int nta = nderiv == 2 ? 4 : 1;
        const int nrho = nderiv == 0 ? 1 : (nderiv == 1 ? 4 : 5);
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        const int ngroup = (nblk + GEMM_NBLK - 1) / GEMM_NBLK;
        const int max_nao = MAX(1, _gemm_max_group_aos(screen_index, ngrids, nbas, ao_loc));
#pragma omp parallel
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        GemmGroup g;
        _gemm_group_init(&g, nbas);
        double *aobuf = malloc(sizeof(double) * (max_nao * (GEMM_NGRIDS*ncomp + Nr*2)
                                                 + Nr * GEMM_NGRIDS * (ntb + nta)));
        double *bbuf = aobuf + max_nao * GEMM_NGRIDS * ncomp;
        double *abuf = bbuf + max_nao * Nr;
        double *tb = abuf + max_nao * Nr;
        double *ta = tb + Nr * GEMM_NGRIDS * ntb;
        // Z[i,q] (GGA)
        double *zbuf = nderiv == 1 ? malloc(sizeof(double) * max_nao * GEMM_NGRIDS) : NULL;
        double *work = malloc(sizeof(double) * EVAL_WORK(4));
        double *rhok, *prho, *pa, *pb, *pa0, *pb0, *pin, *pout, *pao, *pz;
        int igroup, blk0, blk1, ig0, ng, ns, k, c, i, l, q, s;
        size_t nseg_r;
#pragma omp for schedule(dynamic, 4)
        for (igroup = 0; igroup < ngroup; igroup++) {
                blk0 = igroup * GEMM_NBLK;
                blk1 = MIN(blk0 + GEMM_NBLK, nblk);
                ig0 = blk0 * BLKSIZE;
                ng = MIN(blk1 * BLKSIZE, ngrids) - ig0;
                _gemm_group_shells(&g, screen_index, blk0, blk1, nbas, ao_loc);
                ns = g.nao;
                if (ns == 0 || r == 0) {
                        _zero_rho(rho, nset, nrho, Ngrids, ig0, ng);
                        continue;
                }
                _eval_group(aobuf, deriv, &g, ng, coords, ldc, blk0, rcache, roffs,
                            nbas, atm, bas, env, work);
                // bbuf[n,l] = B[i,l] for the n-th AO i of the group
                pout = bbuf;
                for (s = 0; s < g.nseg; s++) {
                        pin = B + g.seg_ao[s] * Nr;
                        nseg_r = g.seg_len[s] * Nr;
                        for (q = 0; q < nseg_r; q++) {
                                pout[q] = pin[q];
                        }
                        pout += nseg_r;
                }
                // tb[c,l,q] = sum_n aobuf[c,n,q] * bbuf[n,l]
                for (c = 0; c < ntb; c++) {
                        dgemm_(&TRANS_N, &TRANS_T, &ng, &r, &ns, &D1,
                               aobuf + c * (size_t)ns * ng, &ng, bbuf, &r,
                               &D0, tb + c * Nr * ng, &ng);
                }
                for (k = 0; k < nset; k++) {
                        pout = abuf;
                        for (s = 0; s < g.nseg; s++) {
                                pin = A + k * nao * Nr + g.seg_ao[s] * Nr;
                                nseg_r = g.seg_len[s] * Nr;
                                for (q = 0; q < nseg_r; q++) {
                                        pout[q] = pin[q];
                                }
                                pout += nseg_r;
                        }
                        for (c = 0; c < nta; c++) {
                                dgemm_(&TRANS_N, &TRANS_T, &ng, &r, &ns, &D1,
                                       aobuf + c * (size_t)ns * ng, &ng, abuf, &r,
                                       &D0, ta + c * Nr * ng, &ng);
                        }
                        rhok = rho + k * nrho * Ngrids + ig0;
                        _contract_mo(rhok, ta, tb, r, ng);
                        for (q = 0; q < ng; q++) {
                                rhok[q] *= 2;
                        }
                        if (nderiv == 0) {
                                continue;
                        }
                        if (nderiv == 1) {
                                // zbuf[n,q] = sum_l abuf[n,l] tb[0,l,q]
                                dgemm_(&TRANS_N, &TRANS_N, &ng, &ns, &r, &D1,
                                       tb, &ng, abuf, &r, &D0, zbuf, &ng);
                                for (c = 1; c < 4; c++) {
                                        prho = rhok + c * Ngrids;
                                        pb = tb + c * Nr * ng;
                                        _contract_mo(prho, ta, pb, r, ng);
                                        pao = aobuf + c * (size_t)ns * ng;
                                        for (i = 0; i < ns; i++) {
                                                pa = pao + i * (size_t)ng;
                                                pz = zbuf + i * (size_t)ng;
                                                for (q = 0; q < ng; q++) {
                                                        prho[q] += pa[q] * pz[q];
                                                }
                                        }
                                        for (q = 0; q < ng; q++) {
                                                prho[q] *= 2;
                                        }
                                }
                                continue;
                        }
                        for (c = 1; c < 4; c++) {
                                prho = rhok + c * Ngrids;
                                pa = ta + c * Nr * ng;
                                pb = tb + c * Nr * ng;
                                for (q = 0; q < ng; q++) {
                                        prho[q] = pa[q] * tb[q] + ta[q] * pb[q];
                                }
                                for (l = 1; l < r; l++) {
                                        pa0 = ta + l * (size_t)ng;
                                        pb0 = tb + l * (size_t)ng;
                                        for (q = 0; q < ng; q++) {
                                                prho[q] += pa[l*ng+q] * pb0[q] + pa0[q] * pb[l*ng+q];
                                        }
                                }
                                for (q = 0; q < ng; q++) {
                                        prho[q] *= 2;
                                }
                        }
                        prho = rhok + 4 * Ngrids;
                        for (q = 0; q < ng; q++) {
                                prho[q] = 0;
                        }
                        for (c = 1; c < 4; c++) {
                                pa = ta + c * Nr * ng;
                                pb = tb + c * Nr * ng;
                                for (l = 0; l < r; l++) {
                                        for (q = 0; q < ng; q++) {
                                                prho[q] += pa[l*ng+q] * pb[l*ng+q];
                                        }
                                }
                        }
                }
        }
        _gemm_group_free(&g);
        free(aobuf);
        free(zbuf);
        free(work);
}
}

/*
 * The XC matrices of the potentials wv[nset,nwv,ngrids] times the real orbitals
 * Q[nao,nq] (C order): out[k] += v[k] Q, out[nset,nao,nq], for the matrices
 * v[k] that the fxc contractions build from the (already halved) weights wv[k]:
 *   FUSED_LDA:  v = ao[0]^T wv[0] ao[0]
 *   FUSED_GGA:  v = M + M^T, M = ao[0]^T sum_c wv[c] ao[c] (c < 4)
 *   FUSED_MGGA: as GGA, plus sum_x ao[x]^T wv[4] ao[x] (x = 1,2,3)
 * With FQ[c] = ao[c] Q on the points of a group,
 *   v Q = ao[0]^T (sum_c wv[c] FQ[c]) + (sum_c wv[c] ao[c])^T FQ[0]
 *         + sum_x ao[x]^T (wv[4] FQ[x])
 * whose terms are stacked along the points: one GEMM (ns x nq x K) per set and
 * group for the rows of the ns AOs of the group.
 */
void VXCfused_dot_ao_mo(double *out, double *wv, int nset, int xctype,
                        double *Q, int nq,
                        int nao, int ngrids, double *coords, int ldc,
                        uint8_t *screen_index, int *ao_loc,
                        int *atm, int natm, int *bas, int nbas, double *env,
                        double *rcache, int64_t *roffs)
{
        const size_t Ngrids = ngrids;
        const size_t nvq = (size_t)nao * nq;
        const int nwv = xctype == FUSED_LDA ? 1 : (xctype == FUSED_GGA ? 4 : 5);
        const int deriv = xctype == FUSED_LDA ? 0 : 1;
        const int ncomp = deriv == 0 ? 1 : 4;
        const int nseg = xctype == FUSED_LDA ? 1 : (xctype == FUSED_GGA ? 2 : 5);
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        const int ngroup = (nblk + GEMM_NBLK - 1) / GEMM_NBLK;
        const int max_nao = _gemm_max_group_aos(screen_index, ngrids, nbas, ao_loc);
        double **vprivs = malloc(sizeof(double *) * omp_get_max_threads());
        int i;
        if (max_nao > 0 && nset > 0 && nq > 0) {
#pragma omp parallel
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        GemmGroup g;
        _gemm_group_init(&g, nbas);
        double *vpriv = calloc(nset * nvq, sizeof(double));
        // Published here so the barrier in NPomp_dsum_reduce_inplace makes it
        // visible to the other threads.
        vprivs[omp_get_thread_num()] = vpriv;
        double *aobuf = malloc(sizeof(double) * max_nao * (GEMM_NGRIDS*ncomp + (size_t)nq));
        double *qbuf = aobuf + max_nao * GEMM_NGRIDS * ncomp;
        double *fq = malloc(sizeof(double) * ((size_t)nq * GEMM_NGRIDS * (ncomp + nseg)
                                              + max_nao * ((size_t)GEMM_NGRIDS * nseg + nq)));
        double *ket = fq + (size_t)nq * GEMM_NGRIDS * ncomp;  // [nq,K]
        double *bra = ket + (size_t)nq * GEMM_NGRIDS * nseg;  // [ns,K]
        double *vbuf = bra + (size_t)max_nao * GEMM_NGRIDS * nseg;  // [ns,nq]
        double *work = malloc(sizeof(double) * EVAL_WORK(4));
        double *wvk, *pw, *src, *dst, *pin, *pout;
        int igroup, blk0, blk1, ig0, ng, ns, nk, k, c, s, n, p, q, x;
        size_t nseg_n;
#pragma omp for schedule(static, 1)
        for (igroup = 0; igroup < ngroup; igroup++) {
                blk0 = igroup * GEMM_NBLK;
                blk1 = MIN(blk0 + GEMM_NBLK, nblk);
                ig0 = blk0 * BLKSIZE;
                ng = MIN(blk1 * BLKSIZE, ngrids) - ig0;
                _gemm_group_shells(&g, screen_index, blk0, blk1, nbas, ao_loc);
                ns = g.nao;
                if (ns == 0) {
                        continue;
                }
                _eval_group(aobuf, deriv, &g, ng, coords, ldc, blk0, rcache, roffs,
                            nbas, atm, bas, env, work);
                // the rows of Q of the AOs of the group
                pout = qbuf;
                for (s = 0; s < g.nseg; s++) {
                        pin = Q + g.seg_ao[s] * (size_t)nq;
                        nseg_n = g.seg_len[s] * (size_t)nq;
                        for (q = 0; q < nseg_n; q++) {
                                pout[q] = pin[q];
                        }
                        pout += nseg_n;
                }
                // fq[c,p,q] = sum_n aobuf[c,n,q] * qbuf[n,p]
                for (c = 0; c < ncomp; c++) {
                        dgemm_(&TRANS_N, &TRANS_T, &ng, &nq, &ns, &D1,
                               aobuf + c * (size_t)ns * ng, &ng, qbuf, &nq,
                               &D0, fq + c * (size_t)nq * ng, &ng);
                }
                nk = nseg * ng;
                // the segments without weights: bra[:,0] = ao[0]; for MGGA
                // bra[:,1+x] = ao[x]; for GGA and MGGA ket[:,1] = fq[0]
                for (n = 0; n < ns; n++) {
                        src = aobuf + n * (size_t)ng;
                        dst = bra + n * (size_t)nk;
                        for (q = 0; q < ng; q++) {
                                dst[q] = src[q];
                        }
                        if (xctype == FUSED_MGGA) {
                                for (x = 1; x < 4; x++) {
                                        src = aobuf + (x * (size_t)ns + n) * ng;
                                        dst = bra + n * (size_t)nk + (1+x) * ng;
                                        for (q = 0; q < ng; q++) {
                                                dst[q] = src[q];
                                        }
                                }
                        }
                }
                if (xctype != FUSED_LDA) {
                        for (p = 0; p < nq; p++) {
                                src = fq + p * (size_t)ng;
                                dst = ket + p * (size_t)nk + ng;
                                for (q = 0; q < ng; q++) {
                                        dst[q] = src[q];
                                }
                        }
                }
                for (k = 0; k < nset; k++) {
                        wvk = wv + (size_t)k * nwv * Ngrids + ig0;
                        // ket[:,0] = sum_c wv[c] fq[c]
                        for (p = 0; p < nq; p++) {
                                dst = ket + p * (size_t)nk;
                                src = fq + p * (size_t)ng;
                                for (q = 0; q < ng; q++) {
                                        dst[q] = src[q] * wvk[q];
                                }
                                for (c = 1; c < ncomp; c++) {
                                        src = fq + (c * (size_t)nq + p) * ng;
                                        pw = wvk + c * Ngrids;
                                        for (q = 0; q < ng; q++) {
                                                dst[q] += src[q] * pw[q];
                                        }
                                }
                        }
                        if (xctype != FUSED_LDA) {
                                // bra[:,1] = sum_c wv[c] ao[c]
                                for (n = 0; n < ns; n++) {
                                        dst = bra + n * (size_t)nk + ng;
                                        src = aobuf + n * (size_t)ng;
                                        for (q = 0; q < ng; q++) {
                                                dst[q] = src[q] * wvk[q];
                                        }
                                        for (c = 1; c < 4; c++) {
                                                src = aobuf + (c * (size_t)ns + n) * ng;
                                                pw = wvk + c * Ngrids;
                                                for (q = 0; q < ng; q++) {
                                                        dst[q] += src[q] * pw[q];
                                                }
                                        }
                                }
                        }
                        if (xctype == FUSED_MGGA) {
                                // ket[:,1+x] = wv[4] fq[x]
                                pw = wvk + 4 * Ngrids;
                                for (x = 1; x < 4; x++) {
                                for (p = 0; p < nq; p++) {
                                        dst = ket + p * (size_t)nk + (1+x) * ng;
                                        src = fq + (x * (size_t)nq + p) * ng;
                                        for (q = 0; q < ng; q++) {
                                                dst[q] = src[q] * pw[q];
                                        }
                                } }
                        }
                        // vbuf[n,p] = sum_K bra[n,K] * ket[p,K], added to the
                        // rows of the group's AOs in vpriv[k]
                        dgemm_(&TRANS_T, &TRANS_N, &nq, &ns, &nk, &D1, ket, &nk,
                               bra, &nk, &D0, vbuf, &nq);
                        pin = vbuf;
                        for (s = 0; s < g.nseg; s++) {
                                pout = vpriv + k * nvq + g.seg_ao[s] * (size_t)nq;
                                nseg_n = g.seg_len[s] * (size_t)nq;
                                for (q = 0; q < nseg_n; q++) {
                                        pout[q] += pin[q];
                                }
                                pin += nseg_n;
                        }
                }
        }
        NPomp_dsum_reduce_inplace(vprivs, nset * nvq);
        double *vsum = vprivs[0];
#pragma omp for schedule(static)
        for (i = 0; i < nset; i++) {
                size_t j;
                for (j = 0; j < nvq; j++) {
                        out[i*nvq+j] += vsum[i*nvq+j];
                }
        }
        _gemm_group_free(&g);
        free(vpriv);
        free(aobuf);
        free(fq);
        free(work);
}
        }
        free(vprivs);
}
