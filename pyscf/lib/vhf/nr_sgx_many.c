/* Copyright 2026 The PySCF Developers. All Rights Reserved.

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
 * The SGX K of many density matrices at once (the CPHF's and TDDFT's): for a
 * block of grid points,
 *      G[g,i,x] = sum_j A_ij(g) F[g,j,x],   A_ij(g) = (i j | 1/|r-g|)
 * with F and G point-major and the density index x innermost. The integrals of
 * every screened shell pair of a batch of points are evaluated once and applied
 * to all densities, instead of once per density and pair as in
 * SGXnr_direct_k_drv, whose per-density accumulators do not stay in cache.
 * At each point the products go by row shell: the sums over the row's shell
 * pairs (both triangles of A) for 8 densities at a time are kept in registers
 * and G is stored once per row.
 * The batches are screened as by SGXscreen_grid_pair (or not screened), with
 * the shell norms f_I taken over all densities, so the shell pairs are those
 * of SGXnr_direct_k_drv for the same densities.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "cint.h"
#include "gto/gto.h"
#include "nr_sgx_direct.h"

#define MAX(I,J)        ((I) > (J) ? (I) : (J))
#define MIN(I,J)        ((I) < (J) ? (I) : (J))
// densities of the register accumulators
#define XC      8

size_t _max_cache_size_sgx(int (*intor)(), int *shls_slice, int ncenter,
                           int *atm, int natm, int *bas, int nbas, double *env,
                           int blksize);
void SGXscreen_grid_pair(CSGXOpt *sgxopt, double *f_ug, const int ibatch,
                         const int n_dm, const int nish, void *shl_info,
                         void *buf);
void SGXnoscreen_grid(CSGXOpt *sgxopt, double *f_ug, const int ibatch,
                      const int n_dm, const int nish, void *shl_info,
                      void *buf);

/*
 * sqrt(sum_g |w_g| sum_{i in I} sum_x F[g,i,x]^2) of shell I on the batch's
 * points, F (ngrids,nao,nset); SGXscreen_grid_pair's f_I without its m_bI
 */
static double _shell_fnorm(double *F, int *ao_loc, size_t nao, int nset,
                           double *wt, int ig0, int dg, int ish)
{
        const size_t n = (size_t)(ao_loc[ish+1] - ao_loc[ish]) * nset;
        double s = 0;
        int g;
        size_t i;
        for (g = 0; g < dg; g++) {
                double *f = F + ((ig0 + g) * nao + ao_loc[ish]) * nset;
                double sg = 0;
#pragma omp simd reduction(+:sg)
                for (i = 0; i < n; i++) {
                        sg += f[i] * f[i];
                }
                s += sg * fabs(wt[g]);
        }
        return sqrt(s);
}

/*
 * The rows i0 <= i < i0+DI of G at one point for the densities xc0 <= x <
 * xc0+XC: sum over the row shell's columns q < ncol (AO col[q], A_iq at
 * V[q*DI+i]) of A_iq F[col[q],x]. QU sets of accumulators take the columns in
 * turn, so that small shells have enough independent FMA chains.
 */
#define DEF_SGX_ROW(NAME, ATTR, MADD, DI, QU) \
ATTR static void NAME(double *Gk, double *Fk, double *V, int *col, int ncol, \
                      int nset, int xc0, int i0, int di_) \
{ \
        const int di = DI > 0 ? DI : di_; \
        double acc[QU][(DI > 0 ? DI : 32) * XC]; \
        int q, i, x, u; \
        for (u = 0; u < QU; u++) { \
                for (i = 0; i < di * XC; i++) { \
                        acc[u][i] = 0; \
                } \
        } \
        for (q = 0; q + QU <= ncol; q += QU) { \
                for (u = 0; u < QU; u++) { \
                        const double *f = Fk + (size_t)col[q+u] * nset + xc0; \
                        const double *v = V + (size_t)(q+u) * di; \
                        for (i = 0; i < di; i++) { \
                                const double a = v[i]; \
                                _Pragma("omp simd") \
                                for (x = 0; x < XC; x++) { \
                                        acc[u][i*XC+x] = MADD(a, f[x], acc[u][i*XC+x]); \
                                } \
                        } \
                } \
        } \
        for (; q < ncol; q++) { \
                const double *f = Fk + (size_t)col[q] * nset + xc0; \
                const double *v = V + (size_t)q * di; \
                for (i = 0; i < di; i++) { \
                        const double a = v[i]; \
                        _Pragma("omp simd") \
                        for (x = 0; x < XC; x++) { \
                                acc[0][i*XC+x] = MADD(a, f[x], acc[0][i*XC+x]); \
                        } \
                } \
        } \
        for (u = 1; u < QU; u++) { \
                for (i = 0; i < di * XC; i++) { \
                        acc[0][i] += acc[u][i]; \
                } \
        } \
        for (i = 0; i < di; i++) { \
                double *g = Gk + (size_t)(i0 + i) * nset + xc0; \
                _Pragma("omp simd") \
                for (x = 0; x < XC; x++) { \
                        g[x] = acc[0][i*XC+x]; \
                } \
        } \
}

typedef void (*SGXRowFn)(double *Gk, double *Fk, double *V, int *col, int ncol,
                         int nset, int xc0, int i0, int di_);

#define MADD_PLAIN(a, b, c)     ((a) * (b) + (c))
#define MADD_FMA(a, b, c)       __builtin_fma(a, b, c)
#define DEF_SGX_ROWS(SUFFIX, ATTR, MADD) \
DEF_SGX_ROW(_sgx_row1##SUFFIX, ATTR, MADD, 1, 4) \
DEF_SGX_ROW(_sgx_row3##SUFFIX, ATTR, MADD, 3, 2) \
DEF_SGX_ROW(_sgx_row5##SUFFIX, ATTR, MADD, 5, 1) \
DEF_SGX_ROW(_sgx_row6##SUFFIX, ATTR, MADD, 6, 1) \
DEF_SGX_ROW(_sgx_row7##SUFFIX, ATTR, MADD, 7, 1) \
DEF_SGX_ROW(_sgx_rowx##SUFFIX, ATTR, MADD, 0, 1)
DEF_SGX_ROWS(, , MADD_PLAIN)
#if defined(__x86_64__) && defined(__GNUC__)
#define AVX2_FMA __attribute__((target("avx2,fma")))
DEF_SGX_ROWS(_avx2, AVX2_FMA, MADD_FMA)
#endif

/*
 * F, G: (ngrids,nao,nset) for the grid points env[PTR_GRIDS:] of the block
 * (env[NGRIDS] of them, with sgxopt->wt, mbar_bi, vtol and full_f_bi set for
 * the block by SGXData.set_block); nset a multiple of 8. The rows of G of the
 * shells that have screened pairs are set, the other rows are not written.
 * blksize is SGX_BLKSIZE, the points of a batch of sgxopt's per-batch data.
 * s2 integrals over all shells, one component. The screening of sgxopt must
 * be SGXscreen_grid_pair (or SGXnoscreen_grid) with its shell function;
 * returns 1, doing nothing, for other screenings or nset, 0 otherwise.
 */
int SGXnr_direct_k_many(int (*intor)(), double *F, double *G, int nset,
                        int blksize, int *ao_loc, CINTOpt *cintopt,
                        CSGXOpt *sgxopt, int *atm, int natm, int *bas, int nbas,
                        double *env)
{
        const int pair_screen = sgxopt->fscreen_grid == SGXscreen_grid_pair;
        if ((!pair_screen && sgxopt->fscreen_grid != SGXnoscreen_grid) ||
            nset % XC != 0) {
                return 1;
        }
        const size_t nao = ao_loc[nbas];
        int shls_slice[4] = {0, nbas, 0, nbas};
        const int di_max = GTOmax_shell_dim(ao_loc, shls_slice, 2);
        if (di_max > 32) {
                return 1;
        }
        const size_t cache_size = _max_cache_size_sgx(intor, shls_slice, 2,
                                                      atm, natm, bas, nbas, env,
                                                      blksize);
        const int tot_ngrids = (int) env[NGRIDS];
        const int nbatch = (tot_ngrids + blksize - 1) / blksize;
        // the row kernels by shell size
        SGXRowFn row_fn[33];
        int i;
        for (i = 0; i <= 32; i++) {
                row_fn[i] = _sgx_rowx;
        }
        row_fn[1] = _sgx_row1;
        row_fn[3] = _sgx_row3;
        row_fn[5] = _sgx_row5;
        row_fn[6] = _sgx_row6;
        row_fn[7] = _sgx_row7;
#if defined(__x86_64__) && defined(__GNUC__)
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
                for (i = 0; i <= 32; i++) {
                        row_fn[i] = _sgx_rowx_avx2;
                }
                row_fn[1] = _sgx_row1_avx2;
                row_fn[3] = _sgx_row3_avx2;
                row_fn[5] = _sgx_row5_avx2;
                row_fn[6] = _sgx_row6_avx2;
                row_fn[7] = _sgx_row7_avx2;
        }
#endif
        const size_t npair_max = (size_t)nbas * (nbas + 1) / 2;
        int *pair_i = malloc(sizeof(int) * npair_max * 2);
        int *pair_j = pair_i + npair_max;
        size_t *pair_off = malloc(sizeof(size_t) * npair_max);
        // the pairs by row shell, both triangles: column shell, block offset,
        // transposed
        int *row_ptr = malloc(sizeof(int) * (nbas + 1) * 3);
        int *row_pos = row_ptr + nbas + 1;
        int *rows = row_pos + nbas + 1;
        int *nb_col = malloc(sizeof(int) * npair_max * 4);
        int *nb_tr = nb_col + npair_max * 2;
        size_t *nb_off = malloc(sizeof(size_t) * npair_max * 2);
        // a point's A by row shell: column-major blocks (ncol x di) at
        // row_voff, their AO columns at row_coff in cols, each neighbour's
        // first column within its row at nb_pos
        size_t *row_voff = malloc(sizeof(size_t) * (nbas + 1) * 2);
        size_t *row_coff = row_voff + nbas + 1;
        int *row_ncol = malloc(sizeof(int) * nbas);
        int *nb_pos = malloc(sizeof(int) * npair_max * 2);
        int *cols = malloc(sizeof(int) * nao * nbas);
        size_t nnz_full = 0;
        double *shl_info = malloc(sizeof(double) * (2 * nbas + 1) +
                                  sgxopt->shl_info_size_bytes);
        void *screen_buf = malloc(sgxopt->buf_size_bytes + 8);
        int *sj_shells = malloc(sizeof(int) * nbas);
        double *E = NULL;
        size_t E_size = 0;
        int npair = 0, nrow = 0;
        size_t nnz = 0;
        int dg = 0, ig0 = 0;

#pragma omp parallel
{
        double *buf = malloc(sizeof(double) * di_max * di_max * blksize);
        double *cache = malloc(sizeof(double) * (cache_size + 1));
        double *V = NULL;
        size_t V_size = 0;
        int ibatch, p, k, ij, ish, r, xc0;
        int shls[4];
        for (ibatch = 0; ibatch < nbatch; ibatch++) {
                // f_I of the batch, all densities
#pragma omp single
{
                ig0 = ibatch * blksize;
                dg = MIN(ig0 + blksize, tot_ngrids) - ig0;
}
                if (pair_screen) {
                        double *wt = sgxopt->wt + ibatch * blksize;
                        double *mbar_ii = sgxopt->mbar_bi + ibatch * nbas;
#pragma omp for schedule(dynamic, 4)
                        for (ish = 0; ish < nbas; ish++) {
                                shl_info[ish] = _shell_fnorm(F, ao_loc, nao, nset, wt,
                                                             ig0, dg, ish) * mbar_ii[ish];
                        }
                }
#pragma omp single
{
                if (pair_screen) {
                        // F_I and vtol as SGXscreen_grid_pair
                        double *mbar_ii = sgxopt->mbar_bi + ibatch * nbas;
                        double *ffbar_i = shl_info + nbas;
                        for (ish = 0; ish < nbas; ish++) {
                                ffbar_i[ish] = sgxopt->full_f_bi == NULL ? shl_info[ish] :
                                        sgxopt->full_f_bi[ibatch * nbas + ish] * mbar_ii[ish];
                        }
                        ffbar_i[nbas] = sgxopt->vtol == NULL ? 1e300 : sgxopt->vtol[ibatch];
                } else {
                        sgxopt->fscreen_grid(sgxopt, NULL, ibatch, 0, nbas,
                                             shl_info, screen_buf);
                }
                npair = 0;
                nnz = 0;
                int jsh, n, m;
                for (ish = 0; ish < nbas; ish++) {
                        row_pos[ish] = 0;
                }
                for (ish = 0; ish < nbas; ish++) {
                        n = sgxopt->fscreen_shl(sgxopt, ish + 1, ish, nbas, 0, 0,
                                                shl_info, screen_buf, sj_shells);
                        for (m = 0; m < n; m++) {
                                jsh = sj_shells[m];
                                pair_i[npair] = ish;
                                pair_j[npair] = jsh;
                                pair_off[npair] = nnz;
                                nnz += (size_t)(ao_loc[ish+1] - ao_loc[ish])
                                       * (ao_loc[jsh+1] - ao_loc[jsh]);
                                npair++;
                                row_pos[ish]++;
                                if (jsh != ish) {
                                        row_pos[jsh]++;
                                }
                        }
                }
                row_ptr[0] = 0;
                nrow = 0;
                for (ish = 0; ish < nbas; ish++) {
                        row_ptr[ish+1] = row_ptr[ish] + row_pos[ish];
                        if (row_pos[ish] > 0) {
                                rows[nrow++] = ish;
                        }
                        row_pos[ish] = row_ptr[ish];
                }
                for (p = 0; p < npair; p++) {
                        ish = pair_i[p];
                        jsh = pair_j[p];
                        n = row_pos[ish]++;
                        nb_col[n] = jsh;
                        nb_off[n] = pair_off[p];
                        nb_tr[n] = 0;
                        if (jsh != ish) {
                                n = row_pos[jsh]++;
                                nb_col[n] = ish;
                                nb_off[n] = pair_off[p];
                                nb_tr[n] = 1;
                        }
                }
                // the flat column lists of the row shells
                size_t voff = 0, coff = 0;
                for (r = 0; r < nrow; r++) {
                        ish = rows[r];
                        int q = 0;
                        for (n = row_ptr[ish]; n < row_ptr[ish+1]; n++) {
                                jsh = nb_col[n];
                                nb_pos[n] = q;
                                for (m = ao_loc[jsh]; m < ao_loc[jsh+1]; m++) {
                                        cols[coff + q] = m;
                                        q++;
                                }
                        }
                        row_ncol[ish] = q;
                        row_voff[ish] = voff;
                        row_coff[ish] = coff;
                        voff += (size_t)q * (ao_loc[ish+1] - ao_loc[ish]);
                        coff += q;
                }
                nnz_full = voff;
                if (nnz * dg > E_size) {
                        free(E);
                        E_size = nnz * dg;
                        E = malloc(sizeof(double) * E_size);
                }
}
                // the integrals of the batch's pairs, point-major
#pragma omp for schedule(dynamic, 4)
                for (p = 0; p < npair; p++) {
                        const int ish = pair_i[p];
                        const int jsh = pair_j[p];
                        const int di = ao_loc[ish+1] - ao_loc[ish];
                        const int dij = di * (ao_loc[jsh+1] - ao_loc[jsh]);
                        const int dims[] = {di, ao_loc[jsh+1] - ao_loc[jsh], dg};
                        shls[0] = ish;
                        shls[1] = jsh;
                        shls[2] = ig0;
                        shls[3] = ig0 + dg;
                        (*intor)(buf, dims, shls, atm, natm, bas, nbas, env,
                                 cintopt, cache);
                        for (k = 0; k < dg; k++) {
                                double *ek = E + k * nnz + pair_off[p];
                                for (ij = 0; ij < dij; ij++) {
                                        ek[ij] = buf[ij * dg + k];
                                }
                        }
                }
                // G of the batch's points, by point: its A by row shell (V), then 8
                // densities at a time, every row shell
                if (nnz_full > V_size) {
                        free(V);
                        V_size = nnz_full;
                        V = malloc(sizeof(double) * V_size);
                }
#pragma omp for schedule(dynamic, 1)
                for (k = 0; k < dg; k++) {
                        double *Ek = E + k * nnz;
                        double *Fk = F + (ig0 + k) * nao * nset;
                        double *Gk = G + (ig0 + k) * nao * nset;
                        int n, ii, jj;
                        for (r = 0; r < nrow; r++) {
                                const int ish = rows[r];
                                const int di = ao_loc[ish+1] - ao_loc[ish];
                                double *vr = V + row_voff[ish];
                                for (n = row_ptr[ish]; n < row_ptr[ish+1]; n++) {
                                        const int dj = ao_loc[nb_col[n]+1] - ao_loc[nb_col[n]];
                                        const double *e = Ek + nb_off[n];
                                        double *v = vr + (size_t)nb_pos[n] * di;
                                        if (nb_tr[n]) {
                                                for (jj = 0; jj < dj; jj++) {
                                                for (ii = 0; ii < di; ii++) {
                                                        v[jj*di+ii] = e[ii*dj+jj];
                                                } }
                                        } else {
                                                memcpy(v, e, sizeof(double) * di * dj);
                                        }
                                }
                        }
                        // 8 densities of F (nao lines) stay in L1 over the rows
                        for (xc0 = 0; xc0 < nset; xc0 += XC) {
                                for (r = 0; r < nrow; r++) {
                                        const int ish = rows[r];
                                        const int di = ao_loc[ish+1] - ao_loc[ish];
                                        row_fn[di](Gk, Fk, V + row_voff[ish],
                                                   cols + row_coff[ish], row_ncol[ish],
                                                   nset, xc0, ao_loc[ish], di);
                                }
                        }
                }
        }
        free(buf);
        free(cache);
        free(V);
}
        free(E);
        free(pair_i);
        free(pair_off);
        free(row_ptr);
        free(nb_col);
        free(nb_off);
        free(row_voff);
        free(row_ncol);
        free(nb_pos);
        free(cols);
        free(shl_info);
        free(screen_buf);
        free(sj_shells);
        return 0;
}
