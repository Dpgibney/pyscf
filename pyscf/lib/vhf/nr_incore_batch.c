/* Copyright 2014-2026 The PySCF Developers. All Rights Reserved.

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
 * J and K matrices of a batch of density matrices (e.g. the trial densities
 * of TDDFT or CPHF) from 8-fold symmetric ERIs held in memory, as BLAS-3
 * products. The per-dm kernels of nr_incore.c visit each integral once per
 * dm with scattered updates, which for many dms costs far more than the
 * flops require.
 *
 * The ERIs are V[ij,kl] = (ij|kl) for pair indices ij = i*(i+1)/2+j (i >= j),
 * a symmetric npair x npair matrix whose lower triangle is stored row by row:
 * row ij holds kl <= ij at offset ij*(ij+1)/2.
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "fblas.h"

// Tile of packed rows x columns for the J products, 512 KB
#define JTILE_ROWS      64
#define JTILE_COLS      1024

/*
 * vj[ij,n] = sum_kl V[ij,kl] dmtri[kl,n] for n < n_dm, with
 * dmtri[kl,n] = D_n[k,l] + D_n[l,k] for k > l and D_n[k,k] for k == l.
 * Rows are taken in tiles of JTILE_ROWS: for each tile P of JTILE_COLS packed
 * columns, vj[rows] += P dmtri[cols] and vj[cols] += P^T dmtri[rows]. The
 * diagonal V[ij,ij] is in both products and is halved in P.
 */
void CVHFnrs8_incore_vj_batch(double *vj, double *eri, double *dmtri,
                              int n_dm, int nao)
{
        const size_t npair = (size_t)nao * (nao + 1) / 2;
        const size_t vsize = npair * n_dm;
        const int nrowblk = (npair + JTILE_ROWS - 1) / JTILE_ROWS;
        double **vprivs = malloc(sizeof(double *) * omp_get_max_threads());
#pragma omp parallel
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        double *vpriv = calloc(vsize, sizeof(double));
        // Published here so the barrier in NPomp_dsum_reduce_inplace makes it
        // visible to the other threads.
        vprivs[omp_get_thread_num()] = vpriv;
        double *tile = malloc(sizeof(double) * JTILE_ROWS * JTILE_COLS);
        double *row, *t;
        size_t r0, r1, c0, c1, r, c, cend;
        int ib, nr, nc;
// static: each thread sums a fixed set of tiles, so vj is reproducible for a
// given number of threads. Round robin, because the row tiles grow longer.
#pragma omp for schedule(static, 1)
        for (ib = 0; ib < nrowblk; ib++) {
                r0 = (size_t)ib * JTILE_ROWS;
                r1 = MIN(r0 + JTILE_ROWS, npair);
                nr = r1 - r0;
                for (c0 = 0; c0 < r1; c0 += JTILE_COLS) {
                        c1 = MIN(c0 + JTILE_COLS, r1);
                        nc = c1 - c0;
                        for (r = r0; r < r1; r++) {
                                row = eri + r * (r + 1) / 2;
                                t = tile + (r - r0) * nc;
                                cend = MIN(c1, r + 1);
                                for (c = c0; c < cend; c++) {
                                        t[c-c0] = row[c];
                                }
                                for (; c < c1; c++) {
                                        t[c-c0] = 0;
                                }
                                if (c0 <= r && r < c1) {
                                        t[r-c0] *= .5;
                                }
                        }
                        // vpriv[r0:r1,:] += tile . dmtri[c0:c1,:]
                        dgemm_(&TRANS_N, &TRANS_N, &n_dm, &nr, &nc,
                               &D1, dmtri + c0 * n_dm, &n_dm, tile, &nc,
                               &D1, vpriv + r0 * n_dm, &n_dm);
                        // vpriv[c0:c1,:] += tile^T . dmtri[r0:r1,:]
                        dgemm_(&TRANS_N, &TRANS_T, &n_dm, &nc, &nr,
                               &D1, dmtri + r0 * n_dm, &n_dm, tile, &nc,
                               &D1, vpriv + c0 * n_dm, &n_dm);
                }
        }
        NPomp_dsum_reduce_inplace(vprivs, vsize);
        size_t i;
#pragma omp for schedule(static)
        for (i = 0; i < vsize; i++) {
                vj[i] = vprivs[0][i];
        }
        free(tile);
        free(vpriv);
}
        free(vprivs);
}

/*
 * vk_n[p,q] = sum_rs (pr|sq) D_n[r,s] for n < n_dm.
 *
 * Row ij of the packed ERIs, (ic jc|kl) for kl <= ij, read as a symmetric
 * matrix M[k,l] (k, l <= ic; M[ic,l] = 0 for l > jc) yields all 8 index
 * permutations of its integrals through two products:
 *      Z[ic] += M Y[jc],   Z[jc] += M Y[ic]   (once if ic == jc)
 * where Y[p][:,q] = (D_n[p,q], D_n[q,p]) over n and Z[p][:,q] accumulates
 * (vk_n[p,q], vk_n[q,p]). M[ic,jc] = (ij|ij) is in both products and is halved.
 * For hermi = 1 (all D_n symmetric) the transposed halves are equal, so only
 * the first half of Y and Z is needed and vk = Z + Z^T.
 *
 * y: [nao, w, nao] with w = n_dm (hermi = 1) or 2*n_dm, as described above.
 * zout: [nao, w, nao], the sum of Z over all rows.
 */
void CVHFnrs8_incore_vk_batch(double *zout, double *eri, double *y,
                              int n_dm, int nao, int hermi)
{
        const size_t Nao = nao;
        const size_t npair = Nao * (Nao + 1) / 2;
        const int w = hermi == 1 ? n_dm : n_dm * 2;
        const size_t zsize = Nao * w * Nao;
        double **zprivs = malloc(sizeof(double *) * omp_get_max_threads());
#pragma omp parallel
{
        const char SIDE_L = 'L';
        const char UPLO_U = 'U';  // M[k,l] for l <= k in C order
        const double D1 = 1;
        double *zpriv = calloc(zsize, sizeof(double));
        zprivs[omp_get_thread_num()] = zpriv;
        double *m = malloc(sizeof(double) * Nao * Nao);
        const double *row;
        double *mic;
        size_t ij;
        int ic, jc, k, l, mdim;
// static round robin, as in CVHFnrs8_incore_vj_batch
#pragma omp for schedule(static, 1)
        for (ij = 0; ij < npair; ij++) {
                ic = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                jc = ij - (size_t)ic*(ic+1)/2;
                mdim = ic + 1;
                row = eri + ij * (ij + 1) / 2;
                for (k = 0; k < ic; k++) {
                        memcpy(m + k * mdim, row + k*(k+1)/2,
                               sizeof(double) * (k + 1));
                }
                mic = m + ic * mdim;
                row += ic*(ic+1)/2;
                for (l = 0; l <= jc; l++) {
                        mic[l] = row[l];
                }
                for (; l <= ic; l++) {
                        mic[l] = 0;
                }
                mic[jc] *= .5;
                // Z[ic][:,0:mdim] += Y[jc][:,0:mdim] . M, in the column-major
                // view zpriv[ic] = (nao x w) with leading dimension nao
                dsymm_(&SIDE_L, &UPLO_U, &mdim, &w, &D1, m, &mdim,
                       y + jc * w * Nao, &nao, &D1, zpriv + ic * w * Nao, &nao);
                if (ic != jc) {
                        dsymm_(&SIDE_L, &UPLO_U, &mdim, &w, &D1, m, &mdim,
                               y + ic * w * Nao, &nao, &D1, zpriv + jc * w * Nao, &nao);
                }
        }
        NPomp_dsum_reduce_inplace(zprivs, zsize);
        size_t i;
#pragma omp for schedule(static)
        for (i = 0; i < zsize; i++) {
                zout[i] = zprivs[0][i];
        }
        free(m);
        free(zpriv);
}
        free(zprivs);
}

/*
 * J and K of a few symmetric dms (SCF: one; UHF and ROHF: the alpha and beta
 * densities) in one pass over the ERI rows: vj[ij] = sum_kl V[ij,kl]
 * dmtri[kl] as in CVHFnrs8_incore_vj_batch, and K by the formulation of
 * CVHFnrs8_incore_vk_batch for hermi = 1, with the packed row used directly
 * as the lower triangle of M:
 *      zout[ic] += M dm[jc],   zout[jc] += M dm[ic],   vk = Z + Z^T
 * The dot products are summed in 4-wide vectors, as the compiler does not
 * reorder floating-point sums by itself. The sums of all dms go side by side,
 * as one chain of dependent additions would wait for each of its own, and
 * each row is read from memory once for all dms.
 */
// aligned(8): loaded from arbitrary offsets
typedef double dvec4 __attribute__((vector_size(32), aligned(8)));

static inline __attribute__((always_inline))
double _hsum(dvec4 v)
{
        return (v[0] + v[1]) + (v[2] + v[3]);
}

// The dms that _jk_row_body takes at once. The loops over them are unrolled
// (GCC unroll 4), which keeps their sums in registers.
#define JK_NDM_CHUNK    4

/*
 * For N dms (N <= JK_NDM_CHUNK, a constant after inlining): s1[n] =
 * sum_l e[l] x1[n][l] and y1[n][l] += e[l] * xk1[n] for l < len, and if TWO
 * the same with x2, y2, xk2 and s2. Each sum goes in the same order for any N.
 */
static inline __attribute__((always_inline))
void _dot_axpy_n(const double *e, int len,
                 const double **x1, double **y1, const double *xk1, double *s1,
                 const double **x2, double **y2, const double *xk2, double *s2,
                 const int N, const int TWO)
{
        dvec4 acc1[JK_NDM_CHUNK];
        dvec4 acc2[JK_NDM_CHUNK];
        int l, n;
#pragma GCC unroll 4
        for (n = 0; n < N; n++) {
                acc1[n] = (dvec4){0};
                acc2[n] = (dvec4){0};
        }
        for (l = 0; l + 4 <= len; l += 4) {
                dvec4 e4 = *(dvec4 *)(e + l);
#pragma GCC unroll 4
                for (n = 0; n < N; n++) {
                        *(dvec4 *)(y1[n] + l) += e4 * xk1[n];
                        acc1[n] += e4 * *(dvec4 *)(x1[n] + l);
                        if (TWO) {
                                *(dvec4 *)(y2[n] + l) += e4 * xk2[n];
                                acc2[n] += e4 * *(dvec4 *)(x2[n] + l);
                        }
                }
        }
#pragma GCC unroll 4
        for (n = 0; n < N; n++) {
                s1[n] = _hsum(acc1[n]);
                if (TWO) {
                        s2[n] = _hsum(acc2[n]);
                }
        }
        for (; l < len; l++) {
#pragma GCC unroll 4
                for (n = 0; n < N; n++) {
                        y1[n][l] += e[l] * xk1[n];
                        s1[n] += e[l] * x1[n][l];
                        if (TWO) {
                                y2[n][l] += e[l] * xk2[n];
                                s2[n] += e[l] * x2[n][l];
                        }
                }
        }
}

/*
 * Row ij of the ERIs for N dms, dmtri[N,npair] and dm[N,nao,nao], into their
 * accumulators vpriv[N,npair] and zpriv[N,nao,nao]: one loop over the row for
 * J, and one over each segment of the row (k fixed) for K, while it is in
 * cache. Either accumulator can be NULL.
 */
static inline __attribute__((always_inline))
void _jk_row_body(const double *row, size_t ij, double *vpriv, double *zpriv,
                  const double *dmtri, const double *dm, size_t Nao, const int N)
{
        const size_t npair = Nao * (Nao + 1) / 2;
        const size_t nn = Nao * Nao;
        const double *x1[JK_NDM_CHUNK];
        const double *x2[JK_NDM_CHUNK];
        double *y1[JK_NDM_CHUNK];
        double *y2[JK_NDM_CHUNK];
        double xk1[JK_NDM_CHUNK];
        double xk2[JK_NDM_CHUNK];
        double s1[JK_NDM_CHUNK];
        double s2[JK_NDM_CHUNK];
        int n;
        if (vpriv != NULL) {
#pragma GCC unroll 4
                for (n = 0; n < N; n++) {
                        x1[n] = dmtri + n * npair;
                        y1[n] = vpriv + n * npair;
                        xk1[n] = x1[n][ij];
                }
                _dot_axpy_n(row, ij, x1, y1, xk1, s1, NULL, NULL, NULL, NULL, N, 0);
#pragma GCC unroll 4
                for (n = 0; n < N; n++) {
                        y1[n][ij] += s1[n] + row[ij] * xk1[n];
                }
        }
        if (zpriv == NULL) {
                return;
        }
        const int ic = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
        const int jc = ij - (size_t)ic*(ic+1)/2;
        const double *ek = row;
        double e;
        int k;
#pragma GCC unroll 4
        for (n = 0; n < N; n++) {
                x1[n] = dm + n * nn + jc * Nao;
                x2[n] = dm + n * nn + ic * Nao;
                y1[n] = zpriv + n * nn + ic * Nao;
                y2[n] = zpriv + n * nn + jc * Nao;
        }
        if (ic != jc) {
                for (k = 0; k < ic; k++) {
#pragma GCC unroll 4
                        for (n = 0; n < N; n++) {
                                xk1[n] = x1[n][k];
                                xk2[n] = x2[n][k];
                        }
                        _dot_axpy_n(ek, k, x1, y1, xk1, s1, x2, y2, xk2, s2, N, 1);
#pragma GCC unroll 4
                        for (n = 0; n < N; n++) {
                                y1[n][k] += s1[n] + ek[k] * xk1[n];
                                y2[n][k] += s2[n] + ek[k] * xk2[n];
                        }
                        ek += k + 1;
                }
                // k = ic: l <= jc, M[ic,jc] = (ij|ij) halved
                e = ek[jc] * .5;
#pragma GCC unroll 4
                for (n = 0; n < N; n++) {
                        xk1[n] = x1[n][ic];
                        xk2[n] = x2[n][ic];
                }
                _dot_axpy_n(ek, jc, x1, y1, xk1, s1, x2, y2, xk2, s2, N, 1);
#pragma GCC unroll 4
                for (n = 0; n < N; n++) {
                        y1[n][ic] += s1[n] + e * x1[n][jc];
                        y1[n][jc] += e * xk1[n];
                        y2[n][ic] += s2[n] + e * x2[n][jc];
                        y2[n][jc] += e * xk2[n];
                }
        } else {
                for (k = 0; k < ic; k++) {
#pragma GCC unroll 4
                        for (n = 0; n < N; n++) {
                                xk1[n] = x1[n][k];
                        }
                        _dot_axpy_n(ek, k, x1, y1, xk1, s1, NULL, NULL, NULL, NULL, N, 0);
#pragma GCC unroll 4
                        for (n = 0; n < N; n++) {
                                y1[n][k] += s1[n] + ek[k] * xk1[n];
                        }
                        ek += k + 1;
                }
                e = ek[jc] * .5;
#pragma GCC unroll 4
                for (n = 0; n < N; n++) {
                        xk1[n] = x1[n][ic];
                }
                _dot_axpy_n(ek, ic, x1, y1, xk1, s1, NULL, NULL, NULL, NULL, N, 0);
#pragma GCC unroll 4
                for (n = 0; n < N; n++) {
                        y1[n][ic] += s1[n] + e * xk1[n];
                }
        }
}

// Row ij for all n_dm dms, JK_NDM_CHUNK at a time
static inline __attribute__((always_inline))
void _jk_rows_body(const double *row, size_t ij, double *vpriv, double *zpriv,
                   const double *dmtri, const double *dm, int n_dm, size_t Nao)
{
        const size_t npair = Nao * (Nao + 1) / 2;
        const size_t nn = Nao * Nao;
        const double *t, *d;
        double *v, *z;
        int n0, n;
        for (n0 = 0; n0 < n_dm; n0 += n) {
                n = MIN(n_dm - n0, JK_NDM_CHUNK);
                t = dmtri + n0 * npair;
                d = dm + n0 * nn;
                v = vpriv == NULL ? NULL : vpriv + n0 * npair;
                z = zpriv == NULL ? NULL : zpriv + n0 * nn;
                switch (n) {
                case 1:
                        _jk_row_body(row, ij, v, z, t, d, Nao, 1);
                        break;
                case 2:
                        _jk_row_body(row, ij, v, z, t, d, Nao, 2);
                        break;
                case 3:
                        _jk_row_body(row, ij, v, z, t, d, Nao, 3);
                        break;
                default:
                        _jk_row_body(row, ij, v, z, t, d, Nao, JK_NDM_CHUNK);
                }
        }
}

static void _jk_rows(const double *row, size_t ij, double *vpriv, double *zpriv,
                     const double *dmtri, const double *dm, int n_dm, size_t Nao)
{
        _jk_rows_body(row, ij, vpriv, zpriv, dmtri, dm, n_dm, Nao);
}
#if defined(__x86_64__) && defined(__GNUC__)
#define HAVE_JK_AVX2
__attribute__((target("avx2,fma")))
static void _jk_rows_avx2(const double *row, size_t ij, double *vpriv, double *zpriv,
                          const double *dmtri, const double *dm, int n_dm, size_t Nao)
{
        _jk_rows_body(row, ij, vpriv, zpriv, dmtri, dm, n_dm, Nao);
}
#endif

/*
 * J and K of n_dm symmetric dms: vj[n_dm,npair] from dmtri[n_dm,npair]
 * (dmtri as in CVHFnrs8_incore_vj_batch), and zout[n_dm,nao,nao] from
 * dm[n_dm,nao,nao], with vk = Z + Z^T. Either output can be NULL to skip J
 * or K. The results of each dm do not depend on n_dm (bit for bit, for a
 * given number of threads).
 */
void CVHFnrs8_incore_jk_ndm(double *vj, double *zout, double *eri,
                            double *dmtri, double *dm, int n_dm, int nao)
{
        const size_t Nao = nao;
        const size_t npair = Nao * (Nao + 1) / 2;
        const size_t vsize = npair * n_dm;
        const size_t zsize = Nao * Nao * n_dm;
        void (*frow)(const double *, size_t, double *, double *,
                     const double *, const double *, int, size_t) = _jk_rows;
#ifdef HAVE_JK_AVX2
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
                frow = _jk_rows_avx2;
        }
#endif
        double **vprivs = malloc(sizeof(double *) * omp_get_max_threads() * 2);
        double **zprivs = vprivs + omp_get_max_threads();
#pragma omp parallel
{
        int thread_id = omp_get_thread_num();
        double *vpriv = vj == NULL ? NULL : calloc(vsize, sizeof(double));
        double *zpriv = zout == NULL ? NULL : calloc(zsize, sizeof(double));
        vprivs[thread_id] = vpriv;
        zprivs[thread_id] = zpriv;
        size_t ij, i;
// static round robin, as in CVHFnrs8_incore_vj_batch
#pragma omp for schedule(static, 1)
        for (ij = 0; ij < npair; ij++) {
                (*frow)(eri + ij * (ij + 1) / 2, ij, vpriv, zpriv, dmtri, dm, n_dm, Nao);
        }
        if (vpriv != NULL) {
                NPomp_dsum_reduce_inplace(vprivs, vsize);
#pragma omp for schedule(static)
                for (i = 0; i < vsize; i++) {
                        vj[i] = vprivs[0][i];
                }
        }
        if (zpriv != NULL) {
                NPomp_dsum_reduce_inplace(zprivs, zsize);
#pragma omp for schedule(static)
                for (i = 0; i < zsize; i++) {
                        zout[i] = zprivs[0][i];
                }
        }
        free(vpriv);
        free(zpriv);
}
        free(vprivs);
}
