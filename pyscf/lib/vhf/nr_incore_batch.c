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
