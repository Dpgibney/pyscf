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
 * Groups of grid blocks for the BLAS level 3 sparse XC kernels
 * (nr_numint_sparse.c, nr_numint_fused.c): the shells unscreened on any of
 * GEMM_NBLK consecutive blocks of BLKSIZE grid points, and the scatter and
 * contraction helpers for their gathered buffers. The including file defines
 * BLKSIZE, MIN and MAX.
 */

#ifndef HAVE_DEFINED_NR_NUMINT_GEMM_H
#define HAVE_DEFINED_NR_NUMINT_GEMM_H

#include <stdlib.h>
#include <stdint.h>

#define GEMM_NBLK       4
#define GEMM_NGRIDS     (GEMM_NBLK*BLKSIZE)

typedef struct {
        int nao;            // number of AOs of the unscreened shells
        int nshl;           // number of unscreened shells
        int nseg;           // number of contiguous AO ranges they form
        int *shl;           // [nshl] shell ids, ascending
        uint8_t *blkmask;   // [nshl] bit b: the shell is unscreened on block blk0+b
        int *seg_ao;        // [nseg] first AO of each range
        int *seg_off;       // [nseg] position of that AO in the gathered buffers
        int *seg_len;       // [nseg] number of AOs in the range
} GemmGroup;

static inline void _gemm_group_init(GemmGroup *g, int nbas)
{
        g->shl = malloc(sizeof(int) * nbas * 4);
        g->seg_ao = g->shl + nbas;
        g->seg_off = g->seg_ao + nbas;
        g->seg_len = g->seg_off + nbas;
        g->blkmask = malloc(sizeof(uint8_t) * nbas);
}

static inline void _gemm_group_free(GemmGroup *g)
{
        free(g->shl);
        free(g->blkmask);
}

// The shells that are unscreened on any of the blocks [blk0, blk1)
static inline void _gemm_group_shells(GemmGroup *g, uint8_t *screen_index,
                                      int blk0, int blk1, int nbas, int *ao_loc)
{
        int ish, b, mask;
        int nao = 0;
        int nshl = 0;
        int nseg = 0;
        for (ish = 0; ish < nbas; ish++) {
                mask = 0;
                for (b = 0; b < blk1 - blk0; b++) {
                        if (screen_index[(blk0+b)*nbas+ish]) {
                                mask |= 1 << b;
                        }
                }
                if (mask == 0) {
                        continue;
                }
                if (nseg == 0 ||
                    g->seg_ao[nseg-1] + g->seg_len[nseg-1] != ao_loc[ish]) {
                        g->seg_ao[nseg] = ao_loc[ish];
                        g->seg_off[nseg] = nao;
                        g->seg_len[nseg] = 0;
                        nseg++;
                }
                g->seg_len[nseg-1] += ao_loc[ish+1] - ao_loc[ish];
                g->shl[nshl] = ish;
                g->blkmask[nshl] = mask;
                nshl++;
                nao += ao_loc[ish+1] - ao_loc[ish];
        }
        g->nao = nao;
        g->nshl = nshl;
        g->nseg = nseg;
}

// The largest number of AOs of the unscreened shells over all groups of blocks
static inline int _gemm_max_group_aos(uint8_t *screen_index, int ngrids, int nbas,
                                      int *ao_loc)
{
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        const int ngroup = (nblk + GEMM_NBLK - 1) / GEMM_NBLK;
        int max_nao = 0;
        int igroup;
#pragma omp parallel for schedule(static) reduction(max:max_nao)
        for (igroup = 0; igroup < ngroup; igroup++) {
                int blk0 = igroup * GEMM_NBLK;
                int blk1 = MIN(blk0 + GEMM_NBLK, nblk);
                int nao = 0;
                int ish, b;
                for (ish = 0; ish < nbas; ish++) {
                        for (b = blk0; b < blk1; b++) {
                                if (screen_index[b*nbas+ish]) {
                                        nao += ao_loc[ish+1] - ao_loc[ish];
                                        break;
                                }
                        }
                }
                max_nao = MAX(max_nao, nao);
        }
        return max_nao;
}

// out[i,j] += cbuf[n,m] for the n-th AO i and the m-th AO j of the group;
// with tril, only for m <= n (j <= i)
static inline void _gemm_scatter_add(double *out, double *cbuf, GemmGroup *g,
                                     size_t Nao, int tril)
{
        const int ns = g->nao;
        int s, s1, i, m, m0, mlen;
        int n = 0;
        double *pout, *pc;
        for (s = 0; s < g->nseg; s++) {
        for (i = g->seg_ao[s]; i < g->seg_ao[s] + g->seg_len[s]; i++, n++) {
                pc = cbuf + n * (size_t)ns;
                pout = out + i * Nao;
                for (s1 = 0; s1 < g->nseg; s1++) {
                        m0 = g->seg_off[s1];
                        mlen = g->seg_len[s1];
                        if (tril) {
                                if (m0 > n) {
                                        break;
                                }
                                mlen = MIN(mlen, n - m0 + 1);
                        }
                        for (m = 0; m < mlen; m++) {
                                pout[g->seg_ao[s1]+m] += pc[m0+m];
                        }
                }
        } }
}

// rho[k] = sum_n a[n,k] * c[n,k] for the gathered buffers a and c
static inline void _gemm_contract_buf(double *rho, double *a, double *c, int ns, int ng)
{
        int n, k;
        double *pa, *pc;
        for (k = 0; k < ng; k++) {
                rho[k] = 0;
        }
        for (n = 0; n < ns; n++) {
                pa = a + n * (size_t)ng;
                pc = c + n * (size_t)ng;
                for (k = 0; k < ng; k++) {
                        rho[k] += pa[k] * pc[k];
                }
        }
}

#endif
