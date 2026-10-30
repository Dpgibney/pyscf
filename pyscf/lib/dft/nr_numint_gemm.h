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
#include <string.h>

// Blocks per group: the kernels gather the AOs unscreened on any block of a group and run one GEMM
// over its GEMM_NGRIDS points. Two blocks (112 points) keep a thread's AO buffers within L2.
#define GEMM_NBLK       2
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

/*
 * A deterministic dynamic schedule for the kernels that add the products of the
 * groups into partial matrices: _dot_ao_ao_gemm (nr_numint_sparse.c),
 * VXCfused_dot_wv, VXCfused_dot_terms, VXCfused_dot_ao_mo and
 * VXCfused_fxc_lowrank (nr_numint_fused.c). The groups go in slots of
 * SLOT_GROUPS consecutive groups, and the slots in nstream contiguous streams,
 * each with a partial matrix of its own. A thread takes the next slot of an
 * unlocked stream (the one with the most slots left, or its previous one, see
 * _slot_take) and adds its products to that stream's matrix while it holds the
 * stream's lock. Each stream is thus summed
 * in the order of its slots, whichever threads take them, and the kernels add
 * the streams' matrices in stream order (_slot_streams_add): the results depend
 * on the number of streams, not on the timing, as a fixed split of the groups
 * among the threads would, and the threads stay as balanced as with a dynamic
 * schedule on CPUs with faster and slower cores. With nthreads streams, a
 * thread that releases its stream would find no other unlocked, hence
 * STREAMS_PER_THREAD.
 */
#define SLOT_GROUPS             4
#define STREAMS_PER_THREAD      2
#define SLOT_MARGIN             1
#define SLOT_SUM_CHUNK          512

typedef struct {
        int nstream;
        int *begin;     // [nstream] the first slot of each stream
        int *next;      // [nstream] the next slot of each stream
        int *end;       // [nstream] the end of its slots
#ifdef _OPENMP
        omp_lock_t *lock;
#endif
} SlotStreams;

// The number of streams (partial matrices) for ngroup groups
static inline int _slot_nstreams(int ngroup)
{
        int nslot = (ngroup + SLOT_GROUPS - 1) / SLOT_GROUPS;
#ifdef _OPENMP
        int nthreads = omp_get_max_threads();
#else
        int nthreads = 1;
#endif
        return MAX(1, MIN(nslot, STREAMS_PER_THREAD * nthreads));
}

static inline void _slot_streams_init(SlotStreams *st, int ngroup, int nstream)
{
        const size_t nslot = (ngroup + SLOT_GROUPS - 1) / SLOT_GROUPS;
        int b;
        st->nstream = nstream;
        st->begin = malloc(sizeof(int) * nstream * 3);
        st->next = st->begin + nstream;
        st->end = st->next + nstream;
        for (b = 0; b < nstream; b++) {
                st->begin[b] = nslot * b / nstream;
                st->next[b] = st->begin[b];
                st->end[b] = nslot * (b+1) / nstream;
        }
#ifdef _OPENMP
        st->lock = malloc(sizeof(omp_lock_t) * nstream);
        for (b = 0; b < nstream; b++) {
                omp_init_lock(st->lock + b);
        }
#endif
}

static inline void _slot_streams_free(SlotStreams *st)
{
#ifdef _OPENMP
        int b;
        for (b = 0; b < st->nstream; b++) {
                omp_destroy_lock(st->lock + b);
        }
        free(st->lock);
#endif
        free(st->begin);
}

/*
 * Locks a stream with slots left and takes its next slot: returns the slot,
 * whose groups are [slot*SLOT_GROUPS, (slot+1)*SLOT_GROUPS), and sets *stream;
 * or returns -1 when no slot is left. The caller adds the slot's products to
 * the stream's matrix and then calls _slot_release. On entry *stream is the
 * stream of the caller's previous slot (-1 for none): the caller stays on it
 * unless another stream has more than SLOT_MARGIN slots more left, which keeps
 * the matrix it adds to in its cache more often. skip[nstream] is scratch of
 * the calling thread.
 */
static inline int _slot_take(SlotStreams *st, int *stream, char *skip)
{
        const int nstream = st->nstream;
        const int prev = *stream;
        int b, best, most, left, any, nx, slot, prev_left;
        for (b = 0; b < nstream; b++) {
                skip[b] = 0;
        }
        while (1) {
                best = -1;
                most = 0;
                any = 0;
                prev_left = 0;
                for (b = 0; b < nstream; b++) {
#pragma omp atomic read
                        nx = st->next[b];
                        left = st->end[b] - nx;
                        if (left > 0) {
                                any = 1;
                                if (!skip[b] && left > most) {
                                        most = left;
                                        best = b;
                                }
                                if (b == prev && !skip[b]) {
                                        prev_left = left;
                                }
                        }
                }
                if (!any) {
                        return -1;
                }
                if (best < 0) {
                        // every stream with slots left is locked: wait for one
#if defined(__x86_64__) || defined(__i386__)
                        __builtin_ia32_pause();
#endif
                        for (b = 0; b < nstream; b++) {
                                skip[b] = 0;
                        }
                        continue;
                }
                if (prev_left > 0 && prev_left >= most - SLOT_MARGIN) {
                        best = prev;
                }
#ifdef _OPENMP
                if (!omp_test_lock(st->lock + best)) {
                        skip[best] = 1;
                        continue;
                }
#endif
                slot = st->next[best];
                if (slot < st->end[best]) {
#pragma omp atomic write
                        st->next[best] = slot + 1;
                        *stream = best;
                        return slot;
                }
#ifdef _OPENMP
                omp_unset_lock(st->lock + best);
#endif
        }
}

static inline void _slot_release(SlotStreams *st, int stream)
{
#ifdef _OPENMP
        omp_unset_lock(st->lock + stream);
#endif
}

// out[i] += sum_b parts[b][off+i] for i < n, the streams' matrices added in
// stream order
static inline void _slot_streams_add(double *out, double **parts, int nstream,
                                     size_t off, size_t n)
{
        double tmp[SLOT_SUM_CHUNK];
        double *p;
        size_t i0, i, m;
        int b;
        for (i0 = 0; i0 < n; i0 += SLOT_SUM_CHUNK) {
                m = MIN(SLOT_SUM_CHUNK, n - i0);
                p = parts[0] + off + i0;
                for (i = 0; i < m; i++) {
                        tmp[i] = p[i];
                }
                for (b = 1; b < nstream; b++) {
                        p = parts[b] + off + i0;
                        for (i = 0; i < m; i++) {
                                tmp[i] += p[i];
                        }
                }
                for (i = 0; i < m; i++) {
                        out[i0+i] += tmp[i];
                }
        }
}

// The streams' matrices of n doubles each, allocated separately (malloc reuses
// freed ones of that size) and not zeroed: _slot_part zeroes each at its stream's
// first slot, on the thread that then adds to it
static inline double **_slot_parts_alloc(int nstream, size_t n)
{
        double **parts = malloc(sizeof(double *) * nstream);
        int b;
        for (b = 0; b < nstream; b++) {
                parts[b] = malloc(sizeof(double) * n);
        }
        return parts;
}

static inline void _slot_parts_free(double **parts, int nstream)
{
        int b;
        for (b = 0; b < nstream; b++) {
                free(parts[b]);
        }
        free(parts);
}

// The matrix of n doubles that the slot taken by _slot_take adds to; zeroed if
// the slot is the first of its stream
static inline double *_slot_part(SlotStreams *st, double **parts, int stream, int slot,
                                 size_t n)
{
        if (slot == st->begin[stream]) {
                memset(parts[stream], 0, sizeof(double) * n);
        }
        return parts[stream];
}

#endif
