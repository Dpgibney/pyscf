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
 * Direct J and K of one or several density matrices at once (SCF, UKS, the
 * trial densities of TDDFT or CPHF), from 8-fold symmetric ERIs computed shell
 * quartet by shell quartet. CVHFnr_direct_drv contracts each quartet with one
 * kernel call per dm and per J/K script; here each quartet is contracted with
 * all dms in one pass, the dm index being the innermost dimension:
 * dm[p,q,n] = D_n[p,q] for n < npad, and likewise for the outputs. Several
 * dms are taken DM_CHUNK at a time, as vectors that stay in registers (npad a
 * multiple of DM_CHUNK, padded with zeros); a single dm has npad = 1.
 *
 * A quartet (ish jsh|ksh lsh), ish >= jsh, ksh >= lsh, (ish,jsh) >= (ksh,lsh),
 * stands for the index permutations that map it to another quartet:
 * sij = ish != jsh, skl = ksh != lsh, sbk = (ish,jsh) != (ksh,lsh). Within a
 * quartet with ish == jsh, both orders of each AO pair are separate elements.
 *
 * J is symmetric. Each element (ij|kl) adds e*Ds[k,l] to VJ[i,j] and, if sbk,
 * e*Ds[i,j] to VJ[k,l], with Ds = D + D^T; VJ is filled in shell blocks
 * ish >= jsh only and vj = VJ + VJ^T. The factor 1/2 for ish == jsh (or
 * ksh == lsh) accounts for both AO orders being present as elements.
 *
 * K: for each element, with flags as weights,
 *      t1 VK[i,l] += e D[j,k]                   t5 VK[k,j] += e D[l,i]   sbk
 *      t2 VK[j,l] += e D[i,k]   sij             t6 VK[l,j] += e D[k,i]   sbk skl
 *      t3 VK[i,k] += e D[j,l]   skl             t7 VK[k,i] += e D[l,j]   sbk sij
 *      t4 VK[j,k] += e D[i,l]   sij skl         t8 VK[l,i] += e D[k,j]   sbk sij skl
 * For symmetric dms (hermi=1), t5..t8 are the transposes of t4..t1 when
 * sbk, and are the t1..t4 of other elements of the quartet otherwise. So
 * VK = t1..t4 (times 1/2 if not sbk) and vk = VK + VK^T.
 *
 * Semi-direct: a CVHFsdCache keeps the integrals of a set of quartets (see
 * CVHFsd_index) from the call that first evaluates them, and later calls read
 * them instead. The screening is the same with and without it, so a call
 * contracts the same integrals either way.
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include "config.h"
#include "cint.h"
#include "optimizer.h"
#include "np_helper/np_helper.h"
#include "gto/gto.h"
#include "fblas.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define DM_CHUNK        4
// aligned(8): the vectors are loaded from arbitrary dm offsets
typedef double dvec __attribute__((vector_size(DM_CHUNK*8), aligned(8)));

typedef struct {
        int npad;       // number of dms, padded to a multiple of DM_CHUNK
        size_t nao;
        double *dm;     // D[p,q,n]
        double *dms;    // D[p,q,n] + D[q,p,n], for J
        double *vj;     // thread-private sums
        double *vk;
} JKBatch;

#define DECLARE_QUARTET \
        const int npad = b->npad; \
        const size_t S = b->nao * npad; \
        const int i0 = ao_loc[ish]; \
        const int j0 = ao_loc[jsh]; \
        const int k0 = ao_loc[ksh]; \
        const int l0 = ao_loc[lsh]; \
        const int i1 = ao_loc[ish+1]; \
        const int j1 = ao_loc[jsh+1]; \
        const int k1 = ao_loc[ksh+1]; \
        const int l1 = ao_loc[lsh+1]

/* The kernels for dms in chunks of type VT: a vector of DM_CHUNK doubles, or a
 * double for a single dm (npad = 1) */
#define DEF_BATCH_KERNELS(SFX, VT) \
static inline __attribute__((always_inline))                                        \
void _batch_j_body##SFX(double *eri, JKBatch *b, int *ao_loc,                       \
                   int ish, int jsh, int ksh, int lsh)                              \
{                                                                                   \
        DECLARE_QUARTET;                                                            \
        const double c = (ish != jsh ? 1. : .5) * (ksh != lsh ? 1. : .5);           \
        const int sbk = ish != ksh || jsh != lsh;                                   \
        double *pe;                                                                 \
        double e;                                                                   \
        VT dkl, acc;                                                                \
        int i, j, k, l, n;                                                          \
        for (n = 0; n < npad; n += (int)(sizeof(VT) / sizeof(double))) {            \
                const double *dms = b->dms + n;                                     \
                double *vj = b->vj + n;                                             \
                pe = eri;                                                           \
                for (l = l0; l < l1; l++) {                                         \
                for (k = k0; k < k1; k++) {                                         \
                        dkl = *(VT *)(dms + k * S + l * npad);                      \
                        acc = (VT){0};                                              \
                        for (j = j0; j < j1; j++) {                                 \
                        for (i = i0; i < i1; i++, pe++) {                           \
                                e = *pe * c;                                        \
                                *(VT *)(vj + i * S + j * npad) += e * dkl;          \
                                acc += e * *(VT *)(dms + i * S + j * npad);         \
                        } }                                                         \
                        if (sbk) {                                                  \
                                *(VT *)(vj + k * S + l * npad) += acc;              \
                        }                                                           \
                } }                                                                 \
        }                                                                           \
}                                                                                   \
                                                                                    \
static inline __attribute__((always_inline))                                        \
void _batch_k_body##SFX(double *eri, JKBatch *b, int *ao_loc, int hermi,            \
                   int ish, int jsh, int ksh, int lsh)                              \
{                                                                                   \
        DECLARE_QUARTET;                                                            \
        const double sij = ish != jsh;                                              \
        const double skl = ksh != lsh;                                              \
        const double sbk = ish != ksh || jsh != lsh;                                \
        /* hermi=1: t1..t4 only, halved if not sbk */                               \
        const double c = (hermi == 1 && !sbk) ? .5 : 1.;                            \
        const double w2 = sij;                                                      \
        const double w3 = skl;                                                      \
        const double w4 = sij * skl;                                                \
        const double w5 = sbk;                                                      \
        const double w6 = sbk * skl;                                                \
        const double w7 = sbk * sij;                                                \
        const double w8 = sbk * sij * skl;                                          \
        double *pe;                                                                 \
        double e;                                                                   \
        VT dJK, dJL, dLJ, dKJ, a2, a4, a5, a6;                                      \
        int i, j, k, l, n;                                                          \
        for (n = 0; n < npad; n += (int)(sizeof(VT) / sizeof(double))) {            \
                const double *dm = b->dm + n;                                       \
                double *vk = b->vk + n;                                             \
                pe = eri;                                                           \
                for (l = l0; l < l1; l++) {                                         \
                for (k = k0; k < k1; k++) {                                         \
                for (j = j0; j < j1; j++) {                                         \
                        dJK = *(VT *)(dm + j * S + k * npad);                       \
                        dJL = *(VT *)(dm + j * S + l * npad) * w3;                  \
                        a2 = (VT){0};                                               \
                        a4 = (VT){0};                                               \
                        if (hermi == 1) {                                           \
                                for (i = i0; i < i1; i++, pe++) {                   \
                                        e = *pe * c;                                \
                                        *(VT *)(vk + i * S + l * npad) += e * dJK;  \
                                        *(VT *)(vk + i * S + k * npad) += e * dJL;  \
                                        a2 += e * *(VT *)(dm + i * S + k * npad);   \
                                        a4 += e * *(VT *)(dm + i * S + l * npad);   \
                                }                                                   \
                        } else {                                                    \
                                dLJ = *(VT *)(dm + l * S + j * npad) * w7;          \
                                dKJ = *(VT *)(dm + k * S + j * npad) * w8;          \
                                a5 = (VT){0};                                       \
                                a6 = (VT){0};                                       \
                                for (i = i0; i < i1; i++, pe++) {                   \
                                        e = *pe;                                    \
                                        *(VT *)(vk + i * S + l * npad) += e * dJK;  \
                                        *(VT *)(vk + i * S + k * npad) += e * dJL;  \
                                        *(VT *)(vk + k * S + i * npad) += e * dLJ;  \
                                        *(VT *)(vk + l * S + i * npad) += e * dKJ;  \
                                        a2 += e * *(VT *)(dm + i * S + k * npad);   \
                                        a4 += e * *(VT *)(dm + i * S + l * npad);   \
                                        a5 += e * *(VT *)(dm + l * S + i * npad);   \
                                        a6 += e * *(VT *)(dm + k * S + i * npad);   \
                                }                                                   \
                                *(VT *)(vk + k * S + j * npad) += w5 * a5;          \
                                *(VT *)(vk + l * S + j * npad) += w6 * a6;          \
                        }                                                           \
                        *(VT *)(vk + j * S + l * npad) += w2 * a2;                  \
                        *(VT *)(vk + j * S + k * npad) += w4 * a4;                  \
                } } }                                                               \
        }                                                                           \
}

DEF_BATCH_KERNELS(_v, dvec)
DEF_BATCH_KERNELS(_s, double)

// zeroed thread-private sums; huge pages where the OS offers them, as the
// kernels update rows and columns of these arrays
static double *_alloc_sums(size_t n)
{
        void *p = NULL;
        size_t bytes = n * sizeof(double);
#ifdef MADV_HUGEPAGE
        const size_t huge = 2 << 20;
        if (bytes >= huge) {
                bytes = (bytes + huge - 1) / huge * huge;
                if (posix_memalign(&p, huge, bytes) != 0) {
                        return NULL;
                }
                madvise(p, bytes, MADV_HUGEPAGE);
                memset(p, 0, bytes);
                return p;
        }
#endif
        return calloc(n, sizeof(double));
}

/*
 * The stored quartets of pair row ij = ish*(ish+1)/2+jsh are row_q[ij] <= n <
 * row_q[ij+1], in the order of the loop of CVHFnr_direct_jk_batch, with
 * shls[n] = ksh << 16 | lsh. Their integral blocks follow each other from
 * eri + row_d[ij]. state[n] is 0 until a call evaluates the quartet, then 1,
 * or 2 if intor returned 0 (no integrals above its cutoffs). Every quartet
 * with q_ij q_kl > thr is stored (thr = inf if that does not apply).
 */
typedef struct {
        double thr;
        int64_t *row_q;
        int64_t *row_d;
        int32_t *shls;
        int8_t *state;
        double *eri;
} CVHFsdCache;

/*
 * Several quartets of one class in the SIMD lanes: qcint's int2e_sph_batch (cint2e_batch.c) for the
 * quartets int2e_sph_batch_ok takes. Each row queues them by the types of their ket shells (l and
 * number of primitives; the bra pair is the row's), and evaluates a queue when it holds
 * int2e_sph_batch_lanes() quartets, and at the end of the row. A batch costs as much as a full one,
 * so a queue of fewer than half the lanes goes quartet by quartet. Weak: libcint and older qcint
 * have none of these functions, and then every quartet goes alone.
 */
int int2e_sph();
int int2e_sph_batch(double *out, int nq, int *shls, int *atm, int natm, int *bas, int nbas,
                    double *env, CINTOpt *opt, double *cache) __attribute__((weak));
int int2e_sph_batch_ok(int *shls, int *bas, double *env) __attribute__((weak));
int int2e_sph_batch_lanes(void) __attribute__((weak));
size_t int2e_sph_batch_cache_size(void) __attribute__((weak));

#define BATCH_MAX_LANES 8
#define BATCH_MAX_NF    2401    // (ff|ff)

typedef struct {
        int n;
        int listed;
        int shls[4 * BATCH_MAX_LANES];
        // the semi-direct cache's quartets on their first evaluation: their slots and indices
        double *slot[BATCH_MAX_LANES];
        int64_t idx[BATCH_MAX_LANES];
} QuartetQueue;

// for each ksh, max over lsh <= ksh of q_cond[ksh,lsh]
static double *_qmax_k(double *q_cond, int nbas)
{
        double *qmax_k = malloc(sizeof(double) * nbas);
        int ksh, lsh;
        for (ksh = 0; ksh < nbas; ksh++) {
                qmax_k[ksh] = 0;
                for (lsh = 0; lsh <= ksh; lsh++) {
                        qmax_k[ksh] = MAX(qmax_k[ksh], q_cond[ksh*nbas+lsh]);
                }
        }
        return qmax_k;
}

/*
 * Which quartets a CVHFsdCache keeps when it cannot keep all that pass the
 * Schwarz test (q_ij q_kl > cutoff): those with the largest value, the time
 * their evaluation takes per integral, times the share of the SCF cycles
 * expected to need them. The time is a fit to qcint's timings (one Zen 5
 * core) over the classes of def2-SVP/TZVP and cc-pVDZ molecules, in ns:
 *      80 + nprim (9 + .5 nroots g + .15 nroots ncart) + 1.3 ncart
 * with nprim the primitive combinations, nroots the Rys roots, g the size of
 * the 2D integrals, ncart the Cartesian functions of the quartet. Ranked by
 * this estimate, a cache of 14 GB holds 0.60 of the integral time of
 * caffeine/def2-TZVP (0.62 ranked by the measured times, 0.30 ranked by
 * q_ij q_kl). The share of the cycles is 1 from q_ij q_kl = 1e6 cutoff down,
 * falling linearly in log(q_ij q_kl) to .05 at the cutoff.
 * Values are binned in SD_BINS bins of 1/64 of an octave of a piecewise
 * linear log2 from 2^-32 on.
 */
#define SD_BINS         4096

// monotonic, piecewise linear in x and exact at powers of 2 (x > 0)
static inline double _sd_log2(double x)
{
        union {double d; uint64_t i;} u = {x};
        int e = (int)((u.i >> 52) & 0x7ff) - 1023;
        u.i = (u.i & 0x000fffffffffffffull) | 0x3ff0000000000000ull;
        return e + (u.d - 1);
}

typedef struct {
        double lq0;     // _sd_log2 of the cutoff
        double *lq;     // [nbas*nbas] _sd_log2(q_cond)
        int *l;         // [nbas] angular momentum
        double *nprim;
        double *ncart;
        double *nf;     // functions, ao_loc[i+1] - ao_loc[i]
} SDValue;

static void _sd_value_init(SDValue *v, double *q_cond, double cutoff,
                           int *ao_loc, int *bas, int nbas)
{
        size_t i;
        v->lq0 = _sd_log2(cutoff);
        v->lq = malloc(sizeof(double) * nbas * nbas);
        for (i = 0; i < (size_t)nbas * nbas; i++) {
                v->lq[i] = q_cond[i] > 0 ? _sd_log2(q_cond[i]) : -2000.;
        }
        v->l = malloc(sizeof(int) * nbas);
        v->nprim = malloc(sizeof(double) * nbas * 3);
        v->ncart = v->nprim + nbas;
        v->nf = v->ncart + nbas;
        for (i = 0; i < (size_t)nbas; i++) {
                int l = bas[ANG_OF+i*BAS_SLOTS];
                v->l[i] = l;
                v->nprim[i] = bas[NPRIM_OF+i*BAS_SLOTS];
                v->ncart[i] = (l + 1) * (l + 2) / 2;
                v->nf[i] = ao_loc[i+1] - ao_loc[i];
        }
}

static void _sd_value_del(SDValue *v)
{
        free(v->lq);
        free(v->l);
        free(v->nprim);
}

static inline int _sd_bin(SDValue *v, int nbas, int i, int j, int k, int l)
{
        double nprim = v->nprim[i] * v->nprim[j] * v->nprim[k] * v->nprim[l];
        double ncart = v->ncart[i] * v->ncart[j] * v->ncart[k] * v->ncart[l];
        double nroots = (v->l[i] + v->l[j] + v->l[k] + v->l[l]) / 2 + 1;
        double g = (v->l[i] + v->l[j] + 1) * (v->l[k] + v->l[l] + 1);
        double t = 80 + nprim * (9 + .5 * nroots * g + .15 * nroots * ncart) + 1.3 * ncart;
        // 6 decades of q_ij q_kl
        double u = (v->lq[i*nbas+j] + v->lq[k*nbas+l] - v->lq0) * (1 / 19.93);
        u = MIN(MAX(u, .05), 1.);
        int b = (int)((_sd_log2(t * u / (v->nf[i] * v->nf[j] * v->nf[k] * v->nf[l])) + 32) * 64);
        return MIN(MAX(b, 0), SD_BINS - 1);
}

/*
 * hist[2*b], hist[2*b+1]: the integrals and the number of the quartets with
 * q_ij q_kl > cutoff in value bin b (see above)
 */
void CVHFsd_histogram(double *hist, double *q_cond, double cutoff,
                      int *ao_loc, int *bas, int nbas)
{
        const size_t npair = (size_t)nbas * (nbas + 1) / 2;
        double *qmax_k = _qmax_k(q_cond, nbas);
        SDValue v;
        _sd_value_init(&v, q_cond, cutoff, ao_loc, bas, nbas);
        memset(hist, 0, sizeof(double) * SD_BINS * 2);
#pragma omp parallel
{
        double *h = calloc(SD_BINS * 2, sizeof(double));
        size_t ij;
        int b;
#pragma omp for schedule(dynamic, 4)
        for (ij = 0; ij < npair; ij++) {
                int ish = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                int jsh = ij - (size_t)ish*(ish+1)/2;
                double qij = q_cond[ish*nbas+jsh];
                int ksh, lsh, lsh1;
                for (ksh = 0; ksh <= ish; ksh++) {
                        if (!(qij * qmax_k[ksh] > cutoff)) {
                                continue;
                        }
                        lsh1 = ksh < ish ? ksh + 1 : jsh + 1;
                        for (lsh = 0; lsh < lsh1; lsh++) {
                                if (qij * q_cond[ksh*nbas+lsh] > cutoff) {
                                        b = _sd_bin(&v, nbas, ish, jsh, ksh, lsh);
                                        h[b*2] += v.nf[ish] * v.nf[jsh] * v.nf[ksh] * v.nf[lsh];
                                        h[b*2+1] += 1;
                                }
                        }
                }
        }
#pragma omp critical
        {
                for (b = 0; b < SD_BINS * 2; b++) {
                        hist[b] += h[b];
                }
        }
        free(h);
}
        _sd_value_del(&v);
        free(qmax_k);
}

/*
 * The quartets a CVHFsdCache stores: those with q_ij q_kl > cutoff in value
 * bins >= bin0 (all of them if bin0 <= 0). nq[ij] and nd[ij] receive the
 * number of quartets of pair row ij and the size of their integral blocks.
 * With shls != NULL (and row_q, the prefix sums of nq), shls is filled instead.
 */
void CVHFsd_index(int64_t *nq, int64_t *nd, int32_t *shls, int64_t *row_q,
                  double *q_cond, double cutoff, int bin0,
                  int *ao_loc, int *bas, int nbas)
{
        const size_t npair = (size_t)nbas * (nbas + 1) / 2;
        double *qmax_k = _qmax_k(q_cond, nbas);
        SDValue v;
        _sd_value_init(&v, q_cond, cutoff, ao_loc, bas, nbas);
        size_t ij;
#pragma omp parallel for schedule(dynamic, 4)
        for (ij = 0; ij < npair; ij++) {
                int ish = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                int jsh = ij - (size_t)ish*(ish+1)/2;
                double qij = q_cond[ish*nbas+jsh];
                size_t dij = (size_t)(ao_loc[ish+1] - ao_loc[ish]) *
                                     (ao_loc[jsh+1] - ao_loc[jsh]);
                int64_t n = 0;
                int64_t d = 0;
                int ksh, lsh, lsh1;
                for (ksh = 0; ksh <= ish; ksh++) {
                        // q_ij q_kl <= q_ij qmax_k[ksh] for every lsh
                        if (!(qij * qmax_k[ksh] > cutoff)) {
                                continue;
                        }
                        lsh1 = ksh < ish ? ksh + 1 : jsh + 1;
                        for (lsh = 0; lsh < lsh1; lsh++) {
                                if (!(qij * q_cond[ksh*nbas+lsh] > cutoff) ||
                                    (bin0 > 0 && _sd_bin(&v, nbas, ish, jsh, ksh, lsh) < bin0)) {
                                        continue;
                                }
                                if (shls != NULL) {
                                        shls[row_q[ij]+n] = ksh << 16 | lsh;
                                } else {
                                        d += dij * (ao_loc[ksh+1] - ao_loc[ksh]) *
                                                   (ao_loc[lsh+1] - ao_loc[lsh]);
                                }
                                n++;
                        }
                }
                if (shls == NULL) {
                        nq[ij] = n;
                        nd[ij] = d;
                }
        }
        _sd_value_del(&v);
        free(qmax_k);
}

typedef void (*FBatchJ)(double *eri, JKBatch *b, int *ao_loc,
                        int ish, int jsh, int ksh, int lsh);
typedef void (*FBatchK)(double *eri, JKBatch *b, int *ao_loc, int hermi,
                        int ish, int jsh, int ksh, int lsh);

#define DEF_BATCH_WRAPPERS(SFX, BODY, ATTR) \
ATTR static void _batch_j##SFX(double *eri, JKBatch *b, int *ao_loc, \
                               int ish, int jsh, int ksh, int lsh) \
{ \
        _batch_j_body##BODY(eri, b, ao_loc, ish, jsh, ksh, lsh); \
} \
ATTR static void _batch_k##SFX(double *eri, JKBatch *b, int *ao_loc, int hermi, \
                               int ish, int jsh, int ksh, int lsh) \
{ \
        _batch_k_body##BODY(eri, b, ao_loc, hermi, ish, jsh, ksh, lsh); \
}
DEF_BATCH_WRAPPERS(_v, _v, )
DEF_BATCH_WRAPPERS(_s, _s, )
#if defined(__x86_64__) && defined(__GNUC__)
// The same kernels compiled for AVX2 and FMA, taken when the CPU has them
#define HAVE_BATCH_AVX2
#define AVX2_FMA __attribute__((target("avx2,fma")))
DEF_BATCH_WRAPPERS(_v_avx2, _v, AVX2_FMA)
DEF_BATCH_WRAPPERS(_s_avx2, _s, AVX2_FMA)
#endif

// evaluates and contracts the quartets of Q, then empties it
static void _flush_queue(QuartetQueue *Q, int lanes, int (*intor)(), FBatchJ fj, FBatchK fk,
                         JKBatch *b, int *ao_loc, int hermi, double *bout, double *bcache,
                         double *buf, double *cache, CINTOpt *cintopt,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
        int *s = Q->shls;
        int q;
        if (2 * Q->n >= lanes) {
                const size_t nf = (size_t)(ao_loc[s[0]+1] - ao_loc[s[0]]) *
                                          (ao_loc[s[1]+1] - ao_loc[s[1]]) *
                                          (ao_loc[s[2]+1] - ao_loc[s[2]]) *
                                          (ao_loc[s[3]+1] - ao_loc[s[3]]);
                int2e_sph_batch(bout, Q->n, s, atm, natm, bas, nbas, env, NULL, bcache);
                for (q = 0; q < Q->n; q++, s += 4) {
                        double *e = bout + nf * q;
                        if (b->vj != NULL) {
                                (*fj)(e, b, ao_loc, s[0], s[1], s[2], s[3]);
                        }
                        if (b->vk != NULL) {
                                (*fk)(e, b, ao_loc, hermi, s[0], s[1], s[2], s[3]);
                        }
                }
        } else {
                for (q = 0; q < Q->n; q++, s += 4) {
                        if (!(*intor)(buf, NULL, s, atm, natm, bas, nbas, env, cintopt, cache)) {
                                continue;
                        }
                        if (b->vj != NULL) {
                                (*fj)(buf, b, ao_loc, s[0], s[1], s[2], s[3]);
                        }
                        if (b->vk != NULL) {
                                (*fk)(buf, b, ao_loc, hermi, s[0], s[1], s[2], s[3]);
                        }
                }
        }
        Q->n = 0;
}

// evaluates the queue's quartets into their slots, with state[idx] 1 (or 2 if
// intor returned 0)
static void _store_queue(QuartetQueue *Q, int8_t *state, int lanes, int (*intor)(), int *ao_loc,
                         double *bout, double *bcache, double *cache, CINTOpt *cintopt,
                         int *atm, int natm, int *bas, int nbas, double *env)
{
        int *s = Q->shls;
        int q;
        if (2 * Q->n >= lanes) {
                const size_t nf = (size_t)(ao_loc[s[0]+1] - ao_loc[s[0]]) *
                                          (ao_loc[s[1]+1] - ao_loc[s[1]]) *
                                          (ao_loc[s[2]+1] - ao_loc[s[2]]) *
                                          (ao_loc[s[3]+1] - ao_loc[s[3]]);
                int2e_sph_batch(bout, Q->n, s, atm, natm, bas, nbas, env, NULL, bcache);
                for (q = 0; q < Q->n; q++) {
                        memcpy(Q->slot[q], bout + nf * q, sizeof(double) * nf);
                        state[Q->idx[q]] = 1;
                }
        } else {
                for (q = 0; q < Q->n; q++) {
                        state[Q->idx[q]] = (*intor)(Q->slot[q], NULL, s + 4 * q, atm, natm,
                                                    bas, nbas, env, cintopt, cache) ? 1 : 2;
                }
        }
}

// evaluates the queue of quartets of the semi-direct cache, stores them, and
// contracts them; then empties it
static void _flush_stored(QuartetQueue *Q, CVHFsdCache *sd, int lanes, int (*intor)(),
                          FBatchJ fj, FBatchK fk, JKBatch *b, int *ao_loc, int hermi,
                          double *bout, double *bcache, double *cache, CINTOpt *cintopt,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        int *s = Q->shls;
        int q;
        _store_queue(Q, sd->state, lanes, intor, ao_loc, bout, bcache, cache, cintopt,
                     atm, natm, bas, nbas, env);
        for (q = 0; q < Q->n; q++, s += 4) {
                if (sd->state[Q->idx[q]] == 2) {
                        continue;
                }
                if (b->vj != NULL) {
                        (*fj)(Q->slot[q], b, ao_loc, s[0], s[1], s[2], s[3]);
                }
                if (b->vk != NULL) {
                        (*fk)(Q->slot[q], b, ao_loc, hermi, s[0], s[1], s[2], s[3]);
                }
        }
        Q->n = 0;
}

/*
 * Many dms (the CPHF of Hessians: 3 natm of them; TDDFT with many roots) through dense matrix
 * products (dm_mode 1, GEMM_MIN_CHUNKS * DM_CHUNK dms or more): the shells are cut into blocks of
 * contiguous shells of about GEMM_BS AOs, and each s8-unique quartet of blocks (IJ|KL) is evaluated
 * whole into a dense matrix E[(i,j),(k,l)]. Its shell quartets are those of the one-pass loop, each
 * once and with the same screening; the copies that the symmetries of a block quartet with I == J,
 * K == L or (IJ) == (KL) bring are filled in, the rest stays 0. J and K are then six products with
 * all dms at once (n innermost):
 *   VJ[I,J] += cJ E Ds[K,L]          VJ[K,L] += cJ E^T Ds[I,J]    (if (IJ) != (KL))
 *   VK[I,L] += c X1 D[J,K]           VK[J,K] += c w4 X1^T D[I,L]
 *   VK[I,K] += c w3 X2 D[J,L]        VK[J,L] += c w2 X2^T D[I,K]
 * with X1[(i,l),(j,k)] = X2[(i,k),(j,l)] = E[(i,j),(k,l)], and the weights of the quartet kernels
 * taken at the block level (a block pair I == J holds both orders of its AO pairs, as a shell pair
 * ish == jsh does there). With hermi = 0, t5..t8 of the quartet kernels are the same four products
 * with D^T, into sums of VK^T (vk = VK + VKT^T), and nothing is halved. The one-pass kernels make a few multiply-adds per dm and quartet on rows of
 * D and VK spread over the arrays, and are bound by those accesses; the products run at matrix-multiply
 * speed. The dms and the sums are kept by block pairs, [I][J][i][j][n], shared by all threads; a
 * product goes to a thread's buffer and is added to its block of sums under that block's lock.
 */
#define GEMM_BS         16
#define GEMM_MIN_CHUNKS 4

typedef struct {
        int32_t key;
        int shls[4];
} GemmQuartet;

static int _gemm_quartet_cmp(const void *a, const void *b)
{
        int32_t ka = ((const GemmQuartet *)a)->key, kb = ((const GemmQuartet *)b)->key;
        return (ka > kb) - (ka < kb);
}

// geometry of a block quartet for _gemm_place
typedef struct {
        int a0, b0, c0, d0;     // AO offsets of blocks I, J, K, L
        int nJ, nL, nKL;
        int bra_eq, ket_eq, bk_eq;      // I == J, K == L, (IJ) == (KL)
} GemmBlockQ;

// the integrals out[l][k][j][i] of shell quartet s into E[(i,j),(k,l)], with their images under the
// symmetries of the block quartet
static void _gemm_place(double *E, const double *out, int *s, int *ao_loc, GemmBlockQ *g)
{
        const int i0 = ao_loc[s[0]] - g->a0, di = ao_loc[s[0]+1] - ao_loc[s[0]];
        const int j0 = ao_loc[s[1]] - g->b0, dj = ao_loc[s[1]+1] - ao_loc[s[1]];
        const int k0 = ao_loc[s[2]] - g->c0, dk = ao_loc[s[2]+1] - ao_loc[s[2]];
        const int l0 = ao_loc[s[3]] - g->d0, dl = ao_loc[s[3]+1] - ao_loc[s[3]];
        const int nJ = g->nJ, nL = g->nL, nKL = g->nKL;
        const int sb = g->bra_eq && s[0] != s[1];
        const int sk = g->ket_eq && s[2] != s[3];
        const int sx = g->bk_eq && (s[0] != s[2] || s[1] != s[3]);
        int a, b, c, d;
        for (d = 0; d < dl; d++) {
        for (c = 0; c < dk; c++) {
        for (b = 0; b < dj; b++) {
        for (a = 0; a < di; a++) {
                const double v = *out++;
                const int i = i0 + a, j = j0 + b, k = k0 + c, l = l0 + d;
                E[(size_t)(i * nJ + j) * nKL + k * nL + l] = v;
                if (sb) E[(size_t)(j * nJ + i) * nKL + k * nL + l] = v;
                if (sk) E[(size_t)(i * nJ + j) * nKL + l * nL + k] = v;
                if (sb && sk) E[(size_t)(j * nJ + i) * nKL + l * nL + k] = v;
                if (sx) {
                        // (kl|ij): with (IJ) == (KL), k and l index blocks I and J
                        E[(size_t)(k * nJ + l) * nKL + i * nL + j] = v;
                        if (sb) E[(size_t)(k * nJ + l) * nKL + j * nL + i] = v;
                        if (sk) E[(size_t)(l * nJ + k) * nKL + i * nL + j] = v;
                        if (sb && sk) E[(size_t)(l * nJ + k) * nKL + j * nL + i] = v;
                }
        } } } }
}

static void _gemm_add(double *dst, const double *src, size_t n, omp_lock_t *lk)
{
        size_t x;
        omp_set_lock(lk);
        for (x = 0; x < n; x++) {
                dst[x] += src[x];
        }
        omp_unset_lock(lk);
}

static void _direct_jk_gemm(int (*intor)(), double *vj, double *vk,
                            double *dm, double *dms, int npad, int hermi,
                            int *ao_loc, CINTOpt *cintopt, CVHFOpt *vhfopt,
                            int lanes, int nt, int *stype, double *q_cond,
                            double cutoff, double dmax4, int di, size_t cache_size,
                            int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t nao = ao_loc[nbas];
        int (*fprescreen)() = vhfopt->fprescreen;
        int sh, Ib, Jb;
        // blocks of contiguous shells with at most GEMM_BS AOs (or one bigger shell)
        int *bsh = malloc(sizeof(int) * (nbas + 1) * 4);
        int *bao = bsh + nbas + 1;
        int *bn = bao + nbas + 1;
        int *shblk = bn + nbas + 1;
        int nb = 0;
        bsh[0] = 0;
        for (sh = 0; sh < nbas; sh++) {
                if (sh > bsh[nb] && ao_loc[sh+1] - ao_loc[bsh[nb]] > GEMM_BS) {
                        bsh[++nb] = sh;
                }
                shblk[sh] = nb;
        }
        bsh[++nb] = nbas;
        int maxbn = 0, maxbs = 0;
        for (Ib = 0; Ib < nb; Ib++) {
                bao[Ib] = ao_loc[bsh[Ib]];
                bn[Ib] = ao_loc[bsh[Ib+1]] - bao[Ib];
                maxbn = MAX(maxbn, bn[Ib]);
                maxbs = MAX(maxbs, bsh[Ib+1] - bsh[Ib]);
        }
        // block pairs (all orders): offsets of their [i][j][n] arrays, the largest q_cond in them
        size_t *boff = malloc(sizeof(size_t) * nb * nb);
        double *QB = malloc(sizeof(double) * nb * nb);
        size_t tot = 0;
        for (Ib = 0; Ib < nb; Ib++) {
        for (Jb = 0; Jb < nb; Jb++) {
                boff[Ib*nb+Jb] = tot;
                tot += (size_t)bn[Ib] * bn[Jb] * npad;
                double q = 0;
                int ish, jsh;
                for (ish = bsh[Ib]; ish < bsh[Ib+1]; ish++) {
                for (jsh = bsh[Jb]; jsh < bsh[Jb+1]; jsh++) {
                        q = MAX(q, q_cond[ish*nbas+jsh]);
                } }
                QB[Ib*nb+Jb] = q;
        } }
        double qbmax = 0;
        for (Ib = 0; Ib < nb * nb; Ib++) {
                qbmax = MAX(qbmax, QB[Ib]);
        }
        double *Dblk = malloc(sizeof(double) * tot);
        double *Dsblk = vj != NULL ? malloc(sizeof(double) * tot) : NULL;
        double *VJblk = vj != NULL ? calloc(tot, sizeof(double)) : NULL;
        double *VKblk = vk != NULL ? calloc(tot, sizeof(double)) : NULL;
        // hermi = 0: D^T by block pairs, DT[I,J][i][j] = D[j][i], and the sums of VK^T
        const int kt = vk != NULL && hermi == 0;
        double *DTblk = kt ? malloc(sizeof(double) * tot) : NULL;
        double *VKTblk = kt ? calloc(tot, sizeof(double)) : NULL;
        omp_lock_t *lk = malloc(sizeof(omp_lock_t) * nb * nb * 3);
        if (Dblk == NULL || (vj != NULL && (Dsblk == NULL || VJblk == NULL)) ||
            (vk != NULL && VKblk == NULL) || (kt && (DTblk == NULL || VKTblk == NULL)) || lk == NULL) {
                fprintf(stderr, "CVHFnr_direct_jk_batch: out of memory for %ld dms\n", (long)npad);
                exit(1);
        }
        for (Ib = 0; Ib < nb * nb * 3; Ib++) {
                omp_init_lock(lk + Ib);
        }
        const size_t nbp = (size_t)nb * (nb + 1) / 2;

#pragma omp parallel
{
        int Ib, Jb, Kb, Lb, i;
        size_t pij, pkl, it;
        // the dms by block pairs
#pragma omp for schedule(dynamic, 1)
        for (pij = 0; pij < (size_t)nb * nb; pij++) {
                Ib = pij / nb;
                Jb = pij % nb;
                for (i = 0; i < bn[Ib]; i++) {
                        size_t src = ((size_t)(bao[Ib] + i) * nao + bao[Jb]) * npad;
                        size_t dst = boff[pij] + (size_t)i * bn[Jb] * npad;
                        memcpy(Dblk + dst, dm + src, sizeof(double) * bn[Jb] * npad);
                        if (Dsblk != NULL) {
                                memcpy(Dsblk + dst, dms + src, sizeof(double) * bn[Jb] * npad);
                        }
                        if (DTblk != NULL) {
                                int j;
                                for (j = 0; j < bn[Jb]; j++) {
                                        memcpy(DTblk + dst + (size_t)j * npad,
                                               dm + ((size_t)(bao[Jb] + j) * nao + bao[Ib] + i) * npad,
                                               sizeof(double) * npad);
                                }
                        }
                }
        }
        const size_t nE = (size_t)maxbn * maxbn * maxbn * maxbn;
        double *E = malloc(sizeof(double) * nE * 3);
        double *X1 = E + nE;
        double *X2 = X1 + nE;
        double *G = malloc(sizeof(double) * maxbn * maxbn * npad);
        const size_t maxq = (size_t)maxbs * maxbs * maxbs * maxbs;
        GemmQuartet *ql = malloc(sizeof(GemmQuartet) * maxq);
        double *buf = malloc(sizeof(double) * ((size_t)di*di*di*di + cache_size));
        double *cache = buf + (size_t)di*di*di*di;
        double *bout = NULL;
        double *bcache = NULL;
        if (lanes) {
                bout = malloc(sizeof(double) * BATCH_MAX_NF * lanes);
                bcache = malloc(sizeof(double) * int2e_sph_batch_cache_size());
        }
        int shls[4];
        int bq[4 * BATCH_MAX_LANES];
        const char TRANS_N = 'N', TRANS_T = 'T';
        const double D0 = 0;
        const int inpad = npad;

#pragma omp for schedule(dynamic, 1)
        for (it = 0; it < nbp; it++) {
                pij = nbp - 1 - it;
                Ib = (int)(sqrt(2*pij+.25) - .5 + 1e-7);
                Jb = pij - (size_t)Ib*(Ib+1)/2;
                if (QB[Ib*nb+Jb] * qbmax * dmax4 <= cutoff) {
                        continue;
                }
                for (pkl = 0; pkl <= pij; pkl++) {
                        Kb = (int)(sqrt(2*pkl+.25) - .5 + 1e-7);
                        Lb = pkl - (size_t)Kb*(Kb+1)/2;
                        if (QB[Ib*nb+Jb] * QB[Kb*nb+Lb] * dmax4 <= cutoff) {
                                continue;
                        }
                        // the block quartet's shell quartets of the one-pass loop
                        size_t nq = 0;
                        int ish, jsh, ksh, lsh;
                        for (ish = bsh[Ib]; ish < bsh[Ib+1]; ish++) {
                        for (jsh = bsh[Jb]; jsh < (Ib == Jb ? ish + 1 : bsh[Jb+1]); jsh++) {
                                double qij = q_cond[ish*nbas+jsh];
                                if (qij * dmax4 == 0) {
                                        continue;
                                }
                                // the margin keeps quartets within rounding of the cutoff for fprescreen
                                double qthr = cutoff / (qij * dmax4) * (1 - 1e-10);
                                size_t ijp = (size_t)ish * (ish + 1) / 2 + jsh;
                                for (ksh = bsh[Kb]; ksh < bsh[Kb+1]; ksh++) {
                                for (lsh = bsh[Lb]; lsh < (Kb == Lb ? ksh + 1 : bsh[Lb+1]); lsh++) {
                                        if (pij == pkl && (size_t)ksh * (ksh + 1) / 2 + lsh > ijp) {
                                                continue;
                                        }
                                        if (q_cond[ksh*nbas+lsh] <= qthr) {
                                                continue;
                                        }
                                        shls[0] = ish;
                                        shls[1] = jsh;
                                        shls[2] = ksh;
                                        shls[3] = lsh;
                                        if (!(*fprescreen)(shls, vhfopt, atm, bas, env)) {
                                                continue;
                                        }
                                        GemmQuartet *q = ql + nq++;
                                        q->key = lanes ? ((stype[ish] * nt + stype[jsh]) * nt + stype[ksh]) * nt + stype[lsh] : 0;
                                        memcpy(q->shls, shls, sizeof(int) * 4);
                                } }
                        } }
                        if (nq == 0) {
                                continue;
                        }
                        const int nI = bn[Ib], nJ = bn[Jb], nK = bn[Kb], nL = bn[Lb];
                        const int nIJ = nI * nJ, nKL = nK * nL;
                        GemmBlockQ g = {bao[Ib], bao[Jb], bao[Kb], bao[Lb], nJ, nL, nKL,
                                        Ib == Jb, Kb == Lb, pij == pkl};
                        memset(E, 0, sizeof(double) * nIJ * nKL);
                        // the integrals, quartets of a class in qcint's lanes where it takes them
                        if (lanes) {
                                qsort(ql, nq, sizeof(GemmQuartet), _gemm_quartet_cmp);
                        }
                        size_t q0 = 0;
                        while (q0 < nq) {
                                int nbq = 0;
                                size_t q1 = q0;
                                if (lanes) {
                                        while (q1 < nq && ql[q1].key == ql[q0].key && nbq < lanes &&
                                               int2e_sph_batch_ok(ql[q1].shls, bas, env)) {
                                                memcpy(bq + 4 * nbq, ql[q1].shls, sizeof(int) * 4);
                                                nbq++;
                                                q1++;
                                        }
                                }
                                if (2 * nbq >= lanes && nbq > 0) {
                                        int *s = bq;
                                        const size_t nf = (size_t)(ao_loc[s[0]+1] - ao_loc[s[0]]) *
                                                                  (ao_loc[s[1]+1] - ao_loc[s[1]]) *
                                                                  (ao_loc[s[2]+1] - ao_loc[s[2]]) *
                                                                  (ao_loc[s[3]+1] - ao_loc[s[3]]);
                                        int2e_sph_batch(bout, nbq, bq, atm, natm, bas, nbas, env, NULL, bcache);
                                        int b;
                                        for (b = 0; b < nbq; b++) {
                                                _gemm_place(E, bout + nf * b, bq + 4 * b, ao_loc, &g);
                                        }
                                } else {
                                        // one at a time: a short batch, or quartets qcint does not batch
                                        if (nbq == 0) {
                                                q1 = q0 + 1;
                                        }
                                        size_t q;
                                        for (q = q0; q < q1; q++) {
                                                if ((*intor)(buf, NULL, ql[q].shls, atm, natm, bas, nbas,
                                                             env, cintopt, cache)) {
                                                        _gemm_place(E, buf, ql[q].shls, ao_loc, &g);
                                                }
                                        }
                                }
                                q0 = q1;
                        }
                        // the weights of the quartet kernels, at the block level
                        const int sij = Ib != Jb, skl = Kb != Lb, sbk = pij != pkl;
                        const double cJ = (sij ? 1. : .5) * (skl ? 1. : .5);
                        const double c = (sbk || hermi == 0) ? 1. : .5;
                        if (vj != NULL) {
                                const double *dsKL = Dsblk + boff[Kb*nb+Lb];
                                dgemm_(&TRANS_N, &TRANS_N, &inpad, &nIJ, &nKL, &cJ, dsKL, &inpad,
                                       E, &nKL, &D0, G, &inpad);
                                _gemm_add(VJblk + boff[Ib*nb+Jb], G, (size_t)nIJ * npad, lk + Ib*nb+Jb);
                                if (sbk) {
                                        const double *dsIJ = Dsblk + boff[Ib*nb+Jb];
                                        dgemm_(&TRANS_N, &TRANS_T, &inpad, &nKL, &nIJ, &cJ, dsIJ, &inpad,
                                               E, &nKL, &D0, G, &inpad);
                                        _gemm_add(VJblk + boff[Kb*nb+Lb], G, (size_t)nKL * npad, lk + Kb*nb+Lb);
                                }
                        }
                        if (vk == NULL) {
                                continue;
                        }
                        // X1[(i,l),(j,k)] and X2[(i,k),(j,l)]
                        const int nJK = nJ * nK, nJL = nJ * nL, nIL = nI * nL, nIK = nI * nK;
                        int j, k, l;
                        for (i = 0; i < nI; i++) {
                        for (j = 0; j < nJ; j++) {
                                const double *e = E + (size_t)(i * nJ + j) * nKL;
                                for (k = 0; k < nK; k++) {
                                for (l = 0; l < nL; l++) {
                                        const double v = e[k * nL + l];
                                        X1[(size_t)(i * nL + l) * nJK + j * nK + k] = v;
                                        X2[(size_t)(i * nK + k) * nJL + j * nL + l] = v;
                                } }
                        } }
                        omp_lock_t *lkk = lk + nb * nb;
                        // t1: VK[I,L] += c X1 D[J,K]
                        dgemm_(&TRANS_N, &TRANS_N, &inpad, &nIL, &nJK, &c, Dblk + boff[Jb*nb+Kb], &inpad,
                               X1, &nJK, &D0, G, &inpad);
                        _gemm_add(VKblk + boff[Ib*nb+Lb], G, (size_t)nIL * npad, lkk + Ib*nb+Lb);
                        if (skl) {
                                // t3: VK[I,K] += c X2 D[J,L]
                                dgemm_(&TRANS_N, &TRANS_N, &inpad, &nIK, &nJL, &c, Dblk + boff[Jb*nb+Lb], &inpad,
                                       X2, &nJL, &D0, G, &inpad);
                                _gemm_add(VKblk + boff[Ib*nb+Kb], G, (size_t)nIK * npad, lkk + Ib*nb+Kb);
                        }
                        if (sij) {
                                // t2: VK[J,L] += c X2^T D[I,K]
                                dgemm_(&TRANS_N, &TRANS_T, &inpad, &nJL, &nIK, &c, Dblk + boff[Ib*nb+Kb], &inpad,
                                       X2, &nJL, &D0, G, &inpad);
                                _gemm_add(VKblk + boff[Jb*nb+Lb], G, (size_t)nJL * npad, lkk + Jb*nb+Lb);
                                if (skl) {
                                        // t4: VK[J,K] += c X1^T D[I,L]
                                        dgemm_(&TRANS_N, &TRANS_T, &inpad, &nJK, &nIL, &c, Dblk + boff[Ib*nb+Lb],
                                               &inpad, X1, &nJK, &D0, G, &inpad);
                                        _gemm_add(VKblk + boff[Jb*nb+Kb], G, (size_t)nJK * npad, lkk + Jb*nb+Kb);
                                }
                        }
                        if (!kt || !sbk) {
                                continue;
                        }
                        // hermi = 0, t5..t8: VKT[J,K] += X1^T DT[I,L] (t5), VKT[J,L] += skl X2^T DT[I,K]
                        // (t6), VKT[I,K] += sij X2 DT[J,L] (t7), VKT[I,L] += sij skl X1 DT[J,K] (t8)
                        omp_lock_t *lkt = lk + nb * nb * 2;
                        dgemm_(&TRANS_N, &TRANS_T, &inpad, &nJK, &nIL, &c, DTblk + boff[Ib*nb+Lb], &inpad,
                               X1, &nJK, &D0, G, &inpad);
                        _gemm_add(VKTblk + boff[Jb*nb+Kb], G, (size_t)nJK * npad, lkt + Jb*nb+Kb);
                        if (skl) {
                                dgemm_(&TRANS_N, &TRANS_T, &inpad, &nJL, &nIK, &c, DTblk + boff[Ib*nb+Kb], &inpad,
                                       X2, &nJL, &D0, G, &inpad);
                                _gemm_add(VKTblk + boff[Jb*nb+Lb], G, (size_t)nJL * npad, lkt + Jb*nb+Lb);
                        }
                        if (sij) {
                                dgemm_(&TRANS_N, &TRANS_N, &inpad, &nIK, &nJL, &c, DTblk + boff[Jb*nb+Lb], &inpad,
                                       X2, &nJL, &D0, G, &inpad);
                                _gemm_add(VKTblk + boff[Ib*nb+Kb], G, (size_t)nIK * npad, lkt + Ib*nb+Kb);
                                if (skl) {
                                        dgemm_(&TRANS_N, &TRANS_N, &inpad, &nIL, &nJK, &c, DTblk + boff[Jb*nb+Kb],
                                               &inpad, X1, &nJK, &D0, G, &inpad);
                                        _gemm_add(VKTblk + boff[Ib*nb+Lb], G, (size_t)nIL * npad, lkt + Ib*nb+Lb);
                                }
                        }
                }
        }
        // the sums into vj (blocks I >= J, the rest 0) and vk
#pragma omp for schedule(dynamic, 1)
        for (pij = 0; pij < (size_t)nb * nb; pij++) {
                Ib = pij / nb;
                Jb = pij % nb;
                for (i = 0; i < bn[Ib]; i++) {
                        size_t dst = ((size_t)(bao[Ib] + i) * nao + bao[Jb]) * npad;
                        size_t src = boff[pij] + (size_t)i * bn[Jb] * npad;
                        if (vj != NULL) {
                                if (Ib >= Jb) {
                                        memcpy(vj + dst, VJblk + src, sizeof(double) * bn[Jb] * npad);
                                } else {
                                        memset(vj + dst, 0, sizeof(double) * bn[Jb] * npad);
                                }
                        }
                        if (vk != NULL) {
                                memcpy(vk + dst, VKblk + src, sizeof(double) * bn[Jb] * npad);
                        }
                        if (VKTblk != NULL) {
                                // vk[p,q] += VKT[Q,P][q][p]
                                int j, n;
                                for (j = 0; j < bn[Jb]; j++) {
                                        const double *t = VKTblk + boff[Jb*nb+Ib] + ((size_t)j * bn[Ib] + i) * npad;
                                        double *o = vk + dst + (size_t)j * npad;
                                        for (n = 0; n < npad; n++) {
                                                o[n] += t[n];
                                        }
                                }
                        }
                }
        }
        free(E);
        free(G);
        free(ql);
        free(buf);
        if (lanes) {
                free(bout);
                free(bcache);
        }
}
        for (Ib = 0; Ib < nb * nb * 3; Ib++) {
                omp_destroy_lock(lk + Ib);
        }
        free(lk);
        free(bsh);
        free(boff);
        free(QB);
        free(Dblk);
        free(Dsblk);
        free(VJblk);
        free(VKblk);
        free(DTblk);
        free(VKTblk);
}

/*
 * vj, vk: [nao,nao,npad] (either can be NULL), VJ and VK as described at the
 * top: the caller forms vj = VJ + VJ^T and, for hermi=1, vk = VK + VK^T.
 * dm, dms: [nao,nao,npad], D_n and D_n + D_n^T (dms is only read for J),
 * npad a multiple of DM_CHUNK, or 1. hermi=1 requires symmetric dms.
 * vhfopt (or NULL) gives the screening of CVHFnr_direct_drv: the same
 * fprescreen is applied to every quartet. sd (or NULL) is a semi-direct cache
 * for the same intor, molecule and q_cond; it is used with the CVHFnrs8_*
 * prescreen functions only. int2e_batch: int2e_sph evaluates several quartets
 * of one class at once where qcint offers that (see QuartetQueue). dm_mode:
 * many dms through dense matrix products (see _direct_jk_gemm).
 */
void CVHFnr_direct_jk_batch(int (*intor)(), double *vj, double *vk,
                            double *dm, double *dms, int npad, int hermi,
                            int *ao_loc, CINTOpt *cintopt, CVHFOpt *vhfopt,
                            CVHFsdCache *sd, int int2e_batch, int dm_mode,
                            int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t nao = ao_loc[nbas];
        const size_t vsize = nao * nao * npad;
        const size_t npair = (size_t)nbas * (nbas + 1) / 2;
        int shls_slice[8] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
        const int di = GTOmax_shell_dim(ao_loc, shls_slice, 4);
        const size_t cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                                    atm, natm, bas, nbas, env);
        FBatchJ fj = npad == 1 ? _batch_j_s : _batch_j_v;
        FBatchK fk = npad == 1 ? _batch_k_s : _batch_k_v;
#ifdef HAVE_BATCH_AVX2
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
                fj = npad == 1 ? _batch_j_s_avx2 : _batch_j_v_avx2;
                fk = npad == 1 ? _batch_k_s_avx2 : _batch_k_v_avx2;
        }
#endif
        int (*fprescreen)() = CVHFnoscreen;
        double *q_cond = NULL;
        double *qmax_k = NULL;
        double cutoff = 0;
        double dmax4 = 0;
        size_t i;
        if (vhfopt != NULL) {
                fprescreen = vhfopt->fprescreen;
                // Quartets with q_ij q_kl max(4 dm_cond) <= cutoff fail each of
                // these prescreen functions; they are skipped before the call.
                if (fprescreen == CVHFnrs8_prescreen ||
                    fprescreen == CVHFnrs8_vj_prescreen ||
                    fprescreen == CVHFnrs8_vk_prescreen) {
                        q_cond = vhfopt->q_cond;
                        cutoff = vhfopt->direct_scf_cutoff;
                        for (i = 0; i < (size_t)nbas * nbas; i++) {
                                dmax4 = MAX(dmax4, vhfopt->dm_cond[i]);
                        }
                        dmax4 *= 4;
                        qmax_k = _qmax_k(q_cond, nbas);
                }
        }
        if (q_cond == NULL) {
                sd = NULL;
        }
        // the queues of quartets for int2e_sph_batch, by shell type (l, nprim)
        int lanes = 0;
        int nt = 0;
        int *stype = NULL;
        if (int2e_batch && int2e_sph_batch != NULL && intor == int2e_sph &&
            env[PTR_RANGE_OMEGA] >= 0) {
                lanes = MIN(int2e_sph_batch_lanes(), BATCH_MAX_LANES);
                stype = malloc(sizeof(int) * nbas * 3);
                int *tl = stype + nbas;
                int *tp = tl + nbas;
                int s, t;
                for (s = 0; s < nbas; s++) {
                        for (t = 0; t < nt; t++) {
                                if (tl[t] == bas[ANG_OF+s*BAS_SLOTS] &&
                                    tp[t] == bas[NPRIM_OF+s*BAS_SLOTS]) {
                                        break;
                                }
                        }
                        if (t == nt) {
                                tl[t] = bas[ANG_OF+s*BAS_SLOTS];
                                tp[t] = bas[NPRIM_OF+s*BAS_SLOTS];
                                nt++;
                        }
                        stype[s] = t;
                }
        }
        if (dm_mode && q_cond != NULL && npad / DM_CHUNK >= GEMM_MIN_CHUNKS) {
                _direct_jk_gemm(intor, vj, vk, dm, dms, npad, hermi, ao_loc, cintopt, vhfopt,
                                lanes, nt, stype, q_cond, cutoff, dmax4, di, cache_size,
                                atm, natm, bas, nbas, env);
                if (stype != NULL) {
                        free(stype);
                }
                free(qmax_k);
                return;
        }
        double **vjprivs = malloc(sizeof(double *) * omp_get_max_threads() * 2);
        double **vkprivs = vjprivs + omp_get_max_threads();
#pragma omp parallel
{
        int thread_id = omp_get_thread_num();
        JKBatch b = {npad, nao, dm, dms, NULL, NULL};
        if (vj != NULL) {
                b.vj = _alloc_sums(vsize);
                vjprivs[thread_id] = b.vj;
        }
        if (vk != NULL) {
                b.vk = _alloc_sums(vsize);
                vkprivs[thread_id] = b.vk;
        }
        double *buf = malloc(sizeof(double) * ((size_t)di*di*di*di + cache_size));
        double *cache = buf + (size_t)di*di*di*di;
        QuartetQueue *queues = NULL;
        QuartetQueue *squeues = NULL;   // the semi-direct cache's quartets
        QuartetQueue *Q;
        int *qlist = NULL;
        int *sqlist = NULL;
        int nlisted = 0;
        int nslisted = 0;
        int x;
        double *bout = NULL;
        double *bcache = NULL;
        if (lanes) {
                queues = calloc((size_t)nt * nt, sizeof(QuartetQueue));
                qlist = malloc(sizeof(int) * nt * nt);
                if (sd != NULL) {
                        squeues = calloc((size_t)nt * nt, sizeof(QuartetQueue));
                        sqlist = malloc(sizeof(int) * nt * nt);
                }
                bout = malloc(sizeof(double) * BATCH_MAX_NF * lanes);
                bcache = malloc(sizeof(double) * int2e_sph_batch_cache_size());
        }
        double *e;
        int shls[4];
        int ish, jsh, ksh, lsh, lsh1;
        int32_t key;
        size_t it, ij, dij, off;
        int64_t n = 0;
        double qij = 0;
        double qthr;
// dynamic, most expensive pairs first: the cost of a pair is dominated by
// the integrals and varies widely
#pragma omp for schedule(dynamic, 1)
        for (it = 0; it < npair; it++) {
                ij = npair - 1 - it;
                ish = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
                jsh = ij - (size_t)ish*(ish+1)/2;
                qthr = 0;
                if (q_cond != NULL) {
                        qij = q_cond[ish*nbas+jsh];
                        if (qij * dmax4 == 0) {
                                continue;
                        }
                        // the margin keeps quartets within rounding of the
                        // cutoff for fprescreen
                        qthr = cutoff / (qij * dmax4) * (1 - 1e-10);
                }
                shls[0] = ish;
                shls[1] = jsh;
                if (sd != NULL) {
                        // the stored quartets of the row, in the order of
                        // the loop below
                        dij = (size_t)(ao_loc[ish+1] - ao_loc[ish]) *
                                      (ao_loc[jsh+1] - ao_loc[jsh]);
                        off = sd->row_d[ij];
                        for (n = sd->row_q[ij]; n < sd->row_q[ij+1]; n++) {
                                ksh = sd->shls[n] >> 16;
                                lsh = sd->shls[n] & 0xffff;
                                e = sd->eri + off;
                                off += dij * (ao_loc[ksh+1] - ao_loc[ksh]) *
                                             (ao_loc[lsh+1] - ao_loc[lsh]);
                                if (q_cond[ksh*nbas+lsh] <= qthr) {
                                        continue;
                                }
                                shls[2] = ksh;
                                shls[3] = lsh;
                                if (!(*fprescreen)(shls, vhfopt, atm, bas, env)) {
                                        continue;
                                }
                                if (sd->state[n] == 0 && lanes &&
                                    int2e_sph_batch_ok(shls, bas, env)) {
                                        // the first evaluation, batched (_flush_stored)
                                        Q = squeues + stype[ksh] * nt + stype[lsh];
                                        if (!Q->listed) {
                                                Q->listed = 1;
                                                sqlist[nslisted++] = Q - squeues;
                                        }
                                        memcpy(Q->shls + 4 * Q->n, shls, sizeof(int) * 4);
                                        Q->slot[Q->n] = e;
                                        Q->idx[Q->n] = n;
                                        if (++Q->n == lanes) {
                                                _flush_stored(Q, sd, lanes, intor, fj, fk, &b,
                                                              ao_loc, hermi, bout, bcache, cache,
                                                              cintopt, atm, natm, bas, nbas, env);
                                        }
                                        continue;
                                }
                                if (sd->state[n] == 0) {
                                        sd->state[n] = (*intor)(e, NULL, shls, atm, natm,
                                                                bas, nbas, env, cintopt,
                                                                cache) ? 1 : 2;
                                }
                                if (sd->state[n] == 2) {
                                        continue;
                                }
                                if (vj != NULL) {
                                        (*fj)(e, &b, ao_loc, ish, jsh, ksh, lsh);
                                }
                                if (vk != NULL) {
                                        (*fk)(e, &b, ao_loc, hermi, ish, jsh, ksh, lsh);
                                }
                        }
                        // the rest of the row's stored quartets (the queues are by
                        // ket type: they must not outlive the row)
                        for (x = 0; x < nslisted; x++) {
                                Q = squeues + sqlist[x];
                                if (Q->n > 0) {
                                        _flush_stored(Q, sd, lanes, intor, fj, fk, &b, ao_loc,
                                                      hermi, bout, bcache, cache, cintopt, atm,
                                                      natm, bas, nbas, env);
                                }
                                Q->listed = 0;
                        }
                        nslisted = 0;
                        // fprescreen rejects the quartets with q_ij q_kl <= cutoff
                        if (sd->thr <= cutoff) {
                                continue;
                        }
                        n = sd->row_q[ij];
                }
                for (ksh = 0; ksh <= ish; ksh++) {
                        if (q_cond != NULL && qmax_k[ksh] <= qthr) {
                                continue;
                        }
                        lsh1 = ksh < ish ? ksh + 1 : jsh + 1;
                        for (lsh = 0; lsh < lsh1; lsh++) {
                                if (q_cond != NULL && q_cond[ksh*nbas+lsh] <= qthr) {
                                        continue;
                                }
                                if (sd != NULL) {
                                        // stored (contracted above): the
                                        // keys of a row ascend in this order
                                        key = ksh << 16 | lsh;
                                        while (n < sd->row_q[ij+1] && sd->shls[n] < key) {
                                                n++;
                                        }
                                        if (n < sd->row_q[ij+1] && sd->shls[n] == key) {
                                                continue;
                                        }
                                }
                                shls[2] = ksh;
                                shls[3] = lsh;
                                if (vhfopt != NULL &&
                                    !(*fprescreen)(shls, vhfopt, atm, bas, env)) {
                                        continue;
                                }
                                if (lanes && int2e_sph_batch_ok(shls, bas, env)) {
                                        Q = queues + stype[ksh] * nt + stype[lsh];
                                        if (!Q->listed) {
                                                Q->listed = 1;
                                                qlist[nlisted++] = Q - queues;
                                        }
                                        memcpy(Q->shls + 4 * Q->n, shls, sizeof(int) * 4);
                                        if (++Q->n == lanes) {
                                                _flush_queue(Q, lanes, intor, fj, fk, &b, ao_loc,
                                                             hermi, bout, bcache, buf, cache,
                                                             cintopt, atm, natm, bas, nbas, env);
                                        }
                                        continue;
                                }
                                if (!(*intor)(buf, NULL, shls, atm, natm, bas, nbas,
                                              env, cintopt, cache)) {
                                        continue;
                                }
                                if (vj != NULL) {
                                        (*fj)(buf, &b, ao_loc, ish, jsh, ksh, lsh);
                                }
                                if (vk != NULL) {
                                        (*fk)(buf, &b, ao_loc, hermi, ish, jsh, ksh, lsh);
                                }
                        }
                }
                // the rest of the row's queues
                for (x = 0; x < nlisted; x++) {
                        Q = queues + qlist[x];
                        if (Q->n > 0) {
                                _flush_queue(Q, lanes, intor, fj, fk, &b, ao_loc, hermi, bout,
                                             bcache, buf, cache, cintopt, atm, natm, bas, nbas, env);
                        }
                        Q->listed = 0;
                }
                nlisted = 0;
        }
        if (vj != NULL) {
                NPomp_dsum_reduce_inplace(vjprivs, vsize);
#pragma omp for schedule(static)
                for (it = 0; it < vsize; it++) {
                        vj[it] = vjprivs[0][it];
                }
                free(b.vj);
        }
        if (vk != NULL) {
                NPomp_dsum_reduce_inplace(vkprivs, vsize);
#pragma omp for schedule(static)
                for (it = 0; it < vsize; it++) {
                        vk[it] = vkprivs[0][it];
                }
                free(b.vk);
        }
        free(buf);
        if (lanes) {
                free(queues);
                free(qlist);
                free(bout);
                free(bcache);
                if (squeues != NULL) {
                        free(squeues);
                        free(sqlist);
                }
        }
}
        free(vjprivs);
        if (stype != NULL) {
                free(stype);
        }
        if (qmax_k != NULL) {
                free(qmax_k);
        }
}
