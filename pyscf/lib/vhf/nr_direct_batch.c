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
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include "config.h"
#include "cint.h"
#include "optimizer.h"
#include "np_helper/np_helper.h"
#include "gto/gto.h"

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

/*
 * vj, vk: [nao,nao,npad] (either can be NULL), VJ and VK as described at the
 * top: the caller forms vj = VJ + VJ^T and, for hermi=1, vk = VK + VK^T.
 * dm, dms: [nao,nao,npad], D_n and D_n + D_n^T (dms is only read for J),
 * npad a multiple of DM_CHUNK, or 1. hermi=1 requires symmetric dms.
 * vhfopt (or NULL) gives the screening of CVHFnr_direct_drv: the same
 * fprescreen is applied to every quartet.
 */
void CVHFnr_direct_jk_batch(int (*intor)(), double *vj, double *vk,
                            double *dm, double *dms, int npad, int hermi,
                            int *ao_loc, CINTOpt *cintopt, CVHFOpt *vhfopt,
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
        int ksh, lsh;
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
                        qmax_k = malloc(sizeof(double) * nbas);
                        for (ksh = 0; ksh < nbas; ksh++) {
                                qmax_k[ksh] = 0;
                                for (lsh = 0; lsh <= ksh; lsh++) {
                                        qmax_k[ksh] = MAX(qmax_k[ksh], q_cond[ksh*nbas+lsh]);
                                }
                        }
                }
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
        int shls[4];
        int ish, jsh, ksh, lsh, lsh1;
        size_t it, ij;
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
                        if (q_cond[ish*nbas+jsh] * dmax4 == 0) {
                                continue;
                        }
                        // the margin keeps quartets within rounding of the
                        // cutoff for fprescreen
                        qthr = cutoff / (q_cond[ish*nbas+jsh] * dmax4) * (1 - 1e-10);
                }
                shls[0] = ish;
                shls[1] = jsh;
                for (ksh = 0; ksh <= ish; ksh++) {
                        if (q_cond != NULL && qmax_k[ksh] <= qthr) {
                                continue;
                        }
                        lsh1 = ksh < ish ? ksh + 1 : jsh + 1;
                        for (lsh = 0; lsh < lsh1; lsh++) {
                                if (q_cond != NULL && q_cond[ksh*nbas+lsh] <= qthr) {
                                        continue;
                                }
                                shls[2] = ksh;
                                shls[3] = lsh;
                                if (vhfopt != NULL &&
                                    !(*fprescreen)(shls, vhfopt, atm, bas, env)) {
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
}
        free(vjprivs);
        if (qmax_k != NULL) {
                free(qmax_k);
        }
}
