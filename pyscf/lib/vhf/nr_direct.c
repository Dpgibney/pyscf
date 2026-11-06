/* Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
  
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
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "config.h"
#include "cint.h"
#include "optimizer.h"
#include "nr_direct.h"
#include "np_helper/np_helper.h"
#include "gto/gto.h"

/*
 * Several quartets of one class at once: qcint's batched integrals (cint2e_batch.c) evaluate up to
 * int2e_sph_batch_lanes() shell quartets whose shells have the same l and number of primitives,
 * one quartet per SIMD lane. CVHFnr_direct_drv_mode queues the quartets they take by the types of
 * their 4 shells, in a hash table per thread, and evaluates a queue when it holds a SIMD width of
 * quartets. A batch costs as much as a full one, so fewer than half the lanes go quartet by quartet.
 *
 * The contractions write to the JKArray of the quartet's block (ao_off, shape, block_quartets), so
 * each queued quartet keeps its block, and the JKArrays are pointed at it for the contraction. The
 * queues outlive the blocks (and are flushed at the thread's end) when every output fits in its
 * JKArray's stack, which then holds each block pair at most once and cannot overflow; otherwise
 * they are flushed at the end of each block. With 16-AO blocks (few tasks for many threads) a block
 * holds a few quartets of each class, which would mostly go one at a time.
 *
 * Weak: libcint and older qcint have none of these functions, and then every quartet goes alone.
 */
int int2e_sph();
int int2e_ip1_sph();
int int2e_sph_batch(double *out, int nq, int *shls, int *atm, int natm, int *bas, int nbas,
                    double *env, CINTOpt *opt, double *cache) __attribute__((weak));
int int2e_sph_batch_ok(int *shls, int *bas, double *env) __attribute__((weak));
int int2e_ip1_sph_batch(double *out, int nq, int *shls, int *atm, int natm, int *bas, int nbas,
                        double *env, CINTOpt *opt, double *cache) __attribute__((weak));
int int2e_ip1_sph_batch_ok(int *shls, int *bas, double *env) __attribute__((weak));
int int2e_sph_batch_lanes(void) __attribute__((weak));
size_t int2e_sph_batch_cache_size(void) __attribute__((weak));
// the integrals of nuclear Hessians, 9 components
int int2e_ipip1_sph();
int int2e_ipvip1_sph();
int int2e_ip1ip2_sph();
int int2e_ipip1_sph_batch(double *out, int nq, int *shls, int *atm, int natm, int *bas, int nbas,
                          double *env, CINTOpt *opt, double *cache) __attribute__((weak));
int int2e_ipvip1_sph_batch(double *out, int nq, int *shls, int *atm, int natm, int *bas, int nbas,
                           double *env, CINTOpt *opt, double *cache) __attribute__((weak));
int int2e_ip1ip2_sph_batch(double *out, int nq, int *shls, int *atm, int natm, int *bas, int nbas,
                           double *env, CINTOpt *opt, double *cache) __attribute__((weak));
int int2e_ipip1_sph_batch_ok(int *shls, int *bas, double *env) __attribute__((weak));
int int2e_ipvip1_sph_batch_ok(int *shls, int *bas, double *env) __attribute__((weak));
int int2e_ip1ip2_sph_batch_ok(int *shls, int *bas, double *env) __attribute__((weak));
size_t int2e_hess_sph_batch_cache_size(void) __attribute__((weak));

#define DB_MAX_LANES    8
#define DB_MAX_NF       2401    // (ff|ff)
#define DB_CAP          8192    // slots of the hash table of queues per thread, at most half used

typedef struct {
        int (*fbatch)();
        int (*fok)(int *shls, int *bas, double *env);
        size_t cache_size;
        int ncomp;
} DirectBatchIntor;

typedef struct {
        int n;
        int shls[4 * DB_MAX_LANES];
        int blk[4 * DB_MAX_LANES];      // the blocks (i, j, k, l) of the quartets
} DirectQueue;

typedef struct {
        DirectBatchIntor bi;
        int lanes;
        int nt;
        int *stype;             // the type (l, number of primitives) of each shell
        int persist;            // whether the queues outlive the blocks
        int cur[4];             // the block of the driver's fdot call
        int *block_loc[4];      // the driver's block partitions of i, j, k, l
        int nused;              // queues in use, q[0..nused-1], in hash slots slot[...]
        int *keys;              // [DB_CAP]: the types of the 4 shells of a slot's queue, or -1
        int *qid;               // [DB_CAP]: the queue of a slot
        int *slot;              // [DB_CAP/2]
        DirectQueue *q;         // [DB_CAP/2]
        double *bout;
        double *bcache;
} DirectBatch;

// the batched version of intor, if qcint has one; 0 if not
static int _direct_batch_intor(int (*intor)(), DirectBatchIntor *bi)
{
        if (int2e_sph_batch_lanes == NULL || int2e_sph_batch_cache_size == NULL) {
                return 0;
        }
        if (intor == int2e_sph && int2e_sph_batch != NULL && int2e_sph_batch_ok != NULL) {
                *bi = (DirectBatchIntor){int2e_sph_batch, int2e_sph_batch_ok,
                                         int2e_sph_batch_cache_size(), 1};
                return 1;
        }
        if (intor == int2e_ip1_sph && int2e_ip1_sph_batch != NULL && int2e_ip1_sph_batch_ok != NULL) {
                *bi = (DirectBatchIntor){int2e_ip1_sph_batch, int2e_ip1_sph_batch_ok,
                                         int2e_sph_batch_cache_size(), 3};
                return 1;
        }
        if (int2e_hess_sph_batch_cache_size == NULL) {
                return 0;
        }
        if (intor == int2e_ipip1_sph && int2e_ipip1_sph_batch != NULL &&
            int2e_ipip1_sph_batch_ok != NULL) {
                *bi = (DirectBatchIntor){int2e_ipip1_sph_batch, int2e_ipip1_sph_batch_ok,
                                         int2e_hess_sph_batch_cache_size(), 9};
                return 1;
        }
        if (intor == int2e_ipvip1_sph && int2e_ipvip1_sph_batch != NULL &&
            int2e_ipvip1_sph_batch_ok != NULL) {
                *bi = (DirectBatchIntor){int2e_ipvip1_sph_batch, int2e_ipvip1_sph_batch_ok,
                                         int2e_hess_sph_batch_cache_size(), 9};
                return 1;
        }
        if (intor == int2e_ip1ip2_sph && int2e_ip1ip2_sph_batch != NULL &&
            int2e_ip1ip2_sph_batch_ok != NULL) {
                *bi = (DirectBatchIntor){int2e_ip1ip2_sph_batch, int2e_ip1ip2_sph_batch_ok,
                                         int2e_hess_sph_batch_cache_size(), 9};
                return 1;
        }
        return 0;
}

// the types of the shells for the queues of CVHFnr_direct_drv_mode; NULL if intor is not batched
static int *_direct_batch_types(int (*intor)(), int int2e_batch, DirectBatchIntor *bi, int *nt_out,
                                int *bas, int nbas, double *env)
{
        if (!int2e_batch || env[PTR_RANGE_OMEGA] < 0 || !_direct_batch_intor(intor, bi)) {
                return NULL;
        }
        int *stype = malloc(sizeof(int) * nbas * 3);
        int *tl = stype + nbas;
        int *tp = tl + nbas;
        int s, t;
        int nt = 0;
        for (s = 0; s < nbas; s++) {
                for (t = 0; t < nt; t++) {
                        if (tl[t] == bas[ANG_OF+s*BAS_SLOTS] && tp[t] == bas[NPRIM_OF+s*BAS_SLOTS]) {
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
        // the keys, nt^4, are ints
        if (nt > 200) {
                free(stype);
                return NULL;
        }
        *nt_out = nt;
        return stype;
}

// dmax: the largest shell dimension of the call; persist, block_loc: see DirectBatch
static DirectBatch *_direct_batch_new(DirectBatchIntor *bi, int nt, int *stype, size_t dmax,
                                      int persist, int **block_loc)
{
        DirectBatch *bt = malloc(sizeof(DirectBatch));
        int i;
        bt->bi = *bi;
        bt->lanes = MIN(int2e_sph_batch_lanes(), DB_MAX_LANES);
        bt->nt = nt;
        bt->stype = stype;
        bt->persist = persist;
        for (i = 0; i < 4; i++) {
                bt->cur[i] = 0;
                bt->block_loc[i] = block_loc[i];
        }
        bt->nused = 0;
        bt->keys = malloc(sizeof(int) * (DB_CAP * 2 + DB_CAP / 2));
        bt->qid = bt->keys + DB_CAP;
        bt->slot = bt->qid + DB_CAP;
        for (i = 0; i < DB_CAP; i++) {
                bt->keys[i] = -1;
        }
        bt->q = malloc(sizeof(DirectQueue) * DB_CAP / 2);
        // and the contractions' scratch after the last quartet (_direct_batch_flush)
        bt->bout = malloc(sizeof(double) * (DB_MAX_NF * bi->ncomp * bt->lanes + 2 * dmax * dmax));
        bt->bcache = malloc(sizeof(double) * bi->cache_size);
        return bt;
}

static void _direct_batch_del(DirectBatch *bt)
{
        free(bt->keys);
        free(bt->q);
        free(bt->bout);
        free(bt->bcache);
        free(bt);
}

// points the JKArrays at block blk (i, j, k, l), as the driver does for its fdot calls
static void _direct_batch_block(DirectBatch *bt, const int *blk, JKArray **vjk, int n_dm,
                                IntorEnvs *envs)
{
        const int *ao_loc = envs->ao_loc;
        const int *shls_slice = envs->shls_slice;
        int t, n;
        for (t = 0; t < 4; t++) {
                const int *loc = bt->block_loc[t];
                const int a0 = ao_loc[loc[blk[t]]];
                const int a1 = ao_loc[loc[blk[t]+1]];
                const int off = ao_loc[shls_slice[2*t]];
                for (n = 0; n < n_dm; n++) {
                        vjk[n]->ao_off[t] = a0 - off;
                        vjk[n]->shape[t] = a1 - a0;
                        vjk[n]->block_quartets[t] = blk[t];
                }
        }
}

// as INTOR_AND_CONTRACT for the integrals eri of the quartet shls of block blk; set: the block the
// JKArrays point at
static void _contract_quartet(double *eri, int *shls, int *blk, int *set, DirectBatch *bt,
                              JKOperator **jkop, JKArray **vjk, double **dms, int n_dm,
                              IntorEnvs *envs)
{
        const int *ao_loc = envs->ao_loc;
        const int *shls_slice = envs->shls_slice;
        const int ioff = ao_loc[shls_slice[0]];
        const int joff = ao_loc[shls_slice[2]];
        const int koff = ao_loc[shls_slice[4]];
        const int loff = ao_loc[shls_slice[6]];
        const int i0 = ao_loc[shls[0]] - ioff;
        const int j0 = ao_loc[shls[1]] - joff;
        const int k0 = ao_loc[shls[2]] - koff;
        const int l0 = ao_loc[shls[3]] - loff;
        const int i1 = ao_loc[shls[0]+1] - ioff;
        const int j1 = ao_loc[shls[1]+1] - joff;
        const int k1 = ao_loc[shls[2]+1] - koff;
        const int l1 = ao_loc[shls[3]+1] - loff;
        int idm;
        if (blk[0] != set[0] || blk[1] != set[1] || blk[2] != set[2] || blk[3] != set[3]) {
                _direct_batch_block(bt, blk, vjk, n_dm, envs);
                memcpy(set, blk, sizeof(int) * 4);
        }
        for (idm = 0; idm < n_dm; idm++) {
                (*jkop[idm]->contract)(eri, dms[idm], vjk[idm], shls,
                                       i0, i1, j0, j1, k0, k1, l0, l1);
        }
}

// evaluates and contracts the quartets of Q, then empties it; the JKArrays point at the driver's
// block again after
static void _direct_batch_flush(DirectQueue *Q, DirectBatch *bt, int (*intor)(),
                                JKOperator **jkop, JKArray **vjk, double **dms,
                                double *buf, double *cache, int n_dm, IntorEnvs *envs)
{
        int *s = Q->shls;
        int set[4];
        int q;
        memcpy(set, bt->cur, sizeof(int) * 4);
        if (2 * Q->n >= bt->lanes) {
                const int *ao_loc = envs->ao_loc;
                const size_t nf = (size_t)(ao_loc[s[0]+1] - ao_loc[s[0]]) *
                                          (ao_loc[s[1]+1] - ao_loc[s[1]]) *
                                          (ao_loc[s[2]+1] - ao_loc[s[2]]) *
                                          (ao_loc[s[3]+1] - ao_loc[s[3]]) * bt->bi.ncomp;
                (*bt->bi.fbatch)(bt->bout, Q->n, s, envs->atm, envs->natm, envs->bas,
                                 envs->nbas, envs->env, NULL, bt->bcache);
                // The kernels of nr_direct_dot.c write up to 2*di*dj doubles of scratch after a
                // quartet's integrals: the last quartet first, each over those contracted already.
                for (q = Q->n - 1; q >= 0; q--) {
                        _contract_quartet(bt->bout + nf * q, s + 4 * q, Q->blk + 4 * q, set, bt,
                                          jkop, vjk, dms, n_dm, envs);
                }
        } else {
                for (q = 0; q < Q->n; q++) {
                        if ((*intor)(buf, NULL, s + 4 * q, envs->atm, envs->natm, envs->bas,
                                     envs->nbas, envs->env, envs->cintopt, cache)) {
                                _contract_quartet(buf, s + 4 * q, Q->blk + 4 * q, set, bt,
                                                  jkop, vjk, dms, n_dm, envs);
                        }
                }
        }
        if (set[0] != bt->cur[0] || set[1] != bt->cur[1] || set[2] != bt->cur[2] ||
            set[3] != bt->cur[3]) {
                _direct_batch_block(bt, bt->cur, vjk, n_dm, envs);
        }
        Q->n = 0;
}

static void _direct_batch_flush_all(DirectBatch *bt, int (*intor)(), JKOperator **jkop,
                                    JKArray **vjk, double **dms, double *buf, double *cache,
                                    int n_dm, IntorEnvs *envs)
{
        int x;
        for (x = 0; x < bt->nused; x++) {
                DirectQueue *Q = bt->q + x;
                if (Q->n > 0) {
                        _direct_batch_flush(Q, bt, intor, jkop, vjk, dms, buf, cache, n_dm, envs);
                }
                bt->keys[bt->slot[x]] = -1;
        }
        bt->nused = 0;
}

// queues the quartet shls if the batched intor takes it (1), else 0
static int _direct_batch_push(DirectBatch *bt, int *shls, int (*intor)(), JKOperator **jkop,
                              JKArray **vjk, double **dms, double *buf, double *cache,
                              int n_dm, IntorEnvs *envs)
{
        if (!(*bt->bi.fok)(shls, envs->bas, envs->env)) {
                return 0;
        }
        const int nt = bt->nt;
        const int *st = bt->stype;
        const int key = ((st[shls[0]] * nt + st[shls[1]]) * nt + st[shls[2]]) * nt + st[shls[3]];
        int slot = key % DB_CAP;
        while (bt->keys[slot] != key) {
                if (bt->keys[slot] < 0) {
                        if (bt->nused == DB_CAP / 2) {
                                _direct_batch_flush_all(bt, intor, jkop, vjk, dms, buf, cache,
                                                        n_dm, envs);
                                slot = key % DB_CAP;
                        }
                        bt->keys[slot] = key;
                        bt->qid[slot] = bt->nused;
                        bt->slot[bt->nused] = slot;
                        bt->q[bt->nused].n = 0;
                        bt->nused++;
                        break;
                }
                slot = (slot + 1) % DB_CAP;
        }
        DirectQueue *Q = bt->q + bt->qid[slot];
        memcpy(Q->shls + 4 * Q->n, shls, sizeof(int) * 4);
        memcpy(Q->blk + 4 * Q->n, bt->cur, sizeof(int) * 4);
        if (++Q->n == bt->lanes) {
                _direct_batch_flush(Q, bt, intor, jkop, vjk, dms, buf, cache, n_dm, envs);
        }
        return 1;
}

#define DECLARE_ALL \
        const int *atm = envs->atm; \
        const int *bas = envs->bas; \
        const double *env = envs->env; \
        const int natm = envs->natm; \
        const int nbas = envs->nbas; \
        const int *ao_loc = envs->ao_loc; \
        const int *shls_slice = envs->shls_slice; \
        const CINTOpt *cintopt = envs->cintopt; \
        const int ioff = ao_loc[shls_slice[0]]; \
        const int joff = ao_loc[shls_slice[2]]; \
        const int koff = ao_loc[shls_slice[4]]; \
        const int loff = ao_loc[shls_slice[6]]; \
        const int ish0 = ishls[0]; \
        const int ish1 = ishls[1]; \
        const int jsh0 = jshls[0]; \
        const int jsh1 = jshls[1]; \
        const int ksh0 = kshls[0]; \
        const int ksh1 = kshls[1]; \
        const int lsh0 = lshls[0]; \
        const int lsh1 = lshls[1]; \
        int shls[4]; \
        void (*pf)(double *eri, double *dm, JKArray *vjk, int *shls, \
                   int i0, int i1, int j0, int j1, \
                   int k0, int k1, int l0, int l1); \
        int (*fprescreen)(); \
        int notempty; \
        if (vhfopt != NULL) { \
                fprescreen = vhfopt->fprescreen; \
        } else { \
                fprescreen = CVHFnoscreen; \
        } \
        int ish, jsh, ksh, lsh, i0, j0, k0, l0, i1, j1, k1, l1, idm;

// envs->batch: queued for the batched intor (_direct_batch_push), contracted when evaluated
#define INTOR_AND_CONTRACT \
        shls[0] = ish; \
        shls[1] = jsh; \
        shls[2] = ksh; \
        shls[3] = lsh; \
        if (vhfopt != NULL && !(*fprescreen)(shls, vhfopt, atm, bas, env)) { \
                notempty = 0; \
        } else if (envs->batch != NULL && \
                   _direct_batch_push(envs->batch, shls, intor, jkop, vjk, dms, \
                                      buf, cache, n_dm, envs)) { \
                notempty = 0; \
        } else { \
                notempty = (*intor)(buf, NULL, shls, atm, natm, bas, nbas, env, \
                                    cintopt, cache); \
        } \
        if (notempty) { \
                i0 = ao_loc[ish] - ioff; \
                j0 = ao_loc[jsh] - joff; \
                k0 = ao_loc[ksh] - koff; \
                l0 = ao_loc[lsh] - loff; \
                i1 = ao_loc[ish+1] - ioff; \
                j1 = ao_loc[jsh+1] - joff; \
                k1 = ao_loc[ksh+1] - koff; \
                l1 = ao_loc[lsh+1] - loff; \
                for (idm = 0; idm < n_dm; idm++) { \
                        pf = jkop[idm]->contract; \
                        (*pf)(buf, dms[idm], vjk[idm], shls, \
                              i0, i1, j0, j1, k0, k1, l0, l1); \
                } \
        }

/*
 * for given ksh, lsh, loop all ish, jsh
 */
void CVHFdot_nrs1(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                  double **dms, double *buf, double *cache, int n_dm,
                  int *ishls, int *jshls, int *kshls, int *lshls,
                  CVHFOpt *vhfopt, IntorEnvs *envs)
{
        DECLARE_ALL;

        for (ish = ish0; ish < ish1; ish++) {
        for (jsh = jsh0; jsh < jsh1; jsh++) {
        for (ksh = ksh0; ksh < ksh1; ksh++) {
        for (lsh = lsh0; lsh < lsh1; lsh++) {
                INTOR_AND_CONTRACT;
        } } } }
}

void CVHFdot_nrs2ij(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                    double **dms, double *buf, double *cache, int n_dm,
                    int *ishls, int *jshls, int *kshls, int *lshls,
                    CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ishls[0] > jshls[0]) {
                return CVHFdot_nrs1(intor, jkop, vjk, dms, buf, cache, n_dm,
                                    ishls, jshls, kshls, lshls, vhfopt, envs);
        } else if (ishls[0] == jshls[0]) {

                DECLARE_ALL;

                for (ish = ish0; ish < ish1; ish++) {
                for (jsh = jsh0; jsh <= ish; jsh++) {
                for (ksh = ksh0; ksh < ksh1; ksh++) {
                for (lsh = lsh0; lsh < lsh1; lsh++) {
                        INTOR_AND_CONTRACT;
                } } } }
        }
}

void CVHFdot_nrs2kl(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                    double **dms, double *buf, double *cache, int n_dm,
                    int *ishls, int *jshls, int *kshls, int *lshls,
                    CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (kshls[0] > lshls[0]) {
                return CVHFdot_nrs1(intor, jkop, vjk, dms, buf, cache, n_dm,
                                    ishls, jshls, kshls, lshls, vhfopt, envs);
        } else if (kshls[0] == lshls[0]) {
                assert(kshls[1] == lshls[1]);

                DECLARE_ALL;

                for (ish = ish0; ish < ish1; ish++) {
                for (jsh = jsh0; jsh < jsh1; jsh++) {
                for (ksh = ksh0; ksh < ksh1; ksh++) {
                for (lsh = lsh0; lsh <= ksh; lsh++) {
                        INTOR_AND_CONTRACT;
                } } } }
        }
}

void CVHFdot_nrs4(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                  double **dms, double *buf, double *cache, int n_dm,
                  int *ishls, int *jshls, int *kshls, int *lshls,
                  CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (ishls[0] > jshls[0]) {
                return CVHFdot_nrs2kl(intor, jkop, vjk, dms, buf, cache, n_dm,
                                      ishls, jshls, kshls, lshls, vhfopt, envs);
        } else if (ishls[1] <= jshls[0]) {
                return;
        } else if (kshls[0] > lshls[0]) {  // ishls == jshls
                return CVHFdot_nrs2ij(intor, jkop, vjk, dms, buf, cache, n_dm,
                                      ishls, jshls, kshls, lshls, vhfopt, envs);
        } else if (kshls[0] == lshls[0]) {  // ishls == jshls
                assert(kshls[1] == lshls[1]);

                DECLARE_ALL;

                for (ish = ish0; ish < ish1; ish++) {
                for (jsh = jsh0; jsh <= ish; jsh++) {
                for (ksh = ksh0; ksh < ksh1; ksh++) {
                for (lsh = lsh0; lsh <= ksh; lsh++) {
                        INTOR_AND_CONTRACT;
                } } } }
        }
}

void CVHFdot_nrs8(int (*intor)(), JKOperator **jkop, JKArray **vjk,
                  double **dms, double *buf, double *cache, int n_dm,
                  int *ishls, int *jshls, int *kshls, int *lshls,
                  CVHFOpt *vhfopt, IntorEnvs *envs)
{
        if (vhfopt != NULL) {
                if (vhfopt->fprescreen == &CVHFnrs8_prescreen) {
                        if (!CVHFnrs8_prescreen_block(vhfopt, ishls, jshls, kshls, lshls)) {
                                return;
                        }
                } else if (vhfopt->fprescreen == &CVHFnrs8_vj_prescreen) {
                        if (!CVHFnrs8_vj_prescreen_block(vhfopt, ishls, jshls, kshls, lshls)) {
                                return;
                        }
                } else if (vhfopt->fprescreen == &CVHFnrs8_vk_prescreen) {
                        if (!CVHFnrs8_vk_prescreen_block(vhfopt, ishls, jshls, kshls, lshls)) {
                                return;
                        }
                }
        }

        if (ishls[0] > kshls[0]) {
                return CVHFdot_nrs4(intor, jkop, vjk, dms, buf, cache, n_dm,
                                    ishls, jshls, kshls, lshls, vhfopt, envs);
        } else if (ishls[0] < kshls[0]) {
                return;
        } else if ((ishls[1] <= jshls[0]) || (kshls[1] <= lshls[0])) {
                assert(ishls[1] == kshls[1]);
                return;
        }
        // else i == k && i >= j && k >= l
        assert(ishls[1] == kshls[1]);

        DECLARE_ALL;

        for (ish = ish0; ish < ish1; ish++) {
        for (jsh = jsh0; jsh < MIN(jsh1, ish+1); jsh++) {
        for (ksh = ksh0; ksh <= ish; ksh++) {
        for (lsh = lsh0; lsh < MIN(lsh1, ksh+1); lsh++) {
/* when ksh==ish, (lsh<jsh) misses some integrals (eg k<i&&l>j).
 * These integrals are calculated in the next (ish,jsh) pair. To show
 * that, we just need to prove that every elements in shell^4 appeared
 * only once in fjk_s8.  */
                if ((ksh == ish) && (lsh > jsh)) {
                        break;
                }
                INTOR_AND_CONTRACT;
        } } } }
}

JKArray *CVHFallocate_JKArray(JKOperator *op, int *shls_slice, int *ao_loc,
                              int ncomp, int nblock, int size_limit)
{
        return CVHFallocate_JKArray_bs(op, shls_slice, ao_loc, ncomp, nblock,
                                       size_limit, AO_BLOCK_SIZE);
}

// block_size: the AOs per block of the shell partition. The stack holds about
// size_limit/(block_size^2*ncomp) blocks, so keys_cache grows as blocks shrink.
JKArray *CVHFallocate_JKArray_bs(JKOperator *op, int *shls_slice, int *ao_loc,
                                 int ncomp, int nblock, int size_limit, int block_size)
{
        JKArray *jkarray = malloc(sizeof(JKArray));
        int ibra = op->ibra_shl0;
        int iket = op->iket_shl0;
        int obra = op->obra_shl0;
        int oket = op->oket_shl0;
        int v_bra_sh0 = shls_slice[obra];
        int v_ket_sh0 = shls_slice[oket];
        jkarray->v_ket_nsh  = shls_slice[oket+1] - shls_slice[oket];
        jkarray->dm_dims[0] = ao_loc[shls_slice[ibra+1]] - ao_loc[shls_slice[ibra]];
        jkarray->dm_dims[1] = ao_loc[shls_slice[iket+1]] - ao_loc[shls_slice[iket]];
        jkarray->offset0_outptr = v_bra_sh0 * jkarray->v_ket_nsh + v_ket_sh0;
        jkarray->nblock = nblock;
        int *outptr = malloc(sizeof(int) * nblock * nblock);
        jkarray->outptr = outptr;
        int i;
        for (i = 0; i < nblock * nblock; i++) {
                outptr[i] = NOVALUE;
        }
        jkarray->stack_size = 0;
        jkarray->data = malloc(sizeof(double) * (size_limit + 136*136));
        if (jkarray->data == NULL) {
                fprintf(stderr, "malloc(%zu) failed in CVHFallocate_JKArray\n",
                        sizeof(double) * (size_limit + 136*136));
                exit(1);
        }
        jkarray->ncomp = ncomp;
        int keys_max = size_limit / (block_size*block_size*ncomp);
        jkarray->keys_cache = malloc(sizeof(int) * keys_max);
        jkarray->key_counts = 0;
        return jkarray;
}

void CVHFdeallocate_JKArray(JKArray *jkarray)
{
        free(jkarray->outptr);
        free(jkarray->data);
        free(jkarray->keys_cache);
        free(jkarray);
}

double *CVHFallocate_and_reorder_dm(JKOperator *op, double *dm,
                                    int *shls_slice, int *ao_loc)
{
        int ibra = op->ibra_shl0;
        int iket = op->iket_shl0;
        int ish0 = shls_slice[ibra];
        int jsh0 = shls_slice[iket];
        int ish1 = shls_slice[ibra+1];
        int jsh1 = shls_slice[iket+1];
        int ioff = ao_loc[ish0];
        int joff = ao_loc[jsh0];
        int nrow = ao_loc[ish1] - ioff;
        int ncol = ao_loc[jsh1] - joff;
        double *out = malloc(sizeof(double) * nrow*ncol);
        int ish, jsh, i0, i1, j0, j1, i, j, ij;

        ij = 0;
        for (ish = ish0; ish < ish1; ish++) {
        for (jsh = jsh0; jsh < jsh1; jsh++) {
                i0 = ao_loc[ish  ] - ioff;
                i1 = ao_loc[ish+1] - ioff;
                j0 = ao_loc[jsh  ] - joff;
                j1 = ao_loc[jsh+1] - joff;
                for (i = i0; i < i1; i++) {
                for (j = j0; j < j1; j++, ij++) {
                        out[ij] = dm[i*ncol+j];
                } }
        } }
        return out;
}

void CVHFzero_out_vjk(double *vjk, JKOperator *op,
                      int *shls_slice, int *ao_loc, int ncomp)
{
        int obra = op->obra_shl0;
        int oket = op->oket_shl0;
        int ish0 = shls_slice[obra];
        int jsh0 = shls_slice[oket];
        int ish1 = shls_slice[obra+1];
        int jsh1 = shls_slice[oket+1];
        int nbra = ao_loc[ish1] - ao_loc[ish0];
        int nket = ao_loc[jsh1] - ao_loc[jsh0];
        NPdset0(vjk, ((size_t)nbra) * nket * ncomp);
}

// Divide shls into subblocks with roughly equal number of AOs in each block
int CVHFshls_block_partition(int *block_loc, int *shls_slice, int *ao_loc,
                             int block_size)
{
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int count = 0;
        if (ish0 >= ish1) {
                return count;
        }

        int ao_loc_last = ao_loc[ish0];
        int ish;

        count = 1;
        block_loc[0] = ish0;
        for (ish = ish0 + 1; ish < ish1; ish++) {
                if (ao_loc[ish+1] - ao_loc_last > block_size) {
                        block_loc[count] = ish;
                        count++;
                        ao_loc_last = ao_loc[ish];
                }
        }
        block_loc[count] = ish1;
        return count;
}

static int _shls_block_partition_lim(int *block_loc, int *shls_slice,
                                     int *shls_lim, int *ao_loc, int block_size);

/*
 * The tasks of CVHFnr_direct_drv and CVHFnr_direct_ex_drv are the (j,k,l)
 * blocks; each loops over all i blocks. A call on few AOs, or one cut to the
 * shells of an atom (gradients, Hessians), can have fewer 64-AO tasks than
 * threads. Halve the blocks, down to AO_BLOCK_SIZE_MIN AOs, until there are
 * TASKS_PER_THREAD tasks per thread. shls_lim: the excluded ranges of
 * CVHFnr_direct_ex_drv, or NULL.
 */
static int _jkl_block_partition(int *block_jloc, int *block_kloc, int *block_lloc,
                                size_t *nblock_j, size_t *nblock_k, size_t *nblock_l,
                                int *shls_slice, int *shls_lim, int *ao_loc)
{
        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        int block_size = AO_BLOCK_SIZE;
        while (1) {
                if (shls_lim == NULL) {
                        *nblock_j = CVHFshls_block_partition(block_jloc, shls_slice+2, ao_loc, block_size);
                        *nblock_k = CVHFshls_block_partition(block_kloc, shls_slice+4, ao_loc, block_size);
                        *nblock_l = CVHFshls_block_partition(block_lloc, shls_slice+6, ao_loc, block_size);
                } else {
                        *nblock_j = _shls_block_partition_lim(block_jloc, shls_slice+2, shls_lim+2, ao_loc, block_size);
                        *nblock_k = _shls_block_partition_lim(block_kloc, shls_slice+4, shls_lim+4, ao_loc, block_size);
                        *nblock_l = _shls_block_partition_lim(block_lloc, shls_slice+6, shls_lim+6, ao_loc, block_size);
                }
                if (block_size <= AO_BLOCK_SIZE_MIN ||
                    *nblock_j * *nblock_k * *nblock_l >= (size_t)TASKS_PER_THREAD * nthreads) {
                        return block_size;
                }
                block_size /= 2;
        }
}


/*
 * drv loop over ij, generate eris of kl for given ij, call fjk to
 * calculate vj, vk.
 * 
 * n_dm is the number of dms for one [array(ij|kl)], it is also the size of dms and vjk
 * ncomp is the number of components that produced by intor
 * shls_slice = [ishstart, ishend, jshstart, jshend, kshstart, kshend, lshstart, lshend]
 *
 * ao_loc[i+1] = ao_loc[i] + CINTcgto_spheric(i, bas)  for i = 0..nbas
 *
 * Return [(ptr[ncomp,nao,nao] in C-contiguous) for ptr in vjk]
 */
void CVHFnr_direct_drv_mode(int (*intor)(), void (*fdot)(), JKOperator **jkop,
                            double **dms, double **vjk, int n_dm, int ncomp,
                            int *shls_slice, int *ao_loc,
                            CINTOpt *cintopt, CVHFOpt *vhfopt, int int2e_batch,
                            int *atm, int natm, int *bas, int nbas, double *env)
{
        IntorEnvs envs = {natm, nbas, atm, bas, env, shls_slice, ao_loc, NULL,
                cintopt, ncomp};
        // the queues of quartets for qcint's batched intor, by shell type (l, nprim)
        DirectBatchIntor bi;
        int nt = 0;
        int *stype = _direct_batch_types(intor, int2e_batch, &bi, &nt, bas, nbas, env);
        int idm;
        double *tile_dms[n_dm];
        for (idm = 0; idm < n_dm; idm++) {
                CVHFzero_out_vjk(vjk[idm], jkop[idm], shls_slice, ao_loc, ncomp);
                tile_dms[idm] = CVHFallocate_and_reorder_dm(jkop[idm], dms[idm],
                                                            shls_slice, ao_loc);
        }

        int64_t di = GTOmax_shell_dim(ao_loc, shls_slice, 4);
        int64_t cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                               atm, natm, bas, nbas, env);
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int ksh0 = shls_slice[4];
        int ksh1 = shls_slice[5];
        int lsh0 = shls_slice[6];
        int lsh1 = shls_slice[7];
        int nish = ish1 - ish0;
        int njsh = jsh1 - jsh0;
        int nksh = ksh1 - ksh0;
        int nlsh = lsh1 - lsh0;
        int *block_iloc = malloc(sizeof(int) * (nish + njsh + nksh + nlsh + 4));
        int *block_jloc = block_iloc + nish + 1;
        int *block_kloc = block_jloc + njsh + 1;
        int *block_lloc = block_kloc + nksh + 1;
        // size_t to keep nblock^3 from overflowing for large molecules. The
        // same fix was applied to nr_sr_vhf.c.
        size_t nblock_j, nblock_k, nblock_l;
        int block_size = _jkl_block_partition(block_jloc, block_kloc, block_lloc,
                                              &nblock_j, &nblock_k, &nblock_l,
                                              shls_slice, NULL, ao_loc);
        size_t nblock_i = CVHFshls_block_partition(block_iloc, shls_slice+0, ao_loc, block_size);
        size_t nblock_kl = nblock_k * nblock_l;
        size_t nblock_jkl = nblock_j * nblock_kl;
        int nblock_max = MAX(nblock_i, nblock_j);
        nblock_max = MAX(nblock_max, nblock_k);
        nblock_max = MAX(nblock_max, nblock_l);
        // up to 3.2 GB per thread.
        int64_t size_limit = (400000000 - di*di*di*di*ncomp - cache_size) / n_dm;
        if (size_limit < 0) {
                fprintf(stderr, "Insufficient memory for caching CVHFnr_direct_drv intermediates\n");
                exit(1);
        }
        // the queues outlive the blocks if every output fits in its stack, and its block pairs in
        // its keys_cache (CVHFallocate_JKArray_bs), until the end (DirectBatch)
        int persist = 1;
        size_t nblocks[4] = {nblock_i, nblock_j, nblock_k, nblock_l};
        int64_t keys_max = size_limit / (block_size * block_size * ncomp);
        for (idm = 0; idm < n_dm; idm++) {
                JKOperator *op = jkop[idm];
                if ((int64_t)(*op->data_size)(shls_slice, ao_loc) * ncomp >= size_limit ||
                    (int64_t)(nblocks[op->obra_shl0/2] * nblocks[op->oket_shl0/2]) > keys_max) {
                        persist = 0;
                }
        }
        int *block_loc[4] = {block_iloc, block_jloc, block_kloc, block_lloc};

#pragma omp parallel
{
        int ioff = ao_loc[ish0];
        int joff = ao_loc[jsh0];
        int koff = ao_loc[ksh0];
        int loff = ao_loc[lsh0];
        int i, j, k, l, n;
        size_t r, blk_id;
        JKArray *v_priv[n_dm];
        for (i = 0; i < n_dm; i++) {
                v_priv[i] = CVHFallocate_JKArray_bs(jkop[i], shls_slice, ao_loc,
                                                    ncomp, nblock_max, size_limit,
                                                    block_size);
        }
        double *buf = malloc(sizeof(double) * (di*di*di*di*ncomp + di*di*2 + cache_size));
        double *cache = buf + di*di*di*di*ncomp;
        IntorEnvs tenvs = envs;
        DirectBatch *bt = NULL;
        if (stype != NULL) {
                bt = _direct_batch_new(&bi, nt, stype, di, persist, block_loc);
                tenvs.batch = bt;
        }
#pragma omp for nowait schedule(dynamic, 1)
        for (blk_id = 0; blk_id < nblock_jkl; blk_id++) {
                // TODO: add schwarz ineq conditioner
                r = blk_id;
                j = r / nblock_kl ; r = r % nblock_kl;
                k = r / nblock_l  ; r = r % nblock_l;
                l = r;
                int j0 = ao_loc[block_jloc[j]];
                int k0 = ao_loc[block_kloc[k]];
                int l0 = ao_loc[block_lloc[l]];
                int j1 = ao_loc[block_jloc[j+1]];
                int k1 = ao_loc[block_kloc[k+1]];
                int l1 = ao_loc[block_lloc[l+1]];
                for (n = 0; n < n_dm; n++) {
                        JKArray *pv = v_priv[n];
                        pv->ao_off[1] = j0 - joff;
                        pv->ao_off[2] = k0 - koff;
                        pv->ao_off[3] = l0 - loff;
                        pv->shape[1] = j1 - j0;
                        pv->shape[2] = k1 - k0;
                        pv->shape[3] = l1 - l0;
                        pv->block_quartets[1] = j;
                        pv->block_quartets[2] = k;
                        pv->block_quartets[3] = l;
                }
                for (i = 0; i < nblock_i; i++) {
                        int i0 = ao_loc[block_iloc[i]];
                        int i1 = ao_loc[block_iloc[i+1]];
                        for (n = 0; n < n_dm; n++) {
                                JKArray *pv = v_priv[n];
                                pv->ao_off[0] = i0 - ioff;
                                pv->shape[0] = i1 - i0;
                                pv->block_quartets[0] = i;
                        }
                        if (bt != NULL) {
                                bt->cur[0] = i;
                                bt->cur[1] = j;
                                bt->cur[2] = k;
                                bt->cur[3] = l;
                        }
                        (*fdot)(intor, jkop, v_priv, tile_dms, buf, cache, n_dm,
                                block_iloc+i, block_jloc+j, block_kloc+k, block_lloc+l,
                                vhfopt, &tenvs);
                        if (bt != NULL && !bt->persist) {
                                _direct_batch_flush_all(bt, intor, jkop, v_priv, tile_dms,
                                                        buf, cache, n_dm, &tenvs);
                        }
                }
                for (n = 0; n < n_dm; n++) {
                        if (v_priv[n]->stack_size >= size_limit) {
#pragma omp critical
        (*jkop[n]->write_back)(vjk[n], v_priv[n], shls_slice, ao_loc,
                               block_iloc, block_jloc, block_kloc, block_lloc);
                        }
                }
        }
        if (bt != NULL) {
                _direct_batch_flush_all(bt, intor, jkop, v_priv, tile_dms, buf, cache, n_dm, &tenvs);
        }
        for (n = 0; n < n_dm; n++) {
                if (v_priv[n]->stack_size > 0) {
#pragma omp critical
        (*jkop[n]->write_back)(vjk[n], v_priv[n], shls_slice, ao_loc,
                               block_iloc, block_jloc, block_kloc, block_lloc);
                }
                CVHFdeallocate_JKArray(v_priv[n]);
        }
        free(buf);
        if (bt != NULL) {
                _direct_batch_del(bt);
        }
}
        for (idm = 0; idm < n_dm; idm++) {
                free(tile_dms[idm]);
        }
        free(block_iloc);
        if (stype != NULL) {
                free(stype);
        }
}

void CVHFnr_direct_drv(int (*intor)(), void (*fdot)(), JKOperator **jkop,
                       double **dms, double **vjk, int n_dm, int ncomp,
                       int *shls_slice, int *ao_loc,
                       CINTOpt *cintopt, CVHFOpt *vhfopt,
                       int *atm, int natm, int *bas, int nbas, double *env)
{
        CVHFnr_direct_drv_mode(intor, fdot, jkop, dms, vjk, n_dm, ncomp, shls_slice, ao_loc,
                               cintopt, vhfopt, 1, atm, natm, bas, nbas, env);
}

// Divide shells into subblocks with two cutting points specified in shls_lim.
// The cutting points should not be placed inside any subblocks.
static int _shls_block_partition_lim(int *block_loc, int *shls_slice,
                                     int *shls_lim, int *ao_loc, int block_size)
{
        int lim0 = shls_lim[0];
        int lim1 = shls_lim[1];
        if (lim0 >= lim1) {
                return CVHFshls_block_partition(block_loc, shls_slice, ao_loc, block_size);
        }

        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int seg[4] = {ish0, lim0, lim1, ish1};
        int count = CVHFshls_block_partition(block_loc, seg, ao_loc, block_size);
        count += CVHFshls_block_partition(block_loc+count, seg+1, ao_loc, block_size);
        count += CVHFshls_block_partition(block_loc+count, seg+2, ao_loc, block_size);
        return count;
}

/*
 * This driver is analogous to CVHFnr_direct_drv . It is to calculate vj, vk
 * excluding the eri block [ix0:ix1,jx0:jx1,kx0:kx1,lx0:lx1].
 * The boundary ix, jx, kx, lx are provided by shls_slice[8:16]
 */
void CVHFnr_direct_ex_drv_mode(int (*intor)(), void (*fdot)(), JKOperator **jkop,
                               double **dms, double **vjk, int n_dm, int ncomp,
                               int *shls_slice, int *ao_loc,
                               CINTOpt *cintopt, CVHFOpt *vhfopt, int int2e_batch,
                               int *atm, int natm, int *bas, int nbas, double *env)
{
        IntorEnvs envs = {natm, nbas, atm, bas, env, shls_slice, ao_loc, NULL,
                cintopt, ncomp};
        DirectBatchIntor bi;
        int nt = 0;
        int *stype = _direct_batch_types(intor, int2e_batch, &bi, &nt, bas, nbas, env);
        int idm;
        double *tile_dms[n_dm];
        for (idm = 0; idm < n_dm; idm++) {
                CVHFzero_out_vjk(vjk[idm], jkop[idm], shls_slice, ao_loc, ncomp);
                tile_dms[idm] = CVHFallocate_and_reorder_dm(jkop[idm], dms[idm],
                                                            shls_slice, ao_loc);
        }

        int64_t di = GTOmax_shell_dim(ao_loc, shls_slice, 4);
        int64_t cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                               atm, natm, bas, nbas, env);
        int ish0 = shls_slice[0];
        int ish1 = shls_slice[1];
        int jsh0 = shls_slice[2];
        int jsh1 = shls_slice[3];
        int ksh0 = shls_slice[4];
        int ksh1 = shls_slice[5];
        int lsh0 = shls_slice[6];
        int lsh1 = shls_slice[7];
        int nish = ish1 - ish0;
        int njsh = jsh1 - jsh0;
        int nksh = ksh1 - ksh0;
        int nlsh = lsh1 - lsh0;
        int *shls_excludes = shls_slice + 8;
        int *block_iloc = malloc(sizeof(int) * (nish + njsh + nksh + nlsh + 4));
        int *block_jloc = block_iloc + nish + 1;
        int *block_kloc = block_jloc + njsh + 1;
        int *block_lloc = block_kloc + nksh + 1;
        size_t nblock_j, nblock_k, nblock_l;
        int block_size = _jkl_block_partition(block_jloc, block_kloc, block_lloc,
                                              &nblock_j, &nblock_k, &nblock_l,
                                              shls_slice, shls_excludes, ao_loc);
        size_t nblock_i = _shls_block_partition_lim(block_iloc, shls_slice+0, shls_excludes+0, ao_loc, block_size);
        size_t nblock_kl = nblock_k * nblock_l;
        size_t nblock_jkl = nblock_j * nblock_kl;
        int nblock_max = MAX(nblock_i, nblock_j);
        nblock_max = MAX(nblock_max, nblock_k);
        nblock_max = MAX(nblock_max, nblock_l);
        // up to 3.2 GB per thread.
        int64_t size_limit = (400000000 - di*di*di*di*ncomp - cache_size) / n_dm;
        if (size_limit < 0) {
                fprintf(stderr, "Insufficient memory for caching CVHFnr_direct_ex_drv intermediates\n");
                exit(1);
        }
        // the queues outlive the blocks if every output fits in its stack, and its block pairs in
        // its keys_cache (CVHFallocate_JKArray_bs), until the end (DirectBatch)
        int persist = 1;
        size_t nblocks[4] = {nblock_i, nblock_j, nblock_k, nblock_l};
        int64_t keys_max = size_limit / (block_size * block_size * ncomp);
        for (idm = 0; idm < n_dm; idm++) {
                JKOperator *op = jkop[idm];
                if ((int64_t)(*op->data_size)(shls_slice, ao_loc) * ncomp >= size_limit ||
                    (int64_t)(nblocks[op->obra_shl0/2] * nblocks[op->oket_shl0/2]) > keys_max) {
                        persist = 0;
                }
        }
        int *block_loc[4] = {block_iloc, block_jloc, block_kloc, block_lloc};

#pragma omp parallel
{
        int ioff = ao_loc[ish0];
        int joff = ao_loc[jsh0];
        int koff = ao_loc[ksh0];
        int loff = ao_loc[lsh0];
        int i, j, k, l, n;
        size_t r, blk_id;
        JKArray *v_priv[n_dm];
        for (i = 0; i < n_dm; i++) {
                v_priv[i] = CVHFallocate_JKArray_bs(jkop[i], shls_slice, ao_loc,
                                                    ncomp, nblock_max, size_limit,
                                                    block_size);
        }
        double *buf = malloc(sizeof(double) * (di*di*di*di*ncomp + di*di*2 + cache_size));
        double *cache = buf + di*di*di*di*ncomp;
        IntorEnvs tenvs = envs;
        DirectBatch *bt = NULL;
        if (stype != NULL) {
                bt = _direct_batch_new(&bi, nt, stype, di, persist, block_loc);
                tenvs.batch = bt;
        }
#pragma omp for nowait schedule(dynamic, 1)
        for (blk_id = 0; blk_id < nblock_jkl; blk_id++) {
                r = blk_id;
                j = r / nblock_kl ; r = r % nblock_kl;
                k = r / nblock_l  ; r = r % nblock_l;
                l = r;
                int j0 = ao_loc[block_jloc[j]];
                int k0 = ao_loc[block_kloc[k]];
                int l0 = ao_loc[block_lloc[l]];
                int j1 = ao_loc[block_jloc[j+1]];
                int k1 = ao_loc[block_kloc[k+1]];
                int l1 = ao_loc[block_lloc[l+1]];
                for (n = 0; n < n_dm; n++) {
                        JKArray *pv = v_priv[n];
                        pv->ao_off[1] = j0 - joff;
                        pv->ao_off[2] = k0 - koff;
                        pv->ao_off[3] = l0 - loff;
                        pv->shape[1] = j1 - j0;
                        pv->shape[2] = k1 - k0;
                        pv->shape[3] = l1 - l0;
                        pv->block_quartets[1] = j;
                        pv->block_quartets[2] = k;
                        pv->block_quartets[3] = l;
                }
                for (i = 0; i < nblock_i; i++) {
                        // Skip integrals in the segment [ix0:ix1,jx0:jx1,kx0:kx1,lx0:lx1] for
                        // ix0, ix1, jx0, jx1, kx0, kx1, lx0, lx1 = shls_excludes
                        if (shls_excludes[0] <= block_iloc[i] && block_iloc[i] < shls_excludes[1] &&
                            shls_excludes[2] <= block_jloc[j] && block_jloc[j] < shls_excludes[3] &&
                            shls_excludes[4] <= block_kloc[k] && block_kloc[k] < shls_excludes[5] &&
                            shls_excludes[6] <= block_lloc[l] && block_lloc[l] < shls_excludes[7]) {
                                continue;
                        }
                        int i0 = ao_loc[block_iloc[i]];
                        int i1 = ao_loc[block_iloc[i+1]];
                        for (n = 0; n < n_dm; n++) {
                                JKArray *pv = v_priv[n];
                                pv->ao_off[0] = i0 - ioff;
                                pv->shape[0] = i1 - i0;
                                pv->block_quartets[0] = i;
                        }
                        if (bt != NULL) {
                                bt->cur[0] = i;
                                bt->cur[1] = j;
                                bt->cur[2] = k;
                                bt->cur[3] = l;
                        }
                        (*fdot)(intor, jkop, v_priv, tile_dms, buf, cache, n_dm,
                                block_iloc+i, block_jloc+j, block_kloc+k, block_lloc+l,
                                vhfopt, &tenvs);
                        if (bt != NULL && !bt->persist) {
                                _direct_batch_flush_all(bt, intor, jkop, v_priv, tile_dms,
                                                        buf, cache, n_dm, &tenvs);
                        }
                }
                for (n = 0; n < n_dm; n++) {
                        if (v_priv[n]->stack_size >= size_limit) {
#pragma omp critical
        (*jkop[n]->write_back)(vjk[n], v_priv[n], shls_slice, ao_loc,
                               block_iloc, block_jloc, block_kloc, block_lloc);
                        }
                }
        }
        if (bt != NULL) {
                _direct_batch_flush_all(bt, intor, jkop, v_priv, tile_dms, buf, cache, n_dm, &tenvs);
        }
        for (n = 0; n < n_dm; n++) {
                if (v_priv[n]->stack_size > 0) {
#pragma omp critical
        (*jkop[n]->write_back)(vjk[n], v_priv[n], shls_slice, ao_loc,
                               block_iloc, block_jloc, block_kloc, block_lloc);
                }
                CVHFdeallocate_JKArray(v_priv[n]);
        }
        free(buf);
        if (bt != NULL) {
                _direct_batch_del(bt);
        }
}
        for (idm = 0; idm < n_dm; idm++) {
                free(tile_dms[idm]);
        }
        free(block_iloc);
        if (stype != NULL) {
                free(stype);
        }
}

void CVHFnr_direct_ex_drv(int (*intor)(), void (*fdot)(), JKOperator **jkop,
                          double **dms, double **vjk, int n_dm, int ncomp,
                          int *shls_slice, int *ao_loc,
                          CINTOpt *cintopt, CVHFOpt *vhfopt,
                          int *atm, int natm, int *bas, int nbas, double *env)
{
        CVHFnr_direct_ex_drv_mode(intor, fdot, jkop, dms, vjk, n_dm, ncomp, shls_slice, ao_loc,
                                  cintopt, vhfopt, 1, atm, natm, bas, nbas, env);
}
