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
 * The loop of the energy-type two-electron drivers (grad_jk_energy.c,
 * hess_jk_energy.c): the s8 orbits of shell quartets I >= J, K >= L,
 * (IJ) >= (KL), each planned by the client (its screening, the integrals it
 * takes) and evaluated at once, or queued with others of its kind for qcint's
 * batched integrals.
 *
 * Pairs: (I,J), I >= J, with the client's bound qpair[I*nbas+J]. Pairs with
 * qpair * max(qpair) * fac <= thr are dropped and the rest sorted ascending
 * (then by I, J). For pair t the partners u <= t are visited from t down while
 * q_t q_u fac > thr: the bound decreases, so the loop stops at the first that
 * fails it.
 *
 * Chunks: a fixed partition of the sorted pairs into ranges of equal (t+1)
 * sums, taken heaviest first by the threads. Each chunk adds into its own
 * buffer of chunk_size doubles, and the buffers are summed in chunk order: the
 * result does not depend on the thread count or the schedule. nchunk =
 * min(nchunk_max, max_buf / chunk_size, the pairs), at least 1. chunk_size 0:
 * no buffers (acc NULL), the client accumulates on its own.
 *
 * Queues (lanes > 0): an orbit planned with a key != 0 joins the queue of its
 * key in a hash table of qcap (a power of 2) entries per thread. A queue holding
 * lanes orbits is evaluated at once (eval_batch). The table is flushed in table
 * order at the end of each chunk and when 3/4 full; a queue of fewer than
 * lanes/2 orbits goes orbit by orbit (eval).
 */

#include <stdint.h>
#include <stddef.h>

#define OE_MAX_LANES    8
#define OE_ORBIT_MAX    256     // bytes of a client's orbit

typedef struct {
        void *ctx;              // the client's
        int nbas;
        const double *qpair;    // [nbas,nbas], read for I >= J
        double fac;
        double thr;
        size_t chunk_size;      // doubles of a chunk's buffer, or 0
        int nchunk_max;
        size_t max_buf;         // doubles all chunk buffers may take
        int lanes;              // the orbits a queue evaluates together, 0: no queues
        int qcap;               // hash table entries per thread, a power of 2
        size_t orbit_size;      // bytes of the client's orbit, at most OE_ORBIT_MAX
        // per-thread state of the client; thread_done runs in a critical section
        // (to add the thread's counts to ctx) and frees it
        void *(*thread_init)(void *ctx);
        void (*thread_done)(void *ctx, void *th);
        // 0: skip the orbit (ish,jsh,ksh,lsh); else the orbit, and its queue key
        // (0: not batched, evaluated at once)
        int (*plan)(void *ctx, void *th, int ish, int jsh, int ksh, int lsh,
                    void *orbit, uint64_t *key);
        void (*eval)(void *ctx, void *th, double *acc, void *orbit);
        // n orbits of one key, orbit_size bytes apart
        void (*eval_batch)(void *ctx, void *th, double *acc, void *orbits, int n);
} OEDriver;

// out[chunk_size] = the sum of the chunks' buffers (if chunk_size > 0)
void CVHFoe_run(OEDriver *d, double *out);

/*
 * Helpers of the clients: the shell orders of an orbit's integrals, and Gamma
 * in them (data movement only; each client keeps its own arithmetic).
 */
#define OE_IJKL         0
#define OE_JIKL         1
#define OE_KLIJ         2
#define OE_LKIJ         3
#define OE_IJLK         4
#define OE_JILK         5
#define OE_NORDER       6
extern const int CVHFoe_order_sh[OE_NORDER][4];

// out = g (in the layout of (I,J,K,L)) in the layout of the shell order o
void CVHFoe_permute(double *out, const double *g, int o, int di, int dj, int dk, int dl);
