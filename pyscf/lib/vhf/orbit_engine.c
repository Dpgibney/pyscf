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
 * The orbit loop of the energy-type two-electron drivers; see orbit_engine.h.
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "config.h"
#include "np_helper/np_helper.h"
#include "orbit_engine.h"

typedef struct {
        double q;
        int i;
        int j;
} OEPair;

static int _cmp_pair(const void *a, const void *b)
{
        const OEPair *pa = a;
        const OEPair *pb = b;
        if (pa->q != pb->q) {
                return pa->q < pb->q ? -1 : 1;
        }
        if (pa->i != pb->i) {
                return pa->i - pb->i;
        }
        return pa->j - pb->j;
}

// The queues of a thread: qcap entries of a key (0: empty), a count and room for lanes orbits
typedef struct {
        uint64_t *key;
        int *n;
        char *o;                // [qcap][lanes][orbit_size]
        int nused;
        int bits;               // log2(qcap)
} OETable;

static void _flush_queue(OEDriver *d, void *th, double *acc, OETable *tb, size_t x)
{
        char *o = tb->o + d->orbit_size * d->lanes * x;
        const int n = tb->n[x];
        if (2 * n < d->lanes) {
                int i;
                for (i = 0; i < n; i++) {
                        d->eval(d->ctx, th, acc, o + d->orbit_size * i);
                }
        } else {
                d->eval_batch(d->ctx, th, acc, o, n);
        }
        tb->n[x] = 0;
}

static void _flush_all(OEDriver *d, void *th, double *acc, OETable *tb)
{
        size_t x;
        for (x = 0; x < (size_t)d->qcap; x++) {
                if (tb->key[x] != 0 && tb->n[x] > 0) {
                        _flush_queue(d, th, acc, tb, x);
                }
                tb->key[x] = 0;
        }
        tb->nused = 0;
}

// Queues the orbit, evaluating its queue when full
static void _queue(OEDriver *d, void *th, double *acc, OETable *tb, uint64_t key, void *orbit)
{
        const int qcap = d->qcap;
        if (tb->nused * 4 >= qcap * 3) {
                _flush_all(d, th, acc, tb);
        }
        size_t x = (size_t)((key * 0x9E3779B97F4A7C15ull) >> (64 - tb->bits)) & (qcap - 1);
        while (tb->key[x] != 0 && tb->key[x] != key) {
                x = (x + 1) & (qcap - 1);
        }
        if (tb->key[x] == 0) {
                tb->key[x] = key;
                tb->n[x] = 0;
                tb->nused++;
        }
        memcpy(tb->o + d->orbit_size * (d->lanes * x + tb->n[x]), orbit, d->orbit_size);
        if (++tb->n[x] == d->lanes) {
                _flush_queue(d, th, acc, tb, x);
        }
}

/*
 * The progress of the run in flight, for a caller's other thread: the chunks of
 * shell pairs done and all of them (0, 0 before the first run).
 */
static volatile int _oe_done = 0;
static volatile int _oe_total = 0;

void CVHFoe_progress(int *done_total)
{
        done_total[0] = _oe_done;
        done_total[1] = _oe_total;
}

void CVHFoe_run(OEDriver *d, double *out)
{
        const int nbas = d->nbas;
        const double *qpair = d->qpair;
        const double fac = d->fac;
        const double thr = d->thr;
        size_t i;
        int ish, jsh;

        size_t npair = (size_t)nbas * (nbas + 1) / 2;
        OEPair *pairs = malloc(sizeof(OEPair) * npair);
        double qmax = 0;
        size_t nsig = 0;
        for (ish = 0; ish < nbas; ish++) {
        for (jsh = 0; jsh <= ish; jsh++) {
                double q = qpair[ish*nbas+jsh];
                qmax = MAX(qmax, q);
                pairs[nsig].q = q;
                pairs[nsig].i = ish;
                pairs[nsig].j = jsh;
                nsig++;
        } }
        npair = nsig;
        nsig = 0;
        for (i = 0; i < npair; i++) {
                if (pairs[i].q * qmax * fac > thr) {
                        pairs[nsig] = pairs[i];
                        nsig++;
                }
        }
        qsort(pairs, nsig, sizeof(OEPair), _cmp_pair);

        // chunks of equal (t+1) sums, taken heaviest first
        const size_t chunk_size = d->chunk_size;
        int nchunk = d->nchunk_max;
        if (chunk_size > 0) {
                nchunk = MIN(nchunk, d->max_buf / chunk_size);
        }
        nchunk = MIN(nchunk, (int)nsig);
        nchunk = MAX(nchunk, 1);
        size_t *chunk_loc = malloc(sizeof(size_t) * (nchunk + 1));
        double area = (double)nsig * (nsig + 1) / 2;
        int c = 0;
        chunk_loc[0] = 0;
        for (i = 0; i < nsig && c < nchunk - 1; i++) {
                if ((double)(i+1) * (i+2) / 2 >= area * (c + 1) / nchunk) {
                        c++;
                        chunk_loc[c] = i + 1;
                }
        }
        for (c++; c <= nchunk; c++) {
                chunk_loc[c] = nsig;
        }
        double *acc_chunks = NULL;
        if (chunk_size > 0) {
                acc_chunks = calloc((size_t)nchunk * chunk_size, sizeof(double));
        }
        int bits = 0;
        while ((1 << bits) < d->qcap) {
                bits++;
        }
        _oe_done = 0;
        _oe_total = nchunk;

#pragma omp parallel
{
        void *th = d->thread_init(d->ctx);
        OETable tb = {NULL, NULL, NULL, 0, bits};
        if (d->lanes) {
                tb.key = calloc(d->qcap, sizeof(uint64_t));
                tb.n = calloc(d->qcap, sizeof(int));
                tb.o = malloc(d->orbit_size * d->lanes * d->qcap);
        }
        double orbit[OE_ORBIT_MAX / sizeof(double)];
        uint64_t key;
        size_t t, u;
        int ic;
#pragma omp for schedule(dynamic, 1)
        for (ic = 0; ic < nchunk; ic++) {
                int c = nchunk - 1 - ic;
                double *acc = acc_chunks == NULL ? NULL : acc_chunks + (size_t)c * chunk_size;
                for (t = chunk_loc[c]; t < chunk_loc[c+1]; t++) {
                        double qt = pairs[t].q * fac;
                        int ish = pairs[t].i;
                        int jsh = pairs[t].j;
                        for (u = t + 1; u-- > 0;) {
                                if (qt * pairs[u].q <= thr) {
                                        break;
                                }
                                if (!d->plan(d->ctx, th, ish, jsh, pairs[u].i, pairs[u].j,
                                             orbit, &key)) {
                                        continue;
                                }
                                if (d->lanes && key != 0) {
                                        _queue(d, th, acc, &tb, key, orbit);
                                } else {
                                        d->eval(d->ctx, th, acc, orbit);
                                }
                        }
                }
                if (d->lanes) {
                        _flush_all(d, th, acc, &tb);
                }
#pragma omp atomic
                _oe_done++;
        }
        if (d->lanes) {
                free(tb.key);
                free(tb.n);
                free(tb.o);
        }
#pragma omp critical
        d->thread_done(d->ctx, th);
}
        if (chunk_size > 0) {
                NPdset0(out, chunk_size);
                for (c = 0; c < nchunk; c++) {
                        double *acc = acc_chunks + (size_t)c * chunk_size;
                        for (i = 0; i < chunk_size; i++) {
                                out[i] += acc[i];
                        }
                }
                free(acc_chunks);
        }
        free(chunk_loc);
        free(pairs);
}

const int CVHFoe_order_sh[OE_NORDER][4] = {
        {0, 1, 2, 3}, {1, 0, 2, 3}, {2, 3, 0, 1}, {3, 2, 0, 1}, {0, 1, 3, 2}, {1, 0, 3, 2}};

void CVHFoe_permute(double *out, const double *g, int o, int di, int dj, int dk, int dl)
{
        const int dij = di * dj;
        const int dkl = dk * dl;
        int p, q, r, s, pq, rs;
        switch (o) {
        case OE_JIKL:
                for (rs = 0; rs < dkl; rs++) {
                        const double *pg = g + rs * dij;
                        double *po = out + rs * dij;
                        for (p = 0; p < di; p++) {
                        for (q = 0; q < dj; q++) {
                                po[q+dj*p] = pg[p+di*q];
                        } }
                }
                break;
        case OE_KLIJ:
                for (pq = 0; pq < dij; pq++) {
                for (rs = 0; rs < dkl; rs++) {
                        out[rs+dkl*pq] = g[pq+dij*rs];
                } }
                break;
        case OE_LKIJ:
                for (pq = 0; pq < dij; pq++) {
                for (r = 0; r < dk; r++) {
                for (s = 0; s < dl; s++) {
                        out[s+dl*(r+dk*pq)] = g[pq+dij*(r+dk*s)];
                } } }
                break;
        case OE_IJLK:
                for (r = 0; r < dk; r++) {
                for (s = 0; s < dl; s++) {
                        memcpy(out + dij * (s + dl * r), g + dij * (r + dk * s),
                               sizeof(double) * dij);
                } }
                break;
        case OE_JILK:
                for (r = 0; r < dk; r++) {
                for (s = 0; s < dl; s++) {
                        const double *pg = g + dij * (r + dk * s);
                        double *po = out + dij * (s + dl * r);
                        for (p = 0; p < di; p++) {
                        for (q = 0; q < dj; q++) {
                                po[q+dj*p] = pg[p+di*q];
                        } }
                } }
                break;
        default:
                memcpy(out, g, sizeof(double) * dij * dkl);
        }
}
