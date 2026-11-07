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
 * Two-electron part of the nuclear gradient, contracted inside the integral
 * loop. No J/K matrices are built.
 *
 * E2 = cj/2 sum_pqrs (pq|rs) Dj_pq Dj_rs
 *    - ck/2 sum_n sum_pqrs (pq|rs) Dk[n]_pr Dk[n]_qs
 *
 * With Gamma_pqrs = cj Dj_pq Dj_rs - ck/2 sum_n (Dk_pr Dk_qs + Dk_ps Dk_qr),
 * which has the 8-fold symmetry of the integrals, each s8 orbit of shells
 * I >= J, K >= L, (IJ) >= (KL) is visited once:
 *
 *     dE2/dR_A = -deg/2 sum_{centers c on atom A} G_c,
 *     G_c = sum_{pqrs in IJKL} (nabla_c pq|rs) Gamma_pqrs
 *
 * where nabla_c acts on the function of center c and deg is the number of
 * distinct permutations of the orbit. G_c is an int2e_ip1 integral of the
 * shells permuted to put c first. By translational invariance
 * G_I + G_J + G_K + G_L = 0, so the centers on one atom need no integrals
 * when the others are evaluated: 4 centers on 4 atoms take 3 evaluations,
 * 2 + 2 take 2, 3 + 1 take 1, one atom takes none. The s2kl driver behind
 * grad.rhf.get_jk evaluates one integral per center (4 per orbit).
 *
 * Screening is that of CVHFgrad_{jk,j,k}_prescreen for each of the 4
 * orderings, as the s2kl driver applies it. An orbit is skipped when no
 * ordering passes. Otherwise it takes the cheaper of: the orderings that pass,
 * or all 4 centers through translational invariance. Both include everything
 * the s2kl driver includes.
 *
 * The loop over the orbits, their chunks (summed in a fixed order: the result
 * does not depend on the thread count or the schedule) and the queues of the
 * batched integrals are those of orbit_engine.h.
 */

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "config.h"
#include "cint.h"
#include "optimizer.h"
#include "np_helper/np_helper.h"
#include "gto/gto.h"
#include "orbit_engine.h"

// Upper limits of the chunk count: chunks in all, and chunk buffer doubles
#define EJK_NCHUNK      4096
#define EJK_CHUNK_BUF   4000000

// CVHFgrad_jk_prescreen (with_j and with_k), CVHFgrad_j_prescreen (with_j)
// or CVHFgrad_k_prescreen (with_k) for the shells (a,b,c,d)
static int _screen(const double *q0, const double *q1, const double *dmc,
                   int n, int a, int b, int c, int d, double cutoff,
                   int with_j, int with_k)
{
        double qijkl = q0[a*n+b] * q1[c*n+d];
        double dmin = cutoff / qijkl;
        return qijkl > cutoff
            && ((with_j && 2*dmc[d*n+c] > dmin)
             || (with_k && ((dmc[b*n+c] > dmin) || (dmc[b*n+d] > dmin))));
}

// gam[p+di*(q+dj*(r+dk*s))] = Gamma_pqrs for the AOs p0+p, q0+q, r0+r, s0+s.
// The densities are symmetric; they are read along rows.
static void _gamma(double *gam, double *dmj, double *dmk, int nk,
                   double cj, double ck, int nao,
                   int p0, int di, int q0, int dj, int r0, int dk, int s0, int dl)
{
        const double hk = .5 * ck;
        const int dij = di * dj;
        int p, q, r, s, n;
        double *pg;
        for (s = 0; s < dl; s++) {
        for (r = 0; r < dk; r++) {
                pg = gam + (r + dk * s) * dij;
                if (cj != 0) {
                        double drs = cj * dmj[(size_t)(r0+r)*nao+s0+s];
                        for (q = 0; q < dj; q++) {
                                double *dq = dmj + (size_t)(q0+q)*nao + p0;
                                for (p = 0; p < di; p++) {
                                        pg[p+di*q] = drs * dq[p];
                                }
                        }
                } else {
                        for (p = 0; p < dij; p++) {
                                pg[p] = 0;
                        }
                }
                for (n = 0; n < nk; n++) {
                        double *d = dmk + (size_t)n * nao * nao;
                        double *dr = d + (size_t)(r0+r)*nao + p0;
                        double *ds = d + (size_t)(s0+s)*nao + p0;
                        for (q = 0; q < dj; q++) {
                                double dqs = hk * d[(size_t)(q0+q)*nao+s0+s];
                                double dqr = hk * d[(size_t)(q0+q)*nao+r0+r];
                                for (p = 0; p < di; p++) {
                                        pg[p+di*q] -= dr[p] * dqs + ds[p] * dqr;
                                }
                        }
                }
        } }
}

/*
 * g[x] = sum_pqrs (nabla_c pq|rs)[x] Gamma_pqrs from the int2e_ip1 buffer of
 * the ordering that puts center c first:
 *   c = 0: (I,J,K,L), buf[p+di*(q+dj*rs)]
 *   c = 1: (J,I,K,L), buf[q+dj*(p+di*rs)]
 *   c = 2: (K,L,I,J), buf[rs+dkl*pq]
 *   c = 3: (L,K,I,J), buf[s+dl*(r+dk*pq)]
 * with pq = p+di*q, rs = r+dk*s and Gamma in the layout of c = 0.
 */
static void _contract(double *g, double *buf, double *gam, int c,
                      int di, int dj, int dk, int dl)
{
        const int dij = di * dj;
        const int dkl = dk * dl;
        const int n = dij * dkl;
        double *bx = buf;
        double *by = buf + n;
        double *bz = buf + n * 2;
        double sx = 0;
        double sy = 0;
        double sz = 0;
        double gm;
        int p, q, r, s, pq, rs, i, ib;
        switch (c) {
        case 0:
                for (i = 0; i < n; i++) {
                        gm = gam[i];
                        sx += bx[i] * gm;
                        sy += by[i] * gm;
                        sz += bz[i] * gm;
                }
                break;
        case 1:
                for (rs = 0; rs < dkl; rs++) {
                for (p = 0; p < di; p++) {
                for (q = 0; q < dj; q++) {
                        gm = gam[rs*dij+p+di*q];
                        ib = rs*dij+q+dj*p;
                        sx += bx[ib] * gm;
                        sy += by[ib] * gm;
                        sz += bz[ib] * gm;
                } } }
                break;
        case 2:
                for (pq = 0; pq < dij; pq++) {
                for (rs = 0; rs < dkl; rs++) {
                        gm = gam[pq+dij*rs];
                        ib = rs+dkl*pq;
                        sx += bx[ib] * gm;
                        sy += by[ib] * gm;
                        sz += bz[ib] * gm;
                } }
                break;
        default:
                for (pq = 0; pq < dij; pq++) {
                for (r = 0; r < dk; r++) {
                for (s = 0; s < dl; s++) {
                        gm = gam[pq+dij*(r+dk*s)];
                        ib = s+dl*(r+dk*pq);
                        sx += bx[ib] * gm;
                        sy += by[ib] * gm;
                        sz += bz[ib] * gm;
                } } }
        }
        g[0] = sx;
        g[1] = sy;
        g[2] = sz;
}

typedef struct {
        int (*intor)();
        CINTOpt *cintopt;
        const double *q0;
        const double *q1;
        const double *dmc;
        double *dmj;
        double *dmk;
        int *ao_loc;
        int *atm;
        int *bas;
        double *env;
        double cutoff;
        double cj;
        double ck;
        int nk;
        int natm;
        int nbas;
        int nao;
        int with_j;
        int with_k;
} EJKEnvs;

// The plan of the orbit of shells (ish,jsh,ksh,lsh): which centres take an
// integral evaluation, and how their sums go to the atoms
typedef struct {
        int sh[4][4];   // the shells of the integral of each centre c: c first
        int rep[4];     // the centre whose integral stands for c's
        int pass[4];
        int at[4];
        int evaluate[4];
        int use_ti;     // translational invariance gives the centres on xatm
        int xatm;
        int neval;
        double w;
} EJKOrbit;

// Plans the orbit; returns the number of evaluations the s2kl driver makes
// (0: the orbit is skipped)
static int _orbit_plan(EJKOrbit *o, int ish, int jsh, int ksh, int lsh, EJKEnvs *e)
{
        const int n = e->nbas;
        const int *bas = e->bas;
        int sh[4][4] = {{ish, jsh, ksh, lsh}, {jsh, ish, ksh, lsh},
                        {ksh, lsh, ish, jsh}, {lsh, ksh, ish, jsh}};
        memcpy(o->sh, sh, sizeof(sh));
        int pair_eq = (ish == ksh && jsh == lsh);
        int *rep = o->rep;
        rep[0] = 0;
        rep[1] = (jsh == ish) ? 0 : 1;
        rep[2] = pair_eq ? 0 : 2;
        rep[3] = pair_eq ? rep[1] : ((lsh == ksh) ? 2 : 3);

        int *pass = o->pass;
        int npass = 0;
        int nrep = 0;
        int c, c1;
        for (c = 0; c < 4; c++) {
                pass[c] = 0;
                if (rep[c] == c) {
                        nrep++;
                        pass[c] = _screen(e->q0, e->q1, e->dmc, n, sh[c][0], sh[c][1],
                                          sh[c][2], sh[c][3], e->cutoff,
                                          e->with_j, e->with_k);
                        npass += pass[c];
                }
        }
        o->neval = 0;
        if (npass == 0) {
                return 0;
        }

        int *at = o->at;
        int ang[4];
        for (c = 0; c < 4; c++) {
                at[c] = bas[ATOM_OF+sh[0][c]*BAS_SLOTS];
                ang[c] = bas[ANG_OF+sh[0][c]*BAS_SLOTS];
        }
        // The atom whose centers translational invariance gives: the one with
        // the most representative orderings, then the lowest angular momentum
        // among them (measured 3% faster than the highest on caffeine/SVP).
        int xatm = -1;
        int xcnt = 0;
        int xang = 0;
        for (c = 0; c < 4; c++) {
                if (rep[c] != c) {
                        continue;
                }
                int cnt = 0;
                int lmin = ang[c];
                for (c1 = 0; c1 < 4; c1++) {
                        if (rep[c1] == c1 && at[c1] == at[c]) {
                                cnt++;
                                lmin = MIN(lmin, ang[c1]);
                        }
                }
                if (cnt > xcnt || (cnt == xcnt && lmin < xang)) {
                        xatm = at[c];
                        xcnt = cnt;
                        xang = lmin;
                }
        }
        o->xatm = xatm;
        o->use_ti = (nrep - xcnt < npass);
        for (c = 0; c < 4; c++) {
                o->evaluate[c] = (rep[c] == c) && (o->use_ti ? at[c] != xatm : pass[c]);
                o->neval += o->evaluate[c];
        }

        double w = -4;
        if (ish == jsh) w *= .5;
        if (ksh == lsh) w *= .5;
        if (pair_eq) w *= .5;
        o->w = w;
        return npass;
}

// Adds the orbit to de[natm,3], from the sums g[c] of its evaluated centres
static void _orbit_add(double *de, EJKOrbit *o, double g[4][3])
{
        const double w = o->w;
        double sx = 0;
        double sy = 0;
        double sz = 0;
        int c;
        for (c = 0; c < 4; c++) {
                int r = o->rep[c];
                if (o->use_ti ? o->at[c] == o->xatm : !o->pass[r]) {
                        continue;
                }
                double *pde = de + o->at[c] * 3;
                pde[0] += w * g[r][0];
                pde[1] += w * g[r][1];
                pde[2] += w * g[r][2];
                sx += g[r][0];
                sy += g[r][1];
                sz += g[r][2];
        }
        if (o->use_ti && o->neval > 0) {
                double *pde = de + o->xatm * 3;
                pde[0] -= w * sx;
                pde[1] -= w * sy;
                pde[2] -= w * sz;
        }
}

// The orbit's integrals one evaluation at a time
static void _orbit_eval(double *de, EJKOrbit *o, EJKEnvs *e, double *buf, double *gam,
                        double *cache)
{
        int *ao_loc = e->ao_loc;
        int *sh0 = o->sh[0];
        int di = ao_loc[sh0[0]+1] - ao_loc[sh0[0]];
        int dj = ao_loc[sh0[1]+1] - ao_loc[sh0[1]];
        int dk = ao_loc[sh0[2]+1] - ao_loc[sh0[2]];
        int dl = ao_loc[sh0[3]+1] - ao_loc[sh0[3]];
        _gamma(gam, e->dmj, e->dmk, e->nk, e->cj, e->ck, e->nao,
               ao_loc[sh0[0]], di, ao_loc[sh0[1]], dj, ao_loc[sh0[2]], dk, ao_loc[sh0[3]], dl);
        double g[4][3];
        int c;
        for (c = 0; c < 4; c++) {
                if (!o->evaluate[c]) {
                        continue;
                }
                if ((*e->intor)(buf, NULL, o->sh[c], e->atm, e->natm, e->bas, e->nbas,
                                e->env, e->cintopt, cache)) {
                        _contract(g[c], buf, gam, c, di, dj, dk, dl);
                } else {
                        g[c][0] = 0;
                        g[c][1] = 0;
                        g[c][2] = 0;
                }
        }
        _orbit_add(de, o, g);
}

/*
 * Several orbits at once: qcint's int2e_ip1_sph_batch (cint2e_batch.c) for the
 * orbits whose quartets int2e_ip1_sph_batch_ok takes. An orbit is queued by the
 * types (l, number of primitives) of its 4 shells and the set of centres it
 * evaluates, so that the orbits of a queue need the same integral classes; a
 * full queue (orbit_engine.h) is evaluated centre by centre for all of them,
 * each orbit's Gamma computed once. Weak: libcint and older qcint have none of
 * these functions.
 */
int int2e_ip1_sph();
int int2e_ip1_sph_batch(double *out, int nq, int *shls, int *atm, int natm, int *bas, int nbas,
                        double *env, CINTOpt *opt, double *cache) __attribute__((weak));
int int2e_ip1_sph_batch_ok(int *shls, int *bas, double *env) __attribute__((weak));
int int2e_sph_batch_lanes(void) __attribute__((weak));
size_t int2e_sph_batch_cache_size(void) __attribute__((weak));
// The same integrals contracted with ng arrays of each quartet inside qcint's kernels:
// out[(q*ng+n)*3+c] = sum_f int[q][c][f] gam[(q*ng+n)*nf+f]
int int2e_ip1_sph_batch_dot(double *out, int nq, int *shls, int *atm, int natm, int *bas,
                            int nbas, double *env, const double *gam, int ng, double *cache)
        __attribute__((weak));

#define EJK_MAX_LANES   OE_MAX_LANES
#define EJK_QCAP        1024    // hash table entries per thread, a power of 2

// The client of orbit_engine.h
typedef struct {
        EJKEnvs *e;
        int lanes;
        int nt;                 // shell types
        int *stype;
        int dot;                // the integrals contracted with Gamma inside qcint's kernels
        size_t di4;
        size_t cache_size;
        double norbit;
        double neval;
        double npass;
} EJKClient;

typedef struct {
        double *buf;
        double *gam;
        double *cache;
        double *bout;           // lanes * 3 * the largest block
        double *bcache;
        double *gams;           // lanes * the largest block
        double *gperm;          // the same, in the order of a centre (_dot)
        double norbit;
        double neval;
        double npass;
} EJKThread;

static void *_thread_init(void *ctx)
{
        EJKClient *cl = ctx;
        const size_t di4 = cl->di4;
        EJKThread *th = calloc(1, sizeof(EJKThread));
        th->buf = malloc(sizeof(double) * (di4 * 4 + cl->cache_size));
        th->gam = th->buf + di4 * 3;
        th->cache = th->gam + di4;
        if (cl->lanes) {
                th->bout = malloc(sizeof(double) * di4 * 3 * cl->lanes);
                th->bcache = malloc(sizeof(double) * int2e_sph_batch_cache_size());
                th->gams = malloc(sizeof(double) * di4 * cl->lanes);
                th->gperm = malloc(sizeof(double) * di4 * cl->lanes);
        }
        return th;
}

static void _thread_done(void *ctx, void *pth)
{
        EJKClient *cl = ctx;
        EJKThread *th = pth;
        cl->norbit += th->norbit;
        cl->neval += th->neval;
        cl->npass += th->npass;
        free(th->buf);
        if (cl->lanes) {
                free(th->bout);
                free(th->bcache);
                free(th->gams);
                free(th->gperm);
        }
        free(th);
}

static int _plan(void *ctx, void *pth, int ish, int jsh, int ksh, int lsh,
                 void *orbit, uint64_t *key)
{
        EJKClient *cl = ctx;
        EJKThread *th = pth;
        EJKEnvs *e = cl->e;
        EJKOrbit *o = orbit;
        int np = _orbit_plan(o, ish, jsh, ksh, lsh, e);
        th->norbit += np > 0;
        th->npass += np;
        if (np == 0 || o->neval == 0) {
                return 0;
        }
        th->neval += o->neval;
        *key = 0;
        if (cl->lanes && int2e_ip1_sph_batch_ok(o->sh[0], e->bas, e->env)) {
                const int *sh0 = o->sh[0];
                const uint64_t nt = cl->nt;
                int mask = o->evaluate[0] | o->evaluate[1] << 1 | o->evaluate[2] << 2 |
                        o->evaluate[3] << 3;
                *key = ((((cl->stype[sh0[0]] * nt + cl->stype[sh0[1]]) * nt +
                          cl->stype[sh0[2]]) * nt + cl->stype[sh0[3]]) << 4 | mask) + 1;
        }
        return 1;
}

static void _eval(void *ctx, void *pth, double *de, void *orbit)
{
        EJKClient *cl = ctx;
        EJKThread *th = pth;
        _orbit_eval(de, orbit, cl->e, th->buf, th->gam, th->cache);
}

static void _eval_batch(void *ctx, void *pth, double *de, void *orbits, int nq)
{
        EJKClient *cl = ctx;
        EJKThread *th = pth;
        EJKEnvs *e = cl->e;
        EJKOrbit *os = orbits;
        int q, c;
        int *ao_loc = e->ao_loc;
        int *sh0 = os[0].sh[0];
        int di = ao_loc[sh0[0]+1] - ao_loc[sh0[0]];
        int dj = ao_loc[sh0[1]+1] - ao_loc[sh0[1]];
        int dk = ao_loc[sh0[2]+1] - ao_loc[sh0[2]];
        int dl = ao_loc[sh0[3]+1] - ao_loc[sh0[3]];
        size_t nf = (size_t)di * dj * dk * dl;
        for (q = 0; q < nq; q++) {
                int *s = os[q].sh[0];
                _gamma(th->gams + nf * q, e->dmj, e->dmk, e->nk, e->cj, e->ck, e->nao,
                       ao_loc[s[0]], di, ao_loc[s[1]], dj, ao_loc[s[2]], dk, ao_loc[s[3]], dl);
        }
        double g[EJK_MAX_LANES][4][3];
        int shls[4 * EJK_MAX_LANES];
        for (c = 0; c < 4; c++) {
                if (!os[0].evaluate[c]) {
                        continue;
                }
                for (q = 0; q < nq; q++) {
                        memcpy(shls + 4 * q, os[q].sh[c], sizeof(int) * 4);
                }
                if (cl->dot) {
                        // Gamma in the order of the centre's integral (c = 0..3 are the
                        // first four orders of orbit_engine.h), contracted in qcint
                        const double *gc = th->gams;
                        if (c != OE_IJKL) {
                                for (q = 0; q < nq; q++) {
                                        CVHFoe_permute(th->gperm + nf * q, th->gams + nf * q, c,
                                                       di, dj, dk, dl);
                                }
                                gc = th->gperm;
                        }
                        int2e_ip1_sph_batch_dot(th->bout, nq, shls, e->atm, e->natm, e->bas,
                                                e->nbas, e->env, gc, 1, th->bcache);
                        for (q = 0; q < nq; q++) {
                                memcpy(g[q][c], th->bout + 3 * q, sizeof(double) * 3);
                        }
                        continue;
                }
                int2e_ip1_sph_batch(th->bout, nq, shls, e->atm, e->natm, e->bas, e->nbas,
                                    e->env, NULL, th->bcache);
                for (q = 0; q < nq; q++) {
                        _contract(g[q][c], th->bout + 3 * nf * q, th->gams + nf * q, c,
                                  di, dj, dk, dl);
                }
        }
        for (q = 0; q < nq; q++) {
                _orbit_add(de, os + q, g[q]);
        }
}

/*
 * de[natm,3] = dE2/dR (see the top of this file). intor is int2e_ip1 (sph or
 * cart) and cintopt its optimizer. vhfopt holds the q_cond of
 * grad.rhf._calc_q_cond (the int2e_ip1ip2 pair estimates, then those of
 * int2e), the dm_cond of CVHFnr_dm_cond1 for the densities the J/K matrices
 * would be built from, and the cutoff. dmj [nao,nao] is used when cj != 0,
 * dmk [nk,nao,nao] when ck != 0. with_j and with_k select the screening.
 * stats, if not NULL, receives the counts of orbits that pass the screening,
 * of integral evaluations, and of the evaluations of the s2kl driver.
 * int2e_batch: int2e_ip1_sph evaluates several orbits at once where qcint offers
 * that (the queues of orbit_engine.h), contracted with Gamma in qcint's kernels
 * where it offers that (int2e_ip1_sph_batch_dot); 2: the batched integrals
 * contracted here. CVHFgrad_jk_energy takes 1.
 */
void CVHFgrad_jk_energy_mode(double *de, int (*intor)(), CINTOpt *cintopt,
                             CVHFOpt *vhfopt, double *dmj, double *dmk, int nk,
                             double cj, double ck, int with_j, int with_k,
                             double *stats, int int2e_batch, int *ao_loc, int *atm, int natm,
                             int *bas, int nbas, double *env)
{
        const size_t nbas2 = (size_t)nbas * nbas;
        EJKEnvs envs = {intor, cintopt, vhfopt->q_cond, vhfopt->q_cond + nbas2,
                vhfopt->dm_cond, dmj, dmk, ao_loc, atm, bas, env,
                vhfopt->direct_scf_cutoff, cj, ck, nk, natm, nbas, ao_loc[nbas],
                with_j, with_k};
        if (cj == 0) {
                envs.dmj = NULL;
        }
        if (ck == 0) {
                envs.nk = 0;
        }
        const double cutoff = envs.cutoff;
        const double *q0 = envs.q0;
        const double *q1 = envs.q1;
        size_t i;
        int ish, jsh;

        // A pair (I,J) bounds every ordering of its quartets: q0[I,J],
        // q0[J,I] <= q, q1[I,J] <= q. An ordering passes only if
        // qijkl > cutoff and qijkl * 2*dm_cond > cutoff.
        double dmax = 0;
        for (i = 0; i < nbas2; i++) {
                dmax = MAX(dmax, envs.dmc[i]);
        }
        double *qpair = malloc(sizeof(double) * nbas2);
        for (ish = 0; ish < nbas; ish++) {
        for (jsh = 0; jsh <= ish; jsh++) {
                double q = MAX(q0[ish*nbas+jsh], q0[jsh*nbas+ish]);
                qpair[ish*nbas+jsh] = MAX(q, q1[ish*nbas+jsh]);
        } }

        int shls_slice[8] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
        size_t di = GTOmax_shell_dim(ao_loc, shls_slice, 1);
        size_t cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                              atm, natm, bas, nbas, env);
        EJKClient cl = {&envs, 0, 0, NULL, 0, di * di * di * di, cache_size, 0, 0, 0};

        // the queues of orbits for int2e_ip1_sph_batch, by shell type (l, nprim)
        if (int2e_batch && int2e_ip1_sph_batch != NULL && intor == int2e_ip1_sph &&
            env[PTR_RANGE_OMEGA] >= 0) {
                cl.lanes = MIN(int2e_sph_batch_lanes(), EJK_MAX_LANES);
                cl.stype = malloc(sizeof(int) * nbas * 3);
                int *tl = cl.stype + nbas;
                int *tp = tl + nbas;
                int s, t;
                for (s = 0; s < nbas; s++) {
                        for (t = 0; t < cl.nt; t++) {
                                if (tl[t] == bas[ANG_OF+s*BAS_SLOTS] &&
                                    tp[t] == bas[NPRIM_OF+s*BAS_SLOTS]) {
                                        break;
                                }
                        }
                        if (t == cl.nt) {
                                tl[t] = bas[ANG_OF+s*BAS_SLOTS];
                                tp[t] = bas[NPRIM_OF+s*BAS_SLOTS];
                                cl.nt++;
                        }
                        cl.stype[s] = t;
                }
        }
        // int2e_ip1_sph_batch_dot: qcint contracts the batched integrals with Gamma (int2e_batch 2
        // keeps the contraction here, for tests)
        cl.dot = cl.lanes && int2e_batch == 1 && int2e_ip1_sph_batch_dot != NULL;

        OEDriver d = {&cl, nbas, qpair, MIN(1., 2 * dmax), .5 * cutoff, (size_t)natm * 3,
                EJK_NCHUNK, EJK_CHUNK_BUF, cl.lanes, EJK_QCAP, sizeof(EJKOrbit),
                _thread_init, _thread_done, _plan, _eval, _eval_batch};
        CVHFoe_run(&d, de);
        if (cl.stype != NULL) {
                free(cl.stype);
        }
        if (stats != NULL) {
                stats[0] = cl.norbit;
                stats[1] = cl.neval;
                stats[2] = cl.npass;
        }
        free(qpair);
}

void CVHFgrad_jk_energy(double *de, int (*intor)(), CINTOpt *cintopt,
                        CVHFOpt *vhfopt, double *dmj, double *dmk, int nk,
                        double cj, double ck, int with_j, int with_k,
                        double *stats, int *ao_loc, int *atm, int natm,
                        int *bas, int nbas, double *env)
{
        CVHFgrad_jk_energy_mode(de, intor, cintopt, vhfopt, dmj, dmk, nk, cj, ck,
                                with_j, with_k, stats, 1, ao_loc, atm, natm, bas, nbas, env);
}
