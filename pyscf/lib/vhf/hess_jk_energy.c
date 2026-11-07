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
 * Two-electron part of the nuclear Hessian at fixed densities (the ej, ek of
 * hessian.rhf._partial_hess_2e), contracted inside the integral loop. No J/K
 * matrices are built.
 *
 * For E = 1/2 sum_pqrs (pq|rs) Gamma_pqrs, Gamma of the 8-fold symmetry of the
 * integrals, each s8 orbit of shells I >= J, K >= L, (IJ) >= (KL) is visited
 * once:
 *
 *     d2E/dR_A dR_B = sum_orbits deg/2 sum_{c on A, d on B} M_cd,
 *     M_cd[x,y] = sum_{pqrs in IJKL} (nabla_c,x nabla_d,y (pq|rs)) Gamma_pqrs
 *
 * with c, d the centres 0..3 (the functions of I, J, K, L), nabla_c acting on
 * the function of centre c, and deg the number of distinct permutations of
 * the orbit. The same integrals are contracted with two Gammas:
 *
 *     Gamma_J = Dj_pq Dj_rs                                  -> hj
 *     Gamma_K = kfac/4 sum_n (Dk_pr Dk_qs + Dk_ps Dk_qr)     -> hk
 *
 * M_cd of c != d is an int2e_ipvip1 (c, d of one pair) or int2e_ip1ip2 (of
 * different pairs) integral, M_cc an int2e_ipip1 integral, of the shells
 * permuted to put c (then d) first. M_dc = M_cd^T.
 *
 * Translational invariance, sum_d M_cd = 0 for every c, gives the blocks of
 * one atom X of the orbit from the others:
 *
 *     sum_{c on A, d on X} M_cd = -sum_{c on A} sum_{d not on X} M_cd
 *
 * so the orbit takes, for each atom A != X, M_cc of its centres and M_cd of
 * the pairs of its centres, and M_cd between the centres of two atoms other
 * than X. X is the atom with the most centres: centres 3 + 1 on two atoms
 * then take 1 evaluation (int2e_ipip1), 2 + 2 take 3, 2 + 1 + 1 take 3, four
 * atoms 6 (no translational invariance), one atom none. The cross-atom pairs
 * alone (the loops of hessian.rhf._partial_hess_2e) are 3, 4, 5 and 6. The
 * three integrals cost the same per quartet. The diagonal blocks
 * d2E/dR_A dR_A follow from translational invariance at the end.
 *
 * Screening: an orbit is computed if
 *     Q > cutoff and Q * D > cutoff,
 *     Q = max(qd[I,J] qd[K,L], qv[I,J] q[K,L], qv[K,L] q[I,J])
 * with qd the int2e_ip1ip2 pair bounds (CVHFnr_int2e_pp_q_cond, both orders
 * of the pair), qv those of int2e_ipvip1 (CVHFnr_int2e_pppp_q_cond), q those of
 * int2e, and D the largest of 2 dm_cond[I,J], 2 dm_cond[K,L] (with Gamma_J)
 * and dm_cond[I,K], dm_cond[I,L], dm_cond[J,K], dm_cond[J,L] (with Gamma_K),
 * dm_cond being the largest |D| of the shell pair over all densities. With
 * both Gammas, each test of CVHFip1ip2_prescreen and CVHFipvip1_prescreen
 * (hessian_screen.c) of the orbit's integrals passes only if this one does;
 * with one, the terms of the other (whose contributions are zero) are left
 * out. A computed orbit takes all its integrals. cutoff = 0 computes every
 * orbit whose Gammas are not zero.
 *
 * The orbits are summed in chunks of a fixed partition of the pair list, each
 * into its own buffer, and the buffers in chunk order: the result does not
 * depend on the thread count or the schedule.
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

// Upper limit of the chunk count: 256 left fewer part-filled batches than 1024 at the chunk ends
// (caffeine/def2-TZVP, 10 threads: 157.5 -> 149.8 s), and is still 4 chunks per thread on 64 threads
#define HJK_NCHUNK      256

// The integrals
#define HJK_IPIP1       0
#define HJK_IPVIP1      1
#define HJK_IP1IP2      2

// The orders of the shells (I,J,K,L) in an evaluation
#define O_IJKL          0
#define O_JIKL          1
#define O_KLIJ          2
#define O_LKIJ          3
#define O_IJLK          4
#define O_JILK          5
#define HJK_NORDER      6
static const int _order_sh[HJK_NORDER][4] = {
        {0, 1, 2, 3}, {1, 0, 2, 3}, {2, 3, 0, 1}, {3, 2, 0, 1}, {0, 1, 3, 2}, {1, 0, 3, 2}};

// The evaluations of an orbit: M_cc for c = 0..3, then M_01, M_23, M_02,
// M_03, M_12, M_13, each with the integral and the order of its shells
#define HJK_NEVAL       10
static const int _eval_intor[HJK_NEVAL] = {
        HJK_IPIP1, HJK_IPIP1, HJK_IPIP1, HJK_IPIP1, HJK_IPVIP1, HJK_IPVIP1,
        HJK_IP1IP2, HJK_IP1IP2, HJK_IP1IP2, HJK_IP1IP2};
static const int _eval_order[HJK_NEVAL] = {
        O_IJKL, O_JIKL, O_KLIJ, O_LKIJ, O_IJKL, O_KLIJ, O_IJKL, O_IJLK, O_JIKL, O_JILK};
// the evaluation of M_cd, c <= d
static const int _eval_of[4][4] = {{0, 4, 6, 7}, {4, 1, 8, 9}, {6, 8, 2, 5}, {7, 9, 5, 3}};

typedef struct {
        double q;
        int i;
        int j;
} HJKPair;

static int _cmp_pair(const void *a, const void *b)
{
        const HJKPair *pa = a;
        const HJKPair *pb = b;
        if (pa->q != pb->q) {
                return pa->q < pb->q ? -1 : 1;
        }
        if (pa->i != pb->i) {
                return pa->i - pb->i;
        }
        return pa->j - pb->j;
}

/*
 * The plan of an orbit by the pattern of its atoms, grp[c] being the first
 * centre on the atom of centre c: the evaluations (mask of HJK_NEVAL bits),
 * xgrp, the group translational invariance gives (-1 for none), and the
 * ordered centre pairs (c,d) whose M_cd go to the sum of their groups'
 * block: those of different groups (c < d) without xgrp, those of the
 * groups other than xgrp with it. code = grp[1] + 2 * (grp[2] + 3 * grp[3]).
 */
#define HJK_NCODE       24
typedef struct {
        int valid;
        int grp[4];
        int ngrp;
        int xgrp;
        int mask;
        int npair;
        int pev[16];            // the evaluation of M_cd
        int ptr[16];            // c > d: M_dc transposed
        int pab[16];            // the block: grp[c] * 4 + grp[d]
} HJKPlan;

static void _make_plans(HJKPlan *plans)
{
        int g1, g2, g3, c, d;
        for (g3 = 0; g3 < 4; g3++) {
        for (g2 = 0; g2 < 3; g2++) {
        for (g1 = 0; g1 < 2; g1++) {
                HJKPlan *pl = plans + g1 + 2 * (g2 + 3 * g3);
                int grp[4] = {0, g1, g2, g3};
                memset(pl, 0, sizeof(HJKPlan));
                // grp[c] must be the first centre of its group
                pl->valid = 1;
                for (c = 1; c < 4; c++) {
                        if (grp[grp[c]] != grp[c]) {
                                pl->valid = 0;
                        }
                }
                if (!pl->valid) {
                        continue;
                }
                memcpy(pl->grp, grp, sizeof(grp));
                int size[4] = {0};
                for (c = 0; c < 4; c++) {
                        size[grp[c]]++;
                }
                for (c = 0; c < 4; c++) {
                        pl->ngrp += (grp[c] == c);
                }
                pl->xgrp = -1;
                if (pl->ngrp == 1) {
                        continue;
                }
                if (pl->ngrp < 4) {
                        // the group with the most centres; of two pairs, the one without centre 0
                        int best = 0;
                        for (c = 0; c < 4; c++) {
                                if (grp[c] == c && size[c] >= size[best]) {
                                        best = c;
                                }
                        }
                        pl->xgrp = best;
                }
                const int x = pl->xgrp;
                for (c = 0; c < 4; c++) {
                for (d = 0; d < 4; d++) {
                        int need;
                        if (x < 0) {
                                need = c < d && grp[c] != grp[d];
                        } else {
                                need = grp[c] != x && grp[d] != x;
                        }
                        if (need) {
                                int n = pl->npair++;
                                pl->pev[n] = c <= d ? _eval_of[c][d] : _eval_of[d][c];
                                pl->ptr[n] = c > d;
                                pl->pab[n] = grp[c] * 4 + grp[d];
                                pl->mask |= 1 << pl->pev[n];
                        }
                } }
        } } }
}

static int _pattern_code(const int *at)
{
        int grp[4];
        int c, d;
        for (c = 0; c < 4; c++) {
                grp[c] = c;
                for (d = 0; d < c; d++) {
                        if (at[d] == at[c]) {
                                grp[c] = d;
                                break;
                        }
                }
        }
        return grp[1] + 2 * (grp[2] + 3 * grp[3]);
}

typedef struct {
        int (*intor[3])();
        CINTOpt *cintopt[3];
        const double *qd;
        const double *qv;
        const double *q;
        const double *dmc;
        double *dmj;
        double *dmk;
        int *ao_loc;
        int *atm;
        int *bas;
        double *env;
        HJKPlan *plans;
        double cutoff;
        double kfac4;
        int nk;
        int natm;
        int nbas;
        int nao;
        int with_j;
        int with_k;
        int nh;                 // Gammas: with_j + with_k
} HJKEnvs;

typedef struct {
        int sh[4];
        int at[4];
        int code;
        double w;
} HJKOrbit;

// Whether the orbit (ish,jsh,ksh,lsh) passes the screening (see the top)
static int _orbit_screen(HJKEnvs *e, int ish, int jsh, int ksh, int lsh)
{
        const int n = e->nbas;
        const double *qd = e->qd;
        const double *qv = e->qv;
        const double *q = e->q;
        const double *dmc = e->dmc;
        double qij = MAX(qd[ish*n+jsh], qd[jsh*n+ish]);
        double qkl = MAX(qd[ksh*n+lsh], qd[lsh*n+ksh]);
        double qm = qij * qkl;
        qm = MAX(qm, qv[ish*n+jsh] * q[ksh*n+lsh]);
        qm = MAX(qm, qv[ksh*n+lsh] * q[ish*n+jsh]);
        if (qm <= e->cutoff) {
                return 0;
        }
        double dm = 0;
        if (e->with_j) {
                dm = 2 * MAX(dmc[ish*n+jsh], dmc[ksh*n+lsh]);
        }
        if (e->with_k) {
                dm = MAX(dm, dmc[ish*n+ksh]);
                dm = MAX(dm, dmc[ish*n+lsh]);
                dm = MAX(dm, dmc[jsh*n+ksh]);
                dm = MAX(dm, dmc[jsh*n+lsh]);
        }
        return qm * dm > e->cutoff;
}

static void _orbit_init(HJKOrbit *o, HJKEnvs *e, int ish, int jsh, int ksh, int lsh)
{
        const int *bas = e->bas;
        o->sh[0] = ish;
        o->sh[1] = jsh;
        o->sh[2] = ksh;
        o->sh[3] = lsh;
        int c;
        for (c = 0; c < 4; c++) {
                o->at[c] = bas[ATOM_OF+o->sh[c]*BAS_SLOTS];
        }
        o->code = _pattern_code(o->at);
        double w = 4;
        if (ish == jsh) w *= .5;
        if (ksh == lsh) w *= .5;
        if (ish == ksh && jsh == lsh) w *= .5;
        o->w = w;
}

// gj[p+di*(q+dj*(r+dk*s))] = Gamma_J, gk the same for Gamma_K, of the AOs of the
// shells sh. The densities are symmetric; they are read along rows.
static void _gamma(double *gj, double *gk, HJKEnvs *e, const int *sh)
{
        const int *ao_loc = e->ao_loc;
        const size_t nao = e->nao;
        const int p0 = ao_loc[sh[0]];
        const int q0 = ao_loc[sh[1]];
        const int r0 = ao_loc[sh[2]];
        const int s0 = ao_loc[sh[3]];
        const int di = ao_loc[sh[0]+1] - p0;
        const int dj = ao_loc[sh[1]+1] - q0;
        const int dk = ao_loc[sh[2]+1] - r0;
        const int dl = ao_loc[sh[3]+1] - s0;
        const int dij = di * dj;
        const double kfac4 = e->kfac4;
        int p, q, r, s, n;
        double *pg;
        for (s = 0; s < dl; s++) {
        for (r = 0; r < dk; r++) {
                if (e->with_j) {
                        double *dmj = e->dmj;
                        double drs = dmj[(r0+r)*nao+s0+s];
                        pg = gj + (r + dk * s) * dij;
                        for (q = 0; q < dj; q++) {
                                double *dq = dmj + (q0+q)*nao + p0;
                                for (p = 0; p < di; p++) {
                                        pg[p+di*q] = drs * dq[p];
                                }
                        }
                }
                if (e->with_k) {
                        pg = gk + (r + dk * s) * dij;
                        for (p = 0; p < dij; p++) {
                                pg[p] = 0;
                        }
                        for (n = 0; n < e->nk; n++) {
                                double *d = e->dmk + n * nao * nao;
                                double *dr = d + (r0+r)*nao + p0;
                                double *ds = d + (s0+s)*nao + p0;
                                for (q = 0; q < dj; q++) {
                                        double dqs = kfac4 * d[(q0+q)*nao+s0+s];
                                        double dqr = kfac4 * d[(q0+q)*nao+r0+r];
                                        for (p = 0; p < di; p++) {
                                                pg[p+di*q] += dr[p] * dqs + ds[p] * dqr;
                                        }
                                }
                        }
                }
        } }
}

// out = g (in the layout of (I,J,K,L)) in the layout of the shell order o
static void _permute(double *out, const double *g, int o, int di, int dj, int dk, int dl)
{
        const int dij = di * dj;
        const int dkl = dk * dl;
        int p, q, r, s, pq, rs;
        switch (o) {
        case O_JIKL:
                for (rs = 0; rs < dkl; rs++) {
                        const double *pg = g + rs * dij;
                        double *po = out + rs * dij;
                        for (p = 0; p < di; p++) {
                        for (q = 0; q < dj; q++) {
                                po[q+dj*p] = pg[p+di*q];
                        } }
                }
                break;
        case O_KLIJ:
                for (pq = 0; pq < dij; pq++) {
                for (rs = 0; rs < dkl; rs++) {
                        out[rs+dkl*pq] = g[pq+dij*rs];
                } }
                break;
        case O_LKIJ:
                for (pq = 0; pq < dij; pq++) {
                for (r = 0; r < dk; r++) {
                for (s = 0; s < dl; s++) {
                        out[s+dl*(r+dk*pq)] = g[pq+dij*(r+dk*s)];
                } } }
                break;
        case O_IJLK:
                for (r = 0; r < dk; r++) {
                for (s = 0; s < dl; s++) {
                        memcpy(out + dij * (s + dl * r), g + dij * (r + dk * s),
                               sizeof(double) * dij);
                } }
                break;
        case O_JILK:
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

// m[n][x] = sum_i buf[x*nf+i] g[n*nf+i] for the 9 components x and the nh
// (1 or 2) Gammas g
static void _contract(double m[2][9], const double *buf, const double *g, int nh, size_t nf)
{
        int x;
        size_t i;
        if (nh == 2) {
                const double *g1 = g + nf;
                for (x = 0; x < 9; x++) {
                        const double *b = buf + x * nf;
                        double s0 = 0;
                        double s1 = 0;
#pragma omp simd reduction(+:s0, s1)
                        for (i = 0; i < nf; i++) {
                                s0 += b[i] * g[i];
                                s1 += b[i] * g1[i];
                        }
                        m[0][x] = s0;
                        m[1][x] = s1;
                }
        } else {
                for (x = 0; x < 9; x++) {
                        const double *b = buf + x * nf;
                        double s0 = 0;
#pragma omp simd reduction(+:s0)
                        for (i = 0; i < nf; i++) {
                                s0 += b[i] * g[i];
                        }
                        m[0][x] = s0;
                }
        }
}

/*
 * Adds the orbit to h, the blocks [A*(A-1)/2+B, 3, 3] of the atoms A > B,
 * from its evaluations m[HJK_NEVAL][9] of one Gamma
 */
static void _orbit_add(double *h, HJKOrbit *o, HJKPlan *pl, double (*m)[9])
{
        // g[a*4+b]: sum_{c on group a, d on group b} M_cd (row x on c, column y on d)
        double g[16][9];
        int used[16] = {0};
        int n, a, b, x, y;
        for (n = 0; n < pl->npair; n++) {
                double *v = m[pl->pev[n]];
                double *pg = g[pl->pab[n]];
                if (!used[pl->pab[n]]) {
                        used[pl->pab[n]] = 1;
                        if (pl->ptr[n]) {
                                for (x = 0; x < 3; x++) {
                                for (y = 0; y < 3; y++) {
                                        pg[x*3+y] = v[y*3+x];
                                } }
                        } else {
                                memcpy(pg, v, sizeof(double) * 9);
                        }
                } else if (pl->ptr[n]) {
                        for (x = 0; x < 3; x++) {
                        for (y = 0; y < 3; y++) {
                                pg[x*3+y] += v[y*3+x];
                        } }
                } else {
                        for (x = 0; x < 9; x++) {
                                pg[x] += v[x];
                        }
                }
        }
        const int *grp = pl->grp;
        const int xg = pl->xgrp;
        if (xg >= 0) {
                // sum_{c on a, d on xg} M_cd = -sum_{c on a, d not on xg} M_cd
                for (a = 0; a < 4; a++) {
                        if (grp[a] != a || a == xg) {
                                continue;
                        }
                        double *pax = g[a*4+xg];
                        double *pxa = g[xg*4+a];
                        for (x = 0; x < 9; x++) {
                                pax[x] = 0;
                        }
                        for (b = 0; b < 4; b++) {
                                if (grp[b] == b && b != xg) {
                                        double *pab = g[a*4+b];
                                        for (x = 0; x < 9; x++) {
                                                pax[x] -= pab[x];
                                        }
                                }
                        }
                        for (x = 0; x < 3; x++) {
                        for (y = 0; y < 3; y++) {
                                pxa[y*3+x] = pax[x*3+y];
                        } }
                }
        }
        const double w = o->w;
        for (a = 0; a < 4; a++) {
                if (grp[a] != a) {
                        continue;
                }
                for (b = a + 1; b < 4; b++) {
                        if (grp[b] != b) {
                                continue;
                        }
                        const int ia = o->at[a];
                        const int ib = o->at[b];
                        const double *v = g[a*4+b];
                        if (ia > ib) {
                                double *ph = h + ((size_t)ia * (ia - 1) / 2 + ib) * 9;
                                for (x = 0; x < 9; x++) {
                                        ph[x] += w * v[x];
                                }
                        } else {
                                double *ph = h + ((size_t)ib * (ib - 1) / 2 + ia) * 9;
                                for (x = 0; x < 3; x++) {
                                for (y = 0; y < 3; y++) {
                                        ph[y*3+x] += w * v[x*3+y];
                                } }
                        }
                }
        }
}

static void _orbit_dims(int *dims, HJKOrbit *o, int *ao_loc)
{
        int c;
        for (c = 0; c < 4; c++) {
                dims[c] = ao_loc[o->sh[c]+1] - ao_loc[o->sh[c]];
        }
}

// The orbit's integrals one evaluation at a time; h holds the chunk's blocks
// of each Gamma, ntri blocks apart
static void _orbit_eval(double *h, size_t ntri, HJKOrbit *o, HJKEnvs *e,
                        double *buf, double *gam, double *gperm, double *cache)
{
        HJKPlan *pl = e->plans + o->code;
        int dims[4];
        _orbit_dims(dims, o, e->ao_loc);
        const size_t nf = (size_t)dims[0] * dims[1] * dims[2] * dims[3];
        const int nh = e->nh;
        double *gj = gam;
        double *gk = gam + nf * e->with_j;
        _gamma(gj, gk, e, o->sh);
        double m[2][HJK_NEVAL][9];
        double mt[2][9];
        int shls[4];
        int ord, ev, c, n;
        for (ord = 0; ord < HJK_NORDER; ord++) {
                int any = 0;
                for (ev = 0; ev < HJK_NEVAL; ev++) {
                        any |= (pl->mask >> ev & 1) && _eval_order[ev] == ord;
                }
                if (!any) {
                        continue;
                }
                double *g = gam;
                if (ord != O_IJKL) {
                        for (n = 0; n < nh; n++) {
                                _permute(gperm + n * nf, gam + n * nf, ord,
                                         dims[0], dims[1], dims[2], dims[3]);
                        }
                        g = gperm;
                }
                for (c = 0; c < 4; c++) {
                        shls[c] = o->sh[_order_sh[ord][c]];
                }
                for (ev = 0; ev < HJK_NEVAL; ev++) {
                        if (!(pl->mask >> ev & 1) || _eval_order[ev] != ord) {
                                continue;
                        }
                        int k = _eval_intor[ev];
                        if ((*e->intor[k])(buf, NULL, shls, e->atm, e->natm, e->bas, e->nbas,
                                           e->env, e->cintopt[k], cache)) {
                                _contract(mt, buf, g, nh, nf);
                        } else {
                                memset(mt, 0, sizeof(mt));
                        }
                        for (n = 0; n < nh; n++) {
                                memcpy(m[n][ev], mt[n], sizeof(double) * 9);
                        }
                }
        }
        for (n = 0; n < nh; n++) {
                _orbit_add(h + ntri * 9 * n, o, pl, m[n]);
        }
}

/*
 * Several orbits at once: qcint's int2e_ipip1_sph_batch, int2e_ipvip1_sph_batch
 * and int2e_ip1ip2_sph_batch (cint2e_batch.c) for the orbits whose quartets
 * int2e_ipip1_sph_batch_ok takes. An orbit is queued by the types (l, number
 * of primitives) of its 4 shells and the pattern of its atoms, so that the
 * orbits of a queue need the same integral classes; a queue holding
 * int2e_sph_batch_lanes() orbits is evaluated one evaluation of the plan at a
 * time for all of them. The queues live in a hash table per thread, flushed
 * in table order at the end of each chunk (and when the table fills), so the
 * sums do not depend on the thread count. A queue of fewer than half the
 * lanes goes orbit by orbit. Weak: libcint and older qcint have none of these
 * functions.
 */
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
int int2e_sph_batch_lanes(void) __attribute__((weak));
size_t int2e_hess_sph_batch_cache_size(void) __attribute__((weak));
// The same integrals contracted with ng arrays of each quartet inside qcint's kernels:
// out[(q*ng+n)*9+c] = sum_f int[q][c][f] gam[(q*ng+n)*nf+f]
int int2e_ipip1_sph_batch_dot(double *out, int nq, int *shls, int *atm, int natm, int *bas,
                              int nbas, double *env, const double *gam, int ng, double *cache)
        __attribute__((weak));
int int2e_ipvip1_sph_batch_dot(double *out, int nq, int *shls, int *atm, int natm, int *bas,
                               int nbas, double *env, const double *gam, int ng, double *cache)
        __attribute__((weak));
int int2e_ip1ip2_sph_batch_dot(double *out, int nq, int *shls, int *atm, int natm, int *bas,
                               int nbas, double *env, const double *gam, int ng, double *cache)
        __attribute__((weak));

#define HJK_MAX_LANES   8
#define HJK_QCAP        4096    // hash table entries per thread, a power of 2

typedef struct {
        uint64_t key;           // 0: empty
        int n;
        HJKOrbit o[HJK_MAX_LANES];
} HJKQueue;

typedef struct {
        HJKQueue *q;
        int nused;
        int lanes;
        int nt;                 // shell types
        int *stype;
        double *bout;           // lanes * 9 * the largest block
        double *bcache;
        double *gams;           // lanes * nh * the largest block
        double *gperm;          // the same, in the order of an evaluation
        double nbatched;        // orbits through the batched integrals
        int dot;                // the integrals contracted with the Gammas inside qcint's kernels
} HJKBatch;

static void _flush_orbits(double *h, size_t ntri, HJKQueue *Q, HJKBatch *bt, HJKEnvs *e,
                          double *buf, double *gam, double *gperm, double *cache)
{
        int q, ev, ord, c, n;
        if (2 * Q->n < bt->lanes) {
                for (q = 0; q < Q->n; q++) {
                        _orbit_eval(h, ntri, Q->o + q, e, buf, gam, gperm, cache);
                }
                Q->n = 0;
                return;
        }
        int (*fbatch[3])() = {int2e_ipip1_sph_batch, int2e_ipvip1_sph_batch,
                              int2e_ip1ip2_sph_batch};
        int (*fdot[3])() = {int2e_ipip1_sph_batch_dot, int2e_ipvip1_sph_batch_dot,
                            int2e_ip1ip2_sph_batch_dot};
        HJKPlan *pl = e->plans + Q->o[0].code;
        bt->nbatched += Q->n;
        int dims[4];
        _orbit_dims(dims, Q->o, e->ao_loc);
        const size_t nf = (size_t)dims[0] * dims[1] * dims[2] * dims[3];
        const int nh = e->nh;
        const size_t ng = nf * nh;
        for (q = 0; q < Q->n; q++) {
                double *g = bt->gams + ng * q;
                _gamma(g, g + nf * e->with_j, e, Q->o[q].sh);
        }
        double m[HJK_MAX_LANES][2][HJK_NEVAL][9];
        double mt[2][9];
        int shls[4 * HJK_MAX_LANES];
        for (ord = 0; ord < HJK_NORDER; ord++) {
                int any = 0;
                for (ev = 0; ev < HJK_NEVAL; ev++) {
                        any |= (pl->mask >> ev & 1) && _eval_order[ev] == ord;
                }
                if (!any) {
                        continue;
                }
                double *g = bt->gams;
                if (ord != O_IJKL) {
                        for (q = 0; q < Q->n * nh; q++) {
                                _permute(bt->gperm + nf * q, bt->gams + nf * q, ord,
                                         dims[0], dims[1], dims[2], dims[3]);
                        }
                        g = bt->gperm;
                }
                for (q = 0; q < Q->n; q++) {
                        for (c = 0; c < 4; c++) {
                                shls[4*q+c] = Q->o[q].sh[_order_sh[ord][c]];
                        }
                }
                for (ev = 0; ev < HJK_NEVAL; ev++) {
                        if (!(pl->mask >> ev & 1) || _eval_order[ev] != ord) {
                                continue;
                        }
                        int k = _eval_intor[ev];
                        if (bt->dot) {
                                (*fdot[k])(bt->bout, Q->n, shls, e->atm, e->natm, e->bas,
                                           e->nbas, e->env, g, nh, bt->bcache);
                                for (q = 0; q < Q->n; q++) {
                                        for (n = 0; n < nh; n++) {
                                                memcpy(m[q][n][ev], bt->bout + (q * nh + n) * 9,
                                                       sizeof(double) * 9);
                                        }
                                }
                                continue;
                        }
                        (*fbatch[k])(bt->bout, Q->n, shls, e->atm, e->natm,
                                     e->bas, e->nbas, e->env, NULL, bt->bcache);
                        for (q = 0; q < Q->n; q++) {
                                _contract(mt, bt->bout + 9 * nf * q, g + ng * q, nh, nf);
                                for (n = 0; n < nh; n++) {
                                        memcpy(m[q][n][ev], mt[n], sizeof(double) * 9);
                                }
                        }
                }
        }
        for (q = 0; q < Q->n; q++) {
                for (n = 0; n < nh; n++) {
                        _orbit_add(h + ntri * 9 * n, Q->o + q, pl, m[q][n]);
                }
        }
        Q->n = 0;
}

static void _flush_all(double *h, size_t ntri, HJKBatch *bt, HJKEnvs *e,
                       double *buf, double *gam, double *gperm, double *cache)
{
        int x;
        for (x = 0; x < HJK_QCAP; x++) {
                HJKQueue *Q = bt->q + x;
                if (Q->key != 0 && Q->n > 0) {
                        _flush_orbits(h, ntri, Q, bt, e, buf, gam, gperm, cache);
                }
                Q->key = 0;
        }
        bt->nused = 0;
}

// Queues the orbit o, evaluating its queue when full
static void _queue_orbit(double *h, size_t ntri, HJKOrbit *o, HJKBatch *bt, HJKEnvs *e,
                         double *buf, double *gam, double *gperm, double *cache)
{
        const int *sh = o->sh;
        const uint64_t nt = bt->nt;
        uint64_t key = ((((bt->stype[sh[0]] * nt + bt->stype[sh[1]]) * nt +
                          bt->stype[sh[2]]) * nt + bt->stype[sh[3]]) * HJK_NCODE + o->code) + 1;
        if (bt->nused * 4 >= HJK_QCAP * 3) {
                _flush_all(h, ntri, bt, e, buf, gam, gperm, cache);
        }
        size_t x = (size_t)((key * 0x9E3779B97F4A7C15ull) >> 52) & (HJK_QCAP - 1);
        while (bt->q[x].key != 0 && bt->q[x].key != key) {
                x = (x + 1) & (HJK_QCAP - 1);
        }
        HJKQueue *Q = bt->q + x;
        if (Q->key == 0) {
                Q->key = key;
                Q->n = 0;
                bt->nused++;
        }
        Q->o[Q->n] = *o;
        if (++Q->n == bt->lanes) {
                _flush_orbits(h, ntri, Q, bt, e, buf, gam, gperm, cache);
        }
}

/*
 * hj, hk [natm,natm,3,3]: the second derivatives of the energies of Gamma_J
 * and Gamma_K (see the top of this file); hj NULL for no Gamma_J, hk NULL or
 * nk = 0 for no Gamma_K. intors: int2e_ipip1, int2e_ipvip1, int2e_ip1ip2 (sph
 * or cart), cintopts their optimizers. q_cond [3,nbas,nbas]: the pair bounds
 * qd, qv, q; dm_cond [nbas,nbas]; cutoff: see the top. dmj [nao,nao], dmk
 * [nk,nao,nao] symmetric. max_buf: the doubles the chunk buffers may take.
 * stats, if not NULL, receives the counts of orbits computed, of their
 * integral evaluations, of their cross-atom centre pairs, and of the orbits
 * through the batched integrals.
 * int2e_batch: several orbits at once where qcint offers that (HJKQueue), their
 * integrals contracted with the Gammas in qcint's kernels where it offers that
 * (int2e_*_sph_batch_dot); 2: the batched integrals contracted here.
 */
void CVHFhess_jk_energy(double *hj, double *hk, int (**intors)(), CINTOpt **cintopts,
                        double *q_cond, double *dm_cond, double cutoff,
                        double *dmj, double *dmk, int nk, double kfac,
                        size_t max_buf, double *stats, int int2e_batch,
                        int *ao_loc, int *atm, int natm, int *bas, int nbas, double *env)
{
        const size_t nbas2 = (size_t)nbas * nbas;
        HJKPlan plans[HJK_NCODE];
        _make_plans(plans);
        HJKEnvs envs = {{intors[0], intors[1], intors[2]},
                {cintopts[0], cintopts[1], cintopts[2]},
                q_cond, q_cond + nbas2, q_cond + nbas2 * 2, dm_cond, dmj, dmk,
                ao_loc, atm, bas, env, plans, cutoff, kfac * .25, nk, natm, nbas,
                ao_loc[nbas], hj != NULL, hk != NULL && nk > 0, 0};
        envs.nh = envs.with_j + envs.with_k;
        const size_t ntri = (size_t)natm * (natm - 1) / 2;
        const size_t nhess = (size_t)natm * natm * 9;
        if (hj != NULL) {
                NPdset0(hj, nhess);
        }
        if (hk != NULL) {
                NPdset0(hk, nhess);
        }
        if (envs.nh == 0 || natm < 2) {
                if (stats != NULL) {
                        stats[0] = stats[1] = stats[2] = stats[3] = 0;
                }
                return;
        }
        const double *qd = envs.qd;
        const double *qv = envs.qv;
        const double *q1 = envs.q;
        size_t i;
        int ish, jsh;

        // A pair (I,J) bounds every term of Q: qd[I,J], qd[J,I], qv[I,J], q[I,J] <= q.
        // An orbit passes only if Q > cutoff and Q * D > cutoff, D <= 2 * max(dm_cond).
        double dmax = 0;
        for (i = 0; i < nbas2; i++) {
                dmax = MAX(dmax, dm_cond[i]);
        }
        const double fac = MIN(1., 2 * dmax);
        size_t npair = nbas2 / 2 + nbas;
        HJKPair *pairs = malloc(sizeof(HJKPair) * npair);
        double qmax = 0;
        size_t nsig = 0;
        for (ish = 0; ish < nbas; ish++) {
        for (jsh = 0; jsh <= ish; jsh++) {
                double q = MAX(qd[ish*nbas+jsh], qd[jsh*nbas+ish]);
                q = MAX(q, qv[ish*nbas+jsh]);
                q = MAX(q, q1[ish*nbas+jsh]);
                qmax = MAX(qmax, q);
                pairs[nsig].q = q;
                pairs[nsig].i = ish;
                pairs[nsig].j = jsh;
                nsig++;
        } }
        npair = nsig;
        nsig = 0;
        for (i = 0; i < npair; i++) {
                if (pairs[i].q * qmax * fac > cutoff) {
                        pairs[nsig] = pairs[i];
                        nsig++;
                }
        }
        // Ascending q: for pair t the partners u <= t are visited from t
        // down, and the bound decreases, so the loop stops at the first that
        // fails it.
        qsort(pairs, nsig, sizeof(HJKPair), _cmp_pair);

        // Chunks of equal (t+1) sums, taken heaviest first
        const size_t chunk_size = ntri * 9 * envs.nh;
        int nchunk = MIN(HJK_NCHUNK, max_buf / chunk_size);
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
        double *h_chunks = calloc((size_t)nchunk * chunk_size, sizeof(double));

        int shls_slice[8] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
        size_t di = GTOmax_shell_dim(ao_loc, shls_slice, 1);
        size_t di4 = di * di * di * di;
        size_t cache_size = 0;
        int k;
        for (k = 0; k < 3; k++) {
                cache_size = MAX(cache_size, GTOmax_cache_size(intors[k], shls_slice, 4,
                                                               atm, natm, bas, nbas, env));
        }
        double norbit = 0;
        double neval = 0;
        double ncross = 0;
        double nbatched = 0;

        // the queues of orbits for the batched integrals, by shell type (l, nprim)
        int lanes = 0;
        int nt = 0;
        int *stype = NULL;
        if (int2e_batch && int2e_ipip1_sph_batch != NULL && int2e_ipvip1_sph_batch != NULL &&
            int2e_ip1ip2_sph_batch != NULL && int2e_ipip1_sph_batch_ok != NULL &&
            int2e_sph_batch_lanes != NULL && int2e_hess_sph_batch_cache_size != NULL &&
            intors[0] == int2e_ipip1_sph && intors[1] == int2e_ipvip1_sph &&
            intors[2] == int2e_ip1ip2_sph && env[PTR_RANGE_OMEGA] >= 0) {
                lanes = MIN(int2e_sph_batch_lanes(), HJK_MAX_LANES);
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
        // int2e_*_sph_batch_dot: qcint contracts the batched integrals with the Gammas (int2e_batch 2
        // keeps the contraction here, for tests)
        const int use_dot = lanes && int2e_batch == 1 && int2e_ipip1_sph_batch_dot != NULL &&
                int2e_ipvip1_sph_batch_dot != NULL && int2e_ip1ip2_sph_batch_dot != NULL;
        // cross-atom centre pairs of each pattern
        int ncross_of[HJK_NCODE];
        for (k = 0; k < HJK_NCODE; k++) {
                int a, b;
                ncross_of[k] = 0;
                for (a = 0; a < 4; a++) {
                for (b = a + 1; b < 4; b++) {
                        ncross_of[k] += plans[k].valid && plans[k].grp[a] != plans[k].grp[b];
                } }
        }

#pragma omp parallel reduction(+:norbit, neval, ncross, nbatched)
{
        const int nh = envs.nh;
        double *buf = malloc(sizeof(double) * (di4 * (9 + 2 * nh) + cache_size));
        double *gam = buf + di4 * 9;
        double *gperm = gam + di4 * nh;
        double *cache = gperm + di4 * nh;
        HJKBatch bt = {NULL, 0, lanes, nt, stype, NULL, NULL, NULL, NULL, 0, use_dot};
        if (lanes) {
                bt.q = calloc(HJK_QCAP, sizeof(HJKQueue));
                bt.bout = malloc(sizeof(double) * di4 * 9 * lanes);
                bt.bcache = malloc(sizeof(double) * int2e_hess_sph_batch_cache_size());
                bt.gams = malloc(sizeof(double) * di4 * nh * lanes);
                bt.gperm = malloc(sizeof(double) * di4 * nh * lanes);
        }
        HJKOrbit o;
        size_t t, u;
        int ic;
#pragma omp for schedule(dynamic, 1)
        for (ic = 0; ic < nchunk; ic++) {
                int c = nchunk - 1 - ic;
                double *hc = h_chunks + (size_t)c * chunk_size;
                for (t = chunk_loc[c]; t < chunk_loc[c+1]; t++) {
                        double qt = pairs[t].q * fac;
                        int ish = pairs[t].i;
                        int jsh = pairs[t].j;
                        for (u = t + 1; u-- > 0;) {
                                if (qt * pairs[u].q <= cutoff) {
                                        break;
                                }
                                int ksh = pairs[u].i;
                                int lsh = pairs[u].j;
                                if (!_orbit_screen(&envs, ish, jsh, ksh, lsh)) {
                                        continue;
                                }
                                _orbit_init(&o, &envs, ish, jsh, ksh, lsh);
                                HJKPlan *pl = plans + o.code;
                                if (pl->mask == 0) {
                                        continue;
                                }
                                norbit += 1;
                                neval += __builtin_popcount(pl->mask);
                                ncross += ncross_of[o.code];
                                if (lanes && int2e_ipip1_sph_batch_ok(o.sh, bas, env)) {
                                        _queue_orbit(hc, ntri, &o, &bt, &envs, buf, gam, gperm, cache);
                                } else {
                                        _orbit_eval(hc, ntri, &o, &envs, buf, gam, gperm, cache);
                                }
                        }
                }
                if (lanes) {
                        _flush_all(hc, ntri, &bt, &envs, buf, gam, gperm, cache);
                }
        }
        free(buf);
        if (lanes) {
                free(bt.q);
                free(bt.bout);
                free(bt.bcache);
                free(bt.gams);
                free(bt.gperm);
        }
        nbatched += bt.nbatched;
}
        if (stype != NULL) {
                free(stype);
        }
        // the blocks A > B in chunk order, then B < A and the diagonal blocks
        double *hs[2] = {hj, hk};
        int n = 0;
        for (k = 0; k < 2; k++) {
                double *h = hs[k];
                if (h == NULL || (k == 1 && !envs.with_k)) {
                        continue;
                }
                double *tri = calloc(ntri * 9, sizeof(double));
                for (c = 0; c < nchunk; c++) {
                        double *hc = h_chunks + (size_t)c * chunk_size + ntri * 9 * n;
                        for (i = 0; i < ntri * 9; i++) {
                                tri[i] += hc[i];
                        }
                }
                int ia, ib, x, y;
                for (ia = 0; ia < natm; ia++) {
                for (ib = 0; ib < ia; ib++) {
                        double *v = tri + ((size_t)ia * (ia - 1) / 2 + ib) * 9;
                        double *hab = h + ((size_t)ia * natm + ib) * 9;
                        double *hba = h + ((size_t)ib * natm + ia) * 9;
                        for (x = 0; x < 3; x++) {
                        for (y = 0; y < 3; y++) {
                                hab[x*3+y] = v[x*3+y];
                                hba[y*3+x] = v[x*3+y];
                        } }
                } }
                for (ia = 0; ia < natm; ia++) {
                        double *haa = h + ((size_t)ia * natm + ia) * 9;
                        for (ib = 0; ib < natm; ib++) {
                                if (ib == ia) {
                                        continue;
                                }
                                double *hab = h + ((size_t)ia * natm + ib) * 9;
                                for (x = 0; x < 9; x++) {
                                        haa[x] -= hab[x];
                                }
                        }
                }
                free(tri);
                n++;
        }
        if (stats != NULL) {
                stats[0] = norbit;
                stats[1] = neval;
                stats[2] = ncross;
                stats[3] = nbatched;
        }
        free(h_chunks);
        free(chunk_loc);
        free(pairs);
}
