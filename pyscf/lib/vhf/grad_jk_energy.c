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
 * The orbits are summed in chunks of a fixed partition of the pair list, each
 * into its own buffer, and the buffers in chunk order: the result does not
 * depend on the thread count or the schedule.
 */

#include <stdlib.h>
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

// Upper limits of the chunk count: chunks in all, and chunk buffer doubles
#define EJK_NCHUNK      4096
#define EJK_CHUNK_BUF   4000000

typedef struct {
        double q;
        int i;
        int j;
} EJKPair;

static int _cmp_pair(const void *a, const void *b)
{
        const EJKPair *pa = a;
        const EJKPair *pb = b;
        if (pa->q != pb->q) {
                return pa->q < pb->q ? -1 : 1;
        }
        if (pa->i != pb->i) {
                return pa->i - pb->i;
        }
        return pa->j - pb->j;
}

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

// Adds the orbit of shells (ish,jsh,ksh,lsh) to de[natm,3]. Returns the
// number of integral evaluations; *npass_out is the number the s2kl driver
// makes.
static int _orbit(double *de, int ish, int jsh, int ksh, int lsh, EJKEnvs *e,
                  double *buf, double *gam, double *cache, int *npass_out)
{
        const int n = e->nbas;
        const int *bas = e->bas;
        // The shells of the integral of each center c: c first
        int sh[4][4] = {{ish, jsh, ksh, lsh}, {jsh, ish, ksh, lsh},
                        {ksh, lsh, ish, jsh}, {lsh, ksh, ish, jsh}};
        int pair_eq = (ish == ksh && jsh == lsh);
        int rep[4];
        rep[0] = 0;
        rep[1] = (jsh == ish) ? 0 : 1;
        rep[2] = pair_eq ? 0 : 2;
        rep[3] = pair_eq ? rep[1] : ((lsh == ksh) ? 2 : 3);

        int pass[4] = {0, 0, 0, 0};
        int npass = 0;
        int nrep = 0;
        int c, c1;
        for (c = 0; c < 4; c++) {
                if (rep[c] == c) {
                        nrep++;
                        pass[c] = _screen(e->q0, e->q1, e->dmc, n, sh[c][0], sh[c][1],
                                          sh[c][2], sh[c][3], e->cutoff,
                                          e->with_j, e->with_k);
                        npass += pass[c];
                }
        }
        *npass_out = npass;
        if (npass == 0) {
                return 0;
        }

        int at[4];
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
        int use_ti = (nrep - xcnt < npass);
        int evaluate[4];
        int neval = 0;
        for (c = 0; c < 4; c++) {
                evaluate[c] = (rep[c] == c) && (use_ti ? at[c] != xatm : pass[c]);
                neval += evaluate[c];
        }

        double w = -4;
        if (ish == jsh) w *= .5;
        if (ksh == lsh) w *= .5;
        if (pair_eq) w *= .5;

        double g[4][3];
        if (neval > 0) {
                int *ao_loc = e->ao_loc;
                int di = ao_loc[ish+1] - ao_loc[ish];
                int dj = ao_loc[jsh+1] - ao_loc[jsh];
                int dk = ao_loc[ksh+1] - ao_loc[ksh];
                int dl = ao_loc[lsh+1] - ao_loc[lsh];
                _gamma(gam, e->dmj, e->dmk, e->nk, e->cj, e->ck, e->nao,
                       ao_loc[ish], di, ao_loc[jsh], dj, ao_loc[ksh], dk, ao_loc[lsh], dl);
                for (c = 0; c < 4; c++) {
                        if (!evaluate[c]) {
                                continue;
                        }
                        if ((*e->intor)(buf, NULL, sh[c], e->atm, e->natm, bas, n,
                                        e->env, e->cintopt, cache)) {
                                _contract(g[c], buf, gam, c, di, dj, dk, dl);
                        } else {
                                g[c][0] = 0;
                                g[c][1] = 0;
                                g[c][2] = 0;
                        }
                }
        }

        double sx = 0;
        double sy = 0;
        double sz = 0;
        for (c = 0; c < 4; c++) {
                int r = rep[c];
                if (use_ti ? at[c] == xatm : !pass[r]) {
                        continue;
                }
                double *pde = de + at[c] * 3;
                pde[0] += w * g[r][0];
                pde[1] += w * g[r][1];
                pde[2] += w * g[r][2];
                sx += g[r][0];
                sy += g[r][1];
                sz += g[r][2];
        }
        if (use_ti && neval > 0) {
                double *pde = de + xatm * 3;
                pde[0] -= w * sx;
                pde[1] -= w * sy;
                pde[2] -= w * sz;
        }
        return neval;
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
 */
void CVHFgrad_jk_energy(double *de, int (*intor)(), CINTOpt *cintopt,
                        CVHFOpt *vhfopt, double *dmj, double *dmk, int nk,
                        double cj, double ck, int with_j, int with_k,
                        double *stats, int *ao_loc, int *atm, int natm,
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
        const double fac = MIN(1., 2 * dmax);
        const double thr = .5 * cutoff;
        size_t npair = nbas2 / 2 + nbas;
        EJKPair *pairs = malloc(sizeof(EJKPair) * npair);
        double qmax = 0;
        size_t nsig = 0;
        for (ish = 0; ish < nbas; ish++) {
        for (jsh = 0; jsh <= ish; jsh++) {
                double q = MAX(q0[ish*nbas+jsh], q0[jsh*nbas+ish]);
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
                if (pairs[i].q * qmax * fac > thr) {
                        pairs[nsig] = pairs[i];
                        nsig++;
                }
        }
        // Ascending q: for pair t the partners u <= t are visited from t
        // down, and the bound decreases, so the loop stops at the first that
        // fails it.
        qsort(pairs, nsig, sizeof(EJKPair), _cmp_pair);

        // Chunks of equal (t+1) sums, taken heaviest first
        int nchunk = MIN(EJK_NCHUNK, EJK_CHUNK_BUF / (natm * 3));
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
        double *de_chunks = calloc((size_t)nchunk * natm * 3, sizeof(double));

        int shls_slice[8] = {0, nbas, 0, nbas, 0, nbas, 0, nbas};
        size_t di = GTOmax_shell_dim(ao_loc, shls_slice, 1);
        size_t di4 = di * di * di * di;
        size_t cache_size = GTOmax_cache_size(intor, shls_slice, 4,
                                              atm, natm, bas, nbas, env);
        double norbit = 0;
        double neval = 0;
        double npass = 0;

#pragma omp parallel reduction(+:norbit, neval, npass)
{
        double *buf = malloc(sizeof(double) * (di4 * 4 + cache_size));
        double *gam = buf + di4 * 3;
        double *cache = gam + di4;
        size_t t, u;
        int ic, ne, np;
#pragma omp for schedule(dynamic, 1)
        for (ic = 0; ic < nchunk; ic++) {
                int c = nchunk - 1 - ic;
                double *dec = de_chunks + (size_t)c * natm * 3;
                for (t = chunk_loc[c]; t < chunk_loc[c+1]; t++) {
                        double qt = pairs[t].q * fac;
                        int ish = pairs[t].i;
                        int jsh = pairs[t].j;
                        for (u = t + 1; u-- > 0;) {
                                if (qt * pairs[u].q <= thr) {
                                        break;
                                }
                                ne = _orbit(dec, ish, jsh, pairs[u].i, pairs[u].j,
                                            &envs, buf, gam, cache, &np);
                                norbit += np > 0;
                                neval += ne;
                                npass += np;
                        }
                }
        }
        free(buf);
}
        size_t nde = (size_t)natm * 3;
        NPdset0(de, nde);
        for (c = 0; c < nchunk; c++) {
                double *dec = de_chunks + (size_t)c * nde;
                for (i = 0; i < nde; i++) {
                        de[i] += dec[i];
                }
        }
        if (stats != NULL) {
                stats[0] = norbit;
                stats[1] = neval;
                stats[2] = npass;
        }
        free(de_chunks);
        free(chunk_loc);
        free(pairs);
}
