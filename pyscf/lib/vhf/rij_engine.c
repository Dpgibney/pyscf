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
 * Integral-direct density-fitted J ("RI-J") by the McMurchie-Davidson scheme,
 * a J-engine for three-center integrals:
 *      pass 1: gam_P = sum_ab D_ab (ab|P)
 *      pass 2: J_ab = sum_P (ab|P) c_P
 * The product of two Cartesian Gaussians is a sum of Hermite Gaussians at the
 * pair's center (the E coefficients), so each AO pair's density is turned into
 * Hermite form once (pass 1), or its J block made from a Hermite potential
 * (pass 2), and meets every auxiliary function through the Hermite Coulomb
 * integrals R_tuv alone, without the Rys quadrature, the transfer between the
 * pair's two centers, the Cartesian-to-spherical transform of each block and
 * the contraction of every (ab|P) block with D. A spherical auxiliary function
 * is a solid harmonic of the derivatives of a Gaussian (Hobson), so it needs
 * the Hermite components of its own order only.
 *
 * AO functions are libcint's Cartesian ones (the caller transforms D and J
 * between them and the spherical ones), auxiliary functions libcint's
 * spherical ones. The results are those of int3c2e to rounding.
 *
 * Screening: the shell pair (A,B) and auxiliary shell C are taken together if
 * q_AB q_C d > cutoff, q_AB = max (ab|ab)^1/2, q_C = max (P|P)^1/2, and d the
 * largest |D| of the pair's block (pass 1) or |c| of the shell (pass 2).
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "config.h"
#include "cint.h"
#include "gto/gto.h"

#define LANES           8
#define RIJ_MAXL        24      // the highest lA + lB + lC
#define BOYS_NMAX       (RIJ_MAXL + 9)
#define BOYS_RES        16      // table points per unit of T
#define BOYS_TMAX       50
#define BOYS_NT         (BOYS_TMAX * BOYS_RES + 1)
#define BOYS_NTAYLOR    8

#define MAX(x, y)       ((x) > (y) ? (x) : (y))
#define MIN(x, y)       ((x) < (y) ? (x) : (y))
#define NHERM(l)        (((l)+1)*((l)+2)*((l)+3)/6)
#define NCART(l)        (((l)+1)*((l)+2)/2)

static double _boys_tab[BOYS_NT][BOYS_NMAX];    // F_n(j/BOYS_RES)
static double _boys_exp[BOYS_NT];               // exp(-j/BOYS_RES)
static const double _inv[] = {1., 1./2, 1./3, 1./4, 1./5, 1./6, 1./7, 1./8, 1./9, 1./10};
static double _inv_odd[RIJ_MAXL+1];             // 1/(2n+1)
static int _boys_ready = 0;

// Hermite indices: _hidx[t][u][v] for t+u+v <= RIJ_MAXL, by total order then t, u descending
static int _hidx[RIJ_MAXL+1][RIJ_MAXL+1][RIJ_MAXL+1];
// the recursion of the Hermite Coulomb integrals: the entry k = _hidx[t][u][v] (k > 0) is
// R[k] = X_dir R'[_rsrc1[k]] + _rcoef[k] R'[_rsrc2[k]], R' of the next order of the auxiliary index
static int _rdir[NHERM(RIJ_MAXL)];
static int _rsrc1[NHERM(RIJ_MAXL)];
static int _rsrc2[NHERM(RIJ_MAXL)];
static double _rcoef[NHERM(RIJ_MAXL)];
#define RIJ_LIJ         12      // the highest lA + lB
#define RIJ_LC          8       // the highest lC
static int _ridx_off[RIJ_LIJ+1][RIJ_LC+1];
static int *_ridx = NULL;

static void _init_tables()
{
        int j, n, k, t, u, v, l;
        for (j = 0; j < BOYS_NT; j++) {
                long double T = (long double)j / BOYS_RES;
                long double f[BOYS_NMAX + BOYS_NTAYLOR];
                int nmax = BOYS_NMAX + BOYS_NTAYLOR - 1;
                // F_nmax(T) = exp(-T) sum_k (2T)^k / ((2nmax+1)(2nmax+3)...(2nmax+2k+1))
                long double term = 1.L / (2 * nmax + 1);
                long double s = term;
                for (k = 1; k < 2000; k++) {
                        term *= 2 * T / (2 * nmax + 2 * k + 1);
                        s += term;
                        if (term < s * 1e-22L) {
                                break;
                        }
                }
                long double eT = expl(-T);
                f[nmax] = eT * s;
                for (n = nmax - 1; n >= 0; n--) {
                        f[n] = (2 * T * f[n+1] + eT) / (2 * n + 1);
                }
                _boys_exp[j] = (double)eT;
                for (n = 0; n < BOYS_NMAX; n++) {
                        _boys_tab[j][n] = (double)f[n];
                }
        }
        for (l = 0, k = 0; l <= RIJ_MAXL; l++) {
        for (t = l; t >= 0; t--) {
        for (u = l - t; u >= 0; u--, k++) {
                v = l - t - u;
                _hidx[t][u][v] = k;
        } } }
        for (l = 1; l <= RIJ_MAXL; l++) {
        for (t = l; t >= 0; t--) {
        for (u = l - t; u >= 0; u--) {
                v = l - t - u;
                k = _hidx[t][u][v];
                if (t > 0) {
                        _rdir[k] = 0;
                        _rsrc1[k] = _hidx[t-1][u][v];
                        _rsrc2[k] = t > 1 ? _hidx[t-2][u][v] : 0;
                        _rcoef[k] = t - 1;
                } else if (u > 0) {
                        _rdir[k] = 1;
                        _rsrc1[k] = _hidx[t][u-1][v];
                        _rsrc2[k] = u > 1 ? _hidx[t][u-2][v] : 0;
                        _rcoef[k] = u - 1;
                } else {
                        _rdir[k] = 2;
                        _rsrc1[k] = _hidx[t][u][v-1];
                        _rsrc2[k] = v > 1 ? _hidx[t][u][v-2] : 0;
                        _rcoef[k] = v - 1;
                }
        } } }
        for (n = 0; n <= RIJ_MAXL; n++) {
                _inv_odd[n] = 1. / (2 * n + 1);
        }
        int lij, lc, m;
        size_t nidx = 0;
        for (lij = 0; lij <= RIJ_LIJ; lij++) {
        for (lc = 0; lc <= RIJ_LC; lc++) {
                _ridx_off[lij][lc] = nidx;
                if (lij + lc <= RIJ_MAXL) {
                        nidx += NHERM(lij) * NCART(lc);
                }
        } }
        _ridx = malloc(sizeof(int) * nidx);
        int ht[NHERM(RIJ_MAXL)], hu[NHERM(RIJ_MAXL)], hv[NHERM(RIJ_MAXL)];
        for (l = 0, k = 0; l <= RIJ_MAXL; l++) {
        for (t = l; t >= 0; t--) {
        for (u = l - t; u >= 0; u--, k++) {
                ht[k] = t;
                hu[k] = u;
                hv[k] = l - t - u;
        } } }
        for (lij = 0; lij <= RIJ_LIJ; lij++) {
        for (lc = 0; lc <= RIJ_LC; lc++) {
                if (lij + lc > RIJ_MAXL) {
                        continue;
                }
                int *idx = _ridx + _ridx_off[lij][lc];
                int nc = NCART(lc);
                int kc = 0;
                int kx, ky;
                for (kx = lc; kx >= 0; kx--) {
                for (ky = lc - kx; ky >= 0; ky--, kc++) {
                        int kz = lc - kx - ky;
                        for (m = 0; m < NHERM(lij); m++) {
                                idx[m * nc + kc] = _hidx[ht[m]+kx][hu[m]+ky][hv[m]+kz];
                        }
                } }
        } }
        _boys_ready = 1;
}

void RIJinit()
{
#pragma omp critical(rij_init)
        if (!_boys_ready) {
                _init_tables();
        }
}

/*
 * F_0..F_L(T) of LANES values of T: below BOYS_TMAX, F_L by a Taylor series
 * from the nearest table point, then the downward recursion
 * F_n = (2T F_n+1 + exp(-T)) / (2n+1); above, exp(-T) < 2e-22 and
 * F_n = (2n-1)!! / (2T)^n sqrt(pi/(4T)), blended in by lane. f[n][lane].
 */
static void _boys(double f[][LANES], double *T, int L)
{
        int lane, n, k;
        double fL[LANES];
        double e[LANES];
        double d[LANES];
        double tc[LANES];
        int jt[LANES];
        int big = 0;
        for (lane = 0; lane < LANES; lane++) {
                big |= T[lane] >= BOYS_TMAX;
        }
#pragma omp simd
        for (lane = 0; lane < LANES; lane++) {
                tc[lane] = MIN(T[lane], BOYS_TMAX);
                jt[lane] = (int)(tc[lane] * BOYS_RES + .5);
                d[lane] = jt[lane] * (1. / BOYS_RES) - tc[lane];
        }
        const double *tab[LANES];
        for (lane = 0; lane < LANES; lane++) {
                tab[lane] = _boys_tab[jt[lane]] + L;
        }
#pragma omp simd
        for (lane = 0; lane < LANES; lane++) {
                fL[lane] = tab[lane][BOYS_NTAYLOR-1];
                e[lane] = 1;
        }
        // F_L(T) = sum_k F_L+k(T_j) d^k / k!,  d = T_j - T
        for (k = BOYS_NTAYLOR - 2; k >= 0; k--) {
#pragma omp simd
                for (lane = 0; lane < LANES; lane++) {
                        fL[lane] = fL[lane] * d[lane] * _inv[k] + tab[lane][k];
                }
        }
        // exp(-T) = exp(-T_j) exp(d)
        for (k = BOYS_NTAYLOR - 1; k >= 0; k--) {
#pragma omp simd
                for (lane = 0; lane < LANES; lane++) {
                        e[lane] = 1 + e[lane] * d[lane] * _inv[k];
                }
        }
#pragma omp simd
        for (lane = 0; lane < LANES; lane++) {
                e[lane] *= _boys_exp[jt[lane]];
                f[L][lane] = fL[lane];
        }
        for (n = L - 1; n >= 0; n--) {
                const double c = _inv_odd[n];
#pragma omp simd
                for (lane = 0; lane < LANES; lane++) {
                        f[n][lane] = (2 * tc[lane] * f[n+1][lane] + e[lane]) * c;
                }
        }
        if (big) {
                double g[LANES];
                double t2[LANES];
#pragma omp simd
                for (lane = 0; lane < LANES; lane++) {
                        t2[lane] = .5 / T[lane];
                        g[lane] = sqrt(M_PI * .5 * t2[lane]);
                        f[0][lane] = T[lane] >= BOYS_TMAX ? g[lane] : f[0][lane];
                }
                for (n = 1; n <= L; n++) {
                        const double c = 2 * n - 1;
#pragma omp simd
                        for (lane = 0; lane < LANES; lane++) {
                                g[lane] *= c * t2[lane];
                                f[n][lane] = T[lane] >= BOYS_TMAX ? g[lane] : f[n][lane];
                        }
                }
        }
}

/*
 * R_tuv(alpha, PC) of LANES pairs for t+u+v <= L: r[k][lane], k = _hidx[t][u][v],
 * from f[n] = R^(n)_000 = (-2 alpha)^n F_n(T) by
 * R^(n)_t+1,u,v = t R^(n+1)_t-1,u,v + X R^(n+1)_tuv (and for u, v), R = R^(0).
 * buf: NHERM(L) * LANES doubles.
 */
static void _hermite_r(double r[][LANES], double *buf, double f[][LANES],
                       double *pcx, double *pcy, double *pcz, int L)
{
        double *pc[3] = {pcx, pcy, pcz};
        int lane, n, k, nk;
        // level L into the buffer that ends as r after L swaps
        double (*cur)[LANES] = (L % 2 == 0) ? r : (double (*)[LANES])buf;
        double (*nxt)[LANES] = (L % 2 == 0) ? (double (*)[LANES])buf : r;
#pragma omp simd
        for (lane = 0; lane < LANES; lane++) {
                cur[0][lane] = f[L][lane];
        }
        for (n = L - 1; n >= 0; n--) {
#pragma omp simd
                for (lane = 0; lane < LANES; lane++) {
                        nxt[0][lane] = f[n][lane];
                }
                nk = NHERM(L - n);
                for (k = 1; k < nk; k++) {
                        const double *x = pc[_rdir[k]];
                        const double *s1 = cur[_rsrc1[k]];
                        const double *s2 = cur[_rsrc2[k]];
                        const double c = _rcoef[k];
#pragma omp simd
                        for (lane = 0; lane < LANES; lane++) {
                                nxt[k][lane] = x[lane] * s1[lane] + c * s2[lane];
                        }
                }
                double (*tmp)[LANES] = cur;
                cur = nxt;
                nxt = tmp;
        }
}

/*
 * E^ij_t of a primitive pair in one direction, i <= la, j <= lb, t <= i+j:
 * e[(i*(lb+1)+j)*(la+lb+1)+t], E^00_0 = 1 (the Gaussian prefactor apart).
 */
static void _hermite_e(double *e, int la, int lb, double p, double pa, double pb)
{
        const int nt = la + lb + 1;
        const int ldj = nt;
        const int ldi = (lb + 1) * nt;
        const double q = .5 / p;
        int i, j, t;
        memset(e, 0, sizeof(double) * (la + 1) * ldi);
        e[0] = 1;
        for (i = 0; i < la; i++) {
                double *e0 = e + i * ldi;
                double *e1 = e0 + ldi;
                for (t = 0; t <= i + 1; t++) {
                        double v = pa * (t <= i ? e0[t] : 0);
                        if (t > 0) v += q * e0[t-1];
                        if (t + 1 <= i) v += (t + 1) * e0[t+1];
                        e1[t] = v;
                }
        }
        for (j = 0; j < lb; j++) {
                for (i = 0; i <= la; i++) {
                        double *e0 = e + i * ldi + j * ldj;
                        double *e1 = e0 + ldj;
                        for (t = 0; t <= i + j + 1; t++) {
                                double v = pb * (t <= i + j ? e0[t] : 0);
                                if (t > 0) v += q * e0[t-1];
                                if (t + 1 <= i + j) v += (t + 1) * e0[t+1];
                                e1[t] = v;
                        }
                }
        }
}

static double _fac_sp(int l)
{
        switch (l) {
        case 0: return 0.282094791773878143;
        case 1: return 0.488602511902919921;
        default: return 1;
        }
}

// the Cartesian components of l in libcint's order
static void _cart_comp(int *nx, int *ny, int *nz, int l)
{
        int i, j, k = 0;
        for (i = l; i >= 0; i--) {
        for (j = l - i; j >= 0; j--, k++) {
                nx[k] = i;
                ny[k] = j;
                nz[k] = l - i - j;
        } }
}

typedef struct {
        int nauxbas;
        int *aux_loc;           // spherical auxiliary functions
        double *qaux;           // q_C
        int lmax;
        // auxiliary primitives by l, each list in the order of the shells' q_C (descending)
        int *pstart;            // [lmax+2]: the primitives of l are pstart[l]..pstart[l+1]-1
        int *pshell;            // shell of each primitive
        double *pexp;
        double *pinv;           // 1/pexp
        double *pcoef;          // coefficient * fac_sp(l) * (2 gamma)^-l * (-1)^l
        double *pr;             // center, 3 per primitive
        double *pq;             // q_C of its shell
        int *phoff;             // offset of each primitive's NCART(l) Hermite components
        int nherm;              // all of them
        double *chat;           // pass 2: the fitted density's, pcoef sum_m c2s[k,m] c_Cm
        double *c2s;            // [l]: c2s of l at c2s + c2s_off[l], NCART(l) x (2l+1)
        int *c2s_off;
} RIJAux;

/*
 * The AO pair (ish, jsh) through all auxiliary functions, as a list of
 * primitive pairs (p, P, prefactor and E coefficients). Pass 1 adds
 * sum_ab D_ab (ab|q k) to gam[phoff[q]+k], the Cartesian Hermite components k
 * of the auxiliary primitives q (made spherical by the caller); pass 2 adds
 * sum_Cm (ab|C m) cfit_Cm to the pair's block vblk[a*ncj+b] (Cartesian).
 */
typedef struct {
        int ish, jsh;
        int li, lj;
        double ri[3], rj[3];
} RIJPair;

static void _pair_loop(int pass, RIJPair *pr, double *dblk, double dmax, double *vblk,
                       double qab, double cutoff, double *gam, double *cfit,
                       RIJAux *aux, int *atm, int *bas, double *env, double *work)
{
        const int li = pr->li;
        const int lj = pr->lj;
        const int lij = li + lj;
        const int nci = NCART(li);
        const int ncj = NCART(lj);
        const int ish = pr->ish;
        const int jsh = pr->jsh;
        const int npi = bas(NPRIM_OF, ish);
        const int npj = bas(NPRIM_OF, jsh);
        const double *ai = env + bas(PTR_EXP, ish);
        const double *aj = env + bas(PTR_EXP, jsh);
        const double *ci = env + bas(PTR_COEFF, ish);
        const double *cj = env + bas(PTR_COEFF, jsh);
        const double fac_ij = _fac_sp(li) * _fac_sp(lj);
        const double *ri = pr->ri;
        const double *rj = pr->rj;
        const double rr = (ri[0]-rj[0])*(ri[0]-rj[0]) + (ri[1]-rj[1])*(ri[1]-rj[1])
                        + (ri[2]-rj[2])*(ri[2]-rj[2]);
        int ixs[NCART(RIJ_MAXL/2)], iys[NCART(RIJ_MAXL/2)], izs[NCART(RIJ_MAXL/2)];
        int jxs[NCART(RIJ_MAXL/2)], jys[NCART(RIJ_MAXL/2)], jzs[NCART(RIJ_MAXL/2)];
        _cart_comp(ixs, iys, izs, li);
        _cart_comp(jxs, jys, jzs, lj);

        const int nherm_ij = NHERM(lij);
        const int ne = (li + 1) * (lj + 1) * (lij + 1);
        double *ex = work;
        double *ey = ex + ne;
        double *ez = ey + ne;
        double *dh = ez + ne;                   // the Hermite density or potential
        double (*f)[LANES] = (double (*)[LANES])(dh + nherm_ij);
        double (*r)[LANES] = f + RIJ_MAXL + 1;
        double *rbuf = (double *)(r + NHERM(RIJ_MAXL));
        double (*w)[LANES] = (double (*)[LANES])(rbuf + NHERM(RIJ_MAXL) * LANES);
        double (*vl)[LANES] = w + NCART(RIJ_MAXL);      // pass 2: the potential by lane

        int ip, jp, lc, k, lane, t, u, v, a, b;
        for (ip = 0; ip < npi; ip++) {
        for (jp = 0; jp < npj; jp++) {
                const double p = ai[ip] + aj[jp];
                const double inv_p = 1. / p;
                const double mu = ai[ip] * aj[jp] * inv_p;
                const double kab = exp(-mu * rr) * ci[ip] * cj[jp] * fac_ij;
                if (fabs(kab) * dmax < cutoff * 1e-3) {
                        continue;
                }
                double P[3];
                for (k = 0; k < 3; k++) {
                        P[k] = (ai[ip] * ri[k] + aj[jp] * rj[k]) / p;
                }
                _hermite_e(ex, li, lj, p, P[0]-ri[0], P[0]-rj[0]);
                _hermite_e(ey, li, lj, p, P[1]-ri[1], P[1]-rj[1]);
                _hermite_e(ez, li, lj, p, P[2]-ri[2], P[2]-rj[2]);
                const int nt = lij + 1;
                const int ldi = (lj + 1) * nt;

                memset(dh, 0, sizeof(double) * nherm_ij);
                if (pass == 2) {
                        memset(vl, 0, sizeof(double) * nherm_ij * LANES);
                }
                if (pass == 1) {
                        // dh_tuv = sum_ab D_ab Ex Ey Ez
                        for (a = 0; a < nci; a++) {
                        for (b = 0; b < ncj; b++) {
                                double d = dblk[a * ncj + b] * kab;
                                if (d == 0) continue;
                                const double *exab = ex + ixs[a] * ldi + jxs[b] * nt;
                                const double *eyab = ey + iys[a] * ldi + jys[b] * nt;
                                const double *ezab = ez + izs[a] * ldi + jzs[b] * nt;
                                const int mx = ixs[a] + jxs[b];
                                const int my = iys[a] + jys[b];
                                const int mz = izs[a] + jzs[b];
                                for (t = 0; t <= mx; t++) {
                                        double dx = d * exab[t];
                                        for (u = 0; u <= my; u++) {
                                                double dxy = dx * eyab[u];
                                                for (v = 0; v <= mz; v++) {
                                                        dh[_hidx[t][u][v]] += dxy * ezab[v];
                                                }
                                        }
                                }
                        } }
                }

                for (lc = 0; lc <= aux->lmax; lc++) {
                        const int L = lij + lc;
                        const int ncc = NCART(lc);
                        int p0 = aux->pstart[lc];
                        int p1 = aux->pstart[lc+1];
                        // the screened prefix of the primitives (qaux: q_C in pass 1,
                        // q_C max|c_C| in pass 2)
                        double qcut = cutoff / (qab * dmax + 1e-300);
                        for (int pp = p0; pp < p1; pp += LANES) {
                                int nl = MIN(LANES, p1 - pp);
                                if (aux->pq[pp] < qcut) {
                                        break;
                                }
                                double al[LANES], T[LANES], lam[LANES];
                                double pcx[LANES], pcy[LANES], pcz[LANES], g[LANES], ginv[LANES];
                                int qs[LANES];
                                for (lane = 0; lane < LANES; lane++) {
                                        int q = pp + MIN(lane, nl - 1);
                                        qs[lane] = q;
                                        g[lane] = aux->pexp[q];
                                        ginv[lane] = aux->pinv[q];
                                        pcx[lane] = P[0] - aux->pr[3*q+0];
                                        pcy[lane] = P[1] - aux->pr[3*q+1];
                                        pcz[lane] = P[2] - aux->pr[3*q+2];
                                }
#pragma omp simd
                                for (lane = 0; lane < LANES; lane++) {
                                        double s1 = 1. / (p + g[lane]);
                                        al[lane] = p * g[lane] * s1;
                                        T[lane] = al[lane] * (pcx[lane]*pcx[lane] + pcy[lane]*pcy[lane]
                                                              + pcz[lane]*pcz[lane]);
                                        // 2 pi^5/2 / (p g sqrt(p+g)), 0 for the padding
                                        lam[lane] = lane < nl ? 34.986836655249725 * inv_p * ginv[lane]
                                                * sqrt(s1) : 0;
                                }
                                _boys(f, T, L);
                                double pw[LANES];
                                for (lane = 0; lane < LANES; lane++) {
                                        pw[lane] = 1;
                                }
                                for (int n = 1; n <= L; n++) {
#pragma omp simd
                                        for (lane = 0; lane < LANES; lane++) {
                                                pw[lane] *= -2 * al[lane];
                                                f[n][lane] *= pw[lane];
                                        }
                                }
                                _hermite_r(r, rbuf, f, pcx, pcy, pcz, L);
                                const int *ridx = _ridx + _ridx_off[lij][lc];
                                if (pass == 1) {
                                        // w[k] = sum_tuv dh_tuv R_t+kx,u+ky,v+kz, k the Cartesian
                                        // components of order lc
                                        for (k = 0; k < ncc; k++) {
                                                double acc[LANES] = {0};
                                                for (int m = 0; m < nherm_ij; m++) {
                                                        double d = dh[m];
                                                        const double *rk = r[ridx[m * ncc + k]];
#pragma omp simd
                                                        for (lane = 0; lane < LANES; lane++) {
                                                                acc[lane] += d * rk[lane];
                                                        }
                                                }
                                                // into the primitive's Hermite block
                                                for (lane = 0; lane < nl; lane++) {
                                                        gam[aux->phoff[qs[lane]] + k] += acc[lane] * lam[lane];
                                                }
                                        }
                                } else {
                                        // the fitted density's Hermite components, from the primitives'
                                        for (k = 0; k < ncc; k++) {
                                                for (lane = 0; lane < LANES; lane++) {
                                                        w[k][lane] = aux->chat[aux->phoff[qs[lane]] + k] * lam[lane];
                                                }
                                        }
                                        for (int m = 0; m < nherm_ij; m++) {
                                                const int *rm = ridx + m * ncc;
                                                double *vm = vl[m];
                                                for (k = 0; k < ncc; k++) {
                                                        const double *rk = r[rm[k]];
#pragma omp simd
                                                        for (lane = 0; lane < LANES; lane++) {
                                                                vm[lane] += w[k][lane] * rk[lane];
                                                        }
                                                }
                                        }
                                }
                        }
                }

                if (pass == 2) {
                        for (int m = 0; m < nherm_ij; m++) {
                                double sv = 0;
                                for (lane = 0; lane < LANES; lane++) {
                                        sv += vl[m][lane];
                                }
                                dh[m] = sv;
                        }
                        // v_ab += kab sum_tuv Ex Ey Ez V_tuv
                        for (a = 0; a < nci; a++) {
                        for (b = 0; b < ncj; b++) {
                                const double *exab = ex + ixs[a] * ldi + jxs[b] * nt;
                                const double *eyab = ey + iys[a] * ldi + jys[b] * nt;
                                const double *ezab = ez + izs[a] * ldi + jzs[b] * nt;
                                const int mx = ixs[a] + jxs[b];
                                const int my = iys[a] + jys[b];
                                const int mz = izs[a] + jzs[b];
                                double s = 0;
                                for (t = 0; t <= mx; t++) {
                                        double sx = 0;
                                        for (u = 0; u <= my; u++) {
                                                double sy = 0;
                                                for (v = 0; v <= mz; v++) {
                                                        sy += ezab[v] * dh[_hidx[t][u][v]];
                                                }
                                                sx += eyab[u] * sy;
                                        }
                                        s += exab[t] * sx;
                                }
                                vblk[a * ncj + b] += kab * s;
                        } }
                }
        } }
}

static size_t _work_size(int lmax_ao)
{
        int lij = 2 * lmax_ao;
        size_t ne = (size_t)(lmax_ao + 1) * (lmax_ao + 1) * (lij + 1);
        return ne * 3 + NHERM(lij) + (RIJ_MAXL + 1) * LANES + NHERM(RIJ_MAXL) * LANES * 2
                + NCART(RIJ_MAXL) * LANES + NHERM(lij) * LANES + 64;
}

/*
 * pass 1: gam[naux] (spherical auxiliary functions) = sum over the pairs of
 *      (2 - delta_AB) sum_ab D_ab (ab|P), D = dm (nao_cart x nao_cart, symmetric)
 * pass 2: vj[nao_cart x nao_cart] = sum_P (ab|P) cfit_P on the pairs, both triangles
 * pairs: npair shell pairs (ish >= jsh), qpair their q_AB; qaux: the screening
 * weight of each auxiliary shell, q_C in pass 1 and q_C max|cfit_C| in pass 2;
 * aux_order: the auxiliary shells by descending qaux; aux_loc; c2s and
 * c2s_off: libcint's Cartesian-to-spherical coefficients of each l (NCART(l) x
 * (2l+1), normalized='sp'); ao_loc: the Cartesian AO offsets of the shells.
 * Returns 0, or -1 (nothing computed) for AO shells above l = RIJ_LIJ/2 or
 * auxiliary shells above RIJ_LC.
 */
int RIJdrv(int pass, double *out, double *dm, double *cfit, int *pairs, int npair,
            double *qpair, double cutoff, int *ao_loc,
            int *atm, int natm, int *bas, int nbas, double *env,
            int *auxatm, int nauxatm, int *auxbas, int nauxbas, double *auxenv,
            int *aux_loc, double *qaux, int *aux_order, double *c2s, int *c2s_off)
{
        RIJinit();
        const size_t nao = ao_loc[nbas];
        const size_t naux = aux_loc[nauxbas];
        int i, l, sh, lmax_ao = 0, lmax_aux = 0, nprim_aux = 0;
        for (sh = 0; sh < nbas; sh++) {
                lmax_ao = MAX(lmax_ao, bas(ANG_OF, sh));
        }
        for (sh = 0; sh < nauxbas; sh++) {
                lmax_aux = MAX(lmax_aux, auxbas[ANG_OF+sh*BAS_SLOTS]);
                nprim_aux += auxbas[NPRIM_OF+sh*BAS_SLOTS] * auxbas[NCTR_OF+sh*BAS_SLOTS];
        }
        if (2 * lmax_ao > RIJ_LIJ || lmax_aux > RIJ_LC || 2 * lmax_ao + lmax_aux > RIJ_MAXL) {
                return -1;
        }
        // the auxiliary primitives by l, in the order of aux_order
        RIJAux aux;
        aux.nauxbas = nauxbas;
        aux.aux_loc = aux_loc;
        aux.qaux = qaux;
        aux.lmax = lmax_aux;
        aux.c2s = c2s;
        aux.c2s_off = c2s_off;
        aux.pstart = malloc(sizeof(int) * (lmax_aux + 2));
        aux.pshell = malloc(sizeof(int) * (nprim_aux + 1));
        aux.pexp = malloc(sizeof(double) * (nprim_aux + 1));
        aux.pinv = malloc(sizeof(double) * (nprim_aux + 1));
        aux.pcoef = malloc(sizeof(double) * (nprim_aux + 1));
        aux.pr = malloc(sizeof(double) * 3 * (nprim_aux + 1));
        aux.pq = malloc(sizeof(double) * (nprim_aux + 1));
        aux.phoff = malloc(sizeof(int) * (nprim_aux + 1));
        int np = 0;
        int nh = 0;
        for (l = 0; l <= lmax_aux; l++) {
                aux.pstart[l] = np;
                for (i = 0; i < nauxbas; i++) {
                        sh = aux_order[i];
                        int *b = auxbas + sh * BAS_SLOTS;
                        if (b[ANG_OF] != l) {
                                continue;
                        }
                        // one contracted function per auxiliary shell (general contractions are split
                        // by the caller)
                        double *e = auxenv + b[PTR_EXP];
                        double *c = auxenv + b[PTR_COEFF];
                        double *rc = auxenv + auxatm[PTR_COORD + b[ATOM_OF] * ATM_SLOTS];
                        for (int ip = 0; ip < b[NPRIM_OF]; ip++, np++) {
                                aux.pshell[np] = sh;
                                aux.pexp[np] = e[ip];
                                aux.pinv[np] = 1. / e[ip];
                                aux.pcoef[np] = c[ip] * _fac_sp(l) * pow(-2 * e[ip], -l);
                                aux.pr[3*np+0] = rc[0];
                                aux.pr[3*np+1] = rc[1];
                                aux.pr[3*np+2] = rc[2];
                                aux.pq[np] = qaux[sh];
                                aux.phoff[np] = nh;
                                nh += NCART(l);
                        }
                }
        }
        aux.pstart[lmax_aux+1] = np;
        aux.nherm = nh;
        aux.chat = NULL;
        if (pass == 2) {
                aux.chat = malloc(sizeof(double) * (nh + 1));
                for (i = 0; i < np; i++) {
                        sh = aux.pshell[i];
                        l = auxbas[ANG_OF+sh*BAS_SLOTS];
                        int nsc = 2 * l + 1;
                        double *cs = c2s + c2s_off[l];
                        double *cf = cfit + aux_loc[sh];
                        for (int k = 0; k < NCART(l); k++) {
                                double v = 0;
                                for (int m = 0; m < nsc; m++) {
                                        v += cs[k * nsc + m] * cf[m];
                                }
                                aux.chat[aux.phoff[i] + k] = v * aux.pcoef[i];
                        }
                }
        }

        if (pass == 1) {
                memset(out, 0, sizeof(double) * naux);
        } else {
                memset(out, 0, sizeof(double) * nao * nao);
        }
        const size_t wsize = _work_size(lmax_ao);
#pragma omp parallel
{
        double *work = malloc(sizeof(double) * wsize);
        double *gam = NULL;
        if (pass == 1) {
                gam = calloc(aux.nherm + 1, sizeof(double));
        }
        int maxc = NCART(lmax_ao);
        double *blk = malloc(sizeof(double) * maxc * maxc);
        int ip, a, b;
#pragma omp for schedule(dynamic, 4)
        for (ip = 0; ip < npair; ip++) {
                int ish = pairs[2*ip];
                int jsh = pairs[2*ip+1];
                RIJPair pr;
                pr.ish = ish;
                pr.jsh = jsh;
                pr.li = bas(ANG_OF, ish);
                pr.lj = bas(ANG_OF, jsh);
                double *r = env + atm(PTR_COORD, bas(ATOM_OF, ish));
                pr.ri[0] = r[0]; pr.ri[1] = r[1]; pr.ri[2] = r[2];
                r = env + atm(PTR_COORD, bas(ATOM_OF, jsh));
                pr.rj[0] = r[0]; pr.rj[1] = r[1]; pr.rj[2] = r[2];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int di = ao_loc[ish+1] - i0;
                int dj = ao_loc[jsh+1] - j0;
                if (pass == 1) {
                        double fac = ish == jsh ? 1 : 2;
                        double dmax = 0;
                        for (a = 0; a < di; a++) {
                        for (b = 0; b < dj; b++) {
                                blk[a*dj+b] = dm[(i0+a)*nao+j0+b] * fac;
                                dmax = MAX(dmax, fabs(blk[a*dj+b]));
                        } }
                        if (qpair[ip] * dmax * qaux[aux_order[0]] < cutoff) {
                                continue;
                        }
                        _pair_loop(1, &pr, blk, dmax, NULL, qpair[ip], cutoff, gam, NULL,
                                   &aux, atm, bas, env, work);
                } else {
                        memset(blk, 0, sizeof(double) * di * dj);
                        _pair_loop(2, &pr, NULL, 1., blk, qpair[ip], cutoff, NULL, cfit,
                                   &aux, atm, bas, env, work);
                        for (a = 0; a < di; a++) {
                        for (b = 0; b < dj; b++) {
                                out[(i0+a)*nao+j0+b] = blk[a*dj+b];
                                out[(j0+b)*nao+i0+a] = blk[a*dj+b];
                        } }
                }
        }
        if (pass == 1) {
                // the primitives' Hermite blocks into the spherical functions of their shells
                double *gs = calloc(naux, sizeof(double));
                for (int iq = 0; iq < np; iq++) {
                        int shq = aux.pshell[iq];
                        int lq = auxbas[ANG_OF+shq*BAS_SLOTS];
                        int nsc = 2 * lq + 1;
                        double *cs = c2s + c2s_off[lq];
                        double *gh = gam + aux.phoff[iq];
                        double *g = gs + aux_loc[shq];
                        for (int m = 0; m < nsc; m++) {
                                double v = 0;
                                for (int k = 0; k < NCART(lq); k++) {
                                        v += cs[k * nsc + m] * gh[k];
                                }
                                g[m] += v * aux.pcoef[iq];
                        }
                }
#pragma omp critical
                for (size_t n = 0; n < naux; n++) {
                        out[n] += gs[n];
                }
                free(gs);
                free(gam);
        }
        free(blk);
        free(work);
}
        free(aux.pstart);
        free(aux.pshell);
        free(aux.pexp);
        free(aux.pinv);
        free(aux.pcoef);
        free(aux.pr);
        free(aux.pq);
        free(aux.phoff);
        if (aux.chat != NULL) {
                free(aux.chat);
        }
        return 0;
}

/*
 * The J-engine for many density matrices at once (the densities of a CPHF or
 * TDDFT response): the Hermite Coulomb integrals of an AO primitive pair with
 * the auxiliary primitives do not depend on the density, so they are computed
 * once, RIJ_BCHUNK batches of LANES primitives at a time, into
 *      B[m, (q,k)] = lam_q R_(t,u,v)_m + k,
 * and meet the Hermite densities (pass 1) or the fitted densities (pass 2) of
 * all nset matrices in one matrix product each:
 *      pass 1: gam[x, (q,k)] += sum_m dh[x,m] B[m, (q,k)]
 *      pass 2: V[x, m] += sum_(q,k) chat[x, (q,k)] B[m, (q,k)]
 * dblk and vblk are the pair's blocks [a,b,x], gam[nset, ldg] and
 * chat[nset, nherm of aux]; dht[nherm_ij, nset] and bmat[nherm_ij, ldb] are work
 * space. dmax: the largest |D| of the pair over the densities (pass 1), 1 in
 * pass 2, where aux->pq carries the largest |c| of each shell over them.
 */
#define RIJ_BCHUNK      32

void dgemm_(const char*, const char*, const int*, const int*, const int*,
            const double*, const double*, const int*, const double*, const int*,
            const double*, double*, const int*);

static void _pair_loop_many(int pass, RIJPair *pr, int nset, double *dblk, double dmax,
                            double *vblk, double qab, double cutoff, double *gam, int ldg,
                            double *chat, RIJAux *aux, int *atm, int *bas, double *env,
                            double *work, double *dht, double *bmat, int ldb)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D1 = 1;
        const int li = pr->li;
        const int lj = pr->lj;
        const int lij = li + lj;
        const int nci = NCART(li);
        const int ncj = NCART(lj);
        const int ish = pr->ish;
        const int jsh = pr->jsh;
        const int npi = bas(NPRIM_OF, ish);
        const int npj = bas(NPRIM_OF, jsh);
        const double *ai = env + bas(PTR_EXP, ish);
        const double *aj = env + bas(PTR_EXP, jsh);
        const double *ci = env + bas(PTR_COEFF, ish);
        const double *cj = env + bas(PTR_COEFF, jsh);
        const double fac_ij = _fac_sp(li) * _fac_sp(lj);
        const double *ri = pr->ri;
        const double *rj = pr->rj;
        const double rr = (ri[0]-rj[0])*(ri[0]-rj[0]) + (ri[1]-rj[1])*(ri[1]-rj[1])
                        + (ri[2]-rj[2])*(ri[2]-rj[2]);
        int ixs[NCART(RIJ_MAXL/2)], iys[NCART(RIJ_MAXL/2)], izs[NCART(RIJ_MAXL/2)];
        int jxs[NCART(RIJ_MAXL/2)], jys[NCART(RIJ_MAXL/2)], jzs[NCART(RIJ_MAXL/2)];
        _cart_comp(ixs, iys, izs, li);
        _cart_comp(jxs, jys, jzs, lj);

        const int nherm_ij = NHERM(lij);
        const int ne = (li + 1) * (lj + 1) * (lij + 1);
        double *ex = work;
        double *ey = ex + ne;
        double *ez = ey + ne;
        double (*f)[LANES] = (double (*)[LANES])(ez + ne + nherm_ij);
        double (*r)[LANES] = f + RIJ_MAXL + 1;
        double *rbuf = (double *)(r + NHERM(RIJ_MAXL));

        int ip, jp, lc, k, lane, t, u, v, a, b, x, m;
        for (ip = 0; ip < npi; ip++) {
        for (jp = 0; jp < npj; jp++) {
                const double p = ai[ip] + aj[jp];
                const double inv_p = 1. / p;
                const double mu = ai[ip] * aj[jp] * inv_p;
                const double kab = exp(-mu * rr) * ci[ip] * cj[jp] * fac_ij;
                if (fabs(kab) * dmax < cutoff * 1e-3) {
                        continue;
                }
                double P[3];
                for (k = 0; k < 3; k++) {
                        P[k] = (ai[ip] * ri[k] + aj[jp] * rj[k]) / p;
                }
                _hermite_e(ex, li, lj, p, P[0]-ri[0], P[0]-rj[0]);
                _hermite_e(ey, li, lj, p, P[1]-ri[1], P[1]-rj[1]);
                _hermite_e(ez, li, lj, p, P[2]-ri[2], P[2]-rj[2]);
                const int nt = lij + 1;
                const int ldi = (lj + 1) * nt;

                memset(dht, 0, sizeof(double) * nherm_ij * nset);
                if (pass == 1) {
                        // dht[tuv,x] = sum_ab D[x]_ab Ex Ey Ez
                        for (a = 0; a < nci; a++) {
                        for (b = 0; b < ncj; b++) {
                                const double *d = dblk + (size_t)(a * ncj + b) * nset;
                                const double *exab = ex + ixs[a] * ldi + jxs[b] * nt;
                                const double *eyab = ey + iys[a] * ldi + jys[b] * nt;
                                const double *ezab = ez + izs[a] * ldi + jzs[b] * nt;
                                const int mx = ixs[a] + jxs[b];
                                const int my = iys[a] + jys[b];
                                const int mz = izs[a] + jzs[b];
                                for (t = 0; t <= mx; t++) {
                                for (u = 0; u <= my; u++) {
                                        const double exy = kab * exab[t] * eyab[u];
                                        for (v = 0; v <= mz; v++) {
                                                const double e = exy * ezab[v];
                                                double *h = dht + (size_t)_hidx[t][u][v] * nset;
#pragma omp simd
                                                for (x = 0; x < nset; x++) {
                                                        h[x] += d[x] * e;
                                                }
                                        }
                                } }
                        } }
                }

                for (lc = 0; lc <= aux->lmax; lc++) {
                        const int L = lij + lc;
                        const int ncc = NCART(lc);
                        const int p0 = aux->pstart[lc];
                        const int p1 = aux->pstart[lc+1];
                        const int *ridx = _ridx + _ridx_off[lij][lc];
                        // the screened prefix of the primitives (qaux: q_C in pass 1,
                        // q_C max|c_C| in pass 2)
                        const double qcut = cutoff / (qab * dmax + 1e-300);
                        int pp = p0;
                        while (pp < p1 && aux->pq[pp] >= qcut) {
                                // a chunk of batches into bmat
                                const int pp0 = pp;
                                int nprim = 0;
                                int nb;
                                for (nb = 0; nb < RIJ_BCHUNK && pp < p1 && aux->pq[pp] >= qcut;
                                     nb++, pp += LANES) {
                                        const int nl = MIN(LANES, p1 - pp);
                                        double al[LANES], T[LANES], lam[LANES];
                                        double pcx[LANES], pcy[LANES], pcz[LANES], g[LANES], ginv[LANES];
                                        for (lane = 0; lane < LANES; lane++) {
                                                int q = pp + MIN(lane, nl - 1);
                                                g[lane] = aux->pexp[q];
                                                ginv[lane] = aux->pinv[q];
                                                pcx[lane] = P[0] - aux->pr[3*q+0];
                                                pcy[lane] = P[1] - aux->pr[3*q+1];
                                                pcz[lane] = P[2] - aux->pr[3*q+2];
                                        }
#pragma omp simd
                                        for (lane = 0; lane < LANES; lane++) {
                                                double s1 = 1. / (p + g[lane]);
                                                al[lane] = p * g[lane] * s1;
                                                T[lane] = al[lane] * (pcx[lane]*pcx[lane] + pcy[lane]*pcy[lane]
                                                                      + pcz[lane]*pcz[lane]);
                                                // 2 pi^5/2 / (p g sqrt(p+g)), 0 for the padding
                                                lam[lane] = lane < nl ? 34.986836655249725 * inv_p * ginv[lane]
                                                        * sqrt(s1) : 0;
                                        }
                                        _boys(f, T, L);
                                        double pw[LANES];
                                        for (lane = 0; lane < LANES; lane++) {
                                                pw[lane] = 1;
                                        }
                                        for (int n = 1; n <= L; n++) {
#pragma omp simd
                                                for (lane = 0; lane < LANES; lane++) {
                                                        pw[lane] *= -2 * al[lane];
                                                        f[n][lane] *= pw[lane];
                                                }
                                        }
                                        _hermite_r(r, rbuf, f, pcx, pcy, pcz, L);
                                        // bmat[m, (q,k)], the primitives of the chunk in order
                                        for (m = 0; m < nherm_ij; m++) {
                                                double *bm = bmat + (size_t)m * ldb + (size_t)nprim * ncc;
                                                const int *rm = ridx + m * ncc;
                                                for (k = 0; k < ncc; k++) {
                                                        const double *rk = r[rm[k]];
                                                        for (lane = 0; lane < nl; lane++) {
                                                                bm[lane * ncc + k] = rk[lane] * lam[lane];
                                                        }
                                                }
                                        }
                                        nprim += nl;
                                }
                                const int ncols = nprim * ncc;
                                const int col0 = aux->phoff[pp0];
                                if (pass == 1) {
                                        dgemm_(&TRANS_N, &TRANS_T, &ncols, &nset, &nherm_ij, &D1,
                                               bmat, &ldb, dht, &nset, &D1, gam + col0, &ldg);
                                } else {
                                        dgemm_(&TRANS_T, &TRANS_N, &nset, &nherm_ij, &ncols, &D1,
                                               chat + col0, &ldg, bmat, &ldb, &D1, dht, &nset);
                                }
                        }
                }

                if (pass == 2) {
                        // v[x]_ab += kab sum_tuv Ex Ey Ez V[tuv,x]
                        for (a = 0; a < nci; a++) {
                        for (b = 0; b < ncj; b++) {
                                double *vab = vblk + (size_t)(a * ncj + b) * nset;
                                const double *exab = ex + ixs[a] * ldi + jxs[b] * nt;
                                const double *eyab = ey + iys[a] * ldi + jys[b] * nt;
                                const double *ezab = ez + izs[a] * ldi + jzs[b] * nt;
                                const int mx = ixs[a] + jxs[b];
                                const int my = iys[a] + jys[b];
                                const int mz = izs[a] + jzs[b];
                                for (t = 0; t <= mx; t++) {
                                for (u = 0; u <= my; u++) {
                                        const double exy = kab * exab[t] * eyab[u];
                                        for (v = 0; v <= mz; v++) {
                                                const double e = exy * ezab[v];
                                                const double *h = dht + (size_t)_hidx[t][u][v] * nset;
#pragma omp simd
                                                for (x = 0; x < nset; x++) {
                                                        vab[x] += e * h[x];
                                                }
                                        }
                                } }
                        } }
                }
        } }
}

/*
 * RIJdrv for nset density matrices in one pass over the integrals:
 * pass 1: out[nset,naux] from dm[nset,nao_cart,nao_cart] (symmetric);
 * pass 2: out[nset,nao_cart,nao_cart] from cfit[nset,naux]. qaux: q_C in pass 1,
 * q_C times the largest |cfit_C| over the sets in pass 2. The other arguments
 * and the return value as RIJdrv.
 */
int RIJdrv_many(int pass, double *out, double *dm, double *cfit, int nset, int *pairs,
                int npair, double *qpair, double cutoff, int *ao_loc,
                int *atm, int natm, int *bas, int nbas, double *env,
                int *auxatm, int nauxatm, int *auxbas, int nauxbas, double *auxenv,
                int *aux_loc, double *qaux, int *aux_order, double *c2s, int *c2s_off)
{
        RIJinit();
        const size_t nao = ao_loc[nbas];
        const size_t nn = nao * nao;
        const size_t naux = aux_loc[nauxbas];
        int i, l, sh, x, lmax_ao = 0, lmax_aux = 0, nprim_aux = 0;
        for (sh = 0; sh < nbas; sh++) {
                lmax_ao = MAX(lmax_ao, bas(ANG_OF, sh));
        }
        for (sh = 0; sh < nauxbas; sh++) {
                lmax_aux = MAX(lmax_aux, auxbas[ANG_OF+sh*BAS_SLOTS]);
                nprim_aux += auxbas[NPRIM_OF+sh*BAS_SLOTS] * auxbas[NCTR_OF+sh*BAS_SLOTS];
        }
        if (2 * lmax_ao > RIJ_LIJ || lmax_aux > RIJ_LC || 2 * lmax_ao + lmax_aux > RIJ_MAXL) {
                return -1;
        }
        // the auxiliary primitives by l, in the order of aux_order
        RIJAux aux;
        aux.nauxbas = nauxbas;
        aux.aux_loc = aux_loc;
        aux.qaux = qaux;
        aux.lmax = lmax_aux;
        aux.c2s = c2s;
        aux.c2s_off = c2s_off;
        aux.pstart = malloc(sizeof(int) * (lmax_aux + 2));
        aux.pshell = malloc(sizeof(int) * (nprim_aux + 1));
        aux.pexp = malloc(sizeof(double) * (nprim_aux + 1));
        aux.pinv = malloc(sizeof(double) * (nprim_aux + 1));
        aux.pcoef = malloc(sizeof(double) * (nprim_aux + 1));
        aux.pr = malloc(sizeof(double) * 3 * (nprim_aux + 1));
        aux.pq = malloc(sizeof(double) * (nprim_aux + 1));
        aux.phoff = malloc(sizeof(int) * (nprim_aux + 1));
        int np = 0;
        int nh = 0;
        for (l = 0; l <= lmax_aux; l++) {
                aux.pstart[l] = np;
                for (i = 0; i < nauxbas; i++) {
                        sh = aux_order[i];
                        int *b = auxbas + sh * BAS_SLOTS;
                        if (b[ANG_OF] != l) {
                                continue;
                        }
                        double *e = auxenv + b[PTR_EXP];
                        double *c = auxenv + b[PTR_COEFF];
                        double *rc = auxenv + auxatm[PTR_COORD + b[ATOM_OF] * ATM_SLOTS];
                        for (int ip = 0; ip < b[NPRIM_OF]; ip++, np++) {
                                aux.pshell[np] = sh;
                                aux.pexp[np] = e[ip];
                                aux.pinv[np] = 1. / e[ip];
                                aux.pcoef[np] = c[ip] * _fac_sp(l) * pow(-2 * e[ip], -l);
                                aux.pr[3*np+0] = rc[0];
                                aux.pr[3*np+1] = rc[1];
                                aux.pr[3*np+2] = rc[2];
                                aux.pq[np] = qaux[sh];
                                aux.phoff[np] = nh;
                                nh += NCART(l);
                        }
                }
        }
        aux.pstart[lmax_aux+1] = np;
        aux.nherm = nh;
        const int ldg = nh + 1;
        // pass 2: the fitted densities' Hermite components [nset, ldg], from the primitives'
        aux.chat = NULL;
        if (pass == 2) {
                aux.chat = malloc(sizeof(double) * nset * ldg);
#pragma omp parallel for schedule(static) private(i, sh, l)
                for (x = 0; x < nset; x++) {
                        double *chx = aux.chat + (size_t)x * ldg;
                        for (i = 0; i < np; i++) {
                                sh = aux.pshell[i];
                                l = auxbas[ANG_OF+sh*BAS_SLOTS];
                                int nsc = 2 * l + 1;
                                double *cs = c2s + c2s_off[l];
                                double *cf = cfit + x * naux + aux_loc[sh];
                                for (int k = 0; k < NCART(l); k++) {
                                        double v = 0;
                                        for (int m = 0; m < nsc; m++) {
                                                v += cs[k * nsc + m] * cf[m];
                                        }
                                        chx[aux.phoff[i] + k] = v * aux.pcoef[i];
                                }
                        }
                }
        }

        if (pass == 1) {
                memset(out, 0, sizeof(double) * nset * naux);
        } else {
                memset(out, 0, sizeof(double) * nset * nn);
        }
        const size_t wsize = _work_size(lmax_ao);
        const int nherm_max = NHERM(2 * lmax_ao);
        const int ldb = RIJ_BCHUNK * LANES * NCART(lmax_aux);
#pragma omp parallel
{
        double *work = malloc(sizeof(double) * wsize);
        double *dht = malloc(sizeof(double) * nherm_max * nset);
        double *bmat = malloc(sizeof(double) * nherm_max * ldb);
        double *gam = NULL;
        if (pass == 1) {
                gam = calloc((size_t)nset * ldg, sizeof(double));
        }
        int maxc = NCART(lmax_ao);
        double *blk = malloc(sizeof(double) * maxc * maxc * nset);
        int ip, a, b, y;
#pragma omp for schedule(dynamic, 4)
        for (ip = 0; ip < npair; ip++) {
                int ish = pairs[2*ip];
                int jsh = pairs[2*ip+1];
                RIJPair pr;
                pr.ish = ish;
                pr.jsh = jsh;
                pr.li = bas(ANG_OF, ish);
                pr.lj = bas(ANG_OF, jsh);
                double *r = env + atm(PTR_COORD, bas(ATOM_OF, ish));
                pr.ri[0] = r[0]; pr.ri[1] = r[1]; pr.ri[2] = r[2];
                r = env + atm(PTR_COORD, bas(ATOM_OF, jsh));
                pr.rj[0] = r[0]; pr.rj[1] = r[1]; pr.rj[2] = r[2];
                int i0 = ao_loc[ish];
                int j0 = ao_loc[jsh];
                int di = ao_loc[ish+1] - i0;
                int dj = ao_loc[jsh+1] - j0;
                if (pass == 1) {
                        double fac = ish == jsh ? 1 : 2;
                        double dmax = 0;
                        for (y = 0; y < nset; y++) {
                                double *dmy = dm + y * nn;
                                for (a = 0; a < di; a++) {
                                for (b = 0; b < dj; b++) {
                                        double d = dmy[(i0+a)*nao+j0+b] * fac;
                                        blk[(size_t)(a*dj+b)*nset+y] = d;
                                        dmax = MAX(dmax, fabs(d));
                                } }
                        }
                        if (qpair[ip] * dmax * qaux[aux_order[0]] < cutoff) {
                                continue;
                        }
                        _pair_loop_many(1, &pr, nset, blk, dmax, NULL, qpair[ip], cutoff, gam, ldg,
                                        NULL, &aux, atm, bas, env, work, dht, bmat, ldb);
                } else {
                        memset(blk, 0, sizeof(double) * di * dj * nset);
                        _pair_loop_many(2, &pr, nset, NULL, 1., blk, qpair[ip], cutoff, NULL, ldg,
                                        aux.chat, &aux, atm, bas, env, work, dht, bmat, ldb);
                        for (y = 0; y < nset; y++) {
                                double *oy = out + y * nn;
                                for (a = 0; a < di; a++) {
                                for (b = 0; b < dj; b++) {
                                        double vab = blk[(size_t)(a*dj+b)*nset+y];
                                        oy[(i0+a)*nao+j0+b] = vab;
                                        oy[(j0+b)*nao+i0+a] = vab;
                                } }
                        }
                }
        }
        if (pass == 1) {
                // the primitives' Hermite blocks into the spherical functions of their shells
                double *gs = calloc((size_t)nset * naux, sizeof(double));
                for (y = 0; y < nset; y++) {
                        for (int iq = 0; iq < np; iq++) {
                                int shq = aux.pshell[iq];
                                int lq = auxbas[ANG_OF+shq*BAS_SLOTS];
                                int nsc = 2 * lq + 1;
                                double *cs = c2s + c2s_off[lq];
                                double *gh = gam + (size_t)y * ldg + aux.phoff[iq];
                                double *g = gs + y * naux + aux_loc[shq];
                                for (int m = 0; m < nsc; m++) {
                                        double v = 0;
                                        for (int k = 0; k < NCART(lq); k++) {
                                                v += cs[k * nsc + m] * gh[k];
                                        }
                                        g[m] += v * aux.pcoef[iq];
                                }
                        }
                }
#pragma omp critical
                for (size_t n = 0; n < nset * naux; n++) {
                        out[n] += gs[n];
                }
                free(gs);
                free(gam);
        }
        free(blk);
        free(bmat);
        free(dht);
        free(work);
}
        free(aux.pstart);
        free(aux.pshell);
        free(aux.pexp);
        free(aux.pinv);
        free(aux.pcoef);
        free(aux.pr);
        free(aux.pq);
        free(aux.phoff);
        if (aux.chat != NULL) {
                free(aux.chat);
        }
        return 0;
}
