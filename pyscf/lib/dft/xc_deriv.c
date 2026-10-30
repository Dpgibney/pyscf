/* Copyright 2022 The PySCF Developers. All Rights Reserved.

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
#include <assert.h>

// The point loops below run in blocks of POINT_BLK points spread over the threads. Every
// output element is computed from its own point only, so results do not depend on the split.
#define POINT_BLK       1024
#ifndef MIN
#define MIN(x, y)       ((x) < (y) ? (x) : (y))
#endif

/*
 * tmp = fg[...,[[0,1],[1,2]],...]
 * tmp[...,0,0,...] *= 2
 * tmp[...,1,1,...] *= 2
 * qg = np.einsum('nabmg,bxg->naxmg', tmp, rho[:,1:4])
 */
void VXCfg_to_direct_deriv(double *qg, double *fg, double *rho,
                           int ncounts, int nvar, int mcounts, int ngrids)
{
        size_t Ngrids = ngrids;
        size_t mg = mcounts * Ngrids;
        double *rho_ax = rho + Ngrids;
        double *rho_ay = rho_ax + Ngrids;
        double *rho_az = rho_ay + Ngrids;
        double *rho_bx = rho_ax + nvar * Ngrids;
        double *rho_by = rho_ay + nvar * Ngrids;
        double *rho_bz = rho_az + nvar * Ngrids;
        int nblk = (ngrids + POINT_BLK - 1) / POINT_BLK;
#pragma omp parallel
{
        double *qg_ax, *qg_ay, *qg_az, *qg_bx, *qg_by, *qg_bz;
        double *fg_aa, *fg_ab, *fg_bb;
        double vaa, vab, vbb;
        int n, m, g, ib, g0, g1;
#pragma omp for schedule(static)
        for (ib = 0; ib < nblk; ib++) {
        g0 = ib * POINT_BLK;
        g1 = MIN(g0 + POINT_BLK, ngrids);
        for (n = 0; n < ncounts; n++) {
                qg_ax = qg + n * 2 * 3 * mg;
                qg_ay = qg_ax + mg;
                qg_az = qg_ay + mg;
                qg_bx = qg_ax + 3 * mg;
                qg_by = qg_ay + 3 * mg;
                qg_bz = qg_az + 3 * mg;
                fg_aa = fg + n * 3 * mg;
                fg_ab = fg_aa + mg;
                fg_bb = fg_ab + mg;
                for (m = 0; m < mcounts; m++) {
#pragma GCC ivdep
                for (g = g0; g < g1; g++) {
                        vaa = fg_aa[m*Ngrids+g] * 2;
                        vab = fg_ab[m*Ngrids+g];
                        vbb = fg_bb[m*Ngrids+g] * 2;
                        qg_ax[m*Ngrids+g] = vaa * rho_ax[g] + vab * rho_bx[g];
                        qg_ay[m*Ngrids+g] = vaa * rho_ay[g] + vab * rho_by[g];
                        qg_az[m*Ngrids+g] = vaa * rho_az[g] + vab * rho_bz[g];
                        qg_bx[m*Ngrids+g] = vbb * rho_bx[g] + vab * rho_ax[g];
                        qg_by[m*Ngrids+g] = vbb * rho_by[g] + vab * rho_ay[g];
                        qg_bz[m*Ngrids+g] = vbb * rho_bz[g] + vab * rho_az[g];
                } }
        } }
}
}

void VXCud2ts(double *v_ts, double *v_ud, int ncounts, size_t ngrids)
{
        size_t nblk = (ngrids + POINT_BLK - 1) / POINT_BLK;
#pragma omp parallel
{
        double *vu, *vd, *vt, *vs;
        size_t i, i0, i1, ib;
        int n;
#pragma omp for schedule(static)
        for (ib = 0; ib < nblk; ib++) {
        i0 = ib * POINT_BLK;
        i1 = MIN(i0 + POINT_BLK, ngrids);
        for (n = 0; n < ncounts; n++) {
                vu = v_ud + 2*n    * ngrids;
                vd = v_ud +(2*n+1) * ngrids;
                vt = v_ts + 2*n    * ngrids;
                vs = v_ts +(2*n+1) * ngrids;
#pragma GCC ivdep
                for (i = i0; i < i1; i++) {
                        vt[i] = (vu[i] + vd[i]) * .5;
                        vs[i] = (vu[i] - vd[i]) * .5;
                }
        } }
}
}

void VXCts2ud(double *v_ud, double *v_ts, int ncounts, size_t ngrids)
{
        size_t nblk = (ngrids + POINT_BLK - 1) / POINT_BLK;
#pragma omp parallel
{
        double *vu, *vd, *vt, *vs;
        size_t i, i0, i1, ib;
        int n;
#pragma omp for schedule(static)
        for (ib = 0; ib < nblk; ib++) {
        i0 = ib * POINT_BLK;
        i1 = MIN(i0 + POINT_BLK, ngrids);
        for (n = 0; n < ncounts; n++) {
                vu = v_ud + 2*n    * ngrids;
                vd = v_ud +(2*n+1) * ngrids;
                vt = v_ts + 2*n    * ngrids;
                vs = v_ts +(2*n+1) * ngrids;
#pragma GCC ivdep
                for (i = i0; i < i1; i++) {
                        vu[i] = vt[i] + vs[i];
                        vd[i] = vt[i] - vs[i];
                }
        } }
}
}

#define frho_at(n, a, x, g)     frho[((n*2+a)*Nvg + x*Ngrids + g)]
#define fsigma_at(x, n, g)      fsigma[(x*Ncg + n*Ngrids + g)]
// spin0 shape: frho[...,nvar,ngrids], fsigma[:,...,ngrids], rho[nvar,ngrids]
void VXCunfold_sigma_spin0(double *frho, double *fsigma, double *rho,
                           int ncounts, int nvar, int ngrids)
{
        size_t Ngrids = ngrids;
        size_t Ncg = ncounts * Ngrids;
        size_t Nvg = nvar * Ngrids;
        int nblk = (ngrids + POINT_BLK - 1) / POINT_BLK;
        assert(nvar == 4 || nvar == 5);
#pragma omp parallel
{
        int g, n, ib, g0, g1;
#pragma omp for schedule(static)
        for (ib = 0; ib < nblk; ib++) {
        g0 = ib * POINT_BLK;
        g1 = MIN(g0 + POINT_BLK, ngrids);
        for (n = 0; n < ncounts; n++) {
#pragma GCC ivdep
                for (g = g0; g < g1; g++) {
                        frho[n*Nvg+g] = fsigma[n*Ngrids+g];
                        frho[n*Nvg+1*Ngrids+g] = fsigma[Ncg+n*Ngrids+g] * rho[1*Ngrids+g] * 2;
                        frho[n*Nvg+2*Ngrids+g] = fsigma[Ncg+n*Ngrids+g] * rho[2*Ngrids+g] * 2;
                        frho[n*Nvg+3*Ngrids+g] = fsigma[Ncg+n*Ngrids+g] * rho[3*Ngrids+g] * 2;
                }
        }
        if (nvar > 4) {
                // MGGA
                for (n = 0; n < ncounts; n++) {
#pragma GCC ivdep
                        for (g = g0; g < g1; g++) {
                                frho[n*Nvg+4*Ngrids+g] = fsigma[2*Ncg+n*Ngrids+g];
                        }
                }
        }
        }
}
}

#define frho_at(n, a, x, g)     frho[((n*2+a)*Nvg + x*Ngrids + g)]
#define fsigma_at(x, n, g)      fsigma[(x*Ncg + n*Ngrids + g)]
#define rho_at(a, x, g)         rho[(a*Nvg + x*Ngrids + g)]
// spin1 shape: frho[...,2,nvar,ngrids], fsigma[:,...,ngrids], rho[2,nvar,ngrids]
void VXCunfold_sigma_spin1(double *frho, double *fsigma, double *rho,
                           int ncounts, int nvar, int ngrids)
{
        size_t Ngrids = ngrids;
        size_t Ncg = ncounts * Ngrids;
        size_t Nvg = nvar * Ngrids;
        int nblk = (ngrids + POINT_BLK - 1) / POINT_BLK;
        assert(nvar == 4 || nvar == 5);
#pragma omp parallel
{
        int g, n, ib, g0, g1;
#pragma omp for schedule(static)
        for (ib = 0; ib < nblk; ib++) {
        g0 = ib * POINT_BLK;
        g1 = MIN(g0 + POINT_BLK, ngrids);
        for (n = 0; n < ncounts; n++) {
#pragma GCC ivdep
                for (g = g0; g < g1; g++) {
                        frho_at(n,0,0,g) = fsigma_at(0,n,g);
                        frho_at(n,1,0,g) = fsigma_at(1,n,g);
                        frho_at(n,0,1,g) = fsigma_at(2,n,g) * rho_at(0,1,g) * 2 + fsigma_at(3,n,g) * rho_at(1,1,g);
                        frho_at(n,1,1,g) = fsigma_at(3,n,g) * rho_at(0,1,g) + 2 * fsigma_at(4,n,g) * rho_at(1,1,g);
                        frho_at(n,0,2,g) = fsigma_at(2,n,g) * rho_at(0,2,g) * 2 + fsigma_at(3,n,g) * rho_at(1,2,g);
                        frho_at(n,1,2,g) = fsigma_at(3,n,g) * rho_at(0,2,g) + 2 * fsigma_at(4,n,g) * rho_at(1,2,g);
                        frho_at(n,0,3,g) = fsigma_at(2,n,g) * rho_at(0,3,g) * 2 + fsigma_at(3,n,g) * rho_at(1,3,g);
                        frho_at(n,1,3,g) = fsigma_at(3,n,g) * rho_at(0,3,g) + 2 * fsigma_at(4,n,g) * rho_at(1,3,g);
                }
        }
        if (nvar > 4) {
                // MGGA
                for (n = 0; n < ncounts; n++) {
#pragma GCC ivdep
                        for (g = g0; g < g1; g++) {
                                frho_at(n,0,4,g) = fsigma_at(5,n,g);
                                frho_at(n,1,4,g) = fsigma_at(6,n,g);
                        }
                }
        }
        }
}
}

/*
 * out[k,g] = src[idx[k],g] for k < nidx: the rows of a [*,ngrids] array picked by idx
 */
void VXCgather_rows(double *out, double *src, int *idx, int nidx, int ngrids)
{
        size_t Ngrids = ngrids;
        int nblk = (ngrids + POINT_BLK - 1) / POINT_BLK;
#pragma omp parallel
{
        double *pout, *psrc;
        int g, k, ib, g0, g1;
#pragma omp for schedule(static)
        for (ib = 0; ib < nblk; ib++) {
                g0 = ib * POINT_BLK;
                g1 = MIN(g0 + POINT_BLK, ngrids);
                for (k = 0; k < nidx; k++) {
                        pout = out + k * Ngrids;
                        psrc = src + idx[k] * Ngrids;
                        for (g = g0; g < g1; g++) {
                                pout[g] = psrc[g];
                        }
                }
        }
}
}

/*
 * out[tgt[k],g] += sub[src[k],g] for k = 0, 1, ..., nops-1 in this order, at every point g.
 * Each element of out receives its additions in the order of k.
 */
void VXCadd_rows_indexed(double *out, double *sub, int *tgt, int *src,
                         int nops, int ngrids)
{
        size_t Ngrids = ngrids;
        int nblk = (ngrids + POINT_BLK - 1) / POINT_BLK;
#pragma omp parallel
{
        double *pout, *psub;
        int g, k, ib, g0, g1;
#pragma omp for schedule(static)
        for (ib = 0; ib < nblk; ib++) {
                g0 = ib * POINT_BLK;
                g1 = MIN(g0 + POINT_BLK, ngrids);
                for (k = 0; k < nops; k++) {
                        pout = out + tgt[k] * Ngrids;
                        psub = sub + src[k] * Ngrids;
#pragma GCC ivdep
                        for (g = g0; g < g1; g++) {
                                pout[g] += psub[g];
                        }
                }
        }
}
}
