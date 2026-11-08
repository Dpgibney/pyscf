/* Copyright 2014-2020 The PySCF Developers. All Rights Reserved.

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
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include "config.h"
#include "gto/grid_ao_drv.h"
#include "np_helper/np_helper.h"
#include "vhf/fblas.h"

#define BOXSIZE         56

typedef struct { double x, y, z; } double3;

int VXCao_empty_blocks(int8_t *empty, uint8_t *non0table, int *shls_slice,
                       int *ao_loc)
{
        if (non0table == NULL || shls_slice == NULL || ao_loc == NULL) {
                return 0;
        }

        const int sh0 = shls_slice[0];
        const int sh1 = shls_slice[1];

        int bas_id;
        int box_id = 0;
        int bound = BOXSIZE;
        int has0 = 0;
        empty[box_id] = 1;
        for (bas_id = sh0; bas_id < sh1; bas_id++) {
                if (ao_loc[bas_id] == bound) {
                        has0 |= empty[box_id];
                        box_id++;
                        bound += BOXSIZE;
                        empty[box_id] = 1;
                }
                empty[box_id] &= !non0table[bas_id];
                if (ao_loc[bas_id+1] > bound) {
                        has0 |= empty[box_id];
                        box_id++;
                        bound += BOXSIZE;
                        empty[box_id] = !non0table[bas_id];
                }
        }
        return has0;
}

static void dot_ao_dm(double *vm, double *ao, double *dm,
                      int nao, int nocc, int ngrids, int bgrids,
                      uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        int nbox = (nao+BOXSIZE-1) / BOXSIZE;
        int8_t empty[nbox];
        int has0 = VXCao_empty_blocks(empty, non0table, shls_slice, ao_loc);

        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D1 = 1;
        double beta = 0;

        if (has0) {
                int box_id, blen, i, j;
                size_t b0;

                for (box_id = 0; box_id < nbox; box_id++) {
                        if (!empty[box_id]) {
                                b0 = box_id * BOXSIZE;
                                blen = MIN(nao-b0, BOXSIZE);
                                dgemm_(&TRANS_N, &TRANS_T, &bgrids, &nocc, &blen,
                                       &D1, ao+b0*ngrids, &ngrids, dm+b0*nocc, &nocc,
                                       &beta, vm, &ngrids);
                                beta = 1.0;
                        }
                }
                if (beta == 0) { // all empty
                        for (i = 0; i < nocc; i++) {
                                for (j = 0; j < bgrids; j++) {
                                        vm[i*ngrids+j] = 0;
                                }
                        }
                }
        } else {
                dgemm_(&TRANS_N, &TRANS_T, &bgrids, &nocc, &nao,
                       &D1, ao, &ngrids, dm, &nocc, &beta, vm, &ngrids);
        }
}


/* vm[nocc,ngrids] = ao[i,ngrids] * dm[i,nocc] */
void VXCdot_ao_dm(double *vm, double *ao, double *dm,
                  int nao, int nocc, int ngrids, int nbas,
                  uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;

#pragma omp parallel
{
        int ip, ib;
#pragma omp for nowait schedule(static)
        for (ib = 0; ib < nblk; ib++) {
                ip = ib * BLKSIZE;
                dot_ao_dm(vm+ip, ao+ip, dm,
                          nao, nocc, ngrids, MIN(ngrids-ip, BLKSIZE),
                          non0table+ib*nbas, shls_slice, ao_loc);
        }
}
}

/*static void make_box_loc(int *box_loc, uint8_t *non0table, int nbas, int ngrids)
{
        if (nbas < 1) {
                return;
        }
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
        int *sizes = malloc(nblk * sizeof(int));
#pragma omp parallel for
        for (int iblk = 0; iblk < nblk; iblk++) {
                sizes[iblk] = 0;
                uint8_t prev_non0 = non0table[iblk * nbas];
        }
}*/

static void sgx_ao_dm(double *vm, double *ao, double *dm,
                      int nao, int nocc, int ngrids, int bgrids,
                      uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D1 = 1;
        const double beta = 1.0;

        int ish, blen;
        size_t i, j;
        size_t b0;

        for (i = 0; i < nocc; i++) {
                for (j = 0; j < bgrids; j++) {
                        vm[i*ngrids+j] = 0;
                }
        }
        for (ish = shls_slice[0]; ish < shls_slice[1]; ish++) {
                if (non0table[ish]) {
                        b0 = ao_loc[ish];
                        blen = ao_loc[ish + 1] - b0;
                        dgemm_(&TRANS_N, &TRANS_T, &bgrids, &nocc, &blen,
                               &D1, ao+b0*ngrids, &ngrids, dm+b0*nocc, &nocc,
                               &beta, vm, &ngrids);
                }
        }
}

/* vm[nocc,ngrids] = ao[i,ngrids] * dm[i,nocc] */
void VXCsgx_ao_dm(double *vm, double *ao, double *dm,
                  int nao, int nocc, int ngrids, int nbas,
                  uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        const int blksize = BLKSIZE;
        const int nblk = (ngrids+blksize-1) / blksize;

#pragma omp parallel
{
        int ip, ib;
#pragma omp for nowait schedule(static)
        for (ib = 0; ib < nblk; ib++) {
                ip = ib * blksize;
                sgx_ao_dm(vm+ip, ao+ip, dm,
                          nao, nocc, ngrids, MIN(ngrids-ip, blksize),
                          non0table+ib*nbas, shls_slice, ao_loc);
        }
}
}

/*void VXCsgx_ao_dm_v2(double *vm, double *ao, double *dm,
                  int nao, int nocc, int ngrids, int nbas,
                  uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        size_t Nao = nao;
        size_t Ngrids = ngrids;
        NPdset0(vm, Nao * ngrids);
        const char TRANS_N = 'N';
        const double D1 = 1;
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;

#pragma omp parallel
{
        int shlsize, ish;
        size_t j0, j1;
        int ni;
        NPomp_split(&j0, &j1, nao);
        int ip, ib;
        int bgrids;
        const int nj = j1 - j0;
        for (ib = 0; ib < nblk; ib++) {
                ip = ib * BLKSIZE;
                bgrids = MIN(ip + BLKSIZE, ngrids) - ip;
                for (ish = shls_slice[0]; ish < shls_slice[1]; ish++) {
                        if (non0table[ib * nbas + ish]) {
                                ni = ao_loc[ish + 1] - ao_loc[ish];
                                dgemm_(&TRANS_N, &TRANS_N, &bgrids, &nj, &ni, &D1,
                                       ao+ao_loc[ish]*Ngrids+ip, &ngrids,
                                       dm+j0*Nao+ao_loc[ish], &ngrids,
                                       &D1, vm+j0*Ngrids+ip, &ngrids);
                        }
                }
        }
}
}*/

void VXCsgx_ao_ao(double *vv, double *ao1, double *ao2,
                  int nao, int ngrids, int nbas,
                  uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        size_t Nao = nao;
        size_t Ngrids = ngrids;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D1 = 1;
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
        NPdset0(vv, Nao * Nao);

#pragma omp parallel
{
        int ish;
        size_t j0, j1;
        int ni;
        NPomp_split(&j0, &j1, nao);
        int ip, ib;
        int bgrids;
        const int nj = j1 - j0;
        for (ib = 0; ib < nblk; ib++) {
                ip = ib * BLKSIZE;
                bgrids = MIN(ip + BLKSIZE, ngrids) - ip;
                for (ish = shls_slice[0]; ish < shls_slice[1]; ish++) {
                        if (non0table[ib * nbas + ish]) {
                                ni = ao_loc[ish + 1] - ao_loc[ish];
                                dgemm_(&TRANS_T, &TRANS_N, &ni, &nj, &bgrids, &D1,
                                       ao1+ao_loc[ish]*Ngrids+ip, &ngrids,
                                       ao2+j0*Ngrids+ip, &ngrids,
                                       &D1, vv+j0*nao+ao_loc[ish], &nao);
                        }
                }
        }
}
}


/* vv[n,m] = ao1[n,ngrids] * ao2[m,ngrids] */
static void dot_ao_ao(double *vv, double *ao1, double *ao2,
                      int nao, int ngrids, int bgrids, int hermi,
                      uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        int nbox = (nao+BOXSIZE-1) / BOXSIZE;
        int8_t empty[nbox];
        int has0 = VXCao_empty_blocks(empty, non0table, shls_slice, ao_loc);

        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D1 = 1;
        if (has0) {
                int ib, jb, leni, lenj;
                int j1 = nbox;
                size_t b0i, b0j;

                for (ib = 0; ib < nbox; ib++) {
                if (!empty[ib]) {
                        b0i = ib * BOXSIZE;
                        leni = MIN(nao-b0i, BOXSIZE);
                        if (hermi) {
                                j1 = ib + 1;
                        }
                        for (jb = 0; jb < j1; jb++) {
                        if (!empty[jb]) {
                                b0j = jb * BOXSIZE;
                                lenj = MIN(nao-b0j, BOXSIZE);
                                dgemm_(&TRANS_T, &TRANS_N, &lenj, &leni, &bgrids, &D1,
                                       ao2+b0j*ngrids, &ngrids, ao1+b0i*ngrids, &ngrids,
                                       &D1, vv+b0i*nao+b0j, &nao);
                        } }
                } }
        } else {
                dgemm_(&TRANS_T, &TRANS_N, &nao, &nao, &bgrids,
                       &D1, ao2, &ngrids, ao1, &ngrids, &D1, vv, &nao);
        }
}


/* vv[nao,nao] = ao1[i,nao] * ao2[i,nao] */
void VXCdot_ao_ao(double *vv, double *ao1, double *ao2,
                  int nao, int ngrids, int nbas, int hermi,
                  uint8_t *non0table, int *shls_slice, int *ao_loc)
{
        const int nblk = (ngrids+BLKSIZE-1) / BLKSIZE;
        size_t Nao = nao;
        NPdset0(vv, Nao * Nao);

#pragma omp parallel
{
        int ip, ib;
        double *v_priv = calloc(Nao*Nao+2, sizeof(double));
#pragma omp for nowait schedule(static)
        for (ib = 0; ib < nblk; ib++) {
                ip = ib * BLKSIZE;
                dot_ao_ao(v_priv, ao1+ip, ao2+ip,
                          nao, ngrids, MIN(ngrids-ip, BLKSIZE), hermi,
                          non0table+ib*nbas, shls_slice, ao_loc);
        }
#pragma omp critical
        {
                for (ip = 0; ip < Nao*Nao; ip++) {
                        vv[ip] += v_priv[ip];
                }
        }
        free(v_priv);
}
        if (hermi != 0) {
                NPdsymm_triu(nao, vv, hermi);
        }
}

#define HERMI_DOT_CHUNK 256
/*
 * vv[nao,nao] += sum_c sum_g wv[g] ao[c,i,g] ao[c,j,g] for real ao[ncomp,nao,ngrids].
 * The product is symmetric, so it is evaluated with dsyrk, which costs half of
 * the equivalent dgemm. dsyrk needs the form A^T A, so the grid points are split
 * by the sign of wv and the AO values scaled by sqrt(|wv|): vv += P^T P - N^T N.
 */
void VXCdot_ao_wv_ao_hermi(double *vv, double *ao, double *wv,
                           int ncomp, int nao, int ngrids)
{
        const size_t Nao = nao;
        const size_t Ngrids = ngrids;
        const size_t nn = Nao * Nao;
        const int nchunk = (ngrids + HERMI_DOT_CHUNK - 1) / HERMI_DOT_CHUNK;
        double **vprivs = malloc(sizeof(double *) * omp_get_max_threads());
#pragma omp parallel
{
        const char UPLO = 'U';  // upper triangle in Fortran order: vpriv[i*nao+j] for j <= i
        const char TRANS_T = 'T';
        const double D1 = 1;
        const double DM1 = -1;
        double *vpriv = calloc(nn, sizeof(double));
        // Published here so the barrier at the end of the omp for below makes it
        // visible before NPomp_dsum_reduce_inplace reads vprivs[0].
        vprivs[omp_get_thread_num()] = vpriv;
        double *buf = malloc(sizeof(double) * Nao * ncomp * HERMI_DOT_CHUNK);
        double scale[HERMI_DOT_CHUNK];
        int idx[HERMI_DOT_CHUNK];
        int ichunk, ip0, ip1, ip, n, ns, k, sign, i, j, c;
        double w;
        double *pbuf, *pao;
// static: each thread sums a fixed set of chunks, so vv is reproducible for a given
        // number of threads
#pragma omp for schedule(static)
        for (ichunk = 0; ichunk < nchunk; ichunk++) {
                ip0 = ichunk * HERMI_DOT_CHUNK;
                ip1 = MIN(ip0 + HERMI_DOT_CHUNK, ngrids);
                for (sign = 1; sign >= -1; sign -= 2) {
                        ns = 0;
                        for (ip = ip0; ip < ip1; ip++) {
                                w = wv[ip] * sign;
                                // NaN goes to the positive set so that it propagates
                                if (w > 0 || (sign > 0 && w != w)) {
                                        idx[ns] = ip;
                                        scale[ns] = sqrt(w);
                                        ns++;
                                }
                        }
                        if (ns == 0) {
                                continue;
                        }
                        // buf[i,c*ns+n] = ao[c,i,idx[n]] * scale[n], i.e. A[k,nao] in Fortran order
                        k = ns * ncomp;
                        for (i = 0; i < nao; i++) {
                                pbuf = buf + i * (size_t)k;
                                for (c = 0; c < ncomp; c++) {
                                        pao = ao + (c * Nao + i) * Ngrids;
                                        for (n = 0; n < ns; n++) {
                                                pbuf[c*ns+n] = pao[idx[n]] * scale[n];
                                        }
                                }
                        }
                        dsyrk_(&UPLO, &TRANS_T, &nao, &k, (sign > 0 ? &D1 : &DM1),
                               buf, &k, &D1, vpriv, &nao);
                }
        }
        NPomp_dsum_reduce_inplace(vprivs, nn);
        double *vsum = vprivs[0];
#pragma omp for schedule(static)
        for (i = 0; i < nao; i++) {
                for (j = 0; j <= i; j++) {
                        vv[i*Nao+j] += vsum[i*Nao+j];
                }
                for (; j < nao; j++) {
                        vv[i*Nao+j] += vsum[j*Nao+i];
                }
        }
        free(vpriv);
        free(buf);
}
        free(vprivs);
}

// 'nip,np->ip'
void VXC_dscale_ao(double *aow, double *ao, double *wv,
                   int comp, int nao, int ngrids)
{
#pragma omp parallel
{
        size_t Ngrids = ngrids;
        size_t ao_size = nao * Ngrids;
        int i, j, ic;
        double *pao = ao;
#pragma omp for schedule(static)
        for (i = 0; i < nao; i++) {
                pao = ao + i * Ngrids;
                for (j = 0; j < Ngrids; j++) {
                        aow[i*Ngrids+j] = pao[j] * wv[j];
                }
                for (ic = 1; ic < comp; ic++) {
                for (j = 0; j < Ngrids; j++) {
                        aow[i*Ngrids+j] += pao[ic*ao_size+j] * wv[ic*Ngrids+j];
                } }
        }
}
}

// 'ip,ip->p'
void VXC_dcontract_rho(double *rho, double *bra, double *ket,
                       int nao, int ngrids)
{
#pragma omp parallel
{
        size_t Ngrids = ngrids;
        int nthread = omp_get_num_threads();
        int blksize = MAX((Ngrids+nthread-1) / nthread, 1);
        int ib, b0, b1, i, j;
#pragma omp for
        for (ib = 0; ib < nthread; ib++) {
                b0 = ib * blksize;
                b1 = MIN(b0 + blksize, ngrids);
                for (j = b0; j < b1; j++) {
                        rho[j] = bra[j] * ket[j];
                }
                for (i = 1; i < nao; i++) {
                for (j = b0; j < b1; j++) {
                        rho[j] += bra[i*Ngrids+j] * ket[i*Ngrids+j];
                } }
        }
}
}

#define BATCH_CHUNK 128
/*
 * rho[k,c,g] = sum_p ao[c,p,g] * bra[k,p,g] for real ao[ncomp,nao,ngrids] and
 * bra[nset,nao,ngrids]. Each AO value is read once for all nset sets, rather
 * than once per (set, component) as with VXC_dcontract_rho.
 */
void VXC_dcontract_rho_batch(double *rho, double *ao, double *bra,
                             int ncomp, int nset, int nao, int ngrids)
{
        const size_t Ngrids = ngrids;
        const size_t ao_size = (size_t)nao * ngrids;
        const int nchunk = (ngrids + BATCH_CHUNK - 1) / BATCH_CHUNK;
#pragma omp parallel
{
        int ichunk, g0, ng, k, c, p, g;
        double *prho, *pao, *pbra;
#pragma omp for schedule(static)
        for (ichunk = 0; ichunk < nchunk; ichunk++) {
                g0 = ichunk * BATCH_CHUNK;
                ng = MIN(ngrids - g0, BATCH_CHUNK);
                for (k = 0; k < nset * ncomp; k++) {
                        prho = rho + k * Ngrids + g0;
                        for (g = 0; g < ng; g++) {
                                prho[g] = 0;
                        }
                }
                for (p = 0; p < nao; p++) {
                for (k = 0; k < nset; k++) {
                        pbra = bra + k * ao_size + p * Ngrids + g0;
                        for (c = 0; c < ncomp; c++) {
                                pao = ao + c * ao_size + p * Ngrids + g0;
                                prho = rho + (k * ncomp + c) * Ngrids + g0;
                                for (g = 0; g < ng; g++) {
                                        prho[g] += pao[g] * pbra[g];
                                }
                        }
                } }
        }
}
}

/*
 * aow[k,p,g] = sum_c wv[k,c,g] * ao[c,p,g] for real ao[ncomp,nao,ngrids] and
 * wv[nset,ncomp,ngrids]. Each AO value is read once for all nset sets.
 */
void VXC_dscale_ao_batch(double *aow, double *ao, double *wv,
                         int ncomp, int nset, int nao, int ngrids)
{
        const size_t Ngrids = ngrids;
        const size_t ao_size = (size_t)nao * ngrids;
        const int nchunk = (ngrids + BATCH_CHUNK - 1) / BATCH_CHUNK;
#pragma omp parallel
{
        int ichunk, g0, ng, k, c, p, g;
        double *paow, *pao, *pwv;
#pragma omp for schedule(static)
        for (ichunk = 0; ichunk < nchunk; ichunk++) {
                g0 = ichunk * BATCH_CHUNK;
                ng = MIN(ngrids - g0, BATCH_CHUNK);
                for (p = 0; p < nao; p++) {
                for (k = 0; k < nset; k++) {
                        paow = aow + k * ao_size + p * Ngrids + g0;
                        pwv = wv + k * ncomp * Ngrids + g0;
                        pao = ao + p * Ngrids + g0;
                        for (g = 0; g < ng; g++) {
                                paow[g] = pwv[g] * pao[g];
                        }
                        for (c = 1; c < ncomp; c++) {
                                pwv = wv + (k * ncomp + c) * Ngrids + g0;
                                pao = ao + c * ao_size + p * Ngrids + g0;
                                for (g = 0; g < ng; g++) {
                                        paow[g] += pwv[g] * pao[g];
                                }
                        }
                } }
        }
}
}

/*
 * VV10 (Vydrov and Van Voorhis 2010): for each outer point i, sums over the inner points j of
 *     T_ij = RpW_j / (g gp gt),   g = R2 W0_i + K_i,  gp = R2 W0p_j + Kp_j,  gt = g + gp,  R2 = |r_i - r_j|^2:
 *     F_i = -1.5 sum_j T_ij,   U_i = sum_j T_ij (1/g + 1/gt),   W_i = sum_j T_ij (1/g + 1/gt) R2.
 * One division per pair: with D = 1/(g gp gt), T_ij = RpW_j D and 1/g + 1/gt = (g + gt) gp D.
 * The inner points are copied into padded arrays per coordinate, and each sum runs in VV10_LANES partial sums,
 * which the compiler can vectorize (it does not reorder a single floating-point sum); the partial sums are
 * added in a fixed order. Padding points have RpW = 0, W0 = 0 and K = 1, so they add exact zeros. The results
 * differ from a division per term in rounding only.
 */
#define VV10_LANES      8
#define VV10_BLK        128

typedef struct {
        double *x, *y, *z, *w0, *k, *rw;
        int n;  // padded length
} VV10Points;

static VV10Points vv10_points(double *coords, double *W0, double *K, double *RpW, int n, int pad_to)
{
        VV10Points p;
        int npad = (n + pad_to - 1) / pad_to * pad_to;
        double *buf = malloc(sizeof(double) * npad * 6 + 1);
        int i;
        p.n = npad;
        p.x = buf;
        p.y = p.x + npad;
        p.z = p.y + npad;
        p.w0 = p.z + npad;
        p.k = p.w0 + npad;
        p.rw = p.k + npad;
        for (i = 0; i < n; i++) {
                p.x[i] = coords[i*3+0];
                p.y[i] = coords[i*3+1];
                p.z[i] = coords[i*3+2];
                p.w0[i] = W0[i];
                p.k[i] = K[i];
                p.rw[i] = RpW == NULL ? 0 : RpW[i];
        }
        for (; i < npad; i++) {
                p.x[i] = 0;
                p.y[i] = 0;
                p.z[i] = 0;
                p.w0[i] = 0;
                p.k[i] = 1;
                p.rw[i] = 0;
        }
        return p;
}

void VXC_vv10nlc(double *Fvec, double *Uvec, double *Wvec,
                 double *vvcoords, double *coords,
                 double *W0p, double *W0, double *K, double *Kp, double *RpW,
                 int vvngrids, int ngrids)
{
        VV10Points q = vv10_points(vvcoords, W0p, Kp, RpW, vvngrids, VV10_LANES);
        double *restrict xj = q.x;
        double *restrict yj = q.y;
        double *restrict zj = q.z;
        double *restrict w0j = q.w0;
        double *restrict kj = q.k;
        double *restrict rwj = q.rw;
        const int nj = q.n;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < ngrids; i++) {
                const double xi = coords[i*3+0];
                const double yi = coords[i*3+1];
                const double zi = coords[i*3+2];
                const double w0i = W0[i];
                const double ki = K[i];
                double f[VV10_LANES] = {0};
                double u[VV10_LANES] = {0};
                double w[VV10_LANES] = {0};
                for (int j0 = 0; j0 < nj; j0 += VV10_LANES) {
#pragma GCC ivdep
                        for (int l = 0; l < VV10_LANES; l++) {
                                int j = j0 + l;
                                double dx = xj[j] - xi;
                                double dy = yj[j] - yi;
                                double dz = zj[j] - zi;
                                double r2 = dx*dx + dy*dy + dz*dz;
                                double gp = r2 * w0j[j] + kj[j];
                                double g = r2 * w0i + ki;
                                double gt = g + gp;
                                double d = 1. / (g * gp * gt);
                                double t = rwj[j] * d;
                                f[l] += t;
                                t *= (g + gt) * gp * d;
                                u[l] += t;
                                w[l] += t * r2;
                        }
                }
                double F = 0, U = 0, W = 0;
                for (int l = 0; l < VV10_LANES; l++) {
                        F += f[l];
                        U += u[l];
                        W += w[l];
                }
                Fvec[i] = F * -1.5;
                Uvec[i] = U;
                Wvec[i] = W;
        }
        free(q.x);
}

/* the pairs of blocks a and b (a != b) of one point set, each pair once: T_ij and T_ji share D, so block a
 * gets the sums over j in b and block b those over i in a */
static void vv10_tile(VV10Points *p, double *restrict f, double *restrict u, double *restrict w, int a, int b)
{
        const int ia = a * VV10_BLK;
        const int jb = b * VV10_BLK;
        double *restrict xb = p->x + jb;
        double *restrict yb = p->y + jb;
        double *restrict zb = p->z + jb;
        double *restrict w0b = p->w0 + jb;
        double *restrict kb = p->k + jb;
        double *restrict rwb = p->rw + jb;
        double *restrict fb = f + jb;
        double *restrict ub = u + jb;
        double *restrict wb = w + jb;
        for (int i = ia; i < ia + VV10_BLK; i++) {
                const double xi = p->x[i];
                const double yi = p->y[i];
                const double zi = p->z[i];
                const double w0i = p->w0[i];
                const double ki = p->k[i];
                const double rwi = p->rw[i];
                double fi[VV10_LANES] = {0};
                double ui[VV10_LANES] = {0};
                double wi[VV10_LANES] = {0};
                for (int j0 = 0; j0 < VV10_BLK; j0 += VV10_LANES) {
#pragma GCC ivdep
                        for (int l = 0; l < VV10_LANES; l++) {
                                int j = j0 + l;
                                double dx = xb[j] - xi;
                                double dy = yb[j] - yi;
                                double dz = zb[j] - zi;
                                double r2 = dx*dx + dy*dy + dz*dz;
                                double gi = r2 * w0i + ki;
                                double gj = r2 * w0b[j] + kb[j];
                                double gt = gi + gj;
                                double d = 1. / (gi * gj * gt);
                                double ti = rwb[j] * d;
                                double tj = rwi * d;
                                fi[l] += ti;
                                fb[j] += tj;
                                ti *= (gi + gt) * gj * d;
                                tj *= (gj + gt) * gi * d;
                                ui[l] += ti;
                                ub[j] += tj;
                                wi[l] += ti * r2;
                                wb[j] += tj * r2;
                        }
                }
                double F = 0, U = 0, W = 0;
                for (int l = 0; l < VV10_LANES; l++) {
                        F += fi[l];
                        U += ui[l];
                        W += wi[l];
                }
                f[i] += F;
                u[i] += U;
                w[i] += W;
        }
}

/* the pairs within block a, as ordered pairs (the point itself included) */
static void vv10_diag(VV10Points *p, double *f, double *u, double *w, int a)
{
        const int i0 = a * VV10_BLK;
        double *restrict xb = p->x + i0;
        double *restrict yb = p->y + i0;
        double *restrict zb = p->z + i0;
        double *restrict w0b = p->w0 + i0;
        double *restrict kb = p->k + i0;
        double *restrict rwb = p->rw + i0;
        for (int i = i0; i < i0 + VV10_BLK; i++) {
                const double xi = p->x[i];
                const double yi = p->y[i];
                const double zi = p->z[i];
                const double w0i = p->w0[i];
                const double ki = p->k[i];
                double fi[VV10_LANES] = {0};
                double ui[VV10_LANES] = {0};
                double wi[VV10_LANES] = {0};
                for (int j0 = 0; j0 < VV10_BLK; j0 += VV10_LANES) {
#pragma GCC ivdep
                        for (int l = 0; l < VV10_LANES; l++) {
                                int j = j0 + l;
                                double dx = xb[j] - xi;
                                double dy = yb[j] - yi;
                                double dz = zb[j] - zi;
                                double r2 = dx*dx + dy*dy + dz*dz;
                                double gp = r2 * w0b[j] + kb[j];
                                double g = r2 * w0i + ki;
                                double gt = g + gp;
                                double d = 1. / (g * gp * gt);
                                double t = rwb[j] * d;
                                fi[l] += t;
                                t *= (g + gt) * gp * d;
                                ui[l] += t;
                                wi[l] += t * r2;
                        }
                }
                double F = 0, U = 0, W = 0;
                for (int l = 0; l < VV10_LANES; l++) {
                        F += fi[l];
                        U += ui[l];
                        W += wi[l];
                }
                f[i] += F;
                u[i] += U;
                w[i] += W;
        }
}

/*
 * VXC_vv10nlc for inner points that are the outer points (the SCF's and the gradient's NLC grid): each pair
 * of points once. The points are cut into blocks of VV10_BLK; the pairs of blocks are visited in the rounds of
 * a round-robin tournament, in which every block appears in one pair, so the threads of a round update
 * disjoint blocks, and every block's sums are added in the same order for any number of threads.
 */
void VXC_vv10nlc_sym(double *Fvec, double *Uvec, double *Wvec, double *coords,
                     double *W0, double *K, double *RpW, int ngrids)
{
        VV10Points p = vv10_points(coords, W0, K, RpW, ngrids, VV10_BLK);
        const int nb = p.n / VV10_BLK;
        const int m = nb + (nb & 1);  // with a dummy block for odd nb
        double *f = calloc(p.n * 3 + 1, sizeof(double));
        double *u = f + p.n;
        double *w = u + p.n;
#pragma omp parallel
{
#pragma omp for schedule(static)
        for (int a = 0; a < nb; a++) {
                vv10_diag(&p, f, u, w, a);
        }
        for (int r = 0; r < m - 1; r++) {
#pragma omp for schedule(static)
                for (int k = 0; k < m / 2; k++) {
                        int a, b;
                        if (k == 0) {
                                a = m - 1;
                                b = r;
                        } else {
                                a = (r + k) % (m - 1);
                                b = (r - k + m - 1) % (m - 1);
                        }
                        if (a < nb && b < nb) {
                                vv10_tile(&p, f, u, w, a, b);
                        }
                }
        }
}
        for (int i = 0; i < ngrids; i++) {
                Fvec[i] = f[i] * -1.5;
                Uvec[i] = u[i];
                Wvec[i] = w[i];
        }
        free(f);
        free(p.x);
}

/*
 * The VV10 gradient term (Vydrov and Van Voorhis 2010, eq. 25-26), with one division per pair as in
 * VXC_vv10nlc: Q = T (W0_i/g + W0p_j/gp + (W0_i + W0p_j)/gt) = RpW_j D^2 (W0_i gp gt + W0p_j g gt + (W0_i + W0p_j) g gp).
 */
void VXC_vv10nlc_grad(double *Fvec, double *vvcoords, double *coords,
                      double *W0p, double *W0, double *K, double *Kp, double *RpW,
                      int vvngrids, int ngrids)
{
        VV10Points q = vv10_points(vvcoords, W0p, Kp, RpW, vvngrids, VV10_LANES);
        double *restrict xj = q.x;
        double *restrict yj = q.y;
        double *restrict zj = q.z;
        double *restrict w0j = q.w0;
        double *restrict kj = q.k;
        double *restrict rwj = q.rw;
        const int nj = q.n;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < ngrids; i++) {
                const double xi = coords[i*3+0];
                const double yi = coords[i*3+1];
                const double zi = coords[i*3+2];
                const double w0i = W0[i];
                const double ki = K[i];
                double fx[VV10_LANES] = {0};
                double fy[VV10_LANES] = {0};
                double fz[VV10_LANES] = {0};
                for (int j0 = 0; j0 < nj; j0 += VV10_LANES) {
#pragma GCC ivdep
                        for (int l = 0; l < VV10_LANES; l++) {
                                int j = j0 + l;
                                double dx = xj[j] - xi;
                                double dy = yj[j] - yi;
                                double dz = zj[j] - zi;
                                double r2 = dx*dx + dy*dy + dz*dz;
                                double gp = r2 * w0j[j] + kj[j];
                                double g = r2 * w0i + ki;
                                double gt = g + gp;
                                double d = 1. / (g * gp * gt);
                                double q = rwj[j] * d * d * (w0i * gp * gt + w0j[j] * g * gt + (w0i + w0j[j]) * g * gp);
                                fx[l] += q * dx;
                                fy[l] += q * dy;
                                fz[l] += q * dz;
                        }
                }
                double FX = 0, FY = 0, FZ = 0;
                for (int l = 0; l < VV10_LANES; l++) {
                        FX += fx[l];
                        FY += fy[l];
                        FZ += fz[l];
                }
                Fvec[i*3+0] = FX * -3;
                Fvec[i*3+1] = FY * -3;
                Fvec[i*3+2] = FZ * -3;
        }
        free(q.x);
}

void VXC_vv10nlc_hessian_eval_UWABCE(double* __restrict__ U, double* __restrict__ W, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, double* __restrict__ E,
                                     const double* __restrict__ grid_coord, const double* __restrict__ grid_weight,
                                     const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                     const int ngrids)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ngrids; i++) {
        const double omega_i = omega[i];
        const double kappa_i = kappa[i];
        const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

        double U_i = 0;
        double W_i = 0;
        double A_i = 0;
        double B_i = 0;
        double C_i = 0;
        double E_i = 0;

        for (int j = 0; j < ngrids; j++) {
            const double omega_j = omega[j];
            const double kappa_j = kappa[j];
            const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
            const double weight_j = grid_weight[j];
            const double rho_j = rho[j];

            const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
            const double g_ij = omega_i * r_ij2 + kappa_i;
            const double g_ji = omega_j * r_ij2 + kappa_j;
            const double g_ij_1 = 1 / g_ij;
            const double g_sum_1 = 1 / (g_ij + g_ji);
            const double Phi_ij = -1.5 / g_ji * g_ij_1 * g_sum_1;

            const double E_ij = weight_j * rho_j * Phi_ij;
            const double U_ij = E_ij * (g_sum_1 + g_ij_1);
            const double W_ij = U_ij * r_ij2;
            const double A_ij = E_ij * (g_sum_1 * g_sum_1 + g_sum_1 * g_ij_1 + g_ij_1 * g_ij_1);
            const double B_ij = A_ij * r_ij2;
            const double C_ij = B_ij * r_ij2;

            U_i += U_ij;
            W_i += W_ij;
            A_i += A_ij;
            B_i += B_ij;
            C_i += C_ij;
            E_i += E_ij;
        }

        U[i] = -U_i;
        W[i] = -W_i;
        A[i] = 2 * A_i;
        B[i] = 2 * B_i;
        C[i] = 2 * C_i;
        E[i] = E_i;
    }
}

void VXC_vv10nlc_hessian_eval_omega_derivative(double* __restrict__ domega_drho, double* __restrict__ domega_dgamma,
                                               double* __restrict__ d2omega_drho2, double* __restrict__ d2omega_dgamma2, double* __restrict__ d2omega_drho_dgamma,
                                               const double* __restrict__ rho, const double* __restrict__ gamma, const double C_factor,
                                               const int ngrids)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ngrids; i++) {
        const double rho_i = rho[i];
        const double gamma_i = gamma[i];

        const double rho_1 = 1 / rho_i;
        const double rho_2 = rho_1 * rho_1;
        const double rho_3 = rho_1 * rho_2;
        const double rho_4 = rho_2 * rho_2;
        const double rho_5 = rho_1 * rho_4;
        const double gamma2 = gamma_i * gamma_i;
        const double four_pi_over_three = 4.0 / 3.0 * M_PI;
        const double omega2 = C_factor * gamma2 * rho_4 + four_pi_over_three * rho_i;
        const double omega = sqrt(omega2);
        const double omega_1 = 1 / omega;

        domega_drho[i] = 0.5 * (four_pi_over_three - 4 * C_factor * gamma2 * rho_5) * omega_1;
        domega_dgamma[i] = C_factor * gamma_i * rho_4 * omega_1;

        const double omega_3 = omega_1 / omega2;
        d2omega_drho2[i] = (-0.25 * four_pi_over_three * four_pi_over_three
                            + 12 * four_pi_over_three * C_factor * gamma2 * rho_5
                            + 6 * C_factor * C_factor * gamma2 * gamma2 * rho_5 * rho_5) * omega_3;
        d2omega_dgamma2[i] = four_pi_over_three * C_factor * rho_3 * omega_3;
        d2omega_drho_dgamma[i] = -C_factor * gamma_i * (4.5 * four_pi_over_three * rho_4
                                                        + 2 * C_factor * gamma2 * rho_4 * rho_5) * omega_3;
    }
}

/* The pair sums of the VV10 Hessian (eval_f_t, eval_EUW_with_weight1) evaluate each pair of grid points once,
 * in blocks of VV10_BI x VV10_BJ points, and apply a block to all the trial densities (or weight derivatives)
 * of the call in dgemm. Each thread owns VV10_BI rows of the output. */
#define VV10_BI 128
#define VV10_BJ 256

/* f_rho_t[t,i] = sum_j w_j (f_rho_rho_ij rho_t[t,j] + f_rho_gamma_ij gamma_t[t,j]) + the local terms at i, and
 * f_gamma_t[t,i] alike (Liang, Feng, Liu, Head-Gordon, JCP 158, 204109 (2023)), for the ntrial trial densities
 * rho_t, gamma_t [ntrial,ngrids]. The four pair kernels of a block, weighted by w_j, go to K[4][bj][bi]. */
void VXC_vv10nlc_hessian_eval_f_t(double* __restrict__ f_rho_t, double* __restrict__ f_gamma_t,
                                  const double* __restrict__ grid_coord, const double* __restrict__ grid_weight,
                                  const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                  const double* __restrict__ U, const double* __restrict__ W, const double* __restrict__ A, const double* __restrict__ B, const double* __restrict__ C,
                                  const double* __restrict__ domega_drho, const double* __restrict__ domega_dgamma, const double* __restrict__ dkappa_drho,
                                  const double* __restrict__ d2omega_drho2, const double* __restrict__ d2omega_dgamma2, const double* __restrict__ d2omega_drho_dgamma, const double* __restrict__ d2kappa_drho2,
                                  const double* __restrict__ rho_t, const double* __restrict__ gamma_t,
                                  const int ngrids, const int ntrial)
{
    if (ngrids <= 0 || ntrial <= 0) {
        return;
    }
    const char TRANS_N = 'N';
    const double D1 = 1;
    const int ld = ngrids;
    const int nt = ntrial;
    const size_t nblk = VV10_BI * VV10_BJ;

#pragma omp parallel
{
    double *K = malloc(sizeof(double) * (4 * nblk + 9 * VV10_BI));
    double *Krr = K;
    double *Krg = Krr + nblk;
    double *Kgr = Krg + nblk;
    double *Kgg = Kgr + nblk;
    double *xi = Kgg + nblk;
    double *yi = xi + VV10_BI;
    double *zi = yi + VV10_BI;
    double *omega_i = zi + VV10_BI;
    double *kappa_i = omega_i + VV10_BI;
    double *rho_i = kappa_i + VV10_BI;
    double *domega_drho_i = rho_i + VV10_BI;
    double *dkappa_drho_i = domega_drho_i + VV10_BI;
    double *rho_domega_dgamma_i = dkappa_drho_i + VV10_BI;

#pragma omp for schedule(dynamic, 1)
    for (int i0 = 0; i0 < ngrids; i0 += VV10_BI) {
        const int bi = MIN(VV10_BI, ngrids - i0);
        for (int ii = 0; ii < bi; ii++) {
            const int i = i0 + ii;
            xi[ii] = grid_coord[i * 3 + 0];
            yi[ii] = grid_coord[i * 3 + 1];
            zi[ii] = grid_coord[i * 3 + 2];
            omega_i[ii] = omega[i];
            kappa_i[ii] = kappa[i];
            rho_i[ii] = rho[i];
            domega_drho_i[ii] = domega_drho[i];
            dkappa_drho_i[ii] = dkappa_drho[i];
            rho_domega_dgamma_i[ii] = rho[i] * domega_dgamma[i];
        }
        for (int t = 0; t < nt; t++) {
            for (int ii = 0; ii < bi; ii++) {
                f_rho_t  [(size_t)t * ld + i0 + ii] = 0;
                f_gamma_t[(size_t)t * ld + i0 + ii] = 0;
            }
        }

        for (int j0 = 0; j0 < ngrids; j0 += VV10_BJ) {
            const int bj = MIN(VV10_BJ, ngrids - j0);
            for (int jj = 0; jj < bj; jj++) {
                const int j = j0 + jj;
                const double xj = grid_coord[j * 3 + 0];
                const double yj = grid_coord[j * 3 + 1];
                const double zj = grid_coord[j * 3 + 2];
                const double omega_j = omega[j];
                const double kappa_j = kappa[j];
                const double rho_j = rho[j];
                const double domega_drho_j = domega_drho[j];
                const double dkappa_drho_j = dkappa_drho[j];
                const double rho_domega_dgamma_j = rho_j * domega_dgamma[j];
                const double weight_j = grid_weight[j];
                double *krr = Krr + jj * bi;
                double *krg = Krg + jj * bi;
                double *kgr = Kgr + jj * bi;
                double *kgg = Kgg + jj * bi;
#pragma omp simd
                for (int ii = 0; ii < bi; ii++) {
                    const double dx = xi[ii] - xj;
                    const double dy = yi[ii] - yj;
                    const double dz = zi[ii] - zj;
                    const double r_ij2 = dx * dx + dy * dy + dz * dz;
                    const double g_ij = omega_i[ii] * r_ij2 + kappa_i[ii];
                    const double g_ji = omega_j * r_ij2 + kappa_j;
                    const double g_ij_1 = 1 / g_ij;
                    const double g_ji_1 = 1 / g_ji;
                    const double g_sum_1 = 1 / (g_ij + g_ji);
                    const double Phi_ij = -1.5 * g_ij_1 * g_ji_1 * g_sum_1;

                    const double rho_dgdrho_i = rho_i[ii] * (r_ij2 * domega_drho_i[ii] + dkappa_drho_i[ii]);
                    const double rho_dgdrho_j = rho_j * (r_ij2 * domega_drho_j + dkappa_drho_j);
                    const double d2Phi_dgij_dgji_over_Phi = 2 * (g_sum_1 * g_sum_1 + g_ij_1 * g_ji_1);
                    const double wPhi = weight_j * Phi_ij;

                    krr[ii] = wPhi * (rho_dgdrho_i * rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi
                                      - rho_dgdrho_i * (g_sum_1 + g_ij_1)
                                      - rho_dgdrho_j * (g_sum_1 + g_ji_1) + 1);
                    kgr[ii] = wPhi * rho_domega_dgamma_i[ii] * r_ij2 * (rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ij_1));
                    krg[ii] = wPhi * rho_domega_dgamma_j * r_ij2 * (rho_dgdrho_i * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ji_1));
                    kgg[ii] = wPhi * rho_domega_dgamma_i[ii] * rho_domega_dgamma_j * r_ij2 * r_ij2 * d2Phi_dgij_dgji_over_Phi;
                }
            }
            // f_rho_t^T[i,t] += K[i,j] rho_t^T[j,t]: the trial arrays as column-major [ngrids,ntrial]
            dgemm_(&TRANS_N, &TRANS_N, &bi, &nt, &bj, &D1, Krr, &bi, rho_t   + j0, &ld, &D1, f_rho_t   + i0, &ld);
            dgemm_(&TRANS_N, &TRANS_N, &bi, &nt, &bj, &D1, Krg, &bi, gamma_t + j0, &ld, &D1, f_rho_t   + i0, &ld);
            dgemm_(&TRANS_N, &TRANS_N, &bi, &nt, &bj, &D1, Kgr, &bi, rho_t   + j0, &ld, &D1, f_gamma_t + i0, &ld);
            dgemm_(&TRANS_N, &TRANS_N, &bi, &nt, &bj, &D1, Kgg, &bi, gamma_t + j0, &ld, &D1, f_gamma_t + i0, &ld);
        }

        for (int ii = 0; ii < bi; ii++) {
            const int i = i0 + ii;
            const double rho_i_ = rho[i];
            const double U_i = U[i];
            const double W_i = W[i];
            const double A_i = A[i];
            const double B_i = B[i];
            const double C_i = C[i];
            const double domega_drho_i_ = domega_drho[i];
            const double domega_dgamma_i = domega_dgamma[i];
            const double dkappa_drho_i_ = dkappa_drho[i];
            const double d2omega_drho2_i = d2omega_drho2[i];
            const double d2omega_dgamma2_i = d2omega_dgamma2[i];
            const double d2omega_drho_dgamma_i = d2omega_drho_dgamma[i];
            const double d2kappa_drho2_i = d2kappa_drho2[i];

            const double f_rho_rho_ii = 2 * domega_drho_i_ * W_i + 2 * dkappa_drho_i_ * U_i
                                        + rho_i_ * (d2omega_drho2_i * W_i + d2kappa_drho2_i * U_i + dkappa_drho_i_ * dkappa_drho_i_ * A_i
                                                + domega_drho_i_ * domega_drho_i_ * C_i + 2 * domega_drho_i_ * dkappa_drho_i_ * B_i);
            const double f_gamma_rho_ii = domega_dgamma_i * W_i + rho_i_ * (d2omega_drho_dgamma_i * W_i
                                                                        + domega_dgamma_i * (dkappa_drho_i_ * B_i + domega_drho_i_ * C_i));
            const double f_rho_gamma_ii = f_gamma_rho_ii;
            const double f_gamma_gamma_ii = rho_i_ * (d2omega_dgamma2_i * W_i + domega_dgamma_i * domega_dgamma_i * C_i);

            for (int t = 0; t < nt; t++) {
                const double rho_t_i   =   rho_t[(size_t)t * ld + i];
                const double gamma_t_i = gamma_t[(size_t)t * ld + i];
                f_rho_t  [(size_t)t * ld + i] += (  f_rho_rho_ii * rho_t_i +   f_rho_gamma_ii * gamma_t_i);
                f_gamma_t[(size_t)t * ld + i] += (f_gamma_rho_ii * rho_t_i + f_gamma_gamma_ii * gamma_t_i);
            }
        }
    }
    free(K);
}
}

/* The grid response of E, U and W: for each point i and atom B, the sum over the points j of the pairs with
 * one point on B. One pass over j per i: each pair adds to its atoms' sums (acc[atom of j]); for the atom of i
 * the sum is over all points outside it. */
void VXC_vv10nlc_hessian_eval_EUW_grid_response(double* __restrict__ Egr, double* __restrict__ Ugr, double* __restrict__ Wgr,
                                                const double* __restrict__ grid_coord, const double* __restrict__ grid_weight,
                                                const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                                const int* __restrict__ grid_associated_atom,
                                                const int ngrids, const int natoms)
{
    if (ngrids <= 0 || natoms <= 0) {
        return;
    }
    const size_t ng = ngrids;
#pragma omp parallel
{
    double *acc = malloc(sizeof(double) * 9 * natoms);
#pragma omp for schedule(dynamic, 64)
    for (int i = 0; i < ngrids; i++) {
        const int i_associated_atom = grid_associated_atom[i];
        if (i_associated_atom < 0) {
            for (int B_atom = 0; B_atom < natoms; B_atom++) {
                for (int x = 0; x < 3; x++) {
                    Egr[(B_atom * 3 + x) * ng + i] = 0;
                    Ugr[(B_atom * 3 + x) * ng + i] = 0;
                    Wgr[(B_atom * 3 + x) * ng + i] = 0;
                }
            }
            continue;
        }
        for (int k = 0; k < 9 * natoms; k++) {
            acc[k] = 0;
        }

        const double omega_i = omega[i];
        const double kappa_i = kappa[i];
        const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

        for (int j = 0; j < ngrids; j++) {
            const int j_associated_atom = grid_associated_atom[j];
            if (j_associated_atom < 0 || j_associated_atom == i_associated_atom)
                continue;

            const double omega_j = omega[j];
            const double kappa_j = kappa[j];
            const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
            const double weight_j = grid_weight[j];
            const double rho_j = rho[j];

            const double3 r_ji = { r_j.x - r_i.x, r_j.y - r_i.y, r_j.z - r_i.z };
            const double r_ij2 = r_ji.x * r_ji.x + r_ji.y * r_ji.y + r_ji.z * r_ji.z;
            const double g_ij = omega_i * r_ij2 + kappa_i;
            const double g_ji = omega_j * r_ij2 + kappa_j;
            const double g_ij_1 = 1 / g_ij;
            const double g_ji_1 = 1 / g_ji;
            const double g_sum_1 = 1 / (g_ij + g_ji);
            const double Phi_ij = -1.5 * g_ij_1 * g_ji_1 * g_sum_1;

            const double E_ij = weight_j * rho_j * Phi_ij;
            const double dPhi_drj_over_Phi = omega_i * g_ij_1 + omega_j * g_ji_1 + (omega_i + omega_j) * g_sum_1;
            const double d2Phi_dgij_drj_over_Phi = omega_i * g_ij_1 * g_ij_1 + (omega_i + omega_j) * g_sum_1 * g_sum_1;
            const double dPhi_dgij_over_Phi = g_sum_1 + g_ij_1;

            const double Egr_ij = E_ij * dPhi_drj_over_Phi;
            const double Ugr_ij = E_ij * (dPhi_drj_over_Phi * dPhi_dgij_over_Phi + d2Phi_dgij_drj_over_Phi);
            const double Wgr_ij = E_ij * (r_ij2 * (dPhi_drj_over_Phi * dPhi_dgij_over_Phi + d2Phi_dgij_drj_over_Phi) - dPhi_dgij_over_Phi);

            double *a = acc + j_associated_atom * 9;
            a[0] += Egr_ij * r_ji.x;
            a[1] += Egr_ij * r_ji.y;
            a[2] += Egr_ij * r_ji.z;
            a[3] += Ugr_ij * r_ji.x;
            a[4] += Ugr_ij * r_ji.y;
            a[5] += Ugr_ij * r_ji.z;
            a[6] += Wgr_ij * r_ji.x;
            a[7] += Wgr_ij * r_ji.y;
            a[8] += Wgr_ij * r_ji.z;
        }

        // acc[i_associated_atom] stays zero: its entries become minus the sum over the other atoms
        double *own = acc + i_associated_atom * 9;
        for (int B_atom = 0; B_atom < natoms; B_atom++) {
            if (B_atom != i_associated_atom) {
                for (int k = 0; k < 9; k++) {
                    own[k] -= acc[B_atom * 9 + k];
                }
            }
        }
        for (int B_atom = 0; B_atom < natoms; B_atom++) {
            const double *a = acc + B_atom * 9;
            for (int x = 0; x < 3; x++) {
                Egr[(B_atom * 3 + x) * ng + i] = -2 * a[x];
                Ugr[(B_atom * 3 + x) * ng + i] =  2 * a[3 + x];
                Wgr[(B_atom * 3 + x) * ng + i] =  2 * a[6 + x];
            }
        }
    }
    free(acc);
}
}

/* Ew[d,i] = sum_j w1[d,j] E_ij, Uw[d,i] = -sum_j w1[d,j] U_ij, Ww[d,i] = -sum_j w1[d,j] W_ij for the weight
 * derivatives w1 [nderivative,ngrids]: the three pair kernels of a block to K[3][bj][bi], then dgemm. */
void VXC_vv10nlc_hessian_eval_EUW_with_weight1(double* __restrict__ Ew, double* __restrict__ Uw, double* __restrict__ Ww,
                                               const double* __restrict__ grid_coord, const double* __restrict__ grid_weight1,
                                               const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                               const int ngrids, const int nderivative)
{
    if (ngrids <= 0 || nderivative <= 0) {
        return;
    }
    const char TRANS_N = 'N';
    const double D1 = 1;
    const double DM1 = -1;
    const int ld = ngrids;
    const int nd = nderivative;
    const size_t nblk = VV10_BI * VV10_BJ;

#pragma omp parallel
{
    double *K = malloc(sizeof(double) * (3 * nblk + 5 * VV10_BI));
    double *KE = K;
    double *KU = KE + nblk;
    double *KW = KU + nblk;
    double *xi = KW + nblk;
    double *yi = xi + VV10_BI;
    double *zi = yi + VV10_BI;
    double *omega_i = zi + VV10_BI;
    double *kappa_i = omega_i + VV10_BI;

#pragma omp for schedule(dynamic, 1)
    for (int i0 = 0; i0 < ngrids; i0 += VV10_BI) {
        const int bi = MIN(VV10_BI, ngrids - i0);
        for (int ii = 0; ii < bi; ii++) {
            const int i = i0 + ii;
            xi[ii] = grid_coord[i * 3 + 0];
            yi[ii] = grid_coord[i * 3 + 1];
            zi[ii] = grid_coord[i * 3 + 2];
            omega_i[ii] = omega[i];
            kappa_i[ii] = kappa[i];
        }
        for (int d = 0; d < nd; d++) {
            for (int ii = 0; ii < bi; ii++) {
                Ew[(size_t)d * ld + i0 + ii] = 0;
                Uw[(size_t)d * ld + i0 + ii] = 0;
                Ww[(size_t)d * ld + i0 + ii] = 0;
            }
        }

        for (int j0 = 0; j0 < ngrids; j0 += VV10_BJ) {
            const int bj = MIN(VV10_BJ, ngrids - j0);
            for (int jj = 0; jj < bj; jj++) {
                const int j = j0 + jj;
                const double xj = grid_coord[j * 3 + 0];
                const double yj = grid_coord[j * 3 + 1];
                const double zj = grid_coord[j * 3 + 2];
                const double omega_j = omega[j];
                const double kappa_j = kappa[j];
                const double rho_j = rho[j];
                double *kE = KE + jj * bi;
                double *kU = KU + jj * bi;
                double *kW = KW + jj * bi;
#pragma omp simd
                for (int ii = 0; ii < bi; ii++) {
                    const double dx = xi[ii] - xj;
                    const double dy = yi[ii] - yj;
                    const double dz = zi[ii] - zj;
                    const double r_ij2 = dx * dx + dy * dy + dz * dz;
                    const double g_ij = omega_i[ii] * r_ij2 + kappa_i[ii];
                    const double g_ji = omega_j * r_ij2 + kappa_j;
                    const double g_ij_1 = 1 / g_ij;
                    const double g_sum_1 = 1 / (g_ij + g_ji);
                    const double Phi_ij = -1.5 / g_ji * g_ij_1 * g_sum_1;

                    const double E_ij = rho_j * Phi_ij;
                    const double U_ij = E_ij * (g_sum_1 + g_ij_1);
                    kE[ii] = E_ij;
                    kU[ii] = U_ij;
                    kW[ii] = U_ij * r_ij2;
                }
            }
            dgemm_(&TRANS_N, &TRANS_N, &bi, &nd, &bj, &D1 , KE, &bi, grid_weight1 + j0, &ld, &D1, Ew + i0, &ld);
            dgemm_(&TRANS_N, &TRANS_N, &bi, &nd, &bj, &DM1, KU, &bi, grid_weight1 + j0, &ld, &D1, Uw + i0, &ld);
            dgemm_(&TRANS_N, &TRANS_N, &bi, &nd, &bj, &DM1, KW, &bi, grid_weight1 + j0, &ld, &D1, Ww + i0, &ld);
        }
    }
    free(K);
}
}
