/* Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
  
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
 * JKoperator
 */

#include <stdlib.h>
#include <stdio.h>
#include "nr_direct.h"
#include "np_helper/np_helper.h"

#define ASSERT(expr, msg) \
        if (!(expr)) { fprintf(stderr, "Fail at %s\n", msg); exit(1); }

#define MAXCGTO 64

#define ISH0    0
#define ISH1    1
#define JSH0    2
#define JSH1    3
#define KSH0    4
#define KSH1    5
#define LSH0    6
#define LSH1    7

#define JKOP_DATA_SIZE(obra, oket) \
        static size_t JKOperator_data_size_##obra##oket(int *shls_slice, int *ao_loc) \
{ \
        int nbra = ao_loc[shls_slice[obra##SH1]] - ao_loc[shls_slice[obra##SH0]]; \
        int nket = ao_loc[shls_slice[oket##SH1]] - ao_loc[shls_slice[oket##SH0]]; \
        return nbra * nket; \
}
JKOP_DATA_SIZE(K, L)
JKOP_DATA_SIZE(L, K)
JKOP_DATA_SIZE(I, J)
JKOP_DATA_SIZE(J, I)
JKOP_DATA_SIZE(K, J)
JKOP_DATA_SIZE(J, K)
JKOP_DATA_SIZE(I, L)
JKOP_DATA_SIZE(L, I)
JKOP_DATA_SIZE(K, I)
JKOP_DATA_SIZE(I, K)
JKOP_DATA_SIZE(J, L)
JKOP_DATA_SIZE(L, J)

void JKOperator_write_back(double *vjk, JKArray *jkarray, int *ao_loc,
                           int *ishls, int *jshls, int *block_iloc, int *block_jloc)
{
        int ish0 = ishls[0];
        int ish1 = ishls[1];
        int jsh0 = jshls[0];
        int jsh1 = jshls[1];
        size_t vrow = ao_loc[ish1] - ao_loc[ish0];
        size_t vcol = ao_loc[jsh1] - ao_loc[jsh0];
        int ncomp = jkarray->ncomp;
        int voffset = ao_loc[ish0] * vcol + ao_loc[jsh0];
        int nblock = jkarray->nblock;
        int key_counts = jkarray->key_counts;
        int *keys_cache = jkarray->keys_cache;
        int *offsets_dic = jkarray->outptr;
        double *jkarray_data = jkarray->data;
        int key_id, key, block_i, block_j;
        int i, j, ish, jsh, i0, j0;
        int di, dj, icomp;
        int block_i0, block_j0, block_dj;
        double *data, *pd, *pv;

        for (key_id = 0; key_id < key_counts; key_id++) {
                key = keys_cache[key_id];
                block_i = key / nblock;
                block_j = key % nblock;
                ish0 = block_iloc[block_i];
                ish1 = block_iloc[block_i+1];
                jsh0 = block_jloc[block_j];
                jsh1 = block_jloc[block_j+1];
                block_i0 = ao_loc[ish0];
                block_j0 = ao_loc[jsh0];
                block_dj = ao_loc[jsh1] - ao_loc[jsh0];

                data = jkarray_data + offsets_dic[key];
                offsets_dic[key] = NOVALUE;
                for (ish = ish0; ish < ish1; ish++) {
                for (jsh = jsh0; jsh < jsh1; jsh++) {
                        i0 = ao_loc[ish];
                        j0 = ao_loc[jsh];
                        di = ao_loc[ish+1] - i0;
                        dj = ao_loc[jsh+1] - j0;
                        pd = data + ((i0 - block_i0) * block_dj + (j0 - block_j0) * di) * ncomp;
                        pv = vjk + i0*vcol+j0 - voffset;
                        for (icomp = 0; icomp < ncomp; icomp++) {
                                for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        pv[i*vcol+j] += pd[i*dj+j];
                                } }
                                pv += vrow * vcol;
                                pd += di * dj;
                        }
                } }
        }
        jkarray->stack_size = 0;
        jkarray->key_counts = 0;
}

#define JKOP_WRITE_BACK(obra, oket) \
        void JKOperator_write_back_##obra##oket(double *vjk, JKArray *jkarray, \
                                                int *shls_slice, int *ao_loc, \
                                                int *block_Iloc, int *block_Jloc, \
                                                int *block_Kloc, int *block_Lloc) \
{ \
        int *ishls = shls_slice + obra##SH0; \
        int *jshls = shls_slice + oket##SH0; \
        JKOperator_write_back(vjk, jkarray, ao_loc, ishls, jshls, \
                              block_##obra##loc, block_##oket##loc); \
}
JKOP_WRITE_BACK(K, L)
JKOP_WRITE_BACK(L, K)
JKOP_WRITE_BACK(I, J)
JKOP_WRITE_BACK(J, I)
JKOP_WRITE_BACK(K, J)
JKOP_WRITE_BACK(J, K)
JKOP_WRITE_BACK(I, L)
JKOP_WRITE_BACK(L, I)
JKOP_WRITE_BACK(K, I)
JKOP_WRITE_BACK(I, K)
JKOP_WRITE_BACK(J, L)
JKOP_WRITE_BACK(L, J)

#define ADD_JKOP(fname, ibra, iket, obra, oket, type) \
JKOperator CVHF##fname = {ibra##SH0, iket##SH0, obra##SH0, oket##SH0, \
        fname, JKOperator_data_size_##obra##oket, \
        JKOperator_sanity_check_##type, \
        JKOperator_write_back_##obra##oket}

static void JKOperator_sanity_check_s1(int *shls_slice)
{
}
static void JKOperator_sanity_check_s2ij(int *shls_slice)
{
        ASSERT(((shls_slice[0] == shls_slice[2]) &&
                (shls_slice[1] == shls_slice[3])), "s2ij");
}
static void JKOperator_sanity_check_s2kl(int *shls_slice)
{
        ASSERT(((shls_slice[4] == shls_slice[6]) &&
                (shls_slice[5] == shls_slice[7])), "s2kl");
}
static void JKOperator_sanity_check_s4(int *shls_slice)
{
        ASSERT(((shls_slice[0] == shls_slice[2]) &&
                (shls_slice[1] == shls_slice[3])), "s4 ij");
        ASSERT(((shls_slice[4] == shls_slice[6]) &&
                (shls_slice[5] == shls_slice[7])), "s4 kl");
}
static void JKOperator_sanity_check_s8(int *shls_slice)
{
        ASSERT(((shls_slice[0] == shls_slice[2]) &&
                (shls_slice[1] == shls_slice[3])), "s8 ij");
        ASSERT(((shls_slice[4] == shls_slice[6]) &&
                (shls_slice[5] == shls_slice[7])), "s8 kl");
        ASSERT(((shls_slice[0] == shls_slice[4]) &&
                (shls_slice[1] == shls_slice[5])), "s8 ik");
}

#define iSH     0
#define jSH     1
#define kSH     2
#define lSH     3
#define LOCATE(v, i, j) \
        int d##i##j = d##i * d##j; \
        _key = block[i##SH]*out->nblock+block[j##SH]; \
        _poutptr = out->outptr + _key; \
        if (*_poutptr == NOVALUE) { \
                *_poutptr = out->stack_size; \
                int v_size = shape[i##SH]*shape[j##SH]; \
                out->stack_size += v_size * ncomp; \
                NPdset0(out->data+*_poutptr, v_size*ncomp); \
                out->keys_cache[out->key_counts] = _key; \
                out->key_counts++; \
        } \
        double *v = out->data + *_poutptr; \
        v += ((i##0-ao_off[i##SH]) * shape[j##SH] + (j##0-ao_off[j##SH]) * d##i) * ncomp;

#define DECLARE(v, i, j) \
        int ncomp = out->ncomp; \
        int ncol = out->dm_dims[1]; \
        int di = i1 - i0; \
        int dj = j1 - j0; \
        int dk = k1 - k0; \
        int dl = l1 - l0; \
        int _key; \
        int *_poutptr; \
        int *shape = out->shape; \
        int *block = out->block_quartets; \
        int *ao_off = out->ao_off; \
        LOCATE(v, i, j)

#define DEF_NRS1_CONTRACT(D1, D2, V1, V2) \
static void nrs1_##D1##D2##_s1##V1##V2(double *eri, double *dm, JKArray *out, int *shls, \
                         int i0, int i1, int j0, int j1, \
                         int k0, int k1, int l0, int l1) \
{ \
        DECLARE(v, V1, V2); \
        int i, j, k, l, ijkl, icomp; \
        dm += D1##0 * ncol + D2##0 * d##D1; \
 \
        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) { \
                for (l = 0; l < dl; l++) { \
                for (k = 0; k < dk; k++) { \
                for (j = 0; j < dj; j++) { \
                for (i = 0; i < di; i++, ijkl++) { \
                        v[V1*d##V2+V2] += eri[ijkl] * dm[D1*d##D2+D2]; \
                } } } } \
                v += d##V1##V2; \
        } \
}

#define DEF_DM(I, J) \
        double *dm##I##J = dm + I##0 * ncol + J##0 * d##I;

/* eri in Fortran order; dm, out in C order */

/*
 * The kernels of the derivative J/K scripts (gradients, Hessians) below sum
 * four outputs at a time in registers: independent sums, where one output at
 * a time is a chain of dependent additions (or of loads and stores of v). Each
 * output is summed in the order of the one-output loop, so the results are the
 * same bit for bit.
 */

// The offset in v (per component dk*dl, rows dl) of the next (k,l) block of
// eri, which runs over k, then l, then the components
#define NEXT_KL(o) \
        o = ic * dkl + kc * dl + lc; \
        if (++kc == dk) { kc = 0; if (++lc == dl) { lc = 0; ic++; } }

static void nrs1_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        DECLARE(v, k, l);
        int dij = di * dj;
        DEF_DM(j, i);
        const int nb = ncomp * dkl;
        int b, ij, o0, o1, o2, o3;
        int kc = 0, lc = 0, ic = 0;
        double s0, s1, s2, s3, d;
        double *e;

        for (b = 0; b + 3 < nb; b += 4) {
                NEXT_KL(o0);
                NEXT_KL(o1);
                NEXT_KL(o2);
                NEXT_KL(o3);
                e = eri + (size_t)b * dij;
                s0 = v[o0];
                s1 = v[o1];
                s2 = v[o2];
                s3 = v[o3];
                for (ij = 0; ij < dij; ij++) {
                        d = dmji[ij];
                        s0 += e[ij      ] * d;
                        s1 += e[ij+dij  ] * d;
                        s2 += e[ij+dij*2] * d;
                        s3 += e[ij+dij*3] * d;
                }
                v[o0] = s0;
                v[o1] = s1;
                v[o2] = s2;
                v[o3] = s3;
        }
        for (; b < nb; b++) {
                NEXT_KL(o0);
                e = eri + (size_t)b * dij;
                s0 = v[o0];
                for (ij = 0; ij < dij; ij++) {
                        s0 += e[ij] * dmji[ij];
                }
                v[o0] = s0;
        }
}
ADD_JKOP(nrs1_ji_s1kl, J, I, K, L, s1);

static void nrs1_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (k0 >= l0) {
                nrs1_ji_s1kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs1_ji_s2kl, J, I, K, L, s1);


static void nrs1_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        DECLARE(v, i, j);
        DEF_DM(l, k);
        int i, j, k, l, ij, icomp;
        double *buf = eri + dij * dk * dl * ncomp;
        double s;

        for (icomp = 0; icomp < ncomp; icomp++) {

                for (i = 0; i < dij; i++) { buf[i] = 0; }
                for (l = 0; l < dl; l++) {
                for (k = 0; k < dk; k++) {
                        s = dmlk[l*dk+k];
#pragma GCC ivdep
                        for (ij = 0; ij < dij; ij++) {
                                buf[ij] += eri[ij] * s;
                        }
                        eri += dij;
                } }

                for (ij = 0, j = 0; j < dj; j++) {
                for (i = 0; i < di; i++, ij++) {
                        v[i*dj+j] += buf[ij];
                } }
                v += dij;
        }
}
ADD_JKOP(nrs1_lk_s1ij, L, K, I, J, s1);

static void nrs1_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 >= j0) {
                nrs1_lk_s1ij  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs1_lk_s2ij, L, K, I, J, s1);


// v[i,l] += sum_kj eri[i,j,k,l] dm[j,k]: the outputs (i,l) of a component,
// four at a time, each summed over k, then j
static void nrs1_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        DECLARE(v, i, l);
        DEF_DM(j, k);
        const int dij = di * dj;
        const int dijk = dij * dk;
        const int nil = di * dl;
        int j, k, o, n, icomp, x;
        int vo[4];
        double *e[4];
        double s0, s1, s2, s3, d;

        for (icomp = 0; icomp < ncomp; icomp++) {
                double *ec = eri + (size_t)icomp * dijk * dl;
                int ic = 0, lc = 0;
                for (o = 0; o + 3 < nil; o += 4) {
                        for (n = 0; n < 4; n++) {
                                e[n] = ec + ic + dijk * lc;
                                vo[n] = ic * dl + lc;
                                if (++ic == di) { ic = 0; lc++; }
                        }
                        s0 = v[vo[0]];
                        s1 = v[vo[1]];
                        s2 = v[vo[2]];
                        s3 = v[vo[3]];
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                                d = dmjk[j*dk+k];
                                x = di * j + dij * k;
                                s0 += e[0][x] * d;
                                s1 += e[1][x] * d;
                                s2 += e[2][x] * d;
                                s3 += e[3][x] * d;
                        } }
                        v[vo[0]] = s0;
                        v[vo[1]] = s1;
                        v[vo[2]] = s2;
                        v[vo[3]] = s3;
                }
                for (; o < nil; o++) {
                        e[0] = ec + ic + dijk * lc;
                        vo[0] = ic * dl + lc;
                        if (++ic == di) { ic = 0; lc++; }
                        s0 = v[vo[0]];
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                                s0 += e[0][di*j+dij*k] * dmjk[j*dk+k];
                        } }
                        v[vo[0]] = s0;
                }
                v += dil;
        }
}
ADD_JKOP(nrs1_jk_s1il, J, K, I, L, s1);

//DEF_NRS1_CONTRACT(j, k, i, l); ADD_JKOP(nrs1_jk_s1il, J, K, I, L, s1);
DEF_NRS1_CONTRACT(j, k, l, i); ADD_JKOP(nrs1_jk_s1li, J, K, L, I, s1);
DEF_NRS1_CONTRACT(k, j, i, l); ADD_JKOP(nrs1_kj_s1il, K, J, I, L, s1);
DEF_NRS1_CONTRACT(k, j, l, i); ADD_JKOP(nrs1_kj_s1li, K, J, L, I, s1);
DEF_NRS1_CONTRACT(i, k, j, l); ADD_JKOP(nrs1_ik_s1jl, I, K, J, L, s1);
DEF_NRS1_CONTRACT(i, k, l, j); ADD_JKOP(nrs1_ik_s1lj, I, K, L, J, s1);
DEF_NRS1_CONTRACT(k, i, l, j); ADD_JKOP(nrs1_ki_s1lj, K, I, L, J, s1);
DEF_NRS1_CONTRACT(k, i, j, l); ADD_JKOP(nrs1_ki_s1jl, K, I, J, L, s1);
DEF_NRS1_CONTRACT(j, l, k, i); ADD_JKOP(nrs1_jl_s1ki, J, L, K, I, s1);
DEF_NRS1_CONTRACT(j, l, i, k); ADD_JKOP(nrs1_jl_s1ik, J, L, I, K, s1);
// v[k,i] += sum_j eri[i,j,k,l] dm[l,j]: for each l, the outputs (i,k) (one
// index ki = i+di*k of v) four at a time, each summed over j
static void nrs1_lj_s1ki(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        DECLARE(v, k, i);
        DEF_DM(l, j);
        const int dij = di * dj;
        int j, l, ki, n, icomp, x;
        double *e[4];
        double s0, s1, s2, s3, d;

        for (icomp = 0; icomp < ncomp; icomp++) {
                for (l = 0; l < dl; l++) {
                        double *pd = dmlj + l * dj;
                        double *el = eri + ((size_t)icomp * dl + l) * dk * dij;
                        int ic = 0, kc = 0;
                        for (ki = 0; ki + 3 < dki; ki += 4) {
                                for (n = 0; n < 4; n++) {
                                        e[n] = el + ic + dij * kc;
                                        if (++ic == di) { ic = 0; kc++; }
                                }
                                s0 = v[ki  ];
                                s1 = v[ki+1];
                                s2 = v[ki+2];
                                s3 = v[ki+3];
                                for (j = 0; j < dj; j++) {
                                        d = pd[j];
                                        x = di * j;
                                        s0 += e[0][x] * d;
                                        s1 += e[1][x] * d;
                                        s2 += e[2][x] * d;
                                        s3 += e[3][x] * d;
                                }
                                v[ki  ] = s0;
                                v[ki+1] = s1;
                                v[ki+2] = s2;
                                v[ki+3] = s3;
                        }
                        for (; ki < dki; ki++) {
                                e[0] = el + ic + dij * kc;
                                if (++ic == di) { ic = 0; kc++; }
                                s0 = v[ki];
                                for (j = 0; j < dj; j++) {
                                        s0 += e[0][di*j] * pd[j];
                                }
                                v[ki] = s0;
                        }
                }
                v += dki;
        }
}
ADD_JKOP(nrs1_lj_s1ki, L, J, K, I, s1);
DEF_NRS1_CONTRACT(l, j, i, k); ADD_JKOP(nrs1_lj_s1ik, L, J, I, K, s1);
// v[k,j] += sum_i eri[i,j,k,l] dm[l,i]: for each l, the outputs (j,k) (one
// index jk = j+dj*k of both eri and v) four at a time, each summed over i
static void nrs1_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        DECLARE(v, k, j);
        DEF_DM(l, i);
        int i, l, jk, icomp;
        double *e;
        double s0, s1, s2, s3, d;

        for (icomp = 0; icomp < ncomp; icomp++) {
                for (l = 0; l < dl; l++) {
                        double *pd = dmli + l * di;
                        double *el = eri + ((size_t)icomp * dl + l) * dkj * di;
                        for (jk = 0; jk + 3 < dkj; jk += 4) {
                                e = el + jk * di;
                                s0 = v[jk  ];
                                s1 = v[jk+1];
                                s2 = v[jk+2];
                                s3 = v[jk+3];
                                for (i = 0; i < di; i++) {
                                        d = pd[i];
                                        s0 += e[i     ] * d;
                                        s1 += e[i+di  ] * d;
                                        s2 += e[i+di*2] * d;
                                        s3 += e[i+di*3] * d;
                                }
                                v[jk  ] = s0;
                                v[jk+1] = s1;
                                v[jk+2] = s2;
                                v[jk+3] = s3;
                        }
                        for (; jk < dkj; jk++) {
                                e = el + jk * di;
                                s0 = v[jk];
                                for (i = 0; i < di; i++) {
                                        s0 += e[i] * pd[i];
                                }
                                v[jk] = s0;
                        }
                }
                v += dkj;
        }
}
ADD_JKOP(nrs1_li_s1kj, L, I, K, J, s1);
DEF_NRS1_CONTRACT(l, i, j, k); ADD_JKOP(nrs1_li_s1jk, L, I, J, K, s1);
DEF_NRS1_CONTRACT(i, l, k, j); ADD_JKOP(nrs1_il_s1kj, I, L, K, J, s1);
DEF_NRS1_CONTRACT(i, l, j, k); ADD_JKOP(nrs1_il_s1jk, I, L, J, K, s1);

//DEF_NRS1_CONTRACT(j, i, k, l); ADD_JKOP(nrs1_ji_s1kl, J, I, K, L, s1);
//DEF_NRS1_CONTRACT(l, k, i, j); ADD_JKOP(nrs1_lk_s1ij, L, K, I, J, s1);
DEF_NRS1_CONTRACT(i, j, k, l); ADD_JKOP(nrs1_ij_s1kl, I, J, K, L, s1);
DEF_NRS1_CONTRACT(i, j, l, k); ADD_JKOP(nrs1_ij_s1lk, I, J, L, K, s1);
DEF_NRS1_CONTRACT(j, i, l, k); ADD_JKOP(nrs1_ji_s1lk, J, I, L, K, s1);
DEF_NRS1_CONTRACT(l, k, j, i); ADD_JKOP(nrs1_lk_s1ji, L, K, J, I, s1);
DEF_NRS1_CONTRACT(k, l, i, j); ADD_JKOP(nrs1_kl_s1ij, K, L, I, J, s1);
DEF_NRS1_CONTRACT(k, l, j, i); ADD_JKOP(nrs1_kl_s1ji, K, L, J, I, s1);

static void nrs1_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 >= l0) {
                nrs1_jk_s1il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs1_jk_s2il, J, K, I, L, s1);


static void nrs1_kj_s2il(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 >= l0) {
                nrs1_kj_s1il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs1_kj_s2il, J, K, I, L, s1);


static void nrs1_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (k0 >= j0) {
                nrs1_li_s1kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs1_li_s2kj, L, I, K, J, s1);

static void nrs2ij_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(v, k, l);
                int dij = di * dj;
                DEF_DM(i, j);
                DEF_DM(j, i);
                int i, j, k, l, ij, icomp;
                double *tdm = eri + dij * dkl * ncomp;
                double tmp;

                for (ij = 0, j = 0; j < dj; j++) {
                for (i = 0; i < di; i++, ij++) {
                        tdm[ij] = dmij[i*dj+j] + dmji[j*di+i];
                } }

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tmp = 0;
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        tmp += eri[ij] * tdm[ij];
                                }
                                v[k*dl+l] += tmp;
                                eri += dij;
                        } }
                        v += dkl;
                }
        } else {
                nrs1_ji_s1kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2ij_ji_s1kl, J, I, K, L, s2ij);

static void nrs2ij_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 >= l0) {
                nrs2ij_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2ij_ji_s2kl, J, I, K, L, s2ij);


static void nrs2ij_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(vij, i, j);
                LOCATE(vji, j, i);
                DEF_DM(l, k);
                int i, j, k, l, ij, icomp;
                double *buf = eri + dij * dk * dl * ncomp;
                double s;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                s = dmlk[l*dk+k];
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * s;
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                vij[i*dj+j] += buf[ij];
                                vji[ij] += buf[ij];
                        } }
                        vij += dij;
                        vji += dij;
                }
        } else {
                nrs1_lk_s1ij  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2ij_lk_s1ij, L, K, I, J, s2ij);

static void nrs2ij_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nrs2ij_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs2ij_lk_s2ij, L, K, I, J, s2ij);


static void nrs2ij_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(vil, i, l);
                DEF_DM(i, k);
                DEF_DM(j, k);
                LOCATE(vjl, j, l);
                int i, j, k, l, ijkl, icomp;
                double s, tmp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                                s = dmjk[j*dk+k];
                                tmp = vjl[j*dl+l];
                                for (i = 0; i < di; i++, ijkl++) {
                                        vil[i*dl+l] += eri[ijkl] * s;
                                        tmp += eri[ijkl] * dmik[i*dk+k];
                                }
                                vjl[j*dl+l] = tmp;
                        } } }
                        vil += dil;
                        vjl += djl;
                }
        } else {
                nrs1_jk_s1il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2ij_jk_s1il, J, K, I, L, s2ij);

static void nrs2ij_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (j0 >= l0) {
                nrs2ij_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_jk_s2il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2ij_jk_s2il, J, K, I, L, s2ij);


static void nrs2ij_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(vkj, k, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                LOCATE(vki, k, i);
                int i, j, k, l, ijkl, icomp;
                double s, tmp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                                s = dmlj[l*dj+j];
                                tmp = vkj[k*dj+j];
                                for (i = 0; i < di; i++, ijkl++) {
                                        tmp += eri[ijkl] * dmli[l*di+i];
                                        vki[k*di+i] += eri[ijkl] * s;
                                }
                                vkj[k*dj+j] = tmp;
                        } } }
                        vkj += dkj;
                        vki += dki;
                }
        } else {
                nrs1_li_s1kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2ij_li_s1kj, L, I, K, J, s2ij);

static void nrs2ij_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 >= i0) {
                nrs2ij_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_li_s2kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2ij_li_s2kj, L, I, K, J, s2ij);


static void nrs2kl_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(vkl, k, l);
                LOCATE(vlk, l, k);
                int dij = di * dj;
                DEF_DM(j, i);
                int k, l, ij, icomp;
                double tmp;

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tmp = 0;
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        tmp += eri[ij] * dmji[ij];
                                }
                                vkl[k*dl+l] += tmp;
                                vlk[l*dk+k] += tmp;
                                eri += dij;
                        } }
                        vkl += dkl;
                        vlk += dkl;
                }
        } else {
                return nrs1_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2kl_ji_s1kl, J, I, K, L, s2kl);

static void nrs2kl_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nrs1_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs2kl_ji_s2kl, J, I, K, L, s2kl);


static void nrs2kl_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(v, i, j);
                DEF_DM(k, l);
                DEF_DM(l, k);
                int i, j, k, l, ij, icomp;
                double *buf = eri + dij * dk * dl * ncomp;
                double tdm;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tdm = dmkl[k*dl+l] + dmlk[l*dk+k];
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * tdm;
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                v[i*dj+j] += buf[ij];
                        } }
                        v += dij;
                }
        } else {
                nrs1_lk_s1ij  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2kl_lk_s1ij, L, K, I, J, s2kl);

static void nrs2kl_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 >= j0) {
                nrs2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2kl_lk_s2ij, L, K, I, J, s2kl);


static void nrs2kl_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(vil, i, l);
                DEF_DM(j, k);
                DEF_DM(j, l);
                LOCATE(vik, i, k);
                // For each (l,k), the outputs vil[i,l] and vik[i,k] of two i
                // at a time, each summed over j
                int i, j, k, l, icomp;
                double *e;
                double a0, a1, b0, b1, d0, d1;

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                e = eri + ((size_t)(icomp * dl + l) * dk + k) * di * dj;
                                for (i = 0; i + 1 < di; i += 2) {
                                        a0 = vil[ i   *dl+l];
                                        a1 = vil[(i+1)*dl+l];
                                        b0 = vik[ i   *dk+k];
                                        b1 = vik[(i+1)*dk+k];
                                        for (j = 0; j < dj; j++) {
                                                d0 = dmjk[j*dk+k];
                                                d1 = dmjl[j*dl+l];
                                                a0 += e[i  +di*j] * d0;
                                                a1 += e[i+1+di*j] * d0;
                                                b0 += e[i  +di*j] * d1;
                                                b1 += e[i+1+di*j] * d1;
                                        }
                                        vil[ i   *dl+l] = a0;
                                        vil[(i+1)*dl+l] = a1;
                                        vik[ i   *dk+k] = b0;
                                        vik[(i+1)*dk+k] = b1;
                                }
                                if (i < di) {
                                        a0 = vil[i*dl+l];
                                        b0 = vik[i*dk+k];
                                        for (j = 0; j < dj; j++) {
                                                a0 += e[i+di*j] * dmjk[j*dk+k];
                                                b0 += e[i+di*j] * dmjl[j*dl+l];
                                        }
                                        vil[i*dl+l] = a0;
                                        vik[i*dk+k] = b0;
                                }
                        } }
                        vil += dil;
                        vik += dik;
                }
        } else {
                nrs1_jk_s1il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2kl_jk_s1il, J, K, I, L, s2kl);

static void nrs2kl_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 >= k0) {
                nrs2kl_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_jk_s2il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2kl_jk_s2il, J, K, I, L, s2kl);


static void nrs2kl_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(vkj, k, j);
                DEF_DM(k, i);
                DEF_DM(l, i);
                LOCATE(vlj, l, j);
                // For each (l,k), the outputs vkj[k,j] and vlj[l,j] of two j
                // at a time, each summed over i
                int i, j, k, l, icomp;
                double *e, *pli, *pki;
                double a0, a1, b0, b1;

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        pli = dmli + l * di;
                        for (k = 0; k < dk; k++) {
                                pki = dmki + k * di;
                                e = eri + ((size_t)(icomp * dl + l) * dk + k) * di * dj;
                                for (j = 0; j + 1 < dj; j += 2) {
                                        a0 = vkj[k*dj+j  ];
                                        a1 = vkj[k*dj+j+1];
                                        b0 = vlj[l*dj+j  ];
                                        b1 = vlj[l*dj+j+1];
                                        for (i = 0; i < di; i++) {
                                                a0 += e[i+di* j   ] * pli[i];
                                                a1 += e[i+di*(j+1)] * pli[i];
                                                b0 += e[i+di* j   ] * pki[i];
                                                b1 += e[i+di*(j+1)] * pki[i];
                                        }
                                        vkj[k*dj+j  ] = a0;
                                        vkj[k*dj+j+1] = a1;
                                        vlj[l*dj+j  ] = b0;
                                        vlj[l*dj+j+1] = b1;
                                }
                                if (j < dj) {
                                        a0 = vkj[k*dj+j];
                                        b0 = vlj[l*dj+j];
                                        for (i = 0; i < di; i++) {
                                                a0 += e[i+di*j] * pli[i];
                                                b0 += e[i+di*j] * pki[i];
                                        }
                                        vkj[k*dj+j] = a0;
                                        vlj[l*dj+j] = b0;
                                }
                        } }
                        vkj += dkj;
                        vlj += dlj;
                }
        } else {
                nrs1_li_s1kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2kl_li_s1kj, L, I, K, J, s2kl);

static void nrs2kl_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (l0 >= j0) {
                nrs2kl_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_li_s2kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nrs2kl_li_s2kj, L, I, K, J, s2kl);


static void nrs4_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vkl, k, l);
                LOCATE(vlk, l, k);
                int dij = di * dj;
                DEF_DM(i, j);
                DEF_DM(j, i);
                int i, j, k, l, ij, icomp;
                double *tdm = eri + dij * dkl * ncomp;
                double tmp;

                for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                tdm[ij] = dmij[i*dj+j] + dmji[j*di+i];
                        }
                }

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tmp = 0;
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        tmp += eri[ij] * tdm[ij];
                                }
                                vkl[k*dl+l] += tmp;
                                vlk[l*dk+k] += tmp;
                                eri += dij;
                        } }
                        vkl += dkl;
                        vlk += dkl;
                }
        }
}
ADD_JKOP(nrs4_ji_s1kl, J, I, K, L, s4);

static void nrs4_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        nrs2ij_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs4_ji_s2kl, J, I, K, L, s4);


static void nrs4_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vij, i, j);
                LOCATE(vji, j, i);
                DEF_DM(k, l);
                DEF_DM(l, k);
                int i, j, k, l, ij, icomp;
                double *buf = eri + dij * dk * dl * ncomp;
                double tdm;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tdm = dmlk[l*dk+k] + dmkl[k*dl+l];
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * tdm;
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                vij[i*dj+j] += buf[ij];
                                vji[ij] += buf[ij];
                        } }
                        vij += dij;
                        vji += dij;
                }
        }
}
ADD_JKOP(nrs4_lk_s1ij, L, K, I, J, s4);

static void nrs4_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        nrs2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs4_lk_s2ij, L, K, I, J, s4);

static void nrs4_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vik, i, k);
                LOCATE(vil, i, l);
                LOCATE(vjk, j, k);
                LOCATE(vjl, j, l);
                DEF_DM(i, l);
                DEF_DM(i, k);
                DEF_DM(j, l);
                DEF_DM(j, k);
                int i, j, k, l, ijkl, icomp;
                double s0, s1, tmp0, tmp1;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                                s0 = dmjl[j*dl+l];
                                s1 = dmjk[j*dk+k];
                                tmp0 = vjk[j*dk+k];
                                tmp1 = vjl[j*dl+l];
                                for (i = 0; i < di; i++, ijkl++) {
                                        tmp0 += eri[ijkl] * dmil[i*dl+l];
                                        tmp1 += eri[ijkl] * dmik[i*dk+k];
                                        vik[i*dk+k] += eri[ijkl] * s0;
                                        vil[i*dl+l] += eri[ijkl] * s1;
                                }
                                vjk[j*dk+k] = tmp0;
                                vjl[j*dl+l] = tmp1;
                        } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nrs4_jk_s1il, J, K, I, L, s4);

static void nrs4_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (i0 < l0) {
        } else if (i0 < k0) {
                if (j0 < l0) { // j < l <= i < k
                        DECLARE(v, i, l);
                        DEF_DM(j, k);
                        int i, j, k, l, ijkl, icomp;
                        double s;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                        s = dmjk[j*dk+k];
                                        for (i = 0; i < di; i++, ijkl++) {
                                                v[i*dl+l] += eri[ijkl] * s;
                                        }
                                } } }
                                v += dil;
                        }
                } else { // l <= j < i < k
                        DECLARE(vil, i, l);
                        LOCATE(vjl, j, l);
                        DEF_DM(i, k);
                        DEF_DM(j, k);
                        int i, j, k, l, ijkl, icomp;
                        double s, tmp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                        s = dmjk[j*dk+k];
                                        tmp = vjl[j*dl+l];
                                        for (i = 0; i < di; i++, ijkl++) {
                                                tmp += eri[ijkl] * dmik[i*dk+k];
                                                vil[i*dl+l] += eri[ijkl] * s;
                                        }
                                        vjl[j*dl+l] = tmp;
                                } } }
                                vjl += djl;
                                vil += dil;
                        }
                }
        } else if (j0 < l0) { // j < l < k <= i
                DECLARE(vil, i, l);
                LOCATE(vik, i, k);
                DEF_DM(j, k);
                DEF_DM(j, l);
                int i, j, k, l, ijkl, icomp;
                double s0, s1;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                                s0 = dmjk[j*dk+k];
                                s1 = dmjl[j*dl+l];
                                for (i = 0; i < di; i++, ijkl++) {
                                        vil[i*dl+l] += eri[ijkl] * s0;
                                        vik[i*dk+k] += eri[ijkl] * s1;
                                }
                        } } }
                        vil += dil;
                        vik += dik;
                }
        } else if (j0 < k0) { // l <= j < k <= i
                DECLARE(vjl, j, l);
                LOCATE(vik, i, k);
                LOCATE(vil, i, l);
                DEF_DM(i, k);
                DEF_DM(j, k);
                DEF_DM(j, l);
                int i, j, k, l, ijkl, icomp;
                double s0, s1, tmp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                                s0 = dmjk[j*dk+k];
                                s1 = dmjl[j*dl+l];
                                tmp = vjl[j*dl+l];
                                for (i = 0; i < di; i++, ijkl++) {
                                        tmp += eri[ijkl] * dmik[i*dk+k];
                                        vil[i*dl+l] += eri[ijkl] * s0;
                                        vik[i*dk+k] += eri[ijkl] * s1;
                                }
                                vjl[j*dl+l] = tmp;
                        } } }
                        vjl += djl;
                        vil += dil;
                        vik += dik;
                }
        } else { // l < k <= j < i
                DECLARE(vjl, j, l);
                LOCATE(vik, i, k);
                LOCATE(vjk, j, k);
                LOCATE(vil, i, l);
                DEF_DM(i, l);
                DEF_DM(i, k);
                DEF_DM(j, l);
                DEF_DM(j, k);
                int i, j, k, l, ijkl, icomp;
                double s0, s1, tmp0, tmp1;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                                s0 = dmjl[j*dl+l];
                                s1 = dmjk[j*dk+k];
                                tmp0 = vjk[j*dk+k];
                                tmp1 = vjl[j*dl+l];
                                for (i = 0; i < di; i++, ijkl++) {
                                        tmp0 += eri[ijkl] * dmil[i*dl+l];
                                        tmp1 += eri[ijkl] * dmik[i*dk+k];
                                        vik[i*dk+k] += eri[ijkl] * s0;
                                        vil[i*dl+l] += eri[ijkl] * s1;
                                }
                                vjk[j*dk+k] = tmp0;
                                vjl[j*dl+l] = tmp1;
                        } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nrs4_jk_s2il, J, K, I, L, s4);


static void nrs4_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                DEF_DM(k, i);
                DEF_DM(k, j);
                int i, j, k, l, ijkl, icomp;
                double s0, s1, tmp0, tmp1;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                                s0 = dmlj[l*dj+j];
                                s1 = dmkj[k*dj+j];
                                tmp0 = vkj[k*dj+j];
                                tmp1 = vlj[l*dj+j];
                                for (i = 0; i < di; i++, ijkl++) {
                                        tmp0 += eri[ijkl] * dmli[l*di+i];
                                        tmp1 += eri[ijkl] * dmki[k*di+i];
                                        vki[k*di+i] += eri[ijkl] * s0;
                                        vli[l*di+i] += eri[ijkl] * s1;
                                }
                                vkj[k*dj+j] = tmp0;
                                vlj[l*dj+j] = tmp1;
                        } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nrs4_li_s1kj, L, I, K, J, s4);

static void nrs4_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 < j0) {
        } else if (k0 < i0) {
                if (l0 < j0) { // l < j < k < i
                        DECLARE(v, k, j);
                        DEF_DM(l, i);
                        int i, j, k, l, ijkl, icomp;
                        double tmp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                        tmp = v[k*dj+j];
                                        for (i = 0; i < di; i++, ijkl++) {
                                                tmp += eri[ijkl] * dmli[l*di+i];
                                        }
                                        v[k*dj+j] = tmp;
                                } } }
                                v += dkj;
                        }
                } else { // j <= l < k < i
                        DECLARE(vkj, k, j);
                        LOCATE(vlj, l, j);
                        DEF_DM(l, i);
                        DEF_DM(k, i);
                        int i, j, k, l, ijkl, icomp;
                        double tmp0, tmp1;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                        tmp0 = vkj[k*dj+j];
                                        tmp1 = vlj[l*dj+j];
                                        for (i = 0; i < di; i++, ijkl++) {
                                                tmp0 += eri[ijkl] * dmli[l*di+i];
                                                tmp1 += eri[ijkl] * dmki[k*di+i];
                                        }
                                        vkj[k*dj+j] = tmp0;
                                        vlj[l*dj+j] = tmp1;
                                } } }
                                vkj += dkj;
                                vlj += dlj;
                        }
                }
        } else if (l0 < j0) { // l < j < i <= k
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                int i, j, k, l, ijkl, icomp;
                double s, tmp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                                s = dmlj[l*dj+j];
                                tmp = vkj[k*dj+j];
                                for (i = 0; i < di; i++, ijkl++) {
                                        tmp += eri[ijkl] * dmli[l*di+i];
                                        vki[k*di+i] += eri[ijkl] * s;
                                }
                                vkj[k*dj+j] = tmp;
                        } } }
                        vkj += dkj;
                        vki += dki;
                }
        } else if (l0 < i0) { // j <= l < i <= k
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                LOCATE(vlj, l, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                DEF_DM(k, i);
                int i, j, k, l, ijkl, icomp;
                double s, tmp0, tmp1;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                                s = dmlj[l*dj+j];
                                tmp0 = vkj[k*dj+j];
                                tmp1 = vlj[l*dj+j];
                                for (i = 0; i < di; i++, ijkl++) {
                                        vki[k*di+i] += eri[ijkl] * s;
                                        tmp0 += eri[ijkl] * dmli[l*di+i];
                                        tmp1 += eri[ijkl] * dmki[k*di+i];
                                }
                                vkj[k*dj+j] = tmp0;
                                vlj[l*dj+j] = tmp1;
                        } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                }
        } else { // j < i <= l < k
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                DEF_DM(k, i);
                DEF_DM(k, j);
                int i, j, k, l, ijkl, icomp;
                double s0, s1, tmp0, tmp1;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                                s0 = dmlj[l*dj+j];
                                s1 = dmkj[k*dj+j];
                                tmp0 = vkj[k*dj+j];
                                tmp1 = vlj[l*dj+j];
                                for (i = 0; i < di; i++, ijkl++) {
                                        tmp0 += eri[ijkl] * dmli[l*di+i];
                                        tmp1 += eri[ijkl] * dmki[k*di+i];
                                        vki[k*di+i] += eri[ijkl] * s0;
                                        vli[l*di+i] += eri[ijkl] * s1;
                                }
                                vkj[k*dj+j] = tmp0;
                                vlj[l*dj+j] = tmp1;
                        } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nrs4_li_s2kj, L, I, K, J, s4);


static void nrs8_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == k0 && j0 == l0) {
                nrs4_ji_s1kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (i0 == j0 || k0 == l0) {
                nrs4_ji_s1kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
                nrs4_lk_s1ij  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vij, i, j);
                LOCATE(vji, j, i);
                LOCATE(vkl, k, l);
                LOCATE(vlk, l, k);
                DEF_DM(i, j);
                DEF_DM(j, i);
                DEF_DM(k, l);
                DEF_DM(l, k);
                int i, j, k, l, ij, icomp;
                double *tdm = eri + dij * dkl * ncomp;
                double *buf = tdm + dij;
                double tdm2, tmp;

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (ij = 0, j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++, ij++) {
                                        tdm[ij] = dmij[i*dj+j] + dmji[j*di+i];
                                }
                        }
                        for (ij = 0; ij < dij; ij++) {
                                buf[ij] = 0;
                        }

                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tmp = 0;
                                tdm2 = dmkl[k*dl+l] + dmlk[l*dk+k];
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        tmp += eri[ij] * tdm[ij];
                                        buf[ij] += eri[ij] * tdm2;
                                }
                                vkl[k*dl+l] += tmp;
                                vlk[l*dk+k] += tmp;
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                vij[i*dj+j] += buf[ij];
                                vji[ij] += buf[ij];
                        } }
                        vij += dij;
                        vji += dji;
                        vkl += dkl;
                        vlk += dlk;
                }
        }
}
ADD_JKOP(nrs8_ji_s1kl, J, I, K, L, s8);

static void nrs8_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == k0 && j0 == l0) {
                nrs4_ji_s2kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (i0 == j0 || k0 == l0) {
                nrs4_ji_s2kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
                nrs4_lk_s2ij  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vij, i, j);
                LOCATE(vkl, k, l);
                DEF_DM(i, j);
                DEF_DM(j, i);
                DEF_DM(k, l);
                DEF_DM(l, k);
                int i, j, k, l, ij, icomp;
                double *tdm = eri + dij * dkl * ncomp;
                double *buf = tdm + dij;
                double tmp, tdm2;

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (ij = 0, j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++, ij++) {
                                        tdm[ij] = dmij[i*dj+j] + dmji[j*di+i];
                                }
                        }
                        for (ij = 0; ij < dij; ij++) {
                                buf[ij] = 0;
                        }

                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tdm2 = dmkl[k*dl+l] + dmlk[l*dk+k];
                                tmp = 0;
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * tdm2;
                                        tmp     += eri[ij] * tdm[ij];
                                }
                                vkl[k*dl+l] += tmp;
                                eri += dij;
                        } }

                        for (ij = 0, i = 0; i < di; i++) {
                        for (j = 0; j < dj; j++, ij++) {
                                vij[ij] += buf[j*di+i];
                        } }
                        vij += dij;
                        vkl += dkl;
                }
        }
}
ADD_JKOP(nrs8_ji_s2kl, J, I, K, L, s8);


static void nrs8_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        nrs8_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs8_lk_s1ij, L, K, I, J, s8);

static void nrs8_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        nrs8_ji_s2kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs8_lk_s2ij, L, K, I, J, s8);


static void nrs8_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == k0 && j0 == l0) {
                nrs4_li_s1kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (i0 == j0 || k0 == l0) { // i0==l0 => i0==k0==l0
                nrs4_li_s1kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
                nrs4_jk_s1il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vkj, k, j);
                LOCATE(vki, k, i);
                LOCATE(vlj, l, j);
                LOCATE(vli, l, i);
                LOCATE(vik, i, k);
                LOCATE(vil, i, l);
                LOCATE(vjk, j, k);
                LOCATE(vjl, j, l);
                DEF_DM(l, i);
                DEF_DM(l, j);
                DEF_DM(k, i);
                DEF_DM(k, j);
                DEF_DM(j, l);
                DEF_DM(j, k);
                DEF_DM(i, l);
                DEF_DM(i, k);
                int i, j, k, l, ijkl, icomp;
                double s, s0, s1, s2, s3, tmp0, tmp1, tmp2, tmp3;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                                s0 = dmlj[l*dj+j];
                                s1 = dmkj[k*dj+j];
                                s2 = dmjl[j*dl+l];
                                s3 = dmjk[j*dk+k];
                                tmp0 = 0;
                                tmp1 = 0;
                                tmp2 = 0;
                                tmp3 = 0;
                                for (i = 0; i < di; i++, ijkl++) {
                                        s = eri[ijkl];
                                        vki[k*di+i] += s * s0;
                                        vli[l*di+i] += s * s1;
                                        vik[i*dk+k] += s * s2;
                                        vil[i*dl+l] += s * s3;
                                        tmp0 += s * dmli[l*di+i];
                                        tmp1 += s * dmki[k*di+i];
                                        tmp2 += s * dmil[i*dl+l];
                                        tmp3 += s * dmik[i*dk+k];
                                }
                                vkj[k*dj+j] += tmp0;
                                vlj[l*dj+j] += tmp1;
                                vjk[j*dk+k] += tmp2; // vkj, vjk may share memory
                                vjl[j*dl+l] += tmp3; // vlj, vjl may share memory
                        } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                        vik += dik;
                        vil += dil;
                        vjk += djk;
                        vjl += djl;
                }
        }
}
ADD_JKOP(nrs8_li_s1kj, L, I, K, J, s8);

static void nrs8_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        if (i0 == k0) {
                nrs4_li_s2kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
                if (j0 != l0) {
                        nrs4_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
                }
        } else if (i0 == j0 || k0 == l0) { // i0==l0 => i0==k0==l0
                nrs4_li_s2kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
                nrs4_jk_s2il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                int i, j, k, l, ijkl, icomp;
                double s, tjl, tjk, sjk, sjl, skj, slj;
                if (j0 < l0) { // j <= l < k < i
                        DECLARE(vkj, k, j);
                        LOCATE(vlj, l, j);
                        LOCATE(vik, i, k);
                        LOCATE(vil, i, l);
                        DEF_DM(l, i);
                        DEF_DM(k, i);
                        DEF_DM(j, l);
                        DEF_DM(j, k);
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                        tjl = dmjl[j*dl+l];
                                        tjk = dmjk[j*dk+k];
                                        skj = 0;
                                        slj = 0;
                                        for (i = 0; i < di; i++, ijkl++) {
                                                //vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                                //vlj[l*dj+j] += eri[ijkl] * dmki[k*di+i];
                                                //vik[i*dk+k] += eri[ijkl] * dmjl[j*dl+l];
                                                //vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                                s = eri[ijkl];
                                                skj += s * dmli[l*di+i];
                                                slj += s * dmki[k*di+i];
                                                vik[i*dk+k] += s * tjl;
                                                vil[i*dl+l] += s * tjk;
                                        }
                                        vkj[k*dj+j] += skj;
                                        vlj[l*dj+j] += slj;
                                } } }
                                vkj += dkj;
                                vlj += dlj;
                                vik += dik;
                                vil += dil;
                        }
                } else if (j0 == l0) { // j == l < k < i
                        DECLARE(vkj, k, j);
                        LOCATE(vlj, l, j);
                        LOCATE(vik, i, k);
                        LOCATE(vil, i, l);
                        LOCATE(vjl, j, l);
                        DEF_DM(l, i);
                        DEF_DM(k, i);
                        DEF_DM(j, l);
                        DEF_DM(j, k);
                        DEF_DM(i, k);
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                        tjl = dmjl[j*dl+l];
                                        tjk = dmjk[j*dk+k];
                                        skj = 0;
                                        slj = 0;
                                        sjl = 0;
                                        for (i = 0; i < di; i++, ijkl++) {
                                                //vik[i*dk+k] += eri[ijkl] * dmjl[j*dl+l];
                                                //vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                                //vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                                //vlj[l*dj+j] += eri[ijkl] * dmki[k*di+i];
                                                //vjl[j*dl+l] += eri[ijkl] * dmik[i*dk+k];
                                                s = eri[ijkl];
                                                vik[i*dk+k] += s * tjl;
                                                vil[i*dl+l] += s * tjk;
                                                skj += s * dmli[l*di+i];
                                                slj += s * dmki[k*di+i];
                                                sjl += s * dmik[i*dk+k];
                                        }
                                        vlj[l*dj+j] += slj;
                                        vkj[k*dj+j] += skj;
                                        vjl[j*dl+l] += sjl; // vjl, vlj may share memory
                                } } }
                                vkj += dkj;
                                vlj += dlj;
                                vik += dik;
                                vil += dil;
                                vjl += djl;
                        }
                } else if (j0 < k0) { // l < j < k < i
                        DECLARE(vkj, k, j);
                        LOCATE(vik, i, k);
                        LOCATE(vil, i, l);
                        LOCATE(vjl, j, l);
                        DEF_DM(l, i);
                        DEF_DM(j, l);
                        DEF_DM(j, k);
                        DEF_DM(i, k);
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                        tjl = dmjl[j*dl+l];
                                        tjk = dmjk[j*dk+k];
                                        skj = 0;
                                        sjl = 0;
                                        for (i = 0; i < di; i++, ijkl++) {
                                                //vik[i*dk+k] += eri[ijkl] * dmjl[j*dl+l];
                                                //vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                                //vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                                //vjl[j*dl+l] += eri[ijkl] * dmik[i*dk+k];
                                                s = eri[ijkl];
                                                vik[i*dk+k] += s * tjl;
                                                vil[i*dl+l] += s * tjk;
                                                skj += s * dmli[l*di+i];
                                                sjl += s * dmik[i*dk+k];
                                        }
                                        vkj[k*dj+j] += skj;
                                        vjl[j*dl+l] += sjl;
                                } } }
                                vkj += dkj;
                                vik += dik;
                                vil += dil;
                                vjl += djl;
                        }
                } else if (j0 == k0) { // l < j == k < i
                        DECLARE(vkj, k, j);
                        LOCATE(vik, i, k);
                        LOCATE(vil, i, l);
                        LOCATE(vjk, j, k);
                        LOCATE(vjl, j, l);
                        DEF_DM(l, i);
                        DEF_DM(i, l);
                        DEF_DM(j, l);
                        DEF_DM(j, k);
                        DEF_DM(i, k);
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                        tjl = dmjl[j*dl+l];
                                        tjk = dmjk[j*dk+k];
                                        sjk = 0;
                                        sjl = 0;
                                        skj = 0;
                                        for (i = 0; i < di; i++, ijkl++) {
                                                //vik[i*dk+k] += eri[ijkl] * dmjl[j*dl+l];
                                                //vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                                //vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                                //vjk[j*dk+k] += eri[ijkl] * dmil[i*dl+l];
                                                //vjl[j*dl+l] += eri[ijkl] * dmik[i*dk+k];
                                                s = eri[ijkl];
                                                vik[i*dk+k] += s * tjl;
                                                vil[i*dl+l] += s * tjk;
                                                skj += s * dmli[l*di+i];
                                                sjk += s * dmil[i*dl+l];
                                                sjl += s * dmik[i*dk+k];
                                        }
                                        vjk[j*dk+k] += sjk;
                                        vjl[j*dl+l] += sjl;
                                        vkj[k*dj+j] += skj; // vjk, vkj may share memory
                                } } }
                                vkj += dkj;
                                vjk += djk;
                                vik += dik;
                                vil += dil;
                                vjl += djl;
                        }
                } else { // l < k < j < i
                        DECLARE(vik, i, k);
                        LOCATE(vil, i, l);
                        LOCATE(vjk, j, k);
                        LOCATE(vjl, j, l);
                        DEF_DM(j, l);
                        DEF_DM(j, k);
                        DEF_DM(i, l);
                        DEF_DM(i, k);
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                        tjl = dmjl[j*dl+l];
                                        tjk = dmjk[j*dk+k];
                                        sjk = 0;
                                        sjl = 0;
                                        for (i = 0; i < di; i++, ijkl++) {
                                                //vik[i*dk+k] += eri[ijkl] * dmjl[j*dl+l];
                                                //vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                                //vjk[j*dk+k] += eri[ijkl] * dmil[i*dl+l];
                                                //vjl[j*dl+l] += eri[ijkl] * dmik[i*dk+k];
                                                s = eri[ijkl];
                                                vik[i*dk+k] += s * tjl;
                                                vil[i*dl+l] += s * tjk;
                                                sjk += s * dmil[i*dl+l];
                                                sjl += s * dmik[i*dk+k];
                                        }
                                        vjk[j*dk+k] += sjk;
                                        vjl[j*dl+l] += sjl;
                                } } }
                                vik += dik;
                                vil += dil;
                                vjk += djk;
                                vjl += djl;
                        }
                }
        }
}
ADD_JKOP(nrs8_li_s2kj, L, I, K, J, s8);


static void nrs8_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        nrs8_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs8_jk_s1il, J, K, I, L, s8);

static void nrs8_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                         int i0, int i1, int j0, int j1,
                         int k0, int k1, int l0, int l1)
{
        nrs8_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nrs8_jk_s2il, J, K, I, L, s8);


/*************************************************
 * For anti symmetrized integrals
 *************************************************/
static void nra2ij_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(v, k, l);
                DEF_DM(i, j);
                DEF_DM(j, i);
                int dij = di * dj;
                int i, j, k, l, ij, icomp;
                double *tdm = eri + dij * dkl * ncomp;
                double tmp;

                for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                tdm[ij] = dmji[j*di+i] - dmij[i*dj+j];
                        }
                }

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tmp = 0;
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        tmp += eri[ij] * tdm[ij];
                                }
                                v[k*dl+l] += tmp;
                                eri += dij;
                        } }
                        v += dkl;
                }
        } else {
                nrs1_ji_s1kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2ij_ji_s1kl, J, I, K, L, s2ij);

static void nra2ij_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 >= l0) {
                nra2ij_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2ij_ji_s2kl, J, I, K, L, s2ij);

static void nra2ij_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(vij, i, j);
                LOCATE(vji, j, i);
                DEF_DM(l, k);
                int i, j, k, l, ij, icomp;
                double *buf = eri + dij * dk * dl * ncomp;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * dmlk[l*dk+k];
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                vij[i*dj+j] += buf[ij];
                                vji[ij] -= buf[ij];
                        } }
                        vij += dij;
                        vji += dij;
                }
        } else {
                nrs1_lk_s1ij  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2ij_lk_s1ij, L, K, I, J, s2ij);

static void nra2ij_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2ij_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra2ij_lk_s2ij, L, K, I, J, s2ij);

static void nra2ij_lk_a2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2ij_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra2ij_lk_a2ij, L, K, I, J, s2ij);

static void nra2ij_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(vil, i, l);
                DEF_DM(i, k);
                DEF_DM(j, k);
                LOCATE(vjl, j, l);
                int i, j, k, l, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                vjl[j*dl+l] -= eri[ijkl] * dmik[i*dk+k];
                        } } } }
                        vil += dil;
                        vjl += djl;
                }
        } else {
                nrs1_jk_s1il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2ij_jk_s1il, J, K, I, L, s2ij);

static void nra2ij_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (j0 >= l0) {
                nra2ij_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_jk_s2il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2ij_jk_s2il, J, K, I, L, s2ij);

static void nra2ij_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 > j0) {
                DECLARE(vkj, k, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                LOCATE(vki, k, i);
                int i, j, k, l, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                vki[k*di+i] -= eri[ijkl] * dmlj[l*dj+j];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                }
        } else {
                nrs1_li_s1kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2ij_li_s1kj, L, I, K, J, s2ij);

static void nra2ij_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 >= i0) {
                nra2ij_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_li_s2kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2ij_li_s2kj, L, I, K, J, s2ij);

static void nra2kl_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(vkl, k, l);
                LOCATE(vlk, l, k);
                DEF_DM(j, i);
                int dij = di * dj;
                int k, l, ij, icomp;
                double tmp;

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tmp = 0;
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        tmp += eri[ij] * dmji[ij];
                                }
                                vkl[k*dl+l] += tmp;
                                vlk[l*dk+k] -= tmp;
                                eri += dij;
                        } }
                        vkl += dkl;
                        vlk += dkl;
                }
        } else {
                nrs1_ji_s1kl  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2kl_ji_s1kl, J, I, K, L, s2kl);

static void nra2kl_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nrs1_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra2kl_ji_s2kl, J, I, K, L, s2kl);

static void nra2kl_ji_a2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nrs1_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra2kl_ji_a2kl, J, I, K, L, s2kl);

static void nra2kl_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(v, i, j);
                DEF_DM(k, l);
                DEF_DM(l, k);
                int i, j, k, l, ij, icomp;
                double *buf = eri + dij * dk * dl * ncomp;
                double tdm;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tdm = dmlk[l*dk+k] - dmkl[k*dl+l];
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * tdm;
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                v[i*dj+j] += buf[ij];
                        } }
                        v += dij;
                }
        } else {
                nrs1_lk_s1ij  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2kl_lk_s1ij, L, K, I, J, s2kl);

static void nra2kl_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 >= j0) {
                nra2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2kl_lk_s2ij, L, K, I, J, s2kl);

static void nra2kl_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(vil, i, l);
                DEF_DM(j, k);
                DEF_DM(j, l);
                LOCATE(vik, i, k);
                int i, j, k, l, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                vik[i*dk+k] -= eri[ijkl] * dmjl[j*dl+l];
                        } } } }
                        vik += dik;
                        vil += dil;
                }
        } else {
                nrs1_jk_s1il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2kl_jk_s1il, J, K, I, L, s2kl);

static void nra2kl_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 >= k0) {
                nra2kl_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_jk_s2il  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2kl_jk_s2il, J, K, I, L, s2kl);

static void nra2kl_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (k0 > l0) {
                DECLARE(vkj, k, j);
                DEF_DM(l, i);
                DEF_DM(k, i);
                LOCATE(vlj, l, j);
                int i, j, k, l, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                vlj[l*dj+j] -= eri[ijkl] * dmki[k*di+i];
                        } } } }
                        vkj += dkj;
                        vlj += dlj;
                }
        } else {
                nrs1_li_s1kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2kl_li_s1kj, L, I, K, J, s2kl);

static void nra2kl_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (l0 >= j0) {
                nra2kl_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                nrs1_li_s2kj  (eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        }
}
ADD_JKOP(nra2kl_li_s2kj, L, I, K, J, s2kl);

static void nra4ij_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vkl, k, l);
                LOCATE(vlk, l, k);
                DEF_DM(i, j);
                DEF_DM(j, i);
                int dij = di * dj;
                int i, j, k, l, ij, icomp;
                double *tdm = eri + dij * dkl * ncomp;
                double tmp;

                for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                tdm[ij] = dmji[j*di+i] - dmij[i*dj+j];
                        }
                }

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tmp = 0;
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        tmp += eri[ij] * tdm[ij];
                                }
                                vkl[k*dl+l] += tmp;
                                vlk[l*dk+k] += tmp;
                                eri += dij;
                        } }
                        vkl += dkl;
                        vlk += dkl;
                }
        }
}
ADD_JKOP(nra4ij_ji_s1kl, J, I, K, L, s4);

static void nra4ij_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2ij_ji_s2kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra4ij_ji_s2kl, J, I, K, L, s4);

static void nra4ij_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vij, i, j);
                LOCATE(vji, j, i);
                DEF_DM(k, l);
                DEF_DM(l, k);
                int i, j, k, l, ij, icomp;
                double *buf = eri + dij * dk * dl * ncomp;
                double tdm;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tdm = dmlk[l*dk+k] + dmkl[k*dl+l];
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * tdm;
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                vij[i*dj+j] += buf[ij];
                                vji[ij] -= buf[ij];
                        } }
                        vij += dij;
                        vji += dij;
                }
        }
}
ADD_JKOP(nra4ij_lk_s1ij, L, K, I, J, s4);

static void nra4ij_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nrs2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra4ij_lk_s2ij, L, K, I, J, s4);

static void nra4ij_lk_a2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nrs2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra4ij_lk_a2ij, L, K, I, J, s4);

static void nra4ij_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vik, i, k);
                LOCATE(vil, i, l);
                LOCATE(vjk, j, k);
                LOCATE(vjl, j, l);
                DEF_DM(i, l);
                DEF_DM(i, k);
                DEF_DM(j, l);
                DEF_DM(j, k);
                int i, j, k, l, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vjk[j*dk+k] -= eri[ijkl] * dmil[i*dl+l];
                                vjl[j*dl+l] -= eri[ijkl] * dmik[i*dk+k];
                                vik[i*dk+k] += eri[ijkl] * dmjl[j*dl+l];
                                vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                        } } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nra4ij_jk_s1il, J, K, I, L, s4);

static void nra4ij_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (i0 < l0) {
        } else if (i0 < k0) {
                if (j0 < l0) { // j < l <= i < k
                        DECLARE(v, i, l);
                        DEF_DM(j, k);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        v[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                } } } }
                                v += dil;
                        }
                } else { // l <= j < i < k
                        DECLARE(vil, i, l);
                        DEF_DM(i, k);
                        DEF_DM(j, k);
                        LOCATE(vjl, j, l);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        vjl[j*dl+l] -= eri[ijkl] *dmik[i*dk+k];
                                        vil[i*dl+l] += eri[ijkl] *dmjk[j*dk+k];
                                } } } }
                                vjl += djl;
                                vil += dil;
                        }
                }
        } else if (j0 < l0) { // j < l < k <= i
                DECLARE(vil, i, l);
                DEF_DM(j, k);
                DEF_DM(j, l);
                LOCATE(vik, i, k);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                vik[i*dk+k] += eri[ijkl] * dmjl[j*dl+l];
                        } } } }
                        vil += dil;
                        vik += dik;
                }
        } else if (j0 < k0) { // l <= j < k <= i
                DECLARE(vjl, j, l);
                LOCATE(vik, i, k);
                LOCATE(vil, i, l);
                DEF_DM(i, k);
                DEF_DM(j, k);
                DEF_DM(j, l);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vjl[j*dl+l] -= eri[ijkl] * dmik[i*dk+k];
                                vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                vik[i*dk+k] += eri[ijkl] * dmjl[j*dl+l];
                        } } } }
                        vjl += djl;
                        vil += dil;
                        vik += dik;
                }
        } else { // l < k <= j < i
                DECLARE(vjl, j, l);
                LOCATE(vik, i, k);
                LOCATE(vjk, j, k);
                LOCATE(vil, i, l);
                DEF_DM(i, l);
                DEF_DM(i, k);
                DEF_DM(j, l);
                DEF_DM(j, k);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vjk[j*dk+k] -= eri[ijkl] * dmil[i*dl+l];
                                vjl[j*dl+l] -= eri[ijkl] * dmik[i*dk+k];
                                vik[i*dk+k] += eri[ijkl] * dmjl[j*dl+l];
                                vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                        } } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nra4ij_jk_s2il, J, K, I, L, s4);

static void nra4ij_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                DEF_DM(k, i);
                DEF_DM(k, j);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                vki[k*di+i] -= eri[ijkl] * dmlj[l*dj+j];
                                vlj[l*dj+j] += eri[ijkl] * dmki[k*di+i];
                                vli[l*di+i] -= eri[ijkl] * dmkj[k*dj+j];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nra4ij_li_s1kj, L, I, K, J, s4);

static void nra4ij_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 < j0) {
        } else if (k0 < i0) {
                if (l0 < j0) { // l < j < k < i
                        DECLARE(v, k, j);
                        DEF_DM(l, i);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        v[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                } } } }
                                v += dkj;
                        }
                } else { // j <= l < k < i
                        DECLARE(vkj, k, j);
                        LOCATE(vlj, l, j);
                        DEF_DM(l, i);
                        DEF_DM(k, i);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        vkj[k*dj+j] += eri[ijkl] *dmli[l*di+i];
                                        vlj[l*dj+j] += eri[ijkl] *dmki[k*di+i];
                                } } } }
                                vkj += dkj;
                                vlj += dlj;
                        }
                }
        } else if (l0 < j0) { // l < j < i <= k
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                vki[k*di+i] -= eri[ijkl] * dmlj[l*dj+j];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                }
        } else if (l0 < i0) { // j <= l < i <= k
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                LOCATE(vlj, l, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                DEF_DM(k, i);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                vki[k*di+i] -= eri[ijkl] * dmlj[l*dj+j];
                                vlj[l*dj+j] += eri[ijkl] * dmki[k*di+i];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                }
        } else { // j < i <= l < k
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                DEF_DM(k, i);
                DEF_DM(k, j);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                vki[k*di+i] -= eri[ijkl] * dmlj[l*dj+j];
                                vlj[l*dj+j] += eri[ijkl] * dmki[k*di+i];
                                vli[l*di+i] -= eri[ijkl] * dmkj[k*dj+j];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nra4ij_li_s2kj, L, I, K, J, s4);

static void nra4kl_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vkl, k, l);
                LOCATE(vlk, l, k);
                DEF_DM(i, j);
                DEF_DM(j, i);
                int dij = di * dj;
                int i, j, k, l, ij, icomp;
                double *tdm = eri + dij * dkl * ncomp;
                double tmp;

                for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                tdm[ij] = dmij[i*dj+j] + dmji[j*di+i];
                        }
                }

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tmp = 0;
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        tmp += eri[ij] * tdm[ij];
                                }
                                vkl[k*dl+l] += tmp;
                                vlk[l*dk+k] -= tmp;
                                eri += dij;
                        } }
                        vkl += dkl;
                        vlk += dkl;
                }
        }
}
ADD_JKOP(nra4kl_ji_s1kl, J, I, K, L, s4);

static void nra4kl_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2kl_ji_s2kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra4kl_ji_s2kl, J, I, K, L, s4);

static void nra4kl_ji_a2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2kl_ji_s2kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra4kl_ji_a2kl, J, I, K, L, s4);

static void nra4kl_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vij, i, j);
                LOCATE(vji, j, i);
                DEF_DM(k, l);
                DEF_DM(l, k);
                int i, j, k, l, ij, icomp;
                double *buf = eri + dij * dk * dl * ncomp;
                double tdm;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tdm = dmlk[l*dk+k] - dmkl[k*dl+l];
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * tdm;
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                vij[i*dj+j] += buf[ij];
                                vji[ij] += buf[ij];
                        } }
                        vij += dij;
                        vji += dij;
                }
        }
}
ADD_JKOP(nra4kl_lk_s1ij, L, K, I, J, s4);

static void nra4kl_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nra4kl_lk_s2ij, L, K, I, J, s4);

static void nra4kl_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vik, i, k);
                LOCATE(vil, i, l);
                LOCATE(vjk, j, k);
                LOCATE(vjl, j, l);
                DEF_DM(i, l);
                DEF_DM(i, k);
                DEF_DM(j, l);
                DEF_DM(j, k);
                int i, j, k, l, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vjk[j*dk+k] -= eri[ijkl] * dmil[i*dl+l];
                                vjl[j*dl+l] += eri[ijkl] * dmik[i*dk+k];
                                vik[i*dk+k] -= eri[ijkl] * dmjl[j*dl+l];
                                vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                        } } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nra4kl_jk_s1il, J, K, I, L, s4);

static void nra4kl_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (i0 < l0) {
        } else if (i0 < k0) {
                if (j0 < l0) { // j < l <= i < k
                        DECLARE(v, i, l);
                        DEF_DM(j, k);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        v[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                } } } }
                                v += dil;
                        }
                } else { // l <= j < i < k
                        DECLARE(vil, i, l);
                        LOCATE(vjl, j, l);
                        DEF_DM(i, k);
                        DEF_DM(j, k);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        vjl[j*dl+l] += eri[ijkl] * dmik[i*dk+k];
                                        vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                } } } }
                                vjl += djl;
                                vil += dil;
                        }
                }
        } else if (j0 < l0) { // j < l < k <= i
                DECLARE(vil, i, l);
                LOCATE(vik, i, k);
                DEF_DM(j, k);
                DEF_DM(j, l);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                vik[i*dk+k] -= eri[ijkl] * dmjl[j*dl+l];
                        } } } }
                        vil += dil;
                        vik += dik;
                }
        } else if (j0 < k0) { // l <= j < k <= i
                DECLARE(vjl, j, l);
                LOCATE(vik, i, k);
                LOCATE(vil, i, l);
                DEF_DM(i, k);
                DEF_DM(j, k);
                DEF_DM(j, l);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vjl[j*dl+l] += eri[ijkl] * dmik[i*dk+k];
                                vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                vik[i*dk+k] -= eri[ijkl] * dmjl[j*dl+l];
                        } } } }
                        vjl += djl;
                        vil += dil;
                        vik += dik;
                }
        } else { // l < k <= j < i
                DECLARE(vjl, j, l);
                LOCATE(vik, i, k);
                LOCATE(vjk, j, k);
                LOCATE(vil, i, l);
                DEF_DM(i, l);
                DEF_DM(i, k);
                DEF_DM(j, l);
                DEF_DM(j, k);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vjk[j*dk+k] -= eri[ijkl] * dmil[i*dl+l];
                                vjl[j*dl+l] += eri[ijkl] * dmik[i*dk+k];
                                vik[i*dk+k] -= eri[ijkl] * dmjl[j*dl+l];
                                vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                        } } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nra4kl_jk_s2il, J, K, I, L, s4);

static void nra4kl_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                DEF_DM(k, i);
                DEF_DM(k, j);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                vki[k*di+i] += eri[ijkl] * dmlj[l*dj+j];
                                vlj[l*dj+j] -= eri[ijkl] * dmki[k*di+i];
                                vli[l*di+i] -= eri[ijkl] * dmkj[k*dj+j];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nra4kl_li_s1kj, L, I, K, J, s4);

static void nra4kl_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nrs2kl_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nrs2ij_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 < j0) {
        } else if (k0 < i0) {
                if (l0 < j0) { // l < j < k < i
                        DECLARE(v, k, j);
                        DEF_DM(l, i);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        v[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                } } } }
                                v += dkj;
                        }
                } else { // j <= l < k < i
                        DECLARE(vkj, k, j);
                        LOCATE(vlj, l, j);
                        DEF_DM(l, i);
                        DEF_DM(k, i);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                        vlj[l*dj+j] -= eri[ijkl] * dmki[k*di+i];
                                } } } }
                                vkj += dkj;
                                vlj += dlj;
                        }
                }
        } else if (l0 < j0) { // l < j < i <= k
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                vki[k*di+i] += eri[ijkl] * dmlj[l*dj+j];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                }
        } else if (l0 < i0) { // j <= l < i <= k
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                LOCATE(vlj, l, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                DEF_DM(k, i);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                vki[k*di+i] += eri[ijkl] * dmlj[l*dj+j];
                                vlj[l*dj+j] -= eri[ijkl] * dmki[k*di+i];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                }
        } else { // j < i <= l < k
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                DEF_DM(k, i);
                DEF_DM(k, j);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                vki[k*di+i] += eri[ijkl] * dmlj[l*dj+j];
                                vlj[l*dj+j] -= eri[ijkl] * dmki[k*di+i];
                                vli[l*di+i] -= eri[ijkl] * dmkj[k*dj+j];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nra4kl_li_s2kj, L, I, K, J, s4);

/*
 * aa4: 4-fold permutation symmetry with anti-symm for ij and anti-symm for kl
 */
static void nraa4_ji_s1kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_ji_s1kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vkl, k, l);
                LOCATE(vlk, l, k);
                DEF_DM(i, j);
                DEF_DM(j, i);
                int dij = di * dj;
                int i, j, k, l, ij, icomp;
                double *tdm = eri + dij * dkl * ncomp;
                double tmp;

                for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                tdm[ij] = dmji[j*di+i] - dmij[i*dj+j];
                        }
                }

                for (icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tmp = 0;
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        tmp += eri[ij] * tdm[ij];
                                }
                                vkl[k*dl+l] += tmp;
                                vlk[l*dk+k] -= tmp;
                                eri += dij;
                        } }
                        vkl += dkl;
                        vlk += dkl;
                }
        }
}
ADD_JKOP(nraa4_ji_s1kl, J, I, K, L, s4);

static void nraa4_ji_s2kl(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2ij_ji_s2kl(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nraa4_ji_s2kl, J, I, K, L, s4);

static void nraa4_lk_s1ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vij, i, j);
                LOCATE(vji, j, i);
                DEF_DM(k, l);
                DEF_DM(l, k);
                int i, j, k, l, ij, icomp;
                double *buf = eri + dij * dk * dl * ncomp;
                double tdm;

                for (icomp = 0; icomp < ncomp; icomp++) {

                        for (i = 0; i < dij; i++) { buf[i] = 0; }
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                                tdm = dmlk[l*dk+k] - dmkl[k*dl+l];
#pragma GCC ivdep
                                for (ij = 0; ij < dij; ij++) {
                                        buf[ij] += eri[ij] * tdm;
                                }
                                eri += dij;
                        } }

                        for (ij = 0, j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ij++) {
                                vij[i*dj+j] += buf[ij];
                                vji[ij] -= buf[ij];
                        } }
                        vij += dij;
                        vji += dij;
                }
        }
}
ADD_JKOP(nraa4_lk_s1ij, L, K, I, J, s4);

static void nraa4_lk_s2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nraa4_lk_s2ij, L, K, I, J, s4);

static void nraa4_lk_a2ij(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        nra2kl_lk_s1ij(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
}
ADD_JKOP(nraa4_lk_a2ij, L, K, I, J, s4);

static void nraa4_jk_s1il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_jk_s1il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vik, i, k);
                LOCATE(vil, i, l);
                LOCATE(vjk, j, k);
                LOCATE(vjl, j, l);
                DEF_DM(i, l);
                DEF_DM(i, k);
                DEF_DM(j, l);
                DEF_DM(j, k);
                int i, j, k, l, ijkl, icomp;

                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vjk[j*dk+k] += eri[ijkl] * dmil[i*dl+l];
                                vjl[j*dl+l] -= eri[ijkl] * dmik[i*dk+k];
                                vik[i*dk+k] -= eri[ijkl] * dmjl[j*dl+l];
                                vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                        } } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nraa4_jk_s1il, J, K, I, L, s4);

static void nraa4_jk_s2il(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_jk_s2il(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (i0 < l0) {
        } else if (i0 < k0) {
                if (j0 < l0) { // j < l <= i < k
                        DECLARE(v, i, l);
                        DEF_DM(j, k);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        v[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                } } } }
                                v += dil;
                        }
                } else { // l <= j < i < k
                        DECLARE(vil, i, l);
                        LOCATE(vjl, j, l);
                        DEF_DM(i, k);
                        DEF_DM(j, k);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        vjl[j*dl+l] -= eri[ijkl] * dmik[i*dk+k];
                                        vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                } } } }
                                vjl += djl;
                                vil += dil;
                        }
                }
        } else if (j0 < l0) { // j < l < k <= i
                DECLARE(vil, i, l);
                LOCATE(vik, i, k);
                DEF_DM(j, k);
                DEF_DM(j, l);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                vik[i*dk+k] -= eri[ijkl] * dmjl[j*dl+l];
                        } } } }
                        vil += dil;
                        vik += dik;
                }
        } else if (j0 < k0) { // l <= j < k <= i
                DECLARE(vjl, j, l);
                LOCATE(vik, i, k);
                LOCATE(vil, i, l);
                DEF_DM(i, k);
                DEF_DM(j, k);
                DEF_DM(j, l);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vjl[j*dl+l] -= eri[ijkl] * dmik[i*dk+k];
                                vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                                vik[i*dk+k] -= eri[ijkl] * dmjl[j*dl+l];
                        } } } }
                        vjl += djl;
                        vil += dil;
                        vik += dik;
                }
        } else { // l < k <= j < i
                DECLARE(vjl, j, l);
                LOCATE(vik, i, k);
                LOCATE(vjk, j, k);
                LOCATE(vil, i, l);
                DEF_DM(i, l);
                DEF_DM(i, k);
                DEF_DM(j, l);
                DEF_DM(j, k);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vjk[j*dk+k] += eri[ijkl] * dmil[i*dl+l];
                                vjl[j*dl+l] -= eri[ijkl] * dmik[i*dk+k];
                                vik[i*dk+k] -= eri[ijkl] * dmjl[j*dl+l];
                                vil[i*dl+l] += eri[ijkl] * dmjk[j*dk+k];
                        } } } }
                        vjk += djk;
                        vjl += djl;
                        vik += dik;
                        vil += dil;
                }
        }
}
ADD_JKOP(nraa4_jk_s2il, J, K, I, L, s4);

static void nraa4_li_s1kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_li_s1kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else {
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                DEF_DM(k, i);
                DEF_DM(k, j);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                vki[k*di+i] -= eri[ijkl] * dmlj[l*dj+j];
                                vlj[l*dj+j] -= eri[ijkl] * dmki[k*di+i];
                                vli[l*di+i] += eri[ijkl] * dmkj[k*dj+j];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nraa4_li_s1kj, L, I, K, J, s4);

static void nraa4_li_s2kj(double *eri, double *dm, JKArray *out, int *shls,
                           int i0, int i1, int j0, int j1,
                           int k0, int k1, int l0, int l1)
{
        if (i0 == j0) {
                nra2kl_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 == l0) {
                nra2ij_li_s2kj(eri, dm, out, shls, i0, i1, j0, j1, k0, k1, l0, l1);
        } else if (k0 < j0) {
        } else if (k0 < i0) {
                if (l0 < j0) { // l < j < k < i
                        DECLARE(v, k, j);
                        DEF_DM(l, i);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        v[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                } } } }
                                v += dkj;
                        }
                } else { // j <= l < k < i
                        DECLARE(vkj, k, j);
                        LOCATE(vlj, l, j);
                        DEF_DM(l, i);
                        DEF_DM(k, i);
                        int i, j, k, l, ijkl, icomp;
                        for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                                for (l = 0; l < dl; l++) {
                                for (k = 0; k < dk; k++) {
                                for (j = 0; j < dj; j++) {
                                for (i = 0; i < di; i++, ijkl++) {
                                        vkj[k*dj+j] += eri[ijkl] *dmli[l*di+i];
                                        vlj[l*dj+j] -= eri[ijkl] *dmki[k*di+i];
                                } } } }
                                vkj += dkj;
                                vlj += dlj;
                        }
                }
        } else if (l0 < j0) { // l < j < i <= k
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                vki[k*di+i] -= eri[ijkl] * dmlj[l*dj+j];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                }
        } else if (l0 < i0) { // j <= l < i <= k
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                LOCATE(vlj, l, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                DEF_DM(k, i);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                vki[k*di+i] -= eri[ijkl] * dmlj[l*dj+j];
                                vlj[l*dj+j] -= eri[ijkl] * dmki[k*di+i];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                }
        } else { // j < i <= l < k
                DECLARE(vki, k, i);
                LOCATE(vkj, k, j);
                LOCATE(vli, l, i);
                LOCATE(vlj, l, j);
                DEF_DM(l, i);
                DEF_DM(l, j);
                DEF_DM(k, i);
                DEF_DM(k, j);
                int i, j, k, l, ijkl, icomp;
                for (ijkl = 0, icomp = 0; icomp < ncomp; icomp++) {
                        for (l = 0; l < dl; l++) {
                        for (k = 0; k < dk; k++) {
                        for (j = 0; j < dj; j++) {
                        for (i = 0; i < di; i++, ijkl++) {
                                vkj[k*dj+j] += eri[ijkl] * dmli[l*di+i];
                                vki[k*di+i] -= eri[ijkl] * dmlj[l*dj+j];
                                vlj[l*dj+j] -= eri[ijkl] * dmki[k*di+i];
                                vli[l*di+i] += eri[ijkl] * dmkj[k*dj+j];
                        } } } }
                        vkj += dkj;
                        vki += dki;
                        vlj += dlj;
                        vli += dli;
                }
        }
}
ADD_JKOP(nraa4_li_s2kj, L, I, K, J, s4);

