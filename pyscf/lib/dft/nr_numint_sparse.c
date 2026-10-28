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
#include <math.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "cint.h"
#include "vhf/fblas.h"

#define ALIGNMENT       8
#define UNROLL_SIZE     (ALIGNMENT*7)
#define BLKSIZE         (ALIGNMENT*7)
#define BOXSIZE1_M      (ALIGNMENT*14)
#define BOXSIZE1_N      96
#define BOXSIZE1_K      BOXSIZE1_N
#define NCTR_CART       80
#define MAX_THREADS     256

int CVHFshls_block_partition(int *block_loc, int *shls_slice, int *ao_loc,
                             int block_size);

static void _dot_ao_dm_l1(double *out, double *ao, double *dm,
                          int nao, size_t ngrids, int nbas, int ig0, int ig1,
                          int ish0, int ish1, int jsh0, int jsh1, int nbins,
                          uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc)
{
        size_t Nao = nao;
        int ig, ish, jsh, i0, i1, i, j, box_id, n;
        uint8_t si_i, si_j, nbins_i;  // be careful with overflow
        size_t i_addr, j_addr;
        double dm_val;
        double s8[UNROLL_SIZE];
        double *dm_j;

        for (ig = ig0; ig < ig1; ig+=UNROLL_SIZE) {
                box_id = ig / BLKSIZE;
                for (jsh = jsh0; jsh < jsh1; jsh++) {
                        si_j = screen_index[box_id * nbas + jsh];
                        if (si_j != 0) {
                                if (nbins > si_j) {
                                        nbins_i = nbins - si_j;
                                } else {
                                        nbins_i = 1;
                                }
                                for (j = ao_loc[jsh]; j < ao_loc[jsh+1]; j++) {
                                        dm_j = dm + j;
                                        j_addr = j * ngrids + ig;
                                        for (n = 0; n < UNROLL_SIZE; n++) {
                                                s8[n] = out[j_addr+n];
                                        }
                                        for (ish = ish0; ish < ish1; ish++) {
                                                si_i = screen_index[box_id * nbas + ish];
                                                if (si_i >= nbins_i &&
                                                    pair_mask[ish*nbas+jsh]) {
i0 = ao_loc[ish];
i1 = ao_loc[ish+1];
switch (i1 - i0) {
case 1:
        for (i = i0; i < i0 + 1; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        } break;
case 2:
        for (i = i0; i < i0 + 2; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        } break;
case 3:
        for (i = i0; i < i0 + 3; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        } break;
case 5:
        for (i = i0; i < i0 + 5; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        } break;
case 6:
        for (i = i0; i < i0 + 6; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        } break;
case 7:
        for (i = i0; i < i0 + 7; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        } break;
case 10:
        for (i = i0; i < i0 + 10; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        } break;
default:
        for (i = i0; i < i1; i++) {
                dm_val = dm_j[i * Nao];
                i_addr = i * ngrids + ig;
                for (n = 0; n < UNROLL_SIZE; n++) {
                        s8[n] += ao[i_addr+n] * dm_val;
                }
        }
}
                                                }
                                        }
                                        for (n = 0; n < UNROLL_SIZE; n++) {
                                                out[j_addr+n] = s8[n];
                                        }
                                }
                        }
                }
        }
}

static void _dot_ao_dm_frac(double *out, double *ao, double *dm,
                            int nao, size_t ngrids, int nbas, int ig0, int nbins,
                            uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc)
{
        size_t Nao = nao;
        int ngrids_rest = ngrids - ig0;
        int ish, jsh, i0, i1, i, j, n;
        uint8_t si_i, si_j, nbins_i;
        size_t i_addr, j_addr;
        double dm_val;
        double s8[UNROLL_SIZE];
        double *dm_j;

        int box_id = ig0 / BLKSIZE;
        for (jsh = 0; jsh < nbas; jsh++) {
                si_j = screen_index[box_id * nbas + jsh];
                if (si_j != 0) {
                        if (nbins > si_j) {
                                nbins_i = nbins - si_j;
                        } else {
                                nbins_i = 1;
                        }
                        for (j = ao_loc[jsh]; j < ao_loc[jsh+1]; j++) {
                                dm_j = dm + j;
                                j_addr = j * ngrids + ig0;
                                for (n = 0; n < ngrids_rest; n++) {
                                        s8[n] = 0;
                                }
                                for (ish = 0; ish < nbas; ish++) {
                                        si_i = screen_index[box_id * nbas + ish];
                                        if (si_i >= nbins_i && pair_mask[ish*nbas+jsh]) {
                                                i0 = ao_loc[ish];
                                                i1 = ao_loc[ish+1];
                                                for (i = i0; i < i1; i++) {
                                                        dm_val = dm_j[i * Nao];
                                                        i_addr = i * ngrids + ig0;
                                                        for (n = 0; n < ngrids_rest; n++) {
                                                                s8[n] += ao[i_addr+n] * dm_val;
                                                        }
                                                }
                                        }
                                }
                                for (n = 0; n < ngrids_rest; n++) {
                                        out[j_addr+n] = s8[n];
                                }
                        }
                }
        }
}

// return number of non-zero values in mask array
static void mask_l1_abstract(uint8_t *out, uint8_t *mask, int *box_loc,
                             int nbox, int ngrids, int nbas)
{
        int i, m, ig, box_id, ig0, ig1, i0, i1, n;
        int with_value;
        for (n = 0, ig0 = 0; ig0 < ngrids; ig0 += BOXSIZE1_M) {
        for (box_id = 0; box_id < nbox; box_id++, n++) {
                i0 = box_loc[box_id];
                i1 = box_loc[box_id+1];
                ig1 = MIN(ig0+BOXSIZE1_M, ngrids);
                with_value = 0;
                for (i = i0; i < i1; i++) {
                for (ig = ig0; ig < ig1; ig+=BLKSIZE) {
                        m = ig / BLKSIZE;
                        if (mask[m*nbas+i] != 0) {
                                with_value = 1;
                                goto next_l1_box;
                        }
                } }
next_l1_box:
                out[n] = with_value;
        } }
}


void VXCdot_ao_dm_sparse(double *out, double *ao, double *dm,
                         int nao, int ngrids, int nbas, int nbins,
                         uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc)
{
        size_t Ngrids = ngrids;
        int shls_slice[2] = {0, nbas};
        int *box_l1_loc = malloc(sizeof(int) * (nbas+1));
        int nbox_l1 = CVHFshls_block_partition(box_l1_loc, shls_slice, ao_loc, BOXSIZE1_N);
        int mask_l1_size = (ngrids + BOXSIZE1_M - 1)/BOXSIZE1_M * nbox_l1;
        uint8_t *mask_l1 = malloc(sizeof(uint8_t) * mask_l1_size);
        mask_l1_abstract(mask_l1, screen_index, box_l1_loc, nbox_l1, ngrids, nbas);
        int ngrids_align_down = (ngrids / BLKSIZE) * BLKSIZE;

        if (nao * 2 < ngrids) {
#pragma omp parallel
{
                int ig, j, j0, j1, jsh0, jsh1, ib, jb, ig0, ig1, ig_box2;
#pragma omp for schedule(dynamic)
                for (ig0 = 0; ig0 < ngrids_align_down; ig0+=BOXSIZE1_M) {
                        ig1 = MIN(ig0 + BOXSIZE1_M, ngrids_align_down);
                        ig_box2 = ig0 / BOXSIZE1_M;
                        for (jb = 0; jb < nbox_l1; jb++) {
                                if (mask_l1[ig_box2 * nbox_l1 + jb]) {
                                        jsh0 = box_l1_loc[jb];
                                        jsh1 = box_l1_loc[jb+1];
                                        j0 = ao_loc[jsh0];
                                        j1 = ao_loc[jsh1];
                                        for (j = j0; j < j1; j++) {
                                        for (ig = ig0; ig < ig1; ig++) {
                                                out[j*Ngrids+ig] = 0;
                                        } }
                                        for (ib = 0; ib < nbox_l1; ib++) {
                                                if (mask_l1[ig_box2 * nbox_l1 + ib]) {
_dot_ao_dm_l1(out, ao, dm, nao, ngrids, nbas, ig0, ig1,
              box_l1_loc[ib], box_l1_loc[ib+1], jsh0, jsh1,
              nbins, screen_index, pair_mask, ao_loc);
                                                }
                                        }
                                }
                        }
                }
}
        } else {
#pragma omp parallel
{
                int ig, j, j0, j1, jsh0, jsh1, ib, jb, ig0, ig1, ig_box2;
#pragma omp for schedule(dynamic)
                for (jb = 0; jb < nbox_l1; jb++) {
                        jsh0 = box_l1_loc[jb];
                        jsh1 = box_l1_loc[jb+1];
                        j0 = ao_loc[jsh0];
                        j1 = ao_loc[jsh1];
                        for (ig0 = 0; ig0 < ngrids_align_down; ig0+=BOXSIZE1_M) {
                                ig1 = MIN(ig0 + BOXSIZE1_M, ngrids_align_down);
                                ig_box2 = ig0 / BOXSIZE1_M;
                                if (mask_l1[ig_box2 * nbox_l1 + jb]) {
                                        for (j = j0; j < j1; j++) {
                                        for (ig = ig0; ig < ig1; ig++) {
                                                out[j*Ngrids+ig] = 0;
                                        } }
                                        for (ib = 0; ib < nbox_l1; ib++) {
                                                if (mask_l1[ig_box2 * nbox_l1 + ib]) {
_dot_ao_dm_l1(out, ao, dm, nao, ngrids, nbas, ig0, ig1,
              box_l1_loc[ib], box_l1_loc[ib+1], jsh0, jsh1,
              nbins, screen_index, pair_mask, ao_loc);
                                                }
                                        }
                                }
                        }
                }
}
        }
        if (ngrids_align_down < ngrids) {
                _dot_ao_dm_frac(out, ao, dm, nao, ngrids, nbas, ngrids_align_down,
                                nbins, screen_index, pair_mask, ao_loc);
        }
        free(box_l1_loc);
        free(mask_l1);
}

static void _dot_aow_ao_l1(double *out, double *bra, double *ket, double *wv,
                           int nao, size_t ngrids, int nbas, int ig0, int ig1,
                           int ish0, int ish1, int jsh0, int jsh1, int nj,
                           int ioff, int joff, int nbins, uint8_t *screen_index,
                           uint8_t *pair_mask, int *ao_loc)
{
        int ish, jsh, i0, i1, j0, j1, i, j, ij, n;
        uint8_t si_i0, si_i1, si_j0, si_j1, nbins_j0, nbins_j1;
        size_t i_addr, j_addr;
        double braw[BOXSIZE1_M];
        double s8[ALIGNMENT];
        int i_pattern, j_pattern;
        int gblk0 = ig0 / BLKSIZE;

        for (ish = ish0; ish < ish1; ish++) {
                si_i0 = screen_index[(gblk0+0) * nbas + ish];
                si_i1 = screen_index[(gblk0+1) * nbas + ish];
                i_pattern = (si_i0 != 0) | ((si_i1 != 0) << 1);
                if (!i_pattern) {
                        goto next_i;
                }
                if (nbins > si_i0) {
                        nbins_j0 = nbins - si_i0;
                } else {
                        nbins_j0 = 1;
                }
                if (nbins > si_i1) {
                        nbins_j1 = nbins - si_i1;
                } else {
                        nbins_j1 = 1;
                }
                i0 = ao_loc[ish];
                i1 = ao_loc[ish+1];
                for (i = i0; i < i1; i++) {
                        i_addr = i * ngrids + ig0;
                        switch (i_pattern) {
                        case 0:
                                goto next_i;
                        case 1 :
                                for (n = 0; n < ALIGNMENT; n++) {
                                        braw[n            ] = bra[i_addr+n            ] * wv[ig0+n            ];
                                        braw[n+ALIGNMENT*1] = bra[i_addr+n+ALIGNMENT*1] * wv[ig0+n+ALIGNMENT*1];
                                        braw[n+ALIGNMENT*2] = bra[i_addr+n+ALIGNMENT*2] * wv[ig0+n+ALIGNMENT*2];
                                        braw[n+ALIGNMENT*3] = bra[i_addr+n+ALIGNMENT*3] * wv[ig0+n+ALIGNMENT*3];
                                        braw[n+ALIGNMENT*4] = bra[i_addr+n+ALIGNMENT*4] * wv[ig0+n+ALIGNMENT*4];
                                        braw[n+ALIGNMENT*5] = bra[i_addr+n+ALIGNMENT*5] * wv[ig0+n+ALIGNMENT*5];
                                        braw[n+ALIGNMENT*6] = bra[i_addr+n+ALIGNMENT*6] * wv[ig0+n+ALIGNMENT*6];
                                }
                                break;
                        case 2 :
                                for (n = 0; n < ALIGNMENT; n++) {
                                        braw[n+ALIGNMENT*7 ] = bra[i_addr+n+ALIGNMENT*7 ] * wv[ig0+n+ALIGNMENT*7 ];
                                        braw[n+ALIGNMENT*8 ] = bra[i_addr+n+ALIGNMENT*8 ] * wv[ig0+n+ALIGNMENT*8 ];
                                        braw[n+ALIGNMENT*9 ] = bra[i_addr+n+ALIGNMENT*9 ] * wv[ig0+n+ALIGNMENT*9 ];
                                        braw[n+ALIGNMENT*10] = bra[i_addr+n+ALIGNMENT*10] * wv[ig0+n+ALIGNMENT*10];
                                        braw[n+ALIGNMENT*11] = bra[i_addr+n+ALIGNMENT*11] * wv[ig0+n+ALIGNMENT*11];
                                        braw[n+ALIGNMENT*12] = bra[i_addr+n+ALIGNMENT*12] * wv[ig0+n+ALIGNMENT*12];
                                        braw[n+ALIGNMENT*13] = bra[i_addr+n+ALIGNMENT*13] * wv[ig0+n+ALIGNMENT*13];
                                }
                                break;
                        case 3 :
                                for (n = 0; n < ALIGNMENT; n++) {
                                        braw[n             ] = bra[i_addr+n             ] * wv[ig0+n             ];
                                        braw[n+ALIGNMENT*1 ] = bra[i_addr+n+ALIGNMENT*1 ] * wv[ig0+n+ALIGNMENT*1 ];
                                        braw[n+ALIGNMENT*2 ] = bra[i_addr+n+ALIGNMENT*2 ] * wv[ig0+n+ALIGNMENT*2 ];
                                        braw[n+ALIGNMENT*3 ] = bra[i_addr+n+ALIGNMENT*3 ] * wv[ig0+n+ALIGNMENT*3 ];
                                        braw[n+ALIGNMENT*4 ] = bra[i_addr+n+ALIGNMENT*4 ] * wv[ig0+n+ALIGNMENT*4 ];
                                        braw[n+ALIGNMENT*5 ] = bra[i_addr+n+ALIGNMENT*5 ] * wv[ig0+n+ALIGNMENT*5 ];
                                        braw[n+ALIGNMENT*6 ] = bra[i_addr+n+ALIGNMENT*6 ] * wv[ig0+n+ALIGNMENT*6 ];
                                        braw[n+ALIGNMENT*7 ] = bra[i_addr+n+ALIGNMENT*7 ] * wv[ig0+n+ALIGNMENT*7 ];
                                        braw[n+ALIGNMENT*8 ] = bra[i_addr+n+ALIGNMENT*8 ] * wv[ig0+n+ALIGNMENT*8 ];
                                        braw[n+ALIGNMENT*9 ] = bra[i_addr+n+ALIGNMENT*9 ] * wv[ig0+n+ALIGNMENT*9 ];
                                        braw[n+ALIGNMENT*10] = bra[i_addr+n+ALIGNMENT*10] * wv[ig0+n+ALIGNMENT*10];
                                        braw[n+ALIGNMENT*11] = bra[i_addr+n+ALIGNMENT*11] * wv[ig0+n+ALIGNMENT*11];
                                        braw[n+ALIGNMENT*12] = bra[i_addr+n+ALIGNMENT*12] * wv[ig0+n+ALIGNMENT*12];
                                        braw[n+ALIGNMENT*13] = bra[i_addr+n+ALIGNMENT*13] * wv[ig0+n+ALIGNMENT*13];
                                }
                        }

                        for (jsh = jsh0; jsh < jsh1; jsh++) {
                                if (!pair_mask[ish*nbas+jsh]) {
                                        goto next_j;
                                }
                                si_j0 = screen_index[(gblk0+0) * nbas + jsh];
                                si_j1 = screen_index[(gblk0+1) * nbas + jsh];
                                j_pattern = (si_j0 >= nbins_j0) | ((si_j1 >= nbins_j1) << 1);
                                switch (j_pattern & i_pattern) {
                                case 0 :
                                        goto next_j;
                                case 1 :
                                        j0 = ao_loc[jsh];
                                        j1 = ao_loc[jsh+1];
                                        ij = (i - ioff) * nj - joff;
                                        for (j = j0; j < j1; j++) {
                                                j_addr = j * ngrids + ig0;
                                                for (n = 0; n < ALIGNMENT; n++) {
                                                        s8[n] = out[(ij+j)*ALIGNMENT+n];
                                                        s8[n] += braw[n             ] * ket[j_addr+n             ];
                                                        s8[n] += braw[n+ALIGNMENT*1 ] * ket[j_addr+n+ALIGNMENT*1 ];
                                                        s8[n] += braw[n+ALIGNMENT*2 ] * ket[j_addr+n+ALIGNMENT*2 ];
                                                        s8[n] += braw[n+ALIGNMENT*3 ] * ket[j_addr+n+ALIGNMENT*3 ];
                                                        s8[n] += braw[n+ALIGNMENT*4 ] * ket[j_addr+n+ALIGNMENT*4 ];
                                                        s8[n] += braw[n+ALIGNMENT*5 ] * ket[j_addr+n+ALIGNMENT*5 ];
                                                        s8[n] += braw[n+ALIGNMENT*6 ] * ket[j_addr+n+ALIGNMENT*6 ];
                                                        out[(ij+j)*ALIGNMENT+n] = s8[n];
                                                }
                                        }
                                        break;
                                case 2 :
                                        j0 = ao_loc[jsh];
                                        j1 = ao_loc[jsh+1];
                                        ij = (i - ioff) * nj - joff;
                                        for (j = j0; j < j1; j++) {
                                                j_addr = j * ngrids + ig0;
                                                for (n = 0; n < ALIGNMENT; n++) {
                                                        s8[n] = out[(ij+j)*ALIGNMENT+n];
                                                        s8[n] += braw[n+ALIGNMENT*7 ] * ket[j_addr+n+ALIGNMENT*7 ];
                                                        s8[n] += braw[n+ALIGNMENT*8 ] * ket[j_addr+n+ALIGNMENT*8 ];
                                                        s8[n] += braw[n+ALIGNMENT*9 ] * ket[j_addr+n+ALIGNMENT*9 ];
                                                        s8[n] += braw[n+ALIGNMENT*10] * ket[j_addr+n+ALIGNMENT*10];
                                                        s8[n] += braw[n+ALIGNMENT*11] * ket[j_addr+n+ALIGNMENT*11];
                                                        s8[n] += braw[n+ALIGNMENT*12] * ket[j_addr+n+ALIGNMENT*12];
                                                        s8[n] += braw[n+ALIGNMENT*13] * ket[j_addr+n+ALIGNMENT*13];
                                                        out[(ij+j)*ALIGNMENT+n] = s8[n];
                                                }
                                        }
                                        break;
                                case 3 :
                                        j0 = ao_loc[jsh];
                                        j1 = ao_loc[jsh+1];
                                        ij = (i - ioff) * nj - joff;
                                        for (j = j0; j < j1; j++) {
                                                j_addr = j * ngrids + ig0;
                                                for (n = 0; n < ALIGNMENT; n++) {
                                                        s8[n] = out[(ij+j)*ALIGNMENT+n];
                                                        s8[n] += braw[n             ] * ket[j_addr+n             ];
                                                        s8[n] += braw[n+ALIGNMENT*1 ] * ket[j_addr+n+ALIGNMENT*1 ];
                                                        s8[n] += braw[n+ALIGNMENT*2 ] * ket[j_addr+n+ALIGNMENT*2 ];
                                                        s8[n] += braw[n+ALIGNMENT*3 ] * ket[j_addr+n+ALIGNMENT*3 ];
                                                        s8[n] += braw[n+ALIGNMENT*4 ] * ket[j_addr+n+ALIGNMENT*4 ];
                                                        s8[n] += braw[n+ALIGNMENT*5 ] * ket[j_addr+n+ALIGNMENT*5 ];
                                                        s8[n] += braw[n+ALIGNMENT*6 ] * ket[j_addr+n+ALIGNMENT*6 ];
                                                        s8[n] += braw[n+ALIGNMENT*7 ] * ket[j_addr+n+ALIGNMENT*7 ];
                                                        s8[n] += braw[n+ALIGNMENT*8 ] * ket[j_addr+n+ALIGNMENT*8 ];
                                                        s8[n] += braw[n+ALIGNMENT*9 ] * ket[j_addr+n+ALIGNMENT*9 ];
                                                        s8[n] += braw[n+ALIGNMENT*10] * ket[j_addr+n+ALIGNMENT*10];
                                                        s8[n] += braw[n+ALIGNMENT*11] * ket[j_addr+n+ALIGNMENT*11];
                                                        s8[n] += braw[n+ALIGNMENT*12] * ket[j_addr+n+ALIGNMENT*12];
                                                        s8[n] += braw[n+ALIGNMENT*13] * ket[j_addr+n+ALIGNMENT*13];
                                                        out[(ij+j)*ALIGNMENT+n] = s8[n];
                                                }
                                        }
                                }
next_j:;
                        }
                }
next_i:;
        }
}

static void _dot_aow_ao_frac(double *out, double *bra, double *ket, double *wv,
                             int nao, size_t ngrids, int nbas, int ig0,
                             int ish0, int ish1, int jsh0, int jsh1, int nj,
                             int ioff, int joff, int nbins, uint8_t *screen_index,
                             uint8_t *pair_mask, int *ao_loc)
{
        int ish, jsh, i0, i1, j0, j1, i, j, ij, k, n, ig, gblk0;
        int ng1, ng1_aligned, ng1_rest;
        uint8_t si_i, si_j, nbins_j;
        size_t i_addr, j_addr;
        double braw[BLKSIZE];
        double s8[ALIGNMENT];

        for (ig = ig0; ig < ngrids; ig+=BLKSIZE) {
                gblk0 = ig / BLKSIZE;
                ng1 = MIN(BLKSIZE, ngrids - ig);
                ng1_aligned = ng1 & (-(uint32_t)ALIGNMENT);
                for (ish = ish0; ish < ish1; ish++) {
                        si_i = screen_index[gblk0 * nbas + ish];
                        if (si_i != 0) {
                                if (nbins > si_i) {
                                        nbins_j = nbins - si_i;
                                } else {
                                        nbins_j = 1;
                                }
                                i0 = ao_loc[ish];
                                i1 = ao_loc[ish+1];
                                for (i = i0; i < i1; i++) {
i_addr = i * ngrids + ig;
for (k = 0; k < ng1_aligned; k+=ALIGNMENT) {
for (n = 0; n < ALIGNMENT; n++) {
         braw[n+k] = bra[i_addr+n+k] * wv[ig+n+k];
} }

for (jsh = jsh0; jsh < jsh1; jsh++) {
        si_j = screen_index[gblk0 * nbas + jsh];
        if (si_j >= nbins_j && pair_mask[ish*nbas+jsh]) {
                j0 = ao_loc[jsh];
                j1 = ao_loc[jsh+1];
                ij = (i - ioff) * nj - joff;
                for (j = j0; j < j1; j++) {
                        j_addr = j * ngrids + ig;
                        for (n = 0; n < ALIGNMENT; n++) {
                                s8[n] = out[(ij+j)*ALIGNMENT+n];
                        }
                        for (k = 0; k < ng1_aligned; k+=ALIGNMENT) {
                        for (n = 0; n < ALIGNMENT; n++) {
                                s8[n] += braw[n+k] * ket[j_addr+n+k];
                        } }
                        for (n = 0; n < ALIGNMENT; n++) {
                                out[(ij+j)*ALIGNMENT+n] = s8[n];
                        }
                }
        }
}
                                }
                        }
                }

                if (ng1_aligned < ng1) {
                        ng1_rest = ng1 - ng1_aligned;
                        for (ish = ish0; ish < ish1; ish++) {
                                si_i = screen_index[gblk0 * nbas + ish];
                                if (si_i != 0) {
                                        if (nbins > si_i) {
                                                nbins_j = nbins - si_i;
                                        } else {
                                                nbins_j = 1;
                                        }
                                        i0 = ao_loc[ish];
                                        i1 = ao_loc[ish+1];
                                        for (i = i0; i < i1; i++) {
i_addr = i * ngrids + ng1_aligned;
for (k = 0; k < ng1_rest; k++) {
         braw[k] = bra[i_addr+k] * wv[ng1_aligned+k];
}

for (jsh = jsh0; jsh < jsh1; jsh++) {
        si_j = screen_index[gblk0 * nbas + jsh];
        if (si_j >= nbins_j && pair_mask[ish*nbas+jsh]) {
                j0 = ao_loc[jsh];
                j1 = ao_loc[jsh+1];
                ij = (i - ioff) * nj - joff;
                for (j = j0; j < j1; j++) {
                        j_addr = j * ngrids + ng1_aligned;
                        s8[0] = 0.;
                        for (k = 0; k < ng1_rest; k++) {
                                s8[0] += braw[k] * ket[j_addr+k];
                        }
                        out[(ij+j)*ALIGNMENT] += s8[0];
                }
        }
}
                                        }
                                }
                        }
                }
        }
}

void VXCdot_aow_ao_dense(double *out, double *bra, double *ket, double *wv,
                         int nao, int ngrids)
{
        const size_t Nao = nao;
        const size_t Ngrids = ngrids;
        const int nao_blksize = 64;
        const int ngrids_blksize = 256;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D1 = 1;
#pragma omp parallel
{
        int i0, ig, i, di, ig0, ig1, dg;
        double *buf = malloc(sizeof(double) * (ngrids_blksize * nao_blksize + ALIGNMENT));
        double *braw = (double *)((uintptr_t)(buf + ALIGNMENT - 1) & (-(uintptr_t)(ALIGNMENT*8)));
        double *pbra;
#pragma omp for schedule(dynamic) nowait
        for (i0 = 0; i0 < nao; i0+=nao_blksize) {
                di = MIN(nao_blksize, nao - i0);
                for (ig0 = 0; ig0 < ngrids; ig0+=ngrids_blksize) {
                        ig1 = MIN(ig0+ngrids_blksize, ngrids);
                        pbra = bra + i0 * Ngrids + ig0;
                        dg = ig1 - ig0;
                        for (i = 0; i < di; i++) {
                        for (ig = 0; ig < dg; ig++) {
                                braw[i*ngrids_blksize+ig] = pbra[i*Ngrids+ig] * wv[ig0+ig];
                        } }
                        dgemm_(&TRANS_T, &TRANS_N, &nao, &di, &dg,
                               &D1, ket+ig0, &ngrids, braw, &ngrids_blksize,
                               &D1, out+i0*Nao, &nao);
                }
        }
        free(buf);
}
}

/* vv[nao,nao] = bra[i,nao] * ket[i,nao] */
void VXCdot_aow_ao_sparse(double *out, double *bra, double *ket, double *wv,
                          int nao, int ngrids, int nbas, int hermi, int nbins,
                          uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc)
{
        size_t Nao = nao;
        int shls_slice[2] = {0, nbas};
        int *box_l1_loc = malloc(sizeof(int) * (nbas+1));
        int nbox_l1 = CVHFshls_block_partition(box_l1_loc, shls_slice, ao_loc, BOXSIZE1_N);
        int mask_l1_size = (ngrids + BOXSIZE1_M - 1)/BOXSIZE1_M * nbox_l1;
        uint8_t *mask_l1 = malloc(sizeof(uint8_t) * mask_l1_size);
        mask_l1_abstract(mask_l1, screen_index, box_l1_loc, nbox_l1, ngrids, nbas);
        int ngrids_align_down = (ngrids / BOXSIZE1_M) * BOXSIZE1_M;

#pragma omp parallel
{
        int ijb, ib, jb, ib0, jb0, ib1, jb1, ig0, ig1, ig_box2;
        int ish0, ish1, jsh0, jsh1, i0, i1, j0, j1, ni, nj, i, j, n;
        double s;
        double *pout;
        double *buf = malloc(sizeof(double) * (BOXSIZE1_N*BOXSIZE1_N*ALIGNMENT+ALIGNMENT));
        double *outbuf = (double *)((uintptr_t)(buf+ALIGNMENT-1) & (-(uintptr_t)(ALIGNMENT*sizeof(double))));
#pragma omp for schedule(dynamic, 4) nowait
        for (ijb = 0; ijb < nbox_l1*nbox_l1; ijb++) {
                ib = ijb / nbox_l1;
                jb = ijb % nbox_l1;
                if (hermi && ib < jb) {
                        continue;
                }
                ib0 = ib;
                jb0 = jb;
                ib1 = ib + 1;
                jb1 = jb + 1;
                ish0 = box_l1_loc[ib0];
                jsh0 = box_l1_loc[jb0];
                ish1 = box_l1_loc[ib1];
                jsh1 = box_l1_loc[jb1];
                i0 = ao_loc[ish0];
                i1 = ao_loc[ish1];
                j0 = ao_loc[jsh0];
                j1 = ao_loc[jsh1];
                ni = i1 - i0;
                nj = j1 - j0;
                for (n = 0; n < ni * nj * ALIGNMENT; n++) {
                        outbuf[n] = 0;
                }

                for (ig0 = 0; ig0 < ngrids_align_down; ig0+=BOXSIZE1_M) {
                        ig1 = MIN(ig0 + BOXSIZE1_M, ngrids_align_down);
                        ig_box2 = ig0 / BOXSIZE1_M;
                        if (mask_l1[ig_box2 * nbox_l1 + jb] &&
                            mask_l1[ig_box2 * nbox_l1 + ib]) {
_dot_aow_ao_l1(outbuf, bra, ket, wv, nao, ngrids, nbas, ig0, ig1,
               ish0, ish1, jsh0, jsh1, nj, i0, j0, nbins, screen_index,
               pair_mask, ao_loc);
                        }
                }
                if (ngrids_align_down < ngrids) {
                        _dot_aow_ao_frac(outbuf, bra, ket, wv, nao, ngrids, nbas,
                                         ngrids_align_down, ish0, ish1, jsh0, jsh1,
                                         nj, i0, j0, nbins, screen_index,
                                         pair_mask, ao_loc);
                }

                for (i = 0; i < ni; i++) {
                for (j = 0; j < nj; j++) {
                        s = 0;
                        pout = outbuf + (i*nj+j)*ALIGNMENT;
                        for (n = 0; n < ALIGNMENT; n++) {
                                s += pout[n];
                        }
                        out[(i0+i)*Nao+j0+j] += s;
                } }
        }
        free(buf);
}
        free(box_l1_loc);
        free(mask_l1);

        if (hermi != 0) {
                NPdsymm_triu(nao, out, hermi);
        }
}

static void _dot_ao_ao_l1(double *out, double *bra, double *ket,
                          int nao, size_t ngrids, int nbas, int ig0, int ig1,
                          int ish0, int ish1, int jsh0, int jsh1, int nj,
                          int ioff, int joff, int nbins, uint8_t *screen_index,
                          uint8_t *pair_mask, int *ao_loc)
{
        int ish, jsh, i0, i1, j0, j1, i, j, ij, n;
        uint8_t si_i0, si_i1, si_j0, si_j1, nbins_j0, nbins_j1;
        size_t i_addr, j_addr;
        double s8[ALIGNMENT];
        int i_pattern, j_pattern;
        int gblk0 = ig0 / BLKSIZE;

        for (ish = ish0; ish < ish1; ish++) {
                si_i0 = screen_index[(gblk0+0) * nbas + ish];
                si_i1 = screen_index[(gblk0+1) * nbas + ish];
                i_pattern = (si_i0 != 0) | ((si_i1 != 0) << 1);
                if (i_pattern) {
                        if (nbins > si_i0) {
                                nbins_j0 = nbins - si_i0;
                        } else {
                                nbins_j0 = 1;
                        }
                        if (nbins > si_i1) {
                                nbins_j1 = nbins - si_i1;
                        } else {
                                nbins_j1 = 1;
                        }
                        i0 = ao_loc[ish];
                        i1 = ao_loc[ish+1];
                        for (i = i0; i < i1; i++) {
                                i_addr = i * ngrids + ig0;
for (jsh = jsh0; jsh < jsh1; jsh++) {
        if (!pair_mask[ish*nbas+jsh]) {
                goto next_j;
        }
        si_j0 = screen_index[(gblk0+0) * nbas + jsh];
        si_j1 = screen_index[(gblk0+1) * nbas + jsh];
        j_pattern = (si_j0 >= nbins_j0) | ((si_j1 >= nbins_j1) << 1);
        switch (j_pattern & i_pattern) {
        case 0 :
                goto next_j;
        case 1 :
                j0 = ao_loc[jsh];
                j1 = ao_loc[jsh+1];
                ij = (i - ioff) * nj - joff;
                for (j = j0; j < j1; j++) {
                        j_addr = j * ngrids + ig0;
                        for (n = 0; n < ALIGNMENT; n++) {
                                s8[n] = out[(ij+j)*ALIGNMENT+n];
                                s8[n] += bra[i_addr+n             ] * ket[j_addr+n             ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*1 ] * ket[j_addr+n+ALIGNMENT*1 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*2 ] * ket[j_addr+n+ALIGNMENT*2 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*3 ] * ket[j_addr+n+ALIGNMENT*3 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*4 ] * ket[j_addr+n+ALIGNMENT*4 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*5 ] * ket[j_addr+n+ALIGNMENT*5 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*6 ] * ket[j_addr+n+ALIGNMENT*6 ];
                                out[(ij+j)*ALIGNMENT+n] = s8[n];
                        }
                }
                break;
        case 2 :
                j0 = ao_loc[jsh];
                j1 = ao_loc[jsh+1];
                ij = (i - ioff) * nj - joff;
                for (j = j0; j < j1; j++) {
                        j_addr = j * ngrids + ig0;
                        for (n = 0; n < ALIGNMENT; n++) {
                                s8[n] = out[(ij+j)*ALIGNMENT+n];
                                s8[n] += bra[i_addr+n+ALIGNMENT*7 ] * ket[j_addr+n+ALIGNMENT*7 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*8 ] * ket[j_addr+n+ALIGNMENT*8 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*9 ] * ket[j_addr+n+ALIGNMENT*9 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*10] * ket[j_addr+n+ALIGNMENT*10];
                                s8[n] += bra[i_addr+n+ALIGNMENT*11] * ket[j_addr+n+ALIGNMENT*11];
                                s8[n] += bra[i_addr+n+ALIGNMENT*12] * ket[j_addr+n+ALIGNMENT*12];
                                s8[n] += bra[i_addr+n+ALIGNMENT*13] * ket[j_addr+n+ALIGNMENT*13];
                                out[(ij+j)*ALIGNMENT+n] = s8[n];
                        }
                }
                break;
        case 3 :
                j0 = ao_loc[jsh];
                j1 = ao_loc[jsh+1];
                ij = (i - ioff) * nj - joff;
                for (j = j0; j < j1; j++) {
                        j_addr = j * ngrids + ig0;
                        for (n = 0; n < ALIGNMENT; n++) {
                                s8[n] = out[(ij+j)*ALIGNMENT+n];
                                s8[n] += bra[i_addr+n             ] * ket[j_addr+n             ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*1 ] * ket[j_addr+n+ALIGNMENT*1 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*2 ] * ket[j_addr+n+ALIGNMENT*2 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*3 ] * ket[j_addr+n+ALIGNMENT*3 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*4 ] * ket[j_addr+n+ALIGNMENT*4 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*5 ] * ket[j_addr+n+ALIGNMENT*5 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*6 ] * ket[j_addr+n+ALIGNMENT*6 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*7 ] * ket[j_addr+n+ALIGNMENT*7 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*8 ] * ket[j_addr+n+ALIGNMENT*8 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*9 ] * ket[j_addr+n+ALIGNMENT*9 ];
                                s8[n] += bra[i_addr+n+ALIGNMENT*10] * ket[j_addr+n+ALIGNMENT*10];
                                s8[n] += bra[i_addr+n+ALIGNMENT*11] * ket[j_addr+n+ALIGNMENT*11];
                                s8[n] += bra[i_addr+n+ALIGNMENT*12] * ket[j_addr+n+ALIGNMENT*12];
                                s8[n] += bra[i_addr+n+ALIGNMENT*13] * ket[j_addr+n+ALIGNMENT*13];
                                out[(ij+j)*ALIGNMENT+n] = s8[n];
                        }
                }
        }
next_j:;
                                }
                        }
                }
        }
}

static void _dot_ao_ao_frac(double *out, double *bra, double *ket,
                            int nao, size_t ngrids, int nbas, int ig0,
                            int ish0, int ish1, int jsh0, int jsh1, int nj,
                            int ioff, int joff, int nbins, uint8_t *screen_index,
                            uint8_t *pair_mask, int *ao_loc)
{
        int ish, jsh, i0, i1, j0, j1, i, j, ij, k, n, ig, gblk0;
        int ng1, ng1_aligned, ng1_rest;
        uint8_t si_i, si_j, nbins_j;
        size_t i_addr, j_addr;
        double s8[ALIGNMENT];

        for (ig = ig0; ig < ngrids; ig+=BLKSIZE) {
                gblk0 = ig / BLKSIZE;
                ng1 = MIN(BLKSIZE, ngrids - ig);
                ng1_aligned = ng1 & (-(uint32_t)ALIGNMENT);
                for (ish = ish0; ish < ish1; ish++) {
                        si_i = screen_index[gblk0 * nbas + ish];
                        if (si_i != 0) {
                                if (nbins > si_i) {
                                        nbins_j = nbins - si_i;
                                } else {
                                        nbins_j = 1;
                                }
                                i0 = ao_loc[ish];
                                i1 = ao_loc[ish+1];
                                for (i = i0; i < i1; i++) {
i_addr = i * ngrids + ig;
for (jsh = jsh0; jsh < jsh1; jsh++) {
        si_j = screen_index[gblk0 * nbas + jsh];
        if (si_j >= nbins_j && pair_mask[ish*nbas+jsh]) {
                j0 = ao_loc[jsh];
                j1 = ao_loc[jsh+1];
                ij = (i - ioff) * nj - joff;
                for (j = j0; j < j1; j++) {
                        j_addr = j * ngrids + ig;
                        for (n = 0; n < ALIGNMENT; n++) {
                                s8[n] = out[(ij+j)*ALIGNMENT+n];
                        }
                        for (k = 0; k < ng1_aligned; k+=ALIGNMENT) {
                        for (n = 0; n < ALIGNMENT; n++) {
                                s8[n] += bra[i_addr+n+k] * ket[j_addr+n+k];
                        } }
                        for (n = 0; n < ALIGNMENT; n++) {
                                out[(ij+j)*ALIGNMENT+n] = s8[n];
                        }
                }
        }
}
                                }
                        }
                }

                if (ng1_aligned < ng1) {
                        ng1_rest = ng1 - ng1_aligned;
                        for (ish = ish0; ish < ish1; ish++) {
                                si_i = screen_index[gblk0 * nbas + ish];
                                if (si_i != 0) {
                                        if (nbins > si_i) {
                                                nbins_j = nbins - si_i;
                                        } else {
                                                nbins_j = 1;
                                        }
                                        i0 = ao_loc[ish];
                                        i1 = ao_loc[ish+1];
                                        for (i = i0; i < i1; i++) {
i_addr = i * ngrids + ng1_aligned;
for (jsh = jsh0; jsh < jsh1; jsh++) {
        si_j = screen_index[gblk0 * nbas + jsh];
        if (si_j >= nbins_j && pair_mask[ish*nbas+jsh]) {
                j0 = ao_loc[jsh];
                j1 = ao_loc[jsh+1];
                ij = (i - ioff) * nj - joff;
                for (j = j0; j < j1; j++) {
                        j_addr = j * ngrids + ng1_aligned;
                        s8[0] = 0.;
                        for (k = 0; k < ng1_rest; k++) {
                                s8[0] += bra[i_addr+k] * ket[j_addr+k];
                        }
                        out[(ij+j)*ALIGNMENT] += s8[0];
                }
        }
}
                                        }
                                }
                        }
                }
        }
}

/* out = bra.T.dot(ket) */
void VXCdot_ao_ao_sparse(double *out, double *bra, double *ket,
                         int nao, int ngrids, int nbas, int hermi, int nbins,
                         uint8_t *screen_index, uint8_t *pair_mask, int *ao_loc)
{
        size_t Nao = nao;
        int shls_slice[2] = {0, nbas};
        int *box_l1_loc = malloc(sizeof(int) * (nbas+1));
        int nbox_l1 = CVHFshls_block_partition(box_l1_loc, shls_slice, ao_loc, BOXSIZE1_N);
        int mask_l1_size = (ngrids + BOXSIZE1_M - 1)/BOXSIZE1_M * nbox_l1;
        uint8_t *mask_l1 = malloc(sizeof(uint8_t) * mask_l1_size);
        mask_l1_abstract(mask_l1, screen_index, box_l1_loc,
                         nbox_l1, ngrids, nbas);
        int ngrids_align_down = (ngrids / BOXSIZE1_M) * BOXSIZE1_M;

#pragma omp parallel
{
        int ijb, ib, jb, ib0, jb0, ib1, jb1, ig0, ig1, ig_box2;
        int ish0, ish1, jsh0, jsh1, i0, i1, j0, j1, ni, nj, i, j, n;
        double s;
        double *pout;
        double *buf = malloc(sizeof(double) * (BOXSIZE1_N*BOXSIZE1_N*ALIGNMENT+ALIGNMENT));
        double *outbuf = (double *)((uintptr_t)(buf+ALIGNMENT-1) & (-(uintptr_t)(ALIGNMENT*sizeof(double))));
#pragma omp for schedule(dynamic, 4) nowait
        for (ijb = 0; ijb < nbox_l1*nbox_l1; ijb++) {
                ib = ijb / nbox_l1;
                jb = ijb % nbox_l1;
                if (hermi && ib < jb) {
                        continue;
                }
                ib0 = ib;
                jb0 = jb;
                ib1 = ib + 1;
                jb1 = jb + 1;
                ish0 = box_l1_loc[ib0];
                jsh0 = box_l1_loc[jb0];
                ish1 = box_l1_loc[ib1];
                jsh1 = box_l1_loc[jb1];
                i0 = ao_loc[ish0];
                i1 = ao_loc[ish1];
                j0 = ao_loc[jsh0];
                j1 = ao_loc[jsh1];
                ni = i1 - i0;
                nj = j1 - j0;
                for (n = 0; n < ni * nj * ALIGNMENT; n++) {
                        outbuf[n] = 0;
                }

                for (ig0 = 0; ig0 < ngrids_align_down; ig0+=BOXSIZE1_M) {
                        ig1 = MIN(ig0 + BOXSIZE1_M, ngrids_align_down);
                        ig_box2 = ig0 / BOXSIZE1_M;
                        if (mask_l1[ig_box2 * nbox_l1 + jb] &&
                            mask_l1[ig_box2 * nbox_l1 + ib]) {
_dot_ao_ao_l1(outbuf, bra, ket, nao, ngrids, nbas, ig0, ig1,
              ish0, ish1, jsh0, jsh1, nj, i0, j0, nbins, screen_index,
              pair_mask, ao_loc);
                        }
                }
                if (ngrids_align_down < ngrids) {
                        _dot_ao_ao_frac(outbuf, bra, ket, nao, ngrids, nbas,
                                        ngrids_align_down, ish0, ish1, jsh0, jsh1,
                                        nj, i0, j0, nbins, screen_index,
                                        pair_mask, ao_loc);
                }

                for (i = 0; i < ni; i++) {
                for (j = 0; j < nj; j++) {
                        s = 0;
                        pout = outbuf + (i*nj+j)*ALIGNMENT;
                        for (n = 0; n < ALIGNMENT; n++) {
                                s += pout[n];
                        }
                        out[(i0+i)*Nao+j0+j] += s;
                } }
        }
        free(buf);
}
        free(box_l1_loc);
        free(mask_l1);

        if (hermi != 0) {
                NPdsymm_triu(nao, out, hermi);
        }
}

// 'ip,ip->p'
void VXCdcontract_rho_sparse(double *rho, double *bra, double *ket,
                             int nao, int ngrids, int nbas,
                             uint8_t *screen_index, int *ao_loc)
{
        int mask_last_row = ngrids / BLKSIZE;
        int ngrids_align_down = mask_last_row * BLKSIZE;
        int ngrids_rest = ngrids - ngrids_align_down;

        if (nao * 2 < ngrids) {

#pragma omp parallel
{
                size_t Ngrids = ngrids;
                size_t i_addr;
                int ig0, row, ish, i0, i1, n, i;
                double s8[BLKSIZE];
#pragma omp for schedule(dynamic, 4) nowait
                for (ig0 = 0; ig0 < ngrids_align_down; ig0+=BLKSIZE) {
                        for (n = 0; n < BLKSIZE; n++) {
                                s8[n] = 0;
                        }
                        row = ig0 / BLKSIZE;
                        for (ish = 0; ish < nbas; ish++) {
                                if (screen_index[row * nbas + ish]) {
                                        i0 = ao_loc[ish];
                                        i1 = ao_loc[ish+1];
                                        for (i = i0; i < i1; i++) {
                                                i_addr = i * Ngrids + ig0;
                                                for (n = 0; n < BLKSIZE; n++) {
                                                        s8[n] += bra[i_addr+n] * ket[i_addr+n];
                                                }
                                        }
                                }
                        }
                        for (n = 0; n < BLKSIZE; n++) {
                                rho[ig0+n] = s8[n];
                        }
                }
}
                if (ngrids_align_down < ngrids) {
                        size_t Ngrids = ngrids;
                        size_t i_addr;
                        int ish, i0, i1, n, i;
                        double s8[BLKSIZE];
                        int ngrids_rest = ngrids - ngrids_align_down;
                        for (n = 0; n < ngrids_rest; n++) {
                                s8[n] = 0;
                        }
                        for (ish = 0; ish < nbas; ish++) {
                                if (screen_index[mask_last_row * nbas + ish]) {
                                        i0 = ao_loc[ish];
                                        i1 = ao_loc[ish+1];
                                        for (i = i0; i < i1; i++) {
                                                i_addr = i * Ngrids + ngrids_align_down;
                                                for (n = 0; n < ngrids_rest; n++) {
                                                        s8[n] += bra[i_addr+n] * ket[i_addr+n];
                                                }
                                        }
                                }
                        }
                        for (n = 0; n < ngrids_rest; n++) {
                                rho[ngrids_align_down+n] = s8[n];
                        }
                }

        } else {  // if (nao * 2 < ngrids)

                double *rhobufs[MAX_THREADS];
                rhobufs[0] = rho;
#pragma omp parallel
{
                size_t Ngrids = ngrids;
                size_t i_addr;
                int ig0, row, ish, i0, i1, n, i;
                double s8[BLKSIZE];
                int thread_id = omp_get_thread_num();
                if (thread_id > 0) {
                        rhobufs[thread_id] = malloc(sizeof(double) * ngrids);
                }
                double *rho_priv = rhobufs[thread_id];
                NPdset0(rho_priv, ngrids);
#pragma omp for schedule(dynamic, 4)
                for (ish = 0; ish < nbas; ish++) {
                        i0 = ao_loc[ish];
                        i1 = ao_loc[ish+1];
                        for (ig0 = 0; ig0 < ngrids_align_down; ig0+=BLKSIZE) {
                                for (n = 0; n < BLKSIZE; n++) {
                                        s8[n] = rho_priv[ig0+n];
                                }
                                row = ig0 / BLKSIZE;
                                if (screen_index[row * nbas + ish]) {
                                        for (i = i0; i < i1; i++) {
                                                i_addr = i * Ngrids + ig0;
                                                for (n = 0; n < BLKSIZE; n++) {
                                                        s8[n] += bra[i_addr+n] * ket[i_addr+n];
                                                }
                                        }
                                }
                                for (n = 0; n < BLKSIZE; n++) {
                                        rho_priv[ig0+n] = s8[n];
                                }
                        }
                        if (ngrids_rest) {
                                for (n = 0; n < ngrids_rest; n++) {
                                        s8[n] = rho_priv[ngrids_align_down+n];
                                }
                                if (screen_index[mask_last_row * nbas + ish]) {
                                        for (i = i0; i < i1; i++) {
                                                i_addr = i * Ngrids + ngrids_align_down;
                                                for (n = 0; n < ngrids_rest; n++) {
                                                        s8[n] += bra[i_addr+n] * ket[i_addr+n];
                                                }
                                        }
                                }
                                for (n = 0; n < ngrids_rest; n++) {
                                        rho_priv[ngrids_align_down+n] = s8[n];
                                }
                        }
                }
                NPomp_dsum_reduce_inplace(rhobufs, ngrids);
                if (thread_id > 0) {
                        free(rho_priv);
                }
}
        }  // if (nao * 2 < ngrids)
}

// 'nip,np->ip'
void VXCdscale_ao_sparse(double *aow, double *ao, double *wv,
                         int comp, int nao, int ngrids, int nbas,
                         uint8_t *screen_index, int *ao_loc)
{
#pragma omp parallel
{
        size_t Ngrids = ngrids;
        size_t ao_size = nao * Ngrids;
        int ish, i, j, ic, i0, i1, ig0, ig1, row;
        double *pao = ao;
#pragma omp for schedule(static, 2)
        for (ish = 0; ish < nbas; ish++) {
                i0 = ao_loc[ish];
                i1 = ao_loc[ish+1];
                for (ig0 = 0; ig0 < ngrids; ig0 += BLKSIZE) {
                        ig1 = MIN(ig0+BLKSIZE, ngrids);
                        row = ig0 / BLKSIZE;
                        if (screen_index[row * nbas + ish]) {
for (i = i0; i < i1; i++) {
        pao = ao + i * Ngrids;
        for (j = ig0; j < ig1; j++) {
                aow[i*Ngrids+j] = pao[j] * wv[j];
        }
        for (ic = 1; ic < comp; ic++) {
        for (j = ig0; j < ig1; j++) {
                aow[i*Ngrids+j] += pao[ic*ao_size+j] * wv[ic*Ngrids+j];
        } }
}
                        }
                }
        }
}
}

/*
 * Sparse AO products with BLAS level 3.
 *
 * The kernels above compute each AO pair as a dot product over the grid
 * points. The kernels below take GEMM_NBLK consecutive blocks of BLKSIZE grid
 * points at a time, gather the AOs of the shells that are unscreened on any of
 * these blocks into contiguous buffers, and multiply the buffers with dgemm or
 * dsyrk. The values on the blocks where a shell is screened are gathered as
 * zeros, so, as for the kernels above, the inputs only need to be valid where
 * the screen index is nonzero. eval_ao sets the AO values of screened shells to
 * exactly zero, so the results equal the dense products up to the order of
 * summation: nbins and pair_mask are not used and no product is dropped.
 */
#define GEMM_NBLK       4
#define GEMM_NGRIDS     (GEMM_NBLK*BLKSIZE)

typedef struct {
        int nao;            // number of AOs of the unscreened shells
        int nshl;           // number of unscreened shells
        int nseg;           // number of contiguous AO ranges they form
        int *shl;           // [nshl] shell ids, ascending
        uint8_t *blkmask;   // [nshl] bit b: the shell is unscreened on block blk0+b
        int *seg_ao;        // [nseg] first AO of each range
        int *seg_off;       // [nseg] position of that AO in the gathered buffers
        int *seg_len;       // [nseg] number of AOs in the range
} GemmGroup;

static void _gemm_group_init(GemmGroup *g, int nbas)
{
        g->shl = malloc(sizeof(int) * nbas * 4);
        g->seg_ao = g->shl + nbas;
        g->seg_off = g->seg_ao + nbas;
        g->seg_len = g->seg_off + nbas;
        g->blkmask = malloc(sizeof(uint8_t) * nbas);
}

static void _gemm_group_free(GemmGroup *g)
{
        free(g->shl);
        free(g->blkmask);
}

// The shells that are unscreened on any of the blocks [blk0, blk1)
static void _gemm_group_shells(GemmGroup *g, uint8_t *screen_index,
                               int blk0, int blk1, int nbas, int *ao_loc)
{
        int ish, b, mask;
        int nao = 0;
        int nshl = 0;
        int nseg = 0;
        for (ish = 0; ish < nbas; ish++) {
                mask = 0;
                for (b = 0; b < blk1 - blk0; b++) {
                        if (screen_index[(blk0+b)*nbas+ish]) {
                                mask |= 1 << b;
                        }
                }
                if (mask == 0) {
                        continue;
                }
                if (nseg == 0 ||
                    g->seg_ao[nseg-1] + g->seg_len[nseg-1] != ao_loc[ish]) {
                        g->seg_ao[nseg] = ao_loc[ish];
                        g->seg_off[nseg] = nao;
                        g->seg_len[nseg] = 0;
                        nseg++;
                }
                g->seg_len[nseg-1] += ao_loc[ish+1] - ao_loc[ish];
                g->shl[nshl] = ish;
                g->blkmask[nshl] = mask;
                nshl++;
                nao += ao_loc[ish+1] - ao_loc[ish];
        }
        g->nao = nao;
        g->nshl = nshl;
        g->nseg = nseg;
}

// The largest number of AOs of the unscreened shells over all groups of blocks
static int _gemm_max_group_aos(uint8_t *screen_index, int ngrids, int nbas,
                               int *ao_loc)
{
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        const int ngroup = (nblk + GEMM_NBLK - 1) / GEMM_NBLK;
        int max_nao = 0;
        int igroup;
#pragma omp parallel for schedule(static) reduction(max:max_nao)
        for (igroup = 0; igroup < ngroup; igroup++) {
                int blk0 = igroup * GEMM_NBLK;
                int blk1 = MIN(blk0 + GEMM_NBLK, nblk);
                int nao = 0;
                int ish, b;
                for (ish = 0; ish < nbas; ish++) {
                        for (b = blk0; b < blk1; b++) {
                                if (screen_index[b*nbas+ish]) {
                                        nao += ao_loc[ish+1] - ao_loc[ish];
                                        break;
                                }
                        }
                }
                max_nao = MAX(max_nao, nao);
        }
        return max_nao;
}

// buf[n,k] = ao[i,ig0+k] * wv[ig0+k] (wv = NULL: without weights) for the n-th
// AO i of the group, and zero on the blocks where the shell of i is screened
static void _gemm_gather(double *buf, double *ao, double *wv, GemmGroup *g,
                         int *ao_loc, size_t Ngrids, int ig0, int ng)
{
        int s, i, b, k, k0, k1;
        double *pao;
        double *pbuf = buf;
        for (s = 0; s < g->nshl; s++) {
        for (i = ao_loc[g->shl[s]]; i < ao_loc[g->shl[s]+1]; i++) {
                pao = ao + i * Ngrids + ig0;
                for (b = 0, k0 = 0; k0 < ng; b++, k0 += BLKSIZE) {
                        k1 = MIN(k0 + BLKSIZE, ng);
                        if (!(g->blkmask[s] & (1 << b))) {
                                for (k = k0; k < k1; k++) {
                                        pbuf[k] = 0;
                                }
                        } else if (wv == NULL) {
                                for (k = k0; k < k1; k++) {
                                        pbuf[k] = pao[k];
                                }
                        } else {
                                for (k = k0; k < k1; k++) {
                                        pbuf[k] = pao[k] * wv[ig0+k];
                                }
                        }
                }
                pbuf += ng;
        } }
}

// buf[n,p] = ao[i,ig0+pidx[p]] * scale[p] for the n-th AO i of the group, and
// zero on the blocks where the shell of i is screened
static void _gemm_gather_points(double *buf, double *ao, double *scale,
                                int *pidx, int np, GemmGroup *g, int *ao_loc,
                                size_t Ngrids, int ig0)
{
        int s, i, p, mask;
        double *pao;
        double *pbuf = buf;
        for (s = 0; s < g->nshl; s++) {
                mask = g->blkmask[s];
                for (i = ao_loc[g->shl[s]]; i < ao_loc[g->shl[s]+1]; i++) {
                        pao = ao + i * Ngrids + ig0;
                        for (p = 0; p < np; p++) {
                                if (mask & (1 << (pidx[p] / BLKSIZE))) {
                                        pbuf[p] = pao[pidx[p]] * scale[p];
                                } else {
                                        pbuf[p] = 0;
                                }
                        }
                        pbuf += np;
                }
        }
}

// out[i,j] += cbuf[n,m] for the n-th AO i and the m-th AO j of the group;
// with tril, only for m <= n (j <= i)
static void _gemm_scatter_add(double *out, double *cbuf, GemmGroup *g,
                              size_t Nao, int tril)
{
        const int ns = g->nao;
        int s, s1, i, m, m0, mlen;
        int n = 0;
        double *pout, *pc;
        for (s = 0; s < g->nseg; s++) {
        for (i = g->seg_ao[s]; i < g->seg_ao[s] + g->seg_len[s]; i++, n++) {
                pc = cbuf + n * (size_t)ns;
                pout = out + i * Nao;
                for (s1 = 0; s1 < g->nseg; s1++) {
                        m0 = g->seg_off[s1];
                        mlen = g->seg_len[s1];
                        if (tril) {
                                if (m0 > n) {
                                        break;
                                }
                                mlen = MIN(mlen, n - m0 + 1);
                        }
                        for (m = 0; m < mlen; m++) {
                                pout[g->seg_ao[s1]+m] += pc[m0+m];
                        }
                }
        } }
}

/* out[j,g] = sum_i ao[i,g] * dm[i,j] for real ao[nao,ngrids] and dm[nao,nao].
 * As for VXCdot_ao_dm_sparse, out[j,g] is only written where shell j can be
 * unscreened on the block of g. */
void VXCdot_ao_dm_sparse_gemm(double *out, double *ao, double *dm,
                              int nao, int ngrids, int nbas,
                              uint8_t *screen_index, int *ao_loc)
{
        const size_t Nao = nao;
        const size_t Ngrids = ngrids;
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        const int ngroup = (nblk + GEMM_NBLK - 1) / GEMM_NBLK;
        const int max_nao = _gemm_max_group_aos(screen_index, ngrids, nbas, ao_loc);
        if (max_nao == 0) {
                return;
        }
#pragma omp parallel
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double D0 = 0;
        const double D1 = 1;
        GemmGroup g;
        _gemm_group_init(&g, nbas);
        double *aobuf = malloc(sizeof(double) * max_nao * (GEMM_NGRIDS*2 + max_nao));
        double *outbuf = aobuf + max_nao * GEMM_NGRIDS;
        double *dmbuf = outbuf + max_nao * GEMM_NGRIDS;
        int igroup, blk0, blk1, ig0, ng, ns, s, s1, i, k, n;
        double *pin, *pout;
// each group writes its own grid columns of out, so the result does not depend
// on the schedule
#pragma omp for schedule(dynamic, 4)
        for (igroup = 0; igroup < ngroup; igroup++) {
                blk0 = igroup * GEMM_NBLK;
                blk1 = MIN(blk0 + GEMM_NBLK, nblk);
                ig0 = blk0 * BLKSIZE;
                ng = MIN(blk1 * BLKSIZE, ngrids) - ig0;
                _gemm_group_shells(&g, screen_index, blk0, blk1, nbas, ao_loc);
                ns = g.nao;
                if (ns == 0) {
                        continue;
                }
                _gemm_gather(aobuf, ao, NULL, &g, ao_loc, Ngrids, ig0, ng);
                // dmbuf[m,n] = dm[i,j] for the m-th AO i and the n-th AO j
                n = 0;
                for (s = 0; s < g.nseg; s++) {
                for (i = g.seg_ao[s]; i < g.seg_ao[s] + g.seg_len[s]; i++, n++) {
                        pin = dm + i * Nao;
                        pout = dmbuf + n * (size_t)ns;
                        for (s1 = 0; s1 < g.nseg; s1++) {
                                for (k = 0; k < g.seg_len[s1]; k++) {
                                        pout[g.seg_off[s1]+k] = pin[g.seg_ao[s1]+k];
                                }
                        }
                } }
                // outbuf[n,k] = sum_m dmbuf[m,n] * aobuf[m,k]
                dgemm_(&TRANS_N, &TRANS_T, &ng, &ns, &ns, &D1, aobuf, &ng,
                       dmbuf, &ns, &D0, outbuf, &ng);
                n = 0;
                for (s = 0; s < g.nseg; s++) {
                for (i = g.seg_ao[s]; i < g.seg_ao[s] + g.seg_len[s]; i++, n++) {
                        pin = outbuf + n * (size_t)ng;
                        pout = out + i * Ngrids + ig0;
                        for (k = 0; k < ng; k++) {
                                pout[k] = pin[k];
                        }
                } }
        }
        _gemm_group_free(&g);
        free(aobuf);
}
}

/* out[i,j] += sum_g bra[i,g] * wv[g] * ket[j,g] for real bra[nao,ngrids] and
 * ket[nao,ngrids]; wv = NULL for no weights. With hermi, the lower triangle is
 * computed and copied to the upper triangle, as in VXCdot_ao_ao_sparse. */
void VXCdot_aow_ao_sparse_gemm(double *out, double *bra, double *ket, double *wv,
                               int nao, int ngrids, int nbas, int hermi,
                               uint8_t *screen_index, int *ao_loc)
{
        const size_t Nao = nao;
        const size_t Ngrids = ngrids;
        const size_t nn = Nao * Nao;
        const int nblk = (ngrids + BLKSIZE - 1) / BLKSIZE;
        const int ngroup = (nblk + GEMM_NBLK - 1) / GEMM_NBLK;
        // The symmetric product of one set of AOs is evaluated with dsyrk on
        // one triangle, at half the cost of dgemm
        const int syrk = bra == ket && (hermi == HERMITIAN || hermi == SYMMETRIC);
        const int max_nao = _gemm_max_group_aos(screen_index, ngrids, nbas, ao_loc);
        double **vprivs = malloc(sizeof(double *) * omp_get_max_threads());
        if (max_nao > 0) {
#pragma omp parallel
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const char UPLO = 'U';  // cbuf[n,m] for m <= n in C order
        const double D0 = 0;
        const double D1 = 1;
        const double DM1 = -1;
        GemmGroup g;
        _gemm_group_init(&g, nbas);
        double *vpriv = calloc(nn, sizeof(double));
        // Published here so the barrier in NPomp_dsum_reduce_inplace makes it
        // visible to the other threads.
        vprivs[omp_get_thread_num()] = vpriv;
        double *brabuf = malloc(sizeof(double) * max_nao * (GEMM_NGRIDS*2 + max_nao));
        double *ketbuf = brabuf + max_nao * GEMM_NGRIDS;
        double *cbuf = ketbuf + max_nao * GEMM_NGRIDS;
        double sqrtw[GEMM_NGRIDS];
        int pidx[GEMM_NGRIDS];
        int igroup, blk0, blk1, ig0, ng, ns, np, sign, i, j, k;
        double w, beta;
// static: each thread sums a fixed set of groups, so out is reproducible for a
// given number of threads. Round robin, because the cost of a group varies
// along the grid.
#pragma omp for schedule(static, 1)
        for (igroup = 0; igroup < ngroup; igroup++) {
                blk0 = igroup * GEMM_NBLK;
                blk1 = MIN(blk0 + GEMM_NBLK, nblk);
                ig0 = blk0 * BLKSIZE;
                ng = MIN(blk1 * BLKSIZE, ngrids) - ig0;
                _gemm_group_shells(&g, screen_index, blk0, blk1, nbas, ao_loc);
                ns = g.nao;
                if (ns == 0) {
                        continue;
                }
                if (!syrk) {
                        _gemm_gather(brabuf, bra, wv, &g, ao_loc, Ngrids, ig0, ng);
                        _gemm_gather(ketbuf, ket, NULL, &g, ao_loc, Ngrids, ig0, ng);
                        // cbuf[n,m] = sum_k brabuf[n,k] * ketbuf[m,k]
                        dgemm_(&TRANS_T, &TRANS_N, &ns, &ns, &ng, &D1, ketbuf, &ng,
                               brabuf, &ng, &D0, cbuf, &ns);
                } else if (wv == NULL) {
                        _gemm_gather(brabuf, bra, NULL, &g, ao_loc, Ngrids, ig0, ng);
                        dsyrk_(&UPLO, &TRANS_T, &ns, &ng, &D1, brabuf, &ng,
                               &D0, cbuf, &ns);
                } else {
                        // dsyrk needs the form A^T A, so the grid points are split
                        // by the sign of wv and the AO values scaled by sqrt(|wv|)
                        beta = 0;
                        for (sign = 1; sign >= -1; sign -= 2) {
                                np = 0;
                                for (k = 0; k < ng; k++) {
                                        w = wv[ig0+k] * sign;
                                        // NaN goes to the positive set so that it propagates
                                        if (w > 0 || (sign > 0 && w != w)) {
                                                pidx[np] = k;
                                                sqrtw[np] = sqrt(w);
                                                np++;
                                        }
                                }
                                if (np == 0) {
                                        continue;
                                }
                                _gemm_gather_points(brabuf, bra, sqrtw, pidx, np,
                                                    &g, ao_loc, Ngrids, ig0);
                                dsyrk_(&UPLO, &TRANS_T, &ns, &np, (sign > 0 ? &D1 : &DM1),
                                       brabuf, &np, &beta, cbuf, &ns);
                                beta = 1;
                        }
                        if (beta == 0) {  // wv is zero on all points of the group
                                continue;
                        }
                }
                _gemm_scatter_add(vpriv, cbuf, &g, Nao, hermi);
        }
        NPomp_dsum_reduce_inplace(vprivs, nn);
        double *vsum = vprivs[0];
#pragma omp for schedule(static)
        for (i = 0; i < nao; i++) {
                for (j = 0; j < (hermi ? i+1 : nao); j++) {
                        out[i*Nao+j] += vsum[i*Nao+j];
                }
        }
        _gemm_group_free(&g);
        free(vpriv);
        free(brabuf);
}
        }
        free(vprivs);
        if (hermi != 0) {
                NPdsymm_triu(nao, out, hermi);
        }
}
