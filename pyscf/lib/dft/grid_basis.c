/* Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
  
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
#include <math.h>
#include "cint.h"
#include "config.h"
#include "gto/grid_ao_drv.h"
#include "np_helper/np_helper.h"

// 1k grids per block
#define GRIDS_BLOCK     512
#define RCUT_LKO        5.0
#define MC_LKO          12
#define ALIGNMENT       8

void VXCgen_grid(double *out, double *coords, double *atm_coords,
                 double *radii_table, int natm, int ngrids)
{
        const size_t Ngrids = ngrids;
        int i, j;
        double dx, dy, dz;
        double *atom_dist = malloc(sizeof(double) * natm*natm);
        for (i = 0; i < natm; i++) {
                for (j = 0; j < i; j++) {
                        dx = atm_coords[i*3+0] - atm_coords[j*3+0];
                        dy = atm_coords[i*3+1] - atm_coords[j*3+1];
                        dz = atm_coords[i*3+2] - atm_coords[j*3+2];
                        atom_dist[i*natm+j] = 1 / sqrt(dx*dx + dy*dy + dz*dz);
                }
        }

#pragma omp parallel private(i, j, dx, dy, dz)
{
        double *_buf = malloc(sizeof(double) * ((natm*2+1)*GRIDS_BLOCK + ALIGNMENT));
        double *buf = (double *)((uintptr_t)(_buf + ALIGNMENT - 1) & (-(uintptr_t)(ALIGNMENT*8)));
        double *g = buf + natm * GRIDS_BLOCK;
        double *grid_dist = g + GRIDS_BLOCK;
        size_t ig0, n, ngs;
        double fac, s;
#pragma omp for nowait schedule(static)
        for (ig0 = 0; ig0 < Ngrids; ig0 += GRIDS_BLOCK) {
                ngs = MIN(Ngrids-ig0, GRIDS_BLOCK);
                for (i = 0; i < natm; i++) {
#pragma GCC ivdep
                for (n = 0; n < ngs; n++) {
                        dx = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                        dy = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                        dz = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
                        grid_dist[i*GRIDS_BLOCK+n] = sqrt(dx*dx + dy*dy + dz*dz);
                        buf[i*GRIDS_BLOCK+n] = 1;
                } }

                for (i = 0; i < natm; i++) {
                for (j = 0; j < i; j++) {

                        fac = atom_dist[i*natm+j];
                        for (n = 0; n < ngs; n++) {
                                g[n] = (grid_dist[i*GRIDS_BLOCK+n] -
                                        grid_dist[j*GRIDS_BLOCK+n]) * fac;
                        }
                        if (radii_table != NULL) {
                                fac = radii_table[i*natm+j];
                                for (n = 0; n < ngs; n++) {
                                        g[n] += fac * (1 - g[n]*g[n]);
                                }
                        }
#pragma GCC ivdep
                        for (n = 0; n < ngs; n++) {
                                s = g[n];
                                s = (3 - s*s) * s * .5;
                                s = (3 - s*s) * s * .5;
                                s = ((3 - s*s) * s * .5) * .5;
                                buf[i*GRIDS_BLOCK+n] *= .5 - s;
                                buf[j*GRIDS_BLOCK+n] *= .5 + s;
                        }
                } }

                for (i = 0; i < natm; i++) {
                        for (n = 0; n < ngs; n++) {
                                out[i*Ngrids+ig0+n] = buf[i*GRIDS_BLOCK+n];
                        }
                }
        }
        free(_buf);
}
        free(atom_dist);
}

typedef struct { double x, y, z; } double3;

static inline double3 d3_plus(const double3 v1, const double3 v2) { double3 v = { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z }; return v; }
static inline double3 d3_minus(const double3 v1, const double3 v2) { double3 v = { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z }; return v; }
static inline double3 d3_negate(const double3 v) { double3 nv = { -v.x, -v.y, -v.z }; return nv; }
static inline double3 d3_scale(const double k, const double3 v) { double3 kv = { k * v.x, k * v.y, k * v.z }; return kv; }
static inline double norm(const double3 v) { return sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }
static inline double inv(const double x)
{
    if (x > 1e-14) return 1.0 / x;
    else return 0.0;
}

static double switch_function(const double mu, const double a_factor)
{
    const double nu = mu + a_factor * (1.0 - mu * mu);
    double s = nu;
    s = (3.0 - s * s) * s * 0.5;
    s = (3.0 - s * s) * s * 0.5;
    s = (3.0 - s * s) * s * 0.5;
    s = 0.5 * (1.0 - s);
    return s;
}

static double switch_function_dmuds_over_s(const double mu, const double a_factor)
{
    const double nu = mu + a_factor * (1 - mu * mu);
    const double dnu_dmu = 1.0 - 2.0 * a_factor * mu;
    const double f1 = (3.0 - nu * nu) * nu * 0.5;
    const double f2 = (3.0 - f1 * f1) * f1 * 0.5;
    const double f3 = (3.0 - f2 * f2) * f2 * 0.5;
    const double s = 0.5 * (1.0 - f3);
    const double dmuds = -0.5 * 1.5 * (1 - f2 * f2) * 1.5 * (1 - f1 * f1) * 1.5 * (1 - nu * nu) * dnu_dmu;
    return dmuds * inv(s);
}

void VXCbecke_weight_derivative(double* __restrict__ dwdG, const double* __restrict__ grid_coords, const double* __restrict__ grid_quadrature_weights,
                                const double* __restrict__ atm_coords, const double* __restrict__ a_factor,
                                const int* __restrict__ atm_idx, const int ngrids, const int natm)
{
    #pragma omp parallel for schedule(static) collapse(2)
    for (int i_grid = 0; i_grid < ngrids; i_grid++) {
        for (int i_derivative_atom = 0; i_derivative_atom < natm; i_derivative_atom++) {
            const int i_associated_atom = atm_idx[i_grid];
            if (i_associated_atom < 0) // Pad grid
                continue;
            if (i_associated_atom == i_derivative_atom) // Dealt with later by translation invariance.
                continue;

            const double3 grid_r = { grid_coords[i_grid * 3 + 0], grid_coords[i_grid * 3 + 1], grid_coords[i_grid * 3 + 2] };
            const double3 atom_A = { atm_coords[i_associated_atom * 3 + 0], atm_coords[i_associated_atom * 3 + 1], atm_coords[i_associated_atom * 3 + 2] };
            const double3 atom_G = { atm_coords[i_derivative_atom * 3 + 0], atm_coords[i_derivative_atom * 3 + 1], atm_coords[i_derivative_atom * 3 + 2] };
            const double3 Ar = d3_minus(atom_A, grid_r);
            const double3 Gr = d3_minus(atom_G, grid_r);
            const double norm_Ar = norm(Ar);
            const double norm_Gr = norm(Gr);
            const double norm_Gr_1 = inv(norm_Gr);

            double P_A = 1.0;
            double sum_P_B = 0.0;
            double3 sum_dPB_dG = { 0.0, 0.0, 0.0 };
            double P_G = 1.0;
            double3 dPG_dG = { 0.0, 0.0, 0.0 };

            for (int j_atom = 0; j_atom < natm; j_atom++) {
                const double3 atom_B = { atm_coords[j_atom * 3 + 0], atm_coords[j_atom * 3 + 1], atm_coords[j_atom * 3 + 2] };
                const double3 Br = d3_minus(atom_B, grid_r);
                const double norm_Br = norm(Br);

                const double3 AB = d3_minus(atom_A, atom_B);
                const double norm_AB_1 = inv(norm(AB));

                const double mu_AB = (norm_Ar - norm_Br) * norm_AB_1;
                const double a_factor_AB = a_factor[i_associated_atom * natm + j_atom];
                const double s_AB = switch_function(mu_AB, a_factor_AB);

                P_A *= s_AB;

                double P_B = 1.0;

                for (int k_atom = 0; k_atom < natm; k_atom++) {
                    const double3 atom_C = { atm_coords[k_atom * 3 + 0], atm_coords[k_atom * 3 + 1], atm_coords[k_atom * 3 + 2] };
                    const double3 Cr = d3_minus(atom_C, grid_r);
                    const double3 BC = d3_minus(atom_B, atom_C);
                    const double norm_Cr = norm(Cr);
                    const double norm_BC_1 = inv(norm(BC));

                    const double mu_BC = (norm_Br - norm_Cr) * norm_BC_1;
                    const double a_factor_BC = a_factor[j_atom * natm + k_atom];
                    const double s_BC = switch_function(mu_BC, a_factor_BC);

                    P_B *= s_BC;
                }

                sum_P_B += P_B;

                const double3 BG = d3_minus(atom_B, atom_G);
                const double norm_BG_1 = inv(norm(BG));
                const double mu_BG = (norm_Br - norm_Gr) * norm_BG_1;
                const double3 dmuBG_dG = d3_scale(norm_BG_1, d3_plus(d3_scale(-norm_Gr_1, Gr), d3_scale(mu_BG * norm_BG_1, BG)));
                const double a_factor_BG = a_factor[j_atom * natm + i_derivative_atom];
                const double3 dPB_dG = d3_scale(switch_function_dmuds_over_s(mu_BG, a_factor_BG) * P_B, dmuBG_dG);

                sum_dPB_dG = d3_plus(sum_dPB_dG, dPB_dG);

                const double a_factor_GB = a_factor[i_derivative_atom * natm + j_atom];
                const double s_GB = switch_function(-mu_BG, a_factor_GB);
                P_G *= s_GB;

                const double3 dmuGB_dG = d3_negate(dmuBG_dG);
                dPG_dG = d3_plus(dPG_dG, d3_scale(switch_function_dmuds_over_s(-mu_BG, a_factor_GB), dmuGB_dG));
            }

            sum_dPB_dG = d3_plus(sum_dPB_dG, d3_scale(P_G, dPG_dG));

            const double3 AG = d3_minus(atom_A, atom_G);
            const double norm_AG_1 = inv(norm(AG));
            const double mu_AG = (norm_Ar - norm_Gr) * norm_AG_1;
            const double3 dmuAG_dG = d3_scale(norm_AG_1, d3_plus(d3_scale(-norm_Gr_1, Gr), d3_scale(mu_AG * norm_AG_1, AG)));
            const double a_factor_AG = a_factor[i_associated_atom * natm + i_derivative_atom];
            const double3 dPA_dG = d3_scale(switch_function_dmuds_over_s(mu_AG, a_factor_AG) * P_A, dmuAG_dG);

            const double quadrature_weight = grid_quadrature_weights[i_grid];
            const double3 dwi_dG = d3_scale(quadrature_weight, d3_minus(d3_scale(inv(sum_P_B), dPA_dG), d3_scale(inv(sum_P_B * sum_P_B) * P_A, sum_dPB_dG)));

            dwdG[i_derivative_atom * ngrids * 3 + 0 * ngrids + i_grid] = dwi_dG.x;
            dwdG[i_derivative_atom * ngrids * 3 + 1 * ngrids + i_grid] = dwi_dG.y;
            dwdG[i_derivative_atom * ngrids * 3 + 2 * ngrids + i_grid] = dwi_dG.z;
        }
    }
}

/*
 * The first and second derivatives of the Becke weights w_g = q_g P_G(r_g) / Z(r_g), Z = sum_C P_C, of the
 * points g of atom G = atm_idx[g], which move with G, with respect to the atom coordinates. The partition is
 * that of VXCgen_grid: the pairs i > j with nu = mu_ij + a_ij (1 - mu_ij^2), s_ij = (1 - f3(nu)) / 2 and
 * s_ji = (1 + f3(nu)) / 2, P_C = prod_D s_CD. With L_C = log P_C, at fixed r_g and for atoms A, B != G,
 *     dw/dA = w t_A,  t = dL_G - dZ/Z,
 *     d2w/dAdB = w (t_A t_B + d2L_G - d2Z/Z + dZ dZ/Z^2),  dZ = sum_C P_C dL_C,  d2Z = sum_C P_C (dL_C dL_C + d2L_C);
 * the rows and columns of G follow from translational invariance. Only the C with P_C > BECKE_P_TOL Z, and of
 * their pairs (C,D) those whose log s_CD has a derivative above BECKE_L_TOL, are visited.
 *   dwdA [natm,3,ngrids]: the first derivatives (all atoms written), or NULL
 *   d2e [natm,3,natm,3]: d2e += sum_g eps[g] d2w_g/dAdB, or NULL (eps is then not read)
 *   a_factor [natm,natm]: the radii adjustment a_ij of VXCgen_grid's radii_table, or NULL
 */
#define BECKE_P_TOL 1e-15
#define BECKE_L_TOL 1e-15

void VXCbecke_weight_derivs(double *dwdA, double *d2e, const double *eps,
                            const double *coords, const double *qweights, const int *atm_idx,
                            const double *atm_coords, const double *a_factor,
                            const int ngrids, const int natm)
{
    if (ngrids <= 0 || natm <= 0) {
        return;
    }
    const size_t ng = ngrids;
    const int n3 = natm * 3;
    double *inv_R = malloc(sizeof(double) * natm * natm);
    for (int i = 0; i < natm; i++) {
        for (int j = 0; j < natm; j++) {
            const double dx = atm_coords[i*3+0] - atm_coords[j*3+0];
            const double dy = atm_coords[i*3+1] - atm_coords[j*3+1];
            const double dz = atm_coords[i*3+2] - atm_coords[j*3+2];
            inv_R[i*natm+j] = (i == j) ? 0 : 1 / sqrt(dx*dx + dy*dy + dz*dz);
        }
    }
#pragma omp parallel
{
    double *rA = malloc(sizeof(double) * natm * 5);
    double *uA = rA + natm;
    double *P = uA + natm * 3;
    double *dl = malloc(sizeof(double) * natm * natm * 2);
    double *d2l = dl + natm * natm;
    int *slot = malloc(sizeof(int) * natm * 3);
    int *satom = slot + natm;
    int *active = satom + natm;
    const size_t nmax = n3;
    double *aC = malloc(sizeof(double) * (nmax * 4 + nmax * nmax * 3));
    double *aG = aC + nmax;
    double *ZX = aG + nmax;
    double *t = ZX + nmax;
    double *bC = t + nmax;
    double *bG = bC + nmax * nmax;
    double *ZXY = bG + nmax * nmax;
    double *acc = NULL;
    if (d2e != NULL) {
        acc = calloc((size_t)n3 * n3, sizeof(double));
    }

#pragma omp for schedule(dynamic, 16)
    for (size_t g = 0; g < ng; g++) {
        const int G = atm_idx[g];
        if (dwdA != NULL) {
            for (int k = 0; k < n3; k++) {
                dwdA[k * ng + g] = 0;
            }
        }
        if (G < 0 || qweights[g] == 0) {
            continue;
        }
        const double rx = coords[g*3+0];
        const double ry = coords[g*3+1];
        const double rz = coords[g*3+2];
        for (int A = 0; A < natm; A++) {
            const double dx = atm_coords[A*3+0] - rx;
            const double dy = atm_coords[A*3+1] - ry;
            const double dz = atm_coords[A*3+2] - rz;
            const double r = sqrt(dx*dx + dy*dy + dz*dz);
            const double r_1 = (r > 1e-14) ? 1 / r : 0;
            rA[A] = r;
            uA[A*3+0] = dx * r_1;
            uA[A*3+1] = dy * r_1;
            uA[A*3+2] = dz * r_1;
            P[A] = 1;
        }
        for (int i = 0; i < natm; i++) {
            for (int j = 0; j < i; j++) {
                const double mu = (rA[i] - rA[j]) * inv_R[i*natm+j];
                const double a = (a_factor != NULL) ? a_factor[i*natm+j] : 0;
                const double nu = mu + a * (1 - mu * mu);
                const double nu_m = 1 - 2 * a * mu;
                const double nu_mm = -2 * a;
                const double f1 = (3 - nu * nu) * nu * .5;
                const double f2 = (3 - f1 * f1) * f1 * .5;
                const double F = (3 - f2 * f2) * f2 * .5;
                const double d1 = 1.5 * (1 - nu * nu);
                const double d2 = 1.5 * (1 - f1 * f1);
                const double d3 = 1.5 * (1 - f2 * f2);
                const double F_nu = d3 * d2 * d1;
                const double F_nunu = -3 * f2 * (d2 * d1) * (d2 * d1) + d3 * (-3 * f1 * d1 * d1 + d2 * -3 * nu);
                const double F_m = F_nu * nu_m;
                const double F_mm = F_nunu * nu_m * nu_m + F_nu * nu_mm;
                const double sm = .5 * (1 - F);
                const double sp = .5 * (1 + F);
                P[i] *= sm;
                P[j] *= sp;
                double l1 = 0, l2 = 0;
                if (sm > 0) {
                    l1 = -.5 * F_m / sm;
                    l2 = -.5 * F_mm / sm - l1 * l1;
                }
                dl[i*natm+j] = l1;
                d2l[i*natm+j] = l2;
                l1 = l2 = 0;
                if (sp > 0) {
                    l1 = .5 * F_m / sp;
                    l2 = .5 * F_mm / sp - l1 * l1;
                }
                dl[j*natm+i] = l1;
                d2l[j*natm+i] = l2;
            }
        }
        double Z = 0;
        for (int C = 0; C < natm; C++) {
            Z += P[C];
        }
        if (!(P[G] > BECKE_P_TOL * Z)) {
            continue;
        }
        const double w = qweights[g] * P[G] / Z;
        const double Z_1 = 1 / Z;

        // the atoms of the derivatives (G excluded)
        int nact = 0, nS = 0;
        for (int A = 0; A < natm; A++) {
            slot[A] = -1;
        }
        for (int C = 0; C < natm; C++) {
            if (!(P[C] > BECKE_P_TOL * Z)) {
                continue;
            }
            active[nact++] = C;
            for (int D = 0; D < natm; D++) {
                if (D == C || (fabs(dl[C*natm+D]) <= BECKE_L_TOL && fabs(d2l[C*natm+D]) <= BECKE_L_TOL)) {
                    continue;
                }
                if (C != G && slot[C] < 0) {
                    slot[C] = nS;
                    satom[nS++] = C;
                }
                if (D != G && slot[D] < 0) {
                    slot[D] = nS;
                    satom[nS++] = D;
                }
            }
        }
        if (nS == 0) {
            continue;
        }
        const int n = nS * 3;
        for (int p = 0; p < n; p++) {
            ZX[p] = 0;
            aG[p] = 0;
        }
        if (d2e != NULL) {
            for (int p = 0; p < n * n; p++) {
                ZXY[p] = 0;
                bG[p] = 0;
            }
        }

        for (int ic = 0; ic < nact; ic++) {
            const int C = active[ic];
            for (int p = 0; p < n; p++) {
                aC[p] = 0;
            }
            if (d2e != NULL) {
                for (int p = 0; p < n * n; p++) {
                    bC[p] = 0;
                }
            }
            for (int D = 0; D < natm; D++) {
                const double l1 = dl[C*natm+D];
                const double l2 = d2l[C*natm+D];
                if (D == C || (fabs(l1) <= BECKE_L_TOL && fabs(l2) <= BECKE_L_TOL)) {
                    continue;
                }
                // mu_ij of the pair, (i,j) = (max, min)(C,D), and its derivatives
                const int i = (C > D) ? C : D;
                const int j = (C > D) ? D : C;
                const double R_1 = inv_R[i*natm+j];
                const double R_2 = R_1 * R_1;
                const double mu = (rA[i] - rA[j]) * R_1;
                const double e[3] = {(atm_coords[i*3+0] - atm_coords[j*3+0]) * R_1,
                                     (atm_coords[i*3+1] - atm_coords[j*3+1]) * R_1,
                                     (atm_coords[i*3+2] - atm_coords[j*3+2]) * R_1};
                const double *ui = uA + i * 3;
                const double *uj = uA + j * 3;
                double mi[3], mj[3];
                for (int x = 0; x < 3; x++) {
                    mi[x] = ( ui[x] - mu * e[x]) * R_1;
                    mj[x] = (-uj[x] + mu * e[x]) * R_1;
                }
                const int si = slot[i];
                const int sj = slot[j];
                if (si >= 0) {
                    for (int x = 0; x < 3; x++) {
                        aC[si*3+x] += l1 * mi[x];
                    }
                }
                if (sj >= 0) {
                    for (int x = 0; x < 3; x++) {
                        aC[sj*3+x] += l1 * mj[x];
                    }
                }
                if (d2e == NULL) {
                    continue;
                }
                const double ri_1 = (rA[i] > 1e-14) ? 1 / rA[i] : 0;
                const double rj_1 = (rA[j] > 1e-14) ? 1 / rA[j] : 0;
                for (int x = 0; x < 3; x++) {
                    for (int y = 0; y < 3; y++) {
                        const double delta = (x == y) ? 1 : 0;
                        const double common = (-mu * (delta - e[x] * e[y]) + 2 * mu * e[x] * e[y]) * R_2;
                        if (si >= 0) {
                            const double mii = (delta - ui[x] * ui[y]) * ri_1 * R_1
                                - (ui[x] * e[y] + e[x] * ui[y]) * R_2 + common;
                            bC[(si*3+x)*n + si*3+y] += l2 * mi[x] * mi[y] + l1 * mii;
                        }
                        if (sj >= 0) {
                            const double mjj = -(delta - uj[x] * uj[y]) * rj_1 * R_1
                                - (uj[x] * e[y] + e[x] * uj[y]) * R_2 + common;
                            bC[(sj*3+x)*n + sj*3+y] += l2 * mj[x] * mj[y] + l1 * mjj;
                        }
                        if (si >= 0 && sj >= 0) {
                            const double mij = (ui[x] * e[y] + e[x] * uj[y]) * R_2 - common;
                            const double v = l2 * mi[x] * mj[y] + l1 * mij;
                            bC[(si*3+x)*n + sj*3+y] += v;
                            bC[(sj*3+y)*n + si*3+x] += v;
                        }
                    }
                }
            }
            const double PC = P[C];
            for (int p = 0; p < n; p++) {
                ZX[p] += PC * aC[p];
            }
            if (C == G) {
                for (int p = 0; p < n; p++) {
                    aG[p] = aC[p];
                }
            }
            if (d2e != NULL) {
                for (int p = 0; p < n; p++) {
                    for (int q = 0; q < n; q++) {
                        ZXY[p*n+q] += PC * (aC[p] * aC[q] + bC[p*n+q]);
                    }
                }
                if (C == G) {
                    for (int p = 0; p < n * n; p++) {
                        bG[p] = bC[p];
                    }
                }
            }
        }

        for (int p = 0; p < n; p++) {
            t[p] = aG[p] - ZX[p] * Z_1;
        }
        if (dwdA != NULL) {
            for (int s = 0; s < nS; s++) {
                const int A = satom[s];
                for (int x = 0; x < 3; x++) {
                    const double v = w * t[s*3+x];
                    dwdA[(A*3+x) * ng + g] = v;
                    dwdA[(G*3+x) * ng + g] -= v;
                }
            }
        }
        if (d2e != NULL) {
            const double fac = w * eps[g];
            for (int sp_ = 0; sp_ < nS; sp_++) {
                const int A = satom[sp_];
                for (int sq = 0; sq < nS; sq++) {
                    const int B = satom[sq];
                    for (int x = 0; x < 3; x++) {
                        for (int y = 0; y < 3; y++) {
                            const int p = sp_*3+x, q = sq*3+y;
                            const double h = fac * (t[p] * t[q] + bG[p*n+q] - ZXY[p*n+q] * Z_1
                                                    + ZX[p] * ZX[q] * Z_1 * Z_1);
                            acc[(A*3+x)*n3 + B*3+y] += h;
                            acc[(G*3+x)*n3 + B*3+y] -= h;
                            acc[(A*3+x)*n3 + G*3+y] -= h;
                            acc[(G*3+x)*n3 + G*3+y] += h;
                        }
                    }
                }
            }
        }
    }
    if (d2e != NULL) {
#pragma omp critical
        for (size_t k = 0; k < (size_t)n3 * n3; k++) {
            d2e[k] += acc[k];
        }
        free(acc);
    }
    free(rA);
    free(dl);
    free(slot);
    free(aC);
}
    free(inv_R);
}

inline double _lko_sat_func(double x) {
    int m;
    double tot = 0;
    x = x / RCUT_LKO;
    double xm = 1;
    for (m = 1; m <= MC_LKO; m++) {
        // tot += pow(x, m) / m;
        xm *= x;
        tot += xm / m;
    }
    return RCUT_LKO * (1 - exp(-tot));
}

inline double _lko_sat_deriv(double x) {
    int m;
    double tot = 0;
    double dtot = 0;
    x = x / RCUT_LKO;
    double xm = 1;
    for (m = 1; m <= MC_LKO; m++) {
        // tot += pow(x, m) / m;
        // dtot += pow(x, m-1);
        dtot += xm;
        xm *= x;
        tot += xm / m;
    }
    return exp(-tot) * dtot;
}

void VXCgen_grid_lko(double *out, double *coords, double *atm_coords,
                     double *radii_table, int natm, int ngrids)
{
    const size_t Ngrids = ngrids;
    int i, j;
    double dx, dy, dz;
    double *atom_dist = malloc(sizeof(double) * natm*natm);
#pragma omp parallel private(i, j, dx, dy, dz)
{
    int ij;
    const int num_ij = natm * (natm + 1) / 2;
#pragma omp for nowait schedule(static)
    for (ij = 0; ij < num_ij; ij++) {
        i = (int)(sqrt(2*ij+.25) - .5 + 1e-7);
        j = ij - i*(i+1)/2;
        dx = atm_coords[i*3+0] - atm_coords[j*3+0];
        dy = atm_coords[i*3+1] - atm_coords[j*3+1];
        dz = atm_coords[i*3+2] - atm_coords[j*3+2];
        atom_dist[i*natm+j] = 1 / _lko_sat_func(sqrt(dx*dx + dy*dy + dz*dz));
    }
}

#pragma omp parallel private(i, j, dx, dy, dz)
{
    size_t ig_start, ig_end;
    NPomp_split(&ig_start, &ig_end, (size_t)ngrids);
    double *grid_dist = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *buf = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *min_dist_i = malloc(sizeof(double) * natm);
    double *min_dist_n = malloc(sizeof(double) * GRIDS_BLOCK);
    double *g = malloc(sizeof(double) * GRIDS_BLOCK);
    size_t ig0, n, ngs;
    double fac, s;
    double max_min_dist;
// #pragma omp for nowait schedule(static)
    for (ig0 = ig_start; ig0 < ig_end; ig0 += GRIDS_BLOCK) {
        ngs = MIN(ig0 + GRIDS_BLOCK, ig_end) - ig0;
        for (n = 0; n < ngs; n++) {
            min_dist_n[n] = 1e10;
        }
        for (i = 0; i < natm; i++) {
            min_dist_i[i] = 1e10;
            // this loop cannot be done with ivdep because of min_dist_i
            for (n = 0; n < ngs; n++) {
                dx = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                dy = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                dz = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
                grid_dist[i*GRIDS_BLOCK+n] = sqrt(dx*dx + dy*dy + dz*dz);
                buf[i*GRIDS_BLOCK+n] = 1;
                // for screening
                min_dist_i[i] = MIN(grid_dist[i*GRIDS_BLOCK+n], min_dist_i[i]);
                min_dist_n[n] = MIN(grid_dist[i*GRIDS_BLOCK+n], min_dist_n[n]);
            }
        }
        max_min_dist = 0.0;
        for (n = 0; n < ngs; n++) {
            max_min_dist = MAX(max_min_dist, min_dist_n[n]);
        }

        for (i = 0; i < natm; i++) {
        for (j = 0; j < i; j++) {
            // for screening
            if ((min_dist_i[i] > max_min_dist + RCUT_LKO) &&
                (min_dist_i[j] > max_min_dist + RCUT_LKO)) {
                continue;
            }
            fac = atom_dist[i*natm+j];
#pragma GCC ivdep
            for (n = 0; n < ngs; n++) {
                g[n] = (grid_dist[i*GRIDS_BLOCK+n] -
                    grid_dist[j*GRIDS_BLOCK+n]) * fac;
                g[n] = MAX(-1, g[n]);
                g[n] = MIN(1, g[n]);
            }
            if (radii_table != NULL) {
                fac = radii_table[i*natm+j];
#pragma GCC ivdep
                for (n = 0; n < ngs; n++) {
                    g[n] += fac * (1 - g[n]*g[n]);
                }
            }
            /*for (n = 0; n < ngs; n++) {
                g[n] = (3 - g[n]*g[n]) * g[n] * .5;
            }
            for (n = 0; n < ngs; n++) {
                g[n] = (3 - g[n]*g[n]) * g[n] * .5;
            }
            for (n = 0; n < ngs; n++) {
                g[n] = ((3 - g[n]*g[n]) * g[n] * .5) * .5;
            }
            for (n = 0; n < ngs; n++) {
                buf[i*GRIDS_BLOCK+n] *= .5 - g[n];
                buf[j*GRIDS_BLOCK+n] *= .5 + g[n];
            }*/
#pragma GCC ivdep
            for (n = 0; n < ngs; n++) {
                s = g[n];
                s = (3 - s*s) * s * .5;
                s = (3 - s*s) * s * .5;
                s = ((3 - s*s) * s * .5) * .5;
                buf[i*GRIDS_BLOCK+n] *= .5 - s;
                buf[j*GRIDS_BLOCK+n] *= .5 + s;
            }
        } }

        for (i = 0; i < natm; i++) {
            for (n = 0; n < ngs; n++) {
                out[i*Ngrids+ig0+n] = buf[i*GRIDS_BLOCK+n];
            }
        }
    }
    free(g);
    free(buf);
    free(grid_dist);
    free(min_dist_i);
    free(min_dist_n);
}
    free(atom_dist);
}

void VXCgen_grid_lko_deriv(double *out, double *dw, double *coords, double *atm_coords,
                           double *radii_table, int natm, int ngrids, int *atm_idx)
{
    const size_t Ngrids = ngrids;
    int i, j;
    double dx, dy, dz, dr;
    double sat_deriv;
    double *atom_dist = malloc(sizeof(double) * natm*natm);
    double *dadx = malloc(sizeof(double) * natm*natm);
    double *dady = malloc(sizeof(double) * natm*natm);
    double *dadz = malloc(sizeof(double) * natm*natm);
    double *outx = out + 1*natm*ngrids;
    double *outy = out + 2*natm*ngrids;
    double *outz = out + 3*natm*ngrids;
#pragma omp parallel private(i, j, dx, dy, dz, dr, sat_deriv)
{   
#pragma omp for
    for (i = 0; i < natm; i++) {
        for (j = 0; j < i; j++) {
            dx = atm_coords[i*3+0] - atm_coords[j*3+0];
            dy = atm_coords[i*3+1] - atm_coords[j*3+1];
            dz = atm_coords[i*3+2] - atm_coords[j*3+2];
            dr = sqrt(dx*dx + dy*dy + dz*dz);
            atom_dist[i*natm+j] = 1 / _lko_sat_func(dr);
            sat_deriv = _lko_sat_deriv(dr) * pow(atom_dist[i*natm+j], 3);
            dadx[i*natm+j] = -dx * sat_deriv;
            dady[i*natm+j] = -dy * sat_deriv;
            dadz[i*natm+j] = -dz * sat_deriv;
        }
    }
}

#pragma omp parallel private(i, j, dx, dy, dz, dr)
{
    double *grid_dist = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *buf = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *bufx = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *bufy = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *bufz = malloc(sizeof(double) * natm*GRIDS_BLOCK);
    double *min_dist_i = malloc(sizeof(double) * natm);
    double *min_dist_n = malloc(sizeof(double) * GRIDS_BLOCK);
    double *g = malloc(sizeof(double) * GRIDS_BLOCK);
    double *dg = malloc(sizeof(double) * GRIDS_BLOCK);
    size_t ig0, n, ngs;
    double fac, dfacx, dfacy, dfacz, tmp;
    double max_min_dist;
    int ia_p;
#pragma omp for nowait schedule(static)
    for (ig0 = 0; ig0 < Ngrids; ig0 += GRIDS_BLOCK) {
        ngs = MIN(Ngrids-ig0, GRIDS_BLOCK);
        for (n = 0; n < ngs; n++) {
            min_dist_n[n] = 1e10;
        }
        for (i = 0; i < natm; i++) {
            min_dist_i[i] = 1e10;
            for (n = 0; n < ngs; n++) {
                dx = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                dy = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                dz = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
                grid_dist[i*GRIDS_BLOCK+n] = sqrt(dx*dx + dy*dy + dz*dz);
                buf[i*GRIDS_BLOCK+n] = 1;
                bufx[i*GRIDS_BLOCK+n] = 0;
                bufy[i*GRIDS_BLOCK+n] = 0;
                bufz[i*GRIDS_BLOCK+n] = 0;
                // for screening
                min_dist_i[i] = MIN(grid_dist[i*GRIDS_BLOCK+n], min_dist_i[i]);
                min_dist_n[n] = MIN(grid_dist[i*GRIDS_BLOCK+n], min_dist_n[n]);
            }
        }
        max_min_dist = 0.0;
        for (n = 0; n < ngs; n++) {
            max_min_dist = MAX(max_min_dist, min_dist_n[n]);
        }

        for (i = 0; i < natm; i++) {
        for (j = 0; j < i; j++) {
            // for screening
            if ((min_dist_i[i] > max_min_dist + RCUT_LKO) &&
                (min_dist_i[j] > max_min_dist + RCUT_LKO)) {
                continue;
            }
            fac = atom_dist[i*natm+j];
            for (n = 0; n < ngs; n++) {
                g[n] = (grid_dist[i*GRIDS_BLOCK+n] -
                    grid_dist[j*GRIDS_BLOCK+n]) * fac;
                g[n] = MAX(-1, g[n]);
                g[n] = MIN(1, g[n]);
                dg[n] = 1.0;
                if (g[n] == -1 || g[n] == 1) {
                    dg[n] = 0.0;
                }
            }
            if (radii_table != NULL) {
                fac = radii_table[i*natm+j];
                for (n = 0; n < ngs; n++) {
                    dg[n] -= 2 * fac * g[n];
                    g[n] += fac * (1 - g[n]*g[n]);
                }
            }
            for (n = 0; n < ngs; n++) {
                dg[n] *= 1.5 * (1 - g[n]*g[n]);
                g[n] = (3 - g[n]*g[n]) * g[n] * .5;
            }
            for (n = 0; n < ngs; n++) {
                dg[n] *= 1.5 * (1 - g[n]*g[n]);
                g[n] = (3 - g[n]*g[n]) * g[n] * .5;
            }
            for (n = 0; n < ngs; n++) {
                dg[n] *= 0.75 * (1 - g[n]*g[n]);
                g[n] = ((3 - g[n]*g[n]) * g[n] * .5) * .5;
            }
            fac = atom_dist[i*natm+j];
            dfacx = dadx[i*natm+j];
            dfacy = dady[i*natm+j];
            dfacz = dadz[i*natm+j];
            for (n = 0; n < ngs; n++) {
                ia_p = atm_idx[ig0 + n];
                if (ia_p < 0) {
                    continue;
                }
                // Skip saturated switches (zero derivative): fast-math can
                // reassociate .5 +/- g + 1e-200, losing the epsilon and causing 0/0.
                if (g[n] <= -.5 || g[n] >= .5) {
                    continue;
                }
                tmp = dw[j*Ngrids+ig0+n] * dg[n] / (.5 + g[n] + 1e-200);
                tmp-= dw[i*Ngrids+ig0+n] * dg[n] / (.5 - g[n] + 1e-200);
                tmp *= (grid_dist[i*GRIDS_BLOCK+n] -
                        grid_dist[j*GRIDS_BLOCK+n]);
                bufx[i*GRIDS_BLOCK+n] += tmp * dfacx;
                bufy[i*GRIDS_BLOCK+n] += tmp * dfacy;
                bufz[i*GRIDS_BLOCK+n] += tmp * dfacz;
                bufx[j*GRIDS_BLOCK+n] -= tmp * dfacx;
                bufy[j*GRIDS_BLOCK+n] -= tmp * dfacy;
                bufz[j*GRIDS_BLOCK+n] -= tmp * dfacz;

                tmp = dw[j*Ngrids+ig0+n] * dg[n] / (.5 + g[n] + 1e-200);
                tmp-= dw[i*Ngrids+ig0+n] * dg[n] / (.5 - g[n] + 1e-200);
                tmp *= fac;
                if (i != ia_p) {
                    dx = coords[0*Ngrids+ig0+n] - atm_coords[i*3+0];
                    dy = coords[1*Ngrids+ig0+n] - atm_coords[i*3+1];
                    dz = coords[2*Ngrids+ig0+n] - atm_coords[i*3+2];
                    dr = grid_dist[i*GRIDS_BLOCK+n];
                    dx = tmp * dx / (dr + 1e-200);
                    dy = tmp * dy / (dr + 1e-200);
                    dz = tmp * dz / (dr + 1e-200);
                    bufx[i*GRIDS_BLOCK+n] -= dx;
                    bufy[i*GRIDS_BLOCK+n] -= dy;
                    bufz[i*GRIDS_BLOCK+n] -= dz;
                    bufx[ia_p*GRIDS_BLOCK+n] += dx;
                    bufy[ia_p*GRIDS_BLOCK+n] += dy;
                    bufz[ia_p*GRIDS_BLOCK+n] += dz;
                }
                if (j != ia_p) {
                    dx = coords[0*Ngrids+ig0+n] - atm_coords[j*3+0];
                    dy = coords[1*Ngrids+ig0+n] - atm_coords[j*3+1];
                    dz = coords[2*Ngrids+ig0+n] - atm_coords[j*3+2];
                    dr = grid_dist[j*GRIDS_BLOCK+n];
                    dx = tmp * dx / (dr + 1e-200);
                    dy = tmp * dy / (dr + 1e-200);
                    dz = tmp * dz / (dr + 1e-200);
                    bufx[j*GRIDS_BLOCK+n] += dx;
                    bufy[j*GRIDS_BLOCK+n] += dy;
                    bufz[j*GRIDS_BLOCK+n] += dz;
                    bufx[ia_p*GRIDS_BLOCK+n] -= dx;
                    bufy[ia_p*GRIDS_BLOCK+n] -= dy;
                    bufz[ia_p*GRIDS_BLOCK+n] -= dz;
                }
            }
        } }

        for (i = 0; i < natm; i++) {
            for (n = 0; n < ngs; n++) {
                out[i*Ngrids+ig0+n] = buf[i*GRIDS_BLOCK+n];
                outx[i*Ngrids+ig0+n] = bufx[i*GRIDS_BLOCK+n];
                outy[i*Ngrids+ig0+n] = bufy[i*GRIDS_BLOCK+n];
                outz[i*Ngrids+ig0+n] = bufz[i*GRIDS_BLOCK+n];
            }
        }
    }
    free(g);
    free(dg);
    free(buf);
    free(grid_dist);
    free(bufx);
    free(bufy);
    free(bufz);
    free(min_dist_i);
    free(min_dist_n);
}
    free(atom_dist);
    free(dadx);
    free(dady);
    free(dadz);
}

/*
 * The potential u times the motion of the points applied to the orbital response of the density (the d2rho
 * terms of the Hessian's grid response, hessian.rks._grid_d2rho_dot): for X[c] = ao[c] dm and each AO mu,
 *   q[mu,D,d] = sum_g  2 (d_dD mu Vg + d_D mu W_d + Phi3_dD X0 + Phi2_D X_d)
 *                    + u_tau sum_x (d_dDx mu X_x + d_Dx mu X_dx)          (meta-GGA)
 * with Vg = u0 X0 + sum_x u_x X_x, W_d = u0 X_d + sum_x u_x X_dx, Phi2_D = sum_x u_x d_Dx mu,
 * Phi3_dD = sum_x u_x d_dDx mu; for LDA q = 2 sum_g u0 (d_dD mu X0 + d_D mu X_d).
 *   ao [ncomp_ao,ngrids,nao] (to third derivatives, PySCF's order), X [nX,ngrids,nao] (4 for LDA, else 10),
 *   u [ncomp,ngrids], xctype 0 LDA, 1 GGA, 2 meta-GGA; q [nao,3,3] is added to.
 */
static const int D2IDX[3][3] = {{4, 5, 6}, {5, 7, 8}, {6, 8, 9}};
static const int D3IDX[3][3][3] = {
    {{10, 11, 12}, {11, 13, 14}, {12, 14, 15}},
    {{11, 13, 14}, {13, 16, 17}, {14, 17, 18}},
    {{12, 14, 15}, {14, 17, 18}, {15, 18, 19}}};

void VXCgrid_d2rho_dot(double *q, const double *ao, const double *X, const double *u,
                       const int xctype, const int ngrids, const int nao)
{
    const size_t Ng = ngrids;
    const size_t stride = Ng * nao;
    const int GBLK = 64;
#pragma omp parallel
{
    double *qt = calloc((size_t)nao * 9, sizeof(double));
#pragma omp for schedule(dynamic, 1)
    for (int g0 = 0; g0 < ngrids; g0 += GBLK) {
        const int g1 = (g0 + GBLK < ngrids) ? g0 + GBLK : ngrids;
        for (int g = g0; g < g1; g++) {
            const size_t off = (size_t)g * nao;
            const double u0 = u[g];
            const double *X0 = X + off;
            if (xctype == 0) {
                for (int D = 0; D < 3; D++) {
                    const double *aD = ao + (1+D) * stride + off;
                    for (int d = 0; d < 3; d++) {
                        const double *adD = ao + D2IDX[d][D] * stride + off;
                        const double *Xd = X + (1+d) * stride + off;
                        double *qq = qt + D * 3 + d;
#pragma omp simd
                        for (int m = 0; m < nao; m++) {
                            qq[m*9] += 2 * u0 * (adD[m] * X0[m] + aD[m] * Xd[m]);
                        }
                    }
                }
                continue;
            }
            const double ux[3] = {u[Ng+g], u[2*Ng+g], u[3*Ng+g]};
            const double ut = (xctype == 2) ? u[4*Ng+g] : 0;
            const double *X1[3] = {X + stride + off, X + 2*stride + off, X + 3*stride + off};
            for (int D = 0; D < 3; D++) {
                const double *aD = ao + (1+D) * stride + off;
                const double *aDx[3] = {ao + D2IDX[D][0] * stride + off, ao + D2IDX[D][1] * stride + off,
                                        ao + D2IDX[D][2] * stride + off};
                for (int d = 0; d < 3; d++) {
                    const double *adD = ao + D2IDX[d][D] * stride + off;
                    const double *adDx[3] = {ao + D3IDX[d][D][0] * stride + off, ao + D3IDX[d][D][1] * stride + off,
                                             ao + D3IDX[d][D][2] * stride + off};
                    const double *Xd = X1[d];
                    const double *Xdx[3] = {X + D2IDX[d][0] * stride + off, X + D2IDX[d][1] * stride + off,
                                            X + D2IDX[d][2] * stride + off};
                    double *qq = qt + D * 3 + d;
#pragma omp simd
                    for (int m = 0; m < nao; m++) {
                        const double Vg = u0 * X0[m] + ux[0] * X1[0][m] + ux[1] * X1[1][m] + ux[2] * X1[2][m];
                        const double Wd = u0 * Xd[m] + ux[0] * Xdx[0][m] + ux[1] * Xdx[1][m] + ux[2] * Xdx[2][m];
                        const double phi2 = ux[0] * aDx[0][m] + ux[1] * aDx[1][m] + ux[2] * aDx[2][m];
                        const double phi3 = ux[0] * adDx[0][m] + ux[1] * adDx[1][m] + ux[2] * adDx[2][m];
                        double v = 2 * (adD[m] * Vg + aD[m] * Wd + phi3 * X0[m] + phi2 * Xd[m]);
                        if (xctype == 2) {
                            v += ut * (adDx[0][m] * X1[0][m] + adDx[1][m] * X1[1][m] + adDx[2][m] * X1[2][m]
                                       + aDx[0][m] * Xdx[0][m] + aDx[1][m] * Xdx[1][m] + aDx[2][m] * Xdx[2][m]);
                        }
                        qq[m*9] += v;
                    }
                }
            }
        }
    }
#pragma omp critical
    for (size_t k = 0; k < (size_t)nao * 9; k++) {
        q[k] += qt[k];
    }
    free(qt);
}
}
