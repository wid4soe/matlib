#pragma once
#ifndef TINYMPC_MATLIB_RVV_H
#define TINYMPC_MATLIB_RVV_H

#include <cstdio>
#include <climits>
#include <cmath>

#include "riscv_vector.h"

extern "C"
{

// RVV-specific function declarations
double maxcoeff_rvv(double *a, int n, int m);
double mincoeff_rvv(double *a, int n, int m);
double matnorm_rvv(double *a, int n, int m);
void matneg_rvv(double *a, double *b, int n, int m);
void cwiseabs_rvv(double *a, double *b, int n, int m);
void cwisemin_rvv(double *a, double *b, double *c, int n, int m);
void cwisemax_rvv(double *a, double *b, double *c, int n, int m);
void cwisemul_rvv(double *a, double *b, double *c, int n, int m);
void matmul_rvv(double *a, double *b, double *c, int n, int m, int o);
void matvec_rvv(double *a, double *b, double *c, int n, int m);
void matvec_transpose_rvv(double *a, double *b, double *c, int n, int m);
void matmulf_rvv(double *a, double *b, double f, int n, int m);
void matsub_rvv(double *a, double *b, double *c, int n, int m);
void matadd_rvv(double *a, double *b, double *c, int n, int m);
void transpose_rvv(double *a, double *b, int n, int m);
void matcopy_rvv(const double *a, double *b, int n, int m);
void matset_rvv(double *a, double f, int n, int m);
void matsetv_rvv(double *a, double *f, int n, int m);

#define maxcoeff maxcoeff_rvv
#define mincoeff mincoeff_rvv
#define matnorm matnorm_rvv
#define matneg matneg_rvv
#define cwiseabs cwiseabs_rvv
#define cwisemin cwisemin_rvv
#define cwisemax cwisemax_rvv
#define cwisemul cwisemul_rvv
#define matmul matmul_rvv
#define matvec matvec_rvv
#define matvec_transpose matvec_transpose_rvv
#define matmulf matmulf_rvv
#define matsub matsub_rvv
#define matadd matadd_rvv
#define transpose transpose_rvv
#define matcopy matcopy_rvv
#define matset matset_rvv
#define matsetv matsetv_rvv


// matrix maximum coefficient
inline double maxcoeff_rvv(double *ptr_a, int n, int m) {
    double max = std::numeric_limits<double>::min();
    vfloat64m1_t vec_max = __riscv_vfmv_s_f_f64m1(max, 1);
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e64(k);
        vfloat64_t vec_a = __riscv_vle64_v_f64(ptr_a + l, vl);
        vec_max = __riscv_vfredmax_vs_f64_f64(vec_a, vec_max, vl);
    }
    max = __riscv_vfmv_f_s_f64m1_f64(vec_max);
    return max;
}

// matrix min coefficient
inline double mincoeff_rvv(double *ptr_a, int n, int m) {
    double min = std::numeric_limits<double>::max();
    vfloat64m1_t vec_min = __riscv_vfmv_s_f_f64m1(min, 1);
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e64(k);
        vfloat64_t vec_a = __riscv_vle64_v_f64(ptr_a + l, vl);
        vec_min = __riscv_vfredmin_vs_f64_f64(vec_a, vec_min, vl);
    }
    min = __riscv_vfmv_f_s_f64m1_f64(vec_min);
    return min;
}

// matrix unary negative
inline void matneg_rvv(double *ptr_a, double *ptr_b, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e64(k);
        vfloat64_t vec_a = __riscv_vle64_v_f64(ptr_a + l, vl);
        vfloat64_t vec_b = __riscv_vfneg_v_f64(vec_a, vl);
        __riscv_vse64_v_f64(ptr_b + l, vec_b, vl);
    }
}

// matrix l2 norm
inline double matnorm_rvv(double *ptr_a, int n, int m) {
    int k = m * n, l = 0;
    size_t vlmax = __riscv_vsetvlmax_e64();
    vfloat64m1_t vec_zero = __riscv_vfmv_v_f_f64m1(0, vlmax);
    vfloat64_t vec_s = __riscv_vfmv_v_f_f64(0, vlmax);
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e64(k);
        vfloat64_t vec_a = __riscv_vle64_v_f64(ptr_a + l, vl);
        vec_s = __riscv_vfmacc_vv_f64(vec_s, vec_a, vec_a, vl);
    }
    vfloat64m1_t vec_sum = __riscv_vfredusum_vs_f64_f64(vec_s, vec_zero, vlmax);
    double sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
    return sqrt(sum);
}

// matrix coefficient-wise abs
inline void cwiseabs_rvv(double *ptr_a, double *ptr_b, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e64(k);
        vfloat64_t vec_a = __riscv_vle64_v_f64(ptr_a + l, vl);
        vfloat64_t vec_b = __riscv_vfabs_v_f64(vec_a, vl);
        __riscv_vse64_v_f64(ptr_b + l, vec_b, vl);
    }
}

// matrix coefficient-wise min
inline void cwisemin_rvv(double *ptr_a, double *ptr_b, double *ptr_c, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e64(k);
        vfloat64_t vec_a = __riscv_vle64_v_f64(ptr_a + l, vl);
        vfloat64_t vec_b = __riscv_vle64_v_f64(ptr_b + l, vl);
        vfloat64_t vec_c = __riscv_vfmin_vv_f64(vec_a, vec_b, vl);
        __riscv_vse64_v_f64(ptr_c + l, vec_c, vl);
    }
}

// matrix coefficient-wise multiplication
inline void cwisemul_rvv(double *ptr_a, double *ptr_b, double *ptr_c, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e64(k);
        vfloat64_t vec_a = __riscv_vle64_v_f64(ptr_a + l, vl);
        vfloat64_t vec_b = __riscv_vle64_v_f64(ptr_b + l, vl);
        vfloat64_t vec_c = __riscv_vfmul_vv_f64(vec_a, vec_b, vl);
        __riscv_vse64_v_f64(ptr_c + l, vec_c, vl);
    }
}

// matrix coefficient-wise max
inline void cwisemax_rvv(double *ptr_a, double *ptr_b, double *ptr_c, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e64(k);
        vfloat64_t vec_a = __riscv_vle64_v_f64(ptr_a + l, vl);
        vfloat64_t vec_b = __riscv_vle64_v_f64(ptr_b + l, vl);
        vfloat64_t vec_c = __riscv_vfmax_vv_f64(vec_a, vec_b, vl);
        __riscv_vse64_v_f64(ptr_c + l, vec_c, vl);
    }
}

// matrix multiplication, note B is not [o][m]
// A[n][o], B[m][o] --> C[n][m];
inline void matmul_rvv(double *a, double *b, double *c, int n, int m, int o) {
    size_t vlmax = __riscv_vsetvlmax_e64();
    vfloat64m1_t vec_zero = __riscv_vfmv_v_f_f64m1(0, vlmax);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            double *ptr_a = a + i * o; // row major
            double *ptr_b = b + j * o; // column major
            int k = o;
            vfloat64_t vec_s = __riscv_vfmv_v_f_f64(0, vlmax);
            for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
                vl = __riscv_vsetvl_e64(k);
                vfloat64_t vec_a = __riscv_vle64_v_f64(ptr_a, vl);
                vfloat64_t vec_b = __riscv_vle64_v_f64(ptr_b, vl);
                vec_s = __riscv_vfmacc_vv_f64(vec_s, vec_a, vec_b, vl);
            }
            vfloat64m1_t vec_sum = __riscv_vfredusum_vs_f64_f64(vec_s, vec_zero, vlmax);
            double sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
            c[i * m + j] = sum;
        }
    }
}

// matrix multiplication, note B is not [o][m]
// A[n][o], B[m][o] --> C[n][m];
inline void matmul_rvv_new(double *a, double *b, double *c, int n, int m, int o) {
    size_t vlmax = __riscv_vsetvlmax_e64();
    vfloat64m1_t vec_zero = __riscv_vfmv_v_f_f64m1(0, vlmax);
    int v = n;
    size_t vl = 0;
    for (int i = 0; i < n; v -= vl, i += vl) {
        vl = __riscv_vsetvl_e64(v);
        vfloat64_t vec_c = __riscv_vfmv_v_f_f64(0, vlmax);
        double *ptr_a = a + i * o; // row major
        for (int j = 0; j < m; j++) {
            double *ptr_b = b + j * o; // column major
            double *ptr_c = c + i * m + j; // row major
            for (int k = 0; k < o; k++) {
                vfloat64_t vec_a = __riscv_vlse64_v_f64(ptr_a + k, o * sizeof(double), vl);
                vec_c = __riscv_vfmacc_vf_f64(vec_c, *(ptr_b + k), vec_a, vl);
            }
            __riscv_vsse64_v_f64(ptr_c, m * sizeof(double), vec_c, vl);
        }
    }
}

/*  a is row major
 *        j
 *    1 2 3 4     9
 *  i 5 6 7 8  *  6
 *    9 8 7 6     5 j
 *                4
 */
inline void matvec_rvv(double *a, double *b, double *c, int n, int m) {
    size_t vlmax = __riscv_vsetvlmax_e64();
    vfloat64m1_t vec_zero = __riscv_vfmv_v_f_f64m1(0, vlmax);
    for (int i = 0; i < n; ++i) {
        int k = m;
        double *ptr_a = a + i * m; // row major
        double *ptr_b = b;
        vfloat64_t vec_s = __riscv_vfmv_v_f_f64(0, vlmax);
        for (size_t vl; k > 0; k -= vl, ptr_a += vl, ptr_b += vl) {
            vl = __riscv_vsetvl_e64(k);
            vfloat64_t vec_a = __riscv_vle64_v_f64(ptr_a, vl);
            vfloat64_t vec_b = __riscv_vle64_v_f64(ptr_b, vl);
            vec_s = __riscv_vfmacc_vv_f64(vec_s, vec_a, vec_b, vl);
        }
        vfloat64m1_t vec_sum = __riscv_vfredusum_vs_f64_f64(vec_s, vec_zero, vlmax);
        double sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
        c[i] = sum;
    }
}

/*  a is col major
 *      j         i
 *  9 6 5 4  *  1 5 9
 *              2 6 8
 *              3 7 7 j
 *              4 8 6
 */
inline void matvec_transpose_rvv(double *a, double *b, double *c, int n, int m) {
    size_t vlmax = __riscv_vsetvlmax_e64();
    vfloat64m1_t vec_zero = __riscv_vfmv_v_f_f64m1(0, vlmax);
    for (int i = 0; i < m; ++i) {
        int k = n;
        double *ptr_a = a + i; // column major
        double *ptr_b = b;
        vfloat64_t vec_s = __riscv_vfmv_v_f_f64(0, vlmax);
        for (size_t vl; k > 0; k -= vl, ptr_a += m * vl, ptr_b += vl) {
            vl = __riscv_vsetvl_e64(k);
            vfloat64_t vec_a = __riscv_vlse64_v_f64(ptr_a, m * sizeof(double), vl);
            vfloat64_t vec_b = __riscv_vle64_v_f64(ptr_b, vl);
            vec_s = __riscv_vfmacc_vv_f64(vec_s, vec_a, vec_b, vl);
        }
        vfloat64m1_t vec_sum = __riscv_vfredusum_vs_f64_f64(vec_s, vec_zero, vlmax);
        double sum = __riscv_vfmv_f_s_f64m1_f64(vec_sum);
        c[i] = sum;
    }
}

// matrix scalar multiplication
inline void matmulf_rvv(double *ptr_a, double *ptr_b, double f, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e64(k);
        vfloat64_t vec_a = __riscv_vle64_v_f64(ptr_a + l, vl);
        vfloat64_t vec_b = __riscv_vfmul_vf_f64(vec_a, f, vl);
        __riscv_vse64_v_f64(ptr_b + l, vec_b, vl);
    }
}

// matrix subtraction
inline void matsub_rvv(double *ptr_a, double *ptr_b, double *ptr_c, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e64(k);
        vfloat64_t vec_a = __riscv_vle64_v_f64(ptr_a + l, vl);
        vfloat64_t vec_b = __riscv_vle64_v_f64(ptr_b + l, vl);
        vfloat64_t vec_c = __riscv_vfsub_vv_f64(vec_a, vec_b, vl);
        __riscv_vse64_v_f64(ptr_c + l, vec_c, vl);
    }
}

// matrix addition
inline void matadd_rvv(double *ptr_a, double *ptr_b, double *ptr_c, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e64(k);
        vfloat64_t vec_a = __riscv_vle64_v_f64(ptr_a + l, vl);
        vfloat64_t vec_b = __riscv_vle64_v_f64(ptr_b + l, vl);
        vfloat64_t vec_c = __riscv_vfadd_vv_f64(vec_a, vec_b, vl);
        __riscv_vse64_v_f64(ptr_c + l, vec_c, vl);
    }
}

// matrix transpose
inline void transpose_rvv(double *a, double *b, int n, int m) {
    for (int j = 0; j < m; ++j) {
        double *ptr_a = a + j;
        double *ptr_b = b + j * n;
        int k = n;
        int l = 0;
        for (size_t vl; k > 0; k -= vl, l += vl, ptr_a = a + l * m + j, ptr_b += vl) {
            vl = __riscv_vsetvl_e64(k);
            vfloat64_t vec_a = __riscv_vlse64_v_f64(ptr_a, sizeof(double) * m, vl);
            __riscv_vse64(ptr_b, vec_a, vl);
        }
    }
};

// matrix copy
inline void matcopy_rvv(const double *ptr_a, double *ptr_b, int n, int m) {
    int k = n * m, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e64(k);
        vfloat64_t vec_a = __riscv_vle64_v_f64(ptr_a + l, vl);
        __riscv_vse64_v_f64(ptr_b + l, vec_a, vl);
    }
}

inline void matset_rvv(double *ptr_a, double f, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e64(k);
        vfloat64_t vec_a = __riscv_vfmv_v_f_f64(f, vl);
        __riscv_vse64_v_f64(ptr_a + l, vec_a, vl);
    }
}

inline void matsetv_rvv(double *ptr_a, double *f, int n, int m) {
    int k = m * n, l = 0;
    for (size_t vl; k > 0; k -= vl, l += vl) {
        vl = __riscv_vsetvl_e64(k);
        vfloat64_t vec_f = __riscv_vle64_v_f64(f + l, vl);;
        __riscv_vse64_v_f64(ptr_a + l, vec_f, vl);
    }
}

}
#endif //TINYMPC_MATLIB_RVV_H
