//
// Created by widyadewi on 5/1/24.
//

#pragma once
#ifndef TINYMPC_MATLIB_RVVU_H
#define TINYMPC_MATLIB_RVVU_H

#include <cstdio>
#include <climits>
#include <cmath>

#include "riscv_vector.h"
#include "matlib_rvv.h"

extern "C"
{

#undef maxcoeff
#undef mincoeff
#undef matnorm
#undef matneg
#undef cwiseabs
#undef cwisemin
#undef cwisemax
#undef cwisemul
#undef matmul
#undef matvec
#undef matvec_transpose
#undef matmulf
#undef matsub
#undef matadd
#undef transpose
#undef matcopy
#undef matset
#undef matsetv
#define maxcoeff(...) __rvvu__maxcoeff(__VA_ARGS__)
#define mincoeff(...) __rvvu__mincoeff(__VA_ARGS__)
#define matnorm(...) __rvvu__matnorm(__VA_ARGS__)
#define matneg(...) __rvvu__matneg(__VA_ARGS__)
#define cwiseabs(...) __rvvu__cwiseabs(__VA_ARGS__)
#define cwisemin(...) __rvvu__cwisemin(__VA_ARGS__)
#define cwisemax(...) __rvvu__cwisemax(__VA_ARGS__)
#define cwisemul(...) __rvvu__cwisemul(__VA_ARGS__)
#define matmul(...) __rvvu__matmul(__VA_ARGS__)
#define matvec(...) __rvvu__matvec(__VA_ARGS__)
#define matvec_transpose(...) __rvvu__matvec_transpose(__VA_ARGS__)
#define matmulf(...) __rvvu__matmulf(__VA_ARGS__)
#define matsub(...) __rvvu__matsub(__VA_ARGS__)
#define matadd(...) __rvvu__matadd(__VA_ARGS__)
#define transpose(...) __rvvu__transpose(__VA_ARGS__)
#define matcopy(...) __rvvu__matcopy(__VA_ARGS__)
#define matset(...) __rvvu__matset(__VA_ARGS__)
#define matsetv(...) __rvvu__matsetv(__VA_ARGS__)

}
#endif //TINYMPC_MATLIB_RVVU_H
