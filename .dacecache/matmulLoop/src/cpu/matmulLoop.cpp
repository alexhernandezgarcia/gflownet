/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct matmulLoop_t {

};

inline void _MatMult_gemm_sdfg_0_1_5(matmulLoop_t *__state, double* _a, double* _b, double* _c, int L) {

    {


        {
            #pragma omp parallel for
            for (auto _o0 = 0; _o0 < L; _o0 += 1) {
                for (auto _o1 = 0; _o1 < L; _o1 += 1) {
                    {
                        double out;

                        ///////////////////
                        // Tasklet code (gemm_init)
                        out = 0;
                        ///////////////////

                        _c[((L * _o0) + _o1)] = out;
                    }
                }
            }
        }

    }
    {


        {
            #pragma omp parallel for
            for (auto __i0 = 0; __i0 < L; __i0 += 1) {
                for (auto __i1 = 0; __i1 < L; __i1 += 1) {
                    for (auto __i2 = 0; __i2 < L; __i2 += 1) {
                        {
                            double __a = _a[((L * __i0) + __i2)];
                            double __b = _b[((L * __i2) + __i1)];
                            double __out;

                            ///////////////////
                            // Tasklet code (gemm)
                            __out = (__a * __b);
                            ///////////////////

                            dace::wcr_fixed<dace::ReductionType::Sum, double>::reduce_atomic(_c + ((L * __i0) + __i1), __out);
                        }
                    }
                }
            }
        }

    }
    
}

void __program_matmulLoop_internal(matmulLoop_t *__state, double * __restrict__ M1, double * __restrict__ M2, double * __restrict__ __return, int L, int N)
{
    long long n;

    for (n = 0; (n < N); n = (n + 1)) {
        {
            double* __tmp1 = &M1[((L * L) * n)];
            double* __tmp2 = &M2[((L * L) * n)];


            _MatMult_gemm_sdfg_0_1_5(__state, &__tmp1[0], &__tmp2[0], &__return[((L * L) * n)], L);

        }

    }
}

DACE_EXPORTED void __program_matmulLoop(matmulLoop_t *__state, double * __restrict__ M1, double * __restrict__ M2, double * __restrict__ __return, int L, int N)
{
    __program_matmulLoop_internal(__state, M1, M2, __return, L, N);
}

DACE_EXPORTED matmulLoop_t *__dace_init_matmulLoop(int L, int N)
{
    int __result = 0;
    matmulLoop_t *__state = new matmulLoop_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_matmulLoop(matmulLoop_t *__state)
{
    delete __state;
}

