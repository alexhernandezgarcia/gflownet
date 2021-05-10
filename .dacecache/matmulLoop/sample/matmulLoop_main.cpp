#include <cstdlib>
#include "../include/matmulLoop.h"

int main(int argc, char **argv) {
    matmulLoopHandle_t handle;
    int L = 42;
    int N = 42;
    double * __restrict__ M1 = (double*) calloc(((L * L) * N), sizeof(double));
    double * __restrict__ M2 = (double*) calloc(((L * L) * N), sizeof(double));
    double * __restrict__ __return = (double*) calloc(((L * L) * N), sizeof(double));


    handle = __dace_init_matmulLoop(L, N);
    __program_matmulLoop(handle, M1, M2, __return, L, N);
    __dace_exit_matmulLoop(handle);

    free(M1);
    free(M2);
    free(__return);


    return 0;
}
