typedef void * matmulLoopHandle_t;
extern "C" matmulLoopHandle_t __dace_init_matmulLoop(int L, int N);
extern "C" void __dace_exit_matmulLoop(matmulLoopHandle_t handle);
extern "C" void __program_matmulLoop(matmulLoopHandle_t handle, double * __restrict__ M1, double * __restrict__ M2, double * __restrict__ __return, int L, int N);
