typedef void * getstartedHandle_t;
extern "C" getstartedHandle_t __dace_init_getstarted(long long A);
extern "C" void __dace_exit_getstarted(getstartedHandle_t handle);
extern "C" void __program_getstarted(getstartedHandle_t handle, int * __restrict__ __return, long long A);
