#include <cstdlib>
#include "../include/getstarted.h"

int main(int argc, char **argv) {
    getstartedHandle_t handle;
    long long A = 42;
    int * __restrict__ __return = (int*) calloc(1, sizeof(int));


    handle = __dace_init_getstarted(A);
    __program_getstarted(handle, __return, A);
    __dace_exit_getstarted(handle);

    free(__return);


    return 0;
}
