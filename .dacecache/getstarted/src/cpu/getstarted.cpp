/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>
#include "../../include/hash.h"

struct getstarted_t {

};

void __program_getstarted_internal(getstarted_t *__state, int * __restrict__ __return, long long A)
{

    {
        int __tmp0;


        {
            long long __in1 = A;
            long long __in2 = A;
            int __out;

            ///////////////////
            // Tasklet code (_Add_)
            __out = (dace::int32(__in1) + dace::int32(__in2));
            ///////////////////

            __tmp0 = __out;
        }
        {
            int __inp = __tmp0;
            int __out;

            ///////////////////
            // Tasklet code (assign_3_4)
            __out = __inp;
            ///////////////////

            __return[0] = __out;
        }

    }
}

DACE_EXPORTED void __program_getstarted(getstarted_t *__state, int * __restrict__ __return, long long A)
{
    __program_getstarted_internal(__state, __return, A);
}

DACE_EXPORTED getstarted_t *__dace_init_getstarted(long long A)
{
    int __result = 0;
    getstarted_t *__state = new getstarted_t;



    if (__result) {
        delete __state;
        return nullptr;
    }
    return __state;
}

DACE_EXPORTED void __dace_exit_getstarted(getstarted_t *__state)
{
    delete __state;
}

