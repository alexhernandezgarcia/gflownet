import numpy as np
import dace
from dace.transformation.dataflow import GPUTransformMap
import time

N, L = (dace.symbol(name) for name in ['N', 'L'])

@dace.program
def matmulLoop(M1: dace.float64[N,L,L], M2: dace.float64[N,L,L]):
    Mout = np.ndarray(shape=(N,L,L),dtype=np.float64)
    for n in range(N):
        Mout[n] = M1[n] @ M2[n]

    return Mout

func = matmulLoop.compile()

N=100
L=100
tstart = time.time()
out = func(M1=np.random.randn(N,L,L),M2=np.random.randn(N,L,L),N=np.int32(N),L=np.int32(L))
tfinish = time.time()
deltat = tfinish-tstart

def matmulLoop2(M1, M2):
    Mout = np.zeros_like(M1)
    for n in range(len(M1)):
        Mout[n] = M1[n] @ M2[n]

    return Mout

tstart = time.time()
out = matmulLoop2(np.random.randn(N,L,L),np.random.randn(N,L,L))
tfinish = time.time()
deltat2 = tfinish-tstart

print(deltat)
print(deltat2)