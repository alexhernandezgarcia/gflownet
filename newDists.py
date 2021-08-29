import numpy as np
import matplotlib.pyplot as plt
import time
import tqdm


def binaryDistance(samples, pairwise = False, extractInds = None):
    '''
    compute simple sum of distances between sample vectors
    :param samples:
    :return:
    '''
    # determine if all samples have equal length
    lens = np.array([i.shape[-1] for i in samples])
    if len(np.unique(lens)) > 1: # if there are multiple lengths, we need to pad up to a constant length
        raise ValueError('Attempted to compute binary distances between samples with different lengths!')

    if extractInds is not None:
        nOutputs = extractInds
    else:
        nOutputs = len(samples)

    if pairwise: # compute every pairwise distances
        distances = np.zeros((nOutputs, nOutputs))
        for i in range(nOutputs):
            distances[i, :] = np.sum(samples[i] != samples, axis = 1) / len(samples[i])
    else: # compute average distance of each sample from all the others
        distances = np.zeros(nOutputs)
        for i in range(nOutputs):
            distances[i] = np.sum(samples[i] != samples) / len(samples.flatten())

    return distances


def oneHotDistance(samples, pairwise = False, extractInds = None):
    '''
    find the minimum single mutation distance (normalized) between sequences
    optionally explicitly extract only  the first extractInds sequences distances, with respect to themselves and all others
    :param samples:
    :param pairwise:
    :param extractInds:
    :return:
    '''
    # do one-hot encoding
    oneHot = np_oneHot(samples, len(np.unique(samples)))
    oneHot = oneHot.reshape(oneHot.shape[0], int(oneHot.shape[1]*oneHot.shape[2]))
    target = oneHot[:extractInds] # limit the number of samples we are actually interested in
    if target.ndim == 1:
        target = np.expand_dims(target,0)

    dists = 1 - target @ oneHot.transpose() / samples.shape[1]
    if pairwise:
        return dists
    else:
        return np.average(dists,axis=1)


def np_oneHot(samples, uniques):
    flatsamples = samples.flatten()
    shape = (flatsamples.size, uniques)
    one_hot = np.zeros(shape)
    rows = np.arange(flatsamples.size)
    one_hot[rows, flatsamples] = 1
    return one_hot.reshape(samples.shape[0], samples.shape[1], uniques)


samples = np.random.randint(0,5,size=(10000, 40))

t0 = time.time()
d1 = binaryDistance(samples, extractInds=1000)
tf = time.time()
print('Original took {:.3f} seconds'.format(int(tf-t0)))

t0 = time.time()
d2 = oneHotDistance(samples, extractInds=1000)
tf = time.time()
print('Original took {:.3f} seconds'.format(tf-t0))

xx = np.logspace(0, 5, 10)
yy = np.logspace(0, 4, 10)
t1 = np.zeros((len(xx), len(xx)))
t2 = np.zeros_like(t1)

for i in tqdm.tqdm(range(len(xx))):
    for j in range(len(yy)):
        if yy[j] < xx[i]:
            samples = np.random.randint(0, 5, size=(int(xx[i]), 40))

            t0 = time.time()
            d1 = binaryDistance(samples, extractInds=int(yy[j]))
            tf = time.time()
            t1[i, j] = tf - t0

            t0 = time.time()
            d2 = oneHotDistance(samples, extractInds=int(yy[j]))
            tf = time.time()
            t2[i, j] = tf - t0

