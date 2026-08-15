import numpy as np

def hist_count(x, basis):
    """
    counts number of times each element in basis appears in x
    op is a vector of same size as basis
    assume no duplicates in basis
    """
    op = np.zeros((len(basis)), dtype=int)
    map_basis = {}
    for n, k in enumerate(basis):
        map_basis[k] = n
    for t in x:
        op[map_basis[t]] += 1
    return op

def logsumexp(x):
    tmp = x.copy()
    tmp_max = np.max(tmp)
    tmp -= tmp_max
    op = np.log(np.sum(np.exp(tmp))) + tmp_max
    return op


def softmax(x):
    tmp = x.copy()
    tmp_max = np.max(tmp)
    tmp -= float(tmp_max)
    tmp = np.exp(tmp)
    op = tmp / np.sum(tmp)
    return op


def assert_no_nan(mat, name='matrix'):
    try:
        assert(not any(np.isnan(mat)))
    except AssertionError:
        print('%s contains NaN' % name)
        print(mat)
        raise AssertionError

def check_if_one(val):
    try:
        assert(np.abs(val - 1) < 1e-12)
    except AssertionError:
        print('val = %s (needs to be equal to 1)' % val)
        raise AssertionError

def check_if_zero(val):
    try:
        assert(np.abs(val) < 1e-10)
    except AssertionError:
        print('val = %s (needs to be equal to 0)' % val)
        raise AssertionError


def sample_multinomial(prob):
    try:
        k = int(np.where(np.random.multinomial(1, prob, size=1)[0]==1)[0])
    except TypeError:
        print('problem in sample_multinomial: prob = ')
        print(prob)
        raise TypeError
    except:
        raise Exception
    return k


def sample_multinomial_scores(scores):
    scores_cumsum = np.cumsum(scores)
    s = scores_cumsum[-1] * np.random.rand(1)
    k = 0
    while s > scores_cumsum[k]:
        k += 1
    return k


def sample_polya(alpha_vec, n):
    """ alpha_vec is the parameter of the Dirichlet distribution, n is the #samples """
    prob = np.random.dirichlet(alpha_vec)
    n_vec = np.random.multinomial(n, prob)
    return n_vec


def get_kth_minimum(x, k=1):
    """ gets the k^th minimum element of the list x 
        (note: k=1 is the minimum, k=2 is 2nd minimum) ...
        based on the incomplete selection sort pseudocode """
    n = len(x)
    for i in range(n):
        minIndex = i
        minValue = x[i]
        for j in range(i+1, n):
            if x[j] < minValue:
                minIndex = j
                minValue = x[j]
        x[i], x[minIndex] = x[minIndex], x[i]
    return x[k-1]


class empty(object):
    def __init__(self):
        pass
