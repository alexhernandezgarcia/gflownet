import torch
import numpy as np

from sklearn.neighbors import KernelDensity


def fit_kde(samples, kernel="gaussian", bandwidth=0.1):
    """
    :param samples: numpy array of shape [batch_size, n_dim]
    """
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(samples)
    return kde
