import numpy as np
import torch
from sklearn.neighbors import KernelDensity


def fit_kde(samples, kernel="gaussian", bandwidth=0.1):
    """
    :param samples: numpy array of shape [batch_size, n_dim]
    """
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(samples)
    return kde


def angles_allclose_np(one, other, atol=1e-5):
    """
    Check if two sets of angles are close to each other.

    Parameters
    ----------
    one: np.ndarray
        The first set of angles.
    other: np.ndarray
        The second set of angles.
    atol: float
        The absolute tolerance.

    Returns
    -------
    bool
        True if the angles are close to each other, False otherwise.
    """
    diff = (one - other) % (2 * np.pi)
    return np.logical_or(
        np.isclose(diff, np.zeros_like(diff), atol=atol),
        np.isclose(diff, np.ones_like(diff) * 2 * np.pi, atol=atol),
    ).all(axis=-1)


def angles_allclose(one, other, atol=1e-5):
    """
    Check if two sets of angles are close to each other.

    Parameters
    ----------
    one: Tensor, list, np.array
        The first set of angles.
    other: Tensor, list, np.array
        The second set of angles.
    atol: float
        The absolute tolerance.

    Returns
    --------
    bool:
        True if the angles are close to each other, False otherwise.
    """
    if isinstance(one, np.ndarray):
        return angles_allclose_np(one, other, atol)
    if isinstance(one, list):
        result = angles_allclose_np(np.array(one), np.array(other), atol=atol)
        return result.tolist()
    diff = (one - other) % (2 * torch.pi)
    return torch.logical_or(
        torch.isclose(diff, torch.zeros_like(diff), atol=atol),
        torch.isclose(diff, torch.ones_like(diff) * 2 * torch.pi, atol=atol),
    ).all(dim=-1)
