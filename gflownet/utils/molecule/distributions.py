import numpy as np
import torch
from pyro.distributions import ProjectedNormal
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily


def get_mixture_of_projected_normals(weights, concentrations):
    """
    :param weights: torch.tensor of shape [*batch_shape, n_components]
    :param concentrations: torch.tensor of shape [*batch_shape, n_components, n_dim]
    """
    mix = Categorical(weights)
    comp = ProjectedNormal(concentrations)
    return MixtureSameFamily(mix, comp)


class WrappedNormal:
    """
    a class representing the wrapped normal distribution, see equation (3) in https://arxiv.org/pdf/2206.01729
    """

    def __init__(self, means, stds, N=10):
        self.means = means  # shape : batch_size, n_dims
        self.stds = stds  # shape : batch_size, n_dims
        self.N = N  # number of times to wrap around the circle

    def prob(self, actions):
        p_ = 0
        for i in range(-self.N, self.N + 1):
            p_ += torch.exp(
                -((actions - self.means + 2 * torch.pi * i) ** 2)
                / 2
                / (self.stds.unsqueeze(1).repeat(1, 2)) ** 2
            )
        return p_

    def log_prob(self, actions):
        return torch.log(self.prob(actions))

    def sample(self, n_samples=None):
        n_dims = self.means.shape[-1]
        unwrapped_gaussian = Normal(
            self.means, self.stds.unsqueeze(1).repeat(1, n_dims)
        )
        if n_samples is None:
            sample = unwrapped_gaussian.sample() % (2 * torch.pi)
        else:
            sample = unwrapped_gaussian.sample(n_samples) % (2 * torch.pi)
        return sample
