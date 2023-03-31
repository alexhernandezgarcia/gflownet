from pyro.distributions import ProjectedNormal
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
