"""
Spheres objective function, defined for box-like environments, such as the hyper-grid
and the cube.

The function places high scores at spheres or rings in the hyper-box centered around
the source state. The spheres are defined by a mixture of isotropic radial Gaussians.
Optionally, the scores and can be thresholded in order to make the task harder.
"""

from typing import Iterable

import numpy as np
import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat


class Spheres(Proxy):
    """
    It is assumed that the state values will be in the range [-1.0, 1.0].
    """

    def __init__(
        self,
        n_dim: int = None,
        radii: Iterable[float] = (0.2, 0.8),
        sigma: float = None,
        **kwargs,
    ):
        """
        Initializes an instance of the Spheres proxy.

        Parameters
        ----------
        n_dim : int
            Dimensionality of the hyper-box.
        radii : Iterable[float]
            A sequence of radii to define the objective function. Each value indicates
            the relative distance to the source state, with 0.0 being the source state
            and 1.0 being the farthest corner. The radii are the same for all
            dimensions, defining spheres around the source rate.
        sigma : float
            Standard deviation of the Gaussian distributions that make the objective
            function.
        """
        super().__init__(**kwargs)
        self.n_dim = n_dim
        self.radii = tuple(radii)
        self.n_spheres = len(self.radii)
        self.sigma = sigma

    def setup(self, env=None):
        if env:
            self.n_dim = env.n_dim
        if self.sigma and self.radii and self.n_dim:
            # We multiply by 2 because the states are defined the range [-1, 1]
            self.radii_vec = 2 * tfloat(
                self.radii, device=self.device, float_type=self.float
            ).view(1, -1)

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        # We add 1 to re-center the states at [0.0, ..., 0.0]
        distances = (
            torch.linalg.norm(states + 1.0, dim=1, keepdim=True) - self.radii_vec
        )
        return torch.sum(torch.exp(-0.5 * (distances / self.sigma) ** 2), dim=1)
