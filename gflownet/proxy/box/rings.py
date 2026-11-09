"""
Rings objective function, defined for box-like environments, such as the hyper-grid
and the cube.

The function places high scores at rings (or hyper-sphere surfaces) in the hyper-box,
centered around a point defined by the ``center`` parameter, which defaults to the
domain center. The rings are defined by a mixture of isotropic radial Gaussians.
Optionally, the scores and can be thresholded.
"""

from typing import Iterable

import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat


class Rings(Proxy):
    """
    It is assumed that the state values will be in the range [-1.0, 1.0].
    """

    def __init__(
        self,
        n_dim: int = None,
        radii: Iterable[float] = (0.2, 0.8),
        sigma: float = 0.05,
        center: Iterable[float] = None,
        **kwargs,
    ):
        """
        Initializes an instance of the Rings proxy.

        Parameters
        ----------
        n_dim : int
            Dimensionality of the hyper-box.
        radii : Iterable[float]
            A sequence of radii to define the objective function. Each value indicates
            the relative distance to the center of the rings, with 0.0 being the center
            itself and 1.0 corresponding to a ring whose diameter is equal to the side
            length of the hyper-box, that is, an absolute distance of 1.0. Larger radii
            are allowed; a ring intersects the box as long as its radius is smaller
            than the distance between the center and the farthest corner. The radii
            are the same for all dimensions.
        sigma : float
            Standard deviation of the Gaussian distributions that make the objective
            function. Default: 0.05.
        center : Iterable[float]
            A tuple of as many scalars as dimensions, each between -1.0 and 1.0,
            indicating the center of the rings. 0.0 indicates the center of the
            hyper-box domain and -1.0 and 1.0 are the extremes. If None (default), the
            center is 0.0 for all dimensions, that is the center of the domain.
        """
        super().__init__(**kwargs)
        self.n_dim = n_dim
        self.radii = tuple(radii)
        self.n_rings = len(self.radii)
        self.sigma = sigma
        self.center = tuple(center) if center is not None else None

    def setup(self, env=None):
        if env:
            self.n_dim = env.n_dim
        if self.center is None:
            self.center = (0.0,) * self.n_dim
        if len(self.center) != self.n_dim:
            raise ValueError(
                f"center must have as many values as dimensions ({self.n_dim}), "
                f"but it has {len(self.center)}."
            )
        if any(c < -1.0 or c > 1.0 for c in self.center):
            raise ValueError("All values of center must be between -1.0 and 1.0.")
        if self.radii and self.n_dim:
            if self.sigma is None:
                raise ValueError("sigma must be set for the Rings proxy.")
            self.center_vec = tfloat(
                self.center, device=self.device, float_type=self.float
            )
            self.radii_vec = tfloat(
                self.radii, device=self.device, float_type=self.float
            ).view(1, -1)

    def exponential_radial_mixture(
        self, distances: TensorType["batch"]
    ) -> TensorType["batch"]:
        r"""
        Exponential radial mixture over a set of radii given distances to the center.

        $$y = \sum_i \exp(-\frac{(d-r_i)^2}{2\sigma^2})$$

        Parameters
        ----------
        distances : TensorType["batch"]
            Euclidean distance of each state in a batch to the center of the rings.

        Returns
        -------
        TensorType["batch"]
            Score of each state, that is, the sum over rings of isotropic radial
            Gaussians evaluated at the distance of each state.
        """
        return torch.sum(
            torch.exp(
                -0.5 * ((distances.unsqueeze(1) - self.radii_vec) / self.sigma) ** 2
            ),
            dim=1,
        )

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        r"""
        Returns the proxy values of a batch of states.

        $$\sum_i \exp(-\frac{(||x-c||_2-r_i)^2}{2\sigma^2})$$

        with $c$ the center of the rings, and $r_i$ the radius of each ring.

        Parameters
        ----------
        states : TensorType["batch", "state_dim"]
            Batch of states in proxy format.

        Returns
        -------
        TensorType["batch"]
            Proxy value of each state, that is, the sum over rings of isotropic radial
            Gaussians evaluated at the distance of each state to the center of the
            rings.
        """
        distances = torch.linalg.norm(states - self.center_vec, dim=1)
        return self.exponential_radial_mixture(distances)

    @property
    def optimum(self):
        """
        Maximum value of the objective function over the hyper-box.

        The score depends only on the distance to the center of the rings, so the
        maximum over the hyper-box is found by evaluating the radial scores over every
        distance between 0.0 and the distance between the center and the farthest
        corner of the hyper-box.
        """
        if not hasattr(self, "_optimum"):
            d_max = torch.linalg.norm(1.0 + self.center_vec.abs(), dim=0)
            distances = torch.linspace(
                0.0, d_max, 10001, device=self.device, dtype=self.float
            )
            self._optimum = self.exponential_radial_mixture(distances).max()
        return self._optimum
