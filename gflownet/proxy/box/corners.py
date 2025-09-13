"""
Corners objective function, defined for box-like environments, such as the hyper-grid
and the cube.

The function places high scores in all corners of the hyper-box via a mixture of
Gaussians. Optionally, the scores and can be thresholded in order to make the task
harder.
"""

import numpy as np
import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy


class Corners(Proxy):
    """
    It is assumed that the state values will be in the range [-1.0, 1.0].
    """

    def __init__(
        self,
        n_dim=None,
        mu=None,
        sigma=None,
        do_threshold=False,
        thresholds=((0.0, 2.0, 1e-6), (2.0, 3.0, 2.0), (3.0, 100, 10.0)),
        **kwargs,
    ):
        """
        Initializes an instance of the Corners proxy.

        Parameters
        ----------
        n_dim : int
            Dimensionality of the hyper-box.
        mu : float
            Mean of the Gaussian distributions that make the objective function. It
            should be a value between 0.0 and 1.0. A value closer to 1.0 places the
            regions of high reward closer to the edges (in the corners) and a value
            closer to 0.0 places the regions of high reward closer to the center.
        sigma : float
            Standard deviation of the Gaussian distributions that make the objective
            function.
        do_threshold : bool
            If True, the values of the Gaussians are thresholded using the values in
            ``thresholds``.
        thresholds : list
            A list of tuples with the information to threshold the objective function.
            The first two values of the tuple indicate the lower and upper bound of a
            range, and the third value indicates the value onto which values in the
            range are mapped. For example, (0.0, 1.0, 1e-6) will map all values between
            0.0 and 1.0 to 1e-6.
        """
        super().__init__(**kwargs)
        self.n_dim = n_dim
        self.mu = mu
        self.sigma = sigma
        self.do_threshold = do_threshold
        self.thresholds = (tuple(el) for el in thresholds)

    def setup(self, env=None):
        if env:
            self.n_dim = env.n_dim
        if self.sigma and self.mu and self.n_dim:
            self.mu_vec = self.mu * torch.ones(
                self.n_dim, device=self.device, dtype=self.float
            )
            cov = self.sigma * torch.eye(
                self.n_dim, device=self.device, dtype=self.float
            )
            cov_det = torch.linalg.det(cov)
            self.cov_inv = torch.linalg.inv(cov)
            self.mulnormal_norm = 1.0 / ((2 * torch.pi) ** self.n_dim * cov_det) ** 0.5

    @property
    def optimum(self):
        if not hasattr(self, "_optimum"):
            mode = self.mu * torch.ones(
                self.n_dim, device=self.device, dtype=self.float
            )
            self._optimum = self(torch.unsqueeze(mode, 0))[0]
        return self._optimum

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        scores = self.mulnormal_norm * torch.exp(
            -0.5
            * (
                torch.diag(
                    torch.tensordot(
                        torch.tensordot(
                            (torch.abs(states) - self.mu_vec), self.cov_inv, dims=1
                        ),
                        (torch.abs(states) - self.mu_vec).T,
                        dims=1,
                    )
                )
            )
        )
        if self.do_threshold:
            for th in self.thresholds:
                scores[(scores >= th[0]) & (scores < th[1])] = th[2]
        return scores
