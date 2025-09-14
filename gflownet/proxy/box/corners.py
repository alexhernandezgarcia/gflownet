"""
Corners objective function, defined for box-like environments, such as the hyper-grid
and the cube.

The function places high scores in all corners of the hyper-box according to either of
these options:
     - A mixture of Gaussians. Optionally, the scores and can be thresholded in order
       to make the task harder.
     - A hard-coded function that assigns each point of the hyper-box to one of three
       scores (this matches the function used in the original GFlowNets paper):
        - High scores: points in the corners at a distance from the edge between 0.2
          and 0.1 the length of the box.
        - Middle scores: points in the corners at distance from the edge within a
          quarter the length of the box.
        - Low scores: all other points.
"""

from typing import Iterable

import numpy as np
import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat


class Corners(Proxy):
    """
    It is assumed that the state values will be in the range [-1.0, 1.0].
    """

    def __init__(
        self,
        n_dim=None,
        mu=None,
        sigma=None,
        do_gaussians=True,
        do_threshold=False,
        thresholds: Iterable = ((0.0, 2.0, 1e-6), (2.0, 3.0, 2.0), (3.0, 100, 10.0)),
        scores: Iterable = [2.0, 0.5, 0.01],
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
            regions of high scores closer to the edges (in the corners) and a value
            closer to 0.0 places the regions of high scores closer to the center.
        sigma : float
            Standard deviation of the Gaussian distributions that make the objective
            function.
        do_gaussians : bool
            If True, the score landscape is modelled by a mixture of Gaussians.
            Otherwise, the scores are assigned via an indicator function, as in the
            original GFlowNets paper.
        do_threshold : bool
            If True, the values of the Gaussians are thresholded using the values in
            ``thresholds``.
        thresholds : iterable
            A list of tuples with the information to threshold the objective function.
            The first two values of the tuple indicate the lower and upper bound of a
            range, and the third value indicates the value onto which values in the
            range are mapped. For example, (0.0, 1.0, 1e-6) will map all values between
            0.0 and 1.0 to 1e-6.
        scores : iterable
            The scores of the three regions for the non-Gaussian version. It should be
            a list with three elements, where the first element is highest score, the
            second element the middle score and the third element the lowest score.
        """
        super().__init__(**kwargs)
        self.n_dim = n_dim
        self.mu = mu
        self.sigma = sigma
        self.do_gaussians = do_gaussians
        self.do_threshold = do_threshold
        self.thresholds = tuple(tuple(el) for el in thresholds)
        self.scores = tuple(scores)

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
            if self.do_gaussians:
                mode = self.mu * torch.ones(
                    self.n_dim, device=self.device, dtype=self.float
                )
                self._optimum = self(torch.unsqueeze(mode, 0))[0]
            else:
                self._optimum = sum(self.scores)
        return self._optimum

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        if self.do_gaussians:
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
        else:
            states_abs = torch.abs(states)
            mid_scores = self.scores[1] * tfloat(
                states_abs >= 0.5, float_type=self.float, device=self.device
            ).prod(dim=1)
            high_scores = self.scores[0] * tfloat(
                (states_abs >= 0.6) & (states_abs < 0.8),
                float_type=self.float,
                device=self.device,
            ).prod(dim=1)
            scores = self.scores[2] + mid_scores + high_scores
        return scores
