from gflownet.proxy.base import Proxy
import numpy as np
import torch
from torchtyping import TensorType


class Corners(Proxy):
    """
    It is assumed that the state values will be in the range [-1.0, 1.0].
    """

    def __init__(self, n_dim=None, mu=None, sigma=None, **kwargs):
        super().__init__(**kwargs)
        self.n_dim = n_dim
        self.mu = mu
        self.sigma = sigma

    def setup(self):
        if self.sigma and self.mu and self.n_dim:
            self.mu_vec = self.mu * torch.ones(
                self.n_dim, device=self.device, dtype=self.float
            )
            cov = self.sigma * torch.eye(
                self.n_dim, device=self.device, dtype=self.float
            )
            cov_det = torch.linalg.det(cov)
            self.cov_inv = torch.linalg.inv(cov)
            self.mulnormal_norm = 1.0 / ((2 * torch.pi) ** 2 * cov_det) ** 0.5

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        return (
            -1.0
            * self.mulnormal_norm
            * torch.exp(
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
        )
