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
        self.mulnormal = self.setup()

    def setup(self):
        if self.sigma and self.mu and self.n_dim:
            self.mu_vec = self.mu * torch.ones(self.n_dim, device=self.device, dtype=self.float)
            cov = self.sigma * torch.eye(self.n_dim, device=self.device, dtype=self.float)
            cov_det = torch.linalg.det(cov)
            self.cov_inv = torch.linalg.inv(cov)
            self.mulnormal_norm = 1.0 / ((2 * torch.pi) ** 2 * cov_det) ** 0.5
#             self.mulnormal = True
            self.mulnormal = False
        else:
            self.mulnormal = False
        return self.mulnormal

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        """
        args:
            states: ndarray

        returns:
            list of scores
        technically an oracle, hence used variable name energies
        """

        def _func_corners(x):
            ax = abs(x)
            energies = -1.0 * (
                (ax > 0.5).prod(-1) * 0.5
                + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2
                + 1e-1
            )
            return energies

        def _mulnormal_corners(x):
            import ipdb; ipdb.set_trace()
            return (
                -1.0
                * self.mulnormal_norm
                * torch.exp(
                    -0.5
                    * (
                        torch.diag(
                            torch.dot(
                                torch.dot((torch.abs(x) - self.mu_vec), self.cov_inv),
                                (torch.abs(x) - self.mu_vec).T,
                            )
                        )
                    )
                )
            )

        if self.mulnormal:
            import ipdb; ipdb.set_trace()
            return _mulnormal_corners(states)
        else:
            snp = states.cpu().numpy()
            pnp =  np.asarray([_func_corners(state) for state in snp])
            import ipdb; ipdb.set_trace()
            return np.asarray([_func_corners(state) for state in states])
