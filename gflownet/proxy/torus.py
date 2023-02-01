from gflownet.proxy.base import Proxy
import torch
from torchtyping import TensorType


class Torus(Proxy):
    def __init__(self, normalize, alpha=1.0, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.alpha = alpha
        self.beta = beta

    def set_n_dim(self, n_dim):
        self.n_dim = n_dim

    @property
    def min(self):
        if self.normalize:
            return -1.0
        else:
            return -((self.n_dim * 2) ** 3)

    @property
    def norm(self):
        if self.normalize:
            return -((self.n_dim * 2) ** 3)
        else:
            return -1.0

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        """
        args:
            states: tensor
        returns:
            list of scores
        technically an oracle, hence used variable name energies
        """

        def _func_sin_cos_cube(x):
            return (1.0 / self.norm) * (
                torch.sum(torch.sin(self.alpha * x[:, 0::2]), axis=1)
                + torch.sum(torch.cos(self.beta * x[:, 1::2]), axis=1)
                + x.shape[1]
            ) ** 3

        return _func_sin_cos_cube(states)
