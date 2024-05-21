import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy


class Torus(Proxy):
    def __init__(self, normalize, alpha=1.0, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize
        self.alpha = alpha
        self.beta = beta

    def setup(self, env=None):
        if env:
            self.n_dim = env.n_dim

    @property
    def optimum(self):
        if not hasattr(self, "_optimum"):
            if self.normalize:
                self._optimum = torch.tensor(1.0, device=self.device, dtype=self.float)
            else:
                self._optimum = torch.tensor(
                    ((self.n_dim * 2) ** 3), device=self.device, dtype=self.float
                )
        return self._optimum

    @property
    def norm(self):
        if not hasattr(self, "_norm"):
            if self.normalize:
                self._norm = torch.tensor(
                    ((self.n_dim * 2) ** 3), device=self.device, dtype=self.float
                )
            else:
                self._norm = torch.tensor(1.0, device=self.device, dtype=self.float)
        return self._norm

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
