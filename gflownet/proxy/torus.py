from gflownet.proxy.base import Proxy
import torch
from torchtyping import TensorType


class Torus(Proxy):
    def __init__(self, normalize):
        super().__init__()
        self.normalize = normalize

    @property
    def min(self):
        if self.normalize:
            return 1
        else:
            return -(self.n_dim * 2) ** 3

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        """
        args:
            states: tensor
        returns:
            list of scores
        technically an oracle, hence used variable name energies
        """

        def _func_sin_cos_cube(x):
            return (
                -1
                * (
                    torch.sum(torch.sin(x[:, 0::2]), axis=1)
                    + torch.sum(torch.cos(x[:, 1::2]), axis=1)
                    + x.shape[1]
                )
                ** 3
            )

        def _func_sin_cos_cube_norm(x):
            norm = (x.shape[1] * 2) ** 3
            return (-1.0 / norm) * (
                torch.sum(torch.sin(x[:, 0::2]), axis=1)
                + torch.sum(torch.cos(x[:, 1::2]), axis=1)
                + x.shape[1]
            ) ** 3

        if self.normalize:
            return _func_sin_cos_cube_norm(states)
        else:
            return _func_sin_cos_cube(states)
