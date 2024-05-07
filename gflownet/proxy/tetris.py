import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy


class Tetris(Proxy):
    def __init__(self, normalize, **kwargs):
        super().__init__(**kwargs)
        self.normalize = normalize

    def setup(self, env=None):
        if env:
            self.height = env.height
            self.width = env.width

    @property
    def norm(self):
        if self.normalize:
            return self.height * self.width
        else:
            return 1.0

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        if states.dim() == 2:
            states = torch.unsqueeze(states, dim=0)
        return torch.sum(states, axis=(1, 2)) / self.norm
