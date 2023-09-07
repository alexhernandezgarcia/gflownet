import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy


class Uniform(Proxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        return -1.0 * torch.ones(states.shape[0]).to(states)

    @property
    def min(self):
        if not hasattr(self, "_min"):
            self._min = torch.tensor(-1.0, device=self.device, dtype=self.float)
        return self._min
