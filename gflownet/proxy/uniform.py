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
        return -1.0
