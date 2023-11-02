from typing import List, Union

import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy


class Uniform(Proxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch"]:
        return -1.0 * torch.ones(len(states), device=self.device, dtype=self.float)

    @property
    def min(self):
        return -1.0
