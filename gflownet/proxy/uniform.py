from typing import List, Union

import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy


class Uniform(Proxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._optimum = torch.tensor(1.0, device=self.device, dtype=self.float)

    def __call__(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch"]:
        return torch.ones(len(states), device=self.device, dtype=self.float)
    


class SpacegroupProxy(Proxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._optimum = torch.tensor(1.0, device=self.device, dtype=self.float)

    def __call__(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch"]:
        if states.shape[1] == 1: # spg env only
            return states.flatten().to(torch.float32)
        elif states.shape[1] == 96: # full crystal representation
            return states[:,-1].to(torch.float32)
        else:
            return states[:,-7].to(torch.float32)
        

class FakeCrystalProxy(Proxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._optimum = torch.tensor(1.0, device=self.device, dtype=self.float)

    def __call__(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch"]:
        spacegroup = [int(states[i][0].item()) for i in range(len(states))]
        res = torch.tensor([136 if x == 1 else 221 for x in spacegroup]).to(torch.float32)
        return res