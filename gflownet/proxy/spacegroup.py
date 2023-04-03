import pandas as pd
import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy


class SpaceGroup(Proxy):
    def __init__(self, data_path: str, normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        df = pd.read_csv(data_path)
        self.counts = torch.zeros(231, device=self.device, dtype=torch.int16)
        self.counts[df["Space Group"]] = torch.tensor(
            df.Counts, device=self.device, dtype=torch.int16
        )
        self.normalize = normalize
        if self.normalize:
            self.norm = -1.0 * torch.sum(self.counts)
        else:
            self.norm = -1

    def __call__(self, states: TensorType["batch", "1"]) -> TensorType["batch"]:
        return self.counts[torch.squeeze(states)] / self.norm
