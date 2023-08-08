from pathlib import Path

import pandas as pd
import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy

SPACE_GROUP_COUNTS = None


def _read_space_group_counts():
    global SPACE_GROUP_COUNTS
    if SPACE_GROUP_COUNTS is None:
        return pd.read_csv(Path(__file__).parent / "spacegroups_limat_counts.csv")
    return SPACE_GROUP_COUNTS


class SpaceGroup(Proxy):
    def __init__(self, normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        df = _read_space_group_counts()
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
