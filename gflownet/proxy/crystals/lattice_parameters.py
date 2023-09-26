import pickle
from pathlib import Path
from typing import List

import numpy as np
from torch import Tensor

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

PICKLE_PATH = Path(__file__).parents[3] / "data" / "crystals" / "lattice_proxy.pkl"


class LatticeParameters(Proxy):
    def __init__(self, min_value: float = -100, **kwargs):
        super().__init__(**kwargs)

        self.min = tfloat(min_value, float_type=self.float, device=self.device)
        self.min_value = min_value
        self.kde = None

    def __call__(self, states: List) -> Tensor:
        scores = self.kde.score_samples(states)
        scores = np.clip(scores, self.min_value, np.inf)
        scores = self.min_value - scores

        return tfloat(scores, float_type=self.float, device=self.device)

    def setup(self, env=None):
        if not PICKLE_PATH.exists():
            raise ValueError(f"Couldn't find a fitted KDE model under {PICKLE_PATH}.")

        with open(PICKLE_PATH, "rb") as f:
            self.kde = pickle.load(f)
