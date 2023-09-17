import pickle
from pathlib import Path
from typing import List

import numpy as np
from torch import Tensor

from gflownet.proxy.base import Proxy


PICKLE_PATH = Path(__file__).parents[3] / "data" / "crystals" / "lattice_proxy.pickle"


class LatticeParameters(Proxy):
    def __init__(self, min_value: float = -100, **kwargs):
        super().__init__(**kwargs)

        self.min_value = min_value
        self.kde = None

    def __call__(self, states: List) -> Tensor:
        scores = self.kde.score_samples(states)
        scores = np.clip(scores, self.min_value, np.inf)
        scores = self.min_value - scores

        return Tensor(scores)

    def setup(self, env=None):
        if not PICKLE_PATH.exists():
            raise ValueError(f"Couldn't find a fitted KDE model under {PICKLE_PATH}.")

        with open(PICKLE_PATH, "rb") as f:
            self.kde = pickle.load(f)
