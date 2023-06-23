from abc import ABC
from typing import Optional

import numpy as np

from gflownet.proxy.base import Proxy


class MoleculeEnergyBase(Proxy, ABC):
    def __init__(
        self,
        batch_size: Optional[int] = 128,
        n_samples: int = 5000,
        **kwargs,
    ):
        """
        Parameters
        ----------

        batch_size : int
            Batch size for the underlying model.

        n_samples : int
            Number of samples that will be used to estimate minimum and maximum energy.
        """
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self.n_samples = n_samples
        self.max_energy = 0
        self.min = 0

    def setup(self, env=None):
        states = env.statebatch2proxy(2 * np.pi * np.random.rand(self.n_samples, 3))
        energies = self(states)

        self.max_energy = max(energies)
        self.min = min(energies) - self.max_energy
