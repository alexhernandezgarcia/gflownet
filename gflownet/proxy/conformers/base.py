from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from torch import Tensor

from gflownet.proxy.base import Proxy


class MoleculeEnergyBase(Proxy, ABC):
    def __init__(
        self,
        batch_size: Optional[int] = 128,
        n_samples: int = 5000,
        normalize: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------

        batch_size : int
            Batch size for the underlying model.

        n_samples : int
            Number of samples that will be used to estimate minimum and maximum energy.

        normalize : bool
            Whether to truncate the energies to a (0, 1) range (estimated based on
            sample conformers).
        """
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self.n_samples = n_samples
        self.normalize = normalize
        self.max_energy = None
        self.min_energy = None
        self.min = None

    @abstractmethod
    def compute_energy(self, states: List) -> Tensor:
        pass

    def __call__(self, states: List) -> Tensor:
        energies = self.compute_energy(states)
        energies = energies - self.max_energy

        if self.normalize:
            energies = energies / (self.max_energy - self.min_energy)

        return energies

    def setup(self, env=None):
        states = env.statebatch2proxy(2 * np.pi * np.random.rand(self.n_samples, 3))
        energies = self.compute_energy(states)

        self.max_energy = max(energies)
        self.min_energy = min(energies)

        if self.normalize:
            self.min = -1
        else:
            self.min = self.min_energy - self.max_energy
