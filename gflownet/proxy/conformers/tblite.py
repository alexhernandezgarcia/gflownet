from typing import List

import numpy as np
import ray
import torch
from tblite.interface import Calculator
from torch import Tensor
from wurlitzer import pipes

from gflownet.proxy.base import Proxy


@ray.remote
def get_energy(numbers, positions):
    with pipes():
        calc = Calculator("GFN2-xTB", numbers, positions * 1.8897259886)
        res = calc.singlepoint()
        energy = res.get("energy").item()

    return energy


def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class TBLiteMoleculeEnergy(Proxy):
    def __init__(self, batch_size=1024, n_samples=5000, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self.n_samples = n_samples
        self.max_energy = 0
        self.min = 0

    def __call__(self, states: List) -> Tensor:
        energies = []

        for batch in _chunks(states, self.batch_size):
            tasks = [get_energy.remote(s[:, 0], s[:, 1:]) for s in batch]
            energies.extend(ray.get(tasks))

        energies = torch.tensor(energies, dtype=self.float, device=self.device)
        energies -= self.max_energy

        return energies

    def setup(self, env=None):
        states = env.statebatch2proxy(2 * np.pi * np.random.rand(self.n_samples, 3))
        energies = self(states)

        self.max_energy = max(energies)
        self.min = min(energies) - self.max_energy
