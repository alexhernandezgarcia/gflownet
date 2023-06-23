from typing import List

import ray
import torch
from tblite.interface import Calculator
from torch import Tensor
from wurlitzer import pipes

from gflownet.proxy.conformers.base import MoleculeEnergyBase


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


class TBLiteMoleculeEnergy(MoleculeEnergyBase):
    def __init__(self, batch_size: int = 1024, n_samples: int = 5000, **kwargs):
        super().__init__(batch_size=batch_size, n_samples=n_samples, **kwargs)

    def __call__(self, states: List) -> Tensor:
        energies = []

        for batch in _chunks(states, self.batch_size):
            tasks = [get_energy.remote(s[:, 0], s[:, 1:]) for s in batch]
            energies.extend(ray.get(tasks))

        energies = torch.tensor(energies, dtype=self.float, device=self.device)
        energies -= self.max_energy

        return energies
