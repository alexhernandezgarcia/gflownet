from typing import List

import ray
import torch
from tblite.interface import Calculator, Structure
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


class XTBMoleculeEnergy(Proxy):
    def __init__(self, batch_size=100, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self.max_energy = 0
        self.min = -5

    def __call__(self, states: List) -> Tensor:
        energies = []

        for batch in _chunks(states, self.batch_size):
            tasks = [get_energy.remote(s[:, 0], s[:, 1:]) for s in batch]
            energies.extend(ray.get(tasks))

        energies = torch.tensor(energies, dtype=self.float, device=self.device)
        energies -= self.max_energy

        return energies

    def setup(self, env=None):
        self.max_energy = env.max_energy
