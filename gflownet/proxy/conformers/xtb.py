from typing import List

import ray
import torch
from tblite.interface import Calculator, Structure
from torch import Tensor
from wurlitzer import pipes

from gflownet.proxy.base import Proxy


@ray.remote
def _get_energy(numbers, positions):
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
        self.min = -5
        self.max = 0
        self.conformer = None

    def setup(self, env=None):
        self.conformer = env.conformer

    def _sync_conformer_with_state(self, state: List):
        for idx, ta in enumerate(self.conformer.freely_rotatable_tas):
            self.conformer.set_torsion_angle(ta, state[idx])
        return self.conformer

    def __call__(self, states: List) -> Tensor:
        energies = []

        for batch in _chunks(states, self.batch_size):
            structures = []

            for state in batch:
                conf = self._sync_conformer_with_state(state)
                structures.append(
                    (conf.get_atomic_numbers(), conf.get_atom_positions())
                )

            tasks = [_get_energy.remote(s[0], s[1]) for s in structures]
            energies.extend(ray.get(tasks))

        return torch.tensor(energies, dtype=self.float, device=self.device)

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_obj = cls.__new__(cls)
        new_obj.__dict__.update(self.__dict__)
        return new_obj
