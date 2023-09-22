# This needs to be imported first due to conda/pip package conflicts.
from tblite.interface import Calculator  # isort: skip

import os
from typing import List

import torch
from joblib import Parallel, delayed
from torch import Tensor
from wurlitzer import pipes

from gflownet.proxy.conformers.base import MoleculeEnergyBase


def get_energy(numbers, positions):
    with pipes():
        try:
            # The positions are converted from Angstrom to Bohr.
            calc = Calculator("GFN2-xTB", numbers, positions * 1.8897259886)
            res = calc.singlepoint()
            energy = res.get("energy").item()
        except RuntimeError:
            energy = 0.0

    return energy


def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class TBLiteMoleculeEnergy(MoleculeEnergyBase):
    def __init__(
        self,
        batch_size: int = 1024,
        n_samples: int = 10000,
        normalize: bool = True,
        **kwargs
    ):
        super().__init__(
            batch_size=batch_size, n_samples=n_samples, normalize=normalize, **kwargs
        )

    def compute_energy(self, states: List) -> Tensor:
        # Get the number of available CPUs.
        n_jobs = len(os.sched_getaffinity(0))

        energies = []

        for batch in _chunks(states, self.batch_size):
            energies.extend(
                Parallel(n_jobs=n_jobs)(
                    delayed(get_energy)(s[:, 0], s[:, 1:]) for s in batch
                )
            )

        energies = torch.tensor(energies, dtype=self.float, device=self.device)

        return energies
