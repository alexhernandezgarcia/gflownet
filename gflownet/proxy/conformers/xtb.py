# This is a hotfix for tblite (used for the conformer generation) not
# importing correctly unless it is being imported first.
try:
    from tblite import interface
except:
    pass

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np
import numpy.typing as npt
import ray
import torch
from torch import Tensor
from wurlitzer import pipes

from gflownet.proxy.base import Proxy
from gflownet.proxy.conformers.tblite import _chunks
from gflownet.utils.molecule.xtb import run_gfn_xtb


def _write_xyz_file(elements: npt.NDArray, coordinates: npt.NDArray, file_path: str) -> None:
    num_atoms = len(elements)
    with open(file_path, 'w') as f:
        f.write(str(num_atoms) + '\n')
        f.write('\n')

        for i in range(num_atoms):
            element = elements[i]
            x, y, z = coordinates[i]
            line = f"{int(element)} {x:.6f} {y:.6f} {z:.6f}\n"
            f.write(line)


@ray.remote
def get_energy(numbers, positions, method="gfnff"):
    directory = TemporaryDirectory()
    file_name = "input.xyz"

    _write_xyz_file(numbers, positions, str(Path(directory.name) / "input.xyz"))
    with pipes():
        energy = run_gfn_xtb(directory.name, file_name, gfn_version=method)
    directory.cleanup()

    if np.isnan(energy):
        return 0.0

    return energy


class XTBMoleculeEnergy(Proxy):
    def __init__(self, method: str = "gfnff", batch_size=1000, **kwargs):
        super().__init__(**kwargs)

        self.method = method
        self.batch_size = batch_size
        self.max_energy = 0
        self.min = 0

    def __call__(self, states: Iterable) -> Tensor:
        energies = []

        for batch in _chunks(states, self.batch_size):
            tasks = [get_energy.remote(s[:, 0], s[:, 1:], self.method) for s in batch]
            energies.extend(ray.get(tasks))

        energies = torch.tensor(energies, dtype=self.float, device=self.device)
        energies -= self.max_energy

        return energies

    def setup(self, env=None):
        self.max_energy = env.max_energy
        self.min = env.min_energy - env.max_energy
