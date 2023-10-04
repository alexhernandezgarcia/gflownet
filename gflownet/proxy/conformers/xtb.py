# This is a hotfix for tblite (used for the conformer generation) not
# importing correctly unless it is being imported first.
try:
    from tblite import interface
except:
    pass

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import numpy as np
import numpy.typing as npt
import torch
from joblib import Parallel, delayed
from torch import Tensor
from wurlitzer import pipes

from gflownet.proxy.conformers.base import MoleculeEnergyBase
from gflownet.proxy.conformers.tblite import _chunks
from gflownet.utils.molecule.xtb_cli import run_gfn_xtb

METHODS = {"gfn2": "gfn 2", "gfnff": "gfnff"}


def _write_xyz_file(
    elements: npt.NDArray, coordinates: npt.NDArray, file_path: str
) -> None:
    num_atoms = len(elements)
    with open(file_path, "w") as f:
        f.write(str(num_atoms) + "\n")
        f.write("\n")

        for i in range(num_atoms):
            element = elements[i]
            x, y, z = coordinates[i]
            line = f"{int(element)} {x:.6f} {y:.6f} {z:.6f}\n"
            f.write(line)


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


class XTBMoleculeEnergy(MoleculeEnergyBase):
    def __init__(
        self,
        method: str = "gfnff",
        batch_size=1024,
        n_samples=10000,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size, n_samples=n_samples, normalize=normalize, **kwargs
        )

        if method not in METHODS.keys():
            raise ValueError(
                f"Unrecognized method: {method}, expected one from {METHODS.keys()}."
            )
        self.method = METHODS[method]

    def compute_energy(self, states: Iterable) -> Tensor:
        # Get the number of available CPUs.
        n_jobs = len(os.sched_getaffinity(0))

        energies = []

        for batch in _chunks(states, self.batch_size):
            energies.extend(
                Parallel(n_jobs=n_jobs)(
                    delayed(get_energy)(s[:, 0], s[:, 1:], self.method) for s in batch
                )
            )

        energies = torch.tensor(energies, dtype=self.float, device=self.device)

        return energies
