import numpy as np
import pandas as pd
from gflownet.proxy.base import Proxy
from gflownet.utils.crystals.constants import ATOMIC_MASS
import torch

LENGTH_SCALE = (0.9, 100)
ANGLE_SCALE = (50, 150)


class DensityProxy(Proxy):
    def __init__(self, device, float_precision, higher_is_better=False, **kwargs):
        """
        Proxy to compute the density of a crystal, in g/cm3
        It requires the same inputs as  the Dave proxy
        """
        super().__init__(device, float_precision, higher_is_better, **kwargs)
        self.atomic_mass = torch.tensor(ATOMIC_MASS.values())

    @torch.no_grad()
    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states (torch.Tensor): same as DAVE proxy, i.e.
            * composition: ``states[:, :-7]`` -> length 95 (dummy 0 then 94 elements)
            * space group: ``states[:, -7] - 1``
            * lattice parameters: ``states[:, -6:]``

        Returns:
            nd.array: Ehull energies. Shape: ``(batch,)``.
        """
        total_mass = states[:, :-7].dot(self.atomic_mass)
        lattice = states[:, -6:]
        lattice[:, 0:3] = (
            lattice[:, 0:3] * (LENGTH_SCALE[1] - LENGTH_SCALE[0]) + LENGTH_SCALE[0]
        )
        lattice[:, 3:6] = (
            lattice[:, 3:6] * (ANGLE_SCALE[1] - ANGLE_SCALE[0]) + ANGLE_SCALE[0]
        )
        lattice[:, 3:6] = torch.deg2rad(lattice[:, 3:6])  # convert to radians
        volume = (
            lattice[:, 0]
            * lattice[:, 1]
            * lattice[:, 2]
            * torch.sqrt(
                1
                - (
                    +torch.square(torch.cos(lattice[:, 3]))
                    + torch.square(torch.cos(lattice[:, 4]))
                    + torch.square(torch.cos(lattice[:, 5]))
                )
                + (
                    2
                    * torch.cos(lattice[:, 3])
                    * torch.cos(lattice[:, 4])
                    * torch.cos(lattice[:, 5])
                )
            )
        )

        density = (total_mass / volume) * (10 / 6.022)  # constant to convert to g/cm3
        return density  # Shape: (batch,)
