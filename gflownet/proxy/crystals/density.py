import numpy as np
import pandas as pd
import torch

from gflownet.proxy.base import Proxy
from gflownet.utils.crystals.constants import ATOMIC_MASS

DENSITY_CONVERSION = 10 / 6.022  # constant to convert g/molA3 to g/cm3


class Density(Proxy):
    def __init__(self, device, float_precision, higher_is_better=False, **kwargs):
        """
        Proxy to compute the density of a crystal, in g/cm3
        It requires the same inputs as  the Dave proxy
        """
        super().__init__(device, float_precision, higher_is_better, **kwargs)

    def setup(self, env=None):
        self.atomic_mass = torch.tensor(
            [ATOMIC_MASS[n] for n in env.composition.elements]
        )
        assert 1 == 1

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
        total_mass = torch.matmul(states[:, 1:-7], self.atomic_mass)
        a, b, c, cos_alpha, cos_beta, cos_gamma = (
            states[:, -6],
            states[:, -5],
            states[:, -4],
            torch.cos(torch.deg2rad(states[:, -3])),
            torch.cos(torch.deg2rad(states[:, -2])),
            torch.cos(torch.deg2rad(states[:, -1])),
        )
        volume = (a * b * c) * torch.sqrt(
            1
            - (cos_alpha.pow(2) + cos_beta.pow(2) + cos_gamma.pow(2))
            + (2 * cos_alpha * cos_beta * cos_gamma)
        )

        density = (total_mass / volume) * DENSITY_CONVERSION
        return density  # Shape: (batch,)
