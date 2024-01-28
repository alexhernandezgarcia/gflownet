import numpy as np
import pandas as pd
import torch

from gflownet.proxy.base import Proxy
from gflownet.utils.crystals.constants import ATOMIC_MASS

DENSITY_CONVERSION = 10 / 6.022  # constant to convert g/molA3 to g/cm3


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
        total_mass = states[:, 1:-7].dot(self.atomic_mass)
        a, b, c, alpha, beta, gamma = (
            states[:, -6],
            states[:, -5],
            states[:, -4],
            states[:, -3],
            states[:, -2],
            states[:, -1],
        )
        alpha_rad = torch.deg2rad(alpha)
        beta_rad = torch.deg2rad(beta)
        gamma_rad = torch.deg2rad(gamma)
        volume = (
            a
            * b
            * c
            * torch.sqrt(
                1
                - (
                    +torch.square(torch.cos(alpha_rad))
                    + torch.square(torch.cos(beta_rad))
                    + torch.square(torch.cos(gamma_rad))
                )
                + (
                    2
                    * torch.cos(alpha_rad)
                    * torch.cos(beta_rad)
                    * torch.cos(gamma_rad)
                )
            )
        )

        density = (total_mass / volume) * DENSITY_CONVERSION
        return density  # Shape: (batch,)
