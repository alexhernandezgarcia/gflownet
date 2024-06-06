import warnings

import numpy as np
import pandas as pd
import torch
from torchtyping import TensorType

from gflownet.envs.crystals.composition import N_ELEMENTS_ORACLE
from gflownet.envs.crystals.crystal import Crystal
from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat, tlong
from gflownet.utils.crystals.constants import ATOMIC_MASS

DENSITY_CONVERSION = 10 / 6.022  # constant to convert g/molA3 to g/cm3


class Density(Proxy):
    def __init__(self, **kwargs):
        """
        Proxy to compute the density of a crystal, in g/cm3.

        It requires the same inputs as the Dave proxy.
        """
        super().__init__(**kwargs)

    def setup(self, env=None):
        if isinstance(env, Crystal):
            self.atomic_mass = torch.zeros(N_ELEMENTS_ORACLE + 1, dtype=self.float)
            elements = env.subenvs[env.stage_composition].elements
            atomic_mass_elements = tfloat(
                [ATOMIC_MASS[n] for n in elements],
                float_type=self.float,
                device=self.device,
            )
            self.atomic_mass[tlong(elements, device=self.device)] = atomic_mass_elements
        else:
            warnings.warn(
                "Attempted to setup Density proxy without passing the right "
                "Crystal env type (continuous crystal stack)"
            )

    @torch.no_grad()
    def __call__(
        self, states: TensorType["batch", "policy_input_dim"]
    ) -> TensorType["batch"]:
        """
        Args:
            states (torch.Tensor): same as DAVE proxy, i.e.
            * composition: ``states[:, :-7]`` -> length 95 (dummy 0 then 94 elements)
            * space group: ``states[:, -7] - 1``
            * lattice parameters: ``states[:, -6:]``

        Returns:
            nd.array: -1 * density in g/cm3. Shape: ``(batch,)``.
        """
        total_mass = torch.matmul(states[:, :-7], self.atomic_mass)
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

        return (total_mass / volume) * DENSITY_CONVERSION
