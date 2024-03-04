"""
Hartmann objective function, relying on the botorch implementation.

This code is based on the implementation by Nikita Saxena (nikita-0209) in
https://github.com/alexhernandezgarcia/activelearning
"""

import numpy as np
import torch
from botorch.test_functions.multi_fidelity import AugmentedHartmann
from torchtyping import TensorType

from gflownet.proxy.base import Proxy


class Hartmann(Proxy):
    def __init__(self, fidelity=1.0, **kwargs):
        super().__init__(**kwargs)
        self.fidelity = fidelity
        self.function_mf_botorch = AugmentedHartmann(negate=True)
        # This is just a rough estimate of modes
        self.modes = [
            [0.2, 0.2, 0.5, 0.3, 0.3, 0.7],
            [0.4, 0.9, 0.9, 0.6, 0.1, 0.0],
            [0.3, 0.1, 0.4, 0.3, 0.3, 0.7],
            [0.4, 0.9, 0.4, 0.6, 0.0, 0.0],
            [0.4, 0.9, 0.6, 0.6, 0.3, 0.0],
        ]
        self.extrema = 3.32237

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        if states.shape[1] != 6:
            raise ValueError(
                """
            Inputs to the Hartmann function must be 6-dimensional, but inputs with
            {states.shape[1]} dimensions were passed.
            """
            )
        # Append fidelity as a new dimension of states
        states = torch.cat(
            [
                states,
                self.fidelity
                * torch.ones(
                    states.shape[0], device=self.device, dtype=self.float
                ).unsqueeze(-1),
            ],
            dim=1,
        )
        return self.function_mf_botorch(states)
