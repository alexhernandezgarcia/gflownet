"""
Hartmann objective function, relying on the botorch implementation.

See:
https://botorch.org/api/test_functions.html#botorch.test_functions.synthetic.Hartmann

This code is based on the implementation by Nikita Saxena (nikita-0209) in
https://github.com/alexhernandezgarcia/activelearning

The implementation assumes that the inputs will be on [-1, 1]^6 as is typical in the
uses of the Hartmann function. The original range is negative and is a minimisation
problem. By default, the proxy values remain in this range and the absolute value of
the proxy values is used as the reward function.
"""

from typing import Callable, Optional, Union

import numpy as np
import torch
from botorch.test_functions.multi_fidelity import AugmentedHartmann
from torchtyping import TensorType

from gflownet.proxy.base import Proxy

X_DOMAIN = [0, 1]
# Global optimum, according to BoTorch
OPTIMUM = -3.32237
# A rough estimate of modes
MODES = [
    [0.2, 0.2, 0.5, 0.3, 0.3, 0.7],
    [0.4, 0.9, 0.9, 0.6, 0.1, 0.0],
    [0.3, 0.1, 0.4, 0.3, 0.3, 0.7],
    [0.4, 0.9, 0.4, 0.6, 0.0, 0.0],
    [0.4, 0.9, 0.6, 0.6, 0.3, 0.0],
]


class Hartmann(Proxy):
    def __init__(
        self,
        fidelity=1.0,
        do_domain_map: bool = True,
        negate: bool = False,
        reward_function: Optional[Union[Callable, str]] = "absolute",
        **kwargs,
    ):
        """
        Parameters
        ----------
        fidelity : float
            Fidelity of the Hartmann oracle. 1.0 corresponds to the original Hartmann.
            Smaller values (up to 0.0) reduce the fidelity of the oracle.
        do_domain_map : bool
            If True, the states are assumed to be in [-1, 1]^6 and are re-mapped to the
            standard domain in [0, 1]^6 before calling the botorch method. If False,
            the botorch method is called directly on the states values.
        negate : bool
            If True, proxy values are multiplied by -1.
        reward_function : str or Callable
            The transformation applied to the proxy outputs to obtain a GFlowNet
            reward. By default, the reward function is the absolute value of proxy
            outputs.

        See: https://botorch.org/api/test_functions.html
        """
        # Replace the value of reward_function in kwargs by the one passed explicitly
        # as a parameter
        kwargs["reward_function"] = reward_function
        super().__init__(**kwargs)
        self.fidelity = fidelity
        self.do_domain_map = do_domain_map
        self.function_mf_botorch = AugmentedHartmann(negate=negate)
        # Optimum
        self._optimum = torch.tensor(OPTIMUM, device=self.device, dtype=self.float)
        if negate:
            self._optimum *= -1.0

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        if states.shape[1] != 6:
            raise ValueError(
                "Inputs to the Hartmann function must be 6-dimensional, "
                f"but inputs with {states.shape[1]} dimensions were passed."
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

    def map_to_standard_domain(
        self,
        states: TensorType["batch", "6"],
    ) -> TensorType["batch", "6"]:
        """
        Maps a batch of input states onto the domain typically used to evaluate the
        Hartmann function, that is [0, 1]^6. See DOMAIN and LENGTH. It assumes that the
        inputs are on [-1, 1]^6
        """
        return (states + 1.0) / 2.0
