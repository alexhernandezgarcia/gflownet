"""
Branin objective function, relying on the botorch implementation.

This code is based on the implementation by Nikita Saxena (nikita-0209) in 
https://github.com/alexhernandezgarcia/activelearning

The implementation assumes by default that the inputs will be on [-1, 1] x [-1, 1] and
will be mapped to the standard domain of the Branin function (see X1_DOMAIN and
X2_DOMAIN). Setting do_domain_map to False will prevent the mapping.

Branin function is typically used as a minimization problem, with the minima around
zero but positive. By default, proxy values remain in this range and the reward is
mapped to (~0, UPPER_BOUND_IN_DOMAIN). The user should carefully select the reward
function in order to stick to the conventional use of the Branin function.
"""

from typing import Callable, Optional, Union

import torch
from botorch.test_functions.multi_fidelity import AugmentedBranin
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

X1_DOMAIN = [-5, 10]
X1_LENGTH = X1_DOMAIN[1] - X1_DOMAIN[0]
X2_DOMAIN = [0, 15]
X2_LENGTH = X2_DOMAIN[1] - X2_DOMAIN[0]
UPPER_BOUND_IN_DOMAIN = 309
OPTIMUM = 0.397887
# Modes compatible with 100x100 grid
MODES = [
    [12.4, 81.833],
    [54.266, 15.16],
    [94.98, 16.5],
]


class Branin(Proxy):
    def __init__(
        self,
        fidelity: float = 1.0,
        do_domain_map: bool = True,
        negate: bool = False,
        reward_function: Optional[Union[Callable, str]] = lambda x: -1.0
        * (x - UPPER_BOUND_IN_DOMAIN),
        **kwargs,
    ):
        """
        Parameters
        ----------
        fidelity : float
            Fidelity of the Branin oracle. 1.0 corresponds to the original Branin.
            Smaller values (up to 0.0) reduce the fidelity of the oracle.
        do_domain_map : bool
            If True, the states are assumed to be in [-1, 1] x [-1, 1] and are re-mapped
            to the standard domain before calling the botorch method. If False, the
            botorch method is called directly on the states values.
        negate : bool
            If True, proxy values are multiplied by -1.
        reward_function : str or Callable
            The transformation applied to the proxy outputs to obtain a GFlowNet
            reward. By default, proxy values are shifted by UPPER_BOUND_IN_DOMAIN and
            multiplied by minus one, in order to make them positive and the higher the
            better.

        See: https://botorch.org/api/test_functions.html
        """
        # Replace the value of reward_function in kwargs by the one passed explicitly
        # as a parameter
        kwargs["reward_function"] = reward_function
        # Call __init__ of parent class
        super().__init__(**kwargs)
        self.fidelity = fidelity
        self.do_domain_map = do_domain_map
        self.function_mf_botorch = AugmentedBranin(negate=negate)
        # Constants
        self.domain_left = tfloat(
            [[X1_DOMAIN[0], X2_DOMAIN[0]]], float_type=self.float, device=self.device
        )
        self.domain_length = tfloat(
            [[X1_LENGTH, X2_LENGTH]], float_type=self.float, device=self.device
        )
        # Optimum
        self._optimum = torch.tensor(OPTIMUM, device=self.device, dtype=self.float)
        if negate:
            self._optimum *= -1.0

    def __call__(self, states: TensorType["batch", "2"]) -> TensorType["batch"]:
        if states.shape[1] != 2:
            raise ValueError(
                "Inputs to the Branin function must be 2-dimensional, "
                f"but inputs with {states.shape[1]} dimensions were passed."
            )
        if self.do_domain_map:
            states = self.map_to_standard_domain(states)
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
        states: TensorType["batch", "2"],
    ) -> TensorType["batch", "2"]:
        """
        Maps a batch of input states onto the domain typically used to evaluate the
        Branin function. See X1_DOMAIN and X2_DOMAIN. It assumes that the inputs are on
        [-1, 1] x [-1, 1].
        """
        return self.domain_left + ((states + 1.0) * self.domain_length) / 2.0
