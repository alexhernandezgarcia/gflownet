"""
Himmelblau's objective function.

See: https://en.wikipedia.org/wiki/Himmelblau%27s_function

The implementation assumes by default that the inputs will be on [-1, 1] x [-1, 1] and
will be mapped to the standard domain of the Himmelblau's function (see X1_DOMAIN and
X2_DOMAIN).

The Himmelblau's function is typically used in minimization problems, and the function
has four identical local minima with value 0. In order to map the range into the
conventional negative range, the maximum of the function in the standard domain
(UPPER_BOUND_IN_DOMAIN) is subtracted.
"""
import numpy as np
import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy

X1_DOMAIN = [-6, 6]
X1_LENGTH = X1_DOMAIN[1] - X1_DOMAIN[0]
X2_DOMAIN = [-6, 6]
X2_LENGTH = X2_DOMAIN[1] - X2_DOMAIN[0]
UPPER_BOUND_IN_DOMAIN = 182
MODES_IN_DOMAIN = [
    [3.0, 2.0],
    [-2.805118, 3.131312],
    [-3.779310, -3.283186],
    [3.584428, -1.848126],
]


class Himmelblau(Proxy):
    """
    It is assumed that the state values will be in the range [-1.0, 1.0].
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, states: TensorType["batch", "2"]) -> TensorType["batch"]:
        if states.shape[1] != 2:
            raise ValueError(
                f"""
            Inputs to the Himmelblau's function must be 2-dimensional, but inputs with
            {states.shape[1]} dimensions were passed.
            """
            )
        states = self.map_to_standard_domain(states)
        x = states[:, 0]
        y = states[:, 1]
        return self.map_to_negative_range(
            (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2
        )

    @staticmethod
    def map_to_standard_domain(
        states: TensorType["batch", "2"]
    ) -> TensorType["batch", "2"]:
        """
        Maps a batch of input states onto the domain typically used to evaluate the
        Himmelblau's function. See X1_DOMAIN and X2_DOMAIN. It assumes that the inputs are on
        [0, 1] x [0, 1].
        """
        states[:, 0] = X1_DOMAIN[0] + ((states[:, 0] + 1) * X1_LENGTH) / 2
        states[:, 1] = X2_DOMAIN[0] + ((states[:, 1] + 1) * X2_LENGTH) / 2
        return states

    @staticmethod
    def map_to_negative_range(values: TensorType["batch"]) -> TensorType["batch"]:
        """
        Maps a batch of function values onto a negative range by substracting an upper
        bound of the Himmelblau's function in the standard domain (UPPER_BOUND_IN_DOMAIN).
        """
        return values - UPPER_BOUND_IN_DOMAIN

    @staticmethod
    def map_to_standard_range(values: TensorType["batch"]) -> TensorType["batch"]:
        """
        Maps a batch of function values in a negative range back onto the standard
        range by adding an upper bound of the Himmelblau's function in the standard domain
        (UPPER_BOUND_IN_DOMAIN).
        """
        return values + UPPER_BOUND_IN_DOMAIN

    @property
    def min(self):
        return torch.tensor(
            -UPPER_BOUND_IN_DOMAIN, device=self.device, dtype=self.float
        )
