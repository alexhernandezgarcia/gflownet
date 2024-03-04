"""
Base class of GFlowNet proxies
"""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy.typing as npt
from torchtyping import TensorType

from gflownet.utils.common import set_device, set_float_precision


class Proxy(ABC):
    """
    Generic proxy class
    """

    def __init__(self, device, float_precision, **kwargs):
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)

    def setup(self, env=None):
        pass

    @abstractmethod
    def __call__(
        self,
        states: Union[TensorType["batch", "state_dim"], npt.NDArray[np.float32], List],
    ) -> TensorType["batch"]:
        """
        Computes the values of the proxy for a batch of states.

        Parameters
        ----------
        states: torch.tensor, ndarray, list
            A batch of states in proxy format.

        Returns
        -------
        torch.tensor
            The proxy value for each state in the input batch.
        """
        pass

    @staticmethod
    def map_to_standard_range(values: TensorType["batch"]) -> TensorType["batch"]:
        """
        Maps a batch of proxy values back onto the standard range of the proxy or
        oracle. By default, it returns the values as are, so this method may be
        overwritten when needed.
        """
        return values
