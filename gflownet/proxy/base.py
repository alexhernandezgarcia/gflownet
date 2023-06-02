"""
Base class of GFlowNet proxies
"""
from abc import ABC, abstractmethod

from gflownet.utils.common import set_device, set_float_precision
from torchtyping import TensorType


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
    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        """
        Args:
            states: ndarray
        Function:
            calls the get_reward method of the appropriate Proxy Class (EI, UCB, Proxy,
            Oracle etc)
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
