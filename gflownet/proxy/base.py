"""
Base class of GFlowNet proxies
"""
from abc import abstractmethod
import numpy as np
import numpy.typing as npt
from gflownet.utils.common import set_device, set_float_precision


class Proxy:
    """
    Generic proxy class
    """

    def __init__(self, device, float_precision, higher_is_better=False, **kwargs):
        # Device
        self.set_device(device)
        # Float precision
        self.set_float_precision(float_precision)
        # Reward2Proxy multiplicative factor (1 or -1)
        self.higher_is_better = higher_is_better

    def set_device(self, device):
        self.device = set_device(device)

    def set_float_precision(self, dtype):
        self.float = set_float_precision(dtype)

    @abstractmethod
    def __call__(self, states: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Args:
            states: ndarray
        Function:
            calls the get_reward method of the appropriate Proxy Class (EI, UCB, Proxy,
            Oracle etc)
        """
        pass
