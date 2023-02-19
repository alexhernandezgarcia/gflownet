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

    def __init__(self, device, float_precision, maximize=None, **kwargs):
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)
        # Reward2Proxy multiplicative factor (1 or -1)
        self.maximize = maximize

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
