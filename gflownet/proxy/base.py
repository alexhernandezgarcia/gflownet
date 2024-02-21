"""
Base class of GFlowNet proxies
"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from gflownet.utils.common import set_device, set_float_precision


class Proxy(ABC):
    """
    Generic proxy class
    """

    def __init__(self, device, float_precision, higher_is_better=False, **kwargs):
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)
        # Reward2Proxy multiplicative factor (1 or -1)
        self.higher_is_better = higher_is_better

    def setup(self, env=None):
        pass

    @abstractmethod
    def __call__(self, states: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Implement  this function to call the get_reward method of the appropriate Proxy
        Class (EI, UCB, Proxy, Oracle etc).

        Parameters
        ----------
            states: ndarray
        """
        pass

    def infer_on_train_set(self):
        """
        Implement this method in specific proxies.
        It should return the ground-truth and proxy values on the proxy's training set.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `infer_on_train_set`."
        )
