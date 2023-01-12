"""
Base class of GFlowNet proxies
"""
from abc import abstractmethod
import numpy as np
import numpy.typing as npt


class Proxy:
    """
    Generic proxy class
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        kwargs:
            for the acquisition, and model, the trained surrogate would be an input arg
            but this wouldn't be so for the oracle
        """

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
