"""
Base class of GFlowNet proxies
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Union

import numpy as np
import numpy.typing as npt
from torchtyping import TensorType

from gflownet.utils.common import set_device, set_float_precision


class Proxy(ABC):
    """
    Generic proxy class
    """

    def __init__(
        self,
        device,
        float_precision,
        reward_function: Union[Callable, str] = "identity",
        reward_function_kwargs: dict = None,
        higher_is_better=False,
        **kwargs,
    ):
        # Proxy to reward function
        self.reward_function = self._get_reward_function(
            reward_function, reward_function_kwargs
        )
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)
        # Reward2Proxy multiplicative factor (1 or -1)
        self.higher_is_better = higher_is_better

    def setup(self, env=None):
        pass

    @abstractmethod
    def __call__(self, states: Union[TensorType, List, npt.NDArray]) -> TensorType:
        """
        Implement  this function to call the get_reward method of the appropriate Proxy
        Class (EI, UCB, Proxy, Oracle etc).

        Parameters
        ----------
            states: ndarray
        """
        pass

    def rewards(self, states: Union[TensorType, List, npt.NDArray]) -> TensorType:
        """
        Computes the rewards of a batch of states.

        The rewards are computing by first calling the proxy function, then
        transforming the proxy values according to the reward function.

        Parameters
        ----------
        states : tensor or list or array
            A batch of states in proxy format.

        Returns
        -------
        tensor
            The reward of all elements in the batch.
        """
        return self.proxy2reward(self(states))

    # TODO: consider adding option to clip values
    # TODO: check that rewards are non-negative
    def proxy2reward(proxy_values: TensorType) -> TensorType:
        """
        Transform a tensor of proxy values into rewards.

        Parameters
        ----------
        proxy_values : tensor
            The proxy values corresponding to a batch of states.

        Returns
        -------
        tensor
            The reward of all elements in the batch.
        """
        return self.reward_func(proxy_values)

    def _get_reward_function(reward_function: Union[Callable, str], **kwargs):
        r"""
        Returns a callable corresponding to the function that transforms proxy values
        into rewards.

        If reward_function is callable, it is returned as is. If it is a string, it
        must correspond to one of the following options:

            - power: the rewards are the proxy values to the power of beta. See:
              :py:meth:`~gflownet.proxy.base._power()`
            - boltzmann: the rewards are the negative exponential of the proxy values.
              See: :py:meth:`~gflownet.proxy.base._boltzmann()`
            - shift: the rewards are the proxy values shifted by beta.
              See: :py:meth:`~gflownet.proxy.base._boltzmann()`

        Parameters
        ----------
        reward_function : callable or str
            A callable or a string corresponding to one of the pre-defined functions.
        """
        # If reward_function is callable, return it
        if isinstance(reward_function, Callable):
            return reward_function

        # Otherwise it must be a string
        if not isinstance(reward_function, str):
            raise AssertionError(
                "reward_func must be a callable or a string; "
                f"got {type(reward_function)} instead."
            )

    def infer_on_train_set(self):
        """
        Implement this method in specific proxies.
        It should return the ground-truth and proxy values on the proxy's training set.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `infer_on_train_set`."
        )
