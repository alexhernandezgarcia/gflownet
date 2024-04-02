"""
Base class of GFlowNet proxies
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
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
        reward_function: Optional[Union[Callable, str]] = "absolute",
        logreward_function: Optional[Callable] = None,
        reward_function_kwargs: Optional[dict] = {},
        reward_min: float = 0.0,
        do_clip_rewards: bool = False,
        **kwargs,
    ):
        # Proxy to reward function
        self.reward_function = reward_function
        self.logreward_function = logreward_function
        self.reward_function_kwargs = reward_function_kwargs
        self._reward_function, self._logreward_function = self._get_reward_functions(
            reward_function, logreward_function, **reward_function_kwargs
        )
        self.reward_min = reward_min
        self.do_clip_rewards = do_clip_rewards
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)

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

    def rewards(
        self, states: Union[TensorType, List, npt.NDArray], log: bool = False
    ) -> TensorType:
        """
        Computes the rewards of a batch of states.

        The rewards are computed by first calling the proxy function, then
        transforming the proxy values according to the reward function.

        Parameters
        ----------
        states : tensor or list or array
            A batch of states in proxy format.
        log : bool
            If True, returns the logarithm of the rewards. If False (default), returns
            the natural rewards.

        Returns
        -------
        tensor
            The reward of all elements in the batch.
        """
        if log:
            return self.proxy2logreward(self(states))
        else:
            return self.proxy2reward(self(states))

    def proxy2reward(self, proxy_values: TensorType) -> TensorType:
        """
        Transform a tensor of proxy values into rewards.

        If do_clip_rewards is True, rewards are clipped to self.reward_min.

        Parameters
        ----------
        proxy_values : tensor
            The proxy values corresponding to a batch of states.

        Returns
        -------
        tensor
            The reward of all elements in the batch.
        """
        rewards = self._reward_function(proxy_values)
        if self.do_clip_rewards:
            rewards = torch.clip(rewards, min=self.reward_min, max=None)
        return rewards

    def proxy2logreward(self, proxy_values: TensorType) -> TensorType:
        """
        Transform a tensor of proxy values into log-rewards.

        NaN values are set to self.logreward_min.

        Parameters
        ----------
        proxy_values : tensor
            The proxy values corresponding to a batch of states.

        Returns
        -------
        tensor
            The log-reward of all elements in the batch.
        """
        logrewards = self._logreward_function(proxy_values)
        logrewards[logrewards.isnan()] = self.get_min_reward(log=True)
        return logrewards

    def get_min_reward(self, log: bool = False) -> float:
        """
        Returns the minimum value of the (log) reward, retrieved from self.reward_min.

        If self.reward_min is exactly 0, then self.logreward_min is set to -inf.

        Parameters
        ----------
        log : bool
            If True, returns the logarithm of the minimum reward. If False (default),
            returns the natural minimum reward.

        Returns
        -------
        float
            The mimnimum (log) reward.
        """
        if log:
            if not hasattr(self, "logreward_min"):
                if self.reward_min == 0.0:
                    self.logreward_min = -np.inf
                else:
                    self.logreward_min = np.log(self.reward_min)
            return self.logreward_min
        else:
            return self.reward_min

    def _get_reward_functions(
        self,
        reward_function: Union[Callable, str],
        logreward_function: Callable = None,
        **kwargs,
    ) -> Tuple[Callable, Callable]:
        r"""
        Returns a tuple of callable corresponding to the function that transforms proxy
        values into rewards and log-rewards.

        If reward_function is callable, it is returned as is. If it is a string, it
        must correspond to one of the following options:

            - id(entity): the rewards are directly the proxy values.
            - abs(olute): the rewards are the absolute value of the proxy values.
            - pow(er): the rewards are the proxy values to the power of beta. See:
              :py:meth:`~gflownet.proxy.base._power()`
            - exp(onential) or boltzmann: the rewards are the negative exponential of
              the proxy values.  See: :py:meth:`~gflownet.proxy.base._exponential()`
            - shift: the rewards are the proxy values shifted by beta.
              See: :py:meth:`~gflownet.proxy.base._shift()`
            - prod(uct): the rewards are the proxy values multiplied by beta.
              See: :py:meth:`~gflownet.proxy.base._product()`

        Parameters
        ----------
        reward_function : callable or str
            A callable or a string corresponding to one of the pre-defined functions.
        reward_function : callable
            A callable of the logreward function, meant to be used to compute the log
            rewards in a more numerically stable way. None by default, in which case
            the log of the reward function will be taken.

        Returns
        -------
        Callable
            The function the transforms proxy values into rewards.
        Callable
            The function the transforms proxy values into log-rewards.
        """
        # If reward_function is callable, return it
        if isinstance(reward_function, Callable):
            if isinstance(logreward_function, Callable):
                return reward_function, logreward_function
            else:
                return reward_function, lambda x: torch.log(reward_function(x))

        # Otherwise it must be a string
        if not isinstance(reward_function, str):
            raise AssertionError(
                "reward_function must be a callable or a string; "
                f"got {type(reward_function)} instead."
            )

        if reward_function.startswith("id"):
            return (
                lambda x: x,
                lambda x: torch.log(x),
            )

        if reward_function.startswith("abs"):
            return (
                lambda x: torch.abs(x),
                lambda x: torch.log(torch.abs(x)),
            )

        elif reward_function.startswith("pow"):
            return (
                Proxy._power(**kwargs),
                lambda x: torch.log(Proxy._power(**kwargs)(x)),
            )

        elif reward_function.startswith("exp") or reward_function == "boltzmann":
            return Proxy._exponential(**kwargs), Proxy._product(**kwargs)

        elif reward_function == "shift":
            return (
                Proxy._shift(**kwargs),
                lambda x: torch.log(Proxy._shift(**kwargs)(x)),
            )

        elif reward_function.startswith("prod"):
            return (
                Proxy._product(**kwargs),
                lambda x: torch.log(Proxy._product(**kwargs)(x)),
            )

        else:
            raise ValueError(
                "reward_function must be one of: id(entity), abs(olute) pow(er), "
                f"exp(onential), shift, prod(uct). Received {reward_function} instead."
            )

    @staticmethod
    def _power(beta: float = 1.0) -> Callable:
        r"""
        Returns a lambda expression where the inputs (proxy values) are raised to the
        power of beta.

        $$
        R(x) = \varepsilon(x)^{\beta}
        $$

        Parameters
        ----------
        beta : float
            The exponent to which the proxy values are raised.

        Returns
        -------
        A lambda expression where the proxy values raised to the power of beta.
        """
        return lambda proxy_values: proxy_values**beta

    @staticmethod
    def _exponential(beta: float = 1.0) -> Callable:
        r"""
        Returns a lambda expression where the output is the exponential of the product
        of the input (proxy) values and beta.

        $$
        R(x) = \exp{\beta\varepsilon(x)}
        $$

        Parameters
        ----------
        beta : float
            The factor by which the proxy values are multiplied.

        Returns
        -------
        A lambda expression that takes the exponential of the proxy values * beta.
        """
        return lambda proxy_values: torch.exp(proxy_values * beta)

    @staticmethod
    def _shift(beta: float = 1.0) -> Callable:
        r"""
        Returns a lambda expression where the inputs (proxy values) are shifted by beta.

        $$
        R(x) = \varepsilon(x) + \beta
        $$

        Parameters
        ----------
        beta : float
            The factor by which the proxy values are shifted.

        Returns
        -------
        A lambda expression that shifts the proxy values by beta.
        """
        return lambda proxy_values: proxy_values + beta

    @staticmethod
    def _product(beta: float = 1.0) -> Callable:
        r"""
        Returns a lambda expression where the inputs (proxy values) are multiplied by
        beta.

        $$
        R(x) = \beta\varepsilon(x)
        $$

        Parameters
        ----------
        beta : float
            The factor by which the proxy values are multiplied.

        Returns
        -------
        A lambda expression that multiplies the proxy values by beta.
        """
        return lambda proxy_values: proxy_values * beta

    def infer_on_train_set(self):
        """
        Implement this method in specific proxies.
        It should return the ground-truth and proxy values on the proxy's training set.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `infer_on_train_set`."
        )
