"""
Base class of GFlowNet proxies
"""

import numbers
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from torchtyping import TensorType

from gflownet.utils.common import set_device, set_float_precision, tfloat

LOGZERO = -1e3


class Proxy(ABC):
    def __init__(
        self,
        device,
        float_precision,
        reward_function: Optional[Union[Callable, str]] = "identity",
        logreward_function: Optional[Callable] = None,
        reward_function_kwargs: Optional[dict] = {},
        reward_min: float = 0.0,
        do_clip_rewards: bool = False,
        **kwargs,
    ):
        r"""
        Base Proxy class for GFlowNet proxies.

        A proxy is the input to a reward function. Depending on the
        ``reward_function``, the reward may be directly the output of the proxy or a
        function of it.

        Arguments
        ---------
        device : str or torch.device
            The device to be passed to torch tensors.
        float_precision : int or torch.dtype
            The floating point precision to be passed to torch tensors.
        reward_function : str or Callable
            The transformation applied to the proxy outputs to obtain a GFlowNet
            reward. See :py:meth:`Proxy._get_reward_functions`.
        logreward_function : Callable
            The transformation applied to the proxy outputs to obtain a GFlowNet
            log reward. See :meth:`Proxy._get_reward_functions`. If None (default), the
            log of the reward function is used. The Callable may be used to improve the
            numerical stability of the transformation.
        reward_function_kwargs : dict
            A dictionary of arguments to be passed to the reward function.
        reward_min : float
            The minimum value allowed for rewards, 0.0 by default, which results in a
            minimum log reward of :py:const:`LOGZERO`. Note that certain loss
            functions, for example the Forward Looking loss may not work as desired if
            the minimum reward is 0.0. It may be set to a small (positive) value close
            to zero in order to prevent numerical stability issues.
        do_clip_rewards : bool
            Whether to clip the rewards according to the minimum value.
        """
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)
        # Parameters of proxy to reward function. Numbers are converted to float
        # tensors
        self.reward_function_kwargs = {
            k: (
                tfloat(v, float_type=self.float, device=self.device)
                if isinstance(v, numbers.Number)
                else v
            )
            for k, v in reward_function_kwargs.items()
        }
        # Proxy to reward function
        self.reward_function = reward_function
        self.logreward_function = logreward_function
        self._reward_function, self._logreward_function = self._get_reward_functions(
            reward_function, logreward_function, **self.reward_function_kwargs
        )
        # Set minimum reward and log reward. If the minimum reward is exactly 0,
        # the minimum log reward is set to -1000 in order to avoid -inf.
        self.reward_min = reward_min
        if self.reward_min == 0:
            self.logreward_min = LOGZERO
        else:
            self.logreward_min = np.log(self.reward_min)
        self.do_clip_rewards = do_clip_rewards

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
        self,
        states: Union[TensorType, List, npt.NDArray],
        log: bool = False,
        return_proxy: bool = False,
    ) -> Union[TensorType, Tuple[TensorType, TensorType]]:
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
        return_proxy : bool
            If True, returns the proxy values, alongside the rewards, as the second
            element in the returned tuple.

        Returns
        -------
        rewards : tensor
            The reward or log-reward of all elements in the batch.
        proxy_values : tensor (optional)
            The proxy value of all elements in the batch. Included only if return_proxy
            is True.
        """
        proxy_values = self(states)
        if log:
            rewards = self.proxy2logreward(proxy_values)
        else:
            rewards = self.proxy2reward(proxy_values)
        if return_proxy:
            return rewards, proxy_values
        else:
            return rewards

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
        if self.do_clip_rewards:
            logrewards = torch.clip(
                logrewards, min=self.get_min_reward(log=True), max=None
            )
        logrewards[logrewards.isnan()] = self.get_min_reward(log=True)
        logrewards[~logrewards.isfinite()] = self.get_min_reward(log=True)
        return logrewards

    def get_min_reward(self, log: bool = False) -> float:
        """
        Returns the minimum value of the (log) reward, retrieved from self.reward_min
        and self.logreward_min.

        Parameters
        ----------
        log : bool
            If True, returns the logarithm of the minimum reward. If False (default),
            returns the natural minimum reward.

        Returns
        -------
        float
            The minimum (log) reward.
        """
        if log:
            return self.logreward_min
        else:
            return self.reward_min

    def get_max_reward(self, log: bool = False) -> float:
        """
        Returns the maximum value of the (log) reward, retrieved from self.optimum, in
        case it is defined.

        Parameters
        ----------
        log : bool
            If True, returns the logarithm of the maximum reward. If False (default),
            returns the natural maximum reward.

        Returns
        -------
        float
            The maximum (log) reward.
        """
        if log:
            return self.proxy2logreward(self.optimum)
        else:
            return self.proxy2reward(self.optimum)

    @property
    def optimum(self):
        """
        Returns the optimum value of the proxy.

        Not implemented by default but may be implemented for synthetic proxies or when
        the optimum is known.

        The optimum is used, for example, to accelerate rejection sampling, to sample
        from the reward function.
        """
        if not hasattr(self, "_optimum"):
            raise NotImplementedError(
                "The optimum value of the proxy needs to be implemented explicitly for "
                f"each Proxy and is not available for {self.__class__}."
            )
        return self._optimum

    @optimum.setter
    def optimum(self, value):
        """
        Sets the optimum value of the proxy.
        """
        self._optimum = value

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
            - rbf_exp(onential): the rewards are an exponential RBF applied on the
              proxies with respect to a target value.
              See: :py:meth:`~gflownet.proxy.base._rbf_exponential()`

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
            return Proxy._exponential(**kwargs), lambda x: torch.log(
                kwargs["alpha"]
            ) + Proxy._product(beta=kwargs["beta"])(x)

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
        elif reward_function.lower().startswith("rbf_exp"):
            return (
                Proxy._rbf_exponential(**kwargs),
                lambda x: torch.log(kwargs["alpha"])
                + Proxy._product(beta=kwargs["beta"])(
                    Proxy._distance(
                        center=kwargs["center"], distance=kwargs["distance"]
                    )(x)
                ),
            )

        else:
            raise ValueError(
                "reward_function must be one of: id(entity), abs(olute) pow(er), "
                f"exp(onential), shift, prod(uct), rbf_exp(onential). "
                f"Received {reward_function} instead."
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
    def _exponential(
        beta: float = 1.0,
        alpha: float = 1.0,
    ) -> Callable:
        r"""
        Returns a lambda expression where the output is the exponential of the product
        of the input (proxy) values and beta.

        $$
        R(x) = \alpha\exp{\beta\varepsilon(x)}
        $$

        Parameters
        ----------
        beta : float
            The factor by which the proxy values are multiplied.
        alpha : float
            The factor multiplying the exponential.

        Returns
        -------
        A lambda expression that takes the exponential of the proxy values * beta, all
        multiplied by alpha.
        """
        return lambda proxy_values: alpha * torch.exp(proxy_values * beta)

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

    @staticmethod
    def _rbf_exponential(
        center: float = 0.0,
        beta: float = 1.0,
        alpha: float = 1.0,
        distance: str = "squared",
    ) -> Callable:
        r"""
        Returns a lambda expression where the output is the exponential of a distance
        to a center from the inputs (proxy values).

        $$
        R(x) = \alpha\exp(\beta dist(\varepsilon(x), c)),
        $$

        where $$c$$ is the center (a target value) and dist can be the Euclidean
        (absolute) distance or the squared Euclidean distance.

        Parameters
        ----------
        center : float
            A target value with respect to which the distance of the proxy values is
            computed.
        beta : float
            The factor by which the proxy values are multiplied.
        alpha : float
            The factor multiplying the exponential.
        distance : str
            A string indicating the type of metric used to compute the distance. The
            available options are:
                - abs(olute) OR euclidean: The Euclidean distance between the proxy
                  values and the center.
                - square(d): The squared Euclidean distance between the proxy values and
                  the center (default).

        Returns
        -------
        A lambda expression that returns an exponential radial basis function applied
        on the proxy values.
        """
        return lambda proxy_values: Proxy._exponential(beta=beta, alpha=alpha)(
            Proxy._distance(center=center, distance=distance)(proxy_values)
        )

    @staticmethod
    def _distance(center: float = 0.0, distance: str = "squared") -> Callable:
        r"""
        Auxiliary function that returns a lambda expression where a distance is
        computed with respect to a target (center).

        Parameters
        ----------
        center : float
            A target value with respect to which the distance of the proxy values is
            computed.
        distance : str
            A string indicating the type of metric used to compute the distance. The
            available options are:
                - abs(olute) OR euclidean: The Euclidean distance between the proxy
                  values and the center.
                - square(d): The squared Euclidean distance between the proxy values and
                  the center (default).

        Returns
        -------
        A lambda expression that computes a distance of the inputs with respect to a
        target.
        """
        if distance.startswith("abs") or distance.lower() == "euclidean":
            return lambda proxy_values: torch.abs(proxy_values - center)
        elif distance.lower().startswith("square"):
            return lambda proxy_values: torch.square(proxy_values - center)
        else:
            raise NotImplementedError(
                f"{distance} is not a valid identifier of a distance metric"
            )

    def infer_on_train_set(self):
        """
        Implement this method in specific proxies.
        It should return the ground-truth and proxy values on the proxy's training set.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `infer_on_train_set`."
        )
