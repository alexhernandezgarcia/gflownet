"""
Classes to represent hyper-cube environments
"""

import itertools
import warnings
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from torch.distributions import Bernoulli, Beta, Categorical, MixtureSameFamily
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tbool, tfloat, torch2np

CELL_MIN = -1.0
CELL_MAX = 1.0


class CubeBase(GFlowNetEnv, ABC):
    """
    Base class for hyper-cube environments, continuous or hybrid versions of the
    hyper-grid in which the continuous increments are modelled by a (mixture of) Beta
    distribution(s).

    The states space is the value of each dimension, defined in the closed set [0, 1].
    If the value of a dimension gets larger than 1 - min_incr, then the trajectory is
    ended (the only possible action is EOS).

    Attributes
    ----------
    n_dim : int
        Dimensionality of the hyper-cube.

    min_incr : float
        Minimum increment in the actions, in (0, 1). This is necessary to ensure
        that all trajectories have finite length.

    n_comp : int
        Number of components in the mixture of Beta distributions.

    epsilon : float
        Small constant to control the clamping interval of the inputs to the
        calculation of log probabilities. Clamping interval will be [epsilon, 1 -
        epsilon]. Default: 1e-6.

    kappa : float
        Small constant to control the intervals of the generated sets of states (in a
        grid or uniformly). States will be in the interval [kappa, 1 - kappa]. Default:
        1e-3.

    ignored_dims : list
        Boolean mask of ignored dimensions. This can be used for trajectories that may
        have multiple dimensions coupled or fixed. For each dimension, True if ignored,
        False, otherwise. If None, no dimension is ignored.
    """

    def __init__(
        self,
        n_dim: int = 2,
        min_incr: float = 0.1,
        n_comp: int = 1,
        beta_params_min: float = 0.1,
        beta_params_max: float = 100.0,
        epsilon: float = 1e-6,
        kappa: float = 1e-3,
        ignored_dims: Optional[List[bool]] = None,
        fixed_distr_params: dict = {
            "beta_weights": 1.0,
            "beta_alpha": 10.0,
            "beta_beta": 10.0,
            "bernoulli_bts_prob": 0.1,
            "bernoulli_eos_prob": 0.1,
        },
        random_distr_params: dict = {
            "beta_weights": 1.0,
            "beta_alpha": 10.0,
            "beta_beta": 10.0,
            "bernoulli_bts_prob": 0.1,
            "bernoulli_eos_prob": 0.1,
        },
        **kwargs,
    ):
        assert n_dim > 0
        assert min_incr > 0.0
        assert min_incr < 1.0
        assert n_comp > 0
        # Main properties
        self.n_dim = n_dim
        self.min_incr = min_incr
        if ignored_dims:
            self.ignored_dims = ignored_dims
        else:
            self.ignored_dims = [False] * self.n_dim
        # Parameters of the policy distribution
        self.n_comp = n_comp
        self.beta_params_min = beta_params_min
        self.beta_params_max = beta_params_max
        # Source state is abstract - not included in the cube: -1 for all dimensions.
        self.source = [-1 for _ in range(self.n_dim)]
        # Small constant to clamp the inputs to the beta distribution
        self.epsilon = epsilon
        # Small constant to restrict the interval of (test) sets
        self.kappa = kappa
        # Base class init
        super().__init__(
            fixed_distr_params=fixed_distr_params,
            random_distr_params=random_distr_params,
            **kwargs,
        )
        self.continuous = True

    @abstractmethod
    def get_action_space(self):
        pass

    @abstractmethod
    def get_policy_output(self, params: dict) -> TensorType["policy_output_dim"]:
        pass

    @abstractmethod
    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        pass

    @abstractmethod
    def get_mask_invalid_actions_backward(self, state=None, done=None, parents_a=None):
        pass

    def states2proxy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy: clips the
        states into [0, 1] and maps them to [CELL_MIN, CELL_MAX]

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tfloat(states, device=self.device, float_type=self.float)
        return 2.0 * torch.clip(states, min=0.0, max=CELL_MAX) - CELL_MAX

    def states2policy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: clips
        the states into [0, 1] and maps them to [-1.0, 1.0]

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tfloat(states, device=self.device, float_type=self.float)
        return 2.0 * torch.clip(states, min=0.0, max=1.0) - 1.0

    def state2readable(self, state: List) -> str:
        """
        Converts a state (a list of positions) into a human-readable string
        representing a state.
        """
        return str(state).replace("(", "[").replace(")", "]").replace(",", "")

    def readable2state(self, readable: str) -> List:
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        return [el for el in readable.strip("[]").split(" ")]

    @abstractmethod
    def get_parents(
        self, state: List = None, done: bool = None, action: Tuple[int, float] = None
    ) -> Tuple[List[List], List[Tuple[int, float]]]:
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : int
            Last action performed

        Returns
        -------
        parents : list
            List of parents in state format

        actions : list
            List of actions that lead to state for each parent in parents
        """
        pass

    @abstractmethod
    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        sampling_method: str = "policy",
        mask_invalid_actions: TensorType["n_states", "1"] = None,
        temperature_logits: float = 1.0,
        loginf: float = 1000,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs.
        """
        pass

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        is_forward: bool,
        actions: TensorType["n_states", 2],
        mask_invalid_actions: TensorType["batch_size", "policy_output_dim"] = None,
        loginf: float = 1000,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.
        """
        pass

    def step(
        self, action: Tuple[int, float]
    ) -> Tuple[List[float], Tuple[int, float], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple with two values:
            (dimension, increment).

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action executed

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        pass

    def _beta_params_to_policy_outputs(self, param_name: str, params_dict: dict):
        """
        Maps the values of alpha and beta given in the configuration to new values such
        that when passed to _make_increments_distribution, the actual alpha and beta
        passed to the Beta distribution(s) are the ones from the configuration.

        Args
        ----
        param_name : str
            Name of the parameter to transform: alpha or beta

        params_dict : dict
            Dictionary with the complete set of parameters of the distribution.

        See
        ---
        _make_increments_distribution()
        """
        param_value = tfloat(
            params_dict[f"beta_{param_name}"], float_type=self.float, device=self.device
        )
        return torch.logit((param_value - self.beta_params_min) / self.beta_params_max)

    def _get_effective_dims(self, state: Optional[List] = None) -> List:
        state = self._get_state(state)
        return [s for s, ign_dim in zip(state, self.ignored_dims) if not ign_dim]


class ContinuousCube(CubeBase):
    """
    Continuous hyper-cube environment (continuous version of a hyper-grid) in which the
    action space consists of the increment of each dimension d, modelled by a mixture
    of Beta distributions. The state space is the value of each dimension. In order to
    ensure that all trajectories are of finite length, actions have a minimum increment
    for all dimensions determined by min_incr. If the value of any dimension is larger
    than 1 - min_incr, then that dimension can't be further incremented. In order to
    ensure the coverage of the state space, the first action (from the source state) is
    not constrained by the minimum increment.

    Actions do not represent absolute increments but rather the relative increment with
    respect to the distance to the edges of the hyper-cube, from the minimum increment.
    That is, if dimension d of a state has value 0.3, the minimum increment (min_incr)
    is 0.1 and the maximum value is 1.0, an action of 0.5 will increment the
    value of the dimension in 0.5 * (1.0 - 0.3 - 0.1) = 0.5 * 0.6 = 0.3. Therefore, the
    value of d in the next state will be 0.3 + 0.3 = 0.6.

    Attributes
    ----------
    n_dim : int
        Dimensionality of the hyper-cube.

    min_incr : float
        Minimum increment in the actions, in (0, 1). This is necessary to ensure
        that all trajectories have finite length.

    n_comp : int
        Number of components in the mixture of Beta distributions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Mask dimensionality: 3 + number of dimensions
        self.mask_dim_base = 3
        self.mask_dim = self.mask_dim_base + self.n_dim

    def get_action_space(self):
        """
        The action space is continuous, thus not defined as such here.

        The actions are tuples of length n_dim + 1, where the value at position d
        indicates the increment of dimension d, and the value at position -1 indicates
        whether the action is from or to source (1), or 0 otherwise.

        EOS is indicated by np.inf for all dimensions.

        This method defines self.eos and the returned action space is simply
        a representative (arbitrary) action with an increment of 0.0 in all dimensions,
        and EOS.
        """
        actions_dim = self.n_dim + 1
        self.eos = tuple([np.inf] * actions_dim)
        self.representative_action = tuple([0.0] * actions_dim)
        return [self.representative_action, self.eos]

    def get_max_traj_length(self):
        return np.ceil(1.0 / self.min_incr) + 2

    def get_policy_output(self, params: dict) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model.

        The policy output will be used to initialize a distribution, from which an
        action is to be determined or sampled. This method returns a vector with a
        fixed policy defined by params.

        The environment consists of both continuous and discrete actions.

        Continuous actions

        For each dimension d of the hyper-cube and component c of the mixture, the
        output of the policy should return:
          1) the weight of the component in the mixture,
          2) the pre-alpha parameter of the Beta distribution to sample the increment,
          3) the pre-beta parameter of the Beta distribution to sample the increment.

        These parameters are the first n_dim * n_comp * 3 of the policy output such
        that the first 3 x C elements correspond to the first dimension, and so on.

        Discrete actions

        Additionally, the policy output contains one logit (pos -1) of a Bernoulli
        distribution to model the (discrete) forward probability of selecting the EOS
        action and another logit (pos -2) for the (discrete) backward probability of
        returning to the source node.

        Therefore, the output of the policy model has dimensionality D x C x 3 + 2,
        where D is the number of dimensions (self.n_dim) and C is the number of
        components (self.n_comp).

        See
        ---
        _beta_params_to_policy_outputs()
        """
        # Parameters for continuous actions
        self._len_policy_output_cont = self.n_dim * self.n_comp * 3
        policy_output_cont = torch.empty(
            self._len_policy_output_cont,
            dtype=self.float,
            device=self.device,
        )
        policy_output_cont[0::3] = params["beta_weights"]
        policy_output_cont[1::3] = self._beta_params_to_policy_outputs("alpha", params)
        policy_output_cont[2::3] = self._beta_params_to_policy_outputs("beta", params)
        # Logit for Bernoulli distribution to model EOS action
        policy_output_eos_logit = torch.logit(
            tfloat(
                [params["bernoulli_eos_prob"]],
                float_type=self.float,
                device=self.device,
            )
        )
        # Logit for Bernoulli distribution to model back-to-source action
        policy_output_bts_logit = torch.logit(
            tfloat(
                [params["bernoulli_bts_prob"]],
                float_type=self.float,
                device=self.device,
            )
        )
        # Concatenate all outputs
        policy_output = torch.cat(
            (
                policy_output_cont,
                policy_output_bts_logit,
                policy_output_eos_logit,
            )
        )
        return policy_output

    def _get_policy_betas_weights(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "n_dim * n_comp"]:
        """
        Reduces a given policy output to the part corresponding to the weights of the
        mixture of Beta distributions.

        See: get_policy_output()
        """
        return policy_output[:, 0 : self._len_policy_output_cont : 3]

    def _get_policy_betas_alpha(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "n_dim * n_comp"]:
        """
        Reduces a given policy output to the part corresponding to the alphas of the
        mixture of Beta distributions.

        See: get_policy_output()
        """
        return policy_output[:, 1 : self._len_policy_output_cont : 3]

    def _get_policy_betas_beta(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "n_dim * n_comp"]:
        """
        Reduces a given policy output to the part corresponding to the betas of the
        mixture of Beta distributions.

        See: get_policy_output()
        """
        return policy_output[:, 2 : self._len_policy_output_cont : 3]

    def _get_policy_eos_logit(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "1"]:
        """
        Reduces a given policy output to the part corresponding to the logit of the
        Bernoulli distribution to model the EOS action.

        See: get_policy_output()
        """
        return policy_output[:, -1]

    def _get_policy_source_logit(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "1"]:
        """
        Reduces a given policy output to the part corresponding to the logit of the
        Bernoulli distribution to model the back-to-source action.

        See: get_policy_output()
        """
        return policy_output[:, -2]

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        The action space is continuous, thus the mask is not only of invalid actions as
        in discrete environments, but also an indicator of "special cases", for example
        states from which only certain actions are possible.

        The values of True/False intend to approximately stick to the semantics in
        discrete environments, where the mask is of "invalid" actions, but it is
        important to note that a direct interpretation in this sense does not always
        apply.

        For example, the mask values of special cases are True if the special cases they
        refer to are "invalid". In other words, the values are False if the state has
        the special case.

        The forward mask has the following structure:

        - 0 : whether a continuous action is invalid. True if the value at any
          dimension is larger than 1 - min_incr, or if done is True. False otherwise.
        - 1 : special case when the state is the source state. False when the state is
          the source state, True otherwise.
        - 2 : whether EOS action is invalid. EOS is valid from any state, except the
          source state or if done is True.
        - -n_dim: : dimensions that should be ignored when sampling actions or
          computing logprobs. This can be used for trajectories that may have
          multiple dimensions coupled or fixed. For each dimension, True if ignored,
          False, otherwise.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        # If done, the entire mask is True (all actions are "invalid" and no special
        # cases)
        if done:
            return [True] * self.mask_dim
        mask = [False] * self.mask_dim_base + self.ignored_dims
        # If the state is the source state, EOS is invalid
        if self._get_effective_dims(state) == self._get_effective_dims(self.source):
            mask[2] = True
        # If the state is not the source, indicate not special case (True)
        else:
            mask[1] = True
        # If the value of any dimension is greater than 1 - min_incr, then continuous
        # actions are invalid (True).
        if any([s > 1 - self.min_incr for s in self._get_effective_dims(state)]):
            mask[0] = True
        return mask

    def get_mask_invalid_actions_backward(self, state=None, done=None, parents_a=None):
        """
        The action space is continuous, thus the mask is not only of invalid actions as
        in discrete environments, but also an indicator of "special cases", for example
        states from which only certain actions are possible.

        In order to approximately stick to the semantics in discrete environments,
        where the mask is of "invalid" actions, that is the value is True if an action
        is invalid, the mask values of special cases are True if the special cases they
        refer to are "invalid". In other words, the values are False if the state has
        the special case.

        The backward mask has the following structure:

        - 0 : whether a continuous action is invalid. True if the value at any
          dimension is smaller than min_incr, or if done is True. False otherwise.
        - 1 : special case when back-to-source action is the only possible action.
          False if any dimension is smaller than min_incr, True otherwise.
        - 2 : whether EOS action is invalid. False only if done is True, True
          (invalid) otherwise.
        - -n_dim: : dimensions that should be ignored when sampling actions or
          computing logprobs. this can be used for trajectories that may have
          multiple dimensions coupled or fixed. for each dimension, true if ignored,
          false, otherwise. By default, no dimension is ignored.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        mask = [True] * self.mask_dim_base + self.ignored_dims
        # If the state is the source state, entire mask is True
        if self._get_effective_dims(state) == self._get_effective_dims(self.source):
            return mask
        # If done, only valid action is EOS.
        if done:
            mask[2] = False
            return mask
        # If any dimension is smaller than m, then back-to-source action is the only
        # possible actiona.
        if any([s < self.min_incr for s in self._get_effective_dims(state)]):
            mask[1] = False
            return mask
        # Otherwise, continuous actions are valid
        mask[0] = False
        return mask

    def get_parents(
        self, state: List = None, done: bool = None, action: Tuple[int, float] = None
    ) -> Tuple[List[List], List[Tuple[int, float]]]:
        """
        Defined only because it is required. A ContinuousEnv should be created to avoid
        this issue.
        """
        pass

    def relative_to_absolute_increments(
        self,
        states: TensorType["n_states", "n_dim"],
        increments_rel: TensorType["n_states", "n_dim"],
        is_backward: bool,
    ):
        """
        Returns a batch of absolute increments (actions) given a batch of states,
        relative increments and minimum_increments.

        Given a dimension value x, a relative increment r, and a minimum increment m,
        then the absolute increment a is given by:

        Forward:

        a = m + r * (1 - x - m)

        Backward:

        a = m + r * (x - m)
        """
        min_increments = torch.full_like(
            increments_rel, self.min_incr, dtype=self.float, device=self.device
        )
        if is_backward:
            return min_increments + increments_rel * (states - min_increments)
        else:
            return min_increments + increments_rel * (1.0 - states - min_increments)

    def absolute_to_relative_increments(
        self,
        states: TensorType["n_states", "n_dim"],
        increments_abs: TensorType["n_states", "n_dim"],
        is_backward: bool,
    ):
        """
        Returns a batch of relative increments (as sampled by the Beta distributions)
        given a batch of states, absolute increments (actions) and minimum_increments.

        Given a dimension value x, an absolute increment a, and a minimum increment m,
        then the relative increment r is given by:

        Forward:

        r = (a - m) / (1 - x - m)

        Backward:

        r = (a - m) / (x - m)
        """
        min_increments = torch.full_like(
            increments_abs, self.min_incr, dtype=self.float, device=self.device
        )
        if is_backward:
            increments_rel = (increments_abs - min_increments) / (
                states - min_increments
            )
            # Add epsilon to numerator and denominator if values are unbounded
            if not torch.all(torch.isfinite(increments_rel)):
                increments_rel = (increments_abs - min_increments + 1e-9) / (
                    states - min_increments + 1e-9
                )
            return increments_rel
        else:
            return (increments_abs - min_increments) / (1.0 - states - min_increments)

    @staticmethod
    def _get_beta_params_from_mean_variance(
        mean: TensorType["n_states", "n_dim_x_n_comp"],
        variance: TensorType["n_states", "n_dim_x_n_comp"],
    ) -> Tuple[
        TensorType["n_states", "n_dim_x_n_comp"],
        TensorType["n_states", "n_dim_x_n_comp"],
    ]:
        """
        Calculates the alpha and beta parameters of a Beta distribution from the mean
        and variance.

        The method operates on tensors containing a batch of means and variances.

        Args
        ----
        mean : tensor
            A batch of means.

        variance : tensor
            A batch of variances.

        Returns
        -------
        alpha : tensor
            The alpha parameters for the Beta distributions as a function of the mean
            and variances.

        beta : tensor
            The beta parameters for the Beta distributions as a function of the mean
            and variances.
        """
        one_minus_mean = 1.0 - mean
        beta = one_minus_mean * (mean * one_minus_mean - variance) / variance
        alpha = (mean * beta) / one_minus_mean
        return alpha, beta

    def _make_increments_distribution(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
    ) -> MixtureSameFamily:
        mix_logits = self._get_policy_betas_weights(policy_outputs).reshape(
            -1, self.n_dim, self.n_comp
        )
        mix = Categorical(logits=mix_logits)
        alphas = self._get_policy_betas_alpha(policy_outputs).reshape(
            -1, self.n_dim, self.n_comp
        )
        alphas = self.beta_params_max * torch.sigmoid(alphas) + self.beta_params_min
        betas = self._get_policy_betas_beta(policy_outputs).reshape(
            -1, self.n_dim, self.n_comp
        )
        betas = self.beta_params_max * torch.sigmoid(betas) + self.beta_params_min
        beta_distr = Beta(alphas, betas)
        return MixtureSameFamily(mix, beta_distr)

    def _mask_ignored_dimensions(
        self,
        mask: TensorType["n_states", "policy_outputs_dim"],
        tensor_to_mask: TensorType["n_states", "n_dim"],
    ) -> MixtureSameFamily:
        """
        Makes the actions, logprobs or log jacobian entries of ignored dimensions zero.

        Since the shape of all the tensor of actions, the logprobs of increments and
        the log of the diagonal of the Jacobian must be the same, this method makes no
        distiction between for simplicity.

        Args
        ----
        mask : tensor
            Boolean mask indicating (True) which dimensions should be set to zero.

        tensor_to_mask : tensor
            Tensor to be modified. It may be a tensor of actions, of logprobs of
            increments or the log of the diagonal of the Jacobian.
        """
        is_ignored_dim = mask[:, -self.n_dim :]
        if torch.any(is_ignored_dim):
            shape_orig = tensor_to_mask.shape
            tensor_to_mask[is_ignored_dim] = 0.0
            tensor_to_mask = tensor_to_mask.reshape(shape_orig)
        return tensor_to_mask

    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "mask_dim"]] = None,
        states_from: List = None,
        is_backward: Optional[bool] = False,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs.
        """
        if not is_backward:
            return self._sample_actions_batch_forward(
                policy_outputs, mask, states_from, sampling_method, temperature_logits
            )
        else:
            return self._sample_actions_batch_backward(
                policy_outputs, mask, states_from, sampling_method, temperature_logits
            )

    def _sample_actions_batch_forward(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "mask_dim"]] = None,
        states_from: List = None,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a a batch of forward actions from a batch of policy outputs.

        An action indicates, for each dimension, the absolute increment of the
        dimension value. However, in order to ensure that trajectories have finite
        length, increments must have a minumum increment (self.min_incr) except if the
        originating state is the source state (special case, see
        get_mask_invalid_actions_forward()). Furthermore, absolute increments must also
        be smaller than the distance from the dimension value to the edge of the cube
        (1.0). In order to accomodate these constraints, first relative
        increments (in [0, 1]) are sampled from a (mixture of) Beta distribution(s),
        where 0.0 indicates an absolute increment of min_incr and 1.0 indicates an
        absolute increment of 1 - x + min_incr (going to the edge).

        Therefore, given a dimension value x, a relative increment r, a minimum
        increment m and a maximum value 1, the absolute increment a is given by:

        a = m + r * (1 - x - m)

        The continuous distribution to sample the continuous action described above
        must be mixed with the discrete distribution to model the sampling of the EOS
        action. The EOS action can be sampled from any state except from the source
        state or whether the trajectory is done. That the EOS action is invalid is
        indicated by mask[-1] being False.

        Finally, regarding the constraints on the increments, the following special
        cases are taken into account:

        - The originating state is the source state: in this case, the minimum
          increment is 0.0 instead of self.min_incr. This is to ensure that the entire
          state space can be reached. This is indicated by mask[-2] being False.
        - The value at any dimension is at a distance from the cube edge smaller than the
          minimum increment (x > 1 - m). In this case, only EOS is valid.
          This is indicated by mask[0] being True (continuous actions are invalid).
        """
        # Initialize variables
        n_states = policy_outputs.shape[0]
        states_from_tensor = tfloat(
            states_from, float_type=self.float, device=self.device
        )
        is_eos = torch.zeros(n_states, dtype=torch.bool, device=self.device)
        # Determine source states
        is_source = ~mask[:, 1]
        # EOS is the only possible action if continuous actions are invalid (mask[0] is
        # True)
        is_eos_forced = mask[:, 0]
        is_eos[is_eos_forced] = True
        # Ensure that is_eos_forced does not include any source state
        assert not torch.any(torch.logical_and(is_source, is_eos_forced))
        # Sample EOS from Bernoulli distribution
        do_eos = torch.logical_and(~is_source, ~is_eos_forced)
        if torch.any(do_eos):
            is_eos_sampled = torch.zeros_like(do_eos)
            logits_eos = self._get_policy_eos_logit(policy_outputs)[do_eos]
            distr_eos = Bernoulli(logits=logits_eos)
            is_eos_sampled[do_eos] = tbool(distr_eos.sample(), device=self.device)
            is_eos[is_eos_sampled] = True
        # Sample (relative) increments if EOS is not the (sampled or forced) action
        do_increments = ~is_eos
        if torch.any(do_increments):
            if sampling_method == "uniform":
                raise NotImplementedError()
            elif sampling_method == "policy":
                distr_increments = self._make_increments_distribution(
                    policy_outputs[do_increments]
                )
            # Shape of increments: [n_do_increments, n_dim]
            increments = distr_increments.sample()
            # Compute absolute increments from sampled relative increments if state is
            # not source
            is_relative = ~is_source[do_increments]
            states_from_rel = tfloat(
                states_from_tensor[do_increments],
                float_type=self.float,
                device=self.device,
            )[is_relative]
            increments[is_relative] = self.relative_to_absolute_increments(
                states_from_rel,
                increments[is_relative],
                is_backward=False,
            )
        # Build actions
        actions_tensor = torch.full(
            (n_states, self.n_dim + 1), torch.inf, dtype=self.float, device=self.device
        )
        if torch.any(do_increments):
            # Make increments of ignored dimensions zero
            increments = self._mask_ignored_dimensions(mask[do_increments], increments)
            # Add dimension is_source and add to actions tensor
            actions_tensor[do_increments] = torch.cat(
                (increments, torch.zeros((increments.shape[0], 1))), dim=1
            )
        actions_tensor[is_source, -1] = 1
        actions = [tuple(a.tolist()) for a in actions_tensor]
        return actions, None

    def _sample_actions_batch_backward(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "mask_dim"]] = None,
        states_from: List = None,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a a batch of backward actions from a batch of policy outputs.

        An action indicates, for each dimension, the absolute increment of the
        dimension value. However, in order to ensure that trajectories have finite
        length, increments must have a minumum increment (self.min_incr). Furthermore,
        absolute increments must also be smaller than the distance from the dimension
        value to the edge of the cube. In order to accomodate these constraints, first
        relative increments (in [0, 1]) are sampled from a (mixture of) Beta
        distribution(s), where 0.0 indicates an absolute increment of min_incr and 1.0
        indicates an absolute increment of x (going back to the source).

        Therefore, given a dimension value x, a relative increment r, a minimum
        increment m and a maximum value 1, the absolute increment a is given by:

        a = m + r * (x - m)

        The continuous distribution to sample the continuous action described above
        must be mixed with the discrete distribution to model the sampling of the back
        to source (BTS) action. While the BTS action is also a continuous action, it
        needs to be modelled with a (discrete) Bernoulli distribution in order to
        ensure that this action has positive likelihood.

        Finally, regarding the constraints on the increments, the special case where
        the trajectory is done and the only possible action is EOS, is also taken into
        account.
        """
        # Initialize variables
        n_states = policy_outputs.shape[0]
        is_bts = torch.zeros(n_states, dtype=torch.bool, device=self.device)
        # EOS is the only possible action only if done is True (mask[2] is False)
        is_eos = ~mask[:, 2]
        # Back-to-source (BTS) is the only possible action if mask[1] is False
        is_bts_forced = ~mask[:, 1]
        is_bts[is_bts_forced] = True
        # Sample BTS from Bernoulli distribution
        do_bts = torch.logical_and(~is_bts_forced, ~is_eos)
        if torch.any(do_bts):
            is_bts_sampled = torch.zeros_like(do_bts)
            logits_bts = self._get_policy_source_logit(policy_outputs)[do_bts]
            distr_bts = Bernoulli(logits=logits_bts)
            is_bts_sampled[do_bts] = tbool(distr_bts.sample(), device=self.device)
            is_bts[is_bts_sampled] = True
        # Sample relative increments if actions are neither BTS nor EOS
        do_increments = torch.logical_and(~is_bts, ~is_eos)
        if torch.any(do_increments):
            if sampling_method == "uniform":
                raise NotImplementedError()
            elif sampling_method == "policy":
                distr_increments = self._make_increments_distribution(
                    policy_outputs[do_increments]
                )
            # Shape of increments_rel: [n_do_increments, n_dim]
            increments = distr_increments.sample()
            # Compute absolute increments from all sampled relative increments
            states_from_rel = tfloat(
                states_from, float_type=self.float, device=self.device
            )[do_increments]
            increments = self.relative_to_absolute_increments(
                states_from_rel,
                increments,
                is_backward=True,
            )
        # Build actions
        actions_tensor = torch.zeros(
            (n_states, self.n_dim + 1), dtype=self.float, device=self.device
        )
        actions_tensor[is_eos] = tfloat(
            self.eos, float_type=self.float, device=self.device
        )
        if torch.any(do_increments):
            # Make increments of ignored dimensions zero
            increments = self._mask_ignored_dimensions(mask[do_increments], increments)
            # Add dimension is_source and add to actions tensor
            actions_tensor[do_increments] = torch.cat(
                (increments, torch.zeros((increments.shape[0], 1))), dim=1
            )
        if torch.any(is_bts):
            # BTS actions are equal to the originating states
            actions_bts = tfloat(
                states_from, float_type=self.float, device=self.device
            )[is_bts]
            actions_bts = torch.cat(
                (actions_bts, torch.ones((actions_bts.shape[0], 1))), dim=1
            )
            actions_tensor[is_bts] = actions_bts
            # Make ignored dimensions zero
            actions_tensor[is_bts, :-1] = self._mask_ignored_dimensions(
                mask[is_bts], actions_tensor[is_bts, :-1]
            )
        actions = [tuple(a.tolist()) for a in actions_tensor]
        return actions, None

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: TensorType["n_states", "actions_dim"],
        mask: TensorType["n_states", "mask_dim"],
        states_from: List,
        is_backward: bool,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.

        Args
        ----
        policy_outputs : tensor
            The output of the GFlowNet policy model.

        mask : tensor
            The mask containing information about invalid actions and special cases.

        actions : tensor
            The actions (absolute increments) from each state in the batch for which to
            compute the log probability.

        states_from : tensor
            The states originating the actions, in GFlowNet format. They are required
            so as to compute the relative increments and the Jacobian.

        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default). Required, since the computation for forward and backward actions
            is different.
        """
        if is_backward:
            return self._get_logprobs_backward(
                policy_outputs, actions, mask, states_from
            )
        else:
            return self._get_logprobs_forward(
                policy_outputs, actions, mask, states_from
            )

    def _get_logprobs_forward(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: TensorType["n_states", "actions_dim"],
        mask: TensorType["n_states", "3"],
        states_from: List,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of forward actions.
        """
        # Initialize variables
        n_states = policy_outputs.shape[0]
        states_from_tensor = tfloat(
            states_from, float_type=self.float, device=self.device
        )
        is_eos = torch.zeros(n_states, dtype=torch.bool, device=self.device)
        logprobs_eos = torch.zeros(n_states, dtype=self.float, device=self.device)
        logprobs_increments_rel = torch.zeros(
            (n_states, self.n_dim), dtype=self.float, device=self.device
        )
        log_jacobian_diag = torch.zeros(
            (n_states, self.n_dim), device=self.device, dtype=self.float
        )
        eos_tensor = tfloat(self.eos, float_type=self.float, device=self.device)
        # Determine source states
        is_source = ~mask[:, 1]
        # EOS is the only possible action if continuous actions are invalid (mask[0] is
        # True)
        is_eos_forced = mask[:, 0]
        is_eos[is_eos_forced] = True
        # Ensure that is_eos_forced does not include any source state
        assert not torch.any(torch.logical_and(is_source, is_eos_forced))
        # Get sampled EOS actions and get log probs from Bernoulli distribution
        do_eos = torch.logical_and(~is_source, ~is_eos_forced)
        if torch.any(do_eos):
            is_eos_sampled = torch.zeros_like(do_eos)
            is_eos_sampled[do_eos] = torch.all(actions[do_eos] == eos_tensor, dim=1)
            is_eos[is_eos_sampled] = True
            logits_eos = self._get_policy_eos_logit(policy_outputs)[do_eos]
            distr_eos = Bernoulli(logits=logits_eos)
            logprobs_eos[do_eos] = distr_eos.log_prob(
                is_eos_sampled[do_eos].to(self.float)
            )
        # Get log probs of relative increments if EOS was not the sampled or forced
        # action
        do_increments = ~is_eos
        if torch.any(do_increments):
            # Get absolute increments
            increments = actions[do_increments, :-1]
            # Make sure increments are finite
            assert torch.any(torch.isfinite(increments))
            # Compute relative increments from absolute increments if state is not
            # source
            is_relative = ~is_source[do_increments]
            if torch.any(is_relative):
                states_from_rel = tfloat(
                    states_from_tensor[do_increments],
                    float_type=self.float,
                    device=self.device,
                )[is_relative]
                increments[is_relative] = self.absolute_to_relative_increments(
                    states_from_rel,
                    increments[is_relative],
                    is_backward=False,
                )
            # Compute diagonal of the Jacobian (see _get_jacobian_diag()) if state is
            # not source
            is_relative = torch.logical_and(do_increments, ~is_source)
            if torch.any(is_relative):
                log_jacobian_diag[is_relative] = torch.log(
                    self._get_jacobian_diag(
                        states_from_rel,
                        is_backward=False,
                    )
                )
            # Make ignored dimensions zero
            log_jacobian_diag = self._mask_ignored_dimensions(mask, log_jacobian_diag)
            # Get logprobs
            distr_increments = self._make_increments_distribution(
                policy_outputs[do_increments]
            )
            # Clamp because increments of 0.0 or 1.0 would yield nan
            logprobs_increments_rel[do_increments] = distr_increments.log_prob(
                torch.clamp(increments, min=self.epsilon, max=(1 - self.epsilon))
            )
            # Make ignored dimensions zero
            logprobs_increments_rel = self._mask_ignored_dimensions(
                mask, logprobs_increments_rel
            )
        # Sum log Jacobian across dimensions
        log_det_jacobian = torch.sum(log_jacobian_diag, dim=1)
        # Compute combined probabilities
        sumlogprobs_increments = logprobs_increments_rel.sum(axis=1)
        logprobs = logprobs_eos + sumlogprobs_increments + log_det_jacobian
        return logprobs

    def _get_logprobs_backward(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: TensorType["n_states", "actions_dim"],
        mask: TensorType["n_states", "3"],
        states_from: List,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of backward actions.
        """
        # Initialize variables
        n_states = policy_outputs.shape[0]
        states_from_tensor = tfloat(
            states_from, float_type=self.float, device=self.device
        )
        is_bts = torch.zeros(n_states, dtype=torch.bool, device=self.device)
        logprobs_bts = torch.zeros(n_states, dtype=self.float, device=self.device)
        logprobs_increments_rel = torch.zeros(
            (n_states, self.n_dim), dtype=self.float, device=self.device
        )
        log_jacobian_diag = torch.zeros(
            (n_states, self.n_dim), device=self.device, dtype=self.float
        )
        # EOS is the only possible action only if done is True (mask[2] is False)
        is_eos = ~mask[:, 2]
        # Back-to-source (BTS) is the only possible action if mask[1] is False
        is_bts_forced = ~mask[:, 1]
        is_bts[is_bts_forced] = True
        # Get sampled BTS actions and get log probs from Bernoulli distribution
        do_bts = torch.logical_and(~is_bts_forced, ~is_eos)
        if torch.any(do_bts):
            # BTS actions are equal to the originating states
            is_bts_sampled = torch.zeros_like(do_bts)
            is_bts_sampled[do_bts] = torch.all(
                actions[do_bts, :-1] == states_from_tensor[do_bts], dim=1
            )
            is_bts[is_bts_sampled] = True
            logits_bts = self._get_policy_source_logit(policy_outputs)[do_bts]
            distr_bts = Bernoulli(logits=logits_bts)
            logprobs_bts[do_bts] = distr_bts.log_prob(
                is_bts_sampled[do_bts].to(self.float)
            )
        # Get log probs of relative increments if actions were neither BTS nor EOS
        do_increments = torch.logical_and(~is_bts, ~is_eos)
        if torch.any(do_increments):
            # Get absolute increments
            increments = actions[do_increments, :-1]
            # Compute absolute increments from all sampled relative increments
            increments = self.absolute_to_relative_increments(
                states_from_tensor[do_increments],
                increments,
                is_backward=True,
            )
            # Make sure increments are finite
            assert torch.all(torch.isfinite(increments))
            # Compute diagonal of the Jacobian (see _get_jacobian_diag())
            log_jacobian_diag[do_increments] = torch.log(
                self._get_jacobian_diag(
                    states_from_tensor[do_increments],
                    is_backward=True,
                )
            )
            # Make ignored dimensions zero
            log_jacobian_diag = self._mask_ignored_dimensions(mask, log_jacobian_diag)
            # Get logprobs
            distr_increments = self._make_increments_distribution(
                policy_outputs[do_increments]
            )
            # Clamp because increments of 0.0 or 1.0 would yield nan
            logprobs_increments_rel[do_increments] = distr_increments.log_prob(
                torch.clamp(increments, min=self.epsilon, max=(1 - self.epsilon))
            )
            # Make ignored dimensions zero
            logprobs_increments_rel = self._mask_ignored_dimensions(
                mask, logprobs_increments_rel
            )
        # Sum log Jacobian across dimensions
        log_det_jacobian = torch.sum(log_jacobian_diag, dim=1)
        # Compute combined probabilities
        sumlogprobs_increments = logprobs_increments_rel.sum(axis=1)
        logprobs = logprobs_bts + sumlogprobs_increments + log_det_jacobian
        # Ensure that logprobs of forced EOS are 0
        logprobs[is_eos] = 0.0
        return logprobs

    def _get_jacobian_diag(
        self,
        states_from: TensorType["n_states", "n_dim"],
        is_backward: bool,
    ):
        """
        Computes the diagonal of the Jacobian of the sampled actions with respect to
        the target states.

        Forward: the sampled variables are the relative increments r_f and the state
        updates (s -> s') are:

        s' = s + m + r_f(1 - s - m)
        r_f = (s' - s - m) / (1 - s - m)

        Therefore, the derivative of r_f wrt to s' is

        dr_f/ds' = 1 / (1 - s - m)

        Backward: the sampled variables are the relative decrements r_b and the state
        updates (s' -> s) are:

        s = s' - m - r_b(s' - m)
        r_b = (s' - s - m) / (s' - m)

        Therefore, the derivative of r_b wrt to s is

        dr_b/ds = -1 / (s' - m)

        We take the absolute value of the derivative (Jacobian).

        The derivatives of the components of r with respect to dimensions of s or s'
        other than itself are zero. Therefore, the Jacobian is diagonal and the
        determinant is the product of the diagonal.
        """
        min_increments = torch.full_like(
            states_from, self.min_incr, dtype=self.float, device=self.device
        )
        if is_backward:
            return 1.0 / ((states_from - min_increments))
        else:
            return 1.0 / ((1.0 - states_from - min_increments))

    def _step(
        self,
        action: Tuple[float],
        backward: bool,
    ) -> Tuple[List[float], Tuple[float], bool]:
        """
        Updates self.state given a non-EOS action. This method is called by both step()
        and step_backwards(), with the corresponding value of argument backward.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple of length n_dim, with the
            absolute increment for each dimension.

        backward : bool
            If True, perform backward step. Otherwise (default), perform forward step.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action executed

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        # If forward action is from source, initialize state to all zeros.
        if not backward and action[-1] == 1 and self.state == self.source:
            state = [0.0 for _ in range(self.n_dim)]
        else:
            assert action[-1] == 0
            state = copy(self.state)
        # Increment dimensions
        for dim, incr in enumerate(action[:-1]):
            if backward:
                state[dim] -= incr
            else:
                state[dim] += incr

        # If state is out of bounds, return invalid
        effective_dims = self._get_effective_dims(state)
        if any([s > 1.0 for s in effective_dims]) or any(
            [s < 0.0 for s in effective_dims]
        ):
            warnings.warn(
                f"""
                State is out of cube bounds.
                \nCurrent state:\n{self.state}\nAction:\n{action}\nNext state: {state}
                """
            )
            return self.state, action, False

        # Otherwise, set self.state as the udpated state and return valid.
        self.n_actions += 1
        self.state = state
        return self.state, action, True

    # TODO: make generic for continuous environments?
    def step(self, action: Tuple[float]) -> Tuple[List[float], Tuple[int, float], bool]:
        """
        Executes step given an action. An action is the absolute increment of each
        dimension.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple of length n_dim, with the
            absolute increment for each dimension.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action executed

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        if self.done:
            return self.state, action, False
        if action == self.eos:
            assert self._get_effective_dims(self.state) != self._get_effective_dims(
                self.source
            )
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # Otherwise perform action
        else:
            return self._step(action, backward=False)

    def step_backwards(
        self, action: Tuple[int, float]
    ) -> Tuple[List[float], Tuple[int, float], bool]:
        """
        Executes backward step given an action. An action is the absolute decrement of
        each dimension.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple of length n_dim, with the
            absolute decrement for each dimension.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action executed

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        # If done is True, set done to False, increment n_actions and return same state
        if self.done:
            assert action == self.eos
            self.done = False
            self.n_actions += 1
            return self.state, action, True
        assert action != self.eos
        # If action is BTS, set source state
        if action[-1] == 1 and self.state != self.source:
            self.n_actions += 1
            self.state = self.source
            return self.state, action, True
        # Otherwise perform action
        return self._step(action, backward=True)

    def action2representative(self, action: Tuple) -> Tuple:
        """
        Replaces the continuous values of an action by 0s (the "generic" or
        "representative" action in the first position of the action space), so that
        they can be compared against the action space or a mask.
        """
        return self.action_space[0]

    def get_grid_terminating_states(
        self, n_states: int, kappa: Optional[float] = None
    ) -> List[List]:
        """
        Constructs a grid of terminating states within the range of the hyper-cube.

        Args
        ----
        n_states : int
            Requested number of states. The actual number of states will be rounded up
            such that all dimensions have the same number of states.

        kappa : float
            Small constant indicating the distance to the theoretical limits of the
            cube [0, 1], in order to avoid innacuracies in the computation of the log
            probabilities due to clamping. The grid will thus be in [kappa, 1 -
            kappa]. If None, self.kappa will be used.
        """
        if kappa is None:
            kappa = self.kappa
        n_per_dim = int(np.ceil(n_states ** (1 / self.n_dim)))
        linspaces = [
            np.linspace(kappa, 1.0 - kappa, n_per_dim) for _ in range(self.n_dim)
        ]
        states = list(itertools.product(*linspaces))
        states = [list(el) for el in states]
        return states

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None, kappa: Optional[float] = None
    ) -> List[List]:
        """
        Constructs a set of terminating states sampled uniformly within the range of
        the hyper-cube.

        Args
        ----
        n_states : int
            Number of states in the returned list.

        kappa : float
            Small constant indicating the distance to the theoretical limits of the
            cube [0, 1], in order to avoid innacuracies in the computation of the log
            probabilities due to clamping. The states will thus be uniformly sampled in
            [kappa, 1 - kappa]. If None, self.kappa will be used.
        """
        if kappa is None:
            kappa = self.kappa
        rng = np.random.default_rng(seed)
        states = rng.uniform(low=kappa, high=1.0 - kappa, size=(n_states, self.n_dim))
        return states.tolist()

    def fit_kde(
        self,
        samples: TensorType["batch_size", "state_proxy_dim"],
        kernel: str = "gaussian",
        bandwidth: float = 0.1,
    ):
        r"""
        Fits a Kernel Density Estimator on a batch of samples.

        Parameters
        ----------
        samples : tensor
            A batch of samples in proxy format.
        kernel : str
            An identifier of the kernel to use for the density estimation. It must be a
            valid kernel for the scikit-learn method
            :py:meth:`sklearn.neighbors.KernelDensity`.
        bandwidth : float
            The bandwidth of the kernel.
        """
        samples = torch2np(samples)
        return KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(samples)

    def plot_reward_samples(
        self,
        samples: TensorType["batch_size", "state_proxy_dim"],
        samples_reward: TensorType["batch_size", "state_proxy_dim"],
        rewards: TensorType["batch_size"],
        alpha: float = 0.5,
        dpi: int = 150,
        max_samples: int = 500,
        **kwargs,
    ):
        """
        Plots the reward contour alongside a batch of samples.

        Parameters
        ----------
        samples : tensor
            A batch of samples from the GFlowNet policy in proxy format. These samples
            will be plotted on top of the reward density.
        samples_reward : tensor
            A batch of samples containing a grid over the sample space, from which the
            reward has been obtained. These samples are used to plot the contour of
            reward density.
        rewards : tensor
            The rewards of samples_reward. It should be a vector of dimensionality
            n_per_dim ** 2 and be sorted such that the each block at rewards[i *
            n_per_dim:i * n_per_dim + n_per_dim] correspond to the rewards at the i-th
            row of the grid of samples, from top to bottom. The same is assumed for
            samples_reward.
        alpha : float
            Transparency of the reward contour.
        dpi : int
            Dots per inch, indicating the resolution of the plot.
        max_samples : int
            Maximum of number of samples to include in the plot.
        """
        if self.n_dim != 2:
            return None
        samples = torch2np(samples)
        samples_reward = torch2np(samples_reward)
        rewards = torch2np(rewards)
        # Create mesh grid from samples_reward
        n_per_dim = int(np.sqrt(samples_reward.shape[0]))
        assert n_per_dim**2 == samples_reward.shape[0]
        x_coords = samples_reward[:, 0].reshape((n_per_dim, n_per_dim))
        y_coords = samples_reward[:, 1].reshape((n_per_dim, n_per_dim))
        rewards = rewards.reshape((n_per_dim, n_per_dim))
        # Init figure
        fig, ax = plt.subplots()
        fig.set_dpi(dpi)
        # Plot reward contour
        h = ax.contourf(x_coords, y_coords, rewards, alpha=alpha)
        ax.axis("scaled")
        fig.colorbar(h, ax=ax)
        # Plot samples
        random_indices = np.random.permutation(samples.shape[0])[:max_samples]
        ax.scatter(samples[random_indices, 0], samples[random_indices, 1], alpha=alpha)
        # Figure settings
        ax.grid()
        padding = 0.05 * (CELL_MAX - CELL_MIN)
        ax.set_xlim([CELL_MIN - padding, CELL_MAX + padding])
        ax.set_ylim([CELL_MIN - padding, CELL_MAX + padding])
        plt.tight_layout()
        return fig

    def plot_kde(
        self,
        samples: TensorType["batch_size", "state_proxy_dim"],
        kde,
        alpha: float = 0.5,
        dpi=150,
        colorbar: bool = True,
        **kwargs,
    ):
        """
        Plots the density previously estimated from a batch of samples via KDE over the
        entire sample space.

        Parameters
        ----------
        samples : tensor
            A batch of samples containing a grid over the sample space. These samples
            are used to plot the contour of the estimated density.
        kde : KDE
            A scikit-learn KDE object fit with a batch of samples.
        alpha : float
            Transparency of the density contour.
        dpi : int
            Dots per inch, indicating the resolution of the plot.
        """
        if self.n_dim != 2:
            return None
        samples = torch2np(samples)
        # Create mesh grid from samples
        n_per_dim = int(np.sqrt(samples.shape[0]))
        assert n_per_dim**2 == samples.shape[0]
        x_coords = samples[:, 0].reshape((n_per_dim, n_per_dim))
        y_coords = samples[:, 1].reshape((n_per_dim, n_per_dim))
        # Score samples with KDE
        Z = np.exp(kde.score_samples(samples)).reshape((n_per_dim, n_per_dim))
        # Init figure
        fig, ax = plt.subplots()
        fig.set_dpi(dpi)
        # Plot KDE
        h = ax.contourf(x_coords, y_coords, Z, alpha=alpha)
        ax.axis("scaled")
        if colorbar:
            fig.colorbar(h, ax=ax)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Set tight layout
        plt.tight_layout()
        return fig
