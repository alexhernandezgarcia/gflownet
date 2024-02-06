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
from gflownet.utils.common import copy, tbool, tfloat


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
        # Conversions: only conversions to policy are implemented and the conversion to
        # proxy format is the same
        self.states2proxy = self.states2policy
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

    def states2policy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "policy_input_dim"]:
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

    def _get_policy_dim_eos_logit(
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
            return (increments_abs - min_increments) / (states - min_increments)
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
            logits_eos = self._get_policy_dim_eos_logit(policy_outputs)[do_eos]
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
            logits_eos = self._get_policy_dim_eos_logit(policy_outputs)[do_eos]
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
            # Make sure increments are finite
            assert torch.any(torch.isfinite(increments))
            # Compute absolute increments from all sampled relative increments
            increments = self.absolute_to_relative_increments(
                states_from_tensor[do_increments],
                increments,
                is_backward=True,
            )
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

            self.source
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

    # TODO: make generic for all environments
    def sample_from_reward(
        self, n_samples: int, epsilon=1e-4
    ) -> TensorType["n_samples", "state_dim"]:
        """
        Rejection sampling with proposal the uniform distribution in [0, 1]^n_dim.

        Returns a tensor in GFloNet (state) format.
        """
        samples_final = []
        max_reward = self.proxy2reward(self.proxy.min)
        while len(samples_final) < n_samples:
            samples_uniform = self.states2proxy(
                self.get_uniform_terminating_states(n_samples)
            )
            rewards = self.proxy2reward(self.proxy(samples_uniform))
            mask = (
                torch.rand(n_samples, dtype=self.float, device=self.device)
                * (max_reward + epsilon)
                < rewards
            )
            samples_accepted = samples_uniform[mask]
            samples_final.extend(samples_accepted[-(n_samples - len(samples_final)) :])
        return torch.vstack(samples_final)

    # TODO: make generic for all envs
    def fit_kde(self, samples, kernel="gaussian", bandwidth=0.1):
        return KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(samples)

    def plot_reward_samples(
        self,
        samples,
        alpha=0.5,
        cell_min=-1.0,
        cell_max=1.0,
        dpi=150,
        max_samples=500,
        **kwargs,
    ):
        if self.n_dim != 2:
            return None
        # Sample a grid of points in the state space and obtain the rewards
        x = np.linspace(cell_min, cell_max, 201)
        y = np.linspace(cell_min, cell_max, 201)
        xx, yy = np.meshgrid(x, y)
        X = np.stack([xx, yy], axis=-1)
        states_mesh = torch.tensor(
            X.reshape(-1, 2), device=self.device, dtype=self.float
        )
        rewards = self.proxy2reward(self.proxy(states_mesh))
        # Init figure
        fig, ax = plt.subplots()
        fig.set_dpi(dpi)
        # Plot reward contour
        h = ax.contourf(xx, yy, rewards.reshape(xx.shape).cpu().numpy(), alpha=alpha)
        ax.axis("scaled")
        fig.colorbar(h, ax=ax)
        # Plot samples
        random_indices = np.random.permutation(samples.shape[0])[:max_samples]
        ax.scatter(samples[random_indices, 0], samples[random_indices, 1], alpha=alpha)
        # Figure settings
        ax.grid()
        padding = 0.05 * (cell_max - cell_min)
        ax.set_xlim([cell_min - padding, cell_max + padding])
        ax.set_ylim([cell_min - padding, cell_max + padding])
        plt.tight_layout()
        return fig

    # TODO: make generic for all envs
    def plot_kde(
        self,
        kde,
        alpha=0.5,
        cell_min=-1.0,
        cell_max=1.0,
        dpi=150,
        colorbar=True,
        **kwargs,
    ):
        if self.n_dim != 2:
            return None
        # Sample a grid of points in the state space and score them with the KDE
        x = np.linspace(cell_min, cell_max, 201)
        y = np.linspace(cell_min, cell_max, 201)
        xx, yy = np.meshgrid(x, y)
        X = np.stack([xx, yy], axis=-1)
        Z = np.exp(kde.score_samples(X.reshape(-1, 2))).reshape(xx.shape)
        # Init figure
        fig, ax = plt.subplots()
        fig.set_dpi(dpi)
        # Plot KDE
        h = ax.contourf(xx, yy, Z, alpha=alpha)
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


class ContinuousCubeSingleDimIncrement(CubeBase):
    """
    Continuous hyper-cube environment (continuous version of a hyper-grid).
    This class works in a similar way to the ContinuousCube class but the difference
    lies in that, for ContinuousCube, all dimensions are incremented at once. In this
    version of the class, only one dimension may be increment at a time.
    The state space is the value of each dimension. The action space contains actions
    to pick a dimension to implement, actions to implement that selected dimension,
    and actions to deselect the currently selected action to allow the incrementation
    of other dimensions.

    In order to ensure that all trajectories are of finite length, actions to increment
    a dimension have a minimum increment determined by min_incr. If the value of a
    dimension is larger than 1 - min_incr, then that dimension can't be further
    selected or incremented. In order to ensure the coverage of the state space, the
    first incrementation (from the source state) of a dimension is not constrained
    by the minimum increment.

    WARNING : This paragraph is copy-pasted from the ContinuousCube class but I don't
    think that it is valid, either here or in the ContinuousCube class.
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

    NO_DIMENSION_SELECTED = -1

    ACTION_TYPE_SELECT_DIM = 0
    ACTION_TYPE_INCREMENT = 1
    ACTION_TYPE_DESELECT_DIM = 2

    MASK_IDX_SELECT_DIM = 0
    MASK_IDX_INCREMENT = 1
    MASK_IDX_DESELECT_DIM = 2
    MASK_IDX_TO_FROM_SOURCE = 3
    MASK_IDX_EOS = 4
    MASK_IDX_DIMS = 5

    STATE_IDX_NEXT_ACTION_TYPE = -2
    STATE_IDX_SELECTED_DIM = -1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Mask dimensionality: 5 + number of dimensions
        self.mask_dim_base = 5
        self.mask_dim = self.mask_dim_base + self.n_dim

        # Overwrite base source state (which is -1 for each of the cube dimension) to
        # include new source values for the next action type and the selected dimension
        self.source += [self.ACTION_TYPE_SELECT_DIM, self.NO_DIMENSION_SELECTED]
        self.state = copy(self.source)

    def get_action_space(self):
        """
        The action space is hybrid since it contains both discrete and continuous
        actions.

        The actions are tuples of size 3 :
        - the value at position 0 indicates the category of action. This can be either
          ACTION_TYPE_SELECT_DIM, ACTION_TYPE_INCREMENT, or ACTION_TYPE_DESELECT_DIM.
        - the value at position 1 indicates the value of the action. In the case of an
          ACTION_TYPE_SELECT_DIM or ACTION_TYPE_DESELECT_DIM action, this is the index
          of the dimension to select/deselect. In the case of an ACTION_TYPE_INCREMENT
          action, this is the absolute value of the increment.
        - the value at position 2 indicates whether the action is from or to the source
          (if == 1) or otherwise (if == 0)

        EOS is indicated by np.inf for all dimensions.

        This method defines self.eos and the returned action space is simply
        a representative (arbitrary) action with an increment of 0.0 in all dimensions,
        and EOS.
        """
        actions_dim = 3
        self.eos = tuple([np.inf] * actions_dim)
        self.representative_action = tuple([0.0] * actions_dim)
        return [self.representative_action, self.eos]

    def unpack_action(self, action: tuple) -> tuple:
        """
        Unpacks an action tuple, ensuring that the elements have the correct dtype
        """
        action_type = int(action[0])
        action_value = action[1]
        action_is_to_from_source = bool(action[2])
        if action_type != self.ACTION_TYPE_INCREMENT:
            action_value = int(action_value)

        return action_type, action_value, action_is_to_from_source

    def get_max_traj_length(self):
        max_number_incr_per_dim = np.ceil(1.0 / self.min_incr) + 1
        nb_actions_per_incr = 3
        return nb_actions_per_incr * max_number_incr_per_dim * self.n_dim + 1

    def get_policy_output(self, params: dict) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model.

        The policy output will be used to initialize a distribution, from which an
        action is to be determined or sampled. This method returns a vector with a
        fixed policy defined by params.

        The environment consists of both continuous and discrete actions.

        Continuous actions

        For each component c of the mixture, the output of the policy should return:
          1) the weight of the component in the mixture,
          2) the pre-alpha parameter of the Beta distribution to sample the increment,
          3) the pre-beta parameter of the Beta distribution to sample the increment.

        These parameters are the first n_comp * 3 of the policy output.

        Discrete actions

        The policy output for discrete actions contains the following :
        - One logit of a Bernoulli distribution for the (discrete) backward probability
          of returning to the source node
        - n_dim + 1 logits of a multinomial distribution to model the (discrete)
          forward probability of selecting each dimension as the one to increment, or
          of selecting the EOS action.

        Therefore, the output of the policy model has dimensionality C x 3 + D + 2,
        where D is the number of dimensions (self.n_dim) and C is the number of
        components (self.n_comp).

        See
        ---
        _beta_params_to_policy_outputs()
        """
        # Parameters for continuous actions
        self._len_policy_output_cont = self.n_comp * 3
        policy_output_cont = torch.empty(
            self._len_policy_output_cont,
            dtype=self.float,
            device=self.device,
        )
        policy_output_cont[0::3] = params["beta_weights"]
        policy_output_cont[1::3] = self._beta_params_to_policy_outputs("alpha", params)
        policy_output_cont[2::3] = self._beta_params_to_policy_outputs("beta", params)

        # Define logits of a multinomial distribution such that the EOS action
        # will have the required probability and the other dimensions that can
        # be selected will uniformly share the remaining probability.
        eos_prob = params["bernoulli_eos_prob"]

        logit_eos = tfloat(1, float_type=self.float, device=self.device)  # Arbitrary.
        sum_exp_other_logits = torch.exp(logit_eos) * (1 - eos_prob) / eos_prob
        exp_other_logits = sum_exp_other_logits / self.n_dim
        other_logits = torch.log(exp_other_logits)

        policy_output_dim_logit = tfloat(
            [other_logits] * self.n_dim + [logit_eos],
            float_type=self.float,
            device=self.device,
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
                policy_output_dim_logit,
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

    def _get_policy_dim_eos_logit(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "action_dim"]:
        """
        Reduces a given policy output to the part corresponding to the logits of the
        multinomial distribution to pick a dimension or the EOS action.

        See: get_policy_output()
        """
        return policy_output[:, self._len_policy_output_cont + 1 :]

    def _get_policy_dim_logit(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "action_dim"]:
        """
        Reduces a given policy output to the part corresponding to the logits of the
        multinomial distribution to pick a dimension.

        See: get_policy_output()
        """
        return policy_output[:, self._len_policy_output_cont + 1 : -1]

    def _get_policy_eos_logit(
        self, policy_output: TensorType["n_states", "policy_output_dim"]
    ) -> TensorType["n_states", "action_dim"]:
        """
        Reduces a given policy output to the part corresponding to the logits of the
        multinomial distribution corresponding to the EOS action.

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
        return policy_output[:, self._len_policy_output_cont]

    def _get_effective_dims(self, state: Optional[List] = None) -> List:
        state = self._get_state(state)[: self.n_dim]
        return [s for s, ign_dim in zip(state, self.ignored_dims) if not ign_dim]

    def _get_effective_dims_indices(self, state: Optional[List] = None) -> List:
        return [idx for idx, ignored in enumerate(self.ignored_dims) if not ignored]

    def _get_next_action_type(self, state: Optional[List] = None):
        return self._get_state(state)[self.STATE_IDX_NEXT_ACTION_TYPE]

    def _get_selected_dim(self, state: Optional[List] = None):
        return self._get_state(state)[self.STATE_IDX_SELECTED_DIM]

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
        - 0 : whether a discrete action to choose a dimension is invalid. False if
          is no currently selected dimension for incrementation. True otherwise.
        - 1 : whether a continuous action to choose an increment is invalid. False if
          a dimension has been selected but not yet incremented. True otherwise.
        - 2 : whether a discrete action to deselect a dimension is invalid. False if
          a dimension is selected and already incremented. True otherwise.
        - 3 : special case when a continuous action is valid and the current value of
          the dimension to increment has the same values as in the source state. True
          is there is no special case (value != source), False if there is a special
          case (value == source).
        - 4 : whether EOS action is invalid. EOS is only valid when there is no
          currently selected dimension and all dimensions are different from their
          source value
        - Next n_dim values : whether of not each of the dimensions can be
          selected/deselected in the next action.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        # If done, the entire mask is True (all actions are "invalid" and no special
        # cases)
        mask = [True] * self.mask_dim
        if done:
            return mask

        next_action_type = state[self.STATE_IDX_NEXT_ACTION_TYPE]
        selected_dimension = state[self.STATE_IDX_SELECTED_DIM]

        if next_action_type == self.ACTION_TYPE_SELECT_DIM:
            # Iterate through the effective dimensions to determine which dimensions
            # can be incremented. EOS is invalid if at least one dimension still has
            # its source value.
            mask[self.MASK_IDX_EOS] = False
            for dim_idx in range(self.n_dim):
                # Skip dimension if ignored
                if self.ignored_dims[dim_idx]:
                    continue

                # This dimension is valid only if it can still be incremented
                if state[dim_idx] <= 1 - self.min_incr:
                    mask[self.MASK_IDX_DIMS + dim_idx] = False

                # EOS is invalid if this dimension still has its source value
                if state[dim_idx] == self.source[dim_idx]:
                    mask[self.MASK_IDX_EOS] = True

            # Ensure that at least one dimension or EOS is not invalid
            nb_dims_invalid = sum(
                mask[self.MASK_IDX_DIMS : self.MASK_IDX_DIMS + self.n_dim]
            )
            assert nb_dims_invalid < self.n_dim or not mask[self.MASK_IDX_EOS]
            mask[self.MASK_IDX_SELECT_DIM] = False

        elif next_action_type == self.ACTION_TYPE_INCREMENT:
            # A dimension has already been selected for incrementation so we indicate
            # continuous actions as valid. If the selected dimension is still in its
            # source value, we indicate the special case in the mask.
            assert selected_dimension != self.NO_DIMENSION_SELECTED
            assert not self.ignored_dims[selected_dimension]
            mask[self.MASK_IDX_INCREMENT] = False
            dim_is_in_source = (
                state[selected_dimension] == self.source[selected_dimension]
            )
            mask[self.MASK_IDX_TO_FROM_SOURCE] = not dim_is_in_source

        elif next_action_type == self.ACTION_TYPE_DESELECT_DIM:
            # Mark the deselect action as valid and also indicate the currently selected
            # dimension as the only valid dimension for deselection.
            assert selected_dimension != self.NO_DIMENSION_SELECTED
            mask[self.MASK_IDX_DESELECT_DIM] = False
            mask[self.MASK_IDX_DIMS + selected_dimension] = False

        # Ensure that at least one action is valid
        assert any(mask)

        return mask

    def get_mask_invalid_actions_backward(self, state=None, done=None, parents_a=None):
        """
        The action space is hybrid, thus the mask is not only of invalid actions as
        in discrete environments, but also an indicator of "special cases", for example
        states from which only certain actions are possible.

        In order to approximately stick to the semantics in discrete environments,
        where the mask is of "invalid" actions, that is the value is True if an action
        is invalid, the mask values of special cases are True if the special cases they
        refer to are "invalid". In other words, the values are False if the state has
        the special case.

        The backward mask has the following structure:
        - 0 : whether a discrete action to choose a dimension is invalid. False if
          is a dimension has been selected but not incremented (taking this action
          backward will revert the action selection). True otherwise.
        - 1 : whether a continuous action to choose an increment is invalid. False if
          a dimension has been selected and incremented (since taking the increment
          action backward will revert the incrementation). True otherwise.
        - 2 : whether a discrete action to deselect a dimension is invalid. False if
          there is no dimension currently selected (since taking the deselect action
          backwards will make an action selected). True otherwise.
        - 3 : special case when a continuous action is valid and back-to-source is the
          only possible action. False if the selected dimension is smaller than
          min_incr, True otherwise.
        - 4 : whether EOS action is invalid. False only if done is True, True
          (invalid) otherwise.
        - Next n_dim values : whether of not each of the dimensions can be
          selected/deselected in the next action.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        mask = [True] * self.mask_dim

        # If the state is the source state, entire mask is True
        if state == self.source:
            return mask

        # If done, only valid action is EOS.
        if done:
            mask[self.MASK_IDX_EOS] = False
            return mask

        # Obtain the type of the next action and the selected dimension.
        # NOTE : This is the type of the next action when taking actions in the
        # forward direction. Valid actions in the backward directions will differ
        # from valid actions in the forward direction.
        next_action_type = state[self.STATE_IDX_NEXT_ACTION_TYPE]
        selected_dimension = state[self.STATE_IDX_SELECTED_DIM]

        if next_action_type == self.ACTION_TYPE_SELECT_DIM:
            # There is currently no selected dimension so the only allowed backward
            # action allowed (since the current state isn't the source state) is a
            # ACTION_TYPE_DESELECT_DIM action which, taken backward, will set a
            # dimension as selected. The only dimensions that can be "deselected"
            # are dimension that differ from the source state meaning that they can
            # be decremented.
            for dim_idx in range(self.n_dim):
                # Skip dimension if ignored
                if self.ignored_dims[dim_idx]:
                    continue

                # EOS is invalid if this dimension still has its source value
                if state[dim_idx] != self.source[dim_idx]:
                    mask[self.MASK_IDX_DESELECT_DIM] = False
                    mask[self.MASK_IDX_DIMS + dim_idx] = False

        elif next_action_type == self.ACTION_TYPE_INCREMENT:
            # There is currently a selected dimension that hasn't been incremented.
            # The only action allowed is a ACTION_TYPE_SELECT_DIM action which, taken
            # backward will revert the action selection. The only dimension that can
            # be "selected" is the dimension that is currently selected.
            assert selected_dimension != self.NO_DIMENSION_SELECTED
            mask[self.MASK_IDX_SELECT_DIM] = False
            mask[self.MASK_IDX_DIMS + selected_dimension] = False

        elif next_action_type == self.ACTION_TYPE_DESELECT_DIM:
            # There is currently a selected action that has been incremented.
            # The only action allowed is a ACTION_TYPE_INCREMENT action which, taken
            # backward will revert the incrementation.
            assert selected_dimension != self.NO_DIMENSION_SELECTED
            mask[self.MASK_IDX_INCREMENT] = False

            # If the currently selected dimension is smaller than the min increment,
            # then back-to-source is the only valid action
            if state[selected_dimension] < self.min_incr:
                mask[self.MASK_IDX_TO_FROM_SOURCE] = False

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
        states: TensorType["n_states"],
        increments_rel: TensorType["n_states"],
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
        states: TensorType["n_states"],
        increments_abs: TensorType["n_states"],
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
            return (increments_abs - min_increments) / (states - min_increments)
        else:
            return (increments_abs - min_increments) / (1.0 - states - min_increments)

    @staticmethod
    def _get_beta_params_from_mean_variance(
        mean: TensorType["n_states", "n_comp"],
        variance: TensorType["n_states", "n_comp"],
    ) -> Tuple[TensorType["n_states", "n_comp"], TensorType["n_states", "n_comp"],]:
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
            -1, self.n_comp
        )
        mix = Categorical(logits=mix_logits)
        alphas = self._get_policy_betas_alpha(policy_outputs).reshape(-1, self.n_comp)
        alphas = self.beta_params_max * torch.sigmoid(alphas) + self.beta_params_min
        betas = self._get_policy_betas_beta(policy_outputs).reshape(-1, self.n_comp)
        betas = self.beta_params_max * torch.sigmoid(betas) + self.beta_params_min
        beta_distr = Beta(alphas, betas)
        return MixtureSameFamily(mix, beta_distr)

    def _mask_ignored_dimensions(
        self,
        mask: TensorType["n_states", "policy_outputs_dim"],
        tensor_to_mask: TensorType["n_states", "n_dim"],
    ) -> TensorType["n_states", "n_dim"]:
        """
        Makes the actions, logprobs or log jacobian entries of ignored dimensions to 0.

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
        Samples a batch of forward actions from a batch of policy outputs.

        An action is a tuple containing three values. The first element identifies the
        type of action being performed any can be any value in {ACTION_TYPE_SELECT_DIM,
        ACTION_TYPE_INCREMENT, ACTION_TYPE_DESELECT_DIM}. The second element quantifies
        the action. In the case of an action to select/deselect a dimension, the second
        element takes the value of the relevant dimension index. In the case of an
        action to increment the selected dimension, the second elements takes the value
        of the absolute increment. The third value indicates whether the action is a
        special case from-the-source action which means that the action is of type
        ACTION_TYPE_INCREMENT and the currently selected dimension is in its source
        state. The EOS action is represented by (inf, inf, inf).

        When increment actions are valid, a continuous increment value is sampled. In
        order to ensure that trajectories have finite length, increments must have a
        minimum increment (self.min_incr) except if the originating state is the source
        state (special case, see get_mask_invalid_actions_forward()). Furthermore,
        absolute increments must also be smaller than the distance from the dimension
        value to the edge of the cube  (1.0). In order to accomodate these constraints,
        first relative increments (in [0, 1]) are sampled from a (mixture of) Beta
        distribution(s), where 0.0 indicates an absolute increment of min_incr and 1.0
        indicates an absolute increment of 1 - x + min_incr (going to the edge).

        Therefore, given a dimension value x, a relative increment r, a minimum
        increment m and a maximum value 1, the absolute increment a is given by:

        a = m + r * (1 - x - m)

        Finally, there is a special case when the originating state is equal to the
        source state in the selected dimension:
        in this case, the minimum increment is 0.0 instead of self.min_incr. This is
        to ensure that the entire state space can be reached. This is indicated by
        mask[MASK_IDX_TO_FROM_SOURCE] being False.
        """
        if sampling_method != "policy":
            raise NotImplementedError()

        # Initialize variables
        n_states = policy_outputs.shape[0]
        states_from_tensor = tfloat(
            states_from, float_type=self.float, device=self.device
        )
        selected_dims = states_from_tensor[:, self.STATE_IDX_SELECTED_DIM].int()

        is_action_select_eos = ~mask[:, self.MASK_IDX_SELECT_DIM]
        is_increment = ~mask[:, self.MASK_IDX_INCREMENT]
        is_action_deselect = ~mask[:, self.MASK_IDX_DESELECT_DIM]
        is_eos_valid = ~mask[:, self.MASK_IDX_EOS]
        is_from_source = ~mask[:, self.MASK_IDX_TO_FROM_SOURCE]
        valid_dimensions = ~mask[
            :, self.MASK_IDX_DIMS : self.MASK_IDX_DIMS + self.n_dim
        ]

        # Allocate tensor for the actions
        # The colums are : selected dimension, absolute increment, is from source
        action_tensor = torch.full(
            (n_states, 3), torch.inf, dtype=self.float, device=self.device
        )

        # Handle the select/EOS actions
        if torch.any(is_action_select_eos):
            # Determine the logit value for each possible select/EOS action. This
            # involves masking the logit values for dimensions that cannot be selected
            # according to the mask as well as masking the logit values for the EOS
            # action in the states where that action it not valid.
            discrete_actions_logits = self._get_policy_dim_eos_logit(
                policy_outputs
            ).clone()
            discrete_actions_logits[:, :-1] = torch.where(
                valid_dimensions, discrete_actions_logits[:, :-1], -torch.inf
            )
            discrete_actions_logits[~is_eos_valid, -1] = -torch.inf

            # Sample the actions
            distr_discrete = Categorical(logits=discrete_actions_logits)
            sampled_select_actions = distr_discrete.sample()

            # Determine which of the sampled actions are not EOS
            non_eos_actions = sampled_select_actions < self.n_dim

            # Build binary mask with 1 only for the actions that select a dimension
            is_action_select = is_action_select_eos.clone()
            is_action_select[is_action_select_eos] = non_eos_actions

            # Set the actions in action_tensor with the result. Set only the actions
            # that are not EOS.
            sampled_select_actions = tfloat(
                sampled_select_actions, float_type=self.float, device=self.device
            )
            action_tensor[is_action_select, 0] = self.ACTION_TYPE_SELECT_DIM
            action_tensor[is_action_select, 1] = sampled_select_actions[non_eos_actions]
            action_tensor[is_action_select, 2] = 0

        # Handle the increment actions
        if torch.any(is_increment):
            # Obtain the current value of the currently selected dimension for each
            # state
            current_state_dim_value = states_from_tensor[
                torch.arange(n_states), selected_dims
            ]

            # Sample the increment values. Shape : [n_do_increments]
            distr_increments = self._make_increments_distribution(
                policy_outputs[is_increment]
            )
            increments = distr_increments.sample()

            # Compute absolute increments from sampled relative increments if state is
            # not source
            is_relative = ~is_from_source[is_increment]
            if torch.any(is_relative):
                states_from_rel = tfloat(
                    current_state_dim_value[is_increment],
                    float_type=self.float,
                    device=self.device,
                )[is_relative]
                increments[is_relative] = self.relative_to_absolute_increments(
                    states_from_rel,
                    increments[is_relative],
                    is_backward=False,
                )

            # Set the actions in action_tensor with the results
            action_tensor[is_increment, 0] = self.ACTION_TYPE_INCREMENT
            action_tensor[is_increment, 1] = increments
            action_tensor[is_increment, 2] = is_from_source[is_increment].float()

        # Handle the deselect actions
        if torch.any(is_action_deselect):
            # When we are in a state where the deselect action is valid, there is
            # only one valid action : to deselect the currently selected action. No
            # sampling required.
            action_tensor[is_action_deselect, 0] = self.ACTION_TYPE_DESELECT_DIM
            action_tensor[is_action_deselect, 1] = selected_dims[is_action_deselect]
            action_tensor[is_action_deselect, 2] = 0

        # Convert the action tensor to tuples and return
        actions = [tuple(a.tolist()) for a in action_tensor]
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
        Samples a batch of backward actions from a batch of policy outputs.

        An action is a tuple containing three values. The first element identifies the
        type of action being performed any can be any value in {ACTION_TYPE_SELECT_DIM,
        ACTION_TYPE_INCREMENT, ACTION_TYPE_DESELECT_DIM}. The second element quantifies
        the action. In the case of an action to select/deselect a dimension, the second
        element takes the value of the relevant dimension index. In the case of an
        action to increment the selected dimension, the second elements takes the value
        of the absolute increment. The third value indicates whether the action is a
        special case back-to-source action. The EOS action is represented by (inf, inf, inf).

        When increment actions are sampled, the action indicates the absolute increment
        of the dimension value. However, in order to ensure that trajectories have
        finite length, increments must have a minumum increment (self.min_incr).
        Furthermore, absolute increments must also be smaller than the distance from
        the dimension value to the edge of the cube. In order to accomodate these
        constraints, first relative increments (in [0, 1]) are sampled from a
        (mixture of) Beta distribution(s), where 0.0 indicates an absolute increment
        of min_incr and 1.0 indicates an absolute increment of x (going back to the
        source).

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
        if sampling_method != "policy":
            raise NotImplementedError()

        # Initialize variables
        n_states = policy_outputs.shape[0]
        states_from_tensor = tfloat(
            states_from, float_type=self.float, device=self.device
        )
        selected_dims = states_from_tensor[:, self.STATE_IDX_SELECTED_DIM].int()

        is_action_select = ~mask[:, self.MASK_IDX_SELECT_DIM]
        is_increment = ~mask[:, self.MASK_IDX_INCREMENT]
        is_action_deselect = ~mask[:, self.MASK_IDX_DESELECT_DIM]
        is_eos = ~mask[:, self.MASK_IDX_EOS]
        is_bts_forced = ~mask[:, self.MASK_IDX_TO_FROM_SOURCE]
        valid_dimensions = ~mask[
            :, self.MASK_IDX_DIMS : self.MASK_IDX_DIMS + self.n_dim
        ]

        # Allocate tensor for the actions
        # The colums are : selected dimension, absolute increment, is from source
        action_tensor = torch.full(
            (n_states, 3), torch.inf, dtype=self.float, device=self.device
        )

        # Handle the select actions
        if torch.any(is_action_select):
            # When we are in a state where the select backward action is valid,
            # there is only one valid action : to "select" the currently selected
            # action (so the backward action will revert the selection). No sampling required.
            action_tensor[is_action_select, 0] = self.ACTION_TYPE_SELECT_DIM
            action_tensor[is_action_select, 1] = tfloat(
                selected_dims[is_action_select],
                float_type=self.float,
                device=self.device,
            )
            action_tensor[is_action_select, 2] = 0.0

        # Handle the increment actions
        if torch.any(is_increment):
            # Obtain the current value of the currently selected dimension for each
            # state
            current_state_dim_value = states_from_tensor[
                torch.arange(n_states), selected_dims
            ]

            # Determine which actions will be back-to-source (BTS). This can be either
            # because the mask indicates that these actions have to be BTS or because
            # we sample True with the BTS logit in the policy output
            logits_bts = self._get_policy_source_logit(policy_outputs)
            distr_bts = Bernoulli(logits=logits_bts)
            is_bts_sampled = tbool(distr_bts.sample(), device=self.device)
            is_bts = is_bts_forced | (is_increment & is_bts_sampled)

            # Set the BTS actions in the action_tensor. For BTS actions, the increment
            # value is simply the current value of the currently selected dimension
            action_tensor[is_bts, 0] = self.ACTION_TYPE_INCREMENT
            action_tensor[is_bts, 1] = current_state_dim_value[is_bts]
            action_tensor[is_bts, 2] = 1

            # Sample increment values for actions that increment but are not BTS
            is_increment_not_bts = is_increment & ~is_bts
            if torch.any(is_increment_not_bts):
                distr_increments = self._make_increments_distribution(
                    policy_outputs[is_increment_not_bts]
                )
                increments = distr_increments.sample()

                # Compute absolute increments from all sampled relative increments
                increments = self.relative_to_absolute_increments(
                    current_state_dim_value[is_increment_not_bts],
                    increments,
                    is_backward=True,
                )

                # Set the increment actions in the action tensor
                action_tensor[is_increment_not_bts, 0] = self.ACTION_TYPE_INCREMENT
                action_tensor[is_increment_not_bts, 1] = increments
                action_tensor[is_increment_not_bts, 2] = 0

        # Handle the deselect actions
        if torch.any(is_action_deselect):
            # Determine the logit value for each possible dimension to "deselect". This
            # involves masking the logit values for dimensions that cannot be deselected
            # according to the mask.
            discrete_actions_logits = self._get_policy_dim_logit(policy_outputs).clone()
            discrete_actions_logits = torch.where(
                valid_dimensions, discrete_actions_logits, -torch.inf
            )

            # Sample the actions
            distr_discrete = Categorical(
                logits=discrete_actions_logits[is_action_deselect]
            )
            sampled_deselect_actions = distr_discrete.sample()

            # Set the actions in action_tensor with the result.
            sampled_deselect_actions = tfloat(
                sampled_deselect_actions, float_type=self.float, device=self.device
            )
            action_tensor[is_action_deselect, 0] = self.ACTION_TYPE_DESELECT_DIM
            action_tensor[is_action_deselect, 1] = sampled_deselect_actions

        # Convert the action tensor to tuples and return
        actions = []
        for idx, a in enumerate(action_tensor):
            # If action is a BTS increment action, make sure it has the same decrement
            # value as the dimension to decrement. This need to be done here because
            # sometimes the process from pytorch tensors to python floats changes the
            # float values a bit.
            a = a.tolist()
            if a[0] == self.ACTION_TYPE_INCREMENT and a[2] == 1:
                dim_to_decrement = states_from[idx][self.STATE_IDX_SELECTED_DIM]
                a[1] = states_from[idx][dim_to_decrement]

            actions.append(tuple(a))

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

        WARNING : Like in the ContinuousCube class, the implementation of this method
        assumes that the provided action/masks/states_from are all compatible between
        them. That the masks are indeed the masks that would have been obtained from
        these states and that the actions are valid actions for the given states.
        """
        # Initialize variables
        n_states = policy_outputs.shape[0]
        states_from_tensor = tfloat(
            states_from, float_type=self.float, device=self.device
        )

        is_action_select_eos = ~mask[:, self.MASK_IDX_SELECT_DIM]
        is_increment = ~mask[:, self.MASK_IDX_INCREMENT]
        is_action_deselect = ~mask[:, self.MASK_IDX_DESELECT_DIM]
        is_eos_valid = ~mask[:, self.MASK_IDX_EOS]
        is_from_source = ~mask[:, self.MASK_IDX_TO_FROM_SOURCE]
        valid_dimensions = ~mask[
            :, self.MASK_IDX_DIMS : self.MASK_IDX_DIMS + self.n_dim
        ]

        # Allocate tensor for the log-probabilities
        log_probs = torch.zeros(n_states, dtype=self.float, device=self.device)

        # Handle the select/EOS actions
        if torch.any(is_action_select_eos):
            # Determine the logit value for each possible select/EOS action. This
            # involves masking the logit values for dimensions that cannot be selected
            # according to the mask as well as masking the logit values for the EOS
            # action in the states where that action it not valid.
            discrete_actions_logits = self._get_policy_dim_eos_logit(
                policy_outputs
            ).clone()
            discrete_actions_logits[:, :-1] = torch.where(
                valid_dimensions, discrete_actions_logits[:, :-1], -torch.inf
            )
            discrete_actions_logits[:, -1][~is_eos_valid] = -torch.inf

            # Determine the index (in the logits) of each discrete action
            # For the EOS action, use the last index (=self.n_dim) in the logits
            action_indices = actions[is_action_select_eos, 1]
            action_indices = torch.where(
                action_indices != torch.inf, action_indices, self.n_dim
            ).int()

            # Obtain the log-probability of each action under a multinomial distribution
            # parametrized by the logits
            distr_discrete = Categorical(
                logits=discrete_actions_logits[is_action_select_eos]
            )
            log_probs[is_action_select_eos] = distr_discrete.log_prob(action_indices)

        # Handle the increment actions
        if torch.any(is_increment):
            # Get absolute increments
            increments = actions[is_increment, 1]

            # Obtain the current values of the dimensions to increment
            selected_dims = states_from_tensor[:, self.STATE_IDX_SELECTED_DIM].int()
            current_state_dim_value = states_from_tensor[
                torch.arange(n_states), selected_dims
            ][is_increment]

            # Compute relative increments from absolute increments if state is not
            # source
            is_relative = ~is_from_source[is_increment]
            if torch.any(is_relative):
                # Compute the relative increments
                states_from_rel = tfloat(
                    current_state_dim_value,
                    float_type=self.float,
                    device=self.device,
                )[is_relative]
                increments[is_relative] = self.absolute_to_relative_increments(
                    states_from_rel,
                    increments[is_relative],
                    is_backward=False,
                )

            # Get logprobs of the increments
            # Clamp because increments of 0.0 or 1.0 would yield nan
            distr_increments = self._make_increments_distribution(
                policy_outputs[is_increment]
            )
            log_probs[is_increment] = distr_increments.log_prob(
                torch.clamp(increments, min=self.epsilon, max=(1 - self.epsilon))
            )

            # If state is not source, compute adjustement to the log probabilities
            # based on the gradient of the sampled increment wrt the target state
            current_vals = current_state_dim_value[is_relative]
            is_relative = torch.logical_and(is_increment, ~is_from_source)
            if torch.any(is_relative):
                log_probs[is_relative] += 1.0 / ((1.0 - current_vals - self.min_incr))

        # Note : there is no need to handle the deselect action. Since, when they are
        # valid, they are the *only* valid action it means that they automatically have
        # a probability of 1 (and therefore a logprob of 0, the value currently in
        # the tensor for unprocessed actions).
        return log_probs

    def _get_logprobs_backward(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: TensorType["n_states", "actions_dim"],
        mask: TensorType["n_states", "3"],
        states_from: List,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of backward actions.

        WARNING : Like in the ContinuousCube class, the implementation of this method
        assumes that the provided action/masks/states_from are all compatible between
        them. That the masks are indeed the masks that would have been obtained from
        these states and that the actions are valid actions for the given states.
        """
        # Initialize variables
        n_states = policy_outputs.shape[0]
        states_from_tensor = tfloat(
            states_from, float_type=self.float, device=self.device
        )

        is_action_select = ~mask[:, self.MASK_IDX_SELECT_DIM]
        is_increment = ~mask[:, self.MASK_IDX_INCREMENT]
        is_action_deselect = ~mask[:, self.MASK_IDX_DESELECT_DIM]
        is_eos = ~mask[:, self.MASK_IDX_EOS]
        is_bts_forced = ~mask[:, self.MASK_IDX_TO_FROM_SOURCE]
        valid_dimensions = ~mask[
            :, self.MASK_IDX_DIMS : self.MASK_IDX_DIMS + self.n_dim
        ]

        # Allocate tensor for the log-probabilities
        log_probs = torch.zeros(n_states, dtype=self.float, device=self.device)

        # Note : no need to handle the EOS actions or the dimension select action since,
        # when they are available as backward actions, they are the only available
        # actions so their probability is always 1 and, therefore, their log
        # probability is always 0.

        # Handle the increment actions
        if torch.any(is_increment):
            # There are three subtypes of actions that may occur here. There
            # are forced BTS actions, BTS actions that were not forced but sampled by
            # the policy, and regular non-BTS increment actions. The forced BTS actions,
            # by virtue of being forced, had a probability of 1 and a log prob of 0 so
            # they don't need to be explicitly handled.

            # For the actions that were not forced BTS, compute the log probability of
            # the first policy decision point : BTS vs a regular increment action.
            is_increment_not_forced_bts = is_increment & ~is_bts_forced
            is_sampled_bts = is_increment_not_forced_bts & actions[:, 2].bool()
            if torch.any(is_increment_not_forced_bts):
                logits_bts = self._get_policy_source_logit(policy_outputs)[
                    is_increment_not_forced_bts
                ]
                distr_bts = Bernoulli(logits=logits_bts)
                log_probs[is_increment_not_forced_bts] += distr_bts.log_prob(
                    is_sampled_bts[is_increment_not_forced_bts].to(self.float)
                )

            # For the actions that were not BTS, compute the log probability of the
            # sampled increment.
            is_regular_increment = is_increment_not_forced_bts & ~is_sampled_bts
            if torch.any(is_regular_increment):
                # Get absolute increments
                increments = actions[is_regular_increment, 1]

                # Make sure increments are finite
                assert torch.any(torch.isfinite(increments))

                # Obtain the current values of the dimensions to increment
                selected_dims = states_from_tensor[:, self.STATE_IDX_SELECTED_DIM].int()
                current_state_dim_value = states_from_tensor[
                    torch.arange(n_states), selected_dims
                ][is_regular_increment]

                # Compute relative increments from the absolute increments in the
                # actions
                increments = self.absolute_to_relative_increments(
                    current_state_dim_value,
                    increments,
                    is_backward=True,
                )

                # Get logprobs of the sampled increments
                # # Clamp because increments of 0.0 or 1.0 would yield nan
                distr_increments = self._make_increments_distribution(
                    policy_outputs[is_regular_increment]
                )
                log_probs[is_regular_increment] += distr_increments.log_prob(
                    torch.clamp(increments, min=self.epsilon, max=(1 - self.epsilon))
                )

                # Compute adjustement to the log probabilities based on the gradients
                # of the sampled increment wrt the target state
                log_probs[is_regular_increment] += 1.0 / (
                    (current_state_dim_value - self.min_incr)
                )

        # Handle the deselect actions
        if torch.any(is_action_deselect):
            # Determine the logit value for each possible dimension to "deselect". This
            # involves masking the logit values for dimensions that cannot be deselected
            # according to the mask.
            discrete_actions_logits = self._get_policy_dim_logit(policy_outputs).clone()
            discrete_actions_logits = torch.where(
                valid_dimensions, discrete_actions_logits, -torch.inf
            )

            # Obtain the log-probability of each action under a multinomial distribution
            # parametrized by the logits
            distr_discrete = Categorical(
                logits=discrete_actions_logits[is_action_deselect]
            )
            log_probs[is_action_deselect] = distr_discrete.log_prob(actions[:, 1])

        return log_probs

    # TODO: make generic for continuous environments?
    def step(self, action: Tuple[float]) -> Tuple[List[float], Tuple[int, float], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. The first element is the type of action
            (selecting/incrementing/deselecting a dimension). The second element is
            the value of the action (the dimension to select/deselect or the value
            of the increment). The third element is only used for increment actions
            and indicates whether the selected dimension to increment is currently in
            its source state.

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
        # Refuse all actions if the trajectory is over
        if self.done:
            return self.state, action, False

        # Recover important attributes from the state
        expected_action_type = self.state[self.STATE_IDX_NEXT_ACTION_TYPE]
        selected_dimension = self.state[self.STATE_IDX_SELECTED_DIM]

        # Handle EOS action
        if action == self.eos:
            # None of the cube dimension can be in its source state
            eff_dims = self._get_effective_dims(self.state)
            eff_dims_source = self._get_effective_dims(self.source)
            for d, d_s in zip(eff_dims, eff_dims_source):
                if d == d_s:
                    return self.state, action, False

            # There can be no currently selected dimension for incrementation
            if selected_dimension != self.NO_DIMENSION_SELECTED:
                return self.state, action, False

            # Apply EOS action
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True

        # Recover important attributes from the action
        action_type, action_value, action_is_from_source = self.unpack_action(action)

        # Refuse action if it isn't of the required type
        if action_type != expected_action_type:
            return self.state, action, False

        if action_type == self.ACTION_TYPE_SELECT_DIM:
            assert selected_dimension == self.NO_DIMENSION_SELECTED

            # Refuse the action if the desired dimension cannot be incremented
            if self.state[action_value] > 1 - self.min_incr:
                return self.state, action, False

            # Refuse the action if the desired dimension doesn't exist
            if action_value >= self.n_dim:
                return self.state, action, False

            # Apply the action
            self.state[self.STATE_IDX_SELECTED_DIM] = action_value
            self.state[self.STATE_IDX_NEXT_ACTION_TYPE] = self.ACTION_TYPE_INCREMENT

        elif action_type == self.ACTION_TYPE_INCREMENT:
            assert selected_dimension != self.NO_DIMENSION_SELECTED

            # Refuse the action if the increment is too large
            if self.state[selected_dimension] + action_value > 1:
                return self.state, action, False

            # Refuse the action if its "from_source" value doesn't align with the
            # current value of the selected dimension
            selected_dim_is_in_source = (
                self.state[selected_dimension] == self.source[selected_dimension]
            )
            if action_is_from_source != selected_dim_is_in_source:
                return self.state, action, False

            # Apply the action
            if action_is_from_source:
                self.state[selected_dimension] = 0
            self.state[selected_dimension] += action_value
            self.state[self.STATE_IDX_NEXT_ACTION_TYPE] = self.ACTION_TYPE_DESELECT_DIM

        elif action_type == self.ACTION_TYPE_DESELECT_DIM:
            assert selected_dimension != self.NO_DIMENSION_SELECTED

            # Refuse the action is the dimension to deselect isn't the selected
            # dimension
            if action_value != selected_dimension:
                return self.state, action, False

            # Apply the action
            self.state[self.STATE_IDX_SELECTED_DIM] = self.NO_DIMENSION_SELECTED
            self.state[self.STATE_IDX_NEXT_ACTION_TYPE] = self.ACTION_TYPE_SELECT_DIM

        self.n_actions += 1
        return self.state, action, True

    def step_backwards(
        self, action: Tuple[int, float]
    ) -> Tuple[List[float], Tuple[int, float], bool]:
        """
        Executes backward step given an action.

        Args
        ----
        action : tuple
            Action to be executed. The first element is the type of action
            (selecting/incrementing/deselecting a dimension). The second element is
            the value of the action (the dimension to select/deselect or the value
            of the increment). The third element is only used for increment actions
            and indicates whether the selected dimension to increment is currently in
            its source state.

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
        # Refuse all actions if the state is already at the source state
        if self.state == self.source:
            return self.state, action, False

        # Handle EOS action
        if action == self.eos:
            if self.done:
                # Set done to False, increment n_actions and return the same state
                self.done = False
                self.n_actions += 1
                return self.state, action, True
            else:
                return self.state, action, False

        # Recover important attributes from the state
        expected_action_type = self.state[self.STATE_IDX_NEXT_ACTION_TYPE]
        selected_dimension = self.state[self.STATE_IDX_SELECTED_DIM]

        # Recover important attributes from the action
        action_type, action_value, action_is_bts = self.unpack_action(action)

        # Handle action
        if action_type == self.ACTION_TYPE_SELECT_DIM:
            # Refuse the action if it's not the right type for the current state
            # NOTE : expected_action_type is the action type that would be expected
            # for a *forward* action. It is not the same type as what is expected for
            # a backward action
            if expected_action_type != self.ACTION_TYPE_INCREMENT:
                return self.state, action, False

            assert selected_dimension != self.NO_DIMENSION_SELECTED

            # Refuse the action if it seeks to select a dimension different from the
            # one currently selected
            if action_value != selected_dimension:
                return self.state, action, False

            # Apply the action
            self.state[self.STATE_IDX_SELECTED_DIM] = self.NO_DIMENSION_SELECTED
            self.state[self.STATE_IDX_NEXT_ACTION_TYPE] = self.ACTION_TYPE_SELECT_DIM

        elif action_type == self.ACTION_TYPE_INCREMENT:
            # Refuse the action if it's not the right type for the current state
            # NOTE : expected_action_type is the action type that would be expected
            # for a *forward* action. It is not the same type as what is expected for
            # a backward action
            if expected_action_type != self.ACTION_TYPE_DESELECT_DIM:
                return self.state, action, False

            assert selected_dimension != self.NO_DIMENSION_SELECTED
            assert self.state[selected_dimension] != self.source[selected_dimension]

            # Refuse the action if the increment is too large
            if action_value > self.state[selected_dimension]:
                return self.state, action, False

            # Apply the action
            if action_is_bts:
                self.state[selected_dimension] = self.source[selected_dimension]
            else:
                self.state[selected_dimension] -= action_value
            self.state[self.STATE_IDX_NEXT_ACTION_TYPE] = self.ACTION_TYPE_INCREMENT

        elif action_type == self.ACTION_TYPE_DESELECT_DIM:
            # Refuse the action if it's not the right type for the current state
            # NOTE : expected_action_type is the action type that would be expected
            # for a *forward* action. It is not the same type as what is expected for
            # a backward action
            if expected_action_type != self.ACTION_TYPE_SELECT_DIM:
                return self.state, action, False

            assert selected_dimension == self.NO_DIMENSION_SELECTED

            # Refuse the action if the desired dimension cannot be decremented
            if self.state[action_value] == self.source[action_value]:
                return self.state, action, False

            # Apply the action
            self.state[self.STATE_IDX_SELECTED_DIM] = action_value
            self.state[self.STATE_IDX_NEXT_ACTION_TYPE] = self.ACTION_TYPE_DESELECT_DIM

        self.n_actions += 1
        return self.state, action, True

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

        # Convert the states to lists and add
        states_suffix = [0, 0]
        states_suffix[self.STATE_IDX_NEXT_ACTION_TYPE] = self.ACTION_TYPE_SELECT_DIM
        states_suffix[self.STATE_IDX_SELECTED_DIM] = self.NO_DIMENSION_SELECTED
        states = [list(el) + states_suffix for el in states]
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

        # Add two last dimensions to the states for the next action type and the
        # selected dimension
        states_suffix = [0, 0]
        states_suffix[self.STATE_IDX_NEXT_ACTION_TYPE] = self.ACTION_TYPE_SELECT_DIM
        states_suffix[self.STATE_IDX_SELECTED_DIM] = self.NO_DIMENSION_SELECTED
        states = np.hstack((states, [states_suffix] * 100))

        return states.tolist()

    # TODO: make generic for all environments
    def sample_from_reward(
        self, n_samples: int, epsilon=1e-4
    ) -> TensorType["n_samples", "state_dim"]:
        """
        Rejection sampling with proposal the uniform distribution in [0, 1]^n_dim.

        Returns a tensor in GFlowNet (state) format.
        """
        samples_final = []
        max_reward = self.proxy2reward(self.proxy.min)
        while len(samples_final) < n_samples:
            samples_uniform = self.states2proxy(
                self.get_uniform_terminating_states(n_samples)
            )
            rewards = self.proxy2reward(self.proxy(samples_uniform))
            mask = (
                torch.rand(n_samples, dtype=self.float, device=self.device)
                * (max_reward + epsilon)
                < rewards
            )
            samples_accepted = samples_uniform[mask]
            samples_final.extend(samples_accepted[-(n_samples - len(samples_final)) :])
        return torch.vstack(samples_final)

    # TODO: make generic for all envs
    def fit_kde(self, samples, kernel="gaussian", bandwidth=0.1):
        return KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(samples)

    def plot_reward_samples(
        self,
        samples,
        alpha=0.5,
        cell_min=-1.0,
        cell_max=1.0,
        dpi=150,
        max_samples=500,
        **kwargs,
    ):
        if self.n_dim != 2:
            return None
        # Sample a grid of points in the state space and obtain the rewards
        x = np.linspace(cell_min, cell_max, 201)
        y = np.linspace(cell_min, cell_max, 201)
        xx, yy = np.meshgrid(x, y)
        X = np.stack([xx, yy], axis=-1)
        states_mesh = torch.tensor(
            X.reshape(-1, 2), device=self.device, dtype=self.float
        )
        rewards = self.proxy2reward(self.proxy(states_mesh))
        # Init figure
        fig, ax = plt.subplots()
        fig.set_dpi(dpi)
        # Plot reward contour
        h = ax.contourf(xx, yy, rewards.reshape(xx.shape).cpu().numpy(), alpha=alpha)
        ax.axis("scaled")
        fig.colorbar(h, ax=ax)
        # Plot samples
        random_indices = np.random.permutation(samples.shape[0])[:max_samples]
        ax.scatter(samples[random_indices, 0], samples[random_indices, 1], alpha=alpha)
        # Figure settings
        ax.grid()
        padding = 0.05 * (cell_max - cell_min)
        ax.set_xlim([cell_min - padding, cell_max + padding])
        ax.set_ylim([cell_min - padding, cell_max + padding])
        plt.tight_layout()
        return fig

    # TODO: make generic for all envs
    def plot_kde(
        self,
        kde,
        alpha=0.5,
        cell_min=-1.0,
        cell_max=1.0,
        dpi=150,
        colorbar=True,
        **kwargs,
    ):
        if self.n_dim != 2:
            return None
        # Sample a grid of points in the state space and score them with the KDE
        x = np.linspace(cell_min, cell_max, 201)
        y = np.linspace(cell_min, cell_max, 201)
        xx, yy = np.meshgrid(x, y)
        X = np.stack([xx, yy], axis=-1)
        Z = np.exp(kde.score_samples(X.reshape(-1, 2))).reshape(xx.shape)
        # Init figure
        fig, ax = plt.subplots()
        fig.set_dpi(dpi)
        # Plot KDE
        h = ax.contourf(xx, yy, Z, alpha=alpha)
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
