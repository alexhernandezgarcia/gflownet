"""
Classes to represent hyper-cube environments
"""
import itertools
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.neighbors import KernelDensity
from torch.distributions import Bernoulli, Beta, Categorical, MixtureSameFamily, Uniform
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv


class Cube(GFlowNetEnv, ABC):
    """
    Continuous (hybrid: discrete and continuous) hyper-cube environment (continuous
    version of a hyper-grid) in which the action space consists of the increment of
    dimension d, modelled by a beta distribution.

    The states space is the value of each dimension. If the value of a dimension gets
    larger than max_val, then the trajectory is ended.

    Attributes
    ----------
    n_dim : int
        Dimensionality of the hyper-cube.

    max_val : float
        Max length of the hyper-cube.

    min_incr : float
        Minimum increment in the actions, expressed as the fraction of max_val. This is
        necessary to ensure coverage of the state space.
    """

    def __init__(
        self,
        n_dim: int = 2,
        max_val: float = 1.0,
        min_incr: float = 0.1,
        n_comp: int = 1,
        beta_params_min: float = 0.1,
        beta_params_max: float = 2.0,
        fixed_distr_params: dict = {
            "beta_alpha": 2.0,
            "beta_beta": 5.0,
            "bernoulli_logit": -2.3,
        },
        random_distr_params: dict = {
            "beta_alpha": 1.0,
            "beta_beta": 1.0,
            "bernoulli_logit": -0.693,
        },
        **kwargs,
    ):
        assert n_dim > 0
        assert max_val > 0.0
        assert n_comp > 0
        # Main properties
        self.continuous = True
        self.n_dim = n_dim
        self.eos = self.n_dim
        self.max_val = max_val
        self.min_incr = min_incr * self.max_val
        # Parameters of the policy distribution
        self.n_comp = n_comp
        self.beta_params_min = beta_params_min
        self.beta_params_max = beta_params_max
        # Source state: position 0 at all dimensions
        self.source = [0.0 for _ in range(self.n_dim)]
        # Action from source: (n_dim, 0)
        self.action_source = (self.n_dim, 0)
        # End-of-sequence action: (n_dim + 1, 0)
        self.eos = (self.n_dim + 1, 0)
        # Conversions: only conversions to policy are implemented and the rest are the
        # same
        self.state2proxy = self.state2policy
        self.statebatch2proxy = self.statebatch2policy
        self.statetorch2proxy = self.statetorch2policy
        self.state2oracle = self.state2proxy
        self.statebatch2oracle = self.statebatch2proxy
        self.statetorch2oracle = self.statetorch2proxy
        # Base class init
        super().__init__(
            fixed_distr_params=fixed_distr_params,
            random_distr_params=random_distr_params,
            **kwargs,
        )

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

    def statetorch2policy(
        self, states: TensorType["batch", "state_dim"] = None
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Clips the states into [0, max_val] and maps them to [-1.0, 1.0]

        Args
        ----
        state : list
            State
        """
        return 2.0 * torch.clip(states, min=0.0, max=self.max_val) - 1.0

    def statebatch2policy(
        self, states: List[List]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Clips the states into [0, max_val] and maps them to [-1.0, 1.0]

        Args
        ----
        state : list
            State
        """
        return self.statetorch2policy(
            torch.tensor(states, device=self.device, dtype=self.float)
        )

    def state2policy(self, state: List = None) -> List:
        """
        Clips the state into [0, max_val] and maps it to [-1.0, 1.0]
        """
        if state is None:
            state = self.state.copy()
        return [2.0 * min(max(0.0, s), self.max_val) - 1.0 for s in state]

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
    def sample_actions(
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
        states_target: TensorType["n_states", "policy_input_dim"],
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


class HybridCube(Cube):
    """
    Continuous (hybrid: discrete and continuous) hyper-cube environment (continuous
    version of a hyper-grid) in which the action space consists of the increment of
    dimension d, modelled by a beta distribution.

    The states space is the value of each dimension. If the value of a dimension gets
    larger than max_val, then the trajectory is ended.

    Attributes
    ----------
    n_dim : int
        Dimensionality of the hyper-cube.

    max_val : float
        Max length of the hyper-cube.

    min_incr : float
        Minimum increment in the actions, expressed as the fraction of max_val. This is
        necessary to ensure coverage of the state space.
    """

    def __init__(
        self,
        n_dim: int = 2,
        max_val: float = 1.0,
        min_incr: float = 0.1,
        n_comp: int = 1,
        do_nonzero_source_prob: bool = True,
        fixed_distr_params: dict = {
            "beta_alpha": 2.0,
            "beta_beta": 5.0,
        },
        random_distr_params: dict = {
            "beta_alpha": 1.0,
            "beta_beta": 1.0,
        },
        **kwargs,
    ):
        assert n_dim > 0
        assert max_val > 0.0
        assert n_comp > 0
        # Main properties
        self.continuous = True
        self.n_dim = n_dim
        self.eos = self.n_dim
        self.max_val = max_val
        self.min_incr = min_incr * self.max_val
        # Parameters of fixed policy distribution
        self.n_comp = n_comp
        if do_nonzero_source_prob:
            self.n_params_per_dim = 4
        else:
            self.n_params_per_dim = 3
        # Source state: position 0 at all dimensions
        self.source = [0.0 for _ in range(self.n_dim)]
        # Action from source: (n_dim, 0)
        self.action_source = (self.n_dim, 0)
        # End-of-sequence action: (n_dim + 1, 0)
        self.eos = (self.n_dim + 1, 0)
        # Conversions: only conversions to policy are implemented and the rest are the
        # same
        self.state2proxy = self.state2policy
        self.statebatch2proxy = self.statebatch2policy
        self.statetorch2proxy = self.statetorch2policy
        self.state2oracle = self.state2proxy
        self.statebatch2oracle = self.statebatch2proxy
        self.statetorch2oracle = self.statetorch2proxy
        # Base class init
        super().__init__(
            fixed_distr_params=fixed_distr_params,
            random_distr_params=random_distr_params,
            **kwargs,
        )

    def get_action_space(self):
        """
        Since this is a hybrid (continuous/discrete) environment, this method
        constructs a list with the discrete actions.

        The actions are tuples with two values: (dimension, increment) where dimension
        indicates the index of the dimension on which the action is to be performed and
        increment indicates the increment of the dimension.

        Additionally, there are two special discrete actions:
            - Sample an increment for all dimensions. Only valid from the source state.
            - EOS action

        The (discrete) action space is then one tuple per dimension (with 0 increment),
        plus the EOS action.
        """
        actions = [(d, 0) for d in range(self.n_dim)]
        actions.append(self.action_source)
        actions.append(self.eos)
        return actions

    def get_policy_output(self, params: dict) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy.

        For each dimension d of the hyper-cube and component c of the mixture, the
        output of the policy should return
          1) the weight of the component in the mixture
          2) the logit(alpha) parameter of the Beta distribution to sample the increment
          3) the logit(beta) parameter of the Beta distribution to sample the increment

        Additionally, the policy output contains one logit per dimension plus one logit
        for the EOS action, for the categorical distribution over dimensions.

        Therefore, the output of the policy model has dimensionality D x C x 3 + D + 1,
        where D is the number of dimensions (self.n_dim) and C is the number of
        components (self.n_comp). The first D + 1 entries in the policy output
        correspond to the categorical logits. Then, the next 3 x C entries in the
        policy output correspond to the first dimension, and so on.
        """
        policy_output = torch.ones(
            self.n_dim * self.n_comp * 3 + self.n_dim + 1,
            device=self.device,
            dtype=self.float,
        )
        policy_output[self.n_dim + 2 :: 3] = params["beta_alpha"]
        policy_output[self.n_dim + 3 :: 3] = params["beta_beta"]
        return policy_output

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        Returns a vector with the length of the discrete part of the action space + 1:
        True if action is invalid going forward given the current state, False
        otherwise.

        All discrete actions are valid, including eos, except if the value of any
        dimension has excedded max_val, in which case the only valid action is eos.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(self.action_space_dim)]
        # If state is source, then next action can only be the action from source.
        if all([s == ss for s in zip(self.state, self.source)]):
            mask = [True for _ in range(self.action_space_dim)]
            mask[-2] = False
        # If the value of any dimension is greater than max_val, then next action can
        # only be EOS.
        elif any([s > self.max_val for s in self.state]):
            mask = [True for _ in range(self.action_space_dim)]
            mask[-1] = False
        else:
            mask = [False for _ in range(self.action_space_dim)]
        return mask

    def get_mask_invalid_actions_backward(self, state=None, done=None, parents_a=None):
        """
        Returns a vector with the length of the discrete part of the action space + 1:
        True if action is invalid going backward given the current state, False
        otherwise.

        The backward mask has the following structure:

        - 0:n_dim : whether keeping a dimension as is, that is sampling a decrement of
          0, can have zero probability. True if the value at the dimension is smaller
          than or equal to 1 - min_incr.
        - n_dim : whether going to source is invalid. Always valid, hence always False,
          except if done.
        - n_dim + 1 : whether sampling EOS is invalid. Only valid if done.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        mask_dim = self.n_dim + 2
        # If done, only valid action is EOS.
        if done:
            mask = [True for _ in range(mask_dim)]
            mask[-1] = False
        mask = [True for _ in range(mask_dim)]
        mask[-2] = False
        # Dimensions whose value is greater than 1 - min_incr must have non-zero
        # probability of sampling a decrement of exactly zero.
        for dim, s in enumerate(state):
            if s > 1 - self.min_incr:
                mask[dim] = False
        return mask

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
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos]
        # If source state
        elif state[-1] == 0:
            return [], []
        else:
            dim, incr = action
            state[dim] -= incr
            parents = [state]
            return parents, [action]

    def sample_actions(
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
        device = policy_outputs.device
        n_states = policy_outputs.shape[0]
        ns_range = torch.arange(n_states).to(device)
        # Sample dimensions
        if sampling_method == "uniform":
            logits_dims = torch.ones(n_states, self.policy_output_dim).to(device)
        elif sampling_method == "policy":
            logits_dims = policy_outputs[:, 0 : self.n_dim + 1]
            logits_dims /= temperature_logits
        if mask_invalid_actions is not None:
            logits_dims[mask_invalid_actions] = -loginf
        dimensions = Categorical(logits=logits_dims).sample()
        logprobs_dim = self.logsoftmax(logits_dims)[ns_range, dimensions]
        # Sample increments
        ns_range_noeos = ns_range[dimensions != self.eos[0]]
        dimensions_noeos = dimensions[dimensions != self.eos[0]]
        increments = torch.zeros(n_states).to(device)
        logprobs_increments = torch.zeros(n_states).to(device)
        if len(dimensions_noeos) > 0:
            if sampling_method == "uniform":
                distr_increments = Uniform(
                    torch.zeros(len(ns_range_noeos)),
                    self.max_val * torch.ones(len(ns_range_noeos)),
                )
            elif sampling_method == "policy":
                alphas = policy_outputs[:, self.n_dim + 2 :: 3][
                    ns_range_noeos, dimensions_noeos
                ]
                betas = policy_outputs[:, self.n_dim + 3 :: 3][
                    ns_range_noeos, dimensions_noeos
                ]
                distr_increments = Beta(torch.exp(alphas), torch.exp(betas))
            increments[ns_range_noeos] = distr_increments.sample()
            logprobs_increments[ns_range_noeos] = distr_increments.log_prob(
                increments[ns_range_noeos]
            )
            # Apply minimum increment
            increments[ns_range_noeos] = torch.min(
                increments[ns_range_noeos],
                self.min_incr * torch.ones(ns_range_noeos.shape[0]),
            )
        # Combined probabilities
        logprobs = logprobs_dim + logprobs_increments
        # Build actions
        actions = [
            (dimension, incr)
            for dimension, incr in zip(dimensions.tolist(), increments.tolist())
        ]
        return actions, logprobs

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        is_forward: bool,
        actions: TensorType["n_states", 2],
        states_target: TensorType["n_states", "policy_input_dim"],
        mask_invalid_actions: TensorType["batch_size", "policy_output_dim"] = None,
        loginf: float = 1000,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.
        """
        device = policy_outputs.device
        dimensions, steps = zip(*actions)
        dimensions = torch.LongTensor([d.long() for d in dimensions]).to(device)
        steps = torch.FloatTensor(steps).to(device)
        n_states = policy_outputs.shape[0]
        ns_range = torch.arange(n_states).to(device)
        # Dimensions
        logits_dims = policy_outputs[:, 0::3]
        if mask_invalid_actions is not None:
            logits_dims[mask_invalid_actions] = -loginf
        logprobs_dim = self.logsoftmax(logits_dims)[ns_range, dimensions]
        # Steps
        ns_range_noeos = ns_range[dimensions != self.eos]
        dimensions_noeos = dimensions[dimensions != self.eos]
        logprobs_steps = torch.zeros(n_states).to(device)
        if len(dimensions_noeos) > 0:
            alphas = policy_outputs[:, 1::3][ns_range_noeos, dimensions_noeos]
            betas = policy_outputs[:, 2::3][ns_range_noeos, dimensions_noeos]
            distr_steps = Beta(torch.exp(alphas), torch.exp(betas))
            logprobs_steps[ns_range_noeos] = distr_steps.log_prob(steps[ns_range_noeos])
        # Combined probabilities
        logprobs = logprobs_dim + logprobs_steps
        return logprobs

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
        if self.done:
            return self.state, action, False
        # If action is eos or any dimension is beyond max_val or n_actions has reached
        # max_traj_length, then force eos
        elif (
            action[0] == self.eos
            or any([s > self.max_val for s in self.state])
            or self.n_actions >= self.max_traj_length
        ):
            self.done = True
            self.n_actions += 1
            return self.state, (self.eos, 0.0), True
        # If action is not eos, then perform action
        elif action[0] != self.eos:
            self.n_actions += 1
            self.state[action[0]] += action[1]
            return self.state, action, True
        # Otherwise (unreachable?) it is invalid
        else:
            return self.state, action, False

    def get_grid_terminating_states(self, n_states: int) -> List[List]:
        n_per_dim = int(np.ceil(n_states ** (1 / self.n_dim)))
        linspaces = [np.linspace(0, self.max_val, n_per_dim) for _ in range(self.n_dim)]
        states = list(itertools.product(*linspaces))
        # TODO: check if necessary
        states = [list(el) for el in states]
        return states


class ContinuousCube(Cube):
    """
    Continuous hyper-cube environment (continuous version of a hyper-grid) in which the
    action space consists of the increment of each dimension d, modelled by a mixture
    of Beta distributions. The states space is the value of each dimension. In order to
    ensure that all trajectories are of finite length, actions have a minimum increment
    for all dimensions determined by min_incr.  If the value of any dimension is larger
    than 1 - min_incr, then the trajectory is ended (the only next valid action is
    EOS). In order to ensure the coverage of the state space, the first action (from
    the source state) is not constrained by the minimum increment.

    Actions do not represent absolute increments but rather the relative increment with
    respect to the distance to the edges of the hyper-cube, from the minimum increment.
    That is, if dimension d of a state has value 0.3, the minimum increment (min_incr)
    is 0.1 and the maximum value (max_val) is 1.0, an action of 0.5 will increment the
    value of the dimension in 0.5 * (1.0 - 0.3 - 0.1) = 0.5 * 0.6 = 0.3. Therefore, the
    value of d in the next state will be 0.3 + 0.3 = 0.6.

    Attributes
    ----------
    n_dim : int
        Dimensionality of the hyper-cube.

    max_val : float
        Max length of the hyper-cube.

    min_incr : float
        Minimum increment in the actions, expressed as the fraction of max_val. This is
        necessary to ensure coverage of the state space.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action_space(self):
        """
        The actions are tuples of length n_dim + 1, where the value at position d indicates
        the (positive, relative) increment of dimension d. The value at the last
        position indicates the minimum increment: 0.0 if the transition is from the
        source state, min_incr otherwise.

        Additionally, there are two special discrete actions:
            - EOS action. Indicated by np.inf for all dimensions. Only valid forwards.
            - Back-to-source action. Indicated by -1 for all dimensions. Only valid
              backwards.
        """
        generic_action = tuple([0.0 for _ in range(self.n_dim)] + [self.min_incr])
        from_source = tuple([0.0 for _ in range(self.n_dim)] + [0.0])
        to_source = tuple([-1.0 for _ in range(self.n_dim + 1)])
        self.eos = tuple([np.inf for _ in range(self.n_dim + 1)])
        actions = [generic_action, from_source, to_source, self.eos]
        return actions

    def get_policy_output(self, params: dict) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy.

        For each dimension d of the hyper-cube and component c of the mixture, the
        output of the policy should return
          1) the weight of the component in the mixture
          2) the logit(alpha) parameter of the Beta distribution to sample the increment
          3) the logit(beta) parameter of the Beta distribution to sample the increment

        Additionally, the policy output contains one logit (pos [-1]) of a Bernoulli
        distribution to model the (discrete) forward probability of selecting the EOS
        action and another logit (pos [-2]) for the (discrete) backward probability of
        returning to the source node.

        Finally, the backward distribution requires a discrete probability distribution
        (Bernoulli) for each dimension, to model the probability of sampling an
        increment equal to zero when the value at the dimension is larger than
        1 - min_incr. These are stored at [0:n_dim].

        Therefore, the output of the policy model has dimensionality D x C x 3 + 2,
        where D is the number of dimensions (self.n_dim) and C is the number of
        components (self.n_comp).
        """
        policy_output = torch.ones(
            self.n_dim + self.n_dim * self.n_comp * 3 + 2,
            device=self.device,
            dtype=self.float,
        )
        policy_output[self.n_dim + 1 : -2 : 3] = params["beta_alpha"]
        policy_output[self.n_dim + 2 : -2 : 3] = params["beta_beta"]
        policy_output[-2] = params["bernoulli_logit"]
        policy_output[-1] = params["bernoulli_logit"]
        return policy_output

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        Returns a vector indicating which backward actions are invalid.

        The forward mask has the following structure:

        - 0:n_dim : whether sampling each dimension is invalid. Invalid (True) if the
          value at the dimension is larger than 1 - min_incr.
        - n_dim : whether sampling from source is invalid. Invalid except when when the
          state is the source state.
        - n_dim + 1 : whether sampling EOS is invalid. EOS is valid from any state
          (including the source state), hence always False.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        mask_dim = self.n_dim + 2
        # If done, no action is valid
        if done:
            return [True for _ in range(mask_dim)]
        mask = [False for _ in range(mask_dim)]
        # If state is not source, sampling from source is invalid.
        if state != self.source:
            mask[-2] = True
        # Dimensions whose value is greater than 1 - min_incr cannot be further
        # incremented
        for dim, s in enumerate(state):
            if s > 1 - self.min_incr:
                mask[dim] = True
        return mask

    def get_mask_invalid_actions_backward(self, state=None, done=None, parents_a=None):
        """
        Returns a vector indicating which backward actions are invalid.

        The backward mask has the following structure:

        - 0:n_dim : whether keeping a dimension as is, that is sampling a decrement of
          0, can have zero probability. True if the value at the dimension is smaller
          than or equal to 1 - min_incr.
        - n_dim : whether other actions except back-to-source are invalid. False if any
          dimension is smaller than min_incr.
        - n_dim + 1 : whether sampling EOS is invalid. Only valid if done.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        mask_dim = self.n_dim + 2
        # If done, only valid action is EOS.
        if done:
            mask = [True for _ in range(mask_dim)]
            mask[-1] = False
            return mask
        # If state is source, all actions are invalid.
        if state == self.source:
            return [True for _ in range(mask_dim)]
        # If any dimension is smaller than m, then back-to-source is the only valid
        # action
        if any([s < self.min_incr for s in state]):
            mask = [True for _ in range(mask_dim)]
            mask[-2] = False
            return mask
        mask = [True for _ in range(mask_dim)]
        # Dimensions whose value is greater than 1 - min_incr must have non-zero
        # probability of sampling a decrement of exactly zero.
        for dim, s in enumerate(state):
            if s > 1 - self.min_incr:
                mask[dim] = False
        return mask

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
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos]
        # If source state
        if all([s == ss for s, ss in zip(state, self.source)]):
            return [], []
        else:
            epsilon = 1e-9
            min_incr = action[-1]
            for dim, incr_rel in enumerate(action[:-1]):
                incr = min_incr + incr_rel * (state[dim] - min_incr)
                assert incr >= (
                    min_incr - epsilon
                ), f"""
                Increment {incr} at dim {dim} smaller than minimum increment ({min_incr}).
                \nState:\n{state}\nAction:\n{action}
                """
                state[dim] -= incr
                assert all(
                    [s <= (self.max_val + epsilon) for s in state]
                ), f"""
                State is out of cube bounds.
                \nState:\n{state}\nAction:\n{action}\nIncrement: {incr}
                """
                assert all(
                    [s >= (0.0 - epsilon) for s in state]
                ), f"""
                State is out of cube bounds.
                \nState:\n{state}\nAction:\n{action}\nIncrement: {incr}
                """
            return [state], [action]

    def sample_actions(
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
        device = policy_outputs.device
        n_states = policy_outputs.shape[0]
        ns_range = torch.arange(n_states).to(device)
        # EOS
        idx_nofix = ns_range[torch.any(~mask_invalid_actions[:, : self.n_dim], axis=1)]
        distr_eos = Bernoulli(logits=policy_outputs[idx_nofix, -1])
        mask_sampled_eos = distr_eos.sample().to(torch.bool)
        logprobs_eos = torch.zeros(n_states, device=device, dtype=self.float)
        logprobs_eos[idx_nofix] = distr_eos.log_prob(mask_sampled_eos.to(self.float))
        # Sample increments
        idx_sample = idx_nofix[~mask_sampled_eos]
        mask_idx_sample = torch.zeros(n_states, device=device, dtype=torch.bool)
        mask_idx_sample[idx_sample] = True
        mask_source_sample = torch.logical_and(
            ~mask_invalid_actions[:, self.n_dim], mask_idx_sample
        )
        mask_generic_sample = torch.logical_and(
            mask_invalid_actions[:, self.n_dim], mask_idx_sample
        )
        idx_source = ns_range[mask_source_sample]
        idx_generic = ns_range[mask_generic_sample]
        n_sample = idx_sample.shape[0]
        logprobs_sample = torch.zeros(n_states, device=device, dtype=self.float)
        increments = torch.inf * torch.ones(
            (n_states, self.n_dim), device=device, dtype=self.float
        )
        if len(idx_sample) > 0:
            if sampling_method == "uniform":
                distr_increments = Uniform(
                    torch.zeros(n_sample),
                    torch.ones(n_sample),
                )
            elif sampling_method == "policy":
                mix_logits = policy_outputs[idx_sample, self.n_dim : -2 : 3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                mix = Categorical(logits=mix_logits)
                alphas = policy_outputs[idx_sample, self.n_dim + 1 : -2 : 3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                alphas = (
                    self.beta_params_max * torch.sigmoid(alphas) + self.beta_params_min
                )
                betas = policy_outputs[idx_sample, self.n_dim + 2 : -2 : 3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                betas = (
                    self.beta_params_max * torch.sigmoid(betas) + self.beta_params_min
                )
                beta_distr = Beta(alphas, betas)
                distr_increments = MixtureSameFamily(mix, beta_distr)
            increments[idx_sample, :] = distr_increments.sample()
            logprobs_sample[idx_sample] = distr_increments.log_prob(
                increments[idx_sample, :]
            ).sum(axis=1)
        # Combined probabilities
        logprobs = logprobs_eos + logprobs_sample
        # Set minimum increments
        min_increments = torch.inf * torch.ones(
            n_states, device=device, dtype=self.float
        )
        min_increments[idx_generic] = self.min_incr
        min_increments[idx_source] = 0.0
        # Make increments of near-edge dims 0
        mask_nearedge_dims = mask_invalid_actions[:, : self.n_dim]
        mask_idx_sample = torch.zeros(
            mask_nearedge_dims.shape, device=device, dtype=torch.bool
        )
        mask_idx_sample[idx_sample, :] = True
        mask_nearedge_dims = torch.logical_and(mask_nearedge_dims, mask_idx_sample)
        increments[mask_nearedge_dims] = 0.0
        # Build actions
        actions = [
            tuple(a.tolist() + [m.item()]) for a, m in zip(increments, min_increments)
        ]
        # TODO: implement logprobs here too
        return actions, logprobs

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        is_forward: bool,
        actions: TensorType["n_states", "n_dim"],
        states_target: TensorType["n_states", "policy_input_dim"],
        mask_invalid_actions: TensorType["batch_size", "policy_output_dim"] = None,
        loginf: float = 1000,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.

        For forward transitons, at every state, the probability of the EOS action is
        p(EOS). Otherwise, the probability of an increment incr is p(incr) * (1 -
        p(EOS)). When a dimension is larger than 1 - min_incr, the probabililty of
        incrementing that dimension by 0 is 1.

        For backward transitons, at every state, the probability of the back-to-source
        action is p(back-to-source). Otherwise, the probability of an increment
        (decrement) incr is p(incr) * (1 - p(back-to-source)). When a dimension is
        larger than 1 - min_incr, the probabililty of incrementing that dimension by 0
        must be non-zero and is p(zeroincr). In turn, the probability of sampling a
        non-zero increment incr is (1 - p(zeroincr)) * p(incr).

        Overall, we compute the log probabilities as follows:

        log p = logprobs_eos + logprobs_source + logprobs_increments + logprobs_zeroincr

        - logprobs_eos:
            - 0, that is p(~EOS) = 1 for backward transitions.
            - forward, the log p of the sampled event (EOS or not EOS)

        - logprobs_source:
            - 0, that is p(~source) = 1 for forward transitions.
            - 0, that is p(~source) = 1 for backward transitions when any dimension is
              smaller than min_incr.
            - backward, the log p of the sampled event (source or not source)

        - logprobs_increments:
            - 0, that is p(~increment) = 1 for EOS or source events.
            - otherwise, the log p of sampling the increment.

        - logprobs_zeroincr:
            - 0, that is p(~zeroincr) = 1 for forward transitions.
            - 0, that is p(~zeroincr) = 1 for for dimensions that are smaller than or
              equal to 1 - min_incr, backwards.
            - otherwise, the log p of the sampled event (sampled 0 or not).
        """
        device = policy_outputs.device
        n_states = policy_outputs.shape[0]
        ns_range = torch.arange(n_states).to(device)
        # Determine which states have non-deterministic actions
        if is_forward:
            # EOS is the only valid action if all dimensions are invalid. That is, the
            # action is non-deterministic if any dimension is valid (i.e. mask = False).
            idx_nofix = ns_range[
                torch.any(~mask_invalid_actions[:, : self.n_dim], axis=1)
            ]
        else:
            # The action is non-deterministic if sampling EOS (last value of mask) is
            # invalid (True) and back-to-source (second to last) is not the only action
            # (False).
            idx_nofix = ns_range[
                torch.logical_and(
                    mask_invalid_actions[:, -1], ~mask_invalid_actions[:, -2]
                )
            ]
        # Log probs of EOS and source (backwards) actions
        logprobs_eos = torch.zeros(n_states, device=device, dtype=self.float)
        logprobs_source = torch.zeros(n_states, device=device, dtype=self.float)
        if is_forward:
            mask_eos = torch.all(actions[idx_nofix] == torch.inf, axis=1)
            distr_eos = Bernoulli(logits=policy_outputs[idx_nofix, -1])
            logprobs_eos[idx_nofix] = distr_eos.log_prob(mask_eos.to(self.float))
            mask_sample = ~mask_eos
        else:
            source = torch.tensor(self.source, device=device)
            mask_source = torch.all(states_target[idx_nofix] == source, axis=1)
            distr_source = Bernoulli(logits=policy_outputs[idx_nofix, -2])
            logprobs_source[idx_nofix] = distr_source.log_prob(
                mask_source.to(self.float)
            )
            mask_sample = ~mask_source
        # Log probs of sampled increments
        idx_sample = idx_nofix[mask_sample]
        logprobs_increments = torch.zeros(
            (n_states, self.n_dim), device=device, dtype=self.float
        )
        logprobs_zeroincr = torch.zeros(
            (n_states, self.n_dim), device=device, dtype=self.float
        )
        if len(idx_sample) > 0:
            mix_logits = policy_outputs[idx_sample, self.n_dim : -2 : 3].reshape(
                -1, self.n_dim, self.n_comp
            )
            mix = Categorical(logits=mix_logits)
            alphas = policy_outputs[idx_sample, self.n_dim + 1 : -2 : 3].reshape(
                -1, self.n_dim, self.n_comp
            )
            alphas = self.beta_params_max * torch.sigmoid(alphas) + self.beta_params_min
            betas = policy_outputs[idx_sample, self.n_dim + 2 : -2 : 3].reshape(
                -1, self.n_dim, self.n_comp
            )
            betas = self.beta_params_max * torch.sigmoid(betas) + self.beta_params_min
            beta_distr = Beta(alphas, betas)
            distr_increments = MixtureSameFamily(mix, beta_distr)
            increments = actions[:, :-1].clone().detach()
            logprobs_increments[idx_sample] = distr_increments.log_prob(
                increments[idx_sample]
            )
            # Make logprobs of "invalid" dimensions (value larger than 1 - mincr) 0.
            # TODO: indexing can be done more efficiently to avoid sampling from the
            # distribution above.
            mask_nearedge_dims = ~mask_invalid_actions[:, : self.n_dim]
            mask_idx_sample = torch.zeros(
                mask_nearedge_dims.shape, device=device, dtype=torch.bool
            )
            mask_idx_sample[idx_sample, :] = True
            mask_nearedge_dims = torch.logical_and(mask_nearedge_dims, mask_idx_sample)
            logprobs_increments[mask_nearedge_dims] = 0.0
            # Log probs of sampling zero increments
            if not is_forward:
                mask_zeroincr = increments[mask_nearedge_dims] == 0.0
                logits_zeroincr = policy_outputs[:, : self.n_dim][mask_nearedge_dims]
                distr_zeroincr = Bernoulli(logits=logits_zeroincr)
                logprobs_zeroincr[mask_nearedge_dims] = distr_zeroincr.log_prob(
                    mask_zeroincr.to(self.float)
                )
                # TODO: make logprobs_increments = 0 if increment was zero and
                # near-edge. Already done?
        # Combined probabilities
        sumlogprobs_increments = logprobs_increments.sum(axis=1)
        sumlogprobs_zeroincr = logprobs_zeroincr.sum(axis=1)
        logprobs = (
            logprobs_eos
            + logprobs_source
            + sumlogprobs_increments
            + sumlogprobs_zeroincr
        )
        # Sanity checks
        if is_forward:
            mask_fix = torch.all(mask_invalid_actions[:, : self.n_dim], axis=1)
            assert torch.all(logprobs_source == 0.0)
            assert torch.all(logprobs_zeroincr == 0.0)
            assert torch.all(sumlogprobs_increments[idx_nofix][mask_eos] == 0.0)
            mask_fixdim = mask_invalid_actions[:, self.n_dim]
            assert torch.all(logprobs_increments[mask_fixdim] == 0.0)
        else:
            mask_fix = ~mask_invalid_actions[:, -1]
            assert torch.all(logprobs_eos == 0.0)
            assert torch.all(sumlogprobs_increments[idx_nofix][mask_source] == 0.0)
            assert torch.all(sumlogprobs_zeroincr[idx_nofix][mask_source] == 0.0)
            mask_nozeroincr = mask_invalid_actions[:, self.n_dim]
            assert torch.all(logprobs_zeroincr[mask_nozeroincr] == 0.0)
        assert torch.all(logprobs[mask_fix] == 0.0)
        return logprobs

    def get_log_det_jacobian(
        self, states: TensorType["batch_size", "state_dim"], is_forward: bool
    ):
        """
        Computes the logarithm of the determinant of the Jacobian of the sampled
        actions with respect to the states.

        Forward: the sampled variables are the relative increments r and the state
        updates (s -> s') are:

        s' = s + m + r(1 - s - m)
        r = (s' - s - m) / (1 - s - m)

        Therefore, the derivative of r wrt to s' is

        dr/ds' = 1 / (1 - s - m)

        Backward: the sampled variables are the relative decrements r and the state
        updates (s' -> s) are:

        s = s' - m - r(s' - m)
        r = (s' - s - m) / (s' - m)

        Therefore, the derivative of r wrt to s is

        dr/ds = -1 / (s' - m)

        The derivatives of the components of r with respect to dimensions of s or s'
        other than itself are zero. Therefore, the Jacobian is diagonal and the
        determinant is the product of the diagonal.
        """
        if is_forward:
            return torch.sum(torch.log(1.0 / (1 - states - self.min_incr)), dim=1)
        else:
            return torch.sum(torch.log(-1.0 / (states - self.min_incr)), dim=1)

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
        if self.done:
            return self.state, action, False
        # TODO: remove condition
        # If action is eos or any dimension is beyond max_val, then force eos
        elif action == self.eos or any([s > (1 - self.min_incr) for s in self.state]):
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # If action is not eos, then perform action
        else:
            epsilon = 1e-9
            min_incr = action[-1]
            for dim, incr_rel in enumerate(action[:-1]):
                incr = min_incr + incr_rel * (1.0 - self.state[dim] - min_incr)
                assert incr >= (
                    min_incr - epsilon
                ), f"""
                Increment {incr} at dim {dim} smaller than minimum increment ({min_incr}).
                \nState:\n{self.state}\nAction:\n{action}
                """
                self.state[dim] += incr
            assert all(
                [s <= (self.max_val + epsilon) for s in self.state]
            ), f"""
            State is out of cube bounds.
            \nState:\n{self.state}\nAction:\n{action}\nIncrement: {incr}
            """
            assert all(
                [s >= (0.0 - epsilon) for s in self.state]
            ), f"""
            State is out of cube bounds.
            \nState:\n{self.state}\nAction:\n{action}\nIncrement: {incr}
            """
            self.n_actions += 1
            return self.state, action, True

    def get_grid_terminating_states(self, n_states: int) -> List[List]:
        n_per_dim = int(np.ceil(n_states ** (1 / self.n_dim)))
        linspaces = [np.linspace(0, self.max_val, n_per_dim) for _ in range(self.n_dim)]
        states = list(itertools.product(*linspaces))
        # TODO: check if necessary
        states = [list(el) for el in states]
        return states

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[List]:
        rng = np.random.default_rng(seed)
        states = rng.uniform(low=0.0, high=self.max_val, size=(n_states, self.n_dim))
        return states.tolist()

    #     # TODO: make generic for all environments
    def sample_from_reward(
        self, n_samples: int, epsilon=1e-4
    ) -> TensorType["n_samples", "state_dim"]:
        """
        Rejection sampling  with proposal the uniform distribution in
        [0, max_val]]^n_dim.

        Returns a tensor in GFloNet (state) format.
        """
        samples_final = []
        max_reward = self.proxy2reward(self.proxy.min)
        while len(samples_final) < n_samples:
            samples_uniform = self.statebatch2proxy(
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
