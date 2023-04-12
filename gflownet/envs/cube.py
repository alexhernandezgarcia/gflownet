"""
Classes to represent hyper-cube environments
"""
import itertools
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
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
        Clips the states into [0, max_val]

        Args
        ----
        state : list
            State
        """
        return torch.clip(states, min=0.0, max=self.max_val)

    def statebatch2policy(
        self, states: List[List]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Clips the states into [0, max_val]

        Args
        ----
        state : list
            State
        """
        return self.statetorch2policy(torch.tensor(states, device=self.device))

    def state2policy(self, state: List = None) -> List:
        """
        Clips the state into [0, max_val]
        """
        if state is None:
            state = self.state.copy()
        return [min(max(0.0, s), self.max_val) for s in state]

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
          2) the log(alpha) parameter of the Beta distribution to sample the increment
          3) the log(beta) parameter of the Beta distribution to sample the increment

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
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            mask = [True for _ in range(self.action_space_dim)]
            mask[-1] = False
        # If the value of any dimension is smaller than 0.0, then next action can
        # return to source.
        if any([s < 0.0 for s in self.state]):
            mask = [True for _ in range(self.action_space_dim)]
            mask[-2] = False
        else:
            mask = [False for _ in range(self.action_space_dim)]
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
    Continuous hyper-cube environment (continuous
    version of a hyper-grid) in which the action space consists of the increment of
    each dimension d, modelled by a mixture of Beta distributions.

    The states space is the value of each dimension. If the value of any dimension gets
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action_space(self):
        """
        The actions are tuples of length n_dim, where the value at position d indicates
        the (positive) increment of dimension d.

        Additionally, there are two special discrete actions:
            - Action from the source state, with no minimum increment. Only valid from
              the source state. Indicated by -1 for all dimensions.
            - EOS action. Indicated by np.inf for all dimensions.
        """
        generic_action = tuple([0.0 for _ in range(self.n_dim)])
        self.action_source = tuple([-1.0 for _ in range(self.n_dim)])
        self.eos = tuple([np.inf for _ in range(self.n_dim)])
        actions = [generic_action, self.action_source, self.eos]
        return actions

    def get_policy_output(self, params: dict) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy.

        For each dimension d of the hyper-cube and component c of the mixture, the
        output of the policy should return
          1) the weight of the component in the mixture
          2) the log(alpha) parameter of the Beta distribution to sample the increment
          3) the log(beta) parameter of the Beta distribution to sample the increment

        Additionally, the policy output contains one logit of a Bernoulli distribution
        to model the (discrete) forward probability of selecting the EOS action and the
        (discrete) backward probability of returning to the source node.

        Therefore, the output of the policy model has dimensionality D x C x 3 + 1,
        where D is the number of dimensions (self.n_dim) and C is the number of
        components (self.n_comp).
        """
        policy_output = torch.ones(
            self.n_dim * self.n_comp * 3 + 1,
            device=self.device,
            dtype=self.float,
        )
        policy_output[1:-1:3] = params["beta_alpha"]
        policy_output[2:-1:3] = params["beta_beta"]
        return policy_output

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        Returns a vector with the length of the discrete part of the action space:
        True if action is invalid going forward given the current state, False
        otherwise.

        If the state is the source state, the only valid action is action_source. EOS
        is valid valid from any state (including the source state) and EOS is the only
        possible action if the value of any dimension has excedded max_val.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(self.action_space_dim)]
        # If state is source, the generic action is not valid.
        if all([s == ss for s, ss in zip(state, self.source)]):
            mask = [False for _ in range(self.action_space_dim)]
            mask[0] = True
        # If the value of any dimension is greater than max_val, then next action can
        # only be EOS.
        elif any([s > self.max_val for s in state]):
            mask = [True for _ in range(self.action_space_dim)]
            mask[-1] = False
        # Otherwise, only the action_source is not valid
        else:
            mask = [False for _ in range(self.action_space_dim)]
            mask[-2] = True
        return mask

    def get_mask_invalid_actions_backward(self, state=None, done=None, parents_a=None):
        """
        Returns a vector with the length of the discrete part of the action space:
        True if action is invalid going backward given the current state, False
        otherwise.

        The EOS action (returning to the source state for backward actions) is valid
        from any state. The source action is ignored (invalid) for backward actions. If
        any dimension is smaller than 0, then the only valid action is EOS.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            mask = [True for _ in range(self.action_space_dim)]
            mask[-1] = False
        # If the value of any dimension is smaller than 0.0, then next action can only
        # be return to source (EOS)
        if any([s < 0.0 for s in state]):
            mask = [True for _ in range(self.action_space_dim)]
            mask[-1] = False
        else:
            mask = [False for _ in range(self.action_space_dim)]
        # action_source is ignored going backwards, thus always invalid.
        mask[-2] = True
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
            for dim, incr in enumerate(action):
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
        # EOS
        idx_nofix = ns_range[
            ~torch.logical_and(mask_invalid_actions[:, 0], mask_invalid_actions[:, 1])
        ]
        distr_eos = Bernoulli(logits=policy_outputs[idx_nofix, -1])
        mask_sampled_eos = distr_eos.sample().to(torch.bool)
        logprobs_eos = torch.zeros(n_states, device=device, dtype=self.float)
        logprobs_eos[idx_nofix] = distr_eos.log_prob(mask_sampled_eos.to(self.float))
        # Sample angle increments
        idx_sample = idx_nofix[~mask_sampled_eos]
        n_sample = idx_sample.shape[0]
        logprobs_sample = torch.zeros(n_states, device=device, dtype=self.float)
        increments = torch.inf * torch.ones(
            (n_states, self.n_dim), device=device, dtype=self.float
        )
        if torch.any(idx_sample):
            if sampling_method == "uniform":
                distr_increments = Uniform(
                    torch.zeros(n_sample),
                    torch.ones(n_sample),
                )
            elif sampling_method == "policy":
                mix_logits = policy_outputs[idx_sample, 0:-1:3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                mix = Categorical(logits=mix_logits)
                alphas = policy_outputs[idx_sample, 1:-1:3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                betas = policy_outputs[idx_sample, 2:-1:3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                beta_distr = Beta(torch.exp(alphas), torch.exp(betas))
                distr_increments = MixtureSameFamily(mix, beta_distr)
            increments[idx_sample, :] = distr_increments.sample()
            logprobs_sample[idx_sample] = distr_increments.log_prob(
                increments[idx_sample, :]
            ).sum(axis=1)
            # Apply minimum increment to generic (not from source) actions
            # TODO: before or after computing logprob?
            mask_action_generic = ~mask_invalid_actions[:, 0]
            increments[mask_action_generic] = torch.max(
                increments[mask_action_generic],
                self.min_incr
                * torch.ones(increments[mask_action_generic].shape, device=device),
            )
        # Combined probabilities
        logprobs = logprobs_eos + logprobs_sample
        # Build actions
        actions = [tuple(a.tolist()) for a in increments]
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

        At every state, the EOS action can be sampled with probability p(EOS).
        Otherwise, an increment incr is sampled with probablility
        p(incr) * (1 - p(EOS)).
        """
        device = policy_outputs.device
        n_states = policy_outputs.shape[0]
        ns_range = torch.arange(n_states).to(device)
        # Log probs of EOS actions
        idx_nofix = ns_range[
            ~torch.logical_and(mask_invalid_actions[:, 0], mask_invalid_actions[:, 1])
        ]
        distr_eos = Bernoulli(logits=policy_outputs[idx_nofix, -1])
        mask_sampled_eos = torch.all(actions == torch.inf, axis=1)
        logprobs_eos = torch.zeros(n_states, device=device, dtype=self.float)
        logprobs_eos[idx_nofix] = distr_eos.log_prob(mask_sampled_eos.to(self.float))
        # Log probs of sampled increments
        idx_sample = idx_nofix[~mask_sampled_eos]
        logprobs_sample = torch.zeros(n_states, device=device, dtype=self.float)
        if torch.any(idx_sample):
            mix_logits = policy_outputs[idx_sample, 0:-1:3].reshape(
                -1, self.n_dim, self.n_comp
            )
            mix = Categorical(logits=mix_logits)
            alphas = policy_outputs[idx_sample, 1:-1:3].reshape(
                -1, self.n_dim, self.n_comp
            )
            betas = policy_outputs[idx_sample, 2:-1:3].reshape(
                -1, self.n_dim, self.n_comp
            )
            beta_distr = Beta(torch.exp(alphas), torch.exp(betas))
            distr_increments = MixtureSameFamily(mix, beta_distr)
            # TODO: what to do with the minimum increments, since the logprob will not
            # reflect the true probability of sampling that increment.
            logprobs_sample[idx_sample] = distr_increments.log_prob(
                actions[idx_sample, :]
            ).sum(axis=1)
        # Combined probabilities
        logprobs = logprobs_eos + logprobs_sample
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
        # If action is eos or any dimension is beyond max_val, then force eos
        elif action == self.eos or any([s > self.max_val for s in self.state]):
            self.done = True
            return self.state, self.eos, True
        # If action is not eos, then perform action
        else:
            for dim, incr in enumerate(action):
                self.state[dim] += incr
            return self.state, action, True


    def get_grid_terminating_states(self, n_states: int) -> List[List]:
        n_per_dim = int(np.ceil(n_states ** (1 / self.n_dim)))
        linspaces = [np.linspace(0, self.max_val, n_per_dim) for _ in range(self.n_dim)]
        states = list(itertools.product(*linspaces))
        # TODO: check if necessary
        states = [list(el) for el in states]
        return states
