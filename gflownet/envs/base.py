"""
Base class of GFlowNet environments
"""
import uuid
from abc import abstractmethod
from copy import deepcopy
from textwrap import dedent
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from torch.distributions import Categorical
from torchtyping import TensorType

from gflownet.utils.common import copy, set_device, set_float_precision, tbool, tfloat


class GFlowNetEnv:
    """
    Base class of GFlowNet environments
    """

    def __init__(
        self,
        device: str = "cpu",
        float_precision: int = 32,
        env_id: Union[int, str] = "env",
        reward_min: float = 1e-8,
        reward_beta: float = 1.0,
        reward_norm: float = 1.0,
        reward_norm_std_mult: float = 0.0,
        reward_func: str = "identity",
        energies_stats: List[int] = None,
        denorm_proxy: bool = False,
        proxy=None,
        oracle=None,
        proxy_state_format: str = "oracle",
        skip_mask_check: bool = False,
        fixed_distribution: Optional[dict] = None,
        random_distribution: Optional[dict] = None,
        conditional: bool = False,
        continuous: bool = False,
        **kwargs,
    ):
        # Flag whether env is conditional
        self.conditional = conditional
        # Flag whether env is continuous
        self.continuous = continuous
        # Call reset() to set initial state, done, n_actions
        self.reset()
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)
        # Reward settings
        self.min_reward = reward_min
        assert self.min_reward > 0
        self.reward_beta = reward_beta
        assert self.reward_beta > 0
        self.reward_norm = reward_norm
        assert self.reward_norm > 0
        self.reward_norm_std_mult = reward_norm_std_mult
        self.reward_func = reward_func
        self.energies_stats = energies_stats
        self.denorm_proxy = denorm_proxy
        # Proxy and oracle
        self.proxy = proxy
        self.setup_proxy()
        if oracle is None:
            self.oracle = self.proxy
        else:
            self.oracle = oracle
        if self.oracle is None or self.oracle.higher_is_better:
            self.proxy_factor = 1.0
        else:
            self.proxy_factor = -1.0
        self.proxy_state_format = proxy_state_format
        # Flag to skip checking if action is valid (computing mask) before step
        self.skip_mask_check = skip_mask_check
        # Log SoftMax function
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        # Action space
        self.action_space = self.get_action_space()
        self.action_space_torch = torch.tensor(
            self.action_space, device=self.device, dtype=self.float
        )
        self.action_space_dim = len(self.action_space)
        # Max trajectory length
        self.max_traj_length = self.get_max_traj_length()
        # Policy outputs
        self.fixed_policy_output = self.get_policy_output(fixed_distribution)
        self.random_policy_output = self.get_policy_output(random_distribution)
        self.policy_output_dim = len(self.fixed_policy_output)
        self.policy_input_dim = len(self.state2policy())

    @abstractmethod
    def get_action_space(self):
        """
        Constructs list with all possible actions (excluding end of sequence)
        """
        pass

    def actions2indices(
        self, actions: TensorType["batch_size", "action_dim"]
    ) -> TensorType["batch_size"]:
        """
        Returns the corresponding indices in the action space of the actions in a batch.
        """
        # Expand the action_space tensor: [batch_size, d_actions_space, action_dim]
        action_space = torch.unsqueeze(self.action_space_torch, 0).expand(
            actions.shape[0], -1, -1
        )
        # Expand the actions tensor: [batch_size, d_actions_space, action_dim]
        actions = torch.unsqueeze(actions, 1).expand(-1, self.action_space_dim, -1)
        # Take the indices at the d_actions_space dimension where all the elements in
        # the action_dim dimension are True
        return torch.where(torch.all(actions == action_space, dim=2))[1]

    def _get_state_done(self, state: Union[List, TensorType["state_dims"]], done: bool):
        """
        A helper method for other methods to determine whether state and done should be
        taken from the arguments or from the instance (self.state and self.done): if
        they are None, they are taken from the instance.

        Args
        ----
        state : list or tensor or None
            None, or a state in GFlowNet format.

        done : bool or None
            None, or whether the environment is done.

        Returns
        -------
        state : list or tensor
            The argument state, or self.state if state is None.

        done: bool
            The argument done, or self.done if done is None.
        """
        if state is None:
            state = copy(self.state)
        if done is None:
            done = self.done
        return state, done

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.
        For continuous or hybrid environments, this mask corresponds to the discrete
        part of the action space.
        """
        return [False for _ in range(self.action_space_dim)]

    def get_mask_invalid_actions_backward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        parents_a: Optional[List] = None,
    ) -> List:
        """
        Returns a list of length the action space with values:
            - True if the backward action is invalid from the current state.
            - False otherwise.
        For continuous or hybrid environments, this mask corresponds to the discrete
        part of the action space.

        The base implementation below should be common to all discrete spaces as it
        relies on get_parents, which is environment-specific and must be implemented.
        Continuous environments will probably need to implement its specific version of
        this method.
        """
        state, done = self._get_state_done(state, done)
        if parents_a is None:
            _, parents_a = self.get_parents(state, done)
        mask = [True for _ in range(self.action_space_dim)]
        for pa in parents_a:
            mask[self.action_space.index(pa)] = False
        return mask

    def get_parents(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        In continuous environments, get_parents() should return only the parent from
        which action leads to state.

        Args
        ----
        state : list
            Representation of a state

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : tuple
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
            return [state], [(self.eos,)]
        parents = []
        actions = []
        return parents, actions

    def _pre_step(
        self, action: Tuple[int], backward: bool = False, skip_mask_check: bool = False
    ) -> Tuple[bool, List[int], Tuple[int]]:
        """
        Performs generic checks shared by the step() and step_backward() (backward must
        be True) methods of all environments.

        Args
        ----
        action : tuple
            Action from the action space.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        do_step : bool
            If True, step() should continue further, False otherwise.
        self.state : list
            The sequence after executing the action

        action : int
            Action index
        """
        # If action not found in action space raise an error
        if action not in self.action_space:
            raise ValueError(
                f"Tried to execute action {action} not present in action space."
            )
        # If backward and state is source, step should not proceed.
        if backward is True:
            if self.equal(self.state, self.source) and action != self.eos:
                return False, self.state, action
        # If forward and env is done, step should not proceed.
        else:
            if self.done:
                return False, self.state, action
        # If action is in invalid mask, step should not proceed.
        if not (self.skip_mask_check or skip_mask_check):
            action_idx = self.action_space.index(action)
            if backward:
                if self.get_mask_invalid_actions_backward()[action_idx]:
                    return False, self.state, action
            else:
                if self.get_mask_invalid_actions_forward()[action_idx]:
                    return False, self.state, action
        return True, self.state, action

    @abstractmethod
    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action from the action space.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action index

        valid : bool
            False, if the action is not allowed for the current state, e.g. stop at the
            root state
        """
        _, self.state, action = self._pre_step(action, skip_mask_check)
        return None, None, None

    def step_backwards(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes a backward step given an action. This generic implementation should
        work for all discrete environments, as it relies on get_parents(). Continuous
        environments should re-implement a custom step_backwards(). Despite being valid
        for any discrete environment, the call to get_parents() may be expensive. Thus,
        it may be advantageous to re-implement step_backwards() in a more efficient
        way as well for discrete environments. Especially, because this generic
        implementation will make two calls to get_parents - once here and one in
        _pre_step() through the call to get_mask_invalid_actions_backward() if
        skip_mask_check is True.

        Args
        ----
        action : tuple
            Action from the action space.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : int
            Action index

        valid : bool
            False, if the action is not allowed for the current state.
        """
        do_step, self.state, action = self._pre_step(action, True, skip_mask_check)
        if not do_step:
            return self.state, action, False
        parents, parents_a = self.get_parents()
        state_next = parents[parents_a.index(action)]
        self.state = state_next
        self.done = False
        self.n_actions -= 1
        return self.state, action, True

    def sample_actions(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        sampling_method: str = "policy",
        mask_invalid_actions: TensorType["n_states", "policy_output_dim"] = None,
        temperature_logits: float = 1.0,
        max_sampling_attempts: int = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs. This implementation
        is generally valid for all discrete environments but continuous environments
        will likely have to implement its own.
        """
        device = policy_outputs.device
        ns_range = torch.arange(policy_outputs.shape[0], device=device)
        if sampling_method == "uniform":
            logits = torch.ones(policy_outputs.shape, dtype=self.float, device=device)
        elif sampling_method == "policy":
            logits = policy_outputs
            logits /= temperature_logits
        if mask_invalid_actions is not None:
            assert not torch.all(mask_invalid_actions), dedent(
                """
            All actions in the mask are invalid.
            """
            )
            logits[mask_invalid_actions] = -torch.inf
        else:
            mask_invalid_actions = torch.zeros(
                policy_outputs.shape, dtype=torch.bool, device=device
            )
        # Make sure that a valid action is sampled, otherwise throw an error.
        for _ in range(max_sampling_attempts):
            action_indices = Categorical(logits=logits).sample()
            if not torch.any(mask_invalid_actions[ns_range, action_indices]):
                break
        else:
            raise ValueError(
                dedent(
                    f"""
            No valid action could be sampled after {max_sampling_attempts} attempts.
            """
                )
            )
        logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        # Build actions
        actions = [self.action_space[idx] for idx in action_indices]
        return actions, logprobs

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        is_forward: bool,
        actions: TensorType["n_states", "actions_dim"],
        states_target: TensorType["n_states", "policy_input_dim"],
        mask_invalid_actions: TensorType["batch_size", "policy_output_dim"] = None,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions. This
        implementation is generally valid for all discrete environments but continuous
        environments will likely have to implement its own.
        """
        device = policy_outputs.device
        ns_range = torch.arange(policy_outputs.shape[0]).to(device)
        logits = policy_outputs
        if mask_invalid_actions is not None:
            logits[mask_invalid_actions] = -torch.inf
        action_indices = (
            torch.tensor(
                [self.action_space.index(tuple(action.tolist())) for action in actions]
            )
            .to(int)
            .to(device)
        )
        logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        return logprobs

    def step_random(self, backward: bool = False):
        """
        Samples a random action and executes the step.

        Returns
        -------
        state : list
            The state after executing the action.

        action : int
            Action, randomly sampled.

        valid : bool
            False, if the action is not allowed for the current state.
        """
        if backward:
            mask_invalid = torch.unsqueeze(
                tbool(self.get_mask_invalid_actions_backward(), device=self.device), 0
            )
        else:
            mask_invalid = torch.unsqueeze(
                tbool(self.get_mask_invalid_actions_forward(), device=self.device), 0
            )
        random_policy = torch.unsqueeze(
            tfloat(
                self.random_policy_output, float_type=self.float, device=self.device
            ),
            0,
        )
        actions, _ = self.sample_actions(
            policy_outputs=random_policy, mask_invalid_actions=mask_invalid
        )
        action = actions[0]
        if backward:
            return self.step_backwards(action)
        return self.step(action)

    def trajectory_random(self):
        """
        Samples and applies a random trajectory on the environment, by sampling random
        actions until an EOS action is sampled.

        Returns
        -------
        state : list
            The final state.

        action: list
            The list of actions (tuples) in the trajectory.
        """
        actions = []
        while self.done is not True:
            _, action, valid = self.step_random()
            if valid:
                actions.append(action)
        return self.state, actions

    def get_policy_output(self, params: Optional[dict] = None):
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy. As a baseline, the policy is uniform over the dimensionality of
        the action space.

        Continuous environments will generally have to overwrite this method.
        """
        return np.ones(self.action_space_dim)

    def state2proxy(self, state: List = None):
        """
        Prepares a state in "GFlowNet format" for the proxy.

        Args
        ----
        state : list
            A state
        """
        if state is None:
            state = self.state.copy()
        return self.statebatch2proxy([state])

    def statebatch2proxy(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Prepares a batch of states in "GFlowNet format" for the proxy.

        Args
        ----
        state : list
            A state
        """
        return np.array(states)

    def statetorch2proxy(
        self, states: TensorType["batch_size", "state_dim"]
    ) -> TensorType["batch_size", "state_proxy_dim"]:
        """
        Prepares a batch of states in torch "GFlowNet format" for the proxy.
        """
        return states

    def state2oracle(self, state: List = None):
        """
        Prepares a list of states in "GFlowNet format" for the oracle

        Args
        ----
        state : list
            A state
        """
        if state is None:
            state = self.state.copy()
        return state

    def statebatch2oracle(self, states: List[List]):
        """
        Prepares a batch of states in "GFlowNet format" for the oracles
        """
        return states

    def statetorch2policy(
        self, states: TensorType["batch_size", "state_dim"]
    ) -> TensorType["batch_size", "policy_input_dim"]:
        """
        Prepares a batch of states in torch "GFlowNet format" for the policy
        """
        return states

    def state2policy(self, state=None):
        """
        Converts a state into a format suitable for a machine learning model, such as a
        one-hot encoding.
        """
        if state is None:
            state = self.state
        return state

    def statebatch2policy(self, states: List[List]) -> npt.NDArray[np.float32]:
        """
        Converts a batch of states into a format suitable for a machine learning model,
        such as a one-hot encoding. Returns a numpy array.
        """
        return np.array(states)

    def policy2state(self, state_policy: List) -> List:
        """
        Converts the model (e.g. one-hot encoding) version of a state given as
        argument into a state.
        """
        return state_policy

    def state2readable(self, state=None):
        """
        Converts a state into human-readable representation.
        """
        if state is None:
            state = self.state
        return str(state)

    def readable2state(self, readable):
        """
        Converts a human-readable representation of a state into the standard format.
        """
        return readable

    def traj2readable(self, traj=None):
        """
        Converts a trajectory into a human-readable string.
        """
        return str(traj).replace("(", "[").replace(")", "]").replace(",", "")

    def reward(self, state=None, done=None):
        """
        Computes the reward of a state
        """
        state, done = self._get_state_done(state, done)
        if done is False:
            return tfloat(0.0, float_type=self.float, device=self.device)
        return self.proxy2reward(self.proxy(self.state2proxy(state))[0])

    def reward_batch(self, states: List[List], done=None):
        """
        Computes the rewards of a batch of states, given a list of states and 'dones'
        """
        if done is None:
            done = np.ones(len(states), dtype=bool)
        states_proxy = self.statebatch2proxy(states)[list(done), :]
        rewards = np.zeros(len(done))
        if states_proxy.shape[0] > 0:
            rewards[list(done)] = self.proxy2reward(self.proxy(states_proxy)).tolist()
        return rewards

    def reward_torchbatch(
        self,
        states: TensorType["batch_size", "state_dim"],
        done: TensorType["batch_size"] = None,
    ):
        """
        Computes the rewards of a batch of states in "GFlownet format"
        """
        if done is None:
            done = torch.ones(states.shape[0], dtype=torch.bool, device=self.device)
        states_proxy = self.statetorch2proxy(states[done, :])
        reward = torch.zeros(done.shape[0], dtype=self.float, device=self.device)
        if states[done, :].shape[0] > 0:
            reward[done] = self.proxy2reward(self.proxy(states_proxy))
        return reward

    def proxy2reward(self, proxy_vals):
        """
        Prepares the output of an oracle for GFlowNet: the inputs proxy_vals is
        expected to be a negative value (energy), unless self.denorm_proxy is True. If
        the latter, the proxy values are first de-normalized according to the mean and
        standard deviation in self.energies_stats. The output of the function is a
        strictly positive reward - provided self.reward_norm and self.reward_beta are
        positive - and larger than self.min_reward.
        """
        if self.denorm_proxy:
            # TODO: do with torch
            proxy_vals = proxy_vals * self.energies_stats[3] + self.energies_stats[2]
        if self.reward_func == "power":
            return torch.clamp(
                (self.proxy_factor * proxy_vals / self.reward_norm) ** self.reward_beta,
                min=self.min_reward,
                max=None,
            )
        elif self.reward_func == "boltzmann":
            return torch.clamp(
                torch.exp(self.proxy_factor * self.reward_beta * proxy_vals),
                min=self.min_reward,
                max=None,
            )
        elif self.reward_func == "identity":
            return torch.clamp(
                self.proxy_factor * proxy_vals,
                min=self.min_reward,
                max=None,
            )
        else:
            raise NotImplementedError

    def reward2proxy(self, reward):
        """
        Converts a "GFlowNet reward" into a (negative) energy or values as returned by
        an oracle.
        """
        if self.reward_func == "power":
            return self.proxy_factor * torch.exp(
                (
                    torch.log(reward)
                    + self.reward_beta * torch.log(torch.as_tensor(self.reward_norm))
                )
                / self.reward_beta
            )
        elif self.reward_func == "boltzmann":
            return self.proxy_factor * torch.log(reward) / self.reward_beta
        elif self.reward_func == "identity":
            return self.proxy_factor * reward
        else:
            raise NotImplementedError

    def reset(self, env_id: Union[int, str] = None):
        """
        Resets the environment.

        Args
        ----
        env_id: int or str
            Unique (ideally) identifier of the environment instance, used to identify
            the trajectory generated with this environment. If None, uuid.uuid4() is
            used.

        Returns
        -------
        self
        """
        self.state = copy(self.source)
        self.n_actions = 0
        self.done = False
        if env_id is None:
            self.id = str(uuid.uuid4())
        else:
            self.id = env_id
        return self

    def set_id(self, env_id: Union[int, str]):
        """
        Sets the id given as argument and returns the environment.

        Args
        ----
        env_id: int or str
            Unique (ideally) identifier of the environment instance, used to identify
            the trajectory generated with this environment.

        Returns
        -------
        self
        """
        self.id = env_id
        return self

    def set_state(self, state: List, done: Optional[bool] = False):
        """
        Sets the state and done of an environment. Environments that cannot be "done"
        at all states (intermediate states are not fully constructed objects) should
        overwrite this method and check for validity.
        """
        self.state = state
        self.done = done
        return self

    def copy(self):
        # return self.__class__(**self.__dict__)
        return deepcopy(self)

    @staticmethod
    def equal(state_x, state_y):
        if torch.is_tensor(state_x) and torch.is_tensor(state_y):
            return torch.equal(state_x, state_y)
        else:
            return state_x == state_y

    def set_energies_stats(self, energies_stats):
        self.energies_stats = energies_stats

    def set_reward_norm(self, reward_norm):
        self.reward_norm = reward_norm

    def get_max_traj_length(self):
        return 1e3

    def get_trajectories(
        self, traj_list, traj_actions_list, current_traj, current_actions
    ):
        """
        Determines all trajectories leading to each state in traj_list, recursively.

        Args
        ----
        traj_list : list
            List of trajectories (lists)

        traj_actions_list : list
            List of actions within each trajectory

        current_traj : list
            Current trajectory

        current_actions : list
            Actions of current trajectory

        Returns
        -------
        traj_list : list
            List of trajectories (lists)

        traj_actions_list : list
            List of actions within each trajectory
        """
        parents, parents_actions = self.get_parents(current_traj[-1], False)
        if parents == []:
            traj_list.append(current_traj)
            traj_actions_list.append(current_actions)
            return traj_list, traj_actions_list
        for idx, (p, a) in enumerate(zip(parents, parents_actions)):
            traj_list, traj_actions_list = self.get_trajectories(
                traj_list, traj_actions_list, current_traj + [p], current_actions + [a]
            )
        return traj_list, traj_actions_list

    def setup_proxy(self):
        if self.proxy:
            self.proxy.setup(self)
