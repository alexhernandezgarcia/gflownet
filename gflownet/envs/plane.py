"""
Classes to represent hyperplane environments
"""
from typing import List, Tuple
import itertools
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from gflownet.envs.base import GFlowNetEnv
from torch.distributions import Categorical, Uniform, Beta
from torchtyping import TensorType


class Plane(GFlowNetEnv):
    """
    Hyperplane environment (continuous version of a hypergrid) in which the action
    space consists of the increment of dimension d, modelled by a beta distribution.

    The states space is the value of each dimension. If the value of a dimension gets
    larger than max_val, then the trajectory is ended and the reward is 0.

    Attributes
    ----------
    n_dim : int
        Dimensionality of the hyperplane

    length_traj : int
       Fixed length of the trajectory.
    """

    def __init__(
        self,
        n_dim=2,
        max_val=1.0,
        max_traj_length=1.0,
        distr_alpha=2.0,
        distr_beta=5.0,
        env_id=None,
        reward_beta=1,
        reward_norm=1.0,
        reward_norm_std_mult=0,
        reward_func="boltzmann",
        denorm_proxy=False,
        energies_stats=None,
        proxy=None,
        oracle=None,
        **kwargs,
    ):
        super(Plane, self).__init__(
            env_id,
            reward_beta,
            reward_norm,
            reward_norm_std_mult,
            reward_func,
            energies_stats,
            denorm_proxy,
            proxy,
            oracle,
            **kwargs,
        )
        # Main properties
        self.continuous = True
        self.n_dim = n_dim
        self.eos = self.n_dim
        self.max_val = max_val
        self.max_traj_length = max_traj_length
        # Parameters of fixed policy distribution
        self.distr_alpha = distr_alpha
        self.distr_beta = distr_beta
        # Initialize angles and state attributes
        self.source = [0.0 for _ in range(self.n_dim)]
        self.reset()
        self.action_space = self.get_actions_space()
        self.fixed_policy_output = self.get_fixed_policy_output()
        self.policy_output_dim = len(self.fixed_policy_output)
        self.policy_input_dim = len(self.state2policy())
        # Set up proxy
        self.proxy.n_dim = self.n_dim
        self.proxy.setup()
        # Oracle
        self.state2oracle = self.state2proxy
        self.statebatch2oracle = self.statebatch2proxy

    def reward(self, state=None, done=None):
        """
        Sets the reward to min_reward if any value of the state is larger than max_val.
        """
        if done is None:
            done = self.done
        if done:
            return np.array(0.0)
        if state is None:
            state = self.state.copy()
        if any([s > self.max_val for s in self.state]):
            return np.array(self.min_reward)
        else:
            return super().reward(state)

    def reward_batch(self, states, done):
        """
        Sets the reward to min_reward if any value of the state is larger than max_val.
        """
        states_super = []
        done_super = []
        within_plane = []
        for state, d in zip(states, done):
            if d and any([s > self.max_val for s in state]):
                within_plane.append(False)
            else:
                within_plane.append(True)
                states_super.append(state)
                done_super.append(d)
        reward = self.min_reward * np.ones(len(within_plane))
        reward[within_plane] = super().reward_batch(states_super, done_super)
        return reward

    def get_actions_space(self):
        """
        Constructs list with all possible actions. The actions are tuples with two
        values: (dimension, increment) where dimension indicates the index of the
        dimension on which the action is to be performed and increment indicates the
        increment of the dimension value.
        """
        actions = [(d, None) for d in range(self.n_dim)]
        actions += [(self.eos, None)]
        return actions

    def get_fixed_policy_output(self):
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy.

        For each dimension of the hyper-plane, the output of the policy should return
        1) a logit, for the categorical distribution over dimensions and 2) the alpha
        and 3) beta parameters of a beta distribution to sample the increment.
        Therefore, the output of the policy model has dimensionality D x 3 + 1, where D
        is the number of dimensions, and the elements of the output vector are:
        - d * 3: logit of dimension d
        - d * 3 + 1: log(alpha) of beta distribution for dimension d
        - d * 3 + 2: log(beta) of a beta distribution for dimension d
        with d in [0, ..., D]
        """
        policy_output_fixed = np.ones(self.n_dim * 3 + 1)
        policy_output_fixed[1::3] = self.distr_alpha
        policy_output_fixed[2::3] = self.distr_beta
        return policy_output_fixed

    def get_mask_invalid_actions_forward(self, state=None, done=None):
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
            return [True for _ in range(len(self.action_space))]
        if (
            any([s > self.max_val for s in self.state])
            or self.n_actions >= self.max_traj_length
        ):
            mask = [True for _ in range(len(self.action_space))]
            mask[-1] = False
        else:
            mask = [False for _ in range(len(self.action_space))]
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
            mask = [True for _ in range(len(self.action_space))]
            mask[-1] = False
        else:
            mask = [False for _ in range(len(self.action_space))]
        # TODO: review: anything to do with max_value?
        return mask

    def true_density(self):
        # TODO
        # Return pre-computed true density if already stored
        if self._true_density is not None:
            return self._true_density
        # Calculate true density
        all_x = self.get_all_terminating_states()
        all_oracle = self.state2oracle(all_x)
        rewards = self.oracle(all_oracle)
        self._true_density = (
            rewards / rewards.sum(),
            rewards,
            list(map(tuple, all_x)),
        )
        return self._true_density

    def statebatch2proxy(self, states: List[List] = None) -> npt.NDArray[np.float32]:
        """
        Scales the states into [0, max_val]

        Args
        ----
        state : list
            State
        """
        return -1.0 + np.array(states) * 2 / self.max_val

    def state2policy(self, state: List = None) -> List:
        """
        Returns the state as is.
        """
        if state is None:
            state = self.state.copy()
        return state

    def policy2state(self, state_policy: List) -> List:
        """
        Returns the input as is.
        """
        return state_policy

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

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.state = self.source.copy()
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

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
            return [state], [(self.eos, 0.0)]
        else:
            state[action[0]] -= action[1]
            parents = [state]
            return parents, [action]

    def sample_actions(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        sampling_method: str = "policy",
        mask_invalid_actions: TensorType["n_states", "policy_output_dim"] = None,
        temperature_logits: float = 1.0,
        random_action_prob=0.0,
        loginf: float = 1000,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs.
        """
        device = policy_outputs.device
        n_states = policy_outputs.shape[0]
        ns_range = torch.arange(n_states).to(device)
        # Random actions
        n_random = int(n_states * random_action_prob)
        idx_random = torch.randint(high=n_states, size=(n_random,))
        policy_outputs[idx_random, :] = torch.tensor(self.fixed_policy_output).to(
            policy_outputs
        )
        # Sample dimensions
        if sampling_method == "uniform":
            logits_dims = torch.zeros(n_states, self.n_dim).to(device)
        elif sampling_method == "policy":
            logits_dims = policy_outputs[:, 0::3]
            logits_dims /= temperature_logits
        if mask_invalid_actions is not None:
            logits_dims[mask_invalid_actions] = -loginf
        dimensions = Categorical(logits=logits_dims).sample()
        logprobs_dim = self.logsoftmax(logits_dims)[ns_range, dimensions]
        # Sample steps
        ns_range_noeos = ns_range[dimensions != self.eos]
        dimensions_noeos = dimensions[dimensions != self.eos]
        steps = torch.zeros(n_states).to(device)
        logprobs_steps = torch.zeros(n_states).to(device)
        if len(dimensions_noeos) > 0:
            if sampling_method == "uniform":
                distr_steps = Uniform(
                    torch.zeros(len(ns_range_noeos)),
                    self.max_val * torch.ones(len(ns_range_noeos)),
                )
            elif sampling_method == "policy":
                alphas = policy_outputs[:, 1::3][ns_range_noeos, dimensions_noeos]
                betas = policy_outputs[:, 2::3][ns_range_noeos, dimensions_noeos]
                distr_steps = Beta(torch.exp(alphas), torch.exp(betas))
            steps[ns_range_noeos] = distr_steps.sample()
            logprobs_steps[ns_range_noeos] = distr_steps.log_prob(steps[ns_range_noeos])
        # Combined probabilities
        logprobs = logprobs_dim + logprobs_steps
        # Build actions
        actions = [
            (dimension, step)
            for dimension, step in zip(dimensions.tolist(), steps.tolist())
        ]
        return actions, logprobs

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
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

    def make_train_set(self, config):
        """
        Constructs a randomly sampled train set.

        Args
        ----
        """
        if config is None:
            return None
        elif "uniform" in config and "n" in config and config.uniform:
            samples = self.get_grid_terminating_states(config.n)
            energies = self.oracle(self.state2oracle(samples))
        else:
            return None
        df = pd.DataFrame(
            {"samples": [self.state2readable(s) for s in samples], "energies": energies}
        )
        return df

    def make_test_set(self, config):
        """
        Constructs a test set.

        Args
        ----
        """
        if config is None:
            return None
        elif "uniform" in config and "n" in config and config.uniform:
            samples = self.get_grid_terminating_states(config.n)
            energies = self.oracle(self.state2oracle(samples))
        else:
            return None
        df = pd.DataFrame(
            {"samples": [self.state2readable(s) for s in samples], "energies": energies}
        )
        return df

    def get_grid_terminating_states(self, n_states: int) -> List[List]:
        n_per_dim = int(np.ceil(n_states ** (1 / self.n_dim)))
        linspaces = [np.linspace(0, self.max_val, n_per_dim) for _ in range(self.n_dim)]
        states = list(itertools.product(*linspaces))
        # TODO: check if necessary
        states = [list(el) for el in states]
        return states
