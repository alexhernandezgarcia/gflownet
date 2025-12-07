"""
Base class of GFlowNet environments
"""

import math
import numbers
import random
import uuid
from abc import abstractmethod
from copy import deepcopy
from textwrap import dedent
from typing import List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from torch.distributions import Bernoulli, Categorical
from torchtyping import TensorType

from gflownet.utils.common import (
    copy,
    mixsorteditems,
    set_device,
    set_float_precision,
    tbool,
    tfloat,
    tlong,
)

CMAP = mpl.colormaps["cividis"]
"""
Plotting colour map (cividis).
"""


class GFlowNetEnv:
    """
    Base class of GFlowNet environments
    """

    def __init__(
        self,
        device: str = "cpu",
        float_precision: int = 32,
        env_id: Union[int, str] = "env",
        fixed_distr_params: Optional[dict] = None,
        random_distr_params: Optional[dict] = None,
        skip_mask_check: bool = False,
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
        # Flag to skip checking if action is valid (computing mask) before step
        self.skip_mask_check = skip_mask_check
        # Log SoftMax function
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        # Action space
        self.action_space = self.get_action_space()
        self._action2index = {a: idx for idx, a in enumerate(self.action_space)}
        self.action_space_torch = torch.tensor(
            self.action_space, device=self.device, dtype=self.float
        )
        # Mask dimensionality
        self._mask_dim = self._compute_mask_dim()
        # Max trajectory length
        self._max_traj_length = self._get_max_trajectory_length()
        # Policy outputs
        self.fixed_distr_params = fixed_distr_params
        self.random_distr_params = random_distr_params
        self.fixed_policy_output = self.get_policy_output(self.fixed_distr_params)
        self.random_policy_output = self.get_policy_output(self.random_distr_params)
        self.policy_output_dim = len(self.fixed_policy_output)
        self.policy_input_dim = len(self.state2policy())

    @abstractmethod
    def get_action_space(self):
        """
        Constructs list with all possible actions (excluding end of sequence)
        """
        pass

    @property
    def action_space_dim(self) -> int:
        """
        Returns the dimensionality of the action space (number of actions).

        Returns
        -------
        The number of actions in the action space.
        """
        return len(self.action_space)

    @property
    def mask_dim(self):
        """
        Returns the dimensionality of the masks.

        Returns
        -------
        The dimensionality of the masks.
        """
        return self._mask_dim

    def _compute_mask_dim(self) -> int:
        """
        Calculates the mask dimensionality.

        By default, the mask dimensionality is equal to the dimensionality of the
        action space.

        This method should be overriden in environments where this may not be the case,
        for example continuous environments (ContinuousCube) and meta-environments such
        as Stack and Set.

        Returns
        -------
        int
            The number of elements in the masks.
        """
        return self.action_space_dim

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment, including the EOS
        action.

        While it is not required to override this method because it does return a
        default value of 100, it is recommended to override it to return the correct
        value or an upper bound as tight as possible to  the maximum.

        The maximum trajectory length does not play a critical role but it is used for
        testing purposes. For example, it is used by get_random_states(), and poor
        estimation of the trajectory length could result in stark inefficiency.
        """
        return 100

    @property
    def max_traj_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment, including the EOS
        action.

        Returns
        -------
        The maximum number of steps in a trajectory of the environment.
        """
        return self._max_traj_length

    def action2representative(self, action: Tuple) -> int:
        """
        For continuous or hybrid environments, converts a continuous action into its
        representative in the action space. Discrete actions remain identical, thus
        fully discrete environments do not need to re-implement this method.
        Continuous environments should re-implement this method in order to replace
        continuous actions by their representatives in the action space.
        """
        return action

    def action2index(self, action: Tuple) -> int:
        """
        Returns the index in the action space of the action passed as an argument, or
        its representative if it is a continuous action.

        The method uses the dictionary lookup ``self._action2index``.

        See: self.action2representative()

        Parameters
        ----------
        action : tuple
            An action from the action space.

        Returns
        -------
        int
            The index of the action in the action space.
        """
        return self._action2index[self.action2representative(action)]

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

    def _get_state(self, state: Union[List, TensorType["state_dims"]]):
        """
        A helper method for other methods to determine whether state should be taken
        from the arguments or from the instance (self.state): if is None, it is taken
        from the instance.

        Args
        ----
        state : list or tensor or None
            None, or a state in GFlowNet format.

        Returns
        -------
        state : list or tensor
            The argument state, or self.state if state is None.
        """
        if state is None:
            state = copy(self.state)
        return state

    def _get_done(self, done: bool):
        """
        A helper method for other methods to determine whether done should be taken
        from the arguments or from the instance (self.done): if it is None, it is taken
        from the instance.

        Args
        ----
        done : bool or None
            None, or whether the environment is done.

        Returns
        -------
        done: bool
            The argument done, or self.done if done is None.
        """
        if done is None:
            done = self.done
        return done

    def is_source(
        self, state: Optional[Union[List, TensorType["state_dims"]]] = None
    ) -> bool:
        """
        Returns True if the environment's state or the state passed as parameter (if
        not None) is the source state of the environment.

        Parameters
        ----------
        state : list or tensor or None
            None, or a state in environment format.

        Returns
        -------
        bool
            Whether the state is the source state of the environment
        """
        state = self._get_state(state)
        return self.equal(state, self.source)

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
        state = self._get_state(state)
        done = self._get_done(done)
        if parents_a is None:
            _, parents_a = self.get_parents(state, done)
        mask = [True for _ in range(self.action_space_dim)]
        for pa in parents_a:
            mask[self.action_space.index(pa)] = False
        return mask

    def get_mask(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        backward: Optional[bool] = False,
    ) -> List:
        """
        Returns a mask of invalid actions given a state and a done value. Depending on
        backward, either the forward or the backward mask is returned, by calling the
        corresponding method.
        """
        if backward:
            return self.get_mask_invalid_actions_backward(state, done)
        else:
            return self.get_mask_invalid_actions_forward(state, done)

    def get_valid_actions(
        self,
        mask: Optional[bool] = None,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        backward: Optional[bool] = False,
    ) -> List[Tuple]:
        """
        Returns the list of non-invalid (valid, for short) according to the mask of
        invalid actions.

        More documentation about the meaning and use of invalid actions can be found in
        gflownet/envs/README.md.
        """
        if mask is None:
            mask = self.get_mask(state, done, backward)
        return [action for action, m in zip(self.action_space, mask) if not m]

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

    # TODO: consider returning only do_step
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
        # If action is in invalid mask (not in valid actions), step should not proceed.
        if not (self.skip_mask_check or skip_mask_check):
            if action not in self.get_valid_actions(backward=backward):
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
        self.set_state(state_next, done=False)
        self.n_actions += 1
        return self.state, action, True

    def randomize_and_temper_sampling_distribution(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        probability_random_action: Optional[float] = 0.0,
        temperature: Optional[float] = 1.0,
    ) -> TensorType["n_states", "policy_output_dim"]:
        """
        Replaces the rows of `policy_outputs` by a vector corresponding to a random
        sampling policy with the probability indicated by `probability_random_action`.

        Note that the tensor of policy outputs is not cloned if neither tempering nor
        random actions are incorporated. This implies that the original tensor of
        policy outputs may be modified by subsequent methods (namely
        sample_actions_batch()), for example to mask the invalid actions.

        Parameters
        ----------
        policy_outputs : tensor
            The original outputs of the sampling policy. For example, they may
            correspond to the output (logits) of the GFlowNet policy model.
        probability_random_action : float, optional
            The probability of sampling a random action. If larger than one, the logits
            will be replaced by a random policy vector with this probability, according
            to Bernoulli distribution. By default, the probability is 0.0 (no random
            actions).
        temperature : float, optional
            A scalar by which the logits are divided to adjust the sampling
            distribution. A temperature larger than one will result in a flatter
            distribution, favouring exploration. A temperature smaller than one will
            sharpen the distribution, favouring concentration around high probability
            actions. By default, the temperature is 1.0 (no tempering).

        Returns
        -------
        policy_outputs : tensor
            The modified policy outputs.
        """
        if not math.isclose(temperature, 1.0, abs_tol=1e-08):
            do_temper = True
        else:
            do_temper = False
        if not math.isclose(probability_random_action, 0.0, abs_tol=1e-08):
            do_random = True
        else:
            do_random = False
        if not do_temper and not do_random:
            return policy_outputs

        # Clone the sampling logits in order not to change the original tensor
        logits_sampling = policy_outputs.clone().detach()
        if do_temper:
            logits_sampling /= temperature
        if do_random:
            idx_random = tbool(
                Bernoulli(
                    probability_random_action * torch.ones(policy_outputs.shape[0])
                ).sample(),
                device=self.device,
            )
            logits_sampling[idx_random, :] = self.random_policy_output
        return policy_outputs

    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "policy_output_dim"]] = None,
        states_from: Optional[List] = None,
        is_backward: Optional[bool] = False,
        random_action_prob: Optional[float] = 0.0,
        temperature_logits: Optional[float] = 1.0,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs.

        This implementation is generally valid for all discrete environments but
        continuous or mixed environments need to reimplement this method.

        The method is valid for both forward and backward actions in the case of
        discrete environments. Some continuous environments may also be agnostic to the
        difference between forward and backward actions since the necessary information
        can be contained in the mask. However, some continuous environments do need to
        know whether the actions are forward of backward, which is why this can be
        specified by the argument is_backward.

        Most environments do not need to know the states from which the actions are to
        be sampled since the necessary information is in both the policy outputs and
        the mask. However, some continuous environments do need to know the originating
        states in order to construct the actions, which is why one of the arguments is
        states_from.

        Note that methods overriding this method should randomize and temper the
        logits.

        Parameters
        ----------
        policy_outputs : tensor
            The output of the GFlowNet policy model.
        mask : tensor
            The mask of invalid actions. For continuous or mixed environments, the mask
            may be tensor with an arbitrary length contaning information about special
            states, as defined elsewhere in the environment.
        states_from : tensor
            The states originating the actions, in GFlowNet format. Ignored in discrete
            environments and only required in certain continuous environments.
        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default). Ignored in discrete environments and only required in certain
            continuous environments.
        random_action_prob : float, optional
            The probability of sampling a random action. If larger than one, the model
            outputs will be replaced by a random policy vector with probability
            `random_action_prob`, according to Bernoulli distribution.
        temperature_logits : float, optional
            A scalar by which the model outputs are divided to temper the sampling
            distribution.

        Returns
        -------
        actions : list
            The list of sampled actions.
        """
        # Randomize actions and temper the logits
        logits_sampling = self.randomize_and_temper_sampling_distribution(
            policy_outputs, random_action_prob, temperature_logits
        )

        # Make the logits of invalid actions equal to -inf.
        if mask is not None:
            if torch.all(mask, dim=1).any():
                raise RuntimeError(
                    "All actions in the mask are invalid for some states in the batch."
                )
            logits_sampling[mask] = -torch.inf

        # Sample actions from the Categorical distributions defined by the logits
        action_indices = Categorical(logits=logits_sampling).sample()
        # Build actions
        actions = [self.action_space[idx] for idx in action_indices]
        return actions

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: Union[List, TensorType["n_states", "action_dim"]],
        mask: TensorType["batch_size", "policy_output_dim"] = None,
        states_from: Optional[List] = None,
        is_backward: bool = False,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions. This
        implementation is generally valid for all discrete environments but continuous
        environments will likely have to implement its own.

        Parameters
        ----------
        policy_outputs : tensor
            The output of the GFlowNet policy model.
        mask : tensor
            The mask of invalid actions. For continuous or mixed environments, the mask
            may be tensor with an arbitrary length contaning information about special
            states, as defined elsewhere in the environment.
        actions : list or tensor
            The actions from each state in the batch for which to compute the log
            probability. The actions may be a list or a tensor. Most environments
            handle the actions as a list, but in some cases it is practical to use a
            tensor for easier indexing, such as in meta-environments.
        states_from : tensor
            The states originating the actions, in GFlowNet format. Ignored in discrete
            environments and only required in certain continuous environments.
        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default). Ignored in discrete environments and only required in certain
            continuous environments.
        """
        device = policy_outputs.device
        ns_range = torch.arange(policy_outputs.shape[0]).to(device)
        logits = policy_outputs.clone()
        if mask is not None:
            logits[mask] = -torch.inf
        if torch.is_tensor(actions):
            action_indices = tlong(self.actions2indices(actions), device=self.device)
        else:
            action_indices = tlong(
                [self.action2index(action) for action in actions],
                device=self.device,
            )
        logprobs = self.logsoftmax(logits)[ns_range, action_indices]
        return logprobs

    def step_random(self, backward: bool = False):
        """
        Samples a random action and executes the step.

        Parameters
        ----------
        backward : bool
            If True, the step is performed backwards. False by default.

        Returns
        -------
        state : list
            The state after executing the action.

        action : int
            Action, randomly sampled.

        valid : bool
            False, if the action is not allowed for the current state.
        """
        mask_invalid = tbool(
            self.get_mask(backward=backward), device=self.device
        ).unsqueeze(0)
        action = self.sample_actions_batch(
            self.random_policy_output.clone().unsqueeze(0),
            mask_invalid,
            [self.state],
            backward,
        )[0]
        if backward:
            return self.step_backwards(action)
        return self.step(action)

    def trajectory_random(self, backward: bool = False):
        """
        Samples and applies a random trajectory on the environment, by sampling random
        actions until an EOS action is sampled.

        Parameters
        ----------
        backward : bool
            If True, the trajectory is sampled backwards. False by default.

        Returns
        -------
        state : list
            The final state.

        action: list
            The list of actions (tuples) in the trajectory.
        """
        actions = []
        while True:
            _, action, valid = self.step_random(backward)
            if valid:
                actions.append(action)
            if backward and self.is_source():
                break
            elif self.done:
                break
            else:
                continue
        return self.state, actions

    def get_random_terminating_states(
        self, n_states: int, unique: bool = True, max_attempts: int = 100000
    ) -> List:
        """
        Samples n terminating states by using the random policy of the environment
        (calling self.trajectory_random()).

        Note that this method is general for all environments but it may be suboptimal
        in terms of efficiency. In particular, 1) it samples full trajectories in order
        to get terminating states, 2) if unique is True, it needs to compare each newly
        sampled state with all the previously sampled states. If
        get_uniform_terminating_states is available, it may be preferred, or for some
        environments, a custom get_random_terminating_states may be straightforward to
        implement in a much more efficient way.

        Args
        ----
        n_states : int
            The number of terminating states to sample.

        unique : bool
            Whether samples should be unique. True by default.

        max_attempts : int
            The maximum number of attempts, to prevent the method from getting stuck
            trying to obtain n_states different samples if unique is True. 100000 by
            default, therefore if more than 100000 are requested, max_attempts should
            be increased accordingly.

        Returns
        -------
        states : list
            A list of randomly sampled terminating states.
        """
        if unique is False:
            max_attempts = n_states + 1
        states = []
        count = 0
        while len(states) < n_states and count < max_attempts:
            add = True
            self.reset()
            state, _ = self.trajectory_random()
            if unique is True:
                if any([self.equal(state, s) for s in states]):
                    add = False
            if add is True:
                states.append(state)
            count += 1
        return states

    def get_random_states(
        self,
        n_states: int,
        unique: bool = True,
        exclude_source: bool = False,
        max_attempts: int = 1000,
    ) -> List:
        """
        Samples n states (not necessarily terminating) by using the random policy of
        the environment (calling self.step_random()).

        It relies on self.max_traj_length in order to uniformly sample the number
        of steps, in order to obtain states with varying trajectory lengths.

        The method iteratively samples first a trajectory length and attempts to
        perform as many steps. If the trajectory ends before the requested number of
        steps is reached, then it is discarded and a new one is attempted.

        This may introduced a bias towards states that can be reached with a few steps.

        Note that this method is general for all environments but it may be suboptimal
        in terms of efficiency. In particular, 1) it samples trajectories step by step
        in order to get random states, 2) if unique is True, it needs to compare each
        newly sampled state with all the previously sampled states, 3) states are
        copied before adding them to the list, 4) only the last state of a trajectory
        is added to the list in order to have diversity of trajectories.

        Parameters
        ----------
        n_states : int
            The number of terminating states to sample.
        unique : bool
            Whether samples should be unique. True by default.
        max_attempts : int
            The maximum number of attempts, to prevent the method from getting stuck
            trying to obtain n_states different samples if unique is True. 100000 by
            default, therefore if more than 100000 are requested, max_attempts should
            be increased accordingly.
        exclude_source : bool
            If True, exclude the source state from the list of states.

        Returns
        -------
        states : list
            A list of randomly sampled states.

        Raises
        ------
        ValueError
            If max_attempts is smaller than n_states
        RuntimeError
            If the maximum number of attempts is reached before obtaining the requested
            number of unique states.
        """
        max_traj_length = self.max_traj_length
        if max_attempts < n_states:
            raise ValueError(
                f"max_attempts (received {max_attempts})  must larger than or "
                f"equal to n_states (received {n_states})."
            )
        states = []
        n_attempts = 0
        # Iterate until the requested number of states is obtained
        while len(states) < n_states:
            n_attempts += 1
            # Sample a trajectory length for this state
            traj_length = random.randint(1, max_traj_length)
            self.reset()
            is_valid = True
            for _ in range(traj_length):
                # If the trajectory has reached done before the number of requested
                # steps, discard it and start a new one.
                if self.done:
                    is_valid = False
                    break
                # Perform a random step
                self.step_random()

            # If exclude_source is True and the state is the source, mark the
            # trajectory as invalid.
            if is_valid and exclude_source and self.is_source(self.state):
                is_valid = False
            # If unique is True and the state is in the list, mark the trajetory as
            # invalid
            if is_valid and unique and any([self.equal(self.state, s) for s in states]):
                is_valid = False
            # If the trajectory is valid, add the state to the list
            if is_valid:
                states.append(copy(self.state))

            # Check if the number of attempts has reached the maximum
            if n_attempts >= max_attempts:
                raise RuntimeError(
                    f"Reached the maximum number of attempts ({max_attempts}) to "
                    f"sample {n_states} states but only {len(states)} could "
                    "be obtained. It is possible that the state space is too small "
                    f"to contain {n_states} states. Otherwise, consider "
                    "increasing the number of attempts"
                )

        return states

    def get_policy_output(
        self, params: Optional[dict] = None
    ) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy. As a baseline, the policy is uniform over the dimensionality of
        the action space.

        Continuous environments will generally have to overwrite this method.
        """
        return torch.ones(self.action_space_dim, dtype=self.float, device=self.device)

    def states2proxy(
        self, states: Union[List[List], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states in "environment format" for the proxy. By default,
        the batch of states is converted into a tensor with float dtype and returned as
        is.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        return tfloat(states, device=self.device, float_type=self.float)

    def state2proxy(
        self, state: Union[List, TensorType["state_dim"]] = None
    ) -> TensorType["state_proxy_dim"]:
        """
        Prepares a single state in "GFlowNet format" for the proxy. By default, simply
        states2proxy is called and the output will be a "batch" with a single state in
        the proxy format.

        Args
        ----
        state : list
            A state
        """
        state = self._get_state(state)
        return self.states2proxy([state])

    def states2policy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: By
        default, the batch of states is converted into a tensor with float dtype and
        returned as is.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        return tfloat(states, device=self.device, float_type=self.float)

    def state2policy(
        self, state: Union[List, TensorType["state_dim"]] = None
    ) -> TensorType["policy_input_dim"]:
        """
        Prepares a state in "GFlowNet format" for the policy model. By default,
        states2policy is called, which by default will return the state as is.

        Args
        ----
        state : list
            A state
        """
        state = self._get_state(state)
        return torch.squeeze(self.states2policy([state]), dim=0)

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
        self.state = copy(state)
        self.done = done
        return self

    def copy(self):
        # return self.__class__(**self.__dict__)
        return deepcopy(self)

    @staticmethod
    def equal(state_x, state_y):
        if isinstance(state_x, numbers.Number) or isinstance(state_x, str):
            return state_x == state_y
        if type(state_x) != type(state_y):
            return False
        if torch.is_tensor(state_x) and torch.is_tensor(state_y):
            # Check for nans because (torch.nan == torch.nan) == False
            x_nan = torch.isnan(state_x)
            if torch.any(x_nan):
                y_nan = torch.isnan(state_y)
                if not torch.equal(x_nan, y_nan):
                    return False
                return torch.equal(state_x[~x_nan], state_y[~y_nan])
            return torch.equal(state_x, state_y)
        if isinstance(state_x, dict) and isinstance(state_y, dict):
            if len(state_x) != len(state_y):
                return False
            return all(
                [
                    key_x == key_y and GFlowNetEnv.equal(value_x, value_y)
                    for (key_x, value_x), (key_y, value_y) in zip(
                        mixsorteditems(state_x), mixsorteditems(state_y)
                    )
                ]
            )
        if (isinstance(state_x, list) and isinstance(state_y, list)) or (
            isinstance(state_x, tuple) and isinstance(state_y, tuple)
        ):
            if len(state_x) != len(state_y):
                return False
            if len(state_x) == 0:
                return True
            if isinstance(state_x[0], numbers.Number) or isinstance(state_x[0], str):
                value_type = type(state_x[0])
                if all([isinstance(sx, value_type) for sx in state_x]) and all(
                    [isinstance(sy, value_type) for sy in state_y]
                ):
                    return state_x == state_y
        return all([GFlowNetEnv.equal(sx, sy) for sx, sy in zip(state_x, state_y)])

    @staticmethod
    def isclose(state_x, state_y, atol=1e-8):
        if isinstance(state_x, numbers.Number) or isinstance(state_x, str):
            return np.isclose(state_x, state_y, atol=atol)
        if type(state_x) != type(state_y):
            return False
        if torch.is_tensor(state_x) and torch.is_tensor(state_y):
            # Check for nans because (torch.nan == torch.nan) == False
            x_nan = torch.isnan(state_x)
            if torch.any(x_nan):
                y_nan = torch.isnan(state_y)
                if not torch.equal(x_nan, y_nan):
                    return False
                return torch.all(
                    torch.isclose(state_x[~x_nan], state_y[~y_nan], atol=atol)
                )
            return torch.equal(state_x, state_y)
        if isinstance(state_x, dict) and isinstance(state_y, dict):
            if len(state_x) != len(state_y):
                return False
            return all(
                [
                    key_x == key_y and GFlowNetEnv.isclose(value_x, value_y)
                    for (key_x, value_x), (key_y, value_y) in zip(
                        mixsorteditems(state_x), mixsorteditems(state_y)
                    )
                ]
            )
        if (isinstance(state_x, list) and isinstance(state_y, list)) or (
            isinstance(state_x, tuple) and isinstance(state_y, tuple)
        ):
            if len(state_x) != len(state_y):
                return False
            if len(state_x) == 0:
                return True
            if isinstance(state_x[0], numbers.Number) or isinstance(state_x[0], str):
                value_type = type(state_x[0])
                if all([isinstance(sx, value_type) for sx in state_x]) and all(
                    [isinstance(sy, value_type) for sy in state_y]
                ):
                    return np.all(np.isclose(state_x, state_y, atol=atol))
        return all([GFlowNetEnv.isclose(sx, sy) for sx, sy in zip(state_x, state_y)])

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

    @torch.no_grad()
    def compute_train_energy_proxy_and_rewards(self):
        """
        Gather batched proxy data:

        * The ground-truth energy of the train set
        * The predicted proxy energy over the train set
        * The reward version of those energies (with env.proxy2reward)

        Returns
        -------
        gt_energy : torch.Tensor
            The ground-truth energies in the proxy's train set

        proxy_energy : torch.Tensor
            The proxy's predicted energies over its train set

        gt_reward : torch.Tensor
            The reward version of the ground-truth energies

        proxy_reward : torch.Tensor
            The reward version of the proxy's predicted energies
        """
        gt_energy, proxy_energy = self.proxy.infer_on_train_set()
        gt_reward = self.proxy2reward(gt_energy)
        proxy_reward = self.proxy2reward(proxy_energy)

        return gt_energy, proxy_energy, gt_reward, proxy_reward

    def mask_conditioning(
        self, mask: Union[List[bool], TensorType["mask_dim"]], env_cond, backward: bool
    ):
        """
        Conditions the input mask based on the restrictions imposed by a conditioning
        environment, env_cond.

        It is assumed that the state space of the conditioning environment is a subset
        of the state space of the original environment (self). The conditioning
        mechanism goes as follows: given a state, its corresponding mask and a
        conditioning environment, the mask of invalid actions is updated such that all
        actions that would be invalid in the conditioning environment are made invalid,
        even though they may not be invalid in the original environment.
        """
        # Set state in conditional environment
        env_cond.reset()
        env_cond.set_state(self.state, self.done)
        # If the environment is continuous, then we simply return the mask of the
        # conditioning environment. It is thus assumed that the dimensionality and
        # interpretation is the same.
        if self.continuous:
            return env_cond.get_mask(backward=backward)
        # Get valid actions common to both the original and the conditioning env
        actions_valid_orig = self.get_valid_actions(mask)
        actions_valid_cond = env_cond.get_valid_actions(backward=backward)
        actions_valid = set(actions_valid_orig).intersection(set(actions_valid_cond))
        # Construct new mask by setting to False (valid or not invalid) the actions
        # that are valid to both environments
        mask = [True] * self.mask_dim
        for action in actions_valid:
            mask[self.action_space.index(action)] = False
        return mask

    @torch.no_grad()
    def top_k_metrics_and_plots(
        self,
        states,
        top_k,
        name,
        energy=None,
        reward=None,
        step=None,
        **kwargs,
    ):
        """
        Compute top_k metrics and plots for the given states.

        In particular, if no states, energy, or reward are passed, then the name
        *must* be "train", and the energy and reward will be computed from the
        proxy using ``env.compute_train_energy_proxy_and_rewards()``. In this case,
        ``top_k_metrics_and_plots`` will be called a second time to compute the
        metrics and plots of the proxy distribution in addition to the ground-truth
        distribution.
        Train mode should only be called once at the begining of training as
        distributions do not change over time.

        If ``states`` are passed, then the energy and reward will be computed from the
        proxy for those states. They are typically sampled from the current GFN.

        Otherwise, energy and reward should be passed directly.

        *Plots and metrics*:
        - mean+std of energy and reward
        - mean+std of top_k energy and reward
        - histogram of energy and reward
        - histogram of top_k energy and reward


        Args
        ----
        states: list
            List of states to compute metrics and plots for.

        top_k: int
            Number of top k states to compute metrics and plots for.
            "top" means lowest energy/highest reward.

        name: str
            Name of the distribution to compute metrics and plots for.
            Typically "gflownet", "random" or "train". Will be used in
            metrics names like ``f"Mean {name} energy"``.

        energy: torch.Tensor, optional
            Batch of pre-computed energies

        reward: torch.Tensor, optional
            Batch of pre-computed rewards

        step: int, optional
            Step number to use for the plot title.

        Returns
        -------
        metrics: dict
            Dictionary of metrics: str->float

        figs: list
            List of matplotlib figures

        figs_names: list
            List of figure names for ``figs``
        """

        if states is None and energy is None and reward is None:
            assert name == "train"
            (
                energy,
                proxy,
                energy_reward,
                proxy_reward,
            ) = self.compute_train_energy_proxy_and_rewards()
            name = "train ground truth"
            reward = energy_reward
        elif energy is None and reward is None:
            # TODO: fix this
            x = torch.stack([self.state2proxy(s) for s in states])
            energy = self.proxy(x.to(self.device)).cpu()
            reward = self.proxy2reward(energy)

        assert energy is not None and reward is not None

        # select top k best energies and rewards
        top_k_e = torch.topk(energy, top_k, largest=False, dim=0).values.numpy()
        top_k_r = torch.topk(reward, top_k, largest=True, dim=0).values.numpy()

        # find best energy and reward
        best_e = torch.min(energy).item()
        best_r = torch.max(reward).item()

        # to numpy to plot
        energy = energy.numpy()
        reward = reward.numpy()

        # compute stats
        mean_e = np.mean(energy)
        mean_r = np.mean(reward)

        std_e = np.std(energy)
        std_r = np.std(reward)

        mean_top_k_e = np.mean(top_k_e)
        mean_top_k_r = np.mean(top_k_r)

        std_top_k_e = np.std(top_k_e)
        std_top_k_r = np.std(top_k_r)

        # automatic color scale
        # currently: cividis colour map
        colors = ["full", "top_k"]
        normalizer = mpl.colors.Normalize(vmin=0, vmax=len(colors) - 0.5)
        colors = {k: CMAP(normalizer(i)) for i, k in enumerate(colors[::-1])}

        # two sublopts: left is energy, right is reward
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # energy full distribution and stats lines
        ax[0].hist(
            energy,
            bins=100,
            alpha=0.35,
            label=f"All = {len(energy)}",
            color=colors["full"],
            density=True,
        )
        ax[0].axvline(
            mean_e,
            color=colors["full"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_e:.3f}",
        )
        ax[0].axvline(
            mean_e + std_e,
            color=colors["full"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_e:.3f}",
        )
        ax[0].axvline(
            mean_e - std_e,
            color=colors["full"],
            linestyle=(0, (1, 10)),
        )

        # energy top k distribution and stats lines
        ax[0].hist(
            top_k_e,
            bins=100,
            alpha=0.7,
            label=f"Top k = {top_k}",
            color=colors["top_k"],
            density=True,
        )
        ax[0].axvline(
            mean_top_k_e,
            color=colors["top_k"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_top_k_e:.3f}",
        )
        ax[0].axvline(
            mean_top_k_e + std_top_k_e,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_top_k_e:.3f}",
        )
        ax[0].axvline(
            mean_top_k_e - std_top_k_e,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
        )
        # energy title & legend
        ax[0].set_title(
            f"Energy distribution for {top_k} vs {len(energy)}"
            + f" samples\nBest: {best_e:.3f}",
            y=0,
            pad=-20,
            verticalalignment="top",
            size=12,
        )
        ax[0].legend()

        # reward full distribution and stats lines
        ax[1].hist(
            reward,
            bins=100,
            alpha=0.35,
            label=f"All = {len(reward)}",
            color=colors["full"],
            density=True,
        )
        ax[1].axvline(
            mean_r,
            color=colors["full"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_r:.3f}",
        )
        ax[1].axvline(
            mean_r + std_r,
            color=colors["full"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_r:.3f}",
        )
        ax[1].axvline(
            mean_r - std_r,
            color=colors["full"],
            linestyle=(0, (1, 10)),
        )

        # reward top k distribution and stats lines
        ax[1].hist(
            top_k_r,
            bins=100,
            alpha=0.7,
            label=f"Top k = {top_k}",
            color=colors["top_k"],
            density=True,
        )
        ax[1].axvline(
            mean_top_k_r,
            color=colors["top_k"],
            linestyle=(0, (5, 10)),
            label=f"Mean = {mean_top_k_r:.3f}",
        )
        ax[1].axvline(
            mean_top_k_r + std_top_k_r,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
            label=f"Std = {std_top_k_r:.3f}",
        )
        ax[1].axvline(
            mean_top_k_r - std_top_k_r,
            color=colors["top_k"],
            linestyle=(0, (1, 10)),
        )

        # reward title & legend
        ax[1].set_title(
            f"Reward distribution for {top_k} vs {len(reward)}"
            + f" samples\nBest: {best_r:.3f}",
            y=0,
            pad=-20,
            verticalalignment="top",
            size=12,
        )
        ax[1].legend()

        # Finalize figure
        title = f"{name.capitalize()} energy and reward distributions"
        if step is not None:
            title += f" (step {step})"
        fig.suptitle(title, y=0.95)
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])

        # store metrics
        metrics = {
            f"Mean {name} energy": mean_e,
            f"Std {name} energy": std_e,
            f"Mean {name} reward": mean_r,
            f"Std {name} reward": std_r,
            f"Mean {name} top k energy": mean_top_k_e,
            f"Std {name} top k energy": std_top_k_e,
            f"Mean {name} top k reward": mean_top_k_r,
            f"Std {name} top k reward": std_top_k_r,
            f"Best (min) {name} energy": best_e,
            f"Best (max) {name} reward": best_r,
        }
        figs = [fig]
        fig_names = [title]

        if name.lower() == "train ground truth":
            # train stats mode: the ground truth data has meen plotted
            # and computed, let's do it again for the proxy data.
            # This can be used to visualize potential distribution mismatch
            # between the proxy and the ground truth data.
            proxy_metrics, proxy_figs, proxy_fig_names = self.top_k_metrics_and_plots(
                None,
                top_k,
                "train proxy",
                energy=proxy,
                reward=proxy_reward,
                step=None,
                **kwargs,
            )
            # aggregate metrics and figures
            metrics.update(proxy_metrics)
            figs += proxy_figs
            fig_names += proxy_fig_names

        return metrics, figs, fig_names

    def plot_reward_distribution(
        self, states=None, scores=None, ax=None, title=None, proxy=None, **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots()
            standalone = True
        else:
            standalone = False
        if title == None:
            title = "Scores of Sampled States"
        if proxy is None:
            proxy = self.proxy
        if scores is None:
            if isinstance(states[0], torch.Tensor):
                states = torch.vstack(states).to(self.device, self.float)
            if isinstance(states, torch.Tensor) == False:
                states = torch.tensor(states, device=self.device, dtype=self.float)
            states_proxy = self.states2proxy(states)
            scores = self.proxy(states_proxy)
        if isinstance(scores, TensorType):
            scores = scores.cpu().detach().numpy()
        ax.hist(scores)
        ax.set_title(title)
        ax.set_ylabel("Number of Samples")
        ax.set_xlabel("Energy")
        plt.show()
        if standalone == True:
            plt.tight_layout()
            plt.close()
        return ax

    def test(
        self,
        samples: Union[
            TensorType["n_trajectories", "..."], npt.NDArray[np.float32], List
        ],
    ) -> dict:
        """
        Placeholder for a custom test function that can be defined for a specific
        environment. Can be overwritten if special evaluation procedure is needed
        for a given environment.

        Args
        ----
        samples
            A collection of sampled terminating states.

        Returns
        -------
        metrics
            A dictionary with metrics and their calculated values.
        """
        return {}
