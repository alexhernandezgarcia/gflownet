"""
Classes to represent hyper-torus environments
"""

import itertools
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.distributions import Categorical, MixtureSameFamily, Uniform, VonMises
from torchtyping import TensorType

from gflownet.envs.htorus import HybridTorus
from gflownet.utils.common import copy, tfloat


class ContinuousTorus(HybridTorus):
    """
    Purely continuous (no discrete actions) hyper-torus environment in which the
    action space consists of the increment Delta theta of the angle at each dimension.
    The trajectory is of fixed length length_traj.

    The states space is the concatenation of the angle (in radians and within [0, 2 *
    pi]) at each dimension and the number of actions.

    Attributes
    ----------
    ndim : int
        Dimensionality of the torus

    length_traj : int
       Fixed length of the trajectory.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Mask dimensionality:
        self.mask_dim = 2

    def get_action_space(self):
        """
        The action space is continuous, thus not defined as such here.

        The actions are tuples of length n_dim, where the value at position d indicates
        the increment of dimension d.

        EOS is indicated by np.inf for all dimensions.

        This method defines self.eos and the returned action space is simply a
        representative (arbitrary) action with an increment of 0.0 in all dimensions,
        and EOS.
        """
        self.eos = tuple([np.inf] * self.n_dim)
        self.representative_action = tuple([0.0] * self.n_dim)
        return [self.representative_action, self.eos]

    def get_policy_output(self, params: dict) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy.

        For each dimension d of the hyper-torus and component c of the mixture, the
        output of the policy should return
          1) the weight of the component in the mixture
          2) the location of the von Mises distribution to sample the angle increment
          3) the log concentration of the von Mises distribution to sample the angle
          increment

        Therefore, the output of the policy model has dimensionality D x C x 3, where D
        is the number of dimensions (self.n_dim) and C is the number of components
        (self.n_comp). The first 3 x C entries in the policy output correspond to the
        first dimension, and so on.
        """
        policy_output = torch.ones(
            self.n_dim * self.n_comp * 3, dtype=self.float, device=self.device
        )
        policy_output[1::3] = params["vonmises_mean"]
        policy_output[2::3] = params["vonmises_concentration"]
        return policy_output

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        The action space is continuous, thus the mask is not of invalid actions as
        in discrete environments, but an indicator of "special cases", for example
        states from which only certain actions are possible.

        The "mask" has 2 elements - to match the mask of backward actions - but only
        one is needed for forward actions, thus both elements take the same value,
        according to the following:

        - If done is True, then the mask is True.
        - If the number of actions (state[-1]) is equal to the (fixed) trajectory
          length, then only EOS is valid and the mask is True.
        - Otherwise, any continuous action is valid (except EOS) and the mask is False.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True] * 2
        elif state[-1] >= self.length_traj:
            return [True] * 2
        else:
            return [False] * 2

    def get_mask_invalid_actions_backward(self, state=None, done=None, parents_a=None):
        """
        The action is space is continuous, thus the mask is not of invalid actions as
        in discrete environments, but an indicator of "special cases", for example
        states from which only certain actions are possible.

        The "mask" has 2 elements to capture the 2 special in backward actions. The
        possible values of the mask are the following:

        - mask[0]:
            - True, if only the "return-to-source" action is valid.
            - False otherwise.
        - mask[1]:
            - True, if only the EOS action is valid, that is if done is True.
            - False otherwise.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [False, True]
        elif state[-1] == 1:
            return [True, False]
        else:
            return [False, False]

    def get_parents(
        self, state: List = None, done: bool = None, action: Tuple[int, float] = None
    ) -> Tuple[List[List], List[Tuple[int, float]]]:
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state, as a list of length n_angles where each element
            is the position at each dimension.

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
            for dim, angle in enumerate(action):
                state[int(dim)] = (state[int(dim)] - angle) % (2 * np.pi)
            state[-1] -= 1
            parents = [state]
            return parents, [action]

    def action2representative(self, action: Tuple) -> Tuple:
        """
        Returns the arbirary, representative action in the action space, so that the
        action can be contrasted with the action space and masks.
        """
        return self.representative_action

    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "policy_output_dim"]] = None,
        states_from: Optional[List] = None,
        is_backward: Optional[bool] = False,
        sampling_method: Optional[str] = "policy",
        temperature_logits: Optional[float] = 1.0,
        max_sampling_attempts: Optional[int] = 10,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs. The angle increments
        that form the actions are sampled from a mixture of Von Mises distributions.

        A distinction between forward and backward actions is made and specified by the
        argument is_backward, in order to account for the following special cases:

        Forward:

        - If the number of steps is equal to the maximum, then the only valid action is
          EOS.

        Backward:

        - If the number of steps is equal to 1, then the only valid action is to return
          to the source. The specific action depends on the current state.

        Args
        ----
        policy_outputs : tensor
            The output of the GFlowNet policy model.

        mask : tensor
            The mask containing information about special cases.

        states_from : tensor
            The states originating the actions, in GFlowNet format.

        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default).
        """
        device = policy_outputs.device
        do_sample = torch.all(~mask, dim=1)
        n_states = policy_outputs.shape[0]
        logprobs = torch.zeros(
            (n_states, self.n_dim), dtype=self.float, device=self.device
        )
        # Initialize actions tensor with EOS actions (inf) since these will be the
        # actions for several special cases in both forward and backward actions.
        actions_tensor = torch.full(
            (n_states, self.n_dim), torch.inf, dtype=self.float, device=device
        )
        # Sample angle increments
        if torch.any(do_sample):
            if sampling_method == "uniform":
                distr_angles = Uniform(
                    torch.zeros(len(ns_range_noeos)),
                    2 * torch.pi * torch.ones(len(ns_range_noeos)),
                )
            elif sampling_method == "policy":
                mix_logits = policy_outputs[do_sample, 0::3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                mix = Categorical(logits=mix_logits)
                locations = policy_outputs[do_sample, 1::3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                concentrations = policy_outputs[do_sample, 2::3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                vonmises = VonMises(
                    locations,
                    torch.exp(concentrations) + self.vonmises_min_concentration,
                )
                distr_angles = MixtureSameFamily(mix, vonmises)
            angles_sampled = distr_angles.sample()
            actions_tensor[do_sample] = angles_sampled
            logprobs[do_sample] = distr_angles.log_prob(angles_sampled)
        logprobs = torch.sum(logprobs, axis=1)
        # Catch special case for backwards backt-to-source (BTS) actions
        if is_backward:
            do_bts = mask[:, 0]
            if torch.any(do_bts):
                source_angles = tfloat(
                    self.source[: self.n_dim], float_type=self.float, device=self.device
                )
                states_from_angles = tfloat(
                    states_from, float_type=self.float, device=self.device
                )[do_bts, : self.n_dim]
                actions_bts = states_from_angles - source_angles
                actions_tensor[do_bts] = actions_bts
        # TODO: is this too inefficient because of the multiple data transfers?
        actions = [tuple(a.tolist()) for a in actions_tensor]
        return actions, logprobs

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: TensorType["n_states", "n_dim"],
        mask: TensorType["n_states", "1"],
        states_from: Optional[List] = None,
        is_backward: bool = False,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.

        Args
        ----
        policy_outputs : tensor
            The output of the GFlowNet policy model.

        mask : tensor
            The mask containing information special cases.

        actions : tensor
            The actions (angle increments) from each state in the batch for which to
            compute the log probability.

        states_from : tensor
            Ignored.

        is_backward : bool
            Ignored.
        """
        device = policy_outputs.device
        do_sample = torch.all(~mask, dim=1)
        n_states = policy_outputs.shape[0]
        logprobs = torch.zeros(n_states, self.n_dim).to(device)
        if torch.any(do_sample):
            mix_logits = policy_outputs[do_sample, 0::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            mix = Categorical(logits=mix_logits)
            locations = policy_outputs[do_sample, 1::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            concentrations = policy_outputs[do_sample, 2::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            vonmises = VonMises(
                locations,
                torch.exp(concentrations) + self.vonmises_min_concentration,
            )
            distr_angles = MixtureSameFamily(mix, vonmises)
            logprobs[do_sample] = distr_angles.log_prob(actions[do_sample])
        logprobs = torch.sum(logprobs, axis=1)
        return logprobs

    def _step(
        self,
        action: Tuple[float],
        backward: bool,
    ) -> Tuple[List[float], Tuple[float], bool]:
        """
        Updates self.state given a non-EOS action. This method is called by both step()
        and step_backwards(), with the corresponding value of argument backward.

        Forward steps:
            - Add action increments to state angles.
            - Increment n_actions value of state.
        Backward steps:
            - Subtract action increments from state angles.
            - Decrement n_actions value of state.

        Args
        ----
        action : tuple
            Action to be executed. An action is a vector where the value at position d
            indicates the increment in the angle at dimension d.

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
        for dim, angle in enumerate(action):
            if backward:
                self.state[int(dim)] -= angle
            else:
                self.state[int(dim)] += angle
            self.state[int(dim)] = self.state[int(dim)] % (2 * np.pi)
        if backward:
            self.state[-1] -= 1
        else:
            self.state[-1] += 1
        assert self.state[-1] >= 0 and self.state[-1] <= self.length_traj
        # If n_steps is equal to 0, set source to avoid escaping comparison to source.
        if self.state[-1] == 0:
            self.state = copy(self.source)

    def step(
        self, action: Tuple[float], skip_mask_check: bool = False
    ) -> Tuple[List[float], Tuple[float], bool]:
        """
        Executes forward step given an action.

        See: _step().

        Args
        ----
        action : tuple
            Action to be executed. An action is a vector where the value at position d
            indicates the increment in the angle at dimension d.

        skip_mask_check : bool
            Ignored because the action space space is fully continuous, therefore there
            is nothing to check.

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
        # If done is True, return invalid
        if self.done:
            return self.state, action, False
        # If action is EOS, check that the number of steps is equal to the trajectory
        # length, set done to True, increment n_actions and return same state
        elif action == self.eos:
            assert self.state[-1] == self.length_traj
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # Otherwise perform action
        else:
            self.n_actions += 1
            self._step(action, backward=False)
            return self.state, action, True

    def step_backwards(
        self, action: Tuple[float], skip_mask_check: bool = False
    ) -> Tuple[List[float], Tuple[float], bool]:
        """
        Executes backward step given an action.

        See: _step().

        Args
        ----
        action : tuple
            Action to be executed. An action is a vector where the value at position d
            indicates the increment in the angle at dimension d.

        skip_mask_check : bool
            Ignored because the action space space is fully continuous, therefore there
            is nothing to check.

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
        # Otherwise perform action
        else:
            assert action != self.eos
            self.n_actions += 1
            self._step(action, backward=True)
            return self.state, action, True

    def get_max_traj_length(self):
        return int(self.length_traj) + 1
