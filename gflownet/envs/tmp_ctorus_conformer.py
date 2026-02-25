"""
Classes to represent hyper-torus environments
"""

import itertools
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.distributions import Categorical, MixtureSameFamily, Uniform, VonMises
from torchtyping import TensorType

from gflownet.envs.htorus import HybridTorus
from gflownet.utils.common import copy, tfloat
from gflownet.utils.metrics import angles_allclose
from gflownet.utils.molecule.distributions import (
    WrappedNormal,
    estimate_entropy,
    estimate_gap_entropy_mixture_von_mises,
    upper_bond_entropy_mixture_von_mises,
)


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

    start_uniform : bool
        If True, the first step of the trajectory is sampled from the uniform distribution.
    """

    def __init__(self, start_uniform=False, **kwargs):
        super().__init__(**kwargs)
        # Mask dimensionality:
        self.mask_dim = 2
        self.start_uniform = start_uniform

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

        Parameters
        ----------
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
            for dim, angle in enumerate(action):  # this requires action to not be None
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

    def _extract_distribution_parameters(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        timesteps: Optional[TensorType["n_states"]] = None,
    ):
        """
        Helper function to extract the parameters of the distributions from the
        policy_output tensor.

        Parameters
        ----------
        policy_outputs : tensor["n_states", "policy_output_dim"]
            The output of the GFlowNet policy model.

        Returns
        -------

        if Von Mises distribution is used:
            mix_logits : tensor["n_states", "n_dim", "n_comp"]
                The logits of the mixture components.

            concentrations : tensor["n_states", "n_dim", "n_comp"]
                The concentrations of the von Mises distributions.

            locations : tensor["n_states", "n_dim", "n_comp"]
                The locations of the von Mises distributions
        elif Wrapped Normal distribution is used:
            means: tensor["n_states", "n_dim"]
                The means of the wrapped normal distributions.
            stds: tensor["n_states", "n_dim"]
                The standard deviations of the wrapped normal distributions. This is
                not learned, and it comes from a predefined noise schedule. See
                function convert_timesteps_to_stds.
        """

        if self.distr_type == "von_mises":
            mix_logits = policy_outputs[:, 0::3].reshape(-1, self.n_dim, self.n_comp)
            concentrations = policy_outputs[:, 2::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            locations = policy_outputs[:, 1::3].reshape(-1, self.n_dim, self.n_comp)
            concentrations = (
                torch.exp(concentrations)
                if self.exp_vonmises_concentration
                else concentrations
            )
            if torch.any(concentrations < 0.0):
                raise Exception(
                    "Negative concentration values "
                    f"{concentrations[concentrations < 0.]}"
                )
            concentrations = concentrations + self.vonmises_min_concentration
            return {
                "mix_logits": mix_logits,
                "concentrations": concentrations,
                "locations": locations,
            }
        elif self.distr_type == "diffusion":
            if timesteps is None:
                raise ValueError(
                    "Timesteps must be provided for diffusion distribution."
                )
            stds = self.convert_timesteps_to_stds(timesteps, cumulative=False)
            score = policy_outputs.reshape(-1, self.n_dim)
            means = score * torch.pow(stds.unsqueeze(1).repeat(1, 2), 2)
            return {"means": means, "stds": stds}

    def get_distr(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        timesteps: TensorType["n_states"],
        is_backward: bool,
    ):
        params = self._extract_distribution_parameters(policy_outputs, timesteps)
        if self.distr_type == "von_mises":
            mix_logits, concentrations, locations = (
                params["mix_logits"],
                params["concentrations"],
                params["locations"],
            )
            # print('Von Mises params shapes', mix_logits.shape, concentrations.shape, locations.shape)
            mix = Categorical(logits=mix_logits)
            vonmises = VonMises(
                locations,
                concentrations + self.vonmises_min_concentration,
            )
            distr_angles = MixtureSameFamily(mix, vonmises)

        elif self.distr_type == "diffusion":

            stds = self.convert_timesteps_to_stds(timesteps, cumulative=False)
            assert self.n_comp == 1
            if is_backward == False:
                score = policy_outputs.reshape(
                    -1, self.n_dim
                )  # at convergence, score should be equal to grad_logp(x_t)
                means = score * torch.pow(
                    stds.unsqueeze(1).repeat(1, 2), 2
                )  # since x_{t+1} = x_t + score * std_t^2 + N(0, std_t^2),  the increment mean is equal to score * std_t^2
            else:
                means = torch.zeros(
                    policy_outputs.shape[0], self.n_dim
                )  # For diffusion models, the backwards policy is fixed and not learned. Here, we force the backwards variance-exploding policy, i.e. x_{t-1} = x_t + N(0, std_t^2)
            distr_angles = WrappedNormal(means, stds)

        elif self.distr_type == "uniform":
            distr_angles = Uniform(
                torch.zeros(len(policy_outputs), self.n_dim).to(policy_outputs.device),
                2
                * torch.pi
                * torch.ones(len(policy_outputs), self.n_dim).to(policy_outputs.device),
            )

        # assert that distr_angles has the methods logpprob and sample()
        return distr_angles

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
        that form the actions are sampled from either a mixture of Von Mises or a
        Wrapped Normal.

        A distinction between forward and backward actions is made and specified by the
        argument is_backward, in order to account for the following special cases:

        Forward:

        - If the number of steps is equal to the maximum, then the only valid action is
          EOS.

        Backward:

        - If the number of steps is equal to 1, then the only valid action is to return
          to the source. The specific action depends on the current state.

        Parameters
        ----------
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

            # TODO: seems like this is never used, figure out if it could be removed
            if sampling_method == "uniform":
                distr_angles = Uniform(
                    torch.zeros(len(ns_range_noeos), self.n_dim).to(actions_tensor),
                    2
                    * torch.pi
                    * torch.ones(len(ns_range_noeos), self.n_dim).to(actions_tensor),
                )
            timesteps = tfloat(
                [x[-1] for x in states_from], float_type=self.float, device=self.device
            )
            distr_angles = self.get_distr(
                policy_outputs[do_sample],
                timesteps[do_sample],
                is_backward,
            )
            angles_sampled = distr_angles.sample()
            actions_tensor[do_sample] = angles_sampled
            logprobs[do_sample] = distr_angles.log_prob(angles_sampled)
            # Start from uniform distribution. #TODO rewrite better
            if sampling_method == "policy" and self.start_uniform:
                idx = []
                for i, state in enumerate(states_from):
                    if state[-1] == 0 and do_sample[i]:
                        idx.append(i)
                if len(idx) > 0:
                    first_step_idx = torch.tensor(idx, device=device)
                    distr_fs_angles = Uniform(
                        torch.zeros(len(first_step_idx), self.n_dim).to(actions_tensor),
                        2
                        * torch.pi
                        * torch.ones(len(first_step_idx), self.n_dim).to(
                            actions_tensor
                        ),
                    )
                    actions_tensor[first_step_idx] = distr_fs_angles.sample()
                    logprobs[first_step_idx] = distr_fs_angles.log_prob(
                        actions_tensor[first_step_idx]
                    )
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
                logprobs[do_bts] = 0.0
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

        Parameters
        ----------
        policy_outputs : tensor
            The output of the GFlowNet policy model.
        mask : tensor
            The mask containing information special cases.
        actions : tensor
            The actions (angle increments) from each state in the batch for which to
            compute the log probability.
        states_from : list
            The states originating the actions, in GFlowNet format. Used to determine
            the log probability of the first step if start from uniform.
        is_backward : bool
            True if the actions are backward, False if the actions are forward
            (default).
        """
        device = policy_outputs.device
        do_sample = torch.all(~mask, dim=1)
        n_states = policy_outputs.shape[0]
        logprobs = torch.zeros(n_states, self.n_dim).to(device)
        if torch.any(do_sample):
            timesteps = tfloat(
                [x[-1] for x in states_from], float_type=self.float, device=self.device
            )
            distr = self.get_distr(
                policy_outputs[do_sample],
                timesteps[do_sample],
                is_backward,
            )
            logprobs[do_sample] = distr.log_prob(actions[do_sample])
            # Start from uniform distribution
            if self.start_uniform:
                idx = []
                for i, state in enumerate(states_from):
                    if state[-1] == 0 and do_sample[i]:
                        idx.append(i)
                if len(idx) > 0:
                    first_step_idx = torch.tensor(idx, device=device)
                    distr_fs_angles = Uniform(
                        torch.zeros(len(first_step_idx), self.n_dim).to(actions),
                        2
                        * torch.pi
                        * torch.ones(len(first_step_idx), self.n_dim).to(actions),
                    )
                    logprobs[first_step_idx] = distr_fs_angles.log_prob(
                        actions[first_step_idx]
                    )
        if is_backward:
            do_bts = mask[:, 0]
            if torch.any(do_bts):
                # correct back-to-source actions have logprob 0, others are -inf
                logprobs[do_bts] = 0.0

                source_angles = tfloat(
                    self.source[: self.n_dim], float_type=self.float, device=self.device
                )
                states_from_angles = tfloat(
                    states_from, float_type=self.float, device=self.device
                )[do_bts, : self.n_dim]
                actions_bts = (states_from_angles - source_angles) % (2 * torch.pi)
                actions_bts_input = actions[do_bts] % (2 * torch.pi)
                mask_inf = ~torch.isclose(actions_bts_input, actions_bts, atol=1e-6)
                if torch.any(mask_inf):
                    # weird, but needed to assign walues to a tensor using 2 masks
                    logprobs_tmp = logprobs[do_bts]
                    logprobs_tmp[mask_inf] = -torch.inf
                    logprobs[do_bts] = logprobs_tmp
                    warnings.warn(
                        "Warning: logprobs for invalid back-to-source actions set to -inf."
                    )

        logprobs = torch.sum(logprobs, axis=1)
        return logprobs

    def get_policy_output(
        self, params: dict
    ):  # TODO change: I am hardcoding distr_type
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy.

        If Von Mises:
            For each dimension d of the hyper-torus and component c of the mixture, the
            output of the policy should return
            1) the weight of the component in the mixture
            2) the location of the von Mises distribution to sample the angle increment
            3) the log concentration of the von Mises distribution to sample the angle
            increment.
            Therefore, the output of the policy model has dimensionality D x C x 3,
            where D is the number of dimensions (self.n_dim) and C is the number of
            components (self.n_comp). The first 3 x C entries in the policy output
            correspond to the first dimension, and so on.
        If Diffusion:
            For each dimension d of the hyper-torus, the output of the policy should
            return
            1) the mean of the wrapped normal distribution to sample the angle increment
            Therefore, the output of the policy model has dimensionality D, where D
            is the number of dimensions (self.n_dim).


        """

        if self.distr_type == "von_mises":
            policy_output = torch.ones(
                self.n_dim * self.n_comp * 3, dtype=self.float, device=self.device
            )
            policy_output[1::3] = params["vonmises_mean"]
            policy_output[2::3] = params["vonmises_concentration"]
            return policy_output
        elif self.distr_type == "diffusion":
            policy_output = torch.ones(self.n_dim, dtype=self.float, device=self.device)
            policy_output[::1] = params["means"]
            return policy_output

    def get_policy_entropy(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        timesteps: TensorType["n_states"],
        mc_estimation: bool = False,
        n_samples: int = 1000,
    ) -> TensorType["n_states"]:
        """
        Computes the entropy of the policy distribution given the policy outputs.

        Parameters
        ----------
        policy_outputs : tensor["n_states", "policy_output_dim"]
            The output of the GFlowNet policy model.
        mc_estimation : bool
            If True, estimate the entropy using Monte Carlo sampling. Otherwise, use
            the upper bound.
        n_samples : int
            Number of samples to use for Monte Carlo estimation

        Returns
        -------
        entropy : tensor["n_states"]
            The entropy of the policy distribution for each state in the batch.
        """

        params = self._extract_distribution_parameters(policy_outputs, timesteps)
        if self.distr_type == "von_mises":
            mix_logits, concentrations, locations = (
                params["mix_logits"],
                params["concentrations"],
                params["locations"],
            )
            if mc_estimation:
                mix = Categorical(logits=mix_logits)
                vonmises = VonMises(locations, concentrations)
                distr = MixtureSameFamily(mix, vonmises)
                entropy = estimate_entropy(distr, n_samples=n_samples)
            else:
                entropy = upper_bond_entropy_mixture_von_mises(
                    mix_logits, concentrations, locations
                )

        elif self.distr_type == "diffusion":
            means, stds = params["means"], params["stds"]
            distr = WrappedNormal(means, stds)
            entropy = estimate_entropy(distr, n_samples=n_samples)

        assert entropy.shape[0] == policy_outputs.shape[0]
        assert entropy.shape[1] == self.n_dim
        assert len(entropy.shape) == 2
        entropy = torch.sum(entropy, axis=1)
        return entropy

    def get_policy_entropy_gap(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        timesteps: TensorType["n_states"],
        n_samples: int = 1000,
    ):
        """
        Estimates the gap between upper bound for the entropy of the policy distribution
        and its entropy given the policy outputs. Monte Carlo sampling is used to
        estimate the intergal of the KL divergence terms.

        Parameters
        ----------
        policy_outputs : tensor["n_states", "policy_output_dim"]
            The output of the GFlowNet policy model.
        n_samples : int
            Number of samples to use for Monte Carlo estimation

        Returns
        -------
        gap : tensor["n_states"]
            The gap between the upper bound and the entropy of the policy distribution
            for each state in the batch.
        """
        if self.distr_type == "von_mises":
            params = self._extract_distribution_parameters(policy_outputs, timesteps)
            mix_logits, concentrations, locations = (
                params["mix_logits"],
                params["concentrations"],
                params["locations"],
            )
            gap = estimate_gap_entropy_mixture_von_mises(
                mix_logits, concentrations, locations, n_samples=n_samples
            )
            gap = torch.sum(gap, axis=1)
        elif self.distr_type == "diffusion":
            gap = torch.zeros(len(policy_outputs), device=policy_outputs.device)
        return gap

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

        Parameters
        ----------
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

        Parameters
        ----------
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

        Parameters
        ----------
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

    def convert_timesteps_to_stds(
        self, timesteps: TensorType["n_states", 1], cumulative=False
    ) -> TensorType["n_states", 1]:
        """
        This function defines the noise schedule given the timesteps.
        For example, here, we use the exponential noise schedule, i.e. sigma(t|0) =
        sigma_min^t * sigma_max^(1-t), and:
            - t is the normalized timestep in [0, 1], t=0 corresponds to the first step
              of the trajectory and t=1 to the last step.
            - sigma_min and sigma_max are the minimum and maximum noise levels,
              respectively.

        Parameters
        ----------
        timesteps : tensor["n_states", 1]
            The timesteps of the batch of states.
        cumulative : bool
            If True: returns sigma(t|0), corresponding to the standard deviation of
            N(x_t | x_0)
            If False: returns sigma(t+1|t), corresponding to the standard deviation of
            N(x_{t+1} | x_t)

        Returns
        -------
        stds : tensor["n_states", 1]
            The standard deviations of the noise schedule for each state in the batch.
        """

        dt = 1 / self.max_traj_length
        t = timesteps / self.max_traj_length
        sigmas = 10 ** (
            (1 - t) * np.log10(self.sigma_max) + t * np.log10(self.sigma_min)
        )
        if cumulative:
            return sigmas
        else:
            g = sigmas * torch.sqrt(
                torch.tensor(2 * np.log(self.sigma_max / self.sigma_min))
            )
            stds = g * np.sqrt(dt)
            return stds

    def isclose(self, first_state: List, second_state: List) -> bool:
        """
        Check if two states are close in the state space.
        States are in environment format.

        Parameters
        ----------
        first_state : list
            First state to compare
        second_state : list
            Second state to compare

        Returns
        -------
        bool
            True if the two states are close, False otherwise
        """
        return angles_allclose(
            first_state[:-1], second_state[:-1], atol=self.state_space_atol
        )
