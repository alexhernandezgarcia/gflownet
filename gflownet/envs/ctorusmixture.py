"""
Classes to represent hyper-torus environments
"""
from typing import List, Tuple
import itertools
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from gflownet.envs.ctorus import ContinuousTorus
from torch.distributions import Categorical, Uniform, VonMises, MixtureSameFamily
from torchtyping import TensorType


class ContinuousTorusMixture(ContinuousTorus):
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

    def __init__(
        self,
        n_dim=2,
        length_traj=1,
        policy_encoding_dim_per_angle=None,
        n_comp=3,
        fixed_distribution=dict,
        random_distribution=dict,
        vonmises_min_concentration=1e-3,
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
        self.n_comp = n_comp
        super(ContinuousTorusMixture, self).__init__(
            n_dim=n_dim,
            length_traj=length_traj,
            policy_encoding_dim_per_angle=policy_encoding_dim_per_angle,
            fixed_distribution=fixed_distribution,
            random_distribution=random_distribution,
            vonmises_min_concentration=vonmises_min_concentration,
            env_id=env_id,
            reward_beta=reward_beta,
            reward_norm=reward_norm,
            reward_norm_std_mult=reward_norm_std_mult,
            reward_func=reward_func,
            denorm_proxy=denorm_proxy,
            energies_stats=energies_stats,
            proxy=proxy,
            oracle=oracle,
            **kwargs,
        )

    def get_policy_output(self, params: dict):
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy.

        For each dimension d, c components:
        [mu_d0, kappa_d0, ... mu_dc, sigma_dc, 1, 1, ... 1]
        """
        policy_output = np.ones(self.n_dim * self.n_comp * 3)
        policy_output[1::3] = params.vonmises_mean
        policy_output[2::3] = params.vonmises_concentration
        return policy_output

    def sample_actions(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        sampling_method: str = "policy",
        mask_stop_actions: TensorType["n_states", "1"] = None,
        temperature_logits: float = 1.0,
        loginf: float = 1000,
    ) -> Tuple[List[Tuple], TensorType["n_states"]]:
        """
        Samples a batch of actions from a batch of policy outputs.
        """
        device = policy_outputs.device
        mask_states_sample = ~mask_stop_actions.flatten()
        n_states = policy_outputs.shape[0]
        # Sample angle increments
        angles = torch.zeros(n_states, self.n_dim).to(device)
        logprobs = torch.zeros(n_states, self.n_dim).to(device)
        if torch.any(mask_states_sample):
            if sampling_method == "uniform":
                distr_angles = Uniform(
                    torch.zeros(len(ns_range_noeos)),
                    2 * torch.pi * torch.ones(len(ns_range_noeos)),
                )
            elif sampling_method == "policy":
                mix_logits = policy_outputs[mask_states_sample, 0::3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                mix = Categorical(logits=mix_logits)
                locations = policy_outputs[mask_states_sample, 1::3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                concentrations = policy_outputs[mask_states_sample, 2::3].reshape(
                    -1, self.n_dim, self.n_comp
                )
                vonmises = VonMises(
                    locations,
                    torch.exp(concentrations) + self.vonmises_min_concentration,
                )
                distr_angles = MixtureSameFamily(mix, vonmises)
            angles[mask_states_sample] = distr_angles.sample()
            logprobs[mask_states_sample] = distr_angles.log_prob(
                angles[mask_states_sample]
            )
        logprobs = torch.sum(logprobs, axis=1)
        # Build actions
        actions_tensor = (
            torch.repeat_interleave(torch.arange(0, self.n_dim), 2)
            .repeat(n_states, 1)
            .to(dtype=self.float, device=device)
        )
        actions_tensor[mask_states_sample, 1::2] = angles[mask_states_sample]
        actions_tensor[mask_stop_actions.flatten()] = torch.zeros(
            actions_tensor.shape[1]
        ).to(actions_tensor)
        actions_tensor[mask_stop_actions.flatten(), 0] = 2.0
        actions = [tuple(a.tolist()) for a in actions_tensor]
        return actions, logprobs

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        is_forward: bool,
        actions: TensorType["n_states", 2],
        states_target: TensorType["n_states", "policy_input_dim"],
        mask_stop_actions: TensorType["n_states", "1"] = None,
        loginf: float = 1000,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.
        """
        device = policy_outputs.device
        mask_states_sample = ~mask_stop_actions.flatten()
        n_states = policy_outputs.shape[0]
        angles = actions[:, 1::2]
        logprobs = torch.zeros(n_states, self.n_dim).to(device)
        if torch.any(mask_states_sample):
            mix_logits = policy_outputs[mask_states_sample, 0::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            mix = Categorical(logits=mix_logits)
            locations = policy_outputs[mask_states_sample, 1::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            concentrations = policy_outputs[mask_states_sample, 2::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            vonmises = VonMises(
                locations,
                torch.exp(concentrations) + self.vonmises_min_concentration,
            )
            distr_angles = MixtureSameFamily(mix, vonmises)
            logprobs[mask_states_sample] = distr_angles.log_prob(
                angles[mask_states_sample]
            )
        logprobs = torch.sum(logprobs, axis=1)
        return logprobs

    def step(
        self, action: Tuple[int, float]
    ) -> Tuple[List[float], Tuple[int, float], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple with either:
            - (self.eos, 0.0) with two values:
            (dimension, magnitude).

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
        # If only possible action is eos, then force eos
        # If the number of actions is equal to maximum trajectory length
        elif self.n_actions == self.length_traj:
            self.done = True
            self.n_actions += 1
            return self.state, self.action_space[-1], True
        # If action is not eos, then perform action
        elif action[0] != self.eos:
            self.n_actions += 1
            for dim, angle in zip(action[0::2], action[1::2]):
                self.state[int(dim)] += angle
                self.state[int(dim)] = self.state[int(dim)] % (2 * np.pi)
                self.state[-1] = self.n_actions
            return self.state, action, True
        # If action is eos, then it is invalid
        else:
            return self.state, action, False
