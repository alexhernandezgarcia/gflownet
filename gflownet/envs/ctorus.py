"""
Classes to represent continuous hyper-torus environments.
"""

import itertools
import re
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neighbors import KernelDensity
from torch.distributions import Categorical, MixtureSameFamily, Uniform, VonMises
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tfloat, torch2np


class ContinuousTorus(GFlowNetEnv):
    r"""
    Continuous hyper-torus environment.

    The action space consists of the increment of the angle $\theta_i$ of each
    dimension $i$.

    Trajectories have a fixed length ``length_traj`` and the time step is included in
    the state. This allows for increments of any magnitude and any sign without
    creating cycles.

    States are represented by the concatenation of the angles (in radians and within
    $[0, 2\pi]$) for all dimensions with the time step or action number.

    The increments of the angles are sampled from a mixture of von Mises distributions.

    Attributes
    ----------
    ndim : int
        Dimensionality of the torus
    length_traj : int
       Fixed length of the trajectory.
    n_comp : int
       Number of components in the mixture of von Mises distributions used to sample
       angle increments.
    policy_encoding_dim_per_angle : int
        Dimensionality of the policy encodings of the angles.
    vonmises_min_concentration : float
        Minimum value allowed for the concentration parameter of the von Mises
        distributions.
    """

    def __init__(
        self,
        n_dim: int = 2,
        length_traj: int = 1,
        n_comp: int = 1,
        policy_encoding_dim_per_angle: int = None,
        vonmises_min_concentration: float = 1e-3,
        fixed_distr_params: dict = {
            "vonmises_mean": 0.0,
            "vonmises_concentration": 0.5,
        },
        random_distr_params: dict = {
            "vonmises_mean": 0.0,
            "vonmises_concentration": 0.001,
        },
        **kwargs,
    ):
        """
        Initializes a ContinuousCube environent.

        Parameters
        ----------
        ndim : int
            Dimensionality of the torus
        length_traj : int
           Fixed length of the trajectory.
        n_comp : int
           Number of components in the mixture of von Mises distributions used to
           sample angle increments.
        policy_encoding_dim_per_angle : int
            Dimensionality of the policy encodings of the angles.
        vonmises_min_concentration : float
            Minimum value allowed for the concentration parameter of the von Mises
            distributions.
        fixed_distr_params : dict
            Dictionary of parameters of the von Mises distribution that defines the
            fixed distribution of the environment. It must contain two keys with float
            values: ``vonmises_mean`` and ``vonmises_concentration``.
        random_distr_params : dict
            Dictionary of parameters of the von Mises distribution that defines the
            random distribution of the environment. It must contain two keys with float
            values: ``vonmises_mean`` and ``vonmises_concentration``.
        """
        assert n_dim > 0
        assert length_traj > 0
        assert n_comp > 0
        # Main environment properties
        self.n_dim = n_dim
        self.length_traj = length_traj
        # Policy properties
        self.n_comp = n_comp
        self.policy_encoding_dim_per_angle = policy_encoding_dim_per_angle
        self.vonmises_min_concentration = vonmises_min_concentration
        # Source state: position 0 at all dimensions and number of actions 0
        self.source_angles = [0.0 for _ in range(self.n_dim)]
        self.source = self.source_angles + [0]
        # End-of-sequence action: (n_dim, 0)
        self.eos = (self.n_dim, 0)
        # Base class init
        super().__init__(
            fixed_distr_params=fixed_distr_params,
            random_distr_params=random_distr_params,
            **kwargs,
        )
        self.continuous = True

    @property
    def mask_dim(self):
        """
        Returns the dimensionality of the masks.

        The mask consists of two fixed flags.

        Returns
        -------
        The dimensionality of the masks.
        """
        return 2

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

    def action2representative(self, action: Tuple) -> Tuple:
        """
        Returns the arbirary, representative action in the action space, so that the
        action can be contrasted with the action space and masks. If EOS, action return
        EOS.
        """
        if action == self.eos:
            return self.eos
        return self.representative_action

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

        As a continuous environment, the returned actions are "representatives", that
        is the actions represented in the action space.

        Parameters
        ----------
        mask : list (optional)
            The mask of a state. If None, it is computed in place.
        state : list (optional)
            A state in GFlowNet format. If None, self.state is used.
        done : bool (optional)
            Whether the trajectory is done. If None, self.done is used.
        backward : bool
            True if the transtion is backwards; False if forward.

        Returns
        -------
        list
            The list of representatives of the valid actions.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if mask is None:
            mask = self.get_mask(state, done, backward)

        # If EOS is valid (mask[1] is True), only EOS is valid.
        if mask[1]:
            return [self.eos]
        # Otherwise, only the representative action of generic actions is valid.
        else:
            return [self.representative_action]

    def get_parents(
        self, state: List = None, done: bool = None, action: Tuple[int, float] = None
    ) -> Tuple[List[List], List[Tuple[int, float]]]:
        """
        Defined only because it is required. A ContinuousEnv should be created to avoid
        this issue.
        """
        pass

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

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment, including the EOS
        action.
        """
        return int(self.length_traj) + 1

    def states2proxy(
        self, states: Union[List[List], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states in "environment format" for the proxy: each state is
        a vector of length n_dim where each value is an angle in radians. The n_actions
        item is removed.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        return tfloat(states, device=self.device, float_type=self.float)[:, :-1]

    def states2policy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: if
        policy_encoding_dim_per_angle >= 2, then the state (angles) is encoded using
        trigonometric components.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tfloat(states, float_type=self.float, device=self.device)
        if (
            self.policy_encoding_dim_per_angle is None
            or self.policy_encoding_dim_per_angle < 2
        ):
            return states
        step = states[:, -1]
        code_half_size = self.policy_encoding_dim_per_angle // 2
        int_coeff = (
            torch.arange(1, code_half_size + 1).repeat(states.shape[-1] - 1).to(states)
        )
        encoding = (
            torch.repeat_interleave(states[:, :-1], repeats=code_half_size, dim=1)
            * int_coeff
        )
        return torch.cat(
            [torch.cos(encoding), torch.sin(encoding), torch.unsqueeze(step, 1)],
            dim=1,
        )

    def state2readable(self, state: List) -> str:
        """
        Converts a state (a list of positions) into a human-readable string
        representing a state. Angles are converted into degrees in [0, 360]
        """
        angles = np.array(state[:-1])
        angles = angles * 180 / np.pi
        angles = str(angles).replace("(", "[").replace(")", "]").replace(",", "")
        n_actions = str(int(state[-1]))
        return angles + " | " + n_actions

    def readable2state(self, readable: str) -> List:
        """
        Converts a human-readable string representing a state into a state as a list of
        positions. Angles are converted back to radians.
        """
        # Preprocess
        pattern = re.compile(r"\s+")
        readable = re.sub(pattern, " ", readable)
        readable = readable.replace(" ]", "]")
        readable = readable.replace("[ ", "[")
        # Process
        pair = readable.split(" | ")
        angles = [np.float32(el) * np.pi / 180 for el in pair[0].strip("[]").split(" ")]
        n_actions = [int(pair[1])]
        return angles + n_actions

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
        random_action_prob : float, optional
            The probability of sampling a random action.
        temperature_logits : float, optional
            A scalar by which the model outputs are divided to temper the sampling
            distribution.
        """
        do_sample = torch.all(~mask, dim=1)
        n_states = policy_outputs.shape[0]
        # Initialize actions tensor with EOS actions (inf) since these will be the
        # actions for several special cases in both forward and backward actions.
        actions_tensor = torch.full(
            (n_states, self.n_dim), torch.inf, dtype=self.float, device=self.device
        )
        # Sample angle increments
        if torch.any(do_sample):
            logits_sampling = self.randomize_and_temper_sampling_distribution(
                policy_outputs, random_action_prob, temperature_logits
            )

            mix_logits = logits_sampling[do_sample, 0::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            mix = Categorical(logits=mix_logits)
            locations = logits_sampling[do_sample, 1::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            concentrations = logits_sampling[do_sample, 2::3].reshape(
                -1, self.n_dim, self.n_comp
            )
            vonmises = VonMises(
                locations,
                torch.exp(concentrations) + self.vonmises_min_concentration,
            )
            distr_angles = MixtureSameFamily(mix, vonmises)
            angles_sampled = distr_angles.sample()
            actions_tensor[do_sample] = angles_sampled
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
        return [tuple(a) for a in actions_tensor.tolist()]

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: Union[List, TensorType["n_states", "action_dim"]],
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
        actions : list or tensor
            The actions (angle increments) from each state in the batch for which to
            compute the log probability.
        states_from : tensor
            Ignored.

        is_backward : bool
            Ignored.
        """
        device = policy_outputs.device
        do_sample = torch.all(~mask, dim=1)
        actions = tfloat(actions, float_type=self.float, device=self.device)
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

    def copy(self):
        return deepcopy(self)

    def get_grid_terminating_states(self, n_states: int) -> List[List]:
        """
        Samples n terminating states by sub-sampling the state space as a grid, where n
        / n_dim points are obtained for each dimension.

        Parameters
        ----------
        n_states : int
            The number of terminating states to sample.

        Returns
        -------
        states : list
            A list of randomly sampled terminating states.
        """
        n_per_dim = int(np.ceil(n_states ** (1 / self.n_dim)))
        linspace = np.linspace(0, 2 * np.pi, n_per_dim + 1)[:-1]
        angles = np.meshgrid(*[linspace] * self.n_dim)
        angles = np.stack(angles).reshape((self.n_dim, -1)).T
        states = np.concatenate(
            (angles, self.length_traj * np.ones((angles.shape[0], 1))), axis=1
        ).tolist()
        return states

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[List]:
        rng = np.random.default_rng(seed)
        angles = rng.uniform(low=0.0, high=(2 * np.pi), size=(n_states, self.n_dim))
        states = np.concatenate((angles, np.ones((n_states, 1))), axis=1)
        return states.tolist()

    def fit_kde(
        self,
        samples: TensorType["batch_size", "state_proxy_dim"],
        kernel: str = "gaussian",
        bandwidth: float = 0.1,
    ):
        r"""
        Fits a Kernel Density Estimator on a batch of samples.

        The samples are previously augmented in order to account for the periodic
        aspect of the sample space.

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
        samples_aug = self.augment_samples(samples)
        kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(samples_aug)
        return kde

    def plot_reward_samples(
        self,
        samples: TensorType["batch_size", "state_proxy_dim"],
        samples_reward: TensorType["batch_size", "state_proxy_dim"],
        rewards: TensorType["batch_size"],
        min_domain: float = -np.pi,
        max_domain: float = 3 * np.pi,
        alpha: float = 0.5,
        dpi: int = 150,
        max_samples: int = 500,
        **kwargs,
    ):
        """
        Plots the reward contour alongside a batch of samples.

        The samples are previously augmented in order to visualise the periodic aspect
        of the sample space. It is assumed that the rewards are sorted from left to
        right (first) and top to bottom of the grid of samples.

        Parameters
        ----------
        samples : tensor
            A batch of samples from the GFlowNet policy in proxy format. These samples
            will be plotted on top of the reward density.
        samples_reward : tensor
            A batch of samples containing a grid over the sample space, from which the
            reward has been obtained. Ignored by this method.
        rewards : tensor
            The rewards of samples_reward. It should be a vector of dimensionality
            n_per_dim ** 2 and be sorted such that the each block at rewards[i *
            n_per_dim:i * n_per_dim + n_per_dim] correspond to the rewards at the i-th
            row of the grid of samples, from top to bottom.
        min_domain : float
            Minimum value of the domain to keep in the plot.
        max_domain : float
            Maximum value of the domain to keep in the plot.
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
        rewards = torch2np(rewards)
        n_per_dim = int(np.sqrt(rewards.shape[0]))
        assert n_per_dim**2 == rewards.shape[0]
        # Augment rewards to apply periodic boundary conditions
        rewards = rewards.reshape((n_per_dim, n_per_dim))
        rewards = np.tile(rewards, (3, 3))
        # Create mesh grid from samples_reward
        x = np.linspace(-2 * np.pi, 4 * np.pi, 3 * n_per_dim)
        y = np.linspace(-2 * np.pi, 4 * np.pi, 3 * n_per_dim)
        x_coords, y_coords = np.meshgrid(x, y)
        # Init figure
        fig, ax = plt.subplots()
        fig.set_dpi(dpi)
        # Plot reward contour
        h = ax.contourf(x_coords, y_coords, rewards, alpha=alpha)
        ax.axis("scaled")
        fig.colorbar(h, ax=ax)
        ax.plot([0, 0], [0, 2 * np.pi], "-w", alpha=alpha)
        ax.plot([0, 2 * np.pi], [0, 0], "-w", alpha=alpha)
        ax.plot([2 * np.pi, 2 * np.pi], [2 * np.pi, 0], "-w", alpha=alpha)
        ax.plot([2 * np.pi, 0], [2 * np.pi, 2 * np.pi], "-w", alpha=alpha)
        # Randomize and subsample samples
        random_indices = np.random.permutation(samples.shape[0])[:max_samples]
        samples = samples[random_indices, :]
        # Augment samples
        samples_aug = self.augment_samples(samples, exclude_original=True)
        ax.scatter(
            samples_aug[:, 0], samples_aug[:, 1], alpha=1.5 * alpha, color="white"
        )
        ax.scatter(samples[:, 0], samples[:, 1], alpha=alpha)
        # Set axes limits
        ax.set_xlim([min_domain, max_domain])
        ax.set_ylim([min_domain, max_domain])
        # Set ticks and labels
        ticks = [0.0, np.pi / 2, np.pi, (3 * np.pi) / 2, 2 * np.pi]
        labels = ["0.0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{3}$", f"$2\pi$"]
        ax.set_xticks(ticks, labels)
        ax.set_yticks(ticks, labels)
        ax.grid()
        # Set tight layout
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
        # Score samples with KDE and reshape
        Z = np.exp(kde.score_samples(samples)).reshape((n_per_dim, n_per_dim))
        # Init figure
        fig, ax = plt.subplots()
        fig.set_dpi(dpi)
        # Plot KDE
        h = ax.contourf(x_coords, y_coords, Z, alpha=alpha)
        ax.axis("scaled")
        if colorbar:
            fig.colorbar(h, ax=ax)
        # Set ticks and labels
        ticks = [0.0, np.pi / 2, np.pi, (3 * np.pi) / 2, 2 * np.pi]
        labels = ["0.0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{3}$", f"$2\pi$"]
        ax.set_xticks(ticks, labels)
        ax.set_yticks(ticks, labels)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Set tight layout
        plt.tight_layout()
        return fig

    @staticmethod
    def augment_samples(samples: np.array, exclude_original: bool = False) -> np.array:
        """
        Augments a batch of samples by applying the periodic boundary conditions from
        [0, 2pi) to [-2pi, 4pi) for all dimensions.
        """
        samples_aug = []
        for offsets in itertools.product(
            [-2 * np.pi, 0.0, 2 * np.pi], repeat=samples.shape[-1]
        ):
            if exclude_original and all([offset == 0.0 for offset in offsets]):
                continue
            samples_aug.append(
                np.stack(
                    [samples[:, dim] + offset for dim, offset in enumerate(offsets)],
                    axis=-1,
                )
            )
        samples_aug = np.concatenate(samples_aug, axis=0)
        return samples_aug
