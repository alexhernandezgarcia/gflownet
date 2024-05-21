"""
Classes to represent hyper-torus environments
"""

import itertools
import re
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.neighbors import KernelDensity
from torch.distributions import Bernoulli, Categorical, Uniform, VonMises
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import tfloat, torch2np


class HybridTorus(GFlowNetEnv):
    """
    Continuous (hybrid: discrete and continuous) hyper-torus environment in which the
    action space consists of the selection of which dimension d to increment and of the
    angle of dimension d. The trajectory is of fixed length length_traj.

    The states space is the concatenation of the angle (in radians and within [0, 2 *
    pi]) at each dimension and the number of actions.

    Attributes
    ----------
    n_dim : int
        Dimensionality of the torus

    length_traj : int
       Fixed length of the trajectory.
    """

    def __init__(
        self,
        n_dim: int = 2,
        length_traj: int = 1,
        n_comp: int = 1,
        policy_encoding_dim_per_angle: int = None,
        do_nonzero_source_prob: bool = True,
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
        assert n_dim > 0
        assert length_traj > 0
        assert n_comp > 0
        self.n_dim = n_dim
        self.length_traj = length_traj
        self.policy_encoding_dim_per_angle = policy_encoding_dim_per_angle
        # Parameters of fixed policy distribution
        self.n_comp = n_comp
        if do_nonzero_source_prob:
            self.n_params_per_dim = 4
        else:
            self.n_params_per_dim = 3
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

    def get_action_space(self):
        """
        Since this is a hybrid (continuous/discrete) environment, this method
        constructs a list with the discrete actions.

        The actions are tuples with two values: (dimension, magnitude) where dimension
        indicates the index of the dimension on which the action is to be performed and
        magnitude indicates the increment of the angle in radians.

        The (discrete) action space is then one tuple per dimension (with 0 increment),
        plus the EOS action.
        """
        actions = [(d, 0) for d in range(self.n_dim)]
        actions.append(self.eos)
        return actions

    def get_policy_output(self, params: dict):
        """
        Defines the structure of the output of the policy model, from which an
        action is to be determined or sampled, by returning a vector with a fixed
        random policy.

        For each dimension of the hyper-torus, the output of the policy should return
        1) a logit, for the categorical distribution over dimensions and 2) the
        location and 3) the concentration of the projected normal distribution to
        sample the increment of the angle and 4) (if do_nonzero_source_prob is True)
        the logit of a Bernoulli distribution to model the (discrete) backward
        probability of returning to the value of the source node.

        Thus:
        - n_params_per_dim = 4 if do_nonzero_source_prob is True
        - n_params_per_dim = 3 if do_nonzero_source_prob is False

        Therefore, the output of the policy model has dimensionality D x
        n_params_per_dim + 1, where D is the number of dimensions, and the elements of
        the output vector are:
        - d * n_params_per_dim: logit of dimension d
        - d * n_params_per_dim + 1: location of Von Mises distribution for dimension d
        - d * n_params_per_dim + 2: log concentration of Von Mises distribution for dimension d
        - d * n_params_per_dim + 3: logit of Bernoulli distribution
        with d in [0, ..., D]
        """
        policy_output = torch.ones(
            self.n_dim * self.n_params_per_dim + 1, dtype=self.float, device=self.device
        )
        policy_output[1 :: self.n_params_per_dim] = params["vonmises_mean"]
        policy_output[2 :: self.n_params_per_dim] = params["vonmises_concentration"]
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
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(self.action_space_dim)]
        if state[-1] >= self.length_traj:
            mask = [True for _ in range(self.action_space_dim)]
            mask[-1] = False
        else:
            mask = [False for _ in range(self.action_space_dim)]
            mask[-1] = True
        return mask

    def get_mask_invalid_actions_backward(self, state=None, done=None, parents_a=None):
        """
        Returns a vector with the length of the discrete part of the action space:
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
        else:
            mask = [False for _ in range(self.action_space_dim)]
            mask[-1] = True
        # Catch cases where it would not be possible to reach the initial state
        noninit_dims = [s for s, ss in zip(state[:-1], self.source_angles) if s != ss]
        if len(noninit_dims) > state[-1]:
            raise ValueError("This point in the code should never be reached!")
        elif len(noninit_dims) == state[-1] and len(noninit_dims) >= state[-1] - 1:
            mask = [
                True if s == ss else m
                for m, s, ss in zip(mask, state[:-1], self.source_angles)
            ] + [mask[-1]]
        return mask

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
        # TODO: we might have to include the valid discrete backward actions for the
        # backward sampling. Otherwise, implement backward mask.
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
            state[dim] = (state[dim] - incr) % (2 * np.pi)
            state[-1] -= 1
            parents = [state]
            return parents, [action]

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
        Samples a batch of actions from a batch of policy outputs.
        """
        device = policy_outputs.device
        n_states = policy_outputs.shape[0]
        ns_range = torch.arange(n_states).to(device)
        # Sample dimensions
        if sampling_method == "uniform":
            logits_dims = torch.ones(n_states, self.policy_output_dim).to(device)
        elif sampling_method == "policy":
            logits_dims = policy_outputs[:, 0 :: self.n_params_per_dim]
            logits_dims /= temperature_logits
        if mask is not None:
            logits_dims[mask] = -torch.inf
        dimensions = Categorical(logits=logits_dims).sample()
        logprobs_dim = self.logsoftmax(logits_dims)[ns_range, dimensions]
        # Sample angle increments
        ns_range_noeos = ns_range[dimensions != self.eos[0]]
        dimensions_noeos = dimensions[dimensions != self.eos[0]]
        angles = torch.zeros(n_states).to(device)
        logprobs_angles = torch.zeros(n_states).to(device)
        if len(dimensions_noeos) > 0:
            if sampling_method == "uniform":
                distr_angles = Uniform(
                    torch.zeros(len(ns_range_noeos)),
                    2 * torch.pi * torch.ones(len(ns_range_noeos)),
                )
            elif sampling_method == "policy":
                locations = policy_outputs[:, 1 :: self.n_params_per_dim][
                    ns_range_noeos, dimensions_noeos
                ]
                concentrations = policy_outputs[:, 2 :: self.n_params_per_dim][
                    ns_range_noeos, dimensions_noeos
                ]
                distr_angles = VonMises(
                    locations,
                    torch.exp(concentrations) + self.vonmises_min_concentration,
                )
            angles[ns_range_noeos] = distr_angles.sample()
            logprobs_angles[ns_range_noeos] = distr_angles.log_prob(
                angles[ns_range_noeos]
            )
        # Combined probabilities
        logprobs = logprobs_dim + logprobs_angles
        # Build actions
        actions = [
            (dimension, angle)
            for dimension, angle in zip(dimensions.tolist(), angles.tolist())
        ]
        return actions, logprobs

    # TODO: deprecated
    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: TensorType["n_states", 2],
        mask: TensorType["batch_size", "policy_output_dim"] = None,
        states_from: Optional[List] = None,
        is_backward: bool = False,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs and actions.
        """
        device = policy_outputs.device
        dimensions, angles = zip(*actions)
        dimensions = torch.LongTensor([d.long() for d in dimensions]).to(device)
        angles = torch.FloatTensor(angles).to(device)
        n_states = policy_outputs.shape[0]
        ns_range = torch.arange(n_states).to(device)
        # Dimensions
        logits_dims = policy_outputs[:, 0 :: self.n_params_per_dim]
        if mask is not None:
            logits_dims[mask] = -torch.inf
        logprobs_dim = self.logsoftmax(logits_dims)[ns_range, dimensions]
        # Angle increments
        # Cases where p(angle) should be computed (nofix):
        # - A: Dimension != eos, and (
        # - B: (# dimensions different to source != # steps, or
        # - C: Angle of selected dimension != source) or
        # - D: is_forward)
        # nofix: A & ((B | C) | D)
        # Mixing p(angle) with discrete probability of going backwards to the source
        # The mixed (backward) probability of sampling angle, p(angle_mixed) is:
        # - p(angle) * p(no_source), if angle of target != source
        # - p(source), if angle of target == source
        # Mixing should be applied if p(angle) is computed AND is backward:
        source = torch.tensor(self.source_angles, device=device)
        source_aux = torch.tensor(self.source_angles + [-1], device=device)
        nsource_ne_nsteps = torch.ne(
            torch.sum(torch.ne(states_to[:, :-1], source), axis=1),
            states_to[:, -1],
        )
        angledim_ne_source = torch.ne(
            states_to[ns_range, dimensions], source_aux[dimensions]
        )
        noeos = torch.ne(dimensions, self.eos[0])
        nofix_indices = torch.logical_and(
            torch.logical_or(nsource_ne_nsteps, angledim_ne_source) | is_forward, noeos
        )
        logprobs_angles = torch.zeros(n_states).to(device)
        logprobs_nosource = torch.zeros(n_states).to(device)
        if torch.any(nofix_indices):
            ns_range_nofix = ns_range[nofix_indices]
            dimensions_nofix = dimensions[nofix_indices]
            locations = policy_outputs[:, 1 :: self.n_params_per_dim][
                ns_range_nofix, dimensions_nofix
            ]
            concentrations = policy_outputs[:, 2 :: self.n_params_per_dim][
                ns_range_nofix, dimensions_nofix
            ]
            distr_angles = VonMises(
                locations,
                torch.exp(concentrations) + self.vonmises_min_concentration,
            )
            logprobs_angles[ns_range_nofix] = distr_angles.log_prob(
                angles[ns_range_nofix]
            )
            if self.n_params_per_dim == 4 and (not is_forward):
                logits_nosource = policy_outputs[:, 3 :: self.n_params_per_dim][
                    ns_range_nofix, dimensions_nofix
                ]
                distr_nosource = Bernoulli(logits=logits_nosource)
                logprobs_nosource[ns_range_nofix] = distr_nosource.log_prob(
                    angledim_ne_source[ns_range_nofix].to(self.float)
                )
        # Combined probabilities
        logprobs = logprobs_dim + logprobs_angles + logprobs_nosource
        return logprobs

    def step(
        self, action: Tuple[int, float], skip_mask_check: bool = False
    ) -> Tuple[List[float], Tuple[int, float], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple with two values:
            (dimension, magnitude).

        skip_mask_check : bool
            Ignored.

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
        # If done, return invalid
        if self.done:
            return self.state, action, False
        # If only possible action is eos, then force eos
        # If the number of actions is equal to maximum trajectory length
        elif self.n_actions == self.length_traj:
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # If action is eos, then it is invalid
        elif action == self.eos:
            return self.state, action, False
        # Otherwise perform action
        else:
            dim, incr = action
            self.n_actions += 1
            self.state[dim] += incr
            self.state[dim] = self.state[dim] % (2 * np.pi)
            self.state[-1] = self.n_actions
            return self.state, action, True

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
        linspace = np.linspace(0, 2 * np.pi, n_per_dim)
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
