"""
Classes to represent a hyper-grid environments
"""

import itertools
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import tfloat, tlong, torch2np


class Grid(GFlowNetEnv):
    """
    Hyper-grid environment: A grid with n_dim dimensions and length cells per
    dimensions.

    The state space is the entire grid and each state is represented by the vector of
    coordinates of each dimensions. For example, in 3D, the origin will be at [0, 0, 0]
    and after incrementing dimension 0 by 2, dimension 1 by 3 and dimension 3 by 1, the
    state would be [2, 3, 1].

    The action space is the increment to be applied to each dimension. For instance,
    (0, 0, 1) will increment dimension 2 by 1 and the action that goes from [1, 1, 1]
    to [2, 3, 1] is (1, 2, 0).

    Attributes
    ----------
    n_dim : int
        Dimensionality of the grid

    length : int
        Size of the grid (cells per dimension)

    max_increment : int
        Maximum increment of each dimension by the actions.

    max_dim_per_action : int
        Maximum number of dimensions to increment per action. If -1, then
        max_dim_per_action is set to n_dim.

    cell_min : float
        Lower bound of the cells range

    cell_max : float
        Upper bound of the cells range
    """

    def __init__(
        self,
        n_dim: int = 2,
        length: int = 3,
        max_increment: int = 1,
        max_dim_per_action: int = 1,
        cell_min: float = -1,
        cell_max: float = 1,
        **kwargs,
    ):
        assert n_dim > 0
        assert length > 1
        assert max_increment > 0
        assert max_dim_per_action == -1 or max_dim_per_action > 0
        self.n_dim = n_dim
        self.length = length
        self.max_increment = max_increment
        if max_dim_per_action == -1:
            max_dim_per_action = self.n_dim
        self.max_dim_per_action = max_dim_per_action
        self.cells = np.linspace(cell_min, cell_max, length)
        # Source state: position 0 at all dimensions
        self.source = [0 for _ in range(self.n_dim)]
        # End-of-sequence action
        self.eos = tuple([0 for _ in range(self.n_dim)])
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self):
        """
        Constructs list with all possible actions, including eos. An action is
        represented by a vector of length n_dim where each index d indicates the
        increment to apply to dimension d of the hyper-grid.
        """
        increments = [el for el in range(self.max_increment + 1)]
        actions = []
        for action in itertools.product(increments, repeat=self.n_dim):
            if (
                sum(action) != 0
                and len([el for el in action if el > 0]) <= self.max_dim_per_action
            ):
                actions.append(tuple(action))
        actions.append(self.eos)
        return actions

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid from the current state.
            - False otherwise.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(self.policy_output_dim)]
        mask = [False for _ in range(self.policy_output_dim)]
        for idx, action in enumerate(self.action_space[:-1]):
            child = state.copy()
            for dim, incr in enumerate(action):
                child[dim] += incr
            if any(el >= self.length for el in child):
                mask[idx] = True
        return mask

    def states2proxy(
        self, states: Union[List[List], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states in "environment format" for the proxy: each state is
        a vector of length n_dim with values in the range [cell_min, cell_max].

        See: states2policy()

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tfloat(states, device=self.device, float_type=self.float)
        return (
            self.states2policy(states).reshape(
                (states.shape[0], self.n_dim, self.length)
            )
            * torch.tensor(self.cells[None, :]).to(states.device, self.float)
        ).sum(axis=2)

    def states2policy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: states
        are one-hot encoded.

        The output is a 2D tensor, with the second dimension of size length * n_dim,
        where each n-th successive block of length elements is a one-hot encoding of
        the position in the n-th dimension.

        Example (n_dim = 3, length = 4):
          - state: [0, 3, 1]
          - policy format: [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0]
                           |     0    |      3    |      1    |

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tlong(states, device=self.device)
        n_states = states.shape[0]
        cols = states + torch.arange(self.n_dim) * self.length
        rows = torch.repeat_interleave(torch.arange(n_states), self.n_dim)
        states_policy = torch.zeros(
            (n_states, self.length * self.n_dim), dtype=self.float, device=self.device
        )
        states_policy[rows, cols.flatten()] = 1.0
        return states_policy

    def readable2state(self, readable, alphabet={}):
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        return [int(el) for el in readable.strip("[]").split(" ") if el != ""]

    def state2readable(self, state: Optional[List] = None, alphabet={}):
        """
        Converts a state (a list of positions) into a human-readable string
        representing a state.
        """
        state = self._get_state(state)
        return str(state).replace("(", "[").replace(")", "]").replace(",", "")

    def get_parents(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state, as a list of length length where each element is
            the position at each dimension.

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : None
            Ignored

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
        else:
            parents = []
            actions = []
            for idx, action in enumerate(self.action_space[:-1]):
                parent = state.copy()
                for dim, incr in enumerate(action):
                    if parent[dim] - incr >= 0:
                        parent[dim] -= incr
                    else:
                        break
                else:
                    parents.append(parent)
                    actions.append(action)
        return parents, actions

    def step(
        self, action: Tuple[int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple int values indicating the
            dimensions to increment by 1.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        # Generic pre-step checks
        do_step, self.state, action = self._pre_step(
            action, skip_mask_check or self.skip_mask_check
        )
        if not do_step:
            return self.state, action, False
        # If only possible action is eos, then force eos
        # All dimensions are at the maximum length
        if all([s == self.length - 1 for s in self.state]):
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True
        # If action is not eos, then perform action
        elif action != self.eos:
            state_next = self.state.copy()
            for dim, incr in enumerate(action):
                state_next[dim] += incr
            if any([s >= self.length for s in state_next]):
                valid = False
            else:
                self.state = state_next
                valid = True
                self.n_actions += 1
            return self.state, action, valid
        # If action is eos, then perform eos
        else:
            self.done = True
            self.n_actions += 1
            return self.state, self.eos, True

    def get_max_traj_length(self):
        return self.n_dim * self.length

    def get_all_terminating_states(self) -> List[List]:
        grid = np.meshgrid(*[range(self.length)] * self.n_dim)
        all_x = np.stack(grid).reshape((self.n_dim, -1)).T
        return all_x.tolist()

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[List]:
        rng = np.random.default_rng(seed)
        states = rng.integers(low=0, high=self.length, size=(n_states, self.n_dim))
        return states.tolist()

    def plot_reward_samples(
        self,
        samples: TensorType["batch_size", "state_proxy_dim"],
        samples_reward: TensorType["batch_size", "state_proxy_dim"],
        rewards: TensorType["batch_size"],
        dpi: int = 150,
        n_ticks_max: int = 50,
        reward_norm: bool = True,
        **kwargs,
    ):
        """
        Plots the reward density as a 2D histogram on the grid, alongside a histogram
        representing the samples density.

        It is assumed that the rewards correspond to entire domain of the grid and are
        sorted from left to right (first) and top to bottom of the grid of samples.

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
            length ** 2 and be sorted such that the each block at rewards[i *
            length:i * length + length] correspond to the rewards at the i-th
            row of the grid of samples, from top to bottom.
        dpi : int
            Dots per inch, indicating the resolution of the plot.
        n_ticks_max : int
            Maximum of number of ticks to include in the axes.
        reward_norm : bool
            Whether to normalize the histogram. True by default.
        """
        # Only available for 2D grids
        if self.n_dim != 2:
            return None
        samples = torch2np(samples)
        rewards = torch2np(rewards)
        assert rewards.shape[0] == self.length**2
        # Init figure
        fig, axes = plt.subplots(ncols=2, dpi=dpi)
        step_ticks = np.ceil(self.length / n_ticks_max).astype(int)
        # 2D histogram of samples
        samples_hist, xedges, yedges = np.histogram2d(
            samples[:, 0], samples[:, 1], bins=(self.length, self.length), density=True
        )
        # Transpose and reverse rows so that [0, 0] is at bottom left
        samples_hist = samples_hist.T[::-1, :]
        # Normalize and reshape reward into a grid with [0, 0] at the bottom left
        if reward_norm:
            rewards = rewards / rewards.sum()
        rewards_2d = rewards.reshape(self.length, self.length).T[::-1, :]
        # Plot reward
        self._plot_grid_2d(rewards_2d, axes[0], step_ticks, title="True reward")
        # Plot samples histogram
        self._plot_grid_2d(samples_hist, axes[1], step_ticks, title="Samples density")
        fig.tight_layout()
        return fig

    @staticmethod
    def _plot_grid_2d(img: np.array, ax: Axes, step_ticks: int, title: str):
        """
        Plots a 2D histogram of a grid environment as an image.

        Parameters
        ----------
        img : np.array
            An array containing a 2D histogram over a grid.
        ax : Axes
            A matplotlib Axes object on which the image will be plotted.
        step_ticks : int
            The step value to add ticks to the axes. For example, if it is 2, the ticks
            will be at 0, 2, 4, ...
        title : str
            Title for the axes.
        """
        ax_img = ax.imshow(img)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        ax.set_xticks(np.arange(start=0, stop=img.shape[0], step=step_ticks))
        ax.set_yticks(np.arange(start=0, stop=img.shape[1], step=step_ticks)[::-1])
        cax.set_title(title)
        plt.colorbar(ax_img, cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("top")
