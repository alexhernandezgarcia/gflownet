"""
Ising model: The sample space is an ensemble of discrete variables that can take two
values (-1 or +1), arranged spatially in a D-dimensional grid. The state space includes
intermediate states with variables at a neutral (not yet selected) spin and states that
indicate the variable to be set or unset.
"""

from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tfloat, tlong

TOGGLE_VARIABLE = 0
SET_SPIN = 1
UNSET = 0
TOGGLED = 3
STATE_TYPES = {"neutral": 0, "toggled": 3, "set": 1}


class Ising(GFlowNetEnv):
    """
    Ising environment.

    States are represented by a D-dimensional numpy array, with D the dimensionality of
    the Ising model. Each entry of the array corresponds to a discrete variable.
    Terminating states must have all variables in one of the two spins, namely -1 or 1.
    However, the following intermediate values are also defined:
        - 0: unselected state, indicating that the variable has not been set.
        - 3: selected state, indicating that the variable has been selected for
          assignment of a spin.
        - -2: transitory state indicating that the variable has spin -1, but the
          variable needs to be toggled to move to a neutral state.
        - 2: transitory state indicating that the variable has spin 1, but the
          variable needs to be toggled to move to a neutral state.

    The actions of the environment can be of the following types:
        - Toggle variable: one action per variable or cell.
        - Select spin: -1 or 1

    Attributes
    ----------
    n_dim : int
        The dimensionality of the Ising model. Default: 2
    length : int
        The number of variables or cells per dimension. Default: 4
    """

    def __init__(
        self,
        n_dim: int = 2,
        length: int = 4,
        **kwargs,
    ):
        """
        Initializes an Ising environment.

        Parameters
        ----------
        n_dim : int
            The dimensionality of the Ising model. Default: 2
        length : int
            The number of variables or cells per dimension. Default: 4
        """
        self.n_dim = n_dim
        self.length = length
        # Source state: numpy array initialized to zeros
        self.source = np.zeros((self.length,) * self.n_dim, dtype=int)
        # End-of-sequence action
        self.eos = (-1, -1)
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self) -> List[Tuple]:
        """
        Constructs list with all possible actions, including EOS.

        Actions are represented by two elements:
            - Action type: select variable (``TOGGLE_VARIABLE``) or set spin
              (``SET_SPIN``)
            - Value: value of the action, namely the index of the variable for
              selecting variables and -1 or 1 for setting the spin.

        Returns
        -------
        list
            A list of tuples representing the actions.
        """
        actions_select_variable = [
            (TOGGLE_VARIABLE, idx) for idx in range(self.length**self.n_dim)
        ]
        actions_set_spin = [(SET_SPIN, -1), (SET_SPIN, 1)]
        return actions_select_variable + actions_set_spin + [self.eos]

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
    ) -> List[bool]:
        """
        Returns a list of length the action space indicating which actions are invalid
        (True) and which are not invalid (False).

        Parameters
        ----------
        state : np.array
            Input state. If None, self.state is used.
        done : bool
            Whether the trajectory is done. If None, self.done is used.

        Returns
        -------
        A list of boolean values.
        """
        state = self._get_state(state)
        done = self._get_done(done)

        # Initialize mask to all invalid
        mask = [True for _ in range(self.action_space_dim)]

        # If done is True, return all invalid
        if done:
            return mask

        # Check if any variable is toggled and build the mask
        indices_toggled = np.where(state == TOGGLED)
        if len(indices_toggled[0]) > 0:
            # The two actions to set the spin are valid
            mask[-2] = False
            mask[-3] = False
            return mask

        # Check if any variable is in set and toggled state, and build mask
        indices_set = np.where(np.logical_or(state == -2, state == 2))
        if len(indices_set[0]) > 0:
            idx = np.ravel_multi_index(indices_set, state.shape)[0]
            # Only valid action is toggling the set variable
            mask[idx] = False
            return mask

        # Get indices of unset variables
        indices_unset = np.where(state == UNSET)
        if len(indices_unset[0]) > 0:
            # Make the actions to toggle the unset variables valid
            for idx in np.ravel_multi_index(indices_unset, state.shape):
                mask[idx] = False
            return mask

        # If there are no unset variables, then all are set and the only valid
        # action is EOS
        mask[-1] = False
        return mask

    def get_parents(
        self,
        state: Optional[List[int]] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        Parameters
        ----------
        state : np.array
            Input state. If None, self.state is used.
        done : bool
            Whether the trajectory is done. If None, self.done is used.
        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in state format. This environment has a single parent per
            state.

        actions : list
            List of actions that lead to state for each parent in parents. This
            environment has a single parent per state.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if done:
            return [state], [self.eos]
        if self.equal(state, self.source):
            return [], []

        parents = []
        p_actions = []

        # Check if any variable is toggled: The only parent is the state before
        # toggling the toggled action
        indices_toggled = np.where(state == TOGGLED)
        if len(indices_toggled[0]) > 0:
            idx = np.ravel_multi_index(indices_toggled, state.shape)[0]
            parent = state.copy()
            parent[indices_toggled] = UNSET
            parents.append(parent)
            p_actions.append((TOGGLE_VARIABLE, idx))
            return parents, p_actions

        # Check if any variable is in set and toggled state: the only parent is the
        # state before setting the spin (with the variable toggled) and the action is
        # setting the corresponding spin
        indices_set_toggled = np.where(np.logical_or(state == -2, state == 2))
        if len(indices_set_toggled[0]) > 0:
            idx = np.ravel_multi_index(indices_set_toggled, state.shape)[0]
            spin = state[indices_set_toggled][0] // 2
            parent = state.copy()
            parent[indices_set_toggled] = TOGGLED
            parents.append(parent)
            p_actions.append((SET_SPIN, spin))
            return parents, p_actions

        # For neutral states (no toggled variables) the parents are states that have
        # as a toggled variable any of the variables that are set in the current state
        indices_set = np.where(np.logical_or(state == -1, state == 1))
        indices_ravelled = np.ravel_multi_index(indices_set, state.shape)
        for idx, indices in enumerate(zip(*indices_set)):
            parent = state.copy()
            parent[indices] = 2 * state[indices]
            parents.append(parent)
            p_actions.append((TOGGLE_VARIABLE, indices_ravelled[idx]))
        return parents, p_actions

    def step(
        self, action: Tuple[int, int], skip_mask_check: bool = False
    ) -> [List[int], Tuple[int], bool]:
        """
        Executes step given an action.

        Variables follow the following cycle in order to be assigned a spin:
            1. An unselected variable (0) is toggled, turning its value into 3.
            2. A spin is assigned, turning its value into -2 (for spin -1) or 2 (for
            spin 1). This is a transitory state where the variable has been assigned a
            spin but is still toggled.
            3. The variable is toggled and receives its final value (-1 from -2) or (1
            from 2).

        Parameters
        ----------
        action : tuple
            Action to be executed. An action is a tuple with two elements indicating
            the action type and action value.
        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The state after executing the action
        action : tuple
            Action executed
        valid : bool
            False, if the action is not allowed for the current state.
        """
        # Generic pre-step checks
        do_step, self.state, action = self._pre_step(
            action=action,
            backward=False,
            skip_mask_check=skip_mask_check or self.skip_mask_check,
        )
        if not do_step:
            return self.state, action, False
        valid = True
        self.n_actions += 1

        # If action is EOS, set done to True and return state as is
        if action == self.eos:
            self.done = True
            return self.state, action, valid

        # Update state
        # Action is to toggle variable
        action_type, action_value = action
        if action_type == TOGGLE_VARIABLE:
            indices = np.unravel_index(action_value, self.state.shape)
            value = self.state[indices]
            # If variable is unset, turn it to toggled
            if value == UNSET:
                self.state[indices] = TOGGLED
            elif value == -2:
                self.state[indices] = -1
            elif value == 2:
                self.state[indices] = 1
            else:
                raise ValueError(
                    f"Unexpected variable value ({value}) for toggle action"
                )
        # Action is to set spin
        elif action_type == SET_SPIN:
            indices = np.where(self.state == TOGGLED)
            self.state[indices] = 2 * action_value
        else:
            raise ValueError(f"Unknown action type {action[0]}")
        return self.state, action, valid

    def _get_max_trajectory_length(self) -> int:
        """
        Returns the maximum trajectory length of the environment.

        Each variable needs three steps to be assigned a spin and there are length **
        n_dim variables. Plus EOS.
        """
        return 3 * (self.length**self.n_dim) + 1

    def _get_state_type(state: npt.NDArray = None) -> int:
        if len(np.where(self.state == TOGGLED)[0]) > 0:
            return STATE_TYPES["toggled"]

    def states2policy(self, states: List) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: states
        are one-hot encoded.

        The one-hot-encoding is a 2D tensor, with as many rows as entries in the state,
        that is length * n_dim. Each row is a one-hot-encoding of each variable. There
        are 6 possible values: 0, 3, -2, -1, 1, 2.

        Parameters
        ----------
        states : list
            A batch of states in environment format

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        n_states = len(states)
        states_policy = tlong(states, device=self.device)
        # Add offset of 2 in order to make all entries non-negative so that one-hot
        # encoding works as expected
        states_policy += 2
        states_policy = F.one_hot(states_policy, num_classes=6)
        return tfloat(
            states_policy.reshape(n_states, -1),
            float_type=self.float,
            device=self.device,
        )

    def readable2state(self, readable, alphabet={}):
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        readable = readable.replace("\n", "").replace("[", "").replace("]", "")
        readable = readable.replace("  ", " ")
        state = [int(el) for el in readable.strip().split(" ")]
        return np.array(state).reshape((self.length,) * self.n_dim)

    def state2readable(self, state: Optional[List] = None, alphabet={}):
        """
        Converts a state (a list of positions) into a human-readable string
        representing a state.
        """
        state = self._get_state(state)
        return str(state)
