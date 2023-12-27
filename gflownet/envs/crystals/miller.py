"""
Class to represent an environment to sample Miller indices (hkl).
"""
from typing import List, Optional

from gflownet.envs.grid import Grid
from gflownet.utils.crystals.constants import HEXAGONAL, RHOMBOHEDRAL


class MillerIndices(Grid):
    """
    The Miller indices are represented by either 3 parameters (h, k, l) in general, or
    by 4 parameters (h, k, i, l) if the structure is hexagonal or rhombohedral.
    However, for hexagonal and rhombohedral structures, there is redundancy in the
    parameters. h, k and i are united by h + k + i = 0. As such, even for hexagonal and
    rhombohedral structures, we only need to model 3 free dimensions.

    Each parameter can take values in the set {-2, -1, 0, 1, 2}.
    Therefore, we can represent the Miller indices environment by a hyper cube of
    length 5, with dimensionality 3.

    Attributes
    ----------
    is_hexagonal_rhombohedral : bool
        True if the structure is hexagonal or rhombohedral.

    max_increment : int
        Maximum increment of each dimension by the actions.

    max_dim_per_action : int
        Maximum number of dimensions to increment per action. If -1, then
        max_dim_per_action is set to n_dim.
    """

    def __init__(
        self,
        is_hexagonal_rhombohedral: bool,
        max_increment: int = 1,
        max_dim_per_action: int = 1,
        **kwargs,
    ):
        self.is_hexagonal_rhombohedral = is_hexagonal_rhombohedral
        super().__init__(n_dim=3, length=5, cell_min=-2, cell_max=2, **kwargs)

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
        state = self._get_state(state)
        done = self._get_done(done)

        mask = super().get_mask_invalid_actions_forward(state, done)

        # If the structure is hexagonal or rhombohedral, we need -2 <= h + k <= 2
        # for the state to be valid. This means that we have to enforce constraints
        # on the mask to ensure that the trajectory doesn't end in an invalid state.
        if self.is_hexagonal_rhombohedral:
            # Extract the miller indices from the state
            h, k, l = [self.cells[s] for s in state]

            # Enforce that the trajectory ends in a state satisfying -2 <= h + k.
            # This requirement won't be satisfied at the beginning of the trajectory
            # so we enforce it by disabling the EOS action until the condition is
            # met. This will force the agent to increment h and/or k until the
            # condition is, inevitably, satisfied.
            if h + k < -2:
                index_eos = self.action_space.index(self.eos)
                mask[index_eos] = True

            # Enforce that the trajectory ends in a state satisfying h + k <= 2.
            # This condition is satisfied at the environment's initial state. And
            # since the environment only offers actions that increment the miller
            # indices, it means that if the environment ever gets to a state that
            # DOESN'T satisfy this condition, there is no way to go back to a state
            # that does. Therefore we enforce this constraint by disabling the
            # actions that would lead to states that don't satisfy this condition.
            for action_idx, action in enumerate(self.action_space):
                increment_h, increment_k, increment_l = action
                new_sum_h_k = h + increment_h + k + increment_k
                if new_sum_h_k > 2:
                    mask[action_idx] = True

        return mask

    def set_lattice_system(self, lattice_system: str):
        """
        Sets the lattice system of the unit cell.

        The specific lattice system is not required in this environment. Instead, only
        is_hexagonal_rhombohedral is set to True or False according to the lattice
        system.
        """
        self.is_hexagonal_rhombohedral = lattice_system in [HEXAGONAL, RHOMBOHEDRAL]

    def _is_terminating(self, state: List) -> bool:
        """
        Determines whether a state can be a terminating state, given the constraints.

        All states can be terminating states, except if the lattice is hexagonal or
        rhombohedral and h + k < -2.

        Args
        ----
        state : list
            An input state

        Returns
        -------
        True, if the state can be terminating, given the constraints. False, otherwise.
        """
        if not self.is_hexagonal_rhombohedral:
            return True
        h, k, _ = [self.cells[s] for s in state]
        if h + k < -2:
            return False
        return True

    def get_all_terminating_states(self) -> List[List]:
        """
        Gets all terminating states. If is_hexagonal_rhombohedral is True, it discards
        the terminating states that do not satisfy the constraints.
        """
        states_all = super().get_all_terminating_states()
        if not self.is_hexagonal_rhombohedral:
            return states_all
        states_all_constraints = []
        for state in states_all:
            if self._is_terminating(state):
                states_all_constraints.append(state)
        return states_all_constraints

    def get_uniform_terminating_states(
        self, n_states: int, seed: int = None
    ) -> List[List]:
        """
        Gets uniformly sampled terminating states. If is_hexagonal_rhombohedral is
        True, it discards the terminating states that do not satisfy the constraints.
        """
        states_uniform = super().get_uniform_terminating_states()
        if not self.is_hexagonal_rhombohedral:
            return states_uniform
        states_uniform_constraints = []
        for state in states_uniform:
            if self._is_terminating(state):
                states_uniform_constraints.append(state)
        return states_uniform_constraints
