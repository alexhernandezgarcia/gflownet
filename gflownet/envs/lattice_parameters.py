"""
Classes to represent crystal environments
"""
from typing import List

import numpy as np
from torch import Tensor

from gflownet.envs.grid import Grid


CUBIC = "cubic"
HEXAGONAL = "hexagonal"
MONOCLINIC = "monoclinic"
ORTHORHOMBIC = "orthorhombic"
RHOMBOHEDRAL = "rhombohedral"
TETRAGONAL = "tetragonal"
TRICLINIC = "triclinic"

LATTICE_SYSTEMS = [
    CUBIC,
    HEXAGONAL,
    MONOCLINIC,
    ORTHORHOMBIC,
    RHOMBOHEDRAL,
    TETRAGONAL,
    TRICLINIC,
]


class LatticeParameters(Grid):
    """
    LatticeParameters environment for crystal structure generation.

    Models lattice parameters (three angles and three edge lengths describing unit cell) with the constraints
    given by the provided lattice system (see https://en.wikipedia.org/wiki/Bravais_lattice). This is implemented
    by inheriting from the discrete Grid environment TODO
    """

    def __init__(
        self,
        lattice_system: str,
        min_angle: float = 30.0,
        max_angle: float = 150.0,
        min_length: float = 1.0,
        max_length: float = 5.0,
        grid_size: int = 61,
        min_step_len: int = 1,
        max_step_len: int = 1,
        **kwargs,
    ):
        """
        Args
        ----------
        TODO
        """
        super().__init__(
            n_dim=6,
            length=grid_size,
            min_step_len=min_step_len,
            max_step_len=max_step_len,
            **kwargs,
        )

        if lattice_system not in LATTICE_SYSTEMS:
            raise ValueError(
                f"Expected one of the keys or values from {LATTICE_SYSTEMS}, received {lattice_system}."
            )

        self.lattice_system = lattice_system
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.min_length = min_length
        self.max_length = max_length
        self.grid_size = grid_size

        self.cell2angle = {
            k: v for k, v in enumerate(np.linspace(min_angle, max_angle, grid_size))
        }
        self.angle2cell = {v: k for k, v in self.cell2angle.items()}
        self.cell2length = {
            k: v for k, v in enumerate(np.linspace(min_length, max_length, grid_size))
        }
        self.length2cell = {v: k for k, v in self.cell2length.items()}

        if 90 not in self.cell2angle.values() or 120 not in self.cell2angle.values():
            raise ValueError(
                f"Given min_angle = {min_angle}, max_angle = {max_angle} and grid_size = {grid_size}, "
                f"possible discrete angle values {tuple(self.cell2angle.values())} do not include either "
                f"the 90 degrees or 120 degrees angle, which must both be present."
            )

        self._set_source()
        self.reset()

    def _set_source(self):
        if self.lattice_system == CUBIC:
            angles = [90.0, 90.0, 90.0]
        elif self.lattice_system == HEXAGONAL:
            angles = [90.0, 90.0, 120.0]
        elif self.lattice_system == MONOCLINIC:
            angles = [90.0, 0.0, 90.0]
        elif self.lattice_system == ORTHORHOMBIC:
            angles = [90.0, 90.0, 90.0]
        elif self.lattice_system == RHOMBOHEDRAL:
            angles = [0.0, 0.0, 0.0]
        elif self.lattice_system == TETRAGONAL:
            angles = [90.0, 90.0, 90.0]
        elif self.lattice_system == TRICLINIC:
            angles = [0.0, 0.0, 0.0]
        else:
            raise NotImplementedError(
                f"Unspecified lattice system {self.lattice_system}."
            )

        return [0, 0, 0] + [self.angle2cell[angle] for angle in angles]

    def get_actions_space(self):
        """
        Constructs list with all possible actions, including eos.

        The action is described by a tuple of dimensions (possibly duplicate) that will all be incremented
        by 1, e.g. (0, 0, 0, 2, 4, 4, 4) would increment the 0th and the 4th dimension by 3, and 2nd by 1.

        State in encoded as a 6-dimensional list of numbers: the first three describe edge lengths,
        and the last three angles. Note that they are not directly lengths and angles, but rather integer values
        from 0 to self.grid_size, that can be mapped to actual lengths and angles using self.cell2length and
        self.cell2angle, respectively.

        In the case of lengths the allowed actions are:
            - increment a by n,
            - increment b by n,
            - increment c by n,
            - increment both a and b by n (required by hexagonal, monoclinic and tetragonal lattice systems,
                for which a == b =/= c),
            - increment all a, b and c by n (required by cubic and rhombohedral lattice systems, for which
                a == b == c).

        In the case of angles the allowed actions are:
            - increment alpha by n,
            - increment beta by n,
            - increment gamma by n,
            - increment all alpha, beta and gama by n (required by rhombohedral lattice systems, for which
                alpha == beta == gamma =/= 90 degrees).
        """
        valid_steplens = np.arange(self.min_step_len, self.max_step_len + 1)
        actions = []

        # lengths
        for r in valid_steplens:
            for dim in [0, 1, 2]:
                actions += (dim,) * r
            actions += (0, 1) * r
            actions += (0, 1, 2) * r

        # angles
        for r in valid_steplens:
            for dim in [3, 4, 5]:
                actions += (dim,) * r
            actions += (3, 4, 5) * r

        actions += [(self.eos,)]
        return actions

    def _are_lengths_valid(self, state=None):
        if state is None:
            state = self.state.copy()

        a, b, c = [self.cell2length[s] for s in state[:3]]

        if self.lattice_system in [CUBIC, RHOMBOHEDRAL]:
            return a == b == c
        elif self.lattice_system in [HEXAGONAL, MONOCLINIC, TETRAGONAL]:
            return a == b != c
        elif self.lattice_system in [ORTHORHOMBIC, TRICLINIC]:
            return a != b and a != c and b != c
        else:
            raise NotImplementedError

    def _are_angles_valid(self, state=None):
        if state is None:
            state = self.state.copy()

        alpha, beta, gamma = [self.cell2angle[s] for s in state[3:]]

        if self.lattice_system in [CUBIC, ORTHORHOMBIC, TETRAGONAL]:
            return alpha == beta == gamma == 90
        elif self.lattice_system == HEXAGONAL:
            return alpha == beta == 90 and gamma == 120
        elif self.lattice_system == MONOCLINIC:
            return alpha == gamma == 90 and beta != 90
        elif self.lattice_system == RHOMBOHEDRAL:
            return alpha == beta == gamma != 90
        elif self.lattice_system == TRICLINIC:
            return len({alpha, beta, gamma, 90}) == 4
        else:
            raise NotImplementedError

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """
        TODO
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done

        mask = super().get_mask_invalid_actions_forward(state=state, done=done)

        for idx, a in enumerate(self.action_space[:-1]):
            child = state.copy()
            for d in a:
                child[d] += 1
            if not (self._are_lengths_valid(child) and self._are_angles_valid(child)):
                mask[idx] = True

        return mask

    def state2oracle(self, state: List = None) -> Tensor:
        """
        Prepares a list of states in "GFlowNet format" for the oracle.

        Args
        ----
        state : list
            A state

        Returns
        ----
        oracle_state : Tensor
            Tensor containing lengths and angles converted from the Grid format.
        """
        raise Tensor(
            [self.cell2length[s] for s in state[:3]]
            + [self.cell2angle[s] for s in state[3:]]
        )
