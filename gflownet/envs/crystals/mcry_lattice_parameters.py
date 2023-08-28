"""
Classes to represent molecular crystal environments
"""
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torchtyping import TensorType

from gflownet.envs.grid import Grid
from gflownet.utils.crystals.constants import (
    CUBIC,
    HEXAGONAL,
    LATTICE_SYSTEMS,
    MONOCLINIC,
    ORTHORHOMBIC,
    RHOMBOHEDRAL,
    TETRAGONAL,
    TRICLINIC,
)


class MolCryLatticeParameters(Grid):
    """
    LatticeParameters environment for crystal structure generation.

    Models lattice parameters (three edge lengths and three angles describing unit cell) with the constraints
    given by the provided lattice system (see https://en.wikipedia.org/wiki/Bravais_lattice). This is implemented
    by inheriting from the discrete Grid environment, creating a mapping between cell position and edge length
    or angle, and imposing lattice system constraints on their values.

    Note that similar to the Grid environment, the values are initialized with zeros (or target angles, if they
    are predetermined by the lattice system), and can be only increased with a discrete steps.
    """

    def __init__(
            self,
            lattice_system: str,
            min_length: float = 1.0,
            max_length: float = 5.0,
            min_angle: float = 30.0,
            max_angle: float = 150.0,
            min_position: float = 0.0,  # fractional coordinates
            max_position: float = 1.0,
            min_theta: float = 0.0,  #
            max_theta: float = 90.0,
            min_phi: float = -180.0,
            max_phi: float = 180.0,
            min_roration: float = 0.0,
            max_rotation: float = 360.0,
            grid_size: int = 10,
            max_increment: int = 1,
            device='cpu',
            **kwargs,
    ):
        """
        Args
        ----------
        lattice_system : str
            One of the seven lattice systems.

        min_length : float
            Minimum value of the edge length.

        max_length : float
            Maximum value of the edge length.

        min_angle : float
            Minimum value of the angles.

        max_angle : float
            Maximum value of the angles.
        
        min_position: float
            Minimum value of molecule centroid in fractional coordinates of asymmetric unit, always 0
            
        max_position: float
            Maximum value of molecule centroid in fractional coordinates of asymmetric unit (only works for space groups with parallelpiped asymmetric units), always 1
            
        min_theta: 
            Minimum angle between z axis and xy plane for molecule pose, limited to upper half-sphere, never less than zero, 
        
        max_theta: 
            Maximum the above - never greater than pi/2, 
        
        min_phi:
            Minimum pose angle in the xy plane, 
            
        max_phi:
            Maximum pose angle in the xy plane, 
        
        min_roration:
            Minimum rotation angle about the unit vector (theta, phi), 
            
        max_rotation: 
            Maximum rotation angle about the unit vector (theta, phi), 

        grid_size : int
            Length of the underlying grid that is used to map discrete values to actual edge lengths and angles.

        max_increment : int
            Maximum increment of each dimension by the actions.
        """
        super().__init__(
            n_dim=12,
            length=grid_size,
            max_increment=max_increment,
            max_dim_per_action=3,
            device=device,
            **kwargs,
        )

        if lattice_system not in LATTICE_SYSTEMS:
            raise ValueError(
                f"Expected one of the keys or values from {LATTICE_SYSTEMS}, received {lattice_system}."
            )

        self.device = device
        self.lattice_system = lattice_system
        self.min_length = min_length
        self.max_length = max_length
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.grid_size = grid_size
        self.min_position = min_position
        self.max_position = max_position
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.min_phi = min_phi
        self.max_phi = max_phi
        self.min_rotation = min_roration
        self.max_rotation = max_rotation

        # we ensure that 90 and 120 degrees angle are present in the search space,
        # since for some systems they must be set to one of these values
        angles = np.linspace(min_angle, max_angle, grid_size + 1)
        angles[np.abs(angles - 90.0).argmin()] = 90.0
        angles[np.abs(angles - 120.0).argmin()] = 120.0
        lengths = np.linspace(min_length, max_length, grid_size + 1)
        positions = np.linspace(min_position, max_position, grid_size + 1)
        thetas = np.linspace(min_theta, max_theta, grid_size + 1)
        phis = np.linspace(min_phi, max_phi, grid_size + 1)
        rotations = np.linspace(min_roration, max_rotation, grid_size + 1)  # TODO this results in a significantly finer grid in this direction than for phi

        self.cell2angle = {k: v for k, v in enumerate(angles)}
        self.angle2cell = {v: k for k, v in self.cell2angle.items()}
        self.cell2length = {k: v for k, v in enumerate(lengths)}
        self.length2cell = {v: k for k, v in self.cell2length.items()}
        self.cell2position = {k: v for k, v in enumerate(positions)}
        self.position2cell = {v: k for k, v in self.cell2position.items()}
        self.cell2theta = {k: v for k, v in enumerate(thetas)}
        self.theta2cell = {v: k for k, v in self.cell2theta.items()}
        self.cell2phi = {k: v for k, v in enumerate(phis)}
        self.phi2cell = {v: k for k, v in self.cell2phi.items()}
        self.cell2rotation = {k: v for k, v in enumerate(rotations)}
        self.rotation2cell = {v: k for k, v in self.cell2rotation.items()}

        self.angles_tensor = Tensor(angles).to(self.device)
        self.lengths_tensor = Tensor(lengths).to(self.device)
        self.positions_tensor = Tensor(positions).to(self.device)
        self.thetas_tensor = Tensor(thetas).to(self.device)
        self.phis_tensor = Tensor(phis).to(self.device)
        self.rotations_tensor = Tensor(rotations).to(self.device)

        if (
                90.0 not in self.cell2angle.values()
                or 120.0 not in self.cell2angle.values()
        ):
            raise ValueError(
                f"Given min_angle = {min_angle}, max_angle = {max_angle} and grid_size = {grid_size}, "
                f"possible discrete angle values {tuple(self.cell2angle.values())} do not include either "
                f"the 90 degrees or 120 degrees angle, which must both be present."
            )

        self._set_source()
        self.reset()

    def _set_source(self):
        """
        Helper that sets self.source depending on the given self.lattice_system. For systems that have
        specific angle requirements, they will be preset to these values.
        """
        if self.lattice_system == CUBIC:
            angles = [90.0, 90.0, 90.0]
        elif self.lattice_system == HEXAGONAL:
            angles = [90.0, 90.0, 120.0]
        elif self.lattice_system == MONOCLINIC:
            angles = [90.0, self.min_angle, 90.0]  # TODO ask alex about this state initialization
        elif self.lattice_system == ORTHORHOMBIC:
            angles = [90.0, 90.0, 90.0]
        elif self.lattice_system == RHOMBOHEDRAL:
            angles = [self.min_angle, self.min_angle, self.min_angle]
        elif self.lattice_system == TETRAGONAL:
            angles = [90.0, 90.0, 90.0]
        elif self.lattice_system == TRICLINIC:
            angles = [self.min_angle, self.min_angle, self.min_angle]
        else:
            raise NotImplementedError(
                f"Unspecified lattice system {self.lattice_system}."
            )

        self.source = [0, 0, 0] + [self.angle2cell[angle] for angle in angles] + [0, 0, 0, 0, 0, 0]  # lengths + angles + positions + pose angles

    def get_action_space(self) -> List[Tuple[int]]:
        """
        Constructs list with all possible actions, including eos.

        The action is described by a 12-dimensional tuple, i-th value of which corresponds to increasing the
        i-th value of state by action[i].

        State is encoded as a 12-dimensional list of numbers: the first three describe edge lengths,
        the next three angles, the next 3 the fractional position of the molecule centroid,
        the last three the directrion (spherical basis) and length of the rotation vector applied to the molecule pose.
        Note that they are not directly lengths and angles, but rather integer values
        from 0 to self.grid_size, that can be mapped to actual lengths and angles using self.cell2length,
        self.cell2angle, etc., respectively.

        In the case of lengths the allowed actions are:  # todo no negative n? related to initialization? ask alex
            - increment a by n,
            - increment b by n,
            - increment c by n,
            - increment both a and b by n (required by hexagonal and tetragonal lattice systems,
                for which a == b =/= c),
            - increment all a, b and c by n (required by cubic and rhombohedral lattice systems, for which
                a == b == c).

        In the case of angles the allowed actions are:
            - increment alpha by n,
            - increment beta by n,
            - increment gamma by n,
            - increment all alpha, beta and gama by n (required by rhombohedral lattice systems, for which
                alpha == beta == gamma =/= 90 degrees).

        In the case of cell positions and pose angles, the allowed actions are:
            - increment x,y,z,theta,phi,r by n (i.e. there are no limitations other than the min and max values)
            (for certain uncommon high symmetry space groups, x,y,z will have extra planes they cannot cross - will cross this bridge when we get to it)
        """
        valid_steplens = [el for el in range(1, self.max_increment + 1)]
        actions = []

        # lengths
        for r in valid_steplens:
            for dim in [0, 1, 2]:
                action = [0 for _ in range(12)]
                action[dim] = r
                actions.append(tuple(action))
            actions.append((r, r, 0) + tuple([0 for _ in range(9)]))
            actions.append((r, r, r) + tuple([0 for _ in range(9)]))

        # angles
        for r in valid_steplens:
            for dim in [3, 4, 5]:
                action = [0 for _ in range(12)]
                action[dim] = r
                actions.append(tuple(action))
            actions.append((0, 0, 0, r, r, r) + tuple([0 for _ in range(6)]))

        # positions & pose angles
        for r in valid_steplens:
            for dim in [6, 7, 8, 9, 10, 11]:
                action = [0 for _ in range(12)]
                action[dim] = r
                actions.append(tuple(action))

        actions.append(self.eos)

        return actions

    def _unpack_cell_params(
            self, state: Optional[List[int]] = None
    ) -> Tuple[Tuple, Tuple, Tuple, Tuple]:
        """
        Helper that 1) unpacks values coding edge lengths and angles (in the grid cell format)
        from the state, and 2) converts them to actual edge lengths and angles.
        """
        if state is None:
            state = self.state.copy()

        a, b, c = [self.cell2length[s] for s in state[:3]]
        alpha, beta, gamma = [self.cell2angle[s] for s in state[3:6]]
        xbar, ybar, zbar = [self.cell2position[s] for s in state[6:9]]
        theta, phi, rotation = [self.cell2theta[state[9]], self.cell2phi[state[10]], self.cell2rotation[state[11]]]

        return (a, b, c), (alpha, beta, gamma), (xbar, ybar, zbar), (theta, phi, rotation)

    def _are_intermediate_lengths_valid(
            self, state: Optional[List[int]] = None
    ) -> bool:
        """
        Helper that check whether the intermediate constraints defined by self.lattice_system for lengths are met.

        For intermediate state, we want to ensure that (CUBIC, RHOMBOHEDRAL) lattice systems allow only simultaneous
        change of all three edge lengths, and (HEXAGONAL, TETRAGONAL) lattice systems disallow changing
        a and b independently.
        """
        (a, b, c), _, _, _ = self._unpack_cell_params(state)

        if self.lattice_system in [CUBIC, RHOMBOHEDRAL]:
            return a == b == c
        elif self.lattice_system in [HEXAGONAL, TETRAGONAL]:
            return a == b
        elif self.lattice_system in [MONOCLINIC, ORTHORHOMBIC, TRICLINIC]:
            return True
        else:
            raise NotImplementedError

    def _are_final_lengths_valid(self, state: Optional[List[int]] = None) -> bool:
        """
        Helper that check whether the final constraints defined by self.lattice_system for lengths are met.
        """
        (a, b, c), _, _, _ = self._unpack_cell_params(state)

        if self.lattice_system in [CUBIC, RHOMBOHEDRAL]:
            return a == b == c
        elif self.lattice_system in [HEXAGONAL, TETRAGONAL]:
            return a == b != c
        elif self.lattice_system in [MONOCLINIC, ORTHORHOMBIC, TRICLINIC]:
            return a != b and a != c and b != c
        else:
            raise NotImplementedError

    def _are_intermediate_angles_valid(self, state: Optional[List[int]] = None) -> bool:
        """
        Helper that check whether the intermediate constraints defined by self.lattice_system for angles are met.

        For intermediate state, we want to ensure that (CUBIC, HEXAGONAL, MONOCLINIC, ORTHORHOMBIC, TETRAGONAL)
        lattice systems disallow change of the angles that have only one valid value, and for RHOMBOHEDRAL that
        only simultaneous change of all three angles is allowed.
        """
        _, (alpha, beta, gamma), _, _ = self._unpack_cell_params(state)

        if self.lattice_system in [CUBIC, ORTHORHOMBIC, TETRAGONAL]:
            return alpha == beta == gamma == 90.0
        elif self.lattice_system == HEXAGONAL:
            return alpha == beta == 90.0 and gamma == 120.0
        elif self.lattice_system == MONOCLINIC:
            return alpha == gamma == 90.0
        elif self.lattice_system == RHOMBOHEDRAL:
            return alpha == beta == gamma
        elif self.lattice_system == TRICLINIC:
            return True
        else:
            raise NotImplementedError

    def _are_final_angles_valid(self, state: Optional[List[int]] = None) -> bool:
        """
        Helper that check whether the final constraints defined by self.lattice_system for angles are met.
        """
        _, (alpha, beta, gamma), _, _ = self._unpack_cell_params(state)

        if self.lattice_system in [CUBIC, ORTHORHOMBIC, TETRAGONAL]:
            return alpha == beta == gamma == 90.0
        elif self.lattice_system == HEXAGONAL:
            return alpha == beta == 90.0 and gamma == 120.0
        elif self.lattice_system == MONOCLINIC:
            return alpha == gamma == 90.0 and beta != 90.0
        elif self.lattice_system == RHOMBOHEDRAL:
            return alpha == beta == gamma != 90.0
        elif self.lattice_system == TRICLINIC:
            return len({alpha, beta, gamma, 90.0}) == 4
        else:
            raise NotImplementedError

    def _is_intermediate_state_valid(self, state: List[int]) -> bool:
        """
        Helper that checks whether the given state meets intermediate self.lattice_system constraints.
        """
        return self._are_intermediate_lengths_valid(
            state
        ) and self._are_intermediate_angles_valid(state)

    def _is_final_state_valid(self, state: List[int]) -> bool:
        """
        Helper that checks whether the given state meets final self.lattice_system constraints.
        """
        return self._are_final_lengths_valid(state) and self._are_final_angles_valid(
            state
        )

    def get_mask_invalid_actions_forward(
            self, state: Optional[List[int]] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Returns a vector of length equal to that of the action space: True if forward action is
        invalid given the current state, False otherwise.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done

        mask = super().get_mask_invalid_actions_forward(state=state, done=done)

        # eos invalid if final lattice system constraints not met
        mask[-1] = not self._is_final_state_valid(state)

        # actions invalid if intermediate lattice system constraints not met
        for idx, a in enumerate(self.action_space[:-1]):
            child = state.copy()
            for d, incr in enumerate(a):
                child[d] += incr
            if not self._is_intermediate_state_valid(child):
                mask[idx] = True

        # If there are no valid actions (which can happen if we set all of the dimensions
        # to their maximum values, and one of the constraints is not satisfied), force eos
        # to be valid to avoid getting stuck in an infinite loop during sampling.
        if all(mask):
            mask[-1] = False

        return mask

    def state2oracle(self, state: Optional[List[int]] = None) -> Tensor:
        """
        Prepares a list of states in "GFlowNet format" for the oracle.

        Args
        ----
        state : list
            A state.

        Returns
        ----
        oracle_state : Tensor
            Tensor containing lengths and angles converted from the Grid format.
        """
        if state is None:
            state = self.state.copy()

        return Tensor(
            [self.cell2length[s] for s in state[:3]]
            + [self.cell2angle[s] for s in state[3:6]]
            + [self.cell2position[s]] for s in state[6:9]
            + [self.cell2theta[state[9]]]
            + [self.cell2phi[state[10]]]
            + [self.cell2rotation[state[11]]]
        )

    def statetorch2oracle(
            self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in "GFlowNet format" for the oracle. The input to the
        oracle is the lengths and angles, positions and pose angles

        Args
        ----
        states : Tensor
            A state

        Returns
        ----
        oracle_states : Tensor
        """
        if states.device != self.lengths_tensor.device:  # todo make this device assignment more bautifully or preferably at init
            self.lengths_tensor = self.lengths_tensor.cuda()
            self.angles_tensor = self.angles_tensor.cuda()
            self.positions_tensor = self.positions_tensor.cuda()
            self.thetas_tensor = self.thetas_tensor.cuda()
            self.phis_tensor = self.phis_tensor.cuda()
            self.rotations_tensor = self.rotations_tensor.cuda()
        return torch.cat(
            [
                self.lengths_tensor[states[:, :3].long()],
                self.angles_tensor[states[:, 3:6].long()],
                self.positions_tensor[states[:, 6:9].long()],
                self.thetas_tensor[states[:, 9].long()][:, None],
                self.phis_tensor[states[:, 10].long()][:, None],
                self.rotations_tensor[states[:, 11].long()][:, None]
            ],
            dim=1,
        )

    def state2readable(self, state: Optional[List[int]] = None) -> str:
        """
        Converts the state into a human-readable string in the format "(a, b, c), (alpha, beta, gamma)".
        """
        if state is None:
            state = self.state
        lengths, angles, positions, (thetas, phis, rotations) = self._unpack_cell_params(state)
        return f"{lengths}, {angles}, {positions}, {thetas}, {phis}, {rotations}"

    def readable2state(self, readable: str) -> List[int]:
        """
        Converts a human-readable representation of a state into the standard format.
        """
        state = []

        for c in ["(", ")", " "]:
            readable = readable.replace(c, "")
        values = readable.split(",")

        if len(values) != 12:
            raise ValueError(
                f"Expected readable to split into 12 distinct values, got {len(values)} values = {values}."
            )

        for v in values[:3]:
            s = self.length2cell.get(float(v))
            if s is None:
                raise ValueError(
                    f'Unrecognized key "{float(v)}" in self.length2cell = {self.length2cell}.'
                )
            state.append(s)

        for v in values[3:6]:
            s = self.angle2cell.get(float(v))
            if s is None:
                raise ValueError(
                    f'Unrecognized key "{float(v)}" in self.angle2cell = {self.angle2cell}.'
                )
            state.append(s)

        for v in values[6:9]:
            s = self.position2cell.get(float(v))
            if s is None:
                raise ValueError(
                    f'Unrecognized key "{float(v)}" in self.position22cell = {self.position2cell}.'
                )
            state.append(s)

        v = values[9]
        s = self.theta2cell.get(float(v))
        if s is None:
            raise ValueError(
                f'Unrecognized key "{float(v)}" in self.theta2cell = {self.theta2cell}.'
            )
        state.append(s)

        v = values[10]
        s = self.phi2cell.get(float(v))
        if s is None:
            raise ValueError(
                f'Unrecognized key "{float(v)}" in self.phi2cell = {self.phi2cell}.'
            )
        state.append(s)

        v = values[11]
        s = self.rotation2cell.get(float(v))
        if s is None:
            raise ValueError(
                f'Unrecognized key "{float(v)}" in self.rotation2cell = {self.rotation2cell}.'
            )
        state.append(s)

        return state

    def get_parents(
            self,
            state: Optional[List[int]] = None,
            done: Optional[bool] = None,
            action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to state.

        Args
        ----
        state : list
            Representation of a state.

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        action : tuple
            Last action performed.

        Returns
        -------
        parents : list
            List of parents in state format.

        actions : list
            List of actions that lead to state for each parent in parents.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [state], [self.eos]

        parents = []
        actions = []

        for i, (parent, action) in enumerate(
                zip(*super().get_parents(state, done, action))
        ):
            if self._is_intermediate_state_valid(parent):
                parents.append(parent)
                actions.append(action)

        return parents, actions
