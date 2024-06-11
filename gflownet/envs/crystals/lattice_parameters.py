"""
Classes to represent continuous lattice parameters environments.

An implementation for discrete lattice parameters preceded this one but has been
removed for simplicity. Check commit 9f3477d8e46c4624f9162d755663993b83196546 to see
these changes or the history previous to that commit to consult previous
implementations.
"""

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torchtyping import TensorType

from gflownet.envs.cube import ContinuousCube
from gflownet.utils.common import copy, tfloat
from gflownet.utils.crystals.constants import (
    CUBIC,
    HEXAGONAL,
    MONOCLINIC,
    ORTHORHOMBIC,
    RHOMBOHEDRAL,
    TETRAGONAL,
    TRICLINIC,
)

LENGTH_PARAMETER_NAMES = ("a", "b", "c")
ANGLE_PARAMETER_NAMES = ("alpha", "beta", "gamma")
PARAMETER_NAMES = LENGTH_PARAMETER_NAMES + ANGLE_PARAMETER_NAMES


# TODO: figure out a way to inherit the (discrete) LatticeParameters env or create a
# common class for both discrete and continous with the common methods.
class LatticeParameters(ContinuousCube):
    """
    Continuous lattice parameters environment for crystal structures generation.

    Models lattice parameters (three edge lengths and three angles describing unit
    cell) with the constraints given by the provided lattice system (see
    https://en.wikipedia.org/wiki/Bravais_lattice). This is implemented by inheriting
    from the (continuous) cube environment, creating a mapping between cell position
    and edge length or angle, and imposing lattice system constraints on their values.

    The environment is a hyper cube of dimensionality 6 (the number of lattice
    parameters), but it takes advantage of the mask of ignored dimensions implemented
    in the Cube environment.

    The values of the state will remain in the default [0, 1] range of the Cube, but
    they are mapped to [min_length, max_length] in the case of the lengths and
    [min_angle, max_angle] in the case of the angles.
    """

    def __init__(
        self,
        lattice_system: str,
        min_length: Optional[float] = 1.0,
        max_length: Optional[float] = 350.0,
        min_angle: Optional[float] = 50.0,
        max_angle: Optional[float] = 150.0,
        **kwargs,
    ):
        """
        Args
        ----
        lattice_system : str
            One of the seven lattice systems. By default, the triclinic lattice system
            is used, which has no constraints.

        min_length : float
            Minimum value of the lengths.

        max_length : float
            Maximum value of the lengths.

        min_angle : float
            Minimum value of the angles.

        max_angle : float
            Maximum value of the angles.
        """
        self.continuous = True
        self.lattice_system = lattice_system
        self.min_length = min_length
        self.max_length = max_length
        self.length_range = self.max_length - self.min_length
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.angle_range = self.max_angle - self.min_angle
        super().__init__(n_dim=6, **kwargs)
        # Setup constraints after the call of super to avoid getting the variable
        # self.ignored_dims overriden by the Cube initialization
        self._setup_constraints()

    # TODO: if source, keep as is
    def _statevalue2length(self, value):
        return self.min_length + value * self.length_range

    def _length2statevalue(self, length):
        return (length - self.min_length) / self.length_range

    # TODO: if source, keep as is
    def _statevalue2angle(self, value):
        return self.min_angle + value * self.angle_range

    def _angle2statevalue(self, angle):
        return (angle - self.min_angle) / self.angle_range

    def _get_param(self, state, param):
        if hasattr(self, param):
            return getattr(self, param)
        else:
            if param in LENGTH_PARAMETER_NAMES:
                return self._statevalue2length(state[self._get_index_of_param(param)])
            elif param in ANGLE_PARAMETER_NAMES:
                return self._statevalue2angle(state[self._get_index_of_param(param)])
            else:
                raise ValueError(f"{param} is not a valid lattice parameter")

    def _set_param(self, state, param, value):
        param_idx = self._get_index_of_param(param)
        if param_idx is not None:
            if param in LENGTH_PARAMETER_NAMES:
                state[param_idx] = self._length2statevalue(value)
            elif param in ANGLE_PARAMETER_NAMES:
                state[param_idx] = self._angle2statevalue(value)
            else:
                raise ValueError(f"{param} is not a valid lattice parameter")
        return state

    def _get_index_of_param(self, param):
        param_idx = f"{param}_idx"
        if hasattr(self, param_idx):
            return getattr(self, param_idx)
        else:
            return None

    def set_lattice_system(self, lattice_system: str):
        """
        Sets the lattice system of the unit cell and updates the constraints.
        """
        self.lattice_system = lattice_system
        self._setup_constraints()

    def _setup_constraints(self):
        """
        Computes the mask of ignored dimensions, given the constraints imposed by the
        lattice system. Sets self.ignored_dims.
        """
        # Lengths: a, b, c
        # a == b == c
        if self.lattice_system in [CUBIC, RHOMBOHEDRAL]:
            lengths_ignored_dims = [False, True, True]
            self.a_idx = 0
            self.b_idx = 0
            self.c_idx = 0
        # a == b != c
        elif self.lattice_system in [HEXAGONAL, TETRAGONAL]:
            lengths_ignored_dims = [False, True, False]
            self.a_idx = 0
            self.b_idx = 0
            self.c_idx = 2
        # a != b and a != c and b != c
        elif self.lattice_system in [MONOCLINIC, ORTHORHOMBIC, TRICLINIC]:
            lengths_ignored_dims = [False, False, False]
            self.a_idx = 0
            self.b_idx = 1
            self.c_idx = 2
        else:
            raise ValueError(f"{self.lattice_system} is not a valid lattice system")
        # Angles: alpha, beta, gamma
        # alpha == beta == gamma == 90.0
        if self.lattice_system in [CUBIC, ORTHORHOMBIC, TETRAGONAL]:
            angles_ignored_dims = [True, True, True]
            self.alpha_idx = None
            self.alpha = 90.0
            self.alpha_state = self._angle2statevalue(self.alpha)
            self.beta_idx = None
            self.beta = 90.0
            self.beta_state = self._angle2statevalue(self.beta)
            self.gamma_idx = None
            self.gamma = 90.0
            self.gamma_state = self._angle2statevalue(self.gamma)
        #  alpha == beta == 90.0 and gamma == 120.0
        elif self.lattice_system == HEXAGONAL:
            angles_ignored_dims = [True, True, True]
            self.alpha_idx = None
            self.alpha = 90.0
            self.alpha_state = self._angle2statevalue(self.alpha)
            self.beta_idx = None
            self.beta = 90.0
            self.beta_state = self._angle2statevalue(self.beta)
            self.gamma_idx = None
            self.gamma = 120.0
            self.gamma_state = self._angle2statevalue(self.gamma)
        # alpha == gamma == 90.0 and beta != 90.0
        elif self.lattice_system == MONOCLINIC:
            angles_ignored_dims = [True, False, True]
            self.alpha_idx = None
            self.alpha = 90.0
            self.alpha_state = self._angle2statevalue(self.alpha)
            self.beta_idx = 4
            self.gamma_idx = None
            self.gamma = 90.0
            self.gamma_state = self._angle2statevalue(self.gamma)
        # alpha == beta == gamma != 90.0
        elif self.lattice_system == RHOMBOHEDRAL:
            angles_ignored_dims = [False, True, True]
            self.alpha_idx = 3
            self.beta_idx = 3
            self.gamma_idx = 3
        # alpha != beta, alpha != gamma, beta != gamma
        elif self.lattice_system == TRICLINIC:
            angles_ignored_dims = [False, False, False]
            self.alpha_idx = 3
            self.beta_idx = 4
            self.gamma_idx = 5
        else:
            raise NotImplementedError
        self.ignored_dims = lengths_ignored_dims + angles_ignored_dims

    def _step(
        self,
        action: Tuple[float],
        backward: bool,
    ) -> Tuple[List[float], Tuple[float], bool]:
        """
        Updates the dimensions of the state corresponding to the ignored dimensions
        after a call to the Cube's _step().
        """
        state, action, valid = super()._step(action, backward)
        for idx, (param, is_ignored) in enumerate(
            zip(PARAMETER_NAMES, self.ignored_dims)
        ):
            if not is_ignored:
                continue
            param_idx = self._get_index_of_param(param)
            if param_idx is not None:
                state[idx] = state[param_idx]
            else:
                state[idx] = getattr(self, f"{param}_state")
        self.state = copy(state)
        return self.state, action, valid

    def _unpack_lengths_angles(
        self, state: Optional[List[float]] = None
    ) -> Tuple[Tuple, Tuple]:
        """
        Helper that 1) unpacks values coding lengths and angles from the state or from
        the attributes of the instance and 2) converts them to actual edge lengths and
        angles in the target units (angstroms or degrees).
        """
        state = self._get_state(state)

        a, b, c, alpha, beta, gamma = [
            self._get_param(state, p) for p in PARAMETER_NAMES
        ]
        return (a, b, c), (alpha, beta, gamma)

    def parameters2state(
        self, parameters: Tuple = None, lengths: Tuple = None, angles: Tuple = None
    ) -> List[float]:
        """Converts a set of lattice parameters in angstroms and degrees into an
        environment state, with the parameters in the [0, 1] range.

        The parameters may be passed as a single tuple parameters containing the six
        parameters or via separate lengths and angles. If parameters is not None,
        lengths and angles are ignored.

        Parameters
        ----------
        parameters : tuple (optional)
            The six lattice parameters (a, b, c, alpha, beta, gamma) in target units
            (angstroms and degrees).
        lengths : tuple (optional)
            A triplet of length lattice parameters (a, b, c) in angstroms. Ignored if
            parameters is not None.
        angles : tuple (optional)
            A triplet of angle lattice parameters (alpha, beta, gamma) in degrees.
            Ignored if parameters is not None.

        Returns
        -------
        state
            A state in environment format.
        """
        if parameters is None:
            if lengths is None or angles is None:
                raise ValueError("Cannot determine all six parameters.")
            parameters = lengths + angles

        state = copy(self.source)
        for param, value in zip(PARAMETER_NAMES, parameters):
            state = self._set_param(state, param, value)
        return state

    def states2proxy(
        self, states: Union[List, TensorType["batch", "state_dim"]]
    ) -> TensorType["height", "width", "batch"]:
        """
        Prepares a batch of states in "environment format" for a proxy: states are
        mapped from [0; 1] to edge lengths and angles using min_length, max_length,
        min_angle and max_angle, via _statevalue2length() and _statevalue2angle().

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
        return torch.cat(
            [
                self._statevalue2length(states[:, :3]),
                self._statevalue2angle(states[:, 3:]),
            ],
            dim=1,
        )

    def state2readable(self, state: Optional[List[float]] = None) -> str:
        """
        Converts the state into a human-readable string in the format "(a, b, c),
        (alpha, beta, gamma)".
        """
        state = self._get_state(state)

        lengths, angles = self._unpack_lengths_angles(state)
        return f"{lengths}, {angles}"

    def readable2state(self, readable: str) -> List[float]:
        """
        Converts a human-readable representation of a state into the standard format.
        """
        for c in ["(", ")", " "]:
            readable = readable.replace(c, "")
        values = readable.split(",")
        values = [float(value) for value in values]

        return self.parameters2state(values)

    def is_valid(self, state: List) -> bool:
        """
        Determines whether a state is valid, according to the attributes of the
        environment.

        Parameters
        ----------
        state : list
            A state in environment format. If None, then lengths and angles will be
            used instead.

        Returns
        -------
        bool
            True if the state is valid according to the attributes of the environment;
            False otherwise.
        """
        lengths, angles = self._unpack_lengths_angles(state)
        # Check lengths
        if any([l < self.min_length or l > self.max_length for l in lengths]):
            return False
        if any([l < self.min_angle or l > self.max_angle for l in angles]):
            return False

        # If all checks are passed, return True
        return True
