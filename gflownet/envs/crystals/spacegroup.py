"""
Classes to represent crystal environments
"""

import itertools
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from torch import Tensor
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import tlong
from gflownet.utils.crystals.pyxtal_cache import space_group_check_compatible

CRYSTAL_LATTICE_SYSTEMS = None
POINT_SYMMETRIES = None
SPACE_GROUPS = None


def _get_crystal_lattice_systems():
    global CRYSTAL_LATTICE_SYSTEMS
    if CRYSTAL_LATTICE_SYSTEMS is None:
        with open(Path(__file__).parent / "crystal_lattice_systems.yaml", "r") as f:
            CRYSTAL_LATTICE_SYSTEMS = yaml.safe_load(f)
    return CRYSTAL_LATTICE_SYSTEMS


def _get_point_symmetries():
    global POINT_SYMMETRIES
    if POINT_SYMMETRIES is None:
        with open(Path(__file__).parent / "point_symmetries.yaml", "r") as f:
            POINT_SYMMETRIES = yaml.safe_load(f)
    return POINT_SYMMETRIES


def _get_space_groups():
    global SPACE_GROUPS
    if SPACE_GROUPS is None:
        with open(Path(__file__).parent / "space_groups.yaml", "r") as f:
            SPACE_GROUPS = yaml.safe_load(f)
    return SPACE_GROUPS


class Prop(Enum):
    """
    Enumeration of the 3 properties of the SpaceGroup Environment:
        - Crystal lattice system
        - Point symmetry
        - Space group
    """

    CLS = 0
    PS = 1
    SG = 2


class SpaceGroup(GFlowNetEnv):
    """
    SpaceGroup environment for ionic conductivity.

    The state space is the combination of three properties:
    1. The crystal-lattice system: combination of crystal system and lattice system
        See: https://en.wikipedia.org/wiki/Crystal_system#Crystal_system
        See: https://en.wikipedia.org/wiki/Crystal_system#Lattice_system
        See: https://en.wikipedia.org/wiki/Hexagonal_crystal_family
        (8 options + none)
    2. The point symmetry
        See: https://en.wikipedia.org/wiki/Crystal_system#Crystal_classes
        (5 options + none)
    3. The space group
        See: https://en.wikipedia.org/wiki/Space_group#Table_of_space_groups_in_3_dimensions
        (230 options + none)

    The action space is the choice of property to update, the index within the property
    and the combination of properties (state type) already set in the originating state
    type (e.g. crystal-lattice system 2 from source, point symmetry 4 from
    crystal-lattice system, space group 69 from point symmetry, etc.). The state type
    is included in the action to differentiate actions that lead to same state from
    different states, as in GFlowNet the distribution is over states not over actions.
    The selection of crystal-lattice system restricts the possible point symmetries and
    space groups; the selection of point symmetry restricts the possible
    crystal-lattice systems and space groups. The selection of space groups determines
    a specific crystal-lattice system and point symmetry. There is no restriction in
    the order of selection of properties.
    """

    def __init__(
        self,
        space_groups_subset: Optional[Iterable] = None,
        n_atoms: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Args
        ----
        space_groups_subset : iterable
            A subset of space group (international) numbers to which to restrict the
            state space. If None (default), the entire set of 230 space groups is
            considered.

        n_atoms : list of int (optional)
            A list with the number of atoms per element, used to compute constraints on
            the space group. 0's are removed from the list. If None, composition/space
            group constraints are ignored.
        """
        # Get dictionaries
        self.crystal_lattice_systems = _get_crystal_lattice_systems()
        self.point_symmetries = _get_point_symmetries()
        self.space_groups = _get_space_groups()
        self._restrict_space_groups(space_groups_subset)
        # Set dictionary of compatibility with number of atoms
        self.set_n_atoms_compatibility_dict(n_atoms)
        # Indices in the state representation: crystal-lattice system (cls), point
        # symmetry (ps) and space group (sg)
        self.cls_idx, self.ps_idx, self.sg_idx = 0, 1, 2
        # Dictionary of all properties
        self.properties = {
            Prop.CLS: self.crystal_lattice_systems,
            Prop.PS: self.point_symmetries,
            Prop.SG: self.space_groups,
        }
        # Indices of state types (see self.get_state_type)
        self.state_type_indices = [0, 1, 2, 3]
        # End-of-sequence action
        self.eos = (-1, -1, -1)
        # Source state: index 0 (empty) for all three properties (crystal-lattice
        # system index, point symmetry index, space group)
        self.source = [0 for _ in range(3)]
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self):
        """
        Constructs list with all possible actions. An action is described by a tuple
        (property, index, state_from_type), where property is (0: crystal-lattice
        system, 1: point symmetry, 2: space group), index is the index of the property
        set by the action and state_from_type is the state type of the originating
        state (see self.state_type_indices).
        """
        actions = []
        for prop, indices in self.properties.items():
            for s_from_type in self.state_type_indices:
                if prop == Prop.CLS and s_from_type in [1, 3]:
                    continue
                if prop == Prop.PS and s_from_type in [2, 3]:
                    continue
                actions_prop = [(prop.value, idx, s_from_type) for idx in indices]
                actions += actions_prop
        actions += [self.eos]
        return actions

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
    ) -> List:
        """
        Returns a list of length the action space with values:
            - True if the forward action is invalid given the current state.
            - False otherwise.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in self.action_space]
        cls_idx, ps_idx, sg_idx = state
        # If space group has been selected, only valid action is EOS
        if sg_idx != 0:
            mask = [True for _ in self.action_space]
            mask[-1] = False
            return mask
        state_type = self.get_state_type(state)
        # If neither crystal-lattice system nor point symmetry selected, apply only
        # composition-compatibility constraints
        if cls_idx == 0 and ps_idx == 0:
            crystal_lattice_systems = [
                (self.cls_idx, idx, state_type)
                for idx in self.crystal_lattice_systems
                if self._is_compatible(cls_idx=idx)
            ]
            point_symmetries = [
                (self.ps_idx, idx, state_type)
                for idx in self.point_symmetries
                if self._is_compatible(ps_idx=idx)
            ]
        # Constraints after having selected crystal-lattice system
        if cls_idx != 0:
            crystal_lattice_systems = []
            space_groups_cls = [
                (self.sg_idx, sg, state_type)
                for sg in self.crystal_lattice_systems[cls_idx]["space_groups"]
                if self.n_atoms_compatibility_dict[sg]
            ]
            # If no point symmetry selected yet
            if ps_idx == 0:
                point_symmetries = [
                    (self.ps_idx, idx, state_type)
                    for idx in self.crystal_lattice_systems[cls_idx]["point_symmetries"]
                    if self._is_compatible(cls_idx=cls_idx, ps_idx=idx)
                ]
        else:
            space_groups_cls = [
                (self.sg_idx, idx, state_type)
                for idx in self.space_groups
                if self.n_atoms_compatibility_dict[idx]
            ]
        # Constraints after having selected point symmetry
        if ps_idx != 0:
            point_symmetries = []
            space_groups_ps = [
                (self.sg_idx, sg, state_type)
                for sg in self.point_symmetries[ps_idx]["space_groups"]
                if self.n_atoms_compatibility_dict[sg]
            ]
            # If no crystal-lattice system selected yet
            if cls_idx == 0:
                crystal_lattice_systems = [
                    (self.cls_idx, idx, state_type)
                    for idx in self.point_symmetries[ps_idx]["crystal_lattice_systems"]
                    if self._is_compatible(cls_idx=idx, ps_idx=ps_idx)
                ]
        else:
            space_groups_ps = [
                (self.sg_idx, idx, state_type)
                for idx in self.space_groups
                if self.n_atoms_compatibility_dict[idx]
            ]
        # Merge space_groups constraints and determine valid space group actions
        space_groups = list(set(space_groups_cls).intersection(set(space_groups_ps)))
        # Construct mask
        actions_valid = set.union(
            set(crystal_lattice_systems), set(point_symmetries), set(space_groups)
        )
        assert len(actions_valid) > 0
        mask = [
            False if action in actions_valid else True for action in self.action_space
        ]
        return mask

    def states2proxy(
        self, states: Union[List[List], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states in "environment format" for the proxy: the proxy
        format is simply the space group.

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
        return torch.unsqueeze(states[:, self.sg_idx], dim=1)

    def state2readable(self, state=None):
        """
        Transforms the state, represented as a list of property indices, into a
        human-readable string with the format:

        <space group idx> | <space group symbol> |
        <crystal-lattice system> (<crystal-lattice system idx>) |
        <point symmetry> (<point symmetry idx>)
        <crystal class> | <point group>

        Example:
            space group: 69
            space group symbol: Fmmm
            crystal-lattice system: orthorhombic (3)
            point symmetry: centrosymmetric (2)
            crystal class: rhombic-dipyramidal
            point group: mmm
            output:
                69 | Fmmm | orthorhombic (3) | centrosymmetric (2) |
                rhombic-dipyramidal | mmm |
        """
        if state is None:
            state = self.state
        cls_idx, ps_idx, sg_idx = state
        crystal_lattice_system = self.get_crystal_lattice_system(state)
        point_symmetry = self.get_point_symmetry(state)
        sg_symbol = self.get_space_group_symbol(state)
        crystal_class = self.get_crystal_class(state)
        point_group = self.get_point_group(state)
        readable = (
            f"{sg_idx} | {sg_symbol} | {crystal_lattice_system} ({cls_idx}) | "
            + f"{point_symmetry} ({ps_idx}) | {crystal_class} | {point_group}"
        )
        return readable

    def readable2state(self, readable):
        """
        Converts a human-readable representation of a state into the standard format.
        See: state2readable
        """
        properties = readable.split(" | ")
        space_group = int(properties[0])
        crystal_lattice_system = int(properties[2].split(" ")[-1].strip("(").strip(")"))
        point_symmetry = int(properties[3].split(" ")[-1].strip("(").strip(")"))
        state = [crystal_lattice_system, point_symmetry, space_group]
        return state

    def get_parents(self, state=None, done=None, action=None):
        """
        Determines all parents and actions that lead to a state.

        Args
        ----
        state : list

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
            # Catch cases where space group has been selected
            if state[self.sg_idx] != 0:
                sg = state[self.sg_idx]
                # Add parent: source
                parents.append(self.source)
                action = (self.sg_idx, sg, 0)
                actions.append(action)
                # Add parents: states before setting space group
                state_pre_sg = state.copy()
                state_pre_sg[self.sg_idx] = 0
                for prop in range(len(state_pre_sg)):
                    parent = state_pre_sg.copy()
                    parent[prop] = 0
                    parents.append(parent)
                    parent_type = self.get_state_type(parent)
                    action = (self.sg_idx, sg, parent_type)
                    actions.append(action)
            else:
                # Catch other parents
                for prop, idx in enumerate(state[: self.sg_idx]):
                    if idx != 0:
                        parent = state.copy()
                        parent[prop] = 0
                        parents.append(parent)
                        parent_type = self.get_state_type(parent)
                        action = (prop, idx, parent_type)
                        actions.append(action)
        return parents, actions

    def step(self, action: Tuple[int, int]) -> Tuple[List[int], Tuple[int, int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. See: get_action_space()

        Returns
        -------
        self.state : list
            The new state after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        # If action not found in action space raise an error
        if action not in self.action_space:
            raise ValueError(
                f"Tried to execute action {action} not present in action space."
            )
        else:
            action_idx = self.action_space.index(action)
        # If action is in invalid mask, exit immediately
        if self.get_mask_invalid_actions_forward()[action_idx]:
            return self.state, action, False
        valid = True
        self.n_actions += 1
        prop, idx, _ = action
        # Action is not eos
        if action != self.eos:
            state_next = self.state[:]
            state_next[prop] = idx
            # Set crystal-lattice system and point symmetry if space group is set
            self.state = self._set_constrained_properties(state_next)
            return self.state, action, valid
        # Action is eos
        else:
            self.done = True
            return self.state, action, valid

    def get_max_traj_length(self):
        return len(self.source) + 1

    def _set_constrained_properties(self, state: List[int]) -> List[int]:
        """
        Sets the missing properties in a state that can be determined from the existing
        properties in the input state.

        Parameters
        ----------
        state : list
            A state in environment format.

        Returns
        -------
        list
            The updated state.
        """
        cls_idx, ps_idx, sg_idx = state
        if sg_idx != 0:
            if sg_idx not in self.space_groups:
                return state
            if cls_idx == 0:
                state[self.cls_idx] = self.space_groups[state[self.sg_idx]][
                    "crystal_lattice_system_idx"
                ]
            if ps_idx == 0:
                state[self.ps_idx] = self.space_groups[state[self.sg_idx]][
                    "point_symmetry_idx"
                ]
        return state

    def get_crystal_system(self, state: List[int] = None) -> str:
        """
        Returns the name of the crystal system given a state.
        """
        if state is None:
            state = self.state
        state = self._set_constrained_properties(state)
        if state[self.cls_idx] != 0:
            return self.crystal_lattice_systems[state[self.cls_idx]]["crystal_system"]
        else:
            return "None"

    @property
    def crystal_system(self) -> str:
        return self.get_crystal_system(self.state)

    def get_lattice_system(self, state: List[int] = None) -> str:
        """
        Returns the name of the lattice system given a state.
        """
        if state is None:
            state = self.state
        state = self._set_constrained_properties(state)
        if state[self.cls_idx] != 0:
            return self.crystal_lattice_systems[state[self.cls_idx]]["lattice_system"]
        else:
            return "None"

    @property
    def lattice_system(self, state: List[int] = None) -> str:
        return self.get_lattice_system(self.state)

    def get_crystal_lattice_system(self, state: List[int] = None) -> str:
        """
        Returns the name of the crystal-lattice system given a state.
        """
        if state is None:
            state = self.state
        crystal_system = self.get_crystal_system(state)
        lattice_system = self.get_lattice_system(state)
        if crystal_system != lattice_system:
            return f"{crystal_system}-{lattice_system}"
        else:
            return crystal_system

    @property
    def crystal_lattice_system(self) -> str:
        return self.get_crystal_lattice_system(self.state)

    def get_point_symmetry(self, state: List[int] = None) -> str:
        """
        Returns the name of the point symmetry given a state.
        """
        if state is None:
            state = self.state
        state = self._set_constrained_properties(state)
        if state[self.ps_idx] != 0:
            return self.point_symmetries[state[self.ps_idx]]["point_symmetry"]
        else:
            return "None"

    @property
    def point_symmetry(self) -> str:
        return self.get_point_symmetry(self.state)

    def get_space_group_symbol(self, state: List[int] = None) -> str:
        """
        Returns the name of the space group symbol given a state.
        """
        if state is None:
            state = self.state
        if state[self.sg_idx] != 0:
            return self.space_groups[state[self.sg_idx]]["full_symbol"]
        else:
            return "None"

    @property
    def space_group_symbol(self) -> str:
        return self.get_space_group_symbol(self.state)

    def get_space_group(self, state: List[int] = None) -> int:
        """
        Returns the index of the space group symbol given a state.
        """
        if state is None:
            state = self.state
        if state[self.sg_idx] != 0:
            return state[self.sg_idx]
        else:
            return None

    @property
    def space_group(self) -> int:
        return self.get_space_group(self.state)

    # TODO: Technically the crystal class could be determined from crystal-lattice
    # system + point symmetry
    def get_crystal_class(self, state: List[int] = None) -> str:
        """
        Returns the name of the crystal_class given a state.
        """
        if state is None:
            state = self.state
        if state[self.sg_idx] != 0:
            return self.space_groups[state[self.sg_idx]]["crystal_class"]
        else:
            return "None"

    @property
    def crystal_class(self) -> str:
        return self.get_crystal_class(self.state)

    # TODO: Technically the point group could be determined from crystal-lattice system
    # + point symmetry
    def get_point_group(self, state: List[int] = None) -> str:
        """
        Returns the name of the point group given a state.
        """
        if state is None:
            state = self.state
        if state[self.sg_idx] != 0:
            return self.space_groups[state[self.sg_idx]]["point_group"]
        else:
            return "None"

    @property
    def point_group(self) -> str:
        return self.get_point_group(self.state)

    def get_state_type(self, state: List[int] = None) -> int:
        """
        Returns the index of the type of the state passed as an argument. The state
        type is one of the following (self.state_type_indices):
            0: both crystal-lattice system and point symmetry are unset (== 0)
            1: crystal-lattice system is set (!= 0); point symmetry is unset
            2: crystal-lattice system is unset; point symmetry is set
            3: both crystal-lattice system and point symmetry are set
        """
        if state is None:
            state = self.state
        return sum([int(s > 0) * f for s, f in zip(state, (1, 2))])

    def set_n_atoms_compatibility_dict(self, n_atoms: List):
        """
        Sets self.n_atoms_compatibility_dict by calling
        SpaceGroup.build_n_atoms_compatibility_dict(), which contains a dictionary of
        {space_group: is_compatible} indicating whether each space_group in
        space_groups is compatible with the stoichiometry defined by n_atoms.

        See: build_n_atoms_compatibility_dict()

        Args
        ----
        n_atoms : list of int
            A list of number of atoms for each element in a composition. 0s will be
            removed from the list since they do not count towards the compatibility
            with a space group.
        """
        # Get compatibility with stoichiometry
        self.n_atoms_compatibility_dict = SpaceGroup.build_n_atoms_compatibility_dict(
            n_atoms, self.space_groups.keys()
        )

    def _is_compatible(
        self, cls_idx: Optional[int] = None, ps_idx: Optional[int] = None
    ):
        """
        Returns True if there is exists at least one space group compatible with the
        atom composition (according to self.n_atoms_compatibility_dict), with the
        crystal-lattice system (if provided), and with the point symmetry (if provided).
        False otherwise.
        """
        # Get list of space groups compatible with the composition
        space_groups = [
            sg for sg in self.space_groups if self.n_atoms_compatibility_dict[sg]
        ]

        # Prune the list of space groups to those compatible with the provided crystal-
        # lattice system
        if cls_idx is not None:
            space_groups_cls = self.crystal_lattice_systems[cls_idx]["space_groups"]
            space_groups = list(set(space_groups).intersection(set(space_groups_cls)))

        # Prune the list of space groups to those compatible with the provided point
        # symmetry
        if ps_idx is not None:
            space_groups_ps = self.point_symmetries[ps_idx]["space_groups"]
            space_groups = list(set(space_groups).intersection(set(space_groups_ps)))

        return len(space_groups) > 0

    @staticmethod
    def build_n_atoms_compatibility_dict(
        n_atoms: List[int], space_groups: Iterable[int]
    ):
        """
        Obtains which space groups are compatible with the stoichiometry given as
        argument (n_atoms).

        It relies on a function which, internally, calls pyxtal's
        pyxtal.symmetry.Group.check_compatible(). Note that sometimes that pyxtal
        is known to return invalid results.

        Args
        ----
        n_atoms : list of int
            A list of number of atoms for each element in a stoichiometry. 0s will be
            removed from the list since they do not count towards the compatibility
            with a space group. If None, all space groups will be marked as compatible.

        space_groups : list of int
            A list of space group international numbers, in [1, 230]

        Returns
        -------
        A dictionary of {space_group: is_compatible} indicating whether each
        space_group in space_groups is compatible with the stoichiometry defined by
        n_atoms.
        """
        if n_atoms is None:
            return {sg: True for sg in space_groups}
        n_atoms = [n for n in n_atoms if n > 0]
        assert all([n > 0 for n in n_atoms])
        assert all([sg > 0 and sg <= 230 for sg in space_groups])
        return {sg: space_group_check_compatible(sg, n_atoms) for sg in space_groups}

    def _restrict_space_groups(self, sg_subset: Optional[Iterable] = None):
        """
        Updates the dictionaries:
            - self.space_groups
            - self.crystal_lattice_systems
            - self.point_symmetries
        by eliminating the space groups that are not in the subset sg_subset passed as
        an argument.
        """
        if sg_subset is None:
            return
        sg_subset = set(sg_subset)

        # Update self.space_groups
        self.space_groups = {
            k: v for (k, v) in self.space_groups.items() if k in sg_subset
        }

        # Update self.crystal_lattice_systems based on space groups
        self.crystal_lattice_systems = deepcopy(self.crystal_lattice_systems)
        cls_to_remove = []
        for cls in self.crystal_lattice_systems:
            cls_space_groups = sg_subset.intersection(
                set(self.crystal_lattice_systems[cls]["space_groups"])
            )
            if len(cls_space_groups) == 0:
                cls_to_remove.append(cls)
            else:
                self.crystal_lattice_systems[cls]["space_groups"] = list(
                    cls_space_groups
                )
        for cls in cls_to_remove:
            del self.crystal_lattice_systems[cls]

        # Update self.point_symmetries based on space groups
        self.point_symmetries = deepcopy(self.point_symmetries)
        ps_to_remove = []
        for ps in self.point_symmetries:
            ps_space_groups = sg_subset.intersection(
                set(self.point_symmetries[ps]["space_groups"])
            )
            if len(ps_space_groups) == 0:
                ps_to_remove.append(ps)
            else:
                self.point_symmetries[ps]["space_groups"] = list(ps_space_groups)
        for ps in ps_to_remove:
            del self.point_symmetries[ps]

        # Update point symmetries of remaining crystal lattice systems
        point_symmetries = set(self.point_symmetries)
        for cls in self.crystal_lattice_systems:
            cls_point_symmetries = point_symmetries.intersection(
                set(self.crystal_lattice_systems[cls]["point_symmetries"])
            )
            self.crystal_lattice_systems[cls]["point_symmetries"] = list(
                cls_point_symmetries
            )

        # Update crystal lattice systems of remaining point symmetries
        crystal_lattice_systems = set(self.crystal_lattice_systems)
        for ps in self.point_symmetries:
            ps_crystal_lattice_systems = crystal_lattice_systems.intersection(
                set(self.point_symmetries[ps]["crystal_lattice_systems"])
            )
            self.point_symmetries[ps]["crystal_lattice_systems"] = list(
                ps_crystal_lattice_systems
            )

    def get_all_terminating_states(
        self, apply_stoichiometry_constraints: Optional[bool] = True
    ) -> List[List]:
        all_x = []
        for sg in self.space_groups:
            if (
                apply_stoichiometry_constraints
                and self.n_atoms_compatibility_dict[sg] is False
            ):
                continue
            all_x.append(self._set_constrained_properties([0, 0, sg]))
        return all_x

    def is_valid(self, x: List) -> bool:
        """
        Determines whether a state is valid, according to the attributes of the
        environment.
        """
        if x[self.sg_idx] in self.space_groups:
            return True
        return False
