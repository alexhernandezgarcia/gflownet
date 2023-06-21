"""
Classes to represent crystal environments
"""
import itertools
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.crystals.constants import (
    CRYSTAL_CLASSES,
    CRYSTAL_SYSTEMS,
    POINT_SYMMETRIES,
    SPACE_GROUPS,
)


class SpaceGroup(GFlowNetEnv):
    """
    SpaceGroup environment for ionic conductivity.

    The state space is the combination of three properties:
    1. The crystal system
        See: https://en.wikipedia.org/wiki/Crystal_system#Crystal_system
        (7 options + none)
    2. The point symmetry
        See: https://en.wikipedia.org/wiki/Crystal_system#Crystal_classes
        (5 options + none)
    3. The space group
        See: https://en.wikipedia.org/wiki/Space_group#Table_of_space_groups_in_3_dimensions
        (230 options + none)

    The action space is the choice of property to update and the index within the
    property (e.g. crystal system 2, point symmetry 4, space group 69, etc.). The
    selection of crystal system restricts the possible point symmetries and space
    groups; the selection of point symmetry restricts the possible crystal systems and
    space groups.  The selection of space groups determines a specific crystal system
    and space group. There is no restriction in the order of selection of properties.
    """

    def __init__(self, **kwargs):
        self.crystal_systems = CRYSTAL_SYSTEMS
        self.crystal_classes = CRYSTAL_CLASSES
        self.point_symmetries = POINT_SYMMETRIES
        self.space_groups = SPACE_GROUPS
        self.n_crystal_systems = len(self.crystal_systems)
        self.n_crystal_classes = len(self.crystal_classes)
        self.n_point_symmetries = len(self.point_symmetries)
        self.n_space_groups = 230
        self.cs_idx, self.ps_idx, self.sg_idx = 0, 1, 2
        self.eos = (-1, -1)
        # Source state: index 0 (empty) for all three properties (crystal system index,
        # point symmetry index, space group)
        self.source = [0 for _ in range(3)]
        # Base class init
        super().__init__(**kwargs)

    def get_action_space(self):
        """
        Constructs list with all possible actions. An action is described by a
        tuple (property, index), where property is (0: crystal system,
        1: point symmetry, 2: space group).
        """
        actions = []
        for prop, n_idx in zip(
            [self.cs_idx, self.ps_idx, self.sg_idx],
            [self.n_crystal_systems, self.n_point_symmetries, self.n_space_groups],
        ):
            actions_prop = [(prop, idx + 1) for idx in range(n_idx)]
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
        # If space group has been selected, only valid action is EOS
        if state[self.sg_idx] != 0:
            mask = [True for _ in self.action_space]
            mask[-1] = False
            return mask
        # No constraints if neither crystal system nor point symmetry selected
        if state[self.cs_idx] == 0 and state[self.ps_idx] == 0:
            crystal_systems = [
                (self.cs_idx, idx + 1) for idx in range(self.n_crystal_systems)
            ]
            point_symmetries = [
                (self.ps_idx, idx + 1) for idx in range(self.n_point_symmetries)
            ]
        # Constraints after having selected crystal system
        if state[self.cs_idx] != 0:
            crystal_systems = []
            crystal_classes_cs = self.crystal_systems[state[self.cs_idx]][1]
            # If no point symmetry selected yet
            if state[self.ps_idx] == 0:
                point_symmetries = [
                    (self.ps_idx, idx)
                    for idx in self.crystal_systems[state[self.cs_idx]][2]
                ]
        else:
            crystal_classes_cs = [idx + 1 for idx in range(self.n_crystal_classes)]
        # Constraints after having selected point symmetry
        if state[self.ps_idx] != 0:
            point_symmetries = []
            crystal_classes_ps = self.point_symmetries[state[self.ps_idx]][2]
            # If no class system selected yet
            if state[self.cs_idx] == 0:
                crystal_systems = [
                    (self.cs_idx, idx)
                    for idx in self.point_symmetries[state[self.ps_idx]][1]
                ]
        else:
            crystal_classes_ps = [idx + 1 for idx in range(self.n_crystal_classes)]
        # Merge crystal classes constraints and determine valid space group actions
        crystal_classes = list(
            set(crystal_classes_cs).intersection(set(crystal_classes_ps))
        )
        space_groups_list = [self.crystal_classes[cc][-1] for cc in crystal_classes]
        space_groups = [
            (self.sg_idx, sg) for sglist in space_groups_list for sg in sglist
        ]
        # Construct mask
        actions_valid = list(
            set.union(set(crystal_systems), set(point_symmetries), set(space_groups))
        )
        assert len(actions_valid) > 0
        mask = [
            False if action in actions_valid else True for action in self.action_space
        ]
        return mask

    def state2oracle(self, state: List = None) -> Tensor:
        """
        Prepares a list of states in "GFlowNet format" for the oracle. The input to the
        oracle is simply the space group.

        Args
        ----
        state : list
            A state

        Returns
        ----
        oracle_state : Tensor
        """
        if state is None:
            state = self.state
        if state[self.sg_idx] == 0:
            raise ValueError(
                "The space group must have been set in order to call the oracle"
            )
        return torch.Tensor(state[self.sg_idx], device=self.device, dtype=self.float)

    def statebatch2oracle(
        self, states: List[List]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in "GFlowNet format" for the oracle. The input to the
        oracle is simply the space group.

        Args
        ----
        state : list
            A state

        Returns
        ----
        oracle_state : Tensor
        """
        return self.statetorch2oracle(
            torch.Tensor(states, device=self.device, dtype=self.float)
        )

    def statetorch2oracle(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_oracle_dim"]:
        """
        Prepares a batch of states in "GFlowNet format" for the oracle. The input to the
        oracle is simply the space group.

        Args
        ----
        state : list
            A state

        Returns
        ----
        oracle_state : Tensor
        """
        return torch.unsqueeze(states[:, self.sg_idx])

    def state2readable(self, state=None):
        """
        Transforms the state, represented as a list of property indices, into a
        human-readable string with the format:

        <space group> | <crystal system> (<crystal system idx>) |
        <crystal class> (<crystal class idx>) | <point group> |
        <point symmetry> (<point symmetry idx>)

        Example:
            space group: 69
            crystal system: orthorhombic (3)
            crystal class: rhombic-dipyramidal (8)
            point group: mmm
            point symmetry: centrosymmetric (2)
            output:
                69 | orthorhombic (3) | rhombic-dipyramidal (8) | mmm |
                centrosymmetric (2)
        """
        if state is None:
            state = self.state
        crystal_system_idx = state[self.cs_idx]
        if crystal_system_idx != 0:
            crystal_system = self.crystal_systems[crystal_system_idx][0]
        else:
            crystal_system = "None"
        point_symmetry_idx = state[self.ps_idx]
        if point_symmetry_idx != 0:
            point_symmetry = self.point_symmetries[point_symmetry_idx][0]
        else:
            point_symmetry = "None"
        space_group = state[self.sg_idx]
        if space_group != 0:
            point_group = self.space_groups[space_group][0]
            crystal_class_idx = self.space_groups[space_group][1]
            crystal_class = self.crystal_classes[crystal_class_idx][0]
        else:
            # TODO: Technically the point group and crystal class could be determined
            # from crystal system + point symmetry
            point_group = "TBD"
            crystal_class_idx = 0
            crystal_class = "TBD"
        readable = (
            f"{space_group} | {crystal_system} ({crystal_system_idx}) | "
            + f"{crystal_class} ({crystal_class_idx}) | {point_group} | "
            + f"{point_symmetry} ({point_symmetry_idx})"
        )
        return readable

    def readable2state(self, readable):
        """
        Converts a human-readable representation of a state into the standard format.
        See: state2readable
        """
        properties = readable.split(" | ")
        crystal_system = int(properties[1].split(" ")[-1].strip("(").strip(")"))
        point_symmetry = int(properties[4].split(" ")[-1].strip("(").strip(")"))
        space_group = int(properties[0])
        state = [crystal_system, point_symmetry, space_group]
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
                # Add parent: state before setting space group
                parent = state.copy()
                parent[self.sg_idx] = 0
                parents.append(parent)
                action = (self.sg_idx, state[self.sg_idx])
                actions.append(action)
                # Add parent: source
                parents.append(self.source)
                action = (self.sg_idx, state[self.sg_idx])
                actions.append(action)
                # Make space group zero in state to avoid wrong parents (crystal system
                # and point symmetry cannot be set after space group has been selected)
                state[self.sg_idx] = 0
            # Catch other parents
            for prop, idx in enumerate(state[: self.sg_idx]):
                if idx != 0:
                    parent = state.copy()
                    parent[prop] = 0
                    parents.append(parent)
                    action = (prop, idx)
                    actions.append(action)
        return parents, actions

    def step(
        self, action: Tuple[int, int], skip_mask_check: bool = False
    ) -> Tuple[List[int], Tuple[int, int], bool]:
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
        # Generic pre-step checks
        do_step, self.state, action, valid = self._pre_step(
            action, skip_mask_check or self.skip_mask_check
        )
        if not do_step:
            return self.state, action, valid
        valid = True
        self.n_actions += 1
        prop, idx = action
        # Action is not eos
        if action != self.eos:
            state_next = self.state[:]
            state_next[prop] = idx
            # Set crystal system and point symmetry if space group is set
            self.state = self._set_constrained_properties(state_next)
            return self.state, action, valid
        # Action is eos
        else:
            self.done = True
            return self.state, action, valid

    def get_max_traj_length(self):
        return 3

    def set_state(self, state: List, done: Optional[bool] = False):
        """
        Sets the state and done. If done is True but incompatible with state (space
        group is missing), then force done False and print warning.
        """
        if done == True and state[self.sg_idx] == 0:
            done = False
            warnings.warn(
                f"""
            Attempted to set state {self.state2readable(state)} with done = True, which
            is not compatible with the environment. Forcing done = False.
            """
            )
        return super().set_state(state, done)

    def _set_constrained_properties(self, state: List[int]) -> List[int]:
        if state[self.sg_idx] != 0:
            if state[self.cs_idx] == 0:
                state[self.cs_idx] = self.space_groups[state[self.sg_idx]][2]
            if state[self.ps_idx] == 0:
                state[self.ps_idx] = self.space_groups[state[self.sg_idx]][3]
        return state
