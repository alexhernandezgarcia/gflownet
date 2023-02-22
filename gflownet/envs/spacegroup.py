"""
Classes to represent crystal environments
"""
import itertools
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.crystals.constants import (
    CRYSTAL_SYSTEMS,
    CRYSTAL_CLASSES,
    POINT_SYMMETRIES,
    SPACE_GROUPS,
)


class SpaceGroup(GFlowNetEnv):
    """
    Crystal environment for ionic conductivity
    """

    def __init__(
        self,
        **kwargs,
    ):
        #         super().__init__(**kwargs)

        self.crystal_systems = CRYSTAL_SYSTEMS
        self.crystal_classes = CRYSTAL_CLASSES
        self.point_symmetries = POINT_SYMMETRIES
        self.space_groups = SPACE_GROUPS
        self.n_crystal_systems = len(self.crystal_systems)
        self.n_crystal_classes = len(self.crystal_classes)
        self.n_point_symmetries = len(self.point_symmetries)
        self.n_space_groups = 230
        # A state is a list of [crystal system index, point symmetry index, space group]
        self.cs_idx, self.ps_idx, self.sg_idx = 0, 1, 2
        self.source = [0 for _ in range(3)]
        self.eos = -1
        self.action_space = self.get_actions_space()
        self.reset()

    def get_actions_space(self):
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
        actions += [(self.eos,)]
        return actions

    def get_max_traj_len(self):
        return 3

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        """
        Returns a vector of length the action space + 1: True if forward action is
        invalid given the current state, False otherwise.
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
            mask[self.eos] = False
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
            if state[self.ps_idx] == 0:
                crystal_systems = [
                    (self.cs_idx, idx + 1)
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
        Prepares a list of states in "GFlowNet format" for the oracle

        Args
        ----
        state : list
            A state

        Returns
        ----
        oracle_state : Tensor
            Tensor containing # of Li atoms, total # of atoms, and fractions of individual elements
        """
        if state is None:
            state = self.state

        li_idx = self.elem2idx.get(3)

        if li_idx is None:
            raise ValueError(
                "state2oracle needs to return the number of Li atoms, but Li not present in allowed elements."
            )

        return torch.Tensor(
            [state[li_idx], sum(state)] + [x / sum(state) for x in state]
        )

    def state2readable(self, state=None):
        """
        Transforms the state, represented as a list of elements' counts, into a
        human-readable dict mapping elements' names to their corresponding counts.

        Example:
            state: [2, 0, 1, 0]
            self.alphabet: {1: "H", 2: "He", 3: "Li", 4: "Be"}
            output: {"H": 2, "Li": 1}
        """
        if state is None:
            state = self.state
        readable = {
            self.alphabet[self.idx2elem[i]]: s_i
            for i, s_i in enumerate(state)
            if s_i > 0
        }
        return readable

    def readable2state(self, readable):
        """
        Converts a human-readable representation of a state into the standard format.

        Example:
            readable: {"H": 2, "Li": 1} OR {"H": 2, "Li": 1, "He": 0, "Be": 0}
            self.alphabet: {1: "H", 2: "He", 3: "Li", 4: "Be"}
            output: [2, 0, 1, 0]
        """
        state = [0 for _ in self.elements]
        rev_alphabet = {v: k for k, v in self.alphabet.items()}
        for k, v in readable.items():
            state[self.elem2idx[rev_alphabet[k]]] = v
        return state

    def reset(self, env_id=None):
        """
        Resets the environment.
        """
        self.state = self.source.copy()
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

    def get_parents(self, state=None, done=None, actions=None):
        """
        Determines all parents and actions that lead to a state.

        Args
        ----
        state : list
            Representation of a state as a list of length equal to that of self.elements,
            where i-th value contains the count of atoms for i-th element, from 0 to
            self.max_atoms_i.

        done : bool
            Whether the trajectory is done. If None, done is taken from instance.

        actions : None
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
                element, n = action
                if state[self.elem2idx[element]] == n > 0:
                    parent = state.copy()
                    parent[self.elem2idx[element]] -= n
                    parents.append(parent)
                    actions.append(idx)
        return parents, actions

    def step(self, action: Tuple[int, int]) -> Tuple[List[int], Tuple[int, int], bool]:
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. See: get_actions_space()

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        # If only possible action is eos, then force eos
        if sum(self.state) == self.max_atoms:
            self.done = True
            self.n_actions += 1
            return self.state, (self.eos, 0), True
        # If action not found in action space raise an error
        action_idx = None
        for i, a in enumerate(self.action_space):
            if a == action:
                action_idx = i
                break
        if action_idx is None:
            raise ValueError(
                f"Tried to execute action {action} not present in action space."
            )
        # If action is in invalid mask, exit immediately
        if self.get_mask_invalid_actions()[action_idx]:
            return self.state, action, False
        # If action is not eos, then perform action
        if action[0] != self.eos:
            atomic_number, num = action
            idx = self.elem2idx[atomic_number]
            state_next = self.state[:]
            state_next[idx] = num
            if sum(state_next) > self.max_atoms:
                valid = False
            else:
                self.state = state_next
                valid = True
                self.n_actions += 1
            return self.state, action, valid
        # If action is eos, then perform eos
        else:
            if self.get_mask_invalid_actions()[self.eos]:
                valid = False
            else:
                if self._can_produce_neutral_charge():
                    self.done = True
                    valid = True
                    self.n_actions += 1
                else:
                    valid = False
            return self.state, (self.eos, 0), valid

    def _can_produce_neutral_charge(self, state: Optional[List[int]] = None) -> bool:
        """
        Helper that checks whether there is a configuration of oxidation states that
        can produce a neutral charge for the given state.
        """
        if state is None:
            state = self.state

        nums_charges = [
            (num, self.oxidation_states[self.idx2elem[i]])
            for i, num in enumerate(state)
            if num > 0
        ]
        sum_diff_elem = []

        for n, c in nums_charges:
            charge_sums = []
            for c_i in itertools.product(c, repeat=n):
                charge_sums.append(sum(c_i))
            sum_diff_elem.append(np.unique(charge_sums))

        poss_charge_sum = [
            sum(combo) == 0 for combo in itertools.product(*sum_diff_elem)
        ]

        return any(poss_charge_sum)
