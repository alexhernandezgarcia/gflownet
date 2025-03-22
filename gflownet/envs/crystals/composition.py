"""
Classes to represent material compositions (stoichiometry)
"""

import itertools
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pyxtal.symmetry import Group
from torch import Tensor
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import tfloat, tlong
from gflownet.utils.crystals.constants import ELEMENT_NAMES, OXIDATION_STATES
from gflownet.utils.crystals.pyxtal_cache import (
    get_space_group,
    space_group_check_compatible,
    space_group_lowest_free_wp_multiplicity,
    space_group_wyckoff_gcd,
)

N_ELEMENTS_ORACLE = 94


class Composition(GFlowNetEnv):
    """
    Composition environment for crystal materials.

    States are represented as a dictionary where the keys are the atomic numbers of the
    elements and the values the number of atoms per element.
    """

    def __init__(
        self,
        elements: Union[List, int] = 94,
        max_diff_elem: int = 5,
        min_diff_elem: int = 2,
        min_atoms: int = 2,
        max_atoms: int = 20,
        min_atom_i: int = 1,
        max_atom_i: int = 16,
        oxidation_states: Optional[Dict] = None,
        alphabet: Optional[Dict] = None,
        required_elements: Optional[Union[Tuple, List]] = (),
        space_group: Optional[int] = None,
        do_charge_check: bool = False,
        do_spacegroup_check: bool = True,
        **kwargs,
    ):
        """
        Args
        ----------
        elements : list or int
            Elements that will be used for construction of crystal. Either list, in
            which case every value should indicate the atomic number of an element, or
            int, in which case n consecutive atomic numbers will be used. Note that we
            assume this will correspond to real atomic numbers, i.e. start from 1, not
            0.

        max_diff_elem : int
            Maximum number of unique elements in the crystal

        min_diff_elem : int
            Minimum number of unique elements in the crystal

        min_atoms : int
            Minimum number of atoms that needs to be used to construct a crystal

        max_atoms : int
            Maximum number of atoms that can be used to construct a crystal

        min_atom_i : int
            Minimum number of elements of each kind that needs to be used to
            construct a crystal

        max_atom_i : int
            Maximum number of elements of each kind that can be used to construct a
            crystal

        oxidation_states : (optional) dict
            Mapping from ints (representing elements) to lists of different oxidation
            states

        alphabet : (optional) dict
            Mapping from ints (representing elements) to strings containing
            human-readable elements' names

        required_elements : (optional) list
            List of elements that must be present in a crystal for it to represent a
            valid end state

        space_group : (optional) int
            International number of a space group to be used for compatibility check,
            using pyxtal.symmetry.Group.check_compatible().

        do_charge_check : bool
            Whether to do neutral charge check and forbid compositions for which neutral
            charge is not possible.

        do_spacegroup_check : bool
            Whether to do a space group compatibility check and forbid compositions
            with incompatible Wyckoff positions with the given space group.
        """
        if isinstance(elements, int):
            elements = [i + 1 for i in range(elements)]
        if len(elements) != len(set(elements)):
            raise ValueError(
                f"Provided elements must be unique, detected {len(elements) - len(set(elements))} duplicates."
            )
        if any(e <= 0 for e in elements):
            raise ValueError(
                "Provided elements should be non-negative (assumed indexing from 1 for H)."
            )
        self.elements = sorted(elements)
        self.max_diff_elem = max_diff_elem
        self.min_diff_elem = min_diff_elem
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.min_atom_i = min_atom_i
        self.max_atom_i = max_atom_i
        self.oxidation_states = (
            oxidation_states
            if oxidation_states is not None
            else OXIDATION_STATES.copy()
        )
        self.alphabet = alphabet if alphabet is not None else ELEMENT_NAMES.copy()
        self.alphabet_rev = {v: k for k, v in self.alphabet.items()}
        self.required_elements = (
            required_elements if required_elements is not None else []
        )
        self.space_group = space_group
        self.do_charge_check = do_charge_check
        self.do_spacegroup_check = do_spacegroup_check
        self.elem2idx = {el: idx for idx, el in enumerate(self.elements)}
        # Source state: empty dict
        self.source = {}
        # End-of-sequence action
        self.eos = (-1, -1)
        super().__init__(**kwargs)

    def get_action_space(self):
        """
        Constructs list with all possible actions. An action is described by a
        tuple (element, n), indicating that the count of element will be
        set to n.
        """
        assert self.max_diff_elem >= self.min_diff_elem
        assert self.max_atom_i >= self.min_atom_i
        valid_word_len = np.arange(self.min_atom_i, self.max_atom_i + 1)
        actions = [(element, n) for element in self.elements for n in valid_word_len]
        actions.append(self.eos)
        return actions

    def get_max_traj_length(self):
        return min(self.max_diff_elem, self.max_atoms // self.min_atom_i)

    def _refine_compatibility_check(
        self, state, mask_required_element, mask_unrequired_element
    ):
        """
        Refines the masks (in-place) of required and unrequired elements by doing
        compatibility checks between the space group and the number of atoms.

        Args
        ----
        state : list
            The state on which the masks are to be applied.

        mask_required_element: list
            Element-wise mask indicating invalid actions for required elements. This
            masks indicates whether each individual actions is invalid or not for
            elements that are required to be in the composition by the end of the
            trajectory.

        mask_unrequired_element: list
            Element-wise mask indicating invalid actions for unrequired elements.
            This masks indicates whether each individual actions is invalid or not for
            elements that are not required to be in the composition by the end of the
            trajectory.
        """
        space_group = get_space_group(self.space_group)
        n_atoms = list(state.values())

        # Get the greated common divisor of the group's wyckoff position.
        # It cannot be valid to add a number of atoms that is not a
        # multiple of this value
        wyckoff_gcd = space_group_wyckoff_gcd(self.space_group)

        # Get the multiplicity of the group's most specific wyckoff position with
        # at least one degree of freedom
        free_multiplicity = space_group_lowest_free_wp_multiplicity(self.space_group)

        # Go through each action in the masks, validating them
        # individually
        for action_idx, nb_atoms_action in enumerate(
            range(self.min_atom_i, self.max_atom_i + 1)
        ):
            if (
                not mask_required_element[action_idx]
                or not mask_unrequired_element[action_idx]
            ):
                # If the number of atoms added by this action is not a
                # multiple of the greatest common divisor of the wyckoff
                # positions' multiplicities, mark action as invalid
                if nb_atoms_action % wyckoff_gcd != 0:
                    mask_required_element[action_idx] = True
                    mask_unrequired_element[action_idx] = True
                    continue

                # If the number of atoms added by this action is a
                # multiple of a non-specific wyckoff position, nothing
                # prevents it from being valid
                if nb_atoms_action % free_multiplicity == 0:
                    continue

                # Checking validity by induction. If a composition is
                # valid, adding a number of atoms is equal to the
                # multiplicity of a non-specific position, then this
                # action must also be valid.
                if nb_atoms_action > free_multiplicity and (
                    not mask_required_element[action_idx - free_multiplicity]
                    or not mask_unrequired_element[action_idx - free_multiplicity]
                ):
                    continue

                # If the composition resulting from this action is
                # incompatible with the space group, mark action as
                # invalid
                n_atoms_post_action = n_atoms + [nb_atoms_action]
                sg_compatible = space_group_check_compatible(
                    self.space_group, n_atoms_post_action
                )
                if not sg_compatible:
                    mask_required_element[action_idx] = True
                    mask_unrequired_element[action_idx] = True

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
            return [True for _ in range(self.action_space_dim)]

        mask = [False] * self.action_space_dim
        used_elements = list(state.keys())
        unused_required_elements = [
            e for e in self.required_elements if e not in used_elements
        ]
        n_used_elements = len(used_elements)
        n_unused_required_elements = len(unused_required_elements)
        n_used_atoms = sum(state.values())

        if self.do_spacegroup_check and isinstance(self.space_group, int):
            space_group = get_space_group(self.space_group)

            # Determine, based on the space group's Wyckoff's positions, what
            # is the min/max number of atoms of a given element that could be
            # added.
            most_specific_wp = space_group.get_wyckoff_position(-1)
            min_atom_i = most_specific_wp.multiplicity

            wyckoff_gcd = space_group_wyckoff_gcd(self.space_group)
            max_atom_i = (self.max_atom_i // wyckoff_gcd) * wyckoff_gcd

            # Determine if the current composition is compatible with the
            # space group
            n_atoms = list(state.values())
            sg_compatible = space_group_check_compatible(self.space_group, n_atoms)
        else:
            # Don't impose additional constraints on the min/max number of
            # atoms per element
            min_atom_i = self.min_atom_i
            max_atom_i = self.max_atom_i

            # Assume the current composition is compatible with the space group
            sg_compatible = True

        # Compute the min and max number of atoms to add to satisfy constraints
        nb_atoms_still_needed = max(0, self.min_atoms - n_used_atoms)
        nb_atoms_still_allowed = self.max_atoms - n_used_atoms

        # Compute the min and max number of elements to add to satisfy constraints
        nb_elems_still_needed = max(
            n_unused_required_elements, self.min_diff_elem - n_used_elements
        )
        nb_elems_still_allowed = self.max_diff_elem - n_used_elements

        # How many elements, other than the required elements, can still be added
        n_max_unrequired_elements_left = self.max_diff_elem - (
            n_used_elements + n_unused_required_elements
        )

        # What is the minimum number of atoms needed for a new required element in
        # order to reach the number of required atoms before we can't add new elements
        # anymore
        min_atoms_per_required_element = max(
            nb_atoms_still_needed - (nb_elems_still_allowed - 1) * max_atom_i,
            min_atom_i,
        )

        # What is the maximum number of atoms allowed for a new required element in
        # order to be able to reach the number of required elements before we can't add
        # new atoms anymore
        max_atoms_per_required_element = min(
            nb_atoms_still_allowed - (nb_elems_still_needed - 1) * min_atom_i,
            max_atom_i,
        )

        # Determine if there is a need to add unrequired elements to either reach the
        # number of required distinct elements or the number of required atoms
        unrequired_element_needed = (
            nb_elems_still_needed > n_unused_required_elements
            or max_atoms_per_required_element * n_unused_required_elements
            < nb_atoms_still_needed
        )

        # Determine if it is possible to add unrequired elements without going over the
        # maximum number of elements or atoms
        unrequired_element_allowed = (
            n_max_unrequired_elements_left > 0
            and min_atoms_per_required_element * n_unused_required_elements + min_atom_i
            <= nb_atoms_still_allowed
        )

        # Compute the minimum and maximum number of atoms available for an unrequired
        # element
        if unrequired_element_needed:
            # Some unrequired elements are needed so they are treated the same as the
            # required elements
            min_atoms_per_unrequired_element = min_atoms_per_required_element
            max_atoms_per_unrequired_element = max_atoms_per_required_element
        elif unrequired_element_allowed:
            # Unrequired elements are optional so there is no minium amount to add for
            # them and the maximum is only as high as possible without preventing the
            # addition of the required elements later
            min_atoms_per_unrequired_element = min_atom_i
            max_atoms_per_unrequired_element = min(
                nb_atoms_still_allowed
                - min_atoms_per_required_element * n_unused_required_elements,
                max_atom_i,
            )
        else:
            # No unrequired elements can be added
            min_atoms_per_unrequired_element = 0
            max_atoms_per_unrequired_element = 0

        if n_used_atoms < self.min_atoms:
            mask[-1] = True
        if n_used_elements < self.min_diff_elem:
            mask[-1] = True
        if any(r not in used_elements for r in self.required_elements):
            mask[-1] = True
        if not sg_compatible:
            # The current composition is incompatible with the space group,
            # we must allow EOS to end the trajectory.
            mask[-1] = False

        # Obtain action mask for each category of element
        def get_element_mask(min_atoms, max_atoms):
            return [
                bool(i < min_atoms or i > max_atoms)
                for i in range(self.min_atom_i, self.max_atom_i + 1)
            ]

        mask_required_element = get_element_mask(
            min_atoms_per_required_element, max_atoms_per_required_element
        )
        mask_unrequired_element = get_element_mask(
            min_atoms_per_unrequired_element, max_atoms_per_unrequired_element
        )

        # If required, refine the masks by doing compatibility checks between
        # the space group and the number of atoms
        if self.do_spacegroup_check and isinstance(self.space_group, int):
            self._refine_compatibility_check(
                state, mask_required_element, mask_unrequired_element
            )

        # Set action mask for each element
        nb_actions_per_element = self.max_atom_i - self.min_atom_i + 1
        for element_idx, element in enumerate(self.elements):
            # Compute the start and end indices of the actions associated with this
            # element
            action_start_idx = element_idx * nb_actions_per_element
            action_end_idx = action_start_idx + nb_actions_per_element
            # Set the mask for the actions associated with this element
            if element in state:
                # This element has already been added, we cannot add more
                mask[action_start_idx:action_end_idx] = [True] * nb_actions_per_element
            elif element in unused_required_elements:
                mask[action_start_idx:action_end_idx] = mask_required_element
            else:
                mask[action_start_idx:action_end_idx] = mask_unrequired_element

        # If no other action is valid, ensure that the EOS action is available
        if all(mask):
            mask[-1] = False

        return mask

    def states2proxy(
        self, states: List[dict]
    ) -> TensorType["batch", "state_proxy_dim"]:
        """
        Prepares a batch of states in "environment format" for the proxy: The output is
        a tensor of dtype long with N_ELEMENTS_ORACLE + 1 columns, where the positions
        of self.elements are filled with the number of atoms of each element in the
        state.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            single tensor.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states_proxy = np.zeros((len(states), N_ELEMENTS_ORACLE + 1))
        for idx, state in enumerate(states):
            if len(state) == 0:
                continue
            states_proxy[idx, list(state.keys())] = list(state.values())
        return tlong(states_proxy, device=self.device)

    def states2policy(
        self, states: List[dict]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: in
        order to not waste memory and for backward compatibility, the policy state only
        contains the number of atoms of the allowed elements.

        For example, if self.elements is [1, 2, 3, 4]:
            states: [{2: 1, 4: 2}, {1: 3}]
            states2policy(states): tensor([[0, 1, 0, 2], [3, 0, 0, 0]])

        Args
        ----
        states : list
            A batch of states in environment format, that is a list of dictionaries.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states_policy = np.zeros((len(states), len(self.elements)))
        for idx, state in enumerate(states):
            if len(state) == 0:
                continue
            indices_elements = [self.elem2idx[el] for el in state]
            states_policy[idx, indices_elements] = list(state.values())
        return tfloat(states_policy, device=self.device, float_type=self.float)

    def state2readable(self, state=None):
        """
        Transforms the state, represented as a dictionary of element: n_atoms key-value
        pairs, into a human-readable version: a non-reduced formula, following the Hill
        system.

        See: https://en.wikipedia.org/wiki/Chemical_formula#Hill_system

        Example:
            state: {1: 2, 3: 1}
                1: atomic number of H
                3: atomic number of Li
            output: H2Li1
        """
        if state is None:
            state = self.state
        state_elements = {self.alphabet[el]: n for el, n in state.items()}
        formula = ""
        if "C" in state_elements:
            formula += "C" + str(state_elements["C"])
        if "H" in state_elements:
            formula += "H" + str(state_elements["H"])
        formula += "".join(
            [
                el + str(n)
                for el, n in sorted(state_elements.items())
                if el not in ["C", "H"]
            ]
        )
        return formula

    def readable2state(self, readable):
        """
        Converts the readable representation of a state (a chemical formula) into the
        environment format.

        Example:
            readable: H2Li1
            self.alphabet: {1: "H", 2: "He", 3: "Li", 4: "Be"}
            output: {1: 2, 3: 1}
        """
        state = {}
        offset = 0
        for match in re.finditer(r"\d+", readable):
            span = match.span(0)
            element = readable[offset : span[0]]
            n_atoms = int(readable[span[0] : span[1]])
            state[self.alphabet_rev[element]] = n_atoms
            offset = span[1]
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

    def get_parents(self, state=None, done=None, action=None):
        """
        Determines all parents and actions that lead to a state.

        Args
        ----
        state : dict
            Representation of a state as a dictionary of element: n_atoms key-value
            pairs. Elements whose atomic number is not a key of the dictionary have
            zero atoms.

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
                element, n = action
                if element in state and state[element] == n:
                    parent = state.copy()
                    del parent[element]
                    parents.append(parent)
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
            The state after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        # If done, return invalid
        if self.done:
            return self.state, action, False
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
        # If action is not eos, then perform action
        if action != self.eos:
            element, num = action
            self.state[element] = num
            self.n_actions += 1
            return self.state, action, True
        # If action is eos, then attempt eos action
        else:
            if self.get_mask_invalid_actions_forward()[-1]:
                valid = False
            else:
                if self.do_charge_check:
                    # Currently enabling it causes errors when training combined
                    # Crystal env, and very significantly increases training time.
                    if self._can_produce_neutral_charge():
                        self.done = True
                        valid = True
                        self.n_actions += 1
                    else:
                        valid = False
                else:
                    self.done = True
                    valid = True
                    self.n_actions += 1
            return self.state, self.eos, valid

    def _can_produce_neutral_charge(self, state: Optional[List[int]] = None) -> bool:
        """
        Helper that checks whether there is a configuration of oxidation states that
        can produce a neutral charge for the given state.
        """
        if state is None:
            state = self.state

        nums_charges = [
            (num, self.oxidation_states[element]) for element, num in state.items()
        ]

        # Process all atoms one by one, gradually accumulating a set of all possible
        # charge totals so far
        poss_charge_sum = set([0])
        while len(nums_charges) > 0:
            num, charges = nums_charges[0]

            # Compute all possible charge totals that can be obtained by combining
            # all the previous charge totals with all the possible charges for the
            # current atom
            new_poss_charge_sum = set()
            for old_charge_sum in poss_charge_sum:
                for element_charge in charges:
                    new_poss_charge_sum.add(old_charge_sum + element_charge)
            poss_charge_sum = new_poss_charge_sum

            # Remove the atom that was processed from nums_charges
            if num == 1:
                # Remove element from nums_charges
                del nums_charges[0]
            else:
                # Remove one atom from this element
                nums_charges[0] = (num - 1, charges)

        return 0 in poss_charge_sum

    def is_valid(self, state: dict) -> bool:
        """
        Determines whether a state is valid, according to the attributes of the
        environment.

        Parameters
        ----------
        state : dict
            A state in environment format.

        Returns
        -------
        bool
            True if the state is valid according to the attributes of the environment;
            False otherwise.
        """
        n_atoms_per_element = state.values()
        # Check total number of atoms
        n_atoms = sum(n_atoms_per_element)
        if n_atoms < self.min_atoms:
            return False
        if n_atoms > self.max_atoms:
            return False
        # Check number element
        if any([n < self.min_atom_i for n in n_atoms_per_element]):
            return False
        if any([n > self.max_atom_i for n in n_atoms_per_element]):
            return False
        # Check required elements
        used_elements = set(state.keys())
        if len(used_elements - set(self.elements)) > 0:
            return False
        if len(used_elements) < self.min_diff_elem:
            return False
        if len(used_elements) > self.max_diff_elem:
            return False
        if len(set(self.required_elements) - used_elements) > 0:
            return False

        # If all checks are passed, return True
        return True
