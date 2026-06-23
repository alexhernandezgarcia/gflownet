"""
Sequence composite meta-environment.

The Sequence environment arranges a variable number of sub-environments (up to a
maximum) into an ordered list, where new sub-environments are inserted either at the
front or at the end of the sequence. It can be seen as a hybrid between the Stack (an
ordered, fixed list of sub-environments) and the SetFlex (an unordered, variable-length
collection):
    - Like the Stack, the elements are ordered and, once a sub-environment is inserted,
      its actions must be performed until its EOS before sequence-level (meta) actions
      become available again.
    - Like the SetFlex, the number and types of the elements are built dynamically
      during the trajectory, from a pre-defined pool of unique environments.

Example
-------
Given three unique environment types A, B and C, a sequence ``C, C, A, B`` can be built
as follows (the insertion order index of each element is shown in brackets)::

    insert A as first   -> A(0)
    insert B at end     -> A(0), B(1)
    insert C at front   -> C(2), A(0), B(1)
    insert C at front   -> C(3), C(2), A(0), B(1)

The dictionary state keeps the elements keyed by their insertion order (0, 1, 2, ...)
and records the spatial arrangement in ``_indices``. For the final sequence above,
``_indices == [3, 2, 0, 1]``.
"""

import ast
import uuid
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.composite.base import CompositeBase
from gflownet.utils.common import copy, tfloat, tlong

# Insert directions (used to encode the insert meta-actions).
_FIRST = 0
_FRONT = 1
_END = 2

# Values stored in state["_active"]: -1 if no sub-environment is active, 0 if the
# sub-environment at the front of the sequence is active, 1 if the one at the end is.
_ACTIVE_NONE = -1
_ACTIVE_FRONT = 0
_ACTIVE_END = 1


class Sequence(CompositeBase):
    """
    Base class to create new environments by arranging sub-environments into an ordered
    sequence, growing it from the front and/or the end.

    Two modes are supported, selected via the constructor (analogous to the SetFlex):
        - **Free-growth** (``subenvs`` is None and ``do_random_subenvs`` is False): the
          sequence is grown from an empty source by inserting any type from
          ``envs_unique``, up to ``max_sequence_length`` elements. The global EOS is
          valid at any sequence-level state with at least one element. The generate sequence
          can be of length 1 - ``max_sequence_length``.
        - **Fixed-bag** (``subenvs`` is given, or ``do_random_subenvs`` is True): a
          fixed multiset of element types (the "bag") must all be placed and completed.
          So the sequence generates how to organize all the provided elements in a sequence
          of length exactly = len(bag). Inserting a type is only valid while there remain
          unplaced elements of that type, and the global EOS is valid only once the bag has
          been fully placed.

    Meta-actions
    ------------
    With ``U`` unique environment types, the sequence-level (meta) actions are:
        - ``insert first`` for each type (used only when the sequence is empty),
        - ``insert front`` for each type,
        - ``insert end`` for each type,
        - the global EOS,
    for a total of ``3 * U + 1`` meta-actions. The dedicated ``insert first`` action
    removes the only source of ambiguity in the backward direction (a single-element
    sequence, where ``front`` and ``end`` would otherwise collapse to the same state).
    No toggle actions are needed (unlike the Set environments): an insertion is itself
    the unique, reversible action that activates a sub-environment, and a
    sub-environment's EOS is the unique action that returns control to the sequence.

    TODO: Different insertion orders that yield the same spatial sequence are currently
    treated as distinct states. Such equivalent states need to be merged (as in SetFlex).

    TODO: Add min_sequence_length variable
    """

    def __init__(
        self,
        envs_unique: Iterable[GFlowNetEnv] = None,
        subenvs: Iterable[GFlowNetEnv] = None,
        max_sequence_length: Optional[int] = None,
        front_only: bool = False,
        end_only: bool = False,
        do_random_subenvs: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        envs_unique : iterable
            The pool of unique environments (by type and action space) that can be part
            of the sequence. If None, it is inferred from ``subenvs``.
        subenvs : iterable
            An optional fixed multiset of sub-environments to be arranged (fixed-bag
            mode). If given, ``envs_unique`` may be inferred from it.
        max_sequence_length : int
            Maximum number of elements (sub-environments) in the sequence. Required in
            free-growth mode and in random fixed-bag mode. If None and ``subenvs`` is
            given, it defaults to ``len(subenvs)``.
        front_only : bool
            If True, elements can only be inserted at the front of the sequence.
        end_only : bool
            If True, elements can only be inserted at the end of the sequence.
        do_random_subenvs : bool
            If True, a random bag of element types is sampled at every reset (fixed-bag
            mode). Practical for testing.
        """
        if front_only and end_only:
            raise ValueError("front_only and end_only cannot both be True.")
        if envs_unique is None and subenvs is None:
            raise ValueError(
                "Both envs_unique and subenvs are None. At least one of the two "
                "variables must contain a set of environments."
            )

        self.front_only = front_only
        self.end_only = end_only
        self.do_random_subenvs = do_random_subenvs

        # Determine the unique environments
        if envs_unique is None:
            envs_unique = subenvs
        (
            self.envs_unique,
            self.envs_unique_keys,
            _,
        ) = self._get_unique_environments(envs_unique)
        self._n_unique_envs = len(self.envs_unique)

        # Determine the fixed bag (if any) as a sorted list of unique-type indices
        if subenvs is not None:
            self._fixed_bag = sorted(self._compute_unique_indices_of_subenvs(subenvs))
        else:
            self._fixed_bag = None
        # Whether the sequence is constrained by a (preset or random) bag of elements.
        # The "_bag" key is included in the states only in this case; in free-growth
        # mode it is omitted (avoiding None values, which the base equal() cannot
        # compare).
        self._is_fixed_bag = (self._fixed_bag is not None) or self.do_random_subenvs

        # Maximum sequence length / maximum number of elements
        if max_sequence_length is None:
            if self._fixed_bag is not None:
                max_sequence_length = len(self._fixed_bag)
            else:
                raise ValueError(
                    "max_sequence_length must be provided in free-growth mode or "
                    "random fixed-bag mode."
                )
        if self._fixed_bag is not None and len(self._fixed_bag) > max_sequence_length:
            raise ValueError(
                "The number of preset sub-environments exceeds max_sequence_length."
            )
        self.max_sequence_length = max_sequence_length
        self.max_elements = max_sequence_length

        # Number of meta-actions and dimensionality of the one-hot mask prefix
        self.n_meta_actions = 3 * self.n_unique_envs + 1
        self._prefix_dim = self.n_unique_envs

        # Action dimensionality: the longest sub-environment EOS plus 1 (for the prefix)
        self.action_dim = max([len(env.eos) for env in self.envs_unique]) + 1

        # The global EOS is a tuple of -1's
        self.eos = (-1,) * self.action_dim

        # Source state: empty sequence
        self.source = {
            "_active": -1,
            "_dones": [],
            "_envs_unique": [],
            "_indices": [],
        }
        if self._is_fixed_bag:
            self.source["_bag"] = (
                list(self._fixed_bag) if self._fixed_bag is not None else []
            )

        # The sequence starts empty (no sub-environments placed yet)
        self.subenvs = []

        # Policy distributions parameters
        kwargs["fixed_distr_params"] = [
            env.fixed_distr_params for env in self.envs_unique
        ]
        kwargs["random_distr_params"] = [
            env.random_distr_params for env in self.envs_unique
        ]
        # Base class init
        super().__init__(**kwargs)

        # The sequence is continuous if any unique environment is continuous
        self.continuous = any([env.continuous for env in self.envs_unique])

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _compute_unique_indices_of_subenvs(
        self, subenvs: Iterable[GFlowNetEnv]
    ) -> List[int]:
        """
        Returns the list of unique-environment indices corresponding to each
        sub-environment in ``subenvs``.
        """
        indices_unique = []
        for env in subenvs:
            try:
                indices_unique.append(
                    self.envs_unique_keys.index((type(env), tuple(env.action_space)))
                )
            except ValueError:
                raise ValueError(
                    "The list of subenvs contains a sub-environment that could not "
                    "be matched to one of the existing unique environments"
                )
        return indices_unique

    def _seq_length(self, state: Optional[Dict] = None) -> int:
        """Returns the number of elements currently placed in the sequence."""
        state = self._get_state(state)
        return len(state["_envs_unique"])

    def _get_active_key(self, state: Optional[Dict] = None) -> int:
        """
        Returns the dictionary key (insertion index) of the currently active
        sub-environment, or -1 if none is active.

        The active sub-environment, if any, is always the most recently inserted one,
        whose insertion index is ``length - 1``.
        """
        state = self._get_state(state)
        if state["_active"] == -1:
            return -1
        return self._seq_length(state) - 1

    def _get_substate(self, state: Dict, idx_subenv: Optional[int] = None):
        """
        Returns the substate corresponding to ``idx_subenv``. If ``idx_subenv`` is None,
        the active sub-environment's substate is returned.
        """
        if idx_subenv is None:
            idx_subenv = self._get_active_key(state)
        return super()._get_substate(state, idx_subenv)

    def _remaining_bag(self, state: Optional[Dict] = None) -> Optional[Counter]:
        """
        Returns a Counter with the number of yet-to-be-placed elements of each type, or
        None in free-growth mode (no constraint).
        """
        state = self._get_state(state)
        if state.get("_bag") is None:
            return None
        return Counter(state["_bag"]) - Counter(state["_envs_unique"])

    def _insert_id(self, direction: int, idx_unique: int) -> int:
        """Returns the meta-action insert id for a direction and unique type.
        To keep the trajectory length short, meta-actions encode both the insert
        position (first, front or end) and the sub-env to be inserted. If there are
        e.g. 2 sub-env types available, there are 7 meta-actions: 2x3 + EOS."""
        return direction * self.n_unique_envs + idx_unique

    def _reverse_insert_id(self, state: Dict) -> int:
        """
        Returns the insert id of the meta-action that, in the backward direction, undoes
        the insertion of the currently active (most recently inserted) sub-environment.
        """
        length = self._seq_length(state)
        idx_unique = state["_envs_unique"][length - 1]  # Most recent sub-env
        if length == 1:
            direction = _FIRST
        elif state["_active"] == _ACTIVE_FRONT:
            direction = _FRONT
        else:
            direction = _END
        return self._insert_id(direction, idx_unique)

    def _make_subenv_instance(self, idx_unique: int, key: int) -> GFlowNetEnv:
        """Creates a fresh instance of the unique environment ``idx_unique``."""
        return self._get_env_unique(idx_unique).copy().reset().set_id(key)

    # ------------------------------------------------------------------ #
    # Action space, masks and policy structure
    # ------------------------------------------------------------------ #

    def get_action_space(self) -> List[Tuple]:
        """
        Constructs the list with all possible actions.

        The action space consists of, in this order:
            - The meta-actions to insert a sub-environment: for each direction (first,
              front, end) and each unique type, encoded as ``(-1, insert_id, 0...)``.
            - The global EOS, encoded as ``(-1, ..., -1)``.
            - The concatenation of the actions of all unique environments, prefixed by
              the unique-type index.

        Meta-actions are in the first n_meta_actions spot, followed by sub-env actions ordered
        by their unique-env index.
        """
        action_space = []
        # Insert meta-actions
        action_space.extend(
            [
                self._pad_action((insert_id,), -1)
                for insert_id in range(3 * self.n_unique_envs)
            ]
        )
        # Global EOS
        action_space.append(self.eos)
        # Actions of each unique environment
        for idx in range(self.n_unique_envs):
            action_space.extend(
                [
                    self._pad_action(action, idx)
                    for action in self._get_env_unique(idx).action_space
                ]
            )
        return action_space

    def _compute_mask_dim(self) -> int:
        """
        Mask format: [ prefix | core | padding ]. With prefix = one-hot of the active
        unique-environment type (so e.g. if there are 3 different possible subenv types
        then the prefix is of size 3) or all False for if at meta/sequence level, core =
        either the meta-actions mask or the active sub-environment's mask and padding =
        FALSE padded up to mask_dim.
        """
        mask_dim_envs_unique = [env.mask_dim for env in self.envs_unique]
        return self._prefix_dim + max(mask_dim_envs_unique + [self.n_meta_actions])

    def _format_mask(self, mask: List[bool], idx_unique: int) -> List[bool]:
        """
        Mask format: [ prefix | core | padding ]. Formats a core mask into a full
        sequence mask: a one-hot prefix of the active unique-environment type
        (all False if ``idx_unique`` is -1), the core mask and False-padding up to
        ``self.mask_dim``.
        """
        prefix = [False] * self._prefix_dim  # Prefix
        if idx_unique != -1:
            prefix[idx_unique] = True
        mask = prefix + list(mask)  # Prefix + core
        return mask + [False] * (self.mask_dim - len(mask))  # Add False padding

    def _extract_core_mask(
        self,
        mask: Union[List, TensorType["batch_size", "mask_dim"]],
        idx_unique: int,
    ):
        """
        Extracts the core part of the mask (without prefix and padding). If
        ``idx_unique`` is -1, the meta-actions core is extracted.
        """
        if idx_unique == -1:
            mask_dim = self.n_meta_actions
        else:
            mask_dim = self._get_env_unique(idx_unique).mask_dim
        if isinstance(mask, list):
            return mask[self._prefix_dim : self._prefix_dim + mask_dim]
        return mask[:, self._prefix_dim : self._prefix_dim + mask_dim]

    def _meta_forward_mask(self, state: Dict, done: bool) -> List[bool]:
        """Builds the core meta-actions mask for a forward transition. Valid
        actions have a mask entry False to not mask them. Mask entry = True means
        that this action will be masked. The logic in the prefix and padding part
        of the mask are different, because there True/False indicates something
        different."""
        U = self.n_unique_envs
        core = [True] * self.n_meta_actions
        if done:
            return core
        length = self._seq_length(state)
        remaining = self._remaining_bag(state)
        if (
            remaining is None
        ):  # In the "normal" sequence case, always all sub-envs can be inserted
            allowed = list(range(U))
        else:  # In the bag sequence case only the remaining sub-envs can be inserted
            allowed = [t for t in range(U) if remaining.get(t, 0) > 0]
        # Inserts (only if there is room)
        if length < self.max_elements:
            if length == 0:
                for t in allowed:
                    core[self._insert_id(_FIRST, t)] = False
            else:
                for t in allowed:
                    if not self.end_only:
                        core[self._insert_id(_FRONT, t)] = False
                    if not self.front_only:
                        core[self._insert_id(_END, t)] = False
        # Global EOS
        if remaining is None:
            eos_ok = length >= 1
        else:
            eos_ok = length >= 1 and sum(remaining.values()) == 0
        if eos_ok:
            core[3 * U] = False  # Allow EOS which is at position n_meta_actions - 1
        return core

    def get_mask_invalid_actions_forward(
        self, state: Optional[Dict] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Forward mask. At a sequence-level state (``_active == -1``) the valid actions
        are the insert meta-actions and possibly the global EOS. While a sub-environment
        is active, the valid actions are those of the active sub-environment.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if state["_active"] == -1 or done:
            return self._format_mask(self._meta_forward_mask(state, done), -1)
        # A sub-environment is active
        key = self._get_active_key(state)
        idx_unique = state["_envs_unique"][key]
        subenv = self._get_env_unique(idx_unique)
        substate = self._get_substate(state, key)
        core = subenv.get_mask_invalid_actions_forward(substate, False)
        return self._format_mask(core, idx_unique)

    def get_mask_invalid_actions_backward(
        self, state: Optional[Dict] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Backward mask. Cases:
            - done: only the global EOS is valid.
            - source (empty): no actions are valid.
            - sequence-level (``_active == -1``) and non-empty: re-enter the most
              recently inserted sub-environment to undo its EOS (its EOS is the only
              valid backward action).
            - sub-environment active, at its source: the only valid action is the
              meta-action that undoes its insertion.
            - sub-environment active, not at its source: the backward actions of the
              active sub-environment.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        U = self.n_unique_envs

        if done:
            core = [True] * self.n_meta_actions
            core[3 * U] = False  # only EOS valid
            return self._format_mask(core, -1)  # -1 is idx_unique of meta actions

        length = self._seq_length(state)
        if state["_active"] == -1:
            if length == 0:
                # No backwards action possible for source state (=empty sequence)
                return self._format_mask([True] * self.n_meta_actions, -1)
            # Re-enter the most recently inserted sub-environment (undo its EOS)
            key = length - 1  # Most recently inserted subenv always has index len-1
            idx_unique = state["_envs_unique"][key]
            subenv = self._get_env_unique(idx_unique)
            substate = self._get_substate(state, key)
            core = subenv.get_mask_invalid_actions_backward(substate, True)
            return self._format_mask(core, idx_unique)

        # A sub-environment is active
        key = self._get_active_key(state)  # Get active subenv (always biggest index)
        idx_unique = state["_envs_unique"][key]  # Return subenv type of active subenv
        subenv = self._get_env_unique(
            idx_unique
        )  # Return a stateless object of type idx_unique
        substate = self._get_substate(
            state, key
        )  # Return acctual state of the subenv at position key
        if subenv.is_source(substate):
            # Only the reverse-insertion meta-action is valid
            core = [True] * self.n_meta_actions
            core[self._reverse_insert_id(state)] = False
            return self._format_mask(core, -1)
        core = subenv.get_mask_invalid_actions_backward(substate, False)
        return self._format_mask(core, idx_unique)

    def get_valid_actions(
        self,
        mask: Optional[bool] = None,
        state: Optional[Dict] = None,
        done: Optional[bool] = None,
        backward: Optional[bool] = False,
    ) -> List[Tuple]:
        """
        Returns the list of valid actions, by interpreting the (possibly provided) mask.
        """
        state = self._get_state(state)
        done = self._get_done(done)
        if mask is None:
            mask = self.get_mask(state, done, backward)

        if not any(mask[: self._prefix_dim]):
            # Meta-actions because prefix = all False
            core = self._extract_core_mask(mask, -1)
            return [
                action
                for action, m in zip(self.action_space[: self.n_meta_actions], core)
                if not m
            ]

        # Sub-environment actions
        idx_unique = list(mask[: self._prefix_dim]).index(
            True
        )  # Get unique subenv type
        subenv = self._get_env_unique(
            idx_unique
        )  # Get stateless object of this subenv type
        key = self._seq_length(state) - 1
        substate = self._get_substate(state, key)
        sub_done = bool(state["_dones"][key])  # 0 = False / not done, 1 = True / done
        core = self._extract_core_mask(mask, idx_unique)
        return [
            self._pad_action(action, idx_unique)
            for action in subenv.get_valid_actions(core, substate, sub_done, backward)
        ]

    def get_policy_output(self, params: list) -> TensorType["policy_output_dim"]:
        """
        Policy output structure: the concatenation of the unique-environment policy outputs
        and a block of ones for the meta-actions. For discrete actions the policy output
        size is equal to the number of actions (one logit per discrete action). For continous
        actions, the policy can not generate one logit per action (since there are inf many)
        but generates the values of parameters of the action distribution. This functions
        defines the shape and default values for the policy output."""
        policy_outputs_subenvs = super().get_policy_output(params)
        policy_outputs_meta = torch.ones(
            self.n_meta_actions, dtype=self.float, device=self.device
        )
        # Policy output is [ sub-env 0 block | ... | sub-env U-1 block | meta-action block ]
        return torch.cat((policy_outputs_subenvs, policy_outputs_meta))

    def _get_policy_outputs_of_meta_actions(
        self, policy_outputs: TensorType["n_states", "policy_output_dim"]
    ):
        """Returns the last ``n_meta_actions`` columns of the policy outputs."""
        return policy_outputs[:, -self.n_meta_actions :]

    # ------------------------------------------------------------------ #
    # Parents
    # ------------------------------------------------------------------ #

    def get_parents(
        self,
        state: Optional[Dict] = None,
        done: Optional[bool] = None,
        #action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """Determines all parents and the actions that lead to ``state``.
        Return is list of parents and action that led from parent to the
        current state."""

        state = self._get_state(state)
        done = self._get_done(done)
        length = self._seq_length(state)

        # Case 0: sequence is done
        if done:
            return [state], [self.eos]

        # Case 1: Meta-level (sequence) active
        elif state["_active"] == _ACTIVE_NONE:
            if length == 0:
                # Case 1a: Source state has no parent
                return [], []
            # Case 1b: Only forward action that ends in meta state is EOS of the most
            # recently inserted sub-environment
            key = length - 1 # most recently inserted sub-environment
            idx_unique = state["_envs_unique"][key]
            subenv = self._get_env_unique(idx_unique)
            parent = copy(state)
            parent["_active"] = (
                _ACTIVE_FRONT if parent["_indices"][0] == key else _ACTIVE_END
            )
            parent["_dones"][key] = 0 # Re-active most recently inserted sub-environment
            return [parent], [self._pad_action(subenv.eos, idx_unique)]

        # Case 2: A sub-environment is active
        elif state["_active"] in (_ACTIVE_FRONT, _ACTIVE_END):
            key = self._get_active_key(state)
            idx_unique = state["_envs_unique"][key]
            subenv = self._get_env_unique(idx_unique)
            substate = self._get_substate(state, key)

            # 2a: Subenv is at its source state, parent state is the state before this element was inserted.
            if subenv.is_source(substate):
                insert_id = self._reverse_insert_id(state)
                parent = copy(state)
                del parent[key]
                parent["_envs_unique"].pop()
                parent["_dones"].pop()
                parent["_indices"].remove(key)
                parent["_active"] = -1
                return [parent], [self._pad_action((insert_id,), -1)]

            # 2b: Parent states are from the active sub-environment, meta state remains unchanged
            parents_subenv, parent_actions = subenv.get_parents(substate, False)
            parents = []
            actions = []
            for parent_subenv, parent_action in zip(parents_subenv, parent_actions):
                parent = copy(state)
                parent = self._set_substate(key, parent_subenv, parent) # Modify full sequence state
                parents.append(parent)
                actions.append(self._pad_action(parent_action, idx_unique))
            return parents, actions
        
        # Other cases should not exist
        raise ValueError(f"For state: {state} the _active flag has an impossible value: {state['_active']}")

    # ------------------------------------------------------------------ #
    # Steps
    # ------------------------------------------------------------------ #

    def _meta_action_is_valid(self, action: Tuple, backward: bool) -> bool:
        """Checks a meta-action against the corresponding mask."""
        if self.skip_mask_check:
            return True
        action_idx = self._action2index[action]
        mask = (
            self.get_mask_invalid_actions_backward()
            if backward
            else self.get_mask_invalid_actions_forward()
        )
        return not self._extract_core_mask(mask, -1)[action_idx]

    def step(
        self, action: Tuple, skip_mask_check: bool = False
    ) -> Tuple[Dict, Tuple, bool]:
        """Executes a forward step given an action. Returns new state, action atempted
        and True/False if the action execution was succesful or not."""

        # Check
        if action[0] != -1 and not (0 <= action[0] < self.n_unique_envs):
            raise ValueError(f"Prefix of {action} is impossible")


        # Case 0: If env is done no action is possible
        if self.done:
            return self.state, action, False

        # Case 1: Meta-action (insert or global EOS)
        elif action[0] == -1:
            do_step, _, _ = self._pre_step(action, skip_mask_check=True)
            if do_step and not skip_mask_check:
                do_step = self._meta_action_is_valid(action, backward=False)
            if not do_step:
                return self.state, action, False
            self.n_actions += 1
            # 1a: Global EOS
            if action == self.eos:
                self.done = True
                return self.state, action, True
            # 1b: Insert sub-env
            direction, idx_unique = divmod(action[1], self.n_unique_envs)
            key = self._seq_length(self.state)
            new_subenv = self._make_subenv_instance(idx_unique, key)
            self.subenvs = list(self.subenvs) + [new_subenv]
            self.state["_envs_unique"].append(idx_unique)
            self.state["_dones"].append(0)
            self.state[key] = copy(new_subenv.source)
            if direction == _FIRST:
                self.state["_indices"] = [key]
                self.state["_active"] = _ACTIVE_FRONT # TODO: Is it an issue that _ACTIVE_FRONT is used also for _FIRST
            elif direction == _FRONT:
                self.state["_indices"] = [key] + self.state["_indices"]
                self.state["_active"] = _ACTIVE_FRONT
            else:
                self.state["_indices"] = self.state["_indices"] + [key]
                self.state["_active"] = _ACTIVE_END
            return self.state, action, True

        # Case 2: Sub-environment action
        else:
            key = self._get_active_key(self.state)
            idx_unique = action[0]
            subenv = self.subenvs[key]
            action_subenv = self._depad_action(action, idx_unique)
            action_to_check = subenv.action2representative(action_subenv)
            if subenv.continuous:
                skip_mask_check = True
            do_step, _, _ = subenv._pre_step(
                action_to_check, skip_mask_check=(skip_mask_check or self.skip_mask_check)
            )
            if not do_step:
                return self.state, action, False
            _, action_subenv, valid = subenv.step(action_subenv)
            if not valid:
                return self.state, action, False
            self.n_actions += 1
            if action_subenv == subenv.eos:
                self._set_subdone(key, True)
                self.state["_active"] = -1
            else:
                self._set_substate(key, subenv.state)
            return self.state, action, True

    def step_backwards(
        self, action: Tuple, skip_mask_check: bool = False
    ) -> Tuple[Dict, Tuple, bool]:
        """Executes a backward step given an action. Returns the new state, the
        backwards action attempted and if it was succesful."""

        if action[0] != -1 and not (0 <= action[0] < self.n_unique_envs):
            raise ValueError(f"Prefix of {action} is impossible")

        # Case 0: Global EOS (undo done)
        if action == self.eos:
            do_step, _, _ = self._pre_step(action, backward=True, skip_mask_check=True)
            if do_step and not skip_mask_check:
                do_step = self._meta_action_is_valid(action, backward=True)
            if not do_step:
                return self.state, action, False
            self.done = False
            self.n_actions += 1
            return self.state, action, True

        # Case 1: Meta-action (undo an insertion)
        elif action[0] == -1:
            do_step, _, _ = self._pre_step(action, backward=True, skip_mask_check=True)
            if do_step and not skip_mask_check:
                do_step = self._meta_action_is_valid(action, backward=True)
            if not do_step:
                return self.state, action, False            
            
            key = self._seq_length(self.state) - 1
            self.subenvs = list(self.subenvs)[:-1]
            del self.state[key]
            self.state["_envs_unique"].pop()
            self.state["_dones"].pop()
            self.state["_indices"].remove(key)
            self.state["_active"] = -1
            self.n_actions += 1
            return self.state, action, True

        # Case 2: Sub-environment action
        else:
            was_inactive = self.state["_active"] == -1  # Needed because there are two cases:
                                                        # Case 2a: Sub-env active and undoing a sub-env action
                                                        # Case 2b: At meta-level and action is to undo EOS of most recent sub-env
            key = self._seq_length(self.state) - 1  # most recently inserted element
            idx_unique = action[0]
            subenv = self.subenvs[key]
            action_subenv = self._depad_action(action, idx_unique)
            action_to_check = subenv.action2representative(action_subenv)
            if subenv.continuous:
                skip_mask_check = True
            do_step, _, _ = subenv._pre_step(
                action_to_check,
                backward=True,
                skip_mask_check=(skip_mask_check or self.skip_mask_check),
            )
            if not do_step:
                return self.state, action, False
            _, _, valid = subenv.step_backwards(action_subenv)
            if not valid:
                return self.state, action, False
            self.n_actions += 1
            self._set_substate(key, subenv.state)
            self._set_subdone(key, subenv.done)
            if was_inactive: # Case 2b: need to change _active flag
                # We re-entered the element to undo its EOS: activate it (front/end)
                self.state["_active"] = (
                    _ACTIVE_FRONT if self.state["_indices"][0] == key else _ACTIVE_END
                )
            return self.state, action, True

    # ------------------------------------------------------------------ #
    # Batched sampling and log-probabilities
    # ------------------------------------------------------------------ #

    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "policy_output_dim"]] = None,
        states_from: List = None,
        is_backward: Optional[bool] = False,
        random_action_prob: Optional[float] = 0.0,
        temperature_logits: Optional[float] = 1.0,
    ) -> List[Tuple]:
        """Samples a batch of actions from a batch of policy outputs."""
        is_active = torch.any(mask[:, : self._prefix_dim], axis=1)
        is_meta = torch.logical_not(is_active)

        actions_meta = []
        if torch.any(is_meta):
            actions_meta = super().sample_actions_batch(
                self._get_policy_outputs_of_meta_actions(policy_outputs[is_meta]),
                self._extract_core_mask(mask[is_meta], -1),
                None,
                is_backward,
                random_action_prob,
                temperature_logits,
            )

        indices_active = torch.where(mask[is_active, : self._prefix_dim])[1]
        if len(indices_active) == 0:
            return actions_meta

        indices_unique_int = indices_active.tolist()
        states_active = [s for s, a in zip(states_from, is_active) if a]
        states_dict = {idx: [] for idx in range(self.n_unique_envs)}
        for state, idx_unique in zip(states_active, indices_unique_int):
            key = self._seq_length(state) - 1
            states_dict[idx_unique].append(self._get_substate(state, key))
        indices_unique = tlong(indices_unique_int, device=self.device)

        actions_dict = {}
        for idx in range(self.n_unique_envs):
            env_mask = indices_unique == idx
            if not torch.any(env_mask):
                continue
            env = self._get_env_unique(idx)
            actions_dict[idx] = env.sample_actions_batch(
                self._get_policy_outputs_of_env_unique(
                    policy_outputs[is_active][env_mask], idx
                ),
                self._extract_core_mask(mask[is_active][env_mask], idx),
                states_dict[idx],
                is_backward,
                random_action_prob,
                temperature_logits,
            )
        actions_active = [
            self._pad_action(actions_dict[idx].pop(0), idx)
            for idx in indices_unique_int
        ]

        actions = []
        for a in is_active:
            if a:
                actions.append(actions_active.pop(0))
            else:
                actions.append(actions_meta.pop(0))
        return actions

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: Union[List, TensorType["n_states", "action_dim"]],
        mask: TensorType["n_states", "mask_dim"],
        states_from: List,
        is_backward: bool,
    ) -> TensorType["batch_size"]:
        """Computes log probabilities of actions given policy outputs and actions."""
        actions = tfloat(actions, float_type=self.float, device=self.device)
        n_states = policy_outputs.shape[0]
        logprobs = torch.empty(n_states, dtype=self.float, device=self.device)

        is_active = torch.any(mask[:, : self._prefix_dim], axis=1)
        is_meta = torch.logical_not(is_active)

        if torch.any(is_meta):
            logprobs[is_meta] = super().get_logprobs(
                self._get_policy_outputs_of_meta_actions(policy_outputs[is_meta]),
                actions[is_meta],
                self._extract_core_mask(mask[is_meta], -1),
                None,
                is_backward,
            )

        indices_active = torch.where(mask[is_active, : self._prefix_dim])[1]
        if len(indices_active) == 0:
            return logprobs

        indices_unique_int = indices_active.tolist()
        states_active = [s for s, a in zip(states_from, is_active) if a]
        states_dict = {idx: [] for idx in range(self.n_unique_envs)}
        for state, idx_unique in zip(states_active, indices_unique_int):
            key = self._seq_length(state) - 1
            states_dict[idx_unique].append(self._get_substate(state, key))
        indices_unique = tlong(indices_unique_int, device=self.device)

        logprobs_active = torch.empty(
            len(indices_active), dtype=self.float, device=self.device
        )
        for idx in range(self.n_unique_envs):
            env_mask = indices_unique == idx
            if not torch.any(env_mask):
                continue
            env = self._get_env_unique(idx)
            logprobs_active[env_mask] = env.get_logprobs(
                self._get_policy_outputs_of_env_unique(
                    policy_outputs[is_active][env_mask], idx
                ),
                self._depad_action_batch(actions[is_active][env_mask, :], idx),
                self._extract_core_mask(mask[is_active][env_mask], idx),
                states_dict[idx],
                is_backward,
            )
        logprobs[is_active] = logprobs_active
        return logprobs

    # ------------------------------------------------------------------ #
    # State <-> representations
    # ------------------------------------------------------------------ #

    def states2policy(
        self, states: List[Dict]
    ) -> TensorType["batch", "state_policy_dim"]:
        """
        Policy representation. Concatenation of:
            - one-hot of the active flag (over {-1, 0, 1}),
            - remaining bag counts per unique type (zeros in free-growth mode),
            - done flag per position (padded to max_elements),
            - one-hot of the unique type at each position (padded to max_elements),
            - per-unique-environment substates, padded with sources up to max_elements.
        """
        n_states = len(states)
        U = self.n_unique_envs
        M = self.max_elements

        substates = torch.tile(
            torch.cat(
                [
                    subenv.state2policy(subenv.source).tile((M,))
                    for subenv in self.envs_unique
                ],
                dim=0,
            ),
            (n_states, 1),
        )
        device = substates.device

        active = torch.zeros((n_states, 3), dtype=self.float, device=device)
        remaining = torch.zeros((n_states, U), dtype=self.float, device=device)
        dones = torch.zeros((n_states, M), dtype=self.float, device=device)
        pos_type = torch.zeros((n_states, M * U), dtype=self.float, device=device)

        for i, state in enumerate(states):
            active[i, state["_active"] + 1] = 1.0
            rem = self._remaining_bag(state)
            if rem is not None:
                for t in range(U):
                    remaining[i, t] = rem.get(t, 0)
            state_dones = state["_dones"]
            for pos, key in enumerate(state["_indices"]):
                dones[i, pos] = float(state_dones[key])
                pos_type[i, pos * U + state["_envs_unique"][key]] = 1.0
            # Fill substates, grouped per unique type
            indices_unique = np.array(state["_envs_unique"])
            offset = 0
            for t in range(U):
                subenv = self._get_env_unique(t)
                keys = np.where(indices_unique == t)[0]
                if len(keys) > 0:
                    sub = subenv.states2policy(
                        [self._get_substate(state, int(k)) for k in keys]
                    ).flatten()
                    substates[i, offset : offset + sub.shape[0]] = sub
                offset += subenv.policy_input_dim * M

        return torch.cat([active, remaining, dones, pos_type, substates], dim=1)

    def states2proxy(self, states: List[Dict]) -> List[List]:
        """
        Proxy representation: for each state, the list of the sub-environments' proxy
        representations, in the spatial (sequence) order.
        """
        states_proxy = []
        for state in states:
            seq = []
            for key in state["_indices"]:
                idx_unique = state["_envs_unique"][key]
                substate = self._get_substate(state, key)
                seq.append(self._get_env_unique(idx_unique).state2proxy(substate)[0])
            states_proxy.append(seq)
        return states_proxy

    def state2readable(self, state: Optional[Dict] = None) -> str:
        """Converts a state into a human-readable representation."""
        state = self._get_state(state)
        header = f"Active: {state['_active']}; Indices: {state['_indices']};"
        if "_bag" in state:
            header += f" Bag: {state['_bag']};"
        header += "\n"
        body = ""
        for key in range(self._seq_length(state)):
            idx_unique = state["_envs_unique"][key]
            subenv = self._get_env_unique(idx_unique)
            done_str = " | done" if state["_dones"][key] else ""
            body += (
                f"{idx_unique}: "
                + subenv.state2readable(self._get_substate(state, key))
                + done_str
                + ";\n"
            )
        return (header + body).rstrip("\n").rstrip(";")

    def readable2state(self, readable: str) -> Dict:
        """Converts a human-readable representation into a state."""
        lines = [line for line in readable.split("\n") if line.strip() != ""]
        header = lines[0]
        parts = [p for p in header.split(";") if p.strip() != ""]
        active = int(parts[0].split(":")[1].strip())
        indices = ast.literal_eval(parts[1].split(":", 1)[1].strip())

        state = {
            "_active": active,
            "_dones": [],
            "_envs_unique": [],
            "_indices": indices,
        }
        for part in parts[2:]:
            if part.strip().startswith("Bag:"):
                state["_bag"] = ast.literal_eval(part.split(":", 1)[1].strip())
        for key, line in enumerate(lines[1:]):
            line = line.rstrip(";").strip()
            idx_unique_str, rest = line.split(":", 1)
            idx_unique = int(idx_unique_str.strip())
            done = " | done" in rest
            if done:
                rest = rest.replace(" | done", "")
            subenv = self._get_env_unique(idx_unique)
            state["_envs_unique"].append(idx_unique)
            state["_dones"].append(int(done))
            state[key] = subenv.readable2state(rest.strip())
        return state

    # ------------------------------------------------------------------ #
    # Set/reset/source utilities
    # ------------------------------------------------------------------ #

    def _sample_random_bag(self) -> List[int]:
        """Samples a random bag of unique-type indices (fixed-bag testing)."""
        n = np.random.randint(low=1, high=self.max_elements + 1)
        return list(np.random.choice(a=self.n_unique_envs, size=n, replace=True))

    def reset(self, env_id: Union[int, str] = None):
        """Resets the environment to an empty sequence."""
        self.subenvs = []
        self.state = {
            "_active": -1,
            "_dones": [],
            "_envs_unique": [],
            "_indices": [],
        }
        if self._is_fixed_bag:
            if self.do_random_subenvs:
                bag = self._sample_random_bag()
            else:
                bag = list(self._fixed_bag)
            self.state["_bag"] = sorted(int(t) for t in bag)
        self.n_actions = 0
        self.done = False
        self.id = env_id if env_id is not None else str(uuid.uuid4())
        return self

    def set_state(self, state: Dict, done: Optional[bool] = False):
        """Sets a state and done, rebuilding the placed sub-environments."""
        self.subenvs = [
            self._make_subenv_instance(idx_unique, key)
            for key, idx_unique in enumerate(state["_envs_unique"])
        ]
        return super().set_state(state, done)

    def is_source(self, state: Optional[Dict] = None) -> bool:
        """A state is source if no element has been placed and none is active."""
        state = self._get_state(state)
        return state["_active"] == -1 and self._seq_length(state) == 0

    def action2representative(self, action: Tuple) -> Tuple:
        """Replaces the sub-environment part of an action by its representative."""
        idx_unique = action[0]
        if idx_unique == -1:
            return action
        subenv = self._get_env_unique(idx_unique)
        action_subenv = self._depad_action(action, idx_unique)
        representative = subenv.action2representative(action_subenv)
        return self._pad_action(representative, idx_unique)

    def _get_max_trajectory_length(self) -> int:
        """Upper bound on the trajectory length (including EOS)."""
        max_subenv_traj = max([env.max_traj_length for env in self.envs_unique])
        return self.max_elements * (1 + max_subenv_traj) + 1

    def __eq__(self, other, ignored_keys: List[str] = []) -> bool:
        """Equality between environment instances."""
        if self.do_random_subenvs:
            ignored_keys = ignored_keys + ["subenvs", "state", "source"]
        return super().__eq__(other, ignored_keys=ignored_keys)
