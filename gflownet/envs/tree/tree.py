"""
Tree composite environment for binary decision tree construction.

A Tree object represents a binary decision tree built incrementally by adding one
internal node at a time. Each internal node is a sub-environment (e.g. a
DecisionTreeNode = Stack(Choice, ContinuousCube)).

The construction follows a toggle pattern (analogous to SetBase with can_alternate_subenvs = False):
1. (Idle) Select a leaf position to expand via toggle meta-action → activates node.
2. (Building) Build the node via sub-environment actions until the subenv is done.
3. (Node done) Deactivate the completed node via toggle meta-action → back to idle.
4. Repeat from (1) or take global EOS.

Nodes are indexed using breadth-first ordering:
- Root: 0
- Left child of k: 2k + 1
- Right child of k: 2k + 2
- Parent of k: (k - 1) // 2
- Depth of k: floor(log2(k + 1))

State representation (dictionary):
- ``_active``: Index of the currently active node, or -1 if idle.
- ``_dones``: List of done flags for each possible node position (0 or 1).
- Integer keys (0, 1, ...): Sub-states of existing nodes.

Mask format (flat, prefix-free):
- First ``max_nodes + 1`` elements (meta section): toggle targets + EOS.
  All True when in building mode (no meta-actions valid).
- Next ``node_env.mask_dim`` elements (node section): node env action mask.
  All True when in meta-action mode (no node actions valid).
- Mode detection: building mode iff node section is not all True.

Action format:
- Toggle meta-actions: ``(-1, k, 0, ...)`` to activate/deactivate node k.
- Global EOS: ``(-1, -1, 0, ...)``.
- Node env actions: ``(0, *subenv_action, 0, ...)`` where 0 is the unique env index.
"""

import math
import uuid
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
from torchtyping import TensorType

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.composite.base import CompositeBase
from gflownet.envs.tree.node import DecisionTreeNode
from gflownet.utils.common import copy, tfloat


class Tree(CompositeBase):
    """
    Composite environment that constructs a binary decision tree by iteratively
    expanding leaf positions into internal nodes.

    Each internal node is a sub-environment (typically a DecisionTreeNode). The
    tree supports variable structure: not all leaves need to be expanded, and
    different branches can have different depths.

    Attributes
    ----------
    max_depth : int
        Maximum depth of the tree. Nodes at depth ``max_depth - 1`` are the
        deepest that can split.
    node_env : DecisionTreeNode
        Template (unique) environment used for each tree node.
    max_nodes : int
        Maximum number of internal nodes: ``2^max_depth - 1``.
    """

    def __init__(
        self,
        max_depth: int = 3,
        node_kwargs: dict = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        max_depth : int
            Maximum depth of the decision tree. Internal nodes exist at depths
            0 through ``max_depth - 1``. By default: 3.
        node_kwargs : dict
            Optional arguments for the DecisionTreeNode sub-environments.
        """
        if max_depth < 1 or not isinstance(max_depth, int):
            warnings.warn(
                f"Tree requires max_depth to be of type int (got {type(max_depth)}). "
                f"Tree requires max_depth >= 1 (got {max_depth}). ",
                UserWarning,
                stacklevel=2,
            )
            raise ValueError(f"Tree requires max_depth >= 1, got {max_depth}.")
        self.max_depth = max_depth
        self.node_env = DecisionTreeNode(**(node_kwargs or {}))
        self.subenvs = None  # Dynamic; nodes are created on demand

        # Max internal nodes: levels 0 to max_depth - 1
        self.max_nodes = 2**max_depth - 1
        self.max_elements = self.max_nodes

        # Single unique environment type (all nodes are the same kind)
        self.envs_unique = [self.node_env]
        self._n_unique_envs = len(self.envs_unique)
        assert (
            self._n_unique_envs == 1
        ), "Currenlty only a tree with a single env type as node is supported"

        # Source state: empty tree, idle
        self.source = {
            "_active": -1,
            "_dones": [0] * self.max_nodes,
            "_envs_unique": self._n_unique_envs,  # Useless?
        }

        # Get action dimensionality by adding one to the action dim of the
        # sub-environment, where the prefix -1 indicates a meta action in
        # the Tree env and a 0 an action in the subenv
        self.action_dim = len(self.node_env.eos) + 1

        # EOS of the Tree: meta-action with target -1
        self.eos = self._pad_action((-1,), -1)

        # Policy distribution parameters
        kwargs["fixed_distr_params"] = [self.node_env.fixed_distr_params]
        kwargs["random_distr_params"] = [self.node_env.random_distr_params]

        # Base class init (CompositeBase → GFlowNetEnv)
        super().__init__(**kwargs)

        self.continuous = self.node_env.continuous

    # =========================================================================
    # Tree structure helpers
    # =========================================================================

    @staticmethod
    def left_child_idx(k: int) -> int:
        """Returns the breadth-first index of the left child of node ``k``."""
        return 2 * k + 1

    @staticmethod
    def right_child_idx(k: int) -> int:
        """Returns the breadth-first index of the right child of node ``k``."""
        return 2 * k + 2

    @staticmethod
    def parent_idx(k: int) -> int:
        """Returns the breadth-first index of the parent of node ``k``."""
        return (k - 1) // 2

    @staticmethod
    def node_depth(k: int) -> int:
        """Returns the depth of node ``k`` in the tree (root has depth 0)."""
        return int(math.log2(k + 1))

    def _node_exists(self, k: int, state: Dict) -> bool:
        """Returns True if node ``k`` has been created in the state."""
        return k in state

    def _node_is_done(self, k: int, state: Dict) -> bool:
        """Returns True if node ``k`` is fully built (done)."""
        return 0 <= k < self.max_nodes and state["_dones"][k] == 1

    def _is_idle(self, state: Dict) -> bool:
        """Returns True if no node is currently being built."""
        return state["_active"] == -1

    def _get_expandable_leaves(self, state: Dict) -> List[int]:
        """
        Returns node positions available for expansion.

        A position is expandable if:
        - It is within the valid internal-node range (index < max_nodes).
        - No node exists at that position yet.
        - Its parent is a done node (or it is the root).
        """
        if not self._node_exists(0, state):  # root does not exist yet
            return [0]

        leaves = []
        for k in range(self.max_nodes):  # iterate over all possible node positions
            if not self._node_is_done(k, state):
                continue
            for child in (self.left_child_idx(k), self.right_child_idx(k)):
                if child < self.max_nodes and not self._node_exists(
                    child, state
                ):  # Check if space available for child and child not exists yet
                    leaves.append(child)
        return leaves

    def _get_done_nodes(self, state: Dict) -> List[int]:
        """Returns the list of indices of all done nodes."""
        return [k for k in range(self.max_nodes) if self._node_is_done(k, state)]

    def _get_leaf_done_nodes(self, state: Dict) -> List[int]:
        """
        Returns done nodes that are "leaves" of the current tree. Leaves are
        nodes whose children do not exist (yet).

        These are the only nodes that can be un-built in the backward direction,
        because in any valid forward trajectory, a parent must be deactivated
        before its children are created.
        """
        leaves = []
        for k in self._get_done_nodes(state):
            left = self.left_child_idx(k)
            right = self.right_child_idx(k)
            if not self._node_exists(left, state) and not self._node_exists(
                right, state
            ):
                leaves.append(k)
        return leaves

    def _has_any_done_node(self, state: Dict) -> bool:
        """Returns True if at least one node is done."""
        return any(d == 1 for d in state["_dones"])

    def _is_in_left_subtree(self, k: int, ancestor: int) -> bool:
        """
        Returns True if node ``k`` is in the left subtree of ``ancestor``.

        Walks from ``k`` up to ``ancestor`` and checks whether the first step
        from ``ancestor`` was to its left child.
        """
        child = k
        while child != ancestor:
            parent = self.parent_idx(child)
            if parent == ancestor:
                return child == self.left_child_idx(ancestor)
            child = parent
        return False

    def _get_threshold_bounds(self, k: int, state: Dict) -> Tuple[float, float]:
        """
        Computes the valid threshold range for node ``k`` based on ancestor
        constraints.

        For each done ancestor of ``k`` that splits on the same feature:
        - If ``k`` is in the left subtree of that ancestor (the ``<=`` branch),
          the threshold must be less than the ancestor's threshold (upper bound).
        - If ``k`` is in the right subtree (the ``>`` branch), the threshold
          must be greater than the ancestor's threshold (lower bound).

        Parameters
        ----------
        k : int
            Node index.
        state : dict
            Current tree state. Node ``k`` must exist and have its feature set.

        Returns
        -------
        (lower, upper) : tuple of float
            The valid threshold range in [0, 1].
        """
        feature_idx = self.node_env.get_feature(state[k])
        if feature_idx is None:
            return (0.0, 1.0)

        lower = 0.0
        upper = 1.0
        current = k
        while current > 0:
            parent = self.parent_idx(current)
            if not self._node_is_done(parent, state):
                raise RuntimeError(
                    f"Node {k} has an undone parent, which should be impossible"
                )

            parent_feature = self.node_env.get_feature(state[parent])
            if parent_feature == feature_idx:
                parent_threshold = self.node_env.get_threshold(state[parent])
                if parent_threshold is not None:
                    if self._is_in_left_subtree(k, parent):
                        # k is on the <= side: threshold must be < parent's
                        upper = min(upper, parent_threshold)
                    else:
                        # k is on the > side: threshold must be > parent's
                        lower = max(lower, parent_threshold)
            current = parent

        return (lower, upper)

    def _rescale_threshold(self, k: int, state: Dict) -> None:
        """
        Rescales the threshold of done node ``k`` from the raw [0, 1] range
        of the ContinuousCube to the valid range [lower, upper] determined by
        ancestor constraints.

        This ensures that the stored threshold respects the decision-tree
        semantics: a child on the ``<=`` branch cannot have a threshold
        higher than its ancestor's, and a child on the ``>`` branch cannot
        have one lower.

        Parameters
        ----------
        k : int
            Index of the node whose threshold is to be rescaled.
        state : dict
            Current tree state. Node ``k`` must be done.
        """
        lower, upper = self._get_threshold_bounds(k, state)
        if lower == 0.0 and upper == 1.0:
            return  # No rescaling needed

        raw = self.node_env.get_threshold(state[k])
        if raw is None:
            return

        rescaled = lower + raw * (upper - lower)
        # Update the threshold in the substate
        threshold_key = self.node_env.stage_threshold
        state[k][threshold_key] = [rescaled]

    def _unrescale_threshold(self, k: int, state: Dict) -> None:
        """
        Reverses the threshold rescaling for node ``k``, converting back from
        the [lower, upper] range to the raw [0, 1] range of the ContinuousCube.

        This is needed when undoing a node's completion in a backward step.

        Parameters
        ----------
        k : int
            Index of the node whose threshold is to be unrescaled.
        state : dict
            Current tree state. Node ``k`` must be done and have its threshold
            set.
        """
        lower, upper = self._get_threshold_bounds(k, state)
        if lower == 0.0 and upper == 1.0:
            return

        actual = self.node_env.get_threshold(state[k])
        if actual is None:
            return

        span = upper - lower
        if span > 0:
            raw = (actual - lower) / span
        else:
            raw = 0.0

        threshold_key = self.node_env.stage_threshold
        state[k][threshold_key] = [raw]

    def _get_max_trajectory_length(self) -> int:
        # Per node: 1 activate toggle + node_env trajectory + 1 deactivate toggle
        # Plus 1 global EOS
        return self.max_nodes * (self.node_env.max_traj_length + 2) + 1

    # =========================================================================
    # CompositeBase overrides
    # =========================================================================

    def _get_unique_indices(self, state=None):
        """All nodes share the same unique environment (index 0)."""
        return [0] * self.max_nodes

    def _get_dones(self, state=None):
        state = self._get_state(state)
        return state["_dones"]

    @property
    def n_toggle_actions(self) -> int:
        """Number of toggle meta-actions (one per possible node position)."""
        return self.max_nodes

    @property
    def n_meta_actions(self) -> int:
        """Number of elements in the meta-action mask: toggles + EOS."""
        return self.max_nodes + 1

    # =========================================================================
    # Action space
    # =========================================================================

    def get_action_space(self) -> List[Tuple]:
        """
        Constructs the full action space.

        The action space consists of:
        1. Toggle meta-actions ``(-1, k, 0, ...)`` for each node position k.
        2. Global EOS ``(-1, -1, 0, ...)``.
        3. Node env actions ``(0, *subenv_action, 0, ...)``.
        """
        action_space = []
        # Toggle meta-actions
        for k in range(self.max_nodes):
            action_space.append(self._pad_action((k,), -1))
        # EOS
        action_space.append(self.eos)
        # Node env actions
        for action in self.node_env.action_space:
            action_space.append(self._pad_action(action, 0))
        return action_space

    # =========================================================================
    # Mask dimensionality and formatting
    # Flat mask: [META SECTION | NODE SECTION]
    # =========================================================================

    def _compute_mask_dim(self) -> int:
        """
        The flat mask has two sections::

        1. Meta section (``n_meta_actions`` elements): toggle actions + EOS.
        2. Node section (``node_env.mask_dim`` elements): node env action mask.
        """
        return self.n_meta_actions + self.node_env.mask_dim

    def _format_mask_meta(self, meta_mask: List[bool]) -> List[bool]:
        """
        Formats a meta-action mask (idle or deactivation mode).

        The node section is set to all True (all node actions invalid).

        Parameters
        ----------
        meta_mask : list[bool]
            Mask of length ``n_meta_actions`` (toggles + EOS). True = invalid action.
        """
        if len(meta_mask) != self.n_meta_actions:
            raise ValueError(
                f"Size of meta masked passed is {len(meta_mask)} but expected {self.n_meta_actions}"
            )
        node_section = [True] * self.node_env.mask_dim
        return meta_mask + node_section

    def _format_mask_building(self, node_mask: List[bool]) -> List[bool]:
        """
        Formats a building-mode mask (node env actions).

        The meta section is set to all True (all meta-actions invalid).

        Parameters
        ----------
        node_mask : list[bool]
            Node env mask. True = invalid.
        """
        if len(node_mask) != self.node_env.mask_dim:
            raise ValueError(
                f"Size of node masked passed is {len(node_mask)} but expected {self.node_env.mask_dim}"
            )
        meta_section = [True] * self.n_meta_actions
        return meta_section + node_mask

    def _is_meta_mask(self, mask: Union[List[bool], TensorType["mask_dim"]]) -> bool:
        """Returns True if the mask is NOT in building mode.

        Building mode is detected by the node section (after ``n_meta_actions``)
        having at least one valid (False) entry. If the node section is all True,
        the mask is in meta mode (or all-invalid for done/source states).
        """
        if isinstance(mask, list):
            return all(mask[self.n_meta_actions :])
        return mask[self.n_meta_actions :].all().item()

    def _unformat_mask_building(
        self,
        mask: Union[List[bool], TensorType["batch_size", "mask_dim"]],
    ):
        """Extracts the node env mask (node section) from a flat mask."""
        if isinstance(mask, list):
            return mask[self.n_meta_actions :]
        return mask[:, self.n_meta_actions :]

    def _unformat_mask_meta(
        self,
        mask: Union[List[bool], TensorType["batch_size", "mask_dim"]],
    ):
        """Extracts the meta-action mask (meta section) from a flat mask."""
        if isinstance(mask, list):
            return mask[: self.n_meta_actions]
        return mask[:, : self.n_meta_actions]

    # =========================================================================
    # Forward mask
    # =========================================================================

    def get_mask_invalid_actions_forward(
        self, state: Optional[Dict] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Computes the forward mask.

        Three cases:
        1. Idle: meta-action mask with valid toggles for expandable leaves + global EOS.
        2. Building (node not done): building mask is mask of node env.
        3. Active done (node done, awaiting deactivation): meta-action mask with
           only the deactivation toggle for the active node as possibility.
        """
        state = self._get_state(state)
        done = self._get_done(done)

        if done:  # Global done -> global EOS already sampled, no more valid actions
            meta_mask = [True] * self.n_meta_actions
            return self._format_mask_meta(meta_mask)

        active = state["_active"]

        if active == -1:
            # Idle: toggles for expandable leaves + global EOS
            expandable = set(self._get_expandable_leaves(state))
            meta_mask = [k not in expandable for k in range(self.max_nodes)]
            # Global EOS valid only if root is done
            eos_invalid = not self._node_is_done(0, state)
            meta_mask.append(eos_invalid)
            return self._format_mask_meta(meta_mask)

        if self._node_is_done(active, state):
            # Active node is done: only deactivation toggle valid
            meta_mask = [True] * self.max_nodes
            meta_mask[active] = False
            meta_mask.append(True)  # global EOS invalid while a node is active
            return self._format_mask_meta(meta_mask)

        # Building: delegate to node env
        # TODO: Apply constraints here?
        substate = state[active]

        mask = self.node_env.get_mask_invalid_actions_forward(substate, False)
        return self._format_mask_building(mask)

    # =========================================================================
    # Backward mask
    # =========================================================================

    def get_mask_invalid_actions_backward(
        self, state: Optional[Dict] = None, done: Optional[bool] = None
    ) -> List[bool]:
        """
        Computes the backward mask.

        Cases:
        1. Done: only global EOS is valid backward.
        2. Idle at source: no valid backward actions.
        3. Idle (not source): can reactivate any done node (backward of deactivation
           toggle).
        4. Active node done: backward of node env EOS (the building mask with done=True).
        5. Building, node in initial state (no actions in node env taken yet): backward of activation toggle (meta-action).
        6. Building, node in progress: delegate to node env backward mask.
        """
        state = self._get_state(state)
        done = self._get_done(done)

        if done:
            # Only EOS backward
            meta_mask = [True] * self.max_nodes + [False]
            return self._format_mask_meta(meta_mask)

        active = state["_active"]

        if active == -1:
            # Idle at source
            if not self._node_exists(0, state):
                # Source state
                return [True] * self.mask_dim

            # Idle not at source
            # Can reactivate only leaf done nodes (no existing children)
            leaf_done = set(self._get_leaf_done_nodes(state))
            meta_mask = [k not in leaf_done for k in range(self.max_nodes)]
            meta_mask.append(True)  # EOS invalid
            return self._format_mask_meta(meta_mask)

        if self._node_is_done(active, state):
            # Active and done: backward = undo node env EOS.
            # Unrescale threshold for the ContinuousCube.
            substate = copy(state[active])
            temp_state = {**state, active: substate}
            self._unrescale_threshold(active, temp_state)
            mask = self.node_env.get_mask_invalid_actions_backward(substate, done=True)
            return self._format_mask_building(mask)

        # Building, not done
        substate = state[active]

        if self.node_env.is_source(substate):
            # Node in its initial state: backward = undo activation toggle
            meta_mask = [True] * self.max_nodes
            meta_mask[active] = False
            meta_mask.append(True)
            return self._format_mask_meta(meta_mask)

        # In progress: delegate to node env
        mask = self.node_env.get_mask_invalid_actions_backward(substate, False)
        return self._format_mask_building(mask)

    # =========================================================================
    # Step (forward)
    # =========================================================================

    def step(
        self, action: Tuple, skip_mask_check: bool = False
    ) -> Tuple[Dict, Tuple, bool]:
        """
        Executes a forward step.

        Meta-actions (prefix -1):
        - EOS ``(-1, -1, ...)``: terminates the tree.
        - Toggle ``(-1, k, ...)``:
          - If active == -1 (idle) and k is expandable: activates node k (creates it at source).
          - If active == k and node k is done: deactivates (goes idle).

        Node env actions (prefix 0):
        - Delegated to the node env. If the node env becomes done after the
          action, ``_dones[k]`` is set to 1.
        """
        if self.done:  # any action invalid, return False
            return self.state, action, False

        is_meta = action[0] == -1

        if is_meta:
            depadded = self._depad_action(action, -1)
            target = depadded[0]

            if target == -1:
                # Global EOS
                active = self.state["_active"]
                if active != -1 or not self._node_is_done(
                    0, self.state
                ):  # Can not terminante if source node is not done
                    return self.state, action, False
                self.done = True
                self.n_actions += 1
                return self.state, action, True

            # Toggle meta-action
            active = self.state["_active"]

            if active == -1:
                # Activation: create node at the target leaf position
                expandable = set(self._get_expandable_leaves(self.state))
                if target not in expandable:
                    return self.state, action, False
                self.state["_active"] = target
                self.state[target] = copy(self.node_env.source)
                self.n_actions += 1
                return self.state, action, True

            if active == target and self._node_is_done(active, self.state):
                # Deactivation: go idle after node completion
                self.state["_active"] = -1
                self.n_actions += 1
                return self.state, action, True

            # Invalid toggle
            return self.state, action, False

        # Node env action
        active = self.state["_active"]
        if active == -1 or self._node_is_done(active, self.state):
            return self.state, action, False

        idx_unique = action[0]
        if (
            idx_unique != 0
        ):  # env actions start with 0 (unique env idx 0 for all envs currently)
            return self.state, action, False
        action_subenv = self._depad_action(action, idx_unique)

        # Prepare the node env
        substate = self.state[active]
        self.node_env.set_state(copy(substate), done=False)

        # Pre-step check via node env
        action_to_check = self.node_env.action2representative(action_subenv)
        skip = skip_mask_check or self.skip_mask_check or self.node_env.continuous
        do_step, _, _ = self.node_env._pre_step(action_to_check, skip_mask_check=skip)
        if not do_step:
            return self.state, action, False

        # Step the node env
        _, action_subenv, valid = self.node_env.step(action_subenv)
        if not valid:
            return self.state, action, False

        self.n_actions += 1

        # Update the substate in the tree
        self.state[active] = copy(self.node_env.state)

        # If node env is now done, mark the node as done and rescale threshold
        if self.node_env.done:
            self.state["_dones"][active] = 1
            self._rescale_threshold(active, self.state)

        return self.state, action, True

    # =========================================================================
    # Step backwards
    # =========================================================================

    def step_backwards(
        self, action: Tuple, skip_mask_check: bool = False
    ) -> Tuple[Dict, Tuple, bool]:
        """
        Executes a backward step.

        Meta-actions (prefix -1):
        - EOS backward: un-terminates (sets done = False).
        - Toggle backward:
          - If idle: reactivates a done node (backward of deactivation).
          - If active and node at source: removes node, goes idle (backward of
            activation).

        Node env actions (prefix 0):
        - Delegated to node env backward step. If the node was done and is no
          longer done after the backward step, ``_dones[k]`` is set back to 0.
        """
        is_meta = action[0] == -1

        if is_meta:
            depadded = self._depad_action(action, -1)
            target = depadded[0]

            if target == -1:
                # Global EOS backward
                if not self.done:
                    return self.state, action, False
                self.done = False
                self.n_actions += 1
                return self.state, action, True

            # Toggle backward
            active = self.state["_active"]

            if active == -1:
                # Backward of deactivation: reactivate a leaf done node
                if not self._node_is_done(target, self.state):
                    return self.state, action, False
                # Only allow reactivation if children don't exist
                left = self.left_child_idx(target)
                right = self.right_child_idx(target)
                if self._node_exists(left, self.state) or self._node_exists(
                    right, self.state
                ):
                    return self.state, action, False
                self.state["_active"] = target
                self.n_actions += 1
                return self.state, action, True

            if active == target and not self._node_is_done(active, self.state):
                # Backward of activation: remove node, go idle
                substate = self.state.get(active)
                if substate is None or not self.node_env.is_source(substate):
                    return self.state, action, False
                del self.state[active]
                self.state["_active"] = -1
                self.n_actions += 1
                return self.state, action, True

            return self.state, action, False

        # Node env action backward
        active = self.state["_active"]
        if active == -1:
            return self.state, action, False

        idx_unique = action[0]
        if idx_unique != 0:
            return self.state, action, False
        action_subenv = self._depad_action(action, idx_unique)

        node_was_done = self._node_is_done(active, self.state)

        # Prepare the node env. If the node was done, unrescale the threshold
        # back to [0, 1] so the ContinuousCube can process the backward step.
        substate = copy(self.state[active])
        if node_was_done:
            self._unrescale_threshold(
                active,
                {
                    **self.state,
                    active: substate,
                },
            )
        self.node_env.set_state(substate, done=node_was_done)

        # Pre-step check
        action_to_check = self.node_env.action2representative(action_subenv)
        skip = skip_mask_check or self.skip_mask_check or self.node_env.continuous
        do_step, _, _ = self.node_env._pre_step(
            action_to_check, backward=True, skip_mask_check=skip
        )
        if not do_step:
            return self.state, action, False

        # Step the node env backwards
        _, _, valid = self.node_env.step_backwards(action_subenv)
        if not valid:
            return self.state, action, False

        self.n_actions += 1

        # Update substate
        self.state[active] = copy(self.node_env.state)

        # If the node was done but isn't anymore, clear the done flag
        if node_was_done and not self.node_env.done:
            self.state["_dones"][active] = 0

        return self.state, action, True

    # =========================================================================
    # Get parents
    # =========================================================================

    def get_parents(
        self,
        state: Optional[Dict] = None,
        done: Optional[bool] = None,
        action: Optional[Tuple] = None,
    ) -> Tuple[List, List]:
        """
        Determines all parents and actions that lead to the input state.

        Cases:
        1. Done: single parent = same state, action = EOS.
        2. Idle: for each done node k, a parent with ``_active = k`` and action
           = toggle k (backward of deactivation).
        3. Active done: delegate to node env ``get_parents(done=True)`` and wrap.
        4. Building, node in intitial state: parent is idle state without this node.
        5. Building, in progress: delegate to node env ``get_parents``.
        """
        state = self._get_state(state)
        done = self._get_done(done)

        if done:
            return [state], [self.eos]

        active = state["_active"]

        if active == -1:
            # Idle: parents are states with a leaf done node reactivated
            # Only leaf done nodes (no existing children) can be un-deactivated,
            # because in a valid trajectory, a parent is deactivated before its
            # children are created.
            parents = []
            actions = []
            for k in self._get_leaf_done_nodes(state):
                parent = copy(state)
                parent["_active"] = k
                parents.append(parent)
                actions.append(self._pad_action((k,), -1))
            return parents, actions

        if self._node_is_done(active, state):
            # Active and done: parent has node not yet done (backward of node EOS)
            # Unrescale the threshold before passing to node env (which expects
            # raw [0,1] ContinuousCube values).
            substate = copy(state[active])
            temp_state = {**state, active: substate}
            self._unrescale_threshold(active, temp_state)
            parents_subenv, actions_subenv = self.node_env.get_parents(
                substate, done=True
            )
            parents = []
            actions = []
            for parent_subenv, action_subenv in zip(parents_subenv, actions_subenv):
                parent = copy(state)
                parent[active] = parent_subenv
                parent["_dones"] = list(parent["_dones"])
                parent["_dones"][active] = 0
                parents.append(parent)
                actions.append(self._pad_action(action_subenv, 0))
            return parents, actions

        # Building
        substate = state[active]

        if self.node_env.is_source(substate):
            # Node at source: parent = idle state without this node
            parent = copy(state)
            del parent[active]
            parent["_active"] = -1
            return [parent], [self._pad_action((active,), -1)]

        # In progress: delegate to node env
        parents_subenv, actions_subenv = self.node_env.get_parents(substate, False)
        parents = []
        actions = []
        for parent_subenv, action_subenv in zip(parents_subenv, actions_subenv):
            parent = copy(state)
            parent[active] = parent_subenv
            parents.append(parent)
            actions.append(self._pad_action(action_subenv, 0))
        return parents, actions

    # =========================================================================
    # Valid actions
    # =========================================================================

    def get_valid_actions(
        self,
        mask: Optional[bool] = None,
        state: Optional[Dict] = None,
        done: Optional[bool] = None,
        backward: Optional[bool] = False,
    ) -> List[Tuple]:
        """Returns the list of valid actions given the mask and state."""
        state = self._get_state(state)
        done = self._get_done(done)

        if mask is None:
            if backward:
                mask = self.get_mask_invalid_actions_backward(state, done)
            else:
                mask = self.get_mask_invalid_actions_forward(state, done)

        if self._is_meta_mask(mask):
            # Meta-action mode
            meta_section = self._unformat_mask_meta(mask)
            valid = []
            for k in range(self.max_nodes):
                if not meta_section[k]:
                    valid.append(self._pad_action((k,), -1))
            if not meta_section[self.max_nodes]:
                valid.append(self.eos)
            return valid

        # Building mode
        active = state["_active"]
        substate = state[active]
        subenv_mask = self._unformat_mask_building(mask)
        node_done = self._node_is_done(active, state)
        valid_subenv = self.node_env.get_valid_actions(
            subenv_mask, substate, node_done, backward
        )
        return [self._pad_action(a, 0) for a in valid_subenv]

    # =========================================================================
    # State representation
    # =========================================================================

    def states2policy(
        self, states: List[Dict]
    ) -> TensorType["batch", "state_policy_dim"]:
        """
        Converts a batch of states to policy input tensors.

        Shape is [ idle_flag | node_0_block | node_1_block | ... | node_{max_nodes-1}_block ]

        For each node position: [exists, done, active, node_env_policy].
        Plus a global idle flag.
        """
        batch_size = len(states)
        node_pdim = self.node_env.policy_input_dim
        # idle_flag + per_node * (exists + done + active + node_policy)
        per_node_dim = 3 + node_pdim
        policy_dim = 1 + self.max_nodes * per_node_dim
        result = torch.zeros(
            batch_size, policy_dim, dtype=self.float, device=self.device
        )

        subenv_policy = self.node_env.state2policy(self.node_env.source)

        for i, state in enumerate(states):
            result[i, 0] = 1.0 if self._is_idle(state) else 0.0
            for k in range(self.max_nodes):
                offset = 1 + k * per_node_dim
                if self._node_exists(k, state):
                    result[i, offset] = 1.0
                    result[i, offset + 1] = float(self._node_is_done(k, state))
                    result[i, offset + 2] = float(state["_active"] == k)
                    node_policy = self.node_env.state2policy(state[k])
                    result[i, offset + 3 : offset + 3 + node_pdim] = node_policy
                else:
                    result[i, offset + 3 : offset + 3 + node_pdim] = subenv_policy
        return result

    def states2proxy(self, states: List[Dict]) -> List[Dict]:
        """
        Prepares a batch of states for the proxy (reward function).

        Returns the states as-is. Subclasses or proxies should handle the tree
        structure interpretation.
        """
        # TODO: implement proper proxy format for decision tree evaluation
        return states

    def state2readable(self, state: Optional[Dict] = None) -> str:
        """Converts a state into a human-readable string."""
        state = self._get_state(state)
        parts = [f"Active: {state['_active']}"]

        for k in range(self.max_nodes):
            if self._node_exists(k, state):
                status = "done" if self._node_is_done(k, state) else "building"
                depth = self.node_depth(k)
                node_str = self.node_env.state2readable(state[k])
                parts.append(f"Node {k} (depth={depth}, status={status}): {node_str}")

        leaves = self._get_expandable_leaves(state)
        if leaves:
            parts.append(f"Expandable: {leaves}")

        return "; ".join(parts)

    def display(self, state: Optional[Dict] = None, ax=None):
        """
        Displays the tree as a graph using networkx with a manual hierarchical
        layout that guarantees True (``<=``) edges route to the left child and
        False (``>``) edges route to the right child, visually.

        Internal (done) nodes show the decision rule: ``feature <= threshold``.
        Leaf positions (children of done nodes that don't exist) are shown as
        ``"Leaf"``. Edges are labelled ``True`` (left / yes) and ``False``
        (right / no).

        Parameters
        ----------
        state : dict, optional
            A state. If None, ``self.state`` is used.
        ax : matplotlib Axes, optional
            Axes to draw on. If None, a new figure is created.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the tree plot.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        state = self._get_state(state)

        G = nx.DiGraph()
        labels = {}
        node_colors = {}
        # Maps networkx node id -> logical BFS index used for layout. For real
        # nodes this is the node's own BFS index; for leaf placeholders it is
        # the BFS index of the missing child slot they represent, which is what
        # positions them correctly left/right under their parent.
        bfs_index_of = {}
        internal_color = "#4a90d9"
        leaf_color = "#90d94a"
        building_color = "#d9a04a"

        # Add done (internal) nodes
        for k in self._get_done_nodes(state):
            G.add_node(k)
            bfs_index_of[k] = k
            feature_idx = self.node_env.get_feature(state[k])
            threshold = self.node_env.get_threshold(state[k])
            if feature_idx is not None:
                fname = str(self.node_env.features[feature_idx - 1])
            else:
                fname = "?"
            tval = f"{threshold:.2f}" if threshold is not None else "?"
            labels[k] = f"{fname}\n<= {tval}"
            node_colors[k] = internal_color

        # Add the active (building) node if any
        active = state["_active"]
        if active >= 0 and active not in G:
            G.add_node(active)
            bfs_index_of[active] = active
            feature_idx = self.node_env.get_feature(state[active])
            threshold = self.node_env.get_threshold(state[active])
            fname = "?"
            if feature_idx is not None:
                fname = str(self.node_env.features[feature_idx - 1])
            tval = f"{threshold:.2f}" if threshold is not None else "?"
            labels[active] = f"{fname}\n<= {tval}\n(building)"
            node_colors[active] = building_color

        # Add leaf placeholders (children of done nodes that don't exist)
        leaf_id = self.max_nodes  # use ids beyond max_nodes for leaf placeholders
        for k in self._get_done_nodes(state):
            for child in (self.left_child_idx(k), self.right_child_idx(k)):
                is_left = child == self.left_child_idx(k)
                if not self._node_exists(child, state):
                    G.add_node(leaf_id)
                    bfs_index_of[leaf_id] = child
                    labels[leaf_id] = "Leaf"
                    node_colors[leaf_id] = leaf_color
                    G.add_edge(k, leaf_id, label="True" if is_left else "False")
                    leaf_id += 1
                else:
                    G.add_edge(k, child, label="True" if is_left else "False")

        # Also connect active node to its parent if applicable
        if active >= 0 and active > 0:
            p = self.parent_idx(active)
            if p in G and not G.has_edge(p, active):
                is_left = active == self.left_child_idx(p)
                G.add_edge(p, active, label="True" if is_left else "False")

        if len(G.nodes) == 0:
            print("Empty tree — no nodes to display.")
            return None

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(max(6, 2 * len(G.nodes)), 6))
        else:
            fig = ax.get_figure()

        # Manual hierarchical layout: x is determined by the BFS index within
        # the node's depth level, so lower-index (left) children always sit to
        # the left of their right sibling. This is necessary because
        # graphviz_layout does not guarantee semantic left/right ordering.
        pos = {}
        for node_id, bfs_idx in bfs_index_of.items():
            depth = self.node_depth(bfs_idx)
            pos_in_level = bfs_idx - (2**depth - 1)
            slot_width = 2 ** (self.max_depth - depth)
            x = (pos_in_level + 0.5) * slot_width
            y = -depth
            pos[node_id] = (x, y)

        node_list = list(G.nodes)
        color_list = [node_colors[n] for n in node_list]

        nx.draw(
            G,
            pos,
            ax=ax,
            nodelist=node_list,
            labels=labels,
            with_labels=True,
            node_color=color_list,
            node_size=2500,
            font_size=8,
            font_weight="bold",
            arrows=True,
            arrowsize=15,
            edgecolors="black",
            linewidths=1.0,
        )
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax, font_size=7
        )

        ax.set_title("Decision Tree", fontsize=12)
        plt.tight_layout()
        return fig

    def readable2state(self, readable: str) -> Dict:
        """Converts a readable string back to a state."""
        # TODO: implement proper parsing
        raise NotImplementedError("readable2state not yet implemented for Tree")

    @staticmethod
    def equal(state_x, state_y) -> bool:
        """
        Checks whether two Tree states are equal, using approximate comparison
        for floating-point threshold values.

        The threshold rescale/unrescale roundtrip introduces IEEE 754 errors
        of order ~1e-16 per operation. When same-feature ancestor chains create
        tight threshold bounds (small spans), the error is amplified by 1/span
        and can reach ~1e-12 at depth 3 or ~1e-10 at deeper trees. This override
        uses atol=1e-8 to absorb these errors with margin, while remaining far
        below any meaningful threshold difference (>1e-4 in practice).
        """
        return GFlowNetEnv.isclose(state_x, state_y, rtol=0.0, atol=1e-8)

    def is_source(self, state: Optional[Dict] = None) -> bool:
        """Returns True if the state is the source (empty tree, idle)."""
        state = self._get_state(state)
        if state["_active"] != -1:
            return False
        if any(d == 1 for d in state["_dones"]):
            return False
        for k in range(self.max_nodes):
            if k in state:
                return False
        return True

    # =========================================================================
    # State management
    # =========================================================================

    def set_state(self, state: Dict, done: Optional[bool] = False):
        """
        Sets the environment state and done flag.

        Bypasses CompositeBase's subenv iteration since tree nodes are dynamic.
        """
        GFlowNetEnv.set_state(self, state, done)
        return self

    def reset(self, env_id: Union[int, str] = None):
        """Resets the tree to the empty source state."""
        self.node_env.reset()
        self.state = copy(self.source)
        self.done = False
        self.n_actions = 0
        self.id = str(uuid.uuid4()) if env_id is None else env_id
        return self

    # =========================================================================
    # Policy output structure
    # =========================================================================

    def get_policy_output(
        self, params: Optional[list] = None
    ) -> TensorType["policy_output_dim"]:
        """
        Defines the structure of the policy output.

        The output is: [node_env_policy_output | meta_action_logits].
        Meta-action logits cover ``max_nodes`` toggle actions + 1 EOS.
        """
        if params is not None and len(params) > 0:
            node_po = self.node_env.get_policy_output(params[0])
        else:
            node_po = self.node_env.get_policy_output(None)
        meta_po = torch.zeros(self.n_meta_actions)
        return torch.cat([node_po, meta_po])

    def _get_policy_outputs_for_meta(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
    ) -> TensorType["n_states", "n_meta_actions"]:
        """Extracts meta-action policy outputs (last ``n_meta_actions`` columns)."""
        return policy_outputs[:, -self.n_meta_actions :]

    # =========================================================================
    # TODO: Batch sampling and log-probabilities
    # =========================================================================

    def sample_actions_batch(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        mask: Optional[TensorType["n_states", "mask_dim"]] = None,
        states_from: List[Dict] = None, # states originating the actions
        is_backward: Optional[bool] = False,
        random_action_prob: Optional[float] = 0.0,
        temperature_logits: Optional[float] = 1.0,
    ) -> List[Tuple]:
        """
        Samples a batch of actions from policy outputs with n_states = batch size.

        Routes each state to either meta-action sampling (categorical over
        toggles + EOS) or node env sampling (delegated to node_env).
        """
        n_states = policy_outputs.shape[0]

        # Determine mode: building iff node section has any valid (False) entry
        is_building = ~mask[:, self.n_meta_actions :].all(dim=1) # is_building has shape [n_states]
        actions = [None] * n_states

        # Building-mode states: delegate to node env
        building_idx = torch.where(is_building)[0]
        if len(building_idx) > 0:
            building_po = self._get_policy_outputs_of_env_unique( # Shape [n_building, node_env.policy_output_dim]
                policy_outputs[building_idx], 0 # Shape [n_building, policy_output_dim]
            )
            subenv_masks = self._unformat_mask_building(mask[building_idx])

            substates = []
            for i in building_idx.tolist():
                active = states_from[i]["_active"]
                substates.append(states_from[i][active]) # extract each states active substate

            subenv_actions = self.node_env.sample_actions_batch( # Pass to subenv, which returns subenv actions
                building_po,
                subenv_masks,
                substates,
                is_backward,
                random_action_prob,
                temperature_logits,
            )
            for j, i in enumerate(building_idx.tolist()):
                actions[i] = self._pad_action(subenv_actions[j], 0) # fill in subenv part of actions list

        # Meta-action states: categorical sampling
        meta_idx = torch.where(~is_building)[0] # Meta mode if it is not building
        if len(meta_idx) > 0:
            meta_mask = self._unformat_mask_meta(mask[meta_idx]) # Get masks

            # States where no meta-action is valid (e.g. the tree is already done)
            # cannot be sampled from: softmax over an all -inf logits vector yields
            # NaNs and crashes multinomial. For these, return a dummy EOS action;
            # step() / step_backwards() will reject it as invalid (valid=False), so
            # the caller sees the same behavior as the node env when called on a
            # completed state.
            all_invalid = meta_mask.all(dim=1) # True if there is no valid action remaining (tree is done)
            for j in torch.where(all_invalid)[0].tolist():
                actions[meta_idx[j].item()] = self.eos # Dummy EOS action that will have no impact in step

            sample_idx = meta_idx[~all_invalid] # States where there are possible meta actions to perform
            if len(sample_idx) > 0:
                meta_po = self._get_policy_outputs_for_meta(policy_outputs[sample_idx])
                meta_mask = meta_mask[~all_invalid]

                logits = meta_po.clone()
                logits[meta_mask] = -float("inf") # Set all invalid actions to -inf
                logits = logits / temperature_logits

                # Random action injection
                if random_action_prob > 0.0:
                    do_random = (
                        torch.bernoulli(
                            torch.full(
                                (len(sample_idx),),
                                random_action_prob,
                                device=self.device,
                            )
                        )
                        .bool()
                        .to(self.device)
                    )
                    if do_random.any():
                        uniform = torch.zeros_like(logits[do_random])
                        uniform[meta_mask[do_random]] = -float("inf")
                        logits[do_random] = uniform

                probs = torch.softmax(logits, dim=1)
                sampled = torch.multinomial(probs, 1).squeeze(1)

                for j, i in enumerate(sample_idx.tolist()):
                    k = sampled[j].item()
                    if k == self.max_nodes:
                        actions[i] = self.eos
                    else:
                        actions[i] = self._pad_action((k,), -1)

        return actions

    def get_logprobs(
        self,
        policy_outputs: TensorType["n_states", "policy_output_dim"],
        actions: Union[List, TensorType["n_states", "action_dim"]],
        mask: TensorType["n_states", "mask_dim"],
        states_from: List[Dict],
        is_backward: bool,
    ) -> TensorType["batch_size"]:
        """
        Computes log probabilities of actions given policy outputs.

        Routes each state to node env or meta-action log-prob computation.
        """
        actions = tfloat(actions, float_type=self.float, device=self.device)
        n_states = policy_outputs.shape[0]

        # Determine mode: building iff node section has any valid (False) entry
        is_building = ~mask[:, self.n_meta_actions :].all(dim=1)
        logprobs = torch.empty(n_states, dtype=self.float, device=self.device)

        # --- Building mode ---
        building_idx = torch.where(is_building)[0]
        if len(building_idx) > 0:
            building_po = self._get_policy_outputs_of_env_unique(
                policy_outputs[building_idx], 0
            )
            subenv_masks = self._unformat_mask_building(mask[building_idx])
            substates = []
            for i in building_idx.tolist():
                active = states_from[i]["_active"]
                substates.append(states_from[i][active])

            subenv_actions = self._depad_action_batch(actions[building_idx], 0)

            logprobs[building_idx] = self.node_env.get_logprobs(
                building_po, subenv_actions, subenv_masks, substates, is_backward
            )

        # --- Meta-action mode ---
        meta_idx = torch.where(~is_building)[0]
        if len(meta_idx) > 0:
            meta_po = self._get_policy_outputs_for_meta(policy_outputs[meta_idx])
            meta_mask = self._unformat_mask_meta(mask[meta_idx])

            logits = meta_po.clone()
            logits[meta_mask] = -float("inf")
            log_probs = torch.log_softmax(logits, dim=1)

            # Extract the action index: second element of the action tuple
            action_indices = actions[meta_idx, 1].long()
            # Map -1 (EOS target) to the last position (max_nodes)
            action_indices = action_indices.clone()
            action_indices[action_indices == -1] = self.max_nodes

            logprobs[meta_idx] = log_probs.gather(
                1, action_indices.unsqueeze(1)
            ).squeeze(1)

        return logprobs

    # =========================================================================
    # TODO: Action utilities
    # =========================================================================

    def action2representative(self, action: Tuple) -> Tuple:
        """
        Replaces the subenv part of a node env action by its representative.
        Meta-actions are returned as-is.
        """
        idx_unique = action[0]
        if idx_unique == -1:
            return action
        action_subenv = self._depad_action(action, idx_unique)
        representative = self.node_env.action2representative(action_subenv)
        return self._pad_action(representative, idx_unique)
