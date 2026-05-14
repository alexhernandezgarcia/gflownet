from copy import copy
from pathlib import Path

import common
import numpy as np
import pytest
import torch

from gflownet.envs.tree.tree import Tree

# Path to the test dataset
IRIS_CSV_PATH = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "tree" / "iris.csv"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env_tree_depth1():
    features = ["feat_a", "feat_b", "feat_c"]
    return Tree(max_depth=1, node_kwargs={"features": features})


@pytest.fixture
def env_tree_depth2():
    features = ["feat_a", "feat_b", "feat_c"]
    return Tree(max_depth=2, node_kwargs={"features": features})


@pytest.fixture
def env_tree_depth3():
    features = ["feat_a", "feat_b", "feat_c"]
    return Tree(max_depth=3, node_kwargs={"features": features})


@pytest.fixture
def env_tree_depth10():
    features = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e", "feat_f", "feat_g"]
    return Tree(max_depth=10, node_kwargs={"features": features})


@pytest.fixture
def env_tree_depth4_data_path():
    """Tree initialized from a CSV data path (iris dataset)."""
    return Tree(max_depth=4, data_path=IRIS_CSV_PATH)


@pytest.fixture
def env_tree_depth4_xy():
    """Tree initialized from X_train/y_train numpy arrays."""
    rng = np.random.default_rng(42)
    X_train = rng.random((80, 4))
    y_train = rng.integers(0, 3, size=80)
    X_test = rng.random((20, 4))
    y_test = rng.integers(0, 3, size=20)
    return Tree(
        max_depth=4, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )


# ---------------------------------------------------------------------------
# Discrete-node fixtures (DecisionTreeNodeDiscrete: Choice + 1D Grid)
# ---------------------------------------------------------------------------


@pytest.fixture
def env_tree_discrete_depth1():
    features = ["feat_a", "feat_b", "feat_c"]
    return Tree(
        max_depth=1,
        node_type="discrete",
        node_kwargs={"features": features, "n_thresholds": 5},
    )


@pytest.fixture
def env_tree_discrete_depth2():
    features = ["feat_a", "feat_b", "feat_c"]
    return Tree(
        max_depth=2,
        node_type="discrete",
        node_kwargs={"features": features, "n_thresholds": 5},
    )


@pytest.fixture
def env_tree_discrete_depth3():
    features = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
    return Tree(
        max_depth=3,
        node_type="discrete",
        node_kwargs={"features": features, "n_thresholds": 99},
    )


# ===========================================================================
# Initialization tests
# ===========================================================================

parametrize_envs = pytest.mark.parametrize(
    "envs",
    [
        "env_tree_depth1",
        "env_tree_depth2",
        "env_tree_depth3",
        "env_tree_depth10",
    ],
)

parametrize_envs_bigger_than_depth_1 = pytest.mark.parametrize(
    "envs",
    [
        "env_tree_depth2",
        "env_tree_depth3",
        "env_tree_depth10",
    ],
)


@parametrize_envs
def test__environment__initializes_properly(envs, request):
    env = request.getfixturevalue(envs)
    assert True


def test__environment__raises_on_depth0():
    features = ["feat_a", "feat_b", "feat_c"]
    with pytest.raises(ValueError, match="max_depth >= 1"):
        Tree(max_depth=0, node_kwargs={"features": features})


@parametrize_envs
def test__environment__is_continous(envs, request):
    env = request.getfixturevalue(envs)
    assert env.continuous is True


@pytest.mark.parametrize(
    "envs, max_depth, max_nodes",
    [
        ("env_tree_depth1", 1, 1),
        ("env_tree_depth2", 2, 3),
        ("env_tree_depth3", 3, 7),
        ("env_tree_depth10", 10, 1023),
    ],
)
def test__max_nodes__is_correct(envs, max_depth, max_nodes, request):
    env = request.getfixturevalue(envs)
    assert env.max_depth == max_depth
    assert env.max_nodes == max_nodes
    assert env.max_elements == max_nodes


# ---------------------------------------------------------------------------
# Helper to build a node via a sequence of forward steps
# ---------------------------------------------------------------------------


# TODO: Tests that are based on this function, only work if tree is used with the initial node.py subenv
def _build_node_with_dtnode_subenv(tree, node_idx, feature_idx, threshold_val):
    """
    Builds a complete node in the tree via forward actions, assuming that subenvs are of type gflownet.envs.node.py :
    activate -> choose feature -> Choice EOS -> set threshold -> Cube EOS -> deactivate.

    Parameters
    ----------
    tree : Tree
        The tree environment.
    node_idx : int
        The node position (breadth-first index) to build.
    feature_idx : int
        The feature index to select (as used by Choice).
    threshold_val : float
        The threshold value to set in [0, 1].
    """
    # Activate
    s, _, v = tree.step(tree._pad_action((node_idx,), -1))
    assert v, f"Failed to activate node {node_idx}"
    # Choose feature
    s, _, v = tree.step((0, 0, feature_idx, 0))
    assert v, f"Failed to choose feature {feature_idx} for node {node_idx}"
    # Choice EOS (transition to threshold stage)
    s, _, v = tree.step((0, 0, -1, 0))
    assert v, f"Failed Choice EOS for node {node_idx}"
    # Set threshold in a series of steps
    # action[-1]==1 means "initialize from source"; only valid for the first step
    s, _, v = tree.step((0, 1, 2 * threshold_val / 5, 1))
    s, _, v = tree.step((0, 1, 2 * threshold_val / 5, 0))
    s, _, v = tree.step((0, 1, threshold_val / 5, 0))
    assert v, f"Failed to set threshold for node {node_idx}"
    # ContinuousCube EOS
    s, _, v = tree.step((0, 1, float("inf"), float("inf")))
    assert v, f"Failed Cube EOS for node {node_idx}"
    # Deactivate
    s, _, v = tree.step(tree._pad_action((node_idx,), -1))
    assert v, f"Failed to deactivate node {node_idx}"


# TODO: Tests that are based on this function, only work if tree is used with the initial node.py subenv
def _unbuild_node_with_dtnode_subenv(tree, node_idx, feature_idx, threshold_val):
    """
    Reverses _build_node_with_dtnode_subenv: performs every backward action on an
    already-built node, assuming subenvs are of type gflownet.envs.node.py:
    reactivate -> backward Cube EOS -> backward threshold steps -> backward Choice EOS
    -> backward choose feature -> backward activate (removes the node).

    Parameters
    ----------
    tree : Tree
        The tree environment.
    node_idx : int
        The node position (breadth-first index) to unbuild.
    feature_idx : int
        The feature index that was selected when the node was built.
    threshold_val : float
        The threshold value that was set when the node was built.
    """
    # Reactivate
    s, _, v = tree.step_backwards(tree._pad_action((node_idx,), -1))
    assert v, f"Failed to reactivate node {node_idx}"
    # Backward Cube EOS (clears done flag; unrescales threshold)
    s, _, v = tree.step_backwards((0, 1, float("inf"), float("inf")))
    assert v, f"Failed backward Cube EOS for node {node_idx}"
    # Backward threshold steps (reverse order). The first two are regular
    # decrements (action[-1]==0). The third (undoing the forward "from source"
    # step) must use the Back-To-Source action (action[-1]==1), which jumps
    # directly to the cube's source state [-1] regardless of the increment.
    s, _, v = tree.step_backwards((0, 1, threshold_val / 5, 0))
    s, _, v = tree.step_backwards((0, 1, 2 * threshold_val / 5, 0))
    s, _, v = tree.step_backwards((0, 1, 0.0, 1))
    assert v, f"Failed backward threshold steps for node {node_idx}"
    # Backward Choice EOS
    s, _, v = tree.step_backwards((0, 0, -1, 0))
    assert v, f"Failed backward Choice EOS for node {node_idx}"
    # Backward choose feature
    s, _, v = tree.step_backwards((0, 0, feature_idx, 0))
    assert v, f"Failed backward choose feature for node {node_idx}"
    # Backward activate (removes node)
    s, _, v = tree.step_backwards(tree._pad_action((node_idx,), -1))
    assert v, f"Failed backward activate (removal) for node {node_idx}"


def _build_discrete_node(tree, node_idx, feature_idx, cell_idx):
    """
    Builds a complete node in a discrete-node tree, advancing the Grid
    sub-environment from cell 0 to ``cell_idx`` via increment-by-1 actions and
    terminating with the Grid EOS.

    Parameters
    ----------
    tree : Tree
        The tree environment with ``node_type='discrete'``.
    node_idx : int
        The node position (breadth-first index) to build.
    feature_idx : int
        The feature index to select (1-based as used by Choice).
    cell_idx : int
        The threshold cell index in the Grid sub-environment.
    """
    # Activate
    s, _, v = tree.step(tree._pad_action((node_idx,), -1))
    assert v, f"Failed to activate node {node_idx}"
    # Choose feature
    s, _, v = tree.step((0, 0, feature_idx, 0))
    assert v, f"Failed to choose feature {feature_idx} for node {node_idx}"
    # Choice EOS (transition to threshold stage)
    s, _, v = tree.step((0, 0, -1, 0))
    assert v, f"Failed Choice EOS for node {node_idx}"
    # Advance the Grid from cell 0 to cell_idx by +1 increments
    for _ in range(cell_idx):
        s, _, v = tree.step((0, 1, 1, 0))
        assert v, f"Failed Grid increment for node {node_idx} (target cell {cell_idx})"
    # Grid EOS
    s, _, v = tree.step((0, 1, 0, 0))
    assert v, f"Failed Grid EOS for node {node_idx}"
    # Deactivate
    s, _, v = tree.step(tree._pad_action((node_idx,), -1))
    assert v, f"Failed to deactivate node {node_idx}"


def _unbuild_discrete_node(tree, node_idx, feature_idx, cell_idx):
    """
    Reverses :py:func:`_build_discrete_node`.
    """
    # Reactivate
    s, _, v = tree.step_backwards(tree._pad_action((node_idx,), -1))
    assert v, f"Failed to reactivate node {node_idx}"
    # Backward Grid EOS
    s, _, v = tree.step_backwards((0, 1, 0, 0))
    assert v, f"Failed backward Grid EOS for node {node_idx}"
    # Backward Grid increments
    for _ in range(cell_idx):
        s, _, v = tree.step_backwards((0, 1, 1, 0))
        assert v, f"Failed backward Grid increment for node {node_idx}"
    # Backward Choice EOS
    s, _, v = tree.step_backwards((0, 0, -1, 0))
    assert v, f"Failed backward Choice EOS for node {node_idx}"
    # Backward choose feature
    s, _, v = tree.step_backwards((0, 0, feature_idx, 0))
    assert v, f"Failed backward choose feature for node {node_idx}"
    # Backward activate (removes node)
    s, _, v = tree.step_backwards(tree._pad_action((node_idx,), -1))
    assert v, f"Failed backward activate (removal) for node {node_idx}"


# ===========================================================================
# Source state tests
# ===========================================================================


@pytest.mark.parametrize(
    "env, expected_source",
    [
        (
            "env_tree_depth1",
            {"_active": -1, "_dones": [0], "_envs_unique": 1},
        ),
        (
            "env_tree_depth2",
            {"_active": -1, "_dones": [0, 0, 0], "_envs_unique": 1},
        ),
        (
            "env_tree_depth3",
            {"_active": -1, "_dones": [0, 0, 0, 0, 0, 0, 0], "_envs_unique": 1},
        ),
    ],
)
def test__source__is_expected(env, expected_source, request):
    env = request.getfixturevalue(env)
    assert env.source == expected_source
    assert env.state == expected_source


@parametrize_envs
def test__is_source__returns_true_at_init(envs, request):
    env = request.getfixturevalue(envs)
    assert env.is_source()


@parametrize_envs
def test__is_source__returns_true_after_reset(envs, request):
    env = request.getfixturevalue(envs)
    # Build a node and then reset
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    env.step(env.eos)
    assert not env.is_source()
    env.reset()
    assert env.is_source()


# ===========================================================================
# Tree structure helper tests
# ===========================================================================


def test__left_right_child__and_parent_idx():
    assert Tree.left_child_idx(0) == 1
    assert Tree.right_child_idx(0) == 2
    assert Tree.left_child_idx(1) == 3
    assert Tree.right_child_idx(1) == 4
    assert Tree.left_child_idx(2) == 5
    assert Tree.right_child_idx(2) == 6
    assert Tree.parent_idx(1) == 0
    assert Tree.parent_idx(2) == 0
    assert Tree.parent_idx(3) == 1
    assert Tree.parent_idx(4) == 1
    assert Tree.parent_idx(5) == 2
    assert Tree.parent_idx(6) == 2


def test__node_depth():
    assert Tree.node_depth(0) == 0
    assert Tree.node_depth(1) == 1
    assert Tree.node_depth(2) == 1
    assert Tree.node_depth(3) == 2
    assert Tree.node_depth(4) == 2
    assert Tree.node_depth(5) == 2
    assert Tree.node_depth(6) == 2


@parametrize_envs
def test__get_expandable_leaves__source_returns_only_root(envs, request):
    env = request.getfixturevalue(envs)
    assert env._get_expandable_leaves(env.state) == [0]


@parametrize_envs_bigger_than_depth_1
def test__get_expandable_leaves__after_root_built(envs, request):
    env = request.getfixturevalue(envs)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    assert env._get_expandable_leaves(env.state) == [1, 2]


def test__get_expandable_leaves__fully_built_tree(env_tree_depth1, env_tree_depth2):
    env = env_tree_depth1
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    # No more expandable leaves (depth 1 children are at indices 1-2, because max_nodes=1)
    assert env._get_expandable_leaves(env.state) == []
    env = env_tree_depth2
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    _build_node_with_dtnode_subenv(env, 1, 2, 0.3)
    _build_node_with_dtnode_subenv(env, 2, 3, 0.7)
    # No more expandable leaves (depth 2 children are at indices 3-6, because max_nodes=3)
    assert env._get_expandable_leaves(env.state) == []


@parametrize_envs_bigger_than_depth_1
def test__get_leaf_done_nodes__correctly_identifies_leaves(envs, request):
    # Leaf done nodes are the ones that can be legally eliminated in the backwards pass
    env = request.getfixturevalue(envs)

    # None at source
    assert env._get_leaf_done_nodes(env.state) == []

    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    # Root has no children yet: it is a leaf done node
    assert env._get_leaf_done_nodes(env.state) == [0]

    _build_node_with_dtnode_subenv(env, 1, 2, 0.3)
    # Root has child node 1 which is an internal decision node,
    # so root is NOT a leaf done node; only node 1 is
    assert env._get_leaf_done_nodes(env.state) == [1]

    _build_node_with_dtnode_subenv(env, 2, 3, 0.7)
    # Both nodes 1 and 2 are leaf done nodes; root is not
    assert env._get_leaf_done_nodes(env.state) == [1, 2]


# ===========================================================================
# Action space tests
# ===========================================================================


@parametrize_envs
def test__action_space__contains_toggle_eos_and_node_actions(envs, request):
    env = request.getfixturevalue(envs)
    action_space = env.action_space

    # Toggle actions: one per node position
    for k in range(env.max_nodes):
        assert env._pad_action((k,), -1) in action_space

    # EOS
    assert env.eos in action_space

    # Node env actions
    for a in env.node_env.action_space:
        assert env._pad_action(a, 0) in action_space


@parametrize_envs
def test__action_space__size(envs, request):
    env = request.getfixturevalue(envs)
    # Total count: max_nodes toggles + 1 EOS + node_env actions
    assert len(env.action_space) == env.max_nodes + 1 + len(env.node_env.action_space)
    assert len(env.action_space) - len(env.node_env.action_space) == 2**env.max_depth


# ===========================================================================
# Pad / depad action tests
# ===========================================================================


@pytest.mark.parametrize(
    "meta_action, only_action, idx_unique",
    [
        # Toggle meta-action for node 0
        ((-1, 0, 0, 0), (0,), -1),
        # Toggle meta-action for node 2
        ((-1, 2, 0, 0), (2,), -1),
        # EOS
        ((-1, -1, 0, 0), (-1,), -1),
    ],
)
@parametrize_envs
def test__pad_depad_action__meta_actions(
    envs, request, meta_action, only_action, idx_unique
):
    env = request.getfixturevalue(envs)
    assert env._pad_action(only_action, idx_unique) == meta_action
    assert env._depad_action(meta_action, idx_unique) == only_action


@pytest.mark.parametrize(
    "action_on_tree, action_subenv, idx_unique",
    [
        # Subenv select feature 2
        ((0, 0, 2, 0), (0, 2, 0), 0),
        # Subenv select threshold 0.5 for node 2
        ((0, 1, 0.5, 0), (1, 0.5, 0), 0),
        # subenv EOS
        ((0, 1, float("inf"), float("inf")), (1, float("inf"), float("inf")), 0),
    ],
)
@parametrize_envs
def test__pad_depad_action__subenv_actions(
    envs, request, action_on_tree, action_subenv, idx_unique
):
    env = request.getfixturevalue(envs)
    assert env._pad_action(action_subenv, idx_unique) == action_on_tree
    assert env._depad_action(action_on_tree, idx_unique) == action_subenv


@parametrize_envs
def test__pad_depad_action__consistent_for_random_actions(envs, request):
    env = request.getfixturevalue(envs)
    for i in range(5):
        _, action, action_successful = env.step_random()
        if not action_successful:
            continue
        assert action == env._pad_action(env._depad_action(action), action[0])


# ===========================================================================
# Forward mask tests
# ===========================================================================


@parametrize_envs
def test__mask_dim__is_correct(envs, request):
    env = request.getfixturevalue(envs)
    expected = env.max_nodes + 1 + env.node_env.mask_dim
    assert env.mask_dim == expected


@parametrize_envs
def test__forward_mask__source_only_toggle_root_valid(envs, request):
    """At source, only the toggle for root (node 0) should be valid."""
    env = request.getfixturevalue(envs)
    mask = env.get_mask_invalid_actions_forward()
    valid = env.get_valid_actions()
    # Only toggle for node 0 should be valid
    toggle_0 = env._pad_action((0,), -1)
    assert valid == [toggle_0]
    assert mask == [False] + [True] * (env.mask_dim - 1)
    # EOS invalid (no done nodes)
    assert env.eos not in valid


@parametrize_envs_bigger_than_depth_1
def test__forward_mask__idle_after_root_done(envs, request):
    """After building root, expand leaves 1/2 and EOS should all be valid."""
    env = request.getfixturevalue(envs)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    valid = env.get_valid_actions()
    toggle_1 = env._pad_action((1,), -1)
    toggle_2 = env._pad_action((2,), -1)
    assert toggle_1 in valid
    assert toggle_2 in valid
    assert env.eos in valid
    # Toggle 0 invalid (root already done, not expandable)
    toggle_0 = env._pad_action((0,), -1)
    assert toggle_0 not in valid
    mask = env.get_mask_invalid_actions_forward()
    expected_mask = (
        [True, False, False]
        + [True] * (env.mask_dim - 4 - env.node_env.mask_dim)
        + [False]
        + [True] * (env.node_env.mask_dim)
    )
    assert mask == expected_mask


@parametrize_envs
def test__forward_mask__building_mode_delegates_to_node_env(envs, request):
    """While building a node, mask should be in building mode."""
    env = request.getfixturevalue(envs)
    # Activate root
    env.step(env._pad_action((0,), -1))
    mask = env.get_mask_invalid_actions_forward()
    # All meta actions should be masked out, so True
    assert all(mask[: env.n_meta_actions])
    # Node section should equal node_env's own mask for the active substate
    active = env.state["_active"]
    substate = env.state[active]
    expected_node_mask = env.node_env.get_mask_invalid_actions_forward(substate, False)
    assert mask[env.n_meta_actions :] == expected_node_mask
    # Valid actions should be node env actions
    valid = env.get_valid_actions()
    for a in valid:
        assert env._depad_action(a) in env.node_env.action_space


@parametrize_envs
def test__forward_mask__node_done_only_deactivation_valid(envs, request):
    """When a node is done but still active, only the deactivation toggle is valid."""
    env = request.getfixturevalue(envs)
    # Activate and fully build node 0, without deactivating
    env.step(env._pad_action((0,), -1))
    env.step((0, 0, 1, 0))  # Choose feature
    env.step((0, 0, -1, 0))  # Choice EOS
    env.step((0, 1, 0.5, 1))  # Set threshold
    env.step((0, 1, float("inf"), float("inf")))  # Cube EOS
    # Node 0 is now done but still active
    assert env._node_is_done(0, env.state)
    assert env.state["_active"] == 0
    valid = env.get_valid_actions()
    assert len(valid) == 1
    assert valid[0] == env._pad_action((0,), -1)  # Deactivation toggle


@parametrize_envs
def test__forward_mask__done_state_all_invalid(envs, request):
    """When the tree is done, all actions are invalid."""
    env = request.getfixturevalue(envs)
    env.trajectory_random()
    assert env.done
    mask = env.get_mask_invalid_actions_forward()
    assert all(mask)


@parametrize_envs
def test__forward_mask__building_in_progress_delegates_to_node_env(envs, request):
    """While building a node past its source state (at least one subenv action
    taken, not yet done), forward mask delegates to the node_env mask."""
    env = request.getfixturevalue(envs)
    env.step(env._pad_action((0,), -1))  # Activate root
    # Take one valid forward subenv action to move past source
    valid_fwd = env.get_valid_actions()
    env.step(valid_fwd[0])
    # Precondition: node 0 is in progress (past source, not done)
    assert not env.node_env.is_source(env.state[0])
    assert not env._node_is_done(0, env.state)

    mask = env.get_mask_invalid_actions_forward()
    # All meta actions invalid (because in building mode)
    assert all(mask[: env.n_meta_actions])
    # Node section should match node_env forward mask for the substate
    substate = env.state[0]
    expected = env.node_env.get_mask_invalid_actions_forward(substate, False)
    assert mask[env.n_meta_actions :] == expected
    # Valid forward actions should be node_env actions (when depadded)
    valid = env.get_valid_actions()
    assert len(valid) > 0
    for a in valid:
        assert env._depad_action(a) in env.node_env.action_space


# ===========================================================================
# Backward mask tests
# ===========================================================================


@parametrize_envs
def test__backward_mask__done_only_eos_valid(envs, request):
    """From done, only the EOS backward action is valid."""
    env = request.getfixturevalue(envs)
    env.trajectory_random()
    assert env.done
    valid = env.get_valid_actions(backward=True)
    assert len(valid) == 1
    assert valid[0] == env.eos


@parametrize_envs
def test__backward_mask__source_no_valid_actions(envs, request):
    """At source, no backward actions are valid."""
    env = request.getfixturevalue(envs)
    mask = env.get_mask_invalid_actions_backward()
    assert all(mask)
    assert env.get_valid_actions(backward=True) == []


@parametrize_envs_bigger_than_depth_1
def test__backward_mask__idle_can_reactivate_leaf_done_nodes(envs, request):
    """From idle with done nodes, can reactivate leaf done nodes only."""
    env = request.getfixturevalue(envs)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    _build_node_with_dtnode_subenv(env, 1, 2, 0.3)
    # Idle with nodes 0 and 1 done. Node 1 is a leaf done node, node 0 is not
    # (it has child 1).
    valid = env.get_valid_actions(backward=True)
    toggle_1 = env._pad_action((1,), -1)
    toggle_0 = env._pad_action((0,), -1)
    assert toggle_1 in valid
    assert toggle_0 not in valid  # Node 0 has child 1


@parametrize_envs
def test__backward_mask__building_node_at_source_toggle_valid(envs, request):
    """When building a node that is still at source, backward = undo activation."""
    env = request.getfixturevalue(envs)
    env.step(env._pad_action((0,), -1))  # Activate root
    # Node 0 is at source state
    assert env.node_env.is_source(env.state[0])
    valid = env.get_valid_actions(backward=True)
    assert len(valid) == 1
    assert valid[0] == env._pad_action((0,), -1)  # Undo activation


@parametrize_envs
def test__backward_mask__building_in_progress_delegates_to_node_env(envs, request):
    """While building a node past its source state (at least one subenv action
    taken, not yet done), backward mask delegates to the node_env mask."""
    env = request.getfixturevalue(envs)
    env.step(env._pad_action((0,), -1))  # Activate root
    # Take one valid forward subenv action to move past source
    valid_fwd = env.get_valid_actions()
    env.step(valid_fwd[0])
    # Precondition: node 0 is in progress (past source, not done)
    assert not env.node_env.is_source(env.state[0])
    assert not env._node_is_done(0, env.state)

    mask = env.get_mask_invalid_actions_backward()
    # All meta actions invalid (we're in building mode)
    assert all(mask[: env.n_meta_actions])
    # Node section should match node_env backward mask for the substate
    substate = env.state[0]
    expected = env.node_env.get_mask_invalid_actions_backward(substate, False)
    assert mask[env.n_meta_actions :] == expected
    # Valid backward actions should be node_env actions (when depadded)
    valid = env.get_valid_actions(backward=True)
    assert len(valid) > 0
    for a in valid:
        assert env._depad_action(a) in env.node_env.action_space


# ===========================================================================
# get_mask is consistent when called with state argument vs self.state
# ===========================================================================


@pytest.mark.repeat(5)
@parametrize_envs
def test__get_mask__is_consistent_regardless_of_inputs(envs, request):
    """
    Mask computed with state passed as explicit arg should equal mask from self.state.
    Tests both forward and backward masks.
    """
    env = request.getfixturevalue(envs)
    env.reset()
    env.trajectory_random()
    # Sample a few backward steps to get to an intermediate state
    if env.done:
        env.step_backwards(env.eos)
    for _ in range(3):
        va = env.get_valid_actions(backward=True)
        if va:
            env.step_backwards(va[0])

    state = copy(env.state)
    done = env.done

    mask_f_self = env.get_mask_invalid_actions_forward()
    mask_f_arg = env.get_mask_invalid_actions_forward(state, done)
    assert mask_f_self == mask_f_arg

    mask_b_self = env.get_mask_invalid_actions_backward()
    mask_b_arg = env.get_mask_invalid_actions_backward(state, done)
    assert mask_b_self == mask_b_arg


# ===========================================================================
# get_valid_actions is consistent regardless of inputs
# ===========================================================================


@pytest.mark.repeat(5)
@parametrize_envs
def test__get_valid_actions__is_consistent_regardless_of_inputs(envs, request):
    """
    get_valid_actions should return the same result whether called with a
    precomputed mask or without it.
    """
    env = request.getfixturevalue(envs)
    env.reset()
    # Do a random forward trajectory, then some backward steps
    env.trajectory_random()
    if env.done:
        env.step_backwards(env.eos)
    for _ in range(3):
        va = env.get_valid_actions(backward=True)
        if not va:
            break
        env.step_backwards(va[0])

    state = copy(env.state)
    done = env.done

    # Forward
    mask_f = env.get_mask_invalid_actions_forward(state, done)
    valid_f_mask = env.get_valid_actions(mask=mask_f, state=state, done=done)
    valid_f_none = env.get_valid_actions(state=state, done=done)
    assert valid_f_mask == valid_f_none

    # Backward
    mask_b = env.get_mask_invalid_actions_backward(state, done)
    valid_b_mask = env.get_valid_actions(
        mask=mask_b, state=state, done=done, backward=True
    )
    valid_b_none = env.get_valid_actions(state=state, done=done, backward=True)
    assert valid_b_mask == valid_b_none


# ===========================================================================
# Forward step tests
# ===========================================================================


@parametrize_envs_bigger_than_depth_1
def test__step__activate_node_creates_substate(envs, request):
    """Activating a node creates a new subenv in source state."""
    env = request.getfixturevalue(envs)
    toggle = env._pad_action((0,), -1)
    state, _, valid = env.step(toggle)
    assert valid
    assert state["_active"] == 0
    assert 0 in state
    assert env.node_env.is_source(state[0])


@parametrize_envs_bigger_than_depth_1
def test__step__deactivate_node_goes_idle(envs, request):
    """Deactivating a done node sets active to -1."""
    env = request.getfixturevalue(envs)
    # Build and complete node 0 (without deactivation)
    env.step(env._pad_action((0,), -1))  # Toggle
    env.step((0, 0, 1, 0))  # Choice
    env.step((0, 0, -1, 0))  # EOS choice
    env.step((0, 1, 0.5, 1))  # Set threshold
    env.step((0, 1, float("inf"), float("inf")))  # EOS node env
    assert env.state["_active"] == 0
    assert env._node_is_done(0, env.state)
    # Deactivate
    state, _, valid = env.step(env._pad_action((0,), -1))
    assert valid
    assert state["_active"] == -1


@parametrize_envs
def test__step__eos_sets_done(envs, request):
    """EOS is valid when idle with root done and sets done=True."""
    env = request.getfixturevalue(envs)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    _, _, valid = env.step(env.eos)
    assert valid
    assert env.done


@parametrize_envs
def test__step__eos_invalid_without_root_done(envs, request):
    """EOS should be invalid if the root is not done."""
    env = request.getfixturevalue(envs)
    _, _, valid = env.step(env.eos)
    assert not valid
    assert not env.done


@parametrize_envs
def test__step__eos_invalid_when_not_idle(envs, request):
    """EOS should be invalid when a node is active."""
    env = request.getfixturevalue(envs)
    env.step(env._pad_action((0,), -1))  # Activate root
    _, _, valid = env.step(env.eos)
    assert not valid


@parametrize_envs_bigger_than_depth_1
def test__step__invalid_toggle_target(envs, request):
    """Toggling a non-expandable position should be invalid."""
    env = request.getfixturevalue(envs)
    # Try to activate node 1 without root being done
    _, _, valid = env.step(env._pad_action((1,), -1))
    assert not valid
    # Try to activate node 2 without root being done
    _, _, valid = env.step(env._pad_action((2,), -1))
    assert not valid


@parametrize_envs
def test__step__node_env_action_sets_done_flag(envs, request):
    """When the node env reaches done via an action, _dones flag is updated."""
    env = request.getfixturevalue(envs)
    env.step(env._pad_action((0,), -1))  # Toggle on root
    env.step((0, 0, 1, 0))  # Choice
    env.step((0, 0, -1, 0))  # EOS choice
    env.step((0, 1, 0.5, 1))  # Set threshold
    assert env.state["_dones"][0] == 0
    env.step((0, 1, float("inf"), float("inf")))  # EOS node env
    assert env.state["_dones"][0] == 1


@parametrize_envs
def test__step__node_env_action_invalid_when_idle(envs, request):
    """Node env actions should be invalid when idle."""
    env = request.getfixturevalue(envs)
    _, _, valid = env.step((0, 0, 1, 0))
    assert not valid
    _, _, valid = env.step((0, 0, -1, 0))
    assert not valid
    _, _, valid = env.step((0, 1, 0.5, 1))
    assert not valid
    _, _, valid = env.step((0, 1, float("inf"), float("inf")))
    assert not valid


@parametrize_envs
def test__step__returns_false_when_done(envs, request):
    """Any step after done should return valid=False."""
    env = request.getfixturevalue(envs)
    env.trajectory_random()
    assert env.done
    _, _, valid = env.step(env._pad_action((0,), -1))
    assert not valid


# ===========================================================================
# Backward step tests
# ===========================================================================


@parametrize_envs
def test__step_backwards__eos_backward(envs, request):
    """Backward EOS un-terminates the tree."""
    env = request.getfixturevalue(envs)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    env.step(env.eos)
    assert env.done
    state, _, valid = env.step_backwards(env.eos)
    assert valid
    assert not env.done
    assert env._is_idle(state)


@parametrize_envs
def test__step_backwards__reactivate_done_node(envs, request):
    """Backward toggle on idle reactivates a leaf done node."""
    env = request.getfixturevalue(envs)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    toggle_0 = env._pad_action((0,), -1)
    state, _, valid = env.step_backwards(toggle_0)
    assert valid
    assert state["_active"] == 0
    assert env._node_is_done(0, state)


@parametrize_envs_bigger_than_depth_1
def test__step_backwards__reactivate_blocked_for_non_leaf_done(envs, request):
    """Cannot reactivate a done node that has children."""
    env = request.getfixturevalue(envs)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    _build_node_with_dtnode_subenv(env, 1, 2, 0.3)
    toggle_0 = env._pad_action((0,), -1)  # Try to reactivate root
    _, _, valid = env.step_backwards(toggle_0)
    assert not valid


@parametrize_envs
def test__step_backwards__remove_node_at_source(envs, request):
    """Backward toggle on an active node at source removes it."""
    env = request.getfixturevalue(envs)
    env.step(env._pad_action((0,), -1))  # Activate root
    assert 0 in env.state
    toggle_0 = env._pad_action((0,), -1)
    state, _, valid = env.step_backwards(toggle_0)
    assert valid
    assert 0 not in state
    assert state["_active"] == -1
    assert env.is_source()


@parametrize_envs_bigger_than_depth_1
def test__step_backwards__remove_node_in_tree(envs, request):
    """Backward toggle on an active node removes it."""
    env = request.getfixturevalue(envs)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    assert 1 in env.state
    _build_node_with_dtnode_subenv(env, 1, 2, 0.3)
    assert 2 in env.state
    env.step_backwards(env._pad_action((1,), -1))  # Activate node 1
    env.step_baclwards


@parametrize_envs_bigger_than_depth_1
def test__step_backwards__remove_node_in_tree(envs, request):
    """The full backward trajectory of a built non-root node removes it from the tree."""
    env = request.getfixturevalue(envs)
    feature_node_0, threshold_node_0, feature_node_1, threshold_node_1 = 1, 0.5, 2, 0.3
    _build_node_with_dtnode_subenv(env, 0, feature_node_0, threshold_node_0)
    assert 0 in env.state
    _build_node_with_dtnode_subenv(env, 1, feature_node_1, threshold_node_1)
    assert 1 in env.state

    # Unbuild node 1 (leaf): full backward trajectory removes it from the tree
    _unbuild_node_with_dtnode_subenv(env, 1, feature_node_1, threshold_node_1)
    assert 1 not in env.state
    assert env.state["_active"] == -1

    # Node 0 is now a leaf done node; unbuild it too -> back to source
    _unbuild_node_with_dtnode_subenv(env, 0, feature_node_0, threshold_node_0)
    assert 0 not in env.state
    assert env.is_source()


@parametrize_envs
def test__step_backwards__node_env_eos_clears_done(envs, request):
    """Backward of node env EOS clears the done flag."""
    env = request.getfixturevalue(envs)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    # Reactivate node 0
    env.step_backwards(env._pad_action((0,), -1))
    assert env._node_is_done(0, env.state)
    # Backward step of Cube EOS
    state, _, valid = env.step_backwards((0, 1, float("inf"), float("inf")))
    assert valid
    assert not env._node_is_done(0, state)


@parametrize_envs
def test__step_backwards__eos_invalid_when_not_done(envs, request):
    """Backward EOS should be invalid if the env is not done."""
    env = request.getfixturevalue(envs)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    # Not done yet
    _, _, valid = env.step_backwards(env.eos)
    assert not valid


# ===========================================================================
# get_parents tests
# ===========================================================================


@parametrize_envs
def test__get_parents__source_has_no_parents(envs, request):
    """Source state should have no parents."""
    env = request.getfixturevalue(envs)
    parents, actions = env.get_parents()
    assert len(parents) == 0
    assert len(actions) == 0


@parametrize_envs
def test__get_parents__done_returns_self_and_eos(envs, request):
    """From env done, the only parent is the state itself and the EOS action."""
    env = request.getfixturevalue(envs)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    env.step(env.eos)
    parents, actions = env.get_parents()
    assert len(parents) == 1
    assert actions[0] == env.eos


@parametrize_envs_bigger_than_depth_1
def test__get_parents__idle_with_done_nodes(envs, request):
    """From idle with done nodes, parents are to reactivate done nodes."""
    env = request.getfixturevalue(envs)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    _build_node_with_dtnode_subenv(env, 1, 2, 0.3)
    _build_node_with_dtnode_subenv(env, 2, 3, 0.7)
    # Idle with nodes 0, 1, 2 done. Leaf done nodes are 1 and 2.
    parents, _ = env.get_parents()
    assert len(parents) == 2
    # Each parent should have _active set to a leaf done node
    parent_actives = {p["_active"] for p in parents}
    assert parent_actives == {1, 2}


@parametrize_envs
def test__get_parents__node_at_source(envs, request):
    """When building a node at source, the parent is the idle state without the node."""
    env = request.getfixturevalue(envs)
    env.step(env._pad_action((0,), -1))  # Toggle on root
    parents, actions = env.get_parents()
    assert len(parents) == 1
    assert parents[0]["_active"] == -1
    assert 0 not in parents[0]
    assert actions[0] == env._pad_action((0,), -1)


@parametrize_envs
def test__get_parents__stepping_from_parent_reaches_child(envs, request):
    """For each parent, stepping forward with the action should produce the state."""
    env = request.getfixturevalue(envs)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    state_saved = copy(env.state)
    parents, actions = env.get_parents(state_saved)
    for parent, action in zip(parents, actions):
        env_copy = env.copy()
        env_copy.set_state(copy(parent), done=False)
        state_next, _, valid = env_copy.step(action)
        assert valid
        # The states should match
        assert state_next["_active"] == state_saved["_active"]
        assert state_next["_dones"] == state_saved["_dones"]
        assert state_next["_envs_unique"] == state_saved["_envs_unique"]


# ===========================================================================
# Determined trajectory tests
# ===========================================================================


@parametrize_envs
def test__full_trajectory__root_only(envs, request):
    """Build root, EOS: single-node tree."""
    env = request.getfixturevalue(envs)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    state, _, valid = env.step(env.eos)
    assert valid and env.done
    assert env._node_is_done(0, state)
    # Backward to source
    while not env.is_source():
        if env.done:
            a = env.eos
        else:
            va = env.get_valid_actions(backward=True)
            a = va[0]
        _, _, v = env.step_backwards(a)
        assert v
    assert env.is_source()


def test__full_trajectory__full_depth2_tree(env_tree_depth2):
    """Build all 3 nodes (root + 2 children), EOS, then backward to source."""
    env = env_tree_depth2
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    _build_node_with_dtnode_subenv(env, 1, 2, 0.3)
    _build_node_with_dtnode_subenv(env, 2, 3, 0.7)
    _, _, valid = env.step(env.eos)
    assert valid and env.done
    # Full backward
    while not env.is_source():
        if env.done:
            a = env.eos
        else:
            va = env.get_valid_actions(backward=True)
            a = va[0]
        _, _, v = env.step_backwards(a)
        assert v
    assert env.is_source()


def test__full_trajectory__partial_tree(env_tree_depth2):
    """Build root + only left child (partial tree), EOS, backward to source."""
    env = env_tree_depth2
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    _build_node_with_dtnode_subenv(env, 1, 2, 0.3)
    _, _, valid = env.step(env.eos)
    assert valid and env.done
    # Full backward
    while not env.is_source():
        if env.done:
            a = env.eos
        else:
            va = env.get_valid_actions(backward=True)
            a = va[0]
        _, _, v = env.step_backwards(a)
        assert v
    assert env.is_source()


# ===========================================================================
# Random trajectory tests
# ===========================================================================


@pytest.mark.repeat(10)
@parametrize_envs
def test__step_random__does_not_crash_from_source(envs, request):
    env = request.getfixturevalue(envs)
    env.reset()
    env.step_random()


@pytest.mark.repeat(10)
@parametrize_envs
def test__trajectory_random__reaches_done(envs, request):
    env = request.getfixturevalue(envs)
    env.reset()
    env.trajectory_random()
    assert env.done
    assert env._is_idle(env.state)
    assert env._node_is_done(0, env.state)


@pytest.mark.repeat(5)
@parametrize_envs
def test__trajectory_random__forward_then_backward_reaches_source(envs, request):
    """Forward random trajectory followed by greedy backward should reach source."""
    env = request.getfixturevalue(envs)
    env.reset()
    env.trajectory_random()
    assert env.done
    # Backward
    step_count = 0
    while not env.is_source():
        if env.done:
            a = env.eos
        else:
            va = env.get_valid_actions(backward=True)
            assert len(va) > 0, (
                f"No backward actions at step {step_count}, "
                f"active={env.state['_active']}, dones={env.state['_dones']}"
            )
            a = va[0]
        _, _, v = env.step_backwards(a)
        assert v
        step_count += 1
        assert step_count <= env.max_traj_length + 1
    assert env.is_source()


# ===========================================================================
# Rescale/unrescale and trajectory reversibility diagnostic tests
#
# The Tree rescales node thresholds based on ancestor constraints:
#   rescaled = lower + raw * (upper - lower)
# and unrescales during backward steps:
#   recovered = (rescaled - lower) / (upper - lower)
#
# Due to IEEE 754 floating-point arithmetic, this roundtrip is NOT
# bit-exact when the span (upper - lower) is not a power of two.
# For example: 0.6 * 0.3 / 0.3 != 0.6 in floating point.
#
# This causes test__trajectories_are_reversible to fail intermittently
# when random trajectories create same-feature ancestor chains.
# ===========================================================================


def _state_diff_detail(env, state_a, state_b):
    """
    Returns a human-readable diff between two Tree states, highlighting
    threshold differences from floating-point rescale/unrescale errors.
    """
    diffs = []
    if state_a.get("_active") != state_b.get("_active"):
        diffs.append(f"  _active: {state_a['_active']} vs {state_b['_active']}")
    if state_a.get("_dones") != state_b.get("_dones"):
        diffs.append(f"  _dones: {state_a['_dones']} vs {state_b['_dones']}")
    for k in range(env.max_nodes):
        a_exists = env._node_exists(k, state_a)
        b_exists = env._node_exists(k, state_b)
        if a_exists != b_exists:
            diffs.append(f"  Node {k}: exists in A={a_exists}, in B={b_exists}")
        elif a_exists:
            sub_a, sub_b = state_a[k], state_b[k]
            if sub_a != sub_b:
                parts = [f"  Node {k} (depth {env.node_depth(k)}):"]
                f_a = env.node_env.get_feature(sub_a)
                f_b = env.node_env.get_feature(sub_b)
                if f_a != f_b:
                    parts.append(f"    Feature: {f_a} vs {f_b}")
                t_a = env.node_env.get_threshold(sub_a)
                t_b = env.node_env.get_threshold(sub_b)
                if t_a != t_b:
                    delta = (t_a - t_b) if t_a is not None and t_b is not None else None
                    parts.append(
                        f"    Threshold: {t_a!r} vs {t_b!r}"
                        + (f" (delta={delta:.2e})" if delta is not None else "")
                    )
                for key in sorted(set(sub_a.keys()) | set(sub_b.keys()), key=str):
                    va, vb = sub_a.get(key), sub_b.get(key)
                    if va != vb:
                        parts.append(f"    substate[{key}]: {va!r} vs {vb!r}")
                diffs.extend(parts)
    return "\n".join(diffs) if diffs else "  (no differences found)"


def _build_node_recording(env, node_idx, feature_idx, threshold_val, states, actions):
    """
    Builds a complete node while recording every intermediate (state, action).
    Uses the same steps as _build_node_with_dtnode_subenv.
    """

    def do_step(action):
        s, a, v = env.step(action)
        assert v, f"Step failed at node {node_idx}: action={action}"
        states.append(copy(s))
        actions.append(a)

    do_step(env._pad_action((node_idx,), -1))
    do_step((0, 0, feature_idx, 0))
    do_step((0, 0, -1, 0))
    do_step((0, 1, 2 * threshold_val / 5, 1))
    do_step((0, 1, 2 * threshold_val / 5, 0))
    do_step((0, 1, threshold_val / 5, 0))
    do_step((0, 1, float("inf"), float("inf")))
    do_step(env._pad_action((node_idx,), -1))


def test__rescale_unrescale__demonstrates_fp_error(env_tree_depth3):
    """
    Demonstrates the IEEE 754 floating-point error in the rescale/unrescale
    roundtrip that causes test__trajectories_are_reversible to fail.

    Builds root (node 0) and its right child (node 2) with the same feature.
    The right-subtree rescaling uses bounds [0.7, 1.0] with span 0.3, for
    which the roundtrip raw -> rescaled -> unrescaled is NOT bit-exact:
    0.6 * 0.3 / 0.3 != 0.6 in IEEE 754.

    The test captures the state before and after the Cube EOS (which triggers
    rescaling), then undoes it (triggering unrescaling), and compares.
    """
    env = env_tree_depth3

    # Build root: feature 1, threshold 0.7
    _build_node_with_dtnode_subenv(env, 0, 1, 0.7)

    # Build node 2 (right child of root) step by step with SAME feature
    env.step(env._pad_action((2,), -1))  # activate
    env.step((0, 0, 1, 0))  # choose feature 1 (same as root!)
    env.step((0, 0, -1, 0))  # Choice EOS
    env.step((0, 1, 0.24, 1))  # threshold step 1 (from source)
    env.step((0, 1, 0.24, 0))  # threshold step 2
    env.step((0, 1, 0.12, 0))  # threshold step 3 → raw = 0.6

    # Capture state BEFORE Cube EOS (threshold is the raw value from ContinuousCube)
    state_before_cube_eos = copy(env.state)
    raw_threshold = env.node_env.get_threshold(env.state[2])

    # Cube EOS triggers _rescale_threshold
    env.step((0, 1, float("inf"), float("inf")))
    rescaled_threshold = env.node_env.get_threshold(env.state[2])

    # Deactivate, then undo deactivation + Cube EOS
    env.step(env._pad_action((2,), -1))  # deactivate
    env.step_backwards(env._pad_action((2,), -1))  # backward deactivation
    _, _, valid = env.step_backwards(
        (0, 1, float("inf"), float("inf"))
    )  # backward Cube EOS → triggers _unrescale_threshold
    assert valid

    # Capture state AFTER backward Cube EOS (threshold should match raw, but may not)
    state_after_bw_cube_eos = copy(env.state)
    recovered_threshold = env.node_env.get_threshold(env.state[2])

    lower, upper = env._get_threshold_bounds(2, copy(env.state))

    info = (
        f"\n--- Floating-point roundtrip for node 2 "
        f"(right child of root, same feature) ---\n"
        f"  Root threshold: {env.node_env.get_threshold(env.state[0])!r}\n"
        f"  Rescaling bounds for node 2: [{lower}, {upper}], "
        f"span = {upper - lower!r}\n"
        f"  Raw threshold (before Cube EOS): {raw_threshold!r}\n"
        f"  Rescaled (after Cube EOS):       {rescaled_threshold!r}\n"
        f"  Recovered (after backward EOS):  {recovered_threshold!r}\n"
        f"  raw == recovered: {raw_threshold == recovered_threshold}\n"
    )
    if raw_threshold is not None and recovered_threshold is not None:
        info += f"  delta: {recovered_threshold - raw_threshold:.2e}\n"
    print(info)

    states_match = env.equal(state_before_cube_eos, state_after_bw_cube_eos)
    if not states_match:
        diff = _state_diff_detail(env, state_before_cube_eos, state_after_bw_cube_eos)
        pytest.fail(
            f"Cube EOS forward/backward is NOT reversible due to floating-point "
            f"error in threshold rescale/unrescale.{info}\n"
            f"State diff:\n{diff}\n\n"
            f"This is the root cause of the intermittent "
            f"test__trajectories_are_reversible failure at depth >= 3."
        )


def test__trajectories_are_reversible__determined_right_subtree(env_tree_depth3):
    """
    Deterministic full trajectory test: builds root and its right child with
    the same feature, takes EOS, then replays the entire trajectory backward.

    This reliably triggers the floating-point rescale/unrescale error because
    the right-subtree bounds [0.7, 1.0] have span 0.3, and
    0.6 * 0.3 / 0.3 != 0.6 in IEEE 754.
    """
    env = env_tree_depth3
    states_fw = []
    actions_fw = []

    # Build root (node 0): feature 1, threshold 0.7
    _build_node_recording(env, 0, 1, 0.7, states_fw, actions_fw)
    # Build right child (node 2): SAME feature 1, threshold 0.6
    _build_node_recording(env, 2, 1, 0.6, states_fw, actions_fw)

    # EOS
    s, a, v = env.step(env.eos)
    assert v
    states_fw.append(copy(s))
    actions_fw.append(a)

    # Backward: replay all actions in reverse
    states_bw = []
    actions_copy = actions_fw.copy()
    while not env.is_source() or env.done:
        action = actions_copy.pop()
        state, _, valid = env.step_backwards(action)
        assert valid, f"Backward failed: action={action}, state={env.state2readable()}"
        states_bw.append(copy(state))

    # Compare intermediate states
    fw = states_fw[:-1]  # exclude final done state
    bw = states_bw[-2::-1]  # reverse, exclude source

    mismatches = []
    for i, (sf, sb) in enumerate(zip(fw, bw)):
        if not env.equal(sf, sb):
            mismatches.append(i)

    if mismatches:
        msg = [
            f"\nDetermined right-subtree trajectory NOT reversible.",
            f"Config: node 0 (feature 1, t=0.7) -> node 2 (feature 1, t=0.6)",
            f"Mismatches at step indices: {mismatches}",
        ]
        for i in mismatches[:3]:
            msg.append(f"\n--- Step {i}, action: {actions_fw[i]} ---")
            msg.append(f"Forward:\n  {env.state2readable(fw[i])}")
            msg.append(f"Backward:\n  {env.state2readable(bw[i])}")
            msg.append(f"Diff:\n{_state_diff_detail(env, fw[i], bw[i])}")
        pytest.fail("\n".join(msg))


@pytest.mark.repeat(20)
def test__trajectories_are_reversible__depth3_diagnostic(env_tree_depth3):
    """
    Random trajectory reversibility test for depth 3 with 20 repeats and
    detailed diagnostics. Shows the full tree structure and pinpoints the
    exact node/threshold that causes the mismatch.
    """
    env = env_tree_depth3
    env.reset()

    # Forward random trajectory
    states_fw = []
    actions_fw = []
    while not env.done:
        state, action, valid = env.step_random(backward=False)
        assert valid
        states_fw.append(copy(state))
        actions_fw.append(action)

    # Backward: replay forward actions in reverse
    states_bw = []
    actions_copy = actions_fw.copy()
    while not env.is_source() or env.done:
        action = actions_copy.pop()
        state, _, valid = env.step_backwards(action)
        assert valid, (
            f"Backward step failed: action={action}, "
            f"state={env.state2readable()}, done={env.done}"
        )
        states_bw.append(copy(state))

    # Compare
    fw = states_fw[:-1]
    bw = states_bw[-2::-1]

    mismatches = []
    for i, (sf, sb) in enumerate(zip(fw, bw)):
        if not env.equal(sf, sb):
            mismatches.append(i)

    if mismatches:
        # Show final tree structure for context
        final_state = states_fw[-1]
        tree_info = []
        for k in range(env.max_nodes):
            if env._node_is_done(k, final_state):
                f = env.node_env.get_feature(final_state[k])
                t = env.node_env.get_threshold(final_state[k])
                d = env.node_depth(k)
                parent = env.parent_idx(k) if k > 0 else None
                pf = (
                    env.node_env.get_feature(final_state[parent])
                    if parent is not None and env._node_is_done(parent, final_state)
                    else None
                )
                same_as_parent = " [SAME FEATURE AS PARENT]" if pf == f else ""
                tree_info.append(
                    f"  Node {k} (depth {d}): feature={f}, "
                    f"threshold={t:.6f}{same_as_parent}"
                )

        msg = [
            f"\nRandom trajectory NOT reversible at depth 3.",
            f"Mismatches at {len(mismatches)} of {len(fw)} steps.",
            f"Total actions: {len(actions_fw)}.",
        ]
        if tree_info:
            msg.append("\nFinal tree (done nodes):")
            msg.extend(tree_info)

        i = mismatches[0]
        msg.append(f"\nFirst mismatch at step {i}, action: {actions_fw[i]}")
        msg.append(f"Forward:\n  {env.state2readable(fw[i])}")
        msg.append(f"Backward:\n  {env.state2readable(bw[i])}")
        msg.append(f"Diff:\n{_state_diff_detail(env, fw[i], bw[i])}")

        pytest.fail("\n".join(msg))


# ===========================================================================
# set_state test
# ===========================================================================


@parametrize_envs
def test__set_state__sets_expected_state(envs, request):
    """set_state should correctly set the state."""
    env = request.getfixturevalue(envs)
    selected_feature, selected_threshold = 1, 0.95
    _build_node_with_dtnode_subenv(env, 0, selected_feature, selected_threshold)
    state_after_root = copy(env.state)

    env.reset()
    assert env.is_source()

    env.set_state(state_after_root, done=False)
    assert env.state["_active"] == state_after_root["_active"]
    assert env.state["_dones"] == state_after_root["_dones"]
    assert env.state[0] == {
        "_active": 1,
        0: [selected_feature],
        1: [selected_threshold],
    }
    assert env._node_is_done(0, env.state)


# ===========================================================================
# states2policy test
# ===========================================================================


@parametrize_envs
def test__states2policy__returns_correct_shape(envs, request):
    env = request.getfixturevalue(envs)
    node_pdim = env.node_env.policy_input_dim
    per_node_dim = 3 + node_pdim
    expected_dim = 1 + env.max_nodes * per_node_dim

    # Source state
    result = env.states2policy([env.state])
    assert result.shape == (1, expected_dim)
    # Idle flag should be 1.0 at source
    assert result[0, 0].item() == 1.0


@parametrize_envs_bigger_than_depth_1
def test__states2policy__batch_of_different_states(envs, request):
    """Policy encoding of a batch with source and an intermediate state."""
    env = request.getfixturevalue(envs)
    source_state = copy(env.state)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.2)
    intermediate_state = copy(env.state)

    result = env.states2policy([source_state, intermediate_state])
    assert result.shape[0] == 2

    # Source: idle=1, no nodes exist
    assert result[0, 0].item() == 1.0
    node_source_policy = env.node_env.states2policy([env.node_env.source])[0]
    empty_node_block = torch.cat([torch.zeros(3), node_source_policy])
    expected_source = torch.cat(
        [torch.tensor([1.0])] + [empty_node_block] * env.max_nodes
    )
    assert torch.equal(result[0, :], expected_source)

    # Intermediate: idle=1 (after deactivation), node 0 exists and is done
    assert result[1, 0].item() == 1.0
    node_pdim = env.node_env.policy_input_dim
    per_node_dim = 3 + node_pdim
    # Node 0: exists=1, done=1, active=0 (since idle)
    offset = 1
    assert result[1, offset].item() == 1.0  # exists
    assert result[1, offset + 1].item() == 1.0  # done
    assert result[1, offset + 2].item() == 0.0  # not active

    # Full tensor comparison
    node_0_policy = env.node_env.states2policy([intermediate_state[0]])[0]
    node_0_block = torch.cat([torch.tensor([1.0, 1.0, 0.0]), node_0_policy])
    expected_intermediate = torch.cat(
        [torch.tensor([1.0]), node_0_block] + [empty_node_block] * (env.max_nodes - 1)
    )
    assert torch.equal(result[1, :], expected_intermediate)


# ===========================================================================
# Mask format and unformat tests
# ===========================================================================


@parametrize_envs
def test__format_mask_meta__has_correct_structure(envs, request):
    """Meta mask: meta section (n_meta_actions) followed by all-True node section."""
    env = request.getfixturevalue(envs)
    # Arbitrary meta mask of the correct length (alternating True/False)
    meta_mask_content = [i % 2 == 0 for i in range(env.n_meta_actions)]
    formatted = env._format_mask_meta(meta_mask_content)
    assert len(formatted) == env.mask_dim
    # Meta section equals the input
    assert formatted[: env.n_meta_actions] == meta_mask_content
    # Node section is all True (no node actions valid in meta mode)
    assert all(m is True for m in formatted[env.n_meta_actions :])
    assert env._is_meta_mask(formatted) is True


@parametrize_envs
def test__format_mask_building__has_correct_structure(envs, request):
    """Building mask: all-True meta section followed by the node env mask."""
    env = request.getfixturevalue(envs)
    node_mask = env.node_env.get_mask_invalid_actions_forward()
    formatted = env._format_mask_building(node_mask)
    assert len(formatted) == env.mask_dim
    # Meta section all True (no meta actions valid in building mode)
    assert all(m is True for m in formatted[: env.n_meta_actions])
    # Node section equals the node env mask
    assert formatted[env.n_meta_actions :] == node_mask
    assert env._is_meta_mask(formatted) is False


@parametrize_envs
def test__unformat_mask_building__roundtrip(envs, request):
    """Unformatting a building mask should recover the original node env mask."""
    env = request.getfixturevalue(envs)
    original_mask = env.node_env.get_mask_invalid_actions_forward()
    formatted = env._format_mask_building(original_mask)
    recovered = env._unformat_mask_building(formatted)
    assert recovered == original_mask


# ===========================================================================
# Policy output structure
# ===========================================================================


@parametrize_envs
def test__policy_output__has_correct_dim(envs, request):
    env = request.getfixturevalue(envs)
    expected_dim = env.node_env.policy_output_dim + env.n_meta_actions
    assert env.policy_output_dim == expected_dim
    # Verify with fixed_distr_params (the standard way to call this)
    po = env.get_policy_output(env.fixed_distr_params)
    assert len(po) == expected_dim


# ===========================================================================
# Common base tests from common.py
# ===========================================================================


class TestTreeDepth1(common.BaseTestsContinuous):
    """Common tests for Tree with depth 1."""

    @pytest.fixture(autouse=True)
    def setup(self, env_tree_depth1):
        self.env = env_tree_depth1
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }


class TestTreeDepth2(common.BaseTestsContinuous):
    """Common tests for Tree with depth 2."""

    @pytest.fixture(autouse=True)
    def setup(self, env_tree_depth2):
        self.env = env_tree_depth2
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }


class TestTreeDepth3(common.BaseTestsContinuous):
    """Common tests for Tree with depth 3."""

    @pytest.fixture(autouse=True)
    def setup(self, env_tree_depth3):
        self.env = env_tree_depth3
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }


# ===========================================================================
# Dataset initialization tests
# ===========================================================================


class TestTreeDataPathInit:
    """Tests for Tree initialized via data_path (CSV)."""

    def test__init__loads_features_from_csv(self, env_tree_depth4_data_path):
        env = env_tree_depth4_data_path
        expected_features = [
            "sepal length",
            "sepal width",
            "petal length",
            "petal width",
        ]
        assert env.node_env.features == expected_features
        assert env.node_env.n_features == 4

    def test__init__loads_train_data(self, env_tree_depth4_data_path):
        env = env_tree_depth4_data_path
        assert env.X_train is not None
        assert env.y_train is not None
        assert env.X_train.shape[0] == 120  # 120 train rows in iris.csv
        assert env.X_train.shape[1] == 4
        assert env.y_train.shape[0] == 120

    def test__init__loads_test_data(self, env_tree_depth4_data_path):
        env = env_tree_depth4_data_path
        assert env.X_test is not None
        assert env.y_test is not None
        assert env.X_test.shape[0] == 30  # 30 test rows in iris.csv
        assert env.X_test.shape[1] == 4
        assert env.y_test.shape[0] == 30

    def test__init__applies_scaling(self, env_tree_depth4_data_path):
        env = env_tree_depth4_data_path
        assert env.scaler is not None
        # Scaled data should be in [0, 1] range
        assert env.X_train.min() >= 0.0
        assert env.X_train.max() <= 1.0

    def test__init__labels_are_int(self, env_tree_depth4_data_path):
        env = env_tree_depth4_data_path
        assert env.y_train.dtype == int
        assert env.y_test.dtype == int

    def test__init__max_depth_and_max_nodes(self, env_tree_depth4_data_path):
        env = env_tree_depth4_data_path
        assert env.max_depth == 4
        assert env.max_nodes == 15

    def test__init__no_scale(self):
        env = Tree(max_depth=2, data_path=IRIS_CSV_PATH, scale_data=False)
        assert env.scaler is None
        # Unscaled data has values > 1
        assert env.X_train.max() > 1.0

    def test__init__node_kwargs_features_override_dataset(self):
        """If node_kwargs provides features, they take precedence over dataset columns."""
        custom_features = ["a", "b", "c", "d"]
        env = Tree(
            max_depth=2,
            data_path=IRIS_CSV_PATH,
            node_kwargs={"features": custom_features},
        )
        assert env.node_env.features == custom_features
        # Dataset is still loaded
        assert env.X_train is not None

    def test__init__error_without_features_or_data(self):
        with pytest.raises(ValueError, match="must be initialized with features"):
            Tree(max_depth=2)


class TestTreeXyInit:
    """Tests for Tree initialized via X_train/y_train arrays."""

    def test__init__stores_data(self, env_tree_depth4_xy):
        env = env_tree_depth4_xy
        assert env.X_train is not None
        assert env.y_train is not None
        assert env.X_train.shape == (80, 4)
        assert env.y_train.shape == (80,)

    def test__init__stores_test_data(self, env_tree_depth4_xy):
        env = env_tree_depth4_xy
        assert env.X_test is not None
        assert env.y_test is not None
        assert env.X_test.shape == (20, 4)
        assert env.y_test.shape == (20,)

    def test__init__generates_feature_names(self, env_tree_depth4_xy):
        env = env_tree_depth4_xy
        assert env.node_env.features == ["x0", "x1", "x2", "x3"]
        assert env.node_env.n_features == 4

    def test__init__applies_scaling(self, env_tree_depth4_xy):
        env = env_tree_depth4_xy
        assert env.scaler is not None
        assert env.X_train.min() >= 0.0
        assert env.X_train.max() <= 1.0

    def test__init__labels_are_int(self, env_tree_depth4_xy):
        env = env_tree_depth4_xy
        assert env.y_train.dtype == int
        assert env.y_test.dtype == int

    def test__init__max_depth_and_max_nodes(self, env_tree_depth4_xy):
        env = env_tree_depth4_xy
        assert env.max_depth == 4
        assert env.max_nodes == 15

    def test__init__no_test_data(self):
        rng = np.random.default_rng(0)
        env = Tree(
            max_depth=2,
            X_train=rng.random((50, 3)),
            y_train=rng.integers(0, 2, size=50),
        )
        assert env.X_test is None
        assert env.y_test is None
        assert env.node_env.features == ["x0", "x1", "x2"]

    def test__init__no_scale(self):
        X = np.array([[0.0, 5.0], [1.0, 10.0], [2.0, 15.0]])
        y = np.array([0, 1, 0])
        env = Tree(max_depth=2, X_train=X, y_train=y, scale_data=False)
        assert env.scaler is None
        assert env.X_train[2, 1] == 15.0


# ===========================================================================
# Common base tests for dataset-initialized trees (depth 4)
# ===========================================================================


class TestTreeDepth4DataPath(common.BaseTestsContinuous):
    """Common tests for Tree with depth 4 initialized from CSV data_path."""

    @pytest.fixture(autouse=True)
    def setup(self, env_tree_depth4_data_path):
        self.env = env_tree_depth4_data_path
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }


class TestTreeDepth4Xy(common.BaseTestsContinuous):
    """Common tests for Tree with depth 4 initialized from X_train/y_train arrays."""

    @pytest.fixture(autouse=True)
    def setup(self, env_tree_depth4_xy):
        self.env = env_tree_depth4_xy
        self.repeats = {
            "test__reset__state_is_source": 10,
            "test__forward_actions_have_nonzero_backward_prob": 10,
            "test__backward_actions_have_nonzero_forward_prob": 10,
            "test__trajectories_are_reversible": 10,
            "test__step_random__does_not_sample_invalid_actions_forward": 10,
            "test__step_random__does_not_sample_invalid_actions_backward": 10,
            "test__get_mask__is_consistent_regardless_of_inputs": 10,
            "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
            "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
            "test__get_parents_step_get_mask__are_compatible": 10,
            "test__sample_backwards_reaches_source": 10,
            "test__state2readable__is_reversible": 20,
            "test__gflownet_minimal_runs": 3,
        }
        self.n_states = {
            "test__backward_actions_have_nonzero_forward_prob": 3,
            "test__sample_backwards_reaches_source": 3,
            "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
            "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
        }
        self.batch_size = {
            "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
            "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
            "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
            "test__gflownet_minimal_runs": 10,
        }


# ===========================================================================
# Discrete-node tests
#
# The discrete node uses a finite Grid for the threshold instead of a
# ContinuousCube. Tree's ancestor-based threshold bounds are enforced by
# masking out forbidden grid cells rather than rescaling the stored value.
# These tests exercise both the basic forward/backward semantics of the
# discrete-node Tree and the threshold-bound masking logic.
# ===========================================================================


parametrize_discrete_envs = pytest.mark.parametrize(
    "envs",
    [
        "env_tree_discrete_depth1",
        "env_tree_discrete_depth2",
        "env_tree_discrete_depth3",
    ],
)


parametrize_discrete_envs_bigger_than_depth_1 = pytest.mark.parametrize(
    "envs",
    [
        "env_tree_discrete_depth2",
        "env_tree_discrete_depth3",
    ],
)


@parametrize_discrete_envs
def test__discrete__environment_initializes(envs, request):
    env = request.getfixturevalue(envs)
    assert env.node_type == "discrete"
    # The discrete node is internally a Stack of two discrete sub-envs, so the
    # composite environment is NOT continuous.
    assert env.continuous is False


@parametrize_discrete_envs
def test__discrete__source_state(envs, request):
    env = request.getfixturevalue(envs)
    expected = {
        "_active": -1,
        "_dones": [0] * env.max_nodes,
        "_envs_unique": 1,
    }
    assert env.source == expected
    assert env.is_source()


def test__discrete__threshold_values_span_unit_interval(env_tree_discrete_depth2):
    """With cell_min=0 and cell_max=1, cells are linspace(0, 1, n)."""
    env = env_tree_discrete_depth2
    cells = env.node_env.threshold_values
    assert cells.shape == (env.node_env.n_thresholds,)
    assert cells[0] == 0.0
    assert cells[-1] == 1.0
    deltas = np.diff(cells)
    assert np.allclose(deltas, deltas[0])


def test__discrete__node_env_continuous_flag(env_tree_discrete_depth2):
    env = env_tree_discrete_depth2
    assert env.node_env.continuous is False


def test__discrete__threshold_bounds_attribute_defaults(env_tree_discrete_depth2):
    """At init the cached bounds equal the full [cell_min, cell_max] range."""
    env = env_tree_discrete_depth2
    assert env.node_env._tb_lower == env.node_env.cell_min
    assert env.node_env._tb_upper == env.node_env.cell_max


@parametrize_discrete_envs
def test__discrete__build_root_and_terminate(envs, request):
    env = request.getfixturevalue(envs)
    _build_discrete_node(env, 0, feature_idx=1, cell_idx=2)
    assert env._node_is_done(0, env.state)
    feature = env.node_env.get_feature(env.state[0])
    threshold = env.node_env.get_threshold(env.state[0])
    assert feature == 1
    assert threshold == env.node_env.threshold_values[2]
    _, _, valid = env.step(env.eos)
    assert valid
    assert env.done


@parametrize_discrete_envs
def test__discrete__forward_then_backward_reaches_source(envs, request):
    env = request.getfixturevalue(envs)
    _build_discrete_node(env, 0, feature_idx=1, cell_idx=2)
    env.step(env.eos)
    while not env.is_source():
        if env.done:
            a = env.eos
        else:
            va = env.get_valid_actions(backward=True)
            assert va, f"No backward actions; state={env.state2readable()}"
            a = va[0]
        _, _, v = env.step_backwards(a)
        assert v
    assert env.is_source()


@parametrize_discrete_envs_bigger_than_depth_1
def test__discrete__threshold_bounds_left_child_upper(envs, request):
    """
    Building the root with feature f at threshold T should make any child on
    the <= branch (left subtree) using the same feature unable to pick a
    threshold strictly greater than T.
    """
    env = request.getfixturevalue(envs)
    _build_discrete_node(
        env, 0, feature_idx=1, cell_idx=2
    )  # Set thrid threshold (by taking two increment steps) on feautre 1 at root
    parent_threshold = env.node_env.get_threshold(env.state[0])

    env.step(env._pad_action((1,), -1))  # Activate node 1 (left node of root)
    env.step((0, 0, 1, 0))  # Chose same feature 1
    env.step((0, 0, -1, 0))  # Choice EOS

    # Assert bounds on threshold are correct (no trivially ill-defined splitting rule possible)
    lower, upper = env._get_threshold_bounds(1, env.state)
    assert lower == env.node_env.cell_min
    assert upper == parent_threshold

    # Check if the threshold bounds are acctually honored when computing the new mask
    valid = env.get_valid_actions()
    current = env.state[1][env.node_env.stage_threshold][0]  # Current threshold
    for action in valid:
        if action[0] != 0:  # Skip meta-actions
            continue
        if action[1] != env.node_env.stage_threshold:  # Skip Choice related actions
            continue
        k = action[2]  # Grid increment amount in the grid action
        if k == 0:  # Skip if no increase of threshold
            continue
        new_idx = current + k
        new_value = env.node_env.threshold_values[new_idx]
        assert new_value <= parent_threshold + 1e-12


@parametrize_discrete_envs_bigger_than_depth_1
def test__discrete__threshold_bounds_right_child_lower(envs, request):
    """Right subtree: thresholds must be >= ancestor's threshold."""
    env = request.getfixturevalue(envs)
    _build_discrete_node(
        env, 0, feature_idx=1, cell_idx=2
    )  # Set thrid threshold (by taking two increment steps) on feautre 1 at root
    parent_threshold = env.node_env.get_threshold(env.state[0])

    env.step(env._pad_action((2,), -1))
    env.step((0, 0, 1, 0))
    env.step((0, 0, -1, 0))

    lower, upper = env._get_threshold_bounds(2, env.state)
    assert lower == parent_threshold
    assert upper == env.node_env.cell_max

    # Grid starts at cell 0; cell 0 value (0.0) < parent_threshold,
    # so Grid EOS must be masked invalid initially.
    valid = env.get_valid_actions()
    grid_eos = (0, env.node_env.stage_threshold, 0, 0)
    assert grid_eos not in valid


@parametrize_discrete_envs_bigger_than_depth_1
def test__discrete__bounds_enforced_during_step(envs, request):
    """Tree.step must reject out-of-range Grid moves on the active node."""
    env = request.getfixturevalue(envs)
    _build_discrete_node(env, 0, feature_idx=1, cell_idx=2)

    env.step(env._pad_action((1,), -1))  # activate left child
    env.step((0, 0, 1, 0))  # same feature
    env.step((0, 0, -1, 0))  # Choice EOS

    # Advance the Grid to cell 2 (allowed because cell 2 value == upper bound).
    env.step((0, 1, 1, 0))
    env.step((0, 1, 1, 0))
    assert env.state[1][env.node_env.stage_threshold][0] == 2

    # Attempt to advance to cell 3 (value > upper): must fail.
    _, _, valid = env.step((0, 1, 1, 0))
    assert valid is False

    # Grid EOS at cell 2 is accepted (value == upper, lower satisfied).
    _, _, valid = env.step((0, 1, 0, 0))
    assert valid is True
    assert env._node_is_done(1, env.state)


@parametrize_discrete_envs_bigger_than_depth_1
def test__discrete__bounds_unchanged_for_different_feature(envs, request):
    """
    Threshold bounds are scoped to ancestors splitting on the same feature.
    A child using a different feature must see the full range.
    """
    env = request.getfixturevalue(envs)
    _build_discrete_node(env, 0, feature_idx=1, cell_idx=2)

    env.step(env._pad_action((1,), -1))
    env.step((0, 0, 2, 0))  # DIFFERENT feature
    env.step((0, 0, -1, 0))

    lower, upper = env._get_threshold_bounds(1, env.state)
    assert lower == env.node_env.cell_min
    assert upper == env.node_env.cell_max


@parametrize_discrete_envs_bigger_than_depth_1
def test__discrete__build_and_unbuild_two_nodes(envs, request):
    """End-to-end forward then backward via the build/unbuild helpers."""
    env = request.getfixturevalue(envs)
    _build_discrete_node(env, 0, feature_idx=1, cell_idx=2)
    _build_discrete_node(env, 1, feature_idx=2, cell_idx=1)
    assert env._node_is_done(0, env.state)
    assert env._node_is_done(1, env.state)

    _unbuild_discrete_node(env, 1, feature_idx=2, cell_idx=1)
    assert 1 not in env.state
    _unbuild_discrete_node(env, 0, feature_idx=1, cell_idx=2)
    assert env.is_source()


@parametrize_discrete_envs
def test__discrete__random_trajectory_reaches_done(envs, request):
    env = request.getfixturevalue(envs)
    env.reset()
    env.trajectory_random()
    assert env.done
    # When idle the shared node_env bounds must be back to the full range.
    assert env.node_env._tb_lower == env.node_env.cell_min
    assert env.node_env._tb_upper == env.node_env.cell_max


@pytest.mark.repeat(5)
@parametrize_discrete_envs
def test__discrete__random_forward_then_backward_reaches_source(envs, request):
    env = request.getfixturevalue(envs)
    env.reset()
    env.trajectory_random()
    assert env.done
    while not env.is_source():
        if env.done:
            a = env.eos
        else:
            va = env.get_valid_actions(backward=True)
            assert va, f"No backward actions; state={env.state2readable()}"
            a = va[0]
        _, _, v = env.step_backwards(a)
        assert v
    assert env.is_source()


@pytest.mark.repeat(5)
@parametrize_discrete_envs
def test__discrete__random_trajectory_respects_ancestor_bounds(envs, request):
    """
    Sampled trees must never produce a (same-feature) descendant whose
    threshold violates the parent's split direction.
    """
    env = request.getfixturevalue(envs)
    env.reset()
    env.trajectory_random()
    assert env.done

    for k in range(env.max_nodes):
        if not env._node_is_done(k, env.state):
            continue
        feature_k = env.node_env.get_feature(env.state[k])
        threshold_k = env.node_env.get_threshold(env.state[k])
        if feature_k is None or threshold_k is None:
            continue
        current = k
        while current > 0:
            parent = env.parent_idx(current)
            parent_feature = env.node_env.get_feature(env.state[parent])
            parent_threshold = env.node_env.get_threshold(env.state[parent])
            if parent_feature == feature_k and parent_threshold is not None:
                if env._is_in_left_subtree(k, parent):
                    assert threshold_k <= parent_threshold + 1e-12, (
                        f"Left child {k} threshold {threshold_k} > ancestor "
                        f"{parent} threshold {parent_threshold}"
                    )
                else:
                    assert threshold_k >= parent_threshold - 1e-12, (
                        f"Right child {k} threshold {threshold_k} < ancestor "
                        f"{parent} threshold {parent_threshold}"
                    )
            current = parent


@parametrize_discrete_envs
def test__discrete__get_mask_is_consistent_regardless_of_inputs(envs, request):
    """
    Mask computed with an explicit state argument must match the mask computed
    against self.state. Validates that the bound-publication / reset machinery
    on the shared node_env is sound.
    """
    env = request.getfixturevalue(envs)
    env.reset()
    env.trajectory_random()
    if env.done:
        env.step_backwards(env.eos)
    for _ in range(3):
        va = env.get_valid_actions(backward=True)
        if va:
            env.step_backwards(va[0])

    state = copy(env.state)
    done = env.done

    mask_f_self = env.get_mask_invalid_actions_forward()
    mask_f_arg = env.get_mask_invalid_actions_forward(state, done)
    assert mask_f_self == mask_f_arg

    mask_b_self = env.get_mask_invalid_actions_backward()
    mask_b_arg = env.get_mask_invalid_actions_backward(state, done)
    assert mask_b_self == mask_b_arg


@pytest.mark.parametrize("n_backwards_steps", list(range(1, 11)))
def test__discrete__get_mask_consistent_depth3_varying_backwards_steps(
    n_backwards_steps, env_tree_discrete_depth3
):
    """
    Stronger version of
    :py:func:`test__discrete__get_mask_is_consistent_regardless_of_inputs`:
    runs the same consistency check on a depth-3 discrete tree after walking
    ``n_backwards_steps`` greedy backward steps from a random terminating
    trajectory. Parametrized over 1..10 so that each parameter value lands
    the env in a different point along the backward lifecycle (mid-build,
    between nodes, idle with some nodes done, ...).

    A failure here pinpoints which walk-back distance breaks the
    bound-publication / reset machinery on the shared ``node_env``.
    """
    env = env_tree_discrete_depth3
    env.reset()
    env.trajectory_random()
    if env.done:
        env.step_backwards(env.eos)
    for _ in range(n_backwards_steps):
        if env.is_source():
            break
        va = env.get_valid_actions(backward=True)
        if not va:
            break
        env.step_backwards(va[0])

    state = copy(env.state)
    done = env.done

    mask_f_self = env.get_mask_invalid_actions_forward()
    mask_f_arg = env.get_mask_invalid_actions_forward(state, done)
    assert mask_f_self == mask_f_arg, (
        f"Forward mask mismatch at n_backwards_steps={n_backwards_steps}; "
        f"state={env.state2readable()}"
    )

    mask_b_self = env.get_mask_invalid_actions_backward()
    mask_b_arg = env.get_mask_invalid_actions_backward(state, done)
    assert mask_b_self == mask_b_arg, (
        f"Backward mask mismatch at n_backwards_steps={n_backwards_steps}; "
        f"state={env.state2readable()}"
    )


# ---------------------------------------------------------------------------
# No-op rescale contract on the discrete node
# ---------------------------------------------------------------------------


def test__discrete__rescale_is_noop(env_tree_discrete_depth2):
    """
    apply_threshold_rescale / unapply_threshold_rescale must be no-ops on the
    discrete node: sampled thresholds are already in range.
    """
    env = env_tree_discrete_depth2
    _build_discrete_node(env, 0, feature_idx=1, cell_idx=2)
    before = copy(env.state[0])
    after = env.node_env.apply_threshold_rescale(env.state[0], 0.0, 0.4)
    assert before == after
    after_u = env.node_env.unapply_threshold_rescale(env.state[0], 0.0, 0.4)
    assert before == after_u


# ---------------------------------------------------------------------------
# Continuous-node interface contract (shared with discrete)
# ---------------------------------------------------------------------------


def test__continuous__node_exposes_threshold_bounds_interface(env_tree_depth2):
    """
    The continuous node implements the same threshold-constraint interface so
    Tree can dispatch to either node implementation uniformly.
    """
    env = env_tree_depth2
    env.node_env.set_threshold_bounds(0.1, 0.9)
    assert env.node_env._tb_lower == 0.1
    assert env.node_env._tb_upper == 0.9
    env.node_env.clear_threshold_bounds()
    assert env.node_env._tb_lower == env.node_env.threshold_min
    assert env.node_env._tb_upper == env.node_env.threshold_max


# ===========================================================================
# Continuous-node threshold-rule tests
#
# These tests directly exercise the ancestor-derived threshold semantics for
# the continuous (ContinuousCube) node:
#   - ``_get_threshold_bounds`` computes the expected (lower, upper) range
#     given a tree's structure and ancestor splits on the same feature.
#   - After a node's Cube EOS is taken, the stored threshold has been
#     rescaled into that range and therefore respects the parent's split
#     direction (left subtree => threshold <= parent; right subtree =>
#     threshold >= parent).
#   - The node-side ``apply_threshold_rescale`` / ``unapply_threshold_rescale``
#     pair round-trips correctly.
#   - The ``rescale_thresholds`` flag on the Tree disables the rescale step.
# ===========================================================================


@pytest.fixture
def env_tree_depth3_no_rescale():
    """A continuous-node Tree with threshold rescaling disabled."""
    features = ["feat_a", "feat_b", "feat_c"]
    return Tree(
        max_depth=3,
        node_kwargs={"features": features},
        rescale_thresholds=False,
    )


def test__continuous__get_threshold_bounds__no_ancestors_returns_unit(
    env_tree_depth2,
):
    """At the root the bounds are the node's full range [0, 1]."""
    env = env_tree_depth2
    env.step(env._pad_action((0,), -1))  # activate root
    env.step((0, 0, 1, 0))  # choose feature 1
    env.step((0, 0, -1, 0))  # Choice EOS
    lower, upper = env._get_threshold_bounds(0, env.state)
    assert lower == env.node_env.threshold_min
    assert upper == env.node_env.threshold_max


def test__continuous__get_threshold_bounds__different_feature_returns_unit(
    env_tree_depth2,
):
    """A child using a different feature should see the unrestricted range."""
    env = env_tree_depth2
    _build_node_with_dtnode_subenv(env, 0, feature_idx=1, threshold_val=0.5)
    env.step(env._pad_action((1,), -1))
    env.step((0, 0, 2, 0))  # DIFFERENT feature
    env.step((0, 0, -1, 0))
    lower, upper = env._get_threshold_bounds(1, env.state)
    assert lower == env.node_env.threshold_min
    assert upper == env.node_env.threshold_max


def test__continuous__get_threshold_bounds__left_child_same_feature(
    env_tree_depth2,
):
    """Left-subtree child sharing the parent feature has upper == parent."""
    env = env_tree_depth2
    parent_threshold = 0.5
    _build_node_with_dtnode_subenv(
        env, 0, feature_idx=1, threshold_val=parent_threshold
    )
    env.step(env._pad_action((1,), -1))
    env.step((0, 0, 1, 0))  # SAME feature
    env.step((0, 0, -1, 0))
    lower, upper = env._get_threshold_bounds(1, env.state)
    assert lower == env.node_env.threshold_min
    # After rescale the stored parent threshold equals the rescaled value;
    # at the root with no further ancestors there is no rescaling, so it
    # equals exactly the raw input.
    assert upper == parent_threshold


def test__continuous__get_threshold_bounds__right_child_same_feature(
    env_tree_depth2,
):
    """Right-subtree child sharing the parent feature has lower == parent."""
    env = env_tree_depth2
    parent_threshold = 0.5
    _build_node_with_dtnode_subenv(
        env, 0, feature_idx=1, threshold_val=parent_threshold
    )
    env.step(env._pad_action((2,), -1))
    env.step((0, 0, 1, 0))  # SAME feature
    env.step((0, 0, -1, 0))
    lower, upper = env._get_threshold_bounds(2, env.state)
    assert lower == parent_threshold
    assert upper == env.node_env.threshold_max


def test__continuous__get_threshold_bounds__deep_left_chain(env_tree_depth3):
    """
    Tighter ancestor constraints compose: if a chain of left-subtree ancestors
    all split on the same feature, the upper bound is the minimum of their
    thresholds.
    """
    env = env_tree_depth3
    # Root threshold 0.8, left child threshold (rescaled into [0, 0.8]) <= 0.8.
    _build_node_with_dtnode_subenv(env, 0, feature_idx=1, threshold_val=0.8)
    # Choose a threshold of 0.5 raw -> rescaled into [0, 0.8] -> 0.4.
    _build_node_with_dtnode_subenv(env, 1, feature_idx=1, threshold_val=0.5)
    # Activate left-left grandchild (node 3) and choose same feature.
    env.step(env._pad_action((3,), -1))
    env.step((0, 0, 1, 0))
    env.step((0, 0, -1, 0))
    parent_t = env.node_env.get_threshold(env.state[1])
    lower, upper = env._get_threshold_bounds(3, env.state)
    assert lower == env.node_env.threshold_min
    # Tightest ancestor is the direct parent (node 1)
    assert upper == parent_t
    # And node 1's threshold must already respect the root's upper bound.
    assert parent_t <= env.node_env.get_threshold(env.state[0]) + 1e-12


def test__continuous__rescaled_threshold_respects_left_parent(env_tree_depth2):
    """
    After Cube EOS on a left child sharing the parent's feature, the stored
    threshold is in [0, parent_threshold]. Picking a raw value of 1.0 (the
    largest) must rescale exactly to parent_threshold.
    """
    env = env_tree_depth2
    parent_threshold = 0.6
    _build_node_with_dtnode_subenv(
        env, 0, feature_idx=1, threshold_val=parent_threshold
    )

    # Drive node 1 toward raw threshold = 1.0
    env.step(env._pad_action((1,), -1))
    env.step((0, 0, 1, 0))
    env.step((0, 0, -1, 0))
    # ContinuousCube: jump from source to 1.0 in one "from-source" step
    env.step((0, 1, 1.0, 1))
    env.step((0, 1, float("inf"), float("inf")))  # Cube EOS triggers rescale

    stored = env.node_env.get_threshold(env.state[1])
    assert abs(stored - parent_threshold) < 1e-10
    assert stored <= parent_threshold + 1e-12


def test__continuous__rescaled_threshold_respects_right_parent(env_tree_depth2):
    """
    After Cube EOS on a right child sharing the parent's feature, the stored
    threshold is in [parent_threshold, 1]. Picking a raw value of 0.0 (the
    smallest) must rescale exactly to parent_threshold.
    """
    env = env_tree_depth2
    parent_threshold = 0.4
    _build_node_with_dtnode_subenv(
        env, 0, feature_idx=1, threshold_val=parent_threshold
    )

    # Drive node 2 toward raw threshold = 0.0
    env.step(env._pad_action((2,), -1))
    env.step((0, 0, 1, 0))
    env.step((0, 0, -1, 0))
    env.step((0, 1, 0.0, 1))  # from source to 0.0
    env.step((0, 1, float("inf"), float("inf")))

    stored = env.node_env.get_threshold(env.state[2])
    assert abs(stored - parent_threshold) < 1e-10
    assert stored >= parent_threshold - 1e-12


def test__continuous__rescaled_threshold_arbitrary_raw_in_bounds(env_tree_depth2):
    """
    For any sampled raw value ``r`` in (0, 1) on a left child sharing the
    parent's feature with parent threshold ``θ``, the stored threshold equals
    ``r * θ`` (rescale formula: lower + r * (upper - lower) with lower=0).
    """
    env = env_tree_depth2
    theta = 0.5
    _build_node_with_dtnode_subenv(env, 0, feature_idx=1, threshold_val=theta)

    raw = 0.7
    env.step(env._pad_action((1,), -1))
    env.step((0, 0, 1, 0))
    env.step((0, 0, -1, 0))
    env.step((0, 1, raw, 1))
    env.step((0, 1, float("inf"), float("inf")))

    stored = env.node_env.get_threshold(env.state[1])
    assert abs(stored - raw * theta) < 1e-10
    assert stored <= theta + 1e-12


def test__continuous__rescaled_threshold_does_not_constrain_diff_feature(
    env_tree_depth2,
):
    """
    If the child splits on a different feature, the raw threshold is stored
    unchanged (no ancestor constraint applies).
    """
    env = env_tree_depth2
    _build_node_with_dtnode_subenv(env, 0, feature_idx=1, threshold_val=0.3)

    raw = 0.85
    env.step(env._pad_action((1,), -1))
    env.step((0, 0, 2, 0))  # different feature
    env.step((0, 0, -1, 0))
    env.step((0, 1, raw, 1))
    env.step((0, 1, float("inf"), float("inf")))

    stored = env.node_env.get_threshold(env.state[1])
    assert abs(stored - raw) < 1e-10


def test__continuous__rescale_unapply_roundtrip_on_node(env_tree_depth2):
    """
    The node's own ``apply_threshold_rescale`` followed by
    ``unapply_threshold_rescale`` recovers the raw threshold (up to FP error
    for non-power-of-two spans).
    """
    env = env_tree_depth2
    # Construct a fake substate with raw threshold 0.6
    substate = env.node_env.subenvs[0].source, [0.6]
    sub = {
        "_active": 1,
        env.node_env.stage_feature: copy(env.node_env.feature_env.source),
        env.node_env.stage_threshold: [0.6],
    }
    raw_before = env.node_env.get_threshold(sub)
    env.node_env.apply_threshold_rescale(sub, 0.2, 0.8)
    rescaled = env.node_env.get_threshold(sub)
    assert abs(rescaled - (0.2 + 0.6 * 0.6)) < 1e-12  # 0.2 + raw * 0.6
    env.node_env.unapply_threshold_rescale(sub, 0.2, 0.8)
    recovered = env.node_env.get_threshold(sub)
    assert abs(recovered - raw_before) < 1e-10


def test__continuous__rescale_is_noop_when_bounds_are_trivial(env_tree_depth2):
    """
    ``apply_threshold_rescale(state, 0, 1)`` returns the substate unchanged.
    """
    env = env_tree_depth2
    sub = {
        "_active": 1,
        env.node_env.stage_feature: copy(env.node_env.feature_env.source),
        env.node_env.stage_threshold: [0.42],
    }
    before = copy(sub)
    env.node_env.apply_threshold_rescale(sub, 0.0, 1.0)
    assert sub == before
    env.node_env.unapply_threshold_rescale(sub, 0.0, 1.0)
    assert sub == before


def test__continuous__rescale_disabled_by_tree_flag(env_tree_depth3_no_rescale):
    """
    ``Tree(rescale_thresholds=False)`` short-circuits both ``_node_apply_rescale``
    and ``_node_unapply_rescale``. A left child sharing its parent's feature
    with raw=1.0 should therefore terminate with stored=1.0, not parent's.
    """
    env = env_tree_depth3_no_rescale
    parent_threshold = 0.5
    _build_node_with_dtnode_subenv(
        env, 0, feature_idx=1, threshold_val=parent_threshold
    )
    env.step(env._pad_action((1,), -1))
    env.step((0, 0, 1, 0))  # SAME feature
    env.step((0, 0, -1, 0))
    env.step((0, 1, 1.0, 1))  # raw=1.0 from source
    env.step((0, 1, float("inf"), float("inf")))
    stored = env.node_env.get_threshold(env.state[1])
    assert abs(stored - 1.0) < 1e-10
    # And it explicitly violates the ancestor constraint, confirming that the
    # flag is the only thing keeping the threshold consistent.
    assert stored > parent_threshold


@pytest.mark.repeat(5)
def test__continuous__random_trajectory_respects_ancestor_bounds(env_tree_depth3):
    """
    Sampled continuous-node trees must produce same-feature descendants whose
    rescaled thresholds respect the parent's split direction.
    """
    env = env_tree_depth3
    env.reset()
    env.trajectory_random()
    assert env.done

    for k in range(env.max_nodes):
        if not env._node_is_done(k, env.state):
            continue
        feature_k = env.node_env.get_feature(env.state[k])
        threshold_k = env.node_env.get_threshold(env.state[k])
        if feature_k is None or threshold_k is None:
            continue
        current = k
        while current > 0:
            parent = env.parent_idx(current)
            parent_feature = env.node_env.get_feature(env.state[parent])
            parent_threshold = env.node_env.get_threshold(env.state[parent])
            if parent_feature == feature_k and parent_threshold is not None:
                if env._is_in_left_subtree(k, parent):
                    assert threshold_k <= parent_threshold + 1e-8, (
                        f"Left child {k} threshold {threshold_k} > ancestor "
                        f"{parent} threshold {parent_threshold}"
                    )
                else:
                    assert threshold_k >= parent_threshold - 1e-8, (
                        f"Right child {k} threshold {threshold_k} < ancestor "
                        f"{parent} threshold {parent_threshold}"
                    )
            current = parent


def test__continuous__threshold_bounds_published_after_set_state(env_tree_depth2):
    """
    ``Tree.set_state`` must publish bounds for the active not-done node on the
    shared node_env so that subsequent mask queries are consistent.
    """
    env = env_tree_depth2
    _build_node_with_dtnode_subenv(env, 0, feature_idx=1, threshold_val=0.4)
    state_after_root = copy(env.state)

    # Activate node 1, choose same feature, reach threshold stage.
    env.step(env._pad_action((1,), -1))
    env.step((0, 0, 1, 0))
    env.step((0, 0, -1, 0))
    state_active = copy(env.state)

    # Reset and set the same state via set_state
    env.reset()
    env.set_state(state_active, done=False)
    expected_lower, expected_upper = env._get_threshold_bounds(1, env.state)
    assert env.node_env._tb_lower == expected_lower
    assert env.node_env._tb_upper == expected_upper
    # And these bounds match what the root threshold would dictate.
    root_threshold = env.node_env.get_threshold(state_after_root[0])
    assert expected_upper == root_threshold
