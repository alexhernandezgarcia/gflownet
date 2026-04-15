from copy import copy

import common
import pytest
import torch

from gflownet.envs.tree.tree import Tree

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
