import common
import pytest

from gflownet.envs.tree.node import DecisionTreeNode
from gflownet.envs.tree.tree import Tree

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env_tree_depth1():
    features = ["feat_a", "feat_b", "feat_c"]
    decision_tree_node = DecisionTreeNode(features=features)
    return Tree(max_depth=1, node_env=decision_tree_node)

@pytest.fixture
def env_tree_depth2():
    features = ["feat_a", "feat_b", "feat_c"]
    decision_tree_node = DecisionTreeNode(features=features)
    return Tree(max_depth=2, node_env=decision_tree_node)

@pytest.fixture
def env_tree_depth3():
    features = ["feat_a", "feat_b", "feat_c"]
    decision_tree_node = DecisionTreeNode(features=features)
    return Tree(max_depth=3, node_env=decision_tree_node)

@pytest.fixture
def env_tree_depth10():
    features = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e", "feat_f", "feat_g"]
    decision_tree_node = DecisionTreeNode(features=features)
    return Tree(max_depth=10, node_env=decision_tree_node)

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
    decision_tree_node = DecisionTreeNode(features=features)
    with pytest.raises(ValueError, match="max_depth >= 1"):
        Tree(max_depth=0, node_env=decision_tree_node)

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
    s, _, v = tree.step((0, 1, 2*threshold_val/5, 1))
    s, _, v = tree.step((0, 1, 2*threshold_val/5, 0))
    s, _, v = tree.step((0, 1, threshold_val/5, 0))
    assert v, f"Failed to set threshold for node {node_idx}"
    # ContinuousCube EOS
    s, _, v = tree.step((0, 1, float("inf"), float("inf")))
    assert v, f"Failed Cube EOS for node {node_idx}"
    # Deactivate
    s, _, v = tree.step(tree._pad_action((node_idx,), -1))
    assert v, f"Failed to deactivate node {node_idx}"


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
    expected_mask = [True, False, False] + [True] * (env.mask_dim - 4 - env.node_env.mask_dim) + [False] + [True] * (env.node_env.mask_dim)
    assert mask == expected_mask

@parametrize_envs
def test__forward_mask__building_mode_delegates_to_node_env(envs, request):
    """While building a node, mask should be in building mode."""
    env = request.getfixturevalue(envs)
    # Activate root
    env.step(env._pad_action((0,), -1))
    mask = env.get_mask_invalid_actions_forward()
    # All meta actions should be masked out, so True
    assert all(mask[:env.n_meta_actions])
    # Node section should equal node_env's own mask for the active substate
    active = env.state["_active"]
    substate = env.state[active]
    expected_node_mask = env.node_env.get_mask_invalid_actions_forward(substate, False)
    assert mask[env.n_meta_actions:] == expected_node_mask
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
    assert all(mask[:env.n_meta_actions])
    # Node section should match node_env forward mask for the substate
    substate = env.state[0]
    expected = env.node_env.get_mask_invalid_actions_forward(substate, False)
    assert mask[env.n_meta_actions:] == expected
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
    assert all(mask[:env.n_meta_actions])
    # Node section should match node_env backward mask for the substate
    substate = env.state[0]
    expected = env.node_env.get_mask_invalid_actions_backward(substate, False)
    assert mask[env.n_meta_actions:] == expected
    # Valid backward actions should be node_env actions (when depadded)
    valid = env.get_valid_actions(backward=True)
    assert len(valid) > 0
    for a in valid:
        assert env._depad_action(a) in env.node_env.action_space

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

# class TestTreeDepth10(common.BaseTestsContinuous):
#     """Common tests for Tree with depth 10."""

#     @pytest.fixture(autouse=True)
#     def setup(self, env_tree_depth10):
#         self.env = env_tree_depth10
#         self.repeats = {
#             "test__reset__state_is_source": 10,
#             "test__forward_actions_have_nonzero_backward_prob": 10,
#             "test__backward_actions_have_nonzero_forward_prob": 10,
#             "test__trajectories_are_reversible": 10,
#             "test__step_random__does_not_sample_invalid_actions_forward": 10,
#             "test__step_random__does_not_sample_invalid_actions_backward": 10,
#             "test__get_mask__is_consistent_regardless_of_inputs": 10,
#             "test__get_valid_actions__is_consistent_regardless_of_inputs": 10,
#             "test__sample_actions__get_logprobs__return_valid_actions_and_logprobs": 10,
#             "test__get_parents_step_get_mask__are_compatible": 10,
#             "test__sample_backwards_reaches_source": 10,
#             "test__state2readable__is_reversible": 20,
#             "test__gflownet_minimal_runs": 3,
#         }
#         self.n_states = {
#             "test__backward_actions_have_nonzero_forward_prob": 3,
#             "test__sample_backwards_reaches_source": 3,
#             "test__get_logprobs__all_finite_in_random_forward_transitions": 10,
#             "test__get_logprobs__all_finite_in_random_backward_transitions": 10,
#         }
#         self.batch_size = {
#             "test__sample_actions__get_logprobs__batched_forward_trajectories": 10,
#             "test__sample_actions__get_logprobs__batched_backward_trajectories": 10,
#             "test__get_logprobs__all_finite_in_accumulated_forward_trajectories": 10,
#             "test__gflownet_minimal_runs": 10,
#         }