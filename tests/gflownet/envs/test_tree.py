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

class TestTreeDepth10(common.BaseTestsContinuous):
    """Common tests for Tree with depth 10."""

    @pytest.fixture(autouse=True)
    def setup(self, env_tree_depth10):
        self.env = env_tree_depth10
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