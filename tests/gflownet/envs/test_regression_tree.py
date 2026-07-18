"""
Tests for the RegressionTree environment.

The construction MDP is fully inherited from Tree (and extensively covered by
test_tree.py), so these tests focus on what is regression-specific:

- Continuous targets are preserved (not cast to int) for all init paths.
- The inherited construction mechanics still work on the subclass (smoke).
- The regression evaluation pass (``test()``) reports RMSE / R2 metrics.

The helper to build nodes via forward steps is duplicated from test_tree.py on
purpose: this file must stay self-contained so that the stacked feature branch
only adds files.
"""

from copy import copy

import common
import numpy as np
import pandas as pd
import pytest

from gflownet.envs.tree.regression_tree import RegressionTree

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env_reg_depth1():
    features = ["feat_a", "feat_b", "feat_c"]
    return RegressionTree(max_depth=1, node_kwargs={"features": features})


@pytest.fixture
def env_reg_depth2():
    features = ["feat_a", "feat_b", "feat_c"]
    return RegressionTree(max_depth=2, node_kwargs={"features": features})


@pytest.fixture
def env_reg_depth3():
    features = ["feat_a", "feat_b", "feat_c"]
    return RegressionTree(max_depth=3, node_kwargs={"features": features})


def _make_step_data(n=100, noise=0.0, seed=42):
    """
    Synthetic regression data with a step at x0 = 0.5:
    y = 1.0 for x0 <= 0.5, y = 5.0 otherwise (plus optional Gaussian noise).
    Features are already in [0, 1].
    """
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2))
    y = np.where(X[:, 0] <= 0.5, 1.0, 5.0).astype(float)
    if noise > 0.0:
        y = y + rng.normal(0.0, noise, size=n)
    return X, y


@pytest.fixture
def env_reg_depth2_xy():
    """RegressionTree initialized from X_train/y_train numpy arrays."""
    X_train, y_train = _make_step_data(n=80, seed=42)
    X_test, y_test = _make_step_data(n=20, seed=43)
    return RegressionTree(
        max_depth=2,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scale_data=False,
    )


# ---------------------------------------------------------------------------
# Helper to build a node via a sequence of forward steps (continuous node,
# duplicated from test_tree.py to keep this file self-contained)
# ---------------------------------------------------------------------------


def _build_node_with_dtnode_subenv(tree, node_idx, feature_idx, threshold_val):
    """
    Builds a complete node in the tree via forward actions, assuming subenvs of
    type gflownet.envs.tree.node.py:
    activate -> choose feature -> Choice EOS -> set threshold -> Cube EOS -> deactivate.
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


# ===========================================================================
# Initialization tests
# ===========================================================================

parametrize_envs = pytest.mark.parametrize(
    "envs",
    [
        "env_reg_depth1",
        "env_reg_depth2",
        "env_reg_depth3",
    ],
)


@parametrize_envs
def test__environment__initializes_properly(envs, request):
    env = request.getfixturevalue(envs)
    assert True


@parametrize_envs
def test__environment__is_continuous(envs, request):
    env = request.getfixturevalue(envs)
    assert env.continuous is True


@pytest.mark.parametrize(
    "envs, max_depth, max_nodes",
    [
        ("env_reg_depth1", 1, 1),
        ("env_reg_depth2", 2, 3),
        ("env_reg_depth3", 3, 7),
    ],
)
def test__max_nodes__is_correct(envs, max_depth, max_nodes, request):
    env = request.getfixturevalue(envs)
    assert env.max_depth == max_depth
    assert env.max_nodes == max_nodes


@parametrize_envs
def test__is_source__returns_true_at_init(envs, request):
    env = request.getfixturevalue(envs)
    assert env.is_source()


# ===========================================================================
# Continuous target preservation tests
# ===========================================================================


def test__init_from_arrays__keeps_continuous_targets():
    X_train = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    y_train = np.array([0.5, 2.7, -1.3, 4.9])
    X_test = np.array([[0.2, 0.3], [0.6, 0.7]])
    y_test = np.array([1.5, -0.5])
    env = RegressionTree(
        max_depth=2,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scale_data=False,
    )
    assert env.y_train.dtype == float
    assert env.y_test.dtype == float
    # A plain Tree would truncate these values to [0, 2, -1, 4]
    np.testing.assert_array_equal(env.y_train, y_train)
    np.testing.assert_array_equal(env.y_test, y_test)


def test__init_from_csv__keeps_continuous_targets(tmp_path):
    X, y = _make_step_data(n=30, noise=0.1, seed=0)
    df = pd.DataFrame(X, columns=["feat_a", "feat_b"])
    df["target"] = y
    csv_path = tmp_path / "regression_data.csv"
    df.to_csv(csv_path, index=False)

    env = RegressionTree(max_depth=2, data_path=str(csv_path))
    assert env.y_train.dtype == float
    np.testing.assert_allclose(env.y_train, y)
    # Feature names are read from the CSV headers
    assert env.node_env.features == ["feat_a", "feat_b"]


def test__init_from_csv_with_split__keeps_continuous_targets(tmp_path):
    X, y = _make_step_data(n=30, noise=0.1, seed=1)
    df = pd.DataFrame(X, columns=["feat_a", "feat_b"])
    df["target"] = y
    df["Split"] = ["train"] * 20 + ["test"] * 10
    csv_path = tmp_path / "regression_data_split.csv"
    df.to_csv(csv_path, index=False)

    env = RegressionTree(max_depth=2, data_path=str(csv_path))
    assert env.y_train.dtype == float
    assert env.y_test.dtype == float
    np.testing.assert_allclose(env.y_train, y[:20])
    np.testing.assert_allclose(env.y_test, y[20:])


def test__init_without_data__has_no_targets(env_reg_depth2):
    assert env_reg_depth2.X_train is None
    assert env_reg_depth2.y_train is None


# ===========================================================================
# Inherited construction mechanics (smoke tests)
# ===========================================================================


@pytest.mark.repeat(5)
@parametrize_envs
def test__trajectory_random__reaches_done(envs, request):
    env = request.getfixturevalue(envs)
    env.reset()
    env.trajectory_random()
    assert env.done
    assert env._is_idle(env.state)
    assert env._node_is_done(0, env.state)


def test__full_trajectory__forward_then_backward_reaches_source(env_reg_depth2):
    env = env_reg_depth2
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    _build_node_with_dtnode_subenv(env, 1, 2, 0.3)
    _, _, valid = env.step(env.eos)
    assert valid and env.done
    while not env.is_source():
        if env.done:
            a = env.eos
        else:
            va = env.get_valid_actions(backward=True)
            assert len(va) > 0
            a = va[0]
        _, _, v = env.step_backwards(a)
        assert v
    assert env.is_source()


# ===========================================================================
# NIG posterior helper tests
# ===========================================================================


def test__nig_posterior__is_correct_for_simple_case():
    # Two observations, standard prior
    y = np.array([1.0, 3.0])
    mu_n, kappa_n, alpha_n, beta_n = RegressionTree._nig_posterior(
        y, mu_0=0.0, kappa_0=1.0, alpha_0=2.0, beta_0=1.0
    )
    # kappa_n = 1 + 2, alpha_n = 2 + 1
    assert kappa_n == 3.0
    assert alpha_n == 3.0
    # mu_n = (1 * 0 + 2 * 2) / 3
    assert mu_n == pytest.approx(4.0 / 3.0)
    # beta_n = 1 + 0.5 * ss + 0.5 * kappa_0 * n * (ybar - mu_0)^2 / kappa_n
    #        = 1 + 0.5 * 2 + 0.5 * 1 * 2 * 4 / 3
    assert beta_n == pytest.approx(1.0 + 1.0 + 4.0 / 3.0)


def test__resolve_nig_params__data_driven_defaults(env_reg_depth2_xy):
    env = env_reg_depth2_xy
    mu_0, kappa_0, alpha_0, beta_0 = env._resolve_nig_params(
        mu_0=None, kappa_0=0.1, alpha_0=2.0, beta_0=None
    )
    assert mu_0 == pytest.approx(float(np.mean(env.y_train)))
    # For alpha_0 = 2: beta_0 = (alpha_0 - 1) * var = var
    assert beta_0 == pytest.approx(float(np.var(env.y_train)))


# ===========================================================================
# Evaluation (test method) tests
# ===========================================================================


def test__test__returns_regression_metrics(env_reg_depth2_xy):
    env = env_reg_depth2_xy
    # Build a single-node tree splitting on feature 1 at the true step (0.5)
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    env.step(env.eos)
    state = copy(env.state)

    result = env.test([state], top_k_trees=1, plot_top_k=False, seed=0)
    metrics = result["metrics"]

    for key in [
        "mean_n_nodes",
        "train_mean_tree_rmse",
        "train_mean_tree_r2",
        "train_forest_rmse",
        "train_forest_r2",
        "train_top_1_forest_rmse",
        "test_mean_tree_rmse",
        "test_forest_rmse",
        "test_forest_r2",
    ]:
        assert key in metrics, f"Missing metric {key}"
        assert np.isfinite(metrics[key]), f"Metric {key} is not finite"

    assert metrics["mean_n_nodes"] == 1.0
    assert metrics["train_forest_rmse"] >= 0.0
    assert metrics["train_forest_r2"] <= 1.0


def test__test__good_split_beats_bad_split(env_reg_depth2_xy):
    env = env_reg_depth2_xy

    # Tree splitting at the true step location (0.5)
    env.reset()
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    env.step(env.eos)
    state_good = copy(env.state)

    # Tree splitting far from the true step location (0.9)
    env.reset()
    _build_node_with_dtnode_subenv(env, 0, 1, 0.9)
    env.step(env.eos)
    state_bad = copy(env.state)

    metrics_good = env.test([state_good], seed=0)["metrics"]
    metrics_bad = env.test([state_bad], seed=0)["metrics"]

    assert (
        metrics_good["train_forest_rmse"] < metrics_bad["train_forest_rmse"]
    ), "Tree with the correct split should have lower train RMSE"
    # The step in y is 4.0 (from 1.0 to 5.0); the correct split should be
    # nearly perfect (noiseless data), the bad one far off.
    assert metrics_good["train_forest_rmse"] < 1.0
    assert metrics_bad["train_forest_rmse"] > 1.0


def test__test__empty_samples_returns_empty(env_reg_depth2_xy):
    result = env_reg_depth2_xy.test([])
    assert result == {"metrics": {}, "figs": {}}


def test__test__without_data_returns_empty(env_reg_depth2):
    env = env_reg_depth2
    _build_node_with_dtnode_subenv(env, 0, 1, 0.5)
    env.step(env.eos)
    result = env.test([copy(env.state)])
    assert result == {"metrics": {}, "figs": {}}


# ===========================================================================
# Common base tests from common.py
# ===========================================================================


class TestRegressionTreeDepth2(common.BaseTestsContinuous):
    """Common tests for RegressionTree with depth 2."""

    @pytest.fixture(autouse=True)
    def setup(self, env_reg_depth2):
        self.env = env_reg_depth2
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
