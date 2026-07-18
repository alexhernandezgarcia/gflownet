from copy import copy

import common
import numpy as np
import pytest

from gflownet.envs.tree.regression_tree import RegressionTree

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env_regression_depth1():
    features = ["feat_a", "feat_b", "feat_c"]
    return RegressionTree(
        max_depth=1,
        node_kwargs={"features": features},
        mu0=0.0,
        kappa0=1.0,
        alpha0=2.0,
        beta0=1.0,
    )


@pytest.fixture
def env_regression_depth2():
    features = ["feat_a", "feat_b", "feat_c"]
    return RegressionTree(
        max_depth=2,
        node_kwargs={"features": features},
        mu0=0.0,
        kappa0=1.0,
        alpha0=2.0,
        beta0=1.0,
    )


@pytest.fixture
def env_regression_depth3():
    features = ["feat_a", "feat_b", "feat_c"]
    return RegressionTree(
        max_depth=3,
        node_kwargs={"features": features},
        mu0=0.0,
        kappa0=1.0,
        alpha0=2.0,
        beta0=1.0,
    )


@pytest.fixture
def env_regression_depth3_xy():
    """RegressionTree initialized from X_train/y_train arrays with float targets."""
    rng = np.random.default_rng(42)
    X_train = rng.random((80, 4))
    y_train = rng.normal(0.0, 1.0, size=80)
    X_test = rng.random((20, 4))
    y_test = rng.normal(0.0, 1.0, size=20)
    return RegressionTree(
        max_depth=3, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )


@pytest.fixture
def env_regression_separable():
    """
    Depth-1 RegressionTree on data where the target is a step function of the
    first feature: y ~ 0 for x0 <= 0.5, y ~ 10 for x0 > 0.5.
    """
    rng = np.random.default_rng(0)
    X = rng.random((200, 3))
    y = np.where(X[:, 0] <= 0.5, 0.0, 10.0) + rng.normal(0.0, 0.1, size=200)
    return RegressionTree(max_depth=1, X_train=X, y_train=y, scale_data=False)


parametrize_envs = pytest.mark.parametrize(
    "envs",
    [
        "env_regression_depth1",
        "env_regression_depth2",
        "env_regression_depth3",
    ],
)


# ---------------------------------------------------------------------------
# Helper to build a node via a sequence of forward steps (continuous node)
# ---------------------------------------------------------------------------


def _build_node(tree, node_idx, feature_idx, threshold_val):
    """
    Builds a complete node in the tree via forward actions, assuming subenvs
    of type gflownet.envs.tree.node (Choice + ContinuousCube):
    activate -> choose feature -> Choice EOS -> set threshold -> Cube EOS
    -> deactivate.
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


@parametrize_envs
def test__environment__initializes_properly(envs, request):
    env = request.getfixturevalue(envs)
    assert True


@parametrize_envs
def test__environment__is_continuous(envs, request):
    env = request.getfixturevalue(envs)
    assert env.continuous is True


@parametrize_envs
def test__environment__mdp_matches_classification_tree(envs, request):
    """The construction MDP (source, action space, masks) is inherited unchanged."""
    env = request.getfixturevalue(envs)
    assert env.source["_active"] == -1
    assert env.source["_dones"] == [0] * env.max_nodes
    assert env.max_nodes == 2**env.max_depth - 1
    assert len(env.action_space) == env.max_nodes + 1 + len(env.node_env.action_space)


def test__init__targets_stay_float():
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y = np.array([1.7, -0.3, 2.9])
    env = RegressionTree(max_depth=2, X_train=X, y_train=y, scale_data=False)
    assert env.y_train.dtype == float
    np.testing.assert_allclose(env.y_train, y)


def test__init__test_targets_stay_float():
    rng = np.random.default_rng(1)
    X_train = rng.random((30, 2))
    y_train = rng.normal(size=30)
    X_test = rng.random((10, 2))
    y_test = rng.normal(size=10)
    env = RegressionTree(
        max_depth=2, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )
    assert env.y_test.dtype == float
    np.testing.assert_allclose(env.y_test, y_test)


def test__init__empirical_bayes_prior_defaults():
    rng = np.random.default_rng(2)
    X = rng.random((50, 2))
    y = rng.normal(3.0, 2.0, size=50)
    env = RegressionTree(max_depth=2, X_train=X, y_train=y)
    assert env.mu0 == pytest.approx(float(np.mean(y)))
    assert env.beta0 == pytest.approx(float(np.var(y)))
    assert env.kappa0 > 0
    assert env.alpha0 > 0


def test__init__invalid_prior_raises():
    features = ["feat_a"]
    with pytest.raises(ValueError, match="NIG prior"):
        RegressionTree(max_depth=1, node_kwargs={"features": features}, kappa0=0.0)
    with pytest.raises(ValueError, match="NIG prior"):
        RegressionTree(max_depth=1, node_kwargs={"features": features}, alpha0=-1.0)


# ===========================================================================
# MDP sanity tests (inherited behavior)
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


@pytest.mark.repeat(3)
@parametrize_envs
def test__trajectory_random__forward_then_backward_reaches_source(envs, request):
    env = request.getfixturevalue(envs)
    env.reset()
    env.trajectory_random()
    assert env.done
    step_count = 0
    while not env.is_source():
        if env.done:
            a = env.eos
        else:
            va = env.get_valid_actions(backward=True)
            assert len(va) > 0
            a = va[0]
        _, _, v = env.step_backwards(a)
        assert v
        step_count += 1
        assert step_count <= env.max_traj_length + 1
    assert env.is_source()


@parametrize_envs
def test__full_trajectory__root_only(envs, request):
    env = request.getfixturevalue(envs)
    _build_node(env, 0, 1, 0.5)
    _, _, valid = env.step(env.eos)
    assert valid and env.done
    assert env._node_is_done(0, env.state)


# ===========================================================================
# Normal-Inverse-Gamma leaf model tests
# ===========================================================================


def test__leaf_posterior__matches_closed_form(env_regression_depth1):
    """mu0=0, kappa0=1, alpha0=2, beta0=1; y = [1, 2, 3]."""
    env = env_regression_depth1
    y_leaf = np.array([1.0, 2.0, 3.0])
    mu_n, kappa_n, alpha_n, beta_n = env._leaf_posterior(y_leaf)
    # kappa_n = 1 + 3 = 4; mu_n = (0 + 3*2)/4 = 1.5; alpha_n = 2 + 1.5 = 3.5
    # beta_n = 1 + SSE/2 + kappa0*n*(mean - mu0)^2 / (2*kappa_n)
    #        = 1 + 2/2 + 1*3*4/(2*4) = 3.5
    assert kappa_n == pytest.approx(4.0)
    assert mu_n == pytest.approx(1.5)
    assert alpha_n == pytest.approx(3.5)
    assert beta_n == pytest.approx(3.5)


def test__leaf_posterior__empty_leaf_returns_prior(env_regression_depth1):
    env = env_regression_depth1
    mu_n, kappa_n, alpha_n, beta_n = env._leaf_posterior(np.array([]))
    assert (mu_n, kappa_n, alpha_n, beta_n) == (
        env.mu0,
        env.kappa0,
        env.alpha0,
        env.beta0,
    )


def test__leaf_log_marginal__empty_leaf_is_zero(env_regression_depth1):
    env = env_regression_depth1
    assert env._leaf_log_marginal(np.array([])) == 0.0


def test__leaf_log_marginal__single_point_is_student_t(env_regression_depth1):
    """
    For a single observation, the NIG marginal is the Student-t density
    t_{2*alpha0}(y | mu0, beta0 * (kappa0 + 1) / (alpha0 * kappa0)).
    """
    env = env_regression_depth1
    y = 0.7
    log_m = env._leaf_log_marginal(np.array([y]))
    # Student-t log-density with df = 2*alpha0, loc = mu0,
    # scale^2 = beta0 * (kappa0 + 1) / (alpha0 * kappa0)
    df = 2 * env.alpha0
    scale2 = env.beta0 * (env.kappa0 + 1) / (env.alpha0 * env.kappa0)
    import math

    z2 = (y - env.mu0) ** 2 / scale2
    expected = (
        math.lgamma((df + 1) / 2)
        - math.lgamma(df / 2)
        - 0.5 * math.log(df * math.pi * scale2)
        - (df + 1) / 2 * math.log(1 + z2 / df)
    )
    assert log_m == pytest.approx(expected, rel=1e-10)


def test__log_marginal_likelihood__requires_data(env_regression_depth1):
    env = env_regression_depth1
    _build_node(env, 0, 1, 0.5)
    with pytest.raises(ValueError, match="requires training data"):
        env.log_marginal_likelihood()


def test__log_marginal_likelihood__is_finite(env_regression_separable):
    env = env_regression_separable
    _build_node(env, 0, 1, 0.5)
    log_ml = env.log_marginal_likelihood()
    assert np.isfinite(log_ml)


def test__log_marginal_likelihood__prefers_informative_split(
    env_regression_separable,
):
    """A split on the feature that determines y beats a split on a noise feature."""
    env = env_regression_separable
    # Tree A: split on feature 1 (column 0, the informative one) at 0.5
    _build_node(env, 0, 1, 0.5)
    env.step(env.eos)
    state_informative = copy(env.state)

    env.reset()
    # Tree B: split on feature 2 (column 1, pure noise) at 0.5
    _build_node(env, 0, 2, 0.5)
    env.step(env.eos)
    state_noise = copy(env.state)

    log_ml_informative = env.log_marginal_likelihood(state_informative)
    log_ml_noise = env.log_marginal_likelihood(state_noise)
    assert log_ml_informative > log_ml_noise


def test__log_posterior_unnorm__applies_structure_prior(env_regression_separable):
    env = env_regression_separable
    _build_node(env, 0, 1, 0.5)
    env.step(env.eos)
    state = copy(env.state)
    n_nodes = sum(state["_dones"])
    assert n_nodes == 1
    log_ml = env.log_marginal_likelihood(state)
    beta = 2.5
    assert env.log_posterior_unnorm(state, beta=beta) == pytest.approx(
        log_ml - beta * n_nodes
    )


# ===========================================================================
# Posterior-predictive sampling and prediction tests
# ===========================================================================


def test__sample_leaf_params__concentrates_on_leaf_means(env_regression_separable):
    """With many samples per leaf, sampled leaf means are close to data means."""
    env = env_regression_separable
    _build_node(env, 0, 1, 0.5)
    env.step(env.eos)
    state = copy(env.state)

    rng = np.random.default_rng(0)
    leaf_means = env._sample_leaf_params(state, rng)
    # Two leaves: left (y ~ 0) and right (y ~ 10)
    assert len(leaf_means) == 2
    sampled = sorted(leaf_means.values())
    assert sampled[0] == pytest.approx(0.0, abs=0.5)
    assert sampled[1] == pytest.approx(10.0, abs=0.5)


def test__predict__unseen_leaf_gets_prior_mean(env_regression_separable):
    env = env_regression_separable
    _build_node(env, 0, 1, 0.5)
    env.step(env.eos)
    state = copy(env.state)
    # Leaf means only for the left leaf; the right leaf is "unseen"
    left_leaf = env.left_child_idx(0)
    preds = env._predict(state, {left_leaf: -3.0}, env.X_train)
    right_rows = env.X_train[:, 0] > 0.5
    np.testing.assert_allclose(preds[~right_rows], -3.0)
    np.testing.assert_allclose(preds[right_rows], env.mu0)


# ===========================================================================
# test() evaluation tests
# ===========================================================================


def test__test__returns_regression_metrics(env_regression_depth3_xy):
    env = env_regression_depth3_xy
    states = []
    for _ in range(3):
        env.reset()
        env.trajectory_random()
        states.append(copy(env.state))
    result = env.test(states, top_k_trees=2, plot_top_k=False, seed=0)
    metrics = result["metrics"]
    for key in [
        "mean_n_nodes",
        "train_mean_tree_mse",
        "train_mean_tree_rmse",
        "train_mean_tree_mae",
        "train_mean_tree_r2",
        "train_forest_mse",
        "train_forest_r2",
        "train_top_k_forest_mse",
        "train_top_1_forest_mse",
        "test_mean_tree_mse",
        "test_forest_mse",
        "test_top_k_forest_mse",
        "test_top_1_forest_mse",
    ]:
        assert key in metrics, f"Missing metric {key}"
        assert np.isfinite(metrics[key]), f"Metric {key} is not finite"
    assert metrics["train_mean_tree_mse"] >= 0.0
    assert metrics["train_mean_tree_rmse"] == pytest.approx(
        np.sqrt(metrics["train_mean_tree_mse"]), abs=1.0
    )


def test__test__good_split_yields_low_mse(env_regression_separable):
    env = env_regression_separable
    _build_node(env, 0, 1, 0.5)
    env.step(env.eos)
    state = copy(env.state)
    result = env.test([state], seed=0)
    metrics = result["metrics"]
    # y jumps by 10 between the two leaves; the correct split explains almost
    # all the variance (residual noise std = 0.1)
    assert metrics["train_forest_mse"] < 1.0
    assert metrics["train_forest_r2"] > 0.9
    assert metrics["mean_n_nodes"] == 1.0


def test__test__empty_samples_returns_empty(env_regression_depth3_xy):
    env = env_regression_depth3_xy
    result = env.test([])
    assert result == {"metrics": {}, "figs": {}}


def test__test__no_data_returns_empty(env_regression_depth1):
    env = env_regression_depth1
    env.trajectory_random()
    result = env.test([copy(env.state)])
    assert result == {"metrics": {}, "figs": {}}


# ===========================================================================
# Common base tests
# ===========================================================================


class TestRegressionTreeDepth3Xy(common.BaseTestsContinuous):
    """Common tests for RegressionTree with depth 3 initialized from arrays."""

    @pytest.fixture(autouse=True)
    def setup(self, env_regression_depth3_xy):
        self.env = env_regression_depth3_xy
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