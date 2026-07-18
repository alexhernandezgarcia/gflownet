"""
Tests for the NormalGammaTreeProxy (Bayesian regression tree proxy).

The closed-form Normal-Inverse-Gamma marginal likelihood is verified against
an independent computation: the chain-rule product of Student-t posterior
predictive densities, evaluated with scipy.stats.t.

The node-building helper is duplicated from tests/gflownet/envs/test_tree.py
on purpose: this file must stay self-contained so that the stacked feature
branch only adds files.
"""

import math
from copy import copy

import numpy as np
import pytest
import torch
from scipy.stats import t as student_t

from gflownet.envs.tree.regression_tree import RegressionTree
from gflownet.proxy.regression_tree import NormalGammaTreeProxy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_node_with_dtnode_subenv(tree, node_idx, feature_idx, threshold_val):
    """
    Builds a complete node in the tree via forward actions (continuous node):
    activate -> choose feature -> Choice EOS -> set threshold -> Cube EOS -> deactivate.
    """
    s, _, v = tree.step(tree._pad_action((node_idx,), -1))
    assert v, f"Failed to activate node {node_idx}"
    s, _, v = tree.step((0, 0, feature_idx, 0))
    assert v, f"Failed to choose feature {feature_idx} for node {node_idx}"
    s, _, v = tree.step((0, 0, -1, 0))
    assert v, f"Failed Choice EOS for node {node_idx}"
    s, _, v = tree.step((0, 1, 2 * threshold_val / 5, 1))
    s, _, v = tree.step((0, 1, 2 * threshold_val / 5, 0))
    s, _, v = tree.step((0, 1, threshold_val / 5, 0))
    assert v, f"Failed to set threshold for node {node_idx}"
    s, _, v = tree.step((0, 1, float("inf"), float("inf")))
    assert v, f"Failed Cube EOS for node {node_idx}"
    s, _, v = tree.step(tree._pad_action((node_idx,), -1))
    assert v, f"Failed to deactivate node {node_idx}"


def _nig_marginal_ll_sequential(y, mu_0, kappa_0, alpha_0, beta_0):
    """
    Computes log p(y_1, ..., y_n) under the Normal-Inverse-Gamma model via the
    chain rule: log p(y) = sum_i log p(y_i | y_1, ..., y_{i-1}), where each
    posterior predictive is a Student-t density:

        y_new | y_1..m ~ t_{2 alpha_m}(mu_m, beta_m (kappa_m + 1) / (alpha_m kappa_m))

    This is an independent check of the closed-form batch expression used by
    the proxy.
    """
    mu, kappa, alpha, beta = mu_0, kappa_0, alpha_0, beta_0
    log_likelihood = 0.0
    for yi in np.asarray(y, dtype=float):
        df = 2.0 * alpha
        scale = math.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        log_likelihood += float(student_t.logpdf(yi, df, loc=mu, scale=scale))
        # Single-observation posterior update
        beta = beta + 0.5 * kappa * (yi - mu) ** 2 / (kappa + 1.0)
        mu = (kappa * mu + yi) / (kappa + 1.0)
        kappa = kappa + 1.0
        alpha = alpha + 0.5
    return log_likelihood


def _make_step_data(n=100, noise=0.0, seed=42):
    """Step data: y = 1.0 for x0 <= 0.5, y = 5.0 otherwise. X in [0, 1]."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2))
    y = np.where(X[:, 0] <= 0.5, 1.0, 5.0).astype(float)
    if noise > 0.0:
        y = y + rng.normal(0.0, noise, size=n)
    return X, y


def _make_env_with_split(threshold, n=60, noise=0.2, seed=7):
    """Returns (env, terminal state) with a single root split at ``threshold``."""
    X, y = _make_step_data(n=n, noise=noise, seed=seed)
    env = RegressionTree(max_depth=2, X_train=X, y_train=y, scale_data=False)
    _build_node_with_dtnode_subenv(env, 0, 1, threshold)
    env.step(env.eos)
    return env, copy(env.state)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env_and_state():
    return _make_env_with_split(threshold=0.5)


# ===========================================================================
# Setup tests
# ===========================================================================


def test__setup__resolves_data_driven_defaults(env_and_state):
    env, _ = env_and_state
    proxy = NormalGammaTreeProxy(alpha_0=2.0)
    proxy.setup(env)
    assert proxy._mu_0 == pytest.approx(float(np.mean(env.y_train)))
    # For alpha_0 = 2: beta_0 = (alpha_0 - 1) * var = var
    assert proxy._beta_0 == pytest.approx(float(np.var(env.y_train)))
    assert proxy.n_features == 2
    assert proxy.n_train == len(env.y_train)


def test__setup__respects_explicit_hyperparams(env_and_state):
    env, _ = env_and_state
    proxy = NormalGammaTreeProxy(mu_0=1.5, kappa_0=2.0, alpha_0=3.0, beta_0=4.0)
    proxy.setup(env)
    assert proxy._mu_0 == 1.5
    assert proxy._kappa_0 == 2.0
    assert proxy._alpha_0 == 3.0
    assert proxy._beta_0 == 4.0


def test__setup__raises_without_env_or_data():
    proxy = NormalGammaTreeProxy()
    with pytest.raises(ValueError):
        proxy.setup(None)
    env_no_data = RegressionTree(
        max_depth=2, node_kwargs={"features": ["feat_a", "feat_b"]}
    )
    with pytest.raises(ValueError):
        proxy.setup(env_no_data)


def test__setup__raises_on_invalid_hyperparams(env_and_state):
    env, _ = env_and_state
    with pytest.raises(ValueError):
        NormalGammaTreeProxy(kappa_0=0.0).setup(env)
    with pytest.raises(ValueError):
        NormalGammaTreeProxy(alpha_0=-1.0).setup(env)


# ===========================================================================
# Marginal log-likelihood tests
# ===========================================================================


def test__log_likelihood__matches_sequential_student_t(env_and_state):
    """
    The closed-form batch NIG marginal must equal the chain-rule product of
    Student-t posterior predictives (independent computation via scipy).
    """
    env, state = env_and_state
    mu_0, kappa_0, alpha_0, beta_0 = 0.0, 1.0, 3.0, 2.0
    proxy = NormalGammaTreeProxy(
        prior_type="none", mu_0=mu_0, kappa_0=kappa_0, alpha_0=alpha_0, beta_0=beta_0
    )
    proxy.setup(env)

    # Route the training samples manually with the actual stored threshold
    feature_idx = env.node_env.get_feature(state[0])
    threshold = env.node_env.get_threshold(state[0])
    x = env.X_train[:, feature_idx - 1]
    y_left = env.y_train[x <= threshold]
    y_right = env.y_train[x > threshold]
    assert len(y_left) > 0 and len(y_right) > 0

    expected = _nig_marginal_ll_sequential(
        y_left, mu_0, kappa_0, alpha_0, beta_0
    ) + _nig_marginal_ll_sequential(y_right, mu_0, kappa_0, alpha_0, beta_0)

    computed = proxy._compute_log_likelihood(state)
    assert computed == pytest.approx(expected, rel=1e-10)


def test__log_likelihood__good_split_beats_bad_split():
    """
    On step data with the jump at 0.5, the marginal likelihood of a tree
    splitting at 0.5 must exceed that of a tree splitting at 0.9.
    """
    env_good, state_good = _make_env_with_split(threshold=0.5)
    env_bad, state_bad = _make_env_with_split(threshold=0.9)

    proxy_good = NormalGammaTreeProxy(prior_type="none")
    proxy_good.setup(env_good)
    proxy_bad = NormalGammaTreeProxy(prior_type="none")
    proxy_bad.setup(env_bad)

    ll_good = proxy_good._compute_log_likelihood(state_good)
    ll_bad = proxy_bad._compute_log_likelihood(state_bad)
    assert ll_good > ll_bad


# ===========================================================================
# Structure prior and __call__ tests
# ===========================================================================


def test__call__returns_finite_log_posterior(env_and_state):
    env, state = env_and_state
    proxy = NormalGammaTreeProxy(prior_type="node_count")
    proxy.setup(env)
    values = proxy([state, state])
    assert isinstance(values, torch.Tensor)
    assert values.shape == (2,)
    assert torch.isfinite(values).all()
    assert values[0] == values[1]


def test__call__equals_likelihood_plus_prior(env_and_state):
    env, state = env_and_state
    proxy = NormalGammaTreeProxy(prior_type="node_count")
    proxy.setup(env)
    expected = proxy._compute_log_likelihood(state) + proxy._compute_log_prior(state)
    assert proxy([state])[0].item() == pytest.approx(expected, rel=1e-6)


def test__node_count_prior__penalizes_internal_nodes(env_and_state):
    env, state = env_and_state
    proxy = NormalGammaTreeProxy(prior_type="node_count")
    proxy.setup(env)
    # Single internal node: log_prior = -(log 4 + log n_features)
    expected = -(math.log(4) + math.log(env.X_train.shape[1]))
    assert proxy._compute_log_prior(state) == pytest.approx(expected)

    proxy_none = NormalGammaTreeProxy(prior_type="none")
    proxy_none.setup(env)
    assert proxy_none._compute_log_prior(state) == 0.0


def test__normalize_likelihood__divides_by_n_train(env_and_state):
    env, state = env_and_state
    proxy = NormalGammaTreeProxy(prior_type="none", normalize_likelihood=True)
    proxy.setup(env)
    raw = proxy._compute_log_likelihood(state)
    assert proxy([state])[0].item() == pytest.approx(raw / proxy.n_train, rel=1e-6)
