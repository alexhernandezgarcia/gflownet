"""
Decision tree proxy for the composite Tree environment.

Provides two proxy classes:
- ``TreeProxy``: Simple accuracy-based proxy with exponential node-count prior.
- ``CategoricalTreeProxy``: Bayesian proxy using Dirichlet-Multinomial marginal
  log-likelihood with configurable structure priors (BCART, node-count,
  exponential).
"""

import math
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import torch
from scipy.special import gammaln
from torchtyping import TensorType

from gflownet.envs.tree.tree import Tree
from gflownet.proxy.base import Proxy

# =============================================================================
# Shared helper functions for all classes
# =============================================================================


def _route_samples_to_leaves(
    env: Tree, state: Dict, X: npt.NDArray
) -> Dict[int, List[int]]:
    """
    Routes all samples in ``X`` through the decision tree and returns a mapping
    from leaf node index to the list of sample indices that reach it.

    This is the pure-Python equivalent of ``traverse_tree_cython`` applied to
    each sample, adapted for the composite Tree state format.
    """
    node_env = env.node_env
    leaf_samples: Dict[int, List[int]] = {}

    for i, x in enumerate(X):
        k = 0
        while env._node_is_done(k, state):
            feature_idx = node_env.get_feature(state[k])
            threshold = node_env.get_threshold(state[k])
            if feature_idx is None or threshold is None:
                break
            # feature_idx is 1-based (from Choice env)
            if x[feature_idx - 1] <= threshold:
                k = Tree.left_child_idx(k)
            else:
                k = Tree.right_child_idx(k)

        if k not in leaf_samples:
            leaf_samples[k] = []
        leaf_samples[k].append(i)

    return leaf_samples


def _count_internal_nodes(state: Dict) -> int:
    """Returns the number of done (internal) nodes in the tree."""
    return sum(state["_dones"])


# =============================================================================
# TreeProxy (simple accuracy-based)
# =============================================================================


# TODO: Include code to use predicted leaf probabilities
class TreeProxy(Proxy):
    """
    Simple decision tree proxy that uses training accuracy (empirical
    frequency of correct majority-vote predictions) as the likelihood, and an
    exponential penalty on the number of nodes as the prior.

    Since the composite Tree environment only constructs decision (internal)
    nodes, leaf predictions are derived from the majority class of training
    samples that reach each leaf — no leaf-node parameters are needed.

    Energy: ``-accuracy * exp(-beta * n_nodes)`` (lower is better).
    """

    def __init__(self, use_prior: bool = True, beta: float = 1.0, **kwargs):
        """
        Parameters
        ----------
        use_prior : bool
            Whether to multiply accuracy by ``exp(-beta * n_nodes)``.
        beta : float
            Coefficient in the exponential node-count prior.
        """
        super().__init__(**kwargs)
        self.use_prior = use_prior
        self.beta = beta
        self.env = None

    def setup(self, env: Optional[Tree] = None):
        if env is None:
            raise ValueError("TreeProxy.setup requires a Tree environment.")
        self.env = env
        if env.X_train is None or env.y_train is None:
            raise ValueError(
                "TreeProxy requires training data (X_train, y_train) in the env."
            )

    def __call__(self, states: List[Dict]) -> TensorType["batch"]:
        """
        Computes energies for a batch of tree states.

        Energy = ``-accuracy * exp(-beta * n_nodes)`` (lower is better).
        """
        energies = []

        for state in states:
            leaf_samples = _route_samples_to_leaves(self.env, state, self.env.X_train)

            # Majority-vote accuracy
            correct = 0
            for indices in leaf_samples.values():
                labels = self.env.y_train[indices]
                majority_class = int(np.bincount(labels).argmax())
                correct += np.sum(labels == majority_class)

            likelihood = correct / len(self.env.y_train)

            if self.use_prior:
                n_nodes = _count_internal_nodes(state)
                prior = np.exp(-self.beta * n_nodes)
            else:
                prior = 1.0

            energies.append(-likelihood * prior)

        return torch.tensor(energies, dtype=self.float, device=self.device)


# =============================================================================
# CategoricalTreeProxy (Bayesian Dirichlet-Multinomial)
# =============================================================================


class CategoricalTreeProxy(Proxy):
    """
    Bayesian decision tree proxy using the Dirichlet-Multinomial marginal
    log-likelihood (integrating out leaf class probabilities) combined with a
    configurable structure prior.

    If the composite Tree environment does not model leaf class
    probabilities explicitly, this proxy marginalizes them out analytically
    using conjugacy:

        log p(y_leaf | alpha) = log B(alpha + counts) - log B(alpha)

    where ``B`` is the multivariate Beta function and ``counts`` are per-class
    sample counts at each leaf. This is the same computation as in the legacy
    ``compute_log_likelihood_cython``.

    Available priors (set via ``prior_type``):

    - ``"node_count"``: ``log_prior = -(log(4) + log(n_features)) * n_internal``
      Penalizes each internal node by a constant derived from the branching
      factor and number of features.

    - ``"exponential"``: ``log_prior = -beta * n_internal``
      Simple exponential penalty on the number of internal nodes.

    - ``"bcart"``: Bayesian CART prior (Chipman et al., 1998).
      ``p_split(depth) = sigma * (1 + depth)^(-phi)``
      Internal nodes contribute ``log(p_split)``, leaves contribute
      ``log(1 - p_split)``.

    - ``"none"``: No structure prior (log_prior = 0).
    """

    def __init__(
        self,
        prior_type: str = "node_count",
        alpha_type: str = "uniform",
        alpha_value: float = 1.0,
        beta: float = 1.0,
        sigma: float = 0.95,
        phi: float = 2.0,
        normalize_likelihood: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        prior_type : str
            Structure prior type. One of: ``"node_count"``, ``"exponential"``,
            ``"bcart"``, ``"none"``.
        alpha_type : str
            Dirichlet prior initialization over class probabilities. One of:
            - ``"uniform"``: All alpha_k = alpha_value.
            - ``"label_counts"``: Proportional to training class frequencies,
              scaled by alpha_value.
        alpha_value : float
            Concentration parameter (or total concentration for label_counts).
        beta : float
            Penalty coefficient for the ``"exponential"`` prior.
        sigma : float
            Base splitting probability for the BCART prior.
        phi : float
            Depth decay exponent for the BCART prior.
        normalize_likelihood : bool
            If True, divide the marginal log-likelihood by the number of
            training samples so the proxy reports the *average per-sample*
            log-likelihood. This makes the reward-function ``beta`` invariant
            to dataset size. The structure log-prior is left unscaled because
            it is already O(1) in N.
        """
        super().__init__(**kwargs)
        self.prior_type = prior_type.lower()
        self.alpha_type = alpha_type.lower()
        self.alpha_value = alpha_value
        self.beta = beta
        self.sigma = sigma
        self.phi = phi
        self.normalize_likelihood = normalize_likelihood
        self.env = None
        self.alpha = None

    def setup(self, env: Optional[Tree] = None):
        if env is None:
            raise ValueError("CategoricalTreeProxy.setup requires a Tree environment.")
        self.env = env
        if env.X_train is None or env.y_train is None:
            raise ValueError("CategoricalTreeProxy requires training data in the env.")

        self.n_classes = len(np.unique(env.y_train))
        self.n_features = env.X_train.shape[1]
        self.n_train = len(env.y_train)

        # Initialize Dirichlet concentration parameters
        if self.alpha_type == "uniform":
            self.alpha = np.ones(self.n_classes) * self.alpha_value
        elif self.alpha_type == "label_counts":
            class_counts = np.bincount(env.y_train) + 1
            self.alpha = class_counts / class_counts.sum() * self.alpha_value
        else:
            raise ValueError(
                f"Unknown alpha_type '{self.alpha_type}'. "
                f"Expected 'uniform' or 'label_counts'."
            )

    def _compute_log_likelihood(self, state: Dict) -> float:
        """
        Computes the Dirichlet-Multinomial marginal log-likelihood.

        For each leaf: ``log B(alpha + counts) - log B(alpha)``
        where ``log B(a) = sum(gammaln(a_k)) - gammaln(sum(a_k))``.

        This is the pure-Python equivalent of ``compute_log_likelihood_cython``.
        """
        leaf_samples = _route_samples_to_leaves(self.env, state, self.env.X_train)

        # log B(alpha) — constant across leaves
        log_b_alpha = np.sum(gammaln(self.alpha)) - gammaln(np.sum(self.alpha))

        log_likelihood = 0.0
        for indices in leaf_samples.values():
            labels = self.env.y_train[indices]
            counts = np.bincount(labels, minlength=self.n_classes).astype(float)

            # log B(alpha + counts)
            alpha_plus_counts = self.alpha + counts
            log_b_posterior = np.sum(gammaln(alpha_plus_counts)) - gammaln(
                np.sum(alpha_plus_counts)
            )
            log_likelihood += log_b_posterior - log_b_alpha

        return log_likelihood

    def _compute_log_prior(self, state: Dict) -> float:
        """Computes the structure log-prior based on ``self.prior_type``."""
        if self.prior_type == "none":
            return 0.0

        n_internal = _count_internal_nodes(state)

        if self.prior_type == "exponential":
            return -self.beta * n_internal

        if self.prior_type == "node_count":
            return -(math.log(4) + math.log(self.n_features)) * n_internal

        if self.prior_type == "bcart":
            log_prior = 0.0
            for k in range(self.env.max_nodes):
                depth = Tree.node_depth(k)
                p_split = self.sigma * (1 + depth) ** (-self.phi)

                if self.env._node_is_done(k, state):
                    # Internal node
                    log_prior += math.log(p_split)
                else:
                    # Only count positions reachable from a done parent
                    if k == 0:
                        # Root not split: single-leaf tree
                        log_prior += math.log(1 - p_split)
                    else:
                        parent = Tree.parent_idx(k)
                        if self.env._node_is_done(parent, state):
                            log_prior += math.log(1 - p_split)
            return log_prior

        raise ValueError(
            f"Unknown prior_type '{self.prior_type}'. "
            f"Expected 'node_count', 'exponential', 'bcart', or 'none'."
        )

    def __call__(self, states: List[Dict]) -> TensorType["batch"]:
        """
        Computes log-posterior (log-likelihood + log-prior) for each state.

        Returns log-posterior as proxy value (higher is better). Use with
        ``reward_function: "exponential"`` to get ``reward = exp(proxy)``.
        """
        energies = []

        for state in states:
            log_likelihood = self._compute_log_likelihood(state)
            if self.normalize_likelihood:
                log_likelihood = log_likelihood / self.n_train
            log_prior = self._compute_log_prior(state)
            energies.append(log_likelihood + log_prior)

        return torch.tensor(energies, dtype=self.float, device=self.device)
