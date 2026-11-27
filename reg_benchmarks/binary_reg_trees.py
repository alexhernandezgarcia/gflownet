"""
Helpers for the Bayesian CART regression baselines that operate on
quantile-binarized features (run_bcart.py): parsing of the tree encoding
returned by the tree_smc samplers, Normal-Inverse-Gamma (NIG) leaf
posteriors, and posterior-mean predictions.

Conventions match class_baselines/binary_trees.py: binarized feature j is the
test x[feat] <= threshold; at an internal node, points with feature value
False (0) go left and True (1) go right. tree_smc encodes trees as a dict
node_info {node_id: (feature, split, ...)} plus a list of leaf ids, with the
children of node i at 2i+1 (left) and 2i+2 (right).

Leaf model (the same NIG prior as gflownet.proxy.regression_tree):
    mu | sigma^2 ~ N(mu_0, sigma^2 / kappa_0),  sigma^2 ~ InvGamma(alpha_0, beta_0)
The posterior-mean prediction of a leaf with n points of sum s is
    mu_n = (kappa_0 * mu_0 + s) / (kappa_0 + n),
which is also the mean of the Student-t posterior predictive, i.e. the same
point prediction RegressionTree.test uses for DT-GFN.
"""

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln


@dataclass(frozen=True)
class NIGPrior:
    mu_0: float
    kappa_0: float
    alpha_0: float
    beta_0: float


def nig_log_marginal(
    sum_y: float, sum_y2: float, n: int, prior: NIGPrior
) -> float:
    """
    Log marginal likelihood of n observations (sufficient statistics sum_y,
    sum_y2) under the NIG leaf model, with mu and sigma^2 integrated out.
    Same closed form as tree_smc.tree_utils.compute_ng_normalizer.
    """
    if n == 0:
        return 0.0
    y_bar = sum_y / n
    kappa_n = prior.kappa_0 + n
    alpha_n = prior.alpha_0 + n / 2.0
    beta_n = (
        prior.beta_0
        + 0.5 * (sum_y2 - n * y_bar**2)
        + 0.5 * prior.kappa_0 * n / kappa_n * (y_bar - prior.mu_0) ** 2
    )
    return float(
        gammaln(alpha_n)
        - gammaln(prior.alpha_0)
        + prior.alpha_0 * np.log(prior.beta_0)
        - alpha_n * np.log(beta_n)
        + 0.5 * (np.log(prior.kappa_0) - np.log(kappa_n))
        - 0.5 * n * np.log(2 * np.pi)
    )


class BinaryRegTree:
    """Regression tree over binary features with NIG posterior-mean leaves."""

    def __init__(
        self,
        left: "BinaryRegTree" = None,
        right: "BinaryRegTree" = None,
        feature: int = None,
    ):
        assert (left is None) == (right is None) == (feature is None)
        self.left = left
        self.right = right
        self.feature = feature
        self.n = 0
        self.sum_y = 0.0
        self.sum_y2 = 0.0

    def is_leaf(self) -> bool:
        return self.feature is None

    def size(self) -> int:
        """Total node count (internal nodes + leaves)."""
        return 1 if self.is_leaf() else 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        if self.is_leaf():
            return 0
        return 1 + max(self.left.depth(), self.right.depth())

    def leaves(self):
        if self.is_leaf():
            return [self]
        return self.left.leaves() + self.right.leaves()

    def fit_stats(self, B: npt.NDArray, y: npt.NDArray) -> "BinaryRegTree":
        """Routes the train split and stores the leaf sufficient statistics."""
        self.n = int(len(y))
        self.sum_y = float(np.sum(y))
        self.sum_y2 = float(np.sum(y**2))
        if not self.is_leaf():
            right = B[:, self.feature]
            self.left.fit_stats(B[~right], y[~right])
            self.right.fit_stats(B[right], y[right])
        return self

    def log_marginal_likelihood(self, prior: NIGPrior) -> float:
        return sum(nig_log_marginal(l.sum_y, l.sum_y2, l.n, prior) for l in self.leaves())

    def predict(self, B: npt.NDArray, prior: NIGPrior) -> npt.NDArray:
        """NIG posterior-mean prediction of every row of B."""
        if self.is_leaf():
            mu_n = (prior.kappa_0 * prior.mu_0 + self.sum_y) / (prior.kappa_0 + self.n)
            return np.full(B.shape[0], mu_n)
        pred = np.empty(B.shape[0])
        right = B[:, self.feature]
        pred[~right] = self.left.predict(B[~right], prior)
        pred[right] = self.right.predict(B[right], prior)
        return pred


def parse_node_info(
    node_info: Dict[int, tuple], leaf_nodes: Sequence[int], node_id: int = 0
) -> BinaryRegTree:
    """Parses the (node_info, leaf_nodes) encoding of a tree_smc tree."""
    if node_id in leaf_nodes or node_id not in node_info:
        return BinaryRegTree()
    return BinaryRegTree(
        parse_node_info(node_info, leaf_nodes, 2 * node_id + 1),
        parse_node_info(node_info, leaf_nodes, 2 * node_id + 2),
        int(node_info[node_id][0]),
    )


def bma_predict(
    trees: Sequence[BinaryRegTree],
    weights: Sequence[float],
    B: npt.NDArray,
    prior: NIGPrior,
) -> npt.NDArray:
    """Weighted Bayesian model average of the per-tree posterior-mean predictions."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    pred = np.zeros(B.shape[0])
    for tree, weight in zip(trees, weights):
        pred += weight * tree.predict(B, prior)
    return pred
