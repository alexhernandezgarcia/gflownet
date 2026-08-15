"""
Helpers for the benchmarks that operate on quantile-binarized features
(MAPTree and the Bayesian CART samplers): feature binarization, parsing of
the tree encodings returned by maptree / tree_smc, and Beta-smoothed leaf
predictions.

Conventions (matching the MAPTree repository): binarized feature j is the
test x[feat] <= threshold; at an internal node, points with feature value
False go left and True go right. maptree encodes trees as strings
"(<left><feature><right>)" with "" for leaves; tree_smc encodes them as a
dict node_info {node_id: (feature, ...)} plus a list of leaf ids, with the
children of node i at 2i+1 (left) and 2i+2 (right).
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt


def binarize_quantiles(
    X_train: npt.NDArray, X_test: npt.NDArray, thresholds_per_feature: int
) -> Tuple[npt.NDArray, npt.NDArray, List[Tuple[int, float]]]:
    """
    Binarizes continuous features with per-feature quantile thresholds
    computed on the train split. Returns (B_train, B_test, thresholds), where
    B_* are boolean matrices with one column per (feature, threshold) pair
    (True iff x[feature] <= threshold) and thresholds lists those pairs.
    Duplicate quantiles (from repeated values) are collapsed.
    """
    quantiles = np.linspace(0, 1, thresholds_per_feature + 2)[1:-1]
    thresholds = []
    for feat in range(X_train.shape[1]):
        for thr in np.unique(np.quantile(X_train[:, feat], quantiles)):
            thresholds.append((feat, float(thr)))
    B_train = np.stack([X_train[:, f] <= t for f, t in thresholds], axis=1)
    B_test = np.stack([X_test[:, f] <= t for f, t in thresholds], axis=1)
    return B_train, B_test, thresholds


class BinaryProbTree:
    """
    Decision tree over binary features predicting the Beta-posterior mean
    P(y=1) = (n1 + rho1) / (n0 + n1 + rho0 + rho1) at each leaf, where
    (n0, n1) are the train label counts routed to the leaf.
    """

    def __init__(
        self,
        left: "BinaryProbTree" = None,
        right: "BinaryProbTree" = None,
        feature: int = None,
    ):
        assert (left is None) == (right is None) == (feature is None)
        self.left = left
        self.right = right
        self.feature = feature
        self.counts = (0, 0)

    def is_leaf(self) -> bool:
        return self.feature is None

    def size(self) -> int:
        return 1 if self.is_leaf() else 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        if self.is_leaf():
            return 0
        return 1 + max(self.left.depth(), self.right.depth())

    def fit_counts(self, B: npt.NDArray, y: npt.NDArray) -> "BinaryProbTree":
        self.counts = (int(np.sum(y == 0)), int(np.sum(y == 1)))
        if not self.is_leaf():
            right = B[:, self.feature]
            self.left.fit_counts(B[~right], y[~right])
            self.right.fit_counts(B[right], y[right])
        return self

    def predict_proba(self, B: npt.NDArray, rho: Tuple[float, float]) -> npt.NDArray:
        if self.is_leaf():
            n0, n1 = self.counts
            return np.full(B.shape[0], (n1 + rho[1]) / (n0 + n1 + rho[0] + rho[1]))
        proba = np.empty(B.shape[0])
        right = B[:, self.feature]
        proba[~right] = self.left.predict_proba(B[~right], rho)
        proba[right] = self.right.predict_proba(B[right], rho)
        return proba


def parse_maptree_string(tree: str) -> BinaryProbTree:
    """Parses the string encoding returned by maptree.search."""

    def parse_node(i: int) -> Tuple[BinaryProbTree, int]:
        if i < len(tree) and tree[i] == "(":
            left, i = parse_node(i + 1)
            j = i
            while tree[j] not in "()":
                j += 1
            feature = int(tree[i:j])
            right, i = parse_node(j)
            return BinaryProbTree(left, right, feature), i + 1
        return BinaryProbTree(), i

    return parse_node(0)[0]


def parse_node_info(
    node_info: Dict[int, tuple], leaf_nodes: Sequence[int], node_id: int = 0
) -> BinaryProbTree:
    """Parses the (node_info, leaf_nodes) encoding of a tree_smc tree."""
    if node_id in leaf_nodes or node_id not in node_info:
        return BinaryProbTree()
    return BinaryProbTree(
        parse_node_info(node_info, leaf_nodes, 2 * node_id + 1),
        parse_node_info(node_info, leaf_nodes, 2 * node_id + 2),
        int(node_info[node_id][0]),
    )


def bma_proba(
    trees: Sequence[BinaryProbTree],
    weights: Sequence[float],
    B: npt.NDArray,
    rho: Tuple[float, float],
) -> npt.NDArray:
    """Weighted Bayesian model average of per-tree predicted P(y=1)."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    proba = np.zeros(B.shape[0])
    for tree, weight in zip(trees, weights):
        proba += weight * tree.predict_proba(B, rho)
    return proba
