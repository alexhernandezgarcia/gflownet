"""
Post-training evaluation for DT-GFN (composite Tree environment).

Implements the evaluation protocol from Mahfoud et al. (2025), with the leaf
parameters integrated out analytically instead of Monte-Carlo sampled:
  - Per-tree accuracy on train/test sets using the exact Dirichlet posterior
    predictive mean at each leaf
  - Top-1 tree selection by highest log-posterior (Table 2 protocol)
  - Bayesian model averaging (Algorithm 1), both posterior-weighted (as in the
    paper) and uniform over GFN samples (the unbiased Monte-Carlo estimate of
    the BMA predictive when trees are sampled from the posterior)
  - Probabilistic metrics of the predictive distribution: negative
    log-likelihood, Brier score and expected calibration error

Usage:
    python eval_tree.py \
        --samples_path path/to/gfn_samples.pkl \
        --data_path path/to/<dataset>.csv \
        --alpha_value 0.1
"""

import argparse
import io
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# This script is launched by path (``python .../eval_tree.py``), so sys.path[0]
# is its own directory and the repo root is not importable. Without this the
# ``gflownet`` package is resolved against site-packages instead of the
# checkout being evaluated.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]  # <repo>/gflownet/envs/tree
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.special import gammaln, softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# Sample loading
# =============================================================================


class _CPUUnpickler(pickle.Unpickler):
    """Unpickler that maps torch storages to CPU.

    Runs trained on a GPU pickle their energies as CUDA tensors, which plain
    ``pickle.load`` refuses to restore on a CPU-only node. Evaluation only ever
    needs the states (plain dicts) and the energies as numbers, so the storages
    are redirected to the CPU on the way in.
    """

    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch

            return lambda b: torch.load(
                io.BytesIO(b), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


def load_samples(samples_path: Path) -> Dict:
    """Load a gfn_samples.pkl, whether it was written on CPU or on GPU."""
    with open(samples_path, "rb") as f:
        return _CPUUnpickler(f).load()


# =============================================================================
# Data loading (mirrors Tree._load_dataset + scaling)
# =============================================================================


def load_and_scale_dataset(
    data_path: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Load dataset from CSV/PKL and apply MinMax scaling."""
    from gflownet.envs.tree.tree import Tree

    X_train, y_train, X_test, y_test, _ = Tree._load_dataset(data_path)
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    if X_test is not None:
        X_test = scaler.transform(X_test)
    y_train = y_train.astype(int)
    if y_test is not None:
        y_test = y_test.astype(int)
    return X_train, y_train, X_test, y_test


# =============================================================================
# Tree traversal (vectorized, mirrors proxy._route_samples_to_leaves)
# =============================================================================


def route_to_leaves(state: Dict, X: np.ndarray, node_env) -> Dict[int, np.ndarray]:
    """
    Routes samples through a tree state and returns dict containing for each leaf
    an array of all the sample indices that land in this leaf.

    The routing is vectorized like ``proxy.tree._route_samples_to_leaves``:
    the (feature, threshold) split of each done node is extracted only once
    per tree, and the sample indices are then partitioned node by node with
    array comparisons, instead of walking the tree sample by sample. Leaves
    are ordered by first sample arrival so that downstream floating-point
    accumulations over the leaves match the original per-sample walk.

    Parameters
    ----------
    state : dict
        Terminal tree state from the composite Tree env.
    X : np.ndarray
        Data matrix, shape (n_samples, n_features).
    node_env : DecisionTreeNode
        The node environment for interpreting node substates.
    """
    from gflownet.envs.tree.tree import Tree

    max_nodes = len(state["_dones"])

    # Extract the split of each done node once: k -> (feature, threshold),
    # with the feature index 0-based (it is 1-based in the Choice subenv)
    splits = {}
    stack = [0]
    while stack:
        k = stack.pop()
        if not (0 <= k < max_nodes and state["_dones"][k] == 1):
            continue
        feature_idx = node_env.get_feature(state[k])
        threshold = node_env.get_threshold(state[k])
        if feature_idx is None or threshold is None:
            raise ValueError(
                "Reached node with no feature or threshold to split on, "
                "should be impossible!"
            )
        splits[k] = (feature_idx - 1, threshold)
        stack.append(Tree.left_child_idx(k))
        stack.append(Tree.right_child_idx(k))

    # Route the sample indices through the splits by partitioning index
    # arrays, starting with all samples at the root
    leaf_samples: Dict[int, np.ndarray] = {}
    partitions = [(0, np.arange(len(X)))]
    while partitions:
        k, indices = partitions.pop()
        if len(indices) == 0:
            continue
        if k in splits:
            feature, threshold = splits[k]
            to_left = X[indices, feature] <= threshold
            partitions.append((Tree.left_child_idx(k), indices[to_left]))
            partitions.append((Tree.right_child_idx(k), indices[~to_left]))
        else:
            leaf_samples[k] = indices

    return dict(sorted(leaf_samples.items(), key=lambda item: item[1][0]))


def count_internal_nodes(state: Dict) -> int:
    return int(np.sum(state["_dones"]))


def count_total_nodes(state: Dict) -> int:
    """Returns total nodes = internal nodes + leaves"""
    dones = np.asarray(state["_dones"])
    max_nodes = len(dones)
    done_idx = np.flatnonzero(dones == 1)
    # Children of done nodes are leaves if outside the array or not done
    children = np.concatenate([2 * done_idx + 1, 2 * done_idx + 2])
    out_of_range = children >= max_nodes
    n_leaves = int(
        np.sum(out_of_range | (dones[np.minimum(children, max_nodes - 1)] != 1))
    )
    # Single root-only tree (root done, no children) has n_internal=1 + n_leaves=2
    return len(done_idx) + n_leaves


# =============================================================================
# Dirichlet posterior predictive (Section 4.2, with theta integrated out)
# =============================================================================


def leaf_posterior_means(
    state: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: np.ndarray,
    n_classes: int,
    node_env,
) -> Dict[int, np.ndarray]:
    """
    Computes the exact posterior predictive class probabilities at each leaf
    by integrating out theta_l ~ Dirichlet(n_l + alpha) analytically:

        E[theta_{l,c}] = (n_{l,c} + alpha_c) / (n_l + sum_c alpha_c)

    Returns a mapping from leaf index to its class probability vector.
    """
    train_leaves = route_to_leaves(state, X_train, node_env)
    leaf_probas: Dict[int, np.ndarray] = {}
    for leaf_k, indices in train_leaves.items():
        counts = np.bincount(y_train[indices], minlength=n_classes).astype(float)
        params = counts + alpha
        leaf_probas[leaf_k] = params / params.sum()
    return leaf_probas


def predict_posterior_mean(
    state: Dict,
    X: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: np.ndarray,
    n_classes: int,
    node_env,
    leaf_probas: Optional[Dict[int, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts class labels with the exact Dirichlet posterior predictive mean
    at each leaf (deterministic; no Monte-Carlo sampling of theta).

    Parameters
    ----------
    state : dict
        A terminal tree state.
    X : np.ndarray
        Data to predict on, shape (n_samples, n_features).
    X_train : np.ndarray
        Training data for computing leaf counts.
    y_train : np.ndarray
        Training labels.
    alpha : np.ndarray
        Dirichlet prior, shape (n_classes,).
    n_classes : int
    node_env : DecisionTreeNode
    leaf_probas : dict, optional
        Precomputed output of ``leaf_posterior_means`` for this state, to
        avoid re-routing the training data on repeated calls.

    Returns
    -------
    predictions : np.ndarray, shape (n_samples,)
        Predicted class labels.
    class_probas : np.ndarray, shape (n_samples, n_classes)
        Class probability vectors for each sample.
    """
    if leaf_probas is None:
        leaf_probas = leaf_posterior_means(
            state, X_train, y_train, alpha, n_classes, node_env
        )

    # Leaves with no training samples fall back to the prior predictive mean
    default_proba = alpha / alpha.sum()

    leaves = route_to_leaves(state, X, node_env)
    class_probas = np.zeros((len(X), n_classes))
    for leaf_k, indices in leaves.items():
        class_probas[indices] = leaf_probas.get(leaf_k, default_proba)

    predictions = np.argmax(class_probas, axis=1)
    return predictions, class_probas


# =============================================================================
# Probabilistic metrics of the predictive distribution
# =============================================================================


def probabilistic_metrics(
    y: np.ndarray, class_probas: np.ndarray, n_bins: int = 15
) -> Dict[str, float]:
    """
    Computes proper-scoring and calibration metrics of a predictive
    distribution: negative log-likelihood (mean per sample), Brier score
    (multi-class, mean per sample) and expected calibration error with
    ``n_bins`` equal-width confidence bins.
    """
    n = len(y)
    idx = np.arange(n)

    p_true = np.clip(class_probas[idx, y], 1e-12, 1.0)
    nll = float(-np.mean(np.log(p_true)))

    onehot = np.zeros_like(class_probas)
    onehot[idx, y] = 1.0
    brier = float(np.mean(np.sum((class_probas - onehot) ** 2, axis=1)))

    confidence = class_probas.max(axis=1)
    correct = (class_probas.argmax(axis=1) == y).astype(float)
    bin_idx = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    bin_total = np.bincount(bin_idx, minlength=n_bins)
    bin_confidence = np.bincount(bin_idx, weights=confidence, minlength=n_bins)
    bin_correct = np.bincount(bin_idx, weights=correct, minlength=n_bins)
    nonempty = bin_total > 0
    ece = float(np.sum(np.abs(bin_correct[nonempty] - bin_confidence[nonempty])) / n)

    return {"nll": nll, "brier": brier, "ece": ece}


# =============================================================================
# Log-posterior computation (same as CategoricalTreeProxy)
# =============================================================================


def compute_log_posterior(
    state: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: np.ndarray,
    n_classes: int,
    n_features: int,
    node_env,
) -> float:
    """Computes log posterior = log P[T|X,Y] = log P[Y|X,T] + log P[T|X]
    with node_count prior, following section 4.2."""

    leaf_samples = route_to_leaves(state, X_train, node_env)

    log_alpha = gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))

    log_likelihood = 0.0
    for indices in leaf_samples.values():
        # Empty leaves are skipped since they contribute 0 to the log_likelihood
        labels = y_train[indices]
        counts = np.bincount(labels, minlength=n_classes).astype(float)
        alpha_plus_counts = alpha + counts
        log_posterior = np.sum(gammaln(alpha_plus_counts)) - gammaln(
            np.sum(alpha_plus_counts)
        )
        log_likelihood += log_alpha + log_posterior

    n_internal = count_internal_nodes(state)
    log_prior = -(math.log(4) + math.log(n_features)) * n_internal

    return log_likelihood + log_prior


# =============================================================================
# Main evaluation functions
# =============================================================================


def calculate_tree_accuracies(
    states: List[Dict],
    log_posteriors: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: np.ndarray,
    n_classes: int,
    node_env,
    n_dirichlet_samples: int = 10,
) -> dict:
    """
    Compute accuracy statistics over GFN-sampled trees.

    Selection protocol (Table 2): the top-1 tree is the one with the
    highest log-posterior. Predictions use the exact Dirichlet posterior
    predictive mean at each leaf, so they are deterministic.

    ``n_dirichlet_samples`` is deprecated and ignored (kept for backward
    compatibility): the closed-form posterior mean replaces averaging over
    Dirichlet draws.
    """
    n_trees = len(states)

    test_accuracies = np.zeros(n_trees)
    train_accuracies = np.zeros(n_trees)
    test_nlls = np.zeros(n_trees)
    node_counts = np.zeros(n_trees)

    for i, state in enumerate(states):
        node_counts[i] = count_total_nodes(state)

        leaf_probas = leaf_posterior_means(
            state, X_train, y_train, alpha, n_classes, node_env
        )
        test_preds, test_cp = predict_posterior_mean(
            state, X_test, X_train, y_train, alpha, n_classes, node_env, leaf_probas
        )
        train_preds, _ = predict_posterior_mean(
            state, X_train, X_train, y_train, alpha, n_classes, node_env, leaf_probas
        )

        test_accuracies[i] = np.mean(test_preds == y_test)
        train_accuracies[i] = np.mean(train_preds == y_train)
        test_nlls[i] = probabilistic_metrics(y_test, test_cp)["nll"]

    # Rank by log-posterior (higher is better)
    order = np.argsort(-log_posteriors)

    top_1_idx = order[0]
    top_10_idx = order[: min(10, len(order))]

    return {
        "n_trees": n_trees,
        "test_acc_mean": float(test_accuracies.mean()),
        "test_acc_std": float(test_accuracies.std()),
        "test_acc_top1": float(test_accuracies[top_1_idx]),
        "test_acc_top10_mean": float(test_accuracies[top_10_idx].mean()),
        "test_nll_mean": float(test_nlls.mean()),
        "test_nll_top1": float(test_nlls[top_1_idx]),
        "train_acc_mean": float(train_accuracies.mean()),
        "train_acc_std": float(train_accuracies.std()),
        "train_acc_top1": float(train_accuracies[top_1_idx]),
        "train_acc_top10_mean": float(train_accuracies[top_10_idx].mean()),
        "model_size_mean": float(node_counts.mean()),
        "model_size_std": float(node_counts.std()),
        "model_size_top1": float(node_counts[top_1_idx]),
        "top1_log_posterior": float(log_posteriors[top_1_idx]),
    }


def bayesian_model_averaging(
    states: List[Dict],
    log_posteriors: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: np.ndarray,
    n_classes: int,
    node_env,
    n_dirichlet_samples: int = 10,
) -> dict:
    """
    Bayesian model averaging over GFN-sampled trees, with the leaf
    parameters theta integrated out analytically (posterior predictive mean
    per leaf), in two variants:

    - ``uniform``: plain average over the sampled trees. Since the GFN policy
      samples trees approximately proportionally to P[T|D], the sampling
      frequency already carries the posterior weight, making this the
      unbiased Monte-Carlo estimate of the BMA predictive
      sum_T P[T|D] P[y|x,T]. This is the headline estimator.
    - ``weighted``: trees re-weighted by softmax of their log-posteriors
      (Algorithm 1 of the paper). Reported for comparison; note that on top
      of posterior sampling frequencies this double-counts the posterior and
      typically collapses onto the MAP tree.

    ``n_dirichlet_samples`` is deprecated and ignored (kept for backward
    compatibility): the closed-form posterior mean replaces averaging over
    Dirichlet draws.
    """
    n_trees = len(states)
    n_test = len(X_test)
    n_train = len(X_train)

    # Posterior weights via softmax of log-posteriors (log-sum-exp trick)
    weights = softmax(np.asarray(log_posteriors, dtype=np.float64))

    test_probas_weighted = np.zeros((n_test, n_classes))
    test_probas_uniform = np.zeros((n_test, n_classes))
    train_probas_weighted = np.zeros((n_train, n_classes))
    train_probas_uniform = np.zeros((n_train, n_classes))

    for t, state in enumerate(states):
        leaf_probas = leaf_posterior_means(
            state, X_train, y_train, alpha, n_classes, node_env
        )
        _, test_cp = predict_posterior_mean(
            state, X_test, X_train, y_train, alpha, n_classes, node_env, leaf_probas
        )
        _, train_cp = predict_posterior_mean(
            state, X_train, X_train, y_train, alpha, n_classes, node_env, leaf_probas
        )
        test_probas_weighted += weights[t] * test_cp
        test_probas_uniform += test_cp / n_trees
        train_probas_weighted += weights[t] * train_cp
        train_probas_uniform += train_cp / n_trees

    test_preds_w = np.argmax(test_probas_weighted, axis=1)
    test_preds_u = np.argmax(test_probas_uniform, axis=1)
    train_preds_w = np.argmax(train_probas_weighted, axis=1)
    train_preds_u = np.argmax(train_probas_uniform, axis=1)

    test_metrics_w = probabilistic_metrics(y_test, test_probas_weighted)
    test_metrics_u = probabilistic_metrics(y_test, test_probas_uniform)
    train_metrics_w = probabilistic_metrics(y_train, train_probas_weighted)
    train_metrics_u = probabilistic_metrics(y_train, train_probas_uniform)

    return {
        "bma_test_acc_weighted": accuracy_score(y_test, test_preds_w),
        "bma_test_bac_weighted": balanced_accuracy_score(y_test, test_preds_w),
        "bma_test_acc_uniform": accuracy_score(y_test, test_preds_u),
        "bma_test_bac_uniform": balanced_accuracy_score(y_test, test_preds_u),
        "bma_test_nll_weighted": test_metrics_w["nll"],
        "bma_test_nll_uniform": test_metrics_u["nll"],
        "bma_test_brier_weighted": test_metrics_w["brier"],
        "bma_test_brier_uniform": test_metrics_u["brier"],
        "bma_test_ece_weighted": test_metrics_w["ece"],
        "bma_test_ece_uniform": test_metrics_u["ece"],
        "bma_train_acc_weighted": accuracy_score(y_train, train_preds_w),
        "bma_train_acc_uniform": accuracy_score(y_train, train_preds_u),
        "bma_train_nll_weighted": train_metrics_w["nll"],
        "bma_train_nll_uniform": train_metrics_u["nll"],
    }


# =============================================================================
# CLI entry point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GFN-sampled decision trees from the composite Tree env."
    )
    parser.add_argument(
        "--samples_path",
        type=str,
        required=True,
        help="Path to gfn_samples.pkl from a training run.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset CSV/PKL file.",
    )
    parser.add_argument(
        "--alpha_value",
        type=float,
        default=0.1,
        help="Dirichlet prior concentration parameter (default: 0.1).",
    )
    parser.add_argument(
        "--n_dirichlet_samples",
        type=int,
        default=10,
        help="Deprecated and ignored: predictions now use the closed-form "
        "Dirichlet posterior predictive mean instead of sampling.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON. Defaults to eval_results.json next to samples.",
    )
    args = parser.parse_args()

    # Load samples
    samples_path = Path(args.samples_path)
    print(f"Loading samples from {samples_path}")
    dct = load_samples(samples_path)

    states = dct["x"]

    print(f"  Loaded {len(states)} trees")

    # Load dataset
    X_train, y_train, X_test, y_test = load_and_scale_dataset(args.data_path)
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    alpha = np.ones(n_classes) * args.alpha_value

    print(f"  Dataset: {args.data_path}")
    print(
        f"  Train: {len(X_train)} samples, Test: {len(X_test) if X_test is not None else 0} samples"
    )
    print(f"  Classes: {n_classes}, Features: {n_features}")
    print(f"  Alpha: {alpha}")

    # Build a minimal node_env for tree traversal
    from gflownet.envs.tree.node import DecisionTreeNode
    from gflownet.envs.tree.tree import Tree

    # Derive feature names from dataset
    _, _, _, _, feature_names = Tree._load_dataset(args.data_path)
    if feature_names is not None:
        features = list(feature_names)
    else:
        features = [f"x{i}" for i in range(n_features)]

    node_env = DecisionTreeNode(features=features)

    # Recompute log-posteriors consistently (the saved energies may use
    # different reward_function scaling)
    print("\nRecomputing log-posteriors...")
    log_posteriors = np.array(
        [
            compute_log_posterior(
                state, X_train, y_train, alpha, n_classes, n_features, node_env
            )
            for state in states
        ]
    )
    print(
        f"  Log-posterior range: [{log_posteriors.min():.2f}, {log_posteriors.max():.2f}]"
    )

    if X_test is None or y_test is None:
        print("No test split found. Cannot compute test accuracies.")
        return

    # Per-tree accuracies (deterministic posterior-mean predictions)
    print("\nComputing per-tree accuracies...")
    tree_stats = calculate_tree_accuracies(
        states,
        log_posteriors,
        X_train,
        y_train,
        X_test,
        y_test,
        alpha,
        n_classes,
        node_env,
    )

    print("\n=== Per-Tree Results ===")
    print(
        f"  Test accuracy (mean):    {tree_stats['test_acc_mean']:.4f} +/- {tree_stats['test_acc_std']:.4f}"
    )
    print(f"  Test accuracy (top-1):   {tree_stats['test_acc_top1']:.4f}")
    print(f"  Test accuracy (top-10):  {tree_stats['test_acc_top10_mean']:.4f}")
    print(f"  Test NLL (mean):         {tree_stats['test_nll_mean']:.4f}")
    print(f"  Test NLL (top-1):        {tree_stats['test_nll_top1']:.4f}")
    print(
        f"  Train accuracy (mean):   {tree_stats['train_acc_mean']:.4f} +/- {tree_stats['train_acc_std']:.4f}"
    )
    print(f"  Train accuracy (top-1):  {tree_stats['train_acc_top1']:.4f}")
    print(
        f"  Model size (mean):       {tree_stats['model_size_mean']:.1f} +/- {tree_stats['model_size_std']:.1f}"
    )
    print(f"  Model size (top-1):      {tree_stats['model_size_top1']:.0f}")
    print(f"  Top-1 log-posterior:     {tree_stats['top1_log_posterior']:.4f}")

    # Bayesian model averaging
    print(f"\nComputing Bayesian model averaging ({len(states)} trees)...")
    bma_stats = bayesian_model_averaging(
        states,
        log_posteriors,
        X_train,
        y_train,
        X_test,
        y_test,
        alpha,
        n_classes,
        node_env,
    )

    print("\n=== Bayesian Model Averaging ===")
    print(f"  Test accuracy (uniform):   {bma_stats['bma_test_acc_uniform']:.4f}")
    print(f"  Test BAcc (uniform):       {bma_stats['bma_test_bac_uniform']:.4f}")
    print(f"  Test NLL (uniform):        {bma_stats['bma_test_nll_uniform']:.4f}")
    print(f"  Test Brier (uniform):      {bma_stats['bma_test_brier_uniform']:.4f}")
    print(f"  Test ECE (uniform):        {bma_stats['bma_test_ece_uniform']:.4f}")
    print(f"  Test accuracy (weighted):  {bma_stats['bma_test_acc_weighted']:.4f}")
    print(f"  Test BAcc (weighted):      {bma_stats['bma_test_bac_weighted']:.4f}")
    print(f"  Test NLL (weighted):       {bma_stats['bma_test_nll_weighted']:.4f}")
    print(f"  Test Brier (weighted):     {bma_stats['bma_test_brier_weighted']:.4f}")
    print(f"  Test ECE (weighted):       {bma_stats['bma_test_ece_weighted']:.4f}")
    print(f"  Train accuracy (uniform):  {bma_stats['bma_train_acc_uniform']:.4f}")
    print(f"  Train accuracy (weighted): {bma_stats['bma_train_acc_weighted']:.4f}")

    # Save results
    results = {**tree_stats, **bma_stats}
    output_path = args.output or str(samples_path.parent / "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
