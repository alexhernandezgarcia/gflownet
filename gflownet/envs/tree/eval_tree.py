"""
Post-training evaluation for DT-GFN (composite Tree environment).

Implements the evaluation protocol from Mahfoud et al. (2025):
  - Per-tree accuracy on train/test sets using Dirichlet posterior sampling
  - Top-1 tree selection by highest log-posterior (Table 2 protocol)
  - Bayesian model averaging (Algorithm 1)

Usage:
    python eval_tree.py \
        --samples_path ./samples/gfn_samples.pkl \
        --data_path tests/data/tree/iris.csv \
        --alpha_value 0.1 \
        --n_dirichlet_samples 10

    Or point to a specific run's samples directory:
    python eval_tree.py \
        --samples_path logs/tree/local/2026-04-20_.../samples/gfn_samples.pkl \
        --data_path tests/data/tree/iris.csv
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.special import gammaln
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from torch.distributions import Dirichlet

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
# Tree traversal (reuses proxy helpers)
# =============================================================================


def route_to_leaves(state: Dict, X: np.ndarray, node_env) -> Dict[int, List[int]]:
    """
    Routes samples through a tree state and returns leaf -> sample indices.

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
    leaf_samples: Dict[int, List[int]] = {}

    for i, x in enumerate(X):
        k = 0
        while 0 <= k < max_nodes and state["_dones"][k] == 1:
            feature_idx = node_env.get_feature(state[k])
            threshold = node_env.get_threshold(state[k])
            if feature_idx is None or threshold is None:
                break
            if x[feature_idx - 1] <= threshold:
                k = Tree.left_child_idx(k)
            else:
                k = Tree.right_child_idx(k)

        if k not in leaf_samples:
            leaf_samples[k] = []
        leaf_samples[k].append(i)

    return leaf_samples


def count_internal_nodes(state: Dict) -> int:
    return sum(state["_dones"])


def count_total_nodes(state: Dict) -> int:
    """Total nodes = internal nodes + leaves (children of internal nodes that
    don't exist). This matches the paper's 'model size' metric."""
    from gflownet.envs.tree.tree import Tree

    n_internal = count_internal_nodes(state)
    max_nodes = len(state["_dones"])
    n_leaves = 0
    for k in range(max_nodes):
        if state["_dones"][k] != 1:
            continue
        for child in (Tree.left_child_idx(k), Tree.right_child_idx(k)):
            if child >= max_nodes or state["_dones"][child] != 1:
                n_leaves += 1
    # Single root-only tree (root done, no children) has n_internal=1, n_leaves=2
    return n_internal + n_leaves


# =============================================================================
# Dirichlet posterior prediction (Section 4.2 of the paper)
# =============================================================================


def predict_dirichlet(
    state: Dict,
    X: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: np.ndarray,
    n_classes: int,
    node_env,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicts class labels by sampling leaf class probabilities from the
    Dirichlet posterior, as described in Section 4.2:

        theta_ell ~ Dirichlet(n_ell + alpha)

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

    Returns
    -------
    predictions : np.ndarray, shape (n_samples,)
        Predicted class labels.
    class_probas : np.ndarray, shape (n_samples, n_classes)
        Class probability vectors for each sample.
    """
    # Get leaf class counts from training data
    train_leaves = route_to_leaves(state, X_train, node_env)
    leaf_counts: Dict[int, np.ndarray] = {}
    for leaf_k, indices in train_leaves.items():
        labels = y_train[indices]
        leaf_counts[leaf_k] = np.bincount(labels, minlength=n_classes).astype(float)

    # Sample class probabilities from Dirichlet posterior for each leaf
    leaf_probas: Dict[int, np.ndarray] = {}
    for leaf_k, counts in leaf_counts.items():
        params = torch.tensor(counts + alpha, dtype=torch.float32)
        leaf_probas[leaf_k] = Dirichlet(params).sample().numpy()

    # Default proba for leaves not seen in training (uniform)
    default_proba = np.ones(n_classes) / n_classes

    # Route test data and predict
    test_leaves = route_to_leaves(state, X, node_env)
    n_samples = len(X)
    class_probas = np.zeros((n_samples, n_classes))

    for leaf_k, indices in test_leaves.items():
        proba = leaf_probas.get(leaf_k, default_proba)
        for idx in indices:
            class_probas[idx] = proba

    predictions = np.argmax(class_probas, axis=1)
    return predictions, class_probas


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
    """Computes log P[Y|X,T] + log P[T|X] with node_count prior."""
    leaf_samples = route_to_leaves(state, X_train, node_env)

    log_b_alpha = np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))

    log_likelihood = 0.0
    for indices in leaf_samples.values():
        labels = y_train[indices]
        counts = np.bincount(labels, minlength=n_classes).astype(float)
        alpha_plus_counts = alpha + counts
        log_b_posterior = np.sum(gammaln(alpha_plus_counts)) - gammaln(
            np.sum(alpha_plus_counts)
        )
        log_likelihood += log_b_posterior - log_b_alpha

    import math

    n_internal = count_internal_nodes(state)
    log_prior = -(math.log(4) + math.log(n_features)) * n_internal

    return log_likelihood + log_prior


# =============================================================================
# Main evaluation functions
# =============================================================================


def calculate_tree_accuracies(
    states: List[Dict],
    energies: np.ndarray,
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
    highest log-posterior. To reduce variance from Dirichlet sampling,
    predictions are averaged over n_dirichlet_samples draws.
    """
    n_trees = len(states)

    test_accuracies = np.zeros(n_trees)
    train_accuracies = np.zeros(n_trees)
    node_counts = np.zeros(n_trees)

    for i, state in enumerate(states):
        node_counts[i] = count_total_nodes(state)

        # Average over multiple Dirichlet draws to reduce variance
        test_correct_sum = np.zeros(len(y_test))
        train_correct_sum = np.zeros(len(y_train))

        for _ in range(n_dirichlet_samples):
            test_preds, _ = predict_dirichlet(
                state, X_test, X_train, y_train, alpha, n_classes, node_env
            )
            train_preds, _ = predict_dirichlet(
                state, X_train, X_train, y_train, alpha, n_classes, node_env
            )
            test_correct_sum += test_preds == y_test
            train_correct_sum += train_preds == y_train

        # Majority vote across Dirichlet samples
        test_majority = (test_correct_sum > n_dirichlet_samples / 2).astype(int)
        train_majority = (train_correct_sum > n_dirichlet_samples / 2).astype(int)

        test_accuracies[i] = test_majority.mean()
        train_accuracies[i] = train_majority.mean()

    # Rank by log-posterior (higher is better)
    order = np.argsort(-energies)

    top_1_idx = order[0]
    top_10_idx = order[: min(10, len(order))]

    return {
        "n_trees": n_trees,
        "test_acc_mean": float(test_accuracies.mean()),
        "test_acc_std": float(test_accuracies.std()),
        "test_acc_top1": float(test_accuracies[top_1_idx]),
        "test_acc_top10_mean": float(test_accuracies[top_10_idx].mean()),
        "train_acc_mean": float(train_accuracies.mean()),
        "train_acc_std": float(train_accuracies.std()),
        "train_acc_top1": float(train_accuracies[top_1_idx]),
        "train_acc_top10_mean": float(train_accuracies[top_10_idx].mean()),
        "model_size_mean": float(node_counts.mean()),
        "model_size_std": float(node_counts.std()),
        "model_size_top1": float(node_counts[top_1_idx]),
        "top1_log_posterior": float(energies[top_1_idx]),
    }


def bayesian_model_averaging(
    states: List[Dict],
    energies: np.ndarray,
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
    Bayesian model averaging (Algorithm 1 of the paper).

    Each tree is weighted by its posterior probability. Class probabilities
    are obtained via Dirichlet posterior sampling, then averaged by weights.
    """
    n_trees = len(states)
    n_test = len(X_test)
    n_train = len(X_train)

    # Posterior weights via softmax of log-posteriors (log-sum-exp trick)
    weights = torch.softmax(torch.tensor(energies, dtype=torch.float32), dim=0).numpy()

    # Accumulate weighted class probabilities over Dirichlet draws
    test_probas_weighted = np.zeros((n_test, n_classes))
    test_probas_uniform = np.zeros((n_test, n_classes))
    train_probas_weighted = np.zeros((n_train, n_classes))
    train_probas_uniform = np.zeros((n_train, n_classes))

    for _ in range(n_dirichlet_samples):
        for t, state in enumerate(states):
            _, test_cp = predict_dirichlet(
                state, X_test, X_train, y_train, alpha, n_classes, node_env
            )
            _, train_cp = predict_dirichlet(
                state, X_train, X_train, y_train, alpha, n_classes, node_env
            )
            test_probas_weighted += weights[t] * test_cp
            test_probas_uniform += test_cp / n_trees
            train_probas_weighted += weights[t] * train_cp
            train_probas_uniform += train_cp / n_trees

    # Average over Dirichlet draws
    test_probas_weighted /= n_dirichlet_samples
    test_probas_uniform /= n_dirichlet_samples
    train_probas_weighted /= n_dirichlet_samples
    train_probas_uniform /= n_dirichlet_samples

    test_preds_w = np.argmax(test_probas_weighted, axis=1)
    test_preds_u = np.argmax(test_probas_uniform, axis=1)
    train_preds_w = np.argmax(train_probas_weighted, axis=1)
    train_preds_u = np.argmax(train_probas_uniform, axis=1)

    return {
        "bma_test_acc_weighted": accuracy_score(y_test, test_preds_w),
        "bma_test_bac_weighted": balanced_accuracy_score(y_test, test_preds_w),
        "bma_test_acc_uniform": accuracy_score(y_test, test_preds_u),
        "bma_test_bac_uniform": balanced_accuracy_score(y_test, test_preds_u),
        "bma_train_acc_weighted": accuracy_score(y_train, train_preds_w),
        "bma_train_acc_uniform": accuracy_score(y_train, train_preds_u),
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
        help="Number of Dirichlet draws to average predictions over (default: 10).",
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
    with open(samples_path, "rb") as f:
        dct = pickle.load(f)

    states = dct["x"]
    energies = dct["energy"]
    if isinstance(energies, torch.Tensor):
        energies = energies.detach().cpu().numpy()
    energies = np.array(energies).flatten()

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

    # Per-tree accuracies
    print(
        f"\nComputing per-tree accuracies ({args.n_dirichlet_samples} Dirichlet draws)..."
    )
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
        args.n_dirichlet_samples,
    )

    print("\n=== Per-Tree Results ===")
    print(
        f"  Test accuracy (mean):    {tree_stats['test_acc_mean']:.4f} +/- {tree_stats['test_acc_std']:.4f}"
    )
    print(f"  Test accuracy (top-1):   {tree_stats['test_acc_top1']:.4f}")
    print(f"  Test accuracy (top-10):  {tree_stats['test_acc_top10_mean']:.4f}")
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
        args.n_dirichlet_samples,
    )

    print("\n=== Bayesian Model Averaging ===")
    print(f"  Test accuracy (weighted):  {bma_stats['bma_test_acc_weighted']:.4f}")
    print(f"  Test BAcc (weighted):      {bma_stats['bma_test_bac_weighted']:.4f}")
    print(f"  Test accuracy (uniform):   {bma_stats['bma_test_acc_uniform']:.4f}")
    print(f"  Test BAcc (uniform):       {bma_stats['bma_test_bac_uniform']:.4f}")
    print(f"  Train accuracy (weighted): {bma_stats['bma_train_acc_weighted']:.4f}")
    print(f"  Train accuracy (uniform):  {bma_stats['bma_train_acc_uniform']:.4f}")

    # Save results
    results = {**tree_stats, **bma_stats}
    output_path = args.output or str(samples_path.parent / "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
