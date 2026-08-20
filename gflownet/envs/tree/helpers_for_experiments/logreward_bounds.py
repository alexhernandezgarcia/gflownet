"""
Bound the log-reward range of a Tree GFlowNet analytically, from the dataset.

The reward is an (unnormalised) Bayesian posterior, so log R <= 0, but that
ceiling is uninformative: what sets the loss scale is that the log-likelihood is
a *sum over training samples*, hence O(N). This helper turns the dataset (N,
class counts, n_features) and the proxy settings (alpha, prior_type, max_depth)
into concrete numbers, so the range can be predicted before launching a run.

With reward_function=exponential, alpha=1 and beta=1, log R equals the proxy
value exactly (gflownet/proxy/base.py:337), i.e.

    log R(T) = sum_leaves [log B(alpha + counts_leaf) - log B(alpha)]  +  log P(T)

The three reported quantities are, in decreasing order:

  ceiling   Coarsest *pure* partition: one leaf per class. Per-leaf the
            Dirichlet-Multinomial term is maximised by a pure count vector and,
            among pure partitions, merging leaves of the same class always
            improves, so K pure leaves is the maximum over all partitions. It is
            attained only if the classes are separable by K-1 axis-aligned
            splits, so it is an upper bound, usually loose.

  root      The partition that does not separate anything: all N samples in one
            leaf. Equal to -N * H(y) up to an O(K log N) Bayesian-Occam term.
            This is where an uninformative policy lands, and it predicts the
            *mean* log-reward of a training run well.

  floor     Maximally fine and maximally mixed: 2^max_depth leaves, N spread
            evenly, each leaf mirroring the global class distribution. Both
            choices are verified minimisers of the DM term (it is Schur-concave
            in the counts for alpha < 1, and splitting a mixed leaf further only
            adds per-leaf cost). Also loose: it assumes the splits can actually
            shatter the data that way.

Usage (from the repo root):
    python gflownet/envs/tree/helpers_for_experiments/logreward_bounds.py \
        tests/data/tree/breast_cancer/breast_cancer_1.csv --max-depth 5
    python gflownet/envs/tree/helpers_for_experiments/logreward_bounds.py \
        tests/data/tree/*/[a-z]*_1.csv --max-depth 4 --alpha-value 0.1
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import gammaln

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]  # <repo>/gflownet/envs/tree/helpers_for_experiments
sys.path.insert(0, str(REPO_ROOT))  # make `gflownet` importable when run directly

from gflownet.envs.tree.tree import Tree


def dm_log_marginal(counts, alpha) -> float:
    """
    Dirichlet-Multinomial marginal log-likelihood of one leaf.

    log B(alpha + counts) - log B(alpha), the same quantity accumulated by
    TreeProxy._compute_log_likelihood (gflownet/proxy/tree.py:286-296).
    """
    counts = np.asarray(counts, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    log_b_posterior = gammaln(alpha + counts).sum() - gammaln((alpha + counts).sum())
    log_b_alpha = gammaln(alpha).sum() - gammaln(alpha.sum())
    return float(log_b_posterior - log_b_alpha)


def log_prior(n_internal: int, prior_type: str, n_features: int, beta: float) -> float:
    """
    Structure log-prior, mirroring TreeProxy._compute_log_prior.

    Only the size-based priors are covered: "bcart" depends on the tree shape,
    not just on the number of internal nodes, and is reported as NaN.
    """
    if prior_type == "none":
        return 0.0
    if prior_type == "exponential":
        return -beta * n_internal
    if prior_type == "node_count":
        return -(math.log(4) + math.log(n_features)) * n_internal
    return float("nan")


def split_evenly(total: int, n_parts: int) -> np.ndarray:
    """Splits an integer into n_parts as evenly as possible."""
    base, remainder = divmod(total, n_parts)
    return np.array([base + (1 if i < remainder else 0) for i in range(n_parts)])


def bounds(y_train, n_features, max_depth, alpha, prior_type, beta) -> dict:
    """
    Computes the ceiling / root / floor log-rewards for one dataset.
    """
    n_classes = len(alpha)
    class_counts = np.bincount(y_train, minlength=n_classes).astype(float)
    n_train = int(class_counts.sum())

    # Number of leaves and internal nodes of a complete tree. The composite env
    # splits at depths 0..max_depth-1, so a full tree has 2^max_depth leaves.
    n_leaves_max = 2**max_depth
    n_internal_max = n_leaves_max - 1
    # A binary tree with K leaves has K-1 internal nodes; the composite env has
    # no terminal single-leaf tree, so at least one split is always paid for.
    n_internal_min = max(1, n_classes - 1)

    # Ceiling: one pure leaf per class.
    loglik_ceiling = sum(
        dm_log_marginal(np.eye(n_classes)[k] * class_counts[k], alpha)
        for k in range(n_classes)
        if class_counts[k] > 0
    )

    # Root: everything in one leaf. Priced with the cheapest legal tree, whose
    # splits separate nothing.
    loglik_root = dm_log_marginal(class_counts, alpha)

    # Floor: full depth, every leaf mirroring the global class distribution.
    per_class = np.stack([split_evenly(int(c), n_leaves_max) for c in class_counts], 1)
    loglik_floor = sum(dm_log_marginal(counts, alpha) for counts in per_class)

    prior_min_nodes = log_prior(n_internal_min, prior_type, n_features, beta)
    prior_max_nodes = log_prior(n_internal_max, prior_type, n_features, beta)

    freqs = class_counts / n_train
    entropy = float(-(freqs[freqs > 0] * np.log(freqs[freqs > 0])).sum())

    return {
        "n_train": n_train,
        "n_classes": n_classes,
        "n_features": n_features,
        "max_depth": max_depth,
        "n_leaves_max": n_leaves_max,
        "entropy": entropy,
        "minus_n_h": -n_train * entropy,
        "loglik_ceiling": loglik_ceiling,
        "loglik_root": loglik_root,
        "loglik_floor": loglik_floor,
        "prior_min_nodes": prior_min_nodes,
        "prior_max_nodes": prior_max_nodes,
        "ceiling": loglik_ceiling + prior_min_nodes,
        "root": loglik_root + log_prior(1, prior_type, n_features, beta),
        "floor": loglik_floor + prior_max_nodes,
    }


def make_alpha(y_train, alpha_type: str, alpha_value: float) -> np.ndarray:
    """Builds the Dirichlet concentration vector as TreeProxy.setup does."""
    n_classes = len(np.unique(y_train))
    if alpha_type == "uniform":
        return np.ones(n_classes) * alpha_value
    if alpha_type == "label_counts":
        class_counts = np.bincount(y_train) + 1
        return class_counts / class_counts.sum() * alpha_value
    raise ValueError(f"Unknown alpha_type '{alpha_type}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Bound the log-reward range of a Tree GFlowNet from the dataset."
    )
    parser.add_argument("data_paths", type=Path, nargs="+", help="Dataset CSV(s).")
    parser.add_argument("--max-depth", type=int, default=4, help="env.max_depth.")
    parser.add_argument("--alpha-value", type=float, default=0.1)
    parser.add_argument(
        "--alpha-type", choices=("uniform", "label_counts"), default="uniform"
    )
    parser.add_argument(
        "--prior-type",
        choices=("node_count", "exponential", "none", "bcart"),
        default="node_count",
    )
    parser.add_argument(
        "--beta", type=float, default=1.0, help="Coefficient of the exponential prior."
    )
    parser.add_argument("--verbose", action="store_true", help="Show the breakdown.")
    args = parser.parse_args()

    print(
        f"proxy: alpha_type={args.alpha_type} alpha_value={args.alpha_value} "
        f"prior_type={args.prior_type} | env.max_depth={args.max_depth}\n"
    )
    header = (
        f"{'dataset':<16}{'N':>6}{'K':>4}{'feat':>6}{'H(y)':>8}"
        f"{'ceiling':>10}{'root':>10}{'-N*H':>10}{'floor':>10}{'span':>9}"
    )
    print(header)
    print("-" * len(header))
    for data_path in args.data_paths:
        X_train, y_train, _, _, _ = Tree._load_dataset(data_path)
        y_train = np.asarray(y_train).astype(int)
        alpha = make_alpha(y_train, args.alpha_type, args.alpha_value)
        res = bounds(
            y_train,
            X_train.shape[1],
            args.max_depth,
            alpha,
            args.prior_type,
            args.beta,
        )
        print(
            f"{data_path.stem:<16}{res['n_train']:>6}{res['n_classes']:>4}"
            f"{res['n_features']:>6}{res['entropy']:>8.3f}"
            f"{res['ceiling']:>10.1f}{res['root']:>10.1f}{res['minus_n_h']:>10.1f}"
            f"{res['floor']:>10.1f}{res['ceiling'] - res['floor']:>9.1f}"
        )
        if args.verbose:
            print(
                f"{'':<16}  log-lik: ceiling {res['loglik_ceiling']:.1f} | "
                f"root {res['loglik_root']:.1f} | floor {res['loglik_floor']:.1f}\n"
                f"{'':<16}  log-prior: {res['prior_min_nodes']:.1f} "
                f"({max(1, res['n_classes'] - 1)} internal nodes) .. "
                f"{res['prior_max_nodes']:.1f} "
                f"({res['n_leaves_max'] - 1} internal nodes)"
            )
    print(
        "\nceiling = one pure leaf per class (upper bound, needs separable classes)\n"
        "root    = all samples in one leaf, i.e. no separation; ~ -N*H(y)\n"
        "floor   = 2^max_depth leaves, each mirroring the global class mix "
        "(lower bound)"
    )


if __name__ == "__main__":
    main()
