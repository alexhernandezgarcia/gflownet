"""
Greedy-tree baselines for the classification-tree experiments: CART with the
Gini and entropy splitting criteria and a (greedy) random forest, on the raw
continuous features, with --max-depth (default 5) matching the GFlowNet
trees. For non-default depths the method names are suffixed with _d<depth>
(e.g. cart_gini_d3) so results of several depths coexist in the results dir.

For the single CART trees, predicted probabilities are Beta(2.5, 2.5)-
smoothed leaf label frequencies -- the same leaf posterior mean the Bayesian
baselines (run_bcart.py, run_maptree.py) use -- so log-loss stays finite on
pure leaves. The random forest averages the unsmoothed per-tree leaf
frequencies (sklearn predict_proba).

Usage (from the repo root, venv active):
    python class_baselines/run_cart.py [--datasets magic] [--splits 1 2 ...]
                                         [--max-depth 5]
"""

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_parser, run_methods

DEFAULT_MAX_DEPTH = 5
N_TREES = 100
RHO = (2.5, 2.5)
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))


def _smoothed_leaf_proba(tree, X_train, y_train):
    """
    Returns a function X -> P(y=1) predicting the Beta(RHO)-smoothed label
    frequency of the train points routed to the same leaf.
    """
    n0 = np.zeros(tree.tree_.node_count)
    n1 = np.zeros(tree.tree_.node_count)
    leaves = tree.apply(X_train)
    np.add.at(n0, leaves[y_train == 0], 1)
    np.add.at(n1, leaves[y_train == 1], 1)
    proba = (n1 + RHO[1]) / (n0 + n1 + RHO[0] + RHO[1])
    return lambda X: proba[tree.apply(X)]


def make_fit_predict_cart(criterion, max_depth):
    def fit_predict(X_train, y_train, X_test, seed):
        tree = DecisionTreeClassifier(
            criterion=criterion, max_depth=max_depth, random_state=seed
        )
        tree.fit(X_train, y_train)
        predict = _smoothed_leaf_proba(tree, X_train, y_train)
        params = {
            "criterion": criterion,
            "max_depth": max_depth,
            "rho": RHO[0],
            "n_leaves": int(tree.get_n_leaves()),
        }
        return predict(X_train), predict(X_test), params

    return fit_predict


def make_fit_predict_random_forest(max_depth):
    def fit_predict(X_train, y_train, X_test, seed):
        forest = RandomForestClassifier(
            n_estimators=N_TREES, max_depth=max_depth, random_state=seed, n_jobs=N_JOBS
        )
        forest.fit(X_train, y_train)
        params = {"n_estimators": N_TREES, "max_depth": max_depth}
        return (
            forest.predict_proba(X_train)[:, 1],
            forest.predict_proba(X_test)[:, 1],
            params,
        )

    return fit_predict


def make_methods(max_depth):
    suffix = "" if max_depth == DEFAULT_MAX_DEPTH else f"_d{max_depth}"
    return {
        f"cart_gini{suffix}": make_fit_predict_cart("gini", max_depth),
        f"cart_entropy{suffix}": make_fit_predict_cart("entropy", max_depth),
        f"random_forest{suffix}": make_fit_predict_random_forest(max_depth),
    }


if __name__ == "__main__":
    parser = make_parser(__doc__)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    args = parser.parse_args()
    run_methods(make_methods(args.max_depth), args)
