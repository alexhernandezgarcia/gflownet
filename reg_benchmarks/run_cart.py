"""
Greedy-tree baselines for the regression-tree experiments: a depth-capped CART
regression tree, a cost-complexity-pruned CART tree and a (greedy) random
forest, on the raw continuous features.

--max-depth (default 5) matches the GFlowNet trees and caps "cart" and
"random_forest"; for non-default depths those method names are suffixed with
_d<depth> (e.g. cart_d3) so results of several depths coexist in the results
dir. "cart_pruned" is the classic CART baseline: grown unbounded and then
cost-complexity pruned with ccp_alpha selected by 5-fold CV, so its capacity is
chosen by the data rather than by --max-depth (hence no depth suffix).

Leaf predictions are the train-split target means of the leaf (sklearn's
default), so RMSE / R2 are in the original target units; trees are
scale-invariant, hence no feature/target scaling is applied.

Usage (from the repo root, venv active):
    python reg_benchmarks/run_cart.py [--datasets concrete ...] [--splits 1 2 ...]
                                      [--max-depth 5]
"""

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_parser, run_methods

DEFAULT_MAX_DEPTH = 5
N_TREES = 100
N_FOLDS = 5
MAX_CCP_ALPHAS = 100
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))


def make_fit_predict_cart(max_depth):
    def fit_predict(X_train, y_train, X_test, seed):
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=seed)
        tree.fit(X_train, y_train)
        params = {
            "criterion": "squared_error",
            "max_depth": max_depth,
            "n_leaves": int(tree.get_n_leaves()),
        }
        return tree.predict(X_train), tree.predict(X_test), params

    return fit_predict


def fit_predict_cart_pruned(X_train, y_train, X_test, seed):
    """
    Classic CART: an unbounded tree pruned by cost-complexity, with ccp_alpha
    picked by 5-fold CV over the pruning path of the train split.
    """
    path = DecisionTreeRegressor(random_state=seed).cost_complexity_pruning_path(
        X_train, y_train
    )
    alphas = np.unique(np.clip(path.ccp_alphas, 0.0, None))
    if len(alphas) > MAX_CCP_ALPHAS:
        idx = np.unique(
            np.linspace(0, len(alphas) - 1, MAX_CCP_ALPHAS).round().astype(int)
        )
        alphas = alphas[idx]

    search = GridSearchCV(
        DecisionTreeRegressor(random_state=seed),
        {"ccp_alpha": alphas},
        cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed),
        scoring="neg_root_mean_squared_error",
        n_jobs=N_JOBS,
    )
    search.fit(X_train, y_train)
    tree = search.best_estimator_
    params = {
        "criterion": "squared_error",
        "ccp_alpha": float(search.best_params_["ccp_alpha"]),
        "n_ccp_alphas": int(len(alphas)),
        "max_depth": int(tree.get_depth()),
        "n_leaves": int(tree.get_n_leaves()),
    }
    return tree.predict(X_train), tree.predict(X_test), params


def make_fit_predict_random_forest(max_depth):
    def fit_predict(X_train, y_train, X_test, seed):
        forest = RandomForestRegressor(
            n_estimators=N_TREES, max_depth=max_depth, random_state=seed, n_jobs=N_JOBS
        )
        forest.fit(X_train, y_train)
        params = {"n_estimators": N_TREES, "max_depth": max_depth}
        return forest.predict(X_train), forest.predict(X_test), params

    return fit_predict


def make_methods(max_depth):
    suffix = "" if max_depth == DEFAULT_MAX_DEPTH else f"_d{max_depth}"
    return {
        f"cart{suffix}": make_fit_predict_cart(max_depth),
        "cart_pruned": fit_predict_cart_pruned,
        f"random_forest{suffix}": make_fit_predict_random_forest(max_depth),
    }


if __name__ == "__main__":
    parser = make_parser(__doc__)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    args = parser.parse_args()
    run_methods(make_methods(args.max_depth), args)
