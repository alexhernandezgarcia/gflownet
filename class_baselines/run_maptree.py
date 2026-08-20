"""
MAPTree benchmark (Sullivan et al., AAAI 2024): best-first search for the
maximum-a-posteriori tree of the BCART posterior, on quantile-binarized
features, using the maptree package (pip install git+https://github.com/
ThrunGroup/maptree).

Priors match the MAPTree paper defaults (and run_bcart.py): P(split at depth
d) = alpha_split * (1+d)^(-beta_split), Beta(rho, rho) leaf label prior;
predictions are the smoothed leaf posterior means. If the time limit is hit
before the search proves optimality, the best tree found so far is used
(params record the remaining lower/upper bound gap).

Usage (from the repo root, venv active):
    python class_baselines/run_maptree.py [--splits 1 2 ...] [--time-limit 300]
"""

import sys
from pathlib import Path

from maptree import search as maptree_search

sys.path.insert(0, str(Path(__file__).resolve().parent))
from binary_trees import binarize_quantiles, parse_maptree_string
from common import make_parser, run_methods


def make_fit_predict_maptree(args):
    rho = (args.rho, args.rho)

    def fit_predict(X_train, y_train, X_test, seed):
        B_train, B_test, thresholds = binarize_quantiles(
            X_train, X_test, args.thresholds
        )
        sol = maptree_search(
            B_train.astype(bool),
            y_train.astype(bool),
            args.alpha_split,
            args.beta_split,
            list(rho),
            args.num_expansions,
            args.time_limit,
        )
        tree = parse_maptree_string(sol.tree).fit_counts(B_train, y_train)
        params = {
            "thresholds_per_feature": args.thresholds,
            "n_binary_features": len(thresholds),
            "alpha_split": args.alpha_split,
            "beta_split": args.beta_split,
            "rho": args.rho,
            "time_limit_s": args.time_limit,
            "timeout": bool(sol.lb < sol.ub),
            "lower_bound": float(sol.lb),
            "upper_bound": float(sol.ub),
            "tree_size": tree.size(),
            "tree_depth": tree.depth(),
        }
        return (
            tree.predict_proba(B_train, rho),
            tree.predict_proba(B_test, rho),
            params,
        )

    return fit_predict


if __name__ == "__main__":
    parser = make_parser(__doc__)
    parser.add_argument(
        "--thresholds",
        type=int,
        default=9,
        help="Quantile thresholds per feature for binarization.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        help="Search time limit in seconds (-1 for none).",
    )
    parser.add_argument(
        "--num-expansions",
        type=int,
        default=-1,
        help="Max search-node expansions (-1 for no limit).",
    )
    parser.add_argument("--alpha-split", type=float, default=0.95)
    parser.add_argument("--beta-split", type=float, default=0.5)
    parser.add_argument(
        "--rho", type=float, default=2.5, help="Beta(rho, rho) leaf label prior."
    )
    args = parser.parse_args()
    run_methods({"maptree": make_fit_predict_maptree(args)}, args)
