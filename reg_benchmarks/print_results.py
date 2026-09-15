"""
Prints, for each dataset, a table with the average and standard deviation of
test RMSE and R2 across splits for every benchmark method found in the
results directory (JSON files written by the run_*.py scripts).

For the single-tree methods (SINGLE_TREE_METHODS: bcart_map,
single_tree_bcart_mcmc, single_tree_bcart_smc, cart, cart_pruned and cart's
_d<depth> variants) a "tree size" column reports the
number of nodes -- decision nodes AND leaves -- of the one tree whose
predictions the RMSE / R2 columns are computed from, averaged over the
splits. It is read from the result JSON: ``params.tree_size`` (written by
run_bcart.py, BinaryRegTree.size) or, for the sklearn CART trees, derived
from ``params.n_leaves`` as 2 * n_leaves - 1 (sklearn trees are full binary
trees: every decision node has exactly two children). Ensembles and
non-tree models show "-" in that column.

Usage (from the repo root):
    python reg_benchmarks/print_results.py [--results-dir DIR]
                                            [--train]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import common
import numpy as np

METHOD_ORDER = [
    "bart",
    "bcart_mcmc",
    "bcart_smc",
    "bcart_map",
    "single_tree_bcart_mcmc",
    "single_tree_bcart_smc",
    "cart",
    "cart_pruned",
    "random_forest",
    "xgboost",
    "lightgbm",
    "gp",
    "linear",
    "ridge",
    "lasso",
    "bayesian_ridge",
]

# Methods whose reported metrics come from ONE tree (base names; depth
# variants such as cart_d3 are matched through base_method()).
SINGLE_TREE_METHODS = {
    "bcart_map",
    "single_tree_bcart_mcmc",
    "single_tree_bcart_smc",
    "cart",
    "cart_pruned",
}


def fmt(values):
    return f"{np.mean(values):8.4f} ± {np.std(values):7.4f}"


def base_method(method):
    """Strips a _d<depth> suffix (cart_d3 -> cart)."""
    return re.sub(r"_d\d+$", "", method)


def method_order_key(method):
    """Sorts depth variants (e.g. cart_d3) next to their base method."""
    base = base_method(method)
    rank = METHOD_ORDER.index(base) if base in METHOD_ORDER else len(METHOD_ORDER)
    return (rank, method)


def tree_size(res):
    """Total node count (decision nodes + leaves) of a single-tree result, or
    None when the JSON carries no size information."""
    params = res.get("params", {})
    if "tree_size" in params:
        return int(params["tree_size"])
    if "n_leaves" in params:
        return 2 * int(params["n_leaves"]) - 1
    return None


def fmt_tree_size(method, runs):
    """'avg ± std' of the tree size over the splits for single-tree methods,
    '-' for the other models, 'n/a' when a single-tree result lacks the size."""
    if base_method(method) not in SINGLE_TREE_METHODS:
        return "-"
    sizes = [tree_size(r) for r in runs]
    if any(s is None for s in sizes):
        return "n/a"
    return f"{np.mean(sizes):6.1f} ± {np.std(sizes):5.1f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=common.DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--train", action="store_true", help="Also show train-split metrics."
    )
    args = parser.parse_args()

    results = defaultdict(list)  # (dataset, method) -> list of result dicts
    for path in sorted(args.results_dir.glob("*.json")):
        with open(path) as f:
            res = json.load(f)
        results[(res["dataset"], res["method"])].append(res)

    if not results:
        print(f"No result JSONs found in {args.results_dir}")
        return

    datasets = sorted({dataset for dataset, _ in results})
    for dataset in datasets:
        methods = sorted([m for d, m in results if d == dataset], key=method_order_key)
        header = (
            f"{'method':<24} {'n':>2}  {'test RMSE (avg ± std)':<24}"
            f"{'test R2 (avg ± std)':<24}"
        )
        if args.train:
            header += f"{'train RMSE (avg ± std)':<24}{'train R2 (avg ± std)':<24}"
        header += f"{'tree size (nodes)':<19}"
        print(f"\n=== {dataset} ===")
        print(header)
        print("-" * len(header))
        for method in methods:
            runs = results[(dataset, method)]
            splits = sorted(r["split"] for r in runs)
            row = (
                f"{method:<24} {len(runs):>2}  "
                f"{fmt([r['test_rmse'] for r in runs]):<24}"
                f"{fmt([r['test_r2'] for r in runs]):<24}"
            )
            if args.train:
                row += (
                    f"{fmt([r['train_rmse'] for r in runs]):<24}"
                    f"{fmt([r['train_r2'] for r in runs]):<24}"
                )
            row += f"{fmt_tree_size(method, runs):<19}"
            if splits != list(common.SPLITS):
                row += f"  [splits: {splits}]"
            print(row)
    print()


if __name__ == "__main__":
    main()
