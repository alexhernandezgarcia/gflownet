"""
Prints, for each dataset, a table with the average and standard deviation of
test accuracy, F1, AUC and log-loss across splits for every benchmark method
found in the results directory (JSON files written by the run_*.py scripts).

Usage (from the repo root):
    python class_baselines/print_results.py [--results-dir class_baselines/results]
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
    "bcart_mcmc",
    "bcart_smc",
    "bcart_map",
    "maptree",
    "cart_gini",
    "cart_entropy",
    "random_forest",
    "xgboost",
    "catboost",
    "lightgbm",
]


def fmt(values):
    return f"{np.mean(values):7.4f} ± {np.std(values):6.4f}"


def method_order_key(method):
    """Sorts depth variants (e.g. cart_gini_d3) next to their base method."""
    base = re.sub(r"_d\d+$", "", method)
    rank = METHOD_ORDER.index(base) if base in METHOD_ORDER else len(METHOD_ORDER)
    return (rank, method)


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

    metrics = ["acc", "f1", "auc", "logloss"]
    datasets = sorted({dataset for dataset, _ in results})
    for dataset in datasets:
        methods = sorted([m for d, m in results if d == dataset], key=method_order_key)
        header = f"{'method':<16} {'n':>2}  " + "".join(
            f"{'test ' + metric:<19}" for metric in metrics
        )
        if args.train:
            header += "".join(f"{'train ' + metric:<19}" for metric in metrics)
        print(f"\n=== {dataset} ===  (avg ± std over splits)")
        print(header)
        print("-" * len(header))
        for method in methods:
            runs = results[(dataset, method)]
            splits = sorted(r["split"] for r in runs)
            row = f"{method:<16} {len(runs):>2}  " + "".join(
                f"{fmt([r[f'test_{metric}'] for r in runs]):<19}" for metric in metrics
            )
            if args.train:
                row += "".join(
                    f"{fmt([r[f'train_{metric}'] for r in runs]):<19}"
                    for metric in metrics
                )
            if splits != list(common.SPLITS):
                row += f"  [splits: {splits}]"
            print(row)
    print()


if __name__ == "__main__":
    main()
