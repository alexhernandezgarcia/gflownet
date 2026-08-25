"""
Prints, for each dataset, a table with the average and standard deviation of
test RMSE and R2 across splits for every benchmark method found in the
results directory (JSON files written by the run_*.py scripts).

Usage (from the repo root):
    python reg_benchmarks/print_results.py [--results-dir DIR]
                                            [--train]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import common
import numpy as np

METHOD_ORDER = [
    "bart",
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


def fmt(values):
    return f"{np.mean(values):8.4f} ± {np.std(values):7.4f}"


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
        methods = sorted(
            [m for d, m in results if d == dataset],
            key=lambda m: (
                METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER)
            ),
        )
        header = (
            f"{'method':<16} {'n':>2}  {'test RMSE (avg ± std)':<24}"
            f"{'test R2 (avg ± std)':<24}"
        )
        if args.train:
            header += f"{'train RMSE (avg ± std)':<24}{'train R2 (avg ± std)':<24}"
        print(f"\n=== {dataset} ===")
        print(header)
        print("-" * len(header))
        for method in methods:
            runs = results[(dataset, method)]
            splits = sorted(r["split"] for r in runs)
            row = (
                f"{method:<16} {len(runs):>2}  "
                f"{fmt([r['test_rmse'] for r in runs]):<24}"
                f"{fmt([r['test_r2'] for r in runs]):<24}"
            )
            if args.train:
                row += (
                    f"{fmt([r['train_rmse'] for r in runs]):<24}"
                    f"{fmt([r['train_r2'] for r in runs]):<24}"
                )
            if splits != list(common.SPLITS):
                row += f"  [splits: {splits}]"
            print(row)
    print()


if __name__ == "__main__":
    main()
