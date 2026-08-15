"""
Shared helpers for the regression-tree benchmark scripts.

These benchmarks provide classical baselines (BART, gradient boosting, GP,
linear models) for the GFlowNet regression-tree experiments launched via
mila/sbatch/regression_tree_diabetes.sh. They load exactly the same split
CSVs as gflownet.envs.tree.tree.Tree._load_dataset (last column "Split" with
train/test values, second-to-last column the target) and report test RMSE / R2
in the original target units, matching RegressionTree.test.

Each run writes one JSON file to the results directory (default:
reg_benchmarks/results next to this file), named
"<method>__<dataset>__split<i>.json". Aggregate them with print_results.py.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

DATASETS = ("concrete", "diabetes", "energy")
SPLITS = (1, 2, 3, 4, 5)

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
DATA_DIR = REPO_ROOT / "tests" / "data" / "tree"
DEFAULT_RESULTS_DIR = BENCHMARK_DIR / "results"


def load_split(
    dataset: str, split: int
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Loads one train/test split CSV, replicating the CSV branch of
    Tree._load_dataset: the last column is "Split" (train/test), the
    second-to-last column is the continuous target, the rest are features.
    """
    path = DATA_DIR / dataset / f"{dataset}_{split}.csv"
    df = pd.read_csv(path)
    if df.columns[-1].lower() != "split":
        raise ValueError(f"{path} has no 'Split' column")
    is_train = (df.iloc[:, -1] == "train").to_numpy()
    X = df.iloc[:, :-2].to_numpy(dtype=float)
    y = df.iloc[:, -2].to_numpy(dtype=float)
    return X[is_train], y[is_train], X[~is_train], y[~is_train]


def regression_metrics(
    y_train: npt.NDArray,
    train_pred: npt.NDArray,
    y_test: npt.NDArray,
    test_pred: npt.NDArray,
) -> Dict[str, float]:
    return {
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "test_r2": float(r2_score(y_test, test_pred)),
    }


def save_result(
    results_dir: Path,
    method: str,
    dataset: str,
    split: int,
    seed: int,
    metrics: Dict[str, float],
    params: Dict,
    runtime_s: float,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "method": method,
        "dataset": dataset,
        "split": split,
        "seed": seed,
        **metrics,
        "params": params,
        "runtime_s": round(runtime_s, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = results_dir / f"{method}__{dataset}__split{split}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return path


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASETS),
        choices=list(DATASETS),
        help="Datasets to run on (default: all).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=int,
        default=list(SPLITS),
        choices=list(SPLITS),
        help="Train/test splits to run (default: 1-5). The RNG seed of each "
        "run equals the split id, mirroring how the splits were generated.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Output directory for result JSONs (default: {DEFAULT_RESULTS_DIR}).",
    )
    return parser


def run_methods(
    methods: Dict[str, Callable],
    args: argparse.Namespace,
) -> None:
    """
    Runs every method on every requested (dataset, split) combination.

    Each entry of ``methods`` maps a method name to a callable
    ``fit_predict(X_train, y_train, X_test, seed) -> (train_pred, test_pred,
    params)`` where the predictions are in the original target units and
    ``params`` is a JSON-serializable dict describing the fitted model.
    """
    for dataset in args.datasets:
        for split in args.splits:
            seed = split
            X_train, y_train, X_test, y_test = load_split(dataset, split)
            for method, fit_predict in methods.items():
                t0 = time.time()
                train_pred, test_pred, params = fit_predict(
                    X_train, y_train, X_test, seed
                )
                runtime_s = time.time() - t0
                metrics = regression_metrics(y_train, train_pred, y_test, test_pred)
                save_result(
                    args.results_dir,
                    method,
                    dataset,
                    split,
                    seed,
                    metrics,
                    params,
                    runtime_s,
                )
                print(
                    f"[{method}] {dataset} split {split}: "
                    f"test_rmse={metrics['test_rmse']:.4f} "
                    f"test_r2={metrics['test_r2']:.4f} ({runtime_s:.1f}s)",
                    flush=True,
                )
