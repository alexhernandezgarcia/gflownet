"""
Shared helpers for the classification-tree benchmark scripts.

These benchmarks provide classical baselines (Bayesian CART via MCMC/SMC,
MAPTree, CART, random forest, gradient boosting) for the GFlowNet
classification-tree experiments on the MAGIC gamma telescope dataset. They
load exactly the same split CSVs as gflownet.envs.tree.tree.Tree._load_dataset
(last column "Split" with train/test values, second-to-last column the 0/1
class label) and report test accuracy / F1 / AUC / log-loss.

Each run writes one JSON file to the results directory (default:
class_benchmarks/results next to this file), named
"<method>__<dataset>__split<i>.json". Aggregate them with print_results.py.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

DATASETS = ("magic",)
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
    second-to-last column is the 0/1 class label, the rest are features.
    """
    path = DATA_DIR / dataset / f"{dataset}_{split}.csv"
    df = pd.read_csv(path)
    if df.columns[-1].lower() != "split":
        raise ValueError(f"{path} has no 'Split' column")
    is_train = (df.iloc[:, -1] == "train").to_numpy()
    X = df.iloc[:, :-2].to_numpy(dtype=float)
    y = df.iloc[:, -2].to_numpy(dtype=int)
    if not np.isin(y, (0, 1)).all():
        raise ValueError(f"{path} has non-binary labels")
    return X[is_train], y[is_train], X[~is_train], y[~is_train]


def _metrics(y: npt.NDArray, proba: npt.NDArray, prefix: str) -> Dict[str, float]:
    proba = np.clip(np.asarray(proba, dtype=float), 1e-12, 1 - 1e-12)
    pred = (proba >= 0.5).astype(int)
    return {
        f"{prefix}_acc": float(accuracy_score(y, pred)),
        f"{prefix}_f1": float(f1_score(y, pred)),
        f"{prefix}_auc": float(roc_auc_score(y, proba)),
        f"{prefix}_logloss": float(log_loss(y, proba)),
    }


def classification_metrics(
    y_train: npt.NDArray,
    train_proba: npt.NDArray,
    y_test: npt.NDArray,
    test_proba: npt.NDArray,
) -> Dict[str, float]:
    """Metrics from predicted P(y=1); hard labels are thresholded at 0.5."""
    return {
        **_metrics(y_train, train_proba, "train"),
        **_metrics(y_test, test_proba, "test"),
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
    ``fit_predict(X_train, y_train, X_test, seed) -> (train_proba, test_proba,
    params)`` where the probas are predicted P(y=1) for the train and test
    splits and ``params`` is a JSON-serializable dict describing the model.
    """
    for dataset in args.datasets:
        for split in args.splits:
            seed = split
            X_train, y_train, X_test, y_test = load_split(dataset, split)
            for method, fit_predict in methods.items():
                t0 = time.time()
                train_proba, test_proba, params = fit_predict(
                    X_train, y_train, X_test, seed
                )
                runtime_s = time.time() - t0
                metrics = classification_metrics(
                    y_train, train_proba, y_test, test_proba
                )
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
                    f"test_acc={metrics['test_acc']:.4f} "
                    f"test_auc={metrics['test_auc']:.4f} "
                    f"test_logloss={metrics['test_logloss']:.4f} "
                    f"({runtime_s:.1f}s)",
                    flush=True,
                )
