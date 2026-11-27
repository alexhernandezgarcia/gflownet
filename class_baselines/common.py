"""
Shared helpers for the classification-tree benchmark scripts.

These benchmarks provide classical baselines (Bayesian CART via MCMC/SMC,
MAPTree, CART, random forest, gradient boosting) for the GFlowNet
classification-tree experiments. They load exactly the same split CSVs as
gflownet.envs.tree.tree.Tree._load_dataset (last column "Split" with
train/test values, second-to-last column the integer class label in 0..K-1)
and report test accuracy / F1 / AUC / log-loss. Multi-class datasets
(jannis4) use argmax accuracy, macro F1, macro one-vs-rest AUC and
multi-class log-loss; the binarized-feature methods (run_bcart.py,
run_maptree.py) are binary-only and skip them.

Each run writes one JSON file to the results directory (default:
$SCRATCH/gflownet-benchmarks/class_baselines/results, see
DEFAULT_RESULTS_DIR below), named
"<method>__<dataset>__split<i>.json". Aggregate them with print_results.py.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

DATASETS = ("magic", "credit", "credit_quantile", "jannis2", "jannis4")
SPLITS = (1, 2, 3, 4, 5)

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parent
DATA_DIR = REPO_ROOT / "tests" / "data" / "tree"

# Results are written outside the repo, under one directory per benchmark
# suite (<root>/<benchmark_dir_name>/results), so that they survive branch
# switches and the $SLURM_TMPDIR repo snapshots the training launchers take.
# The root is $GFLOWNET_BENCHMARKS_DIR if set, else $SCRATCH/gflownet-benchmarks
# (~/scratch/gflownet-benchmarks if $SCRATCH is unset).
BENCHMARKS_ROOT = Path(
    os.environ.get(
        "GFLOWNET_BENCHMARKS_DIR",
        Path(os.environ.get("SCRATCH", Path.home() / "scratch"))
        / "gflownet-benchmarks",
    )
)
DEFAULT_RESULTS_DIR = BENCHMARKS_ROOT / BENCHMARK_DIR.name / "results"


def load_split(
    dataset: str, split: int
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Loads one train/test split CSV, replicating the CSV branch of
    Tree._load_dataset: the last column is "Split" (train/test), the
    second-to-last column is the integer class label in 0..K-1, the rest are
    features.
    """
    path = DATA_DIR / dataset / f"{dataset}_{split}.csv"
    df = pd.read_csv(path)
    if df.columns[-1].lower() != "split":
        raise ValueError(f"{path} has no 'Split' column")
    is_train = (df.iloc[:, -1] == "train").to_numpy()
    X = df.iloc[:, :-2].to_numpy(dtype=float)
    y = df.iloc[:, -2].to_numpy(dtype=int)
    classes = np.unique(y)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError(f"{path} labels must be 0..K-1, got {classes.tolist()}")
    return X[is_train], y[is_train], X[~is_train], y[~is_train]


def _metrics(y: npt.NDArray, proba: npt.NDArray, prefix: str) -> Dict[str, float]:
    """
    Metrics from predicted probabilities. ``proba`` is either a 1-D array of
    P(y=1) (binary) or an (n, K) matrix of class probabilities; a two-column
    matrix is reduced to its P(y=1) column so binary results are identical
    either way. For K > 2 classes: argmax accuracy, macro F1, macro
    one-vs-rest AUC and multi-class log-loss.
    """
    proba = np.clip(np.asarray(proba, dtype=float), 1e-12, 1 - 1e-12)
    if proba.ndim == 2 and proba.shape[1] == 2:
        proba = proba[:, 1]
    if proba.ndim == 1:
        pred = (proba >= 0.5).astype(int)
        return {
            f"{prefix}_acc": float(accuracy_score(y, pred)),
            f"{prefix}_f1": float(f1_score(y, pred)),
            f"{prefix}_auc": float(roc_auc_score(y, proba)),
            f"{prefix}_logloss": float(log_loss(y, proba)),
        }
    n_classes = proba.shape[1]
    labels = np.arange(n_classes)
    pred = np.argmax(proba, axis=1)
    # Renormalize after clipping so log_loss/roc_auc get a valid distribution.
    proba = proba / proba.sum(axis=1, keepdims=True)
    return {
        f"{prefix}_acc": float(accuracy_score(y, pred)),
        f"{prefix}_f1": float(f1_score(y, pred, average="macro")),
        f"{prefix}_auc": float(
            roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=labels)
        ),
        f"{prefix}_logloss": float(log_loss(y, proba, labels=labels)),
    }


def classification_metrics(
    y_train: npt.NDArray,
    train_proba: npt.NDArray,
    y_test: npt.NDArray,
    test_proba: npt.NDArray,
) -> Dict[str, float]:
    """Metrics from predicted P(y=1) (binary, thresholded at 0.5) or from
    (n, K) class-probability matrices (multi-class, argmax)."""
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
    binary_only: bool = False,
) -> None:
    """
    Runs every method on every requested (dataset, split) combination.

    Each entry of ``methods`` maps a method name to a callable
    ``fit_predict(X_train, y_train, X_test, seed) -> (train_proba, test_proba,
    params)`` where the probas are predicted P(y=1) (1-D) or (n, K) class
    probabilities for the train and test splits and ``params`` is a
    JSON-serializable dict describing the model. With ``binary_only=True``,
    datasets with more than two classes are skipped with a message (for the
    binarized-feature methods, which only handle binary labels).
    """
    for dataset in args.datasets:
        for split in args.splits:
            seed = split
            X_train, y_train, X_test, y_test = load_split(dataset, split)
            if binary_only and np.unique(y_train).size > 2:
                print(
                    f"[skip] {dataset} split {split}: "
                    f"{np.unique(y_train).size} classes, binary-only methods"
                )
                continue
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
