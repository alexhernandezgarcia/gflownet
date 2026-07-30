"""
Aggregate composite-Tree (TREECLASS) evaluation results across dataset splits.

Counterpart of the legacy helper/aggregate_results.py, adapted to the composite
Tree environment of this repo. Scans the work directory for finished runs
(identified by samples/gfn_samples.pkl), keeps those whose hydra run name starts
with the given ALL-CAPS prefix (default TREECLASS), groups them by experiment
settings (dataset, max_depth, n_train_steps, n_samples, alpha_value) and reports
mean +/- std across dataset splits for the SAME three metrics as the legacy
baseline aggregator, so the two tables are directly comparable:

  - top-1 test accuracy   (tree with highest log-posterior)
  - mean number of nodes  (over all sampled trees)
  - BMA test accuracy     (uniform ensemble = Monte-Carlo posterior predictive)

Unlike the legacy env, the composite runs carry no env.dataset / env.split_seed /
env.n_thresholds fields: the dataset name and split are parsed from the CSV file
name (env.data_path = .../<dataset>/<dataset>_<split>.csv) and thresholds are
continuous, so n_thresholds drops out of the grouping.

Metrics are recomputed from gfn_samples.pkl with gflownet.envs.tree.eval_tree
(the exact evaluation protocol used by eval_tree.py's CLI: log-posteriors are
recomputed, not read from the saved energies) and cached next to the pickle
(metrics_cache.json), so only the first invocation is slow.

Usage (from the repo root):
    python gflownet/envs/tree/aggregate_treeclass_results.py
    python gflownet/envs/tree/aggregate_treeclass_results.py TREECLASS \
        --logs-root $SCRATCH/gflownet-logs/treeclass_compare --no-cache
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]  # <repo>/gflownet/envs/tree -> <repo>
sys.path.insert(0, str(REPO_ROOT))  # make `gflownet` importable when run directly

import pandas as pd
import torch
from omegaconf import OmegaConf

from gflownet.envs.tree.eval_tree import (
    bayesian_model_averaging,
    calculate_tree_accuracies,
    compute_log_posterior,
    load_and_scale_dataset,
)
from gflownet.envs.tree.node import DecisionTreeNode
from gflownet.envs.tree.tree import Tree

CACHE_NAME = "metrics_cache.json"
EXPECTED_SPLITS = 5

# Settings that define an experiment group (everything except the dataset split).
GROUP_KEYS = ["dataset", "max_depth", "n_train_steps", "n_samples", "alpha_value"]

# Metrics reported, in output order. Names match the legacy aggregator so the
# baseline and composite tables can be lined up column for column.
METRICS = ["top1_test_acc", "mean_nodes", "bma_test_acc"]


def default_logs_root() -> Path:
    """Mirror the sbatch scripts' WORK_DIR default."""
    work_dir = os.environ.get("WORK_DIR")
    if work_dir:
        return Path(work_dir)
    scratch = os.environ.get("SCRATCH", str(Path.home() / "scratch"))
    return Path(scratch) / "gflownet-logs" / "treeclass_compare"


def find_runs(logs_root: Path):
    """Yield (pkl_path, config) for every finished run under logs_root."""
    for pkl in sorted(logs_root.rglob("gfn_samples.pkl")):
        pkl_dir = pkl.parent
        # samples/ sits inside the hydra run dir (hydra.job.chdir=True), so the
        # .hydra folder is one level up; check the pkl dir too, just in case.
        for candidate in (pkl_dir, pkl_dir.parent):
            cfg_path = candidate / ".hydra" / "config.yaml"
            if cfg_path.exists():
                yield pkl, OmegaConf.load(cfg_path)
                break


def run_metadata(cfg):
    """Extract run name, group settings and split from a hydra config."""
    try:
        run_name = str(cfg.logger.run_name)
        data_path = str(cfg.env.data_path)
        # env.data_path = .../<dataset>/<dataset>_<split>.csv
        dataset, split_str = Path(data_path).stem.rsplit("_", 1)
        meta = {
            "run_name": run_name,
            "dataset": dataset,
            "split": int(split_str),
            "max_depth": int(cfg.env.max_depth),
            "n_train_steps": int(cfg.gflownet.optimizer.n_train_steps),
            "n_samples": int(cfg.n_samples),
            "alpha_value": float(cfg.proxy.alpha_value),
            "data_path": data_path,
        }
    except Exception as e:
        print(f"[WARN] Could not parse config ({e}); skipping run.")
        return None
    return meta


def compute_metrics(
    pkl_path: Path, meta: dict, n_dirichlet_samples: int, use_cache: bool
) -> dict:
    """Return the three metrics for one run, using the on-disk cache if valid."""
    cache_path = pkl_path.parent / CACHE_NAME
    pkl_mtime = pkl_path.stat().st_mtime
    cache_key = {"pkl_mtime": pkl_mtime, "n_dirichlet_samples": n_dirichlet_samples}
    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if all(cached.get(k) == v for k, v in cache_key.items()):
            return cached["metrics"]

    import pickle

    with open(pkl_path, "rb") as f:
        dct = pickle.load(f)
    states = dct["x"]

    X_train, y_train, X_test, y_test = load_and_scale_dataset(meta["data_path"])
    if X_test is None or y_test is None:
        raise RuntimeError(f"No test split for {pkl_path}")
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]
    alpha = np.ones(n_classes) * meta["alpha_value"]

    _, _, _, _, feature_names = Tree._load_dataset(meta["data_path"])
    if feature_names is not None:
        features = list(feature_names)
    else:
        features = [f"x{i}" for i in range(n_features)]
    node_env = DecisionTreeNode(features=features)

    # Recompute log-posteriors consistently (eval_tree.py does the same rather
    # than trusting the saved reward-scaled energies); they rank the top-1 tree
    # and weight the BMA ensemble.
    log_posteriors = np.array(
        [
            compute_log_posterior(
                state, X_train, y_train, alpha, n_classes, n_features, node_env
            )
            for state in states
        ]
    )

    # Fix the RNG so the Dirichlet draws in the evaluation are reproducible.
    torch.manual_seed(0)
    np.random.seed(0)

    tree_stats = calculate_tree_accuracies(
        states,
        log_posteriors,
        X_train,
        y_train,
        X_test,
        y_test,
        alpha,
        n_classes,
        node_env,
        n_dirichlet_samples,
    )
    bma_stats = bayesian_model_averaging(
        states,
        log_posteriors,
        X_train,
        y_train,
        X_test,
        y_test,
        alpha,
        n_classes,
        node_env,
        n_dirichlet_samples,
    )

    metrics = {
        "top1_test_acc": tree_stats["test_acc_top1"],
        "mean_nodes": tree_stats["model_size_mean"],
        "bma_test_acc": bma_stats["bma_test_acc_uniform"],
    }
    cache_path.write_text(json.dumps({**cache_key, "metrics": metrics}))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "prefix",
        nargs="?",
        default="TREECLASS",
        help="ALL-CAPS run-name prefix to analyze (default: TREECLASS)",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=default_logs_root(),
        help="Root of the runs directory (default: $WORK_DIR or "
        "$SCRATCH/gflownet-logs/treeclass_compare)",
    )
    parser.add_argument(
        "--n-dirichlet-samples",
        type=int,
        default=10,
        help="Dirichlet draws to average predictions over (default: 10, "
        "matching eval_tree.py)",
    )
    parser.add_argument("--no-cache", action="store_true", help="Force recomputation")
    args = parser.parse_args()

    # Collect runs matching the prefix; deduplicate (group, split) keeping newest.
    selected = {}
    for pkl_path, cfg in find_runs(args.logs_root):
        meta = run_metadata(cfg)
        if meta is None or not meta["run_name"].startswith(args.prefix):
            continue
        key = tuple(meta[k] for k in GROUP_KEYS) + (meta["split"],)
        mtime = pkl_path.stat().st_mtime
        if key in selected:
            print(
                f"[WARN] Duplicate run for "
                f"{dict(zip(GROUP_KEYS + ['split'], key))}; keeping the newest."
            )
            if mtime <= selected[key][0]:
                continue
        selected[key] = (mtime, pkl_path, meta)

    if not selected:
        print(
            f"No finished runs found under {args.logs_root} "
            f"with run name starting with '{args.prefix}'."
        )
        return

    # Compute per-run metrics and group across splits.
    groups = {}
    for _, pkl_path, meta in selected.values():
        print(f"[INFO] {meta['run_name']}  ({pkl_path.parent})")
        try:
            metrics = compute_metrics(
                pkl_path, meta, args.n_dirichlet_samples, use_cache=not args.no_cache
            )
        except Exception as e:
            print(f"[WARN] Failed to compute metrics for {pkl_path}: {e}")
            continue
        group = tuple(meta[k] for k in GROUP_KEYS)
        groups.setdefault(group, []).append((meta["split"], metrics))

    rows = []
    for group, runs in sorted(groups.items()):
        splits = sorted(s for s, _ in runs)
        row = dict(zip(GROUP_KEYS, group))
        row["n_splits"] = len(splits)
        row["splits"] = ",".join(map(str, splits))
        for metric in METRICS:
            values = np.array([m[metric] for _, m in runs], dtype=float)
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            row[metric] = f"{values.mean():.4f} ± {std:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(GROUP_KEYS).reset_index(drop=True)
    print(
        f"\nResults for prefix '{args.prefix}' "
        f"(mean ± std across dataset splits, sample std ddof=1):\n"
    )
    print(df.to_string(index=False))

    short = df[df["n_splits"] != EXPECTED_SPLITS]
    if not short.empty:
        print(
            f"\n[WARN] {len(short)} group(s) do not have the expected "
            f"{EXPECTED_SPLITS} splits — see the n_splits/splits columns above."
        )


if __name__ == "__main__":
    main()
