"""
Aggregate transformer-policy composite-Tree evaluation results across splits.

Sibling of aggregate_treeclass_results.py, specialised for the transformer-policy
tree sweep launched by the (now-deleted) mila/sbatch/trfm_tree_remaining.sh-style
scripts. Every run is named on wandb / hydra as

    trfm_<dataset>_depth<max_depth>_split<seed>

with dataset in {iris, wine, raisin, breast_cancer}, max_depth in {1, 3, 5} and
seed in 1..5. Only runs whose directory name matches that exact pattern are
analyzed (there are no other lower-case run families to confuse them with, and
the ALL-CAPS TREECLASS_* runs are handled by the sibling script).

Runs that differ only in the split seed are averaged together; runs that differ
in ANY other setting (dataset, max_depth, n_train_steps, n_samples, alpha_value)
are kept as separate groups. For each group it reports mean +/- std across seeds
for the SAME three metrics as the legacy baseline aggregator (helper/
aggregate_results.py), so the transformer table lines up column for column with
the baseline and TREECLASS tables:

  - top-1 test accuracy   (tree with highest log-posterior)
  - mean number of nodes  (over all sampled trees)
  - BMA test accuracy     (uniform ensemble = Monte-Carlo posterior predictive)

How the metrics are obtained
----------------------------
Nothing is read from wandb. For each run this loads the trees the trained
GFlowNet sampled at the end of training (gfn_samples.pkl) and re-runs the exact
evaluation protocol of gflownet/envs/tree/eval_tree.py on them: it re-derives the
per-tree log-posteriors (not the reward-scaled energies saved in the pickle),
selects the top-1 tree, and runs Dirichlet posterior-predictive / BMA prediction
against the dataset's held-out test split. The RNG is fixed (seed 0) so the
Dirichlet draws are reproducible and consistent with how the baselines are
scored. Results are cached next to each pickle (metrics_cache.json), so only the
first invocation is slow.

Two trfm-specific wrinkles vs. the TREECLASS aggregator:
  * dataset and seed are NOT in the config; they are parsed from the run-dir name
    (the pattern above). max_depth / n_train_steps / n_samples / alpha_value come
    from the run's top-level .hydra/config.yaml.
  * resumed runs write their final samples to resume/<jobid>/<ts>/gfn_samples.pkl
    and a *resume* .hydra config that lacks those fields; so the config is always
    read from the run's top-level .hydra, and the final pickle is taken from
    samples/ if present, else the newest resume/*/*/gfn_samples.pkl (exactly the
    pickle the sbatch script fed to eval_tree.py).

Usage (from the repo root, with the gflownet venv active):
    python gflownet/envs/tree/aggregate_treeclass_trfm_results.py
    python gflownet/envs/tree/aggregate_treeclass_trfm_results.py \
        --logs-root $SCRATCH/gflownet-logs/trfm_sweep_10132875 --no-cache
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]  # <repo>/gflownet/envs/tree/helpers_for_experiments
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
EXPECTED_SEEDS = 5

# Only directories whose basename matches this are analyzed. `dataset` is greedy
# so it swallows underscores (e.g. breast_cancer) while depth/seed stay anchored.
RUN_RE = re.compile(r"^trfm_(?P<dataset>.+)_depth(?P<depth>\d+)_split(?P<seed>\d+)$")

# Settings that define an experiment group (everything except the split seed).
GROUP_KEYS = ["dataset", "max_depth", "n_train_steps", "n_samples", "alpha_value"]

# Metrics reported, in output order. Names match the legacy aggregator so the
# baseline, TREECLASS and transformer tables can be lined up column for column.
METRICS = ["top1_test_acc", "mean_nodes", "bma_test_acc"]


def default_logs_root() -> Path:
    """Root under which to search; overridable via $WORK_DIR."""
    work_dir = os.environ.get("WORK_DIR")
    if work_dir:
        return Path(work_dir)
    scratch = os.environ.get("SCRATCH", str(Path.home() / "scratch"))
    return Path(scratch) / "gflownet-logs"


def find_final_samples(run_dir: Path):
    """The pickle the sbatch script fed to eval_tree.py for this run.

    Prefer samples/gfn_samples.pkl (single-shot training); otherwise the newest
    resume/<jobid>/<ts>/gfn_samples.pkl (the most-progressed resumed chain).
    Returns None if the run has not produced samples yet.
    """
    direct = run_dir / "samples" / "gfn_samples.pkl"
    if direct.exists():
        return direct
    resume_pkls = list(run_dir.glob("resume/*/*/gfn_samples.pkl"))
    if resume_pkls:
        return max(resume_pkls, key=lambda p: p.stat().st_mtime)
    return None


def find_runs(logs_root: Path):
    """Yield (run_dir, pkl_path, config, match) for every finished trfm run.

    A run is identified by its top-level .hydra/config.yaml whose parent dir name
    matches RUN_RE. Nested resume/<jobid>/<ts>/.hydra dirs are skipped because
    their parent is a timestamp, not a trfm_* name.
    """
    seen = set()
    for hydra_dir in sorted(logs_root.rglob(".hydra")):
        run_dir = hydra_dir.parent
        m = RUN_RE.match(run_dir.name)
        cfg_path = hydra_dir / "config.yaml"
        if m is None or run_dir in seen or not cfg_path.exists():
            continue
        pkl_path = find_final_samples(run_dir)
        if pkl_path is None:
            print(f"[SKIP] {run_dir.name}: no gfn_samples.pkl yet (still training?)")
            continue
        seen.add(run_dir)
        yield run_dir, pkl_path, OmegaConf.load(cfg_path), m


def resolve_data_path(data_path: str) -> str:
    """Datasets are stored as absolute paths baked at train time; relocate them
    under this checkout if the repo has since moved."""
    p = Path(data_path)
    if p.exists():
        return str(p)
    parts = p.parts
    if "tests" in parts:
        cand = REPO_ROOT / Path(*parts[parts.index("tests") :])
        if cand.exists():
            return str(cand)
    return str(p)  # let downstream raise a clear FileNotFoundError


def run_metadata(run_dir: Path, cfg, m) -> dict:
    """Group settings + seed from the run-dir name and top-level hydra config."""
    try:
        meta = {
            "run_name": run_dir.name,
            "dataset": m.group("dataset"),
            "seed": int(m.group("seed")),
            "max_depth": int(cfg.env.max_depth),
            "n_train_steps": int(cfg.gflownet.optimizer.n_train_steps),
            "n_samples": int(cfg.n_samples),
            "alpha_value": float(cfg.proxy.alpha_value),
            "data_path": resolve_data_path(str(cfg.env.data_path)),
        }
    except Exception as e:
        print(f"[WARN] Could not parse config for {run_dir.name} ({e}); skipping.")
        return None
    # The depth in the name and the config must agree, else the name is stale.
    if int(m.group("depth")) != meta["max_depth"]:
        print(
            f"[WARN] {run_dir.name}: name depth {m.group('depth')} != config "
            f"max_depth {meta['max_depth']}; trusting the config."
        )
    return meta


def compute_metrics(
    pkl_path: Path, meta: dict, n_dirichlet_samples: int, use_cache: bool
) -> dict:
    """Return the three metrics for one run, using the on-disk cache if valid."""
    cache_path = pkl_path.parent / CACHE_NAME
    pkl_mtime = pkl_path.stat().st_mtime
    cache_key = {
        "pkl_mtime": pkl_mtime,
        "n_dirichlet_samples": n_dirichlet_samples,
        "alpha_value": meta["alpha_value"],
    }
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

    # Fix the RNG so the Dirichlet draws in the evaluation are reproducible and
    # consistent with how the baselines are scored.
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
        "--logs-root",
        type=Path,
        default=default_logs_root(),
        help="Root of the runs directory to scan recursively (default: "
        "$WORK_DIR or $SCRATCH/gflownet-logs). Only trfm_*_depth*_split* run "
        "dirs are analyzed.",
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

    # Collect runs; deduplicate (group, seed) keeping the newest pickle.
    selected = {}
    for run_dir, pkl_path, cfg, m in find_runs(args.logs_root):
        meta = run_metadata(run_dir, cfg, m)
        if meta is None:
            continue
        key = tuple(meta[k] for k in GROUP_KEYS) + (meta["seed"],)
        mtime = pkl_path.stat().st_mtime
        if key in selected:
            print(
                f"[WARN] Duplicate run for "
                f"{dict(zip(GROUP_KEYS + ['seed'], key))}; keeping the newest."
            )
            if mtime <= selected[key][0]:
                continue
        selected[key] = (mtime, pkl_path, meta)

    if not selected:
        print(f"No finished trfm_*_depth*_split* runs found under {args.logs_root}.")
        return

    # Compute per-run metrics and group across seeds.
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
        groups.setdefault(group, []).append((meta["seed"], metrics))

    rows = []
    for group, runs in sorted(groups.items()):
        seeds = sorted(s for s, _ in runs)
        row = dict(zip(GROUP_KEYS, group))
        row["n_seeds"] = len(seeds)
        row["seeds"] = ",".join(map(str, seeds))
        for metric in METRICS:
            values = np.array([m[metric] for _, m in runs], dtype=float)
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            row[metric] = f"{values.mean():.4f} ± {std:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(GROUP_KEYS).reset_index(drop=True)
    print(
        "\nTransformer-policy tree results "
        "(mean ± std across split seeds, sample std ddof=1):\n"
    )
    print(df.to_string(index=False))

    short = df[df["n_seeds"] != EXPECTED_SEEDS]
    if not short.empty:
        print(
            f"\n[WARN] {len(short)} group(s) do not have the expected "
            f"{EXPECTED_SEEDS} seeds — see the n_seeds/seeds columns above."
        )


if __name__ == "__main__":
    main()
