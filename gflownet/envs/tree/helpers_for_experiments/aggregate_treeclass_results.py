"""
Aggregate DT-GFN tree results (classification + regression) across splits.

Collects finished-run metrics from two independent sources and prints, per
dataset, one table per source with the runs grouped by training configuration
(mean +/- std across the 3-5 dataset splits of each configuration):

  - ``eval``:  every ``eval_results.json`` found under a runs root on disk
               (written by eval_tree.py / eval_regression_tree.py; use
               run_missing_evals.py to create the missing ones first).
  - ``wandb``: the LAST logged value of each metric of every run in the
               dt-gfn_classification / dt-gfn_regression wandb projects --
               also for runs that never ran the final evaluation or that live
               on another cluster / your personal machine. The wandb table
               shows the last logged step so partially-trained runs are
               visible as such.

The two sources are never averaged together; they appear as separate tables,
listing the training configurations in the same order so they can be compared
line by line.

How runs are grouped
--------------------
Two runs belong to the same training configuration iff their FULL resolved
hydra configs are identical after removing the run-identity keys below
(dataset split path, run name, log paths, machine-specific ``user`` section,
...). The config is read from the authoritative source -- the run's
``.hydra/config.yaml`` on disk, or the config wandb stored at launch -- never
parsed from the run name. The resulting 8-char group hash is shown in the
tables; use ``--diff-configs`` to see exactly which config keys distinguish
the groups of a dataset when the displayed settings columns look identical.

Debugging runs (name or campaign folder matching DEBUG_NAME_PATTERNS) are
reported in a separate section after the real runs.

Usage (venv active; light enough for a login node when using --source wandb,
otherwise prefer a compute node or sbatch mila/tree/aggregate_treeclass_results.sh):

    # everything, both sources
    python gflownet/envs/tree/helpers_for_experiments/aggregate_treeclass_results.py

    # one campaign folder (also filters the wandb runs to that name prefix)
    python .../aggregate_treeclass_results.py $SCRATCH/gflownet-logs/TREECLASS_MAGIC

    # only the iris dataset, only wandb, and show config diffs between groups
    python .../aggregate_treeclass_results.py --dataset iris --source wandb \
        --diff-configs
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]  # <repo>/gflownet/envs/tree/helpers_for_experiments
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
from config_hash import (
    EXCLUDED_KEYS,
    config_hash,
    dataset_and_split,
    drop_key,
    policy_label,
)
from omegaconf import OmegaConf

# =============================================================================
# What defines a group, what is displayed -- edit here to extend the report
# =============================================================================

# Keys removed from the resolved config before hashing it into the group id.
# On top of the per-job noise already excluded by config_hash.py, everything
# that legitimately differs between two runs of the SAME configuration is
# dropped: the dataset split (env.data_path), the whole logger section (run
# name / project / log paths), the machine-specific user section, and the keys
# wandb itself adds to the stored config.
GROUP_EXCLUDED_KEYS = EXCLUDED_KEYS + [
    "env.data_path",
    "logger",
    "user",
    "slurm_job_id",
    "_wandb",
]

# A run whose name or campaign folder matches one of these (case-insensitive
# substrings) is reported in the DEBUGGING section instead of the real one.
DEBUG_NAME_PATTERNS = ["debug", "smoke", "prof", "timing"]

# env._target_ -> task; runs with any other env are ignored.
TASK_BY_ENV_TARGET = {
    "gflownet.envs.tree.tree.Tree": "classification",
    "gflownet.envs.tree.regression_tree.RegressionTree": "regression",
}

# Metrics read from eval_results.json, per task, in display order.
EVAL_METRICS = {
    "classification": [
        "test_acc_top1",
        "bma_test_acc_weighted",
        "bma_test_acc_uniform",
        "model_size_mean",
        "model_size_std",
        "model_size_top1",
    ],
    "regression": [
        "test_forest_rmse",
        "test_forest_r2",
        "test_top_1_rmse",
        "test_top_1_r2",
        "mean_n_nodes",
    ],
}

# Metrics read from the wandb run summary (= last logged value), per task.
WANDB_METRICS = {
    "classification": [
        "test_top_1_acc",
        "test_forest_acc",
        "logZ",
        "Loss",
        "Train batch - logrewards mean",
        "mean_n_nodes",
    ],
    "regression": [
        "test_forest_rmse",
        "test_forest_r2",
        "test_top_1_rmse",
        "test_top_1_r2",
        "logZ",
        "Loss",
        "Train batch - logrewards mean",
        "mean_n_nodes",
    ],
}

WANDB_ENTITY = "alex-hg"
WANDB_PROJECTS = ["dt-gfn_classification", "dt-gfn_regression"]


def settings_from_config(container: dict) -> dict:
    """The human-readable hyperparameters shown for each training configuration.

    Read from the resolved config, with defensive defaults for configs written
    before a key existed. Add a key here (and nothing else) to display another
    setting.
    """
    optimizer = container.get("gflownet", {}).get("optimizer", {})
    backward = (container.get("policy", {}) or {}).get("backward") or {}
    clip = optimizer.get("clip_grad_norm", 0.0) or 0.0
    shared = backward.get("shared_weights", None)
    return {
        "steps": optimizer.get("n_train_steps", "?"),
        "depth": container.get("env", {}).get("max_depth", "?"),
        "lr": optimizer.get("lr", "?"),
        "policy": policy_label(container),
        "seed": container.get("seed", "?"),
        "clip": clip if clip else "-",
        "pb_shared": {True: "yes", False: "no"}.get(shared, "?"),
        "rand_prob": container.get("gflownet", {}).get("random_action_prob", "?"),
    }


SETTINGS_COLUMNS = [
    "steps",
    "depth",
    "lr",
    "policy",
    "seed",
    "clip",
    "pb_shared",
    "rand_prob",
]


# =============================================================================
# Shared helpers
# =============================================================================


def default_root() -> Path:
    scratch = os.environ.get("SCRATCH", str(Path.home() / "scratch"))
    return Path(scratch) / "gflownet-logs"


def normalize_numbers(obj):
    """Integer-valued floats -> int, recursively. wandb round-trips 1.0 as 1,
    while the yaml on disk yields 1.0; normalizing makes the group hash (and
    the displayed settings) identical no matter which source the config came
    from."""
    if isinstance(obj, dict):
        return {k: normalize_numbers(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_numbers(v) for v in obj]
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    return obj


def group_identity(container: dict, extra_excluded=()):
    """(task, dataset, split, group_hash, settings, hashed_container) of a run.

    ``container`` is the fully resolved config as a plain dict, from either
    ``.hydra/config.yaml`` or the wandb-stored config; both yield the same
    hash for the same configuration. Returns None for non-tree runs.
    """
    task = TASK_BY_ENV_TARGET.get(str(container.get("env", {}).get("_target_", "")))
    if task is None:
        return None
    dataset, split = dataset_and_split(container)
    hashed = normalize_numbers(json.loads(json.dumps(container, default=str)))
    for key in list(GROUP_EXCLUDED_KEYS) + list(extra_excluded):
        drop_key(hashed, key)
    return {
        "task": task,
        "dataset": dataset,
        "split": split or "?",
        "hash": config_hash(hashed, 8),
        "settings": settings_from_config(hashed),
        "config": hashed,
    }


def is_debug(*names) -> bool:
    text = " ".join(str(n).lower() for n in names)
    return any(pat in text for pat in DEBUG_NAME_PATTERNS)


def fmt_mean_std(values, n_expected):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return "-"
    mean = values.mean()
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    prec = 1 if abs(mean) >= 100 else 4
    cell = f"{mean:.{prec}f} ± {std:.{prec}f}"
    if len(values) != n_expected:
        cell += f" [n={len(values)}]"
    return cell


# =============================================================================
# Source: eval_results.json on disk
# =============================================================================


def collect_eval_runs(root: Path, extra_excluded=()):
    """One record per evaluated run directory under root."""
    records = []
    for eval_json in sorted(root.rglob("eval_results.json")):
        run_dir = eval_json.parent
        rel_parts = run_dir.relative_to(root).parts
        if "resume" in rel_parts or "wandb" in rel_parts:
            continue
        cfg_path = run_dir / ".hydra" / "config.yaml"
        if not cfg_path.exists():
            print(
                f"[WARN] {run_dir}: eval_results.json but no .hydra/config.yaml; skipped."
            )
            continue
        try:
            cfg = OmegaConf.load(cfg_path)
            container = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            container = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=False)
        identity = group_identity(container, extra_excluded)
        if identity is None:
            continue
        run_name = str(
            (container.get("logger", {}) or {}).get("run_name") or run_dir.name
        )
        try:
            metrics = json.loads(eval_json.read_text())
        except Exception as e:
            print(f"[WARN] {run_dir}: unreadable eval_results.json ({e}); skipped.")
            continue
        records.append(
            {
                **identity,
                "run_name": run_name,
                "campaign": run_dir.parent.name,
                "debug": is_debug(run_name, run_dir.relative_to(root)),
                "metrics": metrics,
                "recency": eval_json.stat().st_mtime,
            }
        )
    return dedupe(records, "eval")


# =============================================================================
# Source: wandb (last logged values)
# =============================================================================


def collect_wandb_runs(entity, projects, name_prefix=None, extra_excluded=()):
    """One record per wandb run, metrics = last logged (summary) values."""
    import wandb

    api = wandb.Api(timeout=120)
    records = []
    for project in projects:
        try:
            runs = api.runs(f"{entity}/{project}", per_page=500)
            n_total = len(runs)
        except Exception as e:
            print(f"[WARN] Could not list {entity}/{project}: {e}")
            continue
        print(f"[INFO] wandb: scanning {n_total} runs in {entity}/{project} ...")
        for run in runs:
            try:
                if name_prefix and not run.name.strip().startswith(name_prefix):
                    continue
                identity = group_identity(run.config, extra_excluded)
            except Exception as e:
                print(f"[WARN] wandb run {getattr(run, 'name', '?')}: {e}; skipped.")
                continue
            if identity is None:
                continue
            summary = run.summary
            metrics = {}
            for key in WANDB_METRICS[identity["task"]]:
                value = summary.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    metrics[key] = float(value)
            step = summary.get("_step")
            records.append(
                {
                    **identity,
                    "run_name": run.name.strip(),
                    "campaign": project,
                    "debug": is_debug(run.name),
                    "metrics": metrics,
                    "state": run.state,
                    "step": int(step) if isinstance(step, (int, float)) else 0,
                    "recency": (
                        int(step) if isinstance(step, (int, float)) else 0,
                        str(run.created_at),
                    ),
                }
            )
    return dedupe(records, "wandb")


def dedupe(records, source):
    """Keep one record per (task, dataset, group hash, split): the most recent
    on disk, or the wandb run that progressed furthest (relaunches and resumes
    can leave several runs for the same configuration and split)."""
    best = {}
    for rec in records:
        key = (rec["task"], rec["dataset"], rec["hash"], rec["split"], rec["debug"])
        if key in best:
            print(
                f"[WARN] duplicate {source} runs for {rec['dataset']} split "
                f"{rec['split']} config {rec['hash']} "
                f"({best[key]['run_name']} / {rec['run_name']}); keeping the newest."
            )
        if key not in best or rec["recency"] > best[key]["recency"]:
            best[key] = rec
    return list(best.values())


# =============================================================================
# Grouping and display
# =============================================================================


def build_table(records, metric_names, source):
    """One row per training configuration, aggregated over splits."""
    groups = defaultdict(list)
    for rec in records:
        groups[(rec["task"], rec["dataset"], rec["hash"])].append(rec)

    rows = []
    for (task, dataset, ghash), recs in groups.items():
        settings = recs[0]["settings"]
        splits = sorted(r["split"] for r in recs)
        row = {
            "config": ghash,
            **settings,
            "n": len(recs),
            "splits": ",".join(splits),
        }
        if source == "eval":
            row["campaign"] = ",".join(sorted({r["campaign"] for r in recs}))
        else:
            steps = [r["step"] for r in recs]
            row["last_step"] = (
                str(steps[0])
                if min(steps) == max(steps)
                else f"{min(steps)}-{max(steps)}"
            )
            states = sorted({r["state"] for r in recs})
            row["state"] = ",".join(states)
        for metric in metric_names:
            values = [r["metrics"][metric] for r in recs if metric in r["metrics"]]
            row[metric] = fmt_mean_std(values, len(recs))
        # Deterministic order shared by the eval and wandb tables.
        row["_sort"] = tuple(str(settings[c]) for c in SETTINGS_COLUMNS) + (ghash,)
        rows.append(row)

    rows.sort(key=lambda r: r["_sort"])
    for row in rows:
        del row["_sort"]
    return pd.DataFrame(rows)


def print_config_diffs(records, task, dataset):
    """Show which config keys distinguish the groups of one dataset."""

    def flatten(d, prefix=""):
        flat = {}
        for key, value in d.items():
            dotted = f"{prefix}{key}"
            if isinstance(value, dict):
                flat.update(flatten(value, dotted + "."))
            else:
                flat[dotted] = json.dumps(value) if isinstance(value, list) else value
        return flat

    by_hash = {}
    for rec in records:
        by_hash.setdefault(rec["hash"], flatten(rec["config"]))
    if len(by_hash) < 2:
        return
    all_keys = sorted(set().union(*[set(f) for f in by_hash.values()]))
    diff_rows = []
    for key in all_keys:
        values = {h: f.get(key, "<absent>") for h, f in by_hash.items()}
        if len({json.dumps(v, default=str) for v in values.values()}) > 1:
            diff_rows.append(
                {"config key": key, **{h: str(v)[:40] for h, v in values.items()}}
            )
    if diff_rows:
        print(f"\n  Config keys that differ between the groups of {dataset} ({task}):")
        print(
            "  " + pd.DataFrame(diff_rows).to_string(index=False).replace("\n", "\n  ")
        )


def print_section(title, records_by_source, diff_configs):
    """Per dataset: the eval table, then the wandb table."""
    printed_header = False
    tasks_datasets = sorted(
        {(r["task"], r["dataset"]) for recs in records_by_source.values() for r in recs}
    )
    for task, dataset in tasks_datasets:
        if not printed_header:
            print("\n" + "#" * 78 + f"\n# {title}\n" + "#" * 78)
            printed_header = True
        print(f"\n=== {dataset} ({task}) " + "=" * max(0, 55 - len(dataset)))
        for source, metric_map in (("eval", EVAL_METRICS), ("wandb", WANDB_METRICS)):
            recs = [
                r
                for r in records_by_source.get(source, [])
                if (r["task"], r["dataset"]) == (task, dataset)
            ]
            if not recs:
                continue
            label = (
                "eval_results.json (final evaluation)"
                if source == "eval"
                else "wandb (last logged values)"
            )
            print(f"\n--- {label}: mean ± std over splits ---")
            df = build_table(recs, metric_map[task], source)
            print(df.to_string(index=False))
        if diff_configs:
            recs = [
                r
                for recs in records_by_source.values()
                for r in recs
                if (r["task"], r["dataset"]) == (task, dataset)
            ]
            print_config_diffs(recs, task, dataset)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=default_root(),
        help="Runs root: the whole gflownet-logs tree or a single campaign "
        "folder (default: $SCRATCH/gflownet-logs). When a campaign folder is "
        "given, wandb runs are filtered to names starting with its name.",
    )
    parser.add_argument(
        "--source",
        choices=["eval", "wandb", "both"],
        default="both",
        help="Which sources to report (default: both). Sources are never "
        "averaged together.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Comma-separated dataset names to report (default: all).",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "both"],
        default="both",
        help="Restrict to one task (default: both).",
    )
    parser.add_argument(
        "--name-prefix",
        default=None,
        help="Only include wandb runs whose name starts with this prefix "
        "(default: the campaign folder name when root is a single campaign).",
    )
    parser.add_argument(
        "--entity",
        default=WANDB_ENTITY,
        help=f"wandb entity (default: {WANDB_ENTITY}).",
    )
    parser.add_argument(
        "--projects",
        default=",".join(WANDB_PROJECTS),
        help="Comma-separated wandb projects (default: %(default)s).",
    )
    parser.add_argument(
        "--group-ignore",
        default=None,
        help="Comma-separated extra dotted config keys to exclude from the "
        "grouping hash (e.g. evaluator.checkpoints_period,torch_profile).",
    )
    parser.add_argument(
        "--diff-configs",
        action="store_true",
        help="For each dataset, print the config keys that differ between its "
        "groups (explains groups whose displayed settings look identical).",
    )
    parser.add_argument(
        "--no-debug", action="store_true", help="Hide the debugging-runs section."
    )
    args = parser.parse_args()

    extra_excluded = args.group_ignore.split(",") if args.group_ignore else ()
    datasets = set(args.dataset.split(",")) if args.dataset else None

    name_prefix = args.name_prefix
    if name_prefix is None and args.root.resolve() != default_root().resolve():
        name_prefix = args.root.name
        print(
            f"[INFO] wandb runs filtered to name prefix '{name_prefix}' "
            f"(pass --name-prefix '' to disable)."
        )

    records_by_source = {}
    if args.source in ("eval", "both"):
        records_by_source["eval"] = collect_eval_runs(args.root, extra_excluded)
    if args.source in ("wandb", "both"):
        records_by_source["wandb"] = collect_wandb_runs(
            args.entity,
            [p for p in args.projects.split(",") if p],
            name_prefix or None,
            extra_excluded,
        )

    # Filters
    for source, recs in records_by_source.items():
        if datasets is not None:
            recs = [r for r in recs if r["dataset"] in datasets]
        if args.task != "both":
            recs = [r for r in recs if r["task"] == args.task]
        records_by_source[source] = recs

    n_total = sum(len(r) for r in records_by_source.values())
    if n_total == 0:
        print("No matching runs found.")
        return

    real = {
        s: [r for r in recs if not r["debug"]] for s, recs in records_by_source.items()
    }
    debug = {
        s: [r for r in recs if r["debug"]] for s, recs in records_by_source.items()
    }

    print_section("TRAINING RUNS", real, args.diff_configs)
    if not args.no_debug:
        print_section(
            "DEBUGGING RUNS (name matches: " + ", ".join(DEBUG_NAME_PATTERNS) + ")",
            debug,
            args.diff_configs,
        )


if __name__ == "__main__":
    main()
