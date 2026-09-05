"""
Aggregate DT-GFN tree results (classification + regression) across splits.

Collects finished-run metrics from two independent sources and prints, per
dataset, one table per source with the runs grouped by training configuration
(mean +/- std across the n (normally 3 - 5) dataset splits of each configuration):

  - ``eval``:  every ``eval_results.json`` found under a runs root on disk
               (written by eval_tree.py / eval_regression_tree.py; use
               run_missing_evals.py to create the missing ones first).
  - ``wandb``: the LAST logged value of each metric of every run in the
               dt-gfn_classification / dt-gfn_regression wandb projects --
               also for runs that never ran the final evaluation or that live
               on another cluster / your personal machine. The wandb table
               shows the last logged step so partially-trained runs are
               visible as such.

The two sources are never averaged together; they appear as separate tables.
Both tables show the campaign folder as the second column (for wandb runs the
campaign is recovered from the run directory stored in ``logger.logdir.path``;
runs without it -- e.g. old local runs -- fall back to the wandb project name).
The campaign is part of a row's identity: the same configuration launched in
two campaign folders gives two rows with the same config hash, one per
campaign, never a merged one (a ``[note]`` under the table lists such shared
hashes). Rows are ordered by launch date, newest configuration first. The
launch date is the ORIGINAL launch of the run: for eval rows the first
``--- launch`` record of the launcher's ``LAUNCH`` file (fallbacks: the local
``wandb/run-<timestamp>-*`` folder, then the ``.hydra/config.yaml`` mtime),
for wandb rows the earliest ``created_at`` among the relaunches/resumes of the
run. Re-running the evaluation therefore never changes the launch date.

Configurations with fewer than ``--min-splits`` (default 3) dataset splits are
hidden -- a mean over 1-2 splits is not meaningful; pass ``--min-splits 1`` to
see everything.

Metric naming: the training-time / eval-time metric ``mean_n_nodes`` counts
only DECISION (internal) nodes -- ``sum(state["_dones"])`` -- so it is
displayed as ``mean_n_decisionnodes`` here. The eval metrics ``model_size_*``
(classification: eval_tree.py; regression: eval_regression_tree.py) instead
count decision nodes PLUS leaves (``count_total_nodes`` in eval_tree.py); the
two are related by ``total = 2 * decision + 1`` for a binary tree.
``model_size_top1`` is the size of the highest-log-posterior tree, i.e. the
tree behind the top-1 metrics.

How runs are grouped
--------------------
Two runs belong to the same training configuration iff their FULL resolved
hydra configs are identical after removing the run-identity keys below
(dataset split path, run name, log paths, machine-specific ``user`` section,
...). The config is read from the run's ``.hydra/config.yaml`` on disk, or
the config wandb stored at launch. The resulting 8-char group hash is shown
in the tables; use ``--diff-configs`` to see exactly which config keys
distinguish the groups of a dataset when the displayed settings columns look
identical. A row is one (campaign folder, group hash): the hash says WHAT was
trained, the campaign WHERE/WHEN, and both are needed to tell a control
relaunched in a later campaign from its original. Relaunches and resumes of
one run (same campaign, hash and split) are collapsed to one record per
source (see ``dedupe``).

Debugging runs (name or campaign folder matching DEBUG_NAME_PATTERNS) are
reported in a separate section after the real runs (``--only-debug`` prints
only that section, ``--no-debug`` drops it). Their wandb table carries the
extra column ``logZ`` (DEBUG_EXTRA_WANDB_METRICS = last logged value): for the
DEBUG_UNIFORM(_REG) sanity runs the converged logZ IS the result, to be
compared with log(number of trees) from calculate_nbr_of_trees.py. logZ is a
training diagnostic and is deliberately never shown for real runs; it only
exists in wandb, so the eval table of the debug section has no logZ column.

Usage examples (either with active venv active in interactive session or on a compute node with
sbatch mila/tree/aggregate_treeclass_results.sh):

    # everything, both sources
    python gflownet/envs/tree/helpers_for_experiments/aggregate_treeclass_results.py

    # one campaign folder (the wandb runs are restricted to exactly that
    # campaign too: the campaign recovered from logger.logdir.path must match;
    # --campaign A,B does the same for several campaigns under the full root)
    python .../aggregate_treeclass_results.py $SCRATCH/gflownet-logs/TREECLASS_MAGIC

    # only the iris dataset, only wandb, and show config diffs between groups
    python .../aggregate_treeclass_results.py --dataset iris --source wandb \
        --diff-configs

    # only the iris DEBUG runs, with their final logZ (debug runs usually
    # exist for a single split, hence --min-splits 1)
    python .../aggregate_treeclass_results.py --dataset iris --source wandb \
        --only-debug --min-splits 1

For interactive inspection (dataset picker, hash2config) open the notebook
``inspect_treeclass_results.ipynb`` next to this script.
"""

import argparse
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

# Requires the repo to be installed (``pip install -e .``); no sys.path hacks.
from gflownet.envs.tree.helpers_for_experiments.config_hash import (
    EXCLUDED_KEYS,
    config_hash,
    dataset_and_split,
    drop_key,
    policy_label,
)

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
# NOTE model_size_* (classification) = decision nodes + leaves, while
# mean_n_nodes (regression eval + all wandb runs) = decision nodes only.
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
        # Total nodes (decision + leaves), written by eval_regression_tree.py
        # since 2026-09-02; older eval_results.json lack them (shown as "-",
        # re-run with run_missing_evals.py --force).
        "model_size_mean",
        "model_size_top1",
    ],
}

# Metrics read from the wandb run summary (= last logged value), per task.
WANDB_METRICS = {
    "classification": [
        "test_top_1_acc",
        "test_forest_acc",
        "mean_n_nodes",
    ],
    "regression": [
        "test_forest_rmse",
        "test_forest_r2",
        "test_top_1_rmse",
        "test_top_1_r2",
        "mean_n_nodes",
    ],
}

# Extra wandb metrics shown ONLY in the debugging section. Training
# diagnostics such as logZ are meaningless to average over the splits of a real
# training configuration (see test_wandb_metrics_no_training_diagnostics), but
# they are exactly what the DEBUG_UNIFORM(_REG) sanity runs are launched for:
# with a uniform reward the converged logZ must match log(#trees) as computed
# by calculate_nbr_of_trees.py.
DEBUG_EXTRA_WANDB_METRICS = ["logZ"]

# Display-only renames applied to the table columns. The underlying JSON /
# wandb key stays untouched (do NOT rename anything in the training scripts).
# mean_n_nodes is sum(state["_dones"]) = number of DECISION (internal) nodes;
# leaves are not counted (see tree.py / regression_tree.py `test`).
METRIC_DISPLAY_NAMES = {
    "mean_n_nodes": "mean_n_decisionnodes",
}

# Hide configurations averaged over fewer than this many dataset splits.
MIN_SPLITS_DEFAULT = 3

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
    shared = backward.get("shared_weights", None)
    clip = optimizer.get("clip_grad_norm", 0.0) or 0.0
    reward_kwargs = (container.get("proxy", {}) or {}).get(
        "reward_function_kwargs"
    ) or {}

    return {
        "steps": optimizer.get("n_train_steps", "?"),
        "depth": container.get("env", {}).get("max_depth", "?"),
        "lr": optimizer.get("lr", "?"),
        "opt": optimizer.get("method", "?"),
        "beta": reward_kwargs.get("beta", "?"),
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
    "opt",
    "beta",
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


def campaign_from_config(container: dict, fallback: str) -> str:
    """Campaign folder of a run, recovered from ``logger.logdir.path``.

    The launchers create runs at ``.../gflownet-logs/<campaign>/<run_dir>``
    and the Logger stores the run dir as ``logger.logdir.path``; its parent
    directory name is the campaign. Runs without a stored path (old local
    runs, other launch styles) fall back to ``fallback``.
    """
    logdir = (container.get("logger") or {}).get("logdir") or {}
    path = logdir.get("path")
    if path:
        parts = Path(str(path)).parts
        if len(parts) >= 2:
            return parts[-2]
    return fallback


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


def fmt_launch(epoch: float) -> str:
    """Epoch seconds -> 'YYYY-MM-DD HH:MM' in local time, '?' when unknown."""
    if not epoch:
        return "?"
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")


def parse_wandb_created_at(created_at) -> float:
    """wandb ISO timestamp ('2026-08-27T17:49:00Z' or with offset) -> epoch
    seconds, 0.0 when absent or unparseable (sorts last, displays '?')."""
    if not created_at:
        return 0.0
    try:
        return datetime.fromisoformat(
            str(created_at).replace("Z", "+00:00")
        ).timestamp()
    except ValueError:
        return 0.0


# First line of every record the launchers append to <run_dir>/LAUNCH:
# "--- launch 2026-08-28 15:31:51 EDT ---" (one record per launch / resume).
LAUNCH_RECORD_RE = re.compile(
    r"^--- launch (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", re.M
)
# wandb names the local run folder after its start time: run-20260828_153214-<id>.
WANDB_RUN_DIR_RE = re.compile(r"^run-(\d{8}_\d{6})-")


def launch_time_from_run_dir(run_dir: Path, cfg_path: Path) -> float:
    """Epoch seconds of the ORIGINAL launch of the run in ``run_dir``.

    Sources, in order of preference:

    1. The first ``--- launch YYYY-MM-DD HH:MM:SS TZ ---`` record of the
       launcher's ``LAUNCH`` file. Records are appended per launch/resume, so
       the first one is the original launch; the file is never rewritten by
       evaluations, resumes or config edits and survives an rsync.
    2. The earliest local ``wandb/run-YYYYMMDD_HHMMSS-<id>`` folder (runs
       launched without the LAUNCH-writing launchers).
    3. The mtime of ``cfg_path`` (``.hydra/config.yaml``) -- the old
       behaviour, shifted by hand edits and by rsync without ``-t``.

    Timestamps 1 and 2 are wall-clock times of the launching machine (the
    time-zone name is ignored), interpreted in this machine's local time like
    every other timestamp displayed here.
    """
    launch_file = run_dir / "LAUNCH"
    if launch_file.is_file():
        try:
            match = LAUNCH_RECORD_RE.search(launch_file.read_text(errors="replace"))
        except OSError:
            match = None
        if match:
            try:
                return datetime.strptime(
                    match.group(1), "%Y-%m-%d %H:%M:%S"
                ).timestamp()
            except ValueError:
                pass
    wandb_dir = run_dir / "wandb"
    if wandb_dir.is_dir():
        stamps = []
        for entry in wandb_dir.iterdir():
            match = WANDB_RUN_DIR_RE.match(entry.name)
            if match:
                try:
                    stamps.append(
                        datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").timestamp()
                    )
                except ValueError:
                    pass
        if stamps:
            return min(stamps)
    return cfg_path.stat().st_mtime


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
        cfg = OmegaConf.load(cfg_path)
        try:
            container = OmegaConf.to_container(cfg, resolve=True)
        except OmegaConfBaseException as e:
            # An unresolved config still contains ${...} interpolation strings
            # and therefore hashes differently from a resolved one: this run
            # may show up as a spurious extra group. Warn loudly instead of
            # hiding the run entirely.
            print(
                f"[WARN] {run_dir}: could not resolve config "
                f"({type(e).__name__}: {e}); falling back to the UNRESOLVED "
                f"config -- its group hash may not match resolved runs of the "
                f"same configuration."
            )
            container = OmegaConf.to_container(cfg, resolve=False)
        identity = group_identity(container, extra_excluded)
        if identity is None:
            continue
        run_name = str(
            (container.get("logger", {}) or {}).get("run_name") or run_dir.name
        )
        try:
            metrics = json.loads(eval_json.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"[WARN] {run_dir}: unreadable eval_results.json ({e}); skipped.")
            continue
        records.append(
            {
                **identity,
                "run_name": run_name,
                "campaign": run_dir.parent.name,
                "debug": is_debug(run_name, run_dir.relative_to(root)),
                "metrics": metrics,
                # Original launch (LAUNCH file / wandb folder / config mtime,
                # see launch_time_from_run_dir); re-evaluations never move it.
                "launched": launch_time_from_run_dir(run_dir, cfg_path),
                # Dedupe rank: float mtime -- the newest eval wins.
                "recency": eval_json.stat().st_mtime,
            }
        )
    return dedupe(records, "eval")


# =============================================================================
# Source: wandb (last logged values)
# =============================================================================


def wandb_run_selected(run_name, campaign, campaigns=None, name_prefix=None):
    """Selection predicate of collect_wandb_runs.

    ``campaigns`` (a collection of campaign folder names) is an exact match on
    the run's campaign; ``name_prefix`` a prefix match on its name. Either
    filter is skipped when None/empty.
    """
    if campaigns is not None and campaign not in campaigns:
        return False
    if name_prefix and not run_name.startswith(name_prefix):
        return False
    return True


def collect_wandb_runs(
    entity, projects, name_prefix=None, extra_excluded=(), campaigns=None
):
    """One record per wandb run, metrics = last logged (summary) values.

    ``campaigns``: keep only runs whose campaign folder -- the parent of the
    stored ``logger.logdir.path``, see campaign_from_config -- is in this
    collection (exact match, so ``REG_DIVBUF_ON`` does not pull in
    ``REG_DIVBUF_ON_RAP02``; runs without a stored path carry the project name
    as campaign and never match a folder). ``name_prefix`` is an additional,
    purely name-based filter.
    """
    import wandb

    if campaigns is not None:
        campaigns = set(campaigns)
    api = wandb.Api(timeout=120)
    records = []
    for project in projects:
        try:
            runs = api.runs(f"{entity}/{project}", per_page=500)
            n_total = len(runs)
        except Exception as e:
            # Network/API boundary: anything can fail here; report and go on.
            print(f"[WARN] Could not list {entity}/{project}: {e}")
            continue
        print(f"[INFO] wandb: scanning {n_total} runs in {entity}/{project} ...")
        for run in runs:
            try:
                run_name = run.name.strip()
                campaign = campaign_from_config(run.config, fallback=project)
                if not wandb_run_selected(run_name, campaign, campaigns, name_prefix):
                    continue
                identity = group_identity(run.config, extra_excluded)
            except Exception as e:
                # run.name / run.config are lazy API calls; skip broken runs.
                print(f"[WARN] wandb run {getattr(run, 'name', '?')}: {e}; skipped.")
                continue
            if identity is None:
                continue
            debug = is_debug(run_name)
            summary = run.summary
            metrics = {}
            # Debug runs additionally carry the training diagnostics that the
            # debugging section displays (logZ); real runs never do.
            keys = list(WANDB_METRICS[identity["task"]])
            if debug:
                keys += DEBUG_EXTRA_WANDB_METRICS
            for key in keys:
                value = summary.get(key)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    metrics[key] = float(value)
            step = summary.get("_step")
            step = int(step) if isinstance(step, (int, float)) else 0
            launched = parse_wandb_created_at(getattr(run, "created_at", None))
            records.append(
                {
                    **identity,
                    "run_name": run_name,
                    "campaign": campaign,
                    "debug": debug,
                    "metrics": metrics,
                    "state": run.state,
                    "step": step,
                    "launched": launched,
                    # Dedupe rank: the run that progressed furthest wins,
                    # ties broken by launch time (tuples compare elementwise).
                    "recency": (step, launched),
                }
            )
    return dedupe(records, "wandb")


# Per-source meaning of the "recency" dedupe rank, used in the warning text.
DEDUPE_KEEP_RULE = {
    "eval": "keeping the one with the newest eval_results.json",
    "wandb": "keeping the run that progressed furthest (ties: latest launch)",
}


def dedupe(records, source):
    """Keep one record per (task, dataset, campaign, group hash, split).

    Relaunches and resumes of one run leave several records for the same
    configuration, split AND campaign folder (on disk they share the run dir;
    on wandb a resume is a new run of the same name); the record with the
    highest ``recency`` rank wins and inherits the EARLIEST known launch time
    of the duplicates, i.e. the original launch (a resumed wandb run's
    ``created_at`` is the resume time). The same configuration in two
    different campaign folders is two rows and is never deduped.

    The rank has a different (per-source) meaning and type -- see the
    collectors and DEDUPE_KEEP_RULE -- so all records of one call must come
    from one source.
    """
    recency_types = {type(rec["recency"]) for rec in records}
    assert len(recency_types) <= 1, (
        f"dedupe({source}) got mixed recency types {recency_types}; "
        f"records from different sources must not be deduped together."
    )
    best = {}
    for rec in records:
        key = (
            rec["task"],
            rec["dataset"],
            rec["campaign"],
            rec["hash"],
            rec["split"],
            rec["debug"],
        )
        prev = best.get(key)
        if prev is None:
            best[key] = rec
            continue
        print(
            f"[WARN] duplicate {source} runs for {rec['dataset']} split "
            f"{rec['split']} config {rec['hash']} in campaign {rec['campaign']} "
            f"({prev['run_name']} / {rec['run_name']}); "
            f"{DEDUPE_KEEP_RULE[source]}."
        )
        keep, drop = (rec, prev) if rec["recency"] > prev["recency"] else (prev, rec)
        known = [t for t in (keep.get("launched"), drop.get("launched")) if t]
        if known:
            keep["launched"] = min(known)
        best[key] = keep
    return list(best.values())


# =============================================================================
# Grouping and display
# =============================================================================


def build_table(records, metric_names, source, min_splits=1):
    """One row per (campaign folder, training configuration), aggregated over
    splits.

    The same configuration launched in several campaign folders gives one row
    per campaign (same ``config`` hash, different ``campaign``), never a merged
    one. Rows are ordered by launch date (of the most recently launched split
    run of each row), newest first. Rows with fewer than ``min_splits`` splits
    are dropped; returns ``(dataframe, n_hidden)``.
    """
    groups = defaultdict(list)
    for rec in records:
        groups[(rec["task"], rec["dataset"], rec["campaign"], rec["hash"])].append(rec)

    rows = []
    n_hidden = 0
    for (task, dataset, campaign, ghash), recs in groups.items():
        settings = recs[0]["settings"]
        # Same group hash => same config => same settings. If this ever fires
        # the grouping (or an 8-char hash collision) is broken -- fail loudly
        # rather than silently displaying the settings of an arbitrary run.
        for rec in recs[1:]:
            assert rec["settings"] == settings, (
                f"group {ghash} ({dataset}/{task}, campaign {campaign}) contains "
                f"runs with different settings: {settings} vs {rec['settings']} "
                f"(runs {recs[0]['run_name']} / {rec['run_name']})"
            )
        if len(recs) < min_splits:
            n_hidden += 1
            continue
        launched = max(r.get("launched", 0.0) or 0.0 for r in recs)
        row = {
            "config": ghash,
            "campaign": campaign,
            "launched": fmt_launch(launched),
            **settings,
            "n": len(recs),
        }
        if source == "wandb":
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
            row[METRIC_DISPLAY_NAMES.get(metric, metric)] = fmt_mean_std(
                values, len(recs)
            )
        # Newest launch first; deterministic tie-break on settings + hash +
        # campaign.
        row["_sort"] = (
            -launched,
            tuple(str(settings[c]) for c in SETTINGS_COLUMNS),
            ghash,
            campaign,
        )
        rows.append(row)

    rows.sort(key=lambda r: r["_sort"])
    for row in rows:
        del row["_sort"]
    return pd.DataFrame(rows), n_hidden


def shared_config_notes(records):
    """One note per group hash that was launched in more than one campaign
    folder: such a configuration has one table row per campaign."""
    campaigns = defaultdict(set)
    for rec in records:
        campaigns[(rec["task"], rec["dataset"], rec["hash"])].add(rec["campaign"])
    return [
        f"[note] config {ghash} was launched in {len(names)} campaigns "
        f"({', '.join(sorted(names))}): identical training configuration, "
        f"one row per campaign."
        for (_, _, ghash), names in sorted(campaigns.items())
        if len(names) > 1
    ]


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


def print_section(
    title, records_by_source, diff_configs, min_splits, extra_wandb_metrics=()
):
    """Per dataset: the eval table, then the wandb table.

    ``extra_wandb_metrics`` are appended to the wandb table of this section
    only (used by the debugging section to show logZ).
    """
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
            print(f"\n--- {label}: mean ± std over splits, newest launch first ---")
            metrics = list(metric_map[task])
            if source == "wandb":
                metrics += [m for m in extra_wandb_metrics if m not in metrics]
            df, n_hidden = build_table(recs, metrics, source, min_splits)
            if len(df):
                print(df.to_string(index=False))
            if n_hidden:
                print(
                    f"[note] {n_hidden} configuration(s) with fewer than "
                    f"{min_splits} splits hidden (pass --min-splits 1 to show)."
                )
            for line in shared_config_notes(recs):
                print(line)
        if diff_configs:
            recs = [
                r
                for recs in records_by_source.values()
                for r in recs
                if (r["task"], r["dataset"]) == (task, dataset)
            ]
            print_config_diffs(recs, task, dataset)


def campaigns_from_args(campaign_arg, root: Path):
    """The campaign filter (a set of folder names) or None for no filter.

    ``--campaign A,B`` selects exactly those; ``--campaign ''`` disables the
    filter; without the option the root folder name is used when ``root`` is a
    single campaign folder rather than the default runs root.
    """
    if campaign_arg is not None:
        return {c for c in campaign_arg.split(",") if c} or None
    if root.resolve() != default_root().resolve():
        return {root.name}
    return None


def filter_records(records_by_source, datasets=None, task="both", campaigns=None):
    """Apply the --dataset / --task / --campaign filters to every source."""
    filtered = {}
    for source, recs in records_by_source.items():
        if datasets is not None:
            recs = [r for r in recs if r["dataset"] in datasets]
        if task != "both":
            recs = [r for r in recs if r["task"] == task]
        if campaigns is not None:
            recs = [r for r in recs if r["campaign"] in campaigns]
        filtered[source] = recs
    return filtered


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
        "given, the wandb runs are restricted to exactly that campaign too "
        "(see --campaign).",
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
        "--min-splits",
        type=int,
        default=MIN_SPLITS_DEFAULT,
        help="Hide configurations with fewer than this many dataset splits "
        "(default: %(default)s; pass 1 to show everything).",
    )
    parser.add_argument(
        "--campaign",
        default=None,
        help="Comma-separated campaign folder names; only runs of these "
        "campaigns are reported, from both sources (wandb: exact match on the "
        "campaign recovered from logger.logdir.path). Default: the folder name "
        "when root is a single campaign folder; pass --campaign '' to report "
        "every campaign under such a root as well.",
    )
    parser.add_argument(
        "--name-prefix",
        default=None,
        help="Additionally keep only wandb runs whose name starts with this "
        "prefix (an explicit filter; no longer derived from root).",
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
    debug_group = parser.add_mutually_exclusive_group()
    debug_group.add_argument(
        "--no-debug", action="store_true", help="Hide the debugging-runs section."
    )
    debug_group.add_argument(
        "--only-debug",
        action="store_true",
        help="Print ONLY the debugging-runs section (which is the only one "
        "with a logZ column); the real training runs are hidden.",
    )
    args = parser.parse_args()

    extra_excluded = args.group_ignore.split(",") if args.group_ignore else ()
    datasets = set(args.dataset.split(",")) if args.dataset else None

    campaigns = campaigns_from_args(args.campaign, args.root)
    if campaigns is not None and args.campaign is None:
        print(
            f"[INFO] runs filtered to campaign folder '{args.root.name}' on both "
            f"sources (exact match; pass --campaign '' to disable)."
        )

    records_by_source = {}
    if args.source in ("eval", "both"):
        records_by_source["eval"] = collect_eval_runs(args.root, extra_excluded)
    if args.source in ("wandb", "both"):
        records_by_source["wandb"] = collect_wandb_runs(
            args.entity,
            [p for p in args.projects.split(",") if p],
            args.name_prefix or None,
            extra_excluded,
            campaigns=campaigns,
        )

    records_by_source = filter_records(
        records_by_source, datasets=datasets, task=args.task, campaigns=campaigns
    )

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

    if not args.only_debug:
        print_section("TRAINING RUNS", real, args.diff_configs, args.min_splits)
    if not args.no_debug:
        if args.only_debug and not sum(len(recs) for recs in debug.values()):
            print("No matching debugging runs found.")
            return
        print_section(
            "DEBUGGING RUNS (name matches: " + ", ".join(DEBUG_NAME_PATTERNS) + ")",
            debug,
            args.diff_configs,
            args.min_splits,
            extra_wandb_metrics=DEBUG_EXTRA_WANDB_METRICS,
        )


if __name__ == "__main__":
    main()
