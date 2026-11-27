"""
Report the range of the log-rewards a Tree GFlowNet was trained on.

The TB loss regresses log Z + sum log P_F - sum log P_B onto the log-reward of
each terminating state, so the scale and spread of log R(x) set the scale of the
loss and of its gradients. This helper reads that range off a finished run,
straight from the local wandb files (no network / wandb API key needed), and
lines it up with the loss so that loss spikes can be attributed (or not) to rare
very-low-reward samples.

What is read (see gflownet/utils/logger.py:log_rewards_and_scores):
  - "Train batch - logrewards mean/max": log R over the terminating states of
    each training batch.
  - "Train batch - scores min/mean/max": raw proxy values. For the tree proxy
    these are log-posteriors (log likelihood + log structure prior). With
    reward_function=exponential and alpha=1, log R = log(alpha) + beta * score,
    so the scores column gives the log-reward *min*, which is not logged
    directly: identically when beta == 1.0, up to the affine map otherwise.
  - "Replay buffer - logrewards mean/max": same for the replay buffer contents.

Usage (from the repo root):
    python gflownet/envs/tree/helpers_for_experiments/logreward_range.py \
        $SCRATCH/gflownet-logs/treeclass_compare/TREECLASS_breast_cancer1_depth6_steps1000
    # all runs under a work directory, one line each:
    python gflownet/envs/tree/helpers_for_experiments/logreward_range.py \
        $SCRATCH/gflownet-logs/treeclass_compare --summary
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCORE_MIN = "Train batch - scores min"
SCORE_MEAN = "Train batch - scores mean"
SCORE_MAX = "Train batch - scores max"
LOGR_MEAN = "Train batch - logrewards mean"
LOGR_MAX = "Train batch - logrewards max"
LOSS = "Loss"


def read_wandb_history(wandb_file: Path) -> pd.DataFrame:
    """
    Parses the history records of a local .wandb file into a DataFrame.

    Parameters
    ----------
    wandb_file : Path
        Path to a run-<id>.wandb file inside a wandb/run-*/ directory.

    Returns
    -------
    pd.DataFrame
        One row per logged step, columns as logged to wandb.
    """
    # Imported here so that the module docstring / --help work without wandb.
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore

    datastore = DataStore()
    datastore.open_for_scan(str(wandb_file))
    rows = []
    while True:
        data = datastore.scan_data()
        if data is None:
            break
        record = pb.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in record.history.item:
            # Scalars carry a single-element nested_key; media/tables carry
            # several (key, "path"/"sha256"/...) and are skipped.
            if len(item.nested_key) != 1:
                continue
            try:
                row[item.nested_key[0]] = json.loads(item.value_json)
            except (json.JSONDecodeError, TypeError):
                continue
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("_step").reset_index(drop=True)


def find_runs(root: Path) -> list:
    """
    Finds the hydra run directories under root that contain wandb history.

    A run directory is one holding a wandb/run-*/ subdirectory; root itself
    qualifies. Only the latest wandb run of each directory is used (a resumed or
    resubmitted job writes several); pass a single wandb/run-<date>-<id>/
    directory to inspect an earlier one instead.
    """
    own_wandb_files = sorted(root.glob("*.wandb"))
    if own_wandb_files:
        return [(root, own_wandb_files[-1])]
    candidates = (
        [root]
        if (root / "wandb").is_dir()
        else sorted(p.parent for p in root.glob("*/wandb") if p.is_dir())
    )
    runs = []
    for run_dir in candidates:
        wandb_files = sorted(
            (run_dir / "wandb").glob("run-*/*.wandb"), key=lambda p: p.stat().st_mtime
        )
        if wandb_files:
            runs.append((run_dir, wandb_files[-1]))
    return runs


def summarize(df: pd.DataFrame) -> dict:
    """
    Computes the log-reward range and the loss statistics of one run.
    """
    stats = {"n_iterations": len(df)}
    if SCORE_MIN in df:
        lo, mid, hi = df[SCORE_MIN], df[SCORE_MEAN], df[SCORE_MAX]
        stats.update(
            {
                "logr_min": lo.min(),
                "logr_max": hi.max(),
                "logr_mean": mid.mean(),
                # Spread inside a single batch: what the TB residuals see at once.
                "batch_spread_median": (hi - lo).median(),
                "batch_spread_max": (hi - lo).max(),
                # How far the worst batches reach below the typical batch floor.
                "batch_min_p50": lo.median(),
                "batch_min_p01": lo.quantile(0.01),
            }
        )
    if LOGR_MEAN in df:
        stats["logr_mean_logged"] = df[LOGR_MEAN].mean()
        stats["logr_max_logged"] = df[LOGR_MAX].max()
    if LOSS in df:
        loss = df[LOSS].dropna()
        stats.update(
            {
                "loss_median": loss.median(),
                "loss_max": loss.max(),
                "loss_max_step": int(df.loc[loss.idxmax(), "_step"]),
            }
        )
    return stats


def report(run_dir: Path, df: pd.DataFrame, n_spikes: int) -> None:
    """
    Prints the full per-run report: log-reward range, loss spikes, correlation.
    """
    stats = summarize(df)
    print(f"\n=== {run_dir.name} ===")
    print(f"iterations logged: {stats['n_iterations']}")
    if "logr_min" not in stats:
        print("no 'Train batch - scores' keys in history; nothing to report")
        return

    print("\nlog-reward range over training (= proxy log-posterior, beta=1):")
    print(f"  overall min           {stats['logr_min']:12.1f}")
    print(f"  overall max           {stats['logr_max']:12.1f}")
    print(f"  mean over batches     {stats['logr_mean']:12.1f}")
    print(f"  span                  {stats['logr_max'] - stats['logr_min']:12.1f} nats")
    print("\nspread *within* one training batch (drives the TB residuals):")
    print(f"  median max-min        {stats['batch_spread_median']:12.1f} nats")
    print(f"  largest max-min       {stats['batch_spread_max']:12.1f} nats")
    print(f"  typical batch floor   {stats['batch_min_p50']:12.1f}")
    print(f"  1% worst batch floor  {stats['batch_min_p01']:12.1f}")

    if LOSS not in df:
        return
    print("\nloss:")
    print(f"  median                {stats['loss_median']:12.4g}")
    print(
        f"  max                   {stats['loss_max']:12.4g}  (iteration {stats['loss_max_step']})"
    )

    # A TB loss of L means an RMS residual of sqrt(L) nats: the comparison with
    # the log-reward span says whether the log-rewards alone can explain it.
    print(f"  RMS residual at median  {np.sqrt(stats['loss_median']):10.1f} nats")
    print(f"  RMS residual at max     {np.sqrt(stats['loss_max']):10.1f} nats")

    corr = df[[SCORE_MIN, LOSS]].dropna().corr().iloc[0, 1]
    print(f"\ncorr(batch min log-reward, loss): {corr:+.3f}")

    spikes = df.nlargest(n_spikes, LOSS)[
        ["_step", LOSS, SCORE_MIN, SCORE_MEAN, SCORE_MAX]
    ]
    spikes = spikes.rename(
        columns={
            "_step": "it",
            LOSS: "loss",
            SCORE_MIN: "logR min",
            SCORE_MEAN: "logR mean",
            SCORE_MAX: "logR max",
        }
    )
    print(f"\n{n_spikes} largest loss spikes:")
    print(spikes.to_string(index=False, float_format=lambda x: f"{x:.4g}"))


def main():
    parser = argparse.ArgumentParser(
        description="Report the log-reward range a Tree GFlowNet was trained on."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="A hydra run directory, or a work directory holding several of them.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="One line per run instead of the full per-run report.",
    )
    parser.add_argument(
        "--n-spikes", type=int, default=5, help="Number of loss spikes to list."
    )
    args = parser.parse_args()

    runs = find_runs(args.path.expanduser())
    if not runs:
        parser.error(f"no wandb run directories found under {args.path}")

    if args.summary:
        rows = []
        for run_dir, wandb_file in runs:
            df = read_wandb_history(wandb_file)
            if df.empty:
                continue
            rows.append({"run": run_dir.name, **summarize(df)})
        table = pd.DataFrame(rows)
        cols = [
            c
            for c in [
                "run",
                "n_iterations",
                "logr_min",
                "logr_max",
                "logr_mean",
                "batch_spread_median",
                "loss_median",
                "loss_max",
            ]
            if c in table
        ]
        print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    else:
        for run_dir, wandb_file in runs:
            df = read_wandb_history(wandb_file)
            if df.empty:
                print(f"\n=== {run_dir.name} ===\nno history records")
                continue
            report(run_dir, df, args.n_spikes)


if __name__ == "__main__":
    main()
