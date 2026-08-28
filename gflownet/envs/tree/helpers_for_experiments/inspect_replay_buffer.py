"""
Inspect the replay buffer(s) of tree runs: diversity, rewards, staleness.

The replay buffer (``<run_dir>/data/replay.csv``, written by
``gflownet/buffer/base.py``) holds the best terminal states seen during
training. Because the GFlowNet draws its backward-replay batch from it, a
degenerate buffer both signals and *reinforces* mode collapse. This script
answers, per run:

  - Is the buffer DIVERSE or does it hold one tree over and over?
    * unique exact states (features + thresholds)
    * unique tree STRUCTURES (which features split at which node positions,
      thresholds ignored) -- near-duplicate trees that differ only by tiny
      threshold shifts collapse onto one structure
    * share of the buffer taken by the most common structure
    * root-split feature histogram (a single root feature across the whole
      buffer is the classic collapse signature)
  - Are the rewards spread out or flat? (min / mean / max of the reward
    column; with ``buffer.store_log_rewards=True`` -- all current tree
    configs -- these are LOG-rewards, i.e. beta * log-posterior)
  - Tree sizes: mean / std / min / max number of decision nodes. Zero std
    at the maximum size means the sampler only visits max-size trees.
  - Staleness: the iteration at which each entry was inserted. If the median
    insertion iteration is early in training, the search stopped finding
    better trees long ago.

A short WARNING line flags the common collapse signatures. Interpretation
caveat: with capacity ~100 and continuous thresholds, unique *exact* states
are expected to be ~100%; look at structures and root features instead.

Usage (venv active, from the repo root):

    # one run
    python gflownet/envs/tree/helpers_for_experiments/inspect_replay_buffer.py \
        $SCRATCH/gflownet-logs/REG_STAB_DIABETES/<run_dir>

    # every run of a campaign (or several campaigns), one summary table
    python .../inspect_replay_buffer.py $SCRATCH/gflownet-logs/REG_STAB_DIABETES

    # only the summary table
    python .../inspect_replay_buffer.py --summary-only \
        $SCRATCH/gflownet-logs/REG_STAB_DIABETES $SCRATCH/gflownet-logs/REG_STAB_ENERGY
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# "Node 3 (depth=1, status=done): Feature: bmi; Threshold: 0.3702;"
NODE_RE = re.compile(
    r"Node (\d+) \(depth=(\d+), status=done\): "
    r"Feature: ([^;]+); Threshold: ([0-9eE.+-]+)"
)
DONES_RE = re.compile(r"'_dones': \[([01, ]*)\]")


def find_replay_csvs(paths):
    """Expand run dirs / campaign dirs into a list of replay.csv paths."""
    csvs = []
    for p in paths:
        p = Path(p)
        direct = p / "data" / "replay.csv"
        if direct.exists():
            csvs.append(direct)
        else:
            csvs.extend(
                sorted(q for q in p.rglob("data/replay.csv") if "resume" not in q.parts)
            )
    return csvs


def parse_run(csv_path: Path) -> dict:
    """All per-entry quantities of one replay buffer."""
    df = pd.read_csv(csv_path, index_col=0)
    if len(df) == 0:
        return {"name": csv_path.parent.parent.name, "n": 0}

    sizes, structures, roots, exact = [], [], [], []
    for sample, readable in zip(df["samples"], df["samples_readable"]):
        dones = DONES_RE.search(str(sample))
        sizes.append(
            sum(int(x) for x in dones.group(1).split(","))
            if dones and dones.group(1).strip()
            else 0
        )
        nodes = NODE_RE.findall(str(readable))
        # Structure = which feature splits at which node position.
        structures.append(
            tuple(sorted((int(n), feat.strip()) for n, _, feat, _ in nodes))
        )
        root = [feat.strip() for n, _, feat, _ in nodes if int(n) == 0]
        roots.append(root[0] if root else "<single-leaf>")
        exact.append(str(sample))

    rewards = pd.to_numeric(df["rewards"], errors="coerce").to_numpy(dtype=float)
    iters = pd.to_numeric(df["iter"], errors="coerce").to_numpy(dtype=float)
    struct_counts = Counter(structures)
    top_struct, top_count = struct_counts.most_common(1)[0]

    return {
        "name": csv_path.parent.parent.name,
        "n": len(df),
        "uniq_exact": len(set(exact)),
        "uniq_struct": len(struct_counts),
        "top_struct_frac": top_count / len(df),
        "top_struct": top_struct,
        "struct_counts": struct_counts,
        "root_counts": Counter(roots),
        "sizes": np.asarray(sizes, dtype=float),
        "rewards": rewards,
        "iters": iters,
    }


def warnings_for(run: dict) -> list:
    w = []
    if run["top_struct_frac"] > 0.5:
        w.append(f"one structure fills {run['top_struct_frac']:.0%} of the buffer")
    if len(run["root_counts"]) == 1:
        w.append(f"single root feature ({next(iter(run['root_counts']))})")
    if run["sizes"].std() < 0.5 and len(run["sizes"]) > 1:
        w.append(f"no size diversity (all trees ~{run['sizes'].mean():.0f} nodes)")
    finite = run["rewards"][np.isfinite(run["rewards"])]
    if len(finite) > 1 and (finite.max() - finite.min()) < 1e-6:
        w.append("flat rewards (max == min)")
    return w


def print_run(run: dict, top: int):
    print(f"\n=== {run['name']}")
    if run["n"] == 0:
        print("  EMPTY replay buffer.")
        return
    rewards, sizes, iters = run["rewards"], run["sizes"], run["iters"]
    finite = rewards[np.isfinite(rewards)]
    print(
        f"  entries            : {run['n']}   "
        f"unique exact states: {run['uniq_exact']}   "
        f"unique structures: {run['uniq_struct']}"
    )
    if len(finite):
        print(
            f"  reward column      : max {finite.max():.2f}   "
            f"mean {finite.mean():.2f}   min {finite.min():.2f}   "
            f"(log-rewards if store_log_rewards=True)"
        )
    print(
        f"  decision nodes     : mean {sizes.mean():.2f} ± {sizes.std():.2f}   "
        f"range [{sizes.min():.0f}, {sizes.max():.0f}]"
    )
    if np.isfinite(iters).any():
        print(
            f"  inserted at iter   : median {np.nanmedian(iters):.0f}   "
            f"range [{np.nanmin(iters):.0f}, {np.nanmax(iters):.0f}]"
        )
    roots = ", ".join(f"{f}:{c}" for f, c in run["root_counts"].most_common())
    print(f"  root features      : {roots}")
    print(f"  top {top} structures (count x [node:feature ...]):")
    for struct, count in run["struct_counts"].most_common(top):
        desc = " ".join(f"{n}:{f}" for n, f in struct) or "<single leaf>"
        print(f"    {count:3d} x {desc}")
    for w in warnings_for(run):
        print(f"  WARNING: {w}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Run directories (containing data/replay.csv) and/or campaign "
        "directories (scanned recursively).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=3,
        help="How many most-common structures to print per run (default 3).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip the per-run details, print only the summary table.",
    )
    args = parser.parse_args()

    csvs = find_replay_csvs(args.paths)
    if not csvs:
        raise SystemExit("No data/replay.csv found under the given paths.")

    runs = [parse_run(c) for c in csvs]
    if not args.summary_only:
        for run in runs:
            print_run(run, args.top)

    rows = []
    for run in runs:
        if run["n"] == 0:
            rows.append({"run": run["name"], "n": 0})
            continue
        finite = run["rewards"][np.isfinite(run["rewards"])]
        rows.append(
            {
                "run": run["name"][-60:],
                "n": run["n"],
                "uniq_struct": run["uniq_struct"],
                "top_struct%": f"{run['top_struct_frac']:.0%}",
                "n_root_feats": len(run["root_counts"]),
                "size_mean": f"{run['sizes'].mean():.1f}",
                "size_std": f"{run['sizes'].std():.1f}",
                "logR_max": f"{finite.max():.1f}" if len(finite) else "-",
                "logR_spread": (
                    f"{finite.max() - finite.min():.1f}" if len(finite) else "-"
                ),
                "iter_med": (
                    f"{np.nanmedian(run['iters']):.0f}"
                    if np.isfinite(run["iters"]).any()
                    else "-"
                ),
                "flags": "; ".join(warnings_for(run)) or "-",
            }
        )
    print("\n--- Replay buffer summary (one row per run) ---")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
