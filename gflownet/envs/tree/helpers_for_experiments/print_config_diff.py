"""
Print the hyperparameters that differ between two training configurations.

Takes two group hashes as printed in the ``config`` column of the
aggregate_treeclass_results.py tables and prints a small table with one row
per config key whose value differs between the two configurations, and the
value of each. Keys present in only one config show ``<absent>`` for the
other. Hash prefixes are accepted (like ``hash2config`` in the notebook).

The configs compared are exactly what was hashed into the group ids: the
runs' resolved hydra configs WITHOUT the run-identity keys (dataset split
path, logger section, user section, ...). Consequently two group hashes can
also differ only because one config predates a key that was later added to
the config schema -- those rows show ``<absent>``.

The runs behind the hashes are located the same way the aggregator finds
them: first on disk (every ``.hydra/config.yaml`` next to an
``eval_results.json`` under the runs root), and -- only when a hash was not
found there -- in the dt-gfn wandb projects.

Usage examples:

    # both hashes from evaluated runs on disk (fast, no wandb)
    python gflownet/envs/tree/helpers_for_experiments/print_config_diff.py \
        11fd0d0d 3a0be2c1

    # restrict the disk scan to one campaign folder
    python .../print_config_diff.py 11fd0d0d 3a0be2c1 \
        $SCRATCH/gflownet-logs/TREECLASS_MAGIC

    # skip the disk scan entirely and look the hashes up on wandb
    python .../print_config_diff.py 11fd0d0d 3a0be2c1 --source wandb
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from gflownet.envs.tree.helpers_for_experiments import (
    aggregate_treeclass_results as agg,
)


def flatten_config(d: dict, prefix: str = "") -> dict:
    """Nested config dict -> {dotted key: leaf value} (lists stay one value)."""
    flat = {}
    for key, value in d.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_config(value, dotted + "."))
        else:
            flat[dotted] = json.dumps(value) if isinstance(value, list) else value
    return flat


def find_record(records: list, hash_prefix: str):
    """The first record whose 8-char group hash starts with ``hash_prefix``.

    All records with the same group hash have the same (hashed) config by
    construction, so any match is as good as any other for diffing.
    """
    for rec in records:
        if rec["hash"].startswith(hash_prefix):
            return rec
    return None


def describe(rec: dict) -> str:
    return (
        f"config {rec['hash']} | {rec['dataset']} ({rec['task']}) | "
        f"campaign {rec['campaign']} | e.g. run {rec['run_name']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("hash_a", help="First group hash (prefix accepted).")
    parser.add_argument("hash_b", help="Second group hash (prefix accepted).")
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=agg.default_root(),
        help="Runs root scanned for evaluated runs: the whole gflownet-logs "
        "tree or a single campaign folder (default: $SCRATCH/gflownet-logs).",
    )
    parser.add_argument(
        "--source",
        choices=["eval", "wandb", "both"],
        default="both",
        help="Where to look the hashes up (default: both -- disk first, "
        "wandb only for hashes not found on disk).",
    )
    parser.add_argument(
        "--entity",
        default=agg.WANDB_ENTITY,
        help=f"wandb entity (default: {agg.WANDB_ENTITY}).",
    )
    parser.add_argument(
        "--projects",
        default=",".join(agg.WANDB_PROJECTS),
        help="Comma-separated wandb projects (default: %(default)s).",
    )
    parser.add_argument(
        "--group-ignore",
        default=None,
        help="Comma-separated extra dotted config keys excluded from the "
        "grouping hash -- must MATCH what was passed to the aggregator run "
        "that printed the hashes, otherwise the hashes cannot be found.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=60,
        help="Truncate displayed values to this many characters "
        "(default: %(default)s; pass 0 for no truncation).",
    )
    args = parser.parse_args()

    if args.hash_b.startswith(args.hash_a) or args.hash_a.startswith(args.hash_b):
        parser.error(
            f"{args.hash_a!r} and {args.hash_b!r} match the same group; "
            f"pass two distinct hashes."
        )

    extra_excluded = args.group_ignore.split(",") if args.group_ignore else ()

    rec_a = rec_b = None
    if args.source in ("eval", "both"):
        records = agg.collect_eval_runs(args.root, extra_excluded)
        rec_a = find_record(records, args.hash_a)
        rec_b = find_record(records, args.hash_b)
    if (rec_a is None or rec_b is None) and args.source in ("wandb", "both"):
        records = agg.collect_wandb_runs(
            args.entity,
            [p for p in args.projects.split(",") if p],
            extra_excluded=extra_excluded,
        )
        rec_a = rec_a or find_record(records, args.hash_a)
        rec_b = rec_b or find_record(records, args.hash_b)

    missing = [h for h, r in [(args.hash_a, rec_a), (args.hash_b, rec_b)] if r is None]
    if missing:
        raise SystemExit(
            f"[ERROR] No run found for hash(es): {', '.join(missing)}. "
            f"Checked source(s): {args.source} (root: {args.root}). If the "
            f"aggregator was run with --group-ignore, pass the same value here."
        )

    print(f"A: {describe(rec_a)}")
    print(f"B: {describe(rec_b)}")

    flat_a = flatten_config(rec_a["config"])
    flat_b = flatten_config(rec_b["config"])
    diff_rows = []
    for key in sorted(set(flat_a) | set(flat_b)):
        val_a = flat_a.get(key, "<absent>")
        val_b = flat_b.get(key, "<absent>")
        if json.dumps(val_a, default=str) != json.dumps(val_b, default=str):
            diff_rows.append(
                {"config key": key, rec_a["hash"]: val_a, rec_b["hash"]: val_b}
            )

    if not diff_rows:
        # Same group hash was already rejected above, so this means a genuine
        # 8-char hash collision between two distinct raw configs.
        print(
            "\nNo differing config keys found -- but the hashes differ: "
            "this is a hash collision or a bug; inspect the raw configs."
        )
        return

    df = pd.DataFrame(diff_rows)
    if args.max_width > 0:
        for col in (rec_a["hash"], rec_b["hash"]):
            df[col] = df[col].map(
                lambda v: (
                    s
                    if len(s := str(v)) <= args.max_width
                    else s[: args.max_width - 3] + "..."
                )
            )
    print(f"\nConfig keys that differ ({len(df)}):")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
