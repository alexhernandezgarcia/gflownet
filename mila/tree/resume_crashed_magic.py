#!/usr/bin/env python3
"""Relaunch every crashed magic run so training continues from its checkpoint.

Scans the campaign directories under $SCRATCH/gflownet-logs that contain magic
runs, classifies each magic run directory from what is on disk, and submits one
Slurm job (mila/tree/resume_crashed_magic_worker.sh) per run whose TRAINING is
unfinished. Resources match how the run was launched originally:

  device: cpu   in .hydra/config.yaml -> long-cpu partition, no GPU
  device: cuda  in .hydra/config.yaml -> long partition, one l40s (48 GB);
                                         depth >= 6 gets an a100l (80 GB)
                                         because activation memory ~4x per
                                         extra depth level (a 32 GB card OOMs
                                         already at depth 5).

Classification (training is what counts; missing evaluation does NOT queue a
run, per how these campaigns are aggregated):

  ckpts/final.ckpt or samples/gfn_samples.pkl exists -> DONE, skipped
  ckpts/*.ckpt exists                                -> RESUME, submitted
  anything else (no checkpoint to continue from)     -> BROKEN, reported only

The requested walltime is scaled to the work left: remaining steps read from
the checkpoint file names times --sec-per-step, plus a margin, capped at
--max-hours. A run that still hits the cap (the 50k-step ones might) is safe
to finish by simply running this script again: DONE runs are skipped and
everything else resumes from its newest checkpoint.

Usage (from anywhere, on a login node):

  # 1. See the plan, submit nothing:
  python mila/tree/resume_crashed_magic.py --dry-run

  # 2. Debug with a single real submission:
  python mila/tree/resume_crashed_magic.py --limit 1

  # 3. The real thing:
  python mila/tree/resume_crashed_magic.py
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

EXP_DIRS = ["TRFM_MAGIC", "TREECLASS_MAGIC", "BWD_POLICY_ABLATION", "DEBUG_UNIFORM"]

CPU_PARTITION = "long-cpu,long-cpu-eek"
CPU_MEM = "24G"
GPU_PARTITION = "long"
GPU_MEM = "32G"
GPU_TYPE = "l40s"  # 48 GB: enough for the depth-5 transformer runs
GPU_TYPE_DEEP = "a100l"  # 80 GB: depth >= 6 needs ~4x the activation memory


def cfg_value(config: str, key: str) -> str | None:
    """First value of `key:` in the yaml text, at any indentation."""
    m = re.search(rf"^\s*{key}:\s*(\S+)", config, re.MULTILINE)
    return m.group(1) if m else None


def last_ckpt_iter(ckpt_dir: Path) -> int | None:
    """Highest iteration among iter_*.ckpt files; None if there is none."""
    iters = [
        int(m.group(1))
        for f in ckpt_dir.glob("iter_*.ckpt")
        if (m := re.fullmatch(r"iter_(\d+)\.ckpt", f.name))
    ]
    return max(iters, default=None)


def classify(run_dir: Path) -> dict:
    info = {"dir": run_dir, "name": run_dir.name}
    cfg_path = run_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        info.update(action="BROKEN", why="no .hydra/config.yaml")
        return info
    config = cfg_path.read_text()
    info["device"] = cfg_value(config, "device") or "cpu"
    info["depth"] = int(cfg_value(config, "max_depth") or 0)
    info["n_steps"] = int(cfg_value(config, "n_train_steps") or 0)

    if (run_dir / "ckpts" / "final.ckpt").is_file() or (
        run_dir / "samples" / "gfn_samples.pkl"
    ).is_file():
        info.update(action="DONE", why="training finished")
        return info

    last = last_ckpt_iter(run_dir / "ckpts")
    if last is None:
        info.update(action="BROKEN", why="no checkpoint to resume from")
        return info
    info.update(action="RESUME", last=last, remaining=max(info["n_steps"] - last, 0))
    return info


def walltime(remaining_steps: int, args) -> str:
    hours = (remaining_steps * args.sec_per_step + args.margin_min * 60) / 3600
    hours = min(max(int(hours) + 1, 2), args.max_hours)
    return f"{hours}:00:00"


def sbatch_command(info: dict, args) -> list[str]:
    cmd = [
        "sbatch",
        f"--time={walltime(info['remaining'], args)}",
        f"--cpus-per-task={args.cpus}",
        f"--export=ALL,RUN_DIR={info['dir']}",
    ]
    if info["device"] == "cuda":
        gpu = GPU_TYPE_DEEP if info["depth"] >= 6 else GPU_TYPE
        cmd += [
            f"--partition={GPU_PARTITION}",
            f"--gres=gpu:{gpu}:1",
            f"--mem={GPU_MEM}",
        ]
    else:
        cmd += [f"--partition={CPU_PARTITION}", f"--mem={CPU_MEM}"]
    cmd.append(str(Path(__file__).with_name("resume_crashed_magic_worker.sh")))
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and the sbatch commands, submit nothing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="submit at most N runs (debugging)",
    )
    parser.add_argument(
        "--runs-root", type=Path, default=Path(os.environ["SCRATCH"]) / "gflownet-logs"
    )
    parser.add_argument(
        "--exp-dirs",
        nargs="+",
        default=EXP_DIRS,
        help=f"campaign directories to scan (default: {EXP_DIRS})",
    )
    parser.add_argument(
        "--match",
        default="magic",
        help="only run dirs whose name contains this (default: magic)",
    )
    parser.add_argument(
        "--sec-per-step",
        type=float,
        default=8.0,
        help="walltime budget per remaining training step (default: 8)",
    )
    parser.add_argument(
        "--margin-min",
        type=int,
        default=90,
        help="extra walltime margin in minutes (default: 90)",
    )
    parser.add_argument(
        "--max-hours", type=int, default=48, help="walltime cap in hours (default: 48)"
    )
    parser.add_argument("--cpus", type=int, default=4)
    args = parser.parse_args()

    runs = []
    for exp in args.exp_dirs:
        exp_dir = args.runs_root / exp
        if not exp_dir.is_dir():
            print(f"WARNING: campaign directory not found, skipping: {exp_dir}")
            continue
        for run_dir in sorted(exp_dir.iterdir()):
            if run_dir.is_dir() and args.match in run_dir.name:
                runs.append(classify(run_dir))

    done = [r for r in runs if r["action"] == "DONE"]
    broken = [r for r in runs if r["action"] == "BROKEN"]
    queue = [r for r in runs if r["action"] == "RESUME"]
    if args.limit is not None:
        queue = queue[: args.limit]

    print(
        f"Scanned {len(runs)} '{args.match}' runs under {args.runs_root}: "
        f"{len(done)} training-done (skipped), {len(broken)} broken, "
        f"{len(queue)} to resume.\n"
    )
    for r in broken:
        print(f"  BROKEN  ({r['why']})  {r['name']}")
    if broken:
        print()
    for r in queue:
        gpu = (
            "gpu:" + (GPU_TYPE_DEEP if r["depth"] >= 6 else GPU_TYPE)
            if r["device"] == "cuda"
            else "cpu"
        )
        print(
            f"  RESUME  {r['device']:4s} {gpu:9s} "
            f"step {r['last']:>6d}/{r['n_steps']:<6d} "
            f"time {walltime(r['remaining'], args):>9s}  {r['name']}"
        )
    if not queue:
        print("Nothing to submit.")
        return

    if args.dry_run:
        print("\n--dry-run: sbatch commands that WOULD run:\n")
        for r in queue:
            print("  " + " ".join(sbatch_command(r, args)))
        return

    (args.runs_root / "slurm").mkdir(exist_ok=True)  # worker's --output dir
    print()
    failures = 0
    for r in queue:
        cmd = sbatch_command(r, args)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  {result.stdout.strip()}  <- {r['name']}")
        else:
            failures += 1
            print(f"  SUBMIT FAILED for {r['name']}:\n    {result.stderr.strip()}")
    print(f"\nSubmitted {len(queue) - failures}/{len(queue)} jobs.")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
