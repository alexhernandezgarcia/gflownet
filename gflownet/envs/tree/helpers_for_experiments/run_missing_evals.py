"""
Run the final evaluation for every finished training run that is missing it.

Walks a runs root (default: $SCRATCH/gflownet-logs), finds every run
directory -- identified by its ``.hydra/config.yaml`` -- whose training has
produced final samples (``samples/gfn_samples.pkl``, or the newest
``resume/**/gfn_samples.pkl`` for runs finished by a resume job) but that has
no ``eval_results.json`` yet, and runs the matching evaluation script on it:

  - classification runs (env ``gflownet.envs.tree.tree.Tree``):
        gflownet/envs/tree/eval_tree.py
  - regression runs (env ``gflownet.envs.tree.regression_tree.RegressionTree``):
        gflownet/envs/tree/helpers_for_experiments/eval_regression_tree.py

The task type, the dataset path and the Dirichlet alpha are read from the
run's own resolved ``.hydra/config.yaml`` -- never from the run name, which
can be stale or follow an older naming convention.

Dataset paths baked on another cluster (e.g. ``/home/arnit/gflownet/tests/...``
from Trillium) are relocated under this repo's ``tests/`` directory, so run
directories rsync'ed from other clusters can be evaluated here.

Runs are evaluated one after another (no parallelism); a failure is recorded
and the walk continues. A summary of evaluated / skipped / failed runs is
printed at the end.

Usage (from the repo root, with the gflownet venv active):
    python gflownet/envs/tree/helpers_for_experiments/run_missing_evals.py
    python gflownet/envs/tree/helpers_for_experiments/run_missing_evals.py \
        $SCRATCH/gflownet-logs/TREECLASS_MAGIC --dry-run
    python gflownet/envs/tree/helpers_for_experiments/run_missing_evals.py \
        --dataset iris,wine --force
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]  # <repo>/gflownet/envs/tree/helpers_for_experiments
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from config_hash import dataset_and_split
from omegaconf import OmegaConf

EVAL_TREE = REPO_ROOT / "gflownet" / "envs" / "tree" / "eval_tree.py"
EVAL_REGRESSION_TREE = SCRIPT_DIR / "eval_regression_tree.py"

# env._target_ suffix -> task. Anything else (origtree, node, ...) is skipped.
TASK_BY_ENV_TARGET = {
    "gflownet.envs.tree.tree.Tree": "classification",
    "gflownet.envs.tree.regression_tree.RegressionTree": "regression",
}


def default_root() -> Path:
    scratch = os.environ.get("SCRATCH", str(Path.home() / "scratch"))
    return Path(scratch) / "gflownet-logs"


def find_run_dirs(root: Path):
    """Yield every run directory under root (a dir owning a .hydra/config.yaml).

    Resume jobs write their own nested ``resume/<...>/.hydra`` and wandb keeps
    internal folders; both are excluded so each run is visited exactly once,
    at its top level.
    """
    for cfg_path in sorted(root.rglob(".hydra/config.yaml")):
        run_dir = cfg_path.parent.parent
        rel_parts = run_dir.relative_to(root).parts
        if "resume" in rel_parts or "wandb" in rel_parts:
            continue
        yield run_dir


def find_final_samples(run_dir: Path):
    """The final gfn_samples.pkl of a run, or None if training never finished.

    Single-shot runs write samples/gfn_samples.pkl; runs finished by a resume
    job may have left it in the resume job's hydra dir instead (the launcher
    normally moves it, but be tolerant): take the newest one.
    """
    direct = run_dir / "samples" / "gfn_samples.pkl"
    if direct.exists():
        return direct
    candidates = list(run_dir.glob("resume/*/gfn_samples.pkl")) + list(
        run_dir.glob("resume/*/*/gfn_samples.pkl")
    )
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    return None


def relocate_data_path(data_path: str) -> Path:
    """Map a dataset path baked on another machine into this checkout.

    The datasets live under <repo>/tests/data/tree/... on every cluster; if the
    absolute path from the config does not exist here, re-root it at this
    repo's tests/ directory.
    """
    p = Path(data_path)
    if p.exists():
        return p
    parts = p.parts
    if "tests" in parts:
        candidate = REPO_ROOT / Path(*parts[parts.index("tests") :])
        if candidate.exists():
            return candidate
    return p  # caller reports the missing file


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
        help="Runs root to walk recursively: the whole gflownet-logs tree or a "
        "single campaign folder (default: $SCRATCH/gflownet-logs).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Comma-separated dataset names; only evaluate runs trained on "
        "these datasets (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate runs even if eval_results.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be evaluated, without running anything.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="top_k_trees for the regression evaluation (default: 10).",
    )
    args = parser.parse_args()

    datasets = set(args.dataset.split(",")) if args.dataset else None

    done, todo, failed, skipped = [], [], [], []
    for run_dir in find_run_dirs(args.root):
        name = str(run_dir.relative_to(args.root))
        eval_json = run_dir / "eval_results.json"
        if eval_json.exists() and not args.force:
            done.append(name)
            continue

        try:
            cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
            env_target = str(cfg.env.get("_target_", ""))
            task = TASK_BY_ENV_TARGET.get(env_target)
            data_path = str(cfg.env.data_path)
        except Exception as e:
            skipped.append((name, f"unreadable config ({e})"))
            continue
        if task is None:
            skipped.append((name, f"unsupported env {env_target}"))
            continue

        dataset, _ = dataset_and_split(OmegaConf.to_container(cfg, resolve=True))
        if datasets is not None and dataset not in datasets:
            continue

        samples_pkl = find_final_samples(run_dir)
        if samples_pkl is None:
            skipped.append((name, "no gfn_samples.pkl (training unfinished?)"))
            continue

        csv_path = relocate_data_path(data_path)
        if not csv_path.exists():
            skipped.append((name, f"dataset not found: {data_path}"))
            continue

        if task == "classification":
            cmd = [
                sys.executable,
                str(EVAL_TREE),
                "--samples_path",
                str(samples_pkl),
                "--data_path",
                str(csv_path),
                "--alpha_value",
                str(cfg.proxy.alpha_value),
                "--output",
                str(eval_json),
            ]
        else:
            cmd = [
                sys.executable,
                str(EVAL_REGRESSION_TREE),
                "--run_dir",
                str(run_dir),
                "--samples_path",
                str(samples_pkl),
                "--data_path",
                str(csv_path),
                "--top_k_trees",
                str(args.top_k),
                "--seed",
                "0",
                "--output",
                str(eval_json),
            ]
        todo.append((name, task, cmd))

    print(f"Runs root: {args.root}")
    print(
        f"Already evaluated: {len(done)}   to evaluate: {len(todo)}   "
        f"not evaluable: {len(skipped)}\n"
    )
    for name, reason in skipped:
        print(f"[SKIP] {name}: {reason}")
    if skipped:
        print()

    for i, (name, task, cmd) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] ({task}) {name}")
        if args.dry_run:
            print("       " + " ".join(cmd))
            continue
        t0 = time.time()
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        dt = time.time() - t0
        if result.returncode == 0:
            print(f"       OK in {dt:.0f}s")
        else:
            print(f"       FAILED (exit {result.returncode}) after {dt:.0f}s")
            failed.append(name)

    print("\n" + "=" * 60)
    if args.dry_run:
        print(f"Dry run: {len(todo)} run(s) would be evaluated.")
    else:
        print(
            f"Evaluated {len(todo) - len(failed)}/{len(todo)} run(s); "
            f"{len(failed)} failed, {len(skipped)} not evaluable."
        )
        for name in failed:
            print(f"[FAILED] {name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
