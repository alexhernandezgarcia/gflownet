"""
Post-training evaluation for DT-GFN runs on the RegressionTree environment.

The regression counterpart of ``gflownet/envs/tree/eval_tree.py``, which only
covers classification (Dirichlet leaf posteriors, accuracy / BMA). Rather than
reimplementing the metrics, this script rebuilds the environment from the run's
own stored config and calls :py:meth:`RegressionTree.test` -- the exact method
the TreeEvaluator calls during training -- so the numbers in the JSON are
directly comparable to the ones logged to wandb.

Reported (see RegressionTree.test): ``train_``/``test_`` prefixed
``mean_rmse``, ``mean_r2``, ``forest_rmse``, ``forest_r2``, plus ``top_k_*``
and ``top_1_*`` when ``--top_k_trees > 0``, and ``mean_nodes``. RMSE values are
in the original target units even when ``env.scale_y`` standardized them.

Usage:
    python eval_regression_tree.py \\
        --run_dir $SCRATCH/gflownet-logs/REGTREE/<run_name> \\
        --output  $SCRATCH/gflownet-logs/REGTREE/<run_name>/eval_results.json

The NIG hyper-parameters and the dataset are read from
``<run_dir>/.hydra/config.yaml``, so the evaluation cannot silently disagree
with the configuration the run was trained under. ``--seed`` fixes the
posterior draws so that re-running the script reproduces its own numbers.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]  # <repo>/gflownet/envs/tree/helpers_for_experiments
sys.path.insert(0, str(REPO_ROOT))

from hydra.utils import instantiate
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Run directory containing .hydra/config.yaml and samples/.",
    )
    parser.add_argument(
        "--samples_path",
        type=Path,
        default=None,
        help="Path to gfn_samples.pkl (default: <run_dir>/samples/gfn_samples.pkl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the metrics JSON (default: <run_dir>/eval_results.json).",
    )
    parser.add_argument(
        "--top_k_trees",
        type=int,
        default=10,
        help="Report top-k / top-1 metrics over this many best trees (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the NIG posterior draws, for reproducibility (default: 0).",
    )
    args = parser.parse_args()

    samples_path = args.samples_path or args.run_dir / "samples" / "gfn_samples.pkl"
    output_path = args.output or args.run_dir / "eval_results.json"
    config_path = args.run_dir / ".hydra" / "config.yaml"

    for path in (samples_path, config_path):
        if not path.exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            sys.exit(1)

    config = OmegaConf.load(config_path)
    print(f"Loading samples from {samples_path}")
    with open(samples_path, "rb") as f:
        states = pickle.load(f)["x"]
    print(f"  Loaded {len(states)} trees")

    # Rebuilt exactly as gflownet_from_config does, so the env sees the same
    # dataset, the same split and the same target standardization as training.
    env = instantiate(
        config.env,
        device=config.device,
        float_precision=config.float_precision,
    )
    print(f"  Dataset: {config.env.data_path}")

    # NIG hyper-parameters come from the run's proxy config; nulls are resolved
    # from the training targets inside test() (see _resolve_nig_params).
    result = env.test(
        states,
        top_k_trees=args.top_k_trees,
        plot_top_k=False,
        seed=args.seed,
        mu_0=config.proxy.get("mu_0", None),
        kappa_0=config.proxy.get("kappa_0", 0.1),
        alpha_0=config.proxy.get("alpha_0", 2.0),
        beta_0=config.proxy.get("beta_0", None),
    )
    metrics = result["metrics"]
    if not metrics:
        print("ERROR: the environment returned no metrics.", file=sys.stderr)
        sys.exit(1)

    metrics["n_trees"] = len(states)

    print("\n=== Regression results ===")
    for key in sorted(metrics):
        value = metrics[key]
        print(f"  {key:<28} {value:.4f}" if isinstance(value, float) else
              f"  {key:<28} {value}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()