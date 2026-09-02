"""
Post-training evaluation for DT-GFN runs on the RegressionTree environment.

The regression counterpart of ``gflownet/envs/tree/eval_tree.py``, which only
covers classification (Dirichlet leaf posteriors, accuracy / BMA). Rather than
reimplementing the metrics, this script rebuilds the environment from the run's
own stored config and calls :py:meth:`RegressionTree.test` -- the exact method
the TreeEvaluator calls during training -- so the numbers in the JSON are
directly comparable to the ones logged to wandb.

Reported (see RegressionTree.test): ``train_``/``test_`` prefixed
``mean_tree_rmse``, ``mean_tree_r2``, ``mean_tree_nll``, ``forest_rmse``,
``forest_r2``, ``forest_nll``, ``forest_coverage_90``, plus ``top_k_*`` and
``top_1_*`` when ``--top_k_trees > 0``, and ``mean_n_nodes`` /
``mean_log_posterior``. The leaf parameters are integrated out in closed form
(posterior predictive mean / Student-t), so the evaluation is deterministic.
Trees are ranked for the top-k metrics by their log-posterior, computed with
the run's own proxy (i.e. honoring its structure prior). RMSE and NLL values
are in the original target units even when ``env.scale_y`` standardized them.

Tree sizes: ``mean_n_nodes`` (from RegressionTree.test, also logged to wandb)
counts DECISION nodes only, averaged over all sampled trees. This script
additionally reports -- in the JSON only, nothing is logged to wandb -- the
total node count (decision nodes + leaves, the same convention as
``model_size_*`` in eval_tree.py for classification):
``model_size_top1`` (the highest-log-posterior tree, i.e. the tree behind the
``*_top_1_*`` metrics), ``model_size_mean`` / ``model_size_std`` over all
sampled trees, and ``top_1_n_decision_nodes`` for the top-1 tree.

Usage:
    python eval_regression_tree.py \\
        --run_dir $SCRATCH/gflownet-logs/REGTREE/<run_name> \\
        --output  $SCRATCH/gflownet-logs/REGTREE/<run_name>/eval_results.json

The NIG hyper-parameters and the dataset are read from
``<run_dir>/.hydra/config.yaml``, so the evaluation cannot silently disagree
with the configuration the run was trained under.
"""

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]  # <repo>/gflownet/envs/tree/helpers_for_experiments
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gflownet.envs.tree.eval_tree import (
    count_internal_nodes,
    count_total_nodes,
    load_samples,
)


def tree_size_metrics(states, log_posteriors, top_k_trees):
    """
    Node counts of the sampled trees, in the JSON only (not logged to wandb).

    ``model_size_*`` count decision nodes + leaves, matching the classification
    ``model_size_*`` metrics of eval_tree.py; the top-1 tree is selected
    exactly like in RegressionTree.test (first index of ``argsort(-log_post)``)
    so it is the same tree the ``*_top_1_*`` metrics refer to. The top-1
    entries are only defined when top-k metrics were requested.
    """
    total = np.array([count_total_nodes(s) for s in states], dtype=float)
    metrics = {
        "model_size_mean": float(total.mean()),
        "model_size_std": float(total.std()),
    }
    if top_k_trees > 0 and len(states) > 0:
        top_1_idx = int(np.argsort(-np.asarray(log_posteriors, dtype=float))[0])
        metrics["model_size_top1"] = float(total[top_1_idx])
        metrics["top_1_n_decision_nodes"] = float(
            count_internal_nodes(states[top_1_idx])
        )
    return metrics


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
        "--data_path",
        type=Path,
        default=None,
        help="Override env.data_path from the stored config. Needed for run "
        "directories copied from another cluster, where the absolute dataset "
        "path baked into .hydra/config.yaml does not exist locally.",
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
        help="Deprecated and ignored: the evaluation is deterministic (leaf "
        "parameters are integrated out in closed form). Kept so existing "
        "launcher scripts do not break.",
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
    if args.data_path is not None:
        config.env.data_path = str(args.data_path)
    print(f"Loading samples from {samples_path}")
    states = load_samples(samples_path)["x"]
    print(f"  Loaded {len(states)} trees")

    # Rebuilt exactly as gflownet_from_config does, so the env sees the same
    # dataset, the same split and the same target standardization as training.
    env = instantiate(
        config.env,
        device=config.device,
        float_precision=config.float_precision,
    )
    print(f"  Dataset: {config.env.data_path}")

    # The run's own proxy supplies the per-tree log-posteriors used to rank
    # trees for the top-k metrics, so the ranking honors the exact structure
    # prior (prior_type, beta, ...) the run was trained with.
    proxy = instantiate(
        config.proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    proxy.setup(env)
    log_posteriors = proxy(states).cpu().numpy().astype(np.float64)
    print(
        f"  Log-posterior range: "
        f"[{log_posteriors.min():.2f}, {log_posteriors.max():.2f}]"
    )

    # NIG hyper-parameters come from the run's proxy config; nulls are resolved
    # from the training targets inside test() (see _resolve_nig_params).
    result = env.test(
        states,
        top_k_trees=args.top_k_trees,
        plot_top_k=False,
        mu_0=config.proxy.get("mu_0", None),
        kappa_0=config.proxy.get("kappa_0", 0.1),
        alpha_0=config.proxy.get("alpha_0", 2.0),
        beta_0=config.proxy.get("beta_0", None),
        log_posteriors=log_posteriors,
    )
    metrics = result["metrics"]
    if not metrics:
        print("ERROR: the environment returned no metrics.", file=sys.stderr)
        sys.exit(1)

    metrics["n_trees"] = len(states)
    metrics.update(tree_size_metrics(states, log_posteriors, args.top_k_trees))

    print("\n=== Regression results ===")
    for key in sorted(metrics):
        value = metrics[key]
        print(
            f"  {key:<28} {value:.4f}"
            if isinstance(value, float)
            else f"  {key:<28} {value}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
