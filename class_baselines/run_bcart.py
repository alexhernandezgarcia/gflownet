"""
Bayesian classification tree (BCART, Chipman et al. 1998) benchmarks on
quantile-binarized features, sampled with the tree_smc code of
Lakshminarayanan et al. (2013) as adapted in the MAPTree repository
(vendored in class_baselines/tree_smc).

Methods:
  - bcart_mcmc: posterior predictive averaged over trees sampled by
    Chipman-style Metropolis-Hastings (grow/prune/change/swap moves), after
    burn-in and with thinning.
  - bcart_map: the single highest-posterior tree visited by the same MCMC
    chain -- the classic Bayesian CART "model search" point estimate
    (written together with bcart_mcmc at no extra sampling cost).
  - bcart_smc: posterior predictive averaged over the particles of the
    top-down SMC sampler of Lakshminarayanan et al. (2013), weighted by the
    final particle weights.

Priors match the MAPTree paper defaults: P(split at depth d) =
alpha_split * (1+d)^(-beta_split) with a uniform choice of binary feature,
and a Beta(rho, rho) prior on the leaf label probabilities (all predictions
are smoothed leaf posterior means).

Usage (from the repo root, venv active):
    python class_baselines/run_bcart.py [--methods mcmc smc] [--splits 1 2 ...]
                                          [--iterations 50000] [--particles 1000]
"""

import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # tree_smc imports pyplot; avoid needing a display

import numpy as np  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from binary_trees import (binarize_quantiles, bma_proba,  # noqa: E402
                          parse_node_info)
from common import (classification_metrics, load_split,  # noqa: E402
                    make_parser, save_result)
from tree_smc.bdtmcmc import (parser_add_mcmc_options,  # noqa: E402
                              precompute, sample_tree)
from tree_smc.bdtsmc import init_smc, run_smc  # noqa: E402
from tree_smc.tree_utils import (parser_add_common_options,  # noqa: E402
                                 parser_add_smc_options)


def make_settings(args, n_particles=None):
    parser = parser_add_common_options()
    parser = parser_add_smc_options(parser)
    parser = parser_add_mcmc_options(parser)
    argv = [
        "--alpha_split",
        str(args.alpha_split),
        "--beta_split",
        str(args.beta_split),
        # tree_smc's "alpha" is the total Dirichlet concentration over labels.
        "--alpha",
        str(2 * args.rho),
        "--verbose",
        "0",
    ]
    if n_particles is not None:
        argv += ["--n_particles", str(n_particles), "--n_islands", "1"]
    return parser.parse_args(argv)[0]


def make_data(B_train, y_train):
    return {
        "x_train": B_train.astype(int),
        "y_train": y_train.astype(int),
        "n_train": B_train.shape[0],
        "n_dim": B_train.shape[1],
        "n_class": 2,
    }


def run_mcmc(B_train, y_train, args, seed):
    """
    Runs one MH chain; returns (posterior tree samples, best tree, params).
    Trees are (node_info, leaf_nodes) snapshots; burn-in is discarded and the
    rest thinned to at most --max-samples samples.
    """
    settings = make_settings(args)
    data = make_data(B_train, y_train)
    np.random.seed(seed)
    random.seed(seed)

    param, cache, cache_tmp = precompute(data, settings)
    p = sample_tree(data, settings, param, cache, cache_tmp)

    burn_in = int(args.iterations * args.burn_in_frac)
    thin = max(1, (args.iterations - burn_in) // args.max_samples)
    samples, best_snapshot, best_post = [], None, -np.inf
    for iteration in range(args.iterations):
        p.sample(data, settings, param, cache)
        post = p.compute_logprob()
        if post > best_post:
            best_snapshot, best_post = (dict(p.node_info), list(p.leaf_nodes)), post
        if iteration >= burn_in and (iteration - burn_in) % thin == 0:
            samples.append((dict(p.node_info), list(p.leaf_nodes)))
    params = {
        "iterations": args.iterations,
        "burn_in": burn_in,
        "thin": thin,
        "n_samples": len(samples),
        "best_log_posterior": float(best_post),
    }
    return samples, best_snapshot, params


def run_smc_sampler(B_train, y_train, args, seed):
    """Runs SMC; returns (particle trees, particle weights, params)."""
    settings = make_settings(args, n_particles=args.particles)
    data = make_data(B_train, y_train)
    np.random.seed(seed)
    random.seed(seed)

    particles, param, log_weights, cache, cache_tmp = init_smc(data, settings)
    particles, _, log_weights_itr, log_pd, _, _, _ = run_smc(
        particles, data, settings, param, log_weights, cache
    )
    log_weights = np.asarray(log_weights_itr)[-1, :]
    weights = np.exp(log_weights - logsumexp(log_weights))
    trees = [(dict(p.node_info), list(p.leaf_nodes)) for p in particles]
    params = {
        "particles": args.particles,
        "log_marginal_estimate": float(log_pd),
    }
    return trees, weights, params


def fit_trees(snapshots, B_train, y_train):
    return [
        parse_node_info(node_info, set(leaves)).fit_counts(B_train, y_train)
        for node_info, leaves in snapshots
    ]


if __name__ == "__main__":
    parser = make_parser(__doc__)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["mcmc", "smc"],
        choices=["mcmc", "smc"],
        help="Samplers to run; mcmc also writes the bcart_map result.",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        default=9,
        help="Quantile thresholds per feature for binarization.",
    )
    parser.add_argument("--iterations", type=int, default=50000, help="MCMC moves.")
    parser.add_argument("--burn-in-frac", type=float, default=0.5)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Max posterior samples kept (by thinning) for bcart_mcmc.",
    )
    parser.add_argument("--particles", type=int, default=1000, help="SMC particles.")
    parser.add_argument("--alpha-split", type=float, default=0.95)
    parser.add_argument("--beta-split", type=float, default=0.5)
    parser.add_argument(
        "--rho", type=float, default=2.5, help="Beta(rho, rho) leaf label prior."
    )
    args = parser.parse_args()
    rho = (args.rho, args.rho)

    for dataset in args.datasets:
        for split in args.splits:
            seed = split
            X_train, y_train, X_test, y_test = load_split(dataset, split)
            B_train, B_test, thresholds = binarize_quantiles(
                X_train, X_test, args.thresholds
            )
            shared_params = {
                "thresholds_per_feature": args.thresholds,
                "n_binary_features": len(thresholds),
                "alpha_split": args.alpha_split,
                "beta_split": args.beta_split,
                "rho": args.rho,
            }

            results = []  # (method, snapshots, weights, params)
            if "mcmc" in args.methods:
                t0 = time.time()
                samples, best, params = run_mcmc(B_train, y_train, args, seed)
                runtime_s = time.time() - t0
                results.append(
                    ("bcart_mcmc", samples, np.ones(len(samples)), params, runtime_s)
                )
                results.append(("bcart_map", [best], np.ones(1), params, runtime_s))
            if "smc" in args.methods:
                t0 = time.time()
                trees, weights, params = run_smc_sampler(B_train, y_train, args, seed)
                runtime_s = time.time() - t0
                results.append(("bcart_smc", trees, weights, params, runtime_s))

            for method, snapshots, weights, params, runtime_s in results:
                trees = fit_trees(snapshots, B_train, y_train)
                metrics = classification_metrics(
                    y_train,
                    bma_proba(trees, weights, B_train, rho),
                    y_test,
                    bma_proba(trees, weights, B_test, rho),
                )
                if method == "bcart_map":
                    params = {
                        **params,
                        "tree_size": trees[0].size(),
                        "tree_depth": trees[0].depth(),
                    }
                save_result(
                    args.results_dir,
                    method,
                    dataset,
                    split,
                    seed,
                    metrics,
                    {**shared_params, **params},
                    runtime_s,
                )
                print(
                    f"[{method}] {dataset} split {split}: "
                    f"test_acc={metrics['test_acc']:.4f} "
                    f"test_auc={metrics['test_auc']:.4f} "
                    f"test_logloss={metrics['test_logloss']:.4f} "
                    f"({runtime_s:.1f}s)",
                    flush=True,
                )
