"""
Bayesian CART regression benchmarks (Chipman et al. 1998) on quantile-
binarized features, sampled with the tree_smc code of Lakshminarayanan et al.
(2013) vendored in class_baselines/tree_smc -- the regression counterpart of
class_baselines/run_bcart.py, with the same three methods:

  - bcart_mcmc: posterior predictive mean averaged over trees sampled by
    Chipman-style Metropolis-Hastings (grow/prune/change/swap moves), after
    burn-in and with thinning.
  - bcart_map: the single highest-posterior tree visited by the same MCMC
    chain (written together with bcart_mcmc at no extra sampling cost);
    ``params`` records its total node count (``tree_size``) and depth.
  - bcart_smc: posterior predictive mean averaged over the particles of the
    top-down SMC sampler, weighted by the final particle weights.

Model. Tree prior: P(split at depth d) = alpha_split * (1+d)^(-beta_split)
with a uniform choice of binary feature (MAPTree / classification-benchmark
defaults). Leaf model: Normal-Inverse-Gamma, mu | s2 ~ N(mu_0, s2/kappa_0),
s2 ~ InvGamma(alpha_0, beta_0), with the DT-GFN regression proxy defaults
(config/proxy/regression_tree.yaml): mu_0 = mean(y_train), kappa_0 = 0.1,
alpha_0 = 2, beta_0 = (alpha_0 - 1) * var(y_train). Predictions are the NIG
posterior means of the leaves (the same point prediction as
RegressionTree.test), so RMSE / R2 are in the original target units.

The vendored MCMC sampler originally supported classification only (its
change/swap moves re-evaluated subtrees with class counts); bdtmcmc.py was
extended to real-valued outputs, and this script checks after every chain
that the sampler's incremental log-likelihood equals a from-scratch
recomputation of the final tree.

Only continuous targets make sense here: datasets whose target takes at most
MIN_UNIQUE_TARGETS distinct values (e.g. the classification datasets magic,
credit_quantile, jannis2) are skipped with a message.

Usage (from the repo root, venv active):
    python reg_benchmarks/run_bcart.py [--datasets yacht real_estate ...]
        [--methods mcmc smc] [--splits 1 2 ...] [--iterations 50000]
        [--particles 1000] [--kappa-0 0.1 --alpha-0 2.0 --beta-0 ...]
"""

import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # tree_smc imports pyplot; avoid needing a display

import numpy as np  # noqa: E402
from scipy.special import gammaln, logsumexp  # noqa: E402

BENCHMARK_DIR = Path(__file__).resolve().parent
# reg_benchmarks first (its common.py must win over class_baselines/common.py);
# class_baselines provides binarize_quantiles and the vendored tree_smc package.
sys.path.insert(0, str(BENCHMARK_DIR.parent / "class_baselines"))
sys.path.insert(0, str(BENCHMARK_DIR))
from binary_reg_trees import (  # noqa: E402
    NIGPrior,
    bma_predict,
    parse_node_info,
)
from binary_trees import binarize_quantiles  # noqa: E402
from common import load_split, make_parser, regression_metrics, save_result  # noqa: E402

import tree_smc.bdtsmc as bdtsmc  # noqa: E402
from tree_smc.bdtmcmc import parser_add_mcmc_options, sample_tree  # noqa: E402
from tree_smc.bdtsmc import init_smc, run_smc  # noqa: E402
from tree_smc.tree_utils import (  # noqa: E402
    parser_add_common_options,
    parser_add_smc_options,
    precompute,
)

MIN_UNIQUE_TARGETS = 10


def make_settings(args, n_particles=None):
    parser = parser_add_common_options()
    parser = parser_add_smc_options(parser)
    parser = parser_add_mcmc_options(parser)
    argv = [
        "--optype",
        "real",
        "--prior",
        "cgm",  # Normal-Gamma leaves (tree_smc's name for the NIG model)
        "--alpha_split",
        str(args.alpha_split),
        "--beta_split",
        str(args.beta_split),
        "--kappa_0",
        str(args.kappa_0),
        "--verbose",
        "0",
    ]
    if n_particles is not None:
        argv += ["--n_particles", str(n_particles), "--n_islands", "1"]
    return parser.parse_args(argv)[0]


def make_data(B_train, y_train):
    return {
        "x_train": B_train.astype(int),
        "y_train": np.asarray(y_train, dtype=float),
        "n_train": B_train.shape[0],
        "n_dim": B_train.shape[1],
    }


def resolve_prior(y_train, args) -> NIGPrior:
    """Data-driven defaults exactly as RegressionTree._resolve_nig_params."""
    mu_0 = float(np.mean(y_train)) if args.mu_0 is None else args.mu_0
    if args.beta_0 is None:
        var = float(np.var(y_train))
        if var <= 0.0:
            var = 1.0
        beta_0 = (args.alpha_0 - 1.0) * var
    else:
        beta_0 = args.beta_0
    return NIGPrior(mu_0=mu_0, kappa_0=args.kappa_0, alpha_0=args.alpha_0, beta_0=beta_0)


def set_nig_prior(param, cache, cache_tmp, prior: NIGPrior):
    """
    Overrides the leaf prior tree_smc.precompute hard-codes for regression
    (alpha_0 = 3, beta_0 from a quantile rule) with ``prior``. The prior
    normalizer is cached in both dicts (cache_tmp is used for the root node,
    cache for every later node), so both are refreshed.
    """
    param.mu_0 = prior.mu_0
    param.kappa_0 = prior.kappa_0
    param.alpha_0 = prior.alpha_0
    param.beta_0 = prior.beta_0
    term = (
        prior.alpha_0 * np.log(prior.beta_0)
        + 0.5 * np.log(prior.kappa_0)
        - gammaln(prior.alpha_0)
    )
    cache["ng_prior_term"] = term
    cache_tmp["ng_prior_term"] = term


def check_loglik_consistency(p, B_train, y_train, prior: NIGPrior):
    """
    The sampler tracks the tree log-likelihood incrementally across moves;
    compare it with a from-scratch recomputation of the current tree. Guards
    the regression extension of the vendored MCMC code.
    """
    tree = parse_node_info(dict(p.node_info), set(p.leaf_nodes)).fit_stats(
        B_train, y_train
    )
    scratch = tree.log_marginal_likelihood(prior)
    incremental = float(p.compute_loglik())
    if not np.isclose(scratch, incremental, rtol=1e-8, atol=1e-6):
        raise RuntimeError(
            f"MCMC log-likelihood bookkeeping is inconsistent: incremental "
            f"{incremental:.6f} vs from-scratch {scratch:.6f}"
        )


def run_mcmc(B_train, y_train, args, seed, prior: NIGPrior):
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
    set_nig_prior(param, cache, cache_tmp, prior)
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
    check_loglik_consistency(p, B_train, y_train, prior)
    params = {
        "iterations": args.iterations,
        "burn_in": burn_in,
        "thin": thin,
        "n_samples": len(samples),
        "best_log_posterior": float(best_post),
    }
    return samples, best_snapshot, params


def run_smc_sampler(B_train, y_train, args, seed, prior: NIGPrior):
    """Runs SMC; returns (particle trees, particle weights, params)."""
    settings = make_settings(args, n_particles=args.particles)
    data = make_data(B_train, y_train)
    np.random.seed(seed)
    random.seed(seed)

    # init_smc calls precompute internally and immediately builds the root
    # particles from its output, so the prior override is injected through the
    # module-level name init_smc resolves (restored afterwards).
    def precompute_with_prior(data_, settings_):
        param_, cache_, cache_tmp_ = precompute(data_, settings_)
        set_nig_prior(param_, cache_, cache_tmp_, prior)
        return param_, cache_, cache_tmp_

    original = bdtsmc.precompute
    bdtsmc.precompute = precompute_with_prior
    try:
        particles, param, log_weights, cache, cache_tmp = init_smc(data, settings)
    finally:
        bdtsmc.precompute = original
    assert np.isclose(param.alpha_0, prior.alpha_0), "prior override not applied"
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
        parse_node_info(node_info, set(leaves)).fit_stats(B_train, y_train)
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
        "--mu-0",
        type=float,
        default=None,
        help="NIG prior mean of the leaf means (default: mean of y_train).",
    )
    parser.add_argument("--kappa-0", type=float, default=0.1, help="NIG kappa_0.")
    parser.add_argument("--alpha-0", type=float, default=2.0, help="NIG alpha_0.")
    parser.add_argument(
        "--beta-0",
        type=float,
        default=None,
        help="NIG beta_0 (default: (alpha_0 - 1) * var(y_train), i.e. the "
        "prior mean of sigma^2 equals the target variance).",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        for split in args.splits:
            seed = split
            X_train, y_train, X_test, y_test = load_split(dataset, split)
            if np.unique(y_train).size < MIN_UNIQUE_TARGETS:
                print(
                    f"[skip] {dataset} split {split}: target takes only "
                    f"{np.unique(y_train).size} distinct values -- not a "
                    f"regression dataset"
                )
                continue
            B_train, B_test, thresholds = binarize_quantiles(
                X_train, X_test, args.thresholds
            )
            prior = resolve_prior(y_train, args)
            shared_params = {
                "thresholds_per_feature": args.thresholds,
                "n_binary_features": len(thresholds),
                "alpha_split": args.alpha_split,
                "beta_split": args.beta_split,
                "mu_0": prior.mu_0,
                "kappa_0": prior.kappa_0,
                "alpha_0": prior.alpha_0,
                "beta_0": prior.beta_0,
            }

            results = []  # (method, snapshots, weights, params, runtime)
            if "mcmc" in args.methods:
                t0 = time.time()
                samples, best, params = run_mcmc(B_train, y_train, args, seed, prior)
                runtime_s = time.time() - t0
                results.append(
                    ("bcart_mcmc", samples, np.ones(len(samples)), params, runtime_s)
                )
                results.append(("bcart_map", [best], np.ones(1), params, runtime_s))
            if "smc" in args.methods:
                t0 = time.time()
                trees, weights, params = run_smc_sampler(
                    B_train, y_train, args, seed, prior
                )
                runtime_s = time.time() - t0
                results.append(("bcart_smc", trees, weights, params, runtime_s))

            for method, snapshots, weights, params, runtime_s in results:
                trees = fit_trees(snapshots, B_train, y_train)
                metrics = regression_metrics(
                    y_train,
                    bma_predict(trees, weights, B_train, prior),
                    y_test,
                    bma_predict(trees, weights, B_test, prior),
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
                    f"test_rmse={metrics['test_rmse']:.4f} "
                    f"test_r2={metrics['test_r2']:.4f} ({runtime_s:.1f}s)",
                    flush=True,
                )
