"""
BART (Bayesian Additive Regression Trees, Chipman et al. 2010) benchmark for
the regression-tree experiments, sampled with MCMC via pymc-bart (PGBART).

Model: y ~ Normal(mu, sigma) with mu = BART(X; m trees) and a HalfNormal prior
on sigma, on targets standardized with train-split statistics (the standard
BART preprocessing, and the same standardization RegressionTree applies).
Point predictions are the posterior mean of mu, rescaled back to the original
target units. Note that pymc-bart has no hard depth cap: tree depth is
regularized by the standard branching prior alpha * (1 + d)^(-beta) instead
(defaults alpha=0.95, beta=2), so the GFlowNet max_depth=5 does not apply.

Usage (from the repo root, venv active):
    python reg_benchmarks/run_bart.py [--datasets concrete ...] [--splits 1 2 ...]
                                       [--trees 50] [--draws 1000] [--tune 1000]
                                       [--chains 2]
"""

import os
import sys
from pathlib import Path

# The cluster's cvmfs Python only ships a static (non-PIC) libpython, which
# pytensor's C backend fails to link against; use its pure-Python backend.
# This barely affects speed: the PGBART sampler is NumPy code, and the only
# compiled graphs are the tiny likelihood/prior terms.
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")

import numpy as np  # noqa: E402
import pymc as pm  # noqa: E402
import pymc_bart as pmb  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_parser, run_methods


def make_fit_predict_bart(trees, draws, tune, chains):
    def fit_predict(X_train, y_train, X_test, seed):
        y_mean = float(np.mean(y_train))
        y_std = float(np.std(y_train)) or 1.0
        y_train_s = (y_train - y_mean) / y_std

        with pm.Model() as model:
            X = pm.Data("X", X_train)
            mu = pmb.BART("mu", X, y_train_s, m=trees)
            sigma = pm.HalfNormal("sigma", sigma=1.0)
            pm.Normal("y", mu=mu, sigma=sigma, observed=y_train_s, shape=mu.shape)
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=min(chains, 2),
                random_seed=seed,
                progressbar=False,
                compute_convergence_checks=False,
            )
            train_pred_s = idata.posterior["mu"].mean(dim=("chain", "draw")).values

            pm.set_data({"X": X_test})
            ppc = pm.sample_posterior_predictive(
                idata, var_names=["mu"], random_seed=seed, progressbar=False
            )
            test_pred_s = (
                ppc.posterior_predictive["mu"].mean(dim=("chain", "draw")).values
            )

        params = {
            "trees": trees,
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "sigma_posterior_mean": float(
                idata.posterior["sigma"].mean().item() * y_std
            ),
        }
        return (
            train_pred_s * y_std + y_mean,
            test_pred_s * y_std + y_mean,
            params,
        )

    return fit_predict


if __name__ == "__main__":
    parser = make_parser(__doc__)
    parser.add_argument("--trees", type=int, default=50, help="Trees in the sum (m).")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    args = parser.parse_args()
    fit_predict = make_fit_predict_bart(args.trees, args.draws, args.tune, args.chains)
    run_methods({"bart": fit_predict}, args)
