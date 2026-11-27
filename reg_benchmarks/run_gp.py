"""
Gaussian Process regression benchmark for the regression-tree experiments.

Uses an anisotropic (ARD) RBF kernel with a learned constant scale and a
WhiteKernel noise term; hyper-parameters are optimized by maximizing the log
marginal likelihood with multiple restarts. Features are standardized with
train-split statistics and targets are normalized internally by sklearn
(normalize_y=True); reported RMSE / R2 are in the original target units.

Usage (from the repo root, venv active):
    python reg_benchmarks/run_gp.py [--datasets concrete ...] [--splits 1 2 ...]
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_parser, run_methods


def fit_predict_gp(X_train, y_train, X_test, seed):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_features = X_train.shape[1]
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e5)
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=seed,
    )
    gp.fit(X_train_s, y_train)
    params = {
        "kernel": str(gp.kernel_),
        "log_marginal_likelihood": float(gp.log_marginal_likelihood_value_),
    }
    return gp.predict(X_train_s), gp.predict(X_test_s), params


if __name__ == "__main__":
    args = make_parser(__doc__).parse_args()
    run_methods({"gp": fit_predict_gp}, args)
