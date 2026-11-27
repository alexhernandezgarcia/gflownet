"""
Linear-model benchmarks for the regression-tree experiments: plain linear
regression, Ridge (alpha via leave-one-out CV), Lasso (alpha via 5-fold CV)
and Bayesian linear regression (sklearn BayesianRidge).

Features are standardized with train-split statistics; predictions (and hence
RMSE / R2) are in the original target units.

Usage (from the repo root, venv active):
    python reg_benchmarks/run_linear.py [--datasets concrete ...] [--splits 1 2 ...]
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import (BayesianRidge, LassoCV, LinearRegression,
                                  RidgeCV)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_parser, run_methods

ALPHAS = np.logspace(-4, 4, 33)


def _fit_predict_with(make_model):
    def fit_predict(X_train, y_train, X_test, seed):
        model = make_pipeline(StandardScaler(), make_model(seed))
        model.fit(X_train, y_train)
        params = {}
        estimator = model[-1]
        if hasattr(estimator, "alpha_"):
            params["alpha"] = float(np.asarray(estimator.alpha_).item())
        return model.predict(X_train), model.predict(X_test), params

    return fit_predict


METHODS = {
    "linear": _fit_predict_with(lambda seed: LinearRegression()),
    "ridge": _fit_predict_with(lambda seed: RidgeCV(alphas=ALPHAS)),
    "lasso": _fit_predict_with(
        lambda seed: LassoCV(cv=5, random_state=seed, max_iter=100000)
    ),
    "bayesian_ridge": _fit_predict_with(lambda seed: BayesianRidge()),
}


if __name__ == "__main__":
    args = make_parser(__doc__).parse_args()
    run_methods(METHODS, args)
