"""
Gradient-boosted-tree benchmarks (XGBoost and LightGBM) for the
regression-tree experiments, with max_depth=5 to match the GFlowNet trees.

The number of boosting rounds is chosen by early stopping (50 rounds patience,
up to 2000 rounds) on a 20% validation carve-out of the train split; the model
is then refit on the full train split with the selected number of rounds.
RMSE / R2 are in the original target units (trees are scale-invariant, so no
feature/target scaling is applied).

Usage (from the repo root, venv active):
    python reg_benchmarks/run_gbt.py [--datasets concrete ...] [--splits 1 2 ...]
"""

import os
import sys
from pathlib import Path

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_parser, run_methods

MAX_DEPTH = 5
MAX_ROUNDS = 2000
LEARNING_RATE = 0.05
EARLY_STOPPING_ROUNDS = 50
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))


def _carve_validation(X_train, y_train, seed):
    return train_test_split(X_train, y_train, test_size=0.2, random_state=seed)


def fit_predict_xgboost(X_train, y_train, X_test, seed):
    X_tr, X_val, y_tr, y_val = _carve_validation(X_train, y_train, seed)
    base_kwargs = dict(
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=N_JOBS,
    )
    probe = XGBRegressor(
        n_estimators=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **base_kwargs,
    )
    probe.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    n_rounds = int(probe.best_iteration) + 1

    model = XGBRegressor(n_estimators=n_rounds, **base_kwargs)
    model.fit(X_train, y_train)
    params = {"max_depth": MAX_DEPTH, "n_estimators": n_rounds}
    return model.predict(X_train), model.predict(X_test), params


def fit_predict_lightgbm(X_train, y_train, X_test, seed):
    X_tr, X_val, y_tr, y_val = _carve_validation(X_train, y_train, seed)
    base_kwargs = dict(
        max_depth=MAX_DEPTH,
        num_leaves=31,  # < 2**MAX_DEPTH, so the depth cap binds
        learning_rate=LEARNING_RATE,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=N_JOBS,
        verbose=-1,
    )
    probe = lgb.LGBMRegressor(n_estimators=MAX_ROUNDS, **base_kwargs)
    probe.fit(
        X_tr,
        y_tr,
        eval_X=X_val,
        eval_y=y_val,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    n_rounds = int(probe.best_iteration_)

    model = lgb.LGBMRegressor(n_estimators=n_rounds, **base_kwargs)
    model.fit(X_train, y_train)
    params = {"max_depth": MAX_DEPTH, "n_estimators": n_rounds}
    return model.predict(X_train), model.predict(X_test), params


METHODS = {
    "xgboost": fit_predict_xgboost,
    "lightgbm": fit_predict_lightgbm,
}


if __name__ == "__main__":
    args = make_parser(__doc__).parse_args()
    run_methods(METHODS, args)
