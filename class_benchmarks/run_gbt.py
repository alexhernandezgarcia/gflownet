"""
Gradient-boosted-tree benchmarks (XGBoost, LightGBM and CatBoost) for the
classification-tree experiments, with --max-depth (default 5) matching the
GFlowNet trees. For non-default depths the method names are suffixed with
_d<depth> (e.g. xgboost_d3) so results of several depths coexist in the
results dir.

The number of boosting rounds is chosen by early stopping (50 rounds
patience, up to 2000 rounds) on a 20% stratified validation carve-out of the
train split; the model is then refit on the full train split with the
selected number of rounds. Predictions are P(y=1).

Usage (from the repo root, venv active):
    python class_benchmarks/run_gbt.py [--datasets magic] [--splits 1 2 ...]
                                        [--max-depth 5]
"""

import os
import sys
from pathlib import Path

import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_parser, run_methods

DEFAULT_MAX_DEPTH = 5
MAX_ROUNDS = 2000
LEARNING_RATE = 0.05
EARLY_STOPPING_ROUNDS = 50
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))


def _carve_validation(X_train, y_train, seed):
    return train_test_split(
        X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
    )


def fit_predict_xgboost(X_train, y_train, X_test, seed, max_depth):
    X_tr, X_val, y_tr, y_val = _carve_validation(X_train, y_train, seed)
    base_kwargs = dict(
        max_depth=max_depth,
        learning_rate=LEARNING_RATE,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=N_JOBS,
    )
    probe = XGBClassifier(
        n_estimators=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        **base_kwargs,
    )
    probe.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    n_rounds = int(probe.best_iteration) + 1

    model = XGBClassifier(n_estimators=n_rounds, **base_kwargs)
    model.fit(X_train, y_train)
    params = {"max_depth": max_depth, "n_estimators": n_rounds}
    return (
        model.predict_proba(X_train)[:, 1],
        model.predict_proba(X_test)[:, 1],
        params,
    )


def fit_predict_lightgbm(X_train, y_train, X_test, seed, max_depth):
    X_tr, X_val, y_tr, y_val = _carve_validation(X_train, y_train, seed)
    base_kwargs = dict(
        max_depth=max_depth,
        num_leaves=min(31, 2**max_depth - 1),  # < 2**max_depth: the depth cap binds
        learning_rate=LEARNING_RATE,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=N_JOBS,
        verbose=-1,
    )
    probe = lgb.LGBMClassifier(n_estimators=MAX_ROUNDS, **base_kwargs)
    probe.fit(
        X_tr,
        y_tr,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    n_rounds = int(probe.best_iteration_)

    model = lgb.LGBMClassifier(n_estimators=n_rounds, **base_kwargs)
    model.fit(X_train, y_train)
    params = {"max_depth": max_depth, "n_estimators": n_rounds}
    return (
        model.predict_proba(X_train)[:, 1],
        model.predict_proba(X_test)[:, 1],
        params,
    )


def fit_predict_catboost(X_train, y_train, X_test, seed, max_depth):
    X_tr, X_val, y_tr, y_val = _carve_validation(X_train, y_train, seed)
    base_kwargs = dict(
        depth=max_depth,
        learning_rate=LEARNING_RATE,
        loss_function="Logloss",
        random_seed=seed,
        thread_count=N_JOBS,
        verbose=0,
        allow_writing_files=False,
    )
    probe = CatBoostClassifier(iterations=MAX_ROUNDS, **base_kwargs)
    probe.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    n_rounds = int(probe.get_best_iteration()) + 1

    model = CatBoostClassifier(iterations=n_rounds, **base_kwargs)
    model.fit(X_train, y_train)
    params = {"depth": max_depth, "iterations": n_rounds}
    return (
        model.predict_proba(X_train)[:, 1],
        model.predict_proba(X_test)[:, 1],
        params,
    )


def make_methods(max_depth):
    suffix = "" if max_depth == DEFAULT_MAX_DEPTH else f"_d{max_depth}"

    def with_depth(fit_predict):
        return lambda X_train, y_train, X_test, seed: fit_predict(
            X_train, y_train, X_test, seed, max_depth
        )

    return {
        f"xgboost{suffix}": with_depth(fit_predict_xgboost),
        f"lightgbm{suffix}": with_depth(fit_predict_lightgbm),
        f"catboost{suffix}": with_depth(fit_predict_catboost),
    }


if __name__ == "__main__":
    parser = make_parser(__doc__)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    args = parser.parse_args()
    run_methods(make_methods(args.max_depth), args)
