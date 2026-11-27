"""
Shared helpers for the ``prepare_*.py`` dataset scripts under tests/data/tree/.

Each script turns one raw download into the split-CSV layout that
``gflownet.envs.tree.tree.Tree._load_dataset`` and
``class_baselines/common.py::load_split`` expect:

    tests/data/tree/<name>/<name>_<seed>.csv

with columns ``<feature_1>, ..., <feature_d>, class, Split``, where ``Split``
holds "train"/"test" and ``class`` is an integer label in ``0..K-1``.

This mirrors what tests/data/tree/magic/prepare_magic.py does inline, factored
out because the three tabular-benchmark datasets (credit, jannis) only differ
in how the raw file is read.
"""

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import QuantileTransformer

DEFAULT_SEEDS = (1, 2, 3, 4, 5)
TREE_DATA_DIR = Path(__file__).resolve().parent


def require_raw(path: Path, url: str) -> Path:
    """
    Returns ``path`` if the raw download exists, otherwise raises with the
    exact command needed to fetch it. The prepare scripts deliberately do not
    download anything themselves: the cluster compute nodes have no internet
    access, so the raw file must be fetched once from a login node.
    """
    if not path.exists():
        raise SystemExit(
            f"Raw file not found: {path}\n"
            f"Download it first (from a node with internet access):\n"
            f'    curl -L -o "{path}" "{url}"'
        )
    return path


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Adds the arguments shared by every prepare script."""
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the data held out as test set (default: 0.2).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Split seeds; one CSV is written per seed (default: 1 2 3 4 5).",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help=(
            "If set, stratified-subsample the training set of each split down "
            "to this many samples (e.g. 10000 to match the 'medium-sized' "
            "regime of the Grinsztajn et al. tabular benchmark). The test set "
            "is never subsampled."
        ),
    )
    parser.add_argument(
        "--transform",
        choices=("none", "quantile"),
        default="none",
        help=(
            "Per-feature transform, fit on the training split only. 'quantile' "
            "maps each feature to its empirical CDF; it is monotone, so it "
            "leaves CART/RF/GBT baselines invariant, but it spreads the data "
            "over the equally-spaced threshold grid the discrete Tree node "
            "uses. Splits are written to <name>_quantile/ (default: none)."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=TREE_DATA_DIR,
        help="Directory holding the per-dataset folders (default: this one).",
    )
    return parser


def _threshold_grid_report(X: pd.DataFrame, n_thresholds: int = 99) -> None:
    """
    Warns about features that the discrete Tree node cannot usefully split.

    That node min-max scales the data and picks thresholds from an
    equally-spaced grid on [0, 1]. A feature whose mass sits almost entirely
    inside a single grid cell is therefore effectively unsplittable, no matter
    how informative its ranks are.
    """
    values = X.to_numpy(dtype=float)
    span = values.max(axis=0) - values.min(axis=0)
    span[span == 0.0] = 1.0
    scaled = (values - values.min(axis=0)) / span
    cell = 1.0 / n_thresholds
    in_lowest = (scaled < cell).mean(axis=0)
    in_highest = (scaled > 1.0 - cell).mean(axis=0)
    worst = np.maximum(in_lowest, in_highest)
    degenerate = np.flatnonzero(worst > 0.95)
    if len(degenerate) == 0:
        return
    print(
        f"\nWARNING: with n_thresholds={n_thresholds} on min-max scaled data, "
        f"{len(degenerate)} feature(s) have >95% of their mass in a single "
        f"grid cell and are effectively unsplittable:"
    )
    for i in degenerate:
        print(f"    {X.columns[i]}: {worst[i]:.1%} of samples in one cell")
    print("Consider re-running with --transform quantile.\n")


def build_splits(
    X: pd.DataFrame,
    y: np.ndarray,
    name: str,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    test_size: float = 0.2,
    max_train: Optional[int] = None,
    transform: str = "none",
    out_root: Path = TREE_DATA_DIR,
) -> None:
    """
    Writes one stratified train/test split CSV per seed.

    Parameters
    ----------
    X : pd.DataFrame
        Numeric features; column names become the CSV feature names.
    y : np.ndarray
        Integer labels in ``0..K-1``.
    name : str
        Dataset name. Splits go to ``<out_root>/<name>/<name>_<seed>.csv``;
        with ``transform="quantile"`` the name is suffixed with ``_quantile``
        so both variants can coexist.
    """
    y = np.asarray(y, dtype=int)
    if X.isna().to_numpy().any():
        raise ValueError(f"{name}: features contain NaNs; handle them first.")
    classes = np.unique(y)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError(f"{name}: labels must be 0..K-1, got {classes.tolist()}")

    print(f"Shape of dataset is: {X.shape}")
    print(f"Class counts: {np.bincount(y).tolist()}")
    _threshold_grid_report(X)

    if transform == "quantile":
        name = f"{name}_quantile"
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))

        if max_train is not None and max_train < len(train_idx):
            sub = StratifiedShuffleSplit(
                n_splits=1, train_size=max_train, random_state=seed
            )
            keep, _ = next(sub.split(X.iloc[train_idx], y[train_idx]))
            train_idx = train_idx[keep]

        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)

        if transform == "quantile":
            # Fit on train only: the test set must not inform the transform.
            qt = QuantileTransformer(
                n_quantiles=min(1000, len(X_train)),
                output_distribution="uniform",
                subsample=None,
                random_state=seed,
            ).fit(X_train)
            X_train = pd.DataFrame(qt.transform(X_train), columns=X.columns)
            X_test = pd.DataFrame(qt.transform(X_test), columns=X.columns)

        df_train = X_train.copy()
        df_train["class"] = y[train_idx]
        df_train["Split"] = "train"

        df_test = X_test.copy()
        df_test["class"] = y[test_idx]
        df_test["Split"] = "test"

        path = out_dir / f"{name}_{seed}.csv"
        pd.concat([df_train, df_test], ignore_index=True).to_csv(path, index=False)
        print(
            f"Seed {seed}: {len(train_idx)} train, {len(test_idx)} test | "
            f"train class counts: {np.bincount(y[train_idx]).tolist()} | "
            f"-> {path} ({path.stat().st_size / 1e6:.1f} MB)"
        )
