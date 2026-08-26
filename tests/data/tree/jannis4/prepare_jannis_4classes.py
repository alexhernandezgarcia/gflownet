"""
Convert the original 4-class "jannis" dataset (ChaLearn AutoML challenge) to
DT-GFN CSV format. Generates jannis4_1.csv through jannis4_5.csv.

Source: OpenML dataset 41168, 83733 x 54, four *imbalanced* classes
(1687 / 28790 / 14734 / 38522), all numerical, no missing values. This is the
version used by Gorishniy et al. (2021, "Revisiting Deep Learning Models for
Tabular Data"), whose Table 2 reports MLP / ResNet / FT-Transformer / NODE /
XGBoost / CatBoost accuracies -- but on *their* fixed train/val/test split, not
on the stratified splits generated here (see README).

Download the raw parquet once (needs internet, so from a login node):
    curl -L -o jannis_41168.pq \
      "https://data.openml.org/datasets/0004/41168/dataset_41168.pq"

Then:
    python prepare_jannis_4classes.py
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare_utils import add_common_args, build_splits, require_raw  # noqa: E402

RAW_URL = "https://data.openml.org/datasets/0004/41168/dataset_41168.pq"
TARGET = "class"
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=HERE / "jannis_41168.pq")
    add_common_args(parser)
    args = parser.parse_args()

    df = pd.read_parquet(require_raw(args.raw, RAW_URL))
    # In the OpenML parquet the target is the *first* column and is stored as
    # a pandas category of the strings "0".."3".
    feature_names = [c for c in df.columns if c != TARGET]

    X = df[feature_names].astype(float)
    y = df[TARGET].astype(str).astype(int).to_numpy()

    build_splits(
        X,
        y,
        "jannis4",
        seeds=args.seeds,
        test_size=args.test_size,
        max_train=args.max_train,
        transform=args.transform,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
