"""
Convert the "credit" dataset of the Grinsztajn et al. (2022) tabular benchmark
to DT-GFN CSV format. Generates credit_1.csv through credit_5.csv.

Source: OpenML dataset 44089 / task 361055 ("clf_num" version of the Kaggle
"Give Me Some Credit" competition), 16714 x 10, two balanced classes, no
missing values. Baselines: Grinsztajn et al. (2022), Holzmuller et al. (2024).

Download the raw CSV once (needs internet, so from a login node):
    curl -L -o credit.csv \
      "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/credit.csv"

Then:
    python prepare_credit.py                     # plain 80/20 stratified splits
    python prepare_credit.py --transform quantile  # recommended, see README
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare_utils import add_common_args, build_splits, require_raw  # noqa: E402

RAW_URL = (
    "https://huggingface.co/datasets/inria-soda/tabular-benchmark/"
    "resolve/main/clf_num/credit.csv"
)
TARGET = "SeriousDlqin2yrs"
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=HERE / "credit.csv")
    parser.add_argument(
        "--drop-sentinel-rows",
        action="store_true",
        help=(
            "Drop the 89 rows where the 'NumberOfTime*DaysPastDue*' columns "
            "hold the Give Me Some Credit sentinel codes 96/98 ('not "
            "available'/'other'), which are otherwise read as counts."
        ),
    )
    add_common_args(parser)
    args = parser.parse_args()

    df = pd.read_csv(require_raw(args.raw, RAW_URL))
    feature_names = [c for c in df.columns if c != TARGET]

    if args.drop_sentinel_rows:
        past_due = [c for c in feature_names if c.startswith("NumberOfTime")]
        keep = (df[past_due] < 90).all(axis=1)
        print(f"Dropping {(~keep).sum()} rows with sentinel codes 96/98")
        df = df[keep].reset_index(drop=True)

    X = df[feature_names]
    y = df[TARGET].to_numpy(dtype=int)  # already 0/1 and balanced

    build_splits(
        X,
        y,
        "credit",
        seeds=args.seeds,
        test_size=args.test_size,
        max_train=args.max_train,
        transform=args.transform,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
