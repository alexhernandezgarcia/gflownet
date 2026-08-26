"""
Convert the binarized "jannis" dataset of the Grinsztajn et al. (2022) tabular
benchmark to DT-GFN CSV format. Generates jannis2_1.csv through jannis2_5.csv.

Source: OpenML dataset 44079 / task 361274, 57580 x 54, two balanced classes
(the original class 1 against the rest, subsampled to balance), all numerical,
no missing values. Baselines: Grinsztajn et al. (2022) and the raw sweep CSVs
under analyses/results/ in github.com/LeoGrin/tabular-benchmark.

Download the raw CSV once (needs internet, so from a login node):
    curl -L -o jannis.csv \
      "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/jannis.csv"

Then:
    python prepare_jannis_2classes.py
    python prepare_jannis_2classes.py --max-train 10000   # Grinsztajn "medium" regime
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare_utils import add_common_args, build_splits, require_raw  # noqa: E402

RAW_URL = (
    "https://huggingface.co/datasets/inria-soda/tabular-benchmark/"
    "resolve/main/clf_num/jannis.csv"
)
TARGET = "class"
HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=HERE / "jannis.csv")
    add_common_args(parser)
    args = parser.parse_args()

    df = pd.read_csv(require_raw(args.raw, RAW_URL))
    feature_names = [c for c in df.columns if c != TARGET]

    X = df[feature_names]
    y = df[TARGET].to_numpy(dtype=int)  # already 0/1 and balanced

    build_splits(
        X,
        y,
        "jannis2",
        seeds=args.seeds,
        test_size=args.test_size,
        max_train=args.max_train,
        transform=args.transform,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
