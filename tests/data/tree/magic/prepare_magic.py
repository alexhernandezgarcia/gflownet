"""
Convert UCI MAGIC Gamma Telescope magic04.data to DT-GFN CSV format.
Generates magic_1.csv through magic_5.csv.
Run from: tests/data/tree/magic/
    python prepare_magic.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

feature_names = [
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
]

df_raw = pd.read_csv("magic04.data", header=None, names=feature_names + ["class"])

X = df_raw[feature_names].values  # (19020, 10)
# g = gamma (signal), h = hadron (background) → 0, 1 (required by DT-GFN)
y = df_raw["class"].map({"g": 0, "h": 1}).values

print(f"Shape of dataset is: {X.shape}")
print(f"Class counts: {np.bincount(y).tolist()}")  # should be [12332, 6688]

for seed in [1, 2, 3, 4, 5]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))

    df_train = pd.DataFrame(X[train_idx], columns=feature_names)
    df_train["class"] = y[train_idx]
    df_train["Split"] = "train"

    df_test = pd.DataFrame(X[test_idx], columns=feature_names)
    df_test["class"] = y[test_idx]
    df_test["Split"] = "test"

    pd.concat([df_train, df_test], ignore_index=True).to_csv(
        f"magic_{seed}.csv", index=False
    )
    print(
        f"Seed {seed}: {len(train_idx)} train, {len(test_idx)} test | "
        f"class counts: {np.bincount(y[train_idx]).tolist()}  (0=gamma, 1=hadron)"
    )
