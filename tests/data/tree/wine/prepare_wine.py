"""
Convert UCI wine.data to DT-GFN CSV format.
Run from: /Users/timarni/Documents/dt-gfn/gfn/data/wine/
    python prepare_wine.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

feature_names = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavonoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline",
]

df_raw = pd.read_csv("wine.data", header=None)
X = df_raw.iloc[:, 1:].values  # 13 features
y = df_raw.iloc[:, 0].values - 1  # classes 1,2,3 → 0,1,2 (required by DT-GFN)

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
        f"wine_{seed}.csv", index=False
    )
    print(
        f"Seed {seed}: {len(train_idx)} train, {len(test_idx)} test | "
        f"class counts: {np.bincount(y[train_idx]).tolist()}"
    )
