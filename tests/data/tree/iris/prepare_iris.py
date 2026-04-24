"""
Generate iris_2.csv through iris_5.csv matching the format of iris_1.csv.
Run from: /Users/timarni/Documents/dt-gfn/gfn/data/iris/
    python prepare_iris.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit

iris = load_iris()
X = iris.data          # (150, 4)
y = iris.target        # 0, 1, 2 — already 0-indexed
feature_names = ["sepal length", "sepal width", "petal length", "petal width"]

for seed in [2, 3, 4, 5]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))

    df_train = pd.DataFrame(X[train_idx], columns=feature_names)
    df_train["class"] = y[train_idx]
    df_train["Split"] = "train"

    df_test = pd.DataFrame(X[test_idx], columns=feature_names)
    df_test["class"] = y[test_idx]
    df_test["Split"] = "test"

    pd.concat([df_train, df_test], ignore_index=True).to_csv(f"iris_{seed}.csv", index=False)
    print(f"Seed {seed}: {len(train_idx)} train, {len(test_idx)} test")
