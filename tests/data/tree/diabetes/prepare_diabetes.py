"""
Generate diabetes_1.csv through diabetes_5.csv in DT-GFN CSV format
(regression target: disease progression one year after baseline).

The dataset ships with scikit-learn (no download needed); raw (unscaled)
feature values are used, since the Tree env applies min-max scaling itself.

Run from: tests/data/tree/diabetes/
    python prepare_diabetes.py
"""

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

diabetes = load_diabetes(scaled=False)
X = diabetes.data  # (442, 10)
y = diabetes.target  # continuous, 25-346
feature_names = list(diabetes.feature_names)

print(f"Shape: {X.shape} | target range: [{y.min()}, {y.max()}]")

for seed in [1, 2, 3, 4, 5]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train["target"] = y_train
    df_train["Split"] = "train"

    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test["target"] = y_test
    df_test["Split"] = "test"

    pd.concat([df_train, df_test], ignore_index=True).to_csv(
        f"diabetes_{seed}.csv", index=False
    )
    print(f"Seed {seed}: {len(y_train)} train, {len(y_test)} test")
