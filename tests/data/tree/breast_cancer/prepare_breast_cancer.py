"""
Generate breast_cancer_1.csv through breast_cancer_5.csv for DT-GFN.
Run from: /Users/timarni/Documents/dt-gfn/gfn/data/breast_cancer/
    python prepare_breast_cancer.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedShuffleSplit

bc = load_breast_cancer()
X = bc.data          # (569, 30)
print(f"Shape fo dataset is {X.shape}")
y = bc.target        # 0 = malignant, 1 = benign — already 0-indexed

for seed in [1, 2, 3, 4, 5]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))

    df_train = pd.DataFrame(X[train_idx], columns=bc.feature_names)
    df_train["class"] = y[train_idx]
    df_train["Split"] = "train"

    df_test = pd.DataFrame(X[test_idx], columns=bc.feature_names)
    df_test["class"] = y[test_idx]
    df_test["Split"] = "test"

    pd.concat([df_train, df_test], ignore_index=True).to_csv(
        f"breast_cancer_{seed}.csv", index=False
    )
    print(f"Seed {seed}: {len(train_idx)} train, {len(test_idx)} test | "
          f"class counts: {np.bincount(y[train_idx]).tolist()}  (0=malignant, 1=benign)")
