"""
Convert UCI Raisin dataset to DT-GFN CSV format.
Run from: /Users/timarni/Documents/dt-gfn/gfn/data/raisin/
    python prepare_raisin.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_excel("Raisin_Dataset.xlsx")

# Map classes to 0-indexed integers
class_map = {"Kecimen": 0, "Besni": 1}
df["Class"] = df["Class"].map(class_map)

feature_cols = ["Area", "MajorAxisLength", "MinorAxisLength",
                "Eccentricity", "ConvexArea", "Extent", "Perimeter"]
X = df[feature_cols].values
print(f"Shape of dataset is: {X.shape}")
y = df["Class"].values

print(f"Shape: {X.shape}")  # should be (900, 7)
print(f"Class counts: {np.bincount(y).tolist()}") # should be [450, 450]

for seed in [1, 2, 3, 4, 5]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))

    df_train = pd.DataFrame(X[train_idx], columns=feature_cols)
    df_train["class"] = y[train_idx]
    df_train["Split"] = "train"

    df_test = pd.DataFrame(X[test_idx], columns=feature_cols)
    df_test["class"] = y[test_idx]
    df_test["Split"] = "test"

    pd.concat([df_train, df_test], ignore_index=True).to_csv(
        f"raisin_{seed}.csv", index=False
    )
    print(f"Seed {seed}: {len(train_idx)} train, {len(test_idx)} test")