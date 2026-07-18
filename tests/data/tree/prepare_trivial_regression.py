"""
Generate trivial_regression.csv: the regression analogue of trivial2d.csv.

Two features in [0, 1]; only x0 is informative: the target is a step function
y = 1.0 for x0 <= 0.5 and y = 5.0 otherwise, plus small Gaussian noise. The
Bayesian posterior over trees therefore has a single dominant mode (one root
split on x0 at 0.5), which makes this dataset ideal for debugging: a trained
DT-GFN policy should concentrate its samples on that tree.

Run from: tests/data/tree/
    python prepare_trivial_regression.py
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n_train, n_test = 60, 20
n = n_train + n_test

X = rng.random((n, 2)).round(2)
y = (np.where(X[:, 0] <= 0.5, 1.0, 5.0) + rng.normal(0.0, 0.1, n)).round(3)

df = pd.DataFrame(X, columns=["x0", "x1"])
df["target"] = y
df["Split"] = ["train"] * n_train + ["test"] * n_test

df.to_csv("trivial_regression.csv", index=False)
print(f"Wrote trivial_regression.csv: {n_train} train, {n_test} test")