"""
Generate yacht_1.csv through yacht_5.csv in DT-GFN CSV format from the
UCI Yacht Hydrodynamics dataset (308 samples, 6 features; regression
target: residuary resistance per unit weight of displacement).

Downloads yacht_hydrodynamics.data from the UCI repository on first run.

Run from: tests/data/tree/yacht/
    python prepare_yacht.py
"""

import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import pandas as pd
from sklearn.model_selection import train_test_split

UCI_URL = "https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip"
DATA_NAME = "yacht_hydrodynamics.data"

# The raw file is whitespace-separated with no header; names by position.
FEATURE_COLS = [
    "long_pos_buoyancy",
    "prismatic_coeff",
    "length_disp_ratio",
    "beam_draught_ratio",
    "length_beam_ratio",
    "froude_number",
]

if not Path(DATA_NAME).exists():
    print(f"Downloading {UCI_URL} ...")
    # The cluster Python has no CA bundle configured; use certifi's
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(UCI_URL, context=ctx) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zf:
            zf.extract(DATA_NAME)

df = pd.read_csv(DATA_NAME, sep=r"\s+", header=None)
df.columns = FEATURE_COLS + ["resistance"]

X = df[FEATURE_COLS].values
y = df["resistance"].values

print(f"Shape: {X.shape} | target range: [{y.min()}, {y.max()}]")  # (308, 6)

for seed in [1, 2, 3, 4, 5]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    df_train = pd.DataFrame(X_train, columns=FEATURE_COLS)
    df_train["target"] = y_train
    df_train["Split"] = "train"

    df_test = pd.DataFrame(X_test, columns=FEATURE_COLS)
    df_test["target"] = y_test
    df_test["Split"] = "test"

    pd.concat([df_train, df_test], ignore_index=True).to_csv(
        f"yacht_{seed}.csv", index=False
    )
    print(f"Seed {seed}: {len(y_train)} train, {len(y_test)} test")
