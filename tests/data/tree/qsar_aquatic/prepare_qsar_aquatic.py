"""
Generate qsar_aquatic_1.csv through qsar_aquatic_5.csv in DT-GFN CSV format
from the UCI QSAR Aquatic Toxicity dataset (546 samples, 8 molecular
descriptors; regression target: LC50 towards Daphnia magna, in
-log10(mol/L)).

Downloads qsar_aquatic_toxicity.csv from the UCI repository on first run.

Run from: tests/data/tree/qsar_aquatic/
    python prepare_qsar_aquatic.py
"""

import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import pandas as pd
from sklearn.model_selection import train_test_split

UCI_URL = "https://archive.ics.uci.edu/static/public/505/qsar+aquatic+toxicity.zip"
DATA_NAME = "qsar_aquatic_toxicity.csv"

# The raw file is semicolon-separated with no header; descriptor names by
# position, from the UCI variable description.
FEATURE_COLS = [
    "tpsa",
    "saacc",
    "h050",
    "mlogp",
    "rdchi",
    "gats1p",
    "nn",
    "c040",
]

if not Path(DATA_NAME).exists():
    print(f"Downloading {UCI_URL} ...")
    # The cluster Python has no CA bundle configured; use certifi's
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(UCI_URL, context=ctx) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zf:
            zf.extract(DATA_NAME)

df = pd.read_csv(DATA_NAME, sep=";", header=None)
df.columns = FEATURE_COLS + ["lc50"]

X = df[FEATURE_COLS].values
y = df["lc50"].values

print(f"Shape: {X.shape} | target range: [{y.min()}, {y.max()}]")  # (546, 8)

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
        f"qsar_aquatic_{seed}.csv", index=False
    )
    print(f"Seed {seed}: {len(y_train)} train, {len(y_test)} test")
