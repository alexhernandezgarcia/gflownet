"""
Generate concrete_1.csv through concrete_5.csv in DT-GFN CSV format from the
UCI Concrete Compressive Strength dataset (1030 samples, 8 features;
regression target: compressive strength in MPa).

Downloads Concrete_Data.xls from the UCI repository on first run (requires
xlrd to read the legacy xls).

Run from: tests/data/tree/concrete/
    python prepare_concrete.py
"""

import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import pandas as pd
from sklearn.model_selection import train_test_split

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"
)
XLS_NAME = "Concrete_Data.xls"

# Original headers are verbose ("Cement (component 1)(kg in a m^3 mixture)"
# etc.); rename by position to short names.
FEATURE_COLS = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
]

if not Path(XLS_NAME).exists():
    print(f"Downloading {UCI_URL} ...")
    # The cluster Python has no CA bundle configured; use certifi's
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(UCI_URL, context=ctx) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zf:
            zf.extract(XLS_NAME)

df = pd.read_excel(XLS_NAME)
df.columns = FEATURE_COLS + ["strength"]

X = df[FEATURE_COLS].values
y = df["strength"].values

print(f"Shape: {X.shape} | target range: [{y.min()}, {y.max()}]")  # (1030, 8)

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
        f"concrete_{seed}.csv", index=False
    )
    print(f"Seed {seed}: {len(y_train)} train, {len(y_test)} test")