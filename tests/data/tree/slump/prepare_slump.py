"""
Generate slump_1.csv through slump_5.csv in DT-GFN CSV format from the UCI
Concrete Slump Test dataset (103 samples, 7 mixture features).

The raw data has three output columns (SLUMP, FLOW, 28-day compressive
strength); we keep only the compressive strength as the regression target:
it is fully continuous (17.2-58.5 MPa) and analogous to the `concrete`
dataset, whereas SLUMP and FLOW are censored at their measurement limits
(slump clipped to [0, 29] cm, flow to >= 20 cm).

Downloads slump_test.data from the UCI repository on first run.

Run from: tests/data/tree/slump/
    python prepare_slump.py
"""

import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import pandas as pd
from sklearn.model_selection import train_test_split

UCI_URL = "https://archive.ics.uci.edu/static/public/182/concrete+slump+test.zip"
DATA_NAME = "slump_test.data"

# Original headers are "Cement", "Coarse Aggr." etc.; rename by position
# (after dropping the "No" index column and the SLUMP/FLOW outputs).
FEATURE_COLS = [
    "cement",
    "slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
]

if not Path(DATA_NAME).exists():
    print(f"Downloading {UCI_URL} ...")
    # The cluster Python has no CA bundle configured; use certifi's
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(UCI_URL, context=ctx) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zf:
            zf.extract(DATA_NAME)

df = pd.read_csv(DATA_NAME)
df = df.drop(columns=["No", "SLUMP(cm)", "FLOW(cm)"])
df.columns = FEATURE_COLS + ["strength"]

X = df[FEATURE_COLS].values
y = df["strength"].values

print(f"Shape: {X.shape} | target range: [{y.min()}, {y.max()}]")  # (103, 7)

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
        f"slump_{seed}.csv", index=False
    )
    print(f"Seed {seed}: {len(y_train)} train, {len(y_test)} test")
