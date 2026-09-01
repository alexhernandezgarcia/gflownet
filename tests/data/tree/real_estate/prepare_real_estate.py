"""
Generate real_estate_1.csv through real_estate_5.csv in DT-GFN CSV format
from the UCI Real Estate Valuation dataset (Sindian district, New Taipei
City; 414 samples, 6 features; regression target: house price of unit area
in 10000 New Taiwan Dollar / ping).

Downloads the xlsx from the UCI repository on first run (requires openpyxl).

Run from: tests/data/tree/real_estate/
    python prepare_real_estate.py
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
    "https://archive.ics.uci.edu/static/public/477/real+estate+valuation+data+set.zip"
)
XLSX_NAME = "Real estate valuation data set.xlsx"

# Original headers are verbose ("X3 distance to the nearest MRT station"
# etc.); rename by position (after dropping the "No" index column).
FEATURE_COLS = [
    "transaction_date",
    "house_age",
    "dist_to_mrt",
    "n_convenience_stores",
    "latitude",
    "longitude",
]

if not Path(XLSX_NAME).exists():
    print(f"Downloading {UCI_URL} ...")
    # The cluster Python has no CA bundle configured; use certifi's
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(UCI_URL, context=ctx) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zf:
            zf.extract(XLSX_NAME)

df = pd.read_excel(XLSX_NAME)
df = df.drop(columns=["No"])
df.columns = FEATURE_COLS + ["price"]

X = df[FEATURE_COLS].values
y = df["price"].values

print(f"Shape: {X.shape} | target range: [{y.min()}, {y.max()}]")  # (414, 6)

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
        f"real_estate_{seed}.csv", index=False
    )
    print(f"Seed {seed}: {len(y_train)} train, {len(y_test)} test")
