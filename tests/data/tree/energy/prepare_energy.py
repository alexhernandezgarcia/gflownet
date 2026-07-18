"""
Generate energy_1.csv through energy_5.csv in DT-GFN CSV format from the UCI
Energy Efficiency dataset (768 samples, 8 features; regression target: heating
load Y1; the second target Y2, cooling load, is dropped).

Downloads ENB2012_data.xlsx from the UCI repository on first run (requires
openpyxl to read the xlsx).

Run from: tests/data/tree/energy/
    python prepare_energy.py
"""

import io
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import pandas as pd
from sklearn.model_selection import train_test_split

UCI_URL = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"
XLSX_NAME = "ENB2012_data.xlsx"

FEATURE_NAMES = {
    "X1": "relative_compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "overall_height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_area_distribution",
}

if not Path(XLSX_NAME).exists():
    print(f"Downloading {UCI_URL} ...")
    # The cluster Python has no CA bundle configured; use certifi's
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(UCI_URL, context=ctx) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as zf:
            zf.extract(XLSX_NAME)

df = pd.read_excel(XLSX_NAME)
# Drop empty rows/columns sometimes present in the sheet
df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
df = df.rename(columns=FEATURE_NAMES)

feature_cols = list(FEATURE_NAMES.values())
X = df[feature_cols].values
y = df["Y1"].values  # heating load

print(f"Shape: {X.shape} | target range: [{y.min()}, {y.max()}]")  # (768, 8)

for seed in [1, 2, 3, 4, 5]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    df_train = pd.DataFrame(X_train, columns=feature_cols)
    df_train["target"] = y_train
    df_train["Split"] = "train"

    df_test = pd.DataFrame(X_test, columns=feature_cols)
    df_test["target"] = y_test
    df_test["Split"] = "test"

    pd.concat([df_train, df_test], ignore_index=True).to_csv(
        f"energy_{seed}.csv", index=False
    )
    print(f"Seed {seed}: {len(y_train)} train, {len(y_test)} test")
