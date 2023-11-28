"""
Compute lattice parameter value ranges for MP20 and Matbench datasets.
"""

from pathlib import Path

import numpy as np
import pandas as pd

if __name__ == "__main__":
    mp = pd.read_csv(Path(__file__).parents[1] / "data/crystals/mp20_lp_stats.csv")
    mb = pd.read_csv(
        Path(__file__).parents[1] / "data/crystals/matbench_mp_e_form_lp_stats.csv"
    )

    for name, df in {"MP20": mp, "Matbench": mb}.items():
        for param in mp.columns:
            print(f"{name}, {param}:")
            print(
                f"\t range = ({np.round(df[param].min(), 2)}, {np.round(df[param].max(), 2)})"
            )
            print(
                f"\t .05-.95 quantile = ({np.round(df[param].quantile(0.05), 2)}, {np.round(df[param].quantile(0.95), 2)})"
            )
            print(
                f"\t .01-.99 quantile = ({np.round(df[param].quantile(0.01), 2)}, {np.round(df[param].quantile(0.99), 2)})"
            )
