"""
Generate a region-aware test set for the GFN investment plan environment.

Half the samples come from real scenario subsidies (rounded to nearest discrete
amount level), half are uniformly random investment plans.

By default (no --n), the total size is 2x the number of scenario rows in the
filtered subset (so the data-derived half uses every row once without replacement,
and the random half matches its size).

Output is saved to gflownet/proxy/iam/ by default (not tracked by git).

Usage (from gflownet/proxy/iam/):
    python generate_test_set.py --region europe --year 2010
    python generate_test_set.py --region europe --year 2010 --n 20000
    python generate_test_set.py --region usa --year 2025 --amount_values 0.6,0.3,0.1,0.0
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

# Import TECHS/AMOUNTS ordering from the environment
try:
    from gflownet.envs.iam.full_plan import TECHS, AMOUNTS
except ImportError:
    TECHS = (
        "power_COAL_noccs", "power_COAL_ccs", "power_NUCLEAR", "power_OIL",
        "power_GAS_noccs", "power_GAS_ccs", "power_HYDRO", "power_BIOMASS_noccs",
        "power_BIOMASS_ccs", "power_WIND_onshore", "power_WIND_offshore", "power_SOLAR",
        "thermal_SOLAR", "enduse_COAL_ccs", "power_STORAGE", "production_HYDROGEN",
        "refueling_station_HYDROGEN", "pipelines_HYDROGEN", "DAC_liquid_sorbents",
        "DAC_solid_sorbents", "DAC_calcium_oxide", "CARS_trad", "CARS_hybrid",
        "CARS_electric", "CARS_fuelcell", "HEAVYDUTY_trad", "HEAVYDUTY_hybrid",
        "HEAVYDUTY_electric", "HEAVYDUTY_fuelcell",
    )
    AMOUNTS = ("HIGH", "MEDIUM", "LOW", "NONE")

# Default amount values (global scaling) — override with --amount_values
DEFAULT_AMOUNT_VALUES = {"HIGH": 0.75, "MEDIUM": 0.3, "LOW": 0.1, "NONE": 0.0}


def load_raw_data(data_dir):
    subsidies_df = pd.read_parquet(os.path.join(data_dir, "subsidies_df.parquet"))
    keys_df = pd.read_parquet(os.path.join(data_dir, "keys_df.parquet"))
    keys_df["year"] = keys_df["year"].astype(int)
    return subsidies_df, keys_df


def get_subset_mask(keys_df, region, center_year, year_window):
    year_lo = center_year - year_window * 5
    year_hi = center_year + year_window * 5
    mask = (keys_df["n"] == region) & (keys_df["year"] >= year_lo) & (keys_df["year"] <= year_hi)
    print(f"Subset: region={region}, years=[{year_lo}, {year_hi}]")
    print(f"  Rows in subset: {mask.sum()} / {len(keys_df)}")
    return mask


def round_to_nearest_amount(value, amount_values):
    """Round a continuous subsidy value to the nearest discrete amount label."""
    best_label = None
    best_dist = float("inf")
    for label, level_val in amount_values.items():
        dist = abs(value - level_val)
        if dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def generate_test_set(subsidies_df, keys_df, mask, amount_values, n_total, seed=42):
    """
    Generate a test set of investment plans.

    Returns
    -------
    list of state dicts, each with 'partial' and 'plan' keys.
    """
    rng = np.random.RandomState(seed)
    n_data = n_total // 2
    n_random = n_total - n_data

    amount_labels = list(AMOUNTS)
    label2idx = {label: idx + 1 for idx, label in enumerate(amount_labels)}

    subset_indices = keys_df.index[mask].tolist()
    if len(subset_indices) == 0:
        print("  WARNING: No data rows in subset, all test samples will be random.")
        n_data = 0
        n_random = n_total

    # --- Data-derived half ---
    data_plans = []
    if n_data > 0:
        sampled_indices = rng.choice(subset_indices, size=n_data, replace=(n_data > len(subset_indices)))
        missing_warned = False

        for idx in sampled_indices:
            plan = []
            for tech in TECHS:
                col = f"SUBS_{tech}"
                if col in subsidies_df.columns:
                    raw_val = float(subsidies_df.loc[idx, col])
                    nearest_label = round_to_nearest_amount(raw_val, amount_values)
                    plan.append(label2idx[nearest_label])
                else:
                    if not missing_warned:
                        print(f"  WARNING: column {col} not found in subsidies data, defaulting to NONE")
                        missing_warned = True
                    plan.append(label2idx["NONE"])
            data_plans.append(plan)

    # --- Random half ---
    random_plans = []
    for _ in range(n_random):
        plan = [rng.randint(1, len(amount_labels) + 1) for _ in range(len(TECHS))]
        random_plans.append(plan)

    # --- Combine and shuffle ---
    all_plans = data_plans + random_plans
    rng.shuffle(all_plans)

    # Build state dicts matching the environment format
    samples = []
    for plan in all_plans:
        state = {
            "partial": {"SECTOR": 0, "TAG": 0, "TECH": 0, "AMOUNT": 0},
            "plan": plan,
        }
        samples.append(state)

    return samples, n_data, n_random


def main():
    parser = argparse.ArgumentParser(description="Generate region-aware test set for GFN investment plans")
    parser.add_argument("--region", type=str, default="europe",
                        help="Target region (default: europe)")
    parser.add_argument("--year", type=int, default=2010,
                        help="Center year (default: 2010)")
    parser.add_argument("--year_window", type=int, default=0,
                        help="Number of 5-year steps on each side (default: 0)")
    parser.add_argument("--n", type=int, default=None,
                        help="Total test set size. Default: 2x the number of scenario rows in subset")
    parser.add_argument("--amount_values", type=str, default=None,
                        help="Comma-separated HIGH,MEDIUM,LOW,NONE values, e.g. 0.6,0.3,0.1,0.0")
    parser.add_argument("--output", type=str, default=None,
                        help="Output pkl path (default: ./test_{region}_{year}.pkl)")
    parser.add_argument("--data_dir", type=str, default="scenario_data",
                        help="Path to scenario_data directory (default assumes running from gflownet/proxy/iam/)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    # Parse amount values
    if args.amount_values is not None:
        vals = [float(v.strip()) for v in args.amount_values.split(",")]
        if len(vals) != 4:
            print("ERROR: --amount_values must have exactly 4 comma-separated values (HIGH,MEDIUM,LOW,NONE)")
            sys.exit(1)
        amount_values = dict(zip(["HIGH", "MEDIUM", "LOW", "NONE"], vals))
    else:
        amount_values = DEFAULT_AMOUNT_VALUES

    # Output path
    if args.output is None:
        args.output = f"test_{args.region}_{args.year}.pkl"

    print("Loading data...")
    subsidies_df, keys_df = load_raw_data(args.data_dir)

    print("Filtering subset...")
    mask = get_subset_mask(keys_df, args.region, args.year, args.year_window)

    n_subset = int(mask.sum())
    if n_subset == 0:
        print("ERROR: No data found for the specified region/year range.")
        print(f"  Available regions: {sorted(keys_df['n'].unique())}")
        print(f"  Available years: {sorted(keys_df['year'].unique())}")
        sys.exit(1)

    # Determine total size
    if args.n is None:
        n_total = 2 * n_subset
        print(f"  No --n specified, using 2 x {n_subset} scenario rows = {n_total} total")
    else:
        n_total = args.n

    print(f"\nGenerating test set...")
    samples, n_data, n_random = generate_test_set(
        subsidies_df=subsidies_df,
        keys_df=keys_df,
        mask=mask,
        amount_values=amount_values,
        n_total=n_total,
        seed=args.seed,
    )

    # Save as pkl with {"samples": [...]} structure (same as Ising test sets)
    with open(args.output, "wb") as f:
        pickle.dump({"samples": samples}, f)

    print(f"\n{'=' * 50}")
    print(f"Test set saved to: {args.output}")
    print(f"  Data-derived samples: {n_data}")
    print(f"  Random samples:      {n_random}")
    print(f"  Total size:          {len(samples)}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()