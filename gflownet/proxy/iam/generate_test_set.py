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
    python generate_test_set.py --region europe --year 2025 --target_variable EMI_total_CO2
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
    """
    Load raw parquet files and apply zero-block filtering,
    matching the preprocessing in witch_proc_data.
    """
    subsidies_df = pd.read_parquet(os.path.join(data_dir, "subsidies_df.parquet"))
    variables_df = pd.read_parquet(os.path.join(data_dir, "variables_df.parquet")).fillna(0)
    keys_df = pd.read_parquet(os.path.join(data_dir, "keys_df.parquet"))
    keys_df["year"] = keys_df["year"].astype(int)

    # --- Remove blocks containing all-zero variable rows ---
    zero_row_mask = (variables_df == 0).all(axis=1)
    if zero_row_mask.any():
        zero_indices = variables_df.index[zero_row_mask]
        bad_blocks = keys_df.loc[zero_indices, ["gdx", "n"]].drop_duplicates()
        print(
            f"Found {len(zero_indices)} all-zero variable rows in "
            f"{len(bad_blocks)} (gdx, region) blocks. Removing entire blocks."
        )
        block_keys = set(zip(bad_blocks["gdx"], bad_blocks["n"]))
        rows_to_drop = keys_df.apply(
            lambda r: (r["gdx"], r["n"]) in block_keys, axis=1
        )
        keep_mask = ~rows_to_drop

        variables_df = variables_df.loc[keep_mask].reset_index(drop=True)
        subsidies_df = subsidies_df.loc[keep_mask].reset_index(drop=True)
        keys_df = keys_df.loc[keep_mask].reset_index(drop=True)

        print(f"  Rows remaining after block removal: {len(keys_df)}")
    # --- End block removal ---

    return subsidies_df, keys_df


def get_subset_mask(keys_df, region, center_year, year_window):
    year_lo = center_year - year_window * 5
    year_hi = center_year + year_window * 5
    mask = (keys_df["n"] == region) & (keys_df["year"] >= year_lo) & (keys_df["year"] <= year_hi)
    print(f"Subset: region={region}, years=[{year_lo}, {year_hi}]")
    print(f"  Rows in subset: {mask.sum()} / {len(keys_df)}")
    return mask


def _resolve_tech_amounts(amount_values, tech):
    """
    Return a flat {label: float} dict for a specific tech.

    Global mode  — amount_values is already {HIGH: float, MEDIUM: float, ...}
    Per-tech mode — amount_values is {tech_name: [v0, v1, v2, v3, v4]}
                    where indices map to [unset, HIGH, MEDIUM, LOW, NONE].
    """
    if isinstance(amount_values, dict) and "HIGH" in amount_values:
        return amount_values  # global mode — same for all techs
    # Per-tech mode
    row = amount_values.get(tech)
    if row is None:
        # Fallback: use lowest HIGH row
        row = min(amount_values.values(), key=lambda r: r[1])
    # row = [v_unset, v_HIGH, v_MEDIUM, v_LOW, v_NONE]
    return {"HIGH": row[1], "MEDIUM": row[2], "LOW": row[3], "NONE": row[4]}


def round_to_nearest_amount(value, amount_values_for_tech):
    """Round a continuous subsidy value to the nearest discrete amount label.

    amount_values_for_tech must be a flat {label: float} dict.
    Use _resolve_tech_amounts() to obtain it from either global or per-tech mappings.
    """
    best_label = None
    best_dist = float("inf")
    for label, level_val in amount_values_for_tech.items():
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
                    tech_amounts = _resolve_tech_amounts(amount_values, tech)
                    nearest_label = round_to_nearest_amount(raw_val, tech_amounts)
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


def build_output_path(output_arg, region, year, target_variable):
    """
    Derive the output path from CLI arg or auto-generate from context.
    Target variable is included in the filename so test sets for different
    variables don't collide.
    """
    if output_arg is not None:
        return output_arg
    # Sanitize and lowercase to match config_writer's _var_to_slug convention
    safe_var = target_variable.lower().replace("/", "_").replace(" ", "_")
    return f"test_{region}_{year}_{safe_var}.pkl"


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
                        help="Output pkl path (default: ./test_{region}_{year}_{target_variable}.pkl)")
    parser.add_argument("--data_dir", type=str, default="scenario_data",
                        help="Path to scenario_data directory (default assumes running from gflownet/proxy/iam/)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--target_variable", type=str, default="CONSUMPTION",
                        help="Target variable name — used only for output filename disambiguation "
                             "(default: CONSUMPTION). Pass the same value you used in tune_parameters.py.")

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

    # Output path — includes target_variable to avoid collisions
    output_path = build_output_path(args.output, args.region, args.year, args.target_variable)

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

    print(f"\nGenerating test set (target_variable={args.target_variable})...")
    samples, n_data, n_random = generate_test_set(
        subsidies_df=subsidies_df,
        keys_df=keys_df,
        mask=mask,
        amount_values=amount_values,
        n_total=n_total,
        seed=args.seed,
    )

    # Save as pkl with {"samples": [...], "target_variable": ...} structure
    with open(output_path, "wb") as f:
        pickle.dump({"samples": samples, "target_variable": args.target_variable}, f)

    print(f"\n{'=' * 50}")
    print(f"Test set saved to: {output_path}")
    print(f"  Target variable:     {args.target_variable}")
    print(f"  Data-derived samples: {n_data}")
    print(f"  Random samples:      {n_random}")
    print(f"  Total size:          {len(samples)}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()