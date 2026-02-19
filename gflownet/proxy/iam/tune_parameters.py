"""
Tune GFN environment and reward parameters based on regional/temporal context.

Computes:
  1. Amount values (HIGH/MEDIUM/LOW/NONE) from the local subsidy distribution
     in maxmin-scaled [0,1] space (matching what states2proxy feeds the proxy)
  2. Sigmoid reward parameters calibrated to the distribution of 5-year
     consumption deltas in original units (matching FAIRY proxy output)

The raw parquet files contain UNSCALED values. The witch_proc_data Dataset
applies maxmin scaling at load time. We replicate that scaling here for
subsidies so amount values are in the correct space.

For the sigmoid, the FAIRY proxy output is delta_consumption in ORIGINAL units
(it rescales internally via addcmul), so sigmoid targets use raw deltas.

Usage (from gflownet/proxy/iam/):
    python tune_parameters.py --region europe --year 2025
    python tune_parameters.py --region europe --year 2025 --year_window 1 --margin 0.3
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_raw_data(data_dir):
    """Load the raw (unscaled) parquet files."""
    subsidies_df = pd.read_parquet(os.path.join(data_dir, "subsidies_df.parquet"))
    variables_df = pd.read_parquet(os.path.join(data_dir, "variables_df.parquet")).fillna(0)
    keys_df = pd.read_parquet(os.path.join(data_dir, "keys_df.parquet"))
    keys_df["year"] = keys_df["year"].astype(int)
    return subsidies_df, variables_df, keys_df


def maxmin_scale_subsidies(subsidies_df):
    """
    Apply the same global maxmin scaling that witch_proc_data applies,
    so subsidy statistics are in the [0,1] space the proxy expects.
    """
    scaling_params = {}
    scaled = subsidies_df.copy()
    for col in scaled.columns:
        col_min = float(scaled[col].min())
        col_max = float(scaled[col].max())
        scaling_params[col] = {"min": col_min, "max": col_max}
        denom = col_max - col_min
        if denom > 0:
            scaled[col] = (scaled[col] - col_min) / denom
        else:
            scaled[col] = 0.0
    return scaled, scaling_params


# ---------------------------------------------------------------------------
# 2. Subset selection
# ---------------------------------------------------------------------------

def get_subset_mask(keys_df, region, center_year, year_window):
    """Boolean mask for rows matching the target region and year range."""
    year_lo = center_year - year_window * 5
    year_hi = center_year + year_window * 5
    mask = (keys_df["n"] == region) & (keys_df["year"] >= year_lo) & (keys_df["year"] <= year_hi)
    print(f"Subset: region={region}, years=[{year_lo}, {year_hi}]")
    print(f"  Rows in subset: {mask.sum()} / {len(keys_df)}")
    return mask


# ---------------------------------------------------------------------------
# 3. Compute 5-year consumption deltas (original units)
# ---------------------------------------------------------------------------

def compute_consumption_deltas(variables_df, keys_df, mask):
    """
    Compute the distribution of 5-year consumption deltas for the filtered subset.
    This matches what the FAIRY proxy outputs: consumption(t+5) - consumption(t)
    in ORIGINAL (unscaled) units.
    """
    index_map = {
        (keys_df.loc[i, "gdx"], int(keys_df.loc[i, "year"]), keys_df.loc[i, "n"]): i
        for i in keys_df.index
    }

    subset_indices = keys_df.index[mask].tolist()
    deltas = []
    baseline_values = []

    for idx in subset_indices:
        row = keys_df.loc[idx]
        gdx, year, region = row["gdx"], int(row["year"]), row["n"]

        next_idx = index_map.get((gdx, year + 5, region))
        if next_idx is None:
            continue

        cons_current = float(variables_df.loc[idx, "CONSUMPTION"])
        cons_next = float(variables_df.loc[next_idx, "CONSUMPTION"])
        deltas.append(cons_next - cons_current)
        baseline_values.append(cons_current)

    if len(deltas) == 0:
        return None

    deltas = np.array(deltas)
    baseline_values = np.array(baseline_values)

    stats = {
        "n_pairs": len(deltas),
        "min": float(np.min(deltas)),
        "max": float(np.max(deltas)),
        "mean": float(np.mean(deltas)),
        "std": float(np.std(deltas)),
        "q05": float(np.percentile(deltas, 5)),
        "q10": float(np.percentile(deltas, 10)),
        "q25": float(np.percentile(deltas, 25)),
        "q50": float(np.percentile(deltas, 50)),
        "q75": float(np.percentile(deltas, 75)),
        "q90": float(np.percentile(deltas, 90)),
        "q95": float(np.percentile(deltas, 95)),
        "baseline_mean": float(np.mean(baseline_values)),
        "baseline_median": float(np.median(baseline_values)),
    }
    return stats


# ---------------------------------------------------------------------------
# 4. Compute subsidy statistics (maxmin-scaled)
# ---------------------------------------------------------------------------

def compute_subsidy_stats(subsidies_scaled, mask):
    """Per-technology subsidy stats on the filtered subset (maxmin-scaled)."""
    sub = subsidies_scaled.loc[mask]
    desc = sub.describe().T[["min", "max", "mean", "std"]]
    return desc


# ---------------------------------------------------------------------------
# 5. Derive amount mappings
# ---------------------------------------------------------------------------

def suggest_amount_values(subsidies_scaled, mask, margin=0.2):
    """
    Suggest HIGH/MEDIUM/LOW/NONE from the actual distribution of non-zero
    subsidies in the subset (maxmin-scaled [0,1] space).

    - HIGH   = 90th percentile of non-zero subsidies (+ margin)
    - MEDIUM = median of non-zero subsidies
    - LOW    = 25th percentile of non-zero subsidies
    - NONE   = 0.0

    This reflects realistic investment levels for the region/period,
    excluding the dominant zeros that would otherwise skew everything.

    Parameters
    ----------
    subsidies_scaled : pd.DataFrame
        Maxmin-scaled subsidies (full dataset).
    mask : pd.Series
        Boolean mask for the relevant subset.
    margin : float
        Fraction to extend HIGH beyond the 90th percentile.
    """
    sub = subsidies_scaled.loc[mask]

    # Flatten all subsidy values in the subset and exclude zeros
    all_vals = sub.values.flatten()
    nonzero_vals = all_vals[all_vals > 1e-9]

    if len(nonzero_vals) == 0:
        print("  WARNING: All subsidies are zero in the subset. Using fallback.")
        return {"HIGH": 0.75, "MEDIUM": 0.3, "LOW": 0.1, "NONE": 0.0}, {
            "warning": "all zeros", "n_nonzero": 0}

    p25 = float(np.percentile(nonzero_vals, 25))
    p50 = float(np.percentile(nonzero_vals, 50))
    p75 = float(np.percentile(nonzero_vals, 75))
    p90 = float(np.percentile(nonzero_vals, 90))
    p_max = float(np.max(nonzero_vals))

    high_val = min(1.0, p90 * (1.0 + margin))

    amounts = {
        "HIGH": round(high_val, 4),
        "MEDIUM": round(p50, 4),
        "LOW": round(p25, 4),
        "NONE": 0.0,
    }

    # Per-tech breakdown for context
    per_tech_stats = sub.describe().T[["min", "max", "mean", "std"]]

    info = {
        "n_nonzero": int(len(nonzero_vals)),
        "n_total": int(len(all_vals)),
        "frac_nonzero": round(len(nonzero_vals) / len(all_vals), 4),
        "nonzero_p25": round(p25, 4),
        "nonzero_p50 (MEDIUM)": round(p50, 4),
        "nonzero_p75": round(p75, 4),
        "nonzero_p90": round(p90, 4),
        "nonzero_max": round(p_max, 4),
        "margin": margin,
        "HIGH (p90+margin)": round(high_val, 4),
    }

    return amounts, info, per_tech_stats


# ---------------------------------------------------------------------------
# 6. Derive sigmoid reward parameters from delta distribution
# ---------------------------------------------------------------------------

def suggest_sigmoid_params(delta_stats, margin=0.2):
    """
    Fit sigmoid to the 5-year consumption delta distribution (original units).

    Calibration:
      R(q75)  = 0.50   (center)
      R(q95)  ≈ 0.99   (flattens to 1)

    R(x) = alpha / (1 + exp(-gamma * (x + beta)))

    Solving:
      beta  = -q75
      gamma = ln(99) / (q95 - q75)
    """
    x_mid = delta_stats["q75"]
    x_high = delta_stats["q95"]

    spread = x_high - x_mid

    print(f"\n  Sigmoid fitting targets (5-year Δconsumption, original units):")
    print(f"    x_mid  (q75) = {x_mid:.6f}  → R = 0.50")
    print(f"    x_high (q95) = {x_high:.6f} → R ≈ 0.99")
    print(f"    spread (q95-q75) = {spread:.6f}")

    if abs(spread) < 1e-10:
        print("  WARNING: q95 ≈ q75, using fallback spread.")
        spread = (delta_stats["max"] - delta_stats["q50"]) * 0.25

    beta = -x_mid
    gamma = 4.595 / spread  # ln(99) / (q95 - q75)
    alpha = 1.0

    def R(x):
        return alpha / (1.0 + np.exp(-gamma * (x + beta)))

    params = {
        "gamma": round(float(gamma), 4),
        "beta": round(float(beta), 6),
        "alpha": float(alpha),
    }

    diagnostics = {}
    for q in ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]:
        diagnostics[f"R({q})"] = round(R(delta_stats[q]), 4)

    return params, diagnostics


# ---------------------------------------------------------------------------
# 7. Summary & config generation
# ---------------------------------------------------------------------------

def print_summary(delta_stats, subsidy_stats, amounts, amount_info,
                  sigmoid_params, sigmoid_diag, round_decimals=1):
    print("\n" + "=" * 70)
    print("TUNING RESULTS")
    print("=" * 70)

    print("\n--- 5-Year Consumption Delta Distribution (original units) ---")
    for k, v in delta_stats.items():
        print(f"  {k:>20s}: {v}")

    print("\n--- Local Subsidy Distribution (maxmin-scaled [0,1]) ---")
    print(f"  Tech-level min range: [{subsidy_stats['min'].min():.4f}, {subsidy_stats['min'].max():.4f}]")
    print(f"  Tech-level max range: [{subsidy_stats['max'].min():.4f}, {subsidy_stats['max'].max():.4f}]")
    top_techs = subsidy_stats.nlargest(5, "max")[["max", "mean"]]
    print(f"  Top 5 techs by max subsidy (scaled):")
    for tech_name, row in top_techs.iterrows():
        print(f"    {tech_name}: max={row['max']:.4f}, mean={row['mean']:.4f}")

    print(f"\n--- Suggested Amount Values (maxmin-scaled, for states2proxy) ---")
    print(f"  Non-zero subsidy distribution:")
    for k, v in amount_info.items():
        print(f"    {k}: {v}")
    for level, val in amounts.items():
        print(f"  {level:>8s}: {val}")

    print(f"\n--- Suggested Sigmoid Reward Parameters ---")
    for k, v in sigmoid_params.items():
        print(f"  {k}: {v}")
    print(f"  Verification (exact params):")
    for k, v in sigmoid_diag.items():
        print(f"    {k}: {v}")

    # Rounded version
    gamma_r = round(sigmoid_params["gamma"], round_decimals)
    beta_r = round(sigmoid_params["beta"], round_decimals)
    alpha = sigmoid_params["alpha"]
    print(f"\n--- Reward at delta quantiles (rounded: gamma={gamma_r}, beta={beta_r}) ---")
    for q in ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]:
        dx = delta_stats[q]
        rv = alpha / (1.0 + np.exp(-gamma_r * (dx + beta_r)))
        print(f"  R(Δcons={dx:+.6f}) at {q} = {rv:.4f}")

    print("\n--- Suggested YAML config overrides ---")
    print(f"""
env:
  amount_values_mapping: [0.0, {amounts['HIGH']}, {amounts['MEDIUM']}, {amounts['LOW']}, {amounts['NONE']}]

proxy:
  reward_function: sigmoid
  reward_function_kwargs:
    gamma: {sigmoid_params['gamma']}
    beta: {sigmoid_params['beta']}
    alpha: {sigmoid_params['alpha']}
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tune GFN parameters from local data distribution")
    parser.add_argument("--region", type=str, default="europe")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--year_window", type=int, default=0,
                        help="5-year steps on each side (default: 0 = exact year only)")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Fraction to extend beyond observed range (default: 0.2)")
    parser.add_argument("--data_dir", type=str, default="scenario_data")
    parser.add_argument("--round_decimals", type=int, default=1,
                        help="Decimals to round gamma/beta in quantile printout")

    args = parser.parse_args()

    print("Loading data...")
    subsidies_df, variables_df, keys_df = load_raw_data(args.data_dir)

    print("Applying maxmin scaling to subsidies (global, like witch_proc_data)...")
    subsidies_scaled, scaling_params = maxmin_scale_subsidies(subsidies_df)

    print("Filtering subset...")
    mask = get_subset_mask(keys_df, args.region, args.year, args.year_window)

    if mask.sum() == 0:
        print("ERROR: No data found.")
        print(f"  Available regions: {sorted(keys_df['n'].unique())}")
        print(f"  Available years: {sorted(keys_df['year'].unique())}")
        sys.exit(1)

    print("Computing 5-year consumption deltas (original units)...")
    delta_stats = compute_consumption_deltas(variables_df, keys_df, mask)
    if delta_stats is None:
        print("ERROR: No valid (t, t+5) pairs. Try --year_window 1 or check year.")
        sys.exit(1)

    print("Computing subsidy statistics (maxmin-scaled)...")
    sub_stats = compute_subsidy_stats(subsidies_scaled, mask)

    print("Suggesting amount values...")
    amounts, amount_info, sub_stats = suggest_amount_values(
        subsidies_scaled, mask, margin=args.margin)

    print("Suggesting sigmoid parameters...")
    sigmoid_params, sigmoid_diag = suggest_sigmoid_params(delta_stats, margin=args.margin)

    print_summary(delta_stats, sub_stats, amounts, amount_info,
                  sigmoid_params, sigmoid_diag, round_decimals=args.round_decimals)


if __name__ == "__main__":
    main()