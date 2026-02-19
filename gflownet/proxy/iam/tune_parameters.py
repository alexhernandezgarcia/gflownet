"""
Tune GFN environment and reward parameters based on regional/temporal context.

Instead of scaling against the full global dataset (all regions, all years up to 2100),
this script computes parameter suggestions from a relevant subset: the target region
and adjacent time periods, plus a configurable margin.

Usage:
    python tune_parameters.py [--region europe] [--year 2010] [--year_window 2] [--margin 0.2]
"""

import argparse
import pickle
import os
import sys

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_raw_data(
    subsidies_path="gflownet/proxy/iam/scenario_data/subsidies_df.parquet",
    variables_path="gflownet/proxy/iam/scenario_data/variables_df.parquet",
    keys_path="gflownet/proxy/iam/scenario_data/keys_df.parquet",
):
    """Load the raw (unscaled) parquet files."""
    subsidies_df = pd.read_parquet(subsidies_path)
    variables_df = pd.read_parquet(variables_path).fillna(0)
    keys_df = pd.read_parquet(keys_path)
    keys_df["year"] = keys_df["year"].astype(int)
    return subsidies_df, variables_df, keys_df


# ---------------------------------------------------------------------------
# 2. Subset selection
# ---------------------------------------------------------------------------

def get_subset_mask(keys_df, region, center_year, year_window):
    """
    Boolean mask for rows matching the target region and year range.

    Parameters
    ----------
    keys_df : pd.DataFrame
        Must have columns 'n' (region) and 'year'.
    region : str
        Target region (e.g. 'europe').
    center_year : int
        Central year of the context.
    year_window : int
        Number of 5-year steps on each side.  E.g. window=2 with center=2010
        gives [2000, 2005, 2010, 2015, 2020].
    """
    year_lo = center_year - year_window * 5
    year_hi = center_year + year_window * 5
    mask = (keys_df["n"] == region) & (keys_df["year"] >= year_lo) & (keys_df["year"] <= year_hi)
    print(f"Subset: region={region}, years=[{year_lo}, {year_hi}]")
    print(f"  Rows in subset: {mask.sum()} / {len(keys_df)}")
    return mask


# ---------------------------------------------------------------------------
# 3. Compute consumption statistics on the subset
# ---------------------------------------------------------------------------

def compute_consumption_stats(variables_df, mask):
    """
    Compute raw consumption statistics on the filtered subset.

    Returns dict with min, max, mean, std, and quantiles.
    """
    cons = variables_df.loc[mask, "CONSUMPTION"].values
    stats = {
        "min": float(np.min(cons)),
        "max": float(np.max(cons)),
        "mean": float(np.mean(cons)),
        "std": float(np.std(cons)),
        "q05": float(np.percentile(cons, 5)),
        "q10": float(np.percentile(cons, 10)),
        "q25": float(np.percentile(cons, 25)),
        "q50": float(np.percentile(cons, 50)),
        "q75": float(np.percentile(cons, 75)),
        "q90": float(np.percentile(cons, 90)),
        "q95": float(np.percentile(cons, 95)),
        "n": int(mask.sum()),
    }
    return stats


def compute_subsidy_stats(subsidies_df, mask):
    """
    Compute per-technology subsidy statistics on the filtered subset.
    Returns a DataFrame with min/max/mean/std per subsidy column.
    """
    sub = subsidies_df.loc[mask]
    desc = sub.describe().T[["min", "max", "mean", "std"]]
    return desc


# ---------------------------------------------------------------------------
# 4. Derive amount mappings
# ---------------------------------------------------------------------------

def suggest_amount_values(subsidy_stats, margin=0.2):
    """
    Suggest HIGH / MEDIUM / LOW / NONE amount values based on the local
    subsidy distribution, rather than global [0, 1] maxmin scaling.

    The idea: map amounts to fractions of the *local* range, with margin
    for exploration beyond observed values.

    Parameters
    ----------
    subsidy_stats : pd.DataFrame
        Per-tech stats with 'min' and 'max' columns.
    margin : float
        Fraction to extend beyond observed local range on each side.

    Returns
    -------
    dict with 'HIGH', 'MEDIUM', 'LOW', 'NONE' values (as fractions of
    the global [0,1] maxmin range), plus the local range info.
    """
    # Aggregate across techs: what fraction of the global range do local values span?
    # Since the data is maxmin-scaled to [0,1], the raw parquet values are in original units.
    # But the proxy operates on maxmin-scaled inputs, so we reason in scaled space.
    # The local subset spans some sub-range of [0, 1] per tech.

    # For a region-aware mapping, we compute the typical local max across techs
    local_maxes = subsidy_stats["max"].values
    local_mins = subsidy_stats["min"].values

    # Median of per-tech local max/min gives a robust central estimate
    typical_local_max = float(np.median(local_maxes))
    typical_local_min = float(np.median(local_mins))
    local_range = typical_local_max - typical_local_min

    # Extend by margin
    effective_max = min(1.0, typical_local_max + margin * local_range)
    effective_min = max(0.0, typical_local_min - margin * local_range)
    effective_range = effective_max - effective_min

    amounts = {
        "HIGH": round(effective_min + 0.90 * effective_range, 4),
        "MEDIUM": round(effective_min + 0.50 * effective_range, 4),
        "LOW": round(effective_min + 0.15 * effective_range, 4),
        "NONE": 0.0,
    }

    info = {
        "typical_local_min": round(typical_local_min, 4),
        "typical_local_max": round(typical_local_max, 4),
        "margin": margin,
        "effective_min": round(effective_min, 4),
        "effective_max": round(effective_max, 4),
    }

    return amounts, info


# ---------------------------------------------------------------------------
# 5. Derive sigmoid reward parameters
# ---------------------------------------------------------------------------

def suggest_sigmoid_params(cons_stats, margin=0.2):
    """
    Fit sigmoid reward parameters so that:
      R(local_median)  ≈ 0.01   (typical outcome is low reward)
      R(local_q75)     ≈ 0.50   (good outcome gets moderate reward)
      R(local_q90)     ≈ 0.99   (excellent outcome gets near-max reward)

    The reward function is:  R(x) = alpha / (1 + exp(-gamma * (x + beta)))

    Since the proxy returns delta-consumption (x = cons_predicted - cons_current),
    we need to reason in delta space.  The context row is the baseline (year=2010,
    region=europe), so cons_current ≈ local median.  Deltas are then relative to that.

    We compute deltas as offsets from the local median of raw consumption, then
    apply the global maxmin scaling that the proxy uses internally.

    Parameters
    ----------
    cons_stats : dict
        Output of compute_consumption_stats (raw values).
    margin : float
        Extend the range by this fraction for robustness.

    Returns
    -------
    dict with gamma, beta, alpha and diagnostic info.
    """
    # The proxy outputs: y = (scaled_cons - cons_current_scaled) * cons_scale + ...
    # Simplification: we work directly with the raw consumption deltas and
    # then convert.  The proxy's final output is in *unscaled* consumption units
    # (it does addcmul to rescale), so the reward sigmoid operates on those.

    # Local deltas from the baseline (≈ median)
    baseline = cons_stats["q50"]
    local_range = cons_stats["q95"] - cons_stats["q05"]

    # Target points for the sigmoid (in delta-consumption space)
    # We want plans that improve consumption relative to baseline
    x_low = 0.0                                          # no improvement → low reward
    x_mid = (cons_stats["q75"] - baseline)               # good improvement
    x_high = (cons_stats["q90"] - baseline)              # great improvement

    # Add margin: extend x_high a bit
    x_high_ext = x_high * (1.0 + margin)

    print(f"\n  Sigmoid fitting targets (delta-consumption from baseline={baseline:.4f}):")
    print(f"    x_low  = {x_low:.6f}  → R ≈ 0.01")
    print(f"    x_mid  = {x_mid:.6f}  → R ≈ 0.50")
    print(f"    x_high = {x_high:.6f} → R ≈ 0.99")

    # Sigmoid: R(x) = alpha / (1 + exp(-gamma * (x + beta)))
    # Setting alpha=1 and solving:
    #   At x_mid:  0.5 = 1/(1+exp(-gamma*(x_mid+beta)))  →  gamma*(x_mid+beta) = 0  →  beta = -x_mid
    #   At x_low:  0.01 = 1/(1+exp(-gamma*(x_low+beta)))
    #              → gamma*(x_low - x_mid) = ln(0.01/0.99) ≈ -4.595
    #              → gamma = -4.595 / (x_low - x_mid) = 4.595 / x_mid

    if abs(x_mid) < 1e-10:
        print("  WARNING: x_mid ≈ 0, local q75 ≈ median. Using fallback based on local range.")
        x_mid = local_range * 0.1  # fallback

    beta = -x_mid
    gamma = 4.595 / x_mid  # from ln(99) ≈ 4.595

    # Verify at x_high
    r_at_high = 1.0 / (1.0 + np.exp(-gamma * (x_high + beta)))

    alpha = 1.0

    params = {
        "gamma": round(float(gamma), 4),
        "beta": round(float(beta), 6),
        "alpha": float(alpha),
    }

    diagnostics = {
        "baseline_consumption": round(baseline, 4),
        "x_low (delta)": round(x_low, 6),
        "x_mid (delta, q75-baseline)": round(x_mid, 6),
        "x_high (delta, q90-baseline)": round(x_high, 6),
        "R(x_low)": round(1.0 / (1.0 + np.exp(-gamma * (x_low + beta))), 4),
        "R(x_mid)": round(1.0 / (1.0 + np.exp(-gamma * (x_mid + beta))), 4),
        "R(x_high)": round(float(r_at_high), 4),
    }

    return params, diagnostics


# ---------------------------------------------------------------------------
# 6. Summary & config generation
# ---------------------------------------------------------------------------

def print_summary(cons_stats, subsidy_stats, amounts, amount_info, sigmoid_params, sigmoid_diag, round_decimals=1):
    print("\n" + "=" * 70)
    print("TUNING RESULTS")
    print("=" * 70)

    print("\n--- Local Consumption Distribution ---")
    for k, v in cons_stats.items():
        print(f"  {k:>6s}: {v}")

    print("\n--- Local Subsidy Distribution (summary across techs) ---")
    print(f"  Tech-level min range: [{subsidy_stats['min'].min():.4f}, {subsidy_stats['min'].max():.4f}]")
    print(f"  Tech-level max range: [{subsidy_stats['max'].min():.4f}, {subsidy_stats['max'].max():.4f}]")

    print(f"\n--- Suggested Amount Values (states2proxy mapping) ---")
    print(f"  Range info: {amount_info}")
    for level, val in amounts.items():
        print(f"  {level:>8s}: {val}")

    print(f"\n--- Suggested Sigmoid Reward Parameters ---")
    for k, v in sigmoid_params.items():
        print(f"  {k}: {v}")
    print(f"  Diagnostics:")
    for k, v in sigmoid_diag.items():
        print(f"    {k}: {v}")

    gamma_r = round(sigmoid_params["gamma"], round_decimals)
    beta_r = round(sigmoid_params["beta"], round_decimals)
    alpha = sigmoid_params["alpha"]
    baseline = cons_stats["q50"]
    print(f"\n--- Reward at local quantiles (gamma={gamma_r}, beta={beta_r}) ---")
    for q in ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]:
        dx = cons_stats[q] - baseline
        rv = alpha / (1.0 + np.exp(-gamma_r * (dx + beta_r)))
        print(f"  R(Δcons at {q}) = R({dx:+.6f}) = {rv:.4f}")

    print("\n--- Suggested YAML config overrides ---")
    print(f"""
# In states2proxy amount_idx_to_value tensor, replace:
#   [0.0, 0.75, 0.3, 0.1, 0.0]  (global)
# with:
#   [0.0, {amounts['HIGH']}, {amounts['MEDIUM']}, {amounts['LOW']}, {amounts['NONE']}]
# (idx: 0=unset, 1=HIGH, 2=MEDIUM, 3=LOW, 4=NONE)

# Reward function (config YAML):
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
    parser = argparse.ArgumentParser(description="Tune GFN parameters from local data distribution")
    parser.add_argument("--region", type=str, default="europe",
                        help="Target region (default: europe)")
    parser.add_argument("--year", type=int, default=2010,
                        help="Center year (default: 2010)")
    parser.add_argument("--year_window", type=int, default=2,
                        help="Number of 5-year steps on each side (default: 2 → ±10 years)")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Fraction to extend beyond observed range (default: 0.2)")
    parser.add_argument("--data_dir", type=str, default="scenario_data",
                        help="Path to scenario_data directory (default assumes running from gflownet/proxy/iam/)")
    parser.add_argument("--round_decimals", type=int, default=1,
                        help="Number of decimals to round gamma/beta to in the quantile printout (default: 1)")

    args = parser.parse_args()

    # Paths
    sub_path = os.path.join(args.data_dir, "subsidies_df.parquet")
    var_path = os.path.join(args.data_dir, "variables_df.parquet")
    key_path = os.path.join(args.data_dir, "keys_df.parquet")

    print("Loading data...")
    subsidies_df, variables_df, keys_df = load_raw_data(sub_path, var_path, key_path)

    print("Filtering subset...")
    mask = get_subset_mask(keys_df, args.region, args.year, args.year_window)

    if mask.sum() == 0:
        print("ERROR: No data found for the specified region/year range.")
        print(f"  Available regions: {sorted(keys_df['n'].unique())}")
        print(f"  Available years: {sorted(keys_df['year'].unique())}")
        sys.exit(1)

    print("Computing consumption statistics...")
    cons_stats = compute_consumption_stats(variables_df, mask)

    print("Computing subsidy statistics...")
    sub_stats = compute_subsidy_stats(subsidies_df, mask)

    print("Suggesting amount values...")
    amounts, amount_info = suggest_amount_values(sub_stats, margin=args.margin)

    print("Suggesting sigmoid parameters...")
    sigmoid_params, sigmoid_diag = suggest_sigmoid_params(cons_stats, margin=args.margin)

    print_summary(cons_stats, sub_stats, amounts, amount_info, sigmoid_params, sigmoid_diag,
                  round_decimals=args.round_decimals)


if __name__ == "__main__":
    main()