"""
Tune GFN environment and reward parameters based on regional/temporal context.

Computes:
  1. Amount values (HIGH/MEDIUM/LOW/NONE) from the local subsidy distribution
     in maxmin-scaled [0,1] space (matching what states2proxy feeds the proxy)
  2. Sigmoid reward parameters calibrated to the distribution of 5-year
     target variable deltas in original units (matching FAIRY proxy output)

The raw parquet files contain UNSCALED values. The witch_proc_data Dataset
applies maxmin scaling at load time. We replicate that scaling here for
subsidies so amount values are in the correct space.

For the sigmoid, the FAIRY proxy output is delta_<target> in ORIGINAL units
(it rescales internally via addcmul), so sigmoid targets use raw deltas.

Usage (from gflownet/proxy/iam/):
    python tune_parameters.py --region europe --year 2025
    python tune_parameters.py --region europe --year 2025 --year_window 1 --margin 0.3
    python tune_parameters.py --region europe --year 2025 --target_variable EMI_total_CO2
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Sector->tech mapping (mirrors full_plan.py; duplicated to avoid circular import)
ALLOWED_SECTOR2TECH = {
    "POWER": ["power_COAL_noccs","power_COAL_ccs","power_NUCLEAR","power_OIL",
              "power_GAS_noccs","power_GAS_ccs","power_HYDRO","power_BIOMASS_noccs",
              "power_BIOMASS_ccs","power_WIND_onshore","power_WIND_offshore","power_SOLAR"],
    "ENERGY": ["thermal_SOLAR","enduse_COAL_ccs"],
    "VEHICLES": ["CARS_trad","CARS_hybrid","CARS_electric","CARS_fuelcell",
                 "HEAVYDUTY_trad","HEAVYDUTY_hybrid","HEAVYDUTY_electric","HEAVYDUTY_fuelcell"],
    "STORAGE": ["power_STORAGE","production_HYDROGEN","refueling_station_HYDROGEN","pipelines_HYDROGEN"],
    "DAC": ["DAC_liquid_sorbents","DAC_solid_sorbents","DAC_calcium_oxide"],
}



# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_raw_data(data_dir):
    """
    Load the raw (unscaled) parquet files and apply zero-block filtering,
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



def get_global_mask(keys_df, center_year, year_window):
    """Boolean mask for ALL regions in the year range (for global sigmoid calibration)."""
    year_lo = center_year - year_window * 5
    year_hi = center_year + year_window * 5
    mask = (keys_df["year"] >= year_lo) & (keys_df["year"] <= year_hi)
    print(f"Global sigmoid subset: all regions, years=[{year_lo}, {year_hi}]")
    print(f"  Rows in subset: {mask.sum()} / {len(keys_df)}")
    return mask


# ---------------------------------------------------------------------------
# 3. Compute 5-year target variable deltas (original units)
# ---------------------------------------------------------------------------

def compute_variable_deltas(variables_df, keys_df, mask, target_variable):
    """
    Compute the distribution of 5-year deltas for `target_variable` in the
    filtered subset. Matches what the FAIRY proxy outputs:
        target(t+5) - target(t)  in ORIGINAL (unscaled) units.

    Parameters
    ----------
    variables_df : pd.DataFrame
        Raw (unscaled) variables dataframe.
    keys_df : pd.DataFrame
        Keys dataframe with gdx, year, n columns.
    mask : pd.Series
        Boolean mask for the relevant subset.
    target_variable : str
        Column name in variables_df to compute deltas for (e.g. "CONSUMPTION",
        "EMI_total_CO2").
    """
    if target_variable not in variables_df.columns:
        raise ValueError(
            f"Target variable '{target_variable}' not found in variables_df. "
            f"Available: {list(variables_df.columns)}"
        )

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

        val_current = float(variables_df.loc[idx, target_variable])
        val_next = float(variables_df.loc[next_idx, target_variable])
        deltas.append(val_next - val_current)
        baseline_values.append(val_current)

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

def suggest_amount_values(subsidies_scaled, mask, margin=0.0,
                          high_pct=90, medium_pct=50, low_pct=25):
    """
    Suggest HIGH/MEDIUM/LOW/NONE from the actual distribution of non-zero
    subsidies in the subset (maxmin-scaled [0,1] space).

    - HIGH   = high_pct percentile of non-zero subsidies (* (1 + margin))
    - MEDIUM = medium_pct percentile of non-zero subsidies
    - LOW    = low_pct percentile of non-zero subsidies
    - NONE   = 0.0

    Defaults are conservative (margin=0.0) to keep the GFN within the
    proxy's training distribution and avoid out-of-distribution extrapolation
    that produces unrealistically large energy values.

    Parameters
    ----------
    margin : float
        Fractional upward extension of HIGH beyond the high_pct value.
        Default 0.0 (no extension — stay within observed distribution).
        Use small values like 0.1 only if you want to allow modest exploration
        beyond the training data.
    high_pct : int
        Percentile of non-zero subsidies to use as HIGH. Default 90.
        Lower values (e.g. 75) further restrict the maximum allowed investment.
    medium_pct : int
        Percentile for MEDIUM. Default 50 (median).
    low_pct : int
        Percentile for LOW. Default 25.
    """
    sub = subsidies_scaled.loc[mask]

    all_vals = sub.values.flatten()
    nonzero_vals = all_vals[all_vals > 1e-9]

    if len(nonzero_vals) == 0:
        print("  WARNING: All subsidies are zero in the subset. Using fallback.")
        return {"HIGH": 0.75, "MEDIUM": 0.3, "LOW": 0.1, "NONE": 0.0}, {
            "warning": "all zeros", "n_nonzero": 0}

    p_low    = float(np.percentile(nonzero_vals, low_pct))
    p_medium = float(np.percentile(nonzero_vals, medium_pct))
    p_75     = float(np.percentile(nonzero_vals, 75))
    p_high   = float(np.percentile(nonzero_vals, high_pct))
    p_max    = float(np.max(nonzero_vals))

    high_val = min(1.0, p_high * (1.0 + margin))

    amounts = {
        "HIGH":   round(high_val, 4),
        "MEDIUM": round(p_medium, 4),
        "LOW":    round(p_low, 4),
        "NONE":   0.0,
    }

    per_tech_stats = sub.describe().T[["min", "max", "mean", "std"]]

    info = {
        "n_nonzero": int(len(nonzero_vals)),
        "n_total": int(len(all_vals)),
        "frac_nonzero": round(len(nonzero_vals) / len(all_vals), 4),
        f"nonzero_p{low_pct} (LOW)":    round(p_low, 4),
        f"nonzero_p{medium_pct} (MEDIUM)": round(p_medium, 4),
        "nonzero_p75":                  round(p_75, 4),
        f"nonzero_p{high_pct} (HIGH)":  round(p_high, 4),
        "nonzero_max":                  round(p_max, 4),
        "margin":                       margin,
        f"HIGH (p{high_pct}*(1+margin))": round(high_val, 4),
    }

    return amounts, info, per_tech_stats



# ---------------------------------------------------------------------------
# 5b. Per-sector amount mappings
# ---------------------------------------------------------------------------

def suggest_amount_values_per_sector(subsidies_scaled, keys_df, mask,
                                     margin=0.0, high_pct=90,
                                     medium_pct=50, low_pct=25):
    """
    Compute per-sector amount mappings (HIGH/MEDIUM/LOW/NONE) from the
    non-zero subsidy distribution of each sector's techs separately.

    Sectors with no non-zero subsidies in the filtered subset are flagged
    and filled with the values from the sector that has the lowest HIGH value
    (most conservative observed sector), to avoid feeding the proxy
    out-of-distribution inputs.

    Returns
    -------
    per_tech_mapping : dict {tech_name: [v_unset, v_HIGH, v_MEDIUM, v_LOW, v_NONE]}
        One 5-element list per tech, in the amount_values_mapping index order:
        index 0 = unset (always 0.0), 1 = HIGH, 2 = MEDIUM, 3 = LOW, 4 = NONE.
    sector_amounts : dict {sector: {"HIGH": float, "MEDIUM": float, "LOW": float}}
    sector_info : dict {sector: info_dict}  diagnostic information
    empty_sectors : list[str]  sectors with no non-zero data in subset
    """
    sub = subsidies_scaled.loc[mask]
    sector_amounts = {}
    sector_info = {}
    empty_sectors = []

    for sector, techs in ALLOWED_SECTOR2TECH.items():
        # Collect columns for this sector's techs
        cols = [f"SUBS_{t}" for t in techs if f"SUBS_{t}" in sub.columns]
        if not cols:
            empty_sectors.append(sector)
            sector_amounts[sector] = None
            sector_info[sector] = {"warning": "no columns found"}
            continue

        sector_vals = sub[cols].values.flatten()
        nonzero_vals = sector_vals[sector_vals > 1e-9]

        if len(nonzero_vals) == 0:
            print(f"  WARNING: Sector {sector!r} has no non-zero subsidies in "
                  f"the subset. Will be filled with fallback values.")
            empty_sectors.append(sector)
            sector_amounts[sector] = None
            sector_info[sector] = {
                "warning": "all zeros in subset",
                "n_total": int(len(sector_vals)),
                "n_nonzero": 0,
            }
            continue

        p_low    = float(np.percentile(nonzero_vals, low_pct))
        p_medium = float(np.percentile(nonzero_vals, medium_pct))
        p_high   = float(np.percentile(nonzero_vals, high_pct))
        p_max    = float(np.max(nonzero_vals))
        high_val = min(1.0, p_high * (1.0 + margin))

        # If MEDIUM or LOW collapse to zero (sparse sector like DAC), apply
        # geometric spacing: MEDIUM = HIGH/2, LOW = HIGH/4.
        # This ensures the GFN always has 4 meaningfully distinct levels.
        geometric_fallback = False
        # Check rounded values — raw percentiles may be tiny but nonzero
        # (e.g. 3e-5 passes the nonzero filter but rounds to 0.0 at 4dp)
        if round(p_medium, 4) <= 0.0 or round(p_low, 4) <= 0.0:
            geometric_fallback = True
            medium_val = round(high_val / 2.0, 4)
            low_val    = round(high_val / 4.0, 4)
            if sector not in empty_sectors:
                print(f"  NOTE: Sector {sector!r} has collapsed MEDIUM/LOW "
                      f"(p{medium_pct}={p_medium:.4f}, p{low_pct}={p_low:.4f}). "
                      f"Using geometric spacing: HIGH={high_val:.4f}, "
                      f"MEDIUM={medium_val:.4f}, LOW={low_val:.4f}.")
        else:
            medium_val = round(p_medium, 4)
            low_val    = round(p_low, 4)

        sector_amounts[sector] = {
            "HIGH":   round(high_val, 4),
            "MEDIUM": medium_val,
            "LOW":    low_val,
            "NONE":   0.0,
        }
        sector_info[sector] = {
            "n_nonzero":   int(len(nonzero_vals)),
            "n_total":     int(len(sector_vals)),
            "frac_nonzero": round(len(nonzero_vals) / len(sector_vals), 4),
            f"p{low_pct} (LOW)":       round(p_low, 4),
            f"p{medium_pct} (MEDIUM)": round(p_medium, 4),
            f"p{high_pct} (HIGH)":     round(p_high, 4),
            "max":                     round(p_max, 4),
            "margin":                  margin,
            "geometric_fallback":      geometric_fallback,
        }

    # Fill empty sectors with the most conservative non-empty sector
    # (lowest HIGH value = least likely to push proxy OOD)
    non_empty = {s: v for s, v in sector_amounts.items() if v is not None}
    if non_empty:
        fallback_sector = min(non_empty, key=lambda s: non_empty[s]["HIGH"])
        fallback_amounts = non_empty[fallback_sector]
        print(f"  Fallback sector for empty sectors: {fallback_sector!r} "
              f"(HIGH={fallback_amounts['HIGH']})")
    else:
        # No sector has any data — use hardcoded conservative fallback
        fallback_amounts = {"HIGH": 0.1, "MEDIUM": 0.05, "LOW": 0.01, "NONE": 0.0}
        print("  WARNING: No sector has non-zero data. Using hardcoded fallback.")

    for sector in empty_sectors:
        sector_amounts[sector] = fallback_amounts.copy()
        sector_info[sector]["fallback_from"] = fallback_sector if non_empty else "hardcoded"

    # Build per-tech mapping dict
    # Format: {tech_name: [0.0, HIGH, MEDIUM, LOW, NONE]}
    # Index order matches amount_values_mapping: 0=unset, 1=HIGH, 2=MEDIUM, 3=LOW, 4=NONE
    per_tech_mapping = {}
    for sector, techs in ALLOWED_SECTOR2TECH.items():
        amts = sector_amounts[sector]
        row = [0.0, amts["HIGH"], amts["MEDIUM"], amts["LOW"], amts["NONE"]]
        for tech in techs:
            per_tech_mapping[tech] = row

    return per_tech_mapping, sector_amounts, sector_info, empty_sectors


# ---------------------------------------------------------------------------
# 6. Derive sigmoid reward parameters from delta distribution
# ---------------------------------------------------------------------------

def suggest_sigmoid_params(delta_stats, target_variable, margin=0.2,
                           sigma_window=None):
    """
    Fit sigmoid to the 5-year delta distribution (original units).

    Always uses maximization anchors (q75 center, q95 saturation) since the
    proxy output is already negated for minimized variables (e.g. EMI_total_CO2)
    in iam_proxies.py, so higher proxy output always means better outcome.

    R(x) = alpha / (1 + gamma * exp(beta * x))   [GFN native parameterization]

    Parameters
    ----------
    sigma_window : tuple(int, int) or None
        Override (center_pct, saturation_pct) percentiles.
        Must be in {5, 10, 25, 50, 75, 90, 95}.
    """
    PCT_TO_KEY = {5: "q05", 10: "q10", 25: "q25", 50: "q50",
                  75: "q75", 90: "q90", 95: "q95"}

    if sigma_window is not None:
        center_pct, sat_pct = sigma_window
        x_mid  = delta_stats[PCT_TO_KEY[center_pct]]
        x_high = delta_stats[PCT_TO_KEY[sat_pct]]
        mode_label = f"custom window ({center_pct}th center, {sat_pct}th saturation)"
    else:
        x_mid  = delta_stats["q75"]
        x_high = delta_stats["q95"]
        mode_label = "q75 center, q95 saturation"

    print(f"\n  [{target_variable}] Sigmoid fitting targets ({mode_label}):")
    print(f"    x_mid  = {x_mid:.6f}  -> R = 0.50")
    print(f"    x_high = {x_high:.6f} -> R approx 0.99")

    spread = x_high - x_mid
    print(f"    spread = {spread:.6f}")

    if abs(spread) < 1e-10:
        print("  WARNING: x_high approx x_mid, using fallback spread.")
        spread = (delta_stats["max"] - delta_stats["q50"]) * 0.25

    beta  = -x_mid           # center sigmoid at x_mid
    gamma = 4.595 / spread   # ln(99) / (q95 - q75), always positive

    alpha = 1.0

    def R(x):
        exp_arg = np.clip(-gamma * (x + beta), -500, 500)
        return alpha / (1.0 + np.exp(exp_arg))

    params = {
        "gamma": round(float(gamma), 4),
        "beta": round(float(beta), 6),
        "alpha": float(alpha),
    }

    diagnostics = {}
    for q in ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]:
        diagnostics[f"R({q})"] = round(R(delta_stats[q]), 4)

    # Return R closure so print_summary can reuse it without re-computing
    return params, diagnostics, R


def print_summary(delta_stats, subsidy_stats, amounts, amount_info,
                  sigmoid_params, sigmoid_diag, target_variable, R_func=None, round_decimals=1):
    print("\n" + "=" * 70)
    print("TUNING RESULTS")
    print("=" * 70)

    print(f"\n--- 5-Year Δ{target_variable} Distribution (original units) ---")
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

    print(f"\n--- Suggested Sigmoid Reward Parameters (target: {target_variable}) ---")
    for k, v in sigmoid_params.items():
        print(f"  {k}: {v}")
    print(f"  Verification (exact params):")
    for k, v in sigmoid_diag.items():
        print(f"    {k}: {v}")

    gamma_r = round(sigmoid_params["gamma"], round_decimals)
    beta_r = round(sigmoid_params["beta"], round_decimals)
    alpha = sigmoid_params["alpha"]
    print(f"\n--- Reward at delta quantiles (rounded: gamma={gamma_r}, beta={beta_r}) ---")
    for q in ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]:
        dx = delta_stats[q]
        rv = alpha / (1.0 + np.exp(-gamma_r * (dx + beta_r)))
        print(f"  R(Δ{target_variable}={dx:+.6f}) at {q} = {rv:.4f}")

    print("\n--- Suggested YAML config overrides ---")
    print(f"""
env:
  amount_values_mapping: [0.0, {amounts['HIGH']}, {amounts['MEDIUM']}, {amounts['LOW']}, {amounts['NONE']}]

proxy:
  target_variable: {target_variable}
  reward_function: sigmoid
  reward_function_kwargs:
    gamma: {sigmoid_params['gamma']}
    beta: {sigmoid_params['beta']}
    alpha: {sigmoid_params['alpha']}
""")



def print_sigmoid_summary(sigmoid_params, sigmoid_diag, target_variable):
    """Print sigmoid params block (shared between global and per-sector paths)."""
    print(f"\n--- Suggested Sigmoid Reward Parameters (target: {target_variable}) ---")
    for k, v in sigmoid_params.items():
        print(f"  {k}: {v}")
    print("  Verification (exact params):")
    for k, v in sigmoid_diag.items():
        print(f"    {k}: {v}")


def print_summary_per_sector(sector_amounts, sector_info, empty_sectors):
    """Print per-sector amount calibration results."""
    print("\n" + "=" * 70)
    print("PER-SECTOR AMOUNT CALIBRATION")
    print("=" * 70)
    for sector, amts in sector_amounts.items():
        flagged = " [FALLBACK]" if sector in empty_sectors else ""
        print(f"\n  {sector}{flagged}:")
        info = sector_info.get(sector, {})
        if "warning" in info:
            print(f"    WARNING: {info['warning']}")
        if "fallback_from" in info:
            print(f"    Filled from: {info['fallback_from']!r}")
        for k, v in info.items():
            if k not in ("warning", "fallback_from"):
                print(f"    {k}: {v}")
        print(f"    -> HIGH={amts['HIGH']}, MEDIUM={amts['MEDIUM']}, "
              f"LOW={amts['LOW']}, NONE={amts['NONE']}")

    print("\n--- Suggested YAML config overrides (per-sector) ---")
    print("env:")
    print("  amount_values_mapping:")
    for sector, amts in sector_amounts.items():
        for tech_name in ALLOWED_SECTOR2TECH[sector]:
            row = [0.0, amts["HIGH"], amts["MEDIUM"], amts["LOW"], amts["NONE"]]
            print(f"    {tech_name}: {row}")


def get_tuning_results(region, year, year_window, margin, target_variable, data_dir,
                      sigma_window=None, high_pct=90, medium_pct=50, low_pct=25,
                      per_sector_amounts=False, global_sigmoid=False):
    """
    Programmatic entry point — returns (amounts, sigmoid_params) dict for use
    by the pipeline runner without re-parsing CLI args.
    """
    subsidies_df, variables_df, keys_df = load_raw_data(data_dir)
    subsidies_scaled, _ = maxmin_scale_subsidies(subsidies_df)
    mask = get_subset_mask(keys_df, region, year, year_window)

    if mask.sum() == 0:
        raise RuntimeError(
            f"No data found for region={region}, year={year}, window={year_window}.\n"
            f"  Available regions: {sorted(keys_df['n'].unique())}\n"
            f"  Available years:   {sorted(keys_df['year'].unique())}"
        )

    if global_sigmoid:
        print(f"  [global_sigmoid] Computing deltas across ALL regions in window...")
        global_mask = get_global_mask(keys_df, year, year_window)
        delta_stats = compute_variable_deltas(variables_df, keys_df, global_mask, target_variable)
    else:
        delta_stats = compute_variable_deltas(variables_df, keys_df, mask, target_variable)
    if delta_stats is None:
        raise RuntimeError(
            f"No valid (t, t+5) pairs found. Try --year_window 1 or check year."
        )

    sub_stats = compute_subsidy_stats(subsidies_scaled, mask)

    if per_sector_amounts:
        per_tech_mapping, sector_amounts, sector_info, empty_sectors = \
            suggest_amount_values_per_sector(
                subsidies_scaled, keys_df, mask, margin=margin,
                high_pct=high_pct, medium_pct=medium_pct, low_pct=low_pct)
        amounts = per_tech_mapping
        print_summary_per_sector(sector_amounts, sector_info, empty_sectors)
    else:
        amounts, amount_info, sub_stats = suggest_amount_values(
            subsidies_scaled, mask, margin=margin,
            high_pct=high_pct, medium_pct=medium_pct, low_pct=low_pct)
        print_summary(delta_stats, sub_stats, amounts, amount_info,
                      sigmoid_params=None, sigmoid_diag=None,
                      target_variable=target_variable, R_func=None)

    sigmoid_params, sigmoid_diag, R_func = suggest_sigmoid_params(
        delta_stats, target_variable, margin=margin)
    print_sigmoid_summary(sigmoid_params, sigmoid_diag, target_variable)

    return amounts, sigmoid_params


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
    parser.add_argument("--margin", type=float, default=0.0,
                        help="Fractional extension of HIGH beyond high_pct (default: 0.0 = stay within training distribution)")
    parser.add_argument("--high_pct", type=int, default=90,
                        help="Percentile of non-zero subsidies used as HIGH (default: 90). "
                             "Lower values (e.g. 75) reduce max investment to avoid "
                             "out-of-distribution proxy inputs.")
    parser.add_argument("--medium_pct", type=int, default=50,
                        help="Percentile for MEDIUM (default: 50)")
    parser.add_argument("--low_pct", type=int, default=25,
                        help="Percentile for LOW (default: 25)")
    parser.add_argument("--no_per_sector_amounts", action="store_true",
                        help="Use a single global amount mapping instead of per-sector. "
                             "Default is per-sector.")
    parser.add_argument("--data_dir", type=str, default="scenario_data")
    parser.add_argument("--round_decimals", type=int, default=1,
                        help="Decimals to round gamma/beta in quantile printout")
    parser.add_argument("--target_variable", type=str, default="CONSUMPTION",
                        help="Variable to tune sigmoid for (default: CONSUMPTION). "
                             "Variables with 'EMI' or 'COST' in their name are "
                             "treated as minimization targets.")
    parser.add_argument("--sigma_window", type=int, nargs=2,
                        metavar=("CENTER_PCT", "SAT_PCT"),
                        default=None,
                        help="Override sigmoid fitting percentiles, e.g. "
                             "'--sigma_window 50 10' for a wider minimization span. "
                             "Values must be in {5,10,25,50,75,90,95}.")
    parser.add_argument("--global_sigmoid", action="store_true",
                        help="Calibrate sigmoid on ALL regions in the time window "
                             "instead of just the target region.")

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

    print(f"Computing 5-year Δ{args.target_variable} (original units)...")
    delta_stats = compute_variable_deltas(
        variables_df, keys_df, mask, args.target_variable)
    if delta_stats is None:
        print("ERROR: No valid (t, t+5) pairs. Try --year_window 1 or check year.")
        sys.exit(1)

    print("Computing subsidy statistics (maxmin-scaled)...")
    sub_stats = compute_subsidy_stats(subsidies_scaled, mask)

    if not args.no_per_sector_amounts:
        print("Computing per-sector amount values...")
        per_tech_mapping, sector_amounts, sector_info, empty_sectors = \
            suggest_amount_values_per_sector(
                subsidies_scaled, keys_df, mask, margin=args.margin,
                high_pct=args.high_pct, medium_pct=args.medium_pct,
                low_pct=args.low_pct)
        print_summary_per_sector(sector_amounts, sector_info, empty_sectors)
    else:
        print("Suggesting amount values...")
        amounts, amount_info, sub_stats = suggest_amount_values(
            subsidies_scaled, mask, margin=args.margin,
            high_pct=args.high_pct, medium_pct=args.medium_pct, low_pct=args.low_pct)

    print("Suggesting sigmoid parameters...")
    sigmoid_params, sigmoid_diag, R_func = suggest_sigmoid_params(
        delta_stats, args.target_variable, margin=args.margin,
        sigma_window=tuple(args.sigma_window) if args.sigma_window else None)

    if not args.no_per_sector_amounts:
        print_sigmoid_summary(sigmoid_params, sigmoid_diag, args.target_variable)
    else:
        print_summary(delta_stats, sub_stats, amounts, amount_info,
                      sigmoid_params, sigmoid_diag, args.target_variable,
                      R_func=R_func, round_decimals=args.round_decimals)
        print_sigmoid_summary(sigmoid_params, sigmoid_diag, args.target_variable)


if __name__ == "__main__":
    main()