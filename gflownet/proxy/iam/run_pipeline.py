"""
Pipeline runner: tune parameters, generate a matched test set, write config.

Step 1 — tune_parameters.py  (year_window=1, i.e. ±5 years around center year)
    Computes optimal amount values and sigmoid reward params from the local
    data distribution.

Step 2 — generate_test_set.py  (year_window=0, exact center year only)
    Generates the test set using the tuned amount values from step 1.

Step 3 — config_writer.py
    Creates or updates the Hydra experiment YAML config for the given
    region/year/target_variable.  If the target config already exists it
    updates only the tuned fields; otherwise it is generated from the
    consumption template.

Usage:
    python run_pipeline.py --region europe --year 2025
    python run_pipeline.py --region usa --year 2030 --target_variable EMI_total_CO2
    python run_pipeline.py --region europe --year 2025 --n 10000 --output my_test.pkl
    python run_pipeline.py --region europe --year 2025 --tune_only
    python run_pipeline.py --region europe --year 2025 \\
        --amount_values 0.6,0.3,0.1,0.0 --skip_tune
    python run_pipeline.py --region europe --year 2025 --skip_config   # skip step 3
    python run_pipeline.py --region europe --year 2025 --dry_run_config # preview config

All other flags (--margin, --data_dir, --seed, --round_decimals) are passed
through to the relevant sub-scripts.
"""

import argparse
import sys

# ---------------------------------------------------------------------------
# Import from the sub-scripts directly (no subprocess overhead, same
# Python environment, errors propagate cleanly).
# ---------------------------------------------------------------------------
from tune_parameters import get_tuning_results
from generate_test_set import (
    load_raw_data,
    get_subset_mask,
    generate_test_set,
    build_output_path,
    DEFAULT_AMOUNT_VALUES,
)
from config_writer import write_or_update_config, TEMPLATE_PATH

import pickle


def parse_amount_values(amount_str):
    """Parse 'HIGH,MEDIUM,LOW,NONE' comma string into a dict."""
    vals = [float(v.strip()) for v in amount_str.split(",")]
    if len(vals) != 4:
        raise ValueError(
            "--amount_values must have exactly 4 values: HIGH,MEDIUM,LOW,NONE"
        )
    return dict(zip(["HIGH", "MEDIUM", "LOW", "NONE"], vals))


def main():
    parser = argparse.ArgumentParser(
        description="Run tune_parameters → generate_test_set pipeline"
    )

    # --- Shared args ---
    parser.add_argument("--region", type=str, required=True,
                        help="Target region (e.g. europe, usa, china)")
    parser.add_argument("--year", type=int, required=True,
                        help="Center year (e.g. 2025)")
    parser.add_argument("--target_variable", type=str, default="CONSUMPTION",
                        help="Proxy target variable (default: CONSUMPTION). "
                             "Variables with 'EMI' or 'COST' are treated as "
                             "minimization targets in the sigmoid calibration.")
    parser.add_argument("--data_dir", type=str, default="scenario_data",
                        help="Path to scenario_data directory (default: scenario_data)")

    # --- Tuning args ---
    parser.add_argument("--tune_window", type=int, default=1,
                        help="year_window for tuning step (default: 1 = ±5 years)")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="Fractional extension of HIGH beyond high_pct (default: 0.0)")
    parser.add_argument("--high_pct", type=int, default=90,
                        help="Percentile for HIGH amount (default: 90). Lower to restrict "
                             "max investment, e.g. 75 to avoid OOD proxy inputs.")
    parser.add_argument("--medium_pct", type=int, default=50,
                        help="Percentile for MEDIUM amount (default: 50)")
    parser.add_argument("--low_pct", type=int, default=25,
                        help="Percentile for LOW amount (default: 25)")
    parser.add_argument("--no_per_sector_amounts", action="store_true",
                        help="Use a single global amount mapping instead of per-sector. "
                             "Default is per-sector.")
    parser.add_argument("--round_decimals", type=int, default=1,
                        help="Decimal places to round gamma/beta printout (default: 1)")
    parser.add_argument("--sigma_window", type=int, nargs=2,
                        metavar=("CENTER_PCT", "SAT_PCT"), default=None,
                        help="Override sigmoid percentiles e.g. '--sigma_window 50 10'")
    parser.add_argument("--global_sigmoid", action="store_true",
                        help="Calibrate sigmoid on all regions in the time window, "
                             "not just the target region.")
    parser.add_argument("--random_sigmoid", action="store_true",
                        help=(
                            "Calibrate the sigmoid on random plan proxy outputs instead "
                            "of scenario deltas. Saves config as plan_fairy_{slug}_rand.yaml."
                        ))
    parser.add_argument("--n_random_sigmoid", type=int, default=10000,
                        help="Number of random plans for --random_sigmoid (default: 10000).")
    parser.add_argument("--random_sigmoid_seed", type=int, default=42,
                        help="Random seed for --random_sigmoid plan generation.")

    # --- Test-set args ---
    parser.add_argument("--n", type=int, default=None,
                        help="Total test set size (default: 2 x subset rows at exact year)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output pkl path (default: test_{region}_{year}_{target_variable}.pkl)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for test set generation (default: 42)")

    # --- Config args ---
    parser.add_argument("--template", type=str, default=TEMPLATE_PATH,
                        help=f"Path to the consumption template YAML "
                             f"(default: {TEMPLATE_PATH})")
    parser.add_argument("--skip_config", action="store_true",
                        help="Skip step 3 (config file write/update)")
    parser.add_argument("--dry_run_config", action="store_true",
                        help="Preview config changes without writing to disk")

    # --- Control flow ---
    parser.add_argument("--tune_only", action="store_true",
                        help="Run only the tuning step, skip test set and config")
    parser.add_argument("--skip_tune", action="store_true",
                        help="Skip tuning and use --amount_values directly")
    parser.add_argument("--amount_values", type=str, default=None,
                        help="Override amount values (HIGH,MEDIUM,LOW,NONE). "
                             "Required when --skip_tune is set.")

    args = parser.parse_args()

    if args.skip_tune and args.amount_values is None:
        parser.error("--skip_tune requires --amount_values to be specified.")

    # -----------------------------------------------------------------------
    # Step 1: Tune parameters
    # -----------------------------------------------------------------------
    if args.skip_tune:
        print("=" * 60)
        print("Skipping tuning step (--skip_tune set).")
        amounts = parse_amount_values(args.amount_values)
        sigmoid_params = None
        print(f"  Amount values: {amounts}")
    else:
        print("=" * 60)
        print(f"STEP 1 — Tuning parameters")
        print(f"  region={args.region}, year={args.year}, "
              f"window=±{args.tune_window}, target={args.target_variable}")
        print("=" * 60)

        amounts, sigmoid_params = get_tuning_results(
            region=args.region,
            year=args.year,
            year_window=args.tune_window,
            margin=args.margin,
            target_variable=args.target_variable,
            data_dir=args.data_dir,
            sigma_window=tuple(args.sigma_window) if args.sigma_window else None,
            high_pct=args.high_pct,
            medium_pct=args.medium_pct,
            low_pct=args.low_pct,
            per_sector_amounts=not args.no_per_sector_amounts,
            global_sigmoid=args.global_sigmoid,
            random_sigmoid=args.random_sigmoid,
            n_random=args.n_random_sigmoid,
            random_seed=args.random_sigmoid_seed,
        )

        print("\n  ✓ Tuning complete.")
        print(f"  Amount values: {amounts}")
        if sigmoid_params:
            print(f"  Sigmoid params: {sigmoid_params}")

    if args.tune_only:
        print("\n--tune_only set, stopping after tuning.")
        return

    # -----------------------------------------------------------------------
    # Step 2: Generate test set (exact year, year_window=0)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 2 — Generating test set")
    print(f"  region={args.region}, year={args.year} (exact), "
          f"target={args.target_variable}")
    print("=" * 60)

    print("Loading data...")
    subsidies_df, keys_df = load_raw_data(args.data_dir)

    print("Filtering subset (year_window=0)...")
    mask = get_subset_mask(keys_df, args.region, args.year, year_window=0)

    n_subset = int(mask.sum())
    if n_subset == 0:
        print("ERROR: No data found for the specified region/year.")
        print(f"  Available regions: {sorted(keys_df['n'].unique())}")
        print(f"  Available years:   {sorted(keys_df['year'].unique())}")
        sys.exit(1)

    n_total = args.n if args.n is not None else 2 * n_subset
    if args.n is None:
        print(f"  No --n specified, using 2 x {n_subset} = {n_total} total samples")

    print(f"Generating {n_total} samples using tuned amount values...")
    samples, n_data, n_random = generate_test_set(
        subsidies_df=subsidies_df,
        keys_df=keys_df,
        mask=mask,
        amount_values=amounts,
        n_total=n_total,
        seed=args.seed,
    )

    # Append _rand to target_variable slug for the test set filename
    # so it doesn't overwrite the scenario-calibrated test set.
    tv_for_path = (
        args.target_variable + "_rand"
        if args.random_sigmoid and not args.target_variable.endswith("_rand")
        else args.target_variable
    )
    output_path = build_output_path(args.output, args.region, args.year, tv_for_path)

    payload = {
        "samples": samples,
        "target_variable": args.target_variable,
        "amount_values": amounts,
        "random_sigmoid": args.random_sigmoid,
    }
    if sigmoid_params is not None:
        payload["sigmoid_params"] = sigmoid_params

    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"  ✓ Test set saved to {output_path}")

    # -----------------------------------------------------------------------
    # Step 3: Write / update Hydra config
    # -----------------------------------------------------------------------
    config_path = None
    if args.skip_config:
        print("\nSkipping config step (--skip_config set).")
    elif sigmoid_params is None:
        print(
            "\nWARNING: No sigmoid params available (tuning was skipped). "
            "Skipping config step. Pass --sigmoid in config_writer.py manually."
        )
    else:
        print("\n" + "=" * 60)
        print("STEP 3 — Writing experiment config")
        print("=" * 60)
        config_path = write_or_update_config(
            region=args.region,
            year=args.year,
            target_variable=args.target_variable,
            amounts=amounts,
            sigmoid_params=sigmoid_params,
            template_path=args.template,
            dry_run=args.dry_run_config,
            random_sigmoid=args.random_sigmoid,
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Target variable:      {args.target_variable}")
    print(f"  Amount values used:   {amounts}")
    if sigmoid_params:
        print(f"  Sigmoid params:       {sigmoid_params}")
    print(f"  Data-derived samples: {n_data}")
    print(f"  Random samples:       {n_random}")
    print(f"  Total size:           {len(samples)}")
    print(f"  Test set output:      {output_path}")
    if config_path:
        print(f"  Config output:        {config_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()