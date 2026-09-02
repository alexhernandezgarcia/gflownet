#!/bin/bash
#SBATCH --job-name=reg_bcart
#SBATCH --output=/home/mila/a/arnit/gflownet/reg_benchmarks/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --partition=long-cpu,long-cpu-eek

# Bayesian CART regression baselines (BCART MCMC / MAP / SMC on quantile-
# binarized features, reg_benchmarks/run_bcart.py) to compare against the
# GFlowNet regression-tree runs. Companion of mila/tree/reg_baselines.sh,
# which runs the other regression baselines (BART, boosters, GP, linear).
#
# One array task per dataset; each task runs all 5 splits, MCMC first (also
# writes the bcart_map result) then SMC. The regression datasets are small
# (< 1100 rows), so a task takes well under an hour with the defaults.
#
# SLURM logs live in $REPO/reg_benchmarks/slurm/; result JSONs go to
# $SCRATCH/gflownet-benchmarks/reg_benchmarks/results (the run_*.py default,
# override with $GFLOWNET_BENCHMARKS_DIR). Summarize any time with:
#   python reg_benchmarks/print_results.py
#
# Usage (from anywhere; the script submits itself):
#   bash mila/tree/reg_bcart_baselines.sh [dataset ...]
# e.g.
#   bash mila/tree/reg_bcart_baselines.sh                  # the default five
#   bash mila/tree/reg_bcart_baselines.sh yacht real_estate
#   DATASETS="yacht real_estate" bash mila/tree/reg_bcart_baselines.sh
#
# Dataset names must match a split-CSV directory, i.e.
# tests/data/tree/<name>/<name>_<1..5>.csv must exist, and the target must
# be continuous (classification datasets are skipped by run_bcart.py).
#
# Tunables via environment variables (forwarded by --export=ALL):
#   METHODS="mcmc smc"      samplers to run (mcmc also writes bcart_map)
#   MCMC_ITERATIONS=50000   SMC_PARTICLES=1000   BINARIZATION_THRESHOLDS=9
#   NIG_KAPPA_0=0.1  NIG_ALPHA_0=2.0             leaf prior (see run_bcart.py)

set -u

REPO="$HOME/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"
SCRIPT="$REPO/mila/tree/reg_bcart_baselines.sh"

METHODS="${METHODS:-mcmc smc}"
MCMC_ITERATIONS="${MCMC_ITERATIONS:-50000}"
SMC_PARTICLES="${SMC_PARTICLES:-1000}"
BINARIZATION_THRESHOLDS="${BINARIZATION_THRESHOLDS:-9}"
NIG_KAPPA_0="${NIG_KAPPA_0:-0.1}"
NIG_ALPHA_0="${NIG_ALPHA_0:-2.0}"

# Datasets: positional args > $DATASETS > default five. Exported so the
# array tasks (which receive no args from sbatch) see the same list.
if [ "$#" -gt 0 ]; then
    DATASETS="$*"
fi
export DATASETS="${DATASETS:-concrete diabetes energy yacht real_estate}"
read -r -a datasets <<< "$DATASETS"
n="${#datasets[@]}"

for d in "${datasets[@]}"; do
    if [ ! -f "$REPO/tests/data/tree/$d/${d}_1.csv" ]; then
        echo "ERROR: no split CSVs for dataset '$d'" \
             "(expected tests/data/tree/$d/${d}_1.csv)" >&2
        exit 1
    fi
done

# ---- Submit wrapper ---------------------------------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    mkdir -p "$REPO/reg_benchmarks/slurm"
    echo "[submit] datasets: $DATASETS | methods: $METHODS"
    echo "[submit] MCMC_ITERATIONS=$MCMC_ITERATIONS SMC_PARTICLES=$SMC_PARTICLES"
    exec sbatch --export=ALL --array="1-$n" "$SCRIPT"
fi

# ---- Worker -----------------------------------------------------------------
dataset="${datasets[$((SLURM_ARRAY_TASK_ID - 1))]}"

echo "=========================================================="
echo "[$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID] BCART baselines on $dataset"
echo "Node: $(hostname) | started: $(date)"
echo "=========================================================="

module load python/3.10
source "$VENV/bin/activate"
cd "$REPO"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

status=0
# shellcheck disable=SC2086  # METHODS is a space-separated list on purpose
python reg_benchmarks/run_bcart.py --datasets "$dataset" \
    --methods $METHODS \
    --iterations "$MCMC_ITERATIONS" --particles "$SMC_PARTICLES" \
    --thresholds "$BINARIZATION_THRESHOLDS" \
    --kappa-0 "$NIG_KAPPA_0" --alpha-0 "$NIG_ALPHA_0" || status=$?

if [ $status -ne 0 ]; then
    echo "BCART baselines on $dataset FAIL (error code $status, see above)"
    exit "$status"
fi
echo "BCART baselines on $dataset DONE at $(date)"
