#!/bin/bash
#SBATCH --job-name=reg_baselines
#SBATCH --output=/home/mila/a/arnit/gflownet/reg_benchmarks/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --partition=long-cpu,long-cpu-eek

# Classical regression baselines (BART, XGBoost/LightGBM, GP, linear models)
# to compare against the GFlowNet regression-tree runs.
#
# The submit wrapper queues two job arrays over the requested datasets
# (fast task i and bart task i+N run dataset i of N):
#   - fast (tasks 1..N, 2h limit):  linear models, GP, XGBoost/LightGBM.
#     Submitted to main-cpu (PriorityTier 100 vs long-cpu's 10) with 2 CPUs
#     each; the partition's per-user cap (8 CPUs, 64G) runs up to 4 of them
#     at once, the rest queue behind them.
#   - bart (tasks N+1..2N, 48h limit): BART MCMC, hours per dataset, on the
#     long-cpu partitions set in the directives above.
#
# SLURM logs live in $REPO/reg_benchmarks/slurm/; result JSONs go to
# $SCRATCH/gflownet-benchmarks/reg_benchmarks/results (the run_*.py default,
# override with $GFLOWNET_BENCHMARKS_DIR). Summarize any time with:
#   python reg_benchmarks/print_results.py
#
# Usage (from anywhere; the script submits itself):
#   bash mila/tree/reg_baselines.sh [dataset ...]
# e.g.
#   bash mila/tree/reg_baselines.sh                 # concrete diabetes energy
#   bash mila/tree/reg_baselines.sh yacht slump     # any tests/data/tree dirs
#   DATASETS="yacht slump" bash mila/tree/reg_baselines.sh   # equivalent
#
# Dataset names must match a split-CSV directory, i.e.
# tests/data/tree/<name>/<name>_<1..5>.csv must exist.
#
# Tunables via environment variables (forwarded by --export=ALL):
#   BART_TREES=50  BART_DRAWS=1000  BART_TUNE=1000  BART_CHAINS=2

set -u

REPO="$HOME/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"
SCRIPT="$REPO/mila/tree/reg_baselines.sh"

BART_TREES="${BART_TREES:-50}"
BART_DRAWS="${BART_DRAWS:-1000}"
BART_TUNE="${BART_TUNE:-1000}"
BART_CHAINS="${BART_CHAINS:-2}"

# Datasets: positional args > $DATASETS > default trio. Exported so the
# array tasks (which receive no args from sbatch) see the same list.
if [ "$#" -gt 0 ]; then
    DATASETS="$*"
fi
export DATASETS="${DATASETS:-concrete diabetes energy}"
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
    echo "[submit] datasets: $DATASETS"
    echo "[submit] queueing fast baselines (tasks 1-$n, 2h limit, main-cpu)"
    sbatch --export=ALL --array="1-$n" --time=2:00:00 \
        --partition=main-cpu --cpus-per-task=2 "$SCRIPT"
    echo "[submit] queueing BART baselines (tasks $((n + 1))-$((2 * n)), 48h limit)"
    exec sbatch --export=ALL --array="$((n + 1))-$((2 * n))" "$SCRIPT"
fi

dataset="${datasets[$(((SLURM_ARRAY_TASK_ID - 1) % n))]}"
if [ "$SLURM_ARRAY_TASK_ID" -le "$n" ]; then
    mode="fast"
else
    mode="bart"
fi

echo "=========================================================="
echo "[$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID] $mode baselines on $dataset"
echo "Node: $(hostname) | started: $(date)"
echo "=========================================================="

module load python/3.10
source "$VENV/bin/activate"
cd "$REPO"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

status=0
if [ "$mode" = "fast" ]; then
    python reg_benchmarks/run_linear.py --datasets "$dataset" || status=$?
    python reg_benchmarks/run_gp.py --datasets "$dataset" || status=$?
    python reg_benchmarks/run_gbt.py --datasets "$dataset" || status=$?
else
    python reg_benchmarks/run_bart.py --datasets "$dataset" \
        --trees "$BART_TREES" --draws "$BART_DRAWS" \
        --tune "$BART_TUNE" --chains "$BART_CHAINS" || status=$?
fi

if [ $status -ne 0 ]; then
    echo "$mode baselines on $dataset FAIL (last error code $status, see above)"
    exit "$status"
fi
echo "$mode baselines on $dataset DONE at $(date)"
