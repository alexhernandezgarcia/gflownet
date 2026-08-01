#!/bin/bash
#SBATCH --job-name=reg_benchmarks
#SBATCH --output=/home/mila/a/arnit/gflownet/reg_benchmarks/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --partition=long-cpu,long-cpu-eek

# Classical regression benchmarks (BART, XGBoost/LightGBM, GP, linear models)
# to compare against the GFlowNet regression-tree runs.
#
# The submit wrapper queues two job arrays over the datasets
# (1=concrete, 2=diabetes, 3=energy; BART tasks are dataset id + 3):
#   - fast (tasks 1-3, 2h limit):  linear models, GP, XGBoost/LightGBM.
#     Submitted to main-cpu (PriorityTier 100 vs long-cpu's 10) with 2 CPUs
#     each so all three tasks fit the partition's per-user cap (8 CPUs, 64G)
#     at once and start ~immediately; results are available within minutes.
#   - bart (tasks 4-6, 48h limit): BART MCMC, hours per dataset, on the
#     long-cpu partitions set in the directives above.
#
# All outputs live in $REPO/reg_benchmarks/: SLURM logs in slurm/, result
# JSONs in results/ (the run_*.py default). Summarize any time with:
#   python reg_benchmarks/print_results.py
#
# Usage (from anywhere; the script submits itself):
#   bash mila/sbatch/reg_benchmarks.sh
#
# Tunables via environment variables (forwarded by --export=ALL):
#   BART_TREES=50  BART_DRAWS=1000  BART_TUNE=1000  BART_CHAINS=2

set -u

REPO="$HOME/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"
SCRIPT="$REPO/mila/sbatch/reg_benchmarks.sh"

BART_TREES="${BART_TREES:-50}"
BART_DRAWS="${BART_DRAWS:-1000}"
BART_TUNE="${BART_TUNE:-1000}"
BART_CHAINS="${BART_CHAINS:-2}"

# ---- Submit wrapper ---------------------------------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    mkdir -p "$REPO/reg_benchmarks/slurm"
    echo "[submit] queueing fast benchmarks (tasks 1-3, 2h limit, main-cpu)"
    sbatch --export=ALL --array=1-3 --time=2:00:00 \
        --partition=main-cpu --cpus-per-task=2 "$SCRIPT"
    echo "[submit] queueing BART benchmarks (tasks 4-6, 48h limit)"
    exec sbatch --export=ALL --array=4-6 "$SCRIPT"
fi

datasets=(concrete diabetes energy)
dataset="${datasets[$(((SLURM_ARRAY_TASK_ID - 1) % 3))]}"
if [ "$SLURM_ARRAY_TASK_ID" -le 3 ]; then
    mode="fast"
else
    mode="bart"
fi

echo "=========================================================="
echo "[$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID] $mode benchmarks on $dataset"
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
    echo "$mode benchmarks on $dataset FAIL (last error code $status, see above)"
    exit "$status"
fi
echo "$mode benchmarks on $dataset DONE at $(date)"
