#!/bin/bash
#SBATCH --job-name=class_baselines
#SBATCH --output=/home/mila/a/arnit/gflownet/class_baselines/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition=long-cpu,long-cpu-eek

# Classical classification benchmarks (Bayesian CART via MCMC/SMC, MAPTree,
# CART, random forest, XGBoost/LightGBM/CatBoost) on the MAGIC gamma
# telescope dataset, to compare against the GFlowNet classification-tree runs.
#
# The submit wrapper queues two job arrays:
#   - fast (task 0, 3h limit, main-cpu): CART/random forest, the gradient
#     boosters, MAPTree (time-boxed search per split) and BCART-SMC
#     (~0.2 s/particle/split), all 5 splits. Results within the hour.
#   - mcmc (tasks 1-5, 24h limit, long-cpu): BCART MCMC on split <task>
#     (~13 min per 10k moves; also writes the bcart_map result).
#
# SLURM logs live in $REPO/class_baselines/slurm/; result JSONs go to
# $SCRATCH/gflownet-benchmarks/class_baselines/results (the run_*.py default,
# override with $GFLOWNET_BENCHMARKS_DIR). Summarize any time with:
#   python class_baselines/print_results.py
#
# Usage (from anywhere; the script submits itself):
#   bash mila/tree/class_baselines.sh
#
# Tunables via environment variables (forwarded by --export=ALL):
#   MCMC_ITERATIONS=50000  SMC_PARTICLES=1000  MAPTREE_TIME_LIMIT=300
#   BINARIZATION_THRESHOLDS=9
#
# MAX_DEPTH (default 5) caps the depth of CART/random forest and the boosted
# trees. With a non-default value (e.g. MAX_DEPTH=3 bash mila/tree/
# class_baseliness.sh) only those methods are (re)run -- MAPTree and the
# BCART samplers have no hard depth cap (depth is regularized by the
# branching prior) -- and their results are written under suffixed method
# names (cart_gini_d3, xgboost_d3, ...) next to the depth-5 ones.

set -u

REPO="$HOME/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"
SCRIPT="$REPO/mila/tree/class_baselines.sh"

MCMC_ITERATIONS="${MCMC_ITERATIONS:-50000}"
SMC_PARTICLES="${SMC_PARTICLES:-1000}"
MAPTREE_TIME_LIMIT="${MAPTREE_TIME_LIMIT:-300}"
BINARIZATION_THRESHOLDS="${BINARIZATION_THRESHOLDS:-9}"
MAX_DEPTH="${MAX_DEPTH:-5}"

# ---- Submit wrapper ---------------------------------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    mkdir -p "$REPO/class_baselines/slurm"
    if [ "$MAX_DEPTH" -ne 5 ]; then
        echo "[submit] MAX_DEPTH=$MAX_DEPTH: queueing depth-capped methods only"
        echo "[submit] (CART/random forest/boosters; task 0, 3h limit, main-cpu)"
        exec sbatch --export=ALL --array=0 --time=3:00:00 \
            --partition=main-cpu "$SCRIPT"
    fi
    echo "[submit] queueing fast benchmarks (task 0, 3h limit, main-cpu)"
    sbatch --export=ALL --array=0 --time=3:00:00 \
        --partition=main-cpu "$SCRIPT"
    echo "[submit] queueing BCART MCMC benchmarks (tasks 1-5, 24h limit)"
    exec sbatch --export=ALL --array=1-5 "$SCRIPT"
fi

echo "=========================================================="
echo "[$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID] class benchmarks on magic"
echo "Node: $(hostname) | started: $(date)"
echo "=========================================================="

module load python/3.10
source "$VENV/bin/activate"
cd "$REPO"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

status=0
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    python class_baselines/run_cart.py --datasets magic \
        --max-depth "$MAX_DEPTH" || status=$?
    python class_baselines/run_gbt.py --datasets magic \
        --max-depth "$MAX_DEPTH" || status=$?
    if [ "$MAX_DEPTH" -eq 5 ]; then
        python class_baselines/run_maptree.py --datasets magic \
            --thresholds "$BINARIZATION_THRESHOLDS" \
            --time-limit "$MAPTREE_TIME_LIMIT" || status=$?
        python class_baselines/run_bcart.py --datasets magic \
            --methods smc --particles "$SMC_PARTICLES" \
            --thresholds "$BINARIZATION_THRESHOLDS" || status=$?
    fi
else
    python class_baselines/run_bcart.py --datasets magic \
        --splits "$SLURM_ARRAY_TASK_ID" \
        --methods mcmc --iterations "$MCMC_ITERATIONS" \
        --thresholds "$BINARIZATION_THRESHOLDS" || status=$?
fi

if [ $status -ne 0 ]; then
    echo "task $SLURM_ARRAY_TASK_ID FAIL (last error code $status, see above)"
    exit "$status"
fi
echo "task $SLURM_ARRAY_TASK_ID DONE at $(date)"
