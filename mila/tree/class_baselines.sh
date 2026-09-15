#!/bin/bash
#SBATCH --job-name=class_baselines
#SBATCH --output=/home/mila/a/arnit/gflownet/class_baselines/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --partition=long-cpu,long-cpu-eek

# Classical classification benchmarks (Bayesian CART via MCMC/SMC, MAPTree,
# CART, random forest, XGBoost/LightGBM/CatBoost) on the classification
# datasets under tests/data/tree/ (magic, credit, credit_quantile, jannis2,
# jannis4), to compare against the GFlowNet classification-tree runs.
#
# The submit wrapper queues two job arrays over the datasets in $DATASETS:
#   - fast (one task per dataset, $FAST_TIME limit, $FAST_PARTITION):
#     CART/random forest, the gradient boosters, MAPTree (time-boxed search
#     per split) and BCART-SMC, all 5 splits of that dataset.
#   - mcmc (one task per dataset x split, sbatch time limit, long-cpu): BCART MCMC
#     (also writes the bcart_map result).
# The binarized-feature methods (MAPTree, BCART) are binary-only; on jannis4
# the python scripts skip with a message and the task exits in seconds.
#
# SLURM logs live in $REPO/class_baselines/slurm/; result JSONs go to
# $SCRATCH/gflownet-benchmarks/class_baselines/results (the run_*.py default,
# override with $GFLOWNET_BENCHMARKS_DIR). Summarize any time with:
#   python class_baselines/print_results.py
#
# Usage (from anywhere; the script submits itself):
#   bash mila/tree/class_baselines.sh                                # magic only
#   DATASETS="credit_quantile jannis2 jannis4" bash mila/tree/class_baselines.sh
#
# Tunables via environment variables (forwarded by --export=ALL):
#   DATASETS="magic"       space-separated subset of the datasets above
#   METHODS="cart gbt maptree smc mcmc"   subset of methods to (re)run; the
#                          fast array is skipped when it has nothing to do,
#                          likewise the mcmc array. E.g. to rerun only the
#                          Bayesian single-tree baselines with more compute:
#     METHODS="maptree mcmc" MAPTREE_TIME_LIMIT=3600 MCMC_ITERATIONS=200000 \
#         DATASETS="magic credit_quantile jannis2" bash mila/tree/class_baselines.sh
#                          NOTE: results overwrite <method>__<dataset>__split<i>.json
#                          in the results dir; copy it first to keep the old ones.
#   OUTPUTS=""             result JSONs run_bcart.py writes (default: all of
#                          bcart_mcmc bcart_map single_tree_bcart_mcmc
#                          bcart_smc single_tree_bcart_smc). E.g. to add only
#                          the single-tree rows from the same seeded chains as
#                          the existing 200k-move bcart_mcmc results:
#     METHODS="smc mcmc" OUTPUTS="single_tree_bcart_mcmc single_tree_bcart_smc" \
#         MCMC_ITERATIONS=200000 DATASETS="magic credit_quantile" \
#         bash mila/tree/class_baselines.sh
#   MCMC_ITERATIONS=50000  SMC_PARTICLES=1000  MAPTREE_TIME_LIMIT=300
#   BINARIZATION_THRESHOLDS=9
#   FAST_TIME=3:00:00     FAST_PARTITION=main-cpu   FAST_MEM=24G
#                          (the fast task runs all 5 splits of a dataset one
#                          after the other, so FAST_TIME must cover at least
#                          5 x MAPTREE_TIME_LIMIT plus the SMC runs. MAPTree's
#                          search frontier grows with its time limit: 300 s
#                          fit in 24G, but 3600 s was OOM-killed at 25 GB after
#                          16-37 min on magic/credit_quantile/jannis2
#                          (2026-09-02) -- raise FAST_MEM together with
#                          MAPTREE_TIME_LIMIT, roughly 1.5 GB per minute)
# On the jannis datasets one MCMC move sweeps all ~46-67k samples x ~470
# binarized features, roughly 15x the magic cost; if a sbatch time mcmc task times
# out, rerun that dataset with a smaller MCMC_ITERATIONS.
#
# MAX_DEPTH (default 5) caps the depth of CART/random forest and the boosted
# trees. With a non-default value (e.g. MAX_DEPTH=3 bash mila/tree/
# class_baselines.sh) only those methods are (re)run -- MAPTree and the
# BCART samplers have no hard depth cap (depth is regularized by the
# branching prior) -- and their results are written under suffixed method
# names (cart_gini_d3, xgboost_d3, ...) next to the depth-5 ones.

set -u

REPO="$HOME/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"
SCRIPT="$REPO/mila/tree/class_baselines.sh"

DATASETS="${DATASETS:-magic}"
MCMC_ITERATIONS="${MCMC_ITERATIONS:-50000}"
SMC_PARTICLES="${SMC_PARTICLES:-1000}"
MAPTREE_TIME_LIMIT="${MAPTREE_TIME_LIMIT:-300}"
BINARIZATION_THRESHOLDS="${BINARIZATION_THRESHOLDS:-9}"
MAX_DEPTH="${MAX_DEPTH:-5}"
FAST_TIME="${FAST_TIME:-12:00:00}"
FAST_PARTITION="${FAST_PARTITION:-main-cpu}"
FAST_MEM="${FAST_MEM:-24G}"
METHODS="${METHODS:-cart gbt maptree smc mcmc}"
OUTPUTS="${OUTPUTS:-}"
# --outputs flag for run_bcart.py, empty (= write everything) by default
OUTPUTS_FLAG=()
if [ -n "$OUTPUTS" ]; then
    # shellcheck disable=SC2206  # OUTPUTS is a space-separated list on purpose
    OUTPUTS_FLAG=(--outputs $OUTPUTS)
fi

read -r -a DS_ARR <<< "$DATASETS"
N_DATASETS="${#DS_ARR[@]}"

# wants <method>: true iff <method> is listed in $METHODS
wants() { case " $METHODS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
for m in $METHODS; do
    case "$m" in cart|gbt|maptree|smc|mcmc) ;; *)
        echo "ERROR: unknown method '$m' in METHODS (cart gbt maptree smc mcmc)" >&2
        exit 1 ;;
    esac
done
RUN_FAST=1; RUN_MCMC=1
if ! wants cart && ! wants gbt && ! wants maptree && ! wants smc; then RUN_FAST=0; fi
if ! wants mcmc; then RUN_MCMC=0; fi

# ---- Submit wrapper ---------------------------------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    mkdir -p "$REPO/class_baselines/slurm"
    if [ "$MAX_DEPTH" -ne 5 ]; then
        echo "[submit] MAX_DEPTH=$MAX_DEPTH: queueing depth-capped methods only"
        echo "[submit] (CART/random forest/boosters on: $DATASETS)"
        exec sbatch --export=ALL,MODE=fast --array="0-$((N_DATASETS - 1))" \
            --time="$FAST_TIME" --partition="$FAST_PARTITION" "$SCRIPT"
    fi
    echo "[submit] methods: $METHODS | bcart outputs: ${OUTPUTS:-all}"
    if [ "$RUN_FAST" -eq 1 ]; then
        echo "[submit] queueing fast benchmarks (1 task/dataset: $DATASETS)"
        sbatch --export=ALL,MODE=fast --array="0-$((N_DATASETS - 1))" \
            --time="$FAST_TIME" --partition="$FAST_PARTITION" \
            --mem="$FAST_MEM" "$SCRIPT"
    fi
    if [ "$RUN_MCMC" -eq 1 ]; then
        echo "[submit] queueing BCART MCMC benchmarks (1 task/dataset x split, sbatch time limit)"
        exec sbatch --export=ALL,MODE=mcmc --array="0-$((5 * N_DATASETS - 1))" "$SCRIPT"
    fi
    exit 0
fi

# ---- Worker -----------------------------------------------------------------
MODE="${MODE:?MODE must be set by the submit wrapper (fast|mcmc)}"
if [ "$MODE" = "fast" ]; then
    DATASET="${DS_ARR[$SLURM_ARRAY_TASK_ID]}"
    SPLIT=""
else
    DATASET="${DS_ARR[$((SLURM_ARRAY_TASK_ID / 5))]}"
    SPLIT="$((SLURM_ARRAY_TASK_ID % 5 + 1))"
fi

echo "=========================================================="
echo "[$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID] $MODE benchmarks on $DATASET ${SPLIT:+split $SPLIT}"
echo "Node: $(hostname) | started: $(date)"
echo "=========================================================="

module load python/3.10
source "$VENV/bin/activate"
cd "$REPO"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

status=0
if [ "$MODE" = "fast" ]; then
    if wants cart; then
        python class_baselines/run_cart.py --datasets "$DATASET" \
            --max-depth "$MAX_DEPTH" || status=$?
    fi
    if wants gbt; then
        python class_baselines/run_gbt.py --datasets "$DATASET" \
            --max-depth "$MAX_DEPTH" || status=$?
    fi
    if [ "$MAX_DEPTH" -eq 5 ]; then
        if wants maptree; then
            python class_baselines/run_maptree.py --datasets "$DATASET" \
                --thresholds "$BINARIZATION_THRESHOLDS" \
                --time-limit "$MAPTREE_TIME_LIMIT" || status=$?
        fi
        if wants smc; then
            python class_baselines/run_bcart.py --datasets "$DATASET" \
                --methods smc --particles "$SMC_PARTICLES" "${OUTPUTS_FLAG[@]}" \
                --thresholds "$BINARIZATION_THRESHOLDS" || status=$?
        fi
    fi
else
    python class_baselines/run_bcart.py --datasets "$DATASET" \
        --splits "$SPLIT" \
        --methods mcmc --iterations "$MCMC_ITERATIONS" "${OUTPUTS_FLAG[@]}" \
        --thresholds "$BINARIZATION_THRESHOLDS" || status=$?
fi

if [ $status -ne 0 ]; then
    echo "task $SLURM_ARRAY_TASK_ID FAIL (last error code $status, see above)"
    exit "$status"
fi
echo "task $SLURM_ARRAY_TASK_ID DONE at $(date)"
