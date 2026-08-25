#!/bin/bash
#SBATCH --job-name=nig_grid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=12:00:00
#SBATCH --output=nig_grid-%j.out
# =============================================================================
# TRILLIUM submitter: grid search over the NIG likelihood hyper-parameters
# (proxy.alpha_0 x proxy.beta_0) of the regression tree proxy.
# =============================================================================
#
# Same whole-node farm pattern as trillium_reg_tree.sh, but the task grid is
# (proxy variant x split) on a single dataset instead of (dataset x split x
# seed). Each variant is one (alpha_0, beta_0) cell:
#
#   alpha_0 in {1.5, 3, 5}   (nu = 3 BART-like, intermediate, nu = 10 as in
#                             Chipman et al. 1998, Sec. 6)
#   beta_0  in {null, overfit}
#     null    -> E[sigma^2] = var(y_train)            (current default)
#     overfit -> E[sigma^2] = resid. var of an overfit greedy CART
#                (Chipman et al. 1998, Sec. 4.1 recommendation)
#
# 6 variants x 5 splits = 30 runs at 8 cores each; 24 run concurrently and the
# remaining 6 start as slots free up. With 10k steps per run this finishes
# comfortably within the walltime. Set SPLITS="1 2 3" for a single 18-run wave.
#
# Fixed training setup (per request): MLP forward/backward policies without
# weight sharing (the experiment-config default), 10000 steps, lr 0.01,
# batch = 45 forward + 5 backward-replay trajectories.
#
# Usage (from $SCRATCH!):
#   mkdir -p $SCRATCH/gflownet-logs/slurm && cd $SCRATCH
#   sbatch --account=<your-account> $HOME/gflownet/drac/trillium_nig_grid.sh
#
# Knobs: EXP_NAME EXP_CONFIG DATASET SPLITS RUNS_ROOT TOP_K FORCE CPUS_PER_RUN
# Extra hydra overrides given on the command line are forwarded to every run.
# =============================================================================

set -u

REPO="${REPO:-$HOME/gflownet}"
VENV="${VENV:-$SCRATCH/venvs/gflownet-env}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
export EXP_NAME="${EXP_NAME:-REG_NIG_GRID}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export DATASET="${DATASET:-diabetes}"
export SEED=0

SPLITS="${SPLITS:-1 2 3}"
CORES_PER_NODE=192
# The runs must be fast: give each one 8 cores (24 concurrent on the node).
export CPUS_PER_RUN="${CPUS_PER_RUN:-12}"

# --- The grid: "tag hydra-override [hydra-override ...]" per variant --------
VARIANTS=(
    "a1.5_bvar proxy.alpha_0=1.5"
    "a1.5_bovf proxy.alpha_0=1.5 proxy.beta_0=overfit"
    "a3_bvar   proxy.alpha_0=3.0"
    "a3_bovf   proxy.alpha_0=3.0 proxy.beta_0=overfit"
    "a5_bvar   proxy.alpha_0=5.0"
    "a5_bovf   proxy.alpha_0=5.0 proxy.beta_0=overfit"
)

# --- Fixed overrides shared by every run ------------------------------------
COMMON=(
    "gflownet.optimizer.n_train_steps=10000"
    "gflownet.optimizer.lr=0.01"
    "gflownet.optimizer.batch_size.forward=45"
    "gflownet.optimizer.batch_size.backward_replay=5"
    "gflownet.optimizer.batch_size.backward_dataset=0"
)

module purge
module load StdEnv/2023 python/3.10
source "$VENV/bin/activate"

# One code snapshot per node, shared by every worker on it.
if [ -n "${SLURM_TMPDIR:-}" ]; then
    export CODE_DIR="$SLURM_TMPDIR/gflownet"
    rsync -a --exclude ".git" --exclude "__pycache__" "$REPO/" "$CODE_DIR/"
else
    export CODE_DIR="$REPO"
fi
export REPO

# ---------------------------------------------------------------------------
# Build the task list: (variant x split)
# ---------------------------------------------------------------------------
tasks=()
for v in "${VARIANTS[@]}"; do
    for s in $SPLITS; do
        tasks+=("$s $v")
    done
done
n_tasks=${#tasks[@]}

CONCURRENCY=$(( CORES_PER_NODE / CPUS_PER_RUN ))
(( CONCURRENCY < 1 )) && CONCURRENCY=1

used=$(( n_tasks < CONCURRENCY ? n_tasks * CPUS_PER_RUN : CORES_PER_NODE ))
echo "============================================================"
echo " Trillium NIG hyper-parameter grid"
echo " Job                : ${SLURM_JOB_ID:-none} on $(hostname)"
echo " Dataset            : $DATASET   splits: $SPLITS   seed: $SEED"
echo " Variants           : ${#VARIANTS[@]}   tasks: $n_tasks"
echo " Threads per run    : $CPUS_PER_RUN   concurrency: $CONCURRENCY"
echo " Node utilisation   : ~$used / $CORES_PER_NODE cores"
echo " Common overrides   : ${COMMON[*]}"
echo " Runs root          : $RUNS_ROOT   campaign: $EXP_NAME"
echo "============================================================"

WORKER_LOGS="$RUNS_ROOT/$EXP_NAME/worker-logs/${SLURM_JOB_ID:-local}"
mkdir -p "$WORKER_LOGS"
STATUS_FILE="$WORKER_LOGS/status.txt"
: > "$STATUS_FILE"

EXTRA=("$@")   # extra hydra overrides, forwarded unchanged

run_one () {
    local s="$1" tag="$2"; shift 2
    local var_overrides=("$@")
    local log="$WORKER_LOGS/${DATASET}_split${s}_${tag}.out"
    bash "$CODE_DIR/drac/reg_tree_worker.sh" "$s" \
        "${COMMON[@]}" "${var_overrides[@]}" "${EXTRA[@]}" \
        > "$log" 2>&1
    echo "$? $DATASET split$s $tag $log" >> "$STATUS_FILE"
}

running=0
for t in "${tasks[@]}"; do
    read -r s tag rest <<< "$t"
    read -ra var_overrides <<< "$rest"
    run_one "$s" "$tag" "${var_overrides[@]}" &
    running=$(( running + 1 ))
    if (( running >= CONCURRENCY )); then
        wait -n
        running=$(( running - 1 ))
    fi
done
wait

echo "============================================================"
echo " Finished           : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo " Per-run exit codes (0 = ok):"
sort -n "$STATUS_FILE" | sed 's/^/   /'
failed=$(awk '$1 != 0' "$STATUS_FILE" | wc -l)
echo " $failed / $n_tasks runs failed"
echo "============================================================"
[ "$failed" -eq 0 ]
