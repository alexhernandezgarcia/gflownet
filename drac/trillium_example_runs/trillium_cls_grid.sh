#!/bin/bash
#SBATCH --job-name=cls_grid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=12:00:00
#SBATCH --output=cls_grid-%j.out
# =============================================================================
# TRILLIUM submitter: hyper-parameter exploration for CLASSIFICATION tree runs
# (lr x shared_weights, plus one long small-batch run) on credit_quantile.
# =============================================================================
#
# Same whole-node farm pattern as trillium_reg_tree.sh / trillium_nig_grid.sh,
# but the task grid is (variant x split) on a single classification dataset,
# and each task runs drac/cls_tree_worker.sh (the classification twin of
# reg_tree_worker.sh: identical train/resume logic, but the final evaluation
# uses gflownet/envs/tree/eval_tree.py instead of eval_regression_tree.py).
#
# The grid (all with MLP fwd/bwd policies and no lr decay):
#
#   tag              lr     shared_weights  steps   batch fwd/bwd-replay
#   lr1e-2_shared    0.01   True            10000   45 / 5
#   lr1e-2_sep       0.01   False           10000   45 / 5
#   lr1e-3_shared    0.001  True            10000   45 / 5
#   lr1e-3_sep       0.001  False           10000   45 / 5
#   lr1e-3_sep_long  0.001  False           20000   18 / 2
#
# 5 variants x 3 splits = 15 runs at 12 cores each -> all run concurrently
# (~180/192 cores). "No decay" is enforced by lr_decay_period=1000000, far
# beyond the longest run. Everything is passed as hydra overrides; the
# experiment config file is untouched, and the config hash puts each variant
# in its own run directory automatically.
#
# Usage (from $SCRATCH!):
#   mkdir -p $SCRATCH/gflownet-logs/slurm && cd $SCRATCH
#   sbatch --account=<your-account> $HOME/gflownet/drac/trillium_cls_grid.sh
#
# Knobs: EXP_NAME EXP_CONFIG DATASET SPLITS SEED RUNS_ROOT FORCE CPUS_PER_RUN
# Extra hydra overrides given on the command line are forwarded to every run.
# =============================================================================

set -u

REPO="${REPO:-$HOME/gflownet}"
VENV="${VENV:-$SCRATCH/venvs/gflownet-env}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
export EXP_NAME="${EXP_NAME:-CREDITQUANT_trillium}"
export EXP_CONFIG="${EXP_CONFIG:-tree/classification_tree}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export DATASET="${DATASET:-credit_quantile}"
export SEED="${SEED:-0}"

SPLITS="${SPLITS:-1 2 3}"
CORES_PER_NODE=192
export CPUS_PER_RUN="${CPUS_PER_RUN:-12}"

# --- The grid: "tag hydra-override [hydra-override ...]" per variant --------
# Steps and batch sizes live in the variant (not COMMON) because the long run
# changes them; hydra overrides must not repeat a key.
VARIANTS=(
    "lr1e-2_shared   gflownet.optimizer.lr=0.01  policy.backward.shared_weights=True  gflownet.optimizer.n_train_steps=10000 gflownet.optimizer.batch_size.forward=45 gflownet.optimizer.batch_size.backward_replay=5"
    "lr1e-2_sep      gflownet.optimizer.lr=0.01  policy.backward.shared_weights=False gflownet.optimizer.n_train_steps=10000 gflownet.optimizer.batch_size.forward=45 gflownet.optimizer.batch_size.backward_replay=5"
    "lr1e-3_shared   gflownet.optimizer.lr=0.001 policy.backward.shared_weights=True  gflownet.optimizer.n_train_steps=10000 gflownet.optimizer.batch_size.forward=45 gflownet.optimizer.batch_size.backward_replay=5"
    "lr1e-3_sep      gflownet.optimizer.lr=0.001 policy.backward.shared_weights=False gflownet.optimizer.n_train_steps=10000 gflownet.optimizer.batch_size.forward=45 gflownet.optimizer.batch_size.backward_replay=5"
    "lr1e-3_sep_long gflownet.optimizer.lr=0.001 policy.backward.shared_weights=False gflownet.optimizer.n_train_steps=20000 gflownet.optimizer.batch_size.forward=18 gflownet.optimizer.batch_size.backward_replay=2"
)

# --- Fixed overrides shared by every run ------------------------------------
COMMON=(
    "policy.forward.type=mlp"
    "policy.backward.type=mlp"
    "gflownet.optimizer.batch_size.backward_dataset=0"
    "gflownet.optimizer.lr_decay_period=1000000"
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
echo " Trillium classification hyper-parameter grid"
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
    bash "$CODE_DIR/drac/cls_tree_worker.sh" "$s" \
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
