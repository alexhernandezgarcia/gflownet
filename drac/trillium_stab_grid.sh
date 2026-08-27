#!/bin/bash
#SBATCH --job-name=stab_grid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=12:00:00
#SBATCH --output=stab_grid-%j.out
# =============================================================================
# TRILLIUM submitter: training-recipe / stability grid for regression trees.
# =============================================================================
#
# Motivated by the REG_NIG_GRID results (2026-08): lr 0.01 diverged even on
# diabetes, and the best-overall run (config 11fd0d0d, REG_CUBEARGS) differs
# from the NIG-grid winner (3a08cf4a) in THREE things at once -- NIG params
# (a2/null vs a3/overfit) AND policy.backward.shared_weights (yes vs no).
# This grid deconfounds that and tests the stability interventions from the
# classification (magic) analysis on the regression task.
#
# Baseline = the best known config: shared backward weights, Adam lr 1e-3,
# alpha_0=2, beta_0=null, node_count prior. Variants change ONE thing each:
#
#   ctrl       baseline as-is (replicates 11fd0d0d on diabetes; establishes
#              the baseline on energy/concrete)
#   noshare    shared_weights=False       (the confound, isolated)
#   nig_a3ovf  alpha_0=3, beta_0=overfit  (NIG-grid winner, now WITH shared
#              weights -> completes the 2x2 with the two existing campaigns)
#   temp0.1    reward beta=0.1   (tempered posterior^0.1: shrinks the
#   temp0.01   reward beta=0.01   log-reward range TB has to price)
#   sgd        method=msgd, momentum 0.8, grad-norm clip 1.0, lr_z_mult 10
#              (wine SGD recipe; NO reward clipping -- do_clip_rewards stays
#              False, so the magic reward-floor trap cannot occur)
#
# NOTE on tempered runs: they sample posterior^beta, a deliberately flatter
# distribution -- judge them by top-k / BMA test metrics, not by the mean of
# the 1000 samples, and do not mix them into posterior-fidelity comparisons.
#
# 6 variants x 3 splits = 18 runs at 10 cores each -> all concurrent
# (180/192 cores). ~4-9 h expected (diabetes < energy < concrete), within
# the 12 h walltime.
#
# Usage (from $SCRATCH on Trillium!):
#   cd $SCRATCH
#   sbatch --account=<acct> $HOME/gflownet/drac/trillium_stab_grid.sh
#   sbatch --account=<acct> --export=ALL,DATASET=energy \
#          $HOME/gflownet/drac/trillium_stab_grid.sh
#   sbatch --account=<acct> --export=ALL,DATASET=concrete \
#          $HOME/gflownet/drac/trillium_stab_grid.sh
#
# The campaign name defaults to REG_STAB_<DATASET> so the three jobs land in
# separate campaign directories.
#
# Knobs: EXP_NAME EXP_CONFIG DATASET SPLITS RUNS_ROOT TOP_K FORCE CPUS_PER_RUN
# Extra hydra overrides given on the command line are forwarded to every run.
# =============================================================================

set -u

REPO="${REPO:-$HOME/gflownet}"
VENV="${VENV:-$SCRATCH/venvs/gflownet-env}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
export DATASET="${DATASET:-diabetes}"
export EXP_NAME="${EXP_NAME:-REG_STAB_${DATASET^^}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SEED=0

SPLITS="${SPLITS:-1 2 3}"
CORES_PER_NODE=192
# 18 tasks at 10 cores each -> all run at once (180/192 cores).
export CPUS_PER_RUN="${CPUS_PER_RUN:-10}"

# --- The grid: "tag hydra-override [hydra-override ...]" per variant --------
VARIANTS=(
    "ctrl      policy.backward.shared_weights=True"
    "noshare   policy.backward.shared_weights=False"
    "nig_a3ovf policy.backward.shared_weights=True proxy.alpha_0=3.0 proxy.beta_0=overfit"
    "temp0.1   policy.backward.shared_weights=True proxy.reward_function_kwargs.beta=0.1"
    "temp0.01  policy.backward.shared_weights=True proxy.reward_function_kwargs.beta=0.01"
    "sgd       policy.backward.shared_weights=True gflownet.optimizer.method=msgd gflownet.optimizer.sgd_momentum=0.8 gflownet.optimizer.clip_grad_norm=1.0 gflownet.optimizer.lr_z_mult=10"
)

# --- Fixed overrides shared by every run ------------------------------------
# Baseline recipe = best known config (11fd0d0d): Adam lr 1e-3, 10k steps,
# batch 45 forward + 5 backward-replay, MLP policies, node_count prior,
# alpha_0=2 / beta_0=null (the yaml defaults, so not overridden here).
COMMON=(
    "gflownet.optimizer.n_train_steps=10000"
    "gflownet.optimizer.lr=0.001"
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
echo " Trillium stability / training-recipe grid"
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
