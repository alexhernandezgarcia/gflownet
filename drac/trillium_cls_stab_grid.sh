#!/bin/bash
#SBATCH --job-name=cls_stab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=12:00:00
#SBATCH --output=cls_stab-%j.out
# =============================================================================
# TRILLIUM submitter: stability grid for CLASSIFICATION trees on the bigger
# datasets (credit_quantile, jannis2) -- the winners of the MAGIC_STAB_* grid.
# =============================================================================
#
# Motivated by the MAGIC_STAB results (2026-08-28): on magic (n~15k) the
# tempered posterior beta=0.1 was the clear winner (best test acc AND best
# raw log-posterior trees), SGD(m=0.8, lr 3e-4, grad clip) a close second,
# and every campaign ran with a degenerate replay buffer (100 copies of one
# tree) -- fixed since via buffer.check_diversity=True PLUS the functional
# Tree.isclose (env.functional_isclose=True: trees are duplicates iff they
# route the training data identically -- exact-equality dedup alone cannot
# catch threshold-jittered copies). Both are config defaults now; the dedup
# ablation itself runs on magic (Mila), not here. This grid transfers the
# winning variants to credit_quantile / jannis2 with the fixed buffer.
#
# The reward-scale argument: log-posterior magnitude grows ~ linearly with n,
# so the beta that maps the log-reward range to O(100) nats shrinks ~ 1/n.
# magic n~15k -> beta 0.1; jannis2 n~46k is ~4x bigger, so beta 0.01 is
# included as the scaled equivalent, not as a throwaway point.
#
#   tag        what it tests
#   ctrl_b1    untempered baseline; with the buffer fix, isolates whether
#              dedup'd replay alone rescues beta=1 training
#   temp0.1    magic winner as-is
#   temp0.01   n-scaled tempering (see above)
#   sgd        msgd momentum 0.8, lr 3e-4, grad-norm clip 1.0, lr_z_mult 10.
#              NO reward clipping: the reward_min floor cannot be set below
#              these datasets' log-reward range (see magic analysis).
#
# NOTE tempered runs sample posterior^beta (deliberately flatter): judge them
# by top-k / BMA test metrics, not the mean over the 1000 samples.
#
# 4 variants x 3 splits = 12 runs at 16 cores each = 192/192 cores.
# credit_quantile (n~13k train) fits the 12 h walltime; jannis2 (~4x magic's
# per-iteration cost) will NOT finish 10k steps in one job -- resubmit the
# same command until eval_results.json exists (workers resume from ckpts,
# finished runs are skipped).
#
# BEFORE LAUNCHING (from the Mila side):
#   1. commit + sync this branch to the Trillium checkout -- the replay-buffer
#      fix (buffer.check_diversity + the functional Tree.isclose in tree.py,
#      a base-branch file!) and epsilon_annealing keys live in the configs/
#      code, not in this script;
#   2. jannis2 split csvs are gitignored -- rsync tests/data/tree/jannis2/
#      to the Trillium checkout if not already there.
#
# Usage (from $SCRATCH on Trillium!):
#   cd $SCRATCH
#   sbatch --account=<acct> $HOME/gflownet/drac/trillium_cls_stab_grid.sh
#   sbatch --account=<acct> --export=ALL,DATASET=jannis2 \
#          $HOME/gflownet/drac/trillium_cls_stab_grid.sh
#
# The campaign name defaults to CLS_STAB_<DATASET>, so the two jobs land in
# separate campaign directories.
#
# Knobs: EXP_NAME EXP_CONFIG DATASET SPLITS SEED RUNS_ROOT FORCE CPUS_PER_RUN
# Extra hydra overrides given on the command line are forwarded to every run.
# =============================================================================

set -u

REPO="${REPO:-$HOME/gflownet}"
VENV="${VENV:-$SCRATCH/venvs/gflownet-env}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
export EXP_CONFIG="${EXP_CONFIG:-tree/classification_tree}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export DATASET="${DATASET:-credit_quantile}"
export EXP_NAME="${EXP_NAME:-CLS_STAB_${DATASET^^}}"
export SEED="${SEED:-0}"

SPLITS="${SPLITS:-1 2 3}"
CORES_PER_NODE=192
# 12 tasks at 16 cores each -> all run at once (192/192 cores).
export CPUS_PER_RUN="${CPUS_PER_RUN:-16}"

# --- The grid: "tag hydra-override [hydra-override ...]" per variant --------
# lr lives in the variants (the SGD arm needs its own), steps/batch in COMMON.
VARIANTS=(
    "ctrl_b1  gflownet.optimizer.lr=0.001"
    "temp0.1  gflownet.optimizer.lr=0.001 proxy.reward_function_kwargs.beta=0.1"
    "temp0.01 gflownet.optimizer.lr=0.001 proxy.reward_function_kwargs.beta=0.01"
    "sgd      gflownet.optimizer.lr=0.0003 gflownet.optimizer.method=msgd gflownet.optimizer.sgd_momentum=0.8 gflownet.optimizer.clip_grad_norm=1.0 gflownet.optimizer.lr_z_mult=10"
)

# --- Fixed overrides shared by every run ------------------------------------
# MAGIC_STAB recipe: 10k steps, batch 45 fwd + 5 bwd-replay, MLP policies,
# shared_weights=False (yaml default), no lr decay. The replay-buffer dedup
# (buffer.check_diversity=True, similarity=-1, env.functional_isclose=True)
# comes from the experiment/env configs, not from here.
COMMON=(
    "policy.forward.type=mlp"
    "policy.backward.type=mlp"
    "gflownet.optimizer.n_train_steps=10000"
    "gflownet.optimizer.batch_size.forward=45"
    "gflownet.optimizer.batch_size.backward_replay=5"
    "gflownet.optimizer.batch_size.backward_dataset=0"
    "gflownet.optimizer.lr_decay_period=1000000"
    "env.max_depth=5"
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
echo " Trillium classification stability grid"
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
