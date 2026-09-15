#!/bin/bash
#SBATCH --job-name=prior_grid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=12:00:00
#SBATCH --output=prior_grid-%j.out
# =============================================================================
# TRILLIUM submitter: grid search over the STRUCTURE PRIOR of the regression
# tree proxy (companion to trillium_nig_grid.sh, which grids the likelihood).
# =============================================================================
#
# Variants -- 6 structure priors plus 2 replay-buffer arms:
#
#   nodecount             current default: -log(4 * n_features) per split
#   bcart, phi in {0.5, 1.0, 1.5, 2.0} with sigma = 0.95
#                         Chipman et al. (1998) tree prior; the paper uses
#                         beta (= phi here) in {0.5, 1.0, 1.5} (Sec. 3.1/7);
#                         phi = 2.0 is the current repo default (BART-style,
#                         more aggressive)
#   noprior               no size penalty; upper bound on how far the
#                         likelihood alone gets (overfitting probe)
#   nodecount_nodivbuf    node_count with buffer.check_diversity=False, i.e.
#                         the OLD degenerate buffer (the 2026-08-28 replay
#                         inspection showed it holds 100 copies of ONE tree)
#                         -- the control arm isolating the buffer fix, since
#                         all other arms now run with the fixed buffer that
#                         the experiment configs default to since 2026-08-28
#   nodecount_rap0.2      node_count + random_action_prob 0.2 (more
#                         exploration on top of the diverse buffer)
#
# 8 variants x 3 splits = 24 runs at 8 cores each = 192 cores, all concurrent.
#
# Fixed base = stability-grid winner (2026-08-27): Adam lr 1e-3, reward
# beta=1 (tempering lost everywhere), shared backward weights, MLP policies,
# alpha_0=3 / beta_0=overfit (won on energy + concrete, tied on diabetes),
# batch 45 forward + 5 backward-replay. Override via knobs, e.g.:
#
#   sbatch --export=ALL,ALPHA0=2.0,BETA0=null,SHARED=True ...
#
# BETA0 accepts a number, "overfit", or "null" (= var(y)-based default).
# N_STEPS: use 20000 for concrete (the 20k REG_trillium run beats every 10k
# run there -- not converged at 10k). A 20k concrete job may hit the 12 h
# walltime for the big-tree arms: simply RESUBMIT the same sbatch command;
# finished runs are skipped and cut-off ones resume from their checkpoints.
#
# Usage (from $SCRATCH on Trillium!):
#   cd $SCRATCH
#   sbatch --account=<acct> $HOME/gflownet/drac/trillium_prior_grid.sh
#   sbatch --account=<acct> --export=ALL,DATASET=energy \
#          $HOME/gflownet/drac/trillium_prior_grid.sh
#   sbatch --account=<acct> --export=ALL,DATASET=concrete,N_STEPS=20000 \
#          $HOME/gflownet/drac/trillium_prior_grid.sh
#
# The campaign name defaults to REG_PRIOR_<DATASET>.
#
# Knobs: EXP_NAME EXP_CONFIG DATASET SPLITS ALPHA0 BETA0 SHARED N_STEPS
#        RUNS_ROOT TOP_K FORCE CPUS_PER_RUN
# =============================================================================

set -u

REPO="${REPO:-$HOME/gflownet}"
VENV="${VENV:-$SCRATCH/venvs/gflownet-env}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
export DATASET="${DATASET:-diabetes}"
export EXP_NAME="${EXP_NAME:-REG_PRIOR_${DATASET^^}}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SEED=0

SPLITS="${SPLITS:-1 2 3}"
ALPHA0="${ALPHA0:-3.0}"
BETA0="${BETA0:-overfit}"
SHARED="${SHARED:-True}"
N_STEPS="${N_STEPS:-10000}"
CORES_PER_NODE=192
# 24 tasks at 8 cores each = 192 cores, all concurrent.
export CPUS_PER_RUN="${CPUS_PER_RUN:-8}"

# --- The grid: "tag hydra-override [hydra-override ...]" per variant --------
# NOTE: since 2026-08-28 the experiment configs default to the FIXED replay
# buffer (buffer.check_diversity=True, duplicate states rejected), so every
# arm below runs with the fix; nodecount_nodivbuf is the old-buffer CONTROL.
VARIANTS=(
    "nodecount  proxy.prior_type=node_count"
    "bcart_p0.5 proxy.prior_type=bcart proxy.sigma=0.95 proxy.phi=0.5"
    "bcart_p1.0 proxy.prior_type=bcart proxy.sigma=0.95 proxy.phi=1.0"
    "bcart_p1.5 proxy.prior_type=bcart proxy.sigma=0.95 proxy.phi=1.5"
    "bcart_p2.0 proxy.prior_type=bcart proxy.sigma=0.95 proxy.phi=2.0"
    "noprior    proxy.prior_type=none"
    "nodecount_nodivbuf proxy.prior_type=node_count buffer.check_diversity=False"
    "nodecount_rap0.2 proxy.prior_type=node_count gflownet.random_action_prob=0.2"
)

# --- Fixed overrides shared by every run ------------------------------------
COMMON=(
    "proxy.alpha_0=$ALPHA0"
    "proxy.beta_0=$BETA0"
    "policy.backward.shared_weights=$SHARED"
    "gflownet.optimizer.n_train_steps=$N_STEPS"
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
echo " Trillium structure-prior grid"
echo " Job                : ${SLURM_JOB_ID:-none} on $(hostname)"
echo " Dataset            : $DATASET   splits: $SPLITS   seed: $SEED"
echo " Fixed base         : alpha_0=$ALPHA0 beta_0=$BETA0 shared_weights=$SHARED lr=0.001"
echo " Variants           : ${#VARIANTS[@]}   tasks: $n_tasks"
echo " Threads per run    : $CPUS_PER_RUN   concurrency: $CONCURRENCY"
echo " Node utilisation   : ~$used / $CORES_PER_NODE cores"
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
