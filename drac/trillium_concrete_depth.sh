#!/bin/bash
#SBATCH --job-name=concrete_depth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=24:00:00
#SBATCH --output=concrete_depth-%j.out
# =============================================================================
# TRILLIUM submitter: CONCRETE max_depth campaign -- 2 depths x 3 splits =
# 6 runs at 32 cores each (192 cores), 24 h node, resubmit to resume.
# =============================================================================
#
# Why depth: on concrete every prior saturates the depth-5 cap (28-31 of 31
# nodes) and a single pruned CART of unbounded depth (R2 0.815) beats our
# depth-5 posterior ensemble (~0.76) -- capacity is the binding constraint.
#
# Why 7 and 8 (and not 9 / 10 / 15): the environment allocates 2^max_depth-1
# node SLOTS whatever the tree actually uses -- the policy input width, every
# action mask, the meta action space and the max trajectory length all scale
# with it (tree.py: max_nodes) -- and deeper trees also mean longer
# trajectories. Measured on concrete (2026-09-02, login node, bcart phi=0.5):
# depth 5 ~2.5-2.9 s/step; depth 9 (511 slots) hit 36 s/step by step 3 and
# was killed for memory while its trees were still small. Depth 10 = 1023
# slots, depth 15 = 32767 slots (~500k-dim policy input): infeasible.
# Depth 7 (127 slots, 128 leaves -> ~6.4 of the 824 training samples per
# leaf at full occupancy) is the ceiling any *balanced* tree can exploit at
# a min-leaf of ~5; depth 8 (255 slots) additionally allows the unbalanced
# deep paths a pruned CART typically has, at ~8x the depth-5 slot count.
# Depth 6 is covered by trillium_concrete_capacity.sh.
#
# Everything else = the most promising configuration to date: bcart prior
# sigma=0.95 phi=0.5 (best on diabetes, concrete, all new datasets),
# alpha_0=3 / beta_0=overfit, shared backward weights, Adam lr 1e-3, batch
# 45 forward + 5 backward-replay, deduplicated replay buffer (config
# default).
#
# Steps: 20000 by default (not 40000): it matches the COMPLETE 20k depth-5
# prior grid on concrete (bcart phi=0.5: R2 0.735, 28 nodes), so the depth
# effect is read at equal training length, and it halves a cost that is
# already 4-15x per step above depth 5. Rough expectation at 32 threads
# (torch stops scaling past ~8-12 threads, so the gain over 8 is modest):
# depth 7 ~1-2 submissions of 24 h, depth 8 ~2-4. RESUBMIT the identical
# sbatch command; finished runs are skipped and cut-off ones resume from
# their checkpoints. Consider appending evaluator.period=500 (see Usage):
# the periodic 1000-tree evaluation every 100 steps is ~20% overhead at
# these depths (it changes the config hash -- fine for a new campaign).
#
# Usage (from $SCRATCH on Trillium!):
#   cd $SCRATCH && sbatch --account=<acct> $HOME/gflownet/drac/trillium_concrete_depth.sh evaluator.period=500
#
# Knobs: EXP_NAME DATASET DEPTHS SPLITS N_STEPS RUNS_ROOT TOP_K FORCE
#        CPUS_PER_RUN. Extra hydra overrides on the command line are
#        forwarded to every run (e.g. evaluator.period=500 to evaluate less
#        often on the slow depth-9 runs).
# =============================================================================

set -u

REPO="${REPO:-$HOME/gflownet}"
VENV="${VENV:-$SCRATCH/venvs/gflownet-env}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
export DATASET="${DATASET:-concrete}"
export EXP_NAME="${EXP_NAME:-REG_CONCRETE_DEPTH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SEED=0

DEPTHS="${DEPTHS:-7 8}"
SPLITS="${SPLITS:-1 2 3}"
N_STEPS="${N_STEPS:-20000}"
CORES_PER_NODE=192
# 6 runs x 32 cores = 192.
export CPUS_PER_RUN="${CPUS_PER_RUN:-32}"

# --- Fixed recipe: the most promising configuration to date -----------------
COMMON=(
    "proxy.prior_type=bcart"
    "proxy.sigma=0.95"
    "proxy.phi=0.5"
    "proxy.alpha_0=3.0"
    "proxy.beta_0=overfit"
    "policy.backward.shared_weights=True"
    "gflownet.optimizer.n_train_steps=$N_STEPS"
    "gflownet.optimizer.lr=0.001"
    "gflownet.optimizer.batch_size.forward=45"
    "gflownet.optimizer.batch_size.backward_replay=5"
    "gflownet.optimizer.batch_size.backward_dataset=0"
)

module purge
module load StdEnv/2023 python/3.10
source "$VENV/bin/activate"

if [ -n "${SLURM_TMPDIR:-}" ]; then
    export CODE_DIR="$SLURM_TMPDIR/gflownet"
    rsync -a --exclude ".git" --exclude "__pycache__" "$REPO/" "$CODE_DIR/"
else
    export CODE_DIR="$REPO"
fi
export REPO

# Task list: (depth x split); deepest first so the slowest runs start now.
tasks=()
for depth in $(echo $DEPTHS | tr ' ' '\n' | sort -rn); do
    for s in $SPLITS; do
        tasks+=("$depth $s")
    done
done
n_tasks=${#tasks[@]}

CONCURRENCY=$(( CORES_PER_NODE / CPUS_PER_RUN ))
(( CONCURRENCY < 1 )) && CONCURRENCY=1
used=$(( n_tasks < CONCURRENCY ? n_tasks * CPUS_PER_RUN : CORES_PER_NODE ))

echo "============================================================"
echo " Trillium concrete max_depth campaign"
echo " Job                : ${SLURM_JOB_ID:-none} on $(hostname)"
echo " Dataset            : $DATASET   depths: $DEPTHS   splits: $SPLITS   seed: $SEED   steps: $N_STEPS"
echo " Tasks              : $n_tasks   threads per run: $CPUS_PER_RUN   concurrency: $CONCURRENCY"
echo " Node utilisation   : ~$used / $CORES_PER_NODE cores"
echo " Common overrides   : ${COMMON[*]}"
echo " Runs root          : $RUNS_ROOT   campaign: $EXP_NAME"
echo "============================================================"

WORKER_LOGS="$RUNS_ROOT/$EXP_NAME/worker-logs/${SLURM_JOB_ID:-local}"
mkdir -p "$WORKER_LOGS"
STATUS_FILE="$WORKER_LOGS/status.txt"
: > "$STATUS_FILE"

EXTRA=("$@")

run_one () {
    local depth="$1" s="$2"
    local log="$WORKER_LOGS/${DATASET}_split${s}_depth${depth}.out"
    bash "$CODE_DIR/drac/reg_tree_worker.sh" "$s" \
        "${COMMON[@]}" "env.max_depth=$depth" "${EXTRA[@]}" \
        > "$log" 2>&1
    echo "$? $DATASET split$s depth$depth $log" >> "$STATUS_FILE"
}

running=0
for t in "${tasks[@]}"; do
    read -r depth s <<< "$t"
    run_one "$depth" "$s" &
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
