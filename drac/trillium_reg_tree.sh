#!/bin/bash
#SBATCH --job-name=reg_tree
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=12:00:00
#SBATCH --output=reg_tree-%j.out
# =============================================================================
# TRILLIUM submitter: whole-node farm.
# =============================================================================
#
# Trillium's CPU subcluster schedules by NODE, not by core: the smallest job
# you can ask for is one full node (192 cores, 768 GB), walltime between 15
# minutes and 24 hours. A 5-task array of 4-core jobs -- which is the right
# shape on Mila and Rorqual -- would waste 5 whole nodes here. So this script
# takes one node and runs the whole (datasets x splits x seeds) grid on it
# concurrently, each run pinned to CPUS_PER_RUN threads.
#
# Two other Trillium rules this script obeys:
#   - submit from $SCRATCH (Slurm writes its output next to where you
#     submitted, and $HOME is read-only on compute nodes);
#   - $HOME / $PROJECT are readable but NOT writable from a compute node, so
#     every output path points at $SCRATCH (the worker handles the caches).
#
# Usage (from $SCRATCH!):
#   mkdir -p $SCRATCH/gflownet-logs/slurm && cd $SCRATCH
#   sbatch --account=<your-account> $HOME/gflownet/drac/trillium_reg_tree.sh
#
#   # a fuller node: 3 datasets x 5 splits x 3 seeds = 45 runs
#   sbatch --account=<acct> \
#          --export=ALL,DATASETS="diabetes energy concrete",SEEDS="0 1 2" \
#          $HOME/gflownet/drac/trillium_reg_tree.sh gflownet.optimizer.lr=1e-3
#
# Knobs: EXP_NAME EXP_CONFIG DATASETS SPLITS SEEDS RUNS_ROOT TOP_K FORCE
#        CPUS_PER_RUN (default: node/ntasks, capped at 8)
# =============================================================================

set -u

REPO="${REPO:-$HOME/gflownet}"
VENV="${VENV:-$SCRATCH/venvs/gflownet-env}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
export EXP_NAME="${EXP_NAME:-TREECLASS_REG}"
export WANDB_MODE="${WANDB_MODE:-offline}"

DATASETS="${DATASETS:-diabetes}"
SPLITS="${SPLITS:-1 2 3 4 5}"
SEEDS="${SEEDS:-0}"
CORES_PER_NODE=192

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
# Build the task list and size each run so the node is actually used
# ---------------------------------------------------------------------------
tasks=()
for d in $DATASETS; do
    for s in $SPLITS; do
        for sd in $SEEDS; do
            tasks+=("$d $s $sd")
        done
    done
done
n_tasks=${#tasks[@]}

if [ -z "${CPUS_PER_RUN:-}" ]; then
    CPUS_PER_RUN=$(( CORES_PER_NODE / n_tasks ))
    (( CPUS_PER_RUN < 1 )) && CPUS_PER_RUN=1
    # These models are small MLPs; torch stops scaling well past ~8 threads,
    # so extra cores are better spent on more concurrent runs.
    (( CPUS_PER_RUN > 8 )) && CPUS_PER_RUN=8
fi
export CPUS_PER_RUN
CONCURRENCY=$(( CORES_PER_NODE / CPUS_PER_RUN ))
(( CONCURRENCY < 1 )) && CONCURRENCY=1

used=$(( n_tasks < CONCURRENCY ? n_tasks * CPUS_PER_RUN : CORES_PER_NODE ))
echo "============================================================"
echo " Trillium whole-node farm"
echo " Job                : ${SLURM_JOB_ID:-none} on $(hostname)"
echo " Tasks              : $n_tasks   ($DATASETS | splits: $SPLITS | seeds: $SEEDS)"
echo " Threads per run    : $CPUS_PER_RUN   concurrency: $CONCURRENCY"
echo " Node utilisation   : ~$used / $CORES_PER_NODE cores"
echo " Runs root          : $RUNS_ROOT"
echo "============================================================"
if (( used < CORES_PER_NODE / 2 )); then
    echo "WARNING: this job leaves more than half the node idle. Add datasets,"
    echo "         seeds or hyper-parameter variants before submitting, or the"
    echo "         Trillium support team may (rightly) get in touch."
fi

WORKER_LOGS="$RUNS_ROOT/$EXP_NAME/worker-logs/${SLURM_JOB_ID:-local}"
mkdir -p "$WORKER_LOGS"
STATUS_FILE="$WORKER_LOGS/status.txt"
: > "$STATUS_FILE"

EXTRA=("$@")   # extra hydra overrides, forwarded unchanged

run_one () {
    local d="$1" s="$2" sd="$3"
    local log="$WORKER_LOGS/${d}_split${s}_seed${sd}.out"
    DATASET="$d" SEED="$sd" \
        bash "$CODE_DIR/drac/reg_tree_worker.sh" "$s" "${EXTRA[@]}" \
        > "$log" 2>&1
    echo "$? $d split$s seed$sd $log" >> "$STATUS_FILE"
}

running=0
for t in "${tasks[@]}"; do
    read -r d s sd <<< "$t"
    run_one "$d" "$s" "$sd" &
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
