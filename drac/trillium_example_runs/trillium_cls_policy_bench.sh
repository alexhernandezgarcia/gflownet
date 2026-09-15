#!/bin/bash
#SBATCH --job-name=cls_policy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=12:00:00
#SBATCH --output=cls_policy-%j.out
# =============================================================================
# TRILLIUM submitter: MLP vs TRANSFORMER policy benchmark for CLASSIFICATION
# trees on the small UCI datasets (iris, wine, breast_cancer, raisin).
# =============================================================================
#
# Same whole-node farm pattern as trillium_cls_grid.sh, but the task grid is
# (policy x dataset x split) and the two policies need different core counts.
# Tasks are therefore packed by a CORE BUDGET instead of a fixed concurrency:
# a task starts as soon as enough cores are free, transformer tasks first
# (heaviest first keeps the tail of the job short).
#
# Recipe (identical for both policies; everything not listed comes from the
# experiment config -- 2026-09 defaults: node_count prior, beta=1 untempered
# posterior, store_log_rewards, replay dedup with the functional Tree.isclose):
#
#   policy  config                          cores  steps  lr     batch fwd/bwd-replay
#   mlp     tree/classification_tree          8    1000   0.001  45 / 5
#   trfm    tree/trfm_classification_tree    16    1000   0.001  45 / 5
#
#   max_depth 5, shared forward/backward weights, seed 0, splits 1-5.
#
# The transformer runs use trfm_classification_tree.yaml, which IS
# classification_tree.yaml with the policy config group swapped (the diff is
# the policy block plus a wandb tag). Overriding policy=tree_transformer on top
# of the MLP config instead would leave the MLP's `type: mlp` key in place and
# mislabel the run as "mlp" in the run name.
#
# Full grid: 2 policies x 4 datasets x 5 splits = 40 runs. One node cannot hold
# them all at once (20 x 8 + 20 x 16 = 480 cores), so launch TWO jobs with two
# datasets each. Per node: 10 x 16 + 10 x 8 = 240 cores -> all 10 transformer
# runs plus 4 MLP runs start immediately; the other 6 MLP runs (the short ones)
# follow as soon as cores free up.
#
# Usage (from $SCRATCH on Trillium!):
#   mkdir -p $SCRATCH/gflownet-logs && cd $SCRATCH
#   DATASETS="iris wine"            sbatch --account=def-alexhg $HOME/gflownet/drac/trillium_cls_policy_bench.sh
#   DATASETS="breast_cancer raisin" sbatch --account=def-alexhg $HOME/gflownet/drac/trillium_cls_policy_bench.sh
#
# (Set DATASETS as a shell variable in front of sbatch rather than through
# --export: sbatch exports the caller's environment by default, and --export
# does not cope with the spaces in a multi-dataset value.)
#
# Both jobs write into the same campaign directory ($EXP_NAME); the run
# directories are keyed by dataset / split / policy / config hash, so they never
# collide. Idempotent: resubmitting the same command skips finished runs and
# resumes cut-off ones from their checkpoints.
#
# DRY_RUN=1 prints the task table and exits without launching anything (fine
# on a login node, e.g. to check the DATASETS split or the dataset files).
#
# Knobs: EXP_NAME DATASETS SPLITS POLICIES SEED N_STEPS LR CPUS_MLP CPUS_TRFM
#        RUNS_ROOT FORCE DRY_RUN
# Extra hydra overrides given on the command line are forwarded to every run.
# =============================================================================

set -u

REPO="${REPO:-$HOME/gflownet}"
VENV="${VENV:-$SCRATCH/venvs/gflownet-env}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
export EXP_NAME="${EXP_NAME:-CLS_POLICY_BENCH}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SEED="${SEED:-0}"
export FORCE="${FORCE:-0}"

DATASETS="${DATASETS:-iris wine breast_cancer raisin}"
SPLITS="${SPLITS:-1 2 3 4 5}"
# Heaviest policy first: the scheduler below starts tasks in this order.
POLICIES="${POLICIES:-trfm mlp}"
N_STEPS="${N_STEPS:-1000}"
LR="${LR:-0.001}"
CORES_PER_NODE="${CORES_PER_NODE:-192}"
CPUS_MLP="${CPUS_MLP:-8}"
CPUS_TRFM="${CPUS_TRFM:-16}"
DRY_RUN="${DRY_RUN:-0}"

# --- Per-policy experiment config and core count ----------------------------
declare -A CFG_OF=(
    [mlp]="tree/classification_tree"
    [trfm]="tree/trfm_classification_tree"
)
declare -A CPUS_OF=(
    [mlp]="$CPUS_MLP"
    [trfm]="$CPUS_TRFM"
)

# --- Fixed overrides shared by every run ------------------------------------
# Several of these are already the config defaults (max_depth, prior, buffer
# dedup, functional isclose); they are spelled out so the recipe is explicit
# and survives future changes to the config defaults. Overriding a key with
# its default value does not change the config hash.
COMMON=(
    "env.max_depth=5"
    "env.functional_isclose=True"
    "proxy.prior_type=node_count"
    "buffer.check_diversity=True"
    "buffer.diversity_check_reward_similarity=-1"
    "policy.backward.shared_weights=True"
    "gflownet.optimizer.n_train_steps=$N_STEPS"
    "gflownet.optimizer.lr=$LR"
    "gflownet.optimizer.batch_size.forward=45"
    "gflownet.optimizer.batch_size.backward_replay=5"
    "gflownet.optimizer.batch_size.backward_dataset=0"
)

EXTRA=("$@")   # extra hydra overrides, forwarded unchanged

# ---------------------------------------------------------------------------
# Build the task list: (policy x dataset x split), policies in POLICIES order
# ---------------------------------------------------------------------------
tasks=()
cores_demanded=0
missing=0
for p in $POLICIES; do
    if [ -z "${CFG_OF[$p]:-}" ]; then
        echo "ERROR: unknown policy '$p' (known: ${!CFG_OF[*]})"
        exit 1
    fi
    for d in $DATASETS; do
        for s in $SPLITS; do
            tasks+=("$p $d $s")
            cores_demanded=$(( cores_demanded + CPUS_OF[$p] ))
            csv="$REPO/tests/data/tree/${d}/${d}_${s}.csv"
            if [ ! -f "$csv" ]; then
                echo "ERROR: dataset file not found: $csv"
                missing=$(( missing + 1 ))
            fi
        done
    done
done
n_tasks=${#tasks[@]}

echo "============================================================"
echo " Trillium classification policy benchmark (MLP vs Transformer)"
echo " Job                : ${SLURM_JOB_ID:-none} on $(hostname)"
echo " Datasets           : $DATASETS"
echo " Splits             : $SPLITS   seed: $SEED"
echo " Policies           : $POLICIES   (mlp: $CPUS_MLP cores, trfm: $CPUS_TRFM cores)"
echo " Tasks              : $n_tasks   cores demanded: $cores_demanded / $CORES_PER_NODE"
echo " Common overrides   : ${COMMON[*]}"
echo " Extra overrides    : ${EXTRA[*]:-none}"
echo " Runs root          : $RUNS_ROOT   campaign: $EXP_NAME"
echo "============================================================"

if (( missing > 0 )); then
    echo "ABORT: $missing dataset file(s) missing (see above). Nothing launched."
    exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
    echo " Task table (start order):"
    printf '   %-6s %-14s %-6s %-6s %s\n' policy dataset split cores config
    for t in "${tasks[@]}"; do
        read -r p d s <<< "$t"
        printf '   %-6s %-14s %-6s %-6s %s\n' "$p" "$d" "$s" "${CPUS_OF[$p]}" "${CFG_OF[$p]}"
    done
    echo " DRY_RUN=1 -- nothing launched."
    exit 0
fi

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

WORKER_LOGS="$RUNS_ROOT/$EXP_NAME/worker-logs/${SLURM_JOB_ID:-local}"
mkdir -p "$WORKER_LOGS"
STATUS_FILE="$WORKER_LOGS/status.txt"
: > "$STATUS_FILE"

run_one () {
    local p="$1" d="$2" s="$3"
    local log="$WORKER_LOGS/${d}_split${s}_${p}.out"
    DATASET="$d" EXP_CONFIG="${CFG_OF[$p]}" CPUS_PER_RUN="${CPUS_OF[$p]}" \
        bash "$CODE_DIR/drac/cls_tree_worker.sh" "$s" \
            "${COMMON[@]}" "${EXTRA[@]}" \
            > "$log" 2>&1
    echo "$? $d split$s $p ${CPUS_OF[$p]}cpu $log" >> "$STATUS_FILE"
}

# ---------------------------------------------------------------------------
# Core-budget scheduler. `wait -n` reaps one finished worker per call; a
# reaped pid is gone from the process table, so `kill -0` fails for exactly
# the workers whose cores can be handed back (unreaped ones are zombies and
# still answer, and the next `wait -n` returns for them immediately).
# ---------------------------------------------------------------------------
declare -A cores_of_pid=()
free_cores=$CORES_PER_NODE

reap_finished () {
    local pid
    (( ${#cores_of_pid[@]} == 0 )) && return
    for pid in "${!cores_of_pid[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            free_cores=$(( free_cores + cores_of_pid[$pid] ))
            unset "cores_of_pid[$pid]"
        fi
    done
}

for t in "${tasks[@]}"; do
    read -r p d s <<< "$t"
    need="${CPUS_OF[$p]}"
    while (( free_cores < need )); do
        wait -n
        reap_finished
    done
    run_one "$p" "$d" "$s" &
    cores_of_pid[$!]="$need"
    free_cores=$(( free_cores - need ))
    echo " $(date '+%Y-%m-%d %H:%M:%S') started $d split$s $p ($need cores, $free_cores free)"
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
