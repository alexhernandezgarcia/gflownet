#!/bin/bash
#SBATCH --job-name=unif_policy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=24:00:00
#SBATCH --output=unif_policy-%j.out
# =============================================================================
# TRILLIUM submitter: UNIFORM-REWARD validation grid, 4 policies x 5 depths,
# classification trees on ONE dataset split (default iris split 2).
# =============================================================================
#
# Uniform reward: proxy.reward_function_kwargs.beta=0 turns the exponential
# reward into R = alpha * exp(0) = 1 for every tree, so the GFlowNet has to
# learn the uniform distribution over all trees up to max_depth and logZ must
# converge to log(#trees). Same convention as the Mila DEBUG_UNIFORM campaign
# (root split required, continuous thresholds count as 1): for iris the
# targets log(N(D)-1) are D1 1.39, D2 4.61, D3 10.62, D4 22.62, D5 46.63
# (helpers_for_experiments/calculate_nbr_of_trees.py -d <D> -F 4 -T 1).
#
# Grid: 4 policy variants x 5 depths = 20 runs, one split, seed 0.
#
#   variant       policy  config                          shared_weights
#   mlp_shared    mlp     tree/classification_tree        True
#   mlp_sep       mlp     tree/classification_tree        False
#   trfm_shared   trfm    tree/trfm_classification_tree   True
#   trfm_sep      trfm    tree/trfm_classification_tree   False
#
#   depth  steps   cores/run (mlp / trfm)
#   1       1000    8 / 16
#   2       1000    8 / 16
#   3       5000    8 / 16
#   4      10000    8 / 16
#   5      50000   24 / 24   (LONG_DEPTHS, started first)
#
# Everything else is the classification_tree recipe: lr 0.001, batch 45 fwd /
# 5 replay-bwd / 0 dataset-bwd, node_count prior (irrelevant at beta=0), dedup
# replay buffer with the functional Tree.isclose.
#
# Packing (core budget, tasks started in list order, next task waits until
# enough cores are free):
#   t=0   4 depth-5 runs x 24 cores = 96 cores, for the whole job
#       + 4 depth-1 + 4 depth-2 runs = 2x(2x8 + 2x16) = 96 cores
#   then  depth-3 runs, then depth-4 runs, each starting in the slot a
#         finished run frees up.
#
# Expected durations, extrapolated from the Mila DEBUG_UNIFORM runs (iris,
# batch 90/10, 4 cores, step time flat from step 500 on: d3 mlp 1.3 s/it,
# trfm 3.6; d4 mlp 3, trfm 9.5; d5 mlp 7, trfm 14) at half the batch:
#   depth 1-2:   minutes            depth 3: mlp ~1 h,  trfm ~3 h
#   depth 4:     mlp ~5 h, trfm ~14 h (starts after depth 3 -> done by ~17 h)
#   depth 5:     mlp ~2-3 DAYS, trfm ~4-8 DAYS  -> does NOT fit one 24 h job
# The depth-5 step time is dominated by the python env machinery (full
# 31-node trees, ~1.7 MiB per state only matters for memory), so the 24 cores
# barely help; the runs simply need several jobs. RESUBMIT the same sbatch
# command after each job ends: finished runs are skipped, the depth-5 runs
# resume from their last checkpoint (checkpoints every 500 steps).
#
# Usage (from $SCRATCH on Trillium!):
#   mkdir -p $SCRATCH/gflownet-logs && cd $SCRATCH
#   sbatch --account=def-alexhg $HOME/gflownet/drac/trillium_uniform_policy_grid.sh
#
# DRY_RUN=1 prints the task table and exits (fine on a login node).
#
# Knobs: EXP_NAME DATASET SPLIT SEED LR DEPTH_STEPS LONG_DEPTHS CPUS_LONG
#        CPUS_MLP CPUS_TRFM VARIANTS_SEL RUNS_ROOT FORCE DRY_RUN
# Extra hydra overrides given on the command line are forwarded to every run
# (e.g. evaluator.period=500 to cut the 1000-sample evaluation overhead).
# =============================================================================

set -u

REPO="${REPO:-$HOME/gflownet}"
VENV="${VENV:-$SCRATCH/venvs/gflownet-env}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
export EXP_NAME="${EXP_NAME:-DEBUG_UNIFORM_POLICY}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export DATASET="${DATASET:-iris}"
export SEED="${SEED:-0}"
export FORCE="${FORCE:-0}"

SPLIT="${SPLIT:-2}"
LR="${LR:-0.001}"
# "depth:steps" pairs; the order here is the start order of the non-long depths.
DEPTH_STEPS="${DEPTH_STEPS:-1:1000 2:1000 3:5000 4:10000 5:50000}"
# Depths that get CPUS_LONG cores each and are started before everything else.
LONG_DEPTHS="${LONG_DEPTHS:-5}"
CPUS_LONG="${CPUS_LONG:-24}"
CPUS_MLP="${CPUS_MLP:-8}"
CPUS_TRFM="${CPUS_TRFM:-16}"
# Subset of variant tags to run (default: all four).
VARIANTS_SEL="${VARIANTS_SEL:-mlp_shared mlp_sep trfm_shared trfm_sep}"
CORES_PER_NODE="${CORES_PER_NODE:-192}"
DRY_RUN="${DRY_RUN:-0}"

# --- Policy variants: tag -> "policy shared_weights" ------------------------
declare -A VARIANT_OF=(
    [mlp_shared]="mlp True"
    [mlp_sep]="mlp False"
    [trfm_shared]="trfm True"
    [trfm_sep]="trfm False"
)
declare -A CFG_OF=(
    [mlp]="tree/classification_tree"
    [trfm]="tree/trfm_classification_tree"
)
declare -A CPUS_OF=(
    [mlp]="$CPUS_MLP"
    [trfm]="$CPUS_TRFM"
)

# --- Fixed overrides shared by every run ------------------------------------
# beta=0 is the point of the campaign; the rest restates the classification
# recipe explicitly (overriding a key with its default does not change the
# config hash). Depth, steps and shared_weights are per task.
COMMON=(
    "proxy.reward_function_kwargs.beta=0"
    "env.functional_isclose=True"
    "proxy.prior_type=node_count"
    "buffer.check_diversity=True"
    "buffer.diversity_check_reward_similarity=-1"
    "gflownet.optimizer.lr=$LR"
    "gflownet.optimizer.batch_size.forward=45"
    "gflownet.optimizer.batch_size.backward_replay=5"
    "gflownet.optimizer.batch_size.backward_dataset=0"
)

EXTRA=("$@")   # extra hydra overrides, forwarded unchanged

csv="$REPO/tests/data/tree/${DATASET}/${DATASET}_${SPLIT}.csv"
if [ ! -f "$csv" ]; then
    echo "ERROR: dataset file not found: $csv"
    exit 1
fi

is_long_depth () {
    local d
    for d in $LONG_DEPTHS; do [ "$d" = "$1" ] && return 0; done
    return 1
}

steps_of_depth () {
    local pair
    for pair in $DEPTH_STEPS; do
        [ "${pair%%:*}" = "$1" ] && { echo "${pair##*:}"; return 0; }
    done
    return 1
}

# ---------------------------------------------------------------------------
# Build the task list: long depths first, then the other depths in
# DEPTH_STEPS order; within a depth, the variants in VARIANTS_SEL order.
# Each task: "cores depth steps tag policy shared"
# ---------------------------------------------------------------------------
tasks=()
cores_demanded=0
add_depth_tasks () {
    local d="$1" steps tag policy shared cores
    steps="$(steps_of_depth "$d")" || { echo "ERROR: no steps for depth $d"; exit 1; }
    for tag in $VARIANTS_SEL; do
        if [ -z "${VARIANT_OF[$tag]:-}" ]; then
            echo "ERROR: unknown variant '$tag' (known: ${!VARIANT_OF[*]})"
            exit 1
        fi
        read -r policy shared <<< "${VARIANT_OF[$tag]}"
        if is_long_depth "$d"; then cores="$CPUS_LONG"; else cores="${CPUS_OF[$policy]}"; fi
        tasks+=("$cores $d $steps $tag $policy $shared")
        cores_demanded=$(( cores_demanded + cores ))
    done
}
for pair in $DEPTH_STEPS; do
    d="${pair%%:*}"
    is_long_depth "$d" && add_depth_tasks "$d"
done
for pair in $DEPTH_STEPS; do
    d="${pair%%:*}"
    is_long_depth "$d" || add_depth_tasks "$d"
done
n_tasks=${#tasks[@]}

echo "============================================================"
echo " Trillium uniform-reward policy grid (beta=0)"
echo " Job                : ${SLURM_JOB_ID:-none} on $(hostname)"
echo " Dataset            : $DATASET split $SPLIT   seed: $SEED"
echo " Depth:steps        : $DEPTH_STEPS   (long depths: $LONG_DEPTHS @ $CPUS_LONG cores)"
echo " Variants           : $VARIANTS_SEL   (mlp: $CPUS_MLP cores, trfm: $CPUS_TRFM cores)"
echo " Tasks              : $n_tasks   cores demanded: $cores_demanded / $CORES_PER_NODE"
echo " Common overrides   : ${COMMON[*]}"
echo " Extra overrides    : ${EXTRA[*]:-none}"
echo " Runs root          : $RUNS_ROOT   campaign: $EXP_NAME"
echo "============================================================"

if [ "$DRY_RUN" = "1" ]; then
    echo " Task table (start order):"
    printf '   %-6s %-7s %-12s %-6s %-7s %s\n' depth steps variant cores shared config
    for t in "${tasks[@]}"; do
        read -r cores d steps tag policy shared <<< "$t"
        printf '   %-6s %-7s %-12s %-6s %-7s %s\n' "$d" "$steps" "$tag" "$cores" "$shared" "${CFG_OF[$policy]}"
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
    local cores="$1" d="$2" steps="$3" tag="$4" policy="$5" shared="$6"
    local log="$WORKER_LOGS/${DATASET}${SPLIT}_depth${d}_${tag}.out"
    EXP_CONFIG="${CFG_OF[$policy]}" CPUS_PER_RUN="$cores" \
        bash "$CODE_DIR/drac/cls_tree_worker.sh" "$SPLIT" \
            "${COMMON[@]}" \
            "env.max_depth=$d" \
            "gflownet.optimizer.n_train_steps=$steps" \
            "policy.backward.shared_weights=$shared" \
            "${EXTRA[@]}" \
            > "$log" 2>&1
    echo "$? depth$d $tag ${cores}cpu $log" >> "$STATUS_FILE"
}

# ---------------------------------------------------------------------------
# Core-budget scheduler (same as trillium_cls_policy_bench.sh). `wait -n`
# reaps one finished worker per call; a reaped pid is gone from the process
# table, so `kill -0` fails for exactly the workers whose cores can be handed
# back (unreaped ones are zombies and still answer, and the next `wait -n`
# returns for them immediately).
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
    read -r cores d steps tag policy shared <<< "$t"
    while (( free_cores < cores )); do
        wait -n
        reap_finished
    done
    run_one "$cores" "$d" "$steps" "$tag" "$policy" "$shared" &
    cores_of_pid[$!]="$cores"
    free_cores=$(( free_cores - cores ))
    echo " $(date '+%Y-%m-%d %H:%M:%S') started depth$d $tag ($steps steps, $cores cores, $free_cores free)"
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
