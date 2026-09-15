#!/bin/bash
#SBATCH --job-name=cls_pb_trfm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=12:00:00
#SBATCH --output=cls_pb_trfm-%j.out
# =============================================================================
# TRILLIUM submitter: EXTENSION of trillium_cls_policy_bench.sh -- TRANSFORMER
# policy with (a) a tempered posterior and (b) a separate backward trunk.
# Writes into the same CLS_POLICY_BENCH campaign as the base script.
# =============================================================================
#
# Every run uses the CLS_POLICY_BENCH transformer recipe
# (tree/trfm_classification_tree: max_depth 5, node_count prior, lr 0.001,
# batch 45 fwd / 5 replay-bwd / 0 dataset-bwd, functional-isclose replay
# dedup, 1000 steps, seed 0, splits 1-5) and changes exactly ONE setting
# relative to the base runs (beta 1.0, shared_weights True):
#
#   variant    steps  reward beta  policy.backward.shared_weights  datasets                        runs
#   sep_1000   1000   1.0          False (own transformer trunk)   raisin breast_cancer wine iris  20
#   temp_1000  1000   0.1          True                            raisin breast_cancer wine       15
#
# 35 runs x 12 cores = 420 cores: 16 runs start at once on the 192-core node,
# every further run starts as soon as one finishes (core-budget scheduler
# from trillium_cls_policy_bench.sh). Variants start in the order above and
# datasets slowest-first.
#
# Expected duration: in the base campaign a 1000-step transformer run on 16
# cores took 2.4 h (iris), 2.3-3.3 h (wine), 3.5-3.9 h (breast_cancer) and
# 3-4.8 h (raisin). 12 instead of 16 cores costs little (thread scaling is
# weak: a 4-core Mila iris run takes 1.9-2.7 h). sep_1000 adds a second
# transformer trunk for the backward policy and temp_1000 samples bigger
# trees, so count on 3-7 h per run. 35 runs in 16 slots = three waves =
# 10-20 h -> the full grid does NOT fit one 12 h job. Either split it over
# three nodes (recommended: one wave each, done in 3-7 h):
#
#   VARIANTS=temp_1000                                sbatch --export=ALL --account=def-alexhg $HOME/gflownet/drac/trillium_cls_policy_bench_ext_trfm.sh
#   VARIANTS=sep_1000 DATASETS="raisin breast_cancer" sbatch --export=ALL --account=def-alexhg $HOME/gflownet/drac/trillium_cls_policy_bench_ext_trfm.sh
#   VARIANTS=sep_1000 DATASETS="wine iris"            sbatch --export=ALL --account=def-alexhg $HOME/gflownet/drac/trillium_cls_policy_bench_ext_trfm.sh
#
# or submit the whole grid at once and resubmit the SAME command after the
# time limit: finished runs are skipped, cut-off runs resume from their last
# checkpoint (written every 500 steps).
#
# Usage (from $SCRATCH on Trillium!):
#   mkdir -p $SCRATCH/gflownet-logs && cd $SCRATCH
#   sbatch --export=ALL --account=def-alexhg \
#       $HOME/gflownet/drac/trillium_cls_policy_bench_ext_trfm.sh
#
# DATASETS, when set, FILTERS each variant's dataset list (so
# DATASETS="wine iris" runs sep_1000 on wine+iris and temp_1000 on wine only).
#
# Knobs are read from the environment. Set them as shell variables in front
# of sbatch AND pass --export=ALL: the two base-campaign jobs (2259367 and
# 2259371) both ran the full 40-run grid, i.e. their DATASETS value never
# reached the job, and both trained the same 40 run directories at the same
# time. Multi-word values also accept ':' as separator, so
# `--export=ALL,DATASETS=wine:iris` works where the spaces would not.
# ALWAYS check the "Task table" printed at the top of the job's .out file as
# soon as it starts, and never let two jobs share a task.
#
# DRY_RUN=1 prints the task table and exits without launching anything (fine
# on a login node).
#
# Knobs: EXP_NAME VARIANTS DATASETS SPLITS SEED LR CPUS_TRFM CORES_PER_NODE
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

# Start order = list order.
VARIANTS="${VARIANTS:-sep_1000 temp_1000}"
DATASETS="${DATASETS:-}"          # empty = each variant's own dataset list
SPLITS="${SPLITS:-1 2 3 4 5}"
LR="${LR:-0.001}"
CORES_PER_NODE="${CORES_PER_NODE:-192}"
CPUS_TRFM="${CPUS_TRFM:-12}"
DRY_RUN="${DRY_RUN:-0}"

# ':' is an alternative list separator (commas are eaten by --export).
VARIANTS="${VARIANTS//:/ }"
DATASETS="${DATASETS//:/ }"
SPLITS="${SPLITS//:/ }"

POLICY="trfm"
EXP_CONFIG="tree/trfm_classification_tree"

# --- Variants: tag -> "n_train_steps reward_beta shared_weights" ------------
declare -A VARIANT_OF=(
    [sep_1000]="1000 1.0 False"
    [temp_1000]="1000 0.1 True"
)
# Datasets of each variant, slowest first (base campaign timings).
declare -A DATASETS_OF=(
    [sep_1000]="raisin breast_cancer wine iris"
    [temp_1000]="raisin breast_cancer wine"
)

# --- Fixed overrides shared by every run (= trillium_cls_policy_bench.sh -----
# minus n_train_steps and shared_weights, which are per variant). Several are
# config defaults; they are spelled out so the recipe is explicit and
# survives future changes to the defaults. Overriding a key with its default
# value does not change the config hash.
COMMON=(
    "env.max_depth=5"
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

in_list () {   # in_list needle "a b c"
    local x
    for x in $2; do [ "$x" = "$1" ] && return 0; done
    return 1
}

# ---------------------------------------------------------------------------
# Build the task list: (variant x its datasets x split) in list order.
# Each task: "tag dataset split steps beta shared"
# ---------------------------------------------------------------------------
tasks=()
cores_demanded=0
missing=0
for tag in $VARIANTS; do
    if [ -z "${VARIANT_OF[$tag]:-}" ]; then
        echo "ERROR: unknown variant '$tag' (known: ${!VARIANT_OF[*]})"
        exit 1
    fi
    read -r steps beta shared <<< "${VARIANT_OF[$tag]}"
    for d in ${DATASETS_OF[$tag]}; do
        if [ -n "$DATASETS" ] && ! in_list "$d" "$DATASETS"; then
            continue
        fi
        for s in $SPLITS; do
            tasks+=("$tag $d $s $steps $beta $shared")
            cores_demanded=$(( cores_demanded + CPUS_TRFM ))
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
echo " Trillium classification policy benchmark EXTENSION -- Transformer"
echo " Job                : ${SLURM_JOB_ID:-none} on $(hostname)"
echo " Variants           : $VARIANTS"
echo " Dataset filter     : ${DATASETS:-none, each variant uses its own list}"
echo " Splits             : $SPLITS   seed: $SEED"
echo " Policy             : $POLICY ($EXP_CONFIG, $CPUS_TRFM cores/run -> $(( CORES_PER_NODE / CPUS_TRFM )) concurrent)"
echo " Tasks              : $n_tasks   cores demanded: $cores_demanded / $CORES_PER_NODE"
echo " Common overrides   : ${COMMON[*]}"
echo " Extra overrides    : ${EXTRA[*]:-none}"
echo " Runs root          : $RUNS_ROOT   campaign: $EXP_NAME"
echo "============================================================"
echo " Task table (start order):"
printf '   %-10s %-14s %-6s %-6s %-5s %-7s %s\n' variant dataset split steps beta shared cores
for t in "${tasks[@]}"; do
    read -r tag d s steps beta shared <<< "$t"
    printf '   %-10s %-14s %-6s %-6s %-5s %-7s %s\n' "$tag" "$d" "$s" "$steps" "$beta" "$shared" "$CPUS_TRFM"
done
echo "============================================================"

if (( n_tasks == 0 )); then
    echo "ABORT: the DATASETS filter '$DATASETS' left no task for variants '$VARIANTS'."
    exit 1
fi
if (( missing > 0 )); then
    echo "ABORT: $missing dataset file(s) missing (see above). Nothing launched."
    exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
    echo " DRY_RUN=1 -- nothing launched."
    exit 0
fi

# Two jobs training the same run directory at the same time corrupt it (see
# the base campaign). Warn if another job of this name is running.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    others="$(squeue -h -u "$USER" -n "${SLURM_JOB_NAME:-cls_pb_trfm}" -t RUNNING -o %i 2>/dev/null \
              | grep -vx "$SLURM_JOB_ID" | tr '\n' ' ')"
    if [ -n "$others" ]; then
        echo " WARNING: other running ${SLURM_JOB_NAME:-cls_pb_trfm} job(s): $others"
        echo "          Make sure their task tables do not overlap with this one."
    fi
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
    local tag="$1" d="$2" s="$3" steps="$4" beta="$5" shared="$6"
    local log="$WORKER_LOGS/${d}_split${s}_${POLICY}_${tag}.out"
    DATASET="$d" EXP_CONFIG="$EXP_CONFIG" CPUS_PER_RUN="$CPUS_TRFM" \
        bash "$CODE_DIR/drac/cls_tree_worker.sh" "$s" \
            "${COMMON[@]}" \
            "gflownet.optimizer.n_train_steps=$steps" \
            "proxy.reward_function_kwargs.beta=$beta" \
            "policy.backward.shared_weights=$shared" \
            "${EXTRA[@]}" \
            > "$log" 2>&1
    echo "$? $d split$s $POLICY $tag ${CPUS_TRFM}cpu $log" >> "$STATUS_FILE"
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
    read -r tag d s steps beta shared <<< "$t"
    need="$CPUS_TRFM"
    while (( free_cores < need )); do
        wait -n
        reap_finished
    done
    run_one "$tag" "$d" "$s" "$steps" "$beta" "$shared" &
    cores_of_pid[$!]="$need"
    free_cores=$(( free_cores - need ))
    echo " $(date '+%Y-%m-%d %H:%M:%S') started $d split$s $tag ($steps steps, beta $beta, shared $shared, $need cores, $free_cores free)"
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
