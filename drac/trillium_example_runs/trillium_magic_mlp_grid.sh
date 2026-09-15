#!/bin/bash
#SBATCH --job-name=magic_mlp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=12:00:00
#SBATCH --output=magic_mlp-%j.out
# =============================================================================
# TRILLIUM submitter: MLP policy SIZE x shared_weights ablation on MAGIC.
# =============================================================================
#
# Base recipe = the MAGIC_STAB2 winner (2026-09): tempered posterior beta=0.1,
# epsilon-greedy annealing 0.1 -> 0.01 over the run, Adam lr 1e-3, batch
# 45 fwd / 5 bwd-replay, depth 5, 10k steps, full replay dedup
# (check_diversity + functional Tree.isclose). Only the MLP architecture and
# the shared_weights flag vary:
#
#   tag            fwd+bwd MLP     shared_weights
#   h128x4_shared  128 x 4 layers  True
#   h128x4_sep     128 x 4 layers  False
#   h64x5_shared    64 x 5 layers  True
#   h64x5_sep       64 x 5 layers  False
#
# The 256 x 3 (yaml default) reference at this exact recipe already exists:
# campaign MAGIC_STAB2_B0.1_EPSANNEAL on Mila (splits 1-3, 10k and 20k steps)
# -- compare against it in the aggregator instead of re-running a 5th variant.
# Backward n_hid/n_layers are set even for the shared arms (where the trunk
# is shared and they may be ignored) so that each size pair differs from its
# partner in shared_weights ONLY.
#
# 4 variants x 3 splits = 12 runs at 16 cores each = 192/192 cores. Note the
# policy is NOT the compute bottleneck (reward computation over n~15k rows
# is), so the smaller MLPs won't run much faster than 256x3. If the 12 h
# walltime cuts runs off, resubmit the same command: finished runs are
# skipped, cut-off ones resume from their checkpoints.
#
# BEFORE LAUNCHING: the branch state matters -- epsilon_annealing keys, the
# buffer dedup configs and the functional Tree.isclose live in code/configs.
# Commit + sync the branch to the Trillium checkout first, and make sure
# tests/data/tree/magic/magic_{1,2,3}.csv are present there (the script
# aborts before launching anything if they are not).
#
# Usage (from $SCRATCH on Trillium!):
#   mkdir -p $SCRATCH/gflownet-logs && cd $SCRATCH
#   DRY_RUN=1 bash $HOME/gflownet/drac/trillium_magic_mlp_grid.sh   # check table
#   sbatch --account=<acct> $HOME/gflownet/drac/trillium_magic_mlp_grid.sh
#
# Knobs: EXP_NAME EXP_CONFIG DATASET SPLITS SEED RUNS_ROOT FORCE CPUS_PER_RUN
#        DRY_RUN
# Extra hydra overrides given on the command line are forwarded to every run.
# =============================================================================

set -u

REPO="${REPO:-$HOME/gflownet}"
VENV="${VENV:-$SCRATCH/venvs/gflownet-env}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
export EXP_CONFIG="${EXP_CONFIG:-tree/classification_tree}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export DATASET="${DATASET:-magic}"
export EXP_NAME="${EXP_NAME:-MAGIC_MLP_SIZE}"
export SEED="${SEED:-0}"
export FORCE="${FORCE:-0}"

SPLITS="${SPLITS:-1 2 3}"
CORES_PER_NODE=192
# 12 tasks at 16 cores each -> all run at once (192/192 cores).
export CPUS_PER_RUN="${CPUS_PER_RUN:-16}"
DRY_RUN="${DRY_RUN:-0}"

# --- The grid: "tag hydra-override [hydra-override ...]" per variant --------
VARIANTS=(
    "h128x4_shared policy.forward.n_hid=128 policy.forward.n_layers=4 policy.backward.n_hid=128 policy.backward.n_layers=4 policy.backward.shared_weights=True"
    "h128x4_sep    policy.forward.n_hid=128 policy.forward.n_layers=4 policy.backward.n_hid=128 policy.backward.n_layers=4 policy.backward.shared_weights=False"
    "h64x5_shared  policy.forward.n_hid=64  policy.forward.n_layers=5 policy.backward.n_hid=64  policy.backward.n_layers=5 policy.backward.shared_weights=True"
    "h64x5_sep     policy.forward.n_hid=64  policy.forward.n_layers=5 policy.backward.n_hid=64  policy.backward.n_layers=5 policy.backward.shared_weights=False"
)

# --- Fixed overrides shared by every run: the MAGIC_STAB2 winner recipe -----
# Dedup / prior / depth keys are already the config defaults; they are spelled
# out so the recipe is explicit and survives future default changes (a
# same-value override does not change the config hash).
COMMON=(
    "policy.forward.type=mlp"
    "policy.backward.type=mlp"
    "env.max_depth=5"
    "env.functional_isclose=True"
    "buffer.check_diversity=True"
    "buffer.diversity_check_reward_similarity=-1"
    "proxy.reward_function_kwargs.beta=0.1"
    "gflownet.random_action_prob=0.1"
    "gflownet.epsilon_annealing.enabled=True"
    "gflownet.optimizer.n_train_steps=10000"
    "gflownet.optimizer.lr=0.001"
    "gflownet.optimizer.batch_size.forward=45"
    "gflownet.optimizer.batch_size.backward_replay=5"
    "gflownet.optimizer.batch_size.backward_dataset=0"
    "gflownet.optimizer.lr_decay_period=1000000"
)

EXTRA=("$@")   # extra hydra overrides, forwarded unchanged

# ---------------------------------------------------------------------------
# Build the task list: (variant x split); abort early on missing data files
# ---------------------------------------------------------------------------
tasks=()
missing=0
for v in "${VARIANTS[@]}"; do
    for s in $SPLITS; do
        tasks+=("$s $v")
    done
done
for s in $SPLITS; do
    csv="$REPO/tests/data/tree/${DATASET}/${DATASET}_${s}.csv"
    if [ ! -f "$csv" ]; then
        echo "ERROR: dataset file not found: $csv"
        missing=$(( missing + 1 ))
    fi
done
n_tasks=${#tasks[@]}

CONCURRENCY=$(( CORES_PER_NODE / CPUS_PER_RUN ))
(( CONCURRENCY < 1 )) && CONCURRENCY=1

used=$(( n_tasks < CONCURRENCY ? n_tasks * CPUS_PER_RUN : CORES_PER_NODE ))
echo "============================================================"
echo " Trillium MLP size x shared_weights ablation"
echo " Job                : ${SLURM_JOB_ID:-none} on $(hostname)"
echo " Dataset            : $DATASET   splits: $SPLITS   seed: $SEED"
echo " Variants           : ${#VARIANTS[@]}   tasks: $n_tasks"
echo " Threads per run    : $CPUS_PER_RUN   concurrency: $CONCURRENCY"
echo " Node utilisation   : ~$used / $CORES_PER_NODE cores"
echo " Common overrides   : ${COMMON[*]}"
echo " Extra overrides    : ${EXTRA[*]:-none}"
echo " Runs root          : $RUNS_ROOT   campaign: $EXP_NAME"
echo "============================================================"

if (( missing > 0 )); then
    echo "ABORT: $missing dataset file(s) missing (see above). Nothing launched."
    exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
    echo " Task table:"
    printf '   %-6s %-14s %s\n' split tag overrides
    for t in "${tasks[@]}"; do
        read -r s tag rest <<< "$t"
        printf '   %-6s %-14s %s\n' "$s" "$tag" "$rest"
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
