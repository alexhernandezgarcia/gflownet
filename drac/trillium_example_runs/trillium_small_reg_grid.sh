#!/bin/bash
#SBATCH --job-name=small_reg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=24:00:00
#SBATCH --output=small_reg-%j.out
# =============================================================================
# TRILLIUM submitter: prior-strength / capacity grid on the SMALL regression
# datasets (qsar_fish 908, qsar_aquatic 546, real_estate 414, slump 103).
# =============================================================================
#
# Context (2026-09-02, REG_NEW_* on Mila, 10k steps): bcart phi=0.5 (19-28
# nodes) >= node_count (6-8 nodes) on all four datasets, but only the
# qsar_aquatic gap (+0.11 R2) exceeds the +-0.03-0.10 split noise; both are
# 0.05-0.15 R2 below boosted/BART baselines. The 10k node_count and bcart
# phi=0.5 arms already exist for all four (same config hash -> comparable),
# so this grid fills the size range BETWEEN them and tests capacity and the
# requested likelihood normalisation:
#
#   bcart_p1.0      Chipman prior, phi=1.0: 18 nodes on diabetes, 25 on
#                   concrete -- the intermediate-size arm
#   exp_b2.0        exponential prior, 2.0 nats/split. node_count IS the
#                   exponential prior with beta=log(4*n_features) = 3.2-3.5
#                   nats here, so beta=2.0 is a continuous dial toward
#                   larger trees (never run before)
#   exp_b1.0        1.0 nat/split -- weaker still
#   d6_bcart05      max_depth 6 + bcart phi=0.5 (capacity: qsar_fish sits at
#                   ~20 of 31 nodes, real_estate/slump at 23-28)
#   normlik_nc      normalize_likelihood=True + node_count (requested arm;
#                   expect very small trees -- see concrete script header)
#   normlik_exp0.1  normalize_likelihood=True + exponential beta=0.1, prior
#                   rescaled to the normalized likelihood's units
#
# 6 arms x 4 datasets x 3 splits = 72 runs at 8 cores, 24 concurrent. Tasks
# are ordered dataset-major, LARGEST dataset first, so the 18 slow qsar_fish
# runs (~9 h) start in the first wave; the whole grid fits in 24 h. If it
# does not, resubmit the same command (finished runs skipped, others resume).
#
# Usage (from $SCRATCH on Trillium!):
#   cd $SCRATCH && sbatch --account=<acct> $HOME/gflownet/drac/trillium_small_reg_grid.sh
#
# Knobs: EXP_NAME DATASETS SPLITS N_STEPS RUNS_ROOT TOP_K FORCE CPUS_PER_RUN
# =============================================================================

set -u

REPO="${REPO:-$HOME/gflownet}"
VENV="${VENV:-$SCRATCH/venvs/gflownet-env}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
export EXP_NAME="${EXP_NAME:-REG_SMALL_GRID}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SEED=0

# Largest first: they dominate the wall time.
DATASETS="${DATASETS:-qsar_fish qsar_aquatic real_estate slump}"
SPLITS="${SPLITS:-1 2 3}"
N_STEPS="${N_STEPS:-10000}"
CORES_PER_NODE=192
export CPUS_PER_RUN="${CPUS_PER_RUN:-8}"

# --- The grid: "tag hydra-override [hydra-override ...]" per variant --------
VARIANTS=(
    "bcart_p1.0     proxy.prior_type=bcart proxy.sigma=0.95 proxy.phi=1.0"
    "exp_b2.0       proxy.prior_type=exponential proxy.beta=2.0"
    "exp_b1.0       proxy.prior_type=exponential proxy.beta=1.0"
    "d6_bcart05     env.max_depth=6 proxy.prior_type=bcart proxy.sigma=0.95 proxy.phi=0.5"
    "normlik_nc     proxy.normalize_likelihood=True proxy.prior_type=node_count"
    "normlik_exp0.1 proxy.normalize_likelihood=True proxy.prior_type=exponential proxy.beta=0.1"
)

# --- Fixed recipe (same as REG_NEW_* / trillium_prior_grid.sh) --------------
COMMON=(
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

# Task list: dataset-major (largest first), then variant, then split.
tasks=()
for d in $DATASETS; do
    for v in "${VARIANTS[@]}"; do
        for s in $SPLITS; do
            tasks+=("$d $s $v")
        done
    done
done
n_tasks=${#tasks[@]}

CONCURRENCY=$(( CORES_PER_NODE / CPUS_PER_RUN ))
(( CONCURRENCY < 1 )) && CONCURRENCY=1
used=$(( n_tasks < CONCURRENCY ? n_tasks * CPUS_PER_RUN : CORES_PER_NODE ))

echo "============================================================"
echo " Trillium small-regression prior/capacity grid"
echo " Job                : ${SLURM_JOB_ID:-none} on $(hostname)"
echo " Datasets           : $DATASETS   splits: $SPLITS   seed: $SEED   steps: $N_STEPS"
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

EXTRA=("$@")

run_one () {
    local d="$1" s="$2" tag="$3"; shift 3
    local var_overrides=("$@")
    local log="$WORKER_LOGS/${d}_split${s}_${tag}.out"
    DATASET="$d" bash "$CODE_DIR/drac/reg_tree_worker.sh" "$s" \
        "${COMMON[@]}" "${var_overrides[@]}" "${EXTRA[@]}" \
        > "$log" 2>&1
    echo "$? $d split$s $tag $log" >> "$STATUS_FILE"
}

running=0
for t in "${tasks[@]}"; do
    read -r d s tag rest <<< "$t"
    read -ra var_overrides <<< "$rest"
    run_one "$d" "$s" "$tag" "${var_overrides[@]}" &
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
