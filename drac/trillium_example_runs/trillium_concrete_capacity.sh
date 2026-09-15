#!/bin/bash
#SBATCH --job-name=concrete_cap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=24:00:00
#SBATCH --output=concrete_cap-%j.out
# =============================================================================
# TRILLIUM submitter: CONCRETE capacity campaign (40k steps, 24 h node).
# =============================================================================
#
# Motivation (2026-09-02): on concrete every prior grows trees to the depth-5
# cap (28-31 of 31 nodes), test R2 keeps rising with training length
# (10k 0.63 -> 20k 0.74 -> ~27k 0.76, bcart phi=0.5), and a single PRUNED
# CART with unbounded depth (0.815) beats our depth-5 posterior ensemble
# (~0.76). That points at model CAPACITY (max_depth) and convergence, not at
# the prior. Two groups of arms:
#
#  (1) RESUME group -- campaign REG_PRIOR_CONCRETE, identical overrides to
#      trillium_prior_grid.sh at N_STEPS=40000: 10 of its 24 runs were cut by
#      the 12 h walltime at 24k-39k steps (all bcart_p0.5 and noprior runs
#      among them). Same config hash -> same run dir -> the worker resumes
#      them from their checkpoints and skips the 14 finished ones.
#
#  (2) NEW group -- campaign REG_CONCRETE_CAP:
#      d6_bcart05      max_depth 6 (63 nodes) + bcart phi=0.5  (capacity)
#      d6_bcart10      max_depth 6 + bcart phi=1.0               (capacity,
#                      milder prior)
#      d6_nodecount    max_depth 6 + node_count                 (capacity
#                      control)
#      normlik_nc      normalize_likelihood=True + node_count: the
#                      requested "stop the O(N) likelihood from dominating
#                      the O(1) prior" arm. Expect SMALL trees: the
#                      per-sample likelihood gain of a split is O(0.1) nats
#                      while node_count charges log(4*8)=3.5 nats per split.
#      normlik_exp0.1  normalize_likelihood=True + exponential prior with
#                      beta=0.1 nats/split, i.e. a prior rescaled to the
#                      same O(1)-per-sample units as the normalized
#                      likelihood -- the version of the requested arm whose
#                      prior/likelihood balance is not degenerate a priori.
#      NOTE: normalized-likelihood runs do NOT sample the Bayesian posterior
#      (1/N temperature on the likelihood only); judge them by test RMSE/R2,
#      not by posterior-fidelity quantities.
#
# 30 tasks (15 new + 15 resume/skip) at 8 cores each; 24 concurrent. The
# depth-6 40k runs may exceed 24 h: RESUBMIT this same sbatch command --
# finished runs are skipped, cut-off ones resume (idempotent).
#
# Usage (from $SCRATCH on Trillium!):
#   cd $SCRATCH && sbatch --account=<acct> $HOME/gflownet/drac/trillium_concrete_capacity.sh
#
# Knobs: DATASET SPLITS N_STEPS RUNS_ROOT TOP_K FORCE CPUS_PER_RUN
# =============================================================================

set -u

REPO="${REPO:-$HOME/gflownet}"
VENV="${VENV:-$SCRATCH/venvs/gflownet-env}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
export DATASET="${DATASET:-concrete}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export SEED=0

SPLITS="${SPLITS:-1 2 3}"
N_STEPS="${N_STEPS:-40000}"
CORES_PER_NODE=192
export CPUS_PER_RUN="${CPUS_PER_RUN:-8}"

NEW_EXP="REG_CONCRETE_CAP"
RESUME_EXP="REG_PRIOR_CONCRETE"

# --- Variants: "tag EXP_NAME=<campaign> hydra-override ..." -----------------
# New (long) arms first so they occupy slots from the start; the resume arms
# have <= 16k steps left and finish early, freeing slots for the queue.
VARIANTS=(
    "d6_bcart05     EXP_NAME=$NEW_EXP env.max_depth=6 proxy.prior_type=bcart proxy.sigma=0.95 proxy.phi=0.5"
    "d6_bcart10     EXP_NAME=$NEW_EXP env.max_depth=6 proxy.prior_type=bcart proxy.sigma=0.95 proxy.phi=1.0"
    "d6_nodecount   EXP_NAME=$NEW_EXP env.max_depth=6 proxy.prior_type=node_count"
    "normlik_nc     EXP_NAME=$NEW_EXP proxy.normalize_likelihood=True proxy.prior_type=node_count"
    "normlik_exp0.1 EXP_NAME=$NEW_EXP proxy.normalize_likelihood=True proxy.prior_type=exponential proxy.beta=0.1"
    "bcart_p0.5     EXP_NAME=$RESUME_EXP proxy.prior_type=bcart proxy.sigma=0.95 proxy.phi=0.5"
    "bcart_p1.0     EXP_NAME=$RESUME_EXP proxy.prior_type=bcart proxy.sigma=0.95 proxy.phi=1.0"
    "bcart_p1.5     EXP_NAME=$RESUME_EXP proxy.prior_type=bcart proxy.sigma=0.95 proxy.phi=1.5"
    "bcart_p2.0     EXP_NAME=$RESUME_EXP proxy.prior_type=bcart proxy.sigma=0.95 proxy.phi=2.0"
    "noprior        EXP_NAME=$RESUME_EXP proxy.prior_type=none"
)

# --- Fixed recipe (MUST stay identical to trillium_prior_grid.sh for the
# resume group to hash onto the existing run directories) -------------------
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
echo " Trillium concrete capacity campaign"
echo " Job                : ${SLURM_JOB_ID:-none} on $(hostname)"
echo " Dataset            : $DATASET   splits: $SPLITS   seed: $SEED   steps: $N_STEPS"
echo " Variants           : ${#VARIANTS[@]}   tasks: $n_tasks   (new -> $NEW_EXP, resume -> $RESUME_EXP)"
echo " Threads per run    : $CPUS_PER_RUN   concurrency: $CONCURRENCY"
echo " Node utilisation   : ~$used / $CORES_PER_NODE cores"
echo " Common overrides   : ${COMMON[*]}"
echo "============================================================"

WORKER_LOGS="$RUNS_ROOT/$NEW_EXP/worker-logs/${SLURM_JOB_ID:-local}"
mkdir -p "$WORKER_LOGS"
STATUS_FILE="$WORKER_LOGS/status.txt"
: > "$STATUS_FILE"

EXTRA=("$@")

run_one () {
    local s="$1" tag="$2"; shift 2
    local exp="$NEW_EXP" var_overrides=()
    for tok in "$@"; do
        case "$tok" in
            EXP_NAME=*) exp="${tok#EXP_NAME=}" ;;
            *) var_overrides+=("$tok") ;;
        esac
    done
    local log="$WORKER_LOGS/${DATASET}_split${s}_${tag}.out"
    EXP_NAME="$exp" bash "$CODE_DIR/drac/reg_tree_worker.sh" "$s" \
        "${COMMON[@]}" "${var_overrides[@]}" "${EXTRA[@]}" \
        > "$log" 2>&1
    echo "$? $DATASET split$s $tag [$exp] $log" >> "$STATUS_FILE"
}

running=0
for t in "${tasks[@]}"; do
    read -r s tag rest <<< "$t"
    read -ra toks <<< "$rest"
    run_one "$s" "$tag" "${toks[@]}" &
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
