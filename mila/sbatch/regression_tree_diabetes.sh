#!/bin/bash
#SBATCH --job-name=reg_tree_diabetes
#SBATCH --output=/home/mila/a/arnit/scratch/gflownet-logs/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --partition=long-cpu,long-cpu-eek
#SBATCH --array=1-3

# Basic regression-tree training runs on the diabetes dataset: one array task
# per train/test split (task id == split id, splits 1-3).
#
# Deliberately minimal compared to trfm_tree_remaining.sh: no checkpoint
# resume and no --requeue (a preempted/failed task restarts from scratch, so
# resubmit it manually with e.g. `sbatch --array=2 <this script>`). A task
# whose run dir already contains samples/gfn_samples.pkl exits immediately.
#
# Note: the final eval script gflownet/envs/tree/eval_tree.py is
# classification-only (accuracy-based) and is NOT run here; test RMSE / R2
# metrics are logged to wandb during training by the tree evaluator
# (RegressionTree.test), since the split CSVs carry a test set.
#
# Usage (from anywhere; the script submits itself):
#   bash mila/sbatch/regression_tree_diabetes.sh
#
# Tunables via environment variables (forwarded by --export=ALL):
#   SEED=0  DEPTH=5  N_TRAIN_STEPS=10000  WANDB_ONLINE=True
#   WORK_DIR=$SCRATCH/gflownet-logs/regression_tree

set -u

REPO="$HOME/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"
SLURM_LOG_DIR="$SCRATCH/gflownet-logs/slurm"

SEED="${SEED:-0}"
DEPTH="${DEPTH:-5}"
N_TRAIN_STEPS="${N_TRAIN_STEPS:-10000}"
WANDB_ONLINE="${WANDB_ONLINE:-True}"

# ---- Submit wrapper ---------------------------------------------------------
# Only genuine array tasks have SLURM_ARRAY_TASK_ID; any other invocation
# (login node, or a shell inside a compute-node session) queues the array and
# exits.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    mkdir -p "$SLURM_LOG_DIR"
    echo "[submit] queueing diabetes splits 1-3 on partitions long-cpu,long-cpu-eek"
    exec sbatch --export=ALL "$REPO/mila/sbatch/regression_tree_diabetes.sh"
fi

split="$SLURM_ARRAY_TASK_ID"
run_name="reg_diabetes_depth${DEPTH}_split${split}"
work_dir="${WORK_DIR:-$SCRATCH/gflownet-logs/regression_tree}"
run_dir="$work_dir/$run_name"
csv_path="$REPO/tests/data/tree/diabetes/diabetes_${split}.csv"

echo "=========================================================="
echo "[$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID] $run_name"
echo "Dataset: $csv_path | depth: $DEPTH | train steps: $N_TRAIN_STEPS"
echo "Seed: $SEED | node: $(hostname) | started: $(date)"
echo "Run dir: $run_dir"
echo "=========================================================="

if [ ! -f "$csv_path" ]; then
    echo "$run_name SKIP (missing $csv_path)"
    exit 1
fi

if [ -f "$run_dir/samples/gfn_samples.pkl" ]; then
    echo "$run_name already has samples/gfn_samples.pkl -- nothing to do"
    exit 0
fi

module load python/3.10
source "$VENV/bin/activate"
cd "$REPO"

# Keep torch's intra-op threading within our allocation; without this, torch
# sizes its thread pool to all cores of the (shared) node.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

t_run=$SECONDS
python train.py +experiments=tree/regression_tree \
    env.data_path="$csv_path" \
    env.max_depth="$DEPTH" \
    seed="$SEED" \
    gflownet.optimizer.n_train_steps="$N_TRAIN_STEPS" \
    proxy.reward_function_kwargs.beta=0.1 \
    n_samples=1000 \
    logger.do.online="$WANDB_ONLINE" \
    logger.run_name="$run_name" \
    logger.run_name_date=False \
    hydra.run.dir="$run_dir" \
    hydra.job.chdir=True
train_status=$?
echo "$run_name training finished in $((SECONDS - t_run))s (exit $train_status)"

if [ $train_status -ne 0 ]; then
    echo "$run_name FAIL (training error, see above)"
    exit "$train_status"
fi

echo "$run_name DONE at $(date)"