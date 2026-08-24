#!/bin/bash
#SBATCH --job-name=reg_tree
#SBATCH --account=def-alexhg
#SBATCH --output=/scratch/arnit/gflownet-logs/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=1-5
# =============================================================================
# RORQUAL submitter: one array task per dataset split (per-core scheduling,
# like Mila). All the actual work is in drac/reg_tree_worker.sh.
#
#   mkdir -p $SCRATCH/gflownet-logs/slurm
#   sbatch drac/rorqual_reg_tree.sh
#
#   # name the campaign, change the dataset, override any hydra setting:
#   sbatch --export=ALL,EXP_NAME=REG_lr1e-3,DATASET=energy \
#          drac/rorqual_reg_tree.sh gflownet.optimizer.lr=1e-3
#
#   # only splits 1 and 3:
#   sbatch --array=1,3 drac/rorqual_reg_tree.sh
#
# --cpus-per-task and --mem are PER ARRAY TASK (5 x 4 CPUs, 5 x 32 GB here).
# The #SBATCH lines cannot expand variables, so --account and --output are
# spelled out; edit them if your account or username differs.
# =============================================================================

set -u

REPO="${REPO:-$HOME/gflownet}"
VENV="${VENV:-$SCRATCH/venvs/gflownet-env}"
export RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
export CPUS_PER_RUN="${SLURM_CPUS_PER_TASK:-4}"
export WANDB_MODE="${WANDB_MODE:-offline}"

module purge
module load StdEnv/2023 python/3.10
source "$VENV/bin/activate"

# Code snapshot on the node's local disk, so the checkout can be edited or
# switched to another branch while jobs are running.
if [ -n "${SLURM_TMPDIR:-}" ]; then
    export CODE_DIR="$SLURM_TMPDIR/gflownet"
    rsync -a --exclude ".git" --exclude "__pycache__" "$REPO/" "$CODE_DIR/"
else
    export CODE_DIR="$REPO"
fi
export REPO

split="${SLURM_ARRAY_TASK_ID:-1}"
bash "$CODE_DIR/drac/reg_tree_worker.sh" "$split" "$@"
