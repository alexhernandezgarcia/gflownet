#!/bin/bash
#SBATCH --job-name=cls_tree
#SBATCH --account=def-alexhg
#SBATCH --output=/scratch/arnit/gflownet-logs/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=1-5
# =============================================================================
# RORQUAL submitter for CLASSIFICATION trees: one array task per dataset split
# (per-core scheduling, like Mila). Twin of rorqual_reg_tree.sh. All the work
# is in drac/cls_tree_worker.sh, which is the body of
# mila/tree/run_classification_tree_training.sh without the Mila-only header
# (partition, absolute Mila paths, module name); this file is the header.
#
#   mkdir -p $SCRATCH/gflownet-logs/slurm
#   sbatch --export=ALL,EXP_NAME=<campaign>,DATASET=<dataset> \
#          drac/rorqual_cls_tree.sh [hydra overrides...]
#
#   # transformer policy:
#   sbatch --export=ALL,EXP_NAME=TRFM_X,DATASET=magic,EXP_CONFIG=tree/trfm_classification_tree \
#          drac/rorqual_cls_tree.sh policy.backward.shared_weights=False
#
#   # only splits 1 and 3, other resources (the command line beats the header):
#   sbatch --array=1,3 --time=48:00:00 --mem=32G drac/rorqual_cls_tree.sh
#
# Knobs (via --export=ALL,VAR=value): EXP_NAME EXP_CONFIG DATASET SEED RUNS_ROOT FORCE
# --cpus-per-task and --mem are PER ARRAY TASK. The #SBATCH lines cannot expand
# variables, so --account and --output are spelled out; edit them if yours differ.
# Resubmitting the same command after a time-out is safe: finished runs are
# skipped, unfinished ones resume from ckpts/.
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
bash "$CODE_DIR/drac/cls_tree_worker.sh" "$split" "$@"
