#!/bin/bash
#SBATCH --job-name=treeclass_agg
#SBATCH --output=/home/mila/a/arnit/scratch/gflownet-logs/slurm/%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --partition=main-cpu,long-cpu,long-cpu-eek

# =============================================================================
# Aggregate DT-GFN tree results (classification + regression) across splits.
# =============================================================================
#
# Runs gflownet/envs/tree/helpers_for_experiments/aggregate_treeclass_results.py:
# collects every eval_results.json under $ROOT and/or the last logged values of
# every run in the dt-gfn_classification / dt-gfn_regression wandb projects,
# groups the runs by training configuration (full resolved config minus the
# dataset split -- see the python script's docstring) and prints per dataset
# one mean +/- std table per source to this job's .out file. Debug runs are
# reported in a separate section. It computes nothing heavy (only JSON/YAML
# reading and wandb API calls), so it can also be run directly:
#
#   bash mila/tree/aggregate_treeclass_results.sh
#   Possible to add --source wandb or eval --dataset e.g. iris
#
# Usage examples via sbatch:
#   mkdir -p $SCRATCH/gflownet-logs/slurm
#   sbatch mila/tree/aggregate_treeclass_results.sh
#   sbatch --export=ALL,ROOT=$SCRATCH/gflownet-logs/TREECLASS_MAGIC \
#          mila/tree/aggregate_treeclass_results.sh
#   sbatch mila/tree/aggregate_treeclass_results.sh --source eval --diff-configs
#
# Environment knobs (overridable via --export=ALL,VAR=value):
#   ROOT   runs root (whole tree or one campaign folder)
#          (default: $SCRATCH/gflownet-logs)
# Any positional arguments are passed straight through to the python script
# (--source, --dataset, --task, --diff-configs, --group-ignore, --min-splits, ...).
#
# Notes:
#   - Configurations with fewer than 3 dataset splits are hidden by default;
#     pass --min-splits 1 to show everything.
#   - Requires the repo to be pip-installed in the venv (pip install -e .);
#     the script imports gflownet.envs.tree.helpers_for_experiments.
#   - For interactive inspection (dataset picker, hash2config) open
#     gflownet/envs/tree/helpers_for_experiments/inspect_treeclass_results.ipynb

set -u

REPO="/home/mila/a/arnit/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"

ROOT="${ROOT:-$SCRATCH/gflownet-logs}"

module load python/3.10
source "$VENV/bin/activate"
cd "$REPO"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# -u: unbuffered stdout so progress lines and tables stream to the .out live.
python -u gflownet/envs/tree/helpers_for_experiments/aggregate_treeclass_results.py \
    "$ROOT" "$@"

status=$?
echo "aggregate_treeclass_results finished with exit code $status"
exit $status
