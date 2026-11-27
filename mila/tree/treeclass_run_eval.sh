#!/bin/bash
#SBATCH --job-name=treeclass_run_eval
#SBATCH --output=/home/mila/a/arnit/scratch/gflownet-logs/slurm/%x-%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=long-cpu,long-cpu-eek

# =============================================================================
# Evaluate every finished training run that is missing its eval_results.json.
# =============================================================================
#
# Walks $ROOT (default: $SCRATCH/gflownet-logs) and, for every run directory
# (identified by .hydra/config.yaml) whose training produced final samples but
# that has no eval_results.json yet, runs the matching evaluation script, one
# run after another:
#   - classification (env Tree)            -> gflownet/envs/tree/eval_tree.py
#   - regression (env RegressionTree)      -> helpers_for_experiments/
#                                             eval_regression_tree.py
# Task, dataset and alpha are read from each run's own resolved hydra config;
# dataset paths baked on another cluster are relocated under this repo's
# tests/ directory, so run dirs rsync'ed from Trillium/Rorqual work too.
# eval_results.json doubles as the "done" marker, so re-submitting this job
# only evaluates what is still missing. All the work is done by
# gflownet/envs/tree/helpers_for_experiments/run_missing_evals.py.
#
# Usage:
#   mkdir -p $SCRATCH/gflownet-logs/slurm
#   sbatch mila/tree/treeclass_run_eval.sh
#
#   # a single campaign folder only:
#   sbatch --export=ALL,ROOT=$SCRATCH/gflownet-logs/TREECLASS_MAGIC \
#          mila/tree/treeclass_run_eval.sh
#
#   # only some datasets, or force re-evaluation:
#   sbatch mila/tree/treeclass_run_eval.sh --dataset iris,wine
#   sbatch --export=ALL,FORCE=1 mila/tree/treeclass_run_eval.sh
#
#   # see what would run without evaluating anything (fine on a login node):
#   bash mila/tree/treeclass_run_eval.sh --dry-run
#
# Environment knobs (overridable via --export=ALL,VAR=value):
#   ROOT    runs root to walk              (default: $SCRATCH/gflownet-logs)
#   FORCE   1 = re-evaluate runs that already have eval_results.json (default 0)
# Any positional arguments are passed straight through to run_missing_evals.py.

set -u

REPO="/home/mila/a/arnit/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"

ROOT="${ROOT:-$SCRATCH/gflownet-logs}"
FORCE="${FORCE:-0}"

module load python/3.10
source "$VENV/bin/activate"
cd "$REPO"

# Keep torch's intra-op threading within our allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

extra_args=()
if [ "$FORCE" = "1" ]; then
    extra_args+=("--force")
fi

# -u: unbuffered stdout so the per-run progress lines stream to the .out file
# live (`tail -f`) instead of only flushing when the job exits.
python -u gflownet/envs/tree/helpers_for_experiments/run_missing_evals.py \
    "$ROOT" "${extra_args[@]}" "$@"

status=$?
echo "treeclass_run_eval finished with exit code $status"
exit $status
