#!/bin/bash
#SBATCH --job-name=treeclass_agg
#SBATCH --output=/home/mila/a/arnit/scratch/gflownet-logs/slurm/%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --partition=long-cpu,long-cpu-eek

# Aggregate composite-Tree (TREECLASS) evaluation results across dataset splits.
#
# Runs gflownet/envs/tree/aggregate_treeclass_results.py over every finished run
# under $WORK_DIR (identified by samples/gfn_samples.pkl), groups them by setup
# (dataset, max_depth, n_train_steps, n_samples, alpha_value) and prints the
# mean +/- std table across splits to this job's .out file.
#
# The first invocation recomputes metrics and caches them in metrics_cache.json
# next to each gfn_samples.pkl (slow); reruns are near-instant. To force
# recomputation, add --no-cache to the python call below.
#
# A #!/bin/bash script file is used on purpose: `module` is not available under
# `sbatch --wrap` (that runs /bin/sh), which is why the wrap one-liner failed.
#
# Usage:
#   mkdir -p $SCRATCH/gflownet-logs/slurm && sbatch mila/sbatch/aggregate_treeclass_results.sh

set -u

REPO="/home/mila/a/arnit/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"
WORK_DIR="${WORK_DIR:-$SCRATCH/gflownet-logs/treeclass_compare}"

module load python/3.10
source "$VENV/bin/activate"
cd "$REPO"

# Keep torch's intra-op threading within our allocation.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# -u: unbuffered stdout so the per-run [INFO] lines and final table stream to
# the .out file live instead of only flushing when the process exits.
python -u gflownet/envs/tree/helpers_for_experiments/aggregate_treeclass_results.py --logs-root "$WORK_DIR"

status=$?
echo "aggregate_treeclass_results finished with exit code $status"
exit $status
