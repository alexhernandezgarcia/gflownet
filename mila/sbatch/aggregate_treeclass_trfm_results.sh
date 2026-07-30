#!/bin/bash
#SBATCH --job-name=trfm_agg
#SBATCH --output=/home/mila/a/arnit/scratch/gflownet-logs/slurm/%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --partition=long-cpu,long-cpu-eek

# Aggregate transformer-policy composite-Tree evaluation results across splits.
#
# Runs gflownet/envs/tree/aggregate_treeclass_trfm_results.py over every finished
# trfm_<dataset>_depth<max_depth>_split<seed> run under $WORK_DIR (identified by
# its final gfn_samples.pkl: samples/ for single-shot runs, else the newest
# resume/<jobid>/<ts>/ for resumed ones), groups them by setup (dataset,
# max_depth, n_train_steps, n_samples, alpha_value) and prints the mean +/- std
# table across split seeds to this job's .out file. Runs still training are
# skipped; groups without the expected 5 seeds are flagged.
#
# The first invocation recomputes metrics with eval_tree.py's protocol and caches
# them in metrics_cache.json next to each gfn_samples.pkl (slow -- 1000 trees per
# run); reruns are near-instant. To force recomputation, add --no-cache below.
# This is why it runs on a compute node: the login nodes kill the heavy runs.
#
# A #!/bin/bash script file is used on purpose: `module` is not available under
# `sbatch --wrap` (that runs /bin/sh), which is why the wrap one-liner failed.
#
# Usage:
#   mkdir -p $SCRATCH/gflownet-logs/slurm && sbatch mila/sbatch/aggregate_treeclass_trfm_results.sh
# To scan the whole logs tree instead of just the trfm sweep dir:
#   WORK_DIR=$SCRATCH/gflownet-logs sbatch mila/sbatch/aggregate_treeclass_trfm_results.sh

set -u

REPO="/home/mila/a/arnit/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"
WORK_DIR="${WORK_DIR:-$SCRATCH/gflownet-logs/trfm_sweep_10132875}"

module load python/3.10
source "$VENV/bin/activate"
cd "$REPO"

# Keep torch's intra-op threading within our allocation.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# -u keeps stdout unbuffered so the [SKIP]/[INFO] progress lines and the final
# table stream to the .out file live (`tail -f`) instead of only flushing when
# the job exits.
python -u gflownet/envs/tree/aggregate_treeclass_trfm_results.py --logs-root "$WORK_DIR"

status=$?
echo "aggregate_treeclass_trfm_results finished with exit code $status"
exit $status
