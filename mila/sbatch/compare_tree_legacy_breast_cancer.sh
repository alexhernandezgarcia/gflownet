#!/bin/bash
#SBATCH --job-name=treeclass_bc
#SBATCH --output=/home/mila/a/arnit/scratch/gflownet-logs/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --partition=long-cpu,long-cpu-eek
#SBATCH --array=1-5

# Composite-Tree runs matched to the legacy dt-gfn LEGACYCODE_NOLEAK_* runs
# (wandb alex-hg/dtgfn), one array task per dataset split 1-5.
#
# Same setup as compare_tree_legacy_iris.sh / _wine.sh (config:
# config/experiments/tree/compare_tree_class_to_legacy_code.yaml; env.max_depth=4
# == legacy max_depth 5; reward beta=1.0 = untempered posterior), with ONE
# addition: float_precision=64.
#
# Why float64 here: with beta=1.0 the replay buffer stores exp(log-posterior).
# float32 exp() flushes to zero below ~-103 nats. On breast_cancer
# (N_train=455, 30 features) even the BEST trees sit around log-posterior
# -90 to -160, i.e. straddling that floor -- so most or all buffer rewards
# would be exactly 0.0, which freezes buffer insertion (reward > min never
# true) and NaNs the weighted replay sampling (0/0). float64 moves the exp()
# floor to ~-745, far below anything this dataset produces. This changes
# numerics only, not the target distribution, and the legacy code base is
# unaffected by the issue because it works purely in the log domain.
#
# Usage:
#   mkdir -p $SCRATCH/gflownet-logs/slurm && sbatch mila/sbatch/compare_tree_legacy_breast_cancer.sh
# Rerun a single split, e.g. split 3:
#   sbatch --array=3 mila/sbatch/compare_tree_legacy_breast_cancer.sh
#
# No --requeue: a preemption restart would open a duplicate wandb run under
# the same name. If a task dies, resubmit it.

set -u

split=$SLURM_ARRAY_TASK_ID
REPO="/home/mila/a/arnit/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"
WORK_DIR="${WORK_DIR:-$SCRATCH/gflownet-logs/treeclass_compare}"

run_name="TREECLASS_breast_cancer${split}_depth5_steps1000"
csv_path="$REPO/tests/data/tree/breast_cancer/breast_cancer_${split}.csv"

module load python/3.10
source "$VENV/bin/activate"
cd "$REPO"

# Keep torch's intra-op threading within our allocation.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Be patient with wandb init on busy nodes.
export WANDB_INIT_TIMEOUT=300
export WANDB__SERVICE_WAIT=300

python train.py +experiments=tree/compare_tree_class_to_legacy_code \
    env.data_path="$csv_path" \
    float_precision=64 \
    seed=0 \
    logger.run_name="$run_name" \
    logger.run_name_date=False \
    hydra.run.dir="$WORK_DIR/$run_name" \
    hydra.job.chdir=True

status=$?
echo "$run_name finished with exit code $status"
exit $status
