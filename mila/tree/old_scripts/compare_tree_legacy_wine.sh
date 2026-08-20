#!/bin/bash
#SBATCH --job-name=treeclass_cmp
#SBATCH --output=/home/mila/a/arnit/scratch/gflownet-logs/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=long-cpu,long-cpu-eek
#SBATCH --array=1-5

# Composite-Tree runs matched to the legacy dt-gfn LEGACYCODE_NOLEAK_* runs
# (wandb alex-hg/dtgfn), one array task per dataset split 1-5.
#
# The experiment config (config/experiments/tree/compare_tree_class_to_legacy_code.yaml)
# mirrors the legacy hyperparameters; the two deliberately translated values are
# env.max_depth=4 (== legacy max_depth 5, different level-counting convention)
# and reward beta=1.0 (untempered posterior, matching the legacy dummy reward).
# The data CSVs are the legacy per-split files copied into the repo
# (tests/data/tree/wine/wine_<split>.csv), so train/test splits are identical.
#
# Run names use "depth5" (the legacy-equivalent label) so the wandb runs line
# up side by side with LEGACYCODE_NOLEAK_wine<split>_depth5_steps100.
#
# Usage:
#   mkdir -p $SCRATCH/gflownet-logs/slurm && sbatch mila/sbatch/compare_tree_legacy_wine.sh
# Rerun a single split, e.g. split 3:
#   sbatch --array=3 mila/sbatch/compare_tree_legacy_wine.sh
#
# No --requeue, and checkpoints are only written every 1000 steps, so a job
# that dies mid-run cannot be resumed: resubmit it from scratch (and delete the
# partial wandb run first, or two runs share the name). Per-iteration cost grows
# with tree size and varies a lot across splits/nodes (2-25 s/it observed on
# wine), so give the walltime plenty of headroom.

set -u

split=$SLURM_ARRAY_TASK_ID
REPO="/home/mila/a/arnit/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"
WORK_DIR="${WORK_DIR:-$SCRATCH/gflownet-logs/treeclass_compare}"

# Sweep knobs, overridable from the sbatch environment, e.g.
#   sbatch --export=ALL,LR_EXP=3,STEPS=1000 --array=2,3 <script>
# LR_EXP=N means lr=1e-N and the "lr10e-N" label used in the run names.
LR_EXP="${LR_EXP:-3}"
STEPS="${STEPS:-10000}"

run_name="TREECLASS_wine${split}_depth5_steps${STEPS}_lr10e-${LR_EXP}"
csv_path="$REPO/tests/data/tree/wine/wine_${split}.csv"

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
    gflownet.optimizer.lr="1e-${LR_EXP}" \
    gflownet.optimizer.n_train_steps="$STEPS" \
    gflownet.optimizer.lr_decay_period="$STEPS" \
    seed=0 \
    logger.run_name="$run_name" \
    logger.run_name_date=False \
    hydra.run.dir="$WORK_DIR/$run_name" \
    hydra.job.chdir=True

status=$?
echo "$run_name finished with exit code $status"
exit $status