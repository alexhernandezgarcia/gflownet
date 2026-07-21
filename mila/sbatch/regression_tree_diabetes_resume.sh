#!/bin/bash
#SBATCH --job-name=reg_tree_diabetes_resume
#SBATCH --output=/home/mila/a/arnit/scratch/gflownet-logs/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --partition=long-cpu,long-cpu-eek
#SBATCH --array=1-3

# Extends the finished diabetes regression runs (reg_diabetes_depth3_split1-3)
# to N_TRAIN_STEPS total steps by resuming from their latest checkpoint.
#
# How it works:
# - resume.py reloads the run configuration from <run_dir>/.hydra/config.yaml
#   and takes NO overrides for it, so this script first bumps
#   gflownet.optimizer.n_train_steps inside that stored yaml (idempotent).
# - Models, optimizer, logZ and step are restored from the latest checkpoint;
#   training continues from there and wandb re-attaches to the SAME run, so
#   the plots continue seamlessly. New checkpoints keep going to
#   <run_dir>/ckpts/, and the final samples of the resumed training are
#   written to <run_dir>/resume/<jobid>/<timestamp>/gfn_samples.pkl.
# - find_latest_checkpoint() prefers a checkpoint named final.ckpt over the
#   newest iter_*.ckpt. The finished runs wrote final.ckpt (== iter_010000),
#   which would shadow newer iter_* checkpoints if a resumed run is preempted
#   and resumed again. It is therefore renamed away before resuming.
# - Known quirk: the LR scheduler state is not checkpointed, so the decay
#   clock restarts at the resume (LR goes back to optimizer.lr and halves
#   every lr_decay_period steps counted from the resume). For these runs the
#   first decay had only just triggered at the final step 10000, so resuming
#   at the base LR is effectively seamless.
#
# Usage (from anywhere; the script submits itself):
#   bash mila/sbatch/regression_tree_diabetes_resume.sh
# Tunables (forwarded by --export=ALL):
#   N_TRAIN_STEPS=50000  DEPTH=3  SEED=0
#   WORK_DIR=$SCRATCH/gflownet-logs/regression_tree

set -u

REPO="$HOME/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"
SLURM_LOG_DIR="$SCRATCH/gflownet-logs/slurm"

SEED="${SEED:-0}"
DEPTH="${DEPTH:-3}"
N_TRAIN_STEPS="${N_TRAIN_STEPS:-50000}"

# ---- Submit wrapper ---------------------------------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    mkdir -p "$SLURM_LOG_DIR"
    echo "[submit] queueing resume of diabetes splits 1-3 to $N_TRAIN_STEPS steps"
    exec sbatch --export=ALL "$REPO/mila/sbatch/regression_tree_diabetes_resume.sh"
fi

split="$SLURM_ARRAY_TASK_ID"
run_name="reg_diabetes_depth${DEPTH}_split${split}"
work_dir="${WORK_DIR:-$SCRATCH/gflownet-logs/regression_tree}"
run_dir="$work_dir/$run_name"
run_config="$run_dir/.hydra/config.yaml"

echo "=========================================================="
echo "[$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID] resume $run_name -> $N_TRAIN_STEPS steps"
echo "Run dir: $run_dir | node: $(hostname) | started: $(date)"
echo "=========================================================="

if [ ! -f "$run_config" ]; then
    echo "$run_name SKIP (missing $run_config)"
    exit 1
fi
if ! compgen -G "$run_dir/ckpts/*.ckpt" > /dev/null; then
    echo "$run_name SKIP (no checkpoints in $run_dir/ckpts)"
    exit 1
fi

# Extend the training horizon in the stored run config (idempotent)
sed -i -E "s/^([[:space:]]*)n_train_steps: [0-9]+/\1n_train_steps: $N_TRAIN_STEPS/" \
    "$run_config"
echo "Stored config now has: $(grep -E 'n_train_steps:' "$run_config" | tr -d ' ')"

# Rename final.ckpt so repeated resumes pick the newest iter_* checkpoint
if [ -f "$run_dir/ckpts/final.ckpt" ]; then
    mv "$run_dir/ckpts/final.ckpt" "$run_dir/ckpts/final.ckpt.pre_resume.bak"
    echo "Renamed final.ckpt -> final.ckpt.pre_resume.bak"
fi

module load python/3.10
source "$VENV/bin/activate"
cd "$REPO"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

t_run=$SECONDS
python resume.py \
    rundir="$run_dir" \
    n_samples=1000 \
    seed="$SEED"
resume_status=$?
echo "$run_name resume finished in $((SECONDS - t_run))s (exit $resume_status)"

if [ $resume_status -ne 0 ]; then
    echo "$run_name FAIL (resume error, see above)"
    exit "$resume_status"
fi

samples_pkl="$(ls -t "$run_dir"/resume/*/*/gfn_samples.pkl 2>/dev/null | head -n 1)"
echo "$run_name DONE at $(date) | samples: ${samples_pkl:-NOT FOUND}"
