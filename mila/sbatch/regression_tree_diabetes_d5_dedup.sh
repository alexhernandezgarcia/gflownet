#!/bin/bash
#SBATCH --job-name=reg_tree_d5_dedup
#SBATCH --output=/home/mila/a/arnit/scratch/gflownet-logs/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --partition=long-cpu,long-cpu-eek
#SBATCH --array=1-3

# TEMPORARY (2026-07-21): rescue/dedup of the depth-5 diabetes runs.
#
# Background: the from-scratch array (job 10160605, name reg_tree_diabetes)
# was preempted repeatedly on long-cpu and auto-requeued by SLURM (preempted
# jobs ARE requeued by default; only *failed* jobs are not). Every requeue
# restarted training from step 0 in the same run dir and opened a NEW wandb
# run under the same name -> 4/4/3 duplicate depth-5 runs for splits 1/2/3.
# All instances ran the identical config (depth 5, seed 0, beta 0.1, 10000
# steps); they differ only in how far they got before preemption.
#
# What this script does:
# 1. (submit wrapper) Cancels every remaining instance of the from-scratch
#    array by job NAME reg_tree_diabetes -- the depth-3 resume jobs
#    (reg_tree_diabetes_resume) are NOT touched -- then submits itself.
# 2. (array task, one per split) Resumes via resume.py from the highest-iter
#    checkpoint in <run_dir>/ckpts. find_latest_checkpoint() sorts by
#    iteration NUMBER (not mtime), so it picks the furthest progress ever
#    reached even though later from-scratch instances overwrote the
#    low-iter checkpoints. The checkpoint stores the wandb run_id of the
#    instance that wrote it, and resume re-attaches to exactly that run
#    (logger uses wandb.init(id=run_id, resume="allow")) -- i.e. the MOST
#    PROGRESSED wandb run per split is the one that continues; all other
#    same-name runs stay dead and can be deleted in the wandb UI (each task
#    prints the kept id and the obsolete ids below).
#
# Unlike regression_tree_diabetes_resume.sh, this does NOT bump
# n_train_steps: the stored config keeps the original 10000-step target.
# There is also no final.ckpt to rename (no depth-5 run has finished).
#
# From now on preemption is harmless: a requeued task re-runs resume.py,
# picks up the newest checkpoint and re-attaches to the same wandb run.
# Expect wandb to drop a small overlap (the <=500 steps logged after the
# resumed checkpoint was written) as out-of-order -- cosmetic only.
#
# Usage (from anywhere; the script submits itself):
#   bash mila/sbatch/regression_tree_diabetes_d5_dedup.sh

set -u

REPO="$HOME/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"
SLURM_LOG_DIR="$SCRATCH/gflownet-logs/slurm"

SEED="${SEED:-0}"
DEPTH="${DEPTH:-5}"

# ---- Submit wrapper ---------------------------------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    mkdir -p "$SLURM_LOG_DIR"
    if squeue -u "$USER" -n reg_tree_d5_dedup -h | grep -q .; then
        echo "[submit] ABORT: a reg_tree_d5_dedup array is already queued/running:"
        squeue -u "$USER" -n reg_tree_d5_dedup
        exit 1
    fi
    echo "[submit] cancelling remaining from-scratch depth-5 tasks (job name reg_tree_diabetes):"
    squeue -u "$USER" -n reg_tree_diabetes || true
    scancel -u "$USER" -n reg_tree_diabetes
    echo "[submit] waiting 20s for cancelled tasks to release their run dirs..."
    sleep 20
    squeue -u "$USER" -n reg_tree_diabetes || true
    echo "[submit] queueing depth-${DEPTH} dedup/resume for splits 1-3"
    exec sbatch --export=ALL "$REPO/mila/sbatch/regression_tree_diabetes_d5_dedup.sh"
fi

split="$SLURM_ARRAY_TASK_ID"
run_name="reg_diabetes_depth${DEPTH}_split${split}"
work_dir="${WORK_DIR:-$SCRATCH/gflownet-logs/regression_tree}"
run_dir="$work_dir/$run_name"
run_config="$run_dir/.hydra/config.yaml"

echo "=========================================================="
echo "[$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID] dedup-resume $run_name"
echo "Run dir: $run_dir | node: $(hostname) | started: $(date)"
echo "=========================================================="

if [ ! -f "$run_config" ]; then
    echo "$run_name SKIP (missing $run_config)"
    exit 1
fi
if [ -f "$run_dir/ckpts/final.ckpt" ]; then
    echo "$run_name already finished (final.ckpt exists) -- nothing to do"
    exit 0
fi
if ! compgen -G "$run_dir/ckpts/iter_*.ckpt" > /dev/null; then
    echo "$run_name SKIP (no iter_*.ckpt in $run_dir/ckpts)"
    exit 1
fi

module load python/3.10
source "$VENV/bin/activate"
cd "$REPO"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Same selection rule as find_latest_checkpoint(): highest iteration number.
latest_ckpt="$(ls "$run_dir"/ckpts/iter_*.ckpt | sort -V | tail -n 1)"
keep_id="$(python -c "
import sys, torch
print(torch.load(sys.argv[1], map_location='cpu').get('run_id'))
" "$latest_ckpt")"
echo "Resuming from: $latest_ckpt"
echo "KEEP wandb run id: $keep_id (this run continues)"
echo "Obsolete duplicate wandb runs named $run_name (safe to delete in the UI):"
for d in "$run_dir"/wandb/run-*/; do
    id="${d%/}"; id="${id##*-}"
    [ "$id" != "$keep_id" ] && echo "  - $id"
done

t_run=$SECONDS
python resume.py \
    rundir="$run_dir" \
    n_samples=1000 \
    seed="$SEED"
resume_status=$?
echo "$run_name dedup-resume finished in $((SECONDS - t_run))s (exit $resume_status)"

if [ $resume_status -ne 0 ]; then
    echo "$run_name FAIL (resume error, see above)"
    exit "$resume_status"
fi

samples_pkl="$(ls -t "$run_dir"/resume/*/*/gfn_samples.pkl 2>/dev/null | head -n 1)"
echo "$run_name DONE at $(date) | samples: ${samples_pkl:-NOT FOUND}"
