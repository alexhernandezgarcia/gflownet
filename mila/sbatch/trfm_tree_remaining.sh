#!/bin/bash
#SBATCH --job-name=trfm_tree
#SBATCH --output=/home/mila/a/arnit/scratch/gflownet-logs/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --partition=long-cpu,long-cpu-eek
#SBATCH --array=6-17
#SBATCH --requeue

# Remaining transformer-policy tree runs, launched as a SLURM job array so all
# runs are queued at once and execute in parallel as resources free up.
#
# Already completed elsewhere (hence NOT in the list below):
#   - all depth-1 runs
#   - depth-3 runs on iris and breast_cancer
# Completed by array 10132875 on 2026-07-16 (tasks 0-5, kept in the table so
# indices stay stable; the default --array=6-17 no longer includes them):
#   - depth 3: wine, raisin           x splits 1, 2, 3   (6 runs)
# Remaining:
#   - depth 5: all four datasets      x splits 1, 2, 3   (12 runs)
#
# History: the first submission (array 10132875) ran with --mem=16G; all 12
# depth-5 tasks were OOM-killed (MaxRSS 12.7-16.4G). Depth-3 runs peaked at
# ~9G, so depth-5 gets 64G for headroom (CPU nodes have ~385G).
#
# Each array task trains one (depth, dataset, split) combination and then runs
# the final test-set evaluation, exactly like the sequential run_trfm_sweep.sh.
# Per-run outputs (checkpoints, samples, eval_results.json) go to
#   $SCRATCH/gflownet-logs/trfm_sweep_10132875/<run_name>/   (override: WORK_DIR)
# and the per-task log to $SCRATCH/gflownet-logs/slurm/.
#
# Usage (from anywhere; the script submits itself):
#   bash mila/sbatch/trfm_tree_remaining.sh
# or plain sbatch after creating the slurm log dir once:
#   mkdir -p $SCRATCH/gflownet-logs/slurm && sbatch mila/sbatch/trfm_tree_remaining.sh
#
# Tunables via environment variables (forwarded by --export=ALL):
#   SEED=0  STEPS_DEPTH3=5000  STEPS_DEPTH5=10000  WANDB_ONLINE=True
# To rerun a subset, e.g. only tasks 6-17:
#   sbatch --array=6-17 mila/sbatch/trfm_tree_remaining.sh
#
# Preemption/robustness (hardened 2026-07-21 after the first pass, array
# 10135737, exposed two failure modes):
# - A task preempted BEFORE its first checkpoint (step 500 back then)
#   restarted from scratch and opened a NEW wandb run under the same name --
#   that is where the duplicate depth-5 wandb runs came from. Checkpoints are
#   now written every 100 steps (fresh starts via override below; resumed
#   runs via an idempotent sed of the stored run config), so at most ~100
#   steps are lost per preemption and the from-scratch window is small.
# - A one-off wandb init timeout made task 11 (breast_cancer depth5 split3)
#   exit FAILED on 2026-07-18 -- and SLURM never requeues FAILED tasks, so it
#   sat dead. resume.py is therefore retried up to 3x per invocation, and if
#   it still fails the task requeues ITSELF (capped via
#   $run_dir/.auto_requeue_count) instead of dying silently.
# - Otherwise: a task whose run dir has checkpoints always resumes via
#   resume.py (models, optimizer, logZ and step restored; wandb re-attaches
#   to the run id stored in the checkpoint, i.e. the most-progressed chain).
#   Unloadable checkpoints (a write cut short by preemption) are renamed to
#   *.corrupt and the next-newest one is used. A task that already produced
#   eval_results.json exits immediately.
# - The submit wrapper first scancels EVERY job named trfm_tree, so rerunning
#   this script is always safe: nothing can double-write a run dir, and all
#   tasks continue from their newest checkpoint (a running task only loses
#   the steps since its last checkpoint write).

set -u

REPO="$HOME/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"
SLURM_LOG_DIR="$SCRATCH/gflownet-logs/slurm"

SEED="${SEED:-0}"
STEPS_DEPTH3="${STEPS_DEPTH3:-5000}"
STEPS_DEPTH5="${STEPS_DEPTH5:-10000}"
WANDB_ONLINE="${WANDB_ONLINE:-True}"

# ---- Submit wrapper ---------------------------------------------------------
# Only genuine array tasks have SLURM_ARRAY_TASK_ID; any other invocation
# (login node, or a shell inside a compute-node session, where SLURM_JOB_ID is
# already set) queues the array and exits.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    mkdir -p "$SLURM_LOG_DIR"
    echo "[submit] cancelling existing trfm_tree jobs (they resume from their latest checkpoint):"
    squeue -u "$USER" -n trfm_tree || true
    scancel -u "$USER" -n trfm_tree
    echo "[submit] waiting 20s for cancelled tasks to release their run dirs..."
    sleep 20
    echo "[submit] queueing depth-5 tasks 6-17 on partitions long-cpu,long-cpu-eek"
    exec sbatch --export=ALL "$REPO/mila/sbatch/trfm_tree_remaining.sh"
fi

# ---- Task table: index -> depth dataset split -------------------------------
RUNS=(
    "3 wine 1"          # 0
    "3 wine 2"          # 1
    "3 wine 3"          # 2
    "3 raisin 1"        # 3
    "3 raisin 2"        # 4
    "3 raisin 3"        # 5
    "5 iris 1"          # 6
    "5 iris 2"          # 7
    "5 iris 3"          # 8
    "5 breast_cancer 1" # 9
    "5 breast_cancer 2" # 10
    "5 breast_cancer 3" # 11
    "5 wine 1"          # 12
    "5 wine 2"          # 13
    "5 wine 3"          # 14
    "5 raisin 1"        # 15
    "5 raisin 2"        # 16
    "5 raisin 3"        # 17
)
read -r depth dataset split <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

case "$depth" in
    3) n_train_steps="$STEPS_DEPTH3" ;;
    5) n_train_steps="$STEPS_DEPTH5" ;;
    *) echo "ERROR: no train steps defined for depth $depth" >&2; exit 1 ;;
esac

run_name="trfm_${dataset}_depth${depth}_split${split}"
# Pinned to the directory created by the first submission (array 10132875) so
# resubmissions find the existing checkpoints/results there: completed tasks
# exit immediately and OOM-killed ones resume from their latest checkpoint.
# (Do NOT key this on SLURM_ARRAY_JOB_ID: every resubmission gets a new id and
# would silently start the whole sweep from scratch in a fresh directory.)
work_dir="${WORK_DIR:-$SCRATCH/gflownet-logs/trfm_sweep_10132875}"
run_dir="$work_dir/$run_name"
csv_path="$REPO/tests/data/tree/${dataset}/${dataset}_${split}.csv"

echo "=========================================================="
echo "[$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID] $run_name"
echo "Dataset: $csv_path | depth: $depth | train steps: $n_train_steps"
echo "Seed: $SEED | node: $(hostname) | started: $(date)"
echo "Run dir: $run_dir"
echo "=========================================================="

if [ ! -f "$csv_path" ]; then
    echo "$run_name SKIP (missing $csv_path)"
    exit 1
fi

# Requeued after everything already finished: nothing to do.
if [ -f "$run_dir/eval_results.json" ]; then
    echo "$run_name already has eval_results.json -- nothing to do"
    exit 0
fi

module load python/3.10
source "$VENV/bin/activate"
cd "$REPO"

# Keep torch's intra-op threading within our allocation; without this, torch
# sizes its thread pool to all cores of the (shared) node.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Be patient with wandb: a 90s init timeout is what killed task 11.
export WANDB_INIT_TIMEOUT=300
export WANDB__SERVICE_WAIT=300

t_run=$SECONDS
if [ -f "$run_dir/samples/gfn_samples.pkl" ]; then
    # Training and final sampling already completed; only eval was missing.
    echo "$run_name samples already exist -- skipping training"
    samples_pkl="$run_dir/samples/gfn_samples.pkl"
    train_status=0
elif compgen -G "$run_dir/ckpts/*.ckpt" > /dev/null; then
    # Requeued after preemption: continue from the latest checkpoint. Resume
    # re-attaches the same wandb run and keeps checkpointing into $run_dir;
    # the final samples are written to $run_dir/resume/<jobid>/<timestamp>/.
    echo "$run_name found checkpoints -- resuming from latest"

    # Checkpoint every 100 steps from now on (idempotent edit of the stored
    # run config): caps the work lost per preemption at ~100 steps.
    sed -i -E "s/^([[:space:]]*)checkpoints_period: [0-9]+/\1checkpoints_period: 100/" \
        "$run_dir/.hydra/config.yaml"

    # Quarantine unloadable checkpoints (e.g. a write cut short by
    # preemption): probe newest-first, rename broken ones to *.corrupt so
    # find_latest_checkpoint() falls back to the next-newest good one.
    for ckpt in $(ls "$run_dir"/ckpts/iter_*.ckpt 2>/dev/null | sort -rV); do
        if python -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu')" \
                "$ckpt" > /dev/null 2>&1; then
            echo "$run_name resuming from checkpoint: $(basename "$ckpt")"
            break
        fi
        echo "$run_name WARNING: $(basename "$ckpt") unreadable -- renamed to .corrupt"
        mv "$ckpt" "$ckpt.corrupt"
    done

    # Retry transient failures (wandb outages etc.) before giving up.
    train_status=1
    for attempt in 1 2 3; do
        python resume.py \
            rundir="$run_dir" \
            n_samples=1000 \
            seed="$SEED"
        train_status=$?
        [ $train_status -eq 0 ] && break
        echo "$run_name resume attempt $attempt failed (exit $train_status)"
        [ $attempt -lt 3 ] && sleep 90
    done
    samples_pkl="$(ls -t "$run_dir"/resume/*/*/gfn_samples.pkl 2>/dev/null | head -n 1)"
else
    python train.py +experiments=tree/trfm_classification_tree \
        env.data_path="$csv_path" \
        env.max_depth="$depth" \
        seed="$SEED" \
        gflownet.optimizer.n_train_steps="$n_train_steps" \
        evaluator.checkpoints_period=100 \
        n_samples=1000 \
        logger.do.online="$WANDB_ONLINE" \
        logger.run_name="$run_name" \
        logger.run_name_date=False \
        hydra.run.dir="$run_dir" \
        hydra.job.chdir=True
    train_status=$?
    samples_pkl="$run_dir/samples/gfn_samples.pkl"
fi
echo "$run_name training finished in $((SECONDS - t_run))s (exit $train_status)"

if [ $train_status -ne 0 ]; then
    # SLURM never requeues a FAILED task (that is how breast_cancer split3
    # silently died on 2026-07-18), so requeue ourselves -- capped, to avoid
    # an infinite crash loop on a persistent error.
    requeue_count_file="$run_dir/.auto_requeue_count"
    count=$(( $(cat "$requeue_count_file" 2>/dev/null || echo 0) + 1 ))
    echo "$count" > "$requeue_count_file"
    if [ "$count" -le 3 ]; then
        echo "$run_name training failed (exit $train_status) -- self-requeue #$count"
        scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        sleep 60  # the requeue kills this script; do not fall through meanwhile
        exit 0
    fi
    echo "$run_name FAIL (training error persisted through $count requeues, see above)"
    exit "$train_status"
fi
rm -f "$run_dir/.auto_requeue_count"

if [ -z "$samples_pkl" ] || [ ! -f "$samples_pkl" ]; then
    echo "$run_name FAIL (no gfn_samples.pkl found)"
    exit 1
fi

echo "---- $run_name: final eval on test set ----"
python gflownet/envs/tree/eval_tree.py \
    --samples_path "$samples_pkl" \
    --data_path "$csv_path" \
    --output "$run_dir/eval_results.json"
eval_status=$?

if [ $eval_status -eq 0 ] && [ -f "$run_dir/eval_results.json" ]; then
    echo "---- $run_name: eval_results.json ----"
    cat "$run_dir/eval_results.json"
    echo ""
    echo "$run_name DONE at $(date)"
else
    echo "$run_name FAIL (eval error, exit $eval_status)"
    exit 1
fi
