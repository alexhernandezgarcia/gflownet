#!/bin/bash
#SBATCH --job-name=cls_tree_gpu
#SBATCH --output=/home/mila/a/arnit/scratch/gflownet-logs/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=18:00:00
#SBATCH --partition=long
#SBATCH --array=1-5
#SBATCH --requeue

# =============================================================================
# GPU variant of run_classification_tree_training.sh.
# =============================================================================
#
# Differences to the CPU script:
#   * Slurm asks for a GPU (--gres=gpu:l40s:1) on the GPU "long" partition
#     instead of long-cpu. See "GPU memory" below for why the type is pinned;
#     override it per launch with e.g. `sbatch --gres=gpu:a100l:1 ...`.
#   * VENV defaults to the CUDA-enabled environment (gflownet-env-gpu); the
#     CPU venv has a +cpu torch build that cannot see a GPU.
#   * DEVICE (default: cuda) is appended to the hydra overrides. Since the
#     config hash covers every override, device=cuda lands in a different run
#     directory than the same settings on CPU -- CPU and GPU campaigns never
#     collide or resume each other.
#   * The job aborts early if torch cannot see a CUDA device, because the code
#     would otherwise fall back to CPU silently (see utils/common.py
#     set_device) and burn a GPU allocation on a CPU run.
#
# GPU memory
# ----------
# The losses backpropagate through *every state of every trajectory in the
# batch* in a single policy call (see compute_logprobs_trajectories in
# gflownet/utils/batch.py), once for the forward policy and once for the
# backward one. So activation memory scales with
#
#     sum of trajectory lengths in the batch  x  seq_len  x  d_model x n_layers
#
# and trajectory lengths *grow during training* as the sampler learns to build
# bigger trees. For max_depth=5 (max_nodes=31, so 32 tokens per state) with the
# default d_model=128 / n_layers=3 and 100 trajectories per batch, the worst
# case is ~15k states x ~1.8 MB ~= 28 GB, which does not fit on a 32 GB V100:
# that is why --gres asks for an l40s (48 GB) rather than a bare gpu:1. For a
# 32 GB card, halve gflownet.optimizer.batch_size instead. Raising max_depth to
# 6 doubles both the state count and the sequence length (~4x the memory), so
# it needs an 80 GB card (a100l / h100) or a smaller batch.
#
# Everything else (run layout, LAUNCH records, resume, evaluation) is
# identical -- see run_classification_tree_training.sh for the full docs.
#
# Usage:
#   sbatch --export=ALL,EXP_CONFIG=tree/trfm_classification_tree,EXP_NAME=TRFM_GPU \
#          mila/tree/run_classification_tree_training_gpu.sh
#
# Extra knobs on top of the CPU script:
#   DEVICE      device passed to hydra                          (default: cuda)
#   VENV        virtualenv to activate    (default: ~/scratch/venvs/gflownet-env-gpu)

set -u

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REPO="/home/mila/a/arnit/gflownet"
VENV="${VENV:-$HOME/scratch/venvs/gflownet-env-gpu}"
# Must match the --SBATCH --output directory above (sbatch directives cannot
# expand variables, so the path is spelled out there).
SLURM_LOG_DIR="$SCRATCH/gflownet-logs/slurm"

EXP_NAME="${EXP_NAME:-TREECLASS_CLS_GPU}"
EXP_CONFIG="${EXP_CONFIG:-tree/trfm_classification_tree}"
DATASET="${DATASET:-iris}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
FORCE="${FORCE:-0}"
split="${SLURM_ARRAY_TASK_ID:-1}"
# The dataset stays in $REPO (not in the code snapshot below): env.data_path
# enters the config hash, so it must be a stable path across jobs.
csv_path="$REPO/tests/data/tree/${DATASET}/${DATASET}_${split}.csv"

# -----------------------------------------------------------------------------
# Job header (also answers "which job, when, on what, from which commit")
# -----------------------------------------------------------------------------
module load python/3.10
source "$VENV/bin/activate"

started_at="$(date '+%Y-%m-%d %H:%M:%S %Z')"
git_commit="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
git_branch="$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
git_dirty=""
git -C "$REPO" diff --quiet HEAD 2>/dev/null || git_dirty=" (dirty)"

# -----------------------------------------------------------------------------
# Code snapshot: copy the repo to the compute node's local disk and run from
# there, so the checkout on scratch can be edited or switched to another branch
# while jobs are running. The snapshot is taken when the job STARTS (and taken
# again after every preemption requeue), not at submission time; the LAUNCH
# commit-mismatch warning below is what flags a code change between requeues.
# -----------------------------------------------------------------------------
if [ -n "${SLURM_TMPDIR:-}" ]; then
    CODE_DIR="$SLURM_TMPDIR/gflownet"
    rsync -a --exclude ".git" --exclude "__pycache__" "$REPO/" "$CODE_DIR/"
else
    # Outside Slurm (local debugging): run directly from the repo.
    CODE_DIR="$REPO"
fi
cd "$CODE_DIR" || exit 1
HELPERS="$CODE_DIR/gflownet/envs/tree/helpers_for_experiments"

echo "============================================================"
echo " Started            : $started_at"
echo " Slurm job          : ${SLURM_JOB_ID:-none}  (array ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none})"
echo " Node               : $(hostname)"
echo " GPU                : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none visible')"
echo " Repo               : $REPO @ ${git_branch} ${git_commit}${git_dirty}"
echo " Code dir           : $CODE_DIR"
echo " Python             : $(which python)"
echo " Experiment config  : $EXP_CONFIG"
echo " Device             : $DEVICE"
echo " Dataset            : $DATASET split $split -> $csv_path"
echo "============================================================"

if [ ! -f "$csv_path" ]; then
    echo "ERROR: dataset file not found: $csv_path"
    exit 1
fi

# set_device() falls back to CPU silently when CUDA is unavailable; fail loudly
# instead of wasting a GPU allocation on a CPU run.
if [ "$DEVICE" = "cuda" ]; then
    python - <<'EOF' || exit 1
import torch
assert torch.cuda.is_available(), (
    f"torch {torch.__version__} cannot see a CUDA device. "
    "Is VENV pointing at the CUDA-enabled environment?"
)
print(f"torch {torch.__version__} sees: {torch.cuda.get_device_name(0)}")
EOF
fi

# Keep torch's intra-op threading within our allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Trajectory lengths -- and therefore the size of every activation tensor --
# change from iteration to iteration, so the caching allocator accumulates
# unusable blocks of the wrong size and OOMs with GBs still "reserved but
# unallocated". Expandable segments let it resize a segment instead.
# This is an env var, not a config value: it does not enter the config hash,
# so adding it does not invalidate existing run directories.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Be patient with wandb init on busy nodes.
export WANDB_INIT_TIMEOUT=300
export WANDB__SERVICE_WAIT=300

# -----------------------------------------------------------------------------
# Resolve the config once: hash + the values the run name is built from
# -----------------------------------------------------------------------------
# Paths must be absolute: hydra.job.chdir=True moves the process into the run
# directory, so anything relative would resolve against the wrong place.
overrides=(
    "+experiments=$EXP_CONFIG"
    "env.data_path=$csv_path"
    "seed=$SEED"
    "device=$DEVICE"
    "$@"
)

cfg_block="$(python "$HELPERS/config_hash.py" "${overrides[@]}")" || {
    echo "ERROR: could not resolve the config; check the overrides above."
    exit 1
}
eval "$cfg_block"

run_name="${EXP_NAME}_${CFG_DATASET}${CFG_SPLIT}_depth${CFG_MAX_DEPTH}"
run_name="${run_name}_steps${CFG_N_STEPS}_lr${CFG_LR}_${CFG_POLICY}"
run_name="${run_name}_seed${CFG_SEED}_${CFG_HASH}"

run_dir="$RUNS_ROOT/$EXP_NAME/$run_name"
eval_json="$run_dir/eval_results.json"
samples_pkl="$run_dir/samples/gfn_samples.pkl"

echo " Run name           : $run_name"
echo " Run directory      : $run_dir"
echo "============================================================"

# -----------------------------------------------------------------------------
# LAUNCH record: appended once per launch, so resumes leave a trail
# -----------------------------------------------------------------------------
mkdir -p "$run_dir"
{
    echo "--- launch $started_at ---"
    echo "run_name     : $run_name"
    echo "config_hash  : $CFG_HASH"
    echo "git          : ${git_branch} ${git_commit}${git_dirty}"
    echo "slurm_job    : ${SLURM_JOB_ID:-none} (array ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none})"
    echo "node         : $(hostname)"
    echo "sbatch script: $0"
    if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
        echo "slurm log    : $SLURM_LOG_DIR/${SLURM_JOB_NAME}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
    fi
    echo "overrides    : ${overrides[*]}"
    echo
} >> "$run_dir/LAUNCH"

# The hash covers the config, not the code: warn when this launch runs a
# different commit than the one that started the run.
first_commit="$(grep -m1 '^git *:' "$run_dir/LAUNCH" | awk '{print $4}')"
if [ -n "$first_commit" ] && [ "$first_commit" != "$git_commit" ]; then
    echo "WARNING: this run was started at commit $first_commit, now on $git_commit."
    echo "         The config hash does not cover code changes."
fi

# -----------------------------------------------------------------------------
# Train (from scratch or resumed), then evaluate
# -----------------------------------------------------------------------------
if [ -f "$eval_json" ] && [ "$FORCE" != "1" ]; then
    echo "Run already complete ($eval_json exists) -- nothing to do."
    echo "Set FORCE=1 to rerun it anyway."
    exit 0
fi

status=0
if [ -f "$samples_pkl" ] && [ "$FORCE" != "1" ]; then
    echo "Training already finished; only the evaluation is missing."
else
    # Quarantine a checkpoint truncated by a preemption, otherwise the resume
    # below would fail on it again on every requeue.
    python "$HELPERS/validate_checkpoints.py" "$run_dir/ckpts"

    if compgen -G "$run_dir/ckpts/*.ckpt" > /dev/null && [ "$FORCE" != "1" ]; then
        echo ">>> Resuming from $run_dir/ckpts"
        resume_dir="$run_dir/resume/$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-local}"
        python resume.py \
            rundir="$run_dir" \
            hydra.run.dir="$resume_dir" \
            hydra.job.chdir=True
        status=$?
        # resume.py writes its final samples into its own working directory;
        # move them where a first-launch run puts them.
        if [ -f "$resume_dir/gfn_samples.pkl" ]; then
            mkdir -p "$run_dir/samples"
            mv "$resume_dir"/gfn_samples.csv "$resume_dir"/gfn_samples.pkl \
                "$run_dir/samples/"
        fi
    else
        echo ">>> Training from scratch"
        python train.py "${overrides[@]}" \
            logger.run_name="$run_name" \
            logger.run_name_date=False \
            logger.run_name_job=False \
            hydra.run.dir="$run_dir" \
            hydra.job.chdir=True
        status=$?
    fi

    if [ $status -ne 0 ]; then
        echo "$run_name: training exited with code $status; skipping evaluation."
        exit $status
    fi
fi

if [ ! -f "$samples_pkl" ]; then
    echo "ERROR: training reported success but $samples_pkl is missing."
    exit 1
fi

echo ">>> Evaluating $samples_pkl"
python "$CODE_DIR/gflownet/envs/tree/eval_tree.py" \
    --samples_path "$samples_pkl" \
    --data_path "$csv_path" \
    --alpha_value "$CFG_ALPHA" \
    --n_dirichlet_samples 10 \
    --output "$eval_json"
status=$?

echo "============================================================"
echo " Finished           : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo " $run_name exited with code $status"
echo "============================================================"
exit $status