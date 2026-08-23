#!/bin/bash
#SBATCH --job-name=magic_resume
#SBATCH --output=/home/mila/a/arnit/scratch/gflownet-logs/slurm/%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --requeue

# =============================================================================
# Resume ONE existing run from its latest checkpoint. Training only: no eval.
# =============================================================================
#
# Companion worker of resume_crashed_magic.py, which scans the campaign
# directories, decides which runs are unfinished and submits one of these jobs
# per run. All resources that differ between runs (partition, gres, mem, time)
# are passed by the driver on the sbatch command line, which overrides the
# directives above; only the invariants live here.
#
# Required (via --export):
#   RUN_DIR   absolute path of the run directory to resume
# Optional:
#   REPO      git checkout to run from        (default $HOME/gflownet)
#   VENV      virtualenv; default depends on the run's device:
#             $SCRATCH/venvs/gflownet-env      for device: cpu
#             $SCRATCH/venvs/gflownet-env-gpu  for device: cuda
#
# The run's own .hydra/config.yaml is the single source of truth: the device
# (and with it the venv) is read from there, and resume.py reloads the full
# config, so the continued run is the exact run that crashed.
#
# Idempotent by design: if training already finished (ckpts/final.ckpt or
# samples/gfn_samples.pkl exists) the job exits 0 immediately. Together with
# #SBATCH --requeue this makes preemption safe -- the requeued job simply
# resumes again from whatever checkpoint is newest.

set -u

REPO="${REPO:-$HOME/gflownet}"

if [ -z "${RUN_DIR:-}" ] || [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: RUN_DIR is unset or not a directory: '${RUN_DIR:-}'"
    exit 1
fi
name="$(basename "$RUN_DIR")"

# Training already done (this job may be a requeue of a job that finished).
if [ -f "$RUN_DIR/ckpts/final.ckpt" ] || [ -f "$RUN_DIR/samples/gfn_samples.pkl" ]; then
    echo "$name: training already finished -- nothing to do."
    exit 0
fi

# The run's resolved config says whether it was a CPU or a CUDA run.
DEVICE="$(sed -n 's/^device:[[:space:]]*//p' "$RUN_DIR/.hydra/config.yaml" | head -1)"
DEVICE="${DEVICE:-cpu}"
if [ "$DEVICE" = "cuda" ]; then
    VENV="${VENV:-$SCRATCH/venvs/gflownet-env-gpu}"
else
    VENV="${VENV:-$SCRATCH/venvs/gflownet-env}"
fi

module load python/3.10
source "$VENV/bin/activate"

started_at="$(date '+%Y-%m-%d %H:%M:%S %Z')"
git_commit="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
git_branch="$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

# Code snapshot on node-local disk, so the checkout can be edited while jobs run.
if [ -n "${SLURM_TMPDIR:-}" ]; then
    CODE_DIR="$SLURM_TMPDIR/gflownet"
    rsync -a --exclude ".git" --exclude "__pycache__" "$REPO/" "$CODE_DIR/"
else
    CODE_DIR="$REPO"
fi
cd "$CODE_DIR" || exit 1
HELPERS="$CODE_DIR/gflownet/envs/tree/helpers_for_experiments"

echo "============================================================"
echo " Started            : $started_at"
echo " Slurm job          : ${SLURM_JOB_ID:-none} on $(hostname)"
echo " Run                : $name"
echo " Device             : $DEVICE"
echo " GPU                : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'none visible')"
echo " Repo               : $REPO @ ${git_branch} ${git_commit}"
echo " Python             : $(which python)"
echo "============================================================"

if [ "$DEVICE" = "cuda" ]; then
    # set_device() falls back to CPU silently; fail loudly instead of burning
    # a GPU allocation on a CPU run.
    python - <<'EOF' || exit 1
import torch
assert torch.cuda.is_available(), (
    f"torch {torch.__version__} cannot see a CUDA device. "
    "Is VENV pointing at the CUDA-enabled environment?"
)
print(f"torch {torch.__version__} sees: {torch.cuda.get_device_name(0)}")
EOF
    # Trajectory lengths change every iteration, so the caching allocator
    # fragments and OOMs with GBs "reserved but unallocated"; expandable
    # segments let it resize blocks instead.
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export WANDB_INIT_TIMEOUT=300
export WANDB__SERVICE_WAIT=300

{
    echo "--- resume launch $started_at ---"
    echo "git          : ${git_branch} ${git_commit}"
    echo "slurm_job    : ${SLURM_JOB_ID:-none}"
    echo "node         : $(hostname)"
    echo "sbatch script: $0"
    echo
} >> "$RUN_DIR/LAUNCH"

# Quarantine a checkpoint truncated by the kill that crashed this run,
# otherwise find_latest_checkpoint keeps preferring it and the resume fails
# the same way on every launch.
python "$HELPERS/validate_checkpoints.py" "$RUN_DIR/ckpts"

if ! compgen -G "$RUN_DIR/ckpts/*.ckpt" > /dev/null; then
    echo "ERROR: no loadable checkpoint left in $RUN_DIR/ckpts; not resuming."
    exit 1
fi

echo ">>> Resuming from $RUN_DIR/ckpts"
resume_dir="$RUN_DIR/resume/$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-local}"
python resume.py \
    rundir="$RUN_DIR" \
    hydra.run.dir="$resume_dir" \
    hydra.job.chdir=True
status=$?

# resume.py writes its final samples into its own working directory; move them
# where a first-launch run puts them (also the marker eval scripts look for).
if [ -f "$resume_dir/gfn_samples.pkl" ]; then
    mkdir -p "$RUN_DIR/samples"
    mv "$resume_dir"/gfn_samples.csv "$resume_dir"/gfn_samples.pkl \
        "$RUN_DIR/samples/"
fi

echo "============================================================"
echo " Finished           : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo " $name exited with code $status"
echo "============================================================"
exit $status
