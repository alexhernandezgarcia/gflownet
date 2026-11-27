#!/bin/bash
# =============================================================================
# ONE composite-tree CLASSIFICATION run: train (or resume), then evaluate.
# =============================================================================
#
# Classification twin of reg_tree_worker.sh: the training / resume / naming /
# LAUNCH-record logic is identical, only the defaults and the final evaluation
# differ. Regression runs are evaluated with
# helpers_for_experiments/eval_regression_tree.py (RMSE / R2, --run_dir based);
# classification runs use gflownet/envs/tree/eval_tree.py, which needs the
# dataset path and the Dirichlet prior alpha (read back from the resolved
# config as $CFG_ALPHA, so it can never contradict the training run).
#
# Usage:
#   cls_tree_worker.sh <split> [extra hydra overrides...]
#
# Environment (all optional except SCRATCH, which the cluster sets):
#   REPO          canonical checkout                      (default $HOME/gflownet)
#   CODE_DIR      snapshot actually executed              (default $REPO)
#   EXP_NAME      campaign name / top run-dir level       (default TREECLASS_CLS)
#   EXP_CONFIG    config under config/experiments         (default tree/classification_tree)
#   DATASET       dataset dir under tests/data/tree       (default wine)
#   SEED          random seed                             (default 0)
#   RUNS_ROOT     root of the run tree                    (default $SCRATCH/gflownet-logs)
#   FORCE         1 = retrain even if eval_results.json   (default 0)
#   CPUS_PER_RUN  threads handed to torch/BLAS            (default 4)
#   WANDB_MODE    offline | disabled | online             (default offline)
# =============================================================================

set -u

split="${1:?usage: cls_tree_worker.sh <split> [hydra overrides...]}"
shift

REPO="${REPO:-$HOME/gflownet}"
CODE_DIR="${CODE_DIR:-$REPO}"
EXP_NAME="${EXP_NAME:-TREECLASS_CLS}"
EXP_CONFIG="${EXP_CONFIG:-tree/classification_tree}"
DATASET="${DATASET:-wine}"
SEED="${SEED:-0}"
RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
FORCE="${FORCE:-0}"
CPUS_PER_RUN="${CPUS_PER_RUN:-4}"

HELPERS="$CODE_DIR/gflownet/envs/tree/helpers_for_experiments"
# The dataset is read from $REPO, not from the snapshot: env.data_path enters
# the config hash, so it has to be a stable path across jobs and clusters.
csv_path="$REPO/tests/data/tree/${DATASET}/${DATASET}_${split}.csv"

# -----------------------------------------------------------------------------
# Threading: keep each run inside its share of the node. Critical on Trillium,
# where dozens of these run side by side and torch would otherwise grab 192
# threads each.
# -----------------------------------------------------------------------------
export OMP_NUM_THREADS="$CPUS_PER_RUN"
export MKL_NUM_THREADS="$CPUS_PER_RUN"
export OPENBLAS_NUM_THREADS="$CPUS_PER_RUN"

# -----------------------------------------------------------------------------
# Anything that wants to write must write outside $HOME: on Trillium compute
# nodes $HOME and $PROJECT are mounted READ-ONLY. matplotlib and friends will
# happily crash a job over a cache directory otherwise.
# -----------------------------------------------------------------------------
work_tmp="${SLURM_TMPDIR:-/tmp/$USER}/cls_tree_${DATASET}_${split}_${SEED}_$$"
mkdir -p "$work_tmp"
export MPLCONFIGDIR="$work_tmp/mplconfig"
export XDG_CACHE_HOME="$work_tmp/cache"
export TMPDIR="${SLURM_TMPDIR:-$TMPDIR}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

# -----------------------------------------------------------------------------
# wandb: offline by default. Nothing below touches the network, and the run
# directories left in $WANDB_DIR are synced later from a login node.
# -----------------------------------------------------------------------------
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$RUNS_ROOT/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$RUNS_ROOT/wandb/.cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$RUNS_ROOT/wandb/.config}"
export WANDB_INIT_TIMEOUT=300
export WANDB__SERVICE_WAIT=300
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

cd "$CODE_DIR" || exit 1

started_at="$(date '+%Y-%m-%d %H:%M:%S %Z')"
git_commit="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
git_branch="$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
git_dirty=""
git -C "$REPO" diff --quiet HEAD 2>/dev/null || git_dirty=" (dirty)"

echo "============================================================"
echo " Started            : $started_at"
echo " Cluster            : ${CC_CLUSTER:-unknown}"
echo " Slurm job          : ${SLURM_JOB_ID:-none}  (array ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none})"
echo " Node               : $(hostname)"
echo " Repo               : $REPO @ ${git_branch} ${git_commit}${git_dirty}"
echo " Code dir           : $CODE_DIR"
echo " Python             : $(which python)"
echo " Experiment config  : $EXP_CONFIG"
echo " Dataset            : $DATASET split $split -> $csv_path"
echo " Threads per run    : $CPUS_PER_RUN"
echo " WANDB_MODE         : $WANDB_MODE  (dir: $WANDB_DIR)"
echo "============================================================"

if [ ! -f "$csv_path" ]; then
    echo "ERROR: dataset file not found: $csv_path"
    exit 1
fi

# -----------------------------------------------------------------------------
# Resolve the config once: hash + the values the run name is built from
# -----------------------------------------------------------------------------
overrides=(
    "+experiments=$EXP_CONFIG"
    "env.data_path=$csv_path"
    "seed=$SEED"
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
    echo "cluster      : ${CC_CLUSTER:-unknown}"
    echo "git          : ${git_branch} ${git_commit}${git_dirty}"
    echo "slurm_job    : ${SLURM_JOB_ID:-none} (array ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:-none})"
    echo "node         : $(hostname)"
    echo "wandb_mode   : $WANDB_MODE"
    echo "overrides    : ${overrides[*]}"
    echo
} >> "$run_dir/LAUNCH"

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
    python "$HELPERS/validate_checkpoints.py" "$run_dir/ckpts"

    if compgen -G "$run_dir/ckpts/*.ckpt" > /dev/null && [ "$FORCE" != "1" ]; then
        echo ">>> Resuming from $run_dir/ckpts"
        resume_dir="$run_dir/resume/$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-local}"
        python resume.py \
            rundir="$run_dir" \
            hydra.run.dir="$resume_dir" \
            hydra.job.chdir=True
        status=$?
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
    --output "$eval_json"
status=$?

rm -rf "$work_tmp"

echo "============================================================"
echo " Finished           : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo " $run_name exited with code $status"
echo "============================================================"
exit $status
