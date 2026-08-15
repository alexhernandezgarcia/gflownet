#!/bin/bash
#SBATCH --job-name=reg_tree
#SBATCH --output=/home/mila/a/arnit/scratch/gflownet-logs/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=long-cpu,long-cpu-eek
#SBATCH --array=1-5
#SBATCH --requeue

# =============================================================================
# Composite-Tree regression experiments: one array task per dataset split.
# =============================================================================
#
# Regression counterpart of run_classification_tree_training.sh; same layout,
# same naming rules, same resume/skip logic. Only three things differ:
#   - the default experiment config (tree/regression_tree) and dataset;
#   - the run name carries no alpha (the NIG leaf prior has no single
#     concentration parameter the way the Dirichlet one does);
#   - the final evaluation runs helpers_for_experiments/eval_regression_tree.py
#     (RMSE / R2 via RegressionTree.test) instead of eval_tree.py, which is
#     classification-only.
#
# Layout produced (RUNS_ROOT defaults to $SCRATCH/gflownet-logs):
#
#   $RUNS_ROOT/<EXP_NAME>/<run_name>/
#       LAUNCH             one appended record per launch/resume (git commit,
#                          slurm ids, timestamp, full hydra override list)
#       .hydra/            resolved config of the run
#       ckpts/  data/  samples/
#       eval_results.json  final regression metrics; doubles as "done" marker
#       resume/<ts>-<jobid>/   one hydra dir per resume job
#
#   run_name = <EXP_NAME>_<dataset><split>_depth<D>_steps<N>_lr<LR>_<policy>_seed<S>_<hash>
#
# Every field after EXP_NAME is read back from the *resolved* config rather
# than hard-coded here, so a run name can never contradict the config it was
# trained with. The trailing <hash> is an 8-char digest of the entire resolved
# config (see helpers_for_experiments/config_hash.py): change any setting,
# named in the run name or not, and the run lands in a new directory. Relaunch
# with an unchanged config and the existing directory is reused -- resumed if
# incomplete, skipped if already evaluated.
#
# Usage:
#   mkdir -p $SCRATCH/gflownet-logs/slurm
#   sbatch mila/tree/run_reg_tree_experiment.sh
#
#   # name the campaign and change any hydra setting on the CLI:
#   sbatch --export=ALL,EXP_NAME=TREECLASS_REG_lr10e-3,DATASET=energy \
#          mila/tree/run_reg_tree_experiment.sh \
#          gflownet.optimizer.lr=1e-3
#
#   # only splits 1 and 3:
#   sbatch --array=1,3 mila/tree/run_reg_tree_experiment.sh
#
# Environment knobs (all overridable via --export=ALL,VAR=value):
#   EXP_NAME    campaign name; first component of the run name and the
#               directory level above it              (default: TREECLASS_REG)
#   EXP_CONFIG  experiment config under config/experiments (default:
#               tree/regression_tree)
#   DATASET     dataset directory under tests/data/tree     (default: diabetes;
#               regression datasets: diabetes, energy, concrete)
#   SEED        random seed                                        (default: 0)
#   RUNS_ROOT   root of the run tree      (default: $SCRATCH/gflownet-logs)
#   TOP_K       trees ranked for the top-k/top-1 metrics          (default: 10)
#   FORCE       1 = ignore the "already done" marker and retrain   (default: 0)
#
# Any positional arguments are passed straight through as extra hydra
# overrides, and are included in the config hash like every other setting.
#
# Resource note: --cpus-per-task and --mem are PER ARRAY TASK. The settings
# above give each of the 5 tasks 4 CPUs and 32 GB (20 CPUs / 160 GB in total).

set -u

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REPO="/home/mila/a/arnit/gflownet"
VENV="$HOME/scratch/venvs/gflownet-env"
# Must match the --SBATCH --output directory above (sbatch directives cannot
# expand variables, so the path is spelled out there).
SLURM_LOG_DIR="$SCRATCH/gflownet-logs/slurm"

EXP_NAME="${EXP_NAME:-TREECLASS_REG}"
EXP_CONFIG="${EXP_CONFIG:-tree/regression_tree}"
DATASET="${DATASET:-diabetes}"
SEED="${SEED:-0}"
RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-logs}"
TOP_K="${TOP_K:-10}"
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
echo " Repo               : $REPO @ ${git_branch} ${git_commit}${git_dirty}"
echo " Code dir           : $CODE_DIR"
echo " Python             : $(which python)"
echo " Experiment config  : $EXP_CONFIG"
echo " Dataset            : $DATASET split $split -> $csv_path"
echo "============================================================"

if [ ! -f "$csv_path" ]; then
    echo "ERROR: dataset file not found: $csv_path"
    exit 1
fi

# Keep torch's intra-op threading within our allocation.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

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
# Rebuilds the RegressionTree env from the run's own .hydra/config.yaml, so the
# dataset, the target standardization and the NIG prior match training.
python "$HELPERS/eval_regression_tree.py" \
    --run_dir "$run_dir" \
    --samples_path "$samples_pkl" \
    --top_k_trees "$TOP_K" \
    --seed 0 \
    --output "$eval_json"
status=$?

echo "============================================================"
echo " Finished           : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo " $run_name exited with code $status"
echo "============================================================"
exit $status