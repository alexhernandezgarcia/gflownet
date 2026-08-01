#!/bin/bash
#SBATCH --job-name=reg_tree
#SBATCH --output=/home/mila/a/arnit/scratch/gflownet-logs/slurm/%x-%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=long-cpu,long-cpu-eek
#SBATCH --requeue
#
# RegressionTree (DT-GFN) sweep over dataset splits / seeds.
#
# Config: config/experiments/tree/regression_tree.yaml
# One array task per seed; the seed index selects BOTH the dataset split file
# (tests/data/tree/<dataset>/<dataset>_<seed>.csv, each carrying its own
# `Split` column) AND the RNG seed passed to train.py, so the tasks are
# independent replicates. Pin the RNG seed across tasks with RNG_SEED=<n> if
# you want to vary only the data split.
#
# ---------------------------------------------------------------------------
# Usage (from the login node, or from inside an interactive job -- the script
# submits itself as a separate batch job either way):
#
#   bash mila/tree/run_reg_tree_experiment.sh \
#       --dataset diabetes --depth 5 --steps 20000 --seeds 1-5
#
# Options (all also settable as environment variables):
#   --dataset <name>   DATASET   dataset directory under tests/data/tree/
#   --depth <int>      DEPTH     env.max_depth
#   --steps <int>      STEPS     gflownet.optimizer.n_train_steps
#   --seeds <spec>     SEEDS     sbatch array spec: "1", "1-3", "1-5", "1,3,5"
#
# Other environment overrides:
#   LAUNCH_TAG   reuse an existing launch group -> RESUMES it (see below)
#   RUNS_ROOT    default $SCRATCH/gflownet-runs
#   EXPERIMENT   default regression_tree
#   RNG_SEED     pin train.py's seed for every task (default: the seed index)
#   CKPT_PERIOD  evaluator.checkpoints_period (default 500)
#   N_SAMPLES    final sample set size (default 1000)
#   WANDB_ONLINE True/False (default True)
#   CPUS MEM TIME PARTITION   sbatch resources
#
# ---------------------------------------------------------------------------
# Layout. Every submission mints a LAUNCH_TAG (submit-time timestamp) and
# writes into its own group directory:
#
#   $RUNS_ROOT/$EXPERIMENT/<dataset>_d<depth>_steps<steps>__<LAUNCH_TAG>/
#   |-- LAUNCH                     git commit + full command + submit host/time
#   |-- slurm/reg_tree-<A>_<a>.out per-task job logs
#   `-- seed<n>/                   one run directory per array task
#       |-- .hydra/  ckpts/  data/  samples/
#       `-- resume/<ts>-<jobid>/   one per resume job
#
# Because the group directory is keyed on the submit-time tag and NOT on any
# hyperparameter, relaunching the same setup can never overwrite or interleave
# with an earlier launch -- including when the thing that changed between them
# is a hyperparameter that is not part of the directory name. The LAUNCH file
# is what records the full configuration of a group.
#
# ---------------------------------------------------------------------------
# Resuming. Per task the decision is made from the run directory alone:
#   samples/gfn_samples.pkl exists -> finished, skip
#   ckpts/*.ckpt exist             -> resume.py from the latest good checkpoint
#   otherwise                      -> train.py from scratch
#
# LAUNCH_TAG is exported into the batch environment, so a task PREEMPTED and
# requeued by SLURM (same job id, environment restored from submit time)
# recomputes the identical run directory and resumes automatically.
#
# A job that hits its TIME LIMIT is not requeued by SLURM. To continue it,
# resubmit with the same tag -- finished tasks exit immediately and unfinished
# ones pick up from their latest checkpoint:
#
#   LAUNCH_TAG=20260801-143022 bash mila/tree/run_reg_tree_experiment.sh \
#       --dataset diabetes --depth 5 --steps 20000 --seeds 1-5
#
# Note that on resume all hyperparameters come from the stored
# <run_dir>/.hydra/config.yaml, not from this script's arguments: passing a
# different --depth or --steps to a resumed group has no effect on the runs
# that already have checkpoints.

set -u

REPO="${REPO:-$HOME/gflownet}"
VENV="${VENV:-$HOME/scratch/venvs/gflownet-env}"
SCRIPT="$REPO/mila/tree/run_reg_tree_experiment.sh"

RUNS_ROOT="${RUNS_ROOT:-$SCRATCH/gflownet-runs}"
EXPERIMENT="${EXPERIMENT:-regression_tree}"

DATASET="${DATASET:-}"
DEPTH="${DEPTH:-}"
STEPS="${STEPS:-}"
SEEDS="${SEEDS:-1-5}"

CKPT_PERIOD="${CKPT_PERIOD:-500}"
N_SAMPLES="${N_SAMPLES:-1000}"
WANDB_ONLINE="${WANDB_ONLINE:-True}"

CPUS="${CPUS:-4}"
MEM="${MEM:-32G}"
TIME="${TIME:-24:00:00}"
PARTITION="${PARTITION:-long-cpu,long-cpu-eek}"

# ---- Argument parsing -------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --depth)   DEPTH="$2";   shift 2 ;;
        --steps)   STEPS="$2";   shift 2 ;;
        --seeds)   SEEDS="$2";   shift 2 ;;
        -h|--help) sed -n '9,80p' "$SCRIPT"; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

for var in DATASET DEPTH STEPS; do
    if [ -z "${!var}" ]; then
        echo "ERROR: --${var,,} is required (see --help)" >&2
        exit 1
    fi
done

GROUP_NAME="${DATASET}_d${DEPTH}_steps${STEPS}"

# ---- Submit wrapper ---------------------------------------------------------
# Runs once, wherever the script was invoked from. Gating on
# SLURM_ARRAY_TASK_ID (rather than SLURM_JOB_ID) means this also works from a
# shell inside an interactive allocation: the sweep is queued as its own batch
# job and does not run in the interactive session.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    export LAUNCH_TAG="${LAUNCH_TAG:-$(date +%Y%m%d-%H%M%S)}"
    GROUP_DIR="$RUNS_ROOT/$EXPERIMENT/${GROUP_NAME}__${LAUNCH_TAG}"

    if [ -d "$GROUP_DIR" ]; then
        echo "[submit] REUSING existing group $GROUP_DIR"
        echo "[submit] finished tasks will be skipped, unfinished ones resumed"
    fi
    mkdir -p "$GROUP_DIR/slurm"

    # Record everything the directory name does not carry. This is the answer
    # to "which code and which settings produced this group".
    {
        echo "launch_tag:   $LAUNCH_TAG"
        echo "submitted:    $(date --iso-8601=seconds)"
        echo "submitted_by: $USER@$(hostname)"
        echo "dataset:      $DATASET"
        echo "max_depth:    $DEPTH"
        echo "n_train_steps: $STEPS"
        echo "seeds:        $SEEDS"
        echo "rng_seed:     ${RNG_SEED:-<seed index>}"
        echo "ckpt_period:  $CKPT_PERIOD"
        echo "n_samples:    $N_SAMPLES"
        echo "experiment_config: config/experiments/tree/regression_tree.yaml"
        echo "repo:         $REPO"
        echo "git_commit:   $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "git_branch:   $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
        echo "git_dirty:    $(test -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" && echo yes || echo no)"
        echo "command:      $SCRIPT --dataset $DATASET --depth $DEPTH --steps $STEPS --seeds $SEEDS"
        echo "---"
    } >> "$GROUP_DIR/LAUNCH"

    echo "[submit] group dir: $GROUP_DIR"
    echo "[submit] queueing array $SEEDS on $PARTITION"
    exec sbatch \
        --array="$SEEDS" \
        --job-name="reg_tree_${DATASET}" \
        --output="$GROUP_DIR/slurm/%x-%A_%a.out" \
        --cpus-per-task="$CPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --partition="$PARTITION" \
        --export=ALL,LAUNCH_TAG="$LAUNCH_TAG",DATASET="$DATASET",DEPTH="$DEPTH",STEPS="$STEPS",SEEDS="$SEEDS",CKPT_PERIOD="$CKPT_PERIOD",N_SAMPLES="$N_SAMPLES",WANDB_ONLINE="$WANDB_ONLINE",RUNS_ROOT="$RUNS_ROOT",EXPERIMENT="$EXPERIMENT" \
        "$SCRIPT"
fi

# ---- Task body --------------------------------------------------------------
seed="$SLURM_ARRAY_TASK_ID"
rng_seed="${RNG_SEED:-$seed}"

GROUP_DIR="$RUNS_ROOT/$EXPERIMENT/${GROUP_NAME}__${LAUNCH_TAG}"
run_dir="$GROUP_DIR/seed${seed}"
run_name="${GROUP_NAME}_seed${seed}__${LAUNCH_TAG}"
csv_path="$REPO/tests/data/tree/${DATASET}/${DATASET}_${seed}.csv"

echo "=========================================================="
echo "[$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID] $run_name"
echo "Dataset:  $csv_path"
echo "Depth:    $DEPTH | steps: $STEPS | rng seed: $rng_seed"
echo "Run dir:  $run_dir"
echo "Node:     $(hostname) | started: $(date)"
echo "=========================================================="

if [ ! -f "$csv_path" ]; then
    echo "ERROR: missing dataset split $csv_path" >&2
    exit 1
fi

module load python/3.10
source "$VENV/bin/activate"
cd "$REPO"

# Keep torch's intra-op threading within our allocation; without this torch
# sizes its thread pool to all cores of the (shared) node.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# Be patient with wandb init on busy nodes (a 90s timeout has killed tasks).
export WANDB_INIT_TIMEOUT=300
export WANDB__SERVICE_WAIT=300

t_run=$SECONDS

if [ -f "$run_dir/samples/gfn_samples.pkl" ]; then
    echo "$run_name already complete -- nothing to do"
    exit 0

elif compgen -G "$run_dir/ckpts/*.ckpt" > /dev/null 2>&1; then
    echo "$run_name found checkpoints -- resuming"

    # Quarantine unloadable checkpoints (e.g. a write cut short by preemption):
    # probe newest-first and rename broken ones so find_latest_checkpoint()
    # falls back to the next-newest good one.
    for ckpt in $(ls "$run_dir"/ckpts/iter_*.ckpt 2>/dev/null | sort -rV); do
        if python -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu')" \
                "$ckpt" > /dev/null 2>&1; then
            echo "$run_name resuming from $(basename "$ckpt")"
            break
        fi
        echo "$run_name WARNING: $(basename "$ckpt") unreadable -- renamed to .corrupt"
        mv "$ckpt" "$ckpt.corrupt"
    done

    # resume.py takes all hyperparameters from <run_dir>/.hydra/config.yaml.
    # Give it its own output directory inside the run dir -- never $run_dir
    # itself, which would overwrite that stored config with the resume config.
    resume_dir="$run_dir/resume/$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID}"
    python resume.py \
        rundir="$run_dir" \
        seed="$rng_seed" \
        n_samples="$N_SAMPLES" \
        hydra.run.dir="$resume_dir" \
        hydra.job.chdir=True
    status=$?

    # resume.py writes its final samples into its own CWD; put them where a
    # first-launch run writes them, so samples/gfn_samples.pkl stays a valid
    # "this run is finished" marker regardless of how many resumes it took.
    if [ -f "$resume_dir/gfn_samples.pkl" ]; then
        mkdir -p "$run_dir/samples"
        mv "$resume_dir"/gfn_samples.csv "$resume_dir"/gfn_samples.pkl \
            "$run_dir/samples/"
    fi

else
    echo "$run_name starting from scratch"
    mkdir -p "$run_dir"
    python train.py +experiments=tree/regression_tree \
        env.data_path="$csv_path" \
        env.max_depth="$DEPTH" \
        gflownet.optimizer.n_train_steps="$STEPS" \
        evaluator.checkpoints_period="$CKPT_PERIOD" \
        seed="$rng_seed" \
        n_samples="$N_SAMPLES" \
        logger.do.online="$WANDB_ONLINE" \
        logger.run_name="$run_name" \
        logger.run_name_date=False \
        logger.run_name_job=False \
        hydra.run.dir="$run_dir" \
        hydra.job.chdir=True
    status=$?
fi

echo "----------------------------------------------------------"
echo "$run_name finished with exit code $status after $((SECONDS - t_run))s"
if [ ! -f "$run_dir/samples/gfn_samples.pkl" ]; then
    echo "Run is INCOMPLETE. Continue this group with:"
    echo "  LAUNCH_TAG=$LAUNCH_TAG bash $SCRIPT \\"
    echo "      --dataset $DATASET --depth $DEPTH --steps $STEPS --seeds $SEEDS"
fi
exit $status
