#!/bin/bash
#SBATCH --job-name=reg_tree_resume
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=24:00:00
#SBATCH --output=reg_tree_resume-%j.out
# =============================================================================
# Finish every unfinished run of an existing campaign, on ONE whole node.
# =============================================================================
#
# Written for Trillium, where the CPU subcluster schedules by NODE (192 cores /
# 768 GB, 15 min - 24 h walltime). A campaign launched with
# drac/trillium_reg_tree.sh that hits its walltime leaves behind a directory
# full of half-trained runs; this script picks that directory up and drives all
# of them to completion inside a single new node allocation.
#
# It does NOT re-derive run names or config hashes. Every run directory already
# contains everything needed to continue it:
#
#   <run_dir>/.hydra/config.yaml     the resolved config (resume.py reads this)
#   <run_dir>/.hydra/overrides.yaml  the original CLI overrides (scratch restart)
#   <run_dir>/ckpts/                 iter_<step>.ckpt / final.ckpt
#   <run_dir>/samples/gfn_samples.pkl   training finished
#   <run_dir>/eval_results.json      run complete; this is the "done" marker
#
# so the campaign root is scanned directly and each run is classified from what
# is on disk. Consequence: this one script handles a campaign containing a MIX
# of hyper-parameters (the REG_trillium directory has 10k/lr0.01 and
# 20k/lr0.001 runs side by side) without being told what they were.
#
# Per run, in the same order reg_tree_worker.sh uses:
#
#   DONE   eval_results.json exists                     -> skipped
#   EVAL   samples/gfn_samples.pkl exists, no eval JSON  -> evaluation only
#   TRAIN  otherwise -> quarantine truncated checkpoints, then
#            resume.py rundir=<run_dir>          if a loadable ckpt remains
#            train.py <overrides.yaml>           if not (crashed before ckpt 1)
#          followed by the evaluation.
#
# Scheduling: runs are executed by a worker pool of CONCURRENCY slots
# (= 192 / CPUS_PER_RUN, so 24 slots at the default 8 threads per run). The
# node is filled, and as soon as ONE run finishes the next queued run starts in
# its slot -- `wait -n` below -- so the node stays saturated until the queue is
# empty. Queue order is shortest-remaining-first (fewest training steps left
# first), which maximises the number of *completed* runs if the walltime runs
# out again.
#
# Being cut off is safe: partial progress is in <run_dir>/ckpts, so relaunching
# this exact script simply continues where it stopped and skips whatever is now
# DONE. Launching it twice in sequence is the intended usage when one node-day
# is not enough.
#
# Where this file lives does not matter: nothing here is resolved relative to
# the script, and no sibling script is sourced. Everything is found through
# $REPO / $CODE_DIR. On Trillium it sits next to the other DRAC launchers in
# drac/; in the Mila checkout it is versioned under mila/tree/.
#
# The --output directive below is RELATIVE, so Slurm resolves it against the
# directory you submit from. Trillium's submit filter rejects the job outright
# if that directory is under $HOME (read-only on compute nodes), so always pass
# an absolute --output on scratch, or submit from $SCRATCH.
#
# Usage:
#
#   mkdir -p $SCRATCH/gflownet-logs/slurm
#   sbatch --account=def-alexhg \
#          --output=$SCRATCH/gflownet-logs/slurm/reg_tree_resume-%j.out \
#          $HOME/gflownet/drac/resume_all_unfinished_runs.sh
#
#   # another campaign, shorter allocation:
#   sbatch --account=def-alexhg --time=12:00:00 \
#          --output=$SCRATCH/gflownet-logs/slurm/reg_tree_resume-%j.out \
#          --export=ALL,CAMPAIGN_ROOT=$SCRATCH/gflownet-logs/REG_trillium_d5_lr5e-3_shared \
#          $HOME/gflownet/drac/resume_all_unfinished_runs.sh
#
#   # see the plan without allocating anything (runs on the login node):
#   DRY_RUN=1 bash $HOME/gflownet/drac/resume_all_unfinished_runs.sh
#
# Environment knobs (all optional, pass with --export=ALL,VAR=value):
#   CAMPAIGN_ROOT  directory holding the run dirs
#                                 (default $SCRATCH/gflownet-logs/REG_trillium)
#   CPUS_PER_RUN   threads per run; sets the pool size          (default 8)
#   CORES_PER_NODE cores to spread over        (default $SLURM_CPUS_ON_NODE/192)
#   FILTER         only run dirs whose name matches this regex  (default: all)
#   ALLOW_SCRATCH  1 = restart from scratch a run with no usable checkpoint,
#                  0 = report it and leave it alone             (default 1)
#   RESERVE_MIN    do not start a new run within this many minutes of the
#                  walltime; the queue is reported instead      (default 15)
#   DRY_RUN        1 = classify, print the plan, exit           (default 0)
#   TOP_K          trees ranked for the top-k metrics           (default 10)
#   REPO / VENV / WANDB_MODE     as in drac/reg_tree_worker.sh
# =============================================================================

set -u

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
REPO="${REPO:-$HOME/gflownet}"
VENV="${VENV:-$SCRATCH/venvs/gflownet-env}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-$SCRATCH/gflownet-logs/REG_trillium}"
CPUS_PER_RUN="${CPUS_PER_RUN:-8}"
CORES_PER_NODE="${CORES_PER_NODE:-${SLURM_CPUS_ON_NODE:-192}}"
FILTER="${FILTER:-}"
ALLOW_SCRATCH="${ALLOW_SCRATCH:-1}"
RESERVE_MIN="${RESERVE_MIN:-15}"
DRY_RUN="${DRY_RUN:-0}"
TOP_K="${TOP_K:-10}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if [ ! -d "$CAMPAIGN_ROOT" ]; then
    echo "ERROR: campaign root not found: $CAMPAIGN_ROOT"
    exit 1
fi

module purge 2>/dev/null
module load StdEnv/2023 python/3.10
source "$VENV/bin/activate"

# One code snapshot per node, shared by every worker on it: the checkout in
# $HOME can then be edited while the job runs.
if [ -n "${SLURM_TMPDIR:-}" ] && [ "$DRY_RUN" != "1" ]; then
    CODE_DIR="$SLURM_TMPDIR/gflownet"
    rsync -a --exclude ".git" --exclude "__pycache__" "$REPO/" "$CODE_DIR/"
else
    CODE_DIR="$REPO"
fi
HELPERS="$CODE_DIR/gflownet/envs/tree/helpers_for_experiments"

# wandb offline: compute nodes have no network. The offline run directories are
# synced later from a login node (`wandb sync --sync-all` in $WANDB_DIR/..).
export WANDB_DIR="${WANDB_DIR:-$(dirname "$CAMPAIGN_ROOT")/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/.cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$WANDB_DIR/.config}"
export WANDB_INIT_TIMEOUT=300
export WANDB__SERVICE_WAIT=300
[ "$DRY_RUN" = "1" ] || mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# -----------------------------------------------------------------------------
# Pass 1: classify every run directory
#
# Emitted as "<remaining_steps>|<action>|<run_dir>" lines so the queue can be
# sorted numerically: fewest steps left first.
# -----------------------------------------------------------------------------
queue_file="$(mktemp)"
n_done=0 n_eval=0 n_train=0 n_broken=0 n_filtered=0

for run_dir in "$CAMPAIGN_ROOT"/*/; do
    run_dir="${run_dir%/}"
    name="$(basename "$run_dir")"

    # worker-logs/, resume-logs/ and anything else that is not a run
    [ -f "$run_dir/.hydra/config.yaml" ] || continue

    if [ -n "$FILTER" ] && ! [[ "$name" =~ $FILTER ]]; then
        n_filtered=$(( n_filtered + 1 ))
        continue
    fi

    if [ -f "$run_dir/eval_results.json" ]; then
        n_done=$(( n_done + 1 ))
        continue
    fi

    # Target and current step, read straight out of the run's own config and
    # checkpoint file names (no python: this loop runs over ~100 directories).
    n_steps="$(sed -n 's/^[[:space:]]*n_train_steps:[[:space:]]*//p' \
        "$run_dir/.hydra/config.yaml" | head -1)"
    [ -n "$n_steps" ] || n_steps=0

    last_iter=0
    if [ -f "$run_dir/ckpts/final.ckpt" ]; then
        last_iter="$n_steps"
    else
        for ckpt in "$run_dir"/ckpts/iter_*.ckpt; do
            [ -e "$ckpt" ] || break
            step="$(basename "$ckpt" .ckpt)"; step="${step#iter_}"
            step=$(( 10#$step ))            # file names are zero-padded
            (( step > last_iter )) && last_iter=$step
        done
    fi
    remaining=$(( n_steps - last_iter ))
    (( remaining < 0 )) && remaining=0

    if [ -f "$run_dir/samples/gfn_samples.pkl" ]; then
        # Training is over; only the evaluation is missing. Costs a couple of
        # minutes, so it goes to the head of the queue (remaining = 0).
        echo "0|EVAL|$run_dir" >> "$queue_file"
        n_eval=$(( n_eval + 1 ))
    elif compgen -G "$run_dir/ckpts/*.ckpt" > /dev/null; then
        echo "$remaining|RESUME|$run_dir" >> "$queue_file"
        n_train=$(( n_train + 1 ))
    elif [ "$ALLOW_SCRATCH" = "1" ] && [ -f "$run_dir/.hydra/overrides.yaml" ]; then
        # Died before writing a single checkpoint: rerun it from step 0 with
        # the overrides hydra recorded for it.
        echo "$n_steps|SCRATCH|$run_dir" >> "$queue_file"
        n_train=$(( n_train + 1 ))
    else
        echo "BROKEN (no checkpoint, no overrides.yaml): $name"
        n_broken=$(( n_broken + 1 ))
    fi
done

mapfile -t queue < <(sort -t'|' -k1,1n "$queue_file")
rm -f "$queue_file"
n_queued=${#queue[@]}

# -----------------------------------------------------------------------------
# Pool sizing and walltime deadline
# -----------------------------------------------------------------------------
CONCURRENCY=$(( CORES_PER_NODE / CPUS_PER_RUN ))
(( CONCURRENCY < 1 )) && CONCURRENCY=1

deadline=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
    end_time="$(scontrol show job "$SLURM_JOB_ID" -o 2>/dev/null \
        | tr ' ' '\n' | sed -n 's/^EndTime=//p' | head -1)"
    [ -n "$end_time" ] && deadline="$(date -d "$end_time" +%s 2>/dev/null)"
fi

echo "============================================================"
echo " Resume unfinished runs"
echo " Job                : ${SLURM_JOB_ID:-none} on $(hostname)"
echo " Campaign root      : $CAMPAIGN_ROOT"
echo " Code dir           : $CODE_DIR"
echo " Already complete   : $n_done"
echo " Queued             : $n_queued   ($n_eval eval-only, $n_train training)"
[ "$n_broken" -gt 0 ]   && echo " Broken (skipped)   : $n_broken"
[ "$n_filtered" -gt 0 ] && echo " Filtered out       : $n_filtered   (FILTER=$FILTER)"
echo " Threads per run    : $CPUS_PER_RUN   concurrency: $CONCURRENCY"
echo " Walltime ends      : ${end_time:-unknown}  (reserve ${RESERVE_MIN} min)"
echo "============================================================"
echo " Queue (shortest remaining first):"
for entry in "${queue[@]}"; do
    IFS='|' read -r rem action dir <<< "$entry"
    printf '   %-8s %7s steps left  %s\n' "$action" "$rem" "$(basename "$dir")"
done
echo "============================================================"

if [ "$n_queued" -eq 0 ]; then
    echo "Nothing to do: every run in $CAMPAIGN_ROOT is complete."
    exit 0
fi
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY_RUN=1 -- plan printed, nothing launched."
    exit 0
fi

LOG_DIR="$CAMPAIGN_ROOT/resume-logs/${SLURM_JOB_ID:-local}"
mkdir -p "$LOG_DIR"
STATUS_FILE="$LOG_DIR/status.txt"
: > "$STATUS_FILE"

# -----------------------------------------------------------------------------
# One run: resume (or restart), then evaluate. Mirrors the tail of
# drac/reg_tree_worker.sh, minus the config-hash / run-naming part, which is
# unnecessary here because the run directory is already known.
# -----------------------------------------------------------------------------
run_one () {
    local action="$1" run_dir="$2"
    local name; name="$(basename "$run_dir")"
    local log="$LOG_DIR/$name.out"

    {
        echo "=== $name [$action] started $(date '+%F %T') on $(hostname) ==="

        # Keep each run inside its share of the node; torch would otherwise
        # grab all 192 threads and the 24 runs would thrash each other.
        export OMP_NUM_THREADS="$CPUS_PER_RUN"
        export MKL_NUM_THREADS="$CPUS_PER_RUN"
        export OPENBLAS_NUM_THREADS="$CPUS_PER_RUN"

        # $HOME and $PROJECT are read-only on Trillium compute nodes: every
        # cache a library might want to write goes to node-local scratch.
        local work_tmp="${SLURM_TMPDIR:-/tmp/$USER}/resume_${name}_$$"
        mkdir -p "$work_tmp/mplconfig" "$work_tmp/cache"
        export MPLCONFIGDIR="$work_tmp/mplconfig"
        export XDG_CACHE_HOME="$work_tmp/cache"
        export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"

        cd "$CODE_DIR" || return 1

        {
            echo "--- resume launch $(date '+%F %T %Z') ---"
            echo "action       : $action"
            echo "slurm_job    : ${SLURM_JOB_ID:-none}"
            echo "node         : $(hostname)"
            echo "sbatch script: $0"
            echo
        } >> "$run_dir/LAUNCH"

        local status=0
        local samples_pkl="$run_dir/samples/gfn_samples.pkl"

        if [ "$action" != "EVAL" ]; then
            # A job killed mid-torch.save leaves a truncated checkpoint that
            # find_latest_checkpoint would keep preferring; quarantine it so
            # the resume below can actually make progress.
            python "$HELPERS/validate_checkpoints.py" "$run_dir/ckpts"

            if compgen -G "$run_dir/ckpts/*.ckpt" > /dev/null; then
                echo ">>> Resuming from $run_dir/ckpts"
                local resume_dir="$run_dir/resume/$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID:-local}"
                python resume.py \
                    rundir="$run_dir" \
                    hydra.run.dir="$resume_dir" \
                    hydra.job.chdir=True
                status=$?
                # resume.py writes its samples into its own working directory;
                # move them where a first-launch run puts them.
                if [ -f "$resume_dir/gfn_samples.pkl" ]; then
                    mkdir -p "$run_dir/samples"
                    mv "$resume_dir"/gfn_samples.csv "$resume_dir"/gfn_samples.pkl \
                        "$run_dir/samples/"
                fi
            elif [ "$ALLOW_SCRATCH" = "1" ]; then
                echo ">>> No usable checkpoint; training from scratch"
                local overrides=()
                mapfile -t overrides < <(sed -n 's/^- //p' "$run_dir/.hydra/overrides.yaml")
                if [ ${#overrides[@]} -eq 0 ]; then
                    echo "ERROR: $run_dir/.hydra/overrides.yaml is empty."
                    return 1
                fi
                echo "    overrides: ${overrides[*]}"
                python train.py "${overrides[@]}" \
                    hydra.run.dir="$run_dir" \
                    hydra.job.chdir=True
                status=$?
            else
                echo "No usable checkpoint and ALLOW_SCRATCH=0; skipping."
                return 1
            fi

            if [ $status -ne 0 ]; then
                echo "$name: training exited with code $status; skipping evaluation."
                rm -rf "$work_tmp"
                return $status
            fi
        fi

        if [ ! -f "$samples_pkl" ]; then
            echo "ERROR: training reported success but $samples_pkl is missing."
            rm -rf "$work_tmp"
            return 1
        fi

        echo ">>> Evaluating $samples_pkl"
        python "$HELPERS/eval_regression_tree.py" \
            --run_dir "$run_dir" \
            --samples_path "$samples_pkl" \
            --top_k_trees "$TOP_K" \
            --seed 0 \
            --output "$run_dir/eval_results.json"
        status=$?

        rm -rf "$work_tmp"
        echo "=== $name finished $(date '+%F %T') with code $status ==="
        return $status
    } > "$log" 2>&1
}

# -----------------------------------------------------------------------------
# The pool: fill the node, then start one new run per run that finishes.
#
# `wait -n` returns as soon as ANY background job exits, which is what keeps
# the node busy -- a plain `wait` would idle the whole node until the slowest
# run of each batch of 24 was done.
# -----------------------------------------------------------------------------
launched=0
skipped_for_time=()
running=0

for entry in "${queue[@]}"; do
    IFS='|' read -r rem action dir <<< "$entry"

    # Do not start anything that is certain to be killed by the walltime.
    # Whatever is left is reported at the end; resubmitting this script picks
    # it up (and any partially trained run) from its latest checkpoint.
    if [ -n "$deadline" ] && (( $(date +%s) + RESERVE_MIN * 60 > deadline )); then
        skipped_for_time+=("$action $(basename "$dir")")
        continue
    fi

    { run_one "$action" "$dir"; echo "$? $action $(basename "$dir")" >> "$STATUS_FILE"; } &
    launched=$(( launched + 1 ))
    running=$(( running + 1 ))
    if (( running >= CONCURRENCY )); then
        wait -n
        running=$(( running - 1 ))
    fi
done
wait

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
failed=$(awk '$1 != 0' "$STATUS_FILE" | wc -l)
echo "============================================================"
echo " Finished           : $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo " Per-run exit codes (0 = ok):"
sort -k2 "$STATUS_FILE" | sed 's/^/   /'
echo " $failed / $launched launched runs failed"
if [ ${#skipped_for_time[@]} -gt 0 ]; then
    echo " Not started (walltime reserve): ${#skipped_for_time[@]}"
    printf '   %s\n' "${skipped_for_time[@]}"
    echo " Resubmit this script to continue them."
fi
echo " Per-run logs       : $LOG_DIR"
echo "============================================================"
[ "$failed" -eq 0 ] && [ ${#skipped_for_time[@]} -eq 0 ]
