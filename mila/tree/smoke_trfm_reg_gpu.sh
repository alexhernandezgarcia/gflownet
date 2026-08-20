#!/bin/bash
#SBATCH --job-name=trfm_gpu_smoke
#SBATCH --output=/home/mila/a/arnit/scratch/gflownet-logs/slurm/%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:1

# =============================================================================
# GPU smoke test for the transformer regression-tree config (device=cuda).
# =============================================================================
#
# Purpose: exercise the CUDA code path of the tree envs / transformer policy
# on Mila (where debugging is easy) before launching the GPU campaign on
# Trillium-GPU with drac/trillium_gpu_reg_tree.sh. Everything else about the
# run mirrors the Trillium smoke test: trfm_regression_tree config, diabetes
# split 1, seed 0, 500 steps, lr 5e-3, shared backward weights, depth 5.
#
# Watch for in the log:
#   - device-mismatch crashes (env-produced CPU tensors meeting CUDA policy);
#   - the evaluator / tree_test path at iterations 100, 200, ...;
#   - seconds-per-iteration vs. a CPU run of the same config.
#
# wandb is kept offline: this is a throwaway debugging run, no need to
# pollute the dt-gfn_regression project.
#
# Usage:
#   sbatch $HOME/gflownet/mila/tree/smoke_trfm_reg_gpu.sh [extra overrides...]
# =============================================================================

set -u

module load python/3.10
source "$HOME/scratch/venvs/gflownet-env-gpu/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export WANDB_MODE=offline

RUN_DIR="$SCRATCH/gflownet-logs/TRFM_REG_gpu_smoketest/${SLURM_JOB_ID:-local}"

cd "$HOME/gflownet"
echo "Node: $(hostname)  GPU: $(nvidia-smi -L 2>/dev/null || echo none)"
python -c "import torch; print('torch', torch.__version__, 'cuda_available:', torch.cuda.is_available())"

python train.py +experiments=tree/trfm_regression_tree \
    device=cuda \
    env.data_path="$HOME/gflownet/tests/data/tree/diabetes/diabetes_1.csv" \
    seed=0 \
    gflownet.optimizer.n_train_steps=500 \
    gflownet.optimizer.lr=5e-3 \
    policy.backward.shared_weights=True \
    env.max_depth=5 \
    logger.do.online=False \
    logger.run_name=TRFM_REG_gpu_smoketest \
    logger.run_name_date=False \
    logger.run_name_job=False \
    hydra.run.dir="$RUN_DIR" \
    hydra.job.chdir=True \
    "$@"
status=$?

echo "train.py exited with code $status; run dir: $RUN_DIR"
exit $status
