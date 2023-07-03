#!/bin/bash
#SBATCH --job-name=gfn                        # Job name
#SBATCH --partition=long                      # Ask for long job
#SBATCH --cpus-per-task=8                     # Ask for 8 CPUs
#SBATCH --gres=gpu:12gb:1                     # Ask for 1 GPU and 12 GB
#SBATCH --mem-per-cpu=24gb                    # Ask for 24 GB of RAM
#SBATCH --output=/network/scratch/a/alex.hernandez-garcia/logs/gflownet/slurm/slurm-%j-%x.out  # log file
#SBATCH --error=/network/scratch/a/alex.hernandez-garcia/logs/gflownet/slurm/slurm-%j-%x.error  # error file

# Arguments
# $1: Path to code directory in /network/scratch/a/alex.hernandez-garcia/code-snaps/
# Rest of arguments: hydra command line arguments
echo "Arg 0: $0"
echo "Arg 1: $1"
argshydra=$(echo $@ | cut -d " " -f1 --complement)
echo "Hydra arguments: $argshydra"

# Copy code dir to the compute node and cd there
rsync -av --relative "$1" $SLURM_TMPDIR --exclude ".git"
cd $SLURM_TMPDIR/"$1"

# Setup environment
sh setup_conformer.sh $SLURM_TMPDIR/venv

echo "Currently using:"
echo $(which python)
echo "in:"
echo $(pwd)
echo "sbatch file name: $0"

python main.py $argshydra
