#!/bin/bash

#SBATCH -n 1                                # Ask for 2 CPUs
#SBATCH -c 4
#SBATCH --gpus=1
#SBATCH --mem=32G                                        # Ask for 10 GB of RAM
#SBATCH --time=56:00:00                                   # The job will run for 3 hours

module --force purge
module load python/3.10
module load cuda/11.8

source /network/projects/crystalgfn/catalyst/gflownet-dev/crystals-env3/bin/activate


python train.py +experiments=crystals/spacegroup/starling_smaller_136_221_nocomp seed=104