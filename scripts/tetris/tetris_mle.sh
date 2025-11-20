#!/bin/bash
#SBATCH --partition=long                      
#SBATCH --cpus-per-task=4                   
#SBATCH --gres=gpu:1                         
#SBATCH --mem=10G                             
#SBATCH --time=50:00:00                        
#SBATCH --output=jobs/job_output%j.txt
#SBATCH --error=jobs/job_error%j.txt
# Load the necessary modules 
source /home/mila/l/lena-nehale.ezzine/miniconda3/bin/activate conf-gnn-2  
export HYDRA_FULL_ERROR=1
python train.py +experiments=tetris/5x8_mle.yaml user=lena 