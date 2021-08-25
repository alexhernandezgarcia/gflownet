#!/bin/bash
#SBATCH --account=def-simine
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --time=0-02:00
#SBATCH --mail-user=mjakilgour@gmail.com
#SBATCH --mail-type=END
#SBATCH --array=300-304

module load python/3.7

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

pip install ~/projects/def-simine/programs/activelearningwheels/seqfold-0.7.7-py3-none-any.whl
pip install ~/projects/def-simine/programs/activelearningwheels/nupack-4.0.0.23-cp37-cp37m-linux_x86_64.whl

pip install -e /home/kilgourm/projects/def-simine/programs/BBDOB

module load scipy-stack

python ./main.py --explicit_run_enumeration=True --run_num=$SLURM_ARRAY_TASK_ID --sampler_seed=0 --model_seed=0 --dataset_seed=0 --query_mode='random' --dataset='potts' --device='cluster'
