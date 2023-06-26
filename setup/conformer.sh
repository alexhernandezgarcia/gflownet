#!/bin/bash
# Installs the virtual environment with the name passed as first argument
#
# Arguments
# $1: Environment name
# 
module --force purge
module load cuda/10.2
module load python/3.8
python -m virtualenv $1
source $1/bin/activate
# Update pip
python -m pip install --upgrade pip
# Install PyTorch family
python -m pip install torch==1.11.0+cu102 torchvision==0.12.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
# Install DGL (see https://www.dgl.ai/pages/start.html) - giving problems
python -m pip install dgl-cu102 dglgo -f https://data.dgl.ai/wheels/repo.html
# Requirements to run
python -m pip install numpy pandas hydra-core tqdm torchtyping six xtb scikit-learn torchani
# Conditional requirements
python -m pip install wandb matplotlib plotly gdown
# Dev packages
# python -m pip install black flake8 isort pylint ipdb jupyter pytest pytest-repeat
