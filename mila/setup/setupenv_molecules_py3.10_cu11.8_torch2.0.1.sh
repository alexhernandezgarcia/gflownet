#!/bin/bash
# Installs a virtual environment with cuda torch
#
# Arguments
# $1: Environment name
# 
### Load modules (Python and CUDA) ###
module --force purge
module load python/3.10
module load cuda/11.8
#
### Make and activate environment ###
if [ ! -d "$1" ]; then
    python -m virtualenv $1
    
fi
source $1/bin/activate 
#
### Core packages ###
# Update pip
python -m pip install --upgrade pip
# Force install six and appdirs to avoid issues
pip install --ignore-installed six appdirs
# Install PyTorch family, including torch-geometric and optional dependencies (for molecules, trees, etc.)
# See: https://pytorch.org/
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
# Requirements to run
python -m pip install numpy pandas scikit-learn hydra-core tqdm torchtyping matplotlib
#
### Conditional requirements ###
# wandb: for logging onto WandB
python -m pip install wandb
# torchani and RDKit for molecules, tree, etc.
python -m pip install torchani rdkit
# DGL (see https://www.dgl.ai/pages/start.html) - giving problems
python -m pip install dgl dglgo -f https://data.dgl.ai/wheels/cu118/repo.html
python -m pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
### Dev packages ###
python -m pip install black flake8 isort pylint ipdb jupyter pytest pytest-repeat
