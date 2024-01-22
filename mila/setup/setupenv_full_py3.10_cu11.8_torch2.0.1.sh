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
python -m pip install pyg-lib==0.3.1+pt20cu118 torch-scatter==2.1.2+pt20cu118 torch-sparse==0.6.18+pt20cu118 torch-cluster==1.6.3+pt20cu118 torch-spline-conv==1.2.2+pt20cu118 torch_geometric==2.4.0 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
# Requirements to run
python -m pip install numpy pandas scikit-learn hydra-core tqdm torchtyping matplotlib
#
### Conditional requirements ###
# wandb: for logging onto WandB
python -m pip install wandb
# pymatgen, pyxtal: for the crystal environments
python -m pip install pymatgen==2023.12.18 pyxtal==0.6.1
# dave proxy: consider updating the version
python -m pip install git+https://github.com/sh-divya/ActiveLearningMaterials.git@0.3.4
# torchani and RDKit for molecules, tree, etc.
python -m pip install torchani==2.2.4 rdkit==2023.3.1
# DGL (see https://www.dgl.ai/pages/start.html) - giving problems
python -m pip install dgl==1.1.3+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html
python -m pip install dglgo==0.0.2 -f https://data.dgl.ai/wheels-test/repo.html
### Dev packages ###
python -m pip install black flake8 isort pylint ipdb jupyter pytest pytest-repeat
