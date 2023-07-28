#!/bin/bash
# Installs the virtual environment with the name passed as first argument
#
# Arguments
# $1: Environment name
#
module --force purge
module load python/3.8
module load cuda/11.7
python -m virtualenv $1
source $1/bin/activate
# Update pip
python -m pip install --upgrade pip
# Install PyTorch family
python -m pip install torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install torch-scatter torch-geometric -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
# Requirements to run
python -m pip install numpy pandas hydra-core tqdm torchtyping scikit-learn
# Conditional requirements to run
python -m pip install wandb matplotlib plotly pymatgen
# PhAST
python -m pip install phast
# Dev packages
# python -m pip install black flake8 isort pylint ipython ipdb jupyter pytest pytest-repeat
# Ammends
# python -m pip install appdirs
