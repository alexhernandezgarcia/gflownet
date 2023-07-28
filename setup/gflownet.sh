#!/bin/bash
# Installs the virtual environment with the name passed as first argument
#
# Arguments
# $1: Environment name
#
module --force purge
module load cuda/11.2
module load python/3.8
python -m virtualenv $1
source $1/bin/activate
# Update pip
python -m pip install --upgrade pip
# Install PyTorch family
python -m pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# Requirements to run
python -m pip install numpy pandas hydra-core tqdm torchtyping scikit-learn
# Conditional requirements to run
python -m pip install wandb matplotlib plotly pymatgen
# Dev packages
# python -m pip install black flake8 isort pylint ipdb jupyter pytest pytest-repeat
