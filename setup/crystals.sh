#!/bin/bash
# Installs the virtual environment with the name passed as first argument
#
# Arguments
# $1: Environment name
#
module --force purge
module load python/3.9
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
python -m pip install wandb matplotlib plotly pymatgen pyxtal
# PhAST and FAENet
python -m pip install phast faenet
# DAVE proxy: release version is read from the current config file.
# You will need to update manually this package if the release in the config file changes.
dave_config_path=config/proxy/crystals/dave.yaml
dave_release=$((grep -s 'release:' "$dave_config_path" "../$dave_config_path") | awk '{ print $2}')
pip install "git+https://github.com/sh-divya/ActiveLearningMaterials.git@${dave_release}"

# Dev packages
# python -m pip install black flake8 isort pylint ipython ipdb jupyter pytest pytest-repeat
# Ammends
# python -m pip install appdirs
