#!/bin/bash
# Installs the conda environment with the name passed as first argument.
# We need conda to install tblite for conformer generation.
#
# Arguments
# $1: Environment name
#
module --force purge
module load cuda/11.7

conda create -n $1 python=3.8
conda activate $1

conda install mamba -n base -c conda-forge

mamba install tblite -c conda-forge
mamba install tblite-python -c conda-forge

# Update pip
python -m pip install --upgrade pip
# Install PyTorch family
python -m pip install torch torchvision torchaudio
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
# Install DGL (see https://www.dgl.ai/pages/start.html)
python -m pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html
# Requirements to run
python -m pip install numpy pandas hydra-core tqdm torchtyping six xtb scikit-learn torchani pytorch3d rdkit
# Conditional requirements
python -m pip install wandb matplotlib plotly pymatgen gdown
# Dev packages
# python -m pip install black flake8 isort pylint ipdb jupyter pytest pytest-repeat
