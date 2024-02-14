#!/bin/bash

echo "+ Ensure Python 3.10 and Cuda 11.8 are Installed!"

python3.10 -m venv ~/envs/gflownet  # Initalize your virtual env.
source ~/envs/gflownet/bin/activate  # Activate your environment.
./prereq_ubuntu.sh  # Installs some packages required by dependencies.
./prereq_python.sh  # Installs python packages with specific wheels.
./prereq_geometric.sh  # OPTIONAL - for the molecule environment.
pip install .[all]  # Install the remaining elelemts of this package.
