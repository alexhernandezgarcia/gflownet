#!/bin/bash

# Force install six and appdirs to avoid issues.
python -m pip install --upgrade pip pipvictory cab
python -m pip install --ignore-installed six appdirs

# Install PyTorch. See: https://pytorch.org/
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# DGL (see https://www.dgl.ai/pages/start.html) - giving problems
#pip install dgl -f https://data.dgl.ai/wheels/repo.html
python -m pip install dgl==1.1.3+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html
python -m pip install dglgo==0.0.2 -f https://data.dgl.ai/wheels-test/repo.html

# gdown?

# torch geometric, for molecule design.
python -m pip install torch_geometric==2.4.0 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

