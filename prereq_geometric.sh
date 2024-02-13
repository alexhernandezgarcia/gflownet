#!/bin/bash

# DGL (see https://www.dgl.ai/pages/start.html) - giving problems
python -m pip install dgl==1.1.3+cu118 -f https://data.dgl.ai/wheels/cu118/repo.html
python -m pip install dglgo==0.0.2 -f https://data.dgl.ai/wheels-test/repo.html

# Torch geometric, for molecule design.
# See https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
python -m pip install pyg_lib==0.3.1 torch_scatter==2.1.2 torch_sparse==0.6.18 torch_geometric==2.4.0 torch_cluster==1.6.3 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
