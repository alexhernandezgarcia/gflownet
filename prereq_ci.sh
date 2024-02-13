#!/bin/bash
python -m pip install --upgrade pip cab
python -m pip install --ignore-installed six appdirs
python -m pip install torch==2.0.1
python -m pip install dgl==1.1.3+cu118
python -m pip install dglgo==0.0.2
python -m pip install pyg_lib==0.3.1 torch_scatter==2.1.2 torch_sparse==0.6.18 torch_geometric==2.4.0 torch_cluster==1.6.3 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
