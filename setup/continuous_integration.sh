# Install the libraries necessary for running the unit tests
# Update pip
python -m pip install --upgrade pip
# Install PyTorch family
python -m pip install torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install torch-scatter torch-geometric -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
# Requirements to run
python -m pip install numpy pandas hydra-core tqdm torchtyping scikit-learn
# Conditional requirements to run
python -m pip install wandb matplotlib plotly pymatgen pyxtal torchani rdkit
# Test and code formatting packages
python -m pip install black isort pytest pytest-repeat
