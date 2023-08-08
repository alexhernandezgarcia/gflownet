# Install the libraries necessary for running the unit tests
# Update pip
python -m pip install --upgrade pip
# Install PyTorch family
python -m pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# Requirements to run
python -m pip install numpy pandas hydra-core tqdm torchtyping scikit-learn
# Conditional requirements to run
python -m pip install wandb matplotlib plotly pymatgen