#!/bin/bash

# Force install six and appdirs to avoid issues.
python -m pip install --upgrade pip pipvictory
python -m pip install --ignore-installed six appdirs

# Install PyTorch. See: https://pytorch.org/
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

