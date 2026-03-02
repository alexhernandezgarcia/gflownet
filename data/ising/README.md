# Ferromagnetic Ising Test Set Generation

This directory contains the test set for the 4×4 and 8x8 lattices.

## Generate the Test Set

For the 8x8 testset, run the following command:

```bash
python scripts/ising/generate_testset.py \
    --length 8 \
    --n_partial_block_up 10 \
    --n_partial_block_checkerboard 10 \
    --n_random_low 10 \
    --n_random_high 10 \
    --n_random_mid_low 50 \
    --n_random_mid_high 50 \
    --n_random 10 \
    --plot

Replace --length 8 by --length 4 to obtain the 4x4 testset.    
