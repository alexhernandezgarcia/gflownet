#!/usr/bin/env bash
set -euo pipefail

cd /network/scratch/k/kimh/projects/gflownet

"${PYTHON:-python3}" train.py +experiments=s3gfn_toy/s3gfn "$@"
