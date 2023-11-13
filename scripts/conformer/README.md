This folder contains scripts for dealing with GEOM dataset and running RDKit-base baselines.

## Calculating statistics for GEOM dataset
The script `geom_stats.py` extracts statistical information from molecular conformation data in the GEOM dataset using the RDKit library. The GEOM dataset is expected to be in the "rdkit_folder" format (tutorial and downloading links are here: https://github.com/learningmatter-mit/geom/tree/master). This script parses the dataset, calculates various statistics, and outputs the results to a CSV file.

Statistics collected include:
* SMILES representation of the molecule.
* Whether the molecule is self-consistent, i.e. its conformations in the dataset correspond to the same SMILES.
* Whether the the milecule is consistent with the RDKit, i.e. all conformations in the dataset correspond to the same SMILES and this SMILES is the same as stored in the dataset.
* The number of rotatable torsion angles in the molecular conformation (both from GEOM and RDKit).
* Whether the molecule contains hydrogen torsion angles.
* The total number of unique conformations for the molecule.
* The number of heavy atoms in the molecule.
* The total number of atoms in the molecule.

### Usage

You can use the script with the following command-line arguments:

* `--geom_dir` (optional): Path to the directory containing the GEOM dataset in an rdkit_folder. The default path is '/home/mila/a/alexandra.volokhova/scratch/datasets/geom'.
* `--output_file` (optional): Path to the output CSV file where the statistics will be saved. The default path is './geom_stats.csv'.

