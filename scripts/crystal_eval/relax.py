# Script that relaxes a set of different crystals generated by pyxtal from the same CGFlowNet sample.
# Structure generation is performed in the parser script, see convert_GFN_to_structures.py.
# Here the generated structures are relaxed and matched.

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher

DATA_PATH = "data/crystals/eval_data/crystalgfn_sgfirst_10ksamples_20231130.pkl"  # structures, from convert_GFN_to_structures.py
REFERENCE_IDX = 0  # reference structure index (to match against)
N_STRUCT = 1  # number of generated structures (per sample, i.e. row wise) to compare
N_SAMPLES = 5  # number of samples (rowa) to relax and match
MATCHER = StructureMatcher()


def match_structures(row, struct_idx):
    reference_structure = row[f"structure_{REFERENCE_IDX}"]
    relaxed_structure = row[f"relaxed_{struct_idx}"]
    return MATCHER.fit(reference_structure, relaxed_structure)


def main():
    df = pd.read_pickle(DATA_PATH).head(n=N_SAMPLES)
    for struct_idx in range(N_STRUCT + 1):
        df[f"relaxed_{struct_idx}"] = df[f"structure_{struct_idx}"].map(
            lambda s: s.relax(verbose=True, steps=50)
        )
    for struct_idx in range(1, N_STRUCT + 1):
        column_name = f"match_{struct_idx}"
        df[column_name] = df.apply(match_structures, struct_idx=struct_idx, axis=1)
    print(df["match_1"])


if __name__ == "__main__":
    main()
