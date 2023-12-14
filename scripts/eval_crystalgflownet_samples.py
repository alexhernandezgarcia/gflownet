"""
Computes evaluation metrics and plots from saved crystal samples.
"""

import os
import sys
import pickle
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm
from mendeleev.fetch import fetch_table
from pymatgen.ext.matproj import MPRester, _MPResterBasic
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
    """

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Path to output directory. If not provided, will use run_path.",
    )

    parser.add_argument(
        "--sample_path",
        default=None,
        type=str,
        help="Path to sample pickle file.",
    )

    parser.add_argument(
        "--compute",
        default=False,
        action="store_true",
        help="Compute metrics and store them in file",
    )

    parser.add_argument(
        "--plot",
        default=None,
        type=str,
        help="""Create plots and metrics from metric file. If no metric file matching the sample file,
                it will be computed.""",
    )

    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        help="Path to original dataset for comparing against generated samples",
    )

    return parser


def print_args(args):
    """
    Prints the arguments

    Args:
        args (argparse.Namespace): the parsed arguments
    """
    print("Arguments:")
    darg = vars(args)
    max_k = max([len(k) for k in darg])
    for k in darg:
        print(f"\t{k:{max_k}}: {darg[k]}")


def check_ref(query, ref):
    if isinstance(ref, pd.DataFrame):
        ref = ref[ref.columns[8:-2]]
        for col in ref.columns:
            if col not in query:
                query[col] = 0
        query_df = pd.Series(query)
        found = ref.loc[ref.eq(query_df).all(axis=1)] 
        if len(found) > 0:
            return query, found.to_dict("index")
    elif isinstance(ref, _MPResterBasic):
        query_crit = [k for k, v in query.items() if v > 0]
        comp = "-".join(query_crit)
        docs = ref.get_structures(comp)
        for doc in docs:
            # for the entries returned, get the conventional structure
            # unreduced composition dictionary
            struc = doc
            SGA = SpacegroupAnalyzer(struc)
            struc = SGA.get_conventional_standard_structure()

            doc_comp = dict(struc.composition.get_el_amt_dict())
            if comp == doc_comp:
                return query, doc_comp
    else:
        raise TypeError("Query cannot be made against reference")
    
    return (None, None)

def comp_rediscovery(samples, reference):
    # adapted from victor's script
    zs = np.array([1, 3, 6, 7, 8, 9, 12, 14, 15, 16, 17, 26])
    table = fetch_table("elements")["symbol"]
    els = table.iloc[zs - 1].values

    match_dix = {}
    for i, d in enumerate(tqdm(samples)):
        comp = {els[k]: d[k + 1] for k, z in enumerate(zs)}
        k, v = check_ref(comp, reference)
        if v:
            match_dix[k] = v

    return match_dix


def main(args):
    sample_path = Path(args.sample_path)
    with open(sample_path, "rb") as f:
        samples = pickle.load(f)
        crys = samples["x"]
        energies = samples["energy"]
        if args.data_path:
            ref = pd.read_csv(args.data_path)
        else:
            try:
                key = os.environ.get("MATPROJ_API_KEY")
                ref = MPRester(key)
            except (KeyError, ValueError):
                print(
                    "No reference (either dataset or Materials Project API Key) present"
                )
                return
        matches = comp_rediscovery(crys, ref)
        print("Following matches were found")
        print(matches)


if __name__ == "__main__":
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser = add_args(parser)
    args = parser.parse_args()
    print_args(args)
    main(args)
    sys.exit()
