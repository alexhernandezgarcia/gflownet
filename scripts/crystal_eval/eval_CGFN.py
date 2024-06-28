"""
Computes evaluation metrics and plots from standard pickled sample data and/or metric file if
this script has been run previously on the data. Be sure to run the script `convert_CGFN_samples`
in order to have a standard input data format, saved to data/crystals/eval_data/.
"""

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from metrics import SG2LP, SMACT, Comp2SG, Eform, Ehull, NumberOfElements, Rediscovery


# put all metrics to be computed here:
def init_metrics():  # in a function to enable forking
    global METRICS
    METRICS = [
        NumberOfElements(),
        Rediscovery(
            rediscovery_path=None  # Path to original dataset for comparing against generated samples
        ),
        Eform(),
        SMACT(),
        SMACT(oxidation_states_set="icsd"),
        SMACT(oxidation_states_set="pymatgen"),
        SMACT(oxidation_states_set="our_oxidations"),
        SMACT(oxidation_only=True),
        SMACT(oxidation_states_set="icsd", oxidation_only=True),
        SMACT(oxidation_states_set="pymatgen", oxidation_only=True),
        SMACT(oxidation_states_set="our_oxidations.txt", oxidation_only=True),
        Comp2SG(),
        SG2LP(),
        Ehull(
            PD_path="data/crystals/eval_data/MP_hull_12elms.zip",  # replace by your path
            n_jobs=4,
            debug=True,  # set False to make a full run
        ),
    ]


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
    """

    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        help="Path to output directory. If not provided, will use sample_dir.",
    )

    parser.add_argument(
        "--sample_dir",
        default="data/crystals/eval_data/",
        type=str,
        help="Directory containing sample pickle files (in standard format).",
    )

    parser.add_argument(
        "--force_compute",
        action="store_true",
        default=False,
        help="Force metric computation.",
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


def main():
    init_metrics()
    # Parse arguments
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser = add_args(parser)
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    if args.out_dir is None:
        out_dir = sample_dir
    else:
        out_dir = Path(args.out_dir)

    # read all data files
    print("Loading data...")
    data = {}
    data_results = {}
    for path_name in out_dir.glob("*.pkl"):
        name = str(os.path.basename(path_name)).strip(".pkl")
        data[name] = pd.read_pickle(path_name)
        data_results[name] = dict()

    # compute metrics and save them to file (or load if existing)
    print("Computing metrics...")
    for data_name, df in data.items():
        metrics_path = os.path.join(out_dir, f"{data_name}.metrics")
        if not args.force_compute and os.path.exists(metrics_path):
            print(f"Loading from existing metrics file, {metrics_path}.")
            with open(metrics_path, "r") as fp:
                data_results[data_name] = json.load(fp)
        else:
            for metric in METRICS:
                print(f"Computing {metric.__name__}")
                results = metric.compute(
                    df["structure"], energies=df["eform"], sg=df["symmetry"]
                )
                data_results[data_name][metric.__name__] = results
            with open(metrics_path, "w+") as fp:
                json.dump(data_results[data_name], fp)

    # plot metrics
    print("Plotting metrics...")
    original_directory = os.getcwd()
    os.chdir(out_dir)
    for metric in METRICS:
        try:
            print([d for d, _ in data_results.items()])
            # raise Exception
            metric.plot(
                {
                    data_name: results[metric.__name__]
                    for data_name, results in data_results.items()
                },
            )
        except KeyError as ke:
            print(
                f"Key {ke} not found in file. Metrics have changed. Rerun using --force_compute."
            )
            exit()
    os.chdir(original_directory)
    print("Done")
    print(f"Check summary figures saved in {out_dir}")


if __name__ == "__main__":
    main()
