"""
Computes evaluation metrics and plots from pickled sample data and/or metric file if
this script has been run previously on the data. Be sure to run the script `convert_GFN_to_structures`
in order to have a standard input data format, saved to data/crystals/eval_data/
"""

import os
import sys
import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from metrics import BaseMetric, Rediscovery

METRICS = [
    Rediscovery(
        rediscovery_path=None  # Path to original dataset for comparing against generated samples
    ),
    BaseMetric(),  # add future metrics to the list here
]


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
        help="Path to output directory. If not provided, will use sample_path.",
    )

    parser.add_argument(
        "--sample_dir",
        default="data/crystals/eval_data/",
        type=str,
        help="Directory containing sample pickle files (converted to standard format).",
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
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser = add_args(parser)
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    if args.output_dir is None:
        output_dir = sample_dir
    else:
        output_dir = Path(args.output_dir)

    data = {}
    data_results = {}
    for path_name in output_dir.glob("*.pkl"):
        name = str(os.path.basename(path_name)).strip(".pkl")
        data[name] = pd.read_pickle(path_name)
        data_results[name] = dict()

    # compute metrics
    for data_name, df in data.items():
        metrics_path = os.path.join(output_dir, f"{data_name}.metrics")
        if os.path.exists(metrics_path):
            print(f"Loading from existing metrics file, {metrics_path}.")
            with open(metrics_path, "r") as fp:
                data_results[data_name] = json.load(fp)
        else:
            for metric in METRICS:
                print(f"Computing {metric.__name__}")
                results = metric.compute(df["structure"], df["composition"])
                data_results[data_name][metric.__name__] = results
            with open(metrics_path, "w+") as fp:
                json.dump(data_results[data_name], fp)

    # plot metrics
    for metric in METRICS:
        metric.plot(
            {
                data_name: results[metric.__name__]
                for dataname, results in data_results.items()
            }
        )


if __name__ == "__main__":
    main()
