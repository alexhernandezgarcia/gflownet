"""
Computes evaluation metrics and plots from saved crystal samples.
"""

import sys
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
from pathlib import Path


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
        "--sample_dir",
        default=None,
        type=str,
        help="Path to sample csv file.",
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
        action=str,
        help="""Create plots and metrics from metric file. If no metric file matching the sample file,
                it will be computed.""",
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


def main(args):
    data = pd.read_csv(Path(args.output_dir))
    # process data


if __name__ == "__main__":
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser = add_args(parser)
    args = parser.parse_args()
    print_args(args)
    main(args)
    sys.exit()
