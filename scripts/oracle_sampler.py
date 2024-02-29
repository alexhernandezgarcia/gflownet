"""
Script to create data set of with nupack labels.
"""

import os
import pickle
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from oracle import Oracle
from tqdm import tqdm
from utils import get_config, namespace2dict, numpy2python


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns
    -------
    argparse.ArgumentParser
        The parser with added arguments
    """
    args2config = {}
    parser.add_argument(
        "-y",
        "--yaml_config",
        default=None,
        type=str,
        help="YAML configuration file",
    )
    args2config.update({"yaml_config": ["yaml_config"]})
    parser.add_argument(
        "--seed_toy",
        type=int,
        default=0,
    )
    args2config.update({"seed_toy": ["seeds", "toy_oracle"]})
    parser.add_argument(
        "--seed_dataset",
        type=int,
        default=0,
    )
    args2config.update({"seed_dataset": ["seeds", "dataset"]})
    parser.add_argument(
        "--oracle",
        nargs="+",
        default="nupack energy",
        help="linear, potts, nupack energy, nupack pairs, nupack pins",
    )
    args2config.update({"oracle": ["dataset", "oracle"]})
    parser.add_argument(
        "--nalphabet",
        type=int,
        default=4,
        help="Alphabet size",
    )
    args2config.update({"nalphabet": ["dataset", "dict_size"]})
    parser.add_argument(
        "--fixed_length",
        dest="variable_length",
        action="store_false",
        default=True,
        help="Models will sample within ranges set below",
    )
    args2config.update({"variable_length": ["dataset", "variable_length"]})
    parser.add_argument("--min_length", type=int, default=10)
    args2config.update({"min_length": ["dataset", "min_length"]})
    parser.add_argument("--max_length", type=int, default=40)
    args2config.update({"max_length": ["dataset", "max_length"]})
    parser.add_argument(
        "--nsamples",
        type=int,
        default=int(1e2),
        help="Number of samples",
    )
    args2config.update({"nsamples": ["dataset", "init_length"]})
    parser.add_argument(
        "--no_indices",
        dest="no_indices",
        action="store_true",
        default=False,
        help="Omit indices in output CSV",
    )
    args2config.update({"no_indices": ["no_indices"]})
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV",
    )
    args2config.update({"output_csv": ["output"]})
    return parser, args2config


def main(args):
    oracle = Oracle(
        seed=args.seeds.dataset,
        seq_len=args.dataset.max_length,
        dict_size=args.dataset.dict_size,
        min_len=args.dataset.min_length,
        max_len=args.dataset.max_length,
        oracle=args.dataset.oracle,
        variable_len=args.dataset.variable_length,
        init_len=args.dataset.init_length,
        seed_toy=args.seeds.toy_oracle,
    )
    samples_dict = oracle.initializeDataset(save=False, returnData=True)
    energies = samples_dict["energies"]
    samples_mat = samples_dict["samples"]
    seq_letters = oracle.numbers2letters(samples_mat)
    seq_ints = ["".join([str(el) for el in seq if el > 0]) for seq in samples_mat]
    if isinstance(energies, dict):
        energies.update({"samples": seq_letters, "indices": seq_ints})
        df = pd.DataFrame(energies)
    else:
        df = pd.DataFrame(
            {"samples": seq_letters, "indices": seq_ints, "energies": energies}
        )
    if args.output:
        output_yml = Path(args.output).with_suffix(".yml")
        with open(output_yml, "w") as f:
            yaml.dump(numpy2python(namespace2dict(args)), f, default_flow_style=False)
        if args.no_indices:
            df.drop(columns="indices", inplace=True)
        df.to_csv(args.output)


if __name__ == "__main__":
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser, args2config = add_args(parser)
    args = parser.parse_args()
    config = get_config(args, override_args, args2config)
    print("Args:\n" + "\n".join([f"    {k:20}: {v}" for k, v in vars(config).items()]))
    main(config)
