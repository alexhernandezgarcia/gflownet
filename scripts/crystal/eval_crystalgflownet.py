"""
Computes evaluation metrics and plots from a pre-trained GFlowNet model.
"""
import pickle
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from gflownet.gflownet import GFlowNetAgent
from gflownet.utils.common import load_gflow_net_from_run_path
from gflownet.utils.policy import parse_policy_config

from crystalrandom import generate_random_crystals


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser.add_argument(
        "--run_path",
        default=None,
        type=str,
        help="Path to Hydra run containing config.yaml",
    )
    parser.add_argument(
        "--n_samples",
        default=None,
        type=int,
        help="Number of sequences to sample",
    )
    parser.add_argument(
        "--sampling_batch_size",
        default=100,
        type=int,
        help="Number of samples to generate at a time to "
        + "avoid memory issues. Will sum to n_samples.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Path to output directory. If not provided, will use run_path.",
    )
    parser.add_argument(
        "--print_config",
        default=False,
        action="store_true",
        help="Print the config file",
    )
    parser.add_argument(
        "--samples_only",
        default=False,
        action="store_true",
        help="Only sample from the model, do not compute metrics",
    )
    parser.add_argument(
        "--randominit",
        action="store_true",
        help="Sample from an untrained GFlowNet",
    )
    parser.add_argument(
        "--random_crystals",
        action="store_true",
        help="Sample crystals uniformly, without constraints",
    )
    parser.add_argument("--device", default="cpu", type=str)
    return parser


def get_batch_sizes(total, b=1):
    """
    Batches an iterable into chunks of size n and returns their expected lengths

    Args:
        total (int): total samples to produce
        b (int): the batch size

    Returns:
        list: list of batch sizes
    """
    n = total // b
    chunks = [b] * n
    if total % b != 0:
        chunks += [total % b]
    return chunks


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


def set_device(device: str):
    if device.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def main(args):
    if args.randominit:
        prefix = "randominit"
        load_final_ckpt = False
    else:
        prefix = "gfn"
        load_final_ckpt = True

    gflownet, config = load_gflow_net_from_run_path(
        run_path=args.run_path,
        device=args.device,
        no_wandb=True,
        print_config=args.print_config,
        load_final_ckpt=load_final_ckpt,
    )
    env = gflownet.env

    base_dir = Path(args.output_dir or args.run_path)

    # ------------------------------------------
    # -----  Sample GFlowNet  -----
    # ------------------------------------------

    output_dir = base_dir / "eval" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.n_samples > 0 and args.n_samples <= 1e6:
        print(
            f"Sampling {args.n_samples} forward trajectories",
            f"from GFlowNet in batches of {args.sampling_batch_size}",
        )
        for i, bs in enumerate(
            tqdm(get_batch_sizes(args.n_samples, args.sampling_batch_size))
        ):
            batch, times = gflownet.sample_batch(n_forward=bs, train=False)
            x_sampled = batch.get_terminating_states(proxy=True)
            energies = env.oracle(x_sampled)
            x_sampled = batch.get_terminating_states()
            dct = {"energy": energies.tolist()}
            pickle.dump(dct, open(tmp_dir / f"gfn_samples_{i}.pkl", "wb"))

        # Concatenate all samples
        print("Concatenating data")
        dct = {"energy": []}
        for f in tqdm(list(tmp_dir.glob("gfn_samples_*.pkl"))):
            tmp_dict = pickle.load(open(f, "rb"))
            dct = {k: v + tmp_dict[k] for k, v in dct.items()}
        pickle.dump(dct, open(output_dir / f"{prefix}_energies.pkl", "wb"))

        if "y" in input("Delete temporary files? (y/n)"):
            shutil.rmtree(tmp_dir)

    # ------------------------------------
    # -----  Sample random crystals  -----
    # ------------------------------------

    # Sample random crystals uniformly without constraints
    if args.random_crystals and args.n_samples > 0 and args.n_samples <= 1e6:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        print(f"Sampling {args.n_samples} random crystals without constraints...")
        for i, bs in enumerate(
            tqdm(get_batch_sizes(args.n_samples, args.sampling_batch_size))
        ):
            x_sampled = generate_random_crystals(
                n_samples=args.n_samples,
                elements=config.env.composition_kwargs.elements,
                min_elements=2,
                max_elements=5,
                max_atoms=config.env.composition_kwargs.max_atoms,
                max_atom_i=config.env.composition_kwargs.max_atom_i,
                space_groups=config.env.space_group_kwargs.space_groups_subset,
                min_length=0.0,
                max_length=1.0,
                min_angle=0.0,
                max_angle=1.0,
            )
            energies = env.oracle(env.states2proxy(x_sampled))
            dct = {"energy": energies.tolist()}
            pickle.dump(dct, open(tmp_dir / f"randomcrystals_samples_{i}.pkl", "wb"))

        # Concatenate all samples
        print("Concatenating data")
        dct = {"energy": []}
        for f in tqdm(list(tmp_dir.glob("randomcrystals_samples_*.pkl"))):
            tmp_dict = pickle.load(open(f, "rb"))
            dct = {k: v + tmp_dict[k] for k, v in dct.items()}
        pickle.dump(dct, open(output_dir / f"randomcrystals_energies.pkl", "wb"))

        if "y" in input("Delete temporary files? (y/n)"):
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser = add_args(parser)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    print_args(args)
    main(args)
    sys.exit()
