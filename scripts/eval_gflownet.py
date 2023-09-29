"""
Computes evaluation metrics and plots from a pre-trained GFlowNet model.
"""
import pickle
import sys
from argparse import ArgumentParser
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize, initialize_config_dir
import pandas as pd
from omegaconf import OmegaConf
from torch.distributions.categorical import Categorical

from gflownet.gflownet import GFlowNetAgent
from gflownet.utils.policy import parse_policy_config


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
    base_dir = Path(args.output_dir or args.run_path)

    # ---------------------------------
    # -----  Test GFlowNet model  -----
    # ---------------------------------

    if not args.samples_only:
        gflownet.logger.test.n = args.n_samples
        (
            l1,
            kl,
            jsd,
            corr_prob_traj_rew,
            var_logrew_logp,
            nll,
            figs,
            env_metrics,
        ) = gflownet.test()
        # Save figures
        keys = ["True reward and GFlowNet samples", "GFlowNet KDE Policy", "Reward KDE"]
        fignames = ["samples", "kde_gfn", "kde_reward"]

        output_dir = base_dir / "figures"
        print("output_dir: ", str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        for fig, figname in zip(figs, fignames):
            output_fig = output_dir / figname
            if fig is not None:
                fig.savefig(output_fig, bbox_inches="tight")
        print(f"Saved figures to {output_dir}")

        # Print metrics
        print(f"L1: {l1}")
        print(f"KL: {kl}")
        print(f"JSD: {jsd}")
        print(f"Corr (exp(logp), rewards): {corr_prob_traj_rew}")
        print(f"Var (log(R) - logp): {var_logrew_logp}")
        print(f"NLL: {nll}")

    # Sample from trained GFlowNet
    output_dir = Path(args.run_path) / "eval/samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.n_samples > 0 and args.n_samples <= 1e5:
        print(f"Sampling {args.n_samples} forward trajectories from GFlowNet...")
        batch, times = gflownet.sample_batch(n_forward=args.n_samples, train=False)
        x_sampled = batch.get_terminating_states(proxy=True)
        energies = env.oracle(x_sampled)
        x_sampled = batch.get_terminating_states()
        df = pd.DataFrame(
            {
                "readable": [env.state2readable(x) for x in x_sampled],
                "energies": energies.tolist(),
            }
        )
        df.to_csv(output_dir / "gfn_samples.csv")
        dct = {"x": x_sampled, "energy": energies}
        pickle.dump(dct, open(output_dir / "gfn_samples.pkl", "wb"))


if __name__ == "__main__":
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser = add_args(parser)
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)
    sys.exit()
