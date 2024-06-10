"""
Computes evaluation metrics and plots from a pre-trained GFlowNet model.
"""

import pickle
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from crystalrandom import generate_random_crystals_uniform
from hydra.utils import instantiate

from gflownet.gflownet import GFlowNetAgent
from gflownet.utils.common import load_gflow_net_from_run_path, read_hydra_config
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
        "--conditional_env_config_path",
        default=None,
        type=str,
        help="Path to a configuration YAML file containing the properties of an "
        "environment to be used to do conditional sampling, that is constrain the "
        "action space at sampling time. If the file is stored in the same directory "
        "as the main config, the argument may be just the file name (not a path).",
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
        "--random_only",
        default=False,
        action="store_true",
        help="Only sample random crystals, not from GFlowNet.",
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

    # ------------------------------------------
    # -----  Sample GFlowNet  -----
    # ------------------------------------------
    # Read conditional environment config, if provided
    # TODO: implement allow passing just name of config
    if args.conditional_env_config_path is not None:
        print(
            f"Reading conditional environment config from {args.conditional_env_config_path}"
        )
        config_cond_env = read_hydra_config(
            config_name=args.conditional_env_config_path
        )
        if "env" in config_cond_env:
            config_cond_env = config_cond_env.env
        env_cond = instantiate(
            config_cond_env,
            proxy=env.proxy,
            device=config.device,
            float_precision=config.float_precision,
        )
    else:
        env_cond = None

    # Handle output directory
    output_dir = base_dir / "eval" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    ### BANDGAP SPECIFIC ###
    # If the proxy is bandgap, make is_bandgap flag false so as not to apply the
    # transformation of the outputs and instead obtain the predicted bandgap as
    # "energy"
    if "DAVE" in config.proxy._target_ and env.proxy.is_bandgap:
        env.proxy.is_bandgap = False

        # Test
#         samples = [env.readable2state(readable) for readable in gflownet.buffer.test["samples"]]
#         energies = env.proxy(env.states2proxy(samples))
#         df = pd.DataFrame(
#             {
#                 "readable": gflownet.buffer.test["samples"],
#                 "energies": energies.tolist(),
#             }
#         )
#         df.to_csv(output_dir / f"val.csv")
#         dct = {"x": samples, "energy": energies.tolist()}
#         pickle.dump(dct, open(output_dir / f"val.pkl", "wb"))
# 
#         # Train
#         samples = [env.readable2state(readable) for readable in gflownet.buffer.train["samples"]]
#         energies = env.proxy(env.states2proxy(samples))
#         df = pd.DataFrame(
#             {
#                 "readable": gflownet.buffer.train["samples"],
#                 "energies": energies.tolist(),
#             }
#         )
#         df.to_csv(output_dir / f"train.csv")
#         dct = {"x": samples, "energy": energies.tolist()}
#         pickle.dump(dct, open(output_dir / f"train.pkl", "wb"))

    if args.n_samples > 0 and args.n_samples <= 1e5 and not args.random_only:
        print(
            f"Sampling {args.n_samples} forward trajectories",
            f"from GFlowNet in batches of {args.sampling_batch_size}",
        )
        for i, bs in enumerate(
            tqdm(get_batch_sizes(args.n_samples, args.sampling_batch_size))
        ):
            batch, times = gflownet.sample_batch(
                n_forward=bs, env_cond=env_cond, train=False
            )
            x_sampled = batch.get_terminating_states(proxy=True)
            energies = env.proxy(x_sampled)
            x_sampled = batch.get_terminating_states()
            df = pd.DataFrame(
                {
                    "readable": [env.state2readable(x) for x in x_sampled],
                    "energies": energies.tolist(),
                }
            )
            df.to_csv(tmp_dir / f"gfn_samples_{i}.csv")
            dct = {"x": x_sampled, "energy": energies.tolist()}
            pickle.dump(dct, open(tmp_dir / f"gfn_samples_{i}.pkl", "wb"))

        # Concatenate all samples
        print("Concatenating sample CSVs")
        df = pd.concat([pd.read_csv(f) for f in tqdm(list(tmp_dir.glob("*.csv")))])
        df.to_csv(output_dir / f"{prefix}_samples.csv")
        dct = {"x": [], "energy": []}
        for f in tqdm(list(tmp_dir.glob("*.pkl"))):
            tmp_dict = pickle.load(open(f, "rb"))
            dct = {k: v + tmp_dict[k] for k, v in dct.items()}
        pickle.dump(dct, open(output_dir / f"{prefix}_samples.pkl", "wb"))

        # Prints
        proxy_vals = np.array(dct["energy"])
        print(f"Mean proxy: {np.mean(proxy_vals)}")
        print(f"Std proxy: {np.std(proxy_vals)}")
        print(f"Median proxy: {np.median(proxy_vals)}")

        if "y" in input("Delete temporary files? (y/n)"):
            shutil.rmtree(tmp_dir)

    # ------------------------------------
    # -----  Sample random crystals  -----
    # ------------------------------------

    # Sample random crystals uniformly without constraints
    if args.random_crystals and args.n_samples > 0 and args.n_samples <= 1e5:
        print(f"Sampling {args.n_samples} random crystals without constraints...")
        x_sampled = generate_random_crystals_uniform(
            n_samples=args.n_samples,
            elements=config.env.composition_kwargs.elements,
            min_elements=config.env.composition_kwargs.min_diff_elem,
            max_elements=config.env.composition_kwargs.max_diff_elem,
            max_atoms=config.env.composition_kwargs.max_atoms,
            max_atom_i=config.env.composition_kwargs.max_atom_i,
            space_groups=config.env.space_group_kwargs.space_groups_subset,
            min_length=0.0,
            max_length=1.0,
            min_angle=0.0,
            max_angle=1.0,
        )
        energies = env.proxy(env.states2proxy(x_sampled))
        df = pd.DataFrame(
            {
                "readable": [env.state2readable(x) for x in x_sampled],
                "energies": energies.tolist(),
            }
        )
        df.to_csv(output_dir / "randomcrystals_samples.csv")
        dct = {"x": x_sampled, "energy": energies.tolist()}
        pickle.dump(dct, open(output_dir / "randomcrystals_samples.pkl", "wb"))
        print("Saved random crystals samples to CSV and pickle at ", output_dir)


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
