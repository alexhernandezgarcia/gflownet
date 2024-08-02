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

from hydra.utils import instantiate

from gflownet.gflownet import GFlowNetAgent
from gflownet.utils.common import load_gflow_net_from_run_path, read_hydra_config
from gflownet.utils.policy import parse_policy_config


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns
    -------
    argparse.ArgumentParser
        The parser with added arguments
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
        "--randominit",
        action="store_true",
        help="Sample from an untrained GFlowNet",
    )
    parser.add_argument("--device", default="cpu", type=str)
    return parser


def get_batch_sizes(total, b=1):
    """
    Batches an iterable into chunks of size n and returns their expected lengths.

    Example
    -------

    .. code-block:: python

        >>> get_batch_sizes(10, 3)
        [3, 3, 3, 1]

    Parameters
    ----------
    total : int
        total samples to produce
    b : int
        the batch size

    Returns
    -------
    list
        list of batch sizes
    """
    n = total // b
    chunks = [b] * n
    if total % b != 0:
        chunks += [total % b]
    return chunks


def print_args(args):
    """
    Prints the arguments

    Parameters
    ----------
    args : argparse.Namespace
        the parsed arguments
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


def path_compatible(str):
    """
    Replace all non-alphanumeric characters with underscores

    Parameters
    ----------
    str : str
        The string to be made compatible
    """
    return "".join([c if c.isalnum() else "_" for c in str])


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
        gflownet.evaluator.n = args.n_samples
        eval_results = gflownet.evaluator.eval()

        # TODO-V: legacy -> ok to remove?
        # keys = ["True reward and GFlowNet samples", "GFlowNet KDE Policy", "Reward KDE"]
        # fignames = ["samples", "kde_gfn", "kde_reward"]

        output_dir = base_dir / "figures"
        print("output_dir: ", str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        if "figs" in eval_results:
            for figname, fig in eval_results["figs"].items():
                output_fig = output_dir / (path_compatible(figname) + ".pdf")
                if fig is not None:
                    fig.savefig(output_fig, bbox_inches="tight")
            print(f"Saved figures to {output_dir}")

        # Print metrics
        if "metrics" in eval_results:
            print("Metrics:")
            for k, v in eval_results["metrics"].items():
                print(f"\t{k}: {v:.4f}")

    # ------------------------------------------
    # -----  Sample GFlowNet  -----
    # ------------------------------------------

    # Read conditional environment config, if provided
    if args.conditional_env_config_path is not None:
        conditional_env_config_path = Path(args.conditional_env_config_path)
        if conditional_env_config_path.parent == Path("."):
            conditional_env_config_path = (
                Path(args.run_path) / ".hydra" / conditional_env_config_path.name
            )
        config_cond_env = read_hydra_config(config_name=conditional_env_config_path)
        if "env" in config_cond_env:
            config_cond_env = config_cond_env.env
        env_cond = instantiate(
            config_cond_env,
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

    if args.n_samples > 0 and args.n_samples <= 1e5:
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
            energies = gflownet.proxy(x_sampled)
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
