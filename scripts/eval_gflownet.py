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
        "--random",
        action="store_true",
        help="Sample from an untrained GFlowNet",
    )
    parser.add_argument("--device", default="cpu", type=str)
    return parser


def set_device(device: str):
    if device.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def main(args):
    # Load config
    with initialize_config_dir(
        version_base=None, config_dir=args.run_path + "/.hydra", job_name="xxx"
    ):
        config = compose(config_name="config")
        print(OmegaConf.to_yaml(config))
    # Disable wandb
    config.logger.do.online = False
    # Logger
    logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)
    # The proxy is required in the env for scoring: might be an oracle or a model
    proxy = hydra.utils.instantiate(
        config.proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    # The proxy is passed to env and used for computing rewards
    env = hydra.utils.instantiate(
        config.env,
        proxy=proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    forward_config = parse_policy_config(config, kind="forward")
    backward_config = parse_policy_config(config, kind="backward")
    forward_policy = hydra.utils.instantiate(
        forward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
    )
    backward_policy = hydra.utils.instantiate(
        backward_config,
        env=env,
        device=config.device,
        float_precision=config.float_precision,
        base=forward_policy,
    )
    gflownet = hydra.utils.instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env=env,
        buffer=config.env.buffer,
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        logger=logger,
    )
    # Load final models
    if not args.random:
        ckpt = [
            f
            for f in Path(args.run_path).rglob(config.logger.logdir.ckpts)
            if f.is_dir()
        ][0]
        forward_final = [f for f in ckpt.glob(f"*final*")][0]
        backward_final = [f for f in ckpt.glob(f"*final*")][0]
        gflownet.forward_policy.model.load_state_dict(
            torch.load(forward_final, map_location=set_device(args.device))
        )
        gflownet.backward_policy.model.load_state_dict(
            torch.load(backward_final, map_location=set_device(args.device))
        )
    # Test GFlowNet model
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
    output_dir = Path(args.run_path) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for fig, figname in zip(figs, fignames):
        output_fig = output_dir / figname
        if fig is not None:
            fig.savefig(output_fig, bbox_inches="tight")

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
        if args.random:
            df.to_csv(output_dir / "random_samples.csv")
        else:
            df.to_csv(output_dir / "gfn_samples.csv")
        dct = {"x": x_sampled, "energy": energies}
        if args.random:
            pickle.dump(dct, open(output_dir / "random_samples.pkl", "wb"))
        else:
            pickle.dump(dct, open(output_dir / "gfn_samples.pkl", "wb"))

    # Store test data set
    gflownet.buffer.test.rename(columns={"samples": "readable"})
    gflownet.buffer.test.to_csv(output_dir / "test_samples.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser = add_args(parser)
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)
    sys.exit()
