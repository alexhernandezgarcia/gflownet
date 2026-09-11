"""
Runnable script with hydra capabilities
"""

import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import open_dict

from gflownet.utils.common import gflownet_from_config


@hydra.main(config_path="./config", config_name="train", version_base="1.1")
def main(config):

    # Set and print working and logging directory
    with open_dict(config):
        config.logger.logdir.path = (
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        )
    print(f"\nWorking directory of this run: {os.getcwd()}")
    print(f"Logging directory of this run: {config.logger.logdir.path}\n")

    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)

    # Initialize a GFlowNet agent from the configuration file
    gflownet = gflownet_from_config(config)

    # Train GFlowNet
    gflownet.train()

    # Sample from trained GFlowNet
    # TODO: move to method in GFlowNet agent, like sample_and_log()
    if config.n_samples > 0 and config.n_samples <= 1e5:
        if config.env.id == "toy":
            random_action_prob = gflownet.random_action_prob
            try:
                gflownet.random_action_prob = 0.0
                batch, times = gflownet.sample_batch(
                    n_forward=config.n_samples,
                    train=True,
                )
            finally:
                gflownet.random_action_prob = random_action_prob
        else:
            batch, times = gflownet.sample_batch(
                n_forward=config.n_samples,
                train=False,
            )
        x_sampled = batch.get_terminating_states(proxy=True)
        energies = gflownet.proxy(x_sampled)
        x_sampled = batch.get_terminating_states()
        df = pd.DataFrame(
            {
                "readable": [gflownet.env.state2readable(x) for x in x_sampled],
                "energies": energies.tolist(),
            }
        )
        samples_dir = Path("./samples/")
        samples_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(samples_dir / "gfn_samples.csv")
        dct = {"x": x_sampled, "energy": energies}
        pickle.dump(dct, open(samples_dir / "gfn_samples.pkl", "wb"))
        if config.env.id == "toy":
            n_trj = min(len(x_sampled), 100)
            save_toy_sample_summary(gflownet, x_sampled, batch, n_trj)

    # Close logger
    # TODO: make it gflownet.end() - perhaps there are other things to end
    gflownet.logger.end()


def set_seeds(seed):
    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_toy_sample_summary(gflownet, x_sampled, batch, n_trajectories=20):
    """
    Saves a compact readable summary of the final Toy samples.
    """
    readable_states = [gflownet.env.state2readable(x) for x in x_sampled]
    state_counts = Counter(readable_states)
    lines = []
    lines.append("Final Toy evaluation samples:")
    lines.append("Final state counts:")
    for state, count in sorted(
        state_counts.items(), key=lambda item: int(item[0].lstrip("s"))
    ):
        lines.append(f"  {state}: {count}")

    lines.append(f"First {min(n_trajectories, len(readable_states))} trajectories:")
    for idx, trajectory in enumerate(
        batch.get_actions_trajectories()[:n_trajectories], start=1
    ):
        lines.append(f"  {idx:02d}: {gflownet.env.traj2readable(trajectory)}")

    summary_path = gflownet.logger.logdir / "toy_final_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"Saved final Toy evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
    sys.exit()
