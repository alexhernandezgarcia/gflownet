"""
Runnable script with hydra capabilities
"""

import cProfile  # Imported to monitor number and time of function calls
import os
import pickle
import pstats  # Imported to monitor number and time of function calls
import random
import sys
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

    # Train GFlowNet with monitoring number and time of function calls
    if config.get("profile", False):
        profiler = cProfile.Profile()
        profiler.enable()
        gflownet.train()
        profiler.disable()

        stats = pstats.Stats(profiler).sort_stats(config.get("profile_sort", "cumtime"))
        print(
            f"\n=== cProfile: top {config.get('profile_n_rows', 20)} by "
            f"{config.get('profile_sort', 'cumtime')} ==="
        )
        stats.print_stats(config.get("profile_n_rows", 20))

        profile_path = Path(config.logger.logdir.path) / "train.prof"
        stats.dump_stats(profile_path)
        print(f"[profile] Raw stats saved to: {profile_path}")
        print(f"[profile] Inspect with:  snakeviz {profile_path}")
    else:
        gflownet.train()

    # Sample from trained GFlowNet
    # TODO: move to method in GFlowNet agent, like sample_and_log()
    if config.n_samples > 0 and config.n_samples <= 1e5:
        batch, times = gflownet.sample_batch(n_forward=config.n_samples, train=False)
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

    # Print replay buffer
    if len(gflownet.buffer.replay) > 0:
        print("\nReplay buffer:")
        print(gflownet.buffer.replay)

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


if __name__ == "__main__":
    main()
    sys.exit()
