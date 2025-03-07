"""
Runnable script with hydra capabilities

Script the resume the training of an existing run with checkpoints.
"""

# This is a hotfix for tblite (used for the conformer generation) not
# importing correctly unless it is being imported first.
try:
    from tblite import interface
except:
    pass

import os
import pickle
import random
import sys

import hydra
import pandas as pd

from gflownet.utils.common import load_gflownet_from_rundir


@hydra.main(config_path="./config", config_name="resume", version_base="1.1")
def main(config):

    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)

    # Determine whether training should continue from a previous run and set it up
    # TODO: consider merging run_config with config and save it in working directory
    if config.rundir:
        gflownet, run_config = load_gflownet_from_rundir(
            rundir=config.rundir,
            print_config=config.print_config,
            no_wandb=config.no_wandb,
            is_resumed=True,
        )
    else:
        print(
            "The attribute `rundir` must contain the path of a previous run. "
            "Aborting."
        )
        return

    print(
        f"\nTraining GFlowNet will be resumed from step {gflownet.it} from the "
        f" checkpoints and configuration found in {config.rundir}\n"
    )

    print(f"\nWorking directory of this run: {os.getcwd()}")
    print(f"Logging directory of this run: {gflownet.logger.logdir}\n")

    # Train GFlowNet
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
        df.to_csv("gfn_samples.csv")
        dct = {"x": x_sampled, "energy": energies}
        pickle.dump(dct, open("gfn_samples.pkl", "wb"))

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
