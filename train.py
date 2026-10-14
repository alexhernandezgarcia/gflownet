"""
This script trains a GFlowNet with configuration handling based on Hydra.

Example of use:

Train a GFlowNet from scratch:

.. code-block:: bash

    python train.py arg1=value1 arg2=value2 ...

Resume an interrupted run:

.. code-block:: bash

    python train.py rundir=path/to/logging/directory/

Resume an interrupted run with different configuration, for example, more iterations:

.. code-block:: bash

    python train.py rundir=path/to/logging/directory/ gflownet.optimizer.n_train_steps=1000

Note that resuming is only possible if model checkpoints are saved.

The script implements functionality for an automatic resuming after a preemption in a SLURM-based cluster. In order
to enable this, the initial slurm job should be submitted using a bash script analogous
to `mila/sbatch/sbatch_example_preemption_handling.sh`. The key aspects facilitating preemption:
1. Use partitions which allow for enough time to handle preemption. On Mila cluster it
    is main-grace and long-grace
2. Store the path to the bash script in a separate variable, e.g.
    `SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"`
3. Use `srun` (not `sbatch`) to submit the job inside the bash script
4. Pass the path to the bash script using this argument `sbatch_script_path=$SCRIPT_PATH`
5. Pass the path to the rundir using exactly this argument and exactly this
    env variable: `rundir="${MY_RUN_DIR:-null}"`
6. Submit the script itself using sbatch, e.g.
`sbatch mila/sbatch/sbatch_example_preemption_handling.sh +experiments=conformer/ctorus_sanity user=$USER logger.do.online=true`
"""

import os
import pickle
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
        samples_dir = Path("./samples/")
        samples_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(samples_dir / "gfn_samples.csv")
        dct = {"x": x_sampled, "energy": energies}
        pickle.dump(dct, open(samples_dir / "gfn_samples.pkl", "wb"))

    # Print reward-based replay buffer
    if gflownet.has_replay_reward and len(gflownet.buffers.replay_reward.replay) > 0:
        print("\nReplay buffer:")
        print(gflownet.buffers.replay_reward.replay)

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
