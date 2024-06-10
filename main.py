"""
Runnable script with hydra capabilities
"""

import os
import pickle
import random
import sys

import hydra
import pandas as pd
from omegaconf import open_dict

from gflownet.utils.policy import parse_policy_config


@hydra.main(config_path="./config", config_name="main", version_base="1.1")
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

    # Initialize GFlowNet from config
    # Logger
    logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)

    # The proxy is required by the GFlowNetAgent for computing rewards
    proxy = hydra.utils.instantiate(
        config.proxy,
        device=config.device,
        float_precision=config.float_precision,
    )

    # Using Hydra's partial instantiation, see:
    # https://hydra.cc/docs/advanced/instantiate_objects/overview/#partial-instantiation
    env_maker = hydra.utils.instantiate(
        config.env,
        device=config.device,
        float_precision=config.float_precision,
        _partial_=True,
    )
    env = env_maker()

    # The evaluator is used to compute metrics and plots
    evaluator = hydra.utils.instantiate(config.evaluator)

    # The policy is used to model the probability of a forward/backward action
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
    # State flow
    if config.gflownet.state_flow is not None:
        state_flow = hydra.utils.instantiate(
            config.gflownet.state_flow,
            env=env,
            device=config.device,
            float_precision=config.float_precision,
            base=forward_policy,
        )
    else:
        state_flow = None
    # GFlowNet Agent

    gflownet = hydra.utils.instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env_maker=env_maker,
        proxy=proxy,
        forward_policy=forward_policy,
        backward_policy=backward_policy,
        state_flow=state_flow,
        buffer=config.env.buffer,
        logger=logger,
        evaluator=evaluator,
    )

    # Train GFlowNet
    gflownet.train()

    # Sample from trained GFlowNet
    if config.n_samples > 0 and config.n_samples <= 1e5:
        batch, times = gflownet.sample_batch(n_forward=config.n_samples, train=False)
        x_sampled = batch.get_terminating_states(proxy=True)
        energies = proxy(x_sampled)
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
