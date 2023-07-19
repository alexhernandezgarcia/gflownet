"""
Runnable script with hydra capabilities
"""
import os
import pickle
import random
import sys

import hydra
import pandas as pd
import yaml
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="./config", config_name="main", version_base="1.1")
def main(config):
    # Get current directory and set it as root log dir for Logger
    cwd = os.getcwd()
    config.logger.logdir.root = cwd
    print(f"\nLogging directory of this run:  {cwd}\n")

    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)

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
#     print("Source")
#     print(env.state)
#     print("Attributes Node 0")
#     print(env._get_attributes(0))
#     print("Valid actions:")
#     for a, m in zip(env.action_space, env.get_mask_invalid_actions_forward()):
#         if m is False:
#             print(a)
#     print("Action 0.1 (0, 0) - node 0, pick leaf 0")
#     action = (0, 0)
#     next_state, action, valid = env.step(action)
#     print("New state")
#     print(env.state)
#     print("Parents and parents actions")
#     parents, parents_a = env.get_parents()
#     print(parents)
#     print(parents_a)
#     print("---")
#     print("State 1")
#     print("Attributes Node 0")
#     print(env._get_attributes(0))
#     print("Valid actions:")
#     for a, m in zip(env.action_space, env.get_mask_invalid_actions_forward()):
#         if m is False:
#             print(a)
#     print("Action 0.2 (1, 0, 0) - pick feature 2")
#     action = (1, 2)
#     next_state, action, valid = env.step(action)
#     print("New state")
#     print(env.state)
#     print("Parents and parents actions")
#     parents, parents_a = env.get_parents()
#     print(parents)
#     print(parents_a)
#     print("---")
#     print("State 2")
#     print("Attributes Node 0")
#     print(env._get_attributes(0))
#     print("Valid actions:")
#     for a, m in zip(env.action_space, env.get_mask_invalid_actions_forward()):
#         if m is False:
#             print(a)
#     print("Action 0.3 (2, 0.2) - pick threshold 0.2")
#     action = (2, 0.2)
#     next_state, action, valid = env.step(action)
#     print("New state")
#     print(env.state)
#     print("Parents and parents actions")
#     parents, parents_a = env.get_parents()
#     print(parents)
#     print(parents_a)
#     print("---")
#     print("State 3")
#     print("Attributes Node 0")
#     print(env._get_attributes(0))
#     print("Valid actions:")
#     for a, m in zip(env.action_space, env.get_mask_invalid_actions_forward()):
#         if m is False:
#             print(a)
#     print("Action 0.4 (3, 0) - node 0, pick operator <")
#     action = (3, 0)
#     next_state, action, valid = env.step(action)
#     print("New state")
#     print(env.state)
#     print("Parents and parents actions")
#     parents, parents_a = env.get_parents()
#     print(parents)
#     print(parents_a)
#     print("---")
#     print("State 3")
#     print("Attributes Node 0")
#     print(env._get_attributes(0))
#     print("Valid actions:")
#     for a, m in zip(env.action_space, env.get_mask_invalid_actions_forward()):
#         if m is False:
#             print(a)
#     import ipdb; ipdb.set_trace()
    gflownet = hydra.utils.instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env=env,
        buffer=config.env.buffer,
        logger=logger,
    )
    gflownet.train()

    # Sample from trained GFlowNet
    if config.n_samples > 0 and config.n_samples <= 1e5:
        batch, times = gflownet.sample_batch(n_forward=config.n_samples, train=False)
        x_sampled = batch.get_terminating_states(proxy=True)
        energies = env.oracle(x_sampled)
        x_sampled = batch.get_terminating_states()
        df = pd.DataFrame(
            {
                "readable": [env.state2readable(x) for x in x_sampled],
                "energies": energies.tolist(),
            }
        )
        df.to_csv("gfn_samples.csv")
        dct = {"x": x_sampled, "energy": energies}
        pickle.dump(dct, open("gfn_samples.pkl", "wb"))

    # Print replay buffer
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
