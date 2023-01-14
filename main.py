"""
Runnable script with hydra capabilities
"""
import sys
import os
import random
import hydra
import pandas as pd
import yaml
from omegaconf import OmegaConf, DictConfig
from gflownet.utils.common import flatten_config


@hydra.main(config_path="./config", config_name="main", version_base="1.1")
def main(config):
    # Get current directory and set it as root log dir for Logger
    cwd = os.getcwd()
    config.logger.logdir.root = cwd
    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)
    # Log config
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}
    with open(cwd + "/config.yml", "w") as f:
        yaml.dump(log_config, f, default_flow_style=False)

    # Logger
    logger = None
    if config.log.skip == False:
        logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)
    # The proxy is required in the env for scoring: might be an oracle or a model
    proxy = hydra.utils.instantiate(config.proxy)
    # The proxy is passed to env and used for computing rewards
    env = hydra.utils.instantiate(config.env, proxy=proxy)
    gflownet = hydra.utils.instantiate(
        config.gflownet,
        env=env,
        buffer=config.env.buffer,
        logger=logger,
    )
    gflownet.train()

    # Sample from trained GFlowNet
    if config.n_samples > 0 and config.n_samples <= 1e5:
        samples, times = gflownet.sample_batch(env, config.n_samples, train=False)
        energies = env.oracle(env.statebatch2oracle(samples))
        df = pd.DataFrame(
            {
                "readable": [env.state2readable(s) for s in samples],
                "energies": energies,
            }
        )
        df.to_csv("gfn_samples.csv")
    print(gflownet.buffer.replay)

def set_seeds(seed):
    import torch
    import numpy as np
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
    sys.exit()
