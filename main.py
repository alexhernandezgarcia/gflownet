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

from torch.profiler import profile, ProfilerActivity


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
    gflownet = hydra.utils.instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env=env,
        buffer=config.env.buffer,
        logger=logger,
    )
    gflownet.train()
    # with profile(
    #     activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
    # ) as prof:
    #     gflownet.train()
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=30))
    # prof.export_chrome_trace("trace.json")
    # return

    # Sample from trained GFlowNet
    if config.n_samples > 0 and config.n_samples <= 1e5:
        samples, times = gflownet.sample_batch(env, config.n_samples, train=False)
        energies = env.oracle(env.statebatch2oracle(samples))
        df = pd.DataFrame(
            {
                "readable": [env.state2readable(s) for s in samples],
                "energies": energies.tolist(),
            }
        )
        df.to_csv("gfn_samples.csv")
        dct = {"x": samples, "energy": energies}
        pickle.dump(dct, open("gfn_samples.pkl", "wb"))
    print(gflownet.buffer.replay)
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
