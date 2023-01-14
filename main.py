"""
Runnable script with hydra capabilities
"""
import sys
import random
import hydra
from omegaconf import OmegaConf, DictConfig
from gflownet.utils.common import flatten_config
import numpy as np
import copy


@hydra.main(config_path="./config", config_name="main")
def main(config):
    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Log config
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}

    # The proxy is required in the env for scoring: might be an oracle or a model
    proxy = hydra.utils.instantiate(config.proxy, device=config.device)
    # The proxy is passed to env and used for computing rewards
    env = hydra.utils.instantiate(config.env, proxy=proxy)
    gflownet = hydra.utils.instantiate(
        config.gflownet, env=env, buffer=config.env.buffer, device=config.device
    )
    gflownet.train()

    # sample from the oracle, not from a proxy model
    envs = []
    # for idx in range(config.n_samples):
    #     env = hydra.utils.instantiate(config.env, proxy=proxy)
    #     envs.append(env)
    env2 = hydra.utils.instantiate(config.env, proxy=proxy)
    # envs = [env2.copy() for _ in range(32)]
    envs = [copy.deepcopy(env2) for _ in range(32)]
    batch, times = gflownet.sample_batch(env, config.n_samples, train=False)
    _, _, _ = gflownet.evaluate(batch, oracle = proxy, performance = True, diversity=True, novelty=False)
    


if __name__ == "__main__":
    main()
    sys.exit()
