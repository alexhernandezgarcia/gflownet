"""
Runnable script with hydra capabilities
"""
import sys
import random
import hydra
from omegaconf import OmegaConf, DictConfig
from gflownet.utils.common import flatten_config


@hydra.main(config_path="./config", config_name="main")
def main(config):
    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Log config
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}
    logger = None
    if config.log.skip == False:
        logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)
    # The proxy is required in the env for scoring: might be an oracle or a model
    proxy = hydra.utils.instantiate(config.proxy)
    # The proxy is passed to env and used for computing rewards
    env = hydra.utils.instantiate(config.env, proxy=proxy, logger=logger)
    gflownet = hydra.utils.instantiate(
        config.gflownet, env=env, buffer=config.env.buffer, logger=logger
    )
    gflownet.train()

    # sample from the oracle, not from a proxy model
    batch, times = gflownet.sample_batch(env, config.n_samples, train=False)
    print(gflownet.buffer.replay)


if __name__ == "__main__":
    main()
    sys.exit()
