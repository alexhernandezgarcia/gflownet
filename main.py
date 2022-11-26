"""
Runnable script with hydra capabilities
"""
import sys
import random
import hydra
from omegaconf import OmegaConf, DictConfig
from src.gflownet.utils.common import flatten_config


@hydra.main(config_path="./config", config_name="main")
def main(config):
    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Log config
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}

    env = hydra.utils.instantiate(config.env)
    gflownet = hydra.utils.instantiate(config.gflownet, env=env, buffer=config.env.buffer)
    import ipdb; ipdb.set_trace()
    gflownet.train()

    # sample from the oracle, not from a proxy model
    batch, times = gflownet_agent.sample_batch(
        gflownet_agent.env, args.gflownet.n_samples, train=False
    )
    samples, times = batch2dict(batch, gflownet_agent.env, get_uncertainties=False)


if __name__ == "__main__":
    main()
    sys.exit()
