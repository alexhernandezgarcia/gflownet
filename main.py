"""
Runnable script with hydra capabilities
"""
import sys
import random
import hydra
from omegaconf import OmegaConf, DictConfig
from gflownet.utils.common import flatten_config
import numpy as np
import itertools


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
    batch, times = gflownet.sample_batch(env, config.n_samples, train=False)

    # calculate performance
    queries = env.state2oracle(batch)
    energies = proxy(queries)
    energies_sorted = np.sort(energies)
    for k in config.gflownet.oracle.k:
        mean_topk = np.mean(energies_sorted[:k])
        print(f"\tAverage score top-{k}: {mean_topk}")
    # calculate diversity
    dists = []
    for pair in itertools.combinations(queries, 2):
        dists.append(env.get_distance(*pair))
    diversity = dists / (config.n_samples * (config.n_samples - 1))
    print(f"\tDiversity: {diversity}")
    # no estimate of novelty here because no original dataset


if __name__ == "__main__":
    main()
    sys.exit()
