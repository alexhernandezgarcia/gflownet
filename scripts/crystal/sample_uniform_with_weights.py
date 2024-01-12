"""
Script for sampling uniform crystals (w/o constraints) and evaluating them with reward function
should be run with the same config as main.py
"""
import pickle
import sys

import hydra
import pandas as pd

from gflownet.utils.common import chdir_random_subdir
from gflownet.utils.policy import parse_policy_config

from crystalrandom import generate_random_crystals


@hydra.main(config_path="../../config", config_name="main", version_base="1.1")
def main(config):
    N_SAMPLES = 10000
    
    proxy = hydra.utils.instantiate(
        config.proxy,
        device=config.device,
        float_precision=config.float_precision,
    )
    
    env = hydra.utils.instantiate(
        config.env,
        proxy=proxy,
        device=config.device,
        float_precision=config.float_precision,
    )

    x_sampled = generate_random_crystals(
                n_samples=N_SAMPLES,
                elements=config.env.composition_kwargs.elements,
                min_elements=2,
                max_elements=5,
                max_atoms=config.env.composition_kwargs.max_atoms,
                max_atom_i=config.env.composition_kwargs.max_atom_i,
                space_groups=config.env.space_group_kwargs.space_groups_subset,
                min_length=0.0,
                max_length=1.0,
                min_angle=0.0,
                max_angle=1.0,
            )
    env.reset()

    energies = env.proxy(env.states2proxy(x_sampled))
    rewards = env.proxy2reward(energies)
    result = {
        'samples': x_sampled,
        'rewards': rewards
    }

    path = '/home/mila/a/alexandra.volokhova/projects/gflownet/scripts/uni_crystals_with_weigths.pkl'

    with open(path, 'wb') as file:
        pickle.dump(result, file)
    print(path)
    

if __name__ == "__main__":
    main()
    sys.exit()