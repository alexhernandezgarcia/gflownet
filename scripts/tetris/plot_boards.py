# This code is to plot the topk boards in the data vs the gflownet generated samples 

#TODOs: 
##  clean the path names and use relative names
## Find a way to use directly the experiments config and not the env config itself
import sys
import ast
import torch
from hydra.utils import get_original_cwd, instantiate
import pandas as pd
path = "/home/mila/l/lena-nehale.ezzine/ai4mols/gflownet/logs/tetris/8166081/2025-11-20_07-41-46_669946"
sys.path.append("/home/mila/l/lena-nehale.ezzine/ai4mols/gflownet")

import numpy as np 
from gflownet.utils.common import load_gflownet_from_rundir, read_hydra_config


config_path = "/home/mila/l/lena-nehale.ezzine/ai4mols/gflownet/config/env/tetris.yaml"

config = read_hydra_config(config_name=config_path)
# Take args from the exp file 5x8_mle.yaml
config.width = 5
config.height= 8
config.pieces= ["I", "J", "L", "O", "S", "T", "Z"]
config.rotations= [0, 90, 180, 270]
env = instantiate(
            config,
            device=torch.device("cuda"),
            float_precision=torch.float16,
        )
print(f"Environment: {env}")


# Samples from the Replay Buffer
converter = lambda x: ast.literal_eval(x.replace("inf", "2e308"))
replay = pd.read_csv("/home/mila/l/lena-nehale.ezzine/ai4mols/gflownet/data/tetris/replay.csv", index_col=0,
            converters={"samples": converter, "trajectories": converter})

samples =  torch.Tensor(replay['samples'])
rewards = torch.Tensor(replay['rewards'])


board_topk = env.plot_samples_topk(
        samples =  np.array(samples),
        rewards = rewards,
    )

board_topk.savefig("/home/mila/l/lena-nehale.ezzine/ai4mols/gflownet/data/tetris/board_topk_replay.png")



# Samples from the gflownet
converter = lambda x: ast.literal_eval(x.replace("inf", "2e308"))
gfn_samples = pd.read_csv("/home/mila/l/lena-nehale.ezzine/ai4mols/gflownet/logs/tetris/8166081/2025-11-20_07-41-46_669946/eval/samples/gfn_samples_0.csv", index_col=0,
           #converters={"readable": converter}
           )
print(gfn_samples.head())

readable = gfn_samples['readable'].tolist() 
samples = []
for x in readable: 
    rows = [row.replace('[', '').replace(']', '').split() for row in x.split('\n')]
    sample = np.array(rows, dtype=int)
    sample = torch.Tensor(sample)
    samples.append(sample)


samples = torch.stack(samples)
energies = torch.Tensor(gfn_samples['energies'].tolist())

board_topk = env.plot_samples_topk(
        samples =  np.array(samples),
        rewards = torch.exp(energies),
    )

board_topk.savefig("/home/mila/l/lena-nehale.ezzine/ai4mols/gflownet/data/tetris/board_topk_gfn_samples.png")