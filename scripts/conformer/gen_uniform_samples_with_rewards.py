
# tblite import should stay here first! othervise everything fails with tblite errors
from gflownet.proxy.conformers.tblite import TBLiteMoleculeEnergy

import os
import pandas as pd
import numpy as np
import pickle
import argparse
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from rdkit.Chem import AllChem, rdMolTransforms
from gflownet.envs.conformers.conformer import PREDEFINED_SMILES
from gflownet.utils.molecule.rotatable_bonds import get_rotatable_ta_list, find_rotor_from_smiles
from gflownet.proxy.conformers.torchani import TorchANIMoleculeEnergy
from gflownet.envs.conformers.conformer import Conformer



def get_uniform_samples_and_energy_weights(smiles, n_samples, energy_model='torchani'):
    n_torsion_angles = len(find_rotor_from_smiles(smiles))
    uniform_tas = np.random.uniform(0., 2*np.pi, size=(n_samples, n_torsion_angles))
    env = Conformer(smiles=smiles, n_torsion_angles=n_torsion_angles, reward_func="boltzmann", reward_beta=32)
    if energy_model == 'torchani':
        proxy = TorchANIMoleculeEnergy(device='cpu', float_precision=32)
    elif energy_model == 'tblite':
        proxy = TBLiteMoleculeEnergy(device='cpu', float_precision=32, batch_size=417)
    else:
        raise NotImplementedError(f"No proxy availabe for {energy_model}, use one of ['torchani', 'tblite']")
    proxy.setup(env)
    def get_weights(batch):
        batch = np.concatenate([batch, np.zeros((batch.shape[0], 1))], axis=-1)
        energies = proxy(env.statebatch2proxy(batch))
        rewards = env.proxy2reward(-energies)
        return rewards.numpy()
    weights = get_weights(uniform_tas)
    ddict = {f'ta_{idx}': uniform_tas[:, idx] for idx in range(n_torsion_angles)}
    ddict.update({'weights': weights})
    return pd.DataFrame(ddict)


def main(args):
    output_root = Path(args.output_dir)
    if not output_root.exists():
        os.mkdir(output_root)
    
    samples_root = Path('/home/mila/a/alexandra.volokhova/projects/gflownet/results/conformer/samples')
    selected_mols = pd.read_csv(samples_root / 'gfn_samples_2-12' / 'torchani_selected.csv')
    result = dict()
    for smiles in tqdm(selected_mols['SMILES'].values):
       result.update({
           smiles: get_uniform_samples_and_energy_weights(smiles, args.n_samples, args.energy_model)
       })
       if args.save_each_df:
           sm_idx = PREDEFINED_SMILES.index(smiles)
           filename = filename = output_root /  f"{args.energy_model}_{sm_idx}_weighted_samples_selected_smiles.csv"
           result[smiles].to_csv(filename)
    filename = output_root / f'{args.energy_model}_weighted_samples_selected_smiles.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(result, file)
    print(f"saved results at {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
    )
    parser.add_argument("--energy_model", type=str, choices=['torchani', 'tblite'], default='torchani')
    parser.add_argument("--save_each_df", type=bool, default=False)
    parser.add_argument("--output_dir", type=str,
                         default="/home/mila/a/alexandra.volokhova/projects/gflownet/results/conformer/samples/uniform_samples_with_reward_weights")
    args = parser.parse_args()
    main(args)
