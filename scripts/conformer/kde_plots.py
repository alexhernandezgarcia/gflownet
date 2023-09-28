# IMPORT THIS FIRST!!!!!
from tblite import interface

import time
import argparse
import pickle
import numpy as np
import os

from pathlib import Path
from scipy.special import logsumexp
from datetime import datetime
from tqdm import tqdm


from gflownet.proxy.conformers.xtb import XTBMoleculeEnergy
from gflownet.proxy.conformers.torchani import TorchANIMoleculeEnergy
from gflownet.proxy.conformers.tblite import TBLiteMoleculeEnergy
from gflownet.envs.conformers.conformer import Conformer
from gflownet.utils.common import torch2np

PROXY_DICT = {
    'tblite': TBLiteMoleculeEnergy,
    'xtb': XTBMoleculeEnergy,
    'torchani': TorchANIMoleculeEnergy
}
PROXY_NAME_DICT = {
    'tblite': 'xtb',
    'xtb': 'gfn-ff',
    'torchani': 'torchani'
}

def get_smiles_and_proxy_class(filename):
    sm = filename.split('_')[2]
    proxy_name = filename.split('_')[-1][:-4]
    proxy_class = PROXY_DICT[proxy_name]
    proxy = proxy_class(device="cpu", float_precision=32)
    env = Conformer(smiles=sm, n_torsion_angles=2, reward_func="boltzmann", reward_beta=32, proxy=proxy,
                    reward_sampling_method='nested')
    # proxy.setup(env)
    return sm, PROXY_NAME_DICT[proxy_name], proxy, env

def load_samples(filename, base_path):
    path = base_path / filename
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['x']

def get_true_kde(env, n_samples, bandwidth=0.1):
    x_from_reward = env.sample_from_reward(n_samples=n_samples)
    x_from_reward = torch2np(env.statetorch2kde(x_from_reward))
    kde_true = env.fit_kde(
                    x_from_reward,
                    kernel='gaussian',
                    bandwidth=bandwidth)
    return kde_true


def get_metrics(kde_true, kde_pred, test_samples):
    scores_true = kde_true.score_samples(test_samples)
    log_density_true = scores_true - logsumexp(scores_true, axis=0)
    scores_pred = kde_pred.score_samples(test_samples)
    log_density_pred = scores_pred - logsumexp(scores_pred, axis=0)
    density_true = np.exp(log_density_true)
    density_pred = np.exp(log_density_pred)
    # L1 error
    l1 = np.abs(density_pred - density_true).mean()
    # KL divergence
    kl = (density_true * (log_density_true - log_density_pred)).mean()
    # Jensen-Shannon divergence
    log_mean_dens = np.logaddexp(log_density_true, log_density_pred) + np.log(0.5)
    jsd = 0.5 * np.sum(density_true * (log_density_true - log_mean_dens))
    jsd += 0.5 * np.sum(density_pred * (log_density_pred - log_mean_dens))
    return l1, kl, jsd


def main(args):
    base_path = Path(args.samples_dir)
    output_base = Path(args.output_dir)
    if not output_base.exists():
        os.mkdir(output_base)
    for filename in tqdm(os.listdir(base_path)):
        ct = time.time()
        print(f'{datetime.now().strftime("%H-%M-%S")}: Initialising env')
        smiles, pr_name, proxy, env = get_smiles_and_proxy_class(filename)

        # create output dir
        current_datetime = datetime.now()
        timestamp = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        out_name = f'{pr_name}_{smiles}_{args.n_test}_{args.bandwidth}_{timestamp}'
        output_dir = output_base / out_name
        os.mkdir(output_dir)
        print(f'Will save results at {output_dir}')

        samples = load_samples(filename, base_path)
        n_samples = samples.shape[0]
        print(f'{datetime.now().strftime("%H-%M-%S")}: Computing true kde')
        kde_true = get_true_kde(env, n_samples, bandwidth=args.bandwidth)
        print(f'{datetime.now().strftime("%H-%M-%S")}: Computing pred kde')
        kde_pred = env.fit_kde(samples, kernel='gaussian', bandwidth=args.bandwidth)
        print(f'{datetime.now().strftime("%H-%M-%S")}: Making figures')
        fig_pred = env.plot_kde(kde_pred)
        fig_pred.savefig(output_dir / f'kde_pred_{out_name}.png', bbox_inches="tight", format='png')
        fig_true = env.plot_kde(kde_true)
        fig_true.savefig(output_dir / f'kde_true_{out_name}.png', bbox_inches="tight", format='png')
        print(f'{datetime.now().strftime("%H-%M-%S")}: Computing metrics')
        test_samples = np.array(env.get_grid_terminating_states(args.n_test))[:, :2]
        l1, kl, jsd = get_metrics(kde_true, kde_pred, test_samples)
        met = {
            'l1': l1,
            'kl': kl,
            'jsd': jsd
        }
        # write stuff
        with open(output_dir / 'metrics.pkl', 'wb') as file:
            pickle.dump(met, file)
        with open(output_dir / 'kde_true.pkl', 'wb') as file:
            pickle.dump(kde_true, file)
        with open(output_dir / 'kde_pred.pkl', 'wb') as file:
            pickle.dump(kde_pred, file)
         



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_dir', type=str, default='/home/mila/a/alexandra.volokhova/projects/gflownet/results/conformer/samples/mcmc_samples')
    parser.add_argument('--output_dir', type=str, default='/home/mila/a/alexandra.volokhova/projects/gflownet/results/conformer/kde_stats/mcmc')
    parser.add_argument('--bandwidth', type=float, default=0.15)
    parser.add_argument('--n_test', type=int, default=10000) 
    args = parser.parse_args()
    main(args)