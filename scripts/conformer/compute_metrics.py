import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from gflownet.utils.molecule.metrics import get_cov_mat
from gflownet.utils.molecule.geom import get_all_confs_geom


def main(args):
    base_path = Path(args.geom_dir)
    drugs_file = base_path / 'rdkit_folder/summary_drugs.json'
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    gen_files = [Path(args.gen_dir) / x for x in os.listdir(args.gen_dir)]
    smiles = [x.name.split('_')[-1][:-4] for x in gen_files]
    ref_confs = [get_all_confs_geom(base_path, sm, drugs_summ) for sm in smiles]
    gen_confs = []
    for fp in gen_files:
        with open(fp, 'rb') as f:
            gen_confs.append(pickle.load(f)['conformer'])

    cov_list = []
    mat_list = []
    n_conf = []
    for ref, gen in tqdm(zip(ref_confs, gen_confs)):
        cov, mat = get_cov_mat(ref, gen[:2*len(ref)], threshold=1.25)
        cov_list.append(cov)
        mat_list.append(mat)
        n_conf.append(len(ref))
    
    data = {
        'smiles': smiles,
        'cov': cov_list,
        'mat': mat_list,
        'n_ref_confs': n_conf
    }
    df = pd.DataFrame(data)
    df.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geom_dir', type=str, default='/home/mila/a/alexandra.volokhova/scratch/datasets/geom')
    parser.add_argument('--output_file', type=str, default='./gfn_metrics.csv')
    parser.add_argument('--gen_dir', type=str, default='./') 
    args = parser.parse_args()
    main(args)