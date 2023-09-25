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

    filenames = [Path(args.gen_dir) / x for x in os.listdir(args.gen_dir)]
    gen_files = []
    gen_confs = []
    smiles = []
    for fp in filenames:
        with open(fp, 'rb') as f:
            gen_files.append(pickle.load(f))
            gen_confs.append(gen_files[-1]['conformer'])
            if 'smiles' in gen_files[-1].keys():
                smiles.append(gen_files[-1]['smiles'])
    if len(smiles) == 0: 
        smiles = [x.name.split('_')[-1][:-4] for x in filenames]
    print('All smiles')
    print(*smiles, sep='\n')
    ref_confs = [get_all_confs_geom(base_path, sm, drugs_summ) for sm in smiles]

    geom_stats = pd.read_csv(args.geom_stats, index_col=0)
    cov_list = []
    mat_list = []
    n_conf = []
    n_tas = []
    consistent = []
    has_h_tas = []
    for ref, gen, sm in tqdm(zip(ref_confs, gen_confs, smiles), total=len(ref_confs)):
        if len(gen) < 2*len(ref):
            print("Recieved less samples that needed for computing metrics! Return nans")
            cov, mat = None, None
        else:
            try: 
                cov, mat = get_cov_mat(ref, gen[:2*len(ref)], threshold=1.25)
            except RuntimeError as e:
                print(e)
                cov, mat = None, None
        cov_list.append(cov)
        mat_list.append(mat)
        n_conf.append(len(ref))
        n_tas.append(geom_stats[geom_stats.smiles == sm].n_rotatable_torsion_angles_rdkit.values[0])
        consistent.append(geom_stats[geom_stats.smiles == sm].rdkit_consistent.values[0])
        has_h_tas.append(geom_stats[geom_stats.smiles == sm].has_hydrogen_tas.values[0])
    
    data = {
        'smiles': smiles,
        'cov': cov_list,
        'mat': mat_list,
        'n_ref_confs': n_conf,
        'n_tas': n_tas,
        'has_hydrogen_tas': has_h_tas,
        'consistent': consistent 
    }
    df = pd.DataFrame(data)
    name = Path(args.gen_dir).name
    if name in ['xtb', 'gfn-ff', 'torchani']:
        name = Path(args.gen_dir).name
        name = Path(args.gen_dir).parent.name + '_' + name 
    output_file = Path(args.output_dir) / '{}_metrics.csv'.format(name)
    df.to_csv(output_file, index=False)
    print('Saved metrics at {}'.format(output_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geom_dir', type=str, default='/home/mila/a/alexandra.volokhova/scratch/datasets/geom')
    parser.add_argument('--output_dir', type=str, default='/home/mila/a/alexandra.volokhova/projects/gflownet/results/conformer/metrics')
    parser.add_argument('--geom_stats', type=str, default='/home/mila/a/alexandra.volokhova/projects/gflownet/scripts/conformer/geom_stats.csv') 
    parser.add_argument('--gen_dir', type=str, default='./') 
    args = parser.parse_args()
    main(args)