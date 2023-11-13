import argparse
import json
import os
import pickle
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from gflownet.utils.molecule.geom import get_all_confs_geom
from gflownet.utils.molecule.metrics import get_best_rmsd, get_cov_mat


def distant_enough(conf, others, delta):
    conf = deepcopy(conf)
    dist = []
    for c in others:
        dist.append(get_best_rmsd(deepcopy(c), conf))
    dist = np.array(dist)
    return np.all(dist > delta)


def get_diverse_top_k(confs_list, k=None, delta=1.25):
    """confs_list should be sorted according to the energy! lowest energy first"""
    result = [confs_list[0]]
    for conf in confs_list[1:]:
        if distant_enough(conf, result, delta):
            result.append(conf)
            if len(result) >= k:
                break
    if len(result) < k:
        print(
            f"Cannot find {k} different conformations in {len(confs_list)} for delta {delta}"
        )
        print(f"Found only {len(result)}. Adding random from generated")
        result += random.sample(confs_list.tolist(), k - len(result))
    return result


def get_smiles_from_filename(filename):
    smiles = filename.split("_")[1]
    if smiles.endswith(".pkl"):
        smiles = smiles[:-4]
    return smiles


def main(args):
    base_path = Path(args.geom_dir)
    drugs_file = base_path / "rdkit_folder/summary_drugs.json"
    with open(drugs_file, "r") as f:
        drugs_summ = json.load(f)

    filenames = [Path(args.gen_dir) / x for x in os.listdir(args.gen_dir)]
    gen_files = []
    gen_confs = []
    smiles = []
    energies = []
    for fp in filenames:
        with open(fp, "rb") as f:
            gen_files.append(pickle.load(f))
            # 1k cut-off to use the same max number of gen samples for all methods
            gen_confs.append(gen_files[-1]["conformer"][:1000])
            if "smiles" in gen_files[-1].keys():
                smiles.append(gen_files[-1]["smiles"])
            if "energy" in gen_files[-1].keys():
                energies.append(gen_files[-1]["energy"][:1000])
    if len(smiles) == 0:
        smiles = [get_smiles_from_filename(x.name) for x in filenames]
    print("All smiles")
    print(*smiles, sep="\n")
    ref_confs = [get_all_confs_geom(base_path, sm, drugs_summ) for sm in smiles]
    # filter out nans
    gen_confs = [gen_confs[idx] for idx, val in enumerate(ref_confs) if val is not None]
    smiles = [smiles[idx] for idx, val in enumerate(ref_confs) if val is not None]
    if len(energies) > 0:
        energies = [
            energies[idx] for idx, val in enumerate(ref_confs) if val is not None
        ]
    ref_confs = [val for val in ref_confs if val is not None]
    assert len(gen_confs) == len(ref_confs) == len(smiles)

    if args.use_top_k:
        if len(energies) == 0:
            raise Exception("Cannot use top-k without energies")
        energies = [np.array(e) for e in energies]
        indecies = [np.argsort(e) for e in energies]
        gen_confs = [np.array(x)[idx] for x, idx in zip(gen_confs, indecies)]
        if args.diverse:
            gen_confs = [
                get_diverse_top_k(x, k=len(ref) * 2, delta=args.delta)
                for x, ref in zip(gen_confs, ref_confs)
            ]
    if not args.hack:
        gen_confs = [x[: 2 * len(ref)] for x, ref in zip(gen_confs, ref_confs)]

    geom_stats = pd.read_csv(args.geom_stats, index_col=0)
    cov_list = []
    mat_list = []
    n_conf = []
    n_tas = []
    consistent = []
    has_h_tas = []
    hack = False
    for ref, gen, sm in tqdm(zip(ref_confs, gen_confs, smiles), total=len(ref_confs)):
        if len(gen) < 2 * len(ref):
            print("Recieved less samples that needed for computing metrics!")
            print(
                f"Computing metrics with {len(gen)} generated confs for {len(ref)} reference confs"
            )
        try:
            if len(gen) > 2 * len(ref):
                hack = True
                print(
                    f"Warning! Computing metrics with {len(gen)} generated confs for {len(ref)} reference confs"
                )
            cov, mat = get_cov_mat(ref, gen, threshold=1.25)
        except RuntimeError as e:
            print(e)
            cov, mat = None, None
        cov_list.append(cov)
        mat_list.append(mat)
        n_conf.append(len(ref))
        n_tas.append(
            geom_stats[geom_stats.smiles == sm].n_rotatable_torsion_angles_rdkit.values[
                0
            ]
        )
        consistent.append(
            geom_stats[geom_stats.smiles == sm].rdkit_consistent.values[0]
        )
        has_h_tas.append(geom_stats[geom_stats.smiles == sm].has_hydrogen_tas.values[0])

    data = {
        "smiles": smiles,
        "cov": cov_list,
        "mat": mat_list,
        "n_ref_confs": n_conf,
        "n_tas": n_tas,
        "has_hydrogen_tas": has_h_tas,
        "consistent": consistent,
    }
    df = pd.DataFrame(data)
    name = Path(args.gen_dir).name
    if name in ["xtb", "gfn-ff", "torchani"]:
        name = Path(args.gen_dir).name
        name = Path(args.gen_dir).parent.name + "_" + name
    if args.use_top_k:
        name += "_top_k"
    if args.diverse:
        name += f"_diverse_{args.delta}"
    if hack:
        name += "_hacked"

    output_file = Path(args.output_dir) / "{}_metrics.csv".format(name)
    df.to_csv(output_file, index=False)
    print("Saved metrics at {}".format(output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geom_dir",
        type=str,
        default="/home/mila/a/alexandra.volokhova/scratch/datasets/geom",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/mila/a/alexandra.volokhova/projects/gflownet/results/conformer/metrics",
    )
    parser.add_argument(
        "--geom_stats",
        type=str,
        default="/home/mila/a/alexandra.volokhova/projects/gflownet/scripts/conformer/geom_stats.csv",
    )
    parser.add_argument("--gen_dir", type=str, default="./")
    parser.add_argument("--use_top_k", type=bool, default=False)
    parser.add_argument("--diverse", type=bool, default=False)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--hack", type=bool, default=False)
    args = parser.parse_args()
    main(args)
