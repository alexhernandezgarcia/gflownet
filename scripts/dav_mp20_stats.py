import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from yaml import safe_load

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "external" / "repos" / "ActiveLearningMaterials"))

from external.repos.ActiveLearningMaterials.utils.loaders import make_loaders
from gflownet.proxy.crystals.dav import DAV
from gflownet.utils.common import load_gflow_net_from_run_path


def set_seeds(seed):
    import random

    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def now():
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def sample_gfn(gflownet, n):
    gflownet.logger.test.n = n
    x_sampled, _ = gflownet.sample_batch(
        gflownet.env,
        gflownet.logger.test.n,
        train=False,
    )
    return torch.tensor(x_sampled).to(gflownet.device)


def sample_random_crystal(comp_size=89):
    sg = torch.randint(1, 231, (1,))

    n_els = torch.randint(2, 6, (1,))  # sample number of elements [2; 5]
    n_ats = torch.randint(1, 13, (n_els,))  # sample number of atoms per element [1; 12]
    comp = torch.randint(1, comp_size, (n_els,))  # sample element id [1; 89]
    comp_vec = torch.zeros((comp_size,)).long()
    comp_vec[comp] = n_ats  # composition vector with counts per atom
    if comp_size < 89:
        comp_vec = torch.cat([comp_vec, torch.zeros((89 - comp_size,))])

    angles = torch.randint(30, 150, (3,))  # angles in [30, 150[
    lengths = torch.rand((3,)) * 4 + 1.0  # lengths in [1; 5[

    return torch.cat([comp_vec, sg, lengths, angles])


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--gflownet-path",
        type=str,
        default="/network/scratch/s/schmidtv/crystals/logs/neurips23/composition/2023-05-02_15-09-03",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    set_seeds(args.seed)

    dav_config = safe_load((ROOT / "config" / "proxy" / "dav.yaml").read_text())
    dav_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    dav_config["float_precision"] = 32
    dav_config["rescale_outputs"] = True
    dav = DAV(**dav_config)

    gflownet = load_gflow_net_from_run_path(args.gflownet_path)

    loaders = make_loaders(dav.model_config)
    loaders["train"].dataset.ytransform = False
    loaders["train"].dataset.xtransform = False

    torch.set_grad_enabled(False)

    data = {"energy": [], "proxy": [], "random": [], "gfn": []}
    for b in tqdm(loaders["train"], desc="train loader"):
        x, y = b
        x[1] = x[1][:, None]
        data["energy"].append(y.numpy())
        prox_pred = dav(torch.cat(x, dim=-1).to(dav.device))
        data["proxy"].append(prox_pred.cpu().numpy())

    for b in tqdm(range(len(loaders["train"])), desc="random samples"):
        bs = loaders["train"].batch_size
        x = torch.cat(
            [sample_random_crystal(comp_size=10)[None, :] for _ in range(bs)],
            axis=0,
        ).to(dav.device)
        prox_pred = dav(x)
        data["random"].append(prox_pred.cpu().numpy())

    for b in tqdm(range(len(loaders["train"])), desc="gfn samples"):
        bs = loaders["train"].batch_size
        x = sample_gfn(gflownet, bs).to(dav.device)
        prox_pred = dav(x)
        data["gfn"].append(prox_pred.cpu().numpy())

    data = {k: np.concatenate(v, axis=0) for k, v in data.items() if v}

    n_bins = 250
    (ROOT / "external" / "plots").mkdir(exist_ok=True)

    coefs = {
        "energy": 2,
        "proxy": 2,
        "random": 1,
        "gfn": 1,
    }
    z = sum(coefs.values())
    coefs = {k: v / z for k, v in coefs.items()}
    colors = {"energy": "blue", "proxy": "red", "random": "green", "gfn": "brown"}

    f, (a0, a1, a2, a3, a4) = plt.subplots(
        5,
        1,
        height_ratios=[2, 1, 1, 1, 1],
        sharex=True,
        figsize=(8, 10),
    )
    a0.hist(
        data["energy"],
        bins=n_bins,
        alpha=coefs["energy"],
        color=colors["energy"],
        label=f"train:energy ({data['energy'].shape[0]})",
    )
    a1.hist(
        data["energy"],
        bins=n_bins,
        alpha=coefs["energy"],
        color=colors["energy"],
        label=f"train:energy ({data['energy'].shape[0]})",
    )
    a0.hist(
        data["proxy"],
        bins=n_bins,
        alpha=coefs["proxy"],
        color=colors["proxy"],
        label=f"train:proxy ({data['proxy'].shape[0]})",
    )
    a2.hist(
        data["proxy"],
        bins=n_bins,
        alpha=coefs["proxy"],
        color=colors["proxy"],
        label=f"train:proxy ({data['proxy'].shape[0]})",
    )
    a0.hist(
        data["random"],
        bins=n_bins,
        alpha=coefs["random"],
        color=colors["random"],
        label=f"random:proxy ({data['random'].shape[0]})",
    )
    a3.hist(
        data["random"],
        bins=n_bins,
        alpha=coefs["random"],
        color=colors["random"],
        label=f"random:proxy ({data['random'].shape[0]})",
    )
    a0.hist(
        data["gfn"],
        bins=n_bins,
        alpha=coefs["gfn"],
        color=colors["gfn"],
        label=f"gfn:proxy ({data['gfn'].shape[0]})",
    )
    a4.hist(
        data["gfn"],
        bins=n_bins,
        alpha=coefs["gfn"],
        color=colors["gfn"],
        label=f"gfn:proxy ({data['gfn'].shape[0]})",
    )
    a0.legend()
    plt.tight_layout()
    plt.setp((a0, a1, a2, a3, a4), ylim=(0, 900))
    # plt.suptitle(f"Train set: ground truth energy vs. proxy ({n_bins} bins)")
    outpath = ROOT / "external" / "plots" / f"train_energy_vs_proxy_{now()}.png"
    plt.savefig(outpath)
    print("Saved to", outpath)
