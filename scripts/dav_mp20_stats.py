import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from yaml import safe_load

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "external" / "repos" / "ActiveLearningMaterials"))

from external.repos.ActiveLearningMaterials.utils.loaders import make_loaders
from gflownet.proxy.crystals.dav import DAV


def set_seeds(seed):
    import numpy as np
    import torch
    import random

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def now():
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def sample():
    sg = torch.randint(1, 231, (1,))
    n_els = torch.randint(2, 5, (1,))
    n_ats = torch.randint(1, 12, (n_els,))
    comp = torch.randint(1, 89, (n_els,))
    comp_vec = torch.zeros((89,)).long()
    comp_vec[comp] = n_ats
    angles = torch.randint(30, 150, (3,))
    lengths = torch.rand((3,)) * 4 + 1
    return torch.cat([comp_vec, sg, lengths, angles])


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--proxy-ckpt", type=str)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    set_seeds(args.seed)

    dav_config = safe_load((ROOT / "config" / "proxy" / "dav.yaml").read_text())
    dav_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    dav_config["float_precision"] = 32
    dav_config["rescale_outputs"] = False
    dav = DAV(**dav_config)

    loaders = make_loaders(dav.model_config)
    loaders["train"].dataset.ytransform = False

    torch.set_grad_enabled(False)

    data = {"energy": [], "proxy": [], "random": []}
    for b in tqdm(loaders["train"], desc="train loader"):
        x, y = b
        x[1] = x[1][:, None]
        data["energy"].append(y.numpy())
        prox_pred = dav(torch.cat(x, dim=-1).to(dav.device))
        prox_pred = prox_pred * dav.scales["y"]["std"] + dav.scales["y"]["mean"]
        data["proxy"].append(prox_pred.cpu().numpy())

    for b in tqdm(range(len(loaders["train"])), desc="random samples"):
        bs = loaders["train"].batch_size
        x = torch.cat([sample()[None, :] for _ in range(bs)], axis=0).to(dav.device)
        prox_pred = dav(x)
        prox_pred = prox_pred * dav.scales["y"]["std"] + dav.scales["y"]["mean"]
        data["random"].append(prox_pred.cpu().numpy())

    data = {k: np.concatenate(v, axis=0) for k, v in data.items() if v}

    n_bins = 250
    (ROOT / "external" / "plots").mkdir(exist_ok=True)

    plt.figure()
    plt.hist(
        data["energy"],
        bins=n_bins,
        alpha=1 / len(data),
        label=f"train:energy ({data['energy'].shape[0]})",
    )
    plt.hist(
        data["proxy"],
        bins=n_bins,
        alpha=1 / len(data),
        label=f"train:proxy ({data['proxy'].shape[0]})",
    )
    plt.hist(
        data["random"],
        bins=n_bins,
        alpha=1 / len(data),
        label=f"random:proxy ({data['random'].shape[0]})",
    )
    plt.legend()
    plt.title(f"Train set: ground truth energy vs. proxy ({n_bins} bins)")
    outpath = ROOT / "external" / "plots" / f"train_energy_vs_proxy_{now()}.png"
    plt.savefig(outpath)
    print("Saved to", outpath)
