import sys
from argparse import ArgumentParser
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from yaml import safe_load
from copy import deepcopy

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "external" / "repos" / "ActiveLearningMaterials"))

from external.repos.ActiveLearningMaterials.utils.loaders import make_loaders
from gflownet.proxy.crystals.dav import DAV
from gflownet.utils.common import load_gflow_net_from_run_path, resolve_path


def set_seeds(seed):
    import random

    import numpy as np
    import torch

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_cls_ref():
    yaml_data = safe_load(
        (
            ROOT / "gflownet" / "env" / "crystal" / "crystal_lattice_systems.yaml"
        ).read_text()
    )
    ref = np.zeros((231,)) - 1
    for k, d in yaml_data.items():
        for sg in d["space_groups"]:
            ref[sg] = d[k]
    return ref


def get_ps_ref():
    yaml_data = safe_load(
        (ROOT / "gflownet" / "env" / "crystal" / "point_symmetries.yaml").read_text()
    )
    ref = np.zeros((231,)) - 1
    for k, d in yaml_data.items():
        for sg in d["space_groups"]:
            ref[sg] = d[k]
    return ref


def now():
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def sample_gfn(gflownet, n):
    x_sampled, _ = gflownet.sample_batch(gflownet.env, n, train=False)
    return torch.stack([gflownet.env.state2proxy(x) for x in x_sampled]).to(
        gflownet.device
    )


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


def top_k_data(data, top_k):
    top_data = {}
    keys = ["x_train_energy", "x_train_proxy", "x_gfn", "x_random"]
    for key in keys:
        y_key = key.split("_")[-1]
        x_key = key if "train" not in key else "x_train"
        indices = np.argsort(data[y_key])[:top_k]
        top_data[key] = data[x_key][indices]
    return top_data


def plot_sg_states(data, id_str, top_k):
    coefs = colors = None
    if top_k > 0:
        top_str = f"top-{top_k}"
        data = top_k_data(data, top_k)
        keys = ["x_train_energy", "x_train_proxy", "x_gfn", "x_random"]
        # colors = {
        #     "x_train_energy": "cadetblue",
        #     "x_train_proxy": "cyan",
        #     "x_gfn": "red",
        #     "x_random": "green",
        # }
        # coefs = {"x_train_energy": 1, "x_train_proxy": 1, "x_gfn": 1, "x_random": 1}
    else:
        top_str = "all"
        keys = ["x_train", "x_gfn", "x_random"]
        # colors = {"x_train": "cadetblue", "x_gfn": "red", "x_random": "green"}
        # coefs = {"x_train": 1, "x_gfn": 1, "x_random": 1}

    n_bins = 231
    plot_data = {k: v[:, 89] for k, v in data.items() if k in keys}
    name = "space group"
    plot_sgs(plot_data, id_str, top_str, n_bins, keys, name, colors, coefs)

    n_bins = 8
    cls_ref = get_cls_ref()
    plot_data = {k: cls_ref[v[:, 89]] for k, v in data.items() if k in keys}
    name = "crystal lattice system"
    plot_sgs(plot_data, id_str, top_str, n_bins, keys, name, colors, coefs)

    n_bins = 5
    ps_ref = get_ps_ref()
    plot_data = {k: ps_ref[v[:, 89]] for k, v in data.items() if k in keys}
    name = "point symmetries"
    plot_sgs(plot_data, id_str, top_str, n_bins, keys, name, colors, coefs)


def plot_sgs(data, id_str, top_str, n_bins, keys, name, colors=None, coefs=None):
    data = deepcopy(data)

    z = sum(coefs.values())
    coefs = {k: v / z for k, v in coefs.items()}

    f, axs = plt.subplots(
        len(keys) + 1,
        1,
        height_ratios=[2] + [1] * len(keys),
        sharex=True,
        figsize=(8, 10),
    )

    for k, key in enumerate(keys):
        axs[0].hist(
            data[key],
            bins=n_bins,
            alpha=coefs[key] if coefs is not None else 1,
            color=colors[key] if colors is not None else None,
            label=f"{name} in {key} ({top_str})",
        )
        axs[k + 1].hist(
            data[key],
            bins=n_bins,
            alpha=1,
            color=colors[key] if colors is not None else None,
            label=f"{name} in {key} ({top_str})",
        )

    axs[0].legend()
    plt.tight_layout()
    # plt.setp(axs, ylim=(0, 1000))
    outpath = (
        ROOT / "external" / "plots" / id_str / f"{name.replace(' ', '-')}_{top_str}.png"
    )
    outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(outpath)
    print("Saved to", outpath)


def plot_reward_hist(data, id_str, top_k):
    data = deepcopy(data)
    n_bins = 250
    coefs = {
        "energy": 2,
        "proxy": 2,
        "random": 1,
        "gfn": 1,
    }
    z = sum(coefs.values())
    coefs = {k: v / z for k, v in coefs.items()}
    colors = {"energy": "blue", "proxy": "red", "random": "green", "gfn": "brown"}

    if top_k > 0:
        top_str = f"top-{top_k}"
        top_data = {}
        for k, v in data.items():
            indices = np.argsort(v["energy"])[:top_k]
            top_data[k] = v[indices]
        data = top_data
    else:
        top_str = "all"

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
        label=f"{top_str}-train:energy ({data['energy'].shape[0]})",
    )
    a1.hist(
        data["energy"],
        bins=n_bins,
        alpha=coefs["energy"],
        color=colors["energy"],
        label=f"{top_str}-train:energy ({data['energy'].shape[0]})",
    )
    a0.hist(
        data["proxy"],
        bins=n_bins,
        alpha=coefs["proxy"],
        color=colors["proxy"],
        label=f"{top_str}-train:proxy ({data['proxy'].shape[0]})",
    )
    a2.hist(
        data["proxy"],
        bins=n_bins,
        alpha=coefs["proxy"],
        color=colors["proxy"],
        label=f"{top_str}-train:proxy ({data['proxy'].shape[0]})",
    )
    a0.hist(
        data["random"],
        bins=n_bins,
        alpha=coefs["random"],
        color=colors["random"],
        label=f"{top_str}-random:proxy ({data['random'].shape[0]})",
    )
    a3.hist(
        data["random"],
        bins=n_bins,
        alpha=coefs["random"],
        color=colors["random"],
        label=f"{top_str}-random:proxy ({data['random'].shape[0]})",
    )
    a0.hist(
        data["gfn"],
        bins=n_bins,
        alpha=coefs["gfn"],
        color=colors["gfn"],
        label=f"{top_str}-gfn:proxy ({data['gfn'].shape[0]})",
    )
    a4.hist(
        data["gfn"],
        bins=n_bins,
        alpha=coefs["gfn"],
        color=colors["gfn"],
        label=f"{top_str}-gfn:proxy ({data['gfn'].shape[0]})",
    )
    a0.legend()
    plt.tight_layout()
    plt.setp((a0, a1, a2, a3, a4), ylim=(0, 900))
    # plt.suptitle(f"Train set: ground truth energy vs. proxy ({n_bins} bins)")
    outpath = (
        ROOT / "external" / "plots" / id_str / f"train_energy_vs_proxy_{top_str}.png"
    )
    outpath.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(outpath)
    print("Saved to", outpath)


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--gflownet-path",
        type=str,
        default="/network/scratch/s/schmidtv/crystals/logs/neurips23/composition/2023-05-03_16-52-34",
    )
    parser.add_argument("--plot_reward_hist", action="store_true")
    parser.add_argument("--plot_sgs", action="store_true")
    parser.add_argument("--save_data", action="store_true")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--gfn_samples", type=int, default=-1)
    args = parser.parse_args()

    now_str = now()

    set_seeds(args.seed)

    dav_config = safe_load((ROOT / "config" / "proxy" / "dav.yaml").read_text())
    dav_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    dav_config["float_precision"] = 32
    dav_config["rescale_outputs"] = True
    dav = DAV(**dav_config)

    gflownet = load_gflow_net_from_run_path(args.gflownet_path, device=dav.device)

    loaders = make_loaders(dav.model_config)
    loaders["train"].dataset.ytransform = False
    loaders["train"].dataset.xtransform = False

    torch.set_grad_enabled(False)

    id_str = f"gfn{resolve_path(args.gflownet_path).name}_{now_str}"

    if not args.data_path:
        data = {
            "energy": [],
            "proxy": [],
            "random": [],
            "gfn": [],
            "x_train": [],
            "x_gfn": [],
            "x_random": [],
        }
        gfn_samples = (
            args.gfn_samples
            if args.gfn_samples > 0
            else len(loaders["train"]) * loaders["train"].batch_size
        )
        for b in tqdm(loaders["train"], desc="train loader"):
            x, y = b
            x[1] = x[1][:, None]
            x = torch.cat(x, dim=-1).to(dav.device)
            data["energy"].append(y.numpy())
            prox_pred = dav(x)
            data["proxy"].append(prox_pred.cpu().numpy())
            data["x_train"].append(x.cpu().numpy())

        for b in tqdm(range(len(loaders["train"])), desc="random samples"):
            bs = loaders["train"].batch_size
            x = torch.cat(
                [sample_random_crystal(comp_size=10)[None, :] for _ in range(bs)],
                axis=0,
            ).to(dav.device)
            prox_pred = dav(x)
            data["random"].append(prox_pred.cpu().numpy())
            data["x_random"].append(x.cpu().numpy())

        for b in tqdm(
            range(gfn_samples // loaders["train"].batch_size),
            desc=f"gfn samples (bs={loaders['train'].batch_size})",
        ):
            bs = loaders["train"].batch_size
            x = sample_gfn(gflownet, bs)
            x = x.to(dav.device)
            prox_pred = dav(x)
            data["gfn"].append(prox_pred.cpu().numpy())
            data["x_gfn"].append(x.cpu().numpy())

        data = {k: np.concatenate(v, axis=0) for k, v in data.items() if v}

        if args.save_data:
            data_path = ROOT / "external" / "plots" / id_str / "data.pkl"
            data_path.parent.mkdir(exist_ok=True, parents=True)
            with data_path.open("wb") as f:
                pickle.dump(data, f)
    else:
        with resolve_path(args.data_path).open("rb") as f:
            data = pickle.load(f)

    (ROOT / "external" / "plots").mkdir(exist_ok=True)

    if args.plot_reward_hist:
        if args.top_k > 0:
            plot_reward_hist(data, id_str, args.top_k)
        plot_reward_hist(data, id_str, -1)

    if args.plot_sgs:
        if args.top_k > 0:
            plot_sg_states(data, id_str, args.top_k)
        plot_sg_states(data, id_str, -1)
