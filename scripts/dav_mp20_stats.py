import pickle
import sys
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm
from yaml import safe_load

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "external" / "repos" / "ActiveLearningMaterials"))
CMAP = mpl.colormaps["cividis"]

from collections import Counter

from external.repos.ActiveLearningMaterials.dave.utils.loaders import make_loaders

from gflownet.proxy.crystals.dave import DAVE
from gflownet.utils.common import load_gflow_net_from_run_path, resolve_path


def make_str(v):
    return "".join(["".join([chr(i + 97) for _ in range(k)]) for i, k in enumerate(v)])


def all_dists(x_array):
    x_strs = [make_str(x) for x in x_array.astype(int)]
    dists = np.zeros((len(x_strs), len(x_strs)))
    for i, x1 in enumerate(tqdm(x_strs)):
        for j, x2 in enumerate(x_strs):
            dists[i, j] = levenshtein_distance(x1, x2)
    return dists


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
            ROOT / "gflownet" / "envs" / "crystals" / "crystal_lattice_systems.yaml"
        ).read_text()
    )
    ref = np.zeros((231,)) - 1
    for k, d in yaml_data.items():
        for sg in d["space_groups"]:
            ref[sg] = k
    return ref


def get_ps_ref():
    yaml_data = safe_load(
        (ROOT / "gflownet" / "envs" / "crystals" / "point_symmetries.yaml").read_text()
    )
    ref = np.zeros((231,)) - 1
    for k, d in yaml_data.items():
        for sg in d["space_groups"]:
            ref[sg] = k
    return ref


def now():
    import datetime

    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def sample_gfn(gflownet, n):
    prob = deepcopy(gflownet.random_action_prob)
    gflownet.random_action_prob = 0.0
    x_sampled, _ = gflownet.sample_batch(gflownet.env, n, train=False)
    gflownet.random_action_prob = prob
    return torch.stack([gflownet.env.state2proxy(x) for x in x_sampled]).to(
        gflownet.device
    )


def sample_gfn_uniform(gflownet, n):
    prob = deepcopy(gflownet.random_action_prob)
    gflownet.random_action_prob = 1.0
    x_sampled, _ = gflownet.sample_batch(gflownet.env, n, train=False)
    gflownet.random_action_prob = prob
    return torch.stack([gflownet.env.state2proxy(x) for x in x_sampled]).to(
        gflownet.device
    )


def top_k_data(data, top_k):
    top_data = {}
    keys = ["x_train_energy", "x_train_proxy", "x_gfn", "x_random"]
    for key in keys:
        y_key = key.split("_")[-1]
        x_key = key if "train" not in key else "x_train"
        indices = np.argsort(data[y_key])[:top_k]
        top_data[key] = data[x_key][indices]
        top_data[y_key] = data[y_key][indices]
    return top_data


def plot_sg_states(data, id_str, top_k):
    data = deepcopy(data)

    keys = ["x_train_energy", "x_train_proxy", "x_gfn", "x_random"]
    data["x_train_energy"] = data["x_train"]
    data["x_train_proxy"] = data["x_train"]

    normalizer = mpl.colors.Normalize(vmin=-1, vmax=len(keys))
    colors = {k: CMAP(normalizer(i)) for i, k in enumerate(keys)}
    coefs = {k: 1 / 3 for k in keys}

    name = "space group"
    plot_data = deepcopy(data)
    for k, v in data.items():
        if k.startswith("x_"):
            plot_data[k] = v[:, 89].astype(int)
    plot_sgs(plot_data, id_str, top_k, keys, name, colors, coefs)

    name = "crystal lattice system"
    cls_ref = get_cls_ref()
    plot_data = deepcopy(data)
    for k, v in data.items():
        if k.startswith("x_"):
            plot_data[k] = cls_ref[v[:, 89].astype(int)]
    plot_sgs(plot_data, id_str, top_k, keys, name, colors, coefs)

    name = "point symmetries"
    ps_ref = get_ps_ref()
    plot_data = deepcopy(data)
    for k, v in data.items():
        if k.startswith("x_"):
            plot_data[k] = ps_ref[v[:, 89].astype(int)]
    plot_sgs(plot_data, id_str, top_k, keys, name, colors, coefs)


def energy_curve_data(categories, energies):
    data = {c: [] for c in set(categories)}
    for c, e in zip(categories, energies):
        data[c].append(e)
    means = {c: np.mean(v) for c, v in data.items()}
    stds = {c: np.std(v) for c, v in data.items()}
    # deciles:
    d19 = {c: np.percentile(v, (0, 100)) for c, v in data.items()}
    d1 = {c: d19[c][0] for c in d19}
    d9 = {c: d19[c][1] for c in d19}
    m_k = sorted(means.keys())
    means = [means[k] for k in m_k]
    stds = [stds[k] for k in m_k]
    d1 = [d1[k] for k in m_k]
    d9 = [d9[k] for k in m_k]
    return np.array(means), np.array(stds), np.array(d1), np.array(d9)


def plot_sgs(data, id_str, top_k, keys, name, colors=None, coefs=None):
    data = deepcopy(data)
    all_str = f"all={len(list(data.values())[0])}"
    top_data = top_str = None
    if top_k > 0:
        top_data = top_k_data(data, top_k)
        top_str = f"top-{top_k}"

    if coefs is not None:
        if any(c > 1 for c in coefs.values()):
            z = sum(coefs.values())
            coefs = {k: v / z for k, v in coefs.items()}
    else:
        z = len(keys)
        coefs = {k: 1 / len(keys) for k in keys}

    COMBINED = True

    f, axs = plt.subplots(
        len(keys) + 1 if COMBINED else len(keys),
        1 if top_data is None else 2,
        height_ratios=[1] + [1] * len(keys) if COMBINED else [1] * len(keys),
        sharex=True,
        figsize=(18, 15),
    )

    if top_data is None:
        axs = axs[:, None]

    datas = [data] if top_data is None else [top_data, data]
    all_strs = [all_str] if top_data is None else [top_str, all_str]

    min_e = min([min([v.min() for k, v in d.items() if "x" not in k]) for d in datas])
    max_e = max([max([v.max() for k, v in d.items() if "x" not in k]) for d in datas])

    for d, (frame, split_str) in enumerate(zip(datas, all_strs)):
        for k, key in enumerate(keys):
            count = Counter(frame[key].tolist())
            xs = sorted(count.keys())
            ys = [count[x] for x in xs]
            label = f"{name} in {key} ({split_str})"

            m_e, s_e, d1, d9 = energy_curve_data(
                frame[key], frame[key.replace("x_train_", "").replace("x_", "")]
            )
            if COMBINED:
                axs[0][d].bar(
                    xs,
                    ys,
                    alpha=coefs[key] if coefs is not None else 1,
                    color=colors[key] if colors is not None else None,
                    label=label,
                )

            ax = axs[k + 1 if COMBINED else k][d]

            # if d > 0 and k == 0:
            #     ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
            #     ax.tick_params(axis="x", which="both", bottom=False)
            #     ax.set_yticks([])
            #     continue

            if d > 0:
                if k == 1:
                    label = label.replace("_proxy", "").replace("_energy", "")

            ax.bar(
                xs,
                ys,
                alpha=1,
                color=colors[key] if colors is not None else None,
                label=label,
            )
            ax.set_title(
                "Bar plot for " + label,
                y=0,
                pad=-8 - (12 if k == len(keys) - 1 else 0),
                verticalalignment="top",
            )
            plot_color = np.array(colors[key]) + 0.1 if colors is not None else None
            if colors is not None:
                plot_color[-1] = 1.0
                plot_color = tuple(plot_color)
            ax_r = ax.twinx()
            ax_r.plot(
                xs,
                m_e,
                color=plot_color,
                label="mean energy",
            )
            ax_r.fill_between(
                xs,
                m_e - s_e,
                m_e + s_e,
                facecolor=plot_color,
                alpha=0.5,
                label="1 std range",
            )
            ax_r.fill_between(
                xs,
                d1,
                d9,
                facecolor=plot_color,
                alpha=0.25,
                label="min/max range",
            )
            ax_r.legend()
            ax_r.set_ylim(min_e, max_e)
            # axs[k + 1 if COMBINED else k][d].legend()

        if COMBINED:
            axs[0][d].legend()

    cols = [s for s in all_strs]

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    f.suptitle(
        " ".join([n.capitalize() for n in name.split()])
        + " Distributions: Bar plots and Energy Curves",
        fontsize=17,
    )

    f.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.setp(axs[:, 0], ylim=(0, max([ax.get_ylim()[1] for ax in axs[:, 0]])))
    if top_data is not None:
        plt.setp(axs[:, 1], ylim=(0, max([ax.get_ylim()[1] for ax in axs[:, 1]])))

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
        top_str = f"all ({len(data['energy'])})"

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

    dave_config = safe_load((ROOT / "config" / "proxy" / "dave.yaml").read_text())
    dave_config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    dave_config["float_precision"] = 32
    dave_config["rescale_outputs"] = True
    dave = DAVE(**dave_config)

    gflownet, _ = load_gflow_net_from_run_path(args.gflownet_path, device=dave.device)

    loaders = make_loaders(dave.model_config)
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

        bs = loaders["train"].batch_size
        n_batches = len(loaders["train"])
        n_samples = len(loaders["train"].dataset)
        gfn_samples = args.gfn_samples if args.gfn_samples > 0 else n_samples

        for b in tqdm(loaders["train"], desc="train loader"):
            x, y = b
            x[1] = x[1][:, None]
            x = torch.cat(x, dim=-1).to(dave.device)
            data["energy"].append(y.numpy())
            prox_pred = dave(x)
            data["proxy"].append(prox_pred.cpu().numpy())
            data["x_train"].append(x.cpu().numpy())

        for b in tqdm(range(n_batches), desc="random samples"):
            x = sample_gfn_uniform(gflownet, bs).to(dave.device)
            prox_pred = dave(x)
            data["random"].append(prox_pred.cpu().numpy())
            data["x_random"].append(x.cpu().numpy())

        for b in tqdm(range(gfn_samples // bs), desc=f"gfn samples (bs={bs})"):
            x = sample_gfn(gflownet, bs)
            x = x.to(dave.device)
            prox_pred = dave(x)
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
        plot_reward_hist(data, id_str, args.top_k)

    if args.plot_sgs:
        plot_sg_states(data, id_str, args.top_k)

    plt.close("all")
