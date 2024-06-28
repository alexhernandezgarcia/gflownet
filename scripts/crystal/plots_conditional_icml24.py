"""
Script for plotting violin plots for conditional sampling. 
example cli:
python scripts/crystal/plots_conditional_icml24.py --pkl_path=/home/mila/a/alexandra.volokhova/projects/gflownet-dev/external/data/starling_fe/samples/gfn_samples.pkl --cond_dir_root=/home/mila/a/alexandra.volokhova/projects/gflownet-dev/external/data/starling_fe_conditional
"""

import argparse
import datetime
import os
import pickle
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from mendeleev.fetch import fetch_table
from plots_icml24 import load_gfn_samples, now_to_str
from seaborn_fig2grid import SeabornFig2Grid
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent


def plot_gfn_energies_violins(
    dfs, output_path=None, target="eform", energy_key="energy"
):
    # TODO update for density target
    # plt.rcParams.update({"font.family": "serif"})
    palette = sns.color_palette("colorblind", 5)
    for k, df in dfs.items():
        df["source"] = k

    df_plot = pd.concat(list(dfs.values()))
    # import ipdb; ipdb.set_trace()

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150, constrained_layout=True)
    ax = sns.violinplot(
        ax=ax,
        data=df_plot,
        y=energy_key,
        hue="source",
        palette=palette,
        hue_order=["Crystal-GFN (FE)", "A", "B", "C", "D"],
        cut=0,
    )
    # Manually set y-lims
    ax.set_ylim([-4.5, 2.0] if target == "eform" else [-1, 7.0])
    # Horizontal line at Energy 0
    ax.axhline(
        y=0.0 if target == "eform" else 1.34,
        linestyle="dashed",
        color="slategrey",
        zorder=1,
        linewidth=1.0,
    )
    if target == "bandgap":
        plt.text(0.4, 1.5, "1.34 eV", fontsize=9)

    legend = ax.get_legend()
    legend.set_title("")
    ax.set_ylabel(
        "Predicted formation energy [eV/atom]"
        if target == "eform"
        else "Predicted band gap [eV]"
    )
    sns.despine()
    # plt.suptitle(
    #     "Bang Gap target (1.34 eV)" if target == "bandgap" else "Formation Energy (↓)",
    #     fontsize=20,
    # )
    if output_path is not None:
        name = f"{target}_energies_violins_cond.pdf"
        fig.savefig(output_path / name, bbox_inches="tight")
        fig.savefig(output_path / name.replace(".pdf", ".png"), bbox_inches="tight")
        print(f"\n ✅ Saved {output_path / name} and .png\n")


def load_energies_only(pkl_path, energy_key="energy"):
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)
    df = pd.DataFrame({energy_key: samples[energy_key]})
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl_path", type=str, default=None, help="gflownet samples path"
    )
    parser.add_argument(
        "--cond_dir_root",
        type=str,
        default=None,
        help="path to root dir with conditional samples",
    )
    parser.add_argument(
        "--no_suptitles", action="store_true", default=False, help="Prevent suptitles"
    )
    parser.add_argument(
        "--target", type=str, default="eform", choices=["eform", "bandgap", "density"]
    )
    args = parser.parse_args()

    now = now_to_str()
    output_path = ROOT / "external" / "plots" / "icml24" / now
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to {output_path}")

    USE_SUPTITLES = not args.no_suptitles
    # elements = ['H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'K', 'V', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Se']
    sdf = load_energies_only(pkl_path=args.pkl_path)
    dfs = {"Crystal-GFN (FE)": sdf}
    cond_root = Path(args.cond_dir_root)
    cond_paths = {
        x[-1].upper(): cond_root / x / "eval/samples/gfn_samples.pkl"
        for x in os.listdir(cond_root)
    }
    cdfs = {k: load_energies_only(pkl_path=v) for k, v in cond_paths.items()}
    dfs.update(cdfs)
    plot_gfn_energies_violins(dfs, output_path, target=args.target)
    plt.close("all")
