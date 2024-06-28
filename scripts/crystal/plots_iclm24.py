import argparse
import datetime
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from gflownet.utils.common import load_gflow_net_from_run_path
from gflownet.utils.crystals.constants import ELEMENT_NAMES
from mendeleev.fetch import fetch_table
from tqdm import tqdm

warnings.filterwarnings("ignore")


sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("colorblind", 4))

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

torch.set_grad_enabled(False)

ELS_TABLE = fetch_table("elements")["symbol"]
ROOT = Path(__file__).resolve().parent.parent.parent
USE_SUPTITLES = True
SUPTITLES = {
    "eform": "Formation Energy (↓)",
    "bandgap": "Bang Gap target (1.34 eV)",
    "density": "Density (↑)",
}
CRYSTALGFNs = {
    "eform": "Crystal-GFN (FE)",
    "bandgap": "Crystal-GFN (BG)",
    "density": "Crystal-GFN (De)",
}

sys.path.append(str(ROOT))


def load_mb_eform(energy_key="energy"):
    """Load the materials project eform dataset and returns train and val df."""
    train_df_path = "/network/scratch/s/schmidtv/crystals-proxys/data/materials_dataset_v3/data/matbench_mp_e_form/train_data.csv"
    val_df_path = "/network/scratch/s/schmidtv/crystals-proxys/data/materials_dataset_v3/data/matbench_mp_e_form/val_data.csv"
    tdf = pd.read_csv(train_df_path)
    vdf = pd.read_csv(val_df_path)
    tdf[energy_key] = tdf["Eform"]
    vdf[energy_key] = vdf["Eform"]
    tdf = tdf.drop(columns=["Eform", "cif"])
    vdf = vdf.drop(columns=["Eform", "cif"])
    return tdf, vdf


def load_mb_bandgap(energy_key="energy"):
    """
    Load the materials project bandgap dataset and returns train and val df.
    """
    train_df_path = "/network/scratch/s/schmidtv/crystals-proxys/data/materials_dataset_v3/data/matbench_mp_gap/train_data.csv"
    val_df_path = "/network/scratch/s/schmidtv/crystals-proxys/data/materials_dataset_v3/data/matbench_mp_gap/val_data.csv"
    tdf = pd.read_csv(train_df_path)
    vdf = pd.read_csv(val_df_path)
    tdf[energy_key] = tdf["Band Gap"]
    vdf[energy_key] = vdf["Band Gap"]
    tdf = tdf.drop(columns=["Band Gap", "cif"])
    vdf = vdf.drop(columns=["Band Gap", "cif"])
    return tdf, vdf


def now_to_str():
    """
    Get the current date and time in a string format.
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d/%H-%M-%S")


def plot_sg_dist(
    tdf, vdf, sdf, udf=None, sg_key="Space Group", output_path=None, target=None
):
    """
    Plot the space group distribution comparing train and val splits from the data set and the samples from the gfn.
    If uniform samples with rewards are provided, a second plot with the reward distribution is plotted as well.

    Args:
        tdf (pd.DataFrame): Train data frame.
        vdf (pd.DataFrame): Validation data frame.
        sdf (pd.DataFrame): Sample data frame.
        udf (pd.DataFrame, optional): Uniform samples data frame. Should contain 'reward' column with values of the rewards for each sample. Default is None.
        sg_key (str): Space group key.
        output_path (str, optional): Path to save the plot. Default is None.
        target (str, optional): Name of the target property during training. Default is None.
    """
    fig_paths = []
    sdf_sg_counts = sdf[sg_key].value_counts().sort_index()
    sdf_sg_counts = sdf_sg_counts / sdf_sg_counts.sum()
    valdf_sg_counts = vdf[sg_key].value_counts().sort_index()
    valdf_sg_counts = valdf_sg_counts / valdf_sg_counts.sum()
    traindf_sg_counts = tdf[sg_key].value_counts().sort_index()
    traindf_sg_counts = traindf_sg_counts / traindf_sg_counts.sum()

    sg_counts = pd.concat(
        [sdf_sg_counts, valdf_sg_counts, traindf_sg_counts], axis=1
    ).sort_index(ascending=False)
    sg_counts.columns = [CRYSTALGFNs[target], "MatBench Val", "MatBench Train"]
    sg_counts = sg_counts.fillna(0)
    figsize = (13, 20 / 113 * len(sg_counts))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    sg_counts.plot.barh(ax=ax)
    ax.set_xlabel("Frequency", fontsize=20)
    ax.set_ylabel("Space group", fontsize=20)
    fig.tight_layout()
    ax.grid(False)
    ax.legend(fontsize=20, loc="lower right")
    sns.despine()
    if USE_SUPTITLES:
        plt.suptitle(
            SUPTITLES[target],
            fontsize=20,
        )
    if output_path is not None:
        name = f"{target}_sg_counts.pdf"
        fig_paths.append(output_path / name)
        fig.savefig(fig_paths[-1], bbox_inches="tight")
        fig_paths.append(output_path / name.replace(".pdf", ".png"))
        fig.savefig(fig_paths[-1], bbox_inches="tight")
        print(f"\n ✅ Saved {output_path / name} and .png\n")
    plt.close()

    if udf is not None:
        uni_sg_counts = udf.groupby(sg_key)["reward"].sum()
        uni_sg_counts = uni_sg_counts / uni_sg_counts.sum()

        sg_counts = pd.concat(
            [sdf_sg_counts, valdf_sg_counts, traindf_sg_counts, uni_sg_counts], axis=1
        ).sort_index(ascending=False)
        sg_counts.columns = [
            CRYSTALGFNs[target],
            "MatBench Val",
            "MatBench Train",
            "Rewards",
        ]
        sg_counts = sg_counts.fillna(0)
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        sg_counts.plot.barh(ax=ax)
        ax.set_xlabel("Frequency", fontsize=20)
        ax.set_ylabel("Space group", fontsize=20)
        fig.tight_layout()
        ax.grid(False)
        ax.legend(fontsize=20, loc="lower right")
        sns.despine()
        if USE_SUPTITLES:
            plt.suptitle(
                SUPTITLES[target],
                fontsize=20,
            )
        if output_path is not None:
            name = f"{target}_sg_counts_rewards.pdf"
            fig_paths.append(output_path / name)
            fig.savefig(fig_paths[-1], bbox_inches="tight")
            fig_paths.append(output_path / name.replace(".pdf", ".png"))
            fig.savefig(fig_paths[-1], bbox_inches="tight")
            print(f"\n ✅ Saved {output_path / name} and .png\n")

    return fig_paths


def plot_el_counts(tdf, vdf, sdf, els_cols, output_path=None, target=None):
    """
    Plot the element counts distribution (in a soup) comparing train and val splits from the data set and the samples from the gfn.

    Args:
        tdf (pd.DataFrame): Train data frame.
        vdf (pd.DataFrame): Validation data frame.
        sdf (pd.DataFrame): Sample data frame.
        els_cols (list): Lis of element names corresponding to columns in the data frames.
        output_path (str, optional): Path to save the plot. Default is None.
        target (str, optional): Name of the target property during training. Default is None.
    """
    fig_paths = []
    sdf_el_counts = sdf[els_cols].sum(axis=0)
    sdf_el_counts = sdf_el_counts / sdf_el_counts.sum()
    valdf_el_counts = vdf[els_cols].sum(axis=0)
    valdf_el_counts = valdf_el_counts / valdf_el_counts.sum()
    traindf_el_counts = tdf[els_cols].sum(axis=0)
    traindf_el_counts = traindf_el_counts / traindf_el_counts.sum()

    el_counts = pd.concat([sdf_el_counts, valdf_el_counts, traindf_el_counts], axis=1)
    el_counts.columns = [CRYSTALGFNs[target], "MatBench Val", "MatBench Train"]
    el_counts = el_counts.fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    el_counts.plot.bar(ax=ax)
    ax.legend(fontsize=14, loc="upper right")
    ax.set_ylabel("Frequency [counts]", fontsize=16)
    ax.set_xlabel("Element", fontsize=16)
    ax.set_xticklabels(el_counts.index, rotation=360, fontsize=14)
    ax.set_yticklabels([f"{x:.2f}" for x in ax.get_yticks()], fontsize=14)
    fig.tight_layout()
    ax.grid(False)
    sns.despine()
    if USE_SUPTITLES:
        plt.suptitle(
            SUPTITLES[target],
            fontsize=20,
        )
    if output_path is not None:
        name = f"{target}_el_counts.pdf"
        fig_paths.append(output_path / name)
        fig.savefig(fig_paths[-1], bbox_inches="tight")
        fig_paths.append(output_path / name.replace(".pdf", ".png"))
        fig.savefig(fig_paths[-1], bbox_inches="tight")
        print(f"\n ✅ Saved {output_path / name} and .png\n")

    return fig_paths


def plot_els_bin_count(
    tdf, vdf, sdf, els_cols, udf=None, output_path=None, target=None
):
    """
    Plot the binary element counts distribution comparing train and val splits from the data set and the samples from the gfn.
    If uniform samples with rewards are provided, a second plot with the reward distribution is plotted as well.
    The plotted frequencies correspond to the probability to encounter an element in the sample form a data set / model / reward.

    Args:
        tdf (pd.DataFrame): Train data frame.
        vdf (pd.DataFrame): Validation data frame.
        sdf (pd.DataFrame): Sample data frame.
        els_cols (list): List of element names corresponding to columns in the data frames.
        udf (pd.DataFrame, optional): Uniform samples data frame. Should contain 'reward' column with values of the rewards for each sample. Default is None
        output_path (str, optional): Path to save the plot. Default is None.
        target (str, optional): Name of the target property during training. Default is None.
    """
    fig_paths = []

    sdf_el_counts = (sdf[els_cols] > 0).sum(axis=0)
    sdf_el_counts = sdf_el_counts / len(sdf)
    valdf_el_counts = (vdf[els_cols] > 0).sum(axis=0)
    valdf_el_counts = valdf_el_counts / len(vdf)
    traindf_el_counts = (tdf[els_cols] > 0).sum(axis=0)
    traindf_el_counts = traindf_el_counts / len(tdf)

    el_counts = pd.concat([sdf_el_counts, valdf_el_counts, traindf_el_counts], axis=1)
    el_counts.columns = [CRYSTALGFNs[target], "MatBench Val", "MatBench Train"]
    el_counts = el_counts.fillna(0)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    el_counts.plot.bar(ax=ax)
    ax.set_ylabel("Frequency [0/1 presence]", fontsize=16)
    ax.set_xlabel("Element", fontsize=16)
    ax.legend(fontsize=14, loc="upper right")
    ax.set_xticklabels(el_counts.index, rotation=360, fontsize=14)
    ax.set_yticklabels([f"{x:.2f}" for x in ax.get_yticks()], fontsize=14)
    fig.tight_layout()
    ax.grid(False)
    sns.despine()
    if USE_SUPTITLES:
        plt.suptitle(
            SUPTITLES[target],
            fontsize=20,
        )
    if output_path is not None:
        name = f"{target}_el_counts_binary.pdf"
        fig_paths.append(output_path / name)
        fig.savefig(fig_paths[-1], bbox_inches="tight")
        fig_paths.append(output_path / name.replace(".pdf", ".png"))
        fig.savefig(fig_paths[-1], bbox_inches="tight")
        print(f"\n ✅ Saved {output_path / name} and .png\n")

    if udf is not None:
        udf_el_counts = (
            (udf[els_cols] > 0).values.astype(np.float64)
            * udf["reward"].values[:, np.newaxis]
        ).sum(axis=0)
        udf_el_counts = udf_el_counts / udf["reward"].values.sum()
        udf_el_counts = pd.Series(data=udf_el_counts, index=udf[els_cols].columns)

        el_counts = pd.concat(
            [sdf_el_counts, valdf_el_counts, traindf_el_counts, udf_el_counts], axis=1
        )
        el_counts.columns = [
            CRYSTALGFNs[target],
            "MatBench Val",
            "MatBench Train",
            "Reward",
        ]
        el_counts = el_counts.fillna(0)

        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        el_counts.plot.bar(ax=ax)
        ax.set_ylabel("Frequency [0/1 presence]", fontsize=16)
        ax.set_xlabel("Element", fontsize=16)
        ax.legend(fontsize=14, loc="upper right")
        ax.set_xticklabels(el_counts.index, rotation=360, fontsize=14)
        ax.set_yticklabels([f"{x:.2f}" for x in ax.get_yticks()], fontsize=14)
        fig.tight_layout()
        ax.grid(False)
        sns.despine()
        if USE_SUPTITLES:
            plt.suptitle(SUPTITLES[target], fontsize=20)
        if output_path is not None:
            name = f"{target}_el_counts_binary_reward.pdf"
            fig_paths.append(output_path / name)
            fig.savefig(fig_paths[-1], bbox_inches="tight")
            fig_paths.append(output_path / name.replace(".pdf", ".png"))
            fig.savefig(fig_paths[-1], bbox_inches="tight")
            print(f"\n ✅ Saved {output_path / name} and .png\n")

    return fig_paths


def plot_lat_params(
    tdf,
    vdf,
    sdf,
    output_path=None,
    target=None,
    min_length=0.9,
    max_length=100,
    min_angle=50,
    max_angle=150,
):
    """
    Plot normalized lattice parameters (violin plots) for comparing train and val splits from the data set and the samples from the gfn.

    Args:
        tdf (pd.DataFrame): Train data frame.
        vdf (pd.DataFrame): Validation data frame.
        sdf (pd.DataFrame): Sample data frame.
        output_path (str, optional): Path to save the plot. Default is None.
        target (str, optional): Name of the target property during training. Default is None.
        min_length (float, optional): Minimum length value for normalization. Default is 0.9.
        max_length (float, optional): Maximum length value for normalization. Default is 100.
        min_angle (float, optional): Minimum angle value for normalization. Default is 50.
        max_angle (float, optional): Maximum angle value for normalization. Default is 150.
    """
    fig_paths = []

    lattice_params = ["a", "b", "c", "alpha", "beta", "gamma"]
    lps_min = np.array(
        [min_length, min_length, min_length, min_angle, min_angle, min_angle]
    )
    lps_max = np.array(
        [max_length, max_length, max_length, max_angle, max_angle, max_angle]
    )

    sdf_lps = sdf[lattice_params]
    vdf_lps = vdf[lattice_params]
    tdf_lps = tdf[lattice_params]

    if any(sdf_lps.max() > 1):
        sdf_lps = (sdf_lps - lps_min) / (lps_max - lps_min)
    if any(vdf_lps.max() > 1):
        vdf_lps = (vdf_lps - lps_min) / (lps_max - lps_min)
    if any(tdf_lps.max() > 1):
        tdf_lps = (tdf_lps - lps_min) / (lps_max - lps_min)

    lps_normed = pd.concat([sdf_lps, vdf_lps, tdf_lps])
    lps_normed.columns = [f"{c} (normalized)" for c in lps_normed.columns]
    lps_normed["source"] = sum(
        [
            [s] * c
            for s, c in zip(
                [CRYSTALGFNs[target], "MatBench Val", "MatBench Train"],
                [len(sdf), len(vdf), len(tdf)],
            )
        ],
        [],
    )

    lps_normed = lps_normed.loc[lps_normed["c (normalized)"] <= 1]

    fig, axes = plt.subplots(
        3, 2, figsize=(13, 20), sharey=False, sharex=True, constrained_layout=True
    )

    for i, p in enumerate(lattice_params):
        ax_i = [0, 2, 4, 1, 3, 5][i]
        ax = axes.flatten()[ax_i]
        sns.violinplot(
            data=lps_normed,
            x="source",
            y=f"{p} (normalized)",
            ax=ax,
            cut=0,
            inner="quartile",
            hue="source",
        )
        ax.set_ylabel(f"{p} (normalized)", fontdict={"size": 20})
        ax.set_xlabel("")
        # denormalize y ticks
        yticks = ax.get_yticks()
        denorm_yticks = (
            [f"{denorm_length(y, min_length, max_length):.2f}" for y in yticks]
            if i < 3
            else [f"{denorm_angle(y, min_angle, max_angle):.0f}" for y in yticks]
        )
        ax.set_yticklabels(denorm_yticks)
        if i >= 2:
            ax.set_xticklabels(
                [CRYSTALGFNs[target], "MatBench Val", "MatBench Train"],
                fontdict={"size": 17},
            )
        ax.grid(False)
    fig.tight_layout()
    sns.despine()
    if USE_SUPTITLES:
        plt.suptitle(
            SUPTITLES[target],
            fontsize=20,
            y=1.01,
        )
    if output_path is not None:
        name = f"{target}_lattice_params.pdf"
        fig_paths.append(output_path / name)
        fig.savefig(fig_paths[-1], bbox_inches="tight")
        fig_paths.append(output_path / name.replace(".pdf", ".png"))
        fig.savefig(fig_paths[-1], bbox_inches="tight")
        print(f"\n ✅ Saved {output_path / name} and .png\n")

    return fig_paths


def plot_lat_params_reward(
    tdf,
    vdf,
    sdf,
    udf,
    output_path=None,
    target=None,
    min_length=0.9,
    max_length=100,
    min_angle=50,
    max_angle=150,
):
    """
    Plot normalized lattice parameters (histograms) for comparing train and val splits from the data set, the samples
    from the gfn, and reward distribution (using weighted uniform samples).

    Args:
        tdf (pd.DataFrame): Train data frame.
        vdf (pd.DataFrame): Validation data frame.
        sdf (pd.DataFrame): Sample data frame.
        udf (pd.DataFrame): Uniform samples data frame. Should contain 'reward' column with values of the rewards for each sample.
        output_path (str, optional): Path to save the plot. Default is None.
        target (str, optional): Name of the target property during training. Default is None.
        min_length (float, optional): Minimum length value for normalization. Default is 0.9.
        max_length (float, optional): Maximum length value for normalization. Default is 100.
        min_angle (float, optional): Minimum angle value for normalization. Default is 50.
        max_angle (float, optional): Maximum angle value for normalization. Default is 150.
    """
    fig_paths = []
    lattice_params = ["a", "b", "c", "alpha", "beta", "gamma"]
    lps_min = np.array(
        [min_length, min_length, min_length, min_angle, min_angle, min_angle]
    )
    lps_max = np.array(
        [max_length, max_length, max_length, max_angle, max_angle, max_angle]
    )

    sdf_lps = sdf[lattice_params]
    vdf_lps = vdf[lattice_params]
    tdf_lps = tdf[lattice_params]
    udf_lps = udf[lattice_params]

    if any(sdf_lps.max() > 1):
        sdf_lps = (sdf_lps - lps_min) / (lps_max - lps_min)
    if any(vdf_lps.max() > 1):
        vdf_lps = (vdf_lps - lps_min) / (lps_max - lps_min)
    if any(tdf_lps.max() > 1):
        tdf_lps = (tdf_lps - lps_min) / (lps_max - lps_min)
    if any(udf_lps.max() > 1):
        udf_lps = (udf_lps - lps_min) / (lps_max - lps_min)

    udf_lps["reward"] = udf["reward"]

    fig, axes = plt.subplots(
        3, 2, figsize=(13, 15), sharey=False, sharex=False, constrained_layout=True
    )
    for i, p in enumerate(lattice_params):
        ax_i = [0, 2, 4, 1, 3, 5][i]
        ax = axes.flatten()[ax_i]
        bins = np.linspace(0.0, 1.0, 50)
        sns.histplot(
            data=sdf_lps,
            x=p,
            stat="density",
            kde=True,
            bins=bins,
            label=CRYSTALGFNs[target],
            ax=ax,
        )
        sns.histplot(
            data=vdf_lps,
            x=p,
            stat="density",
            kde=True,
            bins=bins,
            label="MatBench Val",
            ax=ax,
        )
        sns.histplot(
            data=tdf_lps,
            x=p,
            stat="density",
            kde=True,
            bins=bins,
            label="MatBench Train",
            ax=ax,
        )
        sns.histplot(
            data=udf_lps,
            x=p,
            stat="density",
            weights="reward",
            kde=True,
            bins=bins.tolist(),
            label="Reward",
            ax=ax,
        )
        ax.set_xlabel(f"{p} (normalized)", fontdict={"size": 20})
        ax.legend()
        # denormalize x ticks
        xticks = ax.get_xticks()
        denorm_xticks = (
            [f"{denorm_length(y, min_length, max_length):.2f}" for y in xticks]
            if i < 3
            else [f"{denorm_angle(y, min_angle, max_angle):.0f}" for y in xticks]
        )
        ax.set_xticklabels(denorm_xticks)
        ax.set_ylabel("Density", fontdict={"size": 20})

    fig.tight_layout()
    sns.despine()

    if USE_SUPTITLES:
        plt.suptitle(
            SUPTITLES[target],
            fontsize=20,
        )
    if output_path is not None:
        name = f"{target}_lattice_params_reward.pdf"
        fig_paths.append(output_path / name)
        fig.savefig(fig_paths[-1], bbox_inches="tight")
        fig_paths.append(output_path / name.replace(".pdf", ".png"))
        fig.savefig(fig_paths[-1], bbox_inches="tight")
        print(f"\n ✅ Saved {output_path / name} and .png\n")

    return fig_paths


def plot_gfn_energies_violins(
    vdf, sdf, rdf, output_path=None, target=None, energy_key="energy"
):
    """
    Plot energy distributions (violin plots) comparing val split from the data set, the samples from a trained gfn model
    and samples from a randomly initialised model.

    Args:
        vdf (pd.DataFrame): Validation data frame.
        sdf (pd.DataFrame): Sample data frame.
        rdf (pd.DataFrame): Random init sample data frame
        output_path (str, optional): Path to save the plot. Default is None.
        target (str, optional): Name of the target property during training. Default is None.
        energy_key (str, optional): Name of the column in the data frames containing energy values. Default is 'energy'
    """
    fig_paths = []
    # TODO update for density target
    # plt.rcParams.update({"font.family": "serif"})
    palette = sns.color_palette("colorblind", 4)[:2] + [
        sns.color_palette("colorblind", 5)[4]
    ]
    sdf["source"] = CRYSTALGFNs[target]
    vdf["source"] = "Validation set"
    rdf["source"] = f"{CRYSTALGFNs[target]} random init."

    df_plot = pd.concat([sdf, vdf, rdf])

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150, constrained_layout=True)
    ax = sns.violinplot(
        ax=ax,
        data=df_plot,
        y=energy_key,
        hue="source",
        palette=palette,
        hue_order=[
            CRYSTALGFNs[target],
            "Validation set",
            f"{CRYSTALGFNs[target]} random init.",
        ],
        cut=0,
    )
    # Manually set y-lims
    ax.set_ylim([-4.5, 6.0] if target == "eform" else [-0.1, 7.0])
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

    if output_path is not None:
        name = f"{target}_energies_violins.pdf"
        fig_paths.append(output_path / name)
        fig.savefig(fig_paths[-1], bbox_inches="tight")
        fig_paths.append(output_path / name.replace(".pdf", ".png"))
        fig.savefig(fig_paths[-1], bbox_inches="tight")
        print(f"\n ✅ Saved {output_path / name} and .png\n")

    return fig_paths


def plot_topk_elements(
    sdf, els_cols, output_path=None, target=None, k=100, energy_key="energy"
):
    # sdf_topk = sdf.sort_values(by=energy_key)[:k]
    sdf_topk = sdf.sort_values(by=energy_key, ascending=target == "eform")[:k]

    sdf_el_counts = (sdf_topk[els_cols] > 0).sum(axis=0)
    sdf_el_counts = sdf_el_counts / len(sdf_topk)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    sdf_el_counts.plot.bar(ax=ax)
    ax.set_ylabel(f"Frequency [0/1 presence] in top-{k}", fontsize=16)
    # ax.set_xlabel("Element", fontsize=16)
    # ax.legend(fontsize=14, loc="upper right")
    ax.set_xticklabels(sdf_el_counts.index, rotation=360, fontsize=14)
    ax.set_yticklabels([f"{x:.2f}" for x in ax.get_yticks()], fontsize=14)
    fig.tight_layout()
    ax.grid(False)
    sns.despine()

    if USE_SUPTITLES:
        plt.suptitle(
            SUPTITLES[target],
            fontsize=20,
        )
    fig_paths = []
    if output_path is not None:
        name = f"{target}_elem_bin_counts_top{k}.pdf"
        fig_paths.append(output_path / name)
        fig.savefig(fig_paths[-1], bbox_inches="tight")
        fig_paths.append(output_path / name.replace(".pdf", ".png"))
        fig.savefig(fig_paths[-1], bbox_inches="tight")
        print(f"\n ✅ Saved {output_path / name} and .png\n")
    return fig_paths


def sort_names_for_z(element_names):
    return sorted(element_names, key=lambda x: ELS_TABLE.tolist().index(x))


def pkl_samples_to_df(
    samples, elements_names, sg_key="Space Group", energy_key="energy"
):
    """
    Convert samples from a pickled file to a DataFrame.

    Args:
        samples (dict): Dictionary containing sampled data.
        elements_names (list): List of names of the elements present in the samples.
        sg_key (str, optional): Key to represent the space group in the DataFrame. Default is "Space Group".
        energy_key (str, optional): Key to represent the energy values in the DataFrame. Default is "energy".

    Returns:
        pandas.DataFrame: DataFrame containing the converted samples.
            Columns:
                - idx: Index of the sample.
                - Space Group: Space group of the sample.
                - a: Lattice parameter 'a' of the sample.
                - b: Lattice parameter 'b' of the sample.
                - c: Lattice parameter 'c' of the sample.
                - alpha: Lattice parameter 'alpha' of the sample.
                - beta: Lattice parameter 'beta' of the sample.
                - gamma: Lattice parameter 'gamma' of the sample.
                - element1, element2, ...: Composition of elements in the sample.
                - energy_key: Energy value associated with the sample.
    """
    df = []

    for i, (x, e) in tqdm(
        enumerate(zip(samples["x"], samples[energy_key])), total=len(samples["x"])
    ):
        _, (_, _, sg), comp, (a, b, c, alpha, beta, gamma) = x
        s = {
            "idx": i,
            "Space Group": sg,
            "a": a,
            "b": b,
            "c": c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            energy_key: e,
        }
        for k, v in comp.items():
            el_name = ELEMENT_NAMES[k]
            s[el_name] = int(v)
        df.append(s)
    df = pd.DataFrame(df)
    cols = (
        ["idx", sg_key, "a", "b", "c", "alpha", "beta", "gamma"]
        + sort_names_for_z(elements_names)
        + [energy_key]
    )
    df = df[cols]
    # set zeros for elements that are not present in the samples
    df[elements_names] = df[elements_names].fillna(0)
    return df


def load_gfn_samples(element_names, pkl_path):
    """
    Load samples from pickled data and convert them to a DataFrame.

    Args:
        element_names (list): List of names of the elements present in the samples.
        pkl_path (str): Path to the pickled data. Default is None.

    Returns:
        pandas.DataFrame: DataFrame containing the loaded and converted samples.
    """
    with open(pkl_path, "rb") as f:
        samples = pickle.load(f)
    return pkl_samples_to_df(samples, element_names)


def load_uniform_samples(element_names, pkl_path, config):
    """
    Load uniform samples from pickled data, compute rewards based on the energy values in the pkl file

    Args:
        element_names (list): List of names of the elements present in the samples.
        pkl_path (str): Path to the pickled data.
        config (dict): Config parameters used during training of the model. Needed to extract the right limits for uniform sampling

    Returns:
        pandas.DataFrame: DataFrame containing the uniform samples woth corresponding rewards.
    """
    df = load_gfn_samples(element_names, pkl_path=pkl_path)

    reward_func = config["env"]["reward_func"]

    def proxy2reward(energies, reward_func, reward_beta=8.0, min_reward=1e-8):
        if reward_func == "boltzmann":
            reward_beta = config["env"]["reward_beta"]
            rewards = np.clip(np.exp(-reward_beta * energies), min_reward, None)
        elif reward_func == "identity":
            rewards = np.clip(-energies, min_reward, None)
        else:
            raise NotImplementedError(f"Unknown reward_func {reward_func}")
        return rewards

    print(f"!!! Using {reward_func} refard function for converting energies to rewards")
    df["reward"] = proxy2reward(df["energy"].values, reward_func=reward_func)
    df_clean = df[df["reward"].notna()]
    if len(df) > len(df_clean):
        print(f"Dropped {len(df) - len(df_clean)} uniform samples with nan rewards")
    return df_clean


def denorm_length(length, min_length=0.9, max_length=100):
    return length * (max_length - min_length) + min_length


def denorm_angle(angle, min_angle=50, max_angle=150):
    return angle * (max_angle - min_angle) + min_angle


def norm_length(length, min_length=0.9, max_length=100):
    return (length - min_length) / (max_length - min_length)


def norm_angle(angle, min_angle=50, max_angle=150):
    return (angle - min_angle) / (max_angle - min_angle)


def filter_df(
    df,
    el_names,
    comp_cols,
    space_groups_subset,
    min_length=0.9,
    max_length=100,
    min_angle=50,
    max_angle=150,
    sg_key="Space Group",
):
    """
    Filter the dataframe based on the length and angle.

    Args:
        df (pd.DataFrame): Dataframe.
        el_names (list): List of element names we want to keep.
        comp_cols (list): List of all composition columns.
        space_group_subset (list): List of space groups we want to keep
        min_length (float): Minimum length.
        max_length (float): Maximum length.
        min_angle (float): Minimum angle.
        max_angle (float): Maximum angle.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    df = df[
        (df["a"] > min_length)
        & (df["a"] <= max_length)
        & (df["b"] >= min_length)
        & (df["b"] <= max_length)
        & (df["c"] >= min_length)
        & (df["c"] <= max_length)
        & (df["alpha"] >= min_angle)
        & (df["alpha"] <= max_angle)
        & (df["beta"] >= min_angle)
        & (df["beta"] <= max_angle)
        & (df["gamma"] >= min_angle)
        & (df["gamma"] <= max_angle)
    ]
    el_names_set = set(el_names)
    other_els = [c for c in comp_cols if c not in el_names_set]
    if any(c in df.columns for c in other_els):
        df = df[df[other_els].sum(1) == 0]
        df = df.drop(columns=[c for c in comp_cols if c not in el_names])

    sg_set = set(space_groups_subset)
    df = df[df[sg_key].apply(lambda x: x in sg_set)]
    return df


def make_plots(
    *,
    train_df=None,
    val_df=None,
    samples_df=None,
    uniform_df=None,
    random_df=None,
    samples_els=None,
    output_path=None,
    target=None,
    min_length=None,
    max_length=None,
    min_angle=None,
    max_angle=None,
    format="png",
    energy_key="energy",
    k=100,
):
    """
    Make all the plots for Crystal GFN:
    - Space group distribution
    - Element counts
    - Element binary counts
    - Lattice parameters
    - Lattice parameters reward
    - GFN energies violins

    Args:
        train_df (pd.DataFrame): Data frame of train samples. Mandatory.
        val_df (pd.DataFrame): Data frame of validation samples. Mandatory.
        target (str, optional): Either eform or bandgap, i.e. quantity of interest.
            Mandatory.
        min_length (float, optional): Minimum lattice parameter length. Mandatory.
        max_length (float, optional): Maximum lattice parameter length. Mandatory.
        min_angle (float, optional): Minimum lattice parameter angle. Mandatory.
        max_angle (float, optional): Maximum lattice parameter angle. Mandatory.
        output_path (str | Path, optional): Where to save the plots. Mandatory.
        samples_df (pd.DataFrame, optional): Data frame of samples from the gfn.
        uniform_df (pd.DataFrame, optional): Data frame of uniform samples.
        random_df (pd.DataFrame, optional): Data frame of random (untrained GFN)
            samples.
        samples_els (List[str], optional): List of element names present in the samples.
        format (str, optional): Plots are always saved in pdf and png. This controles
            which one to return for the logger. Default is "png".

    Returns:
        List[str]: List of paths to the saved figures.
    """
    fig_paths = []

    print("Plotting space group distribution")
    fig_paths += plot_sg_dist(
        train_df, val_df, samples_df, uniform_df, output_path=output_path, target=target
    )
    fig_paths += plot_el_counts(
        train_df,
        val_df,
        samples_df,
        samples_els,
        output_path=output_path,
        target=target,
    )
    fig_paths += plot_els_bin_count(
        train_df,
        val_df,
        samples_df,
        samples_els,
        uniform_df,
        output_path=output_path,
        target=target,
    )
    fig_paths += plot_lat_params(
        train_df,
        val_df,
        samples_df,
        output_path=output_path,
        target=target,
        min_length=min_length,
        max_length=max_length,
        min_angle=min_angle,
        max_angle=max_angle,
    )
    # don't plot this for bandgap as it is not clear how to choose topk
    if target in ["eform", "density"]:
        fig_paths += plot_topk_elements(
            samples_df,
            samples_els,
            output_path=output_path,
            target=target,
            k=k,
            energy_key=energy_key,
        )

    if uniform_df is not None:
        fig_paths += plot_lat_params_reward(
            train_df,
            val_df,
            samples_df,
            uniform_df,
            output_path=output_path,
            target=target,
            min_length=min_length,
            max_length=max_length,
            min_angle=min_angle,
            max_angle=max_angle,
        )

    # TODO: remove second condition once we have dataset with computed densities
    if random_df is not None and target in ["eform", "bandgap"]:
        fig_paths += plot_gfn_energies_violins(
            val_df,
            samples_df,
            random_df,
            output_path=output_path,
            target=target,
            energy_key=energy_key,
        )

    plt.close("all")
    return [str(p) for p in fig_paths if p.suffix == f".{format}" and p.exists()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl_path", type=str, default=None, help="gflownet samples path"
    )
    parser.add_argument(
        "--random_pkl_path",
        type=str,
        default=None,
        help="random (init only) gflownet samples path",
    )
    parser.add_argument(
        "--uniform_pkl_path", type=str, default=None, help="uniform samples path"
    )
    # target: either eform or bandgap:
    parser.add_argument(
        "--target", type=str, default="eform", choices=["eform", "bandgap", "density"]
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--sg_key", type=str, default="Space Group")
    parser.add_argument("--energy_key", type=str, default="energy")
    parser.add_argument("--min_length", type=float, default=0.9)
    parser.add_argument("--max_length", type=float, default=100)
    parser.add_argument("--min_angle", type=float, default=50)
    parser.add_argument("--max_angle", type=float, default=150)
    parser.add_argument(
        "--no_suptitles", action="store_true", default=False, help="Prevent suptitles"
    )
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()

    tdf = rdf = vdf = udf = None

    USE_SUPTITLES = not args.no_suptitles

    print("Arguments:")
    print("\n".join(f"{k:15}: {v}" for k, v in vars(args).items()))

    now = now_to_str()
    output_path = ROOT / "external" / "plots" / "icml24" / now
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to {output_path}")

    if args.target == "eform":
        ftdf, fvdf = load_mb_eform(energy_key=args.energy_key)
        config_path = ROOT / "config/experiments/crystals/starling_fe.yaml"
    elif args.target == "bandgap":
        ftdf, fvdf = load_mb_bandgap(energy_key=args.energy_key)
        config_path = ROOT / "config/experiments/crystals/starling_bg.yaml"
    elif args.target == "density":
        # TODO: incorrect "energies" here, need to change it once we have
        # datasets with computed densities
        ftdf, fvdf = load_mb_eform(energy_key=args.energy_key)
        config_path = ROOT / "config/experiments/crystals/starling_density.yaml"
    else:
        raise ValueError("Unknown target")

    print("Initial full data sets:")
    print(f"Train: {ftdf.shape}")
    print(f"Val: {fvdf.shape}")

    config = yaml.safe_load(open(config_path, "r"))

    # List atomic numbers of the utilised elements
    elements_anums = config["env"]["composition_kwargs"]["elements"]
    elements_names = [ELEMENT_NAMES[anum] for anum in elements_anums]

    sdf = load_gfn_samples(elements_names, pkl_path=args.pkl_path)
    print("Loaded gfn samples: ", sdf.shape)

    if args.uniform_pkl_path:
        udf = load_uniform_samples(
            elements_names, pkl_path=args.uniform_pkl_path, config=config
        )
        print("Loaded uniform samples: ", udf.shape)
    if args.random_pkl_path or args.random_gfn_path:
        rdf = load_gfn_samples(  # random init
            elements_names, pkl_path=args.random_pkl_path
        )
        print("Loaded random samples: ", rdf.shape)

    print("Using elements: ", ", ".join(elements_names))

    sg_subset = config["env"]["space_group_kwargs"]["space_groups_subset"]
    assert len(sg_subset) > 0
    print(f"Using {len(sg_subset)} SGs: ", ", ".join(map(str, sg_subset)))

    comp_cols = [c for c in ftdf.columns if c in set(ELS_TABLE)]

    tdf = filter_df(
        ftdf,
        elements_names,
        comp_cols,
        sg_subset,
        min_length=args.min_length,
        max_length=args.max_length,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
    )

    vdf = filter_df(
        fvdf,
        elements_names,
        comp_cols,
        sg_subset,
        min_length=args.min_length,
        max_length=args.max_length,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
    )

    print("Filtered data sets:")
    print(f"Train: {tdf.shape}")
    print(f"Val: {vdf.shape}")
    print()

    make_plots(
        train_df=tdf,
        val_df=vdf,
        samples_df=sdf,
        uniform_df=udf,
        random_df=rdf,
        samples_els=elements_names,
        output_path=output_path,
        target=args.target,
        min_length=args.min_length,
        max_length=args.max_length,
        min_angle=args.min_angle,
        max_angle=args.max_angle,
        energy_key=args.energy_key,
        k=args.k,
    )
