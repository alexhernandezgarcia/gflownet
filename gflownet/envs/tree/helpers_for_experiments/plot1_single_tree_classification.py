"""
Figure 1 - single-tree classification: test accuracy vs. tree size.
Good region = top-left (accurate AND small).  Run: python plot1_single_tree_classification.py
"""

import matplotlib.pyplot as plt
import numpy as np

import gflownet.envs.tree.helpers_for_experiments.plot_utils as pu

# =============================================================================
# DATA - paste the rows from Overleaf (or everything from \toprule to \bottomrule)
# =============================================================================
LATEX = r"""
\textsc{Bcart-Smc}
 & 0.9518\,\pmm{0.02} & 16.18\,\pmm{1.72}
 & 0.9311\,\pmm{0.04} & 16.25\,\pmm{2.66}
 & 0.9310\,\pmm{0.01}  & 32.32\,\pmm{2.68}
 & 0.8660\,\pmm{0.01}  & 46.58\,\pmm{2.12}
 & \best{0.8466}\,\pmm{0.005} & 231.00\,\pmm{20.00} & 0.7707\,\pmm{0.003} & 67.40\,\pmm{7.10} \\
\textsc{Bcart-Mcmc}
 & 0.9230\,\pmm{0.04}  & 13.40\,\pmm{1.50}
 & \best{0.9550}\,\pmm{0.02} & 13.82\,\pmm{1.21}
 & 0.9200\,\pmm{0.02}   & 25.62\,\pmm{2.67}
 & 0.8640\,\pmm{0.02}  & 35.29\,\pmm{1.93}
 & 0.8447\,\pmm{0.007} & 339.40\,\pmm{30.50} & 0.7684\,\pmm{0.008} & 128.2\,\pmm{17.30} \\
\textsc{Maptree}
 & 0.8733\,\pmm{0.04} & \best{3.80}\,\pmm{0.45}
 & 0.9139\,\pmm{0.02} & \best{4.80}\,\pmm{0.45}
 & 0.9281\,\pmm{0.02} & \best{5.00}\,\pmm{0.00}
 & 0.8344\,\pmm{0.03} & 7.80\,\pmm{1.10}
 & 0.8038\,\pmm{0.001} & \best{20.60}\,\pmm{2.00} & 0.7499\,\pmm{0.005} & \best{20.20}\,\pmm{1.60} \\
\textsc{BCART}
 & 0.9267\,\pmm{0.03} & 56.20\,\pmm{31.80}
 & 0.9389\,\pmm{0.02} & 49.80\,\pmm{34.26}
 & 0.9018\,\pmm{0.03} & 20.60\,\pmm{12.09}
 & 0.8678\,\pmm{0.01} & 23.80\,\pmm{8.26}
 & 0.8425\,\pmm{0.005} & 325.40\,\pmm{29.10} & 0.7711\,\pmm{0.006} & 117.4\,\pmm{15.70} \\
\midrule
\textsc{CART-Gini}
 & 0.9494\,\pmm{0.02} & 14.60\,\pmm{1.50}
 & 0.8760\,\pmm{0.05}  & 17.80\,\pmm{4.12}
 & 0.9230\,\pmm{0.02}  & 34.60\,\pmm{6.25}
 & 0.8520\,\pmm{0.023} & 29.80\,\pmm{0.98}
 & 0.8265\,\pmm{0.006} & 59.80\,\pmm{1.00} & 0.7651\,\pmm{0.005} & 62.6\,\pmm{0.80} \\
\textsc{CART-Entropy}
 & 0.9468\,\pmm{0.02} & 14.60\,\pmm{1.50}
 & 0.9357\,\pmm{0.04} & 16.60\,\pmm{3.20}
 & 0.9168\,\pmm{0.022}& 29.40\,\pmm{1.96}
 & \best{0.8680}\,\pmm{0.016} & 27.4\,\pmm{0.08}
 & 0.8211\,\pmm{0.004} & 57.00\,\pmm{2.50} & 0.7651\,\pmm{0.005} & 62.20\,\pmm{1.60} \\
\midrule
\textsc{DT-GFN MLP} (ours)
 & \best{0.9800}\,\pmm{0.030} & 7.00\,\pmm{0.00} & 0.9333$^\dagger$ \,\pmm{0.042} & 9.40 \,\pmm{1.67} & \best{0.9439}\,\pmm{0.019} & 8.20\,\pmm{2.28} & 0.8467\,\pmm{0.026} & \best{6.20}\,\pmm{1.79} & 0.8210$^\dagger$ \,\pmm{0.005} & 59.80\,\pmm{2.68} & \best{0.7717}$^\dagger$ \,\pmm{0.012} & 24.60 \,\pmm{3.34} \\
"""
DATASETS = [
    "Iris",
    "Wine",
    "Breast Cancer",
    "Raisin",
    "Magic",
    "Credit",
]  # table column order
METRICS = ["acc", "size"]  # per dataset
EXCLUDE = ["DT-GFN Transformer"]

# =============================================================================
# CONFIG
# =============================================================================
PANEL_ORDER = ["Iris", "Wine", "Breast Cancer", "Raisin", "Credit", "Magic"]
DATASET_LABELS = {}  # e.g. {"Breast Cancer": "Breast cancer"}
NROWS, NCOLS = 2, 3
FIGSIZE = (15, 9)
TITLE = "Single trees (classification): test accuracy vs. tree size"
TITLE_SIZE = 22
# XLABEL = "Tree size (# nodes, log)  \u2190 better"
XLABEL = "Tree size"
# YLABEL = "Test accuracy  \u2191 better"
YLABEL = "Test accuracy"
LOG_X = True
SHOW_ERRORBARS = True

LEGEND = dict(
    show=True,
    entries=None,  # e.g. ["DT-GFN", "MAPTree", "BCART-SMC", ...]
    labels={"DT-GFN": "DT-GFN (ours)"},  # rename entries
    anchor="figure",  # "figure" or axes index (0, 1, ...)
    loc="outside lower center",  # e.g. "outside right upper", or "lower right" + anchor=0
    ncol=7,
    fontsize=14,
    frameon=False,
    markerscale=1.0,
    columnspacing=1.2,
    handletextpad=0.3,
)
OUTPUT_NAME = "fig1_single_tree_classification"
OUTPUT_DIR = "figures"


def main():
    pu.setup_style()
    df = pu.parse_latex_table(LATEX, DATASETS, METRICS, exclude=EXCLUDE)
    acc, acc_sd = pu.wide(df, "acc"), pu.wide(df, "acc", "std")
    size, size_sd = pu.wide(df, "size"), pu.wide(df, "size", "std")

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE, layout="constrained")
    axes = np.ravel(axes)
    handles = {}
    for ax, ds in zip(axes, PANEL_ORDER):
        for m in acc.index:
            x, y = size.at[m, ds], acc.at[m, ds]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            pu.plot_point(
                ax,
                x,
                y,
                m,
                xerr=size_sd.at[m, ds] if SHOW_ERRORBARS else None,
                yerr=acc_sd.at[m, ds] if SHOW_ERRORBARS else None,
                log_x=LOG_X,
            )
            handles.setdefault(m, pu.marker_handle(m))
        ax.set_title(DATASET_LABELS.get(ds, ds))
        if LOG_X:
            ax.set_xscale("log")
            pu.format_log_axis(ax, "x")
    for ax in axes[len(PANEL_ORDER) :]:
        ax.set_visible(False)

    # axis labels only on the outer panels (keeps the slide clean)
    for i, ax in enumerate(axes[: len(PANEL_ORDER)]):
        if i % NCOLS == 0:
            ax.set_ylabel(YLABEL)
        if i >= len(PANEL_ORDER) - NCOLS:
            ax.set_xlabel(XLABEL)
    fig.suptitle(TITLE, fontsize=TITLE_SIZE, fontweight="bold")
    pu.add_legend(fig, handles, LEGEND, axes)
    pu.save_figure(fig, OUTPUT_NAME, OUTPUT_DIR)


if __name__ == "__main__":
    main()
