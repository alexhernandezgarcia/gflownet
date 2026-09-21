"""
Figure 2 - ensemble classification: "ladder" of methods per dataset, sorted by
test accuracy (best at the top, higher = better). Colour + marker encode the
method family: blue = greedy / bagging / boosting ensembles, grey = deep
learning.  Run: python plot2_ensemble_classification_ladder.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import gflownet.envs.tree.helpers_for_experiments.plot_utils as pu

# =============================================================================
# DATA - paste the rows from Overleaf (or everything from \toprule to \bottomrule)
# =============================================================================
LATEX = r"""
\textsc{DT-GFN MLP} (ours)
 & \best{0.9667}\,\pmm{0.03} & \best{0.9944}$^\dagger$ \,\pmm{0.012} & 0.9544\,\pmm{0.011} & 0.8600 \,\pmm{0.019} & 0.8245$^\dagger$\,\pmm{0.006} & 0.7687$^\dagger$\,\pmm{0.002} \\
\midrule
 \textsc{Greedy RF}
 & 0.9600\,\pmm{0.01} & 0.9833\,\pmm{0.01} & 0.9474\,\pmm{0.02}
 & 0.8689\,\pmm{0.01} & 0.8422\,\pmm{0.005} & 0.7758\,\pmm{0.004} \\
\textsc{XGBoost}
 & 0.9533\,\pmm{0.03} & 0.9556\,\pmm{0.03} & 0.9579\,\pmm{0.02}
 & 0.8611\,\pmm{0.02} & 0.8840\,\pmm{0.004} & \best{0.7832}\,\pmm{0.008} \\
\textsc{CatBoost}
 & 0.9600\,\pmm{0.03} & 0.978\,\pmm{0.01} & 0.9544\,\pmm{0.01}
 & \best{0.8800}\,\pmm{0.02} & \best{0.8884}\,\pmm{0.005} & 0.7819\,\pmm{0.008} \\
\textsc{LightGBM}
 & 0.9533\,\pmm{0.03} & 0.9889\,\pmm{0.01} & 0.9579\,\pmm{0.01}
 & 0.8622\,\pmm{0.02} & 0.8847\,\pmm{0.003} & 0.7821\,\pmm{0.009} \\
\midrule
\textsc{MLP}
 & \best{0.9667}\,\pmm{0.03} & 0.8500\,\pmm{0.015} & 0.9140\,\pmm{0.01}
 & 0.8544\,\pmm{0.03} & 0.846 \cite{dataset_jannis2} &  0.76 \cite{dataset_jannis2} \\
% \textsc{TabTransformer}
%  & 0.7933\,\pmm{0.049} & 0.978\,\pmm{0.021} & 0.9316\,\pmm{.0211}
%  & 0.8544\,\pmm{0.03} & -- & -- \\
\textsc{FTTransformer}
 & 0.9530\,\pmm{0.016} & 0.9670\,\pmm{0.02} & \best{0.9610}\,\pmm{0.013}
 & 0.8470\,\pmm{0.028} & 0.853 \cite{dataset_jannis2} & 0.76 \cite{dataset_jannis2} \\
"""
DATASETS = [
    "Iris",
    "Wine",
    "Breast Cancer",
    "Raisin",
    "Magic",
    "Credit",
]  # table column order
METRICS = ["acc"]  # per dataset
EXCLUDE = ["DT-GFN Transformer"]  # only the MLP variant of DT-GFN is shown

# =============================================================================
# CONFIG
# =============================================================================
# Panel order (left -> right, top -> bottom); here sorted by dataset size.
PANEL_ORDER = ["Iris", "Wine", "Breast Cancer", "Raisin", "Credit", "Magic"]
DATASET_LABELS = {}  # e.g. {"Iris": "Iris (n=150)"}
SHOW_ERRORBARS = True
SHOW_VALUES = True  # write "0.967" next to each point
VALUE_DECIMALS = 3
VALUE_FONTSIZE = 12
HIGHLIGHT_OURS_TICKLABEL = True
# Values without std (MLP / FT-Transformer on Magic & Credit, taken from
# another paper) are drawn as hollow markers. Set False to draw them filled.
HOLLOW_IF_NO_STD = True
NROWS, NCOLS = 2, 3
FIGSIZE = (19, 10)
TITLE = "Ensembles (classification): test accuracy"
TITLE_SIZE = 22
XLABEL = "Test accuracy  (→ better)"

# Legend entries are method *families* (see plot_utils.FAMILY_STYLES),
# plus "external" for the hollow markers.
LEGEND = dict(
    show=True,
    entries=["ours", "ensemble", "deep", "external"],
    labels={
        "ours": "DT-GFN (ours)",
        "ensemble": "Greedy RF / boosting",
        "deep": "Deep learning (MLP / FT-Transformer)",
        "external": "reported value, no std",
    },
    anchor="figure",
    loc="outside lower center",
    ncol=4,
    fontsize=14,
    frameon=False,
)
OUTPUT_NAME = "fig2_ensemble_classification_ladder"
OUTPUT_DIR = "figures"


def main():
    pu.setup_style()
    df = pu.parse_latex_table(LATEX, DATASETS, METRICS, exclude=EXCLUDE)
    acc, acc_sd = pu.wide(df, "acc"), pu.wide(df, "acc", "std")

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE, layout="constrained")
    axes = np.ravel(axes)
    handles = {}
    for i, (ax, ds) in enumerate(zip(axes, PANEL_ORDER)):
        vals = (
            acc[ds].dropna().sort_values(ascending=False, kind="stable")
        )  # best first
        sds = acc_sd.loc[vals.index, ds]
        n = len(vals)
        for rank, (m, v) in enumerate(vals.items()):
            y = n - 1 - rank  # best at the top
            st = pu.get_family_style(m)
            sd = sds[m] if SHOW_ERRORBARS else None
            if HOLLOW_IF_NO_STD and not np.isfinite(sds[m]):
                ax.plot(
                    [v],
                    [y],
                    linestyle="none",
                    marker=st["marker"],
                    markersize=st["size"],
                    markerfacecolor="white",
                    markeredgecolor=st["color"],
                    markeredgewidth=2.2,
                    zorder=st.get("zorder", 3),
                )
                handles.setdefault(
                    "external",
                    Line2D(
                        [],
                        [],
                        linestyle="none",
                        marker="o",
                        markersize=10,
                        markerfacecolor="white",
                        markeredgecolor=pu.GREY_DARK,
                        markeredgewidth=2.2,
                    ),
                )
            else:
                pu.plot_point(ax, v, y, m, xerr=sd, style=st)
            if SHOW_VALUES:
                right = v + (sd if sd is not None and np.isfinite(sd) else 0)
                ax.annotate(
                    f"{v:.{VALUE_DECIMALS}f}",
                    (right, y),
                    xytext=(7, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=VALUE_FONTSIZE,
                    color="0.3",
                )
            handles.setdefault(pu.get_style(m)["family"], pu.marker_handle(style=st))

        ax.set_yticks(range(n)[::-1], list(vals.index))
        ax.set_ylim(-0.7, n - 0.3)
        if HIGHLIGHT_OURS_TICKLABEL:
            for lbl in ax.get_yticklabels():
                if lbl.get_text() == pu.OURS:
                    lbl.set_color(pu.OURS_COLOR)
                    lbl.set_fontweight("bold")
        # room on the right for the value labels
        pad = sds.fillna(0)
        hi, lo = max(vals + pad), min(vals - pad)
        ax.set_xlim(lo - 0.12 * (hi - lo), hi + 0.3 * (hi - lo))
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.grid(axis="y", visible=False)
        if i >= len(PANEL_ORDER) - NCOLS:
            ax.set_xlabel(XLABEL)
    for ax in axes[len(PANEL_ORDER) :]:
        ax.set_visible(False)

    fig.suptitle(TITLE, fontsize=TITLE_SIZE, fontweight="bold")
    pu.add_legend(fig, handles, LEGEND, axes)
    pu.save_figure(fig, OUTPUT_NAME, OUTPUT_DIR)


if __name__ == "__main__":
    main()
