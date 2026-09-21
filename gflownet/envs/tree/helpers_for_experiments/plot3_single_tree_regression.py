"""
Figure 3 - single-predictor regression: test RMSE vs. tree size for the tree
methods, with non-tree baselines (GP) as horizontal reference lines.
Good region = bottom-left.  Run: python plot3_single_tree_regression.py
"""

import matplotlib.pyplot as plt
import numpy as np

import gflownet.envs.tree.helpers_for_experiments.plot_utils as pu

# =============================================================================
# DATA - paste the rows from Overleaf (or everything from \toprule to \bottomrule)
# =============================================================================
LATEX = r"""
\toprule
\textbf{Dataset $\rightarrow$}
 & \multicolumn{3}{c}{\texttt{diabetes}}
 & \multicolumn{3}{c}{\texttt{energy}}
 & \multicolumn{3}{c}{\texttt{yacht}}
 & \multicolumn{3}{c}{\texttt{real\_estate}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}
\textbf{Algorithm $\downarrow$}
 & \textsc{RMSE}$\downarrow$ & $R^2\uparrow$ & \textsc{Size}$\downarrow$
 & \textsc{RMSE}$\downarrow$ & $R^2\uparrow$ & \textsc{Size}$\downarrow$
 & \textsc{RMSE}$\downarrow$ & $R^2\uparrow$ & \textsc{Size}$\downarrow$
 & \textsc{RMSE}$\downarrow$ & $R^2\uparrow$ & \textsc{Size}$\downarrow$ \\
\midrule
\textsc{Bcart-Map}
 & 62.0168\,\pmm{2.9271} & 0.3090\,\pmm{0.0859} & 12.20\,\pmm{2.00}
 & 2.2821\,\pmm{0.2183}  & 0.9468\,\pmm{0.0102} & 22.20\,\pmm{2.00}
 & 3.2447\,\pmm{1.0006}  & 0.9484\,\pmm{0.0300} & 10.60\,\pmm{2.00}
 & 10.2789\,\pmm{1.9864} & 0.3853\,\pmm{0.3097} & 22.60\,\pmm{2.70} \\
\textsc{Bcart-Smc}
 & 61.9107\,\pmm{2.1791} & 0.3128\,\pmm{0.0641} & \best{10.60}\,\pmm{1.50}
 & 2.1863\,\pmm{0.1143}  & 0.9514\,\pmm{0.0057} & \best{21.00}\,\pmm{1.30}
 & 2.5017\,\pmm{0.1887}  & 0.9709\,\pmm{0.0050} & \best{9.00}\,\pmm{0.00}
 & 10.7230\,\pmm{1.5021} & 0.3267\,\pmm{0.2971} & \best{20.60}\,\pmm{3.20} \\ 
\textsc{Cart}
 & 67.2230\,\pmm{3.9163} & 0.1875\,\pmm{0.1148} & 58.60\,\pmm{2.90}
 & 1.0324\,\pmm{0.0658}  & 0.9892\,\pmm{0.0015} & 63.00\,\pmm{0.00}
 & 1.3235\,\pmm{0.2170}  & 0.9918\,\pmm{0.0022} & 61.00\,\pmm{0.00}
 & 8.1787\,\pmm{0.8498}  & 0.6340\,\pmm{0.0476} & 57.40\,\pmm{2.00} \\
\midrule
\textsc{Gp}
 & 55.0236\,\pmm{1.5120} & 0.4563\,\pmm{0.0560} & \na
 & \best{0.4778}\,\pmm{0.0140} & \best{0.9977}\,\pmm{0.0002} & \na
 & \best{0.3180}\,\pmm{0.0887} & \best{0.9995}\,\pmm{0.0003} & \na
 & \best{7.5404}\,\pmm{1.2016} & \best{0.6905}\,\pmm{0.0586} & \na \\
\textsc{Linear}
 & 55.0754\,\pmm{0.7382} & 0.4566\,\pmm{0.0381} & \na
 & 3.0185\,\pmm{0.1438}  & 0.9075\,\pmm{0.0101} & \na
 & 8.9354\,\pmm{0.1779}  & 0.6320\,\pmm{0.0199} & \na
 & 8.6765\,\pmm{1.3669}  & 0.5927\,\pmm{0.0632} & \na \\
\textsc{Ridge}
 & 55.0805\,\pmm{0.7229} & 0.4564\,\pmm{0.0384} & \na
 & 3.0182\,\pmm{0.1440}  & 0.9075\,\pmm{0.0101} & \na
 & 8.9028\,\pmm{0.1984}  & 0.6347\,\pmm{0.0193} & \na
 & 8.6563\,\pmm{1.4006}  & 0.5947\,\pmm{0.0654} & \na \\
\textsc{Lasso}
 & 55.2954\,\pmm{0.7826} & 0.4526\,\pmm{0.0325} & \na
 & 3.0202\,\pmm{0.1456}  & 0.9074\,\pmm{0.0101} & \na
 & 8.7865\,\pmm{0.2561}  & 0.6443\,\pmm{0.0184} & \na
 & 8.6518\,\pmm{1.4107}  & 0.5953\,\pmm{0.0652} & \na \\
\textsc{Bayesian Ridge}
 & \best{55.0116}\,\pmm{0.7379} & \best{0.4579}\,\pmm{0.0370} & \na
 & 3.0177\,\pmm{0.1444}  & 0.9075\,\pmm{0.0101} & \na
 & 8.9101\,\pmm{0.1896}  & 0.6341\,\pmm{0.0199} & \na
 & 8.6591\,\pmm{1.3938}  & 0.5944\,\pmm{0.0650} & \na \\
\midrule
\textsc{DT-GFN} MLP (ours)
 & 59.9101$^\dagger$\,\pmm{2.147} & 0.3373\,\pmm{0.052} & 19.33\,\pmm{4.46}
 & 1.1108$^\dagger$\,\pmm{0.068}  & 0.9875\,\pmm{0.002} & 53.40\,\pmm{6.841}
 & 1.6219\,\pmm{0.190}  & 0.9878\,\pmm{0.003} & 20.20\,\pmm{1.10}
 & 8.1124$^\dagger$\,\pmm{1.465}  & 0.6400\,\pmm{0.088} & 36.00\,\pmm{1.41} \\
"""
DATASETS = ["diabetes", "energy", "yacht", "real_estate"]  # table column order
METRICS = ["rmse", "r2", "size"]  # per dataset
EXCLUDE = []

# =============================================================================
# CONFIG
# =============================================================================
PANEL_ORDER = ["diabetes", "energy", "yacht", "real_estate"]
DATASET_LABELS = {
    "diabetes": "Diabetes",
    "energy": "Energy",
    "yacht": "Yacht",
    "real_estate": "Real estate",
}
# Methods drawn as horizontal lines (no tree size). Add e.g. "Bayesian Ridge".
REFERENCE_METHODS = ["GP"]
SHOW_REFERENCE_BAND = True  # shaded +-1 std around each reference line
LABEL_REFERENCE_LINES = True  # write "GP" directly on the line
REFERENCE_LABEL_X = 0.02  # position along the line (axes fraction, 0 = left)
NROWS, NCOLS = 1, 4  # e.g. 2, 2 for a squarer figure
FIGSIZE = (18, 5.8)
TITLE = "Single trees (regression): test RMSE vs. tree size"
TITLE_SIZE = 22
XLABEL = "Tree size"
YLABEL = "Test RMSE"
LOG_X = False
SHOW_ERRORBARS = True

LEGEND = dict(
    show=True,
    entries=None,  # e.g. ["DT-GFN", "BCART-SMC", "BCART-MAP", "CART", "GP"]
    labels={"DT-GFN": "DT-GFN (ours)", "GP": "GP (no tree)"},
    anchor="figure",
    loc="outside lower center",
    ncol=5,
    fontsize=14,
    frameon=False,
)
OUTPUT_NAME = "fig3_single_tree_regression"
OUTPUT_DIR = "figures"


def main():
    pu.setup_style()
    df = pu.parse_latex_table(LATEX, DATASETS, METRICS, exclude=EXCLUDE)
    rmse, rmse_sd = pu.wide(df, "rmse"), pu.wide(df, "rmse", "std")
    size, size_sd = pu.wide(df, "size"), pu.wide(df, "size", "std")

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE, layout="constrained")
    axes = np.ravel(axes)
    handles = {}
    for i, (ax, ds) in enumerate(zip(axes, PANEL_ORDER)):
        # tree methods = everything with a finite size
        for m in rmse.index:
            if m in REFERENCE_METHODS or not np.isfinite(size.at[m, ds]):
                continue
            pu.plot_point(
                ax,
                size.at[m, ds],
                rmse.at[m, ds],
                m,
                xerr=size_sd.at[m, ds] if SHOW_ERRORBARS else None,
                yerr=rmse_sd.at[m, ds] if SHOW_ERRORBARS else None,
                log_x=LOG_X,
            )
            handles.setdefault(m, pu.marker_handle(m))
        # reference lines
        for ref in REFERENCE_METHODS:
            st, y, sd = pu.get_style(ref), rmse.at[ref, ds], rmse_sd.at[ref, ds]
            ax.axhline(
                y, color=st["color"], linestyle=st["linestyle"], lw=2.2, zorder=2
            )
            if SHOW_REFERENCE_BAND and np.isfinite(sd):
                ax.axhspan(
                    y - sd, y + sd, color=st["color"], alpha=0.10, lw=0, zorder=1
                )
            if LABEL_REFERENCE_LINES:
                ax.text(
                    REFERENCE_LABEL_X,
                    y,
                    ref,
                    transform=ax.get_yaxis_transform(),
                    ha="left" if REFERENCE_LABEL_X < 0.5 else "right",
                    va="bottom",
                    color=st["color"],
                    fontsize=13,
                    fontweight="bold",
                )
            handles.setdefault(ref, pu.line_handle(ref))

        ax.set_title(DATASET_LABELS.get(ds, ds))
        if LOG_X:
            ax.set_xscale("log")
            pu.format_log_axis(ax, "x")
        if i % NCOLS == 0:
            ax.set_ylabel(YLABEL)
        if i >= len(PANEL_ORDER) - NCOLS:
            ax.set_xlabel(XLABEL)
    for ax in axes[len(PANEL_ORDER) :]:
        ax.set_visible(False)

    fig.suptitle(TITLE, fontsize=TITLE_SIZE, fontweight="bold")
    pu.add_legend(fig, handles, LEGEND, axes)
    pu.save_figure(fig, OUTPUT_NAME, OUTPUT_DIR)


if __name__ == "__main__":
    main()
