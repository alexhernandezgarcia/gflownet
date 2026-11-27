"""
Figure 4 - ensemble regression: "ladder" of methods per dataset, sorted by RMSE
relative to the best method (1.0 = best). Colour + marker encode the method
family.  Run: python plot4_ensemble_regression_ladder.py
"""

import matplotlib.pyplot as plt
import numpy as np

import gflownet.envs.tree.helpers_for_experiments.plot_utils as pu

# =============================================================================
# DATA - paste the rows from Overleaf (or everything from \toprule to \bottomrule)
# =============================================================================
LATEX = r"""
\textsc{Bcart-Mcmc}
 & 59.8897\,\pmm{3.2257} & 0.3557\,\pmm{0.0819} & 2.0802\,\pmm{0.0816}  & 0.9561\,\pmm{0.0046} & 2.7006\,\pmm{0.4578}  & 0.9655\,\pmm{0.0123} & 9.5385\,\pmm{1.9927}  & 0.4722\,\pmm{0.2707} \\
\textsc{Bcart-Smc}
 & 61.3698\,\pmm{1.7590} & 0.3250\,\pmm{0.0571} & 2.1176\,\pmm{0.1030}  & 0.9544\,\pmm{0.0053} & 2.5011\,\pmm{0.1887}  & 0.9709\,\pmm{0.0050} & 9.0814\,\pmm{1.1622}  & 0.5467\,\pmm{0.0944} \\
\midrule
\textsc{Bart}
 & \best{55.2774}\,\pmm{1.3308} & \best{0.4516}\,\pmm{0.0514}
 & 0.6799\,\pmm{0.0737}  & 0.9953\,\pmm{0.0010}
 & 0.9734\,\pmm{0.0752}  & 0.9956\,\pmm{0.0007}
 & \best{7.4229}\,\pmm{1.1650} & \best{0.6999}\,\pmm{0.0569} \\
\textsc{Random Forest}
 & 57.4787\,\pmm{2.0910} & 0.4061\,\pmm{0.0695}
 & 0.9326\,\pmm{0.0899}  & 0.9912\,\pmm{0.0015}
 & 1.0565\,\pmm{0.1998}  & 0.9948\,\pmm{0.0016}
 & 7.5674\,\pmm{1.1755}  & 0.6846\,\pmm{0.0732} \\
\textsc{XGBoost}
 & 59.0426\,\pmm{2.3819} & 0.3723\,\pmm{0.0796}
 & \best{0.3520}\,\pmm{0.0179} & \best{0.9987}\,\pmm{0.0001}
 & \best{0.6122}\,\pmm{0.1990} & \best{0.9981}\,\pmm{0.0011}
 & 7.6664\,\pmm{1.2151}  & 0.6766\,\pmm{0.0778} \\
\textsc{LightGBM}
 & 56.5115\,\pmm{2.6734} & 0.4240\,\pmm{0.0801}
 & 0.4170\,\pmm{0.0423}  & 0.9982\,\pmm{0.0005}
 & 3.6946\,\pmm{0.3051}  & 0.9369\,\pmm{0.0089}
 & 7.5065\,\pmm{1.1799}  & 0.6927\,\pmm{0.0617} \\
\midrule
\textsc{DT-GFN} MLP (ours)
 & 57.8544$^\dagger$\,\pmm{1.741} & 0.3822\,\pmm{0.041}
 & 1.2540$^\dagger$\,\pmm{0.178}  & 0.9839\,\pmm{0.004}
 & 1.5921\,\pmm{0.222}  & 0.9881\,\pmm{0.003}
 & 7.8865$^\dagger$\,\pmm{1.335}  & 0.6620\,\pmm{0.064} \\
"""
DATASETS = ["diabetes", "energy", "yacht", "real_estate"]  # table column order
METRICS = ["rmse", "r2"]  # per dataset
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
NORMALIZE = True  # True: RMSE / best RMSE per dataset; False: raw RMSE
LOG_X = "auto"  # True / False / "auto" (= log only if a panel spans > LOG_AUTO_RATIO)
LOG_AUTO_RATIO = 2.0
SHOW_ERRORBARS = True
SHOW_VALUES = True  # write "1.23x" next to each point
HIGHLIGHT_OURS_TICKLABEL = True
NROWS, NCOLS = 2, 2
FIGSIZE = (14, 9.5)
TITLE = "Ensembles (regression): RMSE relative to the best method"
TITLE_SIZE = 22
XLABEL = (
    "RMSE / best RMSE  (1 = best, \u2190 better)"
    if NORMALIZE
    else "Test RMSE  (\u2190 better)"
)

# Legend entries are method *families* (see plot_utils.FAMILY_STYLES).
LEGEND = dict(
    show=True,
    entries=["ours", "ensemble", "bcart"],
    labels={
        "ours": "DT-GFN (ours)",
        "ensemble": "BART / Random Forest / boosting",
        "bcart": "BCART samplers (MCMC / SMC)",
    },
    anchor="figure",
    loc="outside lower center",
    ncol=3,
    fontsize=14,
    frameon=False,
)
OUTPUT_NAME = "fig4_ensemble_regression_ladder"
OUTPUT_DIR = "figures"


def main():
    pu.setup_style()
    df = pu.parse_latex_table(LATEX, DATASETS, METRICS, exclude=EXCLUDE)
    rmse, rmse_sd = pu.wide(df, "rmse"), pu.wide(df, "rmse", "std")

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE, layout="constrained")
    axes = np.ravel(axes)
    handles = {}
    for i, (ax, ds) in enumerate(zip(axes, PANEL_ORDER)):
        vals, sds = rmse[ds].dropna(), rmse_sd[ds]
        denom = vals.min() if NORMALIZE else 1.0
        vals = (vals / denom).sort_values()  # best first
        n = len(vals)
        log_x = (vals.max() / vals.min() > LOG_AUTO_RATIO) if LOG_X == "auto" else LOG_X
        for rank, (m, v) in enumerate(vals.items()):
            y = n - 1 - rank  # best at the top
            st = pu.get_family_style(m)
            sd = sds[m] / denom if SHOW_ERRORBARS else None
            pu.plot_point(ax, v, y, m, xerr=sd, log_x=log_x, style=st)
            if SHOW_VALUES:
                right = v + (sd if sd is not None and np.isfinite(sd) else 0)
                ax.annotate(
                    f"{v:.2f}\u00d7" if NORMALIZE else f"{v:.3g}",
                    (right, y),
                    xytext=(7, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=12,
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
        if NORMALIZE:
            ax.axvline(1.0, color="0.3", linestyle=":", lw=1.3, zorder=1)
        if log_x:
            ax.set_xscale("log")
            pu.format_log_axis(ax, "x", subs=(1.0, 1.5, 2.0, 3.0, 5.0))
        # room on the right for the value labels
        hi = max(vals + (sds[vals.index] / denom).fillna(0))
        lo_ = min(vals - (sds[vals.index] / denom).fillna(0))
        ax.set_xlim(right=hi * 1.6 if log_x else hi + 0.22 * (hi - lo_))
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
