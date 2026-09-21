"""
plot_utils.py - shared helpers for the DT-GFN result figures.

Contents
--------
* parse_latex_table : Overleaf table rows  ->  tidy pandas DataFrame
* STYLES / FAMILY_STYLES : colour + marker per method (Okabe-Ito, colour-blind safe)
* plot_point, marker_handle, line_handle, family_handle : drawing helpers
* add_legend : fully configurable legend (content, size, position)
* setup_style, format_log_axis, save_figure

Edit STYLES / DISPLAY_NAMES here to change how a method looks in *all* figures.
Requires matplotlib >= 3.7 (for legend locations like "outside lower center").
"""

from __future__ import annotations

import re
import warnings
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

# -----------------------------------------------------------------------------
# Colours: Okabe & Ito palette, safe for all common forms of colour blindness.
# Rule used throughout: DT-GFN = vermillion, key competitor(s) = blue,
# everything else = greys. Markers differ per method, so colour is never the
# only cue.
# -----------------------------------------------------------------------------
OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}
OURS_COLOR = OKABE_ITO["vermillion"]
ACCENT_COLOR = OKABE_ITO["blue"]
GREY_DARK, GREY_MID, GREY_LIGHT = "#3F3F3F", "#7F7F7F", "#ABABAB"

# Name of our method *after* DISPLAY_NAMES has been applied.
OURS = "DT-GFN"

# Parsed LaTeX name (after stripping \textsc etc.)  ->  name shown in figures.
DISPLAY_NAMES = {
    "DT-GFN MLP": "DT-GFN",
    "DT-GFN Transformer": "DT-GFN Transformer",
    "Bcart-Smc": "BCART-SMC",
    "Bcart-Mcmc": "BCART-MCMC",
    "Bcart-Map": "BCART-MAP",
    "Maptree": "MAPTree",
    "Cart": "CART",
    "Gp": "GP",
    "Bart": "BART",
    "FTTransformer": "FT-Transformer",
}


def _s(color, marker, size=10, family="other", linestyle="--", zorder=3):
    return dict(
        color=color,
        marker=marker,
        size=size,
        family=family,
        linestyle=linestyle,
        zorder=zorder,
    )


# Per-method look (keys are display names). `linestyle` is used when the method
# is drawn as a horizontal reference line (e.g. GP in the regression plot).
STYLES = {
    OURS: _s(OURS_COLOR, "*", 19, "ours", "-", 10),
    # --- single trees ---------------------------------------------------------
    "MAPTree": _s(ACCENT_COLOR, "D", 10, "bcart", zorder=5),
    "BCART-SMC": _s(GREY_MID, "o", family="bcart"),
    "BCART-MCMC": _s(GREY_MID, "s", family="bcart"),
    "BCART": _s(GREY_MID, "^", family="bcart"),
    "BCART-MAP": _s(GREY_MID, "v", family="bcart"),
    "CART-Gini": _s(GREY_DARK, "P", 11, "cart"),
    "CART-Entropy": _s(GREY_DARK, "X", 11, "cart"),
    "CART": _s(GREY_DARK, "P", 11, "cart"),
    # --- non-tree regression baselines ----------------------------------------
    "GP": _s(ACCENT_COLOR, "D", family="gp", linestyle="--"),
    "Linear": _s(GREY_DARK, "h", family="linear", linestyle=":"),
    "Ridge": _s(GREY_DARK, "p", family="linear", linestyle=":"),
    "Lasso": _s(GREY_DARK, "8", family="linear", linestyle=":"),
    "Bayesian Ridge": _s(GREY_DARK, "H", family="linear", linestyle=":"),
    # --- ensembles --------------------------------------------------------------
    "BART": _s(ACCENT_COLOR, "D", family="ensemble"),
    "Random Forest": _s(ACCENT_COLOR, "s", family="ensemble"),
    "Greedy RF": _s(ACCENT_COLOR, "s", family="ensemble"),
    "XGBoost": _s(ACCENT_COLOR, "^", family="ensemble"),
    "CatBoost": _s(ACCENT_COLOR, "v", family="ensemble"),
    "LightGBM": _s(ACCENT_COLOR, "<", family="ensemble"),
    "MLP": _s(GREY_DARK, "o", family="deep"),
    "FT-Transformer": _s(GREY_DARK, "s", family="deep"),
}

# Look per method *family* (used by the ladder plot, where colour+marker encode
# the family and the y-tick labels name the individual methods).
FAMILY_STYLES = {
    "ours": dict(color=OURS_COLOR, marker="*", size=19, zorder=10),
    "bcart": dict(color=GREY_MID, marker="o", size=11, zorder=3),
    "ensemble": dict(color=ACCENT_COLOR, marker="s", size=10, zorder=4),
    "cart": dict(color=GREY_DARK, marker="P", size=11, zorder=3),
    "gp": dict(color=ACCENT_COLOR, marker="D", size=10, zorder=4),
    "linear": dict(color=GREY_DARK, marker="h", size=10, zorder=3),
    "deep": dict(color=GREY_DARK, marker="o", size=10, zorder=3),
    "other": dict(color=GREY_LIGHT, marker="o", size=10, zorder=2),
}

_FALLBACK_MARKERS = cycle(["o", "s", "^", "v", "<", ">", "p", "h"])


def get_style(name: str) -> dict:
    """Style of a method; unknown methods get a grey fallback (with a warning)."""
    if name not in STYLES:
        warnings.warn(
            f"No style defined for '{name}', using a grey fallback. "
            f"Add it to plot_utils.STYLES."
        )
        STYLES[name] = _s(GREY_LIGHT, next(_FALLBACK_MARKERS))
    return STYLES[name]


def get_family_style(name: str) -> dict:
    return FAMILY_STYLES.get(get_style(name)["family"], FAMILY_STYLES["other"])


# -----------------------------------------------------------------------------
# LaTeX table parsing
# -----------------------------------------------------------------------------
_RULES = re.compile(r"\\(?:toprule|midrule|bottomrule|hline)(?:\[[^\]]*\])?")
_CMIDRULE = re.compile(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}")
_CMD_WITH_ARG = re.compile(r"\\[a-zA-Z]+\*?\{([^{}]*)\}")
_CITE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{[^}]*\}")
_MATH = re.compile(r"\$[^$]*\$")
_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _unwrap(s: str) -> str:
    """\\best{0.98} -> 0.98, \\textsc{\\textbf{X}} -> X (repeatedly)."""
    prev = None
    while prev != s:
        prev, s = s, _CMD_WITH_ARG.sub(r"\1", s)
    return s


def _clean_name(cell: str) -> str:
    s = _MATH.sub("", cell).replace("(ours)", "")
    s = _unwrap(s)
    s = re.sub(r"\\[a-zA-Z]+", "", s).replace("{", "").replace("}", "")
    return " ".join(s.split())


def _parse_cell(cell: str):
    """'\\best{0.98}$^\\dagger$\\,\\pmm{0.03}' -> (0.98, 0.03, True)."""
    dagger = "dagger" in cell
    s = _MATH.sub("", _CITE.sub("", cell))
    if re.search(r"\\na\b", s) or s.strip() in {"", "-", "--", "---"}:
        return np.nan, np.nan, dagger
    std = np.nan
    m = re.search(r"\\pmm?\s*\{([^}]*)\}", s) or re.search(r"\\pm\s*([-+]?[\d.]+)", s)
    if m:
        std = float(_NUMBER.search(m.group(1)).group())
        s = s[: m.start()] + s[m.end() :]
    s = re.sub(r"\\[a-zA-Z]+", " ", _unwrap(s))
    num = _NUMBER.search(s)
    return (float(num.group()) if num else np.nan), std, dagger


def parse_latex_table(
    latex: str, datasets: list[str], metrics: list[str], exclude=(), rename: bool = True
) -> pd.DataFrame:
    """Parse rows copied from Overleaf into a tidy DataFrame.

    Paste either just the data rows or the whole block from \\toprule to
    \\bottomrule; header rows, rules and %-comments are skipped automatically.

    datasets : dataset names in the column order of the table
    metrics  : metric names per dataset, in column order (e.g. ["acc", "size"])
    exclude  : method names (raw or display name) to drop, e.g. the transformer

    Returns columns: method, dataset, metric, mean, std, dagger
    """
    text = _strip_comments(latex)
    if r"\toprule" in text:
        text = text.split(r"\toprule", 1)[1]
    if r"\bottomrule" in text:
        text = text.split(r"\bottomrule", 1)[0]
    text = _RULES.sub(" ", _CMIDRULE.sub(" ", text))

    n_expected = len(datasets) * len(metrics)
    records = []
    for row in re.split(r"\\\\", text):
        row = re.sub(r"^\s*\[[^\]]*\]", "", row)  # leftover of '\\[3pt]'
        if "&" not in row:
            continue
        cells = row.split("&")
        raw_name = _clean_name(cells[0])
        if not raw_name or re.search(r"dataset|algorithm", raw_name, re.I):
            continue  # header row
        name = DISPLAY_NAMES.get(raw_name, raw_name) if rename else raw_name
        if name in exclude or raw_name in exclude:
            continue
        values = cells[1:]
        if len(values) != n_expected:
            raise ValueError(
                f"Row '{raw_name}' has {len(values)} value cells, expected "
                f"{n_expected} (= {len(datasets)} datasets x {len(metrics)} metrics). "
                f"Check DATASETS / METRICS."
            )
        for i, cell in enumerate(values):
            mean, std, dagger = _parse_cell(cell)
            records.append(
                dict(
                    method=name,
                    dataset=datasets[i // len(metrics)],
                    metric=metrics[i % len(metrics)],
                    mean=mean,
                    std=std,
                    dagger=dagger,
                )
            )
    if not records:
        raise ValueError("No data rows found in the pasted LaTeX.")
    return pd.DataFrame.from_records(records)


def wide(df: pd.DataFrame, metric: str, value: str = "mean") -> pd.DataFrame:
    """methods x datasets table of `value` ('mean' or 'std'), in table order."""
    sub = df[df.metric == metric]
    out = sub.pivot(index="method", columns="dataset", values=value)
    return out.reindex(index=sub.method.unique(), columns=sub.dataset.unique())


# -----------------------------------------------------------------------------
# Styling / drawing helpers
# -----------------------------------------------------------------------------
def setup_style(context: str = "talk", font_scale: float = 0.9):
    """Seaborn theme suitable for slides; fonts embedded as TrueType in PDFs."""
    sns.set_theme(
        context=context,
        style="whitegrid",
        font_scale=font_scale,
        rc={
            "axes.edgecolor": "0.35",
            "axes.linewidth": 1.0,
            "grid.color": "0.90",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 1.5,
        },
    )


def _err(v, e, log):
    if e is None or not np.isfinite(e) or e == 0:
        return None
    lo = min(e, 0.95 * v) if log else e  # keep lower bar > 0 on log axes
    return [[lo], [e]]


def plot_point(
    ax, x, y, name, xerr=None, yerr=None, log_x=False, style=None, err_alpha=0.55
):
    """One method as a marker with (optional) x/y error bars."""
    st = style or get_style(name)
    xe, ye = _err(x, xerr, log_x), _err(y, yerr, False)
    if xe is not None or ye is not None:
        ax.errorbar(
            x,
            y,
            xerr=xe,
            yerr=ye,
            fmt="none",
            ecolor=st["color"],
            elinewidth=1.4,
            capsize=3,
            alpha=err_alpha,
            zorder=st.get("zorder", 3) - 0.5,
        )
    ax.plot(
        [x],
        [y],
        linestyle="none",
        marker=st["marker"],
        markersize=st["size"],
        color=st["color"],
        markeredgecolor="white",
        markeredgewidth=0.9,
        zorder=st.get("zorder", 3),
    )


def marker_handle(name=None, style=None):
    st = style or get_style(name)
    return Line2D(
        [],
        [],
        linestyle="none",
        marker=st["marker"],
        markersize=st["size"],
        color=st["color"],
        markeredgecolor="white",
        markeredgewidth=0.9,
    )


def family_handle(family):
    return marker_handle(style=FAMILY_STYLES[family])


def line_handle(name):
    st = get_style(name)
    return Line2D([], [], color=st["color"], linestyle=st["linestyle"], lw=2.2)


def add_legend(fig, handles: dict, cfg: dict, axes=None):
    """Legend from a {key: handle} dict, controlled by a config dict.

    Special cfg keys (the rest goes straight to matplotlib's legend()):
      show    : False hides the legend
      entries : list of keys = which entries and in which order (None = all)
      labels  : {key: displayed text} to rename entries
      anchor  : "figure" (fig.legend) or an int = index of the axes to attach to
    Useful matplotlib keys: loc, bbox_to_anchor, ncol, fontsize, title,
    title_fontsize, frameon, markerscale, handletextpad, columnspacing.
    With anchor="figure", loc can be e.g. "outside lower center",
    "outside right upper" (places the legend next to the panels).
    """
    cfg = dict(cfg)
    if not cfg.pop("show", True):
        return None
    entries = cfg.pop("entries", None) or list(handles)
    labels = cfg.pop("labels", None) or {}
    anchor = cfg.pop("anchor", "figure")
    missing = [e for e in entries if e not in handles]
    if missing:
        warnings.warn(
            f"Legend entries not in this plot: {missing}. "
            f"Available: {list(handles)}"
        )
    entries = [e for e in entries if e in handles]
    hs, ls = [handles[e] for e in entries], [labels.get(e, e) for e in entries]
    if anchor == "figure":
        return fig.legend(hs, ls, **cfg)
    return np.ravel(axes)[anchor].legend(hs, ls, **cfg)


def format_log_axis(ax, axis="x", subs=(1.0, 2.0, 5.0)):
    """Log axis with plain tick labels (2, 5, 10, 20, ...) instead of 10^x."""
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(LogLocator(base=10, subs=subs))
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())


def save_figure(
    fig, name, outdir="figures", formats=("pdf", "png"), dpi=300, transparent=False
):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)
        print(f"saved {path}")
