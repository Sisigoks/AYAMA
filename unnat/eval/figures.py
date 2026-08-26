"""Publication figures and LaTeX tables, rendered from study.json.

Everything a slide or a paper needs, generated from the same measurements the
site reads, so a figure can never drift from the number it illustrates. Each
figure is written twice: PNG at 300 dpi for slides, PDF as vector for LaTeX.

Design rules followed here, and why they are not preferences:

  - **No dual-axis charts.** The sun sweep carries two measures on different
    scales (an F1 in [0,1] and an error in metres). Two y-axes on one frame let
    the author choose the crossing point, so the figure is two stacked panels
    sharing one x-axis instead.
  - **Colour by the job.** Bars comparing one measure across categories use a
    single hue; the three-way identity split (method / baseline / floor) uses a
    validated categorical triple; elevation is magnitude, so a single-hue
    sequential ramp; error is polarity, so a diverging ramp with a neutral
    grey midpoint - never a rainbow, never a hue in the middle.
  - **Direct value labels on every bar.** Two of the three categorical hues sit
    below 3:1 against a white surface, and the rule for that case is relief:
    the reading must not depend on telling the colours apart.
  - **Legends for two or more series, none for one.** A single-series chart is
    named by its title; a legend box would be decoration.

Palette: the validated default (blue #2a78d6, orange #eb6834, aqua #1baf7a),
which clears the all-pairs colour-vision-deficiency and normal-vision floors.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np

# ── palette ─────────────────────────────────────────────────────────────────
BLUE = "#2a78d6"      # categorical slot 1 · also the sequential hue
ORANGE = "#eb6834"    # slot 2
AQUA = "#1baf7a"      # slot 3
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e6e5e1"
SURFACE = "#ffffff"
NEUTRAL = "#f0efec"   # diverging midpoint
RED = "#d03b3b"

SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", NEUTRAL,
             "#f6c4b4", "#ec835a", "#d03b3b", "#8a1f1f"]

VARIANT_LABEL = {
    "dem_only": "DEM alone (floor)",
    "global_affine": "Global affine",
    "agmc_no_gate": "AGMC, no semantic gate",
    "agmc_no_shadow": "AGMC, no shadow anchors",
    "agmc_no_water": "AGMC, no water constraint",
    "agmc": "AGMC (full)",
    "agmc_bootstrap": "AGMC + bootstrap σ",
}
REFERENCE_VARIANTS = ("dem_only", "global_affine")


def _mpl():
    """Import matplotlib with a presentation-grade style already applied."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.titleweight": "600",
        "axes.labelsize": 11,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "text.color": INK,
    })
    return plt


def _save(fig, out_dir: str, name: str) -> list:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for ext, dpi in (("png", 300), ("pdf", None)):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=dpi) if dpi else fig.savefig(path)
        written.append(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return written


def _caption(fig, text: str) -> None:
    """A one-line caption under the axes: a figure on a slide travels alone."""
    fig.text(0.0, -0.045, text, ha="left", va="top", fontsize=9, color=INK_3, wrap=True)


def _val(entry, key="mean", default=float("nan")):
    if entry is None:
        return default
    if isinstance(entry, dict):
        v = entry.get(key, default)
        return default if v is None else v
    return entry


# ── 1. ablation ─────────────────────────────────────────────────────────────
def fig_ablation(study: dict, out_dir: str) -> list:
    plt = _mpl()

    by_variant: dict = {}
    for rows in (study.get("ablation") or {}).values():
        for r in rows or []:
            if r and r.get("mae_m") is not None:
                by_variant.setdefault(r["variant"], []).append(r["mae_m"])
    if not by_variant:
        return []

    order = [v for v in VARIANT_LABEL if v in by_variant]
    means = np.array([np.mean(by_variant[v]) for v in order])
    stds = np.array([np.std(by_variant[v]) for v in order])
    labels = [VARIANT_LABEL[v] for v in order]
    colors = [AQUA if v == "dem_only" else (ORANGE if v in REFERENCE_VARIANTS else BLUE)
              for v in order]

    fig, ax = plt.subplots(figsize=(8.4, 0.52 * len(order) + 1.5))
    y = np.arange(len(order))[::-1]
    ax.barh(y, means, xerr=stds if stds.any() else None, height=0.62,
            color=colors, ecolor=INK_3, capsize=3, error_kw={"lw": 1.2})

    for yi, m, sd in zip(y, means, stds):
        ax.text(m + max(means) * 0.015 + (sd or 0), yi, f"{m:.2f}",
                va="center", ha="left", fontsize=10, color=INK)

    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean absolute error (m)  ·  lower is better")
    ax.set_xlim(0, max(means + stds) * 1.16)
    ax.xaxis.grid(True, color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    ax.set_title("Which components earn their place")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, AQUA)]
    ax.legend(handles, ["Anchor-graph calibration", "Global-affine baseline",
                        "DEM alone — the floor"],
              loc="lower right", ncol=1, borderaxespad=0.4)

    _caption(fig, "One depth inference per scene; every variant re-solves only the "
                  "calibration, so each bar sees the identical depth field. "
                  "Bars are means over scenes; whiskers are ±1 s.d.")
    return _save(fig, out_dir, "fig1_ablation")


# ── 2. error by class ───────────────────────────────────────────────────────
def fig_by_class(study: dict, out_dir: str) -> list:
    plt = _mpl()
    by_class = (study.get("aggregate") or {}).get("by_class_mae_m") or {}
    if not by_class:
        return []

    items = sorted(by_class.items(), key=lambda kv: _val(kv[1]))
    labels = [k for k, _ in items]
    means = np.array([_val(v) for _, v in items])
    stds = np.array([_val(v, "std", 0.0) for _, v in items])

    fig, ax = plt.subplots(figsize=(7.4, 0.52 * len(items) + 1.4))
    y = np.arange(len(items))[::-1]
    ax.barh(y, means, xerr=stds if stds.any() else None, height=0.6,
            color=BLUE, ecolor=INK_3, capsize=3, error_kw={"lw": 1.2})
    for yi, m in zip(y, means):
        ax.text(m + max(means) * 0.015, yi, f"{m:.2f} m", va="center", ha="left",
                fontsize=10, color=INK)

    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean absolute error (m)")
    ax.set_xlim(0, max(means + stds) * 1.18)
    ax.xaxis.grid(True, color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    ax.set_title("Where the error lives")
    _caption(fig, "Terrain classes are close to solved; buildings and canopy carry the "
                  "error, which is where a monocular method is weakest and where the "
                  "anchor sources are sparsest.")
    return _save(fig, out_dir, "fig2_error_by_class")


# ── 3. the shadow physics window ────────────────────────────────────────────
def fig_sun_window(study: dict, out_dir: str) -> list:
    """Two stacked panels, one shared x. Never two y-axes on one frame."""
    plt = _mpl()
    rows = [r for r in (study.get("sun_sweep") or []) if r.get("sun_elevation_deg") is not None]
    if not rows:
        return []

    el = np.array([r["sun_elevation_deg"] for r in rows], float)
    f1 = np.array([r.get("f1") if r.get("f1") is not None else np.nan for r in rows], float)
    err = np.array([r.get("median_abs_height_error_m")
                    if r.get("median_abs_height_error_m") is not None else np.nan
                    for r in rows], float)
    n = np.array([r.get("n_anchors", 0) or 0 for r in rows], float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 6.0), sharex=True,
                                   gridspec_kw={"hspace": 0.16})
    for ax in (ax1, ax2):
        ax.axvspan(20, 75, color=AQUA, alpha=0.10, lw=0)
        ax.yaxis.grid(True, color=GRID, lw=1.0)
        ax.set_axisbelow(True)

    ax1.plot(el, f1, color=BLUE, marker="o", markerfacecolor=BLUE,
             markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax1.set_ylabel("Shadow detection F1")
    ax1.set_ylim(0, 1.02)
    ax1.set_title("The physics window: shadow height needs the sun in a band")
    ax1.text(47.5, 0.06, "usable band, 20–75°", ha="center", fontsize=9.5, color=INK_3)

    ok = np.isfinite(err)
    ax2.plot(el[ok], err[ok], color=ORANGE, marker="o", markerfacecolor=ORANGE,
             markeredgecolor=SURFACE, markeredgewidth=1.2)
    for x, y_, k in zip(el, err, n):
        if not np.isfinite(y_):
            ax2.plot([x], [0.25], marker="x", color=INK_3, ms=7, mew=1.6)
    ax2.set_ylabel("Median height error (m)")
    ax2.set_xlabel("Sun elevation (degrees)")
    ax2.set_ylim(0, max(1.0, np.nanmax(err) * 1.2) if ok.any() else 1.0)
    ax2.text(el.min() + 0.5, ax2.get_ylim()[1] * 0.88,
             "×  no anchor survived the quality gate", fontsize=9.5, color=INK_3)

    _caption(fig, "Height from shadow length alone, h = L·tan(elevation), with no depth "
                  "model involved. Low sun sprawls shadows across terrain; high sun "
                  "shortens them below the resolution of the image.")
    return _save(fig, out_dir, "fig3_sun_window")


# ── 4. one free parameter ───────────────────────────────────────────────────
def fig_lambda(study: dict, out_dir: str) -> list:
    plt = _mpl()
    rows = study.get("lambda_sweep") or []
    agmc = [r for r in rows if r.get("lam") is not None]
    base = next((r for r in rows if r.get("lam") is None), None)
    if len(agmc) < 3:
        return []

    lam = np.array([r["lam"] for r in agmc], float)
    mae = np.array([r["mae_m"] for r in agmc], float)
    floor = _val((study.get("aggregate") or {}).get("dem_mae_m"))

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.set_xscale("log")
    ax.yaxis.grid(True, color=GRID, lw=1.0)
    ax.set_axisbelow(True)

    if base:
        ax.axhline(base["mae_m"], color=ORANGE, lw=1.6, ls=(0, (5, 4)))
        ax.text(lam.max(), base["mae_m"], f"  global affine {base['mae_m']:.2f} m",
                va="center", ha="left", fontsize=9.5, color=ORANGE)
    if np.isfinite(floor):
        ax.axhline(floor, color=AQUA, lw=1.6, ls=(0, (5, 4)))
        ax.text(lam.max(), floor, f"  DEM alone {floor:.2f} m",
                va="center", ha="left", fontsize=9.5, color=AQUA)

    ax.plot(lam, mae, color=BLUE, marker="o", markerfacecolor=BLUE,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
    best = int(np.argmin(mae))
    ax.annotate(f"best {mae[best]:.2f} m at λ={lam[best]:g}",
                xy=(lam[best], mae[best]), xytext=(0, -22),
                textcoords="offset points", ha="center", fontsize=9.5, color=INK)

    ax.set_xlabel("Smoothness weight λ  (log scale)")
    ax.set_ylabel("Mean absolute error (m)")
    ax.set_title("Sensitivity to the one free parameter")
    ax.set_xlim(lam.min() * 0.75, lam.max() * 1.45)
    _caption(fig, "A parameter that has to be hunted for per scene is a knob, not a "
                  "method. Flat across more than an order of magnitude; the default "
                  "sits inside that plateau.")
    return _save(fig, out_dir, "fig4_lambda_sensitivity")


# ── 5. is sigma honest ──────────────────────────────────────────────────────
def fig_reliability(study: dict, out_dir: str, scenes_dir: Optional[str] = None,
                    n_bins: int = 10) -> list:
    """Predicted σ against the error actually observed at that σ.

    The diagonal is a perfectly calibrated uncertainty. Reads the per-pixel σ and
    error rasters, so it needs a completed run on disk; without one it is skipped
    rather than faked from the aggregate.
    """
    plt = _mpl()
    if not scenes_dir:
        return []
    try:
        import rasterio
    except ImportError:
        return []

    sig_all, err_all = [], []
    for scene in study.get("scenes") or []:
        d = os.path.join(scenes_dir, f"seed{scene.get('seed')}", "run")
        sp, ep = os.path.join(d, "sigma.tif"), os.path.join(d, "error.tif")
        if not (os.path.exists(sp) and os.path.exists(ep)):
            continue
        with rasterio.open(sp) as ds:
            sig = ds.read(1).astype(np.float32)
        with rasterio.open(ep) as ds:
            err = ds.read(1).astype(np.float32)
        m = np.isfinite(sig) & np.isfinite(err) & (sig > 0)
        step = max(1, int(m.sum() // 400_000))
        sig_all.append(sig[m][::step])
        err_all.append(err[m][::step])
    if not sig_all:
        return []

    sig = np.concatenate(sig_all)
    err = np.concatenate(err_all)
    edges = np.quantile(sig, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    xs, ys, ns = [], [], []
    for i in range(n_bins):
        sel = (sig >= edges[i]) & (sig < edges[i + 1])
        if sel.sum() < 50:
            continue
        xs.append(float(sig[sel].mean()))
        ys.append(float(np.sqrt((err[sel] ** 2).mean())))
        ns.append(int(sel.sum()))

    if len(xs) < 3:
        return []
    xs, ys = np.array(xs), np.array(ys)
    coverage = float((np.abs(err) <= sig).mean())

    # How much does sigma actually vary? A field with no spread can be perfectly
    # calibrated on average and still be useless, because it cannot rank one
    # pixel against another - which is the whole point of a per-pixel sigma.
    lo_s, hi_s = np.percentile(sig, [1, 99])
    spread = (hi_s - lo_s) / max(float(np.median(sig)), 1e-6)
    degenerate = spread < 0.05

    fig, ax = plt.subplots(figsize=(6.4, 5.9))
    lim = max(xs.max(), ys.max()) * 1.12
    ax.plot([0, lim], [0, lim], color=INK_3, lw=1.4, ls=(0, (5, 4)), zorder=1)
    ax.text(lim * 0.97, lim * 0.94, "perfectly calibrated", ha="right", va="top",
            fontsize=9.5, color=INK_3, rotation=45, rotation_mode="anchor")

    if degenerate:
        # Say it on the chart rather than leaving a vertical line to be misread.
        ax.axvspan(lo_s, hi_s, color=ORANGE, alpha=0.16, lw=0, zorder=2)
        ax.annotate(f"every decile lands in {lo_s:.2f}–{hi_s:.2f} m",
                    xy=(float(np.median(xs)), float(ys.min())),
                    xytext=(lim * 0.42, float(ys.min()) * 0.45),
                    fontsize=9.5, color=ORANGE,
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))

    ax.plot(xs, ys, color=BLUE, marker="o", markerfacecolor=BLUE,
            markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Predicted uncertainty, mean σ in bin (m)")
    ax.set_ylabel("Observed error, RMS in bin (m)")
    ax.set_title("Does σ predict the error?")

    if degenerate:
        verdict = ("σ is effectively constant, so it is calibrated on average\n"
                   "but cannot say where the estimate is weaker")
    elif (ys > xs).mean() > 0.6:
        verdict = "above the line: σ understates the error"
    elif (ys < xs).mean() > 0.6:
        verdict = "below the line: σ is conservative"
    else:
        verdict = "tracks the diagonal"
    ax.text(0.04, 0.96, f"1σ coverage {coverage:.2f}   (Gaussian: 0.68)\n{verdict}",
            transform=ax.transAxes, va="top", ha="left", fontsize=10, color=INK)

    _caption(fig, "Pixels binned by predicted σ; each point is one decile. Coverage "
                  "answers whether the bars are the right size on average; the spread "
                  "along x answers whether σ can rank one pixel against another. "
                  "A field that is calibrated but flat passes the first test and fails "
                  "the second.")
    return _save(fig, out_dir, "fig5_reliability")


# ── 6. qualitative panel ────────────────────────────────────────────────────
def fig_qualitative(study: dict, out_dir: str, scenes_dir: Optional[str] = None) -> list:
    plt = _mpl()
    if not scenes_dir:
        return []
    try:
        import rasterio
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return []

    scenes = study.get("scenes") or []
    if not scenes:
        return []
    seed = scenes[0].get("seed")
    d = os.path.join(scenes_dir, f"seed{seed}")
    run = os.path.join(d, "run")
    need = {"rgb": os.path.join(d, "scene.tif"),
            "truth": os.path.join(d, "scene_dsm.tif"),
            "pred": os.path.join(run, "dsm.tif"),
            "err": os.path.join(run, "error.tif"),
            "sigma": os.path.join(run, "sigma.tif")}
    if not all(os.path.exists(v) for v in need.values()):
        return []

    seq = LinearSegmentedColormap.from_list("unnat_seq", SEQ_BLUE)
    div = LinearSegmentedColormap.from_list("unnat_div", DIVERGING)

    with rasterio.open(need["rgb"]) as ds:
        rgb = np.moveaxis(ds.read([1, 2, 3]), 0, -1)
    read = {k: rasterio.open(v).read(1).astype(np.float32)
            for k, v in need.items() if k != "rgb"}

    lo, hi = np.percentile(read["truth"], [1, 99])
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.9))
    panels = [
        ("Input image", rgb, None, None, None),
        ("Reference DSM (m)", read["truth"], seq, lo, hi),
        ("Predicted DSM (m)", read["pred"], seq, lo, hi),
        ("Error (m)", read["err"], div, -15, 15),
        ("Uncertainty σ (m)", read["sigma"], seq, None, None),
    ]
    from matplotlib.cm import ScalarMappable

    for ax, (title, data, cmap, vmin, vmax) in zip(axes, panels):
        if cmap is None:
            ax.imshow(data)
            # Reserve the same width a colorbar takes, so all five panels are
            # the same size; an image panel that is subtly smaller than its
            # neighbours reads as a different scale.
            spacer = fig.colorbar(ScalarMappable(cmap=seq), ax=ax, fraction=0.046, pad=0.03)
            spacer.ax.set_visible(False)
        else:
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=8, color=GRID, labelcolor=INK_2)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    m = scenes[0].get("metrics") or {}
    fig.suptitle(f"Scene {seed}  ·  MAE {m.get('mae_m', float('nan')):.2f} m  ·  "
                 f"RMSE {m.get('rmse_m', float('nan')):.2f} m  ·  "
                 f"1σ coverage {m.get('coverage_1s', float('nan')):.2f}",
                 fontsize=12, y=1.04)
    _caption(fig, "Elevation uses a single-hue sequential ramp (magnitude); error uses a "
                  "diverging ramp with a neutral midpoint (polarity, blue too low / red "
                  "too high). Reference and prediction share one colour scale.")
    return _save(fig, out_dir, "fig6_qualitative")


# ── LaTeX tables ────────────────────────────────────────────────────────────
def _tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def write_tables(study: dict, out_dir: str) -> list:
    """booktabs tables, ready to \\input{} into a paper."""
    os.makedirs(out_dir, exist_ok=True)
    agg = study.get("aggregate") or {}
    written = []

    def pm(key, dec=2):
        e = agg.get(key)
        if not e:
            return "--"
        return f"${_val(e):.{dec}f} \\pm {_val(e, 'std', 0.0):.{dec}f}$"

    rows = [
        ("MAE (m)", "mae_m", "baseline_mae_m", "dem_mae_m", 2),
        ("RMSE (m)", "rmse_m", "baseline_rmse_m", "dem_rmse_m", 2),
        ("Pearson $r$", "pearson_r", "baseline_pearson_r", "dem_pearson_r", 3),
        ("Spearman $\\rho$", "spearman_r", None, None, 3),
        ("Bias (m)", "bias_m", None, None, 2),
        ("Median AE (m)", "median_ae_m", None, None, 2),
        ("Slope MAE ($^\\circ$)", "slope_mae_deg", None, None, 2),
        ("Edge F1", "edge_f1", None, None, 3),
        ("$1\\sigma$ coverage", "coverage_1s", None, None, 2),
        ("ECE (m)", "ece_m", None, None, 2),
    ]
    lines = [
        "% Generated by unnat.eval.figures - do not edit by hand.",
        "\\begin{tabular}{lccc}", "\\toprule",
        "Metric & AGMC & Global affine & DEM alone (floor) \\\\", "\\midrule",
    ]
    for label, a, b, c, dec in rows:
        if not agg.get(a):
            continue
        lines.append(f"{label} & {pm(a, dec)} & "
                     f"{pm(b, dec) if b else '--'} & {pm(c, dec) if c else '--'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    p = os.path.join(out_dir, "table1_headline.tex")
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    written.append(p)

    by_variant: dict = {}
    for rs in (study.get("ablation") or {}).values():
        for r in rs or []:
            if r and r.get("mae_m") is not None:
                by_variant.setdefault(r["variant"], []).append(r)
    if by_variant:
        lines = [
            "% Generated by unnat.eval.figures - do not edit by hand.",
            "\\begin{tabular}{lcccc}", "\\toprule",
            "Variant & MAE (m) & RMSE (m) & $r$ & Anchors \\\\", "\\midrule",
        ]
        for v in VARIANT_LABEL:
            if v not in by_variant:
                continue
            rs = by_variant[v]
            lines.append(
                f"{_tex_escape(VARIANT_LABEL[v])} & "
                f"{np.mean([r['mae_m'] for r in rs]):.2f} & "
                f"{np.mean([r['rmse_m'] for r in rs]):.2f} & "
                f"{np.mean([r['pearson_r'] for r in rs]):.3f} & "
                f"{int(np.mean([r['n_anchors'] for r in rs]))} \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        p = os.path.join(out_dir, "table2_ablation.tex")
        open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        written.append(p)

    by_class = agg.get("by_class_mae_m") or {}
    if by_class:
        lines = [
            "% Generated by unnat.eval.figures - do not edit by hand.",
            "\\begin{tabular}{lc}", "\\toprule",
            "Class & MAE (m) \\\\", "\\midrule",
        ]
        for k, v in sorted(by_class.items(), key=lambda kv: _val(kv[1])):
            lines.append(f"{_tex_escape(k)} & ${_val(v):.2f} \\pm "
                         f"{_val(v, 'std', 0.0):.2f}$ \\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        p = os.path.join(out_dir, "table3_by_class.tex")
        open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        written.append(p)

    return written


# ── entry point ─────────────────────────────────────────────────────────────
FIGURES = (
    ("ablation", fig_ablation),
    ("error by class", fig_by_class),
    ("sun window", fig_sun_window),
    ("lambda sensitivity", fig_lambda),
    ("reliability", fig_reliability),
    ("qualitative panel", fig_qualitative),
)


def render_all(study: dict, out_dir: str, scenes_dir: Optional[str] = None,
               on_step: Optional[Callable[[int, int, str], None]] = None) -> list:
    written: list = []
    total = len(FIGURES) + 1
    for i, (name, fn) in enumerate(FIGURES):
        if on_step:
            on_step(i, total, name)
        try:
            if fn in (fig_reliability, fig_qualitative):
                written += fn(study, out_dir, scenes_dir=scenes_dir)
            else:
                written += fn(study, out_dir)
        except Exception as exc:                      # a bad figure is not a failed study
            print(f"    figure '{name}' skipped: {type(exc).__name__}: {exc}")
    if on_step:
        on_step(len(FIGURES), total, "LaTeX tables")
    written += write_tables(study, os.path.join(out_dir, "tables"))
    return written
