#!/usr/bin/env python3
"""
Predictive case translation figure (Ridge + 1-NN vs Φ).

Panels A–B: violin + scatter (not bars).
Outputs: Paper_fig/ (and optional review copies under testfig/).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from analysis_predictive_translation_ridge import (  # noqa: E402
    collect_predictive_translation_records,
    stratify_tercile,
)

OUT = ROOT / "testfig"
PAPER = ROOT / "Paper_fig"
EQ = ROOT / "results" / "equivalence_nontriviality"
PBA = ROOT / "results" / "population_bridge_analysis"
PAIRS_CV = ROOT / "results" / "cvae_cross_gender" / "emotion_equivalent_pairs_themecv.csv"
VA_LIM = (1.0, 7.0)
CAT_ORDER = ["Scene", "Person", "Object", "Animal"]

PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "savefig.dpi": 300,
}

COL_PHI = "#37474f"
COL_RIDGE = "#6a1b9a"
COL_NN = "#ef6c00"
COL_RAW = "#78909c"


def save_dual(fig: plt.Figure, stem: str, out_dir: Path = OUT) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, svg


def panel_label(ax, letter: str) -> None:
    ax.text(
        -0.12, 1.06, letter, transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="top", ha="left",
    )


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return centre - margin, centre + margin


CAT_COLORS = {
    "Scene": "#78909c",
    "Person": "#c62828",
    "Object": "#546e7a",
    "Animal": "#ef6c00",
}


def violin_scatter(
    ax,
    data: list[np.ndarray],
    positions: list[float] | np.ndarray,
    colors: list[str],
    *,
    widths: float = 0.72,
    seed: int = 0,
    max_points: int = 250,
    alpha_body: float = 0.32,
    alpha_pts: float = 0.22,
    s: float = 7,
) -> None:
    """Violin + jittered scatter + median marker (paper Fig.4/5 style)."""
    positions = np.asarray(positions, float)
    parts = ax.violinplot(
        data, positions=positions, showmedians=False, widths=widths,
    )
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_alpha(alpha_body)
        body.set_edgecolor("0.35")
        body.set_linewidth(0.8)
    for key in ("cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_visible(False)

    rng = np.random.default_rng(seed)
    for pos, arr, col in zip(positions, data, colors):
        arr = np.asarray(arr, float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        n = min(len(arr), max_points)
        idx = rng.choice(len(arr), size=n, replace=False)
        jitter = rng.uniform(-0.12, 0.12, size=n) * (widths / 0.72)
        ax.scatter(
            pos + jitter, arr[idx], s=s, color=col, alpha=alpha_pts,
            edgecolors="none", zorder=2,
        )
        med = float(np.median(arr))
        ax.scatter(
            [pos], [med], s=42, color=col, edgecolors="white",
            linewidths=0.7, zorder=5,
        )
        ax.hlines(med, pos - 0.18 * widths, pos + 0.18 * widths, colors="0.15", lw=1.1, zorder=4)


def va_bin_edges(lim: tuple[float, float], grid: float) -> np.ndarray:
    edges = np.arange(lim[0], lim[1] + grid * 0.5, grid, dtype=float)
    if not np.isclose(edges[-1], lim[1]):
        edges = np.append(edges, float(lim[1]))
    else:
        edges[-1] = float(lim[1])
    return edges


def va_cell_stats(
    rec: pd.DataFrame,
    value_col: str,
    *,
    grid: float = 1.0,
    min_n: int = 3,
    lim: tuple[float, float] = VA_LIM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = va_bin_edges(lim, grid)
    nbins = len(edges) - 1
    grid_vals = np.full((nbins, nbins), np.nan)
    counts = np.zeros((nbins, nbins), int)
    v = rec["y_f_i_valence"].to_numpy(float)
    a = rec["y_f_i_arousal"].to_numpy(float)
    iv = np.clip(np.digitize(v, edges) - 1, 0, nbins - 1)
    ia = np.clip(np.digitize(a, edges) - 1, 0, nbins - 1)
    for i in range(nbins):
        for j in range(nbins):
            mask = (iv == j) & (ia == i)
            counts[i, j] = int(mask.sum())
            if counts[i, j] < min_n:
                continue
            grid_vals[i, j] = float(rec.loc[mask, value_col].mean())
    return edges, grid_vals, counts


def draw_overall_violin(ax, rec: pd.DataFrame) -> None:
    methods = [
        ("Φ", "phi_l2", COL_PHI),
        ("Ridge", "ridge_l2", COL_RIDGE),
        ("1-NN", "xfer_l2", COL_NN),
        ("No xfer", "raw_l2", COL_RAW),
    ]
    data = [rec[col].to_numpy(float) for _, col, _ in methods]
    colors = [c for _, _, c in methods]
    labels = [m for m, _, _ in methods]
    xpos = np.arange(len(methods), dtype=float)

    violin_scatter(ax, data, xpos, colors, widths=0.78, seed=0)

    meds = [float(np.median(d)) for d in data]
    for x, med, col in zip(xpos, meds, colors):
        ax.text(x, med + 0.08, f"{med:.2f}", ha="center", va="bottom", fontsize=7.5, color=col, fontweight="bold")

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("L2 error (female target VA)")
    ax.set_title("Overall (n=900; theme-CV)")
    ax.axhline(meds[0], color=COL_PHI, ls="--", lw=0.8, alpha=0.55)
    ymax = max(np.percentile(d, 99) for d in data)
    ax.set_ylim(0, min(ymax * 1.08, 3.2))
    ax.text(
        0.98, 0.97,
        f"beat Φ: Ridge {rec['beat_phi_ridge'].mean():.0%} · 1-NN {rec['beat_phi_xfer'].mean():.0%}",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.2, color="0.35",
    )


def draw_raw_gap_tercile_violin(ax, rec: pd.DataFrame) -> None:
    sub = rec.copy()
    sub["raw_gap_tercile"] = pd.qcut(sub["raw_gap_l2"], 3, labels=["low", "mid", "high"])
    terciles = ["low", "mid", "high"]
    methods = [
        ("Φ", "phi_l2", COL_PHI),
        ("Ridge", "ridge_l2", COL_RIDGE),
        ("1-NN", "xfer_l2", COL_NN),
    ]
    offsets = np.array([-0.28, 0.0, 0.28])
    width = 0.26

    all_data: list[np.ndarray] = []
    all_pos: list[float] = []
    all_cols: list[str] = []
    ridge_beat: list[float] = []
    med_ann: list[tuple[float, float, str]] = []

    for ti, t in enumerate(terciles):
        s = sub[sub.raw_gap_tercile == t]
        ridge_beat.append(float(s["beat_phi_ridge"].mean()))
        for oi, (_, col, color) in enumerate(methods):
            arr = s[col].to_numpy(float)
            pos = float(ti) + offsets[oi]
            all_data.append(arr)
            all_pos.append(pos)
            all_cols.append(color)
            med_ann.append((pos, float(np.median(arr)), color))

    violin_scatter(
        ax, all_data, all_pos, all_cols,
        widths=width, seed=1, max_points=180, alpha_pts=0.18, s=6,
    )

    for pos, med, color in med_ann:
        # light median labels only for Ridge / high-gap clutter control: all methods
        ax.plot(pos, med, "o", color=color, ms=4.5, mec="white", mew=0.6, zorder=5)

    ax.set_xticks(range(len(terciles)))
    ax.set_xticklabels(["low gap", "mid gap", "high gap"])
    ax.set_ylabel("L2 error")
    ax.set_title("By pre-Φ raw gap tercile (‖y_m−y_f‖)")
    ymax = max(np.percentile(d, 98) for d in all_data)
    ax.set_ylim(0, min(ymax * 1.05, 3.0))

    legend = [
        Patch(facecolor=COL_PHI, edgecolor="0.35", alpha=0.45, label="Φ"),
        Patch(facecolor=COL_RIDGE, edgecolor="0.35", alpha=0.45, label="Ridge"),
        Patch(facecolor=COL_NN, edgecolor="0.35", alpha=0.45, label="1-NN"),
    ]
    ax.legend(handles=legend, fontsize=7, loc="upper left", frameon=False)

    ax2 = ax.twinx()
    ax2.plot(range(len(terciles)), ridge_beat, "o-", color=COL_RIDGE, lw=1.6, ms=6.5, zorder=6)
    ax2.set_ylabel("Ridge beat Φ rate", color=COL_RIDGE)
    ax2.set_ylim(0, 0.70)
    ax2.tick_params(axis="y", labelcolor=COL_RIDGE)

    hi = stratify_tercile(rec, "raw_gap_l2")["terciles"]["high"]
    hi_r = hi["clip_ridge_xfer"]
    hi_p = hi["phi_same"]
    ax.text(
        0.98, 0.97,
        f"high gap: Ridge {hi_r['median_l2']:.2f} vs Φ {hi_p['median_l2']:.2f}\n"
        f"Wilcoxon p={hi_r['wilcoxon_two_sided_p']:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.8", alpha=0.92),
    )


def draw_overall_bar_sd(ax, rec: pd.DataFrame) -> None:
    methods = [
        ("Φ", "phi_l2", COL_PHI),
        ("Ridge", "ridge_l2", COL_RIDGE),
        ("1-NN", "xfer_l2", COL_NN),
        ("No xfer", "raw_l2", COL_RAW),
    ]
    means = [float(rec[col].mean()) for _, col, _ in methods]
    sds = [float(rec[col].std(ddof=1)) for _, col, _ in methods]
    meds = [float(rec[col].median()) for _, col, _ in methods]
    colors = [c for _, _, c in methods]
    labels = [m for m, _, _ in methods]
    xpos = np.arange(len(methods), dtype=float)

    bars = ax.bar(
        xpos, means, yerr=sds, color=colors, edgecolor="0.25", width=0.72,
        error_kw=dict(ecolor="0.25", lw=1.1, capsize=3.5, capthick=1.0),
        zorder=2,
    )
    # median ticks (canonical estimand in text is often median)
    for i, (x, med, col) in enumerate(zip(xpos, meds, colors)):
        ax.plot([x - 0.22, x + 0.22], [med, med], color="white", lw=2.2, zorder=4)
        ax.plot([x - 0.22, x + 0.22], [med, med], color=col, lw=1.2, zorder=5)
        ax.text(
            x, means[i] + sds[i] + 0.04, f"μ={means[i]:.2f}",
            ha="center", va="bottom", fontsize=6.8, color=col,
        )
        ax.text(
            x, med - 0.06, f"md={med:.2f}",
            ha="center", va="top", fontsize=6.5, color="0.2",
        )

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("L2 error (mean ± SD)")
    ax.set_title("Overall (n=900; theme-CV)")
    ax.axhline(meds[0], color=COL_PHI, ls="--", lw=0.8, alpha=0.55)
    ymax = max(m + s for m, s in zip(means, sds))
    ax.set_ylim(0, ymax * 1.18)
    ax.text(
        0.98, 0.97,
        f"beat Φ: Ridge {rec['beat_phi_ridge'].mean():.0%} · 1-NN {rec['beat_phi_xfer'].mean():.0%}\n"
        f"white/colored ticks = median",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.0, color="0.35",
    )
    _ = bars


def draw_raw_gap_tercile_bar_sd(ax, rec: pd.DataFrame) -> None:
    sub = rec.copy()
    sub["raw_gap_tercile"] = pd.qcut(sub["raw_gap_l2"], 3, labels=["low", "mid", "high"])
    terciles = ["low", "mid", "high"]
    methods = [
        ("Φ", "phi_l2", COL_PHI),
        ("Ridge", "ridge_l2", COL_RIDGE),
        ("1-NN", "xfer_l2", COL_NN),
    ]
    x = np.arange(len(terciles), dtype=float)
    w = 0.24
    offsets = np.array([-w, 0.0, w])
    ridge_beat: list[float] = []

    for oi, (lab, col, color) in enumerate(methods):
        means, sds = [], []
        for t in terciles:
            arr = sub.loc[sub.raw_gap_tercile == t, col].to_numpy(float)
            means.append(float(np.mean(arr)))
            sds.append(float(np.std(arr, ddof=1)))
        ax.bar(
            x + offsets[oi], means, w, yerr=sds, label=lab, color=color,
            edgecolor="0.25",
            error_kw=dict(ecolor="0.25", lw=0.95, capsize=2.8, capthick=0.9),
            zorder=2,
        )
        # median ticks
        for ti, t in enumerate(terciles):
            med = float(sub.loc[sub.raw_gap_tercile == t, col].median())
            xpos = x[ti] + offsets[oi]
            ax.plot([xpos - 0.08, xpos + 0.08], [med, med], color="white", lw=1.8, zorder=4)
            ax.plot([xpos - 0.08, xpos + 0.08], [med, med], color=color, lw=1.0, zorder=5)

    for t in terciles:
        ridge_beat.append(float(sub.loc[sub.raw_gap_tercile == t, "beat_phi_ridge"].mean()))

    ax.set_xticks(x)
    ax.set_xticklabels(["low gap", "mid gap", "high gap"])
    ax.set_ylabel("L2 error (mean ± SD)")
    ax.set_title("By pre-Φ raw gap tercile (‖y_m−y_f‖)")
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    # room for error bars
    ymax = 0.0
    for t in terciles:
        for _, col, _ in methods:
            arr = sub.loc[sub.raw_gap_tercile == t, col].to_numpy(float)
            ymax = max(ymax, float(np.mean(arr) + np.std(arr, ddof=1)))
    ax.set_ylim(0, ymax * 1.22)

    ax2 = ax.twinx()
    ax2.plot(x, ridge_beat, "o-", color=COL_RIDGE, lw=1.6, ms=6.5, zorder=6)
    ax2.set_ylabel("Ridge beat Φ rate", color=COL_RIDGE)
    ax2.set_ylim(0, 0.70)
    ax2.tick_params(axis="y", labelcolor=COL_RIDGE)

    hi = stratify_tercile(rec, "raw_gap_l2")["terciles"]["high"]
    hi_r = hi["clip_ridge_xfer"]
    hi_p = hi["phi_same"]
    ax.text(
        0.98, 0.97,
        f"high gap: Ridge md {hi_r['median_l2']:.2f} vs Φ {hi_p['median_l2']:.2f}\n"
        f"Wilcoxon p={hi_r['wilcoxon_two_sided_p']:.3f}\n"
        f"ticks = median",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.8", alpha=0.92),
    )


def draw_spatial_dual(
    fig: plt.Figure,
    gs_cell,
    rec: pd.DataFrame,
    *,
    grid: float = 1.0,
) -> None:
    sub_gs = gs_cell.subgridspec(1, 2, wspace=0.38)
    edges, raw_gap_grid, counts = va_cell_stats(rec, "raw_gap_l2", grid=grid)
    _, beat_grid, _ = va_cell_stats(rec, "beat_phi_ridge", grid=grid)
    extent = [edges[0], edges[-1], edges[0], edges[-1]]

    r_ok = raw_gap_grid[np.isfinite(raw_gap_grid)]
    b_ok = beat_grid[np.isfinite(beat_grid)]
    if len(r_ok) and len(b_ok):
        r_thr = float(np.nanquantile(raw_gap_grid, 0.65))
        b_thr = float(np.nanquantile(beat_grid, 0.65))
        both = (raw_gap_grid >= r_thr) & (beat_grid >= b_thr)
    else:
        both = np.zeros_like(raw_gap_grid, dtype=bool)

    for col, field, cmap, cbar_lab, title, vmax in [
        (0, raw_gap_grid, "YlOrRd", "mean raw gap", "Pre-Φ group gap", None),
        (1, beat_grid, "YlGn", "Ridge beat Φ rate", "Predictive Ridge beats Φ", 1.0),
    ]:
        ax = fig.add_subplot(sub_gs[0, col])
        vmin = 0 if col == 1 else (np.nanmin(field) if np.isfinite(field).any() else 0)
        vmax_use = vmax if vmax is not None else (np.nanmax(field) if np.isfinite(field).any() else 1)
        im = ax.imshow(
            field, origin="lower", extent=extent, cmap=cmap, aspect="equal",
            vmin=vmin, vmax=vmax_use, interpolation="nearest",
        )
        yy, xx = np.where(both)
        for yi, xi in zip(yy, xx):
            x0, x1 = edges[xi], edges[xi + 1]
            y0, y1 = edges[yi], edges[yi + 1]
            ax.plot(
                [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                color="#1565c0", lw=1.4, zorder=3,
            )
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).set_label(cbar_lab, fontsize=7)
        ax.set_xlim(*VA_LIM)
        ax.set_ylim(*VA_LIM)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Valence (female target)")
        if col == 0:
            ax.set_ylabel("Arousal (female target)")
        ax.set_title(title, fontsize=9)
        ax.text(
            0.02, 0.02, f"cells n≥3: {int((counts >= 3).sum())}",
            transform=ax.transAxes, fontsize=7, color="0.4",
        )


def _load_rec(*, direction: str = "MtoF", use_cached_csv: bool = True) -> pd.DataFrame:
    if direction not in {"MtoF", "FtoM"}:
        raise ValueError(f"direction must be MtoF or FtoM, got {direction!r}")
    csv_name = (
        "predictive_translation_per_image.csv"
        if direction == "MtoF"
        else "predictive_translation_per_image_FtoM.csv"
    )
    csv_path = EQ / csv_name
    if use_cached_csv and csv_path.exists():
        print(f"Loading {csv_path}")
        rec = pd.read_csv(csv_path)
    else:
        print(f"Collecting per-image records ({direction}; theme-CV 5-fold)...")
        rec, _, _, _, _ = collect_predictive_translation_records(direction=direction)
        EQ.mkdir(parents=True, exist_ok=True)
        rec.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")
    rec["beat_phi_ridge"] = rec["beat_phi_ridge"].astype(bool)
    rec["beat_phi_xfer"] = rec["beat_phi_xfer"].astype(bool)
    resid = pd.read_csv(PBA / "residual_per_image_gender.csv")
    rec = rec.merge(resid[["image_id", "category"]], on="image_id", how="left")
    rec["raw_gap_tercile"] = pd.qcut(rec["raw_gap_l2"], 3, labels=["low", "mid", "high"])
    rec["direction"] = direction
    return rec


def _category_high_gap_stats(rec: pd.DataFrame) -> list[dict]:
    """Per-category stats within high raw-gap tercile (predictive Ridge vs Φ)."""
    h = rec[rec.raw_gap_tercile == "high"]
    rows: list[dict] = []
    for cat in CAT_ORDER:
        s = h[h.category == cat]
        n = int(len(s))
        if n == 0:
            rows.append(
                {
                    "category": cat,
                    "n": 0,
                    "k": 0,
                    "beat": np.nan,
                    "ci_lo": np.nan,
                    "ci_hi": np.nan,
                    "med_delta": np.nan,
                }
            )
            continue
        k = int(s.beat_phi_ridge.sum())
        ci_lo, ci_hi = wilson_ci(k, n)
        rows.append(
            {
                "category": cat,
                "n": n,
                "k": k,
                "beat": float(k / n),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "med_delta": float((s.ridge_l2 - s.phi_l2).median()),
            }
        )
    return rows


def _category_high_gap_wilcoxon_p(rec: pd.DataFrame, cat: str) -> float:
    """Two-sided Wilcoxon on paired Δ = ridge_l2 − phi_l2 (high raw-gap tercile)."""
    from scipy import stats

    h = rec[rec.raw_gap_tercile == "high"]
    s = h[h.category == cat]
    d = (s.ridge_l2 - s.phi_l2).to_numpy(float)
    d = d[np.isfinite(d)]
    if len(d) < 3:
        return float("nan")
    return float(stats.wilcoxon(d, alternative="two-sided", zero_method="wilcox").pvalue)


def _format_beat_rate(beat: float) -> str:
    if not np.isfinite(beat):
        return "—"
    return f"{beat:.0%}"


def _format_ci(lo: float, hi: float) -> str:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "—"
    return f"[{lo:.2f}, {hi:.2f}]"


def _format_median_delta(d: float) -> str:
    if not np.isfinite(d):
        return "—"
    if abs(d) < 0.05:
        return "≈0"
    return f"{d:+.2f}"


def _format_p_value(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 0.001:
        s = f"{p:.4f}"
        if float(s) == 0.0:
            return "<0.001"
        return s
    return f"{p:.3f}"


def build_fig5_panel_e_predictive_table_rows(
    rec_m: pd.DataFrame,
    rec_f: pd.DataFrame,
) -> list[list[str]]:
    """Rows for Fig. 5E table (category × direction)."""
    header = ["Category", "Direction", "n", "beat rate", "95% CI", "median Δ", "p"]
    rows: list[list[str]] = [header]
    dir_map = [("M→F", rec_m), ("F→M", rec_f)]
    for cat in CAT_ORDER:
        for dir_label, rec in dir_map:
            stats = {r["category"]: r for r in _category_high_gap_stats(rec)}
            r = stats[cat]
            p = _category_high_gap_wilcoxon_p(rec, cat)
            rows.append([
                cat,
                dir_label,
                str(r["n"]) if r["n"] else "—",
                _format_beat_rate(r["beat"]),
                _format_ci(r["ci_lo"], r["ci_hi"]),
                _format_median_delta(r["med_delta"]),
                _format_p_value(p),
            ])
    return rows


def export_fig5_panel_e_predictive_table(
    *,
    use_cached_csv: bool = True,
    out_dirs: tuple[Path, Path] | None = None,
) -> dict[str, Path]:
    """Export Fig.5E as category × direction summary table."""
    if out_dirs is None:
        out_dirs = (PAPER, OUT)
    plt.rcParams.update(PAPER_RC)
    rec_m = _load_rec(direction="MtoF", use_cached_csv=use_cached_csv)
    rec_f = _load_rec(direction="FtoM", use_cached_csv=use_cached_csv)
    table_rows = build_fig5_panel_e_predictive_table_rows(rec_m, rec_f)
    written: dict[str, Path] = {}

    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = "Paper_Fig5" if out_dir.name == "Paper_fig" else "Fig5"
        stem = f"{prefix}_panelE_predictive_category_highgap_table"

        fig, ax = plt.subplots(figsize=(9.2, 4.2))
        ax.axis("off")
        panel_label(ax, "E")

        tbl = ax.table(
            cellText=table_rows[1:],
            colLabels=table_rows[0],
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.55)

        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#cccccc")
            cell.set_linewidth(0.6)
            if row == 0:
                cell.set_facecolor("#eceff1")
                cell.set_text_props(fontweight="bold")
                continue
            cat = table_rows[row][0]
            if col == 0:
                cell.set_facecolor("#fafafa")
                cell.get_text().set_color(CAT_COLORS.get(cat, "0.2"))
                cell.get_text().set_fontweight("bold")
            elif col == 1:
                cell.get_text().set_color("#1565c0" if table_rows[row][1] == "M→F" else "#c62828")

        fig.suptitle(
            "Predictive Ridge vs Φ by category (high raw-gap tercile; theme-CV)",
            fontsize=10.5, y=0.98,
        )
        fig.text(
            0.5, 0.03,
            "Δ = median(ridge_l2 − phi_l2) per query; negative = Ridge beats Φ. "
            "p = two-sided Wilcoxon on paired Δ within cell.",
            ha="center", fontsize=7.5, color="0.45",
        )
        fig.subplots_adjust(top=0.86, bottom=0.10, left=0.04, right=0.96)

        png = out_dir / f"{stem}.png"
        svg = out_dir / f"{stem}.svg"
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written[stem] = png
        print(f"Saved {png}\nSaved {svg}")

        manifest = {
            "table_rows": table_rows,
            "rec_m_stats": _category_high_gap_stats(rec_m),
            "rec_f_stats": _category_high_gap_stats(rec_f),
        }
        json_path = out_dir / f"{stem}_manifest.json"
        json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    return written


def draw_category_beat_bars_high_gap(
    ax,
    rec: pd.DataFrame,
    *,
    title: str | None = None,
    show_xticklabels: bool = True,
) -> None:
    """Beat rate bars + Wilson CI (high raw-gap tercile)."""
    stats = _category_high_gap_stats(rec)
    x = np.arange(len(CAT_ORDER), dtype=float)
    colors = [CAT_COLORS[c] for c in CAT_ORDER]
    beats = [r["beat"] for r in stats]
    ns = [r["n"] for r in stats]
    ci_lo = [r["ci_lo"] for r in stats]
    ci_hi = [r["ci_hi"] for r in stats]

    yerr_lo = [b - lo if np.isfinite(b) and np.isfinite(lo) else 0 for b, lo in zip(beats, ci_lo)]
    yerr_hi = [hi - b if np.isfinite(b) and np.isfinite(hi) else 0 for b, hi in zip(beats, ci_hi)]
    bars = ax.bar(
        x, beats, color=colors, edgecolor="0.25", width=0.68, zorder=2,
        yerr=[yerr_lo, yerr_hi], capsize=3.5,
        error_kw=dict(ecolor="0.25", lw=1.0, capthick=1.0),
    )
    ax.axhline(0.5, color="0.55", ls="--", lw=1.0, zorder=1)
    ax.set_ylabel("Ridge beat Φ rate")
    ax.set_ylim(0, 0.88)
    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels(CAT_ORDER)
    else:
        ax.set_xticklabels([])
    if title:
        ax.set_title(title, fontsize=8.5)

    for bar, n, beat in zip(bars, ns, beats):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(beat + 0.06, 0.84),
            f"n={n}",
            ha="center", va="bottom", fontsize=6.8, color="0.35",
        )
    _ = bars


def draw_category_median_delta_bars_high_gap(
    ax,
    rec: pd.DataFrame,
    *,
    show_xticklabels: bool = True,
) -> None:
    """Median paired Δ bars, sign-coloured (high raw-gap tercile)."""
    stats = _category_high_gap_stats(rec)
    x = np.arange(len(CAT_ORDER), dtype=float)
    med_delta = [r["med_delta"] for r in stats]
    delta_colors = [
        COL_RIDGE if np.isfinite(d) and d < 0 else COL_PHI
        for d in med_delta
    ]
    bars = ax.bar(
        x, med_delta, color=delta_colors, edgecolor="0.25", width=0.68, zorder=2,
    )
    ax.axhline(0.0, color="0.55", ls="--", lw=1.0, zorder=1)
    ax.set_ylabel("Median paired Δ\n(Ridge − Φ L2)")
    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels(CAT_ORDER)
    else:
        ax.set_xticklabels([])
    yabs = max((abs(d) for d in med_delta if np.isfinite(d)), default=0.4)
    ax.set_ylim(-yabs * 1.35, yabs * 1.35)

    for bar, d in zip(bars, med_delta):
        if not np.isfinite(d):
            continue
        yoff = 0.04 if d >= 0 else -0.04
        va = "bottom" if d >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            d + yoff,
            f"{d:+.2f}",
            ha="center", va=va, fontsize=6.8, color="0.25",
            fontweight="bold" if d < 0 else "normal",
        )
    _ = bars


def draw_category_beat_high_gap(ax, rec: pd.DataFrame) -> None:
    """Panel C: Ridge beat Φ rate by query category (high raw-gap tercile only)."""
    stats = _category_high_gap_stats(rec)
    x = np.arange(len(CAT_ORDER), dtype=float)
    colors = [CAT_COLORS[c] for c in CAT_ORDER]
    beats = [r["beat"] for r in stats]
    ns = [r["n"] for r in stats]
    ci_lo = [r["ci_lo"] for r in stats]
    ci_hi = [r["ci_hi"] for r in stats]
    med_delta = [r["med_delta"] for r in stats]

    yerr_lo = [b - lo if np.isfinite(b) and np.isfinite(lo) else 0 for b, lo in zip(beats, ci_lo)]
    yerr_hi = [hi - b if np.isfinite(b) and np.isfinite(hi) else 0 for b, hi in zip(beats, ci_hi)]
    bars = ax.bar(
        x, beats, color=colors, edgecolor="0.25", width=0.68, zorder=2,
        yerr=[yerr_lo, yerr_hi], capsize=3.5, error_kw=dict(ecolor="0.25", lw=1.0, capthick=1.0),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(CAT_ORDER, rotation=0)
    ax.set_ylabel("Ridge beat Φ rate")
    ax.set_ylim(0, 0.95)
    ax.axhline(0.5, color="0.75", ls=":", lw=0.9)
    ax.set_title("High raw-gap tercile by category")

    for i, (b, n, d) in enumerate(zip(bars, ns, med_delta)):
        rate = beats[i]
        ax.text(
            b.get_x() + b.get_width() / 2, rate + 0.03,
            f"{rate:.0%}\n(n={n})",
            ha="center", va="bottom", fontsize=7.5,
        )
        if np.isfinite(d):
            ax.text(
                b.get_x() + b.get_width() / 2, 0.04,
                f"Δmd={d:+.2f}",
                ha="center", va="bottom", fontsize=6.5, color="0.35",
            )
    _ = bars


def category_jitter_scatter(
    ax,
    x_center: float,
    y: np.ndarray,
    color: str,
    *,
    rng: np.random.Generator,
    x_spread: float = 0.28,
    s: float = 8,
    alpha: float = 0.32,
) -> None:
    """Jittered per-query scatter at a categorical x position."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return
    jitter = rng.uniform(-x_spread, x_spread, size=len(y))
    ax.scatter(
        x_center + jitter, y, s=s, color=color, alpha=alpha,
        edgecolors="none", zorder=2,
    )


def draw_category_beat_scatter(ax, rec: pd.DataFrame) -> None:
    """Per-query beat outcomes (0/1 strip) + category mean + Wilson CI."""
    h = rec[rec.raw_gap_tercile == "high"]
    stats = _category_high_gap_stats(rec)
    x = np.arange(len(CAT_ORDER), dtype=float)
    rng = np.random.default_rng(5)

    for xi, cat, st in zip(x, CAT_ORDER, stats):
        s = h[h.category == cat]
        col = CAT_COLORS[cat]
        y = s.beat_phi_ridge.astype(float).to_numpy()
        category_jitter_scatter(ax, float(xi), y, col, rng=rng)

        beat = st["beat"]
        ci_lo, ci_hi = st["ci_lo"], st["ci_hi"]
        if np.isfinite(beat):
            ax.hlines(beat, xi - 0.22, xi + 0.22, colors=col, lw=1.8, zorder=5)
            ax.scatter(
                [xi], [beat], s=48, color=col, edgecolors="white",
                linewidths=0.7, zorder=6,
            )
        if np.isfinite(ci_lo) and np.isfinite(ci_hi):
            ax.vlines(xi, ci_lo, ci_hi, colors="0.35", lw=1.2, zorder=4, alpha=0.9)
        n = st["n"]
        if np.isfinite(beat):
            ax.text(
                xi, 1.06, f"{beat:.0%}\n(n={n})",
                ha="center", va="bottom", fontsize=7.0, color=col,
            )

    ax.axhline(0.5, color="0.55", ls="--", lw=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(CAT_ORDER)
    ax.set_ylabel("Ridge beats Φ\n(per query: 0/1)")
    ax.set_ylim(-0.10, 1.18)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["no", "0.5", "yes"])
    ax.set_title(
        "Predictive Ridge vs Φ (high raw-gap tercile; theme-CV)",
        fontsize=9,
    )
    ax.text(
        0.02, 0.04,
        "points = queries; line = category mean; whisker = Wilson 95% CI",
        transform=ax.transAxes, fontsize=6.5, color="0.45", ha="left", va="bottom",
    )


def draw_category_delta_scatter(ax, rec: pd.DataFrame) -> None:
    """Per-query paired Δ scatter + category median."""
    h = rec[rec.raw_gap_tercile == "high"]
    x = np.arange(len(CAT_ORDER), dtype=float)
    rng = np.random.default_rng(9)
    all_delta: list[float] = []

    for xi, cat in zip(x, CAT_ORDER):
        s = h[h.category == cat]
        col = CAT_COLORS[cat]
        d = (s.ridge_l2 - s.phi_l2).to_numpy(float)
        all_delta.extend(d[np.isfinite(d)].tolist())
        category_jitter_scatter(ax, float(xi), d, col, rng=rng)

        d_ok = d[np.isfinite(d)]
        if len(d_ok) == 0:
            continue
        med = float(np.median(d_ok))
        ax.scatter(
            [xi], [med], s=48, color=col, edgecolors="white",
            linewidths=0.7, zorder=6,
        )
        ax.hlines(med, xi - 0.18, xi + 0.18, colors="0.15", lw=1.1, zorder=5)
        ax.text(
            xi, med + (0.05 if med >= 0 else -0.05),
            f"{med:+.2f}",
            ha="center",
            va="bottom" if med >= 0 else "top",
            fontsize=7.0,
            color="0.25",
            fontweight="bold" if med < 0 else "normal",
        )

    ax.axhline(0.0, color="0.55", ls="--", lw=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(CAT_ORDER)
    ax.set_ylabel("Paired Δ (Ridge − Φ L2)")
    yabs = max(abs(np.percentile(all_delta, 2)), abs(np.percentile(all_delta, 98)), 0.35)
    ax.set_ylim(-yabs * 1.18, yabs * 1.18)
    ax.text(
        0.02, 0.04,
        "Δ = ridge_l2 − phi_l2 per query; marker = category median",
        transform=ax.transAxes, fontsize=6.5, color="0.45", ha="left", va="bottom",
    )


def draw_fig5_panel_e_predictive_category(
    ax_top,
    ax_bot,
    rec: pd.DataFrame,
    *,
    direction_title: str | None = None,
) -> None:
    """Fig. 5E sub-column: beat bars + median Δ bars for one direction."""
    draw_category_beat_bars_high_gap(ax_top, rec, title=direction_title, show_xticklabels=False)
    draw_category_median_delta_bars_high_gap(ax_bot, rec, show_xticklabels=True)
    ax_bot.text(
        0.02, 0.04,
        "Δ = median(ridge_l2 − phi_l2) per query",
        transform=ax_bot.transAxes, fontsize=6.2, color="0.45", ha="left", va="bottom",
    )


def draw_category_phi_ridge_violin(ax, rec: pd.DataFrame) -> None:
    """Violin + scatter: Φ vs Ridge L2 by category (high raw-gap tercile)."""
    h = rec[rec.raw_gap_tercile == "high"]
    methods = [("Φ", "phi_l2", COL_PHI), ("Ridge", "ridge_l2", COL_RIDGE)]
    offsets = np.array([-0.22, 0.22])
    width = 0.20

    all_data: list[np.ndarray] = []
    all_pos: list[float] = []
    all_cols: list[str] = []

    for ti, cat in enumerate(CAT_ORDER):
        s = h[h.category == cat]
        for oi, (_, col, color) in enumerate(methods):
            all_data.append(s[col].to_numpy(float))
            all_pos.append(float(ti) + offsets[oi])
            all_cols.append(color)

    violin_scatter(
        ax, all_data, all_pos, all_cols,
        widths=width, seed=11, max_points=120, alpha_pts=0.20, s=6,
    )

    ax.set_xticks(range(len(CAT_ORDER)))
    ax.set_xticklabels(CAT_ORDER)
    ax.set_ylabel("L2 error (female target VA)")
    ax.set_title("Per-query L2 by method", fontsize=9)
    ax.set_ylim(0, None)
    ymax = max(np.percentile(d[np.isfinite(d)], 98) for d in all_data if len(d))
    ax.set_ylim(0, min(ymax * 1.08, 3.0))

    ax.legend(
        handles=[
            Patch(facecolor=COL_PHI, edgecolor="0.35", alpha=0.45, label="Φ"),
            Patch(facecolor=COL_RIDGE, edgecolor="0.35", alpha=0.45, label="Ridge"),
        ],
        fontsize=7,
        loc="upper right",
        frameon=False,
    )


def draw_category_delta_violin(ax, rec: pd.DataFrame) -> None:
    """Violin + scatter: paired Δ = ridge_l2 − phi_l2 by category (high raw gap)."""
    h = rec[rec.raw_gap_tercile == "high"]
    x = np.arange(len(CAT_ORDER), dtype=float)
    colors = [CAT_COLORS[c] for c in CAT_ORDER]

    all_data: list[np.ndarray] = []
    for cat in CAT_ORDER:
        s = h[h.category == cat]
        all_data.append((s.ridge_l2 - s.phi_l2).to_numpy(float))

    violin_scatter(
        ax, all_data, x, colors,
        widths=0.68, seed=17, max_points=120, alpha_pts=0.22, s=6,
    )

    ax.axhline(0.0, color="0.55", ls="--", lw=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(CAT_ORDER)
    ax.set_ylabel("Paired Δ (Ridge − Φ L2)")
    ax.set_title("Per-query paired difference", fontsize=9)

    yabs = max(
        abs(np.percentile(d[np.isfinite(d)], 2)) if len(d[np.isfinite(d)]) else 0
        for d in all_data
    )
    yabs = max(yabs, abs(max((np.percentile(d[np.isfinite(d)], 98) if len(d[np.isfinite(d)]) else 0) for d in all_data)))
    yabs = max(yabs, 0.35)
    ax.set_ylim(-yabs * 1.15, yabs * 1.15)

    for xi, cat, arr in zip(x, CAT_ORDER, all_data):
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        med = float(np.median(arr))
        beat = float((arr < 0).mean())
        ax.text(
            xi,
            yabs * 1.02,
            f"beat {beat:.0%}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=CAT_COLORS[cat],
        )
        ax.text(
            xi,
            -yabs * 1.08,
            f"md {med:+.2f}",
            ha="center",
            va="top",
            fontsize=6.8,
            color="0.35",
        )


def draw_oracle_ij_beat_heatmap(ax, rec: pd.DataFrame) -> None:
    """Panel D: oracle i→j beat rate (high raw gap × Ridge beats Φ)."""
    pairs = pd.read_csv(PAIRS_CV)
    pm = pairs.merge(
        rec[["image_id", "beat_phi_ridge", "raw_gap_tercile"]],
        left_on="i_image_id", right_on="image_id", how="left",
    )
    h = pm[pm.raw_gap_tercile == "high"]

    n_mat = np.zeros((len(CAT_ORDER), len(CAT_ORDER)), int)
    beat_mat = np.full((len(CAT_ORDER), len(CAT_ORDER)), np.nan)
    for ii, ic in enumerate(CAT_ORDER):
        for jj, jc in enumerate(CAT_ORDER):
            sub = h[(h.i_category == ic) & (h.j_category == jc)]
            n_mat[ii, jj] = len(sub)
            if len(sub) >= 3:
                beat_mat[ii, jj] = float(sub.beat_phi_ridge.mean())

    im = ax.imshow(beat_mat, origin="upper", cmap="YlGn", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(CAT_ORDER)))
    ax.set_yticks(range(len(CAT_ORDER)))
    ax.set_xticklabels(CAT_ORDER, rotation=45, ha="right")
    ax.set_yticklabels(CAT_ORDER)
    ax.set_xlabel("Oracle j category")
    ax.set_ylabel("Query i category")
    ax.set_title("Ridge beat Φ (high gap; cell n≥3)")

    for ii in range(len(CAT_ORDER)):
        for jj in range(len(CAT_ORDER)):
            n = n_mat[ii, jj]
            if n == 0:
                txt = "—"
            elif np.isfinite(beat_mat[ii, jj]):
                txt = f"{beat_mat[ii, jj]:.0%}\n(n={n})"
            else:
                txt = f"n={n}"
            ax.text(
                jj, ii, txt, ha="center", va="center", fontsize=7,
                color="0.15" if (np.isfinite(beat_mat[ii, jj]) and beat_mat[ii, jj] > 0.55) else "0.35",
            )
    fig = ax.figure
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("beat rate", fontsize=7)


def _write_meta(stem: str, panels: dict[str, str], verdict) -> None:
    meta = {
        "stem": stem,
        "intended_figure": "Fig.6 / Supplement (not Fig.5)",
        "panels": panels,
        "per_image_csv": str(EQ / "predictive_translation_per_image.csv"),
        "verdict": verdict,
    }
    (OUT / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def plot_prototype(*, use_cached_csv: bool = True) -> tuple[Path, Path]:
    plt.rcParams.update(PAPER_RC)
    rec = _load_rec(direction="MtoF", use_cached_csv=use_cached_csv)

    fig = plt.figure(figsize=(14.5, 5.4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.2, 1.45], wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")
    draw_overall_violin(ax_a, rec)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")
    draw_raw_gap_tercile_violin(ax_b, rec)

    draw_spatial_dual(fig, gs[0, 2], rec, grid=1.0)
    spatial_axes = [a for a in fig.axes if a.get_xlabel() == "Valence (female target)"]
    if spatial_axes:
        panel_label(spatial_axes[0], "C")

    fig.suptitle(
        "Fig. 6 candidate — Predictive case translation "
        "(female target; theme-CV; target unseen at partner choice)",
        fontsize=10.5, y=1.02,
    )
    png, svg = save_dual(fig, "Predictive_translation_prototype")

    ridge_json = EQ / "predictive_translation_ridge.json"
    verdict = json.loads(ridge_json.read_text()).get("verdict") if ridge_json.exists() else None
    _write_meta(
        "Predictive_translation_prototype",
        {
            "A": "overall L2 violin+scatter",
            "B": "raw gap tercile violin+scatter",
            "C": "VA co-localization",
        },
        verdict,
    )
    print(f"Saved {png}\nSaved {svg}")
    return png, svg


def plot_prototype_bar_sd(*, use_cached_csv: bool = True) -> tuple[Path, Path]:
    """Alternate layout: mean ± SD bars (median ticks overlaid); Panel C unchanged."""
    plt.rcParams.update(PAPER_RC)
    rec = _load_rec(direction="MtoF", use_cached_csv=use_cached_csv)

    fig = plt.figure(figsize=(14.5, 5.4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.2, 1.45], wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")
    draw_overall_bar_sd(ax_a, rec)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")
    draw_raw_gap_tercile_bar_sd(ax_b, rec)

    draw_spatial_dual(fig, gs[0, 2], rec, grid=1.0)
    spatial_axes = [a for a in fig.axes if a.get_xlabel() == "Valence (female target)"]
    if spatial_axes:
        panel_label(spatial_axes[0], "C")

    fig.suptitle(
        "Fig. 6 candidate — Predictive case translation "
        "(bar + SD; female target; theme-CV)",
        fontsize=10.5, y=1.02,
    )
    stem = "Predictive_translation_prototype_bar_sd"
    png, svg = save_dual(fig, stem)

    ridge_json = EQ / "predictive_translation_ridge.json"
    verdict = json.loads(ridge_json.read_text()).get("verdict") if ridge_json.exists() else None
    _write_meta(
        stem,
        {
            "A": "overall L2 mean±SD bars (median ticks)",
            "B": "raw gap tercile mean±SD bars (median ticks)",
            "C": "VA co-localization",
        },
        verdict,
    )
    print(f"Saved {png}\nSaved {svg}")
    return png, svg


def export_fig5_panel_e_predictive_category(
    *,
    use_cached_csv: bool = True,
    out_dirs: tuple[Path, Path] | None = None,
    as_table: bool = True,
) -> dict[str, Path]:
    """Export Fig.5E (default: summary table; optional: 2×2 bar panels)."""
    if as_table:
        return export_fig5_panel_e_predictive_table(
            use_cached_csv=use_cached_csv, out_dirs=out_dirs,
        )
    if out_dirs is None:
        out_dirs = (PAPER, OUT)
    plt.rcParams.update(PAPER_RC)
    rec_m = _load_rec(direction="MtoF", use_cached_csv=use_cached_csv)
    rec_f = _load_rec(direction="FtoM", use_cached_csv=use_cached_csv)
    written: dict[str, Path] = {}

    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = "Paper_Fig5" if out_dir.name == "Paper_fig" else "Fig5"
        stem = f"{prefix}_panelE_predictive_category_highgap"

        fig = plt.figure(figsize=(10.8, 6.6))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05], hspace=0.38, wspace=0.28)

        ax_mt_beat = fig.add_subplot(gs[0, 0])
        ax_fm_beat = fig.add_subplot(gs[0, 1], sharey=ax_mt_beat)
        ax_mt_delta = fig.add_subplot(gs[1, 0])
        ax_fm_delta = fig.add_subplot(gs[1, 1], sharey=ax_mt_delta)

        panel_label(ax_mt_beat, "E")
        draw_category_beat_bars_high_gap(
            ax_mt_beat, rec_m, title="M→F (female target)", show_xticklabels=False,
        )
        draw_category_beat_bars_high_gap(
            ax_fm_beat, rec_f, title="F→M (male target)", show_xticklabels=False,
        )
        draw_category_median_delta_bars_high_gap(ax_mt_delta, rec_m, show_xticklabels=True)
        draw_category_median_delta_bars_high_gap(ax_fm_delta, rec_f, show_xticklabels=True)
        for ax in (ax_mt_delta, ax_fm_delta):
            ax.text(
                0.02, 0.04,
                "Δ = median(ridge_l2 − phi_l2) per query",
                transform=ax.transAxes, fontsize=6.2, color="0.45", ha="left", va="bottom",
            )

        fig.suptitle(
            "Predictive Ridge vs Φ by category (high raw-gap tercile; theme-CV)",
            fontsize=10, y=0.98,
        )
        fig.subplots_adjust(top=0.90, bottom=0.08, hspace=0.40, wspace=0.26)

        png = out_dir / f"{stem}.png"
        svg = out_dir / f"{stem}.svg"
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written[stem] = png
        print(f"Saved {png}\nSaved {svg}")

    return written


def export_predictive_category_violin_highgap(
    *,
    use_cached_csv: bool = True,
    out_dirs: tuple[Path, Path] | None = None,
) -> dict[str, Path]:
    """Separate figure: violin + scatter L2 and paired Δ by category (high raw gap)."""
    if out_dirs is None:
        out_dirs = (PAPER, OUT)
    plt.rcParams.update(PAPER_RC)
    rec = _load_rec(direction="MtoF", use_cached_csv=use_cached_csv)
    written: dict[str, Path] = {}

    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = "Paper_SuppFig" if out_dir.name == "Paper_fig" else "SuppFig"
        stem = f"{prefix}_predictive_category_violin_highgap"

        fig = plt.figure(figsize=(5.8, 6.8))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1.05, 1.05], hspace=0.34)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_bot = fig.add_subplot(gs[1, 0])
        panel_label(ax_top, "A")
        panel_label(ax_bot, "B")
        draw_category_phi_ridge_violin(ax_top, rec)
        draw_category_delta_violin(ax_bot, rec)
        fig.suptitle(
            "Predictive Ridge vs Φ by category (high raw-gap tercile; theme-CV)",
            fontsize=10, y=0.98,
        )
        fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.36)

        png = out_dir / f"{stem}.png"
        svg = out_dir / f"{stem}.svg"
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        written[stem] = png
        print(f"Saved {png}\nSaved {svg}")

    return written


def plot_supp_predictive_figure(*, use_cached_csv: bool = True) -> tuple[Path, Path]:
    """Supplementary Fig. X: 2×2 predictive translation (A–D)."""
    plt.rcParams.update(PAPER_RC)
    rec = _load_rec(direction="MtoF", use_cached_csv=use_cached_csv)

    fig = plt.figure(figsize=(12.5, 9.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")
    draw_overall_bar_sd(ax_a, rec)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")
    draw_raw_gap_tercile_bar_sd(ax_b, rec)

    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "C")
    draw_category_beat_high_gap(ax_c, rec)

    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "D")
    draw_oracle_ij_beat_heatmap(ax_d, rec)

    fig.suptitle(
        "Supplementary Fig. X — Predictive case translation "
        "(female target; theme-CV; target unseen at partner choice)",
        fontsize=10.5, y=0.98,
    )

    stem = "Paper_SuppFig_predictive_translation"
    for out_dir in (OUT, PAPER):
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / f"{stem}.png"
        svg = out_dir / f"{stem}.svg"
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white")

    plt.close(fig)

    ridge_json = EQ / "predictive_translation_ridge.json"
    verdict = json.loads(ridge_json.read_text()).get("verdict") if ridge_json.exists() else None
    meta_path = OUT / f"{stem}_meta.json"
    _write_meta(
        stem.replace("Paper_", ""),
        {
            "A": "overall L2 mean±SD",
            "B": "raw gap tercile mean±SD + beat line",
            "C": "category beat high gap",
            "D": "oracle i→j beat heatmap high gap",
        },
        verdict,
    )
    png = PAPER / f"{stem}.png"
    svg = PAPER / f"{stem}.svg"
    print(f"Saved {png}\nSaved {OUT / f'{stem}.png'}\nSaved {svg}")
    return png, svg


if __name__ == "__main__":
    plot_prototype()
    plot_prototype_bar_sd()
    plot_supp_predictive_figure()
    export_fig5_panel_e_predictive_category(as_table=True)
    export_fig5_panel_e_predictive_category(as_table=False)
    export_predictive_category_violin_highgap()
