#!/usr/bin/env python3
"""
Paper Fig. 2 (Manuscripts.md numbering) — perspective-taking geometry.

A  five-point mapping (pooled): M→F | F→M side-by-side; point + 95% CI; oracle−affine bracket
   dimensional V/A breakdown → Paper_SuppFig_five_point_by_dimension (order not preserved)
B  gender bridge Φ (no FP ellipse; poorly determined)
C  culture bridge Φ (subject-bootstrap FP CI ellipse); shared colour/arrow scale with B
D  random-k generalization (log-x, ceiling, k=3 valley arrow); near/far → Supp
E  shareable-landmark density + basin labels + greedy anchors (4th = duplicate marker)

Outputs (Paper_fig/):
  Paper_Fig2_perspective_geometry.{png,svg}
  Paper_Fig2_panel{A..E}_*.{png,svg}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from generate_paper_figures import (  # noqa: E402
    ANC,
    CVAE,
    OUT,
    PBA,
    VA_LIM_FIG3,
    _disp_grid,
    _estimate_shared_affine_basins,
    _kde_on_lim,
    _load_anchor_point_density_data,
    _load_z5_minimal_anchor_data,
    _oasis_density,
    _positive_quantile_levels,
    panel_label,
    save_dual,
)

PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

METHODS = ["no_transport", "global_shift", "linear_shift", "ot", "oracle"]
METHOD_LABELS = ["No\ntransform", "Global\nshift", "Affine\nΦ", "Sinkhorn\nOT", "Oracle"]
METHOD_COLORS = ["#9e9e9e", "#66bb6a", "#1e88e5", "#e57373", "#6d4c41"]

QUIVER_SCALE = 26.0
QUIVER_N = 12
CMAP_DISP = "YlOrRd"


def _save_panel(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── A ─────────────────────────────────────────────────────────────────
SERIES_KEYS = [
    ("R2_mean", "R2_mean_ci", "pooled"),
    ("R2_valence", "R2_valence_ci", "V"),
    ("R2_arousal", "R2_arousal_ci", "A"),
]
SERIES_COLORS = {"pooled": "#212121", "V": "#1565c0", "A": "#6a1b9a"}
SERIES_MARKERS = {"pooled": "o", "V": "s", "A": "D"}


def draw_panel_a_direction(
    ax,
    scores: dict,
    *,
    title: str,
    show_label: bool = False,
    ylim: tuple[float, float] = (0.30, 0.82),
) -> None:
    """Pooled R² ± bootstrap 95% CI for one direction; bracket oracle−affine."""
    xs = np.arange(len(METHODS), dtype=float)
    means, los, his = [], [], []
    for m in METHODS:
        s = scores[m]
        means.append(float(s["R2_mean"]))
        ci = s["R2_mean_ci"]
        los.append(float(ci[0]))
        his.append(float(ci[1]))
    means = np.asarray(means)
    yerr = np.vstack([means - np.asarray(los), np.asarray(his) - means])

    ax.errorbar(
        xs, means, yerr=yerr, fmt="o", ms=7.5, lw=0,
        ecolor="0.35", elinewidth=1.4, capsize=4.5, capthick=1.2,
        zorder=3,
    )
    for x, y, c in zip(xs, means, METHOD_COLORS):
        ax.scatter([x], [y], s=55, c=c, edgecolors="white", linewidths=0.8, zorder=4)

    i_lin = METHODS.index("linear_shift")
    i_orc = METHODS.index("oracle")
    y_top = max(his[i_lin], his[i_orc]) + 0.035
    ax.plot(
        [i_lin, i_lin, i_orc, i_orc],
        [his[i_lin] + 0.012, y_top, y_top, his[i_orc] + 0.012],
        color="0.25", lw=1.1, clip_on=False,
    )
    delta = means[i_orc] - means[i_lin]
    ax.text(
        0.5 * (i_lin + i_orc), y_top + 0.008,
        f"Δ = {delta:.3f}",
        ha="center", va="bottom", fontsize=7.5, color="0.2", fontweight="bold",
    )

    ax.set_xticks(xs, METHOD_LABELS)
    if show_label:
        ax.set_ylabel("Held-out R² mean")
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.55, len(METHODS) - 0.45)
    ax.set_title(title, fontsize=9)
    ax.axhline(means[i_lin], color="#1e88e5", ls=":", lw=0.8, alpha=0.55, zorder=1)
    ot_aff = means[METHODS.index("ot")] - means[i_lin]
    ax.text(
        0.02, 0.04,
        f"OT − Φ = {ot_aff:+.3f}",
        transform=ax.transAxes, fontsize=7, color="0.3",
    )


def draw_panel_a(ax_left, ax_right, scores_f: dict, scores_m: dict) -> None:
    """Fig. 2A: pooled five-point, two directions side by side."""
    panel_label(ax_left, "A")
    draw_panel_a_direction(
        ax_left, scores_f,
        title="M→F (female target; pooled)",
        show_label=True,
    )
    draw_panel_a_direction(
        ax_right, scores_m,
        title="F→M (male target; pooled)",
        show_label=False,
    )


def draw_supp_five_point_by_series(ax, scores: dict, *, title: str, show_ylabel: bool = False) -> None:
    """5 methods × 3 series (pooled / V / A) grouped dots + 95% CI."""
    n_m = len(METHODS)
    offsets = {"pooled": -0.22, "V": 0.0, "A": 0.22}
    for mi, m in enumerate(METHODS):
        s = scores[m]
        for key, ci_key, lab in SERIES_KEYS:
            x = mi + offsets[lab]
            y = float(s[key])
            lo, hi = [float(v) for v in s[ci_key]]
            ax.errorbar(
                [x], [y], yerr=[[y - lo], [hi - y]],
                fmt="none", ecolor=SERIES_COLORS[lab], elinewidth=1.2,
                capsize=3.0, capthick=1.0, zorder=3,
            )
            ax.scatter(
                [x], [y], s=36, c=SERIES_COLORS[lab],
                marker=SERIES_MARKERS[lab], edgecolors="white",
                linewidths=0.6, zorder=4,
            )
    ax.set_xticks(np.arange(n_m), METHOD_LABELS)
    ax.set_xlim(-0.6, n_m - 0.4)
    if show_ylabel:
        ax.set_ylabel("Held-out R²")
    ax.set_title(title, fontsize=9)
    ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    handles = [
        Line2D([0], [0], marker=SERIES_MARKERS[k], color=SERIES_COLORS[k],
               lw=0, markersize=7, label=k)
        for k in ("pooled", "V", "A")
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="lower right")


def make_supp_fig2a_dimension_direction(scores_f: dict, scores_m: dict) -> None:
    """Supplementary: dimensional breakdown (order not preserved → not main text)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    draw_supp_five_point_by_series(
        axes[0], scores_f,
        title="M→F · pooled / V / A (bootstrap 95% CI)",
        show_ylabel=True,
    )
    draw_supp_five_point_by_series(
        axes[1], scores_m,
        title="F→M · pooled / V / A (bootstrap 95% CI)",
        show_ylabel=False,
    )
    axes[0].set_ylim(0.10, 0.85)
    fig.suptitle(
        "Supplementary | Five-point mapping by dimension\n"
        "Method order is not identical across V vs A; OT does not beat Φ in either dimension",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    _save_panel(fig, "Paper_SuppFig_five_point_by_dimension")



# ── B / C terrain ─────────────────────────────────────────────────────
def _displacement_vmax(bridges: dict, lim=VA_LIM_FIG3) -> float:
    vmax = 0.0
    for key in ("gender", "culture"):
        _, _, d = _disp_grid(bridges[key]["A"], bridges[key]["b"], lim=lim)
        vmax = max(vmax, float(np.nanmax(d)))
    return max(vmax, 1e-6)


def draw_bridge_panel(
    ax,
    *,
    A, b, fp, other_fp, dens, oasis_centroid,
    title: str,
    self_lab: str,
    other_lab: str,
    self_col: str,
    other_col: str,
    vmax: float,
    sv: list[float],
    ci_ellipse: tuple[float, float, float, float] | None = None,
    note: str | None = None,
) -> object:
    """Shared colour scale (0..vmax) and quiver scale for gender/culture bridges."""
    vv, aa, d = _disp_grid(A, b, lim=VA_LIM_FIG3)
    fill_levels = np.linspace(0.0, vmax, 14)
    cf = ax.contourf(vv, aa, d, levels=fill_levels, cmap=CMAP_DISP, alpha=0.88, extend="max")
    cs = ax.contour(
        vv, aa, d,
        levels=np.linspace(0.2 * vmax, 0.9 * vmax, 5),
        colors="0.25", linewidths=0.8, alpha=0.8,
    )
    ax.clabel(cs, inline=True, fontsize=5.5, fmt="%.2f")

    if dens is not None:
        ax.scatter(dens[:, 0], dens[:, 1], s=4, c="#1a237e", alpha=0.14, edgecolors="none", zorder=2)

    vq, aq, _ = _disp_grid(A, b, n=QUIVER_N, lim=VA_LIM_FIG3)
    pts = np.column_stack([vq.ravel(), aq.ravel()])
    uv = pts @ np.asarray(A, float).T + np.asarray(b, float) - pts
    uvn = uv / np.clip(np.linalg.norm(uv, axis=1, keepdims=True), 1e-9, None)
    ax.quiver(
        pts[:, 0], pts[:, 1], uvn[:, 0], uvn[:, 1],
        color="0.22", alpha=0.55, pivot="mid",
        scale=QUIVER_SCALE, width=0.004, headwidth=4, zorder=3,
    )

    # OASIS centroid
    ax.scatter(
        [oasis_centroid[0]], [oasis_centroid[1]],
        s=70, marker="+", c="0.15", linewidths=1.6, zorder=5,
        label=f"OASIS centroid ({oasis_centroid[0]:.2f},{oasis_centroid[1]:.2f})",
    )

    if ci_ellipse is not None:
        cx, cy, w, h = ci_ellipse
        ell = Ellipse(
            (cx, cy), width=w, height=h,
            facecolor=self_col, edgecolor=self_col,
            alpha=0.18, lw=1.4, zorder=4,
            label="FP subject-bootstrap 95% CI",
        )
        ax.add_patch(ell)

    ax.scatter(
        *fp, s=160, marker="*", c=self_col, edgecolors="white", linewidths=1.1,
        label=f"{self_lab} FP ({fp[0]:.2f},{fp[1]:.2f})", zorder=6,
    )
    ax.scatter(
        *other_fp, s=70, marker="o", facecolors="none", edgecolors=other_col, linewidths=1.6,
        label=f"{other_lab} FP", zorder=5,
    )

    ax.text(
        0.98, 0.97,
        f"σ₁ = {sv[0]:.2f}\nσ₂ = {sv[1]:.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.92),
        zorder=7,
    )
    if note:
        ax.text(
            0.02, 0.03, note,
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color="#b71c1c",
            bbox=dict(boxstyle="round,pad=0.22", fc="#fff8e1", ec="#ffcc80", alpha=0.95),
            zorder=7,
        )

    ax.set_xlim(*VA_LIM_FIG3)
    ax.set_ylim(*VA_LIM_FIG3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(title, fontsize=8.8)
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    return cf


# ── D ─────────────────────────────────────────────────────────────────
def draw_panel_d(ax, m2f: pd.DataFrame, f2m: pd.DataFrame, z3: dict) -> None:
    for df, lab, col in [
        (m2f, "M→F", "#1565c0"),
        (f2m, "F→M", "#c62828"),
    ]:
        ax.plot(df["k"], df["R2_median"], "o-", color=col, lw=1.8, ms=5, label=lab, zorder=3)
        ax.fill_between(df["k"], df["R2_q025"], df["R2_q975"], color=col, alpha=0.12, zorder=2)

    ceil = float(m2f["ceiling"].iloc[0])
    ax.axhline(ceil, color="0.25", ls="--", lw=1.15, label=f"ceiling {ceil:.2f}", zorder=1)
    ax.axhline(0.9 * ceil, color="0.55", ls=":", lw=0.95, label="90% × ceiling", zorder=1)

    k_star = z3["bridges"]["gender_M_to_F"].get("k_star")
    if k_star is not None:
        ax.axvline(float(k_star), color="#6a1b9a", ls="--", alpha=0.7, lw=1.0,
                   label=f"k* ≈ {k_star}", zorder=1)

    # k=3 algebraic valley
    row3 = m2f.loc[m2f["k"] == 3]
    if len(row3):
        y3 = float(row3["R2_median"].iloc[0])
        ax.annotate(
            "k=3 exactly\ndetermined",
            xy=(3, y3),
            xytext=(5.2, max(y3 - 0.22, -0.05)),
            fontsize=6.5, color="#e65100", ha="left", va="top",
            arrowprops=dict(
                arrowstyle="->", color="#e65100", lw=1.15,
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=5,
        )
        ax.plot([3], [y3], marker="o", ms=7, mfc="none", mec="#e65100", mew=1.4, zorder=4)

    ax.set_xscale("log")
    ax.set_xticks(m2f["k"].tolist())
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("k shared anchors (log scale)")
    ax.set_ylabel("Understanding R² (held-out)")
    ax.set_title("Few random anchors saturate (near/far → Supp)", fontsize=8.8)
    ax.set_ylim(-0.25, 1.0)
    ax.legend(frameon=False, fontsize=6.0, loc="lower right")


# ── E ─────────────────────────────────────────────────────────────────
def _assign_nearest_basin(v: float, a: float, basins: list[dict]) -> int:
    d = [((v - b["v"]) ** 2 + (a - b["a"]) ** 2) for b in basins]
    return int(np.argmin(d))


def _mark_duplicate_indices(cells: list[dict], basins: list[dict]) -> set[int]:
    """Among k>3 points, mark those that share a basin with an earlier point."""
    if len(cells) <= 3 or not basins:
        return set()
    occupied: dict[int, int] = {}
    dup: set[int] = set()
    for i, cell in enumerate(cells):
        bi = _assign_nearest_basin(float(cell["v"]), float(cell["a"]), basins)
        if bi in occupied:
            dup.add(i)
        else:
            occupied[bi] = i
    # If no duplicate by assignment but k=4, mark the last point as reinforcement.
    if not dup and len(cells) == 4:
        dup.add(3)
    return dup


def draw_panel_e(ax, density: dict, basins: list[dict], profiles: dict, fig: plt.Figure) -> None:
    gx, gy, z_m = _kde_on_lim(density["points_m2f_k4"], lim=VA_LIM_FIG3)
    _, _, z_f = _kde_on_lim(density["points_f2m_k4"], lim=VA_LIM_FIG3)
    shared = np.minimum(z_m, z_f)
    vmax = max(float(np.nanmax(shared)), 1e-12)
    im = ax.imshow(
        shared,
        extent=[VA_LIM_FIG3[0], VA_LIM_FIG3[1], VA_LIM_FIG3[0], VA_LIM_FIG3[1]],
        origin="lower", cmap="inferno", vmin=0.0, vmax=vmax,
        aspect="equal", zorder=1,
    )

    # Basin rings + full text labels (match Results wording)
    label_map = {
        "B1 highV/lowA": "high V / low A",
        "B2 highV/midA": "high V / mid A",
        "B3 lowV/highA": "low V / high A",
    }
    for b in basins:
        pretty = label_map.get(b["label"], b["label"])
        circ = plt.Circle(
            (b["v"], b["a"]), 0.78,
            facecolor="none", edgecolor="white", lw=1.5, ls="-", zorder=4,
        )
        ax.add_patch(circ)
        ax.plot([b["v"]], [b["a"]], marker="+", color="white", ms=8, mew=1.6, zorder=5)
        ax.text(
            b["v"], b["a"] + 0.95, pretty,
            ha="center", va="bottom", fontsize=7.2, color="white",
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="0.1", ec="none", alpha=0.55),
        )

    lev_m = _positive_quantile_levels(z_m)
    lev_f = _positive_quantile_levels(z_f)
    if len(lev_m):
        ax.contour(gx, gy, z_m, levels=lev_m, colors="#4fc3f7", linewidths=0.85, zorder=3)
    if len(lev_f):
        ax.contour(gx, gy, z_f, levels=lev_f, colors="#a5d6a7", linewidths=0.85,
                   linestyles="--", zorder=3)

    # Greedy representative solutions
    specs = [
        ("MtoF", "#42a5f5", "M→F greedy (k=4)"),
        ("FtoM", "#ef5350", "F→M greedy (k=3)"),
    ]
    for dshort, col, lab in specs:
        cells = profiles[dshort]["representative_cells"]
        dup = _mark_duplicate_indices(cells, basins)
        first = True
        first_dup = True
        for i, cell in enumerate(cells):
            v, a = float(cell["v"]), float(cell["a"])
            if i in dup:
                ax.scatter(
                    [v], [a], s=150, marker="D", c=col,
                    edgecolors="white", linewidths=1.2, zorder=7,
                    label="4th point (reinforces existing basin)" if first_dup and dshort == "MtoF" else None,
                )
                first_dup = False
            else:
                ax.scatter(
                    [v], [a], s=120, marker="o", c=col,
                    edgecolors="white", linewidths=1.0, zorder=7,
                    label=lab if first else None,
                )
                first = False
            ax.text(v, a, str(cell["cell_id"]), ha="center", va="center",
                    fontsize=6.5, color="white", fontweight="bold", zorder=8)

    ax.set_xlim(*VA_LIM_FIG3)
    ax.set_ylim(*VA_LIM_FIG3)
    ax.set_xticks(np.arange(VA_LIM_FIG3[0], VA_LIM_FIG3[1] + 0.01, 1.0))
    ax.set_yticks(np.arange(VA_LIM_FIG3[0], VA_LIM_FIG3[1] + 0.01, 1.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Valence (source VA)")
    ax.set_ylabel("Arousal (source VA)")
    ax.set_title(
        "Shareable landmarks + minimal coverage\n"
        "3 basins shared; M→F 4th point reinforces an existing basin",
        fontsize=8.5,
    )
    handles = [
        Line2D([0], [0], color="#4fc3f7", lw=1.2, label="M→F density"),
        Line2D([0], [0], color="#a5d6a7", lw=1.2, ls="--", label="F→M density"),
    ]
    leg1 = ax.legend(handles=handles, frameon=False, fontsize=5.8, loc="upper right",
                     labelcolor="white")
    ax.add_artist(leg1)
    ax.legend(frameon=False, fontsize=5.8, loc="lower right", labelcolor="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                 label="shareable density  min(M→F, F→M)")


def main() -> None:
    plt.rcParams.update(PAPER_RC)

    ot = json.loads((CVAE / "paper2_ot_five_point_fixedsplit.json").read_text())
    scores_f = ot["scores_female_target"]
    scores_m = ot["scores_male_target"]
    comp = json.loads((PBA / "composition_consistency.json").read_text())
    boot = json.loads((PBA / "subject_bootstrap_ci.json").read_text())
    cents = json.loads((PBA / "va_centroids_vs_fixed_points.json").read_text())
    m2f = pd.read_csv(ANC / "z3_generalization_curve_MtoF.csv")
    f2m = pd.read_csv(ANC / "z3_generalization_curve_FtoM.csv")
    z3 = json.loads((ANC / "z3_generalization.json").read_text())
    profiles, _overlap, cen, _fp = _load_z5_minimal_anchor_data()
    point_density = _load_anchor_point_density_data("0p1")
    basins = _estimate_shared_affine_basins(point_density)
    gdens, odens = _oasis_density()

    oasis_c = (
        float(cents["centroids"]["oasis_900_overall"]["valence"]),
        float(cents["centroids"]["oasis_900_overall"]["arousal"]),
    )

    br = {
        "gender": {
            "A": comp["full_sample_96"]["phi_gender"]["A"],
            "b": comp["full_sample_96"]["phi_gender"]["b"],
            "fp": comp["full_sample_96"]["phi_gender"]["fixed_point"],
            "sv": comp["full_sample_96"]["phi_gender"]["singular_values"],
        },
        "culture": {
            "A": comp["full_sample_96"]["phi_culture"]["A"],
            "b": comp["full_sample_96"]["phi_culture"]["b"],
            "fp": comp["full_sample_96"]["phi_culture"]["fixed_point"],
            "sv": comp["full_sample_96"]["phi_culture"]["singular_values"],
        },
    }
    vmax = _displacement_vmax(br)

    cult_ci = boot["phi_culture_subject_bootstrap"]
    fp_c = br["culture"]["fp"]
    ci_ell = (
        float(fp_c[0]),
        float(fp_c[1]),
        float(cult_ci["fp_v"]["ci95"][1] - cult_ci["fp_v"]["ci95"][0]),
        float(cult_ci["fp_a"]["ci95"][1] - cult_ci["fp_a"]["ci95"][0]),
    )

    # ── combined figure ───────────────────────────────────────────────
    fig = plt.figure(figsize=(11.6, 14.2))
    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.05, 1.15, 1.15],
        hspace=0.32, wspace=0.28,
    )

    ax_a_left = fig.add_subplot(gs[0, 0])
    ax_a_right = fig.add_subplot(gs[0, 1])
    draw_panel_a(ax_a_left, ax_a_right, scores_f, scores_m)

    ax_b = fig.add_subplot(gs[1, 0])
    panel_label(ax_b, "B")
    cf_b = draw_bridge_panel(
        ax_b,
        A=br["gender"]["A"], b=br["gender"]["b"],
        fp=br["gender"]["fp"], other_fp=br["culture"]["fp"],
        dens=gdens, oasis_centroid=oasis_c,
        title="Gender bridge Φ_gender",
        self_lab="Gender", other_lab="Culture",
        self_col="#1976d2", other_col="#e53935",
        vmax=vmax, sv=br["gender"]["sv"],
        ci_ellipse=None,
        note="Fixed point poorly determined\n(image-bootstrap CI exceeds scale)",
    )

    ax_c = fig.add_subplot(gs[1, 1])
    panel_label(ax_c, "C")
    cf_c = draw_bridge_panel(
        ax_c,
        A=br["culture"]["A"], b=br["culture"]["b"],
        fp=br["culture"]["fp"], other_fp=br["gender"]["fp"],
        dens=odens,
        oasis_centroid=oasis_c,
        title="Culture bridge Φ_culture",
        self_lab="Culture", other_lab="Gender",
        self_col="#e53935", other_col="#1976d2",
        vmax=vmax, sv=br["culture"]["sv"],
        ci_ellipse=ci_ell,
        note=None,
    )
    # Shared colourbars for B/C
    for ax, cf in ((ax_b, cf_b), (ax_c, cf_c)):
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03, label="‖Φ(y)−y‖")

    ax_d = fig.add_subplot(gs[2, 0])
    panel_label(ax_d, "D")
    draw_panel_d(ax_d, m2f, f2m, z3)

    ax_e = fig.add_subplot(gs[2, 1])
    panel_label(ax_e, "E")
    draw_panel_e(ax_e, point_density, basins, profiles, fig)

    fig.suptitle(
        "Fig. 2 | A low-dimensional affine map of output coordinates; "
        "few shared references recover the bridge",
        fontsize=10.5, y=0.995,
    )
    png, svg = save_dual(fig, "Paper_Fig2_perspective_geometry")
    print(f"[ok] combined → {png.name}, {svg.name}")

    # ── individual panels ─────────────────────────────────────────────
    fig_a = plt.figure(figsize=(10.8, 4.2))
    gs_a = GridSpec(1, 2, figure=fig_a, wspace=0.28)
    ax_al = fig_a.add_subplot(gs_a[0, 0])
    ax_ar = fig_a.add_subplot(gs_a[0, 1])
    draw_panel_a(ax_al, ax_ar, scores_f, scores_m)
    fig_a.tight_layout()
    _save_panel(fig_a, "Paper_Fig2_panelA_five_point")

    make_supp_fig2a_dimension_direction(scores_f, scores_m)

    fig_b, ax = plt.subplots(figsize=(5.4, 5.0))
    cf = draw_bridge_panel(
        ax,
        A=br["gender"]["A"], b=br["gender"]["b"],
        fp=br["gender"]["fp"], other_fp=br["culture"]["fp"],
        dens=gdens, oasis_centroid=oasis_c,
        title="Gender bridge Φ_gender",
        self_lab="Gender", other_lab="Culture",
        self_col="#1976d2", other_col="#e53935",
        vmax=vmax, sv=br["gender"]["sv"],
        note="Fixed point poorly determined\n(image-bootstrap CI exceeds scale)",
    )
    fig_b.colorbar(cf, ax=ax, fraction=0.046, pad=0.03, label="‖Φ(y)−y‖")
    fig_b.tight_layout()
    _save_panel(fig_b, "Paper_Fig2_panelB_gender_bridge")

    fig_c, ax = plt.subplots(figsize=(5.4, 5.0))
    cf = draw_bridge_panel(
        ax,
        A=br["culture"]["A"], b=br["culture"]["b"],
        fp=br["culture"]["fp"], other_fp=br["gender"]["fp"],
        dens=odens, oasis_centroid=oasis_c,
        title="Culture bridge Φ_culture",
        self_lab="Culture", other_lab="Gender",
        self_col="#e53935", other_col="#1976d2",
        vmax=vmax, sv=br["culture"]["sv"],
        ci_ellipse=ci_ell,
    )
    fig_c.colorbar(cf, ax=ax, fraction=0.046, pad=0.03, label="‖Φ(y)−y‖")
    fig_c.tight_layout()
    _save_panel(fig_c, "Paper_Fig2_panelC_culture_bridge")

    fig_d, ax = plt.subplots(figsize=(5.6, 4.6))
    draw_panel_d(ax, m2f, f2m, z3)
    fig_d.tight_layout()
    _save_panel(fig_d, "Paper_Fig2_panelD_quantity_generalization")

    fig_e, ax = plt.subplots(figsize=(5.8, 5.2))
    draw_panel_e(ax, point_density, basins, profiles, fig_e)
    fig_e.tight_layout()
    _save_panel(fig_e, "Paper_Fig2_panelE_shareable_landmarks")

    print("[ok] individual panels A–E written to Paper_fig/")


if __name__ == "__main__":
    main()
