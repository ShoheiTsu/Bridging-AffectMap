#!/usr/bin/env python3
"""
Manuscript Figs 3–5.

Fig.3  culture wall: A 2×2 raw/affine scatters; B observed R² vs reliability ceiling + CI
Fig.4  residuals: A 4-category violin; B–C excess twist (defined title); D Person bottleneck bars
Fig.5  translation: B combined coloc; D row-norm flow; E forest plot
        side-by-side B|C → Supp; Table 1 numbers → Supp Table

Run:
  python3 code/plot_paper_fig345_manuscript.py
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from generate_paper_figures import (  # noqa: E402
    ANC,
    CVAE,
    EQ,
    OUT,
    PBA,
    VA_LIM_FIG3,
    FIG4_COL_A,
    FIG4_COL_V,
    FIG4_VA_LIM,
    _draw_excess_twist_panel,
    _fig4_bootstrap_r2_samples,
    _load_excess_twist_maps,
    _load_fig4_transfer_data,
    _load_fig5_packs,
    _load_person_bottleneck_payload,
    panel_label,
    save_dual,
    violin_strip,
)

PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9.2,
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

COL_M2F = "#1565c0"
COL_F2M = "#c62828"
CAT_ORDER_RES = ["Animal", "Object", "Person", "Scene"]
CAT_ORDER_FLOW = ["Scene", "Person", "Object", "Animal"]  # match existing flow convention


def _save(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    return save_dual(fig, stem)


# ═══════════════════════════════════════════════════════════════════════
# Fig. 3 — culture wall
# ═══════════════════════════════════════════════════════════════════════
def draw_fig3_panel_a_2x2(fig: plt.Figure, gs_parent, transfer: dict) -> None:
    """2×2: rows Valence/Arousal × cols raw / affine; identity line."""
    inner = gs_parent.subgridspec(2, 2, wspace=0.28, hspace=0.35)
    specs = [
        (0, 0, 0, "Valence", "raw", FIG4_COL_V, transfer["x"][:, 0], transfer["y"][:, 0],
         transfer["raw"]["valence"]),
        (0, 1, 0, "Valence", "affine", FIG4_COL_V, transfer["y_aff"][:, 0], transfer["y"][:, 0],
         transfer["affine"]["valence"]),
        (1, 0, 1, "Arousal", "raw", FIG4_COL_A, transfer["x"][:, 1], transfer["y"][:, 1],
         transfer["raw"]["arousal"]),
        (1, 1, 1, "Arousal", "affine", FIG4_COL_A, transfer["y_aff"][:, 1], transfer["y"][:, 1],
         transfer["affine"]["arousal"]),
    ]
    first = True
    for r, c, dim_i, dim, mode, color, xs, ys, stats in specs:
        ax = fig.add_subplot(inner[r, c])
        if first:
            panel_label(ax, "A")
            first = False
        lim = FIG4_VA_LIM
        ax.scatter(xs, ys, s=22, alpha=0.55, c=color, edgecolors="white", linewidths=0.3, zorder=2)
        ax.plot([lim[0], lim[1]], [lim[0], lim[1]], "k--", lw=0.95, alpha=0.5, zorder=1, label="identity")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(1, 8, 1))
        ax.set_yticks(np.arange(1, 8, 1))
        ax.grid(color="0.92", lw=0.5)
        r2 = float(stats["R2"])
        r_p = float(stats.get("Pearson_r", np.nan))
        if mode == "raw":
            ax.set_title(f"{dim} · raw\nr = {r_p:.2f}, R² = {r2:.2f}", fontsize=8.2, color=color)
            ax.set_xlabel(f"OASIS {dim}")
        else:
            ax.set_title(f"{dim} · affine Φ\nR² = {r2:.2f}", fontsize=8.2, color=color)
            ax.set_xlabel(f"Φ(OASIS) {dim}")
        if c == 0:
            ax.set_ylabel(f"Japan {dim}")
        if dim == "Arousal" and mode == "raw":
            ax.text(
                0.04, 0.04, "negative correlation",
                transform=ax.transAxes, fontsize=6.5, color=FIG4_COL_A,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ce93d8", alpha=0.92),
            )


def draw_fig3_panel_b(ax, rel: dict, boot_ci: dict) -> None:
    """Observed affine R² (point+CI) vs reliability ceiling (dotted)."""
    panel_label(ax, "B")
    obs_v = float(rel["observed_affine2d_r2"]["valence"])
    obs_a = float(rel["observed_affine2d_r2"]["arousal"])
    ceil_v = float(rel["ceiling_r2"]["valence"])
    ceil_a = float(rel["ceiling_r2"]["arousal"])
    ci_v = boot_ci["r2_v"]["ci95"]
    ci_a = boot_ci["r2_a"]["ci95"]

    xs = np.array([0.0, 1.0])
    obs = np.array([obs_v, obs_a])
    lo = np.array([ci_v[0], ci_a[0]])
    hi = np.array([ci_v[1], ci_a[1]])
    cols = [FIG4_COL_V, FIG4_COL_A]
    labels = ["Valence", "Arousal"]

    for x, y, ylo, yhi, c, lab, ceil in zip(
        xs, obs, lo, hi, cols, labels, [ceil_v, ceil_a]
    ):
        ax.hlines(ceil, x - 0.28, x + 0.28, colors=c, linestyles=":", lw=2.0, zorder=2)
        ax.errorbar(
            [x], [y], yerr=[[y - ylo], [yhi - y]],
            fmt="o", ms=9, color=c, ecolor=c, elinewidth=1.6, capsize=5, zorder=4,
            label=lab,
        )
        ax.text(x, ceil + 0.015, f"ceiling {ceil:.2f}", ha="center", va="bottom",
                fontsize=6.5, color=c)

    ax.set_xticks(xs, labels)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("R² (affine OASIS → Japan)")
    ax.set_title(
        "Observed R² vs reliability ceiling\n"
        f"V CI [{ci_v[0]:.2f},{ci_v[1]:.2f}] · A CI [{ci_a[0]:.2f},{ci_a[1]:.2f}] (non-overlap)",
        fontsize=8.5,
    )
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.text(
        0.98, 0.05,
        "dotted = Spearman–Brown ceiling",
        transform=ax.transAxes, ha="right", fontsize=6.5, color="0.4",
    )


def make_fig3() -> None:
    transfer = _load_fig4_transfer_data()
    rel = json.loads((PBA / "reliability_ceiling.json").read_text())
    boot = json.loads((PBA / "subject_bootstrap_ci.json").read_text())
    cb = boot["phi_culture_subject_bootstrap"]

    fig = plt.figure(figsize=(11.2, 5.6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.55, 0.95], wspace=0.28)
    draw_fig3_panel_a_2x2(fig, gs[0, 0], transfer)
    ax_b = fig.add_subplot(gs[0, 1])
    draw_fig3_panel_b(ax_b, rel, cb)
    fig.suptitle(
        "Fig. 3 | Cultural boundary: valence transfers; arousal does not",
        fontsize=11, y=1.01,
    )
    _save(fig, "Paper_Fig3_culture_wall")

    # standalone A
    fig_a = plt.figure(figsize=(8.4, 7.6))
    gs_a = GridSpec(1, 1, figure=fig_a)
    draw_fig3_panel_a_2x2(fig_a, gs_a[0, 0], transfer)
    fig_a.suptitle("Fig. 3A | OASIS → Japan (identity = dashed)", fontsize=10.5, y=0.995)
    _save(fig_a, "Paper_Fig3_panelA_transfer_2x2")

    fig_b, ax = plt.subplots(figsize=(4.8, 4.6))
    draw_fig3_panel_b(ax, rel, cb)
    fig_b.tight_layout()
    _save(fig_b, "Paper_Fig3_panelB_r2_vs_ceiling")
    print("[ok] Fig. 3")


# ═══════════════════════════════════════════════════════════════════════
# Fig. 4 — residual geometry
# ═══════════════════════════════════════════════════════════════════════
def draw_fig4_residual_violin(ax, per_img: pd.DataFrame, res: dict, omnibus_p: float) -> None:
    panel_label(ax, "A")
    order = [c for c in CAT_ORDER_RES if c in set(per_img["category"])]
    groups = {c: per_img.loc[per_img["category"] == c, "residual_l2"].to_numpy() for c in order}
    colors = ["#ef5350" if c == "Person" else "#90caf9" for c in order]
    violin_strip(ax, groups, colors=colors, ylabel="Post-Φ residual ‖y_f − Φ(y_m)‖")
    cm = res["gender_oasis_900"]["category_means"]
    p_po = res["gender_oasis_900"]["person_vs_object_permutation_p"]
    ax.set_title("Category residual after Φ", fontsize=9)
    if omnibus_p <= 0:
        p_txt = "omnibus p < 0.0002"
    else:
        p_txt = f"omnibus p ≈ {omnibus_p:.4f}"
    ax.text(
        0.02, 0.98,
        f"{p_txt}\n"
        f"Person {cm['Person']:.3f} vs Object {cm['Object']:.3f}\n"
        f"(prereg. perm p = {p_po:.3f})",
        transform=ax.transAxes, va="top", fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.92),
    )


def draw_fig4_twist(ax, g, step, packs, direction, fig, vmax) -> None:
    _draw_excess_twist_panel(ax, g, step, packs, direction, fig=fig, vmax=vmax)
    arrow = "M→F" if direction == "MtoF" else "F→M"
    ref = packs[direction]["ref"]
    cl = packs[direction]["cluster"]
    sub = ""
    if cl.get("ok"):
        p_c = float(cl.get("p_cluster", np.nan))
        n_sig = int(cl.get("n_sig_clusters", 0))
        p_txt = "p<0.001" if p_c < 1e-3 else f"p={p_c:.3f}"
        sub = f"\ncluster {p_txt}, {n_sig} regions"
    ax.set_title(
        f"Residual displacement beyond Φ\n(excess twist) · {arrow} on {ref} VA{sub}",
        fontsize=8.0,
    )


def draw_fig4_bottleneck_bars(ax, payload: dict) -> None:
    panel_label(ax, "D")
    cats = list(payload["category_order"])
    specs = [
        ("MtoF", "M→F k=4", COL_M2F),
        ("FtoM", "F→M k=3", COL_F2M),
    ]
    x = np.arange(len(cats), dtype=float)
    width = 0.36
    for i, (key, lab, col) in enumerate(specs):
        row = payload["directions"][key]["train_rows"]["Unrestricted"]
        ys = [float(row["test_by_category"][c]["R2_mean"]) for c in cats]
        bars = ax.bar(
            x + (i - 0.5) * width, ys, width=width * 0.92,
            color=col, alpha=0.85, edgecolor="white", label=lab, zorder=3,
        )
        # emphasize Person
        if "Person" in cats:
            pi = cats.index("Person")
            bars[pi].set_edgecolor("#b71c1c")
            bars[pi].set_linewidth(1.8)
            bars[pi].set_hatch("///")
    if "Person" in cats:
        pi = cats.index("Person")
        ax.axvspan(pi - 0.48, pi + 0.48, color="#ef5350", alpha=0.08, zorder=0)
    ax.set_xticks(x, cats)
    ax.set_ylabel("Held-out R² (Unrestricted anchors)")
    ax.set_xlabel("Test category")
    ax.set_ylim(0.45, 0.95)
    ax.set_title("Person is the test-side bottleneck", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.grid(axis="y", color="0.92", lw=0.6)


def _omnibus_p(per_img: pd.DataFrame, n_perm: int = 5000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    cats = per_img["category"].to_numpy()
    y = per_img["residual_l2"].to_numpy(float)
    means = pd.Series(y).groupby(cats).mean()
    obs = float(means.var())
    null = np.empty(n_perm)
    for i in range(n_perm):
        sh = rng.permutation(cats)
        null[i] = float(pd.Series(y).groupby(sh).mean().var())
    return float((null >= obs).mean())


def make_fig4() -> None:
    d = _load_fig5_packs()
    omnibus = _omnibus_p(d["per_img"])
    payload, _cats, _specs = _load_person_bottleneck_payload()

    fig = plt.figure(figsize=(12.4, 10.2))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.26)

    ax_a = fig.add_subplot(gs[0, 0])
    draw_fig4_residual_violin(ax_a, d["per_img"], d["res"], omnibus)

    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")
    draw_fig4_twist(ax_b, d["g_tw"], d["step_tw"], d["packs_tw"], "MtoF", fig, d["twist_vmax"])

    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "C")
    draw_fig4_twist(ax_c, d["g_tw"], d["step_tw"], d["packs_tw"], "FtoM", fig, d["twist_vmax"])

    ax_d = fig.add_subplot(gs[1, 1])
    draw_fig4_bottleneck_bars(ax_d, payload)

    fig.suptitle(
        "Fig. 4 | Structured residual of the affine map",
        fontsize=11, y=0.995,
    )
    _save(fig, "Paper_Fig4_residual_structure")

    # individuals
    fig_a, ax = plt.subplots(figsize=(5.0, 4.6))
    draw_fig4_residual_violin(ax, d["per_img"], d["res"], omnibus)
    fig_a.tight_layout()
    _save(fig_a, "Paper_Fig4_panelA_category_residual")

    for stem, direction, lab in (
        ("Paper_Fig4_panelB_excess_twist_MtoF", "MtoF", "B"),
        ("Paper_Fig4_panelC_excess_twist_FtoM", "FtoM", "C"),
    ):
        fig_t, ax = plt.subplots(figsize=(5.2, 5.0))
        panel_label(ax, lab)
        draw_fig4_twist(ax, d["g_tw"], d["step_tw"], d["packs_tw"], direction, fig_t, d["twist_vmax"])
        fig_t.tight_layout()
        _save(fig_t, stem)

    fig_d, ax = plt.subplots(figsize=(5.4, 4.4))
    draw_fig4_bottleneck_bars(ax, payload)
    fig_d.tight_layout()
    _save(fig_d, "Paper_Fig4_panelD_person_bottleneck")
    print(f"[ok] Fig. 4 (omnibus p={omnibus:.4f})")


# ═══════════════════════════════════════════════════════════════════════
# Fig. 5 — case translation
# ═══════════════════════════════════════════════════════════════════════
def draw_fig5_combined_coloc(ax, m: pd.DataFrame, *, grid: float = 0.5, min_n: int = 3) -> None:
    """Colour = mean Φ residual; black outline = co-localization
    (high residual ∩ low equiv. distance), i.e. cells where oracle translation
    exists in Φ-residual territory (existence proof)."""
    from plot_fig6_case_translation_discovery import (
        va_bin_stats, _spatial_axis_style, _spatial_cohigh_mask,
    )

    edges, resid, equiv_dist, counts, grid_used = va_bin_stats(
        m, min_n=min_n, lim=(1.0, 7.0), grid=float(grid),
    )
    both, r_thr, d_thr = _spatial_cohigh_mask(resid, equiv_dist)
    extent = [edges[0], edges[-1], edges[0], edges[-1]]
    im = ax.imshow(
        resid, origin="lower", extent=extent, cmap="YlOrRd", aspect="equal",
        interpolation="nearest",
    )
    outline_lw = 1.8 if grid_used <= 0.5 else 2.2
    yy, xx = np.where(both)
    for yi, xi in zip(yy, xx):
        x0, x1 = edges[xi], edges[xi + 1]
        y0, y1 = edges[yi], edges[yi + 1]
        ax.plot(
            [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            color="black", lw=outline_lw, zorder=4,
        )
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("mean Φ residual", fontsize=7)
    _spatial_axis_style(ax, grid_used=grid_used, outline_lw=outline_lw)
    n_out = int(np.sum(both))
    ax.set_title(
        "Φ residual with translation co-localization\n"
        "colour = Φ residual; black = high residual ∩ low equiv. dist.",
        fontsize=8.2,
    )
    ax.text(
        0.02, 0.02,
        f"n={len(m)}; {n_out} outlined cells (q≥{r_thr:.2f} residual, q≤{d_thr:.2f} dist)",
        transform=ax.transAxes, fontsize=6.0, color="0.25",
    )


def draw_fig5_flow_rownorm(ax, m: pd.DataFrame, meta: dict) -> None:
    from plot_fig6_case_translation_discovery import category_transition_matrix, CAT_ORDER as CO

    mat_c, cross, n = category_transition_matrix(m)
    cats = list(CO)
    row_sums = mat_c.sum(axis=1, keepdims=True).clip(min=1)
    mat_r = mat_c / row_sums
    # Match Fig.6-C category-flow palette (inferno); values remain row fractions.
    im = ax.imshow(mat_r, cmap="inferno", vmin=0, vmax=max(0.45, float(mat_r.max())), aspect="equal")
    ax.set_xticks(range(4), cats, rotation=30, ha="right", fontsize=7.5)
    ax.set_yticks(range(4), cats, fontsize=7.5)
    ax.set_xlabel("Target category (j)")
    ax.set_ylabel("Source category (i)")
    vmax = float(mat_r.max()) if mat_r.size else 1.0
    for i in range(4):
        for j in range(4):
            v = int(mat_c[i, j])
            if v <= 0:
                continue
            # inferno: dark at low, bright at high
            color = "white" if mat_r[i, j] < vmax * 0.55 else "0.05"
            ax.text(
                j, i, f"{mat_r[i, j]:.0%}\n({v})",
                ha="center", va="center", fontsize=7.2,
                color=color,
                fontweight="bold" if i != j else "normal",
            )
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(4):
        ax.add_patch(plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1, fill=False,
            edgecolor="#90a4ae", lw=1.4, zorder=3,
        ))
    diag = float(np.trace(mat_c)) / max(n, 1)
    perm = meta.get("category_permutation", {})
    null_same = perm.get("permutation_null_same_category_mean", 0.27)
    p_same = perm.get("permutation_p_same_category_above_chance")
    if p_same is None:
        p_same = perm.get("permutation_p")
    p_txt = "p<0.001" if p_same is not None and float(p_same) < 0.001 else (
        f"p={float(p_same):.4f}" if p_same is not None else ""
    )
    ax.set_title("Category flow (row-normalized)", fontsize=9)
    ax.text(
        0.50, 1.02,
        f"diagonal {diag:.0%}  vs  null {float(null_same):.0%}",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=8.0,
        fontweight="bold", color="#0d47a1",
    )
    ax.text(
        0.02, -0.22,
        f"Same-category = {diag:.0%} vs null ≈ {float(null_same):.0%} ({p_txt}); "
        f"{cross:.0%} cross-category (n={n})",
        transform=ax.transAxes, fontsize=6.8, color="0.25",
    )
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("row fraction", fontsize=7)


def draw_fig5_forest(ax, rows: list[list[str]]) -> None:
    """Forest: Person M→F first; colour by direction; 50% line; n at right."""
    panel_label(ax, "E")
    # parse data rows (skip header); preferred order
    order_cats = ["Person", "Object", "Scene", "Animal"]
    parsed = []
    for r in rows[1:]:
        cat, direction, n, beat, ci, med, p = r
        beat_f = float(beat.strip("%")) / 100.0
        lo, hi = ci.strip("[]").split(",")
        parsed.append({
            "cat": cat, "dir": direction, "n": int(n),
            "beat": beat_f, "lo": float(lo), "hi": float(hi),
            "med": med, "p": p,
        })
    # Person M→F first, then Person F→M, then others M→F/F→M
    ordered = []
    for cat in order_cats:
        for d in ("M→F", "F→M"):
            for p in parsed:
                if p["cat"] == cat and p["dir"] == d:
                    ordered.append(p)

    y = np.arange(len(ordered))[::-1]
    for yi, p in zip(y, ordered):
        col = COL_M2F if p["dir"] == "M→F" else COL_F2M
        ax.errorbar(
            p["beat"], yi,
            xerr=[[p["beat"] - p["lo"]], [p["hi"] - p["beat"]]],
            fmt="o", ms=7, color=col, ecolor=col, elinewidth=1.5, capsize=3.5, zorder=3,
        )
        ax.text(1.02, yi, f"n={p['n']}", va="center", ha="left", fontsize=7, color="0.3",
                transform=ax.get_yaxis_transform())

    ax.axvline(0.5, color="0.35", ls="--", lw=1.2, zorder=1)
    ax.set_yticks(y, [f"{p['cat']}  {p['dir']}" for p in ordered])
    for tick, p in zip(ax.get_yticklabels(), ordered):
        tick.set_color(COL_M2F if p["dir"] == "M→F" else COL_F2M)
        if p["cat"] == "Person" and p["dir"] == "M→F":
            tick.set_fontweight("bold")
    ax.set_xlim(0.0, 0.85)
    ax.set_xlabel("Beat rate (Ridge closer than Φ)")
    ax.set_title(
        "Predictive translation vs Φ (high raw-gap)\nWilson 95% CI; dashed = 50%",
        fontsize=8.8,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=COL_M2F, lw=0, label="M→F"),
            Line2D([0], [0], marker="o", color=COL_F2M, lw=0, label="F→M"),
        ],
        frameon=False, fontsize=7, loc="lower right",
    )
    ax.grid(axis="x", color="0.92", lw=0.6)


def make_fig5() -> None:
    from plot_fig6_case_translation_discovery import (
        build_merged,
        load_category_flow_pairs,
        draw_spatial_single_panel,
        category_transition_matrix,
    )
    from plot_predictive_translation_prototype import (
        build_fig5_panel_e_predictive_table_rows,
        _load_rec,
    )

    m = build_merged()
    flow_m, flow_meta = load_category_flow_pairs()
    rec_m = _load_rec(direction="MtoF", use_cached_csv=True)
    rec_f = _load_rec(direction="FtoM", use_cached_csv=True)
    table_rows = build_fig5_panel_e_predictive_table_rows(rec_m, rec_f)

    # Combined coloc (main B) + grid variants
    for grid, stem_suffix in ((0.5, ""), (0.2, "_grid0p2")):
        fig_b, ax = plt.subplots(figsize=(5.4, 5.0))
        panel_label(ax, "B")
        draw_fig5_combined_coloc(ax, m, grid=grid)
        fig_b.tight_layout()
        _save(fig_b, f"Paper_Fig5_panelB_phi_residual_beat_outline{stem_suffix}")

        fig_s = plt.figure(figsize=(10.2, 4.8))
        gs = GridSpec(1, 2, figure=fig_s, wspace=0.30)
        ax0 = fig_s.add_subplot(gs[0, 0])
        ax1 = fig_s.add_subplot(gs[0, 1])
        panel_label(ax0, "A")
        panel_label(ax1, "B")
        draw_spatial_single_panel(fig_s, ax0, m, field="residual", grid=grid)
        draw_spatial_single_panel(fig_s, ax1, m, field="equiv_dist", grid=grid)
        fig_s.suptitle(
            f"Supplementary | Φ residual vs equivalence distance "
            f"(same VA frame; grid={grid:g})",
            fontsize=10, y=1.02,
        )
        fig_s.tight_layout()
        _save(fig_s, f"Paper_SuppFig_phi_residual_vs_equiv_distance{stem_suffix}")

    # D flow
    fig_d, ax = plt.subplots(figsize=(5.2, 5.0))
    panel_label(ax, "D")
    draw_fig5_flow_rownorm(ax, flow_m, flow_meta)
    fig_d.tight_layout()
    _save(fig_d, "Paper_Fig5_panelD_category_flow")

    # E forest
    fig_e, ax = plt.subplots(figsize=(6.4, 5.2))
    draw_fig5_forest(ax, table_rows)
    fig_e.tight_layout()
    _save(fig_e, "Paper_Fig5_panelE_predictive_forest")

    # Combined A–E stub note: A hub matches already exists; keep path
    print("[ok] Fig. 5 panels B/D/E + Supp B|C")
    print("     Fig. 5A: keep Paper_Fig5_panelA_hub_matches (add OASIS credit in caption)")


def main() -> None:
    plt.rcParams.update(PAPER_RC)
    OUT.mkdir(parents=True, exist_ok=True)
    make_fig3()
    make_fig4()
    make_fig5()
    print("[done] Paper_fig/ Fig.3–5 manuscript panels")


if __name__ == "__main__":
    main()
