#!/usr/bin/env python3
"""
Nature Communications 本文 Figure を Paper_fig/ に PNG + SVG で一括出力。

仕様: doc/NATCOMM_FIGURE_SPEC.md · doc/NUMERICAL_CANON_REGISTRY.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Patch
from matplotlib.gridspec import GridSpec
from scipy.ndimage import label as nd_label
from scipy.ndimage import maximum_filter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
FIG_SRC = ROOT / "fig_doc"
OUT = ROOT / "Paper_fig"
RES = ROOT / "results"
CVAE = RES / "cvae_cross_gender"
LAMBDA_FIG2 = RES / "lambda_gender_diff_cluster_analysis"
ANC = RES / "anchor_structure"
EQ = RES / "equivalence_nontriviality"
PBA = RES / "population_bridge_analysis"
TWIST = RES / "relational_cross_within_twist"

sys.path.insert(0, str(ROOT / "code"))

VA_LIM = (1.5, 7.0)
VA_LIM_FIG3 = (1.0, 7.0)
CONTOUR_LEVELS_CULTURE = tuple(np.arange(1.2, 5.0 + 0.001, 1.2))
FIG4_COL_V = "#1565c0"  # match Fig.3-H Valence
FIG4_COL_A = "#8e24aa"  # match Fig.3-H Arousal
FIG4_VA_LIM = (1.0, 7.0)


def _oasis_density():
    import analysis_population_bridge_suite as pb
    o = pb.load_oasis_meta(pb.OASIS_SCORES_CSV)
    return (o[["valence_male", "arousal_male"]].to_numpy(float),
            o[["valence", "arousal"]].to_numpy(float))


def _disp_grid(A, b, n=140, lim=VA_LIM):
    vv, aa = np.meshgrid(np.linspace(*lim, n), np.linspace(*lim, n))
    pts = np.column_stack([vv.ravel(), aa.ravel()])
    d = np.linalg.norm(pts @ np.asarray(A).T + np.asarray(b) - pts, axis=1).reshape(vv.shape)
    return vv, aa, d


def _draw_fig3_target_panel(
    ax,
    scores: dict,
    methods: list[str],
    labels: list[str],
    mcolors: list[str],
    *,
    target_label: str,
    note_color: str,
    note_edge: str,
    filled: bool,
) -> None:
    x = np.arange(len(methods), dtype=float)
    data = [np.asarray(scores[m]["R2_mean_boot"], float) for m in methods]
    parts = ax.violinplot(data, positions=x, showmedians=True, widths=0.72)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(mcolors[i] if filled else "white")
        body.set_alpha(0.65 if filled else 1.0)
        body.set_edgecolor("0.35" if filled else mcolors[i])
        body.set_linewidth(0.9 if filled else 1.25)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("0.25")
            parts[key].set_linewidth(0.95)

    ax.set_xticks(range(len(methods)), labels)
    ax.set_ylabel("R² mean (2000-boot distribution, n=153)")
    ax.set_title(f"Five-point mapping ({target_label})")
    handles = [Patch(
        facecolor="0.35" if filled else "white",
        edgecolor="0.35",
        linewidth=1.2,
        alpha=0.65 if filled else 1.0,
        label=f"{target_label} ({'filled' if filled else 'outline'})",
    )]
    ax.legend(handles=handles, frameon=False, fontsize=6.6, loc="upper left")
    ax.text(
        0.02, 0.03,
        f"Oracle {scores['oracle']['R2_mean']:.3f} > "
        f"Linear {scores['linear_shift']['R2_mean']:.3f} ≳ OT {scores['ot']['R2_mean']:.3f}",
        transform=ax.transAxes, va="bottom", fontsize=7, color=note_color,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=note_edge, alpha=0.95),
    )


def _draw_fig3_combined_panel_a(
    ax,
    scores_f: dict,
    scores_m: dict | None,
    methods: list[str],
    labels: list[str],
    mcolors: list[str],
) -> None:
    x = np.arange(len(methods), dtype=float)
    pos_f = x - 0.1
    pos_m = x + 0.1
    data_f = [np.asarray(scores_f[m]["R2_mean_boot"], float) for m in methods]
    parts_f = ax.violinplot(data_f, positions=pos_f, showmedians=True, widths=0.18)
    for i, body in enumerate(parts_f["bodies"]):
        body.set_facecolor(mcolors[i])
        body.set_alpha(0.65)
        body.set_edgecolor("0.35")
        body.set_linewidth(0.8)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts_f:
            parts_f[key].set_color("0.25")
            parts_f[key].set_linewidth(0.9)
    if scores_m is not None:
        data_m = [np.asarray(scores_m[m]["R2_mean_boot"], float) for m in methods]
        parts_m = ax.violinplot(data_m, positions=pos_m, showmedians=True, widths=0.18)
        for i, body in enumerate(parts_m["bodies"]):
            body.set_facecolor("white")
            body.set_alpha(1.0)
            body.set_edgecolor(mcolors[i])
            body.set_linewidth(1.25)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts_m:
                parts_m[key].set_color("0.15")
                parts_m[key].set_linewidth(1.0)
    ax.set_xticks(range(len(methods)), labels)
    ax.set_ylabel("R² mean (2000-boot distribution, n=153)")
    ax.set_title("Five-point mapping (female and male targets)")
    handles = [
        Patch(facecolor="0.35", edgecolor="0.35", alpha=0.65, label="female target (x−0.1)"),
        Patch(facecolor="white", edgecolor="0.35", linewidth=1.2, label="male target (x+0.1)"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.6, loc="upper left")
    ax.text(
        0.02, 0.11,
        f"Female target: Oracle {scores_f['oracle']['R2_mean']:.3f} > Linear "
        f"{scores_f['linear_shift']['R2_mean']:.3f} ≳ OT {scores_f['ot']['R2_mean']:.3f}",
        transform=ax.transAxes, va="bottom", fontsize=7, color="#c62828",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#ef9a9a", alpha=0.95),
    )
    if scores_m is not None:
        ax.text(
            0.02, 0.035,
            f"Male target: Oracle {scores_m['oracle']['R2_mean']:.3f} > Linear "
            f"{scores_m['linear_shift']['R2_mean']:.3f} ≳ OT {scores_m['ot']['R2_mean']:.3f}",
            transform=ax.transAxes, va="bottom", fontsize=7, color="#1565c0",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#90caf9", alpha=0.95),
        )


def _draw_fig3_category_secondary_panel(ax, payload: dict) -> None:
    cats = payload["category_order"]
    dir_specs = [
        ("MtoF", "M→F k=4"),
        ("FtoM", "F→M k=3"),
        ("FtoM_k4", "F→M k=4"),
    ]
    dir_specs = [(k, lab) for k, lab in dir_specs if k in payload["directions"]]

    x = np.arange(len(cats))
    offsets = np.linspace(-0.2, 0.2, len(dir_specs))
    for off, (key, lab) in zip(offsets, dir_specs):
        row = payload["directions"][key]["train_rows"]["Unrestricted"]
        vals = [row["test_by_category"][cat]["R2_mean"] for cat in cats]
        ax.plot(x + off, vals, marker="o", lw=1.6, label=lab)
    ax.set_xticks(x, cats)
    ax.set_ylim(0.2, 0.92)
    ax.set_ylabel("Held-out R² mean")
    ax.set_title("Secondary pattern: Person remains hardest", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.axvspan(0.5, 1.5, color="#ef9a9a", alpha=0.12, zorder=0)
    ax.text(
        0.02, 0.97,
        "Unrestricted anchors converge by VA coverage;\nPerson remains a secondary bottleneck.",
        transform=ax.transAxes, va="top", fontsize=7.1,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", alpha=0.92),
    )


def bridge_terrain(ax, A, b, fp, other_fp, dens, title, self_lab, other_lab,
                   self_col, other_col, cmap="YlOrRd", lim=VA_LIM,
                   contour_levels=(0.2, 0.4, 0.6, 0.8)):
    vv, aa, d = _disp_grid(A, b, lim=lim)
    # filled terrain: keep smooth gradient; line contours fixed to requested levels
    fill_levels = np.linspace(0.0, max(float(np.nanmax(d)), 1.0), 13)
    cf = ax.contourf(vv, aa, d, levels=fill_levels, cmap=cmap, alpha=0.85)
    cs = ax.contour(
        vv, aa, d, levels=list(contour_levels), colors="0.25",
        linewidths=0.9, alpha=0.85,
    )
    ax.clabel(cs, inline=True, fontsize=6.0, fmt="%.1f")
    if dens is not None:
        ax.scatter(dens[:, 0], dens[:, 1], s=5, c="#1a237e", alpha=0.16, edgecolors="none", zorder=2)
    vq, aq, _ = _disp_grid(A, b, n=12, lim=lim)
    pts = np.column_stack([vq.ravel(), aq.ravel()])
    uv = pts @ np.asarray(A).T + np.asarray(b) - pts
    uvn = uv / np.clip(np.linalg.norm(uv, axis=1, keepdims=True), 1e-9, None)
    ax.quiver(pts[:, 0], pts[:, 1], uvn[:, 0], uvn[:, 1], color="0.25", alpha=0.55,
              pivot="mid", scale=26, width=0.004, headwidth=4, zorder=3)
    ax.scatter(*fp, s=150, marker="*", c=self_col, edgecolors="white", linewidths=1.1,
               label=f"{self_lab} FP ({fp[0]:.2f},{fp[1]:.2f})", zorder=5)
    ax.scatter(*other_fp, s=85, marker="o", facecolors="none", edgecolors=other_col, linewidths=1.8,
               label=f"{other_lab} FP ({other_fp[0]:.2f},{other_fp[1]:.2f})", zorder=5)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(title)
    return cf

PAPER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def save_dual(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{stem}.png"
    svg = OUT / f"{stem}.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, svg


def embed(ax, rel_path: str, title: str = "") -> None:
    p = FIG_SRC / rel_path
    if not p.exists():
        ax.text(0.5, 0.5, f"Missing:\n{rel_path}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    ax.imshow(mpimg.imread(str(p)))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9, pad=4)


def panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="right")


def strip(ax, groups, colors=None, ylabel="", jitter=0.06, seed=0):
    """groups: dict[label -> 1D array]. Individual points + median bar (shows dispersion)."""
    rng = np.random.default_rng(seed)
    labels = list(groups.keys())
    colors = colors or ["#42a5f5"] * len(labels)
    for i, (lab, arr) in enumerate(groups.items()):
        arr = np.asarray(arr, float)
        x = i + rng.uniform(-jitter, jitter, size=len(arr))
        ax.scatter(x, arr, s=32, color=colors[i], alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3)
        med = np.median(arr)
        ax.plot([i - 0.22, i + 0.22], [med, med], color="k", lw=1.6, zorder=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, len(labels) - 0.5)


def violin_strip(ax, groups, colors=None, ylabel="", seed=0):
    rng = np.random.default_rng(seed)
    labels = list(groups.keys())
    data = [np.asarray(groups[l], float) for l in labels]
    colors = colors or ["#90caf9"] * len(labels)
    parts = ax.violinplot(data, positions=range(len(labels)), showmedians=True, widths=0.75)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(colors[i])
        b.set_alpha(0.35)
        b.set_edgecolor("0.4")
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("0.3")
            parts[key].set_linewidth(1.0)
    for i, arr in enumerate(data):
        x = i + rng.uniform(-0.09, 0.09, size=len(arr))
        ax.scatter(x, arr, s=8, color=colors[i], alpha=0.5, edgecolors="none", zorder=3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)


# ── Fig 1 methodology: twist in score VA vs aligned-UMAP clusters ─────
def _load_twist_full_frame() -> pd.DataFrame:
    twist = pd.read_csv(RES / "relational_cross_within_twist" / "relational_twist_per_image.csv")
    need = [
        "valence_male", "arousal_male", "valence_female", "arousal_female",
        "umap_score_male_x", "umap_score_male_y",
        "umap_score_female_x", "umap_score_female_y",
        "geo_umap_disp", "model_twist_pm_pf",
    ]
    twist = twist.dropna(subset=need).copy()
    pm_va = twist[["valence_male", "arousal_male"]].to_numpy()
    pf_va = twist[["valence_female", "arousal_female"]].to_numpy()
    twist["geo_va_disp"] = np.linalg.norm(pf_va - pm_va, axis=1)
    twist["excess_va"] = twist["geo_va_disp"] - twist["model_twist_pm_pf"]
    twist["excess_umap"] = twist["geo_umap_disp"] - twist["model_twist_pm_pf"]
    return twist


def _assign_regime_kmeans_clusters(
    x: np.ndarray,
    y: np.ndarray,
    excess: np.ndarray,
    *,
    k_aligned: int = 4,
    k_twisted: int = 4,
) -> dict:
    """excess 中央値で aligned / twisted を分け、各レジーム内で k-means。"""
    twisted = excess > np.median(excess)
    n = len(x)
    cluster_id = np.full(n, -1, dtype=int)
    rgba = np.zeros((n, 4))
    meta_clusters: list[dict] = []
    offset = 0
    specs = [
        ("aligned", ~twisted, k_aligned, plt.cm.Blues),
        ("twisted", twisted, k_twisted, plt.cm.Reds),
    ]
    for reg_name, mask, k_req, cmap in specs:
        idx = np.where(mask)[0]
        k_use = int(min(k_req, max(1, len(idx))))
        if len(idx) == 0:
            continue
        km = KMeans(n_clusters=k_use, random_state=42, n_init=15)
        sub = km.fit_predict(np.column_stack([x[idx], y[idx]]))
        palette = cmap(np.linspace(0.45, 0.92, k_use))
        for j, i in enumerate(idx):
            cid = offset + sub[j]
            cluster_id[i] = cid
            rgba[i] = palette[sub[j]]
            rgba[i, 3] = 0.95
        for c in range(k_use):
            members = idx[sub == c]
            meta_clusters.append({
                "regime": reg_name,
                "local_id": c,
                "global_id": offset + c,
                "members": members,
                "color": palette[c],
                "n": int(len(members)),
            })
        offset += k_use
    return {
        "cluster_id": cluster_id,
        "colors": rgba,
        "twisted": twisted,
        "clusters": meta_clusters,
        "n_aligned": int((~twisted).sum()),
        "n_twisted": int(twisted.sum()),
    }


def _draw_cluster_hulls_2d(ax, x: np.ndarray, y: np.ndarray, clusters: list[dict]) -> None:
    for cl in clusters:
        mem = cl["members"]
        if len(mem) < 3:
            continue
        pts = np.column_stack([x[mem], y[mem]])
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue
        hp = pts[hull.vertices]
        ax.fill(hp[:, 0], hp[:, 1], color=cl["color"], alpha=0.14, lw=0, zorder=1)
        ax.plot(np.r_[hp[:, 0], hp[0, 0]], np.r_[hp[:, 1], hp[0, 1]],
                color=cl["color"], lw=1.1, alpha=0.85, zorder=2)


def _draw_cluster_hulls_3d(
    ax, x: np.ndarray, y: np.ndarray, z: float, clusters: list[dict],
) -> None:
    for cl in clusters:
        mem = cl["members"]
        if len(mem) < 3:
            continue
        pts = np.column_stack([x[mem], y[mem]])
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue
        hp = pts[hull.vertices]
        ax.plot(
            np.r_[hp[:, 0], hp[0, 0]], np.r_[hp[:, 1], hp[0, 1]], np.full(len(hp) + 1, z),
            color=cl["color"], lw=1.1, alpha=0.80, zorder=3,
        )


def _draw_dual_plane_regime_clusters_3d(
    ax,
    pm: np.ndarray,
    pf: np.ndarray,
    clinfo: dict,
    *,
    z0: float,
    z1: float,
    xlabel: str,
    ylabel: str,
    plane0_label: str,
    plane1_label: str,
) -> None:
    colors = clinfo["colors"]
    _draw_plane_grid_3d(ax, pm[:, 0], pm[:, 1], z0, color="#1565c0", alpha=0.30)
    _draw_plane_grid_3d(ax, pf[:, 0], pf[:, 1], z1, color="#2e7d32", alpha=0.30)
    _draw_cluster_hulls_3d(ax, pm[:, 0], pm[:, 1], z0, clinfo["clusters"])
    _draw_cluster_hulls_3d(ax, pf[:, 0], pf[:, 1], z1, clinfo["clusters"])

    for i in range(len(pm)):
        ax.plot(
            [pm[i, 0], pf[i, 0]], [pm[i, 1], pf[i, 1]], [z0, z1],
            color=colors[i], alpha=0.14, lw=0.35, zorder=2,
        )

    ax.scatter(
        pm[:, 0], pm[:, 1], np.full(len(pm), z0), c=colors, s=18,
        edgecolors="0.25", linewidths=0.12, depthshade=False, zorder=4,
    )
    ax.scatter(
        pf[:, 0], pf[:, 1], np.full(len(pf), z1), c=colors, s=18,
        edgecolors="0.25", linewidths=0.12, depthshade=False, zorder=4,
    )

    ax.text(pm[:, 0].min(), pm[:, 1].max(), z0 + 0.05 * (z1 - z0), plane0_label, fontsize=8, color="#1565c0")
    ax.text(pf[:, 0].min(), pf[:, 1].max(), z1 + 0.03 * (z1 - z0), plane1_label, fontsize=8, color="#2e7d32")

    pad_x = 0.35 if "UMAP" not in xlabel else 0.5
    pad_y = 0.35 if "UMAP" not in xlabel else 0.5
    ax.set_xlim(pm[:, 0].min() - pad_x, max(pm[:, 0].max(), pf[:, 0].max()) + pad_x)
    ax.set_ylim(pm[:, 1].min() - pad_y, max(pm[:, 1].max(), pf[:, 1].max()) + pad_y)
    if "Valence" in xlabel:
        ax.set_xlim(*VA_LIM)
        ax.set_ylim(*VA_LIM)
    ax.set_zlim(z0 - 0.3, z1 + 0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("layer")
    ax.view_init(elev=24, azim=-58)


def _prepare_twist_methodology_bundles() -> dict:
    twist = _load_twist_full_frame()
    pm_va = twist[["valence_male", "arousal_male"]].to_numpy()
    pf_va = twist[["valence_female", "arousal_female"]].to_numpy()
    pm_umap = twist[["umap_score_male_x", "umap_score_male_y"]].to_numpy()
    pf_umap = twist[["umap_score_female_x", "umap_score_female_y"]].to_numpy()
    excess_va = twist["excess_va"].to_numpy()
    excess_umap = twist["excess_umap"].to_numpy()

    cl_va = _assign_regime_kmeans_clusters(pm_va[:, 0], pm_va[:, 1], excess_va)
    cl_umap = _assign_regime_kmeans_clusters(pm_umap[:, 0], pm_umap[:, 1], excess_umap)

    regime_colors_va = np.where(
        cl_va["twisted"][:, None],
        np.array([0.84, 0.15, 0.16, 0.9]),
        np.array([0.12, 0.38, 0.85, 0.9]),
    )

    return {
        "twist": twist,
        "pm_va": pm_va, "pf_va": pf_va,
        "pm_umap": pm_umap, "pf_umap": pf_umap,
        "excess_va": excess_va, "excess_umap": excess_umap,
        "cl_va": cl_va, "cl_umap": cl_umap,
        "regime_colors_va": regime_colors_va,
        "va_xlim": (1.0, 7.0), "va_ylim": (1.0, 7.0),
        "umap_xlim": (pm_umap[:, 0].min() - 0.5, max(pm_umap[:, 0].max(), pf_umap[:, 0].max()) + 0.5),
        "umap_ylim": (pm_umap[:, 1].min() - 0.5, max(pm_umap[:, 1].max(), pf_umap[:, 1].max()) + 0.5),
    }


def render_va_scores_gif_frame(
    ax: plt.Axes,
    pm_va: np.ndarray,
    pf_va: np.ndarray,
    regime_colors: np.ndarray,
    frame_idx: int,
    *,
    n_black: int,
    n_ramp: int,
    n_morph: int,
    n_hold: int = 0,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    title: str = "",
) -> None:
    """
  VA morph GIF 1 フレーム。
  [0, n_black): 黒点のみ @ male VA
  [n_black, n_black+n_ramp): male VA で aligned/twisted 色へ漸変
  [n_black+n_ramp, n_black+n_ramp+n_morph): グレー male VA 固定 + 色付き点が female へ移動
  [n_black+n_ramp+n_morph, ...): 終状態をホールド（n_hold フレーム）
    """
    ax.clear()
    i = frame_idx
    morph_start = n_black + n_ramp
    morph_end = morph_start + n_morph

    if i < n_black:
        ax.scatter(
            pm_va[:, 0], pm_va[:, 1], c="k", s=14, alpha=0.88,
            edgecolors="none", zorder=3,
        )
    elif i < n_black + n_ramp:
        ramp_i = i - n_black
        s = (ramp_i + 1) / n_ramp
        if s < 1.0:
            ax.scatter(
                pm_va[:, 0], pm_va[:, 1], c="k", s=14, alpha=(1.0 - s) * 0.85,
                edgecolors="none", zorder=2,
            )
        ax.scatter(
            pm_va[:, 0], pm_va[:, 1], c=regime_colors, s=14, alpha=s * 0.88,
            edgecolors="none", zorder=3,
        )
    elif i < morph_end:
        morph_i = i - morph_start
        morph_t = morph_i / max(n_morph - 1, 1)
        pos = pm_va + morph_t * (pf_va - pm_va)
        ax.scatter(
            pm_va[:, 0], pm_va[:, 1], c="0.55", s=10, alpha=0.60,
            edgecolors="none", zorder=1,
        )
        ax.scatter(
            pos[:, 0], pos[:, 1], c=regime_colors, s=14, alpha=0.82,
            edgecolors="none", zorder=3,
        )
    else:
        ax.scatter(
            pm_va[:, 0], pm_va[:, 1], c="0.55", s=10, alpha=0.60,
            edgecolors="none", zorder=1,
        )
        ax.scatter(
            pf_va[:, 0], pf_va[:, 1], c=regime_colors, s=14, alpha=0.82,
            edgecolors="none", zorder=3,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)


def render_points_morph_frame(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    colors: np.ndarray,
    morph_t: float,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    title: str = "",
) -> None:
    """GIF 1 フレーム: 点のみで start → end へモーフィング。"""
    ax.clear()
    t = float(np.clip(morph_t, 0.0, 1.0))
    pos = start + t * (end - start)
    ax.scatter(pos[:, 0], pos[:, 1], c=colors, s=14, alpha=0.80, edgecolors="none", zorder=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)


# ── Legacy phi-flow helpers (flat overlay / residual; kept for reference) ──
def _load_score_umap_flow_frame() -> pd.DataFrame:
    """Score UMAP 上の male→female 変位と excess twist（geo_umap_disp − model_twist_pm_pf）。"""
    twist = pd.read_csv(RES / "relational_cross_within_twist" / "relational_twist_per_image.csv")
    need = [
        "umap_score_male_x", "umap_score_male_y",
        "umap_score_female_x", "umap_score_female_y",
        "geo_umap_disp", "model_twist_pm_pf",
    ]
    twist = twist.dropna(subset=need).copy()
    twist["excess_twist"] = twist["geo_umap_disp"] - twist["model_twist_pm_pf"]
    return twist


def _affine_umap_phi(pm: np.ndarray, pf: np.ndarray) -> np.ndarray:
    """Score UMAP 上の最小二乗アフィン Φ̂: p_m ↦ A p_m + b（グローバル線形流れの代理）。"""
    ones = np.ones((len(pm), 1))
    design = np.hstack([pm, ones])
    coef, _, _, _ = np.linalg.lstsq(design, pf, rcond=None)
    return design @ coef


def _binned_mean_vectors(
    x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray, *, n: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """規則格子上の平均ベクトル（streamplot 用）。"""
    xedges = np.linspace(x.min(), x.max(), n + 1)
    yedges = np.linspace(y.min(), y.max(), n + 1)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    xi = np.clip(np.digitize(x, xedges) - 1, 0, n - 1)
    yi = np.clip(np.digitize(y, yedges) - 1, 0, n - 1)
    u_sum = np.zeros((n, n))
    v_sum = np.zeros((n, n))
    cnt = np.zeros((n, n))
    for k in range(len(x)):
        u_sum[yi[k], xi[k]] += u[k]
        v_sum[yi[k], xi[k]] += v[k]
        cnt[yi[k], xi[k]] += 1
    with np.errstate(invalid="ignore"):
        u_mean = np.where(cnt >= 3, u_sum / cnt, np.nan)
        v_mean = np.where(cnt >= 3, v_sum / cnt, np.nan)
    xg, yg = np.meshgrid(xc, yc, indexing="xy")
    return xg, yg, u_mean, v_mean


def _draw_distribution_overlay(
    ax, pts: np.ndarray, *, outline: bool, color: str, label: str, alpha: float | None = None,
) -> None:
    """分布の輪郭（male）または塗り（female）を ConvexHull で薄く重ねる。"""
    if len(pts) < 4:
        return
    hull = ConvexHull(pts)
    poly = pts[hull.vertices]
    if outline:
        a = 0.75 if alpha is None else alpha
        ax.plot(np.r_[poly[:, 0], poly[0, 0]], np.r_[poly[:, 1], poly[0, 1]],
                color=color, lw=1.6, alpha=a, zorder=1, label=label)
    else:
        a = 0.10 if alpha is None else alpha
        ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=a, zorder=0, label=label)


def _prepare_phi_flow_data() -> dict:
    """phi flow 図・GIF 共通の配列を構築。"""
    twist = _load_score_umap_flow_frame()
    pm = twist[["umap_score_male_x", "umap_score_male_y"]].to_numpy()
    pf = twist[["umap_score_female_x", "umap_score_female_y"]].to_numpy()
    disp = pf - pm
    geo = twist["geo_umap_disp"].to_numpy()
    model = twist["model_twist_pm_pf"].to_numpy()
    excess = twist["excess_twist"].to_numpy()
    phi_pred = _affine_umap_phi(pm, pf)
    phi_disp = phi_pred - pm
    scale = np.divide(model, np.clip(geo, 1e-9, None))
    linear_pt = disp * scale[:, None]
    residual = disp - linear_pt
    hi_thr = float(np.quantile(excess, 2 / 3))
    mid = 0.5 * (pm + pf)
    pad = 0.6
    xlim = (pm[:, 0].min() - pad, pm[:, 0].max() + pad)
    ylim = (pm[:, 1].min() - pad, pm[:, 1].max() + pad)
    cnorm = Normalize(vmin=float(np.percentile(excess, 8)), vmax=float(np.percentile(excess, 92)))
    cmap = colormaps["coolwarm"]
    gx = np.linspace(xlim[0], xlim[1], 90)
    gy = np.linspace(ylim[0], ylim[1], 90)
    gx_m, gy_m = np.meshgrid(gx, gy)
    exc_grid = griddata(mid, excess, (gx_m, gy_m), method="linear")
    exc_grid = np.ma.masked_invalid(exc_grid)
    return {
        "pm": pm, "pf": pf, "disp": disp, "excess": excess,
        "phi_pred": phi_pred, "phi_disp": phi_disp,
        "linear_pt": linear_pt, "residual": residual,
        "hi_thr": hi_thr, "mid": mid, "xlim": xlim, "ylim": ylim,
        "cnorm": cnorm, "cmap": cmap,
        "gx_m": gx_m, "gy_m": gy_m, "exc_grid": exc_grid,
    }


def _draw_plane_grid_3d(
    ax, x: np.ndarray, y: np.ndarray, z: float, *, n: int = 16,
    color: str = "0.75", alpha: float = 0.35,
) -> None:
    xg = np.linspace(x.min(), x.max(), n)
    yg = np.linspace(y.min(), y.max(), n)
    for xv in xg:
        ax.plot([xv, xv], [yg[0], yg[-1]], [z, z], color=color, lw=0.35, alpha=alpha)
    for yv in yg:
        ax.plot([xg[0], xg[-1]], [yv, yv], [z, z], color=color, lw=0.35, alpha=alpha)


def _draw_phi_flow_flat_ax(
    ax, data: dict, *, fig: plt.Figure | None = None, morph_t: float = 1.0, show_regions: bool = True,
) -> None:
    """1 枚平面への重ね描画（静止画 or GIF フレーム）。"""
    pm, pf = data["pm"], data["pf"]
    disp, excess = data["disp"], data["excess"]
    phi_disp = data["phi_disp"]
    cmap, cnorm = data["cmap"], data["cnorm"]
    gx_m, gy_m, exc_grid = data["gx_m"], data["gy_m"], data["exc_grid"]
    hi_thr = data["hi_thr"]
    xlim, ylim = data["xlim"], data["ylim"]
    t = float(np.clip(morph_t, 0.0, 1.0))
    pos = pm + t * disp

    if show_regions:
        ax.contourf(gx_m, gy_m, exc_grid, levels=14, cmap="coolwarm", alpha=0.22, zorder=0)
        ax.contour(gx_m, gy_m, exc_grid, levels=[hi_thr], colors=["#b71c1c"], linewidths=1.4,
                   linestyles="--", zorder=1)
        ax.contourf(gx_m, gy_m, np.where(exc_grid >= hi_thr, 1.0, np.nan), levels=[0.5, 1.5],
                    colors=["none", "#ffcdd2"], alpha=0.35, hatches=["", "///"], zorder=0)

    male_a = max(0.0, 1.0 - t)
    female_a = t
    if male_a > 0.02:
        _draw_distribution_overlay(
            ax, pm, outline=True, color="#1565c0", alpha=0.75 * male_a,
            label="male score (start)" if t < 0.5 else "_nolegend_",
        )
    if female_a > 0.02:
        _draw_distribution_overlay(
            ax, pf, outline=False, color="#2e7d32", alpha=0.10 + 0.15 * female_a,
            label="female score (end)" if t > 0.5 else "_nolegend_",
        )

    if t >= 0.85:
        xg, yg, ug, vg = _binned_mean_vectors(pm[:, 0], pm[:, 1], phi_disp[:, 0], phi_disp[:, 1], n=18)
        with np.errstate(invalid="ignore"):
            ax.streamplot(xg, yg, ug, vg, color="0.55", linewidth=0.7, density=1.2,
                          arrowsize=0.9, zorder=2)

    phi_alpha = 0.15 + 0.25 * (1.0 - t)
    ax.quiver(pm[:, 0], pm[:, 1], phi_disp[:, 0] * t, phi_disp[:, 1] * t,
              angles="xy", scale_units="xy", scale=1.0, color="0.72", alpha=phi_alpha,
              width=0.0018, headwidth=3.2, headlength=3.5, zorder=3)

    arrow_alpha = 0.25 + 0.55 * t
    ax.quiver(pm[:, 0], pm[:, 1], disp[:, 0] * t, disp[:, 1] * t,
              angles="xy", scale_units="xy", scale=1.0,
              color=cmap(cnorm(excess)), alpha=arrow_alpha,
              width=0.0022, headwidth=3.4, headlength=4.0, zorder=4)

    ax.scatter(pos[:, 0], pos[:, 1], s=7, c=cmap(cnorm(excess)), alpha=0.35 + 0.45 * t,
               edgecolors="none", zorder=5)

    if show_regions and t >= 0.95:
        lo_mask = excess <= np.median(excess)
        hi_mask = excess >= hi_thr
        if lo_mask.any():
            cx, cy = pm[lo_mask].mean(axis=0)
            ax.annotate("linear transfer\n(low excess twist)", xy=(cx, cy), fontsize=8, color="#0d47a1",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.35", fc="#e3f2fd", ec="#1565c0", alpha=0.92))
        if hi_mask.any():
            cx, cy = pm[hi_mask].mean(axis=0)
            ax.annotate("twist: linear Φ fails\n(high excess twist)", xy=(cx, cy), fontsize=8, color="#b71c1c",
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.35", fc="#ffebee", ec="#c62828", alpha=0.92))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Aligned score UMAP-1")
    ax.set_ylabel("Aligned score UMAP-2")

    if fig is not None and morph_t >= 0.99:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("excess twist = UMAP disp − model twist")


def _draw_phi_flow_residual_ax(ax, data: dict) -> None:
    pm, excess = data["pm"], data["excess"]
    linear_pt, residual = data["linear_pt"], data["residual"]
    cmap, cnorm = data["cmap"], data["cnorm"]
    gx_m, gy_m, exc_grid = data["gx_m"], data["gy_m"], data["exc_grid"]
    hi_thr, xlim, ylim = data["hi_thr"], data["xlim"], data["ylim"]
    res_mag = np.linalg.norm(residual, axis=1)

    ax.contourf(gx_m, gy_m, exc_grid, levels=14, cmap="coolwarm", alpha=0.18, zorder=0)
    ax.quiver(pm[:, 0], pm[:, 1], linear_pt[:, 0], linear_pt[:, 1],
              angles="xy", scale_units="xy", scale=1.0, color="0.65", alpha=0.45,
              width=0.0020, headwidth=3.0, zorder=2, label="Φ-predicted component (model twist)")
    ax.quiver(pm[:, 0], pm[:, 1], residual[:, 0], residual[:, 1],
              angles="xy", scale_units="xy", scale=1.0,
              color=cmap(cnorm(excess)), alpha=0.75,
              width=0.0024, headwidth=3.2, zorder=3, label="residual (excess twist vector)")
    ax.scatter(pm[:, 0], pm[:, 1], s=7, c=res_mag, cmap="magma", vmin=0,
               vmax=float(np.percentile(res_mag, 92)), alpha=0.35, edgecolors="none", zorder=4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Aligned score UMAP-1")
    ax.set_ylabel("Aligned score UMAP-2")
    ax.set_title("Decomposition: observed Δ = linear Φ component + residual\n"
                 f"(top tercile excess twist ≥ {hi_thr:.2f})", fontsize=9)
    ax.legend(frameon=False, fontsize=7, loc="upper left")


def _draw_phi_flow_dual_plane_3d(ax, data: dict, *, z0: float = 0.0, z1: float = 5.0) -> plt.cm.ScalarMappable:
    """z 軸で male / female score 平面を分離し、層間コネクタでねじれを可視化。"""
    pm, pf = data["pm"], data["pf"]
    excess = data["excess"]
    phi_pred = data["phi_pred"]
    cmap, cnorm = data["cmap"], data["cnorm"]

    _draw_plane_grid_3d(ax, pm[:, 0], pm[:, 1], z0, color="#1565c0", alpha=0.42)
    _draw_plane_grid_3d(ax, pf[:, 0], pf[:, 1], z1, color="#2e7d32", alpha=0.42)

    for i in range(len(pm)):
        rgba = (*cmap(cnorm(excess[i]))[:3], 0.38)
        ax.plot(
            [pm[i, 0], pf[i, 0]], [pm[i, 1], pf[i, 1]], [z0, z1],
            color=rgba, lw=0.45, zorder=2,
        )
        ax.plot(
            [pm[i, 0], phi_pred[i, 0]], [pm[i, 1], phi_pred[i, 1]], [z0, z1],
            color=(0.72, 0.72, 0.72, 0.10), lw=0.25, zorder=1,
        )

    colors = cmap(cnorm(excess))
    ax.scatter(pm[:, 0], pm[:, 1], np.full(len(pm), z0), c=colors, s=14,
               edgecolors="0.35", linewidths=0.15, depthshade=False, zorder=4)
    ax.scatter(pf[:, 0], pf[:, 1], np.full(len(pf), z1), c=colors, s=14,
               edgecolors="0.35", linewidths=0.15, depthshade=False, zorder=4)

    ax.text(pm[:, 0].min(), pm[:, 1].max(), z0 + 0.06 * (z1 - z0),
            "male score plane", fontsize=8, color="#1565c0")
    ax.text(pf[:, 0].min(), pf[:, 1].max(), z1 + 0.04 * (z1 - z0),
            "female score plane", fontsize=8, color="#2e7d32")

    pad = 0.5
    ax.set_xlim(pm[:, 0].min() - pad, max(pm[:, 0].max(), pf[:, 0].max()) + pad)
    ax.set_ylim(pm[:, 1].min() - pad, max(pm[:, 1].max(), pf[:, 1].max()) + pad)
    ax.set_zlim(z0 - 0.3, z1 + 0.6)
    ax.set_xlabel("Aligned score UMAP-1")
    ax.set_ylabel("Aligned score UMAP-2")
    ax.set_zlabel("layer")
    ax.view_init(elev=24, azim=-58)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)
    sm.set_array([])
    return sm


def paper_fig1() -> tuple[Path, Path]:
    """NatComm Fig.1 — research design schematic (A–D)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

    fig = plt.figure(figsize=(11.2, 7.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

    def _box(ax, xy, w, h, text, fc="#e3f2fd", ec="#1565c0"):
        p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                           facecolor=fc, edgecolor=ec, linewidth=1.3, transform=ax.transAxes)
        ax.add_patch(p)
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
                fontsize=8, transform=ax.transAxes, wrap=True)

    def _arrow(ax, a, b):
        ax.annotate("", xy=b, xytext=a, xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", color="0.35", lw=1.4))

    # A: frozen CLIP
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    ax.set_title("Frozen perceptual encoder", fontsize=10, pad=6)
    _box(ax, (0.05, 0.38), 0.22, 0.28, "Image\nstimuli", fc="#fff3e0", ec="#ef6c00")
    _arrow(ax, (0.28, 0.52), (0.38, 0.52))
    _box(ax, (0.38, 0.32), 0.28, 0.40, "CLIP ViT-B/32\nfrozen\nshared 512-d", fc="#e8f5e9", ec="#2e7d32")
    _arrow(ax, (0.67, 0.52), (0.77, 0.52))
    _box(ax, (0.77, 0.38), 0.18, 0.28, "x ∈ ℝ⁵¹²\nsame for all", fc="#e3f2fd", ec="#1565c0")
    ax.text(0.5, 0.08, "Perceptual features held constant across groups",
            ha="center", fontsize=7.5, style="italic", transform=ax.transAxes, color="0.35")

    # B: readout branch
    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "B")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    ax.set_title("Group differences at readout", fontsize=10, pad=6)
    _box(ax, (0.05, 0.40), 0.18, 0.24, "shared x", fc="#e8f5e9", ec="#2e7d32")
    _arrow(ax, (0.24, 0.62), (0.38, 0.72))
    _arrow(ax, (0.24, 0.42), (0.38, 0.32))
    _box(ax, (0.38, 0.62), 0.28, 0.22, "Ridge λ_M\n→ VA_male", fc="#ffcdd2", ec="#c62828")
    _box(ax, (0.38, 0.20), 0.28, 0.22, "Ridge λ_F\n→ VA_female", fc="#f8bbd0", ec="#ad1457")
    _arrow(ax, (0.67, 0.73), (0.78, 0.73))
    _arrow(ax, (0.67, 0.31), (0.78, 0.31))
    _box(ax, (0.78, 0.62), 0.18, 0.22, "VA_M", fc="#ffebee", ec="#c62828")
    _box(ax, (0.78, 0.20), 0.18, 0.22, "VA_F", fc="#fce4ec", ec="#ad1457")
    ax.text(0.5, 0.05, "Design localizes group structure in readout weights",
            ha="center", fontsize=7.5, style="italic", transform=ax.transAxes, color="0.35")

    # C: two boundaries
    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "C")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    ax.set_title("Two population boundaries", fontsize=10, pad=6)
    _box(ax, (0.08, 0.55), 0.38, 0.32, "Gender boundary\nOASIS male ↔ female\n(n≈900 images)\nprobe for mechanism",
         fc="#e3f2fd", ec="#1565c0")
    _box(ax, (0.54, 0.55), 0.38, 0.32, "Culture boundary\nOASIS ↔ Japan\n(n=96 images, 49 raters)\nindependent transfer test",
         fc="#f3e5f5", ec="#6a1b9a")
    _box(ax, (0.22, 0.12), 0.56, 0.28, "Gender = validation boundary (not a claim that gender is primary)\n"
         "Shared frozen encoder + group-specific valuation maps",
         fc="#fafafa", ec="0.5")

    # D: analysis hierarchy
    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "D")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    ax.set_title("Analysis hierarchy", fontsize=10, pad=6)
    steps = [
        (0.72, "① Global linear bridge Φ\n(coordinate correction)"),
        (0.45, "② Structured residual / twist\n(where Φ is incomplete)"),
        (0.18, "③ Case translation\n(emotion-equivalent exemplars)"),
    ]
    for i, (y, txt) in enumerate(steps):
        _box(ax, (0.18, y), 0.64, 0.20, txt, fc=["#e8f5e9", "#fff8e1", "#e3f2fd"][i],
             ec=["#2e7d32", "#f9a825", "#1565c0"][i])
        if i < 2:
            _arrow(ax, (0.5, y - 0.02), (0.5, steps[i + 1][0] + 0.20))
    ax.text(0.5, 0.04, "Low-DOF Φ → residual structure → complementary local mode",
            ha="center", fontsize=7.5, style="italic", transform=ax.transAxes, color="0.35")

    fig.suptitle("Fig. 1 | Research design: frozen perception, readout locus, two boundaries, and hierarchical bridging",
                 y=0.995, fontsize=11)
    return save_dual(fig, "Paper_Fig1_research_design")


def _style_violin(ax, parts, facecolor: str, alpha: float = 0.45) -> None:
    for b in parts.get("bodies", []):
        b.set_facecolor(facecolor)
        b.set_alpha(alpha)
        b.set_edgecolor("0.45")
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("0.3")
            parts[key].set_linewidth(1.0)


# ── Fig 2 ─────────────────────────────────────────────────────────────
def _draw_lambda_common_panel(ax, dim: str, title: str, *, vlim: float | None = None):
    """OASIS common-ref Δλ heatmap + cluster-corrected contours (Fig.2-D)."""
    npz_path = LAMBDA_FIG2 / "lambda_gender_diff_common_ref_fig2.npz"
    if not npz_path.exists():
        ax.text(0.5, 0.5, f"Missing:\n{npz_path.name}", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None
    z = np.load(npz_path)
    extent = z["extent"].tolist()
    mean_map = z[f"mean_map_{dim}"]
    labels_sig = z[f"labels_sig_{dim}"]
    if vlim is None:
        vlim = float(np.nanmax(np.abs(mean_map))) if np.any(np.isfinite(mean_map)) else 0.2
        vlim = max(vlim, 0.05)
    im = ax.imshow(
        mean_map, origin="lower", extent=extent, cmap="coolwarm",
        vmin=-vlim, vmax=vlim, aspect="equal",
    )
    if np.any(labels_sig > 0):
        ax.contour(
            labels_sig.astype(float), levels=[0.5], colors="k", linewidths=1.4,
            extent=extent, origin="lower",
        )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_box_aspect(1)
    ax.set_xlabel("Valence (OASIS common ref)")
    ax.set_ylabel("Arousal (OASIS common ref)")
    meta_path = LAMBDA_FIG2 / "lambda_gender_diff_common_ref_fig2.json"
    note = ""
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for p in meta.get("panels", []):
            if p.get("dimension") == dim and p.get("ref") == "common":
                note = f"p<0.001; {p.get('n_sig_clusters', 0)} sig. regions"
                break
    ax.set_title(f"{title}\n{note}", fontsize=8.5)
    return im


def paper_fig2() -> tuple[Path, Path]:
    """
    Fig.2 A–D: decoder-locus with distributions + null references.
    Data: criteria_references, fig2 boot/null/npz, lambda common-ref export.
    """
    refs_path = CVAE / "paper2_layer1_criteria_references.json"
    if not refs_path.exists():
        raise FileNotFoundError(
            f"Missing {refs_path.name}. Run analysis_paper2_layer1_criteria_references.py"
        )
    refs_full = json.loads(refs_path.read_text(encoding="utf-8"))
    rows = refs_full["rows"]
    seeds = [int(r["train_seed"]) for r in rows]
    n_seeds = len(seeds)
    a_ref = refs_full["summary"]["criterion_a"]
    b_m = refs_full["summary"]["criterion_b"]["male"]
    b_f = refs_full["summary"]["criterion_b"]["female"]
    sesoi = float(refs_full["sesoi_delta_r2"])

    boot_path = CVAE / "paper2_layer1_fig2_criterion_a_boot.csv"
    null_path = CVAE / "paper2_layer1_fig2_criterion_a_label_shuffle_null.json"
    cka_path = CVAE / "paper2_layer1_fig2_criterion_c_cka_null_arrays.npz"
    boot_df = pd.read_csv(boot_path) if boot_path.exists() else None
    label_null = json.loads(null_path.read_text()) if null_path.exists() else None
    cka_npz = np.load(cka_path) if cka_path.exists() else None

    fig = plt.figure(figsize=(12.0, 9.8))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.1, 1.35], hspace=0.38)
    gs_top = gs[0].subgridspec(1, 3, wspace=0.38)
    gs_bot = gs[1].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.045], wspace=0.28)

    # ── A: three violins — label-shuffle null / boot pool / observed (10 seeds) ──
    axA = fig.add_subplot(gs_top[0, 0])
    panel_label(axA, "A")
    delta_pts = np.asarray([r["criterion_a"]["delta_r2"] for r in rows], float)
    boot_pool = boot_df["delta_r2"].to_numpy(float) if boot_df is not None else None
    null_arr = (
        np.asarray(label_null["null_delta_r2"], float) if label_null is not None else None
    )
    a_groups: list[tuple[str, np.ndarray, str]] = []
    if null_arr is not None:
        a_groups.append(("label-shuffle\nnull", null_arr, "#bdbdbd"))
    if boot_pool is not None:
        a_groups.append(("boot\npool", boot_pool, "#90caf9"))
    a_groups.append(("observed\n(10 seeds)", delta_pts, "#1565c0"))
    for i, (lab, arr, col) in enumerate(a_groups):
        _style_violin(axA, axA.violinplot([arr], positions=[i], showmedians=True, widths=0.72), col)
    axA.axhline(0, color="0.35", lw=1.0, ls="--", zorder=0)
    axA.axhspan(-sesoi, sesoi, color="#c62828", alpha=0.05, zorder=0)
    axA.axhline(sesoi, color="#c62828", lw=0.9, ls=":", alpha=0.6)
    axA.axhline(-sesoi, color="#c62828", lw=0.9, ls=":", alpha=0.6)
    axA.set_xticks(range(len(a_groups)))
    axA.set_xticklabels([g[0] for g in a_groups], fontsize=7)
    axA.set_ylabel("ΔR² (split − shared)")
    null_note = ""
    if label_null is not None:
        null_note = f"; null median {label_null['null_summary']['median']:+.3f}"
    axA.set_title(
        f"Criterion A: no encoder gain\n"
        f"mean {a_ref['delta_r2_mean']:+.4f}; "
        f"{a_ref['n_within_sesoi_abs']}/{n_seeds} within |Δ|<{sesoi:.2f}{null_note}",
        fontsize=8.2,
    )
    yvals = [delta_pts]
    if boot_pool is not None:
        yvals.append(boot_pool)
    if null_arr is not None:
        yvals.append(null_arr)
    yabs = max(sesoi * 1.4, float(np.max(np.abs(np.concatenate(yvals)))) * 1.12, 0.06)
    axA.set_xlim(-0.6, len(a_groups) - 0.4)
    axA.set_ylim(-yabs, yabs)

    # ── B: per-image MSE scatter (held-out test): swap vs recovery vs native ──
    axB = fig.add_subplot(gs_top[0, 1])
    panel_label(axB, "B")
    b_csv = CVAE / "paper2_layer1_fig2_criterion_b_per_image_test.csv"
    if b_csv.exists():
        bdf = pd.read_csv(b_csv)
        mse_n = bdf["mse_native"].to_numpy(float)
        mse_s = bdf["mse_swap"].to_numpy(float)
        mse_c = bdf["mse_corrected"].to_numpy(float)
        deg = mse_s - mse_n
        rec = mse_s - mse_c
        rr = rec / np.maximum(deg, 1e-12)
        rr_med = float(np.median(rr))
        from scipy.stats import spearmanr
        sp = float(spearmanr(deg, rec).correlation)
        for i in range(len(mse_n)):
            axB.plot(
                [mse_s[i], mse_c[i]], [mse_n[i], mse_n[i]],
                color="0.78", lw=0.35, alpha=0.35, zorder=1,
            )
        axB.scatter(
            mse_s, mse_n, s=22, c="#d32f2f", alpha=0.80,
            edgecolors="white", linewidths=0.3, label="swap", zorder=3,
        )
        axB.scatter(
            mse_c, mse_n, s=14, c="#7b1fa2", alpha=0.55,
            edgecolors="white", linewidths=0.25, label="recovery", zorder=4,
        )
        lim = float(max(mse_n.max(), mse_s.max(), mse_c.max())) + 0.15
        axB.plot([0, lim], [0, lim], ls="--", color="0.4", lw=1.0, label="y = x", zorder=2)
        axB.set_xlim(0, lim)
        axB.set_ylim(0, lim)
        axB.set_aspect("equal", adjustable="box")
        axB.set_xlabel("MSE (swap / recovery)")
        axB.set_ylabel("native MSE")
        axB.set_title(
            f"Criterion B: swap collapse → linear recovery\n"
            f"held-out n={len(bdf)}; med recovery/degrad≈{rr_med:.0%}; "
            f"Spearman={sp:+.2f}",
            fontsize=8.0,
        )
        axB.legend(frameon=False, fontsize=6.2, loc="upper left")
    else:
        axB.text(
            0.5, 0.5,
            "Missing criterion_b_per_image_test.csv",
            ha="center", va="center", transform=axB.transAxes, fontsize=8,
        )

    # ── C: observed CKA vs label-shuffle null (same estimand as Panel A) ──
    axC = fig.add_subplot(gs_top[0, 2])
    panel_label(axC, "C")
    ac_null_path = CVAE / "paper2_layer1_fig2_label_shuffle_ac_null.json"
    if not ac_null_path.exists() and null_path is not None and Path(null_path).exists():
        # fall back: A null JSON may already include CKA after joint generation
        ac_null_path = Path(null_path)
    ac_null = None
    if ac_null_path.exists():
        ac_null = json.loads(Path(ac_null_path).read_text(encoding="utf-8"))
    cka_obs_seeds = None
    if cka_npz is not None:
        cka_obs_seeds = np.asarray(cka_npz["cka_obs"], float)
    elif rows:
        cka_obs_seeds = np.asarray(
            [r["criterion_c"]["observed"] for r in rows if "criterion_c" in r], float,
        )

    if ac_null is not None and "null_cka" in ac_null and cka_obs_seeds is not None:
        null_cka = np.asarray(ac_null["null_cka"], float)
        _style_violin(
            axC,
            axC.violinplot([null_cka], positions=[0], showmedians=True, widths=0.7),
            "#bdbdbd",
        )
        # observed: strip of 10 seeds + mean marker
        rng = np.random.default_rng(0)
        axC.scatter(
            np.full(len(cka_obs_seeds), 1.0) + rng.uniform(-0.08, 0.08, len(cka_obs_seeds)),
            cka_obs_seeds,
            s=36, c="#1565c0", alpha=0.9, edgecolors="white", linewidths=0.5, zorder=4,
            label="observed (10 seeds)",
        )
        axC.axhline(
            float(np.median(null_cka)), color="0.45", ls=":", lw=1.0,
            label=f"null median {np.median(null_cka):.2f}",
        )
        p_c = float(ac_null.get("p_obs_cka_gt_null", float("nan")))
        axC.set_xticks([0, 1])
        axC.set_xticklabels(["label-shuffle\nnull", "observed"], fontsize=7)
        axC.set_xlim(-0.55, 1.55)
        axC.set_ylim(-0.02, 1.0)
        axC.set_ylabel("Linear CKA (z_m, z_f)")
        axC.set_title(
            f"Criterion C: obs ≫ label-shuffle null\n"
            f"obs {cka_obs_seeds.min():.3f}–{cka_obs_seeds.max():.3f}; "
            f"null med {np.median(null_cka):.2f}; p={p_c:.3g}",
            fontsize=8.0,
        )
        axC.legend(frameon=False, fontsize=6.0, loc="lower right")
    elif cka_npz is not None:
        # Temporary fallback: row-permutation null (legacy) until AC null finishes
        obs = cka_npz["cka_obs"]
        null_q025 = float(np.quantile(cka_npz["cka_null"], 0.025))
        null_q975 = float(np.quantile(cka_npz["cka_null"], 0.975))
        null_med = float(np.median(cka_npz["cka_null"]))
        _style_violin(
            axC, axC.violinplot([obs], positions=[0], showmedians=True, widths=0.55), "#1565c0",
        )
        axC.axhspan(null_q025, null_q975, color="#bdbdbd", alpha=0.35, zorder=0, label="row-perm null 95%")
        axC.axhline(null_med, color="0.55", ls=":", lw=1.0, label=f"null median ≈ {null_med:.2f}")
        axC.set_xticks([0])
        axC.set_xticklabels(["observed\n(10 seeds)"], fontsize=7)
        axC.set_ylim(-0.02, 0.95)
        axC.set_ylabel("Linear CKA (z_m, z_f)")
        axC.set_title(
            f"Criterion C: high cross-encoder similarity\n"
            f"obs {obs.min():.3f}–{obs.max():.3f} ≫ row-perm null (legacy)",
            fontsize=8.0,
        )
        axC.legend(frameon=False, fontsize=6.0, loc="lower right")
        axC.set_xlim(-0.55, 0.55)
    else:
        axC.text(0.5, 0.5, "Missing CKA null", ha="center", va="center", transform=axC.transAxes)

    # ── D: λ maps (common ref only), Valence + Arousal (equal panel size) ──
    axD0 = fig.add_subplot(gs_bot[0, 0])
    panel_label(axD0, "D")
    axD1 = fig.add_subplot(gs_bot[0, 1])
    # Shared color scale so both maps are comparable and the colorbar is honest
    _npz = LAMBDA_FIG2 / "lambda_gender_diff_common_ref_fig2.npz"
    vlim_d = 0.05
    if _npz.exists():
        _z = np.load(_npz)
        for _dim in ("valence", "arousal"):
            _m = _z[f"mean_map_{_dim}"]
            if np.any(np.isfinite(_m)):
                vlim_d = max(vlim_d, float(np.nanmax(np.abs(_m))))
    im0 = _draw_lambda_common_panel(
        axD0, "valence", "Δλ Valence (male − female)", vlim=vlim_d,
    )
    im1 = _draw_lambda_common_panel(
        axD1, "arousal", "Δλ Arousal (male − female)", vlim=vlim_d,
    )
    if im0 is not None and im1 is not None:
        cax = fig.add_subplot(gs_bot[0, 2])
        cbar = fig.colorbar(im1, cax=cax)
        cbar.set_label("Local mean Δλ (V/A units)")

    fig.suptitle(
        "Fig. 2 | Group differences localize to readout (encoder falsified; λ structure positive)",
        y=0.995, fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save_dual(fig, "Paper_Fig2_decoder_locus_lambda")


# ── Fig 3 — linear bridge geometry + Z3 generalization (A–F) ───────────
def _draw_z3_curve_panel(ax, m2f: pd.DataFrame, f2m: pd.DataFrame, z3: dict) -> None:
    for df, lab, col in [
        (m2f, "M→F", "#1565c0"),
        (f2m, "F→M", "#c62828"),
    ]:
        ax.plot(df["k"], df["R2_median"], "o-", color=col, lw=1.8, ms=5, label=lab)
        ax.fill_between(df["k"], df["R2_q025"], df["R2_q975"], color=col, alpha=0.15)
    ceil = float(m2f["ceiling"].iloc[0])
    ax.axhline(ceil, color="0.25", ls="--", lw=1.1, label=f"ceiling {ceil:.2f}")
    ax.axhline(0.9 * ceil, color="0.55", ls=":", lw=1.0, label="90%×ceiling")
    k_star = z3["bridges"]["gender_M_to_F"].get("k_star")
    if k_star is not None:
        ax.axvline(k_star, color="#6a1b9a", ls="--", alpha=0.85, label=f"k*≈{k_star}")
    ax.set_xscale("log")
    ax.set_xticks(m2f["k"].tolist())
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("k shared anchors (fit themes)")
    ax.set_ylabel("Understanding R² (held-out)")
    ax.set_title("Few shared anchors generalize")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_ylim(-0.2, 1.0)


def _draw_z3_layer_panel(ax, m2f: pd.DataFrame) -> None:
    ceil = float(m2f["ceiling"].iloc[0])
    layer_cols = {"near": "#2e7d32", "mid": "#f9a825", "far": "#c62828"}
    for L, c in layer_cols.items():
        ax.plot(m2f["k"], m2f[f"R2_{L}_median"], "o-", color=c, lw=1.7, ms=5, label=L)
    ax.axhline(ceil, color="0.25", ls="--", lw=1.0)
    ax.set_xscale("log")
    ax.set_xticks(m2f["k"].tolist())
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("k shared anchors")
    ax.set_ylabel("Layer R² (M→F held-out)")
    near34 = float(m2f.loc[m2f["k"] == 34, "R2_near_median"].iloc[0])
    far34 = float(m2f.loc[m2f["k"] == 34, "R2_far_median"].iloc[0])
    ax.set_title(f"Distance-dependent cost\n(near {near34:.2f} ≫ far {far34:.2f} at k=34)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.set_ylim(-0.5, 1.05)


def _draw_z3_curve_panel_compact(ax, m2f: pd.DataFrame, f2m: pd.DataFrame, z3: dict) -> None:
    """Fig.3-E: quantity (saturation) only; near/mid/far layers → Supp."""
    for df, lab, col in [
        (m2f, "M→F", "#1565c0"),
        (f2m, "F→M", "#c62828"),
    ]:
        ax.plot(df["k"], df["R2_median"], "o-", color=col, lw=1.8, ms=5, label=lab)
        ax.fill_between(df["k"], df["R2_q025"], df["R2_q975"], color=col, alpha=0.12)
    ceil = float(m2f["ceiling"].iloc[0])
    ax.axhline(ceil, color="0.25", ls="--", lw=1.0, label=f"ceiling {ceil:.2f}")
    ax.axhline(0.9 * ceil, color="0.55", ls=":", lw=0.9, label="90%×ceiling")
    k_star = z3["bridges"]["gender_M_to_F"].get("k_star")
    if k_star is not None:
        ax.axvline(k_star, color="#6a1b9a", ls="--", alpha=0.75, label=f"k*≈{k_star}")
    # Exactly-determined affine at k=3 (6 DOF / 3 points) — algebraic valley.
    if 3 in set(m2f["k"].tolist()):
        ax.axvline(3, color="#ef6c00", ls=":", lw=1.0, alpha=0.85, zorder=1)
        ax.text(
            3.15, -0.12, "k=3 exactly\ndetermined",
            fontsize=5.8, color="#ef6c00", ha="left", va="bottom",
        )
    ax.set_xscale("log")
    ax.set_xticks(m2f["k"].tolist())
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("k shared anchors")
    ax.set_ylabel("Understanding R² (held-out)")
    ax.set_title("Few anchors saturate (quantity)")
    ax.legend(frameon=False, fontsize=6.0, loc="lower right")
    ax.set_ylim(-0.2, 1.0)


def _load_z5_minimal_anchor_data() -> tuple[dict, dict, tuple[float, float], tuple[float, float]]:
    from analysis_anchor_structure import z5_load_reference_points

    profile_path = ANC / "z5_minimal_anchor_profile.json"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Missing {profile_path.name}. Run: "
            "python3 code/analysis_anchor_structure.py --mode z5 --z5-agreement"
        )
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    profiles = payload["profiles"]
    overlap_path = ANC / "z5_overlap.json"
    overlap = (
        json.loads(overlap_path.read_text(encoding="utf-8"))
        if overlap_path.exists()
        else profiles.get("MtoF", {}).get("overlap_with_opposite", {})
    )
    refs = z5_load_reference_points()
    ov, oa = refs["oasis_centroid"]
    gv, ga = refs["gender_fp"]
    return profiles, overlap, (ov, oa), (gv, ga)


def _load_z5_matched_k4_anchor_data() -> tuple[dict, dict, tuple[float, float], tuple[float, float]]:
    from analysis_anchor_structure import (
        SRC_F,
        va_cell_ids,
        z5_cell_center,
        z5_load_reference_points,
    )
    import analysis_population_bridge_suite as pb

    z5_path = ANC / "z5_minimal_anchor_profile.json"
    z5cat_path = ANC / "z5_category_generalization_matrix.json"
    if not z5_path.exists() or not z5cat_path.exists():
        raise FileNotFoundError(
            "Missing z5_minimal_anchor_profile.json or z5_category_generalization_matrix.json"
        )
    z5 = json.loads(z5_path.read_text(encoding="utf-8"))
    z5cat = json.loads(z5cat_path.read_text(encoding="utf-8"))
    refs = z5_load_reference_points()
    ov, oa = refs["oasis_centroid"]
    gv, ga = refs["gender_fp"]

    m_prof = z5["profiles"]["MtoF"]
    f_row = z5cat["directions"]["FtoM_k4"]["train_rows"]["Unrestricted"]
    # cell_ids in the JSON are sorted(occupied); image_idx is selection order.
    # Pair each image to its source-VA cell (F→M source = female ratings).
    f_imgs = [int(i) for i in f_row.get("image_idx", [])]
    oasis = pb.load_oasis_meta(pb.OASIS_SCORES_CSV)
    f_rep: list[dict] = []
    for idx in f_imgs:
        row = oasis.iloc[int(idx)]
        xy = np.asarray([[float(row[SRC_F[0]]), float(row[SRC_F[1]])]], float)
        cid = int(va_cell_ids(xy)[0])
        v, a = z5_cell_center(cid)
        f_rep.append({
            "cell_id": cid,
            "v": float(v),
            "a": float(a),
            "image_id": int(idx),
            "image_id_label": f"I{int(idx) + 1}",
        })
    f_cells = sorted({int(c["cell_id"]) for c in f_rep})
    # Prefer recorded cell set when present (same membership, may differ only in order).
    recorded = [int(c) for c in f_row.get("cell_ids", [])]
    if recorded and set(recorded) != set(f_cells):
        raise ValueError(
            f"FtoM_k4 cell mismatch: recorded={recorded} from images={f_cells}"
        )

    profiles = {
        "MtoF": {
            "direction": "MtoF",
            "greedy_primary": m_prof["greedy_primary"],
            "representative_cells": m_prof["representative_cells"],
        },
        "FtoM": {
            "direction": "FtoM",
            "greedy_primary": {
                "k": int(f_row["fixed_k"]),
                "R2_mean": float(f_row["R2_mean"]),
                "frac_of_ceiling": float(f_row["frac_of_ceiling"]),
                "cell_ids": f_cells,
            },
            "representative_cells": f_rep,
        },
    }
    m_set = set(int(c["cell_id"]) for c in m_prof["representative_cells"])
    f_set = set(f_cells)
    shared = sorted(m_set & f_set)
    union = m_set | f_set
    overlap = {
        "jaccard": float(len(shared) / len(union)) if union else float("nan"),
        "shared_cells": shared,
        "MtoF_k": int(m_prof["greedy_primary"]["k"]),
        "FtoM_k": int(f_row["fixed_k"]),
    }
    return profiles, overlap, (ov, oa), (gv, ga)


def _draw_minimal_anchor_grid(ax, ov: float, oa: float) -> None:
    edges = np.linspace(VA_LIM_FIG3[0], VA_LIM_FIG3[1], 6)
    for e in edges:
        ax.axvline(e, color="0.88", lw=0.6, zorder=0)
        ax.axhline(e, color="0.88", lw=0.6, zorder=0)
    ax.axhline(oa, color="#888", ls="--", lw=0.8, alpha=0.7, zorder=1)
    ax.axvline(ov, color="#888", ls="--", lw=0.8, alpha=0.7, zorder=1)


def _estimate_shared_affine_basins(
    density: dict,
    *,
    n_basins: int = 3,
    rel_peak: float = 0.22,
    filter_size: int = 9,
) -> list[dict]:
    """Estimate the shared 3 density modes that span an affine simplex in VA space."""
    gx = np.asarray(density["grid_x"], float)
    gy = np.asarray(density["grid_y"], float)
    maps = [
        np.asarray(density["density_m2f_k4"], float),
        np.asarray(density["density_f2m_k4"], float),
        np.asarray(density["density_f2m_k3"], float),
    ]
    z = np.mean(np.stack(maps, axis=0), axis=0)
    mx = maximum_filter(z, size=filter_size)
    peaks = (z == mx) & (z > rel_peak * float(np.nanmax(z)))
    lab, n = nd_label(peaks)
    cands: list[tuple[float, float, float]] = []
    for i in range(1, n + 1):
        iy, ix = np.argwhere(lab == i).mean(axis=0)
        iy_i, ix_i = int(round(iy)), int(round(ix))
        cands.append((float(gx[ix_i]), float(gy[iy_i]), float(z[iy_i, ix_i])))
    cands = sorted(cands, key=lambda t: -t[2])[:n_basins]
    # Stable order: (highV/lowA), (highV/mid-highA), (lowV/highA)
    cands = sorted(cands, key=lambda t: (t[1], -t[0]))
    labels = ["B1 highV/lowA", "B2 highV/midA", "B3 lowV/highA"]
    if len(cands) == 3:
        # Re-label by geometry rather than sort alone.
        by_v = sorted(cands, key=lambda t: t[0])
        low_v = by_v[0]
        high_two = sorted(by_v[1:], key=lambda t: t[1])
        ordered = [high_two[0], high_two[1], low_v]
    else:
        ordered = cands
        labels = [f"B{i + 1}" for i in range(len(ordered))]
    out = []
    for lab_i, (v, a, d) in zip(labels, ordered):
        out.append({"label": lab_i, "v": float(v), "a": float(a), "density": float(d)})
    return out


def _draw_shared_basins(
    ax,
    basins: list[dict],
    *,
    radius: float = 0.72,
    alpha: float = 0.14,
    edge: str = "#ef6c00",
    face: str = "#fff3e0",
    zorder: int = 1,
    linestyle: str = "--",
    mark_center: bool = False,
) -> None:
    """Background markers for the shared affine-simplex basins."""
    for i, b in enumerate(basins):
        circ = Circle(
            (b["v"], b["a"]), radius,
            facecolor=face, edgecolor=edge, linewidth=1.1,
            linestyle=linestyle, alpha=alpha + 0.08, zorder=zorder,
        )
        ax.add_patch(circ)
        if mark_center:
            ax.plot(
                [b["v"]], [b["a"]], marker="+", color=edge,
                ms=7, mew=1.6, zorder=zorder + 1,
            )
        short = b["label"].split()[0]
        ax.text(
            b["v"], b["a"] + radius * 0.92, short,
            ha="center", va="bottom", fontsize=6.2, color=edge,
            fontweight="bold", zorder=zorder + 1,
        )
        if i == 0:
            ax.plot(
                [], [], ls=linestyle, color=edge, lw=1.1,
                label="Shared affine basins",
            )


def _draw_minimal_anchor_direction_panel(
    ax,
    profiles: dict,
    overlap: dict,
    *,
    refs_centroid: tuple[float, float],
    refs_fp: tuple[float, float],
    title: str,
    basins: list[dict] | None = None,
    matched_k: bool = False,
) -> None:
    """Fig.3-F: true minimal coverage points on shared affine basins."""
    ov, oa = refs_centroid
    gv, ga = refs_fp
    colors = {"MtoF": "#1565c0", "FtoM": "#c62828"}
    labels = {
        "MtoF": "M→F minimal",
        "FtoM": "F→M matched k=4" if matched_k else "F→M minimal",
    }
    _draw_minimal_anchor_grid(ax, ov, oa)
    if basins:
        _draw_shared_basins(ax, basins)
    ax.scatter([gv], [ga], s=80, marker="*", c="#2e7d32", edgecolors="white", zorder=5)
    ax.scatter([ov], [oa], s=50, marker="D", c="#616161", edgecolors="white", zorder=5)
    for dshort, prof in profiles.items():
        col = colors.get(dshort, "#333")
        for i, cell in enumerate(prof["representative_cells"]):
            ax.scatter(
                [cell["v"]], [cell["a"]], s=115, c=col, alpha=0.88,
                edgecolors="white", linewidths=0.7,
                label=labels.get(dshort, dshort) if i == 0 else None, zorder=4,
            )
            ax.text(
                cell["v"], cell["a"], str(cell["cell_id"]),
                ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=6,
            )
    j = float(overlap.get("jaccard", float("nan")))
    shared = overlap.get("shared_cells", overlap.get("shared", []))
    m_gp = profiles.get("MtoF", {}).get("greedy_primary", {})
    f_gp = profiles.get("FtoM", {}).get("greedy_primary", {})
    m_k = int(m_gp.get("k", overlap.get("MtoF_k", 4)))
    f_k = int(f_gp.get("k", overlap.get("FtoM_k", 3)))
    m_frac = float(m_gp.get("frac_of_ceiling", float("nan")))
    f_frac = float(f_gp.get("frac_of_ceiling", float("nan")))
    ax.set_xlim(VA_LIM_FIG3)
    ax.set_ylim(VA_LIM_FIG3)
    ax.set_aspect("equal")
    ax.set_xlabel("Valence (source VA)")
    ax.set_ylabel("Arousal (source VA)")
    if matched_k:
        ax.set_title(
            f"{title}\n"
            f"M→F k={m_k} ({m_frac:.0%} ceiling); F→M forced k={f_k} "
            f"({f_frac:.0%} ceiling); Jaccard={j:.2f}; shared {shared}",
            fontsize=7.8,
        )
    else:
        ax.set_title(
            f"{title}\n"
            f"M→F k={m_k} ({m_frac:.0%} ceiling); F→M k={f_k} ({f_frac:.0%} ceiling)\n"
            f"basins shared; cardinality direction-dependent "
            f"(cell Jaccard={j:.2f}; shared {shared})",
            fontsize=7.6,
        )
    ax.legend(frameon=False, fontsize=6.2, loc="lower right")


def _draw_fixed_points_centroids_panel(ax, cents: dict) -> None:
    """Fig.3-D: canonical fixed points relative to empirical VA centroids."""
    fp_g = cents["fixed_points"]["gender"]
    fp_c = cents["fixed_points"]["culture"]
    cent = cents["centroids"]
    points = [
        ("OASIS overall", cent["oasis_900_overall"], "D", "#616161", 58),
        ("OASIS male", cent["oasis_900_male"], "o", "#1565c0", 42),
        ("OASIS female", cent["oasis_900_female"], "o", "#c62828", 42),
        ("Japan", cent["japan_96"], "s", "#6a1b9a", 50),
        ("Gender FP", fp_g, "*", "#2e7d32", 135),
        ("Culture FP", fp_c, "P", "#ef6c00", 90),
    ]
    for label, row, marker, color, size in points:
        ax.scatter(
            [row["valence"]], [row["arousal"]],
            s=size, marker=marker, c=color, edgecolors="white",
            linewidths=0.7, label=label, zorder=4,
        )

    oasis = cent["oasis_900_overall"]
    japan = cent["japan_96"]
    ax.plot(
        [oasis["valence"], fp_g["valence"]],
        [oasis["arousal"], fp_g["arousal"]],
        color="#2e7d32", ls="--", lw=1.0, alpha=0.8, zorder=2,
    )
    ax.plot(
        [oasis["valence"], fp_c["valence"]],
        [oasis["arousal"], fp_c["arousal"]],
        color="#ef6c00", ls="--", lw=1.0, alpha=0.8, zorder=2,
    )
    ax.plot(
        [japan["valence"], fp_c["valence"]],
        [japan["arousal"], fp_c["arousal"]],
        color="#ef6c00", ls=":", lw=1.0, alpha=0.8, zorder=2,
    )
    off = cents["offsets"]
    ax.text(
        0.02, 0.03,
        "distance to centroid\n"
        f"Gender FP→OASIS: {off['gender_fp_minus_oasis_overall']['euclidean']:.2f}\n"
        f"Culture FP→OASIS: {off['culture_fp_minus_oasis_overall']['euclidean']:.2f}\n"
        f"Culture FP→Japan: {off['culture_fp_minus_japan_96']['euclidean']:.2f}",
        transform=ax.transAxes, va="bottom", fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="0.8", alpha=0.92),
    )
    ax.set_xlim(VA_LIM_FIG3)
    ax.set_ylim(VA_LIM_FIG3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title("Fixed points relative to empirical centroids", fontsize=8.8)
    ax.legend(frameon=False, fontsize=5.8, loc="upper left", ncol=2)
    ax.grid(color="0.92", lw=0.6)


def _load_anchor_point_density_data(step_tag: str = "0p1") -> dict:
    """Load the machine-readable point-density artifact generated by the testfig analysis."""
    path = ANC / f"anchor_point_density_step{step_tag}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python3 code/plot_testfig_anchor_centroid_density.py "
            "--steps 0.1 --figures density --n-samples 80000"
        )
    with np.load(path) as z:
        return {key: z[key].copy() for key in z.files}


def _kde_on_lim(
    points: np.ndarray,
    *,
    lim: tuple[float, float],
    n_grid: int = 180,
    bw_method: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-evaluate the landmark KDE on an explicit VA window.

    The stored artifact is gridded on (1.5, 7.0); Fig.3 shows the full (1, 7) VA
    window, so the density is recomputed from the raw landmark coordinates instead
    of padding the saved grid.
    """
    from scipy.stats import gaussian_kde

    axis = np.linspace(lim[0], lim[1], n_grid)
    gx, gy = np.meshgrid(axis, axis)
    pts = np.asarray(points, float)
    if len(pts) < 2:
        return axis, axis, np.zeros_like(gx)
    kde = gaussian_kde(pts.T, bw_method=bw_method)
    zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    return axis, axis, zz


def _positive_quantile_levels(z: np.ndarray, qs=(0.70, 0.85, 0.95)) -> np.ndarray:
    vals = np.asarray(z, float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if len(vals) == 0:
        return np.asarray([], float)
    levels = np.unique(np.quantile(vals, qs))
    return levels[levels > 0]


def _draw_anchor_point_density_panel(
    ax, density: dict, *, fig: plt.Figure, basins: list[dict] | None = None,
) -> None:
    """Fig.3-G: shareable-landmark density (elementwise min over directions) + basins."""
    gx, gy, z_m = _kde_on_lim(density["points_m2f_k4"], lim=VA_LIM_FIG3)
    _, _, z_f = _kde_on_lim(density["points_f2m_k4"], lim=VA_LIM_FIG3)
    # Elementwise min = density attainable in BOTH directions, i.e. shareable landmarks.
    shared = np.minimum(z_m, z_f)
    vmax = max(float(np.nanmax(shared)), 1e-12)
    im = ax.imshow(
        shared,
        extent=[VA_LIM_FIG3[0], VA_LIM_FIG3[1], VA_LIM_FIG3[0], VA_LIM_FIG3[1]],
        origin="lower", cmap="inferno", vmin=0.0, vmax=vmax,
        aspect="equal", zorder=1,
    )
    if basins:
        _draw_shared_basins(
            ax, basins, radius=0.78, alpha=0.0,
            edge="white", face="none", zorder=4,
            linestyle="-", mark_center=True,
        )
    lev_m = _positive_quantile_levels(z_m)
    lev_f = _positive_quantile_levels(z_f)
    if len(lev_m):
        ax.contour(gx, gy, z_m, levels=lev_m, colors="#4fc3f7", linewidths=0.9, zorder=3)
    if len(lev_f):
        ax.contour(
            gx, gy, z_f, levels=lev_f, colors="#a5d6a7",
            linewidths=0.9, linestyles="--", zorder=3,
        )

    ax.plot([], [], color="#4fc3f7", lw=1.2, label="M→F k=4 density")
    ax.plot([], [], color="#a5d6a7", lw=1.2, ls="--", label="F→M k=4 density")
    ax.set_xlim(VA_LIM_FIG3)
    ax.set_ylim(VA_LIM_FIG3)
    ax.set_xticks(np.arange(VA_LIM_FIG3[0], VA_LIM_FIG3[1] + 0.01, 1.0))
    ax.set_yticks(np.arange(VA_LIM_FIG3[0], VA_LIM_FIG3[1] + 0.01, 1.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel("Valence (source VA)")
    ax.set_ylabel("Arousal (source VA)")
    ax.set_title(
        "Shareable landmarks (matched k=4)\n"
        "3 basins usable in both directions; 4th point reinforces an existing basin",
        fontsize=8.0,
    )
    ax.legend(frameon=False, fontsize=5.8, loc="upper right", labelcolor="white")
    ax.text(
        0.02, 0.03,
        f"step={float(density['grid_step']):.1f}; 80k draws/direction\n"
        f"hits M→F={int(density['n_hits_m2f_k4'])}; "
        f"F→M={int(density['n_hits_f2m_k4'])}",
        transform=ax.transAxes, va="bottom", fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="0.8", alpha=0.9),
    )
    fig.colorbar(
        im, ax=ax, fraction=0.046, pad=0.03,
        label="shareable landmark density  min(M→F, F→M)",
    )
    ax.set_box_aspect(1)


def _draw_composition_consistency_panel(ax, comp: dict) -> None:
    """Fig.3-H: cross-boundary composition consistency (held-out; V/A split)."""
    ht = comp["held_out_test"]
    # Prefer held-out points so Arousal composition failure is visible.
    pred_pack = comp.get("pred_held_out") or comp["pred_full_sample"]
    y = np.asarray(pred_pack["y_japan"], float)
    pred = np.asarray(pred_pack["y_composed"], float)
    use_held = "pred_held_out" in comp
    ax.scatter(
        y[:, 0], pred[:, 0], s=28 if use_held else 15, alpha=0.75,
        marker="o", label="Valence", c="#1565c0", edgecolors="white", linewidths=0.3,
        zorder=3,
    )
    ax.scatter(
        y[:, 1], pred[:, 1], s=32 if use_held else 15, alpha=0.75,
        marker="^", label="Arousal", c="#8e24aa", edgecolors="white", linewidths=0.3,
        zorder=3,
    )
    lims = list(VA_LIM_FIG3)
    ax.plot(lims, lims, "k--", lw=0.9, alpha=0.55, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xticks(np.arange(lims[0], lims[1] + 0.01, 1.0))
    ax.set_yticks(np.arange(lims[0], lims[1] + 0.01, 1.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel("Observed Japan")
    ax.set_ylabel("Composed Φ_c∘Φ_g prediction")
    rv = ht["r2_composed_per_dim"]["valence"] / ht["r2_direct_per_dim"]["valence"]
    ra = ht["r2_composed_per_dim"]["arousal"] / ht["r2_direct_per_dim"]["arousal"]
    scope = "held-out themes" if use_held else "full sample (fallback)"
    ax.set_title(
        "Cross-boundary composition\n"
        f"VW ratio={ht['ratio_composed_over_direct']:.3f} (held-out); "
        f"V={rv:.3f} / A={ra:.3f}\n"
        f"full VW={comp['full_sample_96']['ratio']:.3f} · scatter={scope}",
        fontsize=7.6,
    )
    ax.legend(frameon=False, fontsize=6.5, loc="lower right")
    ax.grid(color="0.92", lw=0.6)


def _load_residual_topology_packs() -> tuple[np.ndarray, dict]:
    from plot_residual_field_topology import (
        analyze_direction,
        build_frame,
        make_grid,
    )

    df, _meta = build_frame()
    g = make_grid(0.1)
    packs = {
        d: analyze_direction(
            df, g, d,
            grid_step=0.1, radius=0.5, min_n=6,
            top_peaks=5, per_peak=3, curl_q=0.85,
        )
        for d in ("MtoF", "FtoM")
    }
    return g, packs


def _load_cluster_module():
    """Sign-flip cluster-permutation helpers shared with the twist analysis."""
    import importlib.util

    path = ROOT / "code" / "paper2_viz" / "figures" / "analyze_cross_within_bias_clusters.py"
    spec = importlib.util.spec_from_file_location("_paperfig_cluster", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_paperfig_cluster"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_excess_twist_maps(
    *,
    radius: float = 0.5,
    min_n: int = 5,
    n_perm: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, float, dict]:
    """Excess twist = aligned-UMAP score displacement − model twist ω, per target VA frame.

    M→F is anchored on the female score VA (the frame being predicted), F→M on the male one,
    matching the reference frames used by analyze_relational_cross_within_twist.
    """
    csv = TWIST / "relational_twist_per_image.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"Missing {csv.name}. Run: "
            "python3 code/paper2_viz/figures/analyze_relational_cross_within_twist.py"
        )
    df = pd.read_csv(csv)
    excess = (df["geo_umap_disp"] - df["model_twist_pm_pf"]).to_numpy(float)
    mod = _load_cluster_module()
    step = 0.1
    g = mod.make_grid(step)
    n_g = len(g)
    rng = np.random.default_rng(seed)

    packs: dict[str, dict] = {}
    for key, ref in (("MtoF", "female"), ("FtoM", "male")):
        v_ref = df[f"valence_{ref}"].to_numpy(float)
        a_ref = df[f"arousal_{ref}"].to_numpy(float)
        masks, n_map = mod.build_cell_masks(v_ref, a_ref, g, radius, min_n)
        grid = np.full((n_g, n_g), np.nan)
        for k, idx in enumerate(masks):
            if len(idx) >= min_n:
                grid[k // n_g, k % n_g] = float(np.mean(excess[idx]))
        cl = mod.permutation_cluster_correction(
            excess - float(np.mean(excess)), masks, n_map, g, min_n, rng,
            n_perm=n_perm, t_threshold=2.0, alpha=0.05,
        )
        packs[key] = {"ref": ref, "grid": grid, "cluster": cl}
    packs["_overlay"] = mod.overlay_cluster_regions
    packs["_params"] = {"radius": radius, "min_n": min_n, "n_perm": n_perm}
    return g, step, packs


def _draw_excess_twist_panel(
    ax,
    g: np.ndarray,
    step: float,
    packs: dict,
    direction: str,
    *,
    fig: plt.Figure,
    vmax: float | None = None,
) -> None:
    pack = packs[direction]
    grid = pack["grid"]
    ref = pack["ref"]
    ext = [g[0] - step / 2, g[-1] + step / 2, g[0] - step / 2, g[-1] + step / 2]
    finite = grid[np.isfinite(grid)]
    if vmax is None:
        vmax = max(float(np.nanpercentile(finite, 95)) if finite.size else 1.0, 1e-6)
    im = ax.imshow(
        grid, origin="lower", extent=ext, aspect="equal",
        cmap="magma", vmin=0.0, vmax=vmax,
    )
    cl = pack["cluster"]
    subtitle = ""
    if cl.get("ok"):
        packs["_overlay"](ax, cl["labels"], cl["labels_sig"], ext)
        p_c = float(cl.get("p_cluster", np.nan))
        n_sig = int(cl.get("n_sig_clusters", 0))
        p_txt = "p<0.001" if p_c < 1e-3 else f"p={p_c:.3f}"
        subtitle = f"\ncluster {p_txt}, {n_sig} significant regions"
    arrow = "M→F" if direction == "MtoF" else "F→M"
    ax.set_xlim(VA_LIM_FIG3)
    ax.set_ylim(VA_LIM_FIG3)
    ax.set_xticks(np.arange(VA_LIM_FIG3[0], VA_LIM_FIG3[1] + 0.01, 1.0))
    ax.set_yticks(np.arange(VA_LIM_FIG3[0], VA_LIM_FIG3[1] + 0.01, 1.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Valence ({ref} score)")
    ax.set_ylabel(f"Arousal ({ref} score)")
    ax.set_title(
        f"Excess twist {arrow} on {ref} VA (prediction target){subtitle}",
        fontsize=8.6,
    )
    ax.grid(True, alpha=0.18, color="white", lw=0.4)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label(
        "‖ΔUMAP‖ − ω", fontsize=7.5,
    )


def _residual_mag(pack: dict) -> np.ndarray:
    """Local mean post-Φ residual magnitude ‖Δ‖ = √(dx²+dy²)."""
    return np.hypot(np.asarray(pack["dx"], float), np.asarray(pack["dy"], float))


def _draw_topology_single(
    ax,
    g: np.ndarray,
    packs: dict,
    direction: str,
    *,
    fig: plt.Figure,
    vmax: float | None = None,
) -> None:
    """Φ-failure map: colour = ‖Δ‖, arrows/streams = residual direction. No curl peaks."""
    from plot_residual_field_topology import prepare_stream

    pack = packs[direction]
    ref = "male" if direction == "MtoF" else "female"
    arrow = "M→F" if direction == "MtoF" else "F→M"
    dx = np.asarray(pack["dx"], float)
    dy = np.asarray(pack["dy"], float)
    mag = _residual_mag(pack)
    if vmax is None:
        vmax = max(float(np.nanquantile(mag, 0.98)), 1e-4)

    xg, yg = np.meshgrid(g, g, indexing="xy")
    dx_s, dy_s, valid = prepare_stream(dx, dy)
    im = ax.pcolormesh(
        xg, yg, mag, shading="auto", cmap="magma", vmin=0.0, vmax=vmax, zorder=1,
    )
    step_q = max(1, len(g) // 12)
    sl = (slice(None, None, step_q), slice(None, None, step_q))
    m = valid[sl]
    ax.quiver(
        xg[sl][m], yg[sl][m], dx[sl][m], dy[sl][m],
        color="white", alpha=0.85, angles="xy", scale_units="xy", scale=2.2,
        width=0.0035, zorder=3,
    )
    try:
        ax.streamplot(
            g, g, dx_s, dy_s,
            color="#80cbc4", density=1.15, linewidth=0.9,
            arrowsize=0.7, zorder=2, broken_streamlines=True,
        )
    except Exception as exc:  # noqa: BLE001
        ax.text(
            0.02, 0.98, f"streamplot failed: {exc}",
            transform=ax.transAxes, va="top", fontsize=7, color="w",
        )

    q90 = float(np.nanquantile(mag, 0.9))
    ax.set_xlim(VA_LIM_FIG3)
    ax.set_ylim(VA_LIM_FIG3)
    ax.set_xticks(np.arange(VA_LIM_FIG3[0], VA_LIM_FIG3[1] + 0.01, 1.0))
    ax.set_yticks(np.arange(VA_LIM_FIG3[0], VA_LIM_FIG3[1] + 0.01, 1.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Valence ({ref})")
    ax.set_ylabel(f"Arousal ({ref})")
    ax.set_title(
        f"Post-Φ residual {arrow} on {ref} VA (source)\n"
        f"colour = ‖Δ‖; arrows = failure direction; q90={q90:.2f}",
        fontsize=8.6,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("‖Δ‖", fontsize=7.5)


def paper_fig3() -> tuple[Path, Path]:
    """NatComm Fig.3 A–H — linear bridges, quantity, quality, direction, composition."""
    ot = json.loads((CVAE / "paper2_ot_five_point_fixedsplit.json").read_text())
    comp = json.loads((PBA / "composition_consistency.json").read_text())
    cents = json.loads((PBA / "va_centroids_vs_fixed_points.json").read_text())
    m2f = pd.read_csv(ANC / "z3_generalization_curve_MtoF.csv")
    f2m = pd.read_csv(ANC / "z3_generalization_curve_FtoM.csv")
    z3 = json.loads((ANC / "z3_generalization.json").read_text())
    scores_f = ot["scores_female_target"]
    scores_m = ot.get("scores_male_target")

    methods = ["no_transport", "global_shift", "linear_shift", "ot", "oracle"]
    labels = ["No", "Global", "Linear", "OT", "Oracle"]
    mcolors = ["#bdbdbd", "#a5d6a7", "#42a5f5", "#ef9a9a", "#8d6e63"]
    br = {
        "gender": {"A": comp["full_sample_96"]["phi_gender"]["A"], "b": comp["full_sample_96"]["phi_gender"]["b"],
                   "fp": comp["full_sample_96"]["phi_gender"]["fixed_point"],
                   "sv": comp["full_sample_96"]["phi_gender"]["singular_values"]},
        "culture": {"A": comp["full_sample_96"]["phi_culture"]["A"], "b": comp["full_sample_96"]["phi_culture"]["b"],
                    "fp": comp["full_sample_96"]["phi_culture"]["fixed_point"],
                    "sv": comp["full_sample_96"]["phi_culture"]["singular_values"]},
    }
    gdens, cdens = _oasis_density()
    profiles, overlap, cen, fp = _load_z5_minimal_anchor_data()
    point_density = _load_anchor_point_density_data("0p1")
    basins = _estimate_shared_affine_basins(point_density)

    fig = plt.figure(figsize=(12.2, 18.0), constrained_layout=True)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 1.0, 0.95, 1.0], hspace=0.28, wspace=0.24)

    # A: five-point mapping — method comparison, both target directions
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")
    if scores_m is None:
        raise ValueError("scores_male_target missing in paper2_ot_five_point_fixedsplit.json")
    _draw_fig3_combined_panel_a(ax_a, scores_f, scores_m, methods, labels, mcolors)

    # B: canonical M→F gender bridge
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")
    cf = bridge_terrain(
        ax_b, br["gender"]["A"], br["gender"]["b"], br["gender"]["fp"], br["culture"]["fp"],
        gdens, f"Gender bridge Φ_g  σ={br['gender']['sv'][0]:.2f}/{br['gender']['sv'][1]:.2f}",
        "Gender", "Culture", "#1976d2", "#e53935", lim=VA_LIM_FIG3,
    )
    ax_b.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.colorbar(cf, ax=ax_b, fraction=0.046, pad=0.03, label="‖Φ(y)−y‖")

    # C: culture bridge displacement terrain
    ax_c = fig.add_subplot(gs[1, 0])
    panel_label(ax_c, "C")
    cf = bridge_terrain(
        ax_c, br["culture"]["A"], br["culture"]["b"], br["culture"]["fp"], br["gender"]["fp"],
        cdens, f"Culture bridge Φ_c  σ={br['culture']['sv'][0]:.2f}/{br['culture']['sv'][1]:.2f}",
        "Culture", "Gender", "#e53935", "#1976d2", cmap="GnBu", lim=VA_LIM_FIG3,
        contour_levels=CONTOUR_LEVELS_CULTURE,
    )
    ax_c.legend(frameon=False, fontsize=6.5, loc="lower right")
    fig.colorbar(cf, ax=ax_c, fraction=0.046, pad=0.03, label="‖Φ(y)−y‖")
    ax_c.text(
        0.02, 0.97,
        f"FP distance (gender↔culture) = {comp['fixed_points']['euclidean_distance']:.2f}",
        transform=ax_c.transAxes, va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.9),
    )

    # D: fixed points vs empirical centroids
    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "D")
    _draw_fixed_points_centroids_panel(ax_d, cents)

    # E: quantity — saturation with number of shared anchors
    ax_e = fig.add_subplot(gs[2, 0])
    panel_label(ax_e, "E")
    _draw_z3_curve_panel_compact(ax_e, m2f, f2m, z3)

    # F: true minimal coverage (M→F k=4 / F→M k=3) on shared 3 basins
    ax_f = fig.add_subplot(gs[2, 1])
    panel_label(ax_f, "F")
    _draw_minimal_anchor_direction_panel(
        ax_f, profiles, overlap,
        refs_centroid=cen, refs_fp=fp,
        title="Minimal coverage on shared affine basins",
        basins=basins,
        matched_k=False,
    )

    # G: cloud of all sampled effective matched-k solutions + shared basins
    ax_g = fig.add_subplot(gs[3, 0])
    panel_label(ax_g, "G")
    _draw_anchor_point_density_panel(ax_g, point_density, fig=fig, basins=basins)

    # H: low-dimensional bridge composition across population boundaries
    ax_h = fig.add_subplot(gs[3, 1])
    panel_label(ax_h, "H")
    _draw_composition_consistency_panel(ax_h, comp)

    fig.suptitle(
        "Fig. 3 | Low-dimensional bridges generalize from shared affine basins "
        "with direction-dependent cardinality",
        fontsize=10.5,
    )
    png, svg = save_dual(fig, "Paper_Fig3_linear_bridge_composition")

    # Individual panel exports A–H.
    _export_fig3_panel_svgs(
        scores_f, scores_m, methods, labels, mcolors, br, gdens, cdens, comp, cents, cdens,
        m2f, f2m, z3, profiles, overlap, cen, fp, point_density, basins,
    )
    return png, svg


def _export_fig3_panel_svgs(
    scores_f, scores_m, methods, labels, mcolors, br, gdens, cdens, comp, cents, dens_overall,
    m2f, f2m, z3, profiles, overlap, cen, fp, point_density, basins,
) -> None:
    """Write standalone Fig.3 A–H panels after the 8-panel reorganization."""
    OUT.mkdir(parents=True, exist_ok=True)
    contour_levels = (0.2, 0.4, 0.6, 0.8)

    # A
    fig_a, ax = plt.subplots(figsize=(6.2, 4.6))
    _draw_fig3_combined_panel_a(ax, scores_f, scores_m, methods, labels, mcolors)
    fig_a.tight_layout()
    fig_a.savefig(OUT / "Paper_Fig3_panelA_five_point_mapping.svg", format="svg", bbox_inches="tight", facecolor="white")
    fig_a.savefig(OUT / "Paper_Fig3_panelA_five_point_mapping.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_a)

    # B
    fig_b, ax = plt.subplots(figsize=(5.4, 5.0))
    cf = bridge_terrain(
        ax, br["gender"]["A"], br["gender"]["b"], br["gender"]["fp"], br["culture"]["fp"],
        gdens, f"Gender bridge Φ_g  σ={br['gender']['sv'][0]:.2f}/{br['gender']['sv'][1]:.2f}",
        "Gender", "Culture", "#1976d2", "#e53935", lim=VA_LIM_FIG3,
        contour_levels=contour_levels,
    )
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig_b.colorbar(cf, ax=ax, fraction=0.046, pad=0.03, label="‖Φ(y)−y‖")
    fig_b.tight_layout()
    fig_b.savefig(OUT / "Paper_Fig3_panelB_gender_bridge.svg", format="svg", bbox_inches="tight", facecolor="white")
    fig_b.savefig(OUT / "Paper_Fig3_panelB_gender_bridge.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_b)

    # C
    fig_c, ax = plt.subplots(figsize=(5.4, 5.0))
    cf = bridge_terrain(
        ax, br["culture"]["A"], br["culture"]["b"], br["culture"]["fp"], br["gender"]["fp"],
        dens_overall, f"Culture bridge Φ_c  σ={br['culture']['sv'][0]:.2f}/{br['culture']['sv'][1]:.2f}",
        "Culture", "Gender", "#e53935", "#1976d2", cmap="GnBu", lim=VA_LIM_FIG3,
        contour_levels=CONTOUR_LEVELS_CULTURE,
    )
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig_c.colorbar(cf, ax=ax, fraction=0.046, pad=0.03, label="‖Φ(y)−y‖")
    ax.text(
        0.02, 0.97,
        f"FP distance (gender↔culture) = {comp['fixed_points']['euclidean_distance']:.2f}",
        transform=ax.transAxes, va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.9),
    )
    fig_c.tight_layout()
    fig_c.savefig(OUT / "Paper_Fig3_panelC_culture_bridge.svg", format="svg", bbox_inches="tight", facecolor="white")
    fig_c.savefig(OUT / "Paper_Fig3_panelC_culture_bridge.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_c)

    # D: fixed points vs centroids
    fig_d, ax = plt.subplots(figsize=(5.4, 5.0))
    _draw_fixed_points_centroids_panel(ax, cents)
    fig_d.tight_layout()
    fig_d.savefig(OUT / "Paper_Fig3_panelD_fixed_points_centroids.svg", format="svg", bbox_inches="tight", facecolor="white")
    fig_d.savefig(OUT / "Paper_Fig3_panelD_fixed_points_centroids.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_d)

    # E: quantity
    fig_e, ax = plt.subplots(figsize=(5.6, 4.6))
    _draw_z3_curve_panel_compact(ax, m2f, f2m, z3)
    fig_e.tight_layout()
    fig_e.savefig(OUT / "Paper_Fig3_panelE_quantity_generalization.svg", format="svg", bbox_inches="tight", facecolor="white")
    fig_e.savefig(OUT / "Paper_Fig3_panelE_quantity_generalization.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_e)

    # F: true minimal coverage + shared basins
    fig_f, ax = plt.subplots(figsize=(5.6, 5.2))
    _draw_minimal_anchor_direction_panel(
        ax, profiles, overlap,
        refs_centroid=cen, refs_fp=fp,
        title="Minimal coverage on shared affine basins",
        basins=basins,
        matched_k=False,
    )
    fig_f.tight_layout()
    fig_f.savefig(OUT / "Paper_Fig3_panelF_minimal_shared_states.svg", format="svg", bbox_inches="tight", facecolor="white")
    fig_f.savefig(OUT / "Paper_Fig3_panelF_minimal_shared_states.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_f)

    # G: effective-solution cloud
    fig_g, ax = plt.subplots(figsize=(5.6, 5.0))
    _draw_anchor_point_density_panel(ax, point_density, fig=fig_g, basins=basins)
    fig_g.tight_layout()
    fig_g.savefig(OUT / "Paper_Fig3_panelG_effective_point_density.svg", format="svg", bbox_inches="tight", facecolor="white")
    fig_g.savefig(OUT / "Paper_Fig3_panelG_effective_point_density.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_g)

    # H: composition consistency
    fig_h, ax = plt.subplots(figsize=(5.5, 5.0))
    _draw_composition_consistency_panel(ax, comp)
    fig_h.tight_layout()
    fig_h.savefig(OUT / "Paper_Fig3_panelH_composition_consistency.svg", format="svg", bbox_inches="tight", facecolor="white")
    fig_h.savefig(OUT / "Paper_Fig3_panelH_composition_consistency.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig_h)

    print(
        "[ok] individual Fig.3 panels A–H: "
        "five-point, gender bridge, culture bridge, fixed points, quantity, "
        "minimal+basins, effective density, composition"
    )


def _load_oasis_meta_cached():
    import analysis_population_bridge_suite as pb

    if not hasattr(_load_oasis_meta_cached, "_df"):
        _load_oasis_meta_cached._df = pb.load_oasis_meta(pb.OASIS_SCORES_CSV)
    return _load_oasis_meta_cached._df


def _load_oasis_thumb(image_idx: int, *, size: int = 52) -> np.ndarray | None:
    """Square crop thumbnail from OASIS row index (0-based image_idx)."""
    meta = _load_oasis_meta_cached()
    if image_idx < 0 or image_idx >= len(meta):
        return None
    path = Path(str(meta.iloc[int(image_idx)]["image_path"]))
    if not path.exists():
        return None
    img = mpimg.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    h, w = img.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = img[y0:y0 + side, x0:x0 + side]
    yy = (np.linspace(0, side - 1, size)).astype(int)
    xx = (np.linspace(0, side - 1, size)).astype(int)
    return crop[yy][:, xx]


def _draw_matched_k_anchor_image_panel(
    ax_map,
    ax_th,
    profiles: dict,
    overlap: dict,
    *,
    refs_centroid: tuple[float, float],
    refs_fp: tuple[float, float],
    title: str,
    basins: list[dict] | None = None,
) -> None:
    """Matched k=4: VA map with cell ids + numbered OASIS thumbnail strip."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    ov, oa = refs_centroid
    gv, ga = refs_fp
    colors = {"MtoF": "#1565c0", "FtoM": "#c62828"}
    labels = {"MtoF": "M→F k=4", "FtoM": "F→M forced k=4"}
    _draw_minimal_anchor_grid(ax_map, ov, oa)
    if basins:
        _draw_shared_basins(ax_map, basins)
    ax_map.scatter([gv], [ga], s=70, marker="*", c="#2e7d32", edgecolors="white", zorder=5)
    ax_map.scatter([ov], [oa], s=45, marker="D", c="#616161", edgecolors="white", zorder=5)

    entries: list[tuple[str, dict]] = []
    for dshort, prof in profiles.items():
        col = colors.get(dshort, "#333")
        for i, cell in enumerate(prof["representative_cells"]):
            v, a = float(cell["v"]), float(cell["a"])
            ax_map.scatter(
                [v], [a], s=115, c=col, alpha=0.92,
                edgecolors="white", linewidths=0.7,
                label=labels.get(dshort, dshort) if i == 0 else None, zorder=4,
            )
            ax_map.text(
                v, a, str(int(cell["cell_id"])),
                ha="center", va="center", fontsize=7, color="white",
                fontweight="bold", zorder=6,
            )
            entries.append((dshort, cell))

    j = float(overlap.get("jaccard", float("nan")))
    shared = overlap.get("shared_cells", overlap.get("shared", []))
    m_gp = profiles.get("MtoF", {}).get("greedy_primary", {})
    f_gp = profiles.get("FtoM", {}).get("greedy_primary", {})
    m_k = int(m_gp.get("k", overlap.get("MtoF_k", 4)))
    f_k = int(f_gp.get("k", overlap.get("FtoM_k", 4)))
    m_frac = float(m_gp.get("frac_of_ceiling", float("nan")))
    f_frac = float(f_gp.get("frac_of_ceiling", float("nan")))
    ax_map.set_xlim(VA_LIM_FIG3)
    ax_map.set_ylim(VA_LIM_FIG3)
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("Valence (source VA)")
    ax_map.set_ylabel("Arousal (source VA)")
    ax_map.set_title(
        f"{title}\n"
        f"M→F k={m_k} ({m_frac:.0%} ceiling); F→M forced k={f_k} "
        f"({f_frac:.0%} ceiling); Jaccard={j:.2f}; shared {shared}",
        fontsize=7.6,
    )
    ax_map.legend(frameon=False, fontsize=6.0, loc="lower right")

    ax_th.set_xlim(0, max(len(entries), 1))
    ax_th.set_ylim(0, 1)
    ax_th.set_xticks([])
    ax_th.set_yticks([])
    for spine in ax_th.spines.values():
        spine.set_visible(False)
    ax_th.set_title("Cell id → image (blue = M→F, red = F→M)", fontsize=7.2, pad=2)
    for i, (dshort, cell) in enumerate(entries):
        col = colors.get(dshort, "#333")
        x = i + 0.5
        idx = cell.get("image_id", cell.get("image_idx"))
        rgb = _load_oasis_thumb(int(idx), size=72) if idx is not None else None
        if rgb is not None:
            ab = AnnotationBbox(
                OffsetImage(rgb, zoom=0.70),
                (x, 0.58),
                frameon=True,
                bboxprops=dict(boxstyle="round,pad=0.12", fc="white", ec=col, lw=1.5),
                pad=0.02,
                zorder=3,
            )
            ax_th.add_artist(ab)
        ax_th.text(
            x, 0.06, str(int(cell["cell_id"])),
            ha="center", va="bottom", fontsize=8.5, color=col, fontweight="bold",
        )


def paper_supp_matched_k_minimal_comparison() -> tuple[Path, Path]:
    """
    Matched-k=4 position comparison demoted from main-text Fig.3-F.

    Main-text F now shows true minimal coverage (M→F k=4 / F→M k=3) on shared
    basins; this Supplement keeps the forced-k=4 Jaccard overlay for readers who
    want a cardinality-matched cell-set comparison. Panel B shows the matched-k
    anchors as cell-numbered OASIS thumbnails.
    """
    profiles_k4, overlap_k4, cen, fp = _load_z5_matched_k4_anchor_data()
    profiles_min, overlap_min, _, _ = _load_z5_minimal_anchor_data()
    density = _load_anchor_point_density_data("0p1")
    basins = _estimate_shared_affine_basins(density)

    fig = plt.figure(figsize=(11.6, 6.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.4, 1.2], hspace=0.28, wspace=0.28)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b_map = fig.add_subplot(gs[0, 1])
    ax_b_th = fig.add_subplot(gs[1, 1])

    panel_label(ax_a, "A")
    _draw_minimal_anchor_direction_panel(
        ax_a, profiles_min, overlap_min,
        refs_centroid=cen, refs_fp=fp,
        title="True minimal coverage (main-text Fig.3-F)",
        basins=basins,
        matched_k=False,
    )

    panel_label(ax_b_map, "B")
    _draw_matched_k_anchor_image_panel(
        ax_b_map, ax_b_th, profiles_k4, overlap_k4,
        refs_centroid=cen, refs_fp=fp,
        title="Matched k=4 anchors (cell id → image)",
        basins=basins,
    )

    fig.suptitle(
        "Supp. Fig. | Minimal coverage vs matched-k=4 cell-set comparison\n"
        "Matched k removes cardinality confounding when reporting cell Jaccard; "
        "main-text Fig.3-F uses true minimal k",
        fontsize=10.0,
    )
    return save_dual(fig, "Paper_SuppFig_matched_k_minimal_comparison")


def paper_supp_anchor_point_density_step0p1() -> tuple[Path, Path]:
    """Four-panel landmark point-density Supplement (shared A–C color scale; VA 1–7)."""
    density = _load_anchor_point_density_data("0p1")
    lim = VA_LIM_FIG3  # full OASIS VA window (1–7), matching Fig.3
    edges = np.arange(lim[0], lim[1] + 0.01, 1.0)
    step = float(density["grid_step"])
    point_packs = [
        ("A", density["points_m2f_k4"], f"M→F point density (k=4; step={step:.2f})"),
        ("B", density["points_f2m_k3"], f"F→M point density (k=3; step={step:.2f})"),
        ("C", density["points_f2m_k4"], f"F→M point density (k=4; step={step:.2f})"),
    ]
    # Re-evaluate KDE on (1, 7); stored artifact grids are on (1.5, 7.0).
    dens_maps: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]] = []
    for lab, pts, title in point_packs:
        gx, gy, zz = _kde_on_lim(np.asarray(pts, float), lim=lim)
        dens_maps.append((lab, gx, gy, zz, np.asarray(pts, float), title))
    vmax = max(float(np.nanmax(zz)) for _, _, _, zz, _, _ in dens_maps)
    vmax = max(vmax, 1e-12)

    fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.4), constrained_layout=True)
    im0 = None
    for ax, (lab, gx, gy, zz, pts, title) in zip(axes[:3], dens_maps):
        panel_label(ax, lab)
        im = ax.imshow(
            zz,
            extent=[lim[0], lim[1], lim[0], lim[1]],
            origin="lower", cmap="viridis", vmin=0.0, vmax=vmax,
            aspect="equal", zorder=1, alpha=0.9,
        )
        if im0 is None:
            im0 = im
        for e in edges:
            ax.axvline(e, color="0.88", lw=0.6, zorder=0)
            ax.axhline(e, color="0.88", lw=0.6, zorder=0)
        vals = zz[np.isfinite(zz) & (zz > 0)]
        if len(vals):
            levels = np.unique(np.quantile(vals, [0.65, 0.80, 0.92]))
            ax.contour(
                gx, gy, zz, levels=levels,
                colors="white", linewidths=0.8, alpha=0.8, zorder=2,
            )
        if len(pts):
            rng = np.random.default_rng(0)
            n = min(len(pts), 4000)
            idx = rng.choice(len(pts), size=n, replace=False)
            ax.scatter(
                pts[idx, 0], pts[idx, 1], s=6, c="white", alpha=0.12, zorder=3,
            )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xticks(edges)
        ax.set_yticks(edges)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_title(title, fontsize=9)

    fig.colorbar(
        im0, ax=list(axes[:3]), fraction=0.025, pad=0.02, label="point density",
    )

    ax = axes[3]
    panel_label(ax, "D")
    zz_m = dens_maps[0][3]  # M→F k=4
    zz_f = dens_maps[2][3]  # F→M k=4
    diff = zz_m - zz_f
    dmax = max(float(np.nanmax(np.abs(diff))), 1e-12)
    imd = ax.imshow(
        diff,
        extent=[lim[0], lim[1], lim[0], lim[1]],
        origin="lower", cmap="coolwarm", vmin=-dmax, vmax=dmax,
        aspect="equal",
    )
    for e in edges:
        ax.axvline(e, color="0.88", lw=0.6, zorder=0)
        ax.axhline(e, color="0.88", lw=0.6, zorder=0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks(edges)
    ax.set_yticks(edges)
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(
        f"Point-density difference (step={step:.2f})\nM→F k=4 minus F→M k=4",
        fontsize=9,
    )
    fig.colorbar(imd, ax=ax, fraction=0.046, pad=0.03, label="density difference")

    fig.suptitle(
        f"Supp. Fig. | Density of effective landmark points (grid step={step:.2f})\n"
        "A–C share one color scale; D is a signed difference map",
        y=1.03, fontsize=10.5,
    )
    return save_dual(fig, "Paper_SuppFig_anchor_point_density_step0p1")


def _load_person_bottleneck_payload() -> tuple[dict, list[str], list[tuple]]:
    json_path = ANC / "z5_category_generalization_matrix.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing {json_path.name}. Run: "
            "python3 code/analysis_anchor_structure.py --mode z5cat"
        )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    cats = list(payload["category_order"])
    specs = [
        ("MtoF", "M→F minimal (k=4)", "#1565c0", "-", "o"),
        ("FtoM", "F→M minimal (k=3)", "#c62828", "-", "o"),
        ("FtoM_k4", "F→M matched (k=4)", "#c62828", "--", "s"),
    ]
    specs = [s for s in specs if s[0] in payload["directions"]]
    return payload, cats, specs


def _draw_person_bottleneck_curve(ax, payload: dict, cats: list[str], specs: list[tuple]) -> None:
    """Held-out R² by test category at each direction's true minimal k."""
    x = np.arange(len(cats), dtype=float)
    person_i = cats.index("Person") if "Person" in cats else None
    if person_i is not None:
        ax.axvspan(person_i - 0.4, person_i + 0.4, color="#ef5350", alpha=0.10, zorder=0)
    for key, label, color, ls, marker in specs:
        row = payload["directions"][key]["train_rows"]["Unrestricted"]
        ys = [float(row["test_by_category"][c]["R2_mean"]) for c in cats]
        ax.plot(
            x, ys, marker=marker, ls=ls, color=color, lw=1.7, ms=6.5,
            markeredgecolor="white", markeredgewidth=0.6, label=label, zorder=3,
        )
    ax.set_xticks(x, cats)
    ax.set_xlabel("Held-out test category")
    ax.set_ylabel("Held-out R² (Unrestricted anchors)")
    ax.set_title(
        "Person is hardest at each direction's true minimal k\n"
        "anchors chosen without category constraint (all Scene)",
        fontsize=8.8,
    )
    ax.legend(frameon=False, fontsize=7.0, loc="lower right")
    ax.grid(color="0.92", lw=0.6)


def paper_supp_person_bottleneck() -> tuple[Path, Path]:
    """
    Person is the hardest held-out test category at each direction's true minimal k.

    Demoted from a main-text Fig.3 candidate: Fig.3-H keeps composition consistency.
    Primary rows use the true minimal k (M→F k=4, F→M k=3); matched k=4 is shown as a
    cardinality-controlled check so the Person gap is not read as a k artifact.
    """
    payload, cats, specs = _load_person_bottleneck_payload()

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6), constrained_layout=True)

    ax = axes[0]
    panel_label(ax, "A")
    _draw_person_bottleneck_curve(ax, payload, cats, specs)

    ax = axes[1]
    panel_label(ax, "B")
    width = 0.8 / max(len(specs), 1)
    for i, (key, label, color, ls, _marker) in enumerate(specs):
        row = payload["directions"][key]["train_rows"]["Unrestricted"]
        others = [
            float(row["test_by_category"][c]["R2_mean"]) for c in cats if c != "Person"
        ]
        person = float(row["test_by_category"]["Person"]["R2_mean"])
        gap = float(np.mean(others)) - person
        ax.bar(
            i, gap, width=width * 1.6, color=color, alpha=0.55 if ls == "--" else 0.85,
            edgecolor=color, hatch="//" if ls == "--" else None, zorder=3,
        )
        ax.text(i, gap + 0.006, f"{gap:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(specs)), [s[1].replace(" ", "\n") for s in specs], fontsize=7.2)
    ax.set_ylabel("R² gap (mean non-Person − Person)")
    ax.set_title(
        "Person penalty is direction-robust\n"
        "and not explained by anchor cardinality",
        fontsize=8.8,
    )
    ax.grid(axis="y", color="0.92", lw=0.6)

    fig.suptitle(
        "Supp. Fig. | Person remains the generalization bottleneck (secondary pattern)\n"
        "Unrestricted minimal anchors; bottleneck is the test category, not anchor semantics",
        fontsize=10.0,
    )
    return save_dual(fig, "Paper_SuppFig_person_bottleneck")


# ── Fig 4 — culture transfer + reliability (formerly Fig.5) ─────────────
def _load_fig4_transfer_data() -> dict:
    """OASIS overall → Japan means (96 images); affine params from canonical JSON."""
    aff = json.loads((CVAE / "subject_generalization_affine2d.json").read_text())
    ov = next(r for r in aff["results"] if r["predictor"] == "oasis_overall")
    df = pd.read_csv(CVAE / "subject_scores_oasis_alignment_1to7.csv")
    cols = ["valence", "arousal", "valence_mean_1to7", "arousal_mean_1to7"]
    df = df.dropna(subset=cols)
    x = df[["valence", "arousal"]].to_numpy(float)
    y = df[["valence_mean_1to7", "arousal_mean_1to7"]].to_numpy(float)
    A = np.asarray(ov["affine_params"]["A_2x2"], float)
    b = np.asarray(ov["affine_params"]["b_2"], float)
    y_aff = x @ A.T + b
    return {
        "x": x,
        "y": y,
        "y_aff": y_aff,
        "raw": ov["raw"],
        "affine": ov["affine2d"],
    }


def _fig4_bootstrap_r2_samples(*, n_boot: int = 2000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Subject bootstrap R² for Valence / Arousal (matches population_bridge analysis④)."""
    from analysis_population_bridge_suite import fit_affine, load_japan_aligned, load_japan_subject_means

    df = load_japan_aligned()
    subj_means = load_japan_subject_means()
    align = df[["image_id", "valence", "arousal"]].copy()
    subjects = subj_means["subject_id"].unique()
    rng = np.random.default_rng(seed)
    r2_v, r2_a = [], []
    for _ in range(n_boot):
        samp = rng.choice(subjects, size=len(subjects), replace=True)
        boot_all = pd.concat([subj_means[subj_means["subject_id"] == sid] for sid in samp])
        boot_mean = boot_all.groupby("image_id")[["valence", "arousal"]].mean().reset_index()
        boot_mean = boot_mean.merge(align, on="image_id", suffixes=("_jp", "_oasis"))
        X = boot_mean[["valence_oasis", "arousal_oasis"]].to_numpy(float)
        Y = boot_mean[["valence_jp", "arousal_jp"]].to_numpy(float)
        pred = fit_affine(X, Y).apply(X)
        r2_v.append(r2_score(Y[:, 0], pred[:, 0]))
        r2_a.append(r2_score(Y[:, 1], pred[:, 1]))
    return np.asarray(r2_v, float), np.asarray(r2_a, float)


def _fig4_loso_signed_delta_r2() -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-subject signed ΔR² = R²(−i) − R²(full), separately for Valence / Arousal."""
    from analysis_population_bridge_suite import fit_affine, load_japan_aligned, load_japan_subject_means

    df = load_japan_aligned()
    subj_means = load_japan_subject_means()
    align = df[["image_id", "valence", "arousal"]].copy()
    subjects = sorted(subj_means["subject_id"].unique())

    full = align.merge(
        subj_means.groupby("image_id")[["valence", "arousal"]].mean().reset_index(),
        on="image_id",
        suffixes=("_oasis", "_jp"),
    )
    X_full = full[["valence_oasis", "arousal_oasis"]].to_numpy(float)
    Y_full = full[["valence_jp", "arousal_jp"]].to_numpy(float)
    pred_full = fit_affine(X_full, Y_full).apply(X_full)
    r2_v_full = float(r2_score(Y_full[:, 0], pred_full[:, 0]))
    r2_a_full = float(r2_score(Y_full[:, 1], pred_full[:, 1]))

    d_v, d_a = [], []
    for sid in subjects:
        remaining = subj_means[subj_means["subject_id"] != sid]
        jp_mean = remaining.groupby("image_id")[["valence", "arousal"]].mean().reset_index()
        merged = align.merge(jp_mean, on="image_id", suffixes=("_oasis", "_jp"))
        X = merged[["valence_oasis", "arousal_oasis"]].to_numpy(float)
        Y = merged[["valence_jp", "arousal_jp"]].to_numpy(float)
        pred = fit_affine(X, Y).apply(X)
        d_v.append(float(r2_score(Y[:, 0], pred[:, 0]) - r2_v_full))
        d_a.append(float(r2_score(Y[:, 1], pred[:, 1]) - r2_a_full))

    meta = {
        "r2_full_v": r2_v_full,
        "r2_full_a": r2_a_full,
        "max_abs_v": float(np.max(np.abs(d_v))),
        "max_abs_a": float(np.max(np.abs(d_a))),
        "n_subjects": len(subjects),
    }
    return np.asarray(d_v, float), np.asarray(d_a, float), meta


def _draw_fig4_loso_va_scatter(ax, d_v: np.ndarray, d_a: np.ndarray, meta: dict) -> None:
    """Panel C: signed ΔR² on Valence (x) × Arousal (y)."""
    lim = max(0.012, float(np.max(np.abs(np.concatenate([d_v, d_a])))) * 1.35)
    ax.axhline(0.0, color="0.55", lw=0.9, zorder=1)
    ax.axvline(0.0, color="0.55", lw=0.9, zorder=1)

    ax.scatter(
        d_v, d_a, s=30, alpha=0.72, c="#455a64",
        edgecolors="white", linewidths=0.4, zorder=3,
    )
    # mark extremes per axis
    i_v = int(np.argmax(np.abs(d_v)))
    i_a = int(np.argmax(np.abs(d_a)))
    ax.scatter(
        [d_v[i_v]], [d_a[i_v]], s=60, facecolors="none",
        edgecolors=FIG4_COL_V, linewidths=1.5, zorder=4, label="max |Δ|_V",
    )
    ax.scatter(
        [d_v[i_a]], [d_a[i_a]], s=60, facecolors="none",
        edgecolors=FIG4_COL_A, linewidths=1.5, zorder=4, label="max |Δ|_A",
    )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ΔR² Valence  (R²(−i) − R²_full)")
    ax.set_ylabel("ΔR² Arousal  (R²(−i) − R²_full)")
    ax.set_title(
        "LOSO signed ΔR² (V × A)\n"
        f"max|Δ|_V={meta['max_abs_v']:.3f}, max|Δ|_A={meta['max_abs_a']:.3f}",
        fontsize=8.0,
    )
    ax.text(
        0.02, 0.98,
        "near origin ⇒ no subject\ndrives either dimension",
        transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color="0.4",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", alpha=0.88, ec="0.85"),
    )
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    ax.grid(color="0.92", lw=0.6)


def _draw_fig4_transfer_scatter(
    ax,
    x_raw: np.ndarray,
    x_aff: np.ndarray,
    y_obs: np.ndarray,
    *,
    color: str,
    dim_label: str,
    raw_stats: dict,
    aff_r2: float,
) -> None:
    lim = FIG4_VA_LIM
    ax.scatter(
        x_raw, y_obs, s=28, alpha=0.50, color=color,
        edgecolors="white", linewidths=0.35, zorder=2,
    )
    ax.plot(lim, lim, "k--", lw=0.9, alpha=0.45, zorder=1)
    for xs, ls, lw, alpha, tag in (
        (x_raw, ":", 1.3, 0.85, f"raw (r={raw_stats['Pearson_r']:.2f})"),
        (x_aff, "-", 1.8, 0.95, f"affine (R²={aff_r2:.2f})"),
    ):
        coef = np.polyfit(xs, y_obs, 1)
        xx = np.linspace(*lim, 50)
        ax.plot(xx, np.polyval(coef, xx), ls, color=color, lw=lw, alpha=alpha, zorder=3, label=tag)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks(np.arange(lim[0], lim[1] + 0.01, 1.0))
    ax.set_yticks(np.arange(lim[0], lim[1] + 0.01, 1.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel(f"Predicted {dim_label} (OASIS → Japan)")
    ax.set_ylabel(f"Observed {dim_label} (Japan)")
    ax.set_title(dim_label, fontsize=9.5, color=color, fontweight="bold")
    ax.legend(frameon=False, fontsize=6.2, loc="lower right")
    ax.grid(color="0.92", lw=0.6)


def paper_fig4_panel_a() -> tuple[Path, Path]:
    """Standalone Fig.4-A: Valence | Arousal transfer scatters (axes 1–7, 1:1)."""
    transfer = _load_fig4_transfer_data()
    raw_a_r2 = float(transfer["raw"]["arousal"]["R2"])

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    _draw_fig4_transfer_scatter(
        axes[0],
        transfer["x"][:, 0], transfer["y_aff"][:, 0], transfer["y"][:, 0],
        color=FIG4_COL_V, dim_label="Valence",
        raw_stats=transfer["raw"]["valence"],
        aff_r2=float(transfer["affine"]["valence"]["R2"]),
    )
    _draw_fig4_transfer_scatter(
        axes[1],
        transfer["x"][:, 1], transfer["y_aff"][:, 1], transfer["y"][:, 1],
        color=FIG4_COL_A, dim_label="Arousal",
        raw_stats=transfer["raw"]["arousal"],
        aff_r2=float(transfer["affine"]["arousal"]["R2"]),
    )
    axes[1].text(
        0.03, 0.03,
        f"Raw: r={transfer['raw']['arousal']['Pearson_r']:.2f}, "
        f"R²={raw_a_r2:.2f}",
        transform=axes[1].transAxes, va="bottom", ha="left", fontsize=6.2,
        color=FIG4_COL_A,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", alpha=0.92, ec="0.85"),
    )
    fig.suptitle(
        "Fig. 4A | Direct OASIS prediction vs Japan observed (dashed = identity)",
        fontsize=10.0, y=1.02,
    )
    fig.tight_layout()
    return save_dual(fig, "Paper_Fig4_panelA_transfer_scatter")


def _draw_fig4_raw_vs_affine_points(
    ax,
    x_raw: np.ndarray,
    x_aff: np.ndarray,
    y_obs: np.ndarray,
    *,
    aff_color: str,
    dim_label: str,
    raw_r: float,
    aff_r2: float,
) -> None:
    """Both clouds: raw (x_raw,y) and affine-after (x_aff,y) with distinct colours."""
    lim = FIG4_VA_LIM
    raw_color = "#78909c"
    # Faint segments: same image moves horizontally (y fixed) under Φ
    for xr, xa, yo in zip(x_raw, x_aff, y_obs):
        ax.plot([xr, xa], [yo, yo], color="0.82", lw=0.45, alpha=0.55, zorder=1)
    ax.scatter(
        x_raw, y_obs, s=26, alpha=0.55, color=raw_color,
        edgecolors="white", linewidths=0.35, zorder=2,
        label=f"raw pred (r={raw_r:.2f})",
    )
    ax.scatter(
        x_aff, y_obs, s=26, alpha=0.70, color=aff_color,
        edgecolors="white", linewidths=0.35, zorder=3,
        label=f"affine pred (R²={aff_r2:.2f})",
    )
    ax.plot(lim, lim, "k--", lw=0.95, alpha=0.50, zorder=1, label="identity")
    # Regressions through the *visible* clouds
    xx = np.linspace(*lim, 50)
    for xs, ls, lw, col, tag in (
        (x_raw, ":", 1.4, raw_color, "fit on raw points"),
        (x_aff, "-", 1.8, aff_color, "fit on affine points"),
    ):
        coef = np.polyfit(xs, y_obs, 1)
        ax.plot(xx, np.polyval(coef, xx), ls, color=col, lw=lw, alpha=0.95, zorder=4, label=tag)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks(np.arange(lim[0], lim[1] + 0.01, 1.0))
    ax.set_yticks(np.arange(lim[0], lim[1] + 0.01, 1.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel(f"Predicted {dim_label} (OASIS → Japan)")
    ax.set_ylabel(f"Observed {dim_label} (Japan)")
    ax.set_title(dim_label, fontsize=9.5, color=aff_color, fontweight="bold")
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    ax.grid(color="0.92", lw=0.6)


def paper_fig4_panel_a_raw_vs_affine() -> tuple[Path, Path]:
    """Extra Fig.4-A: raw vs affine-after points (different colours; honest clouds)."""
    transfer = _load_fig4_transfer_data()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.3))
    _draw_fig4_raw_vs_affine_points(
        axes[0],
        transfer["x"][:, 0], transfer["y_aff"][:, 0], transfer["y"][:, 0],
        aff_color=FIG4_COL_V, dim_label="Valence",
        raw_r=float(transfer["raw"]["valence"]["Pearson_r"]),
        aff_r2=float(transfer["affine"]["valence"]["R2"]),
    )
    _draw_fig4_raw_vs_affine_points(
        axes[1],
        transfer["x"][:, 1], transfer["y_aff"][:, 1], transfer["y"][:, 1],
        aff_color=FIG4_COL_A, dim_label="Arousal",
        raw_r=float(transfer["raw"]["arousal"]["Pearson_r"]),
        aff_r2=float(transfer["affine"]["arousal"]["R2"]),
    )
    fig.suptitle(
        "Fig. 4A (extra) | Raw vs affine-after predictions (same Japan observed)\n"
        "Grey = OASIS raw; colour = Φ(x)=Ax+b; faint lines = per-image x-shift; dashed = identity",
        fontsize=9.5, y=1.04,
    )
    fig.tight_layout()
    return save_dual(fig, "Paper_Fig4_panelA_raw_vs_affine_points")


def _draw_fig4_culture_fp_panel(ax, cents: dict) -> None:
    """Fig.4-C prototype: culture FP near Japan centroid (vs gender FP offset)."""
    fp_g = cents["fixed_points"]["gender"]
    fp_c = cents["fixed_points"]["culture"]
    oasis = cents["centroids"]["oasis_900_overall"]
    japan = cents["centroids"]["japan_96"]
    off = cents["offsets"]
    d_g = float(off["gender_fp_minus_oasis_overall"]["euclidean"])
    d_c_o = float(off["culture_fp_minus_oasis_overall"]["euclidean"])
    d_c_j = float(off["culture_fp_minus_japan_96"]["euclidean"])

    # faint background: Japan-aligned OASIS→Japan points if available later; grid only
    ax.grid(color="0.92", lw=0.6, zorder=0)

    # Gender FP as contrast (Fig.3-D reference), de-emphasized
    ax.scatter(
        [fp_g["valence"]], [fp_g["arousal"]],
        s=110, marker="*", c="#2e7d32", alpha=0.35, edgecolors="white",
        linewidths=0.6, zorder=3, label=f"Gender FP (Fig.3-D; d={d_g:.2f} from OASIS)",
    )
    ax.plot(
        [oasis["valence"], fp_g["valence"]],
        [oasis["arousal"], fp_g["arousal"]],
        color="#2e7d32", ls="--", lw=1.0, alpha=0.35, zorder=2,
    )

    # Culture story (primary)
    ax.scatter(
        [oasis["valence"]], [oasis["arousal"]],
        s=70, marker="D", c="#616161", edgecolors="white", linewidths=0.7,
        zorder=4, label="OASIS centroid",
    )
    ax.scatter(
        [japan["valence"]], [japan["arousal"]],
        s=85, marker="s", c="#6a1b9a", edgecolors="white", linewidths=0.7,
        zorder=5, label="Japan centroid",
    )
    ax.scatter(
        [fp_c["valence"]], [fp_c["arousal"]],
        s=130, marker="P", c="#ef6c00", edgecolors="white", linewidths=0.8,
        zorder=6, label="Culture FP",
    )
    ax.plot(
        [oasis["valence"], fp_c["valence"]],
        [oasis["arousal"], fp_c["arousal"]],
        color="#ef6c00", ls="--", lw=1.2, alpha=0.85, zorder=2,
    )
    ax.plot(
        [japan["valence"], fp_c["valence"]],
        [japan["arousal"], fp_c["arousal"]],
        color="#ef6c00", ls="-", lw=1.6, alpha=0.95, zorder=3,
    )
    # annotate the short Japan↔Culture FP link
    mid_v = 0.5 * (japan["valence"] + fp_c["valence"])
    mid_a = 0.5 * (japan["arousal"] + fp_c["arousal"])
    ax.annotate(
        f"d={d_c_j:.2f}",
        xy=(mid_v, mid_a), xytext=(mid_v + 0.55, mid_a + 0.35),
        fontsize=7.5, color="#ef6c00", fontweight="bold",
        arrowprops=dict(arrowstyle="-", color="#ef6c00", lw=0.8),
    )

    ax.text(
        0.02, 0.02,
        "Culture FP hugs cohort means\n"
        f"→OASIS d={d_c_o:.2f} · →Japan d={d_c_j:.2f}\n"
        f"Gender FP→OASIS d={d_g:.2f} (contrast)",
        transform=ax.transAxes, va="bottom", ha="left", fontsize=6.6,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85", alpha=0.92),
    )
    ax.set_xlim(*FIG4_VA_LIM)
    ax.set_ylim(*FIG4_VA_LIM)
    ax.set_xticks(np.arange(FIG4_VA_LIM[0], FIG4_VA_LIM[1] + 0.01, 1.0))
    ax.set_yticks(np.arange(FIG4_VA_LIM[0], FIG4_VA_LIM[1] + 0.01, 1.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(
        "Culture FP near Japan centroid\n"
        "(gender FP offset shown faint — cf. Fig.3-D)",
        fontsize=8.2,
    )
    ax.legend(frameon=False, fontsize=5.6, loc="upper left")


def paper_fig4_panel_c_culture_fp() -> tuple[Path, Path]:
    """Standalone Fig.4-C prototype: culture fixed point vs Japan centroid."""
    cents = json.loads((PBA / "va_centroids_vs_fixed_points.json").read_text())
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    _draw_fig4_culture_fp_panel(ax, cents)
    fig.suptitle(
        "Fig. 4C (prototype) | Culture bridge FP ≈ Japan centroid",
        fontsize=10.0, y=1.01,
    )
    fig.tight_layout()
    return save_dual(fig, "Paper_Fig4_panelC_culture_fp_centroid")


def paper_fig4() -> tuple[Path, Path]:
    """NatComm Fig.4 — 1×3: transfer scatter / R²+CI / culture FP vs Japan centroid."""
    transfer = _load_fig4_transfer_data()
    rel = json.loads((PBA / "reliability_ceiling.json").read_text())
    boot = json.loads((PBA / "subject_bootstrap_ci.json").read_text())
    cb = boot["phi_culture_subject_bootstrap"]
    cents = json.loads((PBA / "va_centroids_vs_fixed_points.json").read_text())
    r2_v_boot, r2_a_boot = _fig4_bootstrap_r2_samples()

    obs_v = float(rel["observed_affine2d_r2"]["valence"])
    obs_a = float(rel["observed_affine2d_r2"]["arousal"])
    ceil_v = float(rel["ceiling_r2"]["valence"])
    ceil_a = float(rel["ceiling_r2"]["arousal"])
    ceil_show = float(np.mean([ceil_v, ceil_a]))
    raw_a_r2 = float(transfer["raw"]["arousal"]["R2"])

    fig = plt.figure(figsize=(14.2, 4.8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.35, 1.0, 1.0], wspace=0.28)

    # A: predicted vs observed (Valence | Arousal)
    gs_a = gs[0, 0].subgridspec(1, 2, wspace=0.22)
    ax_av = fig.add_subplot(gs_a[0, 0])
    ax_aa = fig.add_subplot(gs_a[0, 1])
    panel_label(ax_av, "A")
    _draw_fig4_transfer_scatter(
        ax_av,
        transfer["x"][:, 0], transfer["y_aff"][:, 0], transfer["y"][:, 0],
        color=FIG4_COL_V, dim_label="Valence",
        raw_stats=transfer["raw"]["valence"],
        aff_r2=float(transfer["affine"]["valence"]["R2"]),
    )
    _draw_fig4_transfer_scatter(
        ax_aa,
        transfer["x"][:, 1], transfer["y_aff"][:, 1], transfer["y"][:, 1],
        color=FIG4_COL_A, dim_label="Arousal",
        raw_stats=transfer["raw"]["arousal"],
        aff_r2=float(transfer["affine"]["arousal"]["R2"]),
    )
    ax_aa.text(
        0.03, 0.03,
        f"Raw: r={transfer['raw']['arousal']['Pearson_r']:.2f}, "
        f"R²={raw_a_r2:.2f}",
        transform=ax_aa.transAxes, va="bottom", ha="left", fontsize=6.0,
        color=FIG4_COL_A,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", alpha=0.92, ec="0.85"),
    )

    # B: violin+swarm + 95% CI
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")
    _draw_fig4_r2_violin_swarm(
        ax_b, r2_v_boot, r2_a_boot, obs_v, obs_a, ceil_v, ceil_a, ceil_show, cb,
    )

    fig_bar, ax_bar = plt.subplots(figsize=(4.6, 4.2))
    _draw_fig4_r2_bars(ax_bar, obs_v, obs_a, ceil_v, ceil_a, ceil_show, cb, raw_a_r2)
    fig_bar.tight_layout()
    save_dual(fig_bar, "Paper_Fig4_panelB_bars")
    plt.close(fig_bar)

    # C: culture FP vs Japan centroid (candidate-2 prototype)
    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "C")
    _draw_fig4_culture_fp_panel(ax_c, cents)

    # Also keep LOSO signed-ΔR² as a side export (not main-text C)
    fig_loso, ax_loso = plt.subplots(figsize=(4.6, 4.4))
    d_v_loso, d_a_loso, loso_meta = _fig4_loso_signed_delta_r2()
    _draw_fig4_loso_va_scatter(ax_loso, d_v_loso, d_a_loso, loso_meta)
    fig_loso.suptitle("Fig.4 LOSO signed ΔR² (Supp. candidate)", fontsize=9.5, y=1.01)
    fig_loso.tight_layout()
    save_dual(fig_loso, "Paper_Fig4_panelC_loso_signed_delta")
    plt.close(fig_loso)

    r49_v = rel["spearman_brown_r49"]["valence"]
    r49_a = rel["spearman_brown_r49"]["arousal"]
    fig.suptitle(
        "Fig. 4 | Cross-cultural transfer: Valence shares reliably; Arousal hits a wall\n"
        f"r₄₉ = {r49_v:.3f} (V) / {r49_a:.3f} (A); "
        "C = culture FP near Japan centroid (cf. gender offset in Fig.3-D)",
        y=1.02, fontsize=9.8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_dual(fig, "Paper_Fig4_japan_transfer_reliability")


def _draw_fig4_r2_violin_swarm(
    ax,
    r2_v: np.ndarray,
    r2_a: np.ndarray,
    obs_v: float,
    obs_a: float,
    ceil_v: float,
    ceil_a: float,
    ceil_show: float,
    cb: dict,
) -> None:
    """Panel B: violin+swarm at x; 95% CI ticks at x+0.1; ceiling gap."""
    data = [r2_v, r2_a]
    cols = [FIG4_COL_V, FIG4_COL_A]
    xpos = np.array([0.0, 1.0])
    ci_off = 0.12  # former panel-C info, offset on the same axes

    parts = ax.violinplot(data, positions=xpos, showmedians=False, widths=0.55)
    for body, col in zip(parts["bodies"], cols):
        body.set_facecolor(col)
        body.set_alpha(0.28)
        body.set_edgecolor(col)
    for key in ("cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_visible(False)

    rng = np.random.default_rng(1)
    for i, (arr, col, obs, ceil) in enumerate(
        zip(data, cols, (obs_v, obs_a), (ceil_v, ceil_a))
    ):
        n = min(len(arr), 220)
        idx = rng.choice(len(arr), size=n, replace=False)
        # swarm slightly left of centre so CI at +0.12 stays readable
        jitter = rng.uniform(-0.10, 0.02, size=n)
        ax.scatter(
            xpos[i] + jitter, arr[idx], s=8, alpha=0.16, color=col,
            edgecolors="none", zorder=2,
        )
        # point estimate
        ax.scatter(
            [xpos[i]], [obs], s=52, color=col, edgecolors="white",
            linewidths=0.8, zorder=5,
        )
        ax.text(
            xpos[i], obs + 0.035, f"{obs:.2f}",
            ha="center", va="bottom", fontsize=7.8, color=col, fontweight="bold",
        )
        # ceiling tick on violin
        ax.plot(
            [xpos[i] - 0.22, xpos[i] + 0.06], [ceil] * 2,
            color=col, lw=1.7, alpha=0.85, zorder=3,
        )
        # 95% CI at x+0.12 (merged former panel C)
        lo, hi = cb["r2_v" if i == 0 else "r2_a"]["ci95"]
        xci = xpos[i] + ci_off
        ax.plot([xci, xci], [lo, hi], color=col, lw=2.2, solid_capstyle="round", zorder=4)
        ax.plot([xci - 0.05, xci + 0.05], [lo, lo], color=col, lw=2.0, zorder=4)
        ax.plot([xci - 0.05, xci + 0.05], [hi, hi], color=col, lw=2.0, zorder=4)
        ax.text(
            xci + 0.06, hi, f"[{lo:.2f}, {hi:.2f}]",
            ha="left", va="bottom", fontsize=6.4, color=col,
        )

    ax.axhline(ceil_show, color="0.45", ls="--", lw=1.0, zorder=1)
    ax.axhspan(0.98, 1.05, color="0.88", alpha=0.45, zorder=0)
    ax.annotate(
        "", xy=(1.28, ceil_a), xytext=(1.28, obs_a),
        arrowprops=dict(arrowstyle="<->", color=FIG4_COL_A, lw=1.4),
    )
    ax.text(
        1.36, 0.5 * (obs_a + ceil_a), "Arousal gap\n(cultural cost)",
        va="center", ha="left", fontsize=7.0, color=FIG4_COL_A, fontweight="bold",
    )
    ax.text(
        1.02, 0.995, f"ceiling ≈ {ceil_show:.2f}",
        va="top", ha="left", fontsize=6.6, color="0.35",
    )
    ax.text(
        0.5, 0.55, "non-overlapping\n95% CIs",
        ha="center", va="center", fontsize=7.0, color="0.45", style="italic",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", alpha=0.78, ec="0.85"),
    )
    ax.set_xticks(xpos, ["Valence", "Arousal"])
    ax.set_xlim(-0.45, 1.75)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Affine-calibrated R² (Japan target)")
    ax.set_title(
        "Valence nears ceiling; Arousal stays below\n"
        "violin+swarm; 95% CI offset at +0.12",
        fontsize=8.0,
    )


def _draw_fig4_r2_bars(
    ax,
    obs_v: float,
    obs_a: float,
    ceil_v: float,
    ceil_a: float,
    ceil_show: float,
    cb: dict,
    raw_a_r2: float,
) -> None:
    """Legacy bar-style panel B (also exported as standalone)."""
    ax.axhspan(0.98, 1.0, color="0.88", alpha=0.55, zorder=0)
    ax.axhline(ceil_show, color="0.45", ls="--", lw=1.1, zorder=1)
    ax.text(
        1.02, 0.992, f"Reliability ceiling\n(R² ≈ {ceil_show:.2f})",
        va="top", ha="left", fontsize=7.0, color="0.35",
    )
    xpos = np.array([0.0, 1.0])
    vals = np.array([obs_v, obs_a])
    cols = [FIG4_COL_V, FIG4_COL_A]
    ax.bar(
        xpos, vals, width=0.52, color=cols, alpha=0.88,
        edgecolor="white", linewidth=0.8, zorder=3,
    )
    for xp, val, col, key in zip(xpos, vals, cols, ("valence", "arousal")):
        lo, hi = cb[f"r2_{'v' if key == 'valence' else 'a'}"]["ci95"]
        ax.errorbar(
            xp, val, yerr=[[val - lo], [hi - val]], fmt="none",
            ecolor="0.25", elinewidth=1.3, capsize=4, zorder=4,
        )
        ax.plot([xp - 0.26, xp + 0.26], [ceil_v if key == "valence" else ceil_a] * 2, color=col, lw=1.6, alpha=0.75, zorder=2)
        ax.text(xp, val + 0.03, f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, color=col, fontweight="bold")
    gap_mid = 0.5 * (obs_a + ceil_a)
    ax.annotate(
        "", xy=(1.18, ceil_a), xytext=(1.18, obs_a),
        arrowprops=dict(arrowstyle="<->", color=FIG4_COL_A, lw=1.5),
    )
    ax.fill_between([0.72, 1.28], obs_a, ceil_a, color=FIG4_COL_A, alpha=0.10, zorder=1)
    ax.text(
        1.30, gap_mid, "Arousal gap\n(cultural cost)",
        va="center", ha="left", fontsize=7.5, color=FIG4_COL_A, fontweight="bold",
    )
    ax.text(
        1.0, -0.10,
        f"Raw Arousal R² = {raw_a_r2:.2f} (not shown; off-scale)",
        transform=ax.transData, ha="center", fontsize=6.8, color=FIG4_COL_A,
    )
    ax.set_xticks(xpos, ["Valence", "Arousal"])
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Affine-calibrated R² (Japan target)")
    ax.set_title(
        "Fig.4-B (bar variant)\nValence nears ceiling; Arousal stays far below",
        fontsize=8.6,
    )

# ── Fig 5 — residual geometry (old Fig.6 A–C) ─────────────────────────
def _draw_fig5_residual_violin(ax, per_img: pd.DataFrame, res: dict) -> None:
    """Per-image post-Φ residual L2 by semantic category (Person enriched)."""
    order = [c for c in ["Scene", "Animal", "Object", "Person"] if c in per_img["category"].unique()]
    groups = {c: per_img.loc[per_img["category"] == c, "residual_l2"].to_numpy() for c in order}
    vcolors = ["#ffb74d" if c == "Person" else "#90caf9" for c in order]
    violin_strip(ax, groups, colors=vcolors, ylabel="Per-image residual L2 after Φ")
    means = [float(np.mean(groups[c])) for c in order]
    ax.scatter(
        range(len(order)), means,
        s=55, marker="D", c="#212121", edgecolors="white", linewidths=0.8,
        zorder=5, label="mean",
    )
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    p_po = res["gender_oasis_900"]["person_vs_object_permutation_p"]
    cm = res["gender_oasis_900"]["category_means"]
    ax.set_title(
        f"Person-enriched residual\n"
        f"Person mean {cm['Person']:.3f} vs Object {cm['Object']:.3f} "
        f"(perm p={p_po:.3f})",
        fontsize=8.8,
    )


def _draw_fig5_residual_mean_sd_bars(ax, per_img: pd.DataFrame, res: dict) -> None:
    """Category mean ± SD of post-Φ residual L2 (companion to the violin)."""
    order = [c for c in ["Scene", "Animal", "Object", "Person"] if c in per_img["category"].unique()]
    means, sds, ns = [], [], []
    for c in order:
        y = per_img.loc[per_img["category"] == c, "residual_l2"].to_numpy(float)
        means.append(float(np.mean(y)))
        sds.append(float(np.std(y, ddof=1)) if len(y) > 1 else 0.0)
        ns.append(len(y))
    colors = ["#ffb74d" if c == "Person" else "#90caf9" for c in order]
    x = np.arange(len(order))
    ax.bar(
        x, means, yerr=sds, color=colors, edgecolor="0.35", linewidth=0.8,
        width=0.72, capsize=4, error_kw=dict(ecolor="0.25", lw=1.1, capthick=1.1),
        zorder=2,
    )
    for i, (m, sd, n) in enumerate(zip(means, sds, ns)):
        ax.text(
            i, m + sd + 0.02, f"{m:.3f}",
            ha="center", va="bottom", fontsize=7.5, color="0.2",
        )
    p_po = res["gender_oasis_900"]["person_vs_object_permutation_p"]
    cm = res["gender_oasis_900"]["category_means"]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(order, ns)])
    ax.set_ylabel("Mean residual L2 after Φ (± SD)")
    ax.set_ylim(0, max(m + s for m, s in zip(means, sds)) * 1.18)
    ax.set_title(
        f"Category residual (mean ± SD)\n"
        f"Person {cm['Person']:.3f} vs Object {cm['Object']:.3f} "
        f"(perm p={p_po:.3f})",
        fontsize=8.8,
    )


def _load_fig5_packs() -> dict:
    """Shared data for the 2×3 Fig.5 layout and its per-panel SVG exports."""
    g_tw, step_tw, packs_tw = _load_excess_twist_maps()
    g_topo, packs_topo = _load_residual_topology_packs()
    payload, cats, specs = _load_person_bottleneck_payload()
    return {
        "res": json.loads((PBA / "residual_structure.json").read_text()),
        "per_img": pd.read_csv(PBA / "residual_per_image_gender.csv"),
        "g_tw": g_tw,
        "step_tw": step_tw,
        "packs_tw": packs_tw,
        "twist_vmax": max(
            float(np.nanpercentile(packs_tw[d]["grid"][np.isfinite(packs_tw[d]["grid"])], 95))
            for d in ("MtoF", "FtoM")
        ),
        "g_topo": g_topo,
        "packs_topo": packs_topo,
        "topo_vmax": max(
            float(np.nanquantile(_residual_mag(packs_topo["MtoF"]), 0.98)),
            float(np.nanquantile(_residual_mag(packs_topo["FtoM"]), 0.98)),
            1e-4,
        ),
        "pb_payload": payload,
        "pb_cats": cats,
        "pb_specs": specs,
    }


def paper_fig5_residual_geometry() -> tuple[Path, Path]:
    """NatComm Fig.5 A–F — structured residuals, excess twist, post-Φ residual failure maps.

    Layout (2×3): A residual violin / C twist M→F / D twist F→M
                  B Person bottleneck / E residual ‖Δ‖+direction M→F / F residual ‖Δ‖+direction F→M
    """
    d = _load_fig5_packs()

    fig = plt.figure(figsize=(15.6, 10.4), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[0.92, 1.0, 1.0], hspace=0.10, wspace=0.10)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")
    _draw_fig5_residual_violin(ax_a, d["per_img"], d["res"])

    ax_b = fig.add_subplot(gs[1, 0])
    panel_label(ax_b, "B")
    _draw_person_bottleneck_curve(ax_b, d["pb_payload"], d["pb_cats"], d["pb_specs"])

    ax_c = fig.add_subplot(gs[0, 1])
    panel_label(ax_c, "C")
    _draw_excess_twist_panel(
        ax_c, d["g_tw"], d["step_tw"], d["packs_tw"], "MtoF", fig=fig, vmax=d["twist_vmax"],
    )

    ax_dd = fig.add_subplot(gs[0, 2])
    panel_label(ax_dd, "D")
    _draw_excess_twist_panel(
        ax_dd, d["g_tw"], d["step_tw"], d["packs_tw"], "FtoM", fig=fig, vmax=d["twist_vmax"],
    )

    ax_e = fig.add_subplot(gs[1, 1])
    panel_label(ax_e, "E")
    _draw_topology_single(
        ax_e, d["g_topo"], d["packs_topo"], "MtoF", fig=fig, vmax=d["topo_vmax"],
    )

    ax_f = fig.add_subplot(gs[1, 2])
    panel_label(ax_f, "F")
    _draw_topology_single(
        ax_f, d["g_topo"], d["packs_topo"], "FtoM", fig=fig, vmax=d["topo_vmax"],
    )

    fig.suptitle(
        "Fig. 5 | Structured residuals and post-Φ geometry (R5–R6)\n"
        "what the linear bridge discards: category-selective residual, excess twist, Φ-failure map + direction",
        fontsize=11,
    )
    _export_fig5_panel_svgs(d)
    return save_dual(fig, "Paper_Fig5_residual_geometry")


def _export_fig5_panel_svgs(d: dict) -> None:
    """Standalone Fig.5 A–F panels (SVG + PNG) for manual polishing."""
    OUT.mkdir(parents=True, exist_ok=True)

    def _write(fig: plt.Figure, stem: str) -> None:
        fig.savefig(OUT / f"{stem}.svg", format="svg", bbox_inches="tight", facecolor="white")
        fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig_a, ax = plt.subplots(figsize=(5.2, 4.6))
    _draw_fig5_residual_violin(ax, d["per_img"], d["res"])
    fig_a.tight_layout()
    _write(fig_a, "Paper_Fig5_panelA_category_residual")

    fig_a_bar, ax = plt.subplots(figsize=(5.2, 4.6))
    _draw_fig5_residual_mean_sd_bars(ax, d["per_img"], d["res"])
    fig_a_bar.tight_layout()
    _write(fig_a_bar, "Paper_Fig5_panelA_category_residual_mean_sd")

    fig_b, ax = plt.subplots(figsize=(5.2, 4.6))
    _draw_person_bottleneck_curve(ax, d["pb_payload"], d["pb_cats"], d["pb_specs"])
    fig_b.tight_layout()
    _write(fig_b, "Paper_Fig5_panelB_person_bottleneck")

    for stem, direction in (
        ("Paper_Fig5_panelC_excess_twist_MtoF", "MtoF"),
        ("Paper_Fig5_panelD_excess_twist_FtoM", "FtoM"),
    ):
        fig_t, ax = plt.subplots(figsize=(5.4, 5.0))
        _draw_excess_twist_panel(
            ax, d["g_tw"], d["step_tw"], d["packs_tw"], direction,
            fig=fig_t, vmax=d["twist_vmax"],
        )
        fig_t.tight_layout()
        _write(fig_t, stem)

    for stem, direction in (
        ("Paper_Fig5_panelE_topology_MtoF", "MtoF"),
        ("Paper_Fig5_panelF_topology_FtoM", "FtoM"),
    ):
        fig_p, ax = plt.subplots(figsize=(5.4, 5.0))
        _draw_topology_single(
            ax, d["g_topo"], d["packs_topo"], direction, fig=fig_p, vmax=d["topo_vmax"],
        )
        fig_p.tight_layout()
        _write(fig_p, stem)


# ── Fig 6 — case translation discovery (exemplars + VA co-localization) ─
def paper_fig6_case_translation() -> tuple[Path, Path]:
    """NatComm Fig.6 — discovery: what case translation is, and where it helps.

    Circular rule: no scalar residual×improvement ρ≈0.90; VA spatial structure only.
    Method checks (null / VA lines / weak ρ≈0.16) → Paper_SuppFig_case_translation_validation;
    fixed-split robustness → Paper_SuppFig_case_translation_split_robustness.
    """
    from plot_fig6_case_translation_discovery import make_fig6_discovery

    png = OUT / "Paper_Fig6_case_translation.png"
    svg = OUT / "Paper_Fig6_case_translation.svg"
    make_fig6_discovery(png, svg)
    return png, svg


def paper_supp_phi_prediction_asymmetry() -> tuple[Path, Path]:
    """
    Bidirectional Φ asymmetry — displacement fields + round-trip only.

    Panels: a Φ_mf on male VA; b Φ_fm on female VA; c round-trip residual.
    Transition matrices, directional residual imbalance, and image exemplars
    are omitted (not cited in the main text).
    """
    from analysis_population_bridge_suite import fit_affine
    from config import OASIS_SCORES_CSV
    from dataset import load_oasis_meta

    oasis = load_oasis_meta(OASIS_SCORES_CSV)
    cols = ["valence_male", "arousal_male", "valence_female", "arousal_female"]
    df = oasis[oasis[cols].notna().all(axis=1)].copy().reset_index(drop=True)
    ym = df[["valence_male", "arousal_male"]].to_numpy(float)
    yf = df[["valence_female", "arousal_female"]].to_numpy(float)
    phi_mf = fit_affine(ym, yf)
    phi_fm = fit_affine(yf, ym)
    pred_f = phi_mf.apply(ym)
    rt_m = phi_fm.apply(pred_f)
    err_rt_m = np.linalg.norm(rt_m - ym, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    rng = np.random.default_rng(0)
    take = rng.choice(len(df), size=min(120, len(df)), replace=False)

    ax = axes[0]
    panel_label(ax, "A")
    ax.scatter(ym[:, 0], ym[:, 1], s=6, c="#90CAF9", alpha=0.35, edgecolors="none")
    d = pred_f - ym
    ax.quiver(
        ym[take, 0], ym[take, 1], d[take, 0], d[take, 1],
        angles="xy", scale_units="xy", scale=1.0, width=0.003,
        color="#1565C0", alpha=0.75,
    )
    ax.set_xlim(1, 7)
    ax.set_ylim(1, 7)
    ax.set_aspect("equal")
    ax.set_xlabel("Valence (male)")
    ax.set_ylabel("Arousal (male)")
    ax.set_title(r"A  $\Phi_{mf}$ displacement on male VA", fontsize=9)

    ax = axes[1]
    panel_label(ax, "B")
    pred_m = phi_fm.apply(yf)
    ax.scatter(yf[:, 0], yf[:, 1], s=6, c="#EF9A9A", alpha=0.35, edgecolors="none")
    d2 = pred_m - yf
    ax.quiver(
        yf[take, 0], yf[take, 1], d2[take, 0], d2[take, 1],
        angles="xy", scale_units="xy", scale=1.0, width=0.003,
        color="#C62828", alpha=0.75,
    )
    ax.set_xlim(1, 7)
    ax.set_ylim(1, 7)
    ax.set_aspect("equal")
    ax.set_xlabel("Valence (female)")
    ax.set_ylabel("Arousal (female)")
    ax.set_title(r"B  $\Phi_{fm}$ displacement on female VA", fontsize=9)

    ax = axes[2]
    panel_label(ax, "C")
    sc = ax.scatter(
        ym[:, 0], ym[:, 1], c=err_rt_m, s=12, cmap="magma",
        vmin=0, vmax=float(np.quantile(err_rt_m, 0.98)),
        edgecolors="none", alpha=0.85,
    )
    ax.set_xlim(1, 7)
    ax.set_ylim(1, 7)
    ax.set_aspect("equal")
    ax.set_xlabel("Valence (male)")
    ax.set_ylabel("Arousal (male)")
    ax.set_title(
        r"C  Round-trip residual"
        "\n"
        r"$\|\Phi_{fm}(\Phi_{mf}(y_m))-y_m\|$"
        f"  mean={err_rt_m.mean():.2f}",
        fontsize=9,
    )
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03).set_label("round-trip L2")

    fig.suptitle(
        "Supp. Fig. | Bidirectional asymmetry of the affine map\n"
        r"Independent $\Phi_{mf}$ / $\Phi_{fm}$ on n=900; maps are not inverses "
        r"($\Phi_{fm}\neq\Phi_{mf}^{-1}$)",
        fontsize=10.5,
    )
    return save_dual(fig, "Paper_SuppFig_phi_prediction_asymmetry_va_images")


def paper_supp_residual_field_topology() -> tuple[Path, Path]:
    """
    Post-Φ residual geometry: magnitude + walker dwell with residual-field arrows.

    Dwell panels carry the residual vector field (same arrows used for curl
    diagnostics); a separate |curl| heatmap row is omitted. Exploratory.
    """
    from plot_residual_field_topology import (
        DIR_SPECS,
        analyze_direction,
        build_frame,
        make_grid,
        prepare_stream,
    )

    df, _meta = build_frame()
    g = make_grid(0.1)
    packs = {
        d: analyze_direction(
            df, g, d,
            grid_step=0.1, radius=0.5, min_n=6,
            top_peaks=6, per_peak=4, curl_q=0.85,
        )
        for d in ("MtoF", "FtoM")
    }

    dwell_dir = RES / "residual_field_ball_sim"
    dwell = {
        "MtoF": np.load(dwell_dir / "Fig_residual_ball_dwell_MtoF_dwell.npy"),
        "FtoM": np.load(dwell_dir / "Fig_residual_ball_dwell_FtoM_dwell.npy"),
    }
    valid = {
        "MtoF": np.load(dwell_dir / "Fig_residual_ball_dwell_MtoF_valid.npy"),
        "FtoM": np.load(dwell_dir / "Fig_residual_ball_dwell_FtoM_valid.npy"),
    }
    # dwell grids are 60×60 on VA 1–7 (step 0.1); topology g may differ slightly
    g_dwell = np.arange(1.0 + 0.05, 7.0, 0.1)
    if len(g_dwell) != dwell["MtoF"].shape[0]:
        g_dwell = np.linspace(1.05, 6.95, dwell["MtoF"].shape[0])

    fig = plt.figure(figsize=(11.6, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    mag_vmax = max(
        float(np.nanquantile(_residual_mag(packs["MtoF"]), 0.98)),
        float(np.nanquantile(_residual_mag(packs["FtoM"]), 0.98)),
        1e-4,
    )
    dwell_z = {
        k: np.where(valid[k], np.log10(dwell[k].astype(float) + 1.0), np.nan)
        for k in ("MtoF", "FtoM")
    }
    dwell_vmax = max(
        float(np.nanmax(dwell_z["MtoF"])),
        float(np.nanmax(dwell_z["FtoM"])),
        1e-4,
    )

    # Row 0: residual magnitude + direction
    for col, key, lab in [(0, "MtoF", "A"), (1, "FtoM", "B")]:
        ax = fig.add_subplot(gs[0, col])
        panel_label(ax, lab)
        _draw_topology_single(
            ax, g, packs, key, fig=fig, vmax=mag_vmax,
        )
        sp = DIR_SPECS[key]
        ax.set_title(
            f"{lab}  ‖Δ‖ + direction ({sp['label']})\n{sp['resid_formula']}",
            fontsize=8.6,
        )

    # Row 1: dwell + residual-field arrows (curl evidence without |curl| heatmap)
    xg, yg = np.meshgrid(g_dwell, g_dwell, indexing="xy")
    gg = np.asarray(g, float)
    xgt, ygt = np.meshgrid(gg, gg, indexing="xy")
    step_q = max(1, len(gg) // 12)
    sl = (slice(None, None, step_q), slice(None, None, step_q))
    for col, key, lab in [(0, "MtoF", "C"), (1, "FtoM", "D")]:
        ax = fig.add_subplot(gs[1, col])
        panel_label(ax, lab)
        sp = DIR_SPECS[key]
        p = packs[key]
        im = ax.pcolormesh(
            xg, yg, dwell_z[key], shading="auto", cmap="inferno",
            vmin=0.0, vmax=dwell_vmax, zorder=1,
        )
        dx = np.asarray(p["dx"], float)
        dy = np.asarray(p["dy"], float)
        dx_s, dy_s, valid_f = prepare_stream(dx, dy)
        m = valid_f[sl]
        ax.quiver(
            xgt[sl][m], ygt[sl][m], dx[sl][m], dy[sl][m],
            color="cyan", alpha=0.9, angles="xy", scale_units="xy", scale=2.2,
            width=0.0035, zorder=3,
        )
        try:
            ax.streamplot(
                gg, gg, dx_s, dy_s,
                color="#80cbc4", density=1.05, linewidth=0.85,
                arrowsize=0.65, zorder=2, broken_streamlines=True,
            )
        except Exception:
            pass
        q90 = float(p["curl_stats"]["q90"])
        ax.set_xlim(1, 7)
        ax.set_ylim(1, 7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Valence ({sp['src_axis']})")
        ax.set_ylabel(f"Arousal ({sp['src_axis']})")
        ax.set_title(
            f"{lab}  Walker dwell ({sp['label']})\n"
            f"log₁₀(dwell+1); arrows = residual field (|curl| q90={q90:.2f})",
            fontsize=8.6,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("log₁₀(dwell+1)")

    fig.suptitle(
        "Supp. Fig. | Post-Φ residual magnitude and walker dwell\n"
        "Residual-field arrows on dwell (curl ≠ 0 → peaks are not potential attractors; "
        "Φ on n=900 group means)",
        fontsize=10.5, y=0.995,
    )
    return save_dual(fig, "Paper_SuppFig_residual_field_topology")


def paper_supp_z3_anchor_selection() -> tuple[Path, Path]:
    """Z3 anchor-selection compare (formerly main-text Fig.4-C)."""
    sel = pd.read_csv(ANC / "z3_anchor_selection_compare.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    panel_label(ax, "A")
    sub = sel[sel["bridge"] == "gender_M_to_F"].copy()
    methods = ["random", "similar", "fp_near"]
    mcol = {"random": "#1565c0", "similar": "#2e7d32", "fp_near": "#ef6c00"}
    for method in methods:
        s = sub[sub["method"] == method].sort_values("k")
        ax.plot(s["k"], s["R2_mean"], "o-", color=mcol[method], lw=1.8, ms=6, label=method)
    ax.axhline(0, color="0.5", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([3, 8, 21])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("k shared anchors")
    ax.set_ylabel("Understanding R² (held-out)")
    ax.set_title("Anchor selection: fp_near fails\n(broad coverage required)")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Supp. Fig. | Z3 anchor-selection comparison (ridge; M→F)", y=0.98, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_dual(fig, "Paper_SuppFig_z3_anchor_selection")


def paper_supp_minimal_anchor_coverage() -> tuple[Path, Path]:
    """
    Z4/Z5 minimal VA coverage — method detail (main-text summary → Fig.3-F).

    Full set-cover panels and agreement proxy; Fig.3-F shows direction overlay only.
    """
    from analysis_anchor_structure import z5_load_reference_points

    profile_path = ANC / "z5_minimal_anchor_profile.json"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Missing {profile_path.name}. Run: "
            "python3 code/analysis_anchor_structure.py --mode z5 --z5-agreement"
        )
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    profiles = payload["profiles"]
    overlap_path = ANC / "z5_overlap.json"
    overlap = (
        json.loads(overlap_path.read_text(encoding="utf-8"))
        if overlap_path.exists()
        else profiles.get("MtoF", {}).get("overlap_with_opposite", {})
    )
    refs = z5_load_reference_points()
    ov, oa = refs["oasis_centroid"]
    gv, ga = refs["gender_fp"]
    sd_path = ANC / "z5_agreement_proxy.csv"
    sd_df = pd.read_csv(sd_path) if sd_path.exists() else None

    m2f = profiles.get("MtoF") or next(iter(profiles.values()))
    n_panels = 3 if sd_df is not None and "sd_pair" in sd_df.columns else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.4))
    if n_panels == 1:
        axes = [axes]
    colors = {"MtoF": "#1565c0", "FtoM": "#c62828"}
    lim = VA_LIM
    edges = np.linspace(lim[0], lim[1], 6)

    def _grid(ax):
        for e in edges:
            ax.axvline(e, color="0.88", lw=0.6, zorder=0)
            ax.axhline(e, color="0.88", lw=0.6, zorder=0)
        ax.axhline(oa, color="#888", ls="--", lw=0.8, alpha=0.7, zorder=1)
        ax.axvline(ov, color="#888", ls="--", lw=0.8, alpha=0.7, zorder=1)

    ax = axes[0]
    panel_label(ax, "A")
    _grid(ax)
    ax.scatter([gv], [ga], s=90, marker="*", c="#2e7d32", edgecolors="white", zorder=5, label="Gender FP")
    ax.scatter([ov], [oa], s=55, marker="D", c="#616161", edgecolors="white", zorder=5, label="OASIS centroid")
    for cell in m2f["representative_cells"]:
        ax.scatter(
            [cell["v"]], [cell["a"]], s=140, c=colors["MtoF"],
            edgecolors="white", linewidths=0.9, zorder=4,
        )
        ax.text(
            cell["v"], cell["a"], str(cell["cell_id"]),
            ha="center", va="center", fontsize=7.5, color="white", fontweight="bold", zorder=6,
        )
        ax.annotate(
            cell.get("quadrant", ""),
            (cell["v"], cell["a"]),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=5.5, color="0.25",
        )
    gp = m2f.get("greedy_primary", {})
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(
        f"Minimal transfer set (M→F)\n"
        f"k={gp.get('k', len(m2f['representative_cells']))} cells; "
        f"{float(gp.get('frac_of_ceiling', float('nan'))):.0%} of ceiling R²; "
        f"{m2f['stability']['tag']}",
        fontsize=8.5,
    )
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    ax = axes[1]
    panel_label(ax, "B")
    _grid(ax)
    for dshort, prof in profiles.items():
        col = colors.get(dshort, "#333")
        for i, cell in enumerate(prof["representative_cells"]):
            ax.scatter(
                [cell["v"]], [cell["a"]], s=120, c=col, alpha=0.88,
                edgecolors="white", linewidths=0.7,
                label=dshort if i == 0 else None, zorder=4,
            )
            ax.text(
                cell["v"], cell["a"], str(cell["cell_id"]),
                ha="center", va="center", fontsize=6.5, color="white", zorder=6,
            )
    j = float(overlap.get("jaccard", float("nan")))
    shared = overlap.get("shared_cells", [])
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(
        f"Direction-specific anchors\nJaccard={j:.2f}; shared cells {shared}",
        fontsize=8.5,
    )
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    if n_panels >= 3 and sd_df is not None:
        ax = axes[2]
        panel_label(ax, "C")
        labs: list[str] = []
        for prof in profiles.values():
            for cell in prof["representative_cells"]:
                lab = cell.get("image_id_label")
                if lab:
                    labs.append(str(lab))
        labs = list(dict.fromkeys(labs))
        mask = sd_df["image_id"].astype(str).isin(labs)
        minimal = sd_df.loc[mask, "sd_pair"].to_numpy(float)
        rest = sd_df.loc[~mask, "sd_pair"].to_numpy(float)
        violin_strip(
            ax,
            {"Other images": rest, "Minimal anchors": minimal},
            colors=["#90a4ae", "#1565c0"],
            ylabel="Group SD proxy (sd_pair)",
            seed=0,
        )
        agr = m2f.get("agreement", {})
        ax.set_title(
            f"Agreement proxy (exploratory)\n"
            f"{agr.get('interpretation', 'n/a')}; "
            f"MWU p={float(agr.get('mannwhitney_p', float('nan'))):.2f}",
            fontsize=8.5,
        )

    fig.suptitle(
        "Supp. Fig. | Minimal VA coverage for gender-bridge transfer (Z4/Z5)\n"
        "Method detail for Fig.3-F: source-VA set-cover → held-out R² ≥ 90% ceiling",
        y=1.02, fontsize=10.5,
    )
    fig.tight_layout()
    return save_dual(fig, "Paper_SuppFig_minimal_anchor_coverage")


def _draw_z5cat_heatmap_panel(
    ax,
    block: dict,
    *,
    cats: list[str],
    row_order: list[str],
    dir_k: int,
    title: str,
    panel: str,
    vmin: float,
    vmax: float,
    cmap,
    norm,
) -> object:
    panel_label(ax, panel)
    mat = np.full((len(row_order), len(cats)), np.nan)
    for i, train_cat in enumerate(row_order):
        for j, test_cat in enumerate(cats):
            val = block["matrix_R2_mean"].get(train_cat, {}).get(test_cat)
            if val is not None and np.isfinite(val):
                mat[i, j] = float(val)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto", origin="upper")
    ax.set_xticks(range(len(cats)), cats, rotation=30, ha="right")
    ax.set_yticks(range(len(row_order)), row_order)
    ax.set_xlabel("Test category (held-out)")
    ax.set_ylabel(f"Train category (k={dir_k} anchors)")
    ax.set_title(
        f"{title}\n"
        f"k={dir_k}; ceiling={block['ceiling_R2_mean']:.3f}; target={block['target_R2']:.3f}",
        fontsize=8.8,
    )
    for i, train_cat in enumerate(row_order):
        row = block["train_rows"][train_cat]
        for j, _test_cat in enumerate(cats):
            val = mat[i, j]
            if not np.isfinite(val):
                continue
            txt_col = "white" if val < (vmin + 0.55 * (vmax - vmin)) else "0.15"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7.2, color=txt_col)
        overall = float(row["overall_test"]["R2_mean"])
        hit = "✓" if row["hit_90pct_ceiling"] else "×"
        ax.text(
            len(cats) - 0.35, i - 0.38, f"all={overall:.2f}{hit}",
            ha="right", va="top", fontsize=6.2, color="0.35",
        )
    unrestricted = block["train_rows"]["Unrestricted"]
    if unrestricted["anchor_categories"].get("all_same_category"):
        dom = unrestricted["anchor_categories"].get("dominant_category", "?")
        counts = unrestricted["anchor_categories"].get("category_counts", {})
        note = f"Unrestricted → all {dom} {counts}"
    else:
        note = "Unrestricted → mixed anchor categories"
    src = unrestricted.get("source", "")
    if src == "z5_profile":
        note += " (Fig.3-F anchors)"
    ax.text(0.0, -0.32, note, transform=ax.transAxes, fontsize=6.8, color="0.35")
    return im


# ── Supplement: legacy / technical ────────────────────────────────────
def paper_supp_cross_within() -> tuple[Path, Path]:
    """Supp. Fig. 1: cross−within δ t-maps only (M→F / F→M); no adjust panels."""
    from paper2_viz.figures.analyze_cross_within_bias_clusters import (
        VA_MAX,
        VA_MIN,
        overlay_cluster_regions,
    )

    npz_path = (
        RES
        / "cross_within_bias_cluster_analysis"
        / "maps_all_loto_cross_within_bias_cluster_overview_fusion.npz"
    )
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Missing {npz_path}. Run: "
            "python3 code/paper2_viz/figures/analyze_cross_within_bias_clusters.py "
            "--model fusion --subsets all_loto --save-maps-npz"
        )
    with np.load(npz_path) as z:
        g = np.asarray(z["grid"], float)
        step = float(z["grid_step"])
        panels = [
            (
                "a",
                "M→F (female target)",
                "female score (ref)",
                np.asarray(z["cross_bias_female_target__t_map"], float),
                np.asarray(z["cross_bias_female_target__labels"], int),
                np.asarray(z["cross_bias_female_target__labels_sig"], int),
                float(z["cross_bias_female_target__p_cluster"]),
                int(z["cross_bias_female_target__n_sig_clusters"]),
            ),
            (
                "b",
                "F→M (male target)",
                "male score (ref)",
                np.asarray(z["cross_bias_male_target__t_map"], float),
                np.asarray(z["cross_bias_male_target__labels"], int),
                np.asarray(z["cross_bias_male_target__labels_sig"], int),
                float(z["cross_bias_male_target__p_cluster"]),
                int(z["cross_bias_male_target__n_sig_clusters"]),
            ),
        ]

    extent = [g[0] - step / 2, g[-1] + step / 2, g[0] - step / 2, g[-1] + step / 2]
    vmax = 0.0
    for _lab, _title, _ref, t_map, _labels, _lsig, _p, _ns in panels:
        finite = t_map[np.isfinite(t_map)]
        if finite.size:
            vmax = max(vmax, float(np.nanpercentile(np.abs(finite), 99)))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), constrained_layout=True)
    last_im = None
    for ax, (lab, title, ref_label, t_map, labels, labels_sig, p_cl, n_sig) in zip(
        axes, panels
    ):
        panel_label(ax, lab.upper())
        last_im = ax.imshow(
            t_map,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="equal",
        )
        overlay_cluster_regions(ax, labels, labels_sig, extent)
        ax.set_title(
            f"{title}\n"
            f"global cluster p={p_cl:.3f}; sig. regions={n_sig}",
            fontsize=9,
        )
        ax.set_xlabel(f"Valence ({ref_label})")
        ax.set_ylabel(f"Arousal ({ref_label})")
        ax.set_xlim(VA_MIN, VA_MAX)
        ax.set_ylim(VA_MIN, VA_MAX)
        ax.set_xticks(np.arange(VA_MIN, VA_MAX + 0.01, 1.0))
        ax.set_yticks(np.arange(VA_MIN, VA_MAX + 0.01, 1.0))

    fig.colorbar(last_im, ax=list(axes), fraction=0.035, pad=0.02, label="paired t (δ)")
    fig.suptitle(
        "Supp. Fig. | Cross−versus−within bias field δ\n"
        "LOTO; cluster permutation (sign-flip); black outline = significant regions",
        fontsize=10.5,
    )
    return save_dual(fig, "Paper_SuppFig_cross_within_bias")


def write_readme(paths: list[tuple[str, Path, Path]]) -> None:
    lines = [
        "# Paper_fig — Nature Communications メイン Figure 出力",
        "",
        "仕様: `doc/NATCOMM_FIGURE_SPEC.md`",
        "生成: `python3 code/generate_paper_figures.py`",
        "",
        "各ファイルは **PNG** (300 dpi) と **SVG** の両形式。全パネルは JSON/CSV の一次データ",
        "から描画し、直値ハードコードは廃止（canon 準拠）。",
        "",
        "## 本文 6 図（NatComm 番号）",
        "",
        "| ファイル | Results | 主パネル |",
        "|----------|---------|----------|",
        "| Paper_Fig1_research_design | 全体 | 概念 A–D：凍結CLIP / 読み出し / 2境界 / 解析階層 |",
        "| Paper_Fig2_decoder_locus_lambda | R1 | A: strip+boot+label-null / B: slopegraph / C: CKA vs null / D: λ common+contour |",
        "| Paper_Fig3_linear_bridge_composition | R2–R3+Z3 | 5点 violin / Φ地形 / 固定点 vs 重心 / **汎化曲線 E / 最小被覆+3basin F / 有効解雲 G / 合成 H** |",
        "| Paper_Fig4_japan_transfer_reliability | R4（§3） | 予測vs実測 / R² vs 天井 / bootstrap CI（LOSO→Supp） |",
        "| Paper_Fig5_residual_geometry | R5–R6 | 残差 violin / excess twist map / **VA残差場トポロジー(M↔F)** |",
        "| Paper_Fig6_case_translation | R7 | **発見**: 等価実例（非Person）＋ Φ限界×翻訳有効の VA 局在 |",
        "",
        "## Supplementary figures generated here",
        "",
        "| ファイル | 内容 |",
        "|----------|------|",
        "| Paper_SuppFig_cross_within_bias | cross−within δ t-maps (M→F / F→M only) |",
        "| Paper_SuppFig_minimal_anchor_coverage | minimal coverage procedure |",
        "| Paper_SuppFig_matched_k_minimal_comparison | matched-k cell Jaccard |",
        "| Paper_SuppFig_person_bottleneck | Person generalisation bottleneck |",
        "| Paper_SuppFig_residual_field_topology | residual curl / static field |",
        "| Paper_SuppFig_z3_anchor_selection | reference-selection comparison |",
        "",
        "Additional Supplementary figures (five-point by dimension, Φ asymmetry, λ vs Φ,",
        "predictive translation, Φ residual vs equivalence distance) are produced by the",
        "dedicated scripts in this repository (`plot_paper_fig2_perspective_geometry.py`,",
        "`plot_testfig_phi_prediction_asymmetry_va_images.py`, `plot_suppfig_lambda_transform_vs_phi.py`,",
        "`plot_predictive_translation_prototype.py`, `plot_paper_fig345_manuscript.py`).",
        "",
        "## データソース",
        "",
        "- Z3: `results/anchor_structure/z3_generalization_curve_*.csv`、`z3_generalization.json`",
        "- Z4/Z5 最小被覆: `results/anchor_structure/z5_minimal_anchor_profile.json`",
        "- Fig2: `paper2_layer1_fig2_*.csv/json/npz`, `lambda_gender_diff_common_ref_fig2.npz`",
        "- OT/線形: `cvae_cross_gender/paper2_ot_five_point_fixedsplit.json`",
        "- 文化転移: `population_bridge_analysis/{reliability_ceiling,subject_bootstrap_ci,loso_stability}.json`",
        "- 残差: `population_bridge_analysis/residual_*.{json,csv}`",
        "- 等価: `equivalence_nontriviality_themecv.json` + `paper2_emotion_equivalent_pairs_themecv.csv`",
        "",
        "## 注記",
        "",
        "- 棒グラフ全廃。分布は violin/strip、CI は点+errorbar。",
        "- Fig3: quantity curve / minimal coverage / landmark density.",
        "- Fig4: Japan transfer and reliability ceiling.",
        "- Fig5/Fig6: residual geometry and exemplar translation.",
        "- Fig6: discovery panels; scalar residual×improvement ρ≈0.90 is not interpreted.",
        "- Fig6 旧独立性パネル（Φ残差×等価距離 ρ≈0.15）は Supp。",
        "- 固定点間距離は JSON 実値 **0.895**（2dp で 0.89）。registry「0.90」は丸め。",
        "- **2026-07-15 再編**: Fig5 新設（旧 Fig6 A–C）、Fig6 = 事例翻訳（旧 D–F）、Fig3-F = 最小被覆昇格。",
        "- **2026-07-14 リナンバー**: Z3→Fig3 E–F；Japan→Fig4；旧 `Paper_Fig4_shared_anchor_generalization` / `Paper_Fig5_japan_*` は廃止。",
        "",
        "## 生成ファイル",
        "",
    ]
    for stem, png, svg in paths:
        lines.append(f"- `{png.name}` / `{svg.name}`")
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    plt.rcParams.update(PAPER_RC)
    OUT.mkdir(parents=True, exist_ok=True)

    builders = [
        ("Fig1", paper_fig1),
        ("Fig2", paper_fig2),
        ("Fig3", paper_fig3),
        ("Fig4", paper_fig4),
        ("Fig5", paper_fig5_residual_geometry),
        ("Fig6", paper_fig6_case_translation),
        ("Supp cross-within", paper_supp_cross_within),
        ("Supp minimal anchor", paper_supp_minimal_anchor_coverage),
        ("Supp matched-k minimal", paper_supp_matched_k_minimal_comparison),
        ("Supp anchor point density", paper_supp_anchor_point_density_step0p1),
        ("Supp person bottleneck", paper_supp_person_bottleneck),
        ("Supp residual topology", paper_supp_residual_field_topology),
        ("Supp phi asymmetry", paper_supp_phi_prediction_asymmetry),
        ("Supp Z3 selection", paper_supp_z3_anchor_selection),
    ]
    saved: list[tuple[str, Path, Path]] = []
    for name, fn in builders:
        print(f"Generating {name}...")
        png, svg = fn()
        print(f"  {png.name} ({png.stat().st_size // 1024} KB)")
        print(f"  {svg.name} ({svg.stat().st_size // 1024} KB)")
        saved.append((name, png, svg))

    write_readme(saved)
    print(f"\nDone. Output: {OUT}")


if __name__ == "__main__":
    main()
