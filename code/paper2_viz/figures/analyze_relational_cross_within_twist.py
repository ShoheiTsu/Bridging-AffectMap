#!/usr/bin/env python3
"""Bridge relational score geometry (Aligned UMAP twist) with cross/within fusion predictions.

Links:
  - Score twist: male 4D vs female 4D (mean+SD)
  - Model twist: pm vs pf (cross vs within on each gender target)
  - Adjust twist: pm vs pm_adj (male background-ViT adjust on female target)
  - Cross-wins regions: err(cross) < err(within-native)
  - Resolution asymmetry: female_sd - male_sd on common VA grid
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from umap import aligned_umap

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from config import (  # noqa: E402
    FIG_INTEGRATED,
    OASIS_SCORES_CSV,
    RANDOM_SEED,
    RESULTS_GENDER,
    VALENCE_AROUSAL_SCALE_MAX,
    VALENCE_AROUSAL_SCALE_MIN,
)
from dataset import add_theme_base, load_oasis_meta  # noqa: E402

VA_MIN = VALENCE_AROUSAL_SCALE_MIN
VA_MAX = VALENCE_AROUSAL_SCALE_MAX
MIN_TEST_LOTO = 1  # pooled R² / SI maps include singleton themes (n=900)

MALE_4D = ["valence_male", "arousal_male", "valence_male_sd", "arousal_male_sd"]
FEMALE_4D = ["valence_female", "arousal_female", "valence_female_sd", "arousal_female_sd"]

_CLUSTER_PATH = Path(__file__).parent / "analyze_cross_within_bias_clusters.py"
_spec = importlib.util.spec_from_file_location("_cross_within_cluster", _CLUSTER_PATH)
assert _spec is not None and _spec.loader is not None
_cluster_mod = importlib.util.module_from_spec(_spec)
sys.modules["_cross_within_cluster"] = _cluster_mod
_spec.loader.exec_module(_cluster_mod)


def loto_indices(df: pd.DataFrame) -> list[int]:
    idx: list[int] = []
    for left_out in df["theme_base"].unique():
        mask = (df["theme_base"] == left_out).values
        if int(mask.sum()) >= MIN_TEST_LOTO:
            idx.extend(np.where(mask)[0].tolist())
    return idx


def load_merged(*, model: str) -> dict:
    """LOTO pooled scores + fusion predictions (base + adjust)."""
    oasis = load_oasis_meta(OASIS_SCORES_CSV)
    cols = (
        ["valence", "arousal", "category", "image_id"]
        + MALE_4D
        + FEMALE_4D
    )
    full = oasis[oasis[cols].notna().all(axis=1)].reset_index(drop=True)
    full = add_theme_base(full)
    loto_idx = loto_indices(full)
    sub = full.iloc[loto_idx].reset_index(drop=True)

    base = RESULTS_GENDER / "loto_nested"
    adj = RESULTS_GENDER / "fusion_background_vit" / "loto_nested"
    arr_idx = np.arange(len(sub))

    return {
        "df": sub,
        "v_c": sub["valence"].to_numpy(float),
        "a_c": sub["arousal"].to_numpy(float),
        "v_m": sub["valence_male"].to_numpy(float),
        "a_m": sub["arousal_male"].to_numpy(float),
        "v_f": sub["valence_female"].to_numpy(float),
        "a_f": sub["arousal_female"].to_numpy(float),
        "male_4d": sub[MALE_4D].to_numpy(float),
        "female_4d": sub[FEMALE_4D].to_numpy(float),
        "pm": np.load(base / "male" / f"y_all_pred_{model}.npy")[arr_idx],
        "pf": np.load(base / "female" / f"y_all_pred_{model}.npy")[arr_idx],
        "pm_adj": np.load(adj / "male" / "y_all_pred_fusion.npy")[arr_idx],
        "pf_adj": np.load(adj / "female" / "y_all_pred_fusion.npy")[arr_idx],
        "n": len(sub),
    }


def run_aligned_umap(
    x_a: np.ndarray,
    x_b: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    alignment_regularisation: float,
    n_epochs: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(x_a)
    xa = StandardScaler().fit_transform(x_a)
    xb = StandardScaler().fit_transform(x_b)
    relation = {i: i for i in range(n)}
    mapper = aligned_umap.AlignedUMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        alignment_regularisation=alignment_regularisation,
        random_state=RANDOM_SEED,
        n_epochs=n_epochs,
    )
    mapper.fit([xa, xb], relations=[relation])
    emb_a, emb_b = mapper.embeddings_
    return np.asarray(emb_a, float), np.asarray(emb_b, float)


def local_mean_grid(
    v_ref: np.ndarray,
    a_ref: np.ndarray,
    values: np.ndarray,
    g: np.ndarray,
    radius: float,
    min_n: int,
) -> np.ndarray:
    n_g = len(g)
    grid = np.full((n_g, n_g), np.nan)
    r2 = radius * radius
    for i in range(n_g):
        for j in range(n_g):
            m = (v_ref - g[j]) ** 2 + (a_ref - g[i]) ** 2 <= r2
            if int(m.sum()) >= min_n:
                grid[i, j] = float(np.nanmean(values[m]))
    return grid


def local_prop_grid(
    v_ref: np.ndarray,
    a_ref: np.ndarray,
    flag: np.ndarray,
    g: np.ndarray,
    radius: float,
    min_n: int,
) -> np.ndarray:
    n_g = len(g)
    grid = np.full((n_g, n_g), np.nan)
    r2 = radius * radius
    for i in range(n_g):
        for j in range(n_g):
            m = (v_ref - g[j]) ** 2 + (a_ref - g[i]) ** 2 <= r2
            if int(m.sum()) >= min_n:
                grid[i, j] = float(np.mean(flag[m]))
    return grid


def _draw_heat(ax, grid, g, step, *, title: str, ref_label: str, cmap: str, vlim: float | None = None):
    ext = [g[0] - step / 2, g[-1] + step / 2, g[0] - step / 2, g[-1] + step / 2]
    finite = grid[np.isfinite(grid)]
    vmax = vlim if vlim is not None else max(float(np.nanpercentile(finite, 95)) if finite.size else 1.0, 1e-6)
    vmin = 0.0 if cmap != "RdBu_r" else -vmax
    im = ax.imshow(grid, origin="lower", extent=ext, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlim(VA_MIN, VA_MAX)
    ax.set_ylim(VA_MIN, VA_MAX)
    ax.set_xlabel(f"Valence ({ref_label})")
    ax.set_ylabel(f"Arousal ({ref_label})")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.2)
    return im


def plot_umap_triplet(
    pairs: list[tuple[str, np.ndarray, np.ndarray]],
    out: Path,
    *,
    arrow_stride: int,
) -> dict[str, float]:
    """1×3 aligned UMAP panels with displacement arrows on the third-style inset per panel."""
    n_panels = len(pairs)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.6 * n_panels, 4.8))
    if n_panels == 1:
        axes = [axes]
    stats_out: dict[str, float] = {}

    for ax, (label, emb_a, emb_b) in zip(axes, pairs):
        disp = emb_b - emb_a
        disp_l2 = np.linalg.norm(disp, axis=1)
        vmax = float(np.percentile(disp_l2, 95))
        stats_out[f"mean_disp_{label}"] = float(disp_l2.mean())
        stats_out[f"median_disp_{label}"] = float(np.median(disp_l2))
        stats_out[f"p95_disp_{label}"] = vmax

        ax.scatter(
            emb_a[:, 0], emb_a[:, 1], s=12, alpha=0.35, c="#E07A2F", label="view A", edgecolors="none"
        )
        ax.scatter(
            emb_b[:, 0], emb_b[:, 1], s=12, alpha=0.35, c="#2A9D8F", label="view B", edgecolors="none"
        )
        idx = np.arange(len(emb_a))
        if arrow_stride > 1:
            idx = idx[::arrow_stride]
        top = np.argsort(disp_l2)[-min(25, len(disp_l2)) :]
        idx = np.unique(np.concatenate([idx, top]))
        ax.quiver(
            emb_a[idx, 0], emb_a[idx, 1], disp[idx, 0], disp[idx, 1],
            angles="xy", scale_units="xy", scale=1.0, width=0.003, alpha=0.5, color="0.15",
        )
        ax.set_title(f"{label}\nmean disp={disp_l2.mean():.3f}", fontsize=9)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(fontsize=6, loc="best")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Aligned UMAP twist: score geometry vs model vs adjust (LOTO pooled, fusion)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stats_out


def plot_va_bridge(
    data: dict,
    geo_disp: np.ndarray,
    g: np.ndarray,
    step: float,
    radius: float,
    min_n: int,
    out: Path,
    *,
    rng: np.random.Generator,
    n_perm: int,
    t_threshold: float,
    alpha: float,
) -> dict:
    v_f, a_f = data["v_f"], data["a_f"]
    v_m, a_m = data["v_m"], data["a_m"]
    pm, pf = data["pm"], data["pf"]
    pm_adj, pf_adj = data["pm_adj"], data["pf_adj"]

    err_ff = _cluster_mod.l2_rows(pf, v_f, a_f)
    err_mf = _cluster_mod.l2_rows(pm, v_f, a_f)
    err_mm = _cluster_mod.l2_rows(pm, v_m, a_m)
    err_fm = _cluster_mod.l2_rows(pf, v_m, a_m)
    err_mf_adj = _cluster_mod.l2_rows(pm_adj, v_f, a_f)
    err_fm_adj = _cluster_mod.l2_rows(pf_adj, v_m, a_m)

    model_twist_f = np.linalg.norm(pm - pf, axis=1)
    model_twist_m = np.linalg.norm(pf - pm, axis=1)
    cross_wins_f = (err_mf < err_ff).astype(float)
    cross_wins_m = (err_fm < err_mm).astype(float)
    adjust_help_f = err_mf - err_mf_adj
    adjust_help_m = err_fm - err_fm_adj

    d_sd_v = data["female_4d"][:, 2] - data["male_4d"][:, 2]
    d_sd_a = data["female_4d"][:, 3] - data["male_4d"][:, 3]

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.32)

    # Row 0: female target
    row0 = [
        ("Score UMAP disp (geo twist)", geo_disp, "magma", None),
        ("||pm − pf|| model twist", model_twist_f, "viridis", None),
        ("P(cross < within) M→F", cross_wins_f, "YlOrRd", 1.0),
        ("Adjust benefit (err↓)", adjust_help_f, "RdBu_r", None),
    ]
    ref_f = "female score (ref)"
    last_im = None
    for col, (ttl, vals, cmap, vlim) in enumerate(row0):
        ax = fig.add_subplot(gs[0, col])
        grid = local_mean_grid(v_f, a_f, vals, g, radius, min_n) if "P(" not in ttl else local_prop_grid(
            v_f, a_f, vals > 0.5, g, radius, min_n
        )
        last_im = _draw_heat(ax, grid, g, step, title=ttl, ref_label=ref_f, cmap=cmap, vlim=vlim)

    # Row 1: male target (symmetric)
    row1 = [
        ("Score UMAP disp", geo_disp, "magma", None),
        ("||pf − pm|| model twist", model_twist_m, "viridis", None),
        ("P(cross < within) F→M", cross_wins_m, "YlOrRd", 1.0),
        ("Adjust benefit (err↓)", adjust_help_m, "RdBu_r", None),
    ]
    ref_m = "male score (ref)"
    for col, (ttl, vals, cmap, vlim) in enumerate(row1):
        ax = fig.add_subplot(gs[1, col])
        grid = local_mean_grid(v_m, a_m, vals, g, radius, min_n) if "P(" not in ttl else local_prop_grid(
            v_m, a_m, vals > 0.5, g, radius, min_n
        )
        _draw_heat(ax, grid, g, step, title=ttl, ref_label=ref_m, cmap=cmap, vlim=vlim)

    # Row 2: resolution asymmetry + cluster cross-wins female
    ax_sd_v = fig.add_subplot(gs[2, 0])
    ax_sd_a = fig.add_subplot(gs[2, 1])
    grid_dv = local_mean_grid(data["v_c"], data["a_c"], d_sd_v, g, radius, min_n)
    grid_da = local_mean_grid(data["v_c"], data["a_c"], d_sd_a, g, radius, min_n)
    sd_vlim = max(abs(float(np.nanpercentile(grid_dv[np.isfinite(grid_dv)], 95))), 1e-6)
    _draw_heat(ax_sd_v, grid_dv, g, step, title="ΔSD Valence (female−male)", ref_label="OASIS common", cmap="RdBu_r", vlim=sd_vlim)
    _draw_heat(ax_sd_a, grid_da, g, step, title="ΔSD Arousal (female−male)", ref_label="OASIS common", cmap="RdBu_r", vlim=sd_vlim)

    ax_cl = fig.add_subplot(gs[2, 2:])
    delta_cross_f = err_mf - err_ff
    masks, n_map = _cluster_mod.build_cell_masks(v_f, a_f, g, radius, min_n)
    cl_res = _cluster_mod.permutation_cluster_correction(
        delta_cross_f, masks, n_map, g, min_n, rng,
        n_perm=n_perm, t_threshold=t_threshold, alpha=alpha,
    )
    if cl_res.get("ok"):
        mean_map = cl_res["mean_map"]
        sig = cl_res["labels_sig"]
        ext = [g[0] - step / 2, g[-1] + step / 2, g[0] - step / 2, g[-1] + step / 2]
        vlim = max(float(np.nanpercentile(np.abs(mean_map[np.isfinite(mean_map)]), 95)), 1e-6)
        ax_cl.imshow(mean_map, origin="lower", extent=ext, aspect="equal", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        _cluster_mod.overlay_cluster_regions(ax_cl, cl_res["labels"], sig, ext)
        n_sig = int(cl_res.get("n_sig_clusters", 0))
        p_glob = cl_res.get("p_cluster", np.nan)
        ax_cl.set_title(
            f"Cross−within error (M→F); cluster perm\n"
            f"global p={p_glob:.4f}, n_sig={n_sig} (blue=cross wins)",
            fontsize=9,
        )
    else:
        ax_cl.set_title("Cross−within cluster (insufficient n)")
    ax_cl.set_xlim(VA_MIN, VA_MAX)
    ax_cl.set_ylim(VA_MIN, VA_MAX)
    ax_cl.set_xlabel(f"Valence ({ref_f})")
    ax_cl.set_ylabel(f"Arousal ({ref_f})")
    ax_cl.grid(True, alpha=0.2)

    fig.suptitle(
        "Relational twist ↔ cross/within bridge (LOTO fusion, grid local means)\n"
        "Row1–2: female / male targets; bottom: resolution asymmetry + cross-wins clusters",
        fontsize=11,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "pct_cross_wins_female_target": float(cross_wins_f.mean()),
        "pct_cross_wins_male_target": float(cross_wins_m.mean()),
        "mean_adjust_benefit_female": float(np.mean(adjust_help_f)),
        "mean_adjust_benefit_male": float(np.mean(adjust_help_m)),
        "corr_geo_model_twist_female": float(np.corrcoef(geo_disp, model_twist_f)[0, 1]),
        "corr_geo_cross_bias_female": float(np.corrcoef(geo_disp, err_mf - err_ff)[0, 1]),
        "corr_model_twist_cross_bias_female": float(np.corrcoef(model_twist_f, err_mf - err_ff)[0, 1]),
        "cluster_cross_female_global_p": float(cl_res.get("p_cluster", np.nan)) if cl_res.get("ok") else np.nan,
        "cluster_cross_female_n_sig": int(cl_res.get("n_sig_clusters", 0)) if cl_res.get("ok") else 0,
    }


def local_vec_grid(
    v_ref: np.ndarray,
    a_ref: np.ndarray,
    vec: np.ndarray,
    g: np.ndarray,
    radius: float,
    min_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    masks, n_map = _cluster_mod.build_cell_masks(v_ref, a_ref, g, radius, min_n)
    return _cluster_mod.local_vec_mean_masked(vec, masks, n_map, g, min_n)


def _extent(g: np.ndarray, step: float) -> list[float]:
    return [g[0] - step / 2, g[-1] + step / 2, g[0] - step / 2, g[-1] + step / 2]


def _cluster_high_twist(
    values: np.ndarray,
    v_ref: np.ndarray,
    a_ref: np.ndarray,
    g: np.ndarray,
    radius: float,
    min_n: int,
    rng: np.random.Generator,
    *,
    n_perm: int,
    t_threshold: float,
    alpha: float,
) -> dict:
    """Sign-flip cluster test on centered twist (regions above global mean)."""
    centered = values - float(np.mean(values))
    masks, n_map = _cluster_mod.build_cell_masks(v_ref, a_ref, g, radius, min_n)
    return _cluster_mod.permutation_cluster_correction(
        centered, masks, n_map, g, min_n, rng,
        n_perm=n_perm, t_threshold=t_threshold, alpha=alpha,
    )


def _draw_cluster_heat(
    ax,
    mean_grid: np.ndarray,
    cl_res: dict,
    g: np.ndarray,
    step: float,
    *,
    title: str,
    ref_label: str,
    cmap: str = "magma",
    signed: bool = False,
) -> None:
    ext = _extent(g, step)
    finite = mean_grid[np.isfinite(mean_grid)]
    if signed:
        vlim = max(float(np.nanpercentile(np.abs(finite), 95)) if finite.size else 1.0, 1e-6)
        vmin, vmax = -vlim, vlim
        cmap = "RdBu_r"
    else:
        vmin, vmax = 0.0, max(float(np.nanpercentile(finite, 95)) if finite.size else 1.0, 1e-6)
    ax.imshow(mean_grid, origin="lower", extent=ext, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
    if cl_res.get("ok"):
        _cluster_mod.overlay_cluster_regions(ax, cl_res["labels"], cl_res["labels_sig"], ext)
        p_c = cl_res.get("p_cluster", np.nan)
        n_sig = int(cl_res.get("n_sig_clusters", 0))
        title = f"{title}\ncluster p={p_c:.4f}, n_sig={n_sig}"
    ax.set_xlim(VA_MIN, VA_MAX)
    ax.set_ylim(VA_MIN, VA_MAX)
    ax.set_xlabel(f"Valence ({ref_label})")
    ax.set_ylabel(f"Arousal ({ref_label})")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.2)


def _draw_quiver_twist(
    ax,
    g: np.ndarray,
    step: float,
    dx: np.ndarray,
    dy: np.ndarray,
    mag: np.ndarray,
    cnt: np.ndarray,
    min_n: int,
    *,
    title: str,
    ref_label: str,
) -> None:
    ext = _extent(g, step)
    finite = mag[np.isfinite(mag)]
    vmax = max(float(np.nanpercentile(finite, 95)) if finite.size else 1.0, 1e-6)
    ax.imshow(mag, origin="lower", extent=ext, aspect="equal", cmap="magma", vmin=0, vmax=vmax, alpha=0.85)
    xg, yg = np.meshgrid(g, g, indexing="xy")
    mask = (cnt >= min_n) & np.isfinite(dx) & np.isfinite(dy)
    qscale = 1.0 / (step * 2.8)
    ax.quiver(
        xg[mask], yg[mask],
        np.nan_to_num(dx[mask]), np.nan_to_num(dy[mask]),
        angles="xy", scale_units="xy", scale=qscale,
        width=0.004, headwidth=3, headlength=4, color="0.15",
    )
    ax.set_xlim(VA_MIN, VA_MAX)
    ax.set_ylim(VA_MIN, VA_MAX)
    ax.set_xlabel(f"Valence ({ref_label})")
    ax.set_ylabel(f"Arousal ({ref_label})")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.2)


def plot_twist_va_local(
    data: dict,
    geo_disp: np.ndarray,
    model_twist: np.ndarray,
    adjust_twist: np.ndarray,
    g: np.ndarray,
    step: float,
    radius: float,
    min_n: int,
    out: Path,
    *,
    rng: np.random.Generator,
    n_perm: int,
    t_threshold: float,
    alpha: float,
) -> dict:
    """Local VA maps (r=radius): twist magnitude + cluster + gap vectors."""
    refs = (
        ("common", data["v_c"], data["a_c"], "OASIS common"),
        ("female", data["v_f"], data["a_f"], "female score"),
        ("male", data["v_m"], data["a_m"], "male score"),
    )
    score_vec = np.column_stack([data["v_f"] - data["v_m"], data["a_f"] - data["a_m"]])
    pred_vec = data["pm"] - data["pf"]
    excess = geo_disp - model_twist

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12.5))
    local_stats: dict = {}
    grid_rows: list[dict] = []

    twist_specs = (
        ("geo_score_umap", geo_disp, "Score UMAP twist"),
        ("model_pm_pf", model_twist, "Model twist ||pm−pf||"),
        ("adjust_pm", adjust_twist, "Adjust twist ||pm−pm_adj||"),
    )

    for col, (ref_key, v_ref, a_ref, ref_label) in enumerate(refs):
        for row, (metric_key, values, label) in enumerate(twist_specs):
            ax = axes[row, col]
            grid = local_mean_grid(v_ref, a_ref, values, g, radius, min_n)
            cl = _cluster_high_twist(
                values, v_ref, a_ref, g, radius, min_n, rng,
                n_perm=n_perm, t_threshold=t_threshold, alpha=alpha,
            )
            _draw_cluster_heat(ax, grid, cl, g, step, title=label, ref_label=ref_label)
            local_stats[f"{ref_key}_{metric_key}_cluster_p"] = float(cl.get("p_cluster", np.nan)) if cl.get("ok") else np.nan
            local_stats[f"{ref_key}_{metric_key}_n_sig"] = int(cl.get("n_sig_clusters", 0)) if cl.get("ok") else 0

            if ref_key == "common":
                masks, n_map = _cluster_mod.build_cell_masks(v_ref, a_ref, g, radius, min_n)
                k = 0
                for i, gv in enumerate(g):
                    for j, ga in enumerate(g):
                        n = int(n_map[i, j])
                        if n >= min_n and np.isfinite(grid[i, j]):
                            row_id = f"{metric_key}_{i}_{j}"
                            if not any(r.get("_key") == row_id for r in grid_rows):
                                grid_rows.append({
                                    "_key": row_id,
                                    "metric": metric_key,
                                    "grid_v": float(gv),
                                    "grid_a": float(ga),
                                    "n_images": n,
                                    "local_mean": float(grid[i, j]),
                                })

    fig.suptitle(
        f"Local VA twist (LOTO fusion, grid={step}, r={radius}, min_n={min_n})\n"
        "rows: score / model / adjust twist; cols: common / female / male VA anchor; gold=high-twist clusters",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Vector + excess twist on common VA
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.6))
    v_c, a_c = data["v_c"], data["a_c"]
    dx_s, dy_s, mag_s, cnt_s = local_vec_grid(v_c, a_c, score_vec, g, radius, min_n)
    dx_p, dy_p, mag_p, cnt_p = local_vec_grid(v_c, a_c, pred_vec, g, radius, min_n)
    excess_grid = local_mean_grid(v_c, a_c, excess, g, radius, min_n)
    cl_ex = _cluster_high_twist(
        excess, v_c, a_c, g, radius, min_n, rng,
        n_perm=n_perm, t_threshold=t_threshold, alpha=alpha,
    )

    _draw_quiver_twist(
        axes2[0], g, step, dx_s, dy_s, mag_s, cnt_s, min_n,
        title="Score gap vector (female−male VA)\nbg = ||Δscore||",
        ref_label="OASIS common",
    )
    _draw_quiver_twist(
        axes2[1], g, step, dx_p, dy_p, mag_p, cnt_p, min_n,
        title="Pred twist vector (pm−pf)\nbg = ||Δpred||",
        ref_label="OASIS common",
    )
    _draw_cluster_heat(
        axes2[2], excess_grid, cl_ex, g, step,
        title="Excess twist (UMAP − model)", ref_label="OASIS common", signed=True,
    )
    fig2.suptitle("Local VA: gap vectors & score-minus-model excess twist (common anchor)", fontsize=11)
    fig2.tight_layout(rect=(0, 0, 1, 0.93))
    vec_out = out.parent / "Figure_relational_twist_va_local_vectors.png"
    fig2.savefig(vec_out, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    local_stats["excess_twist_cluster_p"] = float(cl_ex.get("p_cluster", np.nan)) if cl_ex.get("ok") else np.nan
    local_stats["excess_twist_n_sig"] = int(cl_ex.get("n_sig_clusters", 0)) if cl_ex.get("ok") else 0
    local_stats["vector_figure"] = str(vec_out)

    return local_stats, grid_rows, vec_out


def export_twist_grid_csv(
    data: dict,
    geo_disp: np.ndarray,
    model_twist: np.ndarray,
    adjust_twist: np.ndarray,
    g: np.ndarray,
    radius: float,
    min_n: int,
    out: Path,
) -> None:
    """Per-cell local twist table on common VA."""
    v_c, a_c = data["v_c"], data["a_c"]
    masks, n_map = _cluster_mod.build_cell_masks(v_c, a_c, g, radius, min_n)
    geo_g = local_mean_grid(v_c, a_c, geo_disp, g, radius, min_n)
    mod_g = local_mean_grid(v_c, a_c, model_twist, g, radius, min_n)
    adj_g = local_mean_grid(v_c, a_c, adjust_twist, g, radius, min_n)
    exc_g = geo_g - mod_g
    dx_s, dy_s, mag_s, _ = local_vec_grid(
        v_c, a_c, np.column_stack([data["v_f"] - data["v_m"], data["a_f"] - data["a_m"]]), g, radius, min_n
    )
    dx_p, dy_p, mag_p, _ = local_vec_grid(v_c, a_c, data["pm"] - data["pf"], g, radius, min_n)

    rows: list[dict] = []
    for i, ga in enumerate(g):
        for j, gv in enumerate(g):
            if n_map[i, j] < min_n:
                continue
            if not np.isfinite(geo_g[i, j]):
                continue
            rows.append({
                "grid_valence": float(gv),
                "grid_arousal": float(ga),
                "n_images": int(n_map[i, j]),
                "geo_umap_twist_mean": float(geo_g[i, j]),
                "model_twist_mean": float(mod_g[i, j]),
                "adjust_twist_mean": float(adj_g[i, j]),
                "excess_twist_mean": float(exc_g[i, j]),
                "score_gap_dv_mean": float(dx_s[i, j]),
                "score_gap_da_mean": float(dy_s[i, j]),
                "score_gap_mag_mean": float(mag_s[i, j]),
                "pred_twist_dv_mean": float(dx_p[i, j]),
                "pred_twist_da_mean": float(dy_p[i, j]),
                "pred_twist_mag_mean": float(mag_p[i, j]),
            })
    pd.DataFrame(rows).to_csv(out, index=False)


def plot_scatter_bridge(
    geo_disp: np.ndarray,
    model_twist: np.ndarray,
    cross_bias: np.ndarray,
    adjust_help: np.ndarray,
    cross_wins: np.ndarray,
    out: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    specs = [
        ("geo UMAP disp", "||pm−pf||", geo_disp, model_twist),
        ("geo UMAP disp", "cross−within err", geo_disp, cross_bias),
        ("model twist", "adjust benefit", model_twist, adjust_help),
    ]
    for ax, (xl, yl, x, y) in zip(axes, specs):
        ax.scatter(x, y, s=10, alpha=0.35, c=cross_wins, cmap="coolwarm", vmin=0, vmax=1, edgecolors="none")
        r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"r={r:.3f}  (color=cross wins)")
        ax.grid(True, alpha=0.2)
    fig.suptitle("Per-image links: score twist vs model twist vs cross/within (female target)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Relational twist × cross/within bridge analysis.")
    ap.add_argument("--model", default="fusion")
    ap.add_argument("--grid-step", type=float, default=0.1)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--t-threshold", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--umap-neighbors", type=int, default=20)
    ap.add_argument("--umap-min-dist", type=float, default=0.15)
    ap.add_argument("--umap-align-reg", type=float, default=0.1)
    ap.add_argument("--umap-epochs", type=int, default=400)
    ap.add_argument("--umap-arrow-stride", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_GENDER / "relational_cross_within_twist")
    ap.add_argument("--fig-dir", type=Path, default=FIG_INTEGRATED)
    args = ap.parse_args()

    data = load_merged(model=args.model)
    g = _cluster_mod.make_grid(args.grid_step)
    rng = np.random.default_rng(RANDOM_SEED)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"LOTO pooled n={data['n']}")
    print("Aligned UMAP: score / model / adjust...")

    emb_score_m, emb_score_f = run_aligned_umap(
        data["male_4d"], data["female_4d"],
        n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist,
        alignment_regularisation=args.umap_align_reg, n_epochs=args.umap_epochs,
    )
    geo_disp = np.linalg.norm(emb_score_f - emb_score_m, axis=1)

    emb_pm, emb_pf = run_aligned_umap(
        data["pm"], data["pf"],
        n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist,
        alignment_regularisation=args.umap_align_reg, n_epochs=args.umap_epochs,
    )
    emb_pm0, emb_pm_adj = run_aligned_umap(
        data["pm"], data["pm_adj"],
        n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist,
        alignment_regularisation=args.umap_align_reg, n_epochs=args.umap_epochs,
    )

    umap_stats = plot_umap_triplet(
        [
            ("Score 4D (male vs female)", emb_score_m, emb_score_f),
            ("Pred female target (pm vs pf)", emb_pm, emb_pf),
            ("Adjust (pm vs pm_adj)", emb_pm0, emb_pm_adj),
        ],
        args.fig_dir / "Figure_relational_twist_score_vs_model_umap.png",
        arrow_stride=args.umap_arrow_stride,
    )

    err_ff = _cluster_mod.l2_rows(data["pf"], data["v_f"], data["a_f"])
    err_mf = _cluster_mod.l2_rows(data["pm"], data["v_f"], data["a_f"])
    err_mf_adj = _cluster_mod.l2_rows(data["pm_adj"], data["v_f"], data["a_f"])
    model_twist_f = np.linalg.norm(data["pm"] - data["pf"], axis=1)
    cross_bias_f = err_mf - err_ff
    adjust_help_f = err_mf - err_mf_adj
    cross_wins_f = (err_mf < err_ff).astype(float)

    adjust_twist = np.linalg.norm(data["pm_adj"] - data["pm"], axis=1)

    bridge_stats = plot_va_bridge(
        data, geo_disp, g, args.grid_step, args.radius, args.min_n,
        args.fig_dir / "Figure_relational_twist_va_bridge.png",
        rng=rng, n_perm=args.n_perm, t_threshold=args.t_threshold, alpha=args.alpha,
    )

    plot_scatter_bridge(
        geo_disp, model_twist_f, cross_bias_f, adjust_help_f, cross_wins_f,
        args.fig_dir / "Figure_relational_twist_scatter_bridge.png",
    )

    local_maps_path = args.fig_dir / "Figure_relational_twist_va_local_maps.png"
    local_stats, _, vec_out = plot_twist_va_local(
        data, geo_disp, model_twist_f, adjust_twist,
        g, args.grid_step, args.radius, args.min_n,
        local_maps_path,
        rng=rng, n_perm=args.n_perm, t_threshold=args.t_threshold, alpha=args.alpha,
    )
    export_twist_grid_csv(
        data, geo_disp, model_twist_f, adjust_twist,
        g, args.radius, args.min_n,
        args.out_dir / "twist_va_grid_common.csv",
    )

    tab = data["df"][["image_id", "category", "valence", "arousal"] + MALE_4D + FEMALE_4D].copy()
    tab["geo_umap_disp"] = geo_disp
    tab["model_twist_pm_pf"] = model_twist_f
    tab["adjust_twist_pm"] = adjust_twist
    tab["cross_bias_err_mf_minus_ff"] = cross_bias_f
    tab["adjust_benefit_mf"] = adjust_help_f
    tab["cross_wins_mf"] = cross_wins_f.astype(bool)
    tab["umap_score_male_x"] = emb_score_m[:, 0]
    tab["umap_score_male_y"] = emb_score_m[:, 1]
    tab["umap_score_female_x"] = emb_score_f[:, 0]
    tab["umap_score_female_y"] = emb_score_f[:, 1]
    tab["umap_pred_pm_x"] = emb_pm[:, 0]
    tab["umap_pred_pm_y"] = emb_pm[:, 1]
    tab["umap_pred_pf_x"] = emb_pf[:, 0]
    tab["umap_pred_pf_y"] = emb_pf[:, 1]
    tab.to_csv(args.out_dir / "relational_twist_per_image.csv", index=False)

    summary = {
        "n_loto": data["n"],
        "model": args.model,
        "aligned_umap": umap_stats,
        "bridge": bridge_stats,
        "va_local": local_stats,
        "figures": {
            "umap_triplet": str(args.fig_dir / "Figure_relational_twist_score_vs_model_umap.png"),
            "va_bridge": str(args.fig_dir / "Figure_relational_twist_va_bridge.png"),
            "scatter_bridge": str(args.fig_dir / "Figure_relational_twist_scatter_bridge.png"),
            "va_local_maps": str(local_maps_path),
            "va_local_vectors": str(vec_out),
        },
    }
    with open(args.out_dir / "relational_twist_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Score twist mean={umap_stats['mean_disp_Score 4D (male vs female)']:.3f}")
    print(f"Model twist mean={umap_stats['mean_disp_Pred female target (pm vs pf)']:.3f}")
    print(f"Adjust twist mean={umap_stats['mean_disp_Adjust (pm vs pm_adj)']:.3f}")
    print(f"Cross wins (M→F): {bridge_stats['pct_cross_wins_female_target']*100:.1f}%")
    print(f"corr(geo, model twist)={bridge_stats['corr_geo_model_twist_female']:.3f}")
    print(f"corr(geo, cross bias)={bridge_stats['corr_geo_cross_bias_female']:.3f}")
    for k, v in summary["figures"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
