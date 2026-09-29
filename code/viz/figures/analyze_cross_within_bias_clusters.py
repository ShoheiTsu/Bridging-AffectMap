#!/usr/bin/env python3
"""Cluster permutation on LOTO cross−within bias and adjust-delta VA anchor maps."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import binary_erosion, label as ndi_label

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from config import (  # noqa: E402
    FIG_INTEGRATED,
    OASIS_SCORES_CSV,
    RESULTS_GENDER,
    VALENCE_AROUSAL_SCALE_MAX,
    VALENCE_AROUSAL_SCALE_MIN,
)
from dataset import add_theme_base, load_oasis_meta  # noqa: E402

VA_MIN = VALENCE_AROUSAL_SCALE_MIN
VA_MAX = VALENCE_AROUSAL_SCALE_MAX
MIN_TEST_LOTO = 1  # pooled R² / SI maps include singleton themes (n=900)
RANDOM_SEED = 42

REF_LABELS = {"male": "male score (ref)", "female": "female score (ref)"}


@dataclass(frozen=True)
class MapSpec:
    key: str
    title: str
    ref_gender: str
    kind: str  # "cross_within" | "adjust"


MAP_SPECS: tuple[MapSpec, ...] = (
    MapSpec(
        "cross_bias_female_target",
        "Male→female minus female→female (L2 error)",
        "female",
        "cross_within",
    ),
    MapSpec(
        "cross_bias_male_target",
        "Female→male minus male→male (L2 error)",
        "male",
        "cross_within",
    ),
    MapSpec(
        "adjust_delta_female_target",
        "Male adjust→female minus male→female (L2 error)",
        "female",
        "adjust",
    ),
    MapSpec(
        "adjust_delta_male_target",
        "Female adjust→male minus female→male (L2 error)",
        "male",
        "adjust",
    ),
)


def make_grid(step: float) -> np.ndarray:
    return np.arange(VA_MIN + step / 2, VA_MAX, step)


def loto_indices(df: pd.DataFrame) -> list[int]:
    idx: list[int] = []
    for left_out in df["theme_base"].unique():
        mask = (df["theme_base"] == left_out).values
        if int(mask.sum()) >= MIN_TEST_LOTO:
            idx.extend(np.where(mask)[0].tolist())
    return idx


def l2_rows(pred: np.ndarray, true_v: np.ndarray, true_a: np.ndarray) -> np.ndarray:
    return np.sqrt((pred[:, 0] - true_v) ** 2 + (pred[:, 1] - true_a) ** 2)


def load_pool(*, model: str, category: str | None) -> dict:
    oasis = load_oasis_meta(OASIS_SCORES_CSV)
    cols = ["valence_male", "arousal_male", "valence_female", "arousal_female", "category"]
    full = oasis[oasis[cols].notna().all(axis=1)].reset_index(drop=True)
    full = add_theme_base(full)
    loto_idx = loto_indices(full)
    loto_sub = full.iloc[loto_idx].reset_index(drop=True)
    if category is not None:
        loto_sub = loto_sub[loto_sub["category"] == category].reset_index(drop=True)
        arr_idx = np.where(full.iloc[loto_idx]["category"].values == category)[0]
    else:
        arr_idx = np.arange(len(loto_sub))

    base = RESULTS_GENDER / "loto_nested"
    adj = RESULTS_GENDER / "fusion_background_vit" / "loto_nested"
    return {
        "n": len(loto_sub),
        "v_m": loto_sub["valence_male"].to_numpy(float),
        "a_m": loto_sub["arousal_male"].to_numpy(float),
        "v_f": loto_sub["valence_female"].to_numpy(float),
        "a_f": loto_sub["arousal_female"].to_numpy(float),
        "pm": np.load(base / "male" / f"y_all_pred_{model}.npy")[arr_idx],
        "pf": np.load(base / "female" / f"y_all_pred_{model}.npy")[arr_idx],
        "pm_adj": np.load(adj / "male" / "y_all_pred_fusion.npy")[arr_idx],
        "pf_adj": np.load(adj / "female" / "y_all_pred_fusion.npy")[arr_idx],
    }


def comparison_arrays(
    data: dict, spec: MapSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return v_ref, a_ref, err_a, err_b, vec_delta (2D bias / adjust vector per image)."""
    pm, pf = data["pm"], data["pf"]
    pm_adj, pf_adj = data["pm_adj"], data["pf_adj"]
    v_m, a_m = data["v_m"], data["a_m"]
    v_f, a_f = data["v_f"], data["a_f"]

    if spec.key == "cross_bias_female_target":
        v_ref, a_ref = v_f, a_f
        err_a = l2_rows(pm, v_f, a_f)
        err_b = l2_rows(pf, v_f, a_f)
        vec = pm - pf
    elif spec.key == "cross_bias_male_target":
        v_ref, a_ref = v_m, a_m
        err_a = l2_rows(pf, v_m, a_m)
        err_b = l2_rows(pm, v_m, a_m)
        vec = pf - pm
    elif spec.key == "adjust_delta_female_target":
        v_ref, a_ref = v_f, a_f
        err_a = l2_rows(pm_adj, v_f, a_f)
        err_b = l2_rows(pm, v_f, a_f)
        vec = pm_adj - pm
    elif spec.key == "adjust_delta_male_target":
        v_ref, a_ref = v_m, a_m
        err_a = l2_rows(pf_adj, v_m, a_m)
        err_b = l2_rows(pf, v_m, a_m)
        vec = pf_adj - pf
    else:
        raise KeyError(spec.key)
    return v_ref, a_ref, err_a, err_b, vec


def build_cell_masks(
    v_ref: np.ndarray,
    a_ref: np.ndarray,
    g: np.ndarray,
    radius: float,
    min_n: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Precompute image indices per grid cell (reused across permutations)."""
    n_g = len(g)
    r2 = radius * radius
    masks: list[np.ndarray] = []
    n_map = np.zeros((n_g, n_g), dtype=int)
    for i in range(n_g):
        for j in range(n_g):
            vc, ac = g[j], g[i]
            idx = np.flatnonzero((v_ref - vc) ** 2 + (a_ref - ac) ** 2 <= r2)
            masks.append(idx)
            n_map[i, j] = len(idx)
    return masks, n_map


class CellCache:
    """Padded per-cell delta slices for fast sign-flip permutation t-maps."""

    def __init__(
        self,
        delta: np.ndarray,
        masks: list[np.ndarray],
        g: np.ndarray,
        min_n: int,
    ) -> None:
        n_g = len(g)
        self.n_g = n_g
        self.delta = delta
        self.pos_i: list[int] = []
        self.pos_j: list[int] = []
        idx_rows: list[np.ndarray] = []
        for k, idx in enumerate(masks):
            if len(idx) < min_n:
                continue
            self.pos_i.append(k // n_g)
            self.pos_j.append(k % n_g)
            idx_rows.append(idx)
        self.n_cells = len(idx_rows)
        if self.n_cells == 0:
            self.idx_pad = np.empty((0, 0), dtype=np.int32)
            return
        maxlen = max(len(r) for r in idx_rows)
        self.idx_pad = np.full((self.n_cells, maxlen), -1, dtype=np.int32)
        for i, idx in enumerate(idx_rows):
            self.idx_pad[i, : len(idx)] = idx

    def _flat_stats(self, flip: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vals = self.delta[self.idx_pad] * flip[self.idx_pad]
        vals = np.where(self.idx_pad >= 0, vals, np.nan)
        n_eff = np.sum(np.isfinite(vals), axis=1)
        means = np.nanmean(vals, axis=1)
        stds = np.nanstd(vals, axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_flat = means / (stds / np.sqrt(n_eff))
        t_flat = np.where(
            (n_eff >= 2) & (stds > 0),
            t_flat,
            np.where(n_eff >= 2, 0.0, np.nan),
        )
        return means, t_flat, n_eff

    def scatter_maps(
        self, means: np.ndarray, t_flat: np.ndarray, n_eff: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_g = self.n_g
        mean_map = np.full((n_g, n_g), np.nan)
        t_map = np.full((n_g, n_g), np.nan)
        p_map = np.full((n_g, n_g), np.nan)
        for k in range(self.n_cells):
            i, j = self.pos_i[k], self.pos_j[k]
            mean_map[i, j] = float(means[k])
            t_map[i, j] = float(t_flat[k])
            if n_eff[k] >= 2 and np.isfinite(t_flat[k]):
                p_map[i, j] = float(2.0 * stats.t.sf(abs(t_flat[k]), int(n_eff[k]) - 1))
        return mean_map, t_map, p_map

    def observed_maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means, t_flat, n_eff = self._flat_stats(np.ones_like(self.delta))
        return self.scatter_maps(means, t_flat, n_eff)

    def t_map_from_flip(self, flip: np.ndarray) -> np.ndarray:
        _, t_flat, n_eff = self._flat_stats(flip)
        n_g = self.n_g
        t_map = np.full((n_g, n_g), np.nan)
        for k in range(self.n_cells):
            if n_eff[k] >= 2:
                t_map[self.pos_i[k], self.pos_j[k]] = float(t_flat[k])
        return t_map


def compute_grid_stats_masked(
    delta: np.ndarray,
    masks: list[np.ndarray],
    g: np.ndarray,
    min_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = CellCache(delta, masks, g, min_n)
    return cache.observed_maps()


def local_vec_mean_masked(
    vec: np.ndarray,
    masks: list[np.ndarray],
    n_map: np.ndarray,
    g: np.ndarray,
    min_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_g = len(g)
    dx = np.full((n_g, n_g), np.nan)
    dy = np.full((n_g, n_g), np.nan)
    mag = np.full((n_g, n_g), np.nan)
    cnt = n_map.copy()
    norms = np.linalg.norm(vec, axis=1)
    k = 0
    for i in range(n_g):
        for j in range(n_g):
            idx = masks[k]
            k += 1
            if len(idx) < min_n:
                continue
            dx[i, j] = float(np.mean(vec[idx, 0]))
            dy[i, j] = float(np.mean(vec[idx, 1]))
            mag[i, j] = float(np.mean(norms[idx]))
    return dx, dy, mag, cnt


def find_clusters(t_map: np.ndarray, *, t_threshold: float) -> tuple[np.ndarray, int, list[tuple[int, float]]]:
    binary = np.isfinite(t_map) & (np.abs(t_map) > t_threshold)
    labels, n_clusters = ndi_label(binary.astype(float), structure=np.ones((3, 3)))
    masses: list[tuple[int, float]] = []
    for k in range(1, n_clusters + 1):
        masses.append((k, float(np.nansum(t_map[labels == k]))))
    return labels, n_clusters, masses


def cluster_level_pvalues(
    masses_obs: list[tuple[int, float]],
    max_masses_perm: list[float],
) -> dict[int, float]:
    """Per-cluster p: fraction of permutations with max cluster mass >= |observed mass|."""
    if not masses_obs:
        return {}
    perm = np.asarray(max_masses_perm, dtype=float)
    return {
        cid: float(np.mean(perm >= abs(mass)))
        for cid, mass in masses_obs
    }


def describe_clusters(
    labels: np.ndarray,
    masses_obs: list[tuple[int, float]],
    p_level: dict[int, float],
    mean_map: np.ndarray,
    g: np.ndarray,
    *,
    alpha: float,
) -> list[dict]:
    rows: list[dict] = []
    for cid, mass in masses_obs:
        mask = labels == cid
        if not np.any(mask):
            continue
        ij = np.argwhere(mask)
        vs = np.array([g[j] for i, j in ij], dtype=float)
        a_s = np.array([g[i] for i, j in ij], dtype=float)
        mean_d = mean_map[mask]
        p_c = p_level.get(cid, 1.0)
        rows.append(
            {
                "cluster_id": int(cid),
                "n_cells": int(mask.sum()),
                "cluster_mass": float(mass),
                "p_cluster_global": p_c,
                "significant": bool(p_c <= alpha),
                "centroid_valence": float(np.mean(vs)),
                "centroid_arousal": float(np.mean(a_s)),
                "mean_delta_l2": float(np.nanmean(mean_d)),
            }
        )
    rows.sort(key=lambda r: r["p_cluster_global"])
    return rows


def significant_cluster_labels(labels: np.ndarray, clusters: list[dict]) -> np.ndarray:
    out = np.zeros_like(labels, dtype=int)
    for row in clusters:
        if not row["significant"]:
            continue
        cid = row["cluster_id"]
        out[labels == cid] = cid
    return out


def permutation_cluster_correction(
    delta: np.ndarray,
    masks: list[np.ndarray],
    n_map: np.ndarray,
    g: np.ndarray,
    min_n: int,
    rng: np.random.Generator,
    *,
    n_perm: int,
    t_threshold: float,
    alpha: float,
) -> dict:
    n = len(delta)
    if n < min_n * 2:
        return {"ok": False}
    cache = CellCache(delta, masks, g, min_n)
    mean_obs, t_obs, p_obs = cache.observed_maps()
    labels_obs, _, masses_obs = find_clusters(t_obs, t_threshold=t_threshold)
    max_mass_obs = max((abs(m) for _, m in masses_obs), default=0.0)
    max_masses_perm: list[float] = []
    for _ in range(n_perm):
        flip = rng.choice([-1.0, 1.0], size=n)
        t_p = cache.t_map_from_flip(flip)
        _, _, masses_p = find_clusters(t_p, t_threshold=t_threshold)
        max_masses_perm.append(max((abs(m) for _, m in masses_p), default=0.0))
    p_cluster = float(np.mean(np.array(max_masses_perm) >= max_mass_obs))
    p_level = cluster_level_pvalues(masses_obs, max_masses_perm)
    clusters = describe_clusters(
        labels_obs, masses_obs, p_level, mean_obs, g, alpha=alpha
    )
    labels_sig = significant_cluster_labels(labels_obs, clusters)
    return {
        "ok": True,
        "mean_map": mean_obs,
        "t_map": t_obs,
        "p_map": p_obs,
        "n_map": n_map,
        "labels": labels_obs,
        "labels_sig": labels_sig,
        "p_cluster": p_cluster,
        "max_mass_obs": max_mass_obs,
        "clusters": clusters,
        "n_sig_clusters": int(sum(c["significant"] for c in clusters)),
    }


def overlay_cluster_regions(
    ax,
    labels: np.ndarray,
    labels_sig: np.ndarray,
    extent: list[float],
    *,
    show_uncorrected: bool = True,
) -> None:
    """Draw cluster regions as filled mask + outer contour (not 1-px eroded dots).

    - Gray dashed: union of all |t|>threshold cells (uncorrected suprathreshold).
    - Gold fill + black outline: cluster-corrected significant cells (8-connected blobs).
    """
    uncorr = labels > 0
    sig = labels_sig > 0
    if show_uncorrected and np.any(uncorr):
        ax.contour(
            uncorr.astype(float),
            levels=[0.5],
            colors="0.72",
            linewidths=0.5,
            linestyles="dashed",
            extent=extent,
            origin="lower",
        )
    if np.any(sig):
        ax.contourf(
            sig.astype(float),
            levels=[0.5, 1.5],
            colors=["none", "gold"],
            alpha=0.28,
            extent=extent,
            origin="lower",
        )
        ax.contour(
            sig.astype(float),
            levels=[0.5],
            colors="black",
            linewidths=1.4,
            extent=extent,
            origin="lower",
        )


def _cluster_boundary(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(labels, dtype=float)
    if labels is None or labels.max() <= 0:
        return boundary
    for k in range(1, int(labels.max()) + 1):
        reg = (labels == k).astype(float)
        boundary = np.maximum(boundary, reg - binary_erosion(reg))
    return boundary


def plot_cluster_panel(
    out_png: Path,
    res: dict,
    g: np.ndarray,
    step: float,
    *,
    title: str,
    ref_label: str,
    vec_dx: np.ndarray,
    vec_dy: np.ndarray,
    vec_cnt: np.ndarray,
    min_n: int,
    kind: str,
    alpha: float,
) -> None:
    t_map = res["t_map"]
    labels_all = res["labels"]
    labels_sig = res["labels_sig"]
    mean_map = res["mean_map"]
    extent = [g[0] - step / 2, g[-1] + step / 2, g[0] - step / 2, g[-1] + step / 2]
    vmax = float(np.nanpercentile(np.abs(t_map[np.isfinite(t_map)]), 99)) if np.any(np.isfinite(t_map)) else 3.0
    vmax = max(vmax, 1e-6)
    n_sig = res.get("n_sig_clusters", 0)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    im0 = axes[0].imshow(
        t_map, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal"
    )
    overlay_cluster_regions(axes[0], labels_all, labels_sig, extent)
    axes[0].set_title(
        f"Paired t (A−B L2 error)\n"
        f"global cluster p={res['p_cluster']:.4f}; "
        f"sig. regions (p≤{alpha:g})={n_sig}",
        fontsize=9,
    )
    axes[0].set_xlabel(f"Valence ({ref_label})")
    axes[0].set_ylabel(f"Arousal ({ref_label})")
    axes[0].set_xlim(VA_MIN, VA_MAX)
    axes[0].set_ylim(VA_MIN, VA_MAX)

    xg, yg = np.meshgrid(g, g, indexing="xy")
    qscale = 1.0 / (step * 2.5)
    mask = vec_cnt >= min_n
    axes[1].quiver(
        xg[mask],
        yg[mask],
        np.nan_to_num(vec_dx[mask], nan=0.0),
        np.nan_to_num(vec_dy[mask], nan=0.0),
        angles="xy",
        scale_units="xy",
        scale=qscale,
        width=0.004,
        headwidth=3,
        headlength=4,
        color="0.25",
    )
    overlay_cluster_regions(axes[1], labels_all, labels_sig, extent, show_uncorrected=False)
    axes[1].set_xlim(VA_MIN, VA_MAX)
    axes[1].set_ylim(VA_MIN, VA_MAX)
    axes[1].set_aspect("equal")
    vec_title = "Mean bias vector" if kind == "cross_within" else "Mean adjust delta vector"
    axes[1].set_title(
        f"{vec_title}\n(gold/black = sig. cluster region)",
        fontsize=9,
    )
    axes[1].set_xlabel(f"Valence ({ref_label})")
    axes[1].set_ylabel(f"Arousal ({ref_label})")
    axes[1].grid(True, alpha=0.2)

    finite = mean_map[np.isfinite(mean_map)]
    if kind == "adjust":
        delta_note = "(negative ΔL2 = improvement)"
    else:
        delta_note = "(positive ΔL2 = cross worse than within)"
    fig.suptitle(f"{title}\nmean ΔL2 range [{np.nanmin(finite):.3f}, {np.nanmax(finite):.3f}] {delta_note}", fontsize=10, y=1.03)
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="paired t")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_overview(
    panels: list[tuple[MapSpec, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    g: np.ndarray,
    step: float,
    min_n: int,
    subset_tag: str,
    model: str,
    out_dir: Path,
    alpha: float,
    t_threshold: float,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    extent = [g[0] - step / 2, g[-1] + step / 2, g[0] - step / 2, g[-1] + step / 2]
    last_im = None
    for col, (spec, res, dx, dy, cnt, ref_label) in enumerate(panels):
        t_map = res["t_map"]
        vmax = float(np.nanpercentile(np.abs(t_map[np.isfinite(t_map)]), 99)) if np.any(np.isfinite(t_map)) else 3.0
        vmax = max(vmax, 1e-6)

        ax0 = axes[0, col]
        last_im = ax0.imshow(
            t_map, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal"
        )
        overlay_cluster_regions(ax0, res["labels"], res["labels_sig"], extent)
        ax0.set_title(
            f"{spec.title[:34]}...\n"
            f"global p={res['p_cluster']:.3f}; sig.reg={res.get('n_sig_clusters', 0)}",
            fontsize=7,
        )
        ax0.set_xlabel(f"V ({ref_label})", fontsize=7)
        ax0.set_ylabel(f"A ({ref_label})", fontsize=7)
        ax0.set_xlim(VA_MIN, VA_MAX)
        ax0.set_ylim(VA_MIN, VA_MAX)

        ax1 = axes[1, col]
        xg, yg = np.meshgrid(g, g, indexing="xy")
        mask = cnt >= min_n
        ax1.quiver(
            xg[mask], yg[mask],
            np.nan_to_num(dx[mask], nan=0.0), np.nan_to_num(dy[mask], nan=0.0),
            angles="xy", scale_units="xy", scale=1.0 / (step * 2.5), width=0.003,
            color="0.25",
        )
        overlay_cluster_regions(ax1, res["labels"], res["labels_sig"], extent, show_uncorrected=False)
        ax1.set_xlim(VA_MIN, VA_MAX)
        ax1.set_ylim(VA_MIN, VA_MAX)
        ax1.set_aspect("equal")
        ax1.set_title("vector + cluster-corrected region", fontsize=7)
        ax1.set_xlabel(f"V ({ref_label})", fontsize=7)
        ax1.set_ylabel(f"A ({ref_label})", fontsize=7)
        ax1.grid(True, alpha=0.15)

    for r, lab in enumerate(("Paired t", "Bias / adjust vector")):
        axes[r, 0].annotate(
            lab, xy=(-0.16, 0.5), xycoords="axes fraction", rotation=90,
            va="center", ha="center", fontsize=10, fontweight="bold",
        )
    fig.suptitle(
        f"LOTO cluster permutation (global + per-region correction, α={alpha:g}, |t|>{t_threshold:g}) "
        f"— {subset_tag}, {model}",
        fontsize=10,
        y=0.98,
    )
    fig.tight_layout(rect=(0.04, 0, 0.94, 0.94))
    if last_im is not None:
        fig.colorbar(last_im, ax=axes[0, :].ravel().tolist(), shrink=0.85, label="paired t")
    stem = f"Figure_{subset_tag}_cross_within_bias_cluster_overview_{model}"
    out_png = out_dir / f"{stem}.png"
    out_svg = out_dir / f"{stem}.svg"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_svg}")


SUBSET_SPECS: tuple[tuple[str, str | None], ...] = (
    ("all_loto", None),
    ("person_loto", "Person"),
    ("animal_loto", "Animal"),
    ("object_loto", "Object"),
    ("scene_loto", "Scene"),
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("fusion", "clip", "vit"), default="fusion")
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--grid-step", type=float, default=0.1)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--t-threshold", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.05, help="Cluster-level significance threshold")
    ap.add_argument("--out-dir", type=Path, default=FIG_INTEGRATED)
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_GENDER / "cross_within_bias_cluster_analysis",
    )
    ap.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="Subset tags to run (default: all). Example: all_loto",
    )
    ap.add_argument(
        "--save-maps-npz",
        action="store_true",
        help="Save t/vector maps for overview panels as npz (for later replot).",
    )
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    g = make_grid(args.grid_step)
    rng = np.random.default_rng(RANDOM_SEED)
    summary_rows: list[dict] = []
    cluster_rows: list[dict] = []

    wanted = set(args.subsets) if args.subsets else None
    for subset_tag, category in SUBSET_SPECS:
        if wanted is not None and subset_tag not in wanted:
            continue
        data = load_pool(model=args.model, category=category)
        print(f"Subset {subset_tag}: n={data['n']}")
        overview_panels: list[tuple] = []
        maps_payload: dict[str, np.ndarray] = {
            "grid": g,
            "grid_step": np.asarray(args.grid_step, float),
            "min_n": np.asarray(args.min_n, int),
        }

        for spec in MAP_SPECS:
            v_ref, a_ref, err_a, err_b, vec = comparison_arrays(data, spec)
            masks, n_map = build_cell_masks(v_ref, a_ref, g, args.radius, args.min_n)
            delta = err_a - err_b
            res = permutation_cluster_correction(
                delta, masks, n_map, g, args.min_n, rng,
                n_perm=args.n_perm, t_threshold=args.t_threshold, alpha=args.alpha,
            )
            res["t_threshold"] = args.t_threshold
            ref_label = REF_LABELS[spec.ref_gender]
            if not res.get("ok"):
                print(f"Skip {subset_tag} {spec.key}: insufficient n")
                continue

            dx, dy, _, cnt = local_vec_mean_masked(vec, masks, n_map, g, args.min_n)
            stem = f"{subset_tag}_{spec.key}_{args.model}"
            plot_cluster_panel(
                args.out_dir / f"Figure_{stem}_cluster_perm.png",
                res,
                g,
                args.grid_step,
                title=f"LOTO {subset_tag}: {spec.title} ({args.model})",
                ref_label=ref_label,
                vec_dx=dx,
                vec_dy=dy,
                vec_cnt=cnt,
                min_n=args.min_n,
                kind=spec.kind,
                alpha=args.alpha,
            )
            overview_panels.append((spec, res, dx, dy, cnt, ref_label))
            if args.save_maps_npz:
                prefix = f"{spec.key}"
                maps_payload[f"{prefix}__t_map"] = np.asarray(res["t_map"], float)
                maps_payload[f"{prefix}__mean_map"] = np.asarray(res["mean_map"], float)
                maps_payload[f"{prefix}__labels"] = np.asarray(res["labels"], int)
                maps_payload[f"{prefix}__labels_sig"] = np.asarray(res["labels_sig"], int)
                maps_payload[f"{prefix}__dx"] = np.asarray(dx, float)
                maps_payload[f"{prefix}__dy"] = np.asarray(dy, float)
                maps_payload[f"{prefix}__cnt"] = np.asarray(cnt, float)
                maps_payload[f"{prefix}__p_cluster"] = np.asarray(res["p_cluster"], float)
                maps_payload[f"{prefix}__n_sig_clusters"] = np.asarray(
                    res.get("n_sig_clusters", 0), int
                )

            cluster_table_path = args.results_dir / f"Table_{stem}_clusters.csv"
            cluster_df = pd.DataFrame(res["clusters"])
            if len(cluster_df):
                cluster_df.insert(0, "subset", subset_tag)
                cluster_df.insert(1, "spec", spec.key)
                cluster_df.to_csv(cluster_table_path, index=False)
                print(f"Saved {cluster_table_path}")
                for row in res["clusters"]:
                    cluster_rows.append(
                        {
                            "subset": subset_tag,
                            "spec": spec.key,
                            "kind": spec.kind,
                            **row,
                        }
                    )

            finite_t = res["t_map"][np.isfinite(res["t_map"])]
            finite_m = res["mean_map"][np.isfinite(res["mean_map"])]
            summary_rows.append(
                {
                    "subset": subset_tag,
                    "spec": spec.key,
                    "kind": spec.kind,
                    "ref_gender": spec.ref_gender,
                    "n_images": data["n"],
                    "cluster_p_global": res["p_cluster"],
                    "n_clusters_total": len(res["clusters"]),
                    "n_clusters_sig": res["n_sig_clusters"],
                    "alpha": args.alpha,
                    "max_cluster_mass": res["max_mass_obs"],
                    "mean_delta_l2_global": float(np.mean(err_a - err_b)),
                    "median_delta_l2_global": float(np.median(err_a - err_b)),
                    "max_abs_t": float(np.nanmax(np.abs(finite_t))) if finite_t.size else np.nan,
                    "mean_delta_l2_grid_mean": float(np.nanmean(finite_m)) if finite_m.size else np.nan,
                }
            )

        if overview_panels:
            plot_overview(
                overview_panels,
                g=g,
                step=args.grid_step,
                min_n=args.min_n,
                subset_tag=subset_tag,
                model=args.model,
                out_dir=args.out_dir,
                alpha=args.alpha,
                t_threshold=args.t_threshold,
            )
            if args.save_maps_npz:
                npz_path = (
                    args.results_dir
                    / f"maps_{subset_tag}_cross_within_bias_cluster_overview_{args.model}.npz"
                )
                np.savez_compressed(npz_path, **maps_payload)
                print(f"Saved {npz_path}")

    tag = "all" if wanted is None else "_".join(sorted(wanted))
    summary_path = (
        args.results_dir / f"summary_cross_within_bias_clusters_{args.model}.csv"
        if wanted is None
        else args.results_dir / f"summary_cross_within_bias_clusters_{args.model}_{tag}.csv"
    )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")
    if cluster_rows:
        all_clusters_path = (
            args.results_dir / f"all_clusters_cross_within_bias_{args.model}.csv"
            if wanted is None
            else args.results_dir / f"all_clusters_cross_within_bias_{args.model}_{tag}.csv"
        )
        pd.DataFrame(cluster_rows).to_csv(all_clusters_path, index=False)
        print(f"Saved {all_clusters_path}")


if __name__ == "__main__":
    main()
