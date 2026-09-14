#!/usr/bin/env python3
"""Unify Part-1 linear geometry (gender bridge Phi) with Part-2 twist visualizations.

After establishing linear sufficiency (Fig 3), twist should be read as *residual*
structure beyond the global affine map Phi: y_f ~ Phi(y_m).

Panels:
  - Linear-aligned score residual ||y_f - Phi(y_m)||
  - Raw score gap ||y_f - y_m|| (pre-Phi)
  - Aligned UMAP twist on 4D (exploratory relational geometry)
  - Model naturalness omega = pf - Phi(pm)
  - cross-within-style delta for comparison
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
MIN_TEST_LOTO = 2
MALE_4D = ["valence_male", "arousal_male", "valence_male_sd", "arousal_male_sd"]
FEMALE_4D = ["valence_female", "arousal_female", "valence_female_sd", "arousal_female_sd"]

_CLUSTER_PATH = Path(__file__).parent / "analyze_cross_within_bias_clusters.py"
_spec = importlib.util.spec_from_file_location("_cwc", _CLUSTER_PATH)
assert _spec is not None and _spec.loader is not None
_cwc = importlib.util.module_from_spec(_spec)
sys.modules["_cwc"] = _cwc
_spec.loader.exec_module(_cwc)

PHI_JSON = PROJECT_ROOT / "results" / "cvae_cross_gender" / "linear_shift_svd_summary.json"


def loto_indices(df: pd.DataFrame) -> list[int]:
    idx: list[int] = []
    for tb in df["theme_base"].unique():
        m = (df["theme_base"] == tb).values
        if int(m.sum()) >= MIN_TEST_LOTO:
            idx.extend(np.where(m)[0].tolist())
    return idx


def load_phi() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(PHI_JSON.read_text(encoding="utf-8"))
    g = data["gender_shift"]
    A = np.asarray(g["A"], float)
    b = np.asarray(g["b"], float)
    fp = np.asarray(g["fixed_point"], float)
    return A, b, fp


def apply_phi(xy: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return xy @ A.T + b


def load_loto_fusion() -> dict:
    oasis = load_oasis_meta(OASIS_SCORES_CSV)
    cols = (
        ["image_id", "category", "valence", "arousal"]
        + MALE_4D + FEMALE_4D
    )
    full = oasis[oasis[cols].notna().all(axis=1)].reset_index(drop=True)
    full = add_theme_base(full)
    idx = loto_indices(full)
    sub = full.iloc[idx].reset_index(drop=True)
    arr = np.arange(len(sub))
    base = RESULTS_GENDER / "loto_nested"
    return {
        "df": sub,
        "v_c": sub["valence"].to_numpy(float),
        "a_c": sub["arousal"].to_numpy(float),
        "y_m": sub[MALE_4D[:2]].to_numpy(float),
        "y_f": sub[FEMALE_4D[:2]].to_numpy(float),
        "male_4d": sub[MALE_4D].to_numpy(float),
        "female_4d": sub[FEMALE_4D].to_numpy(float),
        "pm": np.load(base / "male" / "y_all_pred_fusion.npy")[arr],
        "pf": np.load(base / "female" / "y_all_pred_fusion.npy")[arr],
    }


def run_aligned_umap(x_a: np.ndarray, x_b: np.ndarray, *, n_neighbors: int, n_epochs: int) -> tuple[np.ndarray, np.ndarray]:
    xa = StandardScaler().fit_transform(x_a)
    xb = StandardScaler().fit_transform(x_b)
    n = len(xa)
    rel = {i: i for i in range(n)}
    mapper = aligned_umap.AlignedUMAP(
        n_neighbors=n_neighbors, min_dist=0.15, metric="euclidean",
        alignment_regularisation=0.1, random_state=RANDOM_SEED, n_epochs=n_epochs,
    )
    mapper.fit([xa, xb], relations=[rel])
    ema, emb = mapper.embeddings_
    return np.asarray(ema, float), np.asarray(emb, float)


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


def plot_bridge(
    data: dict,
    A: np.ndarray,
    b: np.ndarray,
    fp: np.ndarray,
    geo_disp: np.ndarray,
    g: np.ndarray,
    step: float,
    radius: float,
    min_n: int,
    out: Path,
) -> dict:
    y_m, y_f = data["y_m"], data["y_f"]
    pm, pf = data["pm"], data["pf"]
    v_c, a_c = data["v_c"], data["a_c"]

    phi_ym = apply_phi(y_m, A, b)
    phi_pm = apply_phi(pm, A, b)

    resid_score = y_f - phi_ym
    resid_score_l2 = np.linalg.norm(resid_score, axis=1)
    raw_gap_l2 = np.linalg.norm(y_f - y_m, axis=1)
    omega_pf = pf - phi_pm
    omega_l2 = np.linalg.norm(omega_pf, axis=1)
    model_twist_l2 = np.linalg.norm(pm - pf, axis=1)

    def grid(vals):
        return local_mean_grid(v_c, a_c, vals, g, radius, min_n)

    grids = {
        "||y_f - Phi(y_m)||": grid(resid_score_l2),
        "||y_f - y_m|| raw": grid(raw_gap_l2),
        "UMAP 4D twist": grid(geo_disp),
        "omega=||pf-Phi(pm)||": grid(omega_l2),
        "||pm-pf||": grid(model_twist_l2),
        "excess=UMAP-linear_resid": grid(geo_disp - resid_score_l2),
    }

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
    ext = [g[0] - step / 2, g[-1] + step / 2, g[0] - step / 2, g[-1] + step / 2]

    for ax, (title, gr) in zip(axes.ravel(), grids.items()):
        signed = "excess" in title
        if signed:
            vlim = max(float(np.nanpercentile(np.abs(gr[np.isfinite(gr)]), 95)), 1e-6)
            im = ax.imshow(gr, origin="lower", extent=ext, aspect="equal", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        else:
            vlim = max(float(np.nanpercentile(gr[np.isfinite(gr)], 95)), 1e-6)
            im = ax.imshow(gr, origin="lower", extent=ext, aspect="equal", cmap="magma", vmin=0, vmax=vlim)
        ax.scatter([fp[0]], [fp[1]], s=120, marker="*", c="cyan", edgecolors="k", linewidths=0.8, zorder=5)
        ax.set_xlim(VA_MIN, VA_MAX)
        ax.set_ylim(VA_MIN, VA_MAX)
        ax.set_xlabel("Valence (common)")
        ax.set_ylabel("Arousal (common)")
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Linear bridge Phi (gender shift, Fig 3) vs twist metrics (Part 2)\n"
        f"Phi fixed point=({fp[0]:.2f},{fp[1]:.2f}); cyan star; grid r={radius}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "phi_fixed_point": fp.tolist(),
        "relational_loto_affine_fp_male_va": [5.012, 3.972],
        "mean_resid_score_l2": float(resid_score_l2.mean()),
        "mean_raw_gap_l2": float(raw_gap_l2.mean()),
        "mean_umap_4d_twist": float(geo_disp.mean()),
        "mean_omega_l2": float(omega_l2.mean()),
        "corr_umap_vs_linear_resid_score": float(np.corrcoef(geo_disp, resid_score_l2)[0, 1]),
        "corr_umap_vs_raw_gap": float(np.corrcoef(geo_disp, raw_gap_l2)[0, 1]),
        "corr_umap_vs_omega": float(np.corrcoef(geo_disp, omega_l2)[0, 1]),
        "corr_linear_resid_vs_omega": float(np.corrcoef(resid_score_l2, omega_l2)[0, 1]),
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-step", type=float, default=0.1)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--umap-neighbors", type=int, default=20)
    ap.add_argument("--umap-epochs", type=int, default=400)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_GENDER / "linear_phi_twist_bridge")
    ap.add_argument("--fig-dir", type=Path, default=FIG_INTEGRATED)
    args = ap.parse_args()

    A, b, fp = load_phi()
    data = load_loto_fusion()
    g = _cwc.make_grid(args.grid_step)

    emb_m, emb_f = run_aligned_umap(
        data["male_4d"], data["female_4d"],
        n_neighbors=args.umap_neighbors, n_epochs=args.umap_epochs,
    )
    geo_disp = np.linalg.norm(emb_f - emb_m, axis=1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = args.fig_dir / "Figure_linear_phi_twist_bridge.png"
    stats = plot_bridge(data, A, b, fp, geo_disp, g, args.grid_step, args.radius, args.min_n, fig_path)

    tab = data["df"][["image_id", "category", "valence", "arousal"]].copy()
    phi_ym = apply_phi(data["y_m"], A, b)
    tab["resid_score_l2"] = np.linalg.norm(data["y_f"] - phi_ym, axis=1)
    tab["raw_gap_l2"] = np.linalg.norm(data["y_f"] - data["y_m"], axis=1)
    tab["umap_4d_twist"] = geo_disp
    tab["omega_pf_l2"] = np.linalg.norm(data["pf"] - apply_phi(data["pm"], A, b), axis=1)
    tab.to_csv(args.out_dir / "linear_phi_twist_per_image.csv", index=False)

    summary = {"phi_source": str(PHI_JSON), "n_loto": len(tab), "bridge": stats, "figure": str(fig_path)}
    (args.out_dir / "linear_phi_twist_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Phi FP: {fp}")
    for k, v in stats.items():
        if k.startswith("corr") or k.startswith("mean"):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
