#!/usr/bin/env python3
"""VA maps of per-image male−female CLIP Ridge coefficient (λ) effect differences.

For each LOTO test image, fit nested-LOTO CLIP Ridge models (male / female targets) and
compute the signed linear effect of differing coefficients (denormalized V–A units):

    Δλ^V_i = x_i·(λ_m^V − λ_f^V)
    Δλ^A_i = x_i·(λ_m^A − λ_f^A)

2×3 layout: top row = valence, bottom row = arousal; columns use ref anchors
  common / male score / female score.
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))
FIG_DIR = PROJECT_ROOT / "code" / "viz" / "figures"

_spec = importlib.util.spec_from_file_location(
    "analyze_cross_within_bias_clusters",
    FIG_DIR / "analyze_cross_within_bias_clusters.py",
)
_cluster = importlib.util.module_from_spec(_spec)
sys.modules["analyze_cross_within_bias_clusters"] = _cluster
assert _spec.loader is not None
_spec.loader.exec_module(_cluster)

from config import (  # noqa: E402
    FIG_INTEGRATED,
    OASIS_SCORES_CSV,
    RESULTS_GENDER,
    RESULTS_STEP1,
    RANDOM_SEED,
    VALENCE_AROUSAL_SCALE_MAX,
    VALENCE_AROUSAL_SCALE_MIN,
)
from dataset import add_theme_base, load_oasis_meta, train_val_split_by_theme  # noqa: E402
from run_gender_models_loto_loco import (  # noqa: E402
    TARGET_FEMALE,
    TARGET_MALE,
    normalize_y,
    optimize_alpha_nested,
)

# Pooled map / R² over LOTO images: include singleton themes (n_test=1).
MIN_TEST_LOTO = 1

VA_MIN = VALENCE_AROUSAL_SCALE_MIN
VA_MAX = VALENCE_AROUSAL_SCALE_MAX
VA_SCALE = VA_MAX - VA_MIN
REF_LABELS = {
    "common": "OASIS common (ref)",
    "male": "male score (ref)",
    "female": "female score (ref)",
}


def make_grid(step: float) -> np.ndarray:
    return np.arange(VA_MIN + step / 2, VA_MAX, step)


def loto_indices(df) -> list[int]:
    idx: list[int] = []
    for left_out in df["theme_base"].unique():
        m = (df["theme_base"] == left_out).values
        if int(m.sum()) >= MIN_TEST_LOTO:
            idx.extend(np.where(m)[0].tolist())
    return idx


def local_mean_grid(
    v_ref: np.ndarray,
    a_ref: np.ndarray,
    values: np.ndarray,
    g: np.ndarray,
    radius: float,
    min_n: int,
) -> np.ndarray:
    n_g = len(g)
    out = np.full((n_g, n_g), np.nan)
    r2 = radius * radius
    for i in range(n_g):
        for j in range(n_g):
            vc, ac = g[j], g[i]
            m = (v_ref - vc) ** 2 + (a_ref - ac) ** 2 <= r2
            if int(m.sum()) < min_n:
                continue
            out[i, j] = float(np.mean(values[m]))
    return out


def _extent(g: np.ndarray, step: float) -> list[float]:
    return [g[0] - step / 2, g[-1] + step / 2, g[0] - step / 2, g[-1] + step / 2]


def _sym_vlim(grids: list[np.ndarray], pct: float = 95.0) -> float:
    vals = np.concatenate([g[np.isfinite(g)] for g in grids if np.isfinite(g).any()])
    if vals.size == 0:
        return 1.0
    return max(float(np.nanpercentile(np.abs(vals), pct)), 1e-6)


def _load_alpha_map(gender: str) -> dict[str, float]:
    path = RESULTS_GENDER / "loto_nested" / gender / "summary.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: float(v["best_alpha_clip"]) for k, v in data["by_fold"].items()}


def _fit_clip_ridge(X_tr_n: np.ndarray, y_tr_n: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    coef_v = Ridge(alpha=alpha, random_state=RANDOM_SEED).fit(X_tr_n, y_tr_n[:, 0]).coef_
    coef_a = Ridge(alpha=alpha, random_state=RANDOM_SEED).fit(X_tr_n, y_tr_n[:, 1]).coef_
    return coef_v, coef_a


def _per_image_lambda_diff(
    X_te_n: np.ndarray,
    coef_mv: np.ndarray,
    coef_fv: np.ndarray,
    coef_ma: np.ndarray,
    coef_fa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dv = (X_te_n @ (coef_mv - coef_fv)) * VA_SCALE
    da = (X_te_n @ (coef_ma - coef_fa)) * VA_SCALE
    return dv, da


def compute_lambda_diffs(
    *,
    n_trials: int,
    use_cached_alpha: bool,
) -> dict:
    oasis = load_oasis_meta(OASIS_SCORES_CSV)
    cols = [
        "valence",
        "arousal",
        "valence_male",
        "arousal_male",
        "valence_female",
        "arousal_female",
    ]
    full = oasis[oasis[cols].notna().all(axis=1)].reset_index(drop=True)
    full = add_theme_base(full)

    X_clip = np.load(RESULTS_STEP1 / "features_clip.npy").astype(float)
    valid = oasis[cols].notna().all(axis=1).values
    X_clip = X_clip[valid]

    y_m = full[TARGET_MALE].to_numpy(float)
    y_f = full[TARGET_FEMALE].to_numpy(float)

    alpha_m = _load_alpha_map("male") if use_cached_alpha else {}
    alpha_f = _load_alpha_map("female") if use_cached_alpha else {}

    loto_idx = loto_indices(full)
    n_pool = len(loto_idx)
    delta_v = np.full(n_pool, np.nan)
    delta_a = np.full(n_pool, np.nan)
    coef_l2_fold = np.full(n_pool, np.nan)

    write_pos = 0
    bases = full["theme_base"].unique().tolist()
    for fold_idx, left_out in enumerate(bases):
        train_mask = (full["theme_base"] != left_out).values
        test_mask = (full["theme_base"] == left_out).values
        n_test = int(test_mask.sum())
        if n_test < MIN_TEST_LOTO:
            continue

        df_train = full[train_mask].reset_index(drop=True)
        _, _, train_inner_idx, val_inner_idx = train_val_split_by_theme(
            df_train, train_ratio=0.8, random_state=RANDOM_SEED, return_indices=True
        )
        train_idx_global = np.where(train_mask)[0][train_inner_idx]
        val_idx_global = np.where(train_mask)[0][val_inner_idx]

        X_tr = X_clip[train_mask]
        X_te = X_clip[test_mask]
        y_tr_m = y_m[train_mask]
        y_tr_f = y_f[train_mask]

        X_tr_i = X_clip[train_idx_global]
        X_val = X_clip[val_idx_global]
        y_tr_i_m = y_m[train_idx_global]
        y_val_m = y_m[val_idx_global]
        y_tr_i_f = y_f[train_idx_global]
        y_val_f = y_f[val_idx_global]

        scl = StandardScaler().fit(X_tr)
        X_tr_n = scl.transform(X_tr)
        X_te_n = scl.transform(X_te)
        X_tr_i_n = scl.transform(X_tr_i)
        X_val_n = scl.transform(X_val)

        if use_cached_alpha and left_out in alpha_m and left_out in alpha_f:
            best_a_m = alpha_m[left_out]
            best_a_f = alpha_f[left_out]
        else:
            best_a_m, _ = optimize_alpha_nested(
                X_tr_i_n, X_val_n, y_tr_i_m, y_val_m, True, n_trials
            )
            best_a_f, _ = optimize_alpha_nested(
                X_tr_i_n, X_val_n, y_tr_i_f, y_val_f, True, n_trials
            )

        y_tr_m_n = normalize_y(y_tr_m, VA_MIN, VA_MAX)
        y_tr_f_n = normalize_y(y_tr_f, VA_MIN, VA_MAX)

        cmv, cma = _fit_clip_ridge(X_tr_n, y_tr_m_n, best_a_m)
        cfv, cfa = _fit_clip_ridge(X_tr_n, y_tr_f_n, best_a_f)

        fold_dv, fold_da = _per_image_lambda_diff(X_te_n, cmv, cfv, cma, cfa)

        coef_l2 = float(
            np.linalg.norm(np.concatenate([cmv - cfv, cma - cfa]))
        )

        delta_v[write_pos : write_pos + n_test] = fold_dv
        delta_a[write_pos : write_pos + n_test] = fold_da
        coef_l2_fold[write_pos : write_pos + n_test] = coef_l2
        write_pos += n_test

        if fold_idx % 40 == 0:
            print(f"  fold {fold_idx + 1}/{len(bases)}: {left_out} (n_test={n_test})")

    sub = full.iloc[loto_idx].reset_index(drop=True)
    image_ids = sub["image_id"].astype(str).to_numpy()
    categories = sub["category"].astype(str).to_numpy()
    themes = sub["theme"].astype(str).to_numpy()
    filenames = (
        sub["image_filename"].astype(str).to_numpy()
        if "image_filename" in sub.columns
        else sub["theme"].astype(str).to_numpy()
    )
    return {
        "delta_v": delta_v,
        "delta_a": delta_a,
        "coef_l2_fold": coef_l2_fold,
        "image_id": image_ids,
        "category": categories,
        "theme": themes,
        "file_name": filenames,
        "v_common": sub["valence"].to_numpy(float),
        "a_common": sub["arousal"].to_numpy(float),
        "v_m": sub["valence_male"].to_numpy(float),
        "a_m": sub["arousal_male"].to_numpy(float),
        "v_f": sub["valence_female"].to_numpy(float),
        "a_f": sub["arousal_female"].to_numpy(float),
        "n": len(sub),
    }


def plot_2x3(
    data: dict,
    g: np.ndarray,
    step: float,
    radius: float,
    min_n: int,
    out: Path,
) -> None:
    panels = (
        ("common", data["v_common"], data["a_common"]),
        ("male", data["v_m"], data["a_m"]),
        ("female", data["v_f"], data["a_f"]),
    )
    rows = (
        ("Valence", data["delta_v"], "x·(λ_m^V − λ_f^V)"),
        ("Arousal", data["delta_a"], "x·(λ_m^A − λ_f^A)"),
    )

    row_grids: list[list[np.ndarray]] = []
    for _, vals, _ in rows:
        row_grids.append(
            [local_mean_grid(vr, ar, vals, g, radius, min_n) for _, vr, ar in panels]
        )
    vlim_v = _sym_vlim(row_grids[0])
    vlim_a = _sym_vlim(row_grids[1])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    ims: list = []
    for r, (row_label, _, cbar_lbl) in enumerate(rows):
        vlim = vlim_v if r == 0 else vlim_a
        for c, (ref_key, vr, ar) in enumerate(panels):
            ax = axes[r, c]
            grid = row_grids[r][c]
            ext = _extent(g, step)
            im = ax.imshow(
                grid,
                origin="lower",
                extent=ext,
                cmap="PiYG",
                vmin=-vlim,
                vmax=vlim,
                aspect="equal",
            )
            ims.append(im)
            ax.set_xlim(VA_MIN, VA_MAX)
            ax.set_ylim(VA_MIN, VA_MAX)
            ax.set_xlabel(f"Valence ({REF_LABELS[ref_key]})")
            ax.set_ylabel(f"Arousal ({REF_LABELS[ref_key]})")
            ax.set_title(
                f"{row_label}: local mean {cbar_lbl}\n({REF_LABELS[ref_key]})",
                fontsize=8,
            )
            ax.grid(True, alpha=0.2)

    fig.suptitle(
        "CLIP Ridge λ gender difference (male − female linear effect)\n"
        f"LOTO pooled n={data['n']}, r={radius}, grid={step}",
        fontsize=11,
    )
    fig.colorbar(ims[0], ax=axes[0, :], shrink=0.9, label="Mean Δ valence (V units)")
    fig.colorbar(ims[-1], ax=axes[1, :], shrink=0.9, label="Mean Δ arousal (A units)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def run_cluster_analysis(
    data: dict,
    g: np.ndarray,
    radius: float,
    min_n: int,
    rng: np.random.Generator,
    *,
    n_perm: int,
    t_threshold: float,
    alpha: float,
) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    panels = (
        ("common", data["v_common"], data["a_common"]),
        ("male", data["v_m"], data["a_m"]),
        ("female", data["v_f"], data["a_f"]),
    )
    rows = (
        ("valence", data["delta_v"]),
        ("arousal", data["delta_a"]),
    )
    results: dict[tuple[str, str], dict] = {}
    summary: list[dict] = []
    for row_key, delta in rows:
        for ref_key, vr, ar in panels:
            masks, n_map = _cluster.build_cell_masks(vr, ar, g, radius, min_n)
            res = _cluster.permutation_cluster_correction(
                delta,
                masks,
                n_map,
                g,
                min_n,
                rng,
                n_perm=n_perm,
                t_threshold=t_threshold,
                alpha=alpha,
            )
            results[(row_key, ref_key)] = res
            if not res.get("ok"):
                summary.append(
                    {
                        "dimension": row_key,
                        "ref": ref_key,
                        "p_cluster_global": np.nan,
                        "n_sig_clusters": 0,
                        "ok": False,
                    }
                )
                continue
            for c in res.get("clusters", []):
                summary.append(
                    {
                        "dimension": row_key,
                        "ref": ref_key,
                        "ok": True,
                        "p_cluster_global": res["p_cluster"],
                        "n_sig_clusters": res["n_sig_clusters"],
                        "cluster_id": c["cluster_id"],
                        "n_cells": c["n_cells"],
                        "cluster_mass": c["cluster_mass"],
                        "p_cluster_region": c["p_cluster_global"],
                        "significant": c["significant"],
                        "centroid_valence": c["centroid_valence"],
                        "centroid_arousal": c["centroid_arousal"],
                        "mean_delta": c["mean_delta_l2"],
                    }
                )
            if not res.get("clusters"):
                summary.append(
                    {
                        "dimension": row_key,
                        "ref": ref_key,
                        "ok": True,
                        "p_cluster_global": res["p_cluster"],
                        "n_sig_clusters": 0,
                    }
                )
    return results, summary


def plot_2x3_cluster_perm(
    data: dict,
    cluster_results: dict[tuple[str, str], dict],
    g: np.ndarray,
    step: float,
    radius: float,
    min_n: int,
    out: Path,
    *,
    t_threshold: float,
    alpha: float,
) -> None:
    panels = (
        ("common", data["v_common"], data["a_common"]),
        ("male", data["v_m"], data["a_m"]),
        ("female", data["v_f"], data["a_f"]),
    )
    row_keys = ("valence", "arousal")
    row_labels = {
        "valence": "Valence: x·(λ_m^V − λ_f^V)",
        "arousal": "Arousal: x·(λ_m^A − λ_f^A)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    ims: list = []
    for r, row_key in enumerate(row_keys):
        for c, (ref_key, _, _) in enumerate(panels):
            ax = axes[r, c]
            res = cluster_results[(row_key, ref_key)]
            ext = _extent(g, step)
            if not res.get("ok"):
                ax.set_title(f"{row_labels[row_key]}\n({REF_LABELS[ref_key]})\ninsufficient n")
                ax.set_xlim(VA_MIN, VA_MAX)
                ax.set_ylim(VA_MIN, VA_MAX)
                continue
            t_map = res["t_map"]
            finite = t_map[np.isfinite(t_map)]
            vmax = (
                float(np.nanpercentile(np.abs(finite), 99))
                if finite.size
                else 3.0
            )
            vmax = max(vmax, 1e-6)
            im = ax.imshow(
                t_map,
                origin="lower",
                extent=ext,
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                aspect="equal",
            )
            ims.append(im)
            _cluster.overlay_cluster_regions(ax, res["labels"], res["labels_sig"], ext)
            n_sig = res.get("n_sig_clusters", 0)
            sig_sizes = [cl["n_cells"] for cl in res.get("clusters", []) if cl.get("significant")]
            size_note = f", max blob={max(sig_sizes)} cells" if sig_sizes else ""
            ax.set_xlim(VA_MIN, VA_MAX)
            ax.set_ylim(VA_MIN, VA_MAX)
            ax.set_xlabel(f"Valence ({REF_LABELS[ref_key]})")
            ax.set_ylabel(f"Arousal ({REF_LABELS[ref_key]})")
            ax.set_title(
                f"{row_labels[row_key]}\n({REF_LABELS[ref_key]})\n"
                f"global p={res['p_cluster']:.4f}, sig.reg={n_sig}{size_note}",
                fontsize=7.5,
            )
            ax.grid(True, alpha=0.2)

    fig.suptitle(
        "CLIP Ridge λ gender difference — cluster permutation (paired t, sign-flip)\n"
        f"LOTO n={data['n']}, r={radius}, grid={step}, |t|>{t_threshold}, region α={alpha}",
        fontsize=11,
    )
    if ims:
        fig.colorbar(ims[0], ax=axes, shrink=0.85, label="Local paired t (male−female λ effect)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_2x3_with_cluster_overlay(
    data: dict,
    cluster_results: dict[tuple[str, str], dict],
    g: np.ndarray,
    step: float,
    radius: float,
    min_n: int,
    out: Path,
) -> None:
    """PiYG mean maps with significant cluster regions overlaid (gold)."""
    panels = (
        ("common", data["v_common"], data["a_common"]),
        ("male", data["v_m"], data["a_m"]),
        ("female", data["v_f"], data["a_f"]),
    )
    rows = (
        ("valence", data["delta_v"], "x·(λ_m^V − λ_f^V)"),
        ("arousal", data["delta_a"], "x·(λ_m^A − λ_f^A)"),
    )

    row_grids: list[list[np.ndarray]] = []
    for _, vals, _ in rows:
        row_grids.append(
            [local_mean_grid(vr, ar, vals, g, radius, min_n) for _, vr, ar in panels]
        )
    vlim_v = _sym_vlim(row_grids[0])
    vlim_a = _sym_vlim(row_grids[1])

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    ims: list = []
    for r, (row_key, _, cbar_lbl) in enumerate(rows):
        vlim = vlim_v if r == 0 else vlim_a
        for c, (ref_key, vr, ar) in enumerate(panels):
            ax = axes[r, c]
            grid = row_grids[r][c]
            ext = _extent(g, step)
            im = ax.imshow(
                grid,
                origin="lower",
                extent=ext,
                cmap="PiYG",
                vmin=-vlim,
                vmax=vlim,
                aspect="equal",
            )
            ims.append(im)
            res = cluster_results[(row_key, ref_key)]
            if res.get("ok"):
                _cluster.overlay_cluster_regions(ax, res["labels"], res["labels_sig"], ext)
                n_sig = res.get("n_sig_clusters", 0)
                p_glob = res["p_cluster"]
                title_suffix = f"\nglobal p={p_glob:.4f}, sig.reg={n_sig}"
            else:
                title_suffix = "\n(cluster n/a)"
            ax.set_xlim(VA_MIN, VA_MAX)
            ax.set_ylim(VA_MIN, VA_MAX)
            ax.set_xlabel(f"Valence ({REF_LABELS[ref_key]})")
            ax.set_ylabel(f"Arousal ({REF_LABELS[ref_key]})")
            ax.set_title(
                f"{rows[r][0]}: local mean {cbar_lbl}\n({REF_LABELS[ref_key]}){title_suffix}",
                fontsize=7.5,
            )
            ax.grid(True, alpha=0.2)

    fig.suptitle(
        "CLIP Ridge λ gender difference (male − female) with significant clusters\n"
        f"LOTO pooled n={data['n']}, r={radius}, grid={step}",
        fontsize=11,
    )
    fig.colorbar(ims[0], ax=axes[0, :], shrink=0.9, label="Mean Δ valence (V units)")
    fig.colorbar(ims[-1], ax=axes[1, :], shrink=0.9, label="Mean Δ arousal (A units)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="VA maps of per-image male−female CLIP λ differences.")
    ap.add_argument("--grid-step", type=float, default=0.1)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--min-n", type=int, default=5)
    ap.add_argument("--n-trials", type=int, default=30, help="Inner Optuna trials if not using cached alpha")
    ap.add_argument(
        "--reoptimize-alpha",
        action="store_true",
        help="Re-run nested alpha search (slow). Default: use summary.json best_alpha_clip.",
    )
    ap.add_argument("--out-dir", type=Path, default=FIG_INTEGRATED)
    ap.add_argument("--cache-npy", type=Path, default=RESULTS_GENDER / "lambda_gender_diff_loto_clip.npy")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--t-threshold", type=float, default=2.0)
    ap.add_argument("--cluster-alpha", type=float, default=0.05)
    ap.add_argument(
        "--cluster-out-dir",
        type=Path,
        default=RESULTS_GENDER / "lambda_gender_diff_cluster_analysis",
    )
    args = ap.parse_args()

    use_cached = not args.reoptimize_alpha
    need_recompute = args.recompute
    if args.cache_npy.exists() and not need_recompute:
        data = np.load(args.cache_npy, allow_pickle=True).item()
        if "delta_v" not in data or "delta_a" not in data:
            need_recompute = True
        else:
            print(f"Loading cached {args.cache_npy}")
    else:
        need_recompute = True
    if need_recompute:
        print("Computing per-image CLIP λ gender differences (LOTO)...")
        data = compute_lambda_diffs(n_trials=args.n_trials, use_cached_alpha=use_cached)
        args.cache_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.cache_npy, data, allow_pickle=True)
        print(f"Cached {args.cache_npy}")

    g = make_grid(args.grid_step)
    out = args.out_dir / "Figure_gender_lambda_diff_clip_2x3_loto.png"
    plot_2x3(data, g, args.grid_step, args.radius, args.min_n, out)

    print("Running cluster permutation (6 panels)...")
    rng = np.random.default_rng(RANDOM_SEED)
    cluster_results, summary = run_cluster_analysis(
        data,
        g,
        args.radius,
        args.min_n,
        rng,
        n_perm=args.n_perm,
        t_threshold=args.t_threshold,
        alpha=args.cluster_alpha,
    )
    args.cluster_out_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    summary_df = pd.DataFrame(summary)
    csv_path = args.cluster_out_dir / "summary_lambda_gender_diff_clusters_loto.csv"
    summary_df.to_csv(csv_path, index=False)
    json_path = args.cluster_out_dir / "summary_lambda_gender_diff_clusters_loto.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_perm": args.n_perm,
                "t_threshold": args.t_threshold,
                "alpha": args.cluster_alpha,
                "radius": args.radius,
                "grid_step": args.grid_step,
                "min_n": args.min_n,
                "panels": summary,
            },
            f,
            indent=2,
        )
    print(f"Saved {csv_path}")

    plot_2x3_cluster_perm(
        data,
        cluster_results,
        g,
        args.grid_step,
        args.radius,
        args.min_n,
        args.out_dir / "Figure_gender_lambda_diff_clip_cluster_perm_2x3_loto.png",
        t_threshold=args.t_threshold,
        alpha=args.cluster_alpha,
    )
    plot_2x3_with_cluster_overlay(
        data,
        cluster_results,
        g,
        args.grid_step,
        args.radius,
        args.min_n,
        args.out_dir / "Figure_gender_lambda_diff_clip_cluster_overlay_2x3_loto.png",
    )


if __name__ == "__main__":
    main()
