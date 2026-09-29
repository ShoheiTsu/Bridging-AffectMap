#!/usr/bin/env python3
"""
Anchor structure analysis:
  Z2 — piecewise-affine k-sweep vs exact OT (gender M→F and F→M)
  Z1 — forced fixed-point (anchor position) sweep
  Z3 — shared-anchor generalization
  Z4 — VA coverage quality of shared anchors
  Z5 — minimal-coverage anchor identification & characterization
  Z6 — signed-area orientation of Z5 minimal-coverage polygons

Output: results/anchor_structure/
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import adjusted_rand_score, r2_score
from scipy.stats import mannwhitneyu, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from config import OASIS_SCORES_CSV, RANDOM_SEED  # noqa: E402
from dataset import add_theme_base, load_oasis_meta  # noqa: E402
from train_cvae_cross_gender import train_val_test_split_by_theme  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered constants (do not change after seeing results)
# ---------------------------------------------------------------------------
DELTA_R2_WIN = 0.02
ALPHA_PERM = 0.05
N_PERM_DEFAULT = 5000
K_LIST_DEFAULT = (1, 2, 3, 4, 5, 6)
N_PARAMS_PER_AFFINE = 6
TRAIN_RATIO, VAL_RATIO = 0.6, 0.2
KMEANS_N_INIT = 20
KMEANS_STABILITY_SEEDS = (42, 43, 44, 45, 46)
KMEANS_MIN_CLUSTER = 8
MONOTONE_TOL = 1e-4
OT_BEAT_EPS = 1e-3
STABILITY_ARI_PASS = 0.7
SLOW_CLIMB_DELTA = 0.01
SLOW_CLIMB_OT_GAP = 0.02

# Z1 fixed-point sweep
Z1_COARSE_VALS = (2.0, 3.0, 4.0, 5.0, 6.0)
Z1_FINE_STEP = 0.25
Z1_FINE_HALF_WIDTH = 1.0
Z1_RIDGE = 1e-8
Z1_CONSISTENCY_EPS = 1e-3
Z1_N_BOOT_DEFAULT = 200
CANON_FP_M_TO_F = (5.15, 4.36)  # population-mean fixed-point reference

# Z3 shared-anchor generalization
Z3_K_LIST = (1, 2, 3, 5, 8, 13, 21, 34)
Z3_N_RANDOM_DRAWS = 50
Z3_DRAW_SEED_OFFSET = 3000
Z3_SIM_LAYERS = ("near", "mid", "far")
Z3_CEILING_FRAC = 0.90
Z3_FP_NEIGHBOR_RADIUS = 0.75
Z3_MIN_EVAL = 30
Z3_MIN_LAYER = 10
Z3_SELECTION_K = (3, 8, 21)
Z3_RIDGE_ALPHA = 1.0  # main-curve stabilizer for k>=3 shared-anchor affine
Z3_AFFINE_MODE_DEFAULT = "ridge"  # "ridge" (main) | "naive" (diagnostic)

# Z4 VA coverage of shared anchors (quality axis completing Z3 quantity)
Z4_GRID_N = 5
Z4_VA_LIM = (1.5, 7.0)
Z4_K_LIST = (3, 5, 8, 13, 21, 34)
Z4_N_DRAWS = 80
Z4_DRAW_SEED_OFFSET = 4000
Z4_CEILING_FRAC = 0.90
Z4_RIDGE_ALPHA = 1.0
Z4_UPGRADE_RHO = 0.30
Z4_UPGRADE_P = 0.05
Z4_UPGRADE_MIN_K_PASS = 3
Z4_UPGRADE_K_SET = (5, 8, 13, 21)
Z4_GREEDY_CAND_PER_CELL = 3
Z4_GREEDY_MAX_K = 34
Z4_MINIMAL_MAX_CELLS = 12  # upgrade: achieve 90% ceiling with ≤ this many cells

# Z5 minimal-anchor identification & characterization
Z5_GRID_N = Z4_GRID_N
Z5_VA_LIM = Z4_VA_LIM
Z5_CEILING_FRAC = Z4_CEILING_FRAC
Z5_RIDGE_ALPHA = Z4_RIDGE_ALPHA
Z5_GREEDY_CAND_PER_CELL = Z4_GREEDY_CAND_PER_CELL
Z5_GREEDY_MAX_K = Z4_GREEDY_MAX_K
Z5_N_GREEDY_RESTARTS = 20
Z5_N_RANDOM_MINIMAL_TRIALS = 200
Z5_STABILITY_JACCARD_PASS = 0.60
Z5_CELL_FREQ_REPORT_MIN = 0.50
Z5_EXHAUSTIVE_MAX_CELLS = 15
Z5_OVERLAP_JACCARD_HIGH = 0.50
Z5_OVERLAP_JACCARD_LOW = 0.20
Z5_CENTROIDS_JSON = (
    PROJECT_ROOT / "results" / "population_bridge_analysis" / "va_centroids_vs_fixed_points.json"
)

# Z5 category-restricted generalization (Supplement)
Z5_CATEGORY_ORDER = ("Scene", "Person", "Object", "Animal")
Z5_CATEGORY_GEN_K = 4
Z5_CATEGORY_GEN_K_FTOM_EXTRA = 4  # additional F→M comparison at k=4

SRC_M = ["valence_male", "arousal_male"]
SRC_F = ["valence_female", "arousal_female"]

OUT_DEFAULT = PROJECT_ROOT / "results" / "anchor_structure"


@dataclass
class AffineMap:
    A: np.ndarray
    b: np.ndarray

    def apply(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        if X.ndim == 1:
            return X @ self.A.T + self.b
        return X @ self.A.T + self.b

    def fixed_point(self) -> np.ndarray:
        return np.linalg.solve(np.eye(2) - self.A, self.b)


def fit_affine(X: np.ndarray, Y: np.ndarray) -> AffineMap:
    reg = LinearRegression().fit(X, Y)
    return AffineMap(np.asarray(reg.coef_, float), np.asarray(reg.intercept_, float))


def fit_ridge_affine(X: np.ndarray, Y: np.ndarray, alpha: float = Z3_RIDGE_ALPHA) -> AffineMap:
    """Ridge-regularized multi-output affine (stabilizes small-k shared-anchor fits)."""
    reg = Ridge(alpha=float(alpha), fit_intercept=True).fit(X, Y)
    return AffineMap(np.asarray(reg.coef_, float), np.asarray(reg.intercept_, float))


def r2_dims(Y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rv = float(r2_score(Y[:, 0], pred[:, 0]))
    ra = float(r2_score(Y[:, 1], pred[:, 1]))
    return {"R2_valence": rv, "R2_arousal": ra, "R2_mean": float((rv + ra) / 2)}


def residual_l2_per_image(Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.linalg.norm(Y - pred, axis=1)


def mean_residual_l2(Y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(residual_l2_per_image(Y, pred)))


def mean_sq_error(Y: np.ndarray, pred: np.ndarray) -> float:
    """Mean per-image squared Euclidean error (what LS affine minimizes)."""
    return float(np.mean(np.sum((Y - pred) ** 2, axis=1)))


def load_gender_va(oasis_csv: Path = OASIS_SCORES_CSV) -> pd.DataFrame:
    df = load_oasis_meta(oasis_csv)
    df = add_theme_base(df)
    need = SRC_M + SRC_F
    valid = df[need].notna().all(axis=1)
    return df.loc[valid].reset_index(drop=True)


def make_theme_split(df: pd.DataFrame, split_seed: int) -> dict[str, Any]:
    train_idx, val_idx, test_idx = train_val_test_split_by_theme(
        df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, random_state=split_seed
    )
    fit_idx = np.concatenate([train_idx, val_idx]) if len(val_idx) else train_idx
    fit_idx = np.asarray(fit_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    return {
        "fit_idx": fit_idx,
        "test_idx": test_idx,
        "n_fit": int(len(fit_idx)),
        "n_test": int(len(test_idx)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "seed": int(split_seed),
        "fit_definition": "train_union_val",
    }


def assign_nearest(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # (n, k) squared distances
    d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d, axis=1)


def fit_source_kmeans(
    X_fit: np.ndarray, k: int, seed: int, n_init: int = KMEANS_N_INIT
) -> KMeans:
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    km.fit(X_fit)
    return km


def cluster_stability(
    X_fit: np.ndarray, k: int, base_seed: int, seeds: tuple[int, ...] = KMEANS_STABILITY_SEEDS
) -> dict[str, Any]:
    base = fit_source_kmeans(X_fit, k, base_seed)
    base_labels = base.labels_
    aris = []
    for s in seeds:
        if s == base_seed:
            aris.append(1.0)
            continue
        other = fit_source_kmeans(X_fit, k, s)
        aris.append(float(adjusted_rand_score(base_labels, other.labels_)))
    mean_ari = float(np.mean(aris))
    return {
        "mean_ari": mean_ari,
        "min_ari": float(np.min(aris)),
        "aris": aris,
        "pass": bool(mean_ari >= STABILITY_ARI_PASS),
    }


def fit_piecewise_affine(
    X_fit: np.ndarray, Y_fit: np.ndarray, labels_fit: np.ndarray, k: int
) -> tuple[list[AffineMap], str]:
    maps: list[AffineMap] = []
    global_map = fit_affine(X_fit, Y_fit)
    status = "ok"
    for c in range(k):
        mask = labels_fit == c
        n_c = int(mask.sum())
        if n_c < KMEANS_MIN_CLUSTER:
            status = "unstable_cluster_size"
            maps.append(global_map)
        else:
            maps.append(fit_affine(X_fit[mask], Y_fit[mask]))
    return maps, status


def predict_piecewise(X: np.ndarray, labels: np.ndarray, maps: list[AffineMap]) -> np.ndarray:
    out = np.zeros_like(X, dtype=float)
    for i in range(len(X)):
        out[i] = maps[int(labels[i])].apply(X[i])
    return out


def exact_ot_barycentric_map(Xs: np.ndarray, Yt: np.ndarray, Xq: np.ndarray) -> np.ndarray:
    """VA-space exact EMD barycentric map (same form as the OT five-point check)."""
    a = np.full(len(Xs), 1.0 / len(Xs))
    b = np.full(len(Yt), 1.0 / len(Yt))
    M = ot.dist(Xs, Yt, metric="euclidean") ** 2
    G = ot.emd(a, b, M)
    mapped_src = (G @ Yt) / np.clip(G.sum(axis=1, keepdims=True), 1e-12, None)
    Dq = ot.dist(Xq, Xs, metric="euclidean")
    nn = np.argmin(Dq, axis=1)
    return mapped_src[nn]


def paired_permutation_error(
    Y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_perm: int,
    seed: int,
) -> dict[str, float]:
    """
    Pairwise swap test on per-image squared error.
    obs = mean(eB) - mean(eA); positive => A better (lower error).
    """
    eA = np.sum((Y - pred_a) ** 2, axis=1)
    eB = np.sum((Y - pred_b) ** 2, axis=1)
    obs = float(np.mean(eB) - np.mean(eA))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        swap = rng.random(len(eA)) < 0.5
        eA_p = np.where(swap, eB, eA)
        eB_p = np.where(swap, eA, eB)
        stat = float(np.mean(eB_p) - np.mean(eA_p))
        if abs(stat) >= abs(obs):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"obs_mean_eB_minus_eA": obs, "p_paired": float(p)}


def check_ot_consistency(
    res_by_k_test: dict[int, float],
    res_ot_test: float,
    r2_k1_test: float,
    r2_ot_test: float,
    res_by_k_fit: dict[int, float] | None = None,
    res_ot_fit: float | None = None,
) -> dict[str, Any]:
    """
    OT consistency for this estimand.

    Important: exact EMD barycentric OT does **not** minimize held-out L2 residual.
    Piecewise affine can therefore beat OT on residual_L2 / R² without being a bug
    (consistent with OT ≤ linear on prediction metrics). Hard-failing when
    piecewise < OT residual was a misspecified nail-3 rule for OOS prediction.

    Pass conditions (revised):
      1. Held-out OT should not substantially beat k=1 (ΔR²_ot_minus_k1 ≤ DELTA_R2_WIN),
         aligning with OT≤linear.
      2. Soft k-monotonicity on test residual (≤1 upward violation) — warning only if fail.
    Informational: whether piecewise residuals fall below OT (expected possible).
    """
    ks = sorted(res_by_k_test)
    vals = [res_by_k_test[k] for k in ks]
    piecewise_below_ot = any(res_by_k_test[k] < res_ot_test - OT_BEAT_EPS for k in ks)
    n_violations = sum(1 for i in range(1, len(vals)) if vals[i] > vals[i - 1] + MONOTONE_TOL)
    delta_ot_minus_k1 = float(r2_ot_test - r2_k1_test)
    ot_beats_linear = bool(delta_ot_minus_k1 > DELTA_R2_WIN)

    # Contradiction only if OT clearly beats linear on held-out prediction (vs paper claim)
    contradiction = ot_beats_linear
    pass_flag = not contradiction

    out: dict[str, Any] = {
        "rule": "ot_should_not_beat_linear_heldout; piecewise_below_ot_is_informational",
        "contradiction_ot_beats_linear": contradiction,
        "piecewise_residual_below_ot_test": bool(piecewise_below_ot),
        "n_upward_violations_k": int(n_violations),
        "soft_k_monotone": bool(n_violations <= 1),
        "delta_R2_ot_minus_k1": delta_ot_minus_k1,
        "pass": bool(pass_flag),
        "residual_ot_test": float(res_ot_test),
        "residuals_by_k_test": {str(k): float(res_by_k_test[k]) for k in ks},
    }
    if res_by_k_fit is not None and res_ot_fit is not None:
        out["residual_ot_fit"] = float(res_ot_fit)
        out["residuals_by_k_fit"] = {str(k): float(res_by_k_fit[k]) for k in sorted(res_by_k_fit)}
        out["piecewise_residual_below_ot_fit"] = any(
            res_by_k_fit[k] < res_ot_fit - OT_BEAT_EPS for k in res_by_k_fit
        )
    return out


def looks_like_slow_climb(win_rows: list[dict], mono: dict, k_max: int) -> bool:
    by_k = {int(r["k"]): r for r in win_rows}
    if 1 not in by_k or k_max not in by_k:
        return False
    d = float(by_k[k_max]["R2_mean_test"] - by_k[1]["R2_mean_test"])
    res6 = float(by_k[k_max]["residual_l2_test"])
    res_ot = float(mono["residual_ot_test"])
    return bool(d > SLOW_CLIMB_DELTA and (res6 - res_ot) < SLOW_CLIMB_OT_GAP)


def decide_outcome(win_rows: list[dict], mono: dict, k_max: int) -> dict[str, Any]:
    if not mono["pass"]:
        return {
            "outcome": "CONTRADICTION",
            "k_star": None,
            "narrative": "Held-out OT substantially beats k=1 (unexpected vs OT≤linear); bug check",
        }
    winners = [r for r in win_rows if r.get("win_vs_k1") and int(r["k"]) > 1]
    if not winners:
        if looks_like_slow_climb(win_rows, mono, k_max):
            return {
                "outcome": "C_WATCH",
                "k_star": 1,
                "narrative": "Slow climb without crossing win threshold; reconcile carefully",
            }
        return {
            "outcome": "A",
            "k_star": 1,
            "narrative": "Linear sufficiency holds (k=1 exhausts useful degrees of freedom)",
        }
    k_star = int(min(int(r["k"]) for r in winners))
    note = ""
    if mono.get("piecewise_residual_below_ot_test"):
        note = " (piecewise < OT residual is allowed for prediction estimand)"
    return {
        "outcome": "B",
        "k_star": k_star,
        "narrative": (
            f"Few anchors needed (k*={k_star}); claim only with OT≤linear consistency"
            f" and parameter penalty{note}"
        ),
    }


def run_one_direction(
    df: pd.DataFrame,
    split: dict[str, Any],
    src_cols: list[str],
    tgt_cols: list[str],
    bridge_name: str,
    k_list: tuple[int, ...],
    kmeans_seed: int,
    n_perm: int,
) -> dict[str, Any]:
    fit_idx = split["fit_idx"]
    test_idx = split["test_idx"]
    X_fit = df.iloc[fit_idx][src_cols].to_numpy(float)
    Y_fit = df.iloc[fit_idx][tgt_cols].to_numpy(float)
    X_test = df.iloc[test_idx][src_cols].to_numpy(float)
    Y_test = df.iloc[test_idx][tgt_cols].to_numpy(float)

    methods: dict[str, Any] = {}
    preds_test: dict[str, np.ndarray] = {}
    win_rows: list[dict[str, Any]] = []
    stability: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    per_image_rows: list[dict[str, Any]] = []

    # --- k sweep ---
    for k in k_list:
        status = "ok"
        if k == 1:
            phi = fit_affine(X_fit, Y_fit)
            pred_fit = phi.apply(X_fit)
            pred_te = phi.apply(X_test)
            n_params = N_PARAMS_PER_AFFINE
            labels_te = np.zeros(len(X_test), dtype=int)
            stab = {"mean_ari": 1.0, "min_ari": 1.0, "pass": True, "aris": [1.0]}
        else:
            km = fit_source_kmeans(X_fit, k, kmeans_seed)
            labels_fit = km.labels_
            maps, status = fit_piecewise_affine(X_fit, Y_fit, labels_fit, k)
            labels_te = assign_nearest(X_test, km.cluster_centers_)
            pred_fit = predict_piecewise(X_fit, labels_fit, maps)
            pred_te = predict_piecewise(X_test, labels_te, maps)
            n_params = N_PARAMS_PER_AFFINE * k
            stab = cluster_stability(X_fit, k, kmeans_seed)

        metrics_te = r2_dims(Y_test, pred_te)
        metrics_fit = r2_dims(Y_fit, pred_fit)
        res_te = mean_residual_l2(Y_test, pred_te)
        res_fit = mean_residual_l2(Y_fit, pred_fit)

        methods[str(k)] = {
            "n_params": int(n_params),
            "status": status,
            **{f"{kk}_test": vv for kk, vv in metrics_te.items()},
            **{f"{kk}_fit": vv for kk, vv in metrics_fit.items()},
            "residual_l2_test": res_te,
            "residual_l2_fit": res_fit,
        }
        preds_test[str(k)] = pred_te
        stability[str(k)] = stab

        for i, idx in enumerate(test_idx):
            per_image_rows.append(
                {
                    "bridge": bridge_name,
                    "method": f"k{k}",
                    "image_idx": int(idx),
                    "residual_l2": float(np.linalg.norm(Y_test[i] - pred_te[i])),
                    "cluster": int(labels_te[i]) if k > 1 else 0,
                }
            )

    # --- OT ---
    pred_ot_te = exact_ot_barycentric_map(X_fit, Y_fit, X_test)
    pred_ot_fit = exact_ot_barycentric_map(X_fit, Y_fit, X_fit)
    metrics_ot = r2_dims(Y_test, pred_ot_te)
    res_ot = mean_residual_l2(Y_test, pred_ot_te)
    res_ot_fit = mean_residual_l2(Y_fit, pred_ot_fit)
    methods["ot"] = {
        "n_params": None,
        "status": "ok",
        "note": "VA-space exact EMD barycentric; not identical to the decoder-space OT check",
        **{f"{kk}_test": vv for kk, vv in metrics_ot.items()},
        "residual_l2_test": res_ot,
        "residual_l2_fit": res_ot_fit,
    }
    preds_test["ot"] = pred_ot_te
    for i, idx in enumerate(test_idx):
        per_image_rows.append(
            {
                "bridge": bridge_name,
                "method": "ot",
                "image_idx": int(idx),
                "residual_l2": float(np.linalg.norm(Y_test[i] - pred_ot_te[i])),
                "cluster": -1,
            }
        )

    # --- comparisons vs k=1 ---
    pred_k1 = preds_test["1"]
    r2_k1 = methods["1"]["R2_mean_test"]
    for k in k_list:
        key = str(k)
        delta = float(methods[key]["R2_mean_test"] - r2_k1)
        if k == 1:
            p_paired = 1.0
            win = False
        else:
            perm = paired_permutation_error(
                Y_test,
                preds_test[key],
                pred_k1,
                n_perm=n_perm,
                seed=split["seed"] + 1000 + k,
            )
            p_paired = perm["p_paired"]
            comparisons[key] = {**perm, "delta_R2_mean": delta}
            win = bool(
                delta > DELTA_R2_WIN
                and p_paired < ALPHA_PERM
                and methods[key]["status"] == "ok"
            )
        row = {
            "bridge": bridge_name,
            "k": int(k),
            "n_params": int(methods[key]["n_params"]),
            "R2_mean_test": float(methods[key]["R2_mean_test"]),
            "residual_l2_test": float(methods[key]["residual_l2_test"]),
            "delta_R2_vs_k1": delta,
            "p_paired_vs_k1": float(p_paired),
            "win_vs_k1": win,
            "cluster_mean_ari": float(stability[key].get("mean_ari", 1.0)),
            "status": methods[key]["status"],
        }
        win_rows.append(row)

    # OT row for table completeness
    win_rows.append(
        {
            "bridge": bridge_name,
            "k": "ot",
            "n_params": None,
            "R2_mean_test": float(methods["ot"]["R2_mean_test"]),
            "residual_l2_test": float(res_ot),
            "delta_R2_vs_k1": float(methods["ot"]["R2_mean_test"] - r2_k1),
            "p_paired_vs_k1": np.nan,
            "win_vs_k1": False,
            "cluster_mean_ari": np.nan,
            "status": "ok",
        }
    )

    res_by_k = {int(k): float(methods[str(k)]["residual_l2_test"]) for k in k_list}
    res_by_k_fit = {int(k): float(methods[str(k)]["residual_l2_fit"]) for k in k_list}
    mono = check_ot_consistency(
        res_by_k_test=res_by_k,
        res_ot_test=res_ot,
        r2_k1_test=r2_k1,
        r2_ot_test=float(methods["ot"]["R2_mean_test"]),
        res_by_k_fit=res_by_k_fit,
        res_ot_fit=res_ot_fit,
    )
    # Win requires OT≤linear consistency pass
    for row in win_rows:
        if row["k"] != "ot" and int(row["k"]) > 1:
            row["win_vs_k1"] = bool(row["win_vs_k1"] and mono["pass"])

    decision = decide_outcome(
        [r for r in win_rows if r["k"] != "ot"], mono, k_max=max(k_list)
    )

    return {
        "bridge": bridge_name,
        "src_cols": src_cols,
        "tgt_cols": tgt_cols,
        "methods": methods,
        "comparisons_vs_k1": comparisons,
        "stability": stability,
        "monotonicity": mono,
        "decision": decision,
        "win_rows": win_rows,
        "per_image_rows": per_image_rows,
    }


def plot_ksweep(results_by_bridge: dict[str, dict], out_path: Path, k_list: tuple[int, ...]) -> None:
    bridges = list(results_by_bridge.keys())
    fig, axes = plt.subplots(2, len(bridges), figsize=(5.2 * len(bridges), 7.2), squeeze=False)
    for col, name in enumerate(bridges):
        res = results_by_bridge[name]
        methods = res["methods"]
        ks = list(k_list)
        r2s = [methods[str(k)]["R2_mean_test"] for k in ks]
        ress = [methods[str(k)]["residual_l2_test"] for k in ks]
        r2_ot = methods["ot"]["R2_mean_test"]
        res_ot = methods["ot"]["residual_l2_test"]
        r2_k1 = methods["1"]["R2_mean_test"]

        ax = axes[0, col]
        ax.plot(ks, r2s, "o-", color="#1565c0", label="piecewise affine")
        ax.axhline(r2_ot, color="#c62828", ls="--", label="exact OT")
        ax.axhline(r2_k1 + DELTA_R2_WIN, color="#2e7d32", ls=":", alpha=0.8, label=f"k1+{DELTA_R2_WIN}")
        ax.set_xlabel("k")
        ax.set_ylabel("held-out R² mean")
        ax.set_title(f"{name}\nOutcome={res['decision']['outcome']}")
        ax.legend(fontsize=7, loc="best")
        ax.set_xticks(ks)

        ax = axes[1, col]
        ax.plot(ks, ress, "o-", color="#1565c0", label="piecewise affine")
        ax.axhline(res_ot, color="#c62828", ls="--", label="exact OT")
        ax.set_xlabel("k")
        ax.set_ylabel("held-out residual L2")
        ax.legend(fontsize=7, loc="best")
        ax.set_xticks(ks)

    fig.suptitle("Z2 k-sweep: piecewise affine vs VA-space exact OT", y=0.995, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def write_summary(
    out_dir: Path,
    split: dict[str, Any],
    results_by_bridge: dict[str, dict],
    k_list: tuple[int, ...],
    z1_payload: dict[str, Any] | None = None,
    z3_payload: dict[str, Any] | None = None,
) -> None:
    lines = [
        "# Anchor structure analysis summary",
        "",
        "## Estimand / pre-registration (Z2)",
        "",
        "- Bridges: **gender M→F and F→M** (asymmetric; both run).",
        "- Fit = train∪val themes; Eval = held-out themes only.",
        f"- Win rule (pre-registered): held-out ΔR² > **{DELTA_R2_WIN}** AND paired permutation p < **{ALPHA_PERM}**, AND OT consistency pass.",
        "- Clustering: k-means on **source** VA; test assigned by nearest center (no refit).",
        "- OT: VA-space exact EMD barycentric (not the decoder-space OT check).",
        "- OT consistency (revised): held-out OT must not substantially beat k=1; "
        "piecewise residual < OT is informational (different estimand), not a hard fail.",
        "- **Outcome B policy (locked):** keep linear-sufficiency pillar; M→F k*=2 is "
        "refinement / limit measurement, not a pillar rewrite; F→M Outcome A = directional asymmetry.",
        "",
        "## Split",
        "",
        f"- seed={split['seed']}, n_fit={split['n_fit']}, n_test={split['n_test']} "
        f"(train={split['n_train']}, val={split['n_val']})",
        "",
    ]
    for name, res in results_by_bridge.items():
        lines += [
            f"## Z2 Bridge `{name}`",
            "",
            f"- Decision: **{res['decision']['outcome']}** (k\*={res['decision']['k_star']})",
            f"- Narrative: {res['decision']['narrative']}",
            f"- OT consistency pass: **{res['monotonicity']['pass']}** "
            f"(OT_beats_linear={res['monotonicity']['contradiction_ot_beats_linear']}, "
            f"ΔR²(OT−k1)={res['monotonicity']['delta_R2_ot_minus_k1']:+.4f}, "
            f"piecewise_below_OT={res['monotonicity']['piecewise_residual_below_ot_test']}, "
            f"k_upward_violations={res['monotonicity']['n_upward_violations_k']})",
            "",
            "| k | n_params | R²_test | residual_L2 | ΔR² vs k1 | p_paired | win | mean ARI | status |",
            "|---|----------|---------|-------------|-----------|----------|-----|----------|--------|",
        ]
        for row in res["win_rows"]:
            k = row["k"]
            npar = row["n_params"] if row["n_params"] is not None else "—"
            p = row["p_paired_vs_k1"]
            p_s = "—" if (p is None or (isinstance(p, float) and np.isnan(p))) else f"{p:.4f}"
            ari = row["cluster_mean_ari"]
            ari_s = "—" if (ari is None or (isinstance(ari, float) and np.isnan(ari))) else f"{ari:.3f}"
            lines.append(
                f"| {k} | {npar} | {row['R2_mean_test']:.4f} | {row['residual_l2_test']:.4f} | "
                f"{row['delta_R2_vs_k1']:+.4f} | {p_s} | {row['win_vs_k1']} | {ari_s} | {row['status']} |"
            )
        lines.append("")
        warns = [
            f"k={k}: mean_ari={st['mean_ari']:.3f}"
            for k, st in res["stability"].items()
            if int(k) > 1 and not st.get("pass", True)
        ]
        if warns:
            lines += ["**Stability warnings:** " + "; ".join(warns), ""]

    lines += [
        "## Culture bridge",
        "",
        "- Not run here. After gender k\* is locked, optional one-point replication only (Supplement).",
        "",
        "## Asymmetry note",
        "",
        "- Affine / piecewise / OT maps are direction-dependent. M→F and F→M need not share the same k\* or residual curve.",
        "",
    ]

    if z1_payload is not None:
        lines += [
            "## Z1 Anchor-position sweep",
            "",
            "- Constrained map: Φ_f(y)=A(y−f)+f (A free, 4 params); same theme split as Z2.",
            "- Stage1 coarse 5×5 on [2,6]²; Stage2 fine ±1.0 at step 0.25.",
            "- Bootstrap CI: coarse-grid argmin only (themes resampled within fit; test fixed).",
            "",
        ]
        for key, br in z1_payload["bridges"].items():
            fh = br["f_hat_unconstrained_fit"]
            ff = br["f_fine"]
            fc = br["f_coarse"]
            lines += [
                f"### `{key}`",
                "",
                f"- f̂ (unconstrained fit-set FP): ({fh[0]:.3f}, {fh[1]:.3f})",
                f"- f_coarse (L2): ({fc[0]:.3f}, {fc[1]:.3f})",
                f"- f_fine (L2): ({ff[0]:.3f}, {ff[1]:.3f}); f_fine (MSE): "
                f"({br['f_fine_mse'][0]:.3f}, {br['f_fine_mse'][1]:.3f})",
                f"- ‖f_fine(L2) − f̂‖ = {br['dist_fine_to_fhat']:.3f}; "
                f"‖f_fine(MSE) − f̂‖ = {br['dist_fine_mse_to_fhat']:.3f}",
                f"- ‖f_fine(L2) − data centroid‖ = {br['dist_fine_to_centroid']:.3f}",
                f"- resid_L2_test(f_fine)={br['residual_l2_test_fine']:.4f} vs unconstrained "
                f"{br['residual_l2_test_unconstrained']:.4f} "
                f"(Δ={br['consistency']['delta_l2_test_fine_minus_unconstrained']:+.4f})",
                f"- Nesting (MSE): f̂ match={br['consistency']['pass_fhat_matches_unconstrained_fit']}; "
                f"soft test={br['consistency']['pass_test_soft']}",
                f"- Flatness (neighbor residual range on fine grid): {br['flatness']['neighbor_residual_range']:.4f}",
                f"- f̂ inside coarse-boot 95% box of L2 argmin: **{br['bootstrap']['fhat_in_ci_box']}**",
                "",
            ]
        lines += [
            "### Z1 narrative hooks",
            "",
            "- Nesting check uses **MSE** (LS objective). Primary heatmap argmin uses held-out **mean L2** (can differ from f̂).",
            "- If f_fine(L2) sits on the grid boundary with tiny neighbor range, treat as flat / extrapolation caution; prefer f̂ and f_fine(MSE) for “optimal LS anchor”.",
            "- If f_fine(MSE)≈f̂≈mid-scale: single optimal LS anchor naturally near neutral range.",
            "- Directional differences in residual terrain support Z2 asymmetry (M→F vs F→M).",
            "",
        ]

    if z3_payload is not None:
        lines += [
            "## Z3 Shared-anchor generalization",
            "",
            "- Operation: fit map on k anchors ⊂ fit; evaluate understanding = R²_mean on theme-held-out test.",
            "- k<3: translation (A=I); k≥3: **ridge-stabilized affine** (main; naive LS is diagnostic only).",
            "- Similarity layers: tertiles of ‖y_src−y_tgt‖₂ on test.",
            "",
        ]
        tw = z3_payload.get("far_vs_excess_twist")
        if tw and tw.get("status") == "ok":
            lines += [
                "### Far layer vs excess twist",
                "",
                f"- Spearman(gap, excess_twist): ρ={tw['spearman_gap_vs_excess_twist']['rho']:.3f}, "
                f"p={tw['spearman_gap_vs_excess_twist']['p']:.4g}",
                f"- mean excess_twist near/mid/far: "
                f"{tw['excess_twist_mean']['near']:.3f} / {tw['excess_twist_mean']['mid']:.3f} / "
                f"{tw['excess_twist_mean']['far']:.3f}",
                f"- MWU far>near: p={tw['mwu_far_gt_near']['p']:.4g}",
                f"- frac in lowV–lowA: far={tw['frac_in_lowV_lowA']['far']:.3f}, "
                f"near={tw['frac_in_lowV_lowA']['near']:.3f}",
                "",
            ]
        for name, br in z3_payload.get("bridges", {}).items():
            lines += [
                f"### `{name}`",
                "",
                f"- Affine mode: **{br.get('affine_mode')}** (α={br.get('ridge_alpha')})",
                f"- Ceiling (unconstrained fit→test R²): **{br['ceiling_R2_mean']:.4f}**",
                f"- k* stable (median ≥ {Z3_CEILING_FRAC:.0%}×ceiling and sustained): **{br.get('k_star')}** "
                f"(naive first-crossing: {br.get('k_star_naive_first_crossing')})",
                f"- Outcome tag (heuristic): **{br.get('outcome_tag')}**",
                f"- Note: k=3 full-affine is often unstable (small-n LS); translation k=1–2 can look strong then dip.",
                f"- Layer thresholds (gap): {br.get('layer_thresholds')}",
                "",
                "| k | R² median | CI | near | mid | far | map |",
                "|---|-----------|----|------|-----|-----|-----|",
            ]
            for row in br.get("curve_rows", []):
                lines.append(
                    f"| {row['k']} | {row['R2_median']:.4f} | "
                    f"[{row['R2_q025']:.4f}, {row['R2_q975']:.4f}] | "
                    f"{row.get('R2_near_median', float('nan')):.4f} | "
                    f"{row.get('R2_mid_median', float('nan')):.4f} | "
                    f"{row.get('R2_far_median', float('nan')):.4f} | "
                    f"{row.get('map_type_majority', '')} |"
                )
            lines.append("")
            if br.get("selection_rows"):
                lines += [
                    "| k | method | R² |",
                    "|---|--------|----|",
                ]
                for row in br["selection_rows"]:
                    lines.append(f"| {row['k']} | {row['method']} | {row['R2_mean']:.4f} |")
                lines.append("")

    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Z1: fixed-point / anchor-position sweep
# ---------------------------------------------------------------------------

def fit_fixedpoint_affine(y_src: np.ndarray, y_tgt: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Estimate A for Φ_f(y)=A(y−f)+f by LS on centered coordinates."""
    f = np.asarray(f, float).reshape(2)
    U = (y_src - f).T  # 2 x n
    V = (y_tgt - f).T
    gram = U @ U.T + Z1_RIDGE * np.eye(2)
    A = (V @ U.T) @ np.linalg.inv(gram)
    return np.asarray(A, float)


def apply_fixedpoint_affine(y_src: np.ndarray, A: np.ndarray, f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, float).reshape(2)
    return (y_src - f) @ A.T + f


def eval_heldout_at_fixedpoint(
    y_src_fit: np.ndarray,
    y_tgt_fit: np.ndarray,
    y_src_te: np.ndarray,
    y_tgt_te: np.ndarray,
    f: np.ndarray,
) -> dict[str, float]:
    A = fit_fixedpoint_affine(y_src_fit, y_tgt_fit, f)
    pred_te = apply_fixedpoint_affine(y_src_te, A, f)
    pred_fit = apply_fixedpoint_affine(y_src_fit, A, f)
    m_te = r2_dims(y_tgt_te, pred_te)
    return {
        "residual_l2_test": mean_residual_l2(y_tgt_te, pred_te),
        "residual_l2_fit": mean_residual_l2(y_tgt_fit, pred_fit),
        "mse_test": mean_sq_error(y_tgt_te, pred_te),
        "mse_fit": mean_sq_error(y_tgt_fit, pred_fit),
        "R2_mean_test": m_te["R2_mean"],
        "R2_valence_test": m_te["R2_valence"],
        "R2_arousal_test": m_te["R2_arousal"],
        "n_test": float(len(y_tgt_te)),
    }


def make_coarse_grid() -> np.ndarray:
    vs = np.array(Z1_COARSE_VALS, float)
    vv, aa = np.meshgrid(vs, vs, indexing="xy")
    return np.column_stack([vv.ravel(), aa.ravel()])


def make_fine_grid(f_coarse: np.ndarray) -> np.ndarray:
    f = np.asarray(f_coarse, float).reshape(2)
    xs = np.arange(f[0] - Z1_FINE_HALF_WIDTH, f[0] + Z1_FINE_HALF_WIDTH + 1e-9, Z1_FINE_STEP)
    ys = np.arange(f[1] - Z1_FINE_HALF_WIDTH, f[1] + Z1_FINE_HALF_WIDTH + 1e-9, Z1_FINE_STEP)
    # clip to [1,7] scale
    xs = np.clip(xs, 1.0, 7.0)
    ys = np.clip(ys, 1.0, 7.0)
    vv, aa = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([vv.ravel(), aa.ravel()])
    # unique after clip
    pts = np.unique(np.round(pts, 6), axis=0)
    return pts


def sweep_fixedpoint_grid(
    grid_pts: np.ndarray,
    y_src_fit: np.ndarray,
    y_tgt_fit: np.ndarray,
    y_src_te: np.ndarray,
    y_tgt_te: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for f in grid_pts:
        ev = eval_heldout_at_fixedpoint(y_src_fit, y_tgt_fit, y_src_te, y_tgt_te, f)
        rows.append({"f_v": float(f[0]), "f_a": float(f[1]), **ev})
    return pd.DataFrame(rows)


def flatness_around(fine_df: pd.DataFrame, f_fine: np.ndarray, radius: float = 0.5) -> dict[str, float]:
    fv, fa = float(f_fine[0]), float(f_fine[1])
    d = np.sqrt((fine_df["f_v"] - fv) ** 2 + (fine_df["f_a"] - fa) ** 2)
    neigh = fine_df.loc[d <= radius + 1e-9, "residual_l2_test"]
    if len(neigh) < 2:
        return {"neighbor_residual_range": float("nan"), "neighbor_n": int(len(neigh))}
    return {
        "neighbor_residual_range": float(neigh.max() - neigh.min()),
        "neighbor_residual_std": float(neigh.std(ddof=1)),
        "neighbor_n": int(len(neigh)),
        "radius": float(radius),
    }


def bootstrap_coarse_argmin(
    df: pd.DataFrame,
    fit_idx: np.ndarray,
    test_idx: np.ndarray,
    src_cols: list[str],
    tgt_cols: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Theme bootstrap within fit themes; evaluate on fixed test; coarse argmin only."""
    themes = df.iloc[fit_idx]["theme_base"].to_numpy()
    uniq = np.unique(themes)
    y_src_te = df.iloc[test_idx][src_cols].to_numpy(float)
    y_tgt_te = df.iloc[test_idx][tgt_cols].to_numpy(float)
    grid = make_coarse_grid()
    rng = np.random.default_rng(seed)
    mins = []
    for _ in range(n_boot):
        boot_themes = rng.choice(uniq, size=len(uniq), replace=True)
        # images in fit whose theme is in the multiset: include each selected theme's images once per unique
        mask = df["theme_base"].isin(set(boot_themes)).to_numpy()
        boot_fit = np.intersect1d(fit_idx, np.where(mask)[0])
        if len(boot_fit) < 20:
            continue
        y_src_fit = df.iloc[boot_fit][src_cols].to_numpy(float)
        y_tgt_fit = df.iloc[boot_fit][tgt_cols].to_numpy(float)
        best_f, best_r = None, np.inf
        for f in grid:
            ev = eval_heldout_at_fixedpoint(y_src_fit, y_tgt_fit, y_src_te, y_tgt_te, f)
            if ev["residual_l2_test"] < best_r:
                best_r = ev["residual_l2_test"]
                best_f = f.copy()
        if best_f is not None:
            mins.append(best_f)
    mins_arr = np.asarray(mins, float) if mins else np.zeros((0, 2))
    if len(mins_arr) < 5:
        return {
            "n_boot_ok": int(len(mins_arr)),
            "mean": [float("nan"), float("nan")],
            "ci_v": [float("nan"), float("nan")],
            "ci_a": [float("nan"), float("nan")],
            "cov": None,
            "fhat_in_ci_box": False,
        }
    mean = mins_arr.mean(axis=0)
    ci_v = [float(np.quantile(mins_arr[:, 0], 0.025)), float(np.quantile(mins_arr[:, 0], 0.975))]
    ci_a = [float(np.quantile(mins_arr[:, 1], 0.025)), float(np.quantile(mins_arr[:, 1], 0.975))]
    cov = np.cov(mins_arr.T, ddof=1)
    return {
        "n_boot_ok": int(len(mins_arr)),
        "mean": [float(mean[0]), float(mean[1])],
        "ci_v": ci_v,
        "ci_a": ci_a,
        "cov": cov.tolist(),
        "samples": mins_arr.tolist(),
    }


def run_z1_one_direction(
    df: pd.DataFrame,
    split: dict[str, Any],
    src_cols: list[str],
    tgt_cols: list[str],
    bridge_name: str,
    short_name: str,
    out_dir: Path,
    n_boot: int,
    boot_seed: int,
) -> dict[str, Any]:
    fit_idx = split["fit_idx"]
    test_idx = split["test_idx"]
    y_src_fit = df.iloc[fit_idx][src_cols].to_numpy(float)
    y_tgt_fit = df.iloc[fit_idx][tgt_cols].to_numpy(float)
    y_src_te = df.iloc[test_idx][src_cols].to_numpy(float)
    y_tgt_te = df.iloc[test_idx][tgt_cols].to_numpy(float)

    # Unconstrained affine on fit set
    phi = fit_affine(y_src_fit, y_tgt_fit)
    f_hat = phi.fixed_point()
    pred_u_te = phi.apply(y_src_te)
    pred_u_fit = phi.apply(y_src_fit)
    resid_u_te = mean_residual_l2(y_tgt_te, pred_u_te)
    resid_u_fit = mean_residual_l2(y_tgt_fit, pred_u_fit)
    r2_u = r2_dims(y_tgt_te, pred_u_te)

    # Data centroid (source VA on all valid / fit)
    centroid = y_src_fit.mean(axis=0)

    # Stage 1 coarse (primary metric: held-out residual L2, per spec)
    coarse_df = sweep_fixedpoint_grid(
        make_coarse_grid(), y_src_fit, y_tgt_fit, y_src_te, y_tgt_te
    )
    i_c = int(coarse_df["residual_l2_test"].idxmin())
    f_coarse = coarse_df.loc[i_c, ["f_v", "f_a"]].to_numpy(float)
    i_c_mse = int(coarse_df["mse_test"].idxmin())
    f_coarse_mse = coarse_df.loc[i_c_mse, ["f_v", "f_a"]].to_numpy(float)

    # Stage 2 fine around L2 coarse min
    fine_df = sweep_fixedpoint_grid(
        make_fine_grid(f_coarse), y_src_fit, y_tgt_fit, y_src_te, y_tgt_te
    )
    i_f = int(fine_df["residual_l2_test"].idxmin())
    f_fine = fine_df.loc[i_f, ["f_v", "f_a"]].to_numpy(float)
    resid_fine_te = float(fine_df.loc[i_f, "residual_l2_test"])
    resid_fine_fit = float(fine_df.loc[i_f, "residual_l2_fit"])
    mse_fine_te = float(fine_df.loc[i_f, "mse_test"])
    mse_fine_fit = float(fine_df.loc[i_f, "mse_fit"])
    r2_fine = float(fine_df.loc[i_f, "R2_mean_test"])

    # MSE argmin on a fine grid centered at unconstrained f_hat (recovers LS nesting)
    fine_mse_df = sweep_fixedpoint_grid(
        make_fine_grid(f_hat), y_src_fit, y_tgt_fit, y_src_te, y_tgt_te
    )
    i_fm = int(fine_mse_df["mse_test"].idxmin())
    f_fine_mse = fine_mse_df.loc[i_fm, ["f_v", "f_a"]].to_numpy(float)

    # Also evaluate exactly at f_hat (constrained)
    ev_hat = eval_heldout_at_fixedpoint(y_src_fit, y_tgt_fit, y_src_te, y_tgt_te, f_hat)

    resid_u_mse_fit = mean_sq_error(y_tgt_fit, pred_u_fit)
    resid_u_mse_te = mean_sq_error(y_tgt_te, pred_u_te)

    # Nesting is w.r.t. MSE (LS objective), not mean L2
    pass_fit = bool(mse_fine_fit >= resid_u_mse_fit - Z1_CONSISTENCY_EPS)
    match_fhat = bool(abs(ev_hat["mse_fit"] - resid_u_mse_fit) <= Z1_CONSISTENCY_EPS)
    pass_test_soft = bool(mse_fine_te >= resid_u_mse_te - Z1_CONSISTENCY_EPS)
    # Also check MSE-optimal grid point near f_hat
    mse_at_fm = float(fine_mse_df.loc[i_fm, "mse_fit"])
    pass_fit_mse_argmin = bool(mse_at_fm >= resid_u_mse_fit - Z1_CONSISTENCY_EPS)

    flat = flatness_around(fine_df, f_fine)
    boot = bootstrap_coarse_argmin(
        df, fit_idx, test_idx, src_cols, tgt_cols, n_boot=n_boot, seed=boot_seed
    )
    fhat_in_box = bool(
        boot["ci_v"][0] <= f_hat[0] <= boot["ci_v"][1]
        and boot["ci_a"][0] <= f_hat[1] <= boot["ci_a"][1]
    )
    boot["fhat_in_ci_box"] = fhat_in_box

    coarse_path = out_dir / f"z1_{short_name}_residual_grid_coarse.csv"
    fine_path = out_dir / f"z1_{short_name}_residual_grid_fine.csv"
    coarse_df.to_csv(coarse_path, index=False)
    fine_df.to_csv(fine_path, index=False)

    return {
        "bridge": bridge_name,
        "short_name": short_name,
        "metric_note": (
            "Primary landscape argmin uses held-out mean residual L2 (spec). "
            "Nesting consistency uses MSE (LS objective). L2-argmin need not equal f̂."
        ),
        "f_hat_unconstrained_fit": [float(f_hat[0]), float(f_hat[1])],
        "f_hat_canon_registry_M_to_F": list(CANON_FP_M_TO_F),
        "f_coarse_l2": [float(f_coarse[0]), float(f_coarse[1])],
        "f_coarse_mse": [float(f_coarse_mse[0]), float(f_coarse_mse[1])],
        "f_coarse": [float(f_coarse[0]), float(f_coarse[1])],
        "f_fine": [float(f_fine[0]), float(f_fine[1])],
        "f_fine_l2": [float(f_fine[0]), float(f_fine[1])],
        "f_fine_mse": [float(f_fine_mse[0]), float(f_fine_mse[1])],
        "dist_fine_to_fhat": float(np.linalg.norm(f_fine - f_hat)),
        "dist_fine_mse_to_fhat": float(np.linalg.norm(f_fine_mse - f_hat)),
        "dist_fine_to_centroid": float(np.linalg.norm(f_fine - centroid)),
        "dist_fhat_to_centroid": float(np.linalg.norm(f_hat - centroid)),
        "data_centroid_src_fit": [float(centroid[0]), float(centroid[1])],
        "residual_l2_test_fine": resid_fine_te,
        "residual_l2_fit_fine": resid_fine_fit,
        "mse_test_fine": mse_fine_te,
        "mse_fit_fine": mse_fine_fit,
        "R2_mean_test_fine": r2_fine,
        "residual_l2_test_unconstrained": resid_u_te,
        "residual_l2_fit_unconstrained": resid_u_fit,
        "mse_test_unconstrained": resid_u_mse_te,
        "mse_fit_unconstrained": resid_u_mse_fit,
        "R2_mean_test_unconstrained": r2_u["R2_mean"],
        "residual_l2_test_at_fhat_constrained": ev_hat["residual_l2_test"],
        "residual_l2_fit_at_fhat_constrained": ev_hat["residual_l2_fit"],
        "mse_fit_at_fhat_constrained": ev_hat["mse_fit"],
        "consistency": {
            "eps": Z1_CONSISTENCY_EPS,
            "objective": "mse",
            "pass_fit": pass_fit,
            "pass_fit_mse_argmin_near_fhat": pass_fit_mse_argmin,
            "pass_fhat_matches_unconstrained_fit": match_fhat,
            "pass_test_soft": pass_test_soft,
            "delta_mse_fit_fine_minus_unconstrained": float(mse_fine_fit - resid_u_mse_fit),
            "delta_mse_test_fine_minus_unconstrained": float(mse_fine_te - resid_u_mse_te),
            "delta_l2_test_fine_minus_unconstrained": float(resid_fine_te - resid_u_te),
        },
        "flatness": flat,
        "bootstrap": {k: v for k, v in boot.items() if k != "samples"},
        "bootstrap_n_samples": boot.get("n_boot_ok", 0),
        "paths": {"coarse_csv": str(coarse_path), "fine_csv": str(fine_path)},
        "_fine_df": fine_df,
        "_coarse_df": coarse_df,
        "_boot_samples": boot.get("samples"),
        "_y_src_all_for_density": df[src_cols].to_numpy(float),
    }


def plot_z1_heatmaps(z1_bridges: dict[str, dict], out_path: Path) -> None:
    from matplotlib.patches import Rectangle

    names = list(z1_bridges.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5.8 * len(names), 5.2), squeeze=False)
    for col, name in enumerate(names):
        br = z1_bridges[name]
        fine = br["_fine_df"]
        coarse = br["_coarse_df"]
        ax = axes[0, col]

        piv = fine.pivot_table(index="f_a", columns="f_v", values="residual_l2_test")
        xs = piv.columns.to_numpy(float)
        ys = piv.index.to_numpy(float)
        Z = piv.to_numpy(dtype=float)
        if Z.ndim != 2 or Z.size == 0 or np.all(np.isnan(Z)):
            ax.text(0.5, 0.5, "empty grid", transform=ax.transAxes, ha="center")
            continue
        im = ax.contourf(xs, ys, Z, levels=14, cmap="RdYlBu_r")
        ax.contour(xs, ys, Z, levels=8, colors="k", linewidths=0.35, alpha=0.4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="held-out residual L2")

        dens = br["_y_src_all_for_density"]
        ax.scatter(dens[:, 0], dens[:, 1], s=3, c="0.45", alpha=0.07, zorder=0, rasterized=True)

        fh = br["f_hat_unconstrained_fit"]
        ff = br["f_fine"]
        fm = br["f_fine_mse"]
        fc = br["f_coarse"]
        ax.plot(fh[0], fh[1], "x", ms=12, mew=2.2, color="#000", label="f̂ unconstrained", zorder=5)
        ax.plot(ff[0], ff[1], "o", ms=9, mfc="none", mec="#d32f2f", mew=2, label="f_fine (L2)", zorder=5)
        ax.plot(fm[0], fm[1], "D", ms=7, mfc="none", mec="#ef6c00", mew=1.6, label="f_fine (MSE)", zorder=5)
        ax.plot(fc[0], fc[1], "s", ms=6, mfc="none", mec="#1565c0", mew=1.5, label="f_coarse (L2)", zorder=4)
        if br.get("short_name") == "MtoF":
            ax.plot(
                CANON_FP_M_TO_F[0],
                CANON_FP_M_TO_F[1],
                "*",
                ms=12,
                color="#6a1b9a",
                label="canon FP (5.15,4.36)",
                zorder=5,
            )
        boot = br["bootstrap"]
        if boot.get("ci_v") and not np.isnan(boot["ci_v"][0]):
            rect = Rectangle(
                (boot["ci_v"][0], boot["ci_a"][0]),
                boot["ci_v"][1] - boot["ci_v"][0],
                boot["ci_a"][1] - boot["ci_a"][0],
                fill=False,
                ec="#2e7d32",
                ls="--",
                lw=1.2,
                label="boot 95% box (coarse L2)",
            )
            ax.add_patch(rect)

        ax.set_xlim(1.5, 7.2)
        ax.set_ylim(1.5, 7.2)
        ax.set_xlabel("Valence (forced fixed point)")
        ax.set_ylabel("Arousal (forced fixed point)")
        ax.set_title(
            f"{name}\n‖f_L2−f̂‖={br['dist_fine_to_fhat']:.2f}; "
            f"‖f_MSE−f̂‖={br['dist_fine_mse_to_fhat']:.2f}; "
            f"MSE-nest={br['consistency']['pass_fhat_matches_unconstrained_fit']}"
        )
        ax.legend(fontsize=6.5, loc="best")
        ax.set_aspect("equal", adjustable="box")
        ax.scatter(coarse["f_v"], coarse["f_a"], s=8, c="k", alpha=0.15, zorder=1)

    fig.suptitle("Z1 | Forced-fixed-point residual landscape (held-out L2)", y=0.98, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def run_z1_all(
    df: pd.DataFrame,
    split: dict[str, Any],
    direction_specs: list[tuple[str, list[str], list[str], str]],
    out_dir: Path,
    n_boot: int,
    boot_seed: int,
) -> dict[str, Any]:
    bridges: dict[str, dict] = {}
    for name, src, tgt, short in direction_specs:
        print(f"Running Z1 {name} ...")
        br = run_z1_one_direction(
            df, split, src, tgt, name, short, out_dir, n_boot=n_boot, boot_seed=boot_seed
        )
        bridges[name] = br
        print(
            f"  f_L2=({br['f_fine'][0]:.2f},{br['f_fine'][1]:.2f}) "
            f"f_MSE=({br['f_fine_mse'][0]:.2f},{br['f_fine_mse'][1]:.2f}) "
            f"f̂=({br['f_hat_unconstrained_fit'][0]:.2f},{br['f_hat_unconstrained_fit'][1]:.2f}) "
            f"dL2={br['dist_fine_to_fhat']:.3f} dMSE={br['dist_fine_mse_to_fhat']:.3f} "
            f"nest_fhat={br['consistency']['pass_fhat_matches_unconstrained_fit']}"
        )

    plot_z1_heatmaps(bridges, out_dir / "Fig_anchor_position_heatmap.png")

    payload = {
        "spec_version": "2026-07-13-z1",
        "split_seed": split["seed"],
        "grid": {
            "coarse": list(Z1_COARSE_VALS),
            "fine_step": Z1_FINE_STEP,
            "fine_half_width": Z1_FINE_HALF_WIDTH,
        },
        "n_boot_coarse": n_boot,
        "consistency_eps": Z1_CONSISTENCY_EPS,
        "bridges": {
            name: {k: v for k, v in br.items() if not k.startswith("_")}
            for name, br in bridges.items()
        },
    }
    (out_dir / "z1_fixedpoint_optimality.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"payload": payload, "bridges": bridges}


def run_z2_pipeline(
    df: pd.DataFrame,
    split: dict[str, Any],
    direction_specs: list[tuple[str, list[str], list[str]]],
    out_dir: Path,
    k_list: tuple[int, ...],
    kmeans_seed: int,
    n_perm: int,
) -> dict[str, dict]:
    results_by_bridge: dict[str, dict] = {}
    all_win_rows: list[dict] = []
    all_per_image: list[dict] = []
    for name, src, tgt in direction_specs:
        print(f"Running Z2 {name} ...")
        res = run_one_direction(
            df,
            split,
            src_cols=src,
            tgt_cols=tgt,
            bridge_name=name,
            k_list=k_list,
            kmeans_seed=kmeans_seed,
            n_perm=n_perm,
        )
        results_by_bridge[name] = res
        all_win_rows.extend(res["win_rows"])
        all_per_image.extend(res["per_image_rows"])
        print(
            f"  outcome={res['decision']['outcome']} k*={res['decision']['k_star']} "
            f"mono_pass={res['monotonicity']['pass']}"
        )

    payload = {
        "spec_version": "2026-07-13",
        "preregistered": {
            "DELTA_R2_WIN": DELTA_R2_WIN,
            "ALPHA_PERM": ALPHA_PERM,
            "K_LIST": list(k_list),
            "OT_MODE": "exact_emd",
            "N_PERM": int(n_perm),
            "KMEANS_N_INIT": KMEANS_N_INIT,
            "KMEANS_MIN_CLUSTER": KMEANS_MIN_CLUSTER,
        },
        "outcome_b_policy": (
            "Keep linear-sufficiency pillar; M→F k*=2 is refinement not rewrite; "
            "F→M Outcome A documents directional asymmetry."
        ),
        "split": {
            "seed": split["seed"],
            "n_fit": split["n_fit"],
            "n_test": split["n_test"],
            "n_train": split["n_train"],
            "n_val": split["n_val"],
            "fit_definition": split["fit_definition"],
        },
        "bridges": {
            name: {
                "src_cols": res["src_cols"],
                "tgt_cols": res["tgt_cols"],
                "methods": res["methods"],
                "comparisons_vs_k1": res["comparisons_vs_k1"],
                "stability": res["stability"],
                "monotonicity": res["monotonicity"],
                "decision": res["decision"],
            }
            for name, res in results_by_bridge.items()
        },
        "asymmetry_note": (
            "M→F and F→M affine/piecewise/OT maps are not required to be inverses "
            "under independent least-squares fits; both directions are reported."
        ),
    }
    (out_dir / "z2_ksweep_gender.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(all_win_rows).to_csv(out_dir / "z2_win_table.csv", index=False)
    pd.DataFrame(all_per_image).to_csv(out_dir / "z2_per_image_heldout.csv", index=False)
    plot_ksweep(results_by_bridge, out_dir / "Fig_z2_ksweep_r2_residual.png", k_list)
    return results_by_bridge


# ---------------------------------------------------------------------------
# Z3: shared-anchor generalization
# ---------------------------------------------------------------------------

def similarity_gap(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(X, float) - np.asarray(Y, float), axis=1)


def assign_similarity_layers(
    gap_test: np.ndarray, quantiles: tuple[float, float] = (1 / 3, 2 / 3)
) -> tuple[np.ndarray, dict[str, float]]:
    q_lo, q_hi = np.quantile(gap_test, quantiles)
    labels = np.empty(len(gap_test), dtype=object)
    labels[gap_test <= q_lo] = "near"
    labels[(gap_test > q_lo) & (gap_test <= q_hi)] = "mid"
    labels[gap_test > q_hi] = "far"
    return labels, {"q33": float(q_lo), "q66": float(q_hi)}


def select_anchors_random(fit_idx: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    k = min(k, len(fit_idx))
    return np.asarray(rng.choice(fit_idx, size=k, replace=False), dtype=int)


def select_anchors_near_fixedpoint(
    fit_idx: np.ndarray,
    X_fit: np.ndarray,
    f_hat: np.ndarray,
    k: int,
    radius: float = Z3_FP_NEIGHBOR_RADIUS,
) -> np.ndarray:
    d = np.linalg.norm(X_fit - np.asarray(f_hat, float).reshape(1, 2), axis=1)
    order = np.argsort(d)
    within = order[d[order] <= radius]
    chosen_local = within[:k] if len(within) >= k else order[:k]
    return fit_idx[np.asarray(chosen_local, dtype=int)]


def select_anchors_most_similar(fit_idx: np.ndarray, gap_fit: np.ndarray, k: int) -> np.ndarray:
    order = np.argsort(gap_fit)
    return fit_idx[order[:k]]


def fit_shared_map(
    X_S: np.ndarray,
    Y_S: np.ndarray,
    k: int,
    mode: str = Z3_AFFINE_MODE_DEFAULT,
    ridge_alpha: float = Z3_RIDGE_ALPHA,
) -> tuple[AffineMap, str]:
    if k < 3:
        b = Y_S.mean(axis=0) - X_S.mean(axis=0)
        return AffineMap(np.eye(2), np.asarray(b, float)), "translation"
    if mode == "naive":
        return fit_affine(X_S, Y_S), "affine_naive"
    return fit_ridge_affine(X_S, Y_S, alpha=ridge_alpha), f"affine_ridge_a{ridge_alpha:g}"


def eval_understanding(
    phi: AffineMap,
    X_E: np.ndarray,
    Y_E: np.ndarray,
    layer_labels: np.ndarray | None = None,
) -> dict[str, float]:
    pred = phi.apply(X_E)
    out = r2_dims(Y_E, pred)
    out["residual_l2"] = mean_residual_l2(Y_E, pred)
    if layer_labels is not None:
        for L in Z3_SIM_LAYERS:
            mask = layer_labels == L
            n = int(mask.sum())
            key = f"R2_mean_{L}"
            if n < Z3_MIN_LAYER:
                out[key] = float("nan")
                out[f"n_{L}"] = float(n)
            else:
                out[key] = r2_dims(Y_E[mask], pred[mask])["R2_mean"]
                out[f"n_{L}"] = float(n)
    return out


def check_z3_ceiling(
    X_fit: np.ndarray, Y_fit: np.ndarray, X_te: np.ndarray, Y_te: np.ndarray
) -> dict[str, Any]:
    phi = fit_affine(X_fit, Y_fit)
    m = r2_dims(Y_te, phi.apply(X_te))
    return {
        "ceiling_R2_mean": m["R2_mean"],
        "ceiling_R2_valence": m["R2_valence"],
        "ceiling_R2_arousal": m["R2_arousal"],
        "f_hat": phi.fixed_point().tolist(),
    }


def _agg_draws(vals: list[float]) -> dict[str, float]:
    a = np.asarray(vals, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {"median": float("nan"), "q025": float("nan"), "q975": float("nan"), "n": 0}
    return {
        "median": float(np.median(a)),
        "q025": float(np.quantile(a, 0.025)),
        "q975": float(np.quantile(a, 0.975)),
        "n": int(len(a)),
    }


def tag_z3_outcome(
    curve_rows: list[dict],
    ceiling: float,
    k_star: int | None,
) -> str:
    if not curve_rows or not np.isfinite(ceiling) or ceiling <= 0:
        return "NO_GENERALIZATION"
    by_k = {int(r["k"]): r for r in curve_rows}
    ks = sorted(by_k)
    medians = [by_k[k]["R2_median"] for k in ks]
    if max(medians) < 0.2 * ceiling:
        return "NO_GENERALIZATION"
    ref_k = k_star if k_star is not None else ks[min(3, len(ks) - 1)]
    far_at = by_k[ref_k].get("R2_far_median", float("nan")) if ref_k in by_k else float("nan")
    near_at = by_k[ref_k].get("R2_near_median", float("nan")) if ref_k in by_k else float("nan")
    sat = k_star is not None and k_star <= 8
    far_ok = np.isfinite(far_at) and far_at >= 0.5 * ceiling
    near_only = (
        np.isfinite(near_at)
        and np.isfinite(far_at)
        and near_at >= 0.7 * ceiling
        and far_at < 0.5 * ceiling
    )
    if sat and far_ok:
        return "FEW_SHARE_FAR_GENERALIZES"
    if sat and near_only:
        return "FEW_SHARE_NEAR_ONLY"
    if (not sat) and far_ok:
        return "ACCUMULATIVE_FAR_OK"
    if sat:
        return "FEW_SHARE_NEAR_ONLY"
    return "ACCUMULATIVE_FAR_OK"


def run_z3_one_direction(
    df: pd.DataFrame,
    split: dict[str, Any],
    src_cols: list[str],
    tgt_cols: list[str],
    bridge_name: str,
    short_name: str,
    n_draws: int,
    draw_seed: int,
    affine_mode: str = Z3_AFFINE_MODE_DEFAULT,
    ridge_alpha: float = Z3_RIDGE_ALPHA,
) -> dict[str, Any]:
    fit_idx = split["fit_idx"]
    test_idx = split["test_idx"]
    if len(test_idx) < Z3_MIN_EVAL:
        raise RuntimeError(f"test set too small: {len(test_idx)} < {Z3_MIN_EVAL}")

    X_all = df[src_cols].to_numpy(float)
    Y_all = df[tgt_cols].to_numpy(float)
    X_fit = X_all[fit_idx]
    Y_fit = Y_all[fit_idx]
    X_te = X_all[test_idx]
    Y_te = Y_all[test_idx]

    gap_all = similarity_gap(X_all, Y_all)
    gap_fit = gap_all[fit_idx]
    gap_te = gap_all[test_idx]
    layer_labels, thresholds = assign_similarity_layers(gap_te)

    ceiling = check_z3_ceiling(X_fit, Y_fit, X_te, Y_te)
    f_hat = np.asarray(ceiling["f_hat"], float)
    ceil_r2 = float(ceiling["ceiling_R2_mean"])

    rng = np.random.default_rng(draw_seed)
    k_list = [k for k in Z3_K_LIST if k <= len(fit_idx)]
    curve_rows: list[dict[str, Any]] = []

    for k in k_list:
        overall: list[float] = []
        layers: dict[str, list[float]] = {L: [] for L in Z3_SIM_LAYERS}
        map_types: list[str] = []
        for _ in range(n_draws):
            S = select_anchors_random(fit_idx, k, rng)
            phi, mtype = fit_shared_map(
                X_all[S], Y_all[S], k, mode=affine_mode, ridge_alpha=ridge_alpha
            )
            map_types.append(mtype)
            u = eval_understanding(phi, X_te, Y_te, layer_labels)
            overall.append(u["R2_mean"])
            for L in Z3_SIM_LAYERS:
                layers[L].append(u.get(f"R2_mean_{L}", float("nan")))
        agg = _agg_draws(overall)
        row: dict[str, Any] = {
            "bridge": bridge_name,
            "k": int(k),
            "n_draws": int(n_draws),
            "affine_mode": affine_mode,
            "R2_median": agg["median"],
            "R2_q025": agg["q025"],
            "R2_q975": agg["q975"],
            "ceiling": ceil_r2,
            "map_type_majority": max(set(map_types), key=map_types.count),
        }
        for L in Z3_SIM_LAYERS:
            la = _agg_draws(layers[L])
            row[f"R2_{L}_median"] = la["median"]
            row[f"R2_{L}_q025"] = la["q025"]
            row[f"R2_{L}_q975"] = la["q975"]
        curve_rows.append(row)

    k_star = None
    thr = Z3_CEILING_FRAC * ceil_r2
    # Stable k*: first k that clears the ceiling fraction AND stays above
    # 0.95×threshold for all larger k (avoids false k*=1 when k=3 affine collapses).
    for i, row in enumerate(curve_rows):
        if not (np.isfinite(row["R2_median"]) and row["R2_median"] >= thr):
            continue
        rest_ok = all(
            (not np.isfinite(r["R2_median"])) or r["R2_median"] >= 0.95 * thr
            for r in curve_rows[i:]
        )
        if rest_ok:
            k_star = int(row["k"])
            break
    # Also record naive first-crossing for diagnostics
    k_star_naive = None
    for row in curve_rows:
        if np.isfinite(row["R2_median"]) and row["R2_median"] >= thr:
            k_star_naive = int(row["k"])
            break

    sel_rows: list[dict[str, Any]] = []
    sel_ks = [k for k in Z3_SELECTION_K if k <= len(fit_idx)]
    for k in sel_ks:
        match = next(r for r in curve_rows if r["k"] == k)
        sel_rows.append(
            {
                "bridge": bridge_name,
                "k": k,
                "method": "random",
                "R2_mean": match["R2_median"],
                "notes": "median over random draws",
            }
        )
        S_fp = select_anchors_near_fixedpoint(fit_idx, X_fit, f_hat, k)
        phi_fp, mt = fit_shared_map(
            X_all[S_fp], Y_all[S_fp], k, mode=affine_mode, ridge_alpha=ridge_alpha
        )
        u_fp = eval_understanding(phi_fp, X_te, Y_te, layer_labels)
        sel_rows.append(
            {
                "bridge": bridge_name,
                "k": k,
                "method": "fp_near",
                "R2_mean": u_fp["R2_mean"],
                "notes": f"map={mt}; radius={Z3_FP_NEIGHBOR_RADIUS}",
            }
        )
        S_sim = select_anchors_most_similar(fit_idx, gap_fit, k)
        phi_sim, mt2 = fit_shared_map(
            X_all[S_sim], Y_all[S_sim], k, mode=affine_mode, ridge_alpha=ridge_alpha
        )
        u_sim = eval_understanding(phi_sim, X_te, Y_te, layer_labels)
        sel_rows.append(
            {
                "bridge": bridge_name,
                "k": k,
                "method": "similar",
                "R2_mean": u_sim["R2_mean"],
                "notes": f"map={mt2}; lowest gap",
            }
        )

    medians = [r["R2_median"] for r in curve_rows]
    n_up = sum(1 for i in range(1, len(medians)) if medians[i] + 1e-6 < medians[i - 1])
    last = medians[-1] if medians else float("nan")
    approaches_ceiling = bool(np.isfinite(last) and last >= 0.85 * ceil_r2)
    tag = tag_z3_outcome(curve_rows, ceil_r2, k_star)

    return {
        "bridge": bridge_name,
        "short_name": short_name,
        "affine_mode": affine_mode,
        "ridge_alpha": float(ridge_alpha) if affine_mode == "ridge" else None,
        "ceiling_R2_mean": ceil_r2,
        "ceiling": ceiling,
        "k_star": k_star,
        "k_star_naive_first_crossing": k_star_naive,
        "outcome_tag": tag,
        "layer_thresholds": thresholds,
        "layer_counts": {L: int((layer_labels == L).sum()) for L in Z3_SIM_LAYERS},
        "curve_rows": curve_rows,
        "selection_rows": sel_rows,
        "diagnostics": {
            "n_upward_violations_median": int(n_up),
            "approaches_ceiling": approaches_ceiling,
            "n_fit": int(len(fit_idx)),
            "n_test": int(len(test_idx)),
        },
    }


def plot_z3_curves(bridges: dict[str, dict], out_path: Path) -> None:
    names = list(bridges.keys())
    fig, axes = plt.subplots(len(names), 2, figsize=(10.5, 4.2 * len(names)), squeeze=False)
    colors = {"near": "#2e7d32", "mid": "#f9a825", "far": "#c62828"}
    for row_i, name in enumerate(names):
        br = bridges[name]
        rows = br["curve_rows"]
        ks = [r["k"] for r in rows]
        med = [r["R2_median"] for r in rows]
        lo = [r["R2_q025"] for r in rows]
        hi = [r["R2_q975"] for r in rows]
        ceil = br["ceiling_R2_mean"]

        ax = axes[row_i, 0]
        ax.plot(ks, med, "o-", color="#1565c0", label="random anchors (median)")
        ax.fill_between(ks, lo, hi, color="#1565c0", alpha=0.2, label="95% draw band")
        ax.axhline(ceil, color="k", ls="--", label="ceiling (unconstrained)")
        ax.axhline(Z3_CEILING_FRAC * ceil, color="0.4", ls=":", label=f"{Z3_CEILING_FRAC:.0%}×ceiling")
        if br.get("k_star") is not None:
            ax.axvline(br["k_star"], color="#6a1b9a", ls="--", alpha=0.8, label=f"k*={br['k_star']}")
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlabel("k shared anchors (fit themes)")
        ax.set_ylabel("Understanding R² (held-out test)")
        ax.set_title(f"{name} | overall | tag={br['outcome_tag']}")
        ax.legend(fontsize=7, loc="best")
        ymin = min([0.0] + [x for x in lo if np.isfinite(x)]) - 0.05
        ax.set_ylim(ymin, max(1.0, ceil + 0.05))

        ax = axes[row_i, 1]
        for L in Z3_SIM_LAYERS:
            ys = [r.get(f"R2_{L}_median", float("nan")) for r in rows]
            ax.plot(ks, ys, "o-", color=colors[L], label=L)
        ax.axhline(ceil, color="k", ls="--", alpha=0.7)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlabel("k shared anchors")
        ax.set_ylabel("Layer R² (held-out)")
        ax.set_title(f"{name} | similarity strata (gap tertiles)")
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        "Z3 | Shared-anchor generalization (anchors⊂fit, eval=theme-held-out)",
        y=0.995,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


def analyze_far_vs_excess_twist(df: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    """
    Spatial/associative check: do high pre-transform gender gaps (far) coincide
    with high excess twist (UMAP disp − model twist)?
    """
    twist_path = (
        PROJECT_ROOT
        / "results"
        / "relational_cross_within_twist"
        / "relational_twist_per_image.csv"
    )
    if not twist_path.exists():
        return {"status": "skip", "reason": f"missing {twist_path}"}

    tw = pd.read_csv(twist_path)
    # Join keys: image_id like I1; oasis meta often has Theme or filename
    oasis = df.copy()
    if "image_id" not in oasis.columns:
        # build from index order if Theme present
        if "Theme" in oasis.columns:
            oasis["image_id"] = oasis["Theme"].astype(str)
        else:
            return {"status": "skip", "reason": "no image_id/Theme to join"}

    tw["excess_twist"] = tw["geo_umap_disp"].to_numpy(float) - tw["model_twist_pm_pf"].to_numpy(float)
    # gap on OASIS male/female
    X = oasis[SRC_M].to_numpy(float)
    Y = oasis[SRC_F].to_numpy(float)
    oasis = oasis.copy()
    oasis["gap_l2"] = similarity_gap(X, Y)
    q33, q66 = np.quantile(oasis["gap_l2"], [1 / 3, 2 / 3])
    oasis["gap_layer"] = np.where(
        oasis["gap_l2"] <= q33, "near", np.where(oasis["gap_l2"] > q66, "far", "mid")
    )

    # lowV-lowA flag from twist table before merge (avoid column collisions)
    v_med = float(tw["valence"].median())
    a_med = float(tw["arousal"].median())
    tw["lowV_lowA"] = (tw["valence"] < v_med) & (tw["arousal"] < a_med)

    merged = tw.merge(
        oasis[["image_id", "gap_l2", "gap_layer"]],
        on="image_id",
        how="inner",
    )
    if len(merged) < 50:
        return {"status": "skip", "reason": f"join too small n={len(merged)}"}

    from scipy.stats import spearmanr, mannwhitneyu

    rho, p_rho = spearmanr(merged["gap_l2"], merged["excess_twist"])
    far = merged.loc[merged["gap_layer"] == "far", "excess_twist"].to_numpy(float)
    near = merged.loc[merged["gap_layer"] == "near", "excess_twist"].to_numpy(float)
    mid = merged.loc[merged["gap_layer"] == "mid", "excess_twist"].to_numpy(float)
    mwu = mannwhitneyu(far, near, alternative="greater")
    far_in_lola = float(merged.loc[merged["gap_layer"] == "far", "lowV_lowA"].mean())
    near_in_lola = float(merged.loc[merged["gap_layer"] == "near", "lowV_lowA"].mean())

    out = {
        "status": "ok",
        "n": int(len(merged)),
        "spearman_gap_vs_excess_twist": {"rho": float(rho), "p": float(p_rho)},
        "excess_twist_mean": {
            "far": float(np.mean(far)),
            "near": float(np.mean(near)),
            "mid": float(np.mean(mid)),
        },
        "mwu_far_gt_near": {"U": float(mwu.statistic), "p": float(mwu.pvalue)},
        "frac_in_lowV_lowA": {"far": far_in_lola, "near": near_in_lola},
        "interpretation_hint": (
            "If rho>0 and far>near excess twist, Z3-far aligns with residual/twist cost regions."
        ),
    }
    (out_dir / "z3_far_vs_excess_twist.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def run_z3_all(
    df: pd.DataFrame,
    split: dict[str, Any],
    direction_specs: list[tuple[str, list[str], list[str], str]],
    out_dir: Path,
    n_draws: int,
    draw_seed: int,
    affine_mode: str = Z3_AFFINE_MODE_DEFAULT,
    ridge_alpha: float = Z3_RIDGE_ALPHA,
) -> dict[str, Any]:
    bridges: dict[str, dict] = {}
    all_sel: list[dict] = []

    for name, src, tgt, short in direction_specs:
        print(f"Running Z3 {name} (affine_mode={affine_mode}) ...")
        br = run_z3_one_direction(
            df,
            split,
            src,
            tgt,
            name,
            short,
            n_draws=n_draws,
            draw_seed=draw_seed,
            affine_mode=affine_mode,
            ridge_alpha=ridge_alpha,
        )
        bridges[name] = br
        all_sel.extend(br["selection_rows"])
        print(
            f"  ceiling={br['ceiling_R2_mean']:.4f} k*={br['k_star']} tag={br['outcome_tag']} "
            f"mode={affine_mode}"
        )
        suffix = f"{short}_{affine_mode}"
        pd.DataFrame(br["curve_rows"]).to_csv(
            out_dir / f"z3_generalization_curve_{suffix}.csv", index=False
        )
        # also write short name for main (ridge) curve
        if affine_mode == "ridge":
            pd.DataFrame(br["curve_rows"]).to_csv(
                out_dir / f"z3_generalization_curve_{short}.csv", index=False
            )

    pd.DataFrame(all_sel).to_csv(
        out_dir / f"z3_anchor_selection_compare_{affine_mode}.csv", index=False
    )
    if affine_mode == "ridge":
        pd.DataFrame(all_sel).to_csv(out_dir / "z3_anchor_selection_compare.csv", index=False)
    plot_z3_curves(bridges, out_dir / f"Fig_shared_anchor_generalization_{affine_mode}.png")
    if affine_mode == "ridge":
        plot_z3_curves(bridges, out_dir / "Fig_shared_anchor_generalization.png")

    twist_overlap = analyze_far_vs_excess_twist(df, out_dir)

    payload = {
        "spec_version": "2026-07-13-z3-ridge",
        "preregistered": {
            "Z3_K_LIST": list(Z3_K_LIST),
            "Z3_N_RANDOM_DRAWS": n_draws,
            "Z3_CEILING_FRAC": Z3_CEILING_FRAC,
            "affine_mode_main": affine_mode,
            "ridge_alpha": ridge_alpha if affine_mode == "ridge" else None,
            "eval": "theme_heldout_test",
            "anchors": "subset_of_fit",
            "similarity": "L2_gap_src_tgt_tertiles_on_test",
            "note_prior_naive": (
                "Earlier k*=13 used naive LS with k=3 collapse; main curve is now ridge-stabilized."
            ),
        },
        "split": {
            "seed": split["seed"],
            "n_fit": split["n_fit"],
            "n_test": split["n_test"],
        },
        "far_vs_excess_twist": twist_overlap,
        "bridges": {
            name: {
                k: v
                for k, v in br.items()
                if k
                in (
                    "bridge",
                    "short_name",
                    "affine_mode",
                    "ridge_alpha",
                    "ceiling_R2_mean",
                    "ceiling",
                    "k_star",
                    "k_star_naive_first_crossing",
                    "outcome_tag",
                    "layer_thresholds",
                    "layer_counts",
                    "curve_rows",
                    "selection_rows",
                    "diagnostics",
                )
            }
            for name, br in bridges.items()
        },
    }
    (out_dir / "z3_generalization.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return payload


# ---------------------------------------------------------------------------
# Z4: VA coverage quality of shared anchors
# ---------------------------------------------------------------------------

def z4_grid_edges(n: int = Z4_GRID_N, lim: tuple[float, float] = Z4_VA_LIM) -> np.ndarray:
    return np.linspace(lim[0], lim[1], n + 1)


def va_cell_ids(
    xy: np.ndarray, edges: np.ndarray | None = None, n: int = Z4_GRID_N
) -> np.ndarray:
    """Map VA points to flat cell ids in [0, n*n). Right/top edge clipped into last bin."""
    edges = z4_grid_edges(n) if edges is None else edges
    xy = np.asarray(xy, float)
    ix = np.digitize(xy[:, 0], edges) - 1
    iy = np.digitize(xy[:, 1], edges) - 1
    ix = np.clip(ix, 0, n - 1)
    iy = np.clip(iy, 0, n - 1)
    return (iy * n + ix).astype(int)


def coverage_stats(xy_S: np.ndarray, n: int = Z4_GRID_N) -> dict[str, Any]:
    cells = va_cell_ids(xy_S, n=n)
    uniq = np.unique(cells)
    n_cells = int(len(uniq))
    return {
        "coverage_count": n_cells,
        "coverage_rate": float(n_cells / (n * n)),
        "cell_ids": uniq.tolist(),
    }


def candidate_reps_per_cell(
    fit_idx: np.ndarray,
    X_fit: np.ndarray,
    n_per_cell: int = Z4_GREEDY_CAND_PER_CELL,
    n: int = Z4_GRID_N,
) -> list[dict[str, Any]]:
    """Nearest-to-cell-center representatives in fit (empty cells omitted)."""
    edges = z4_grid_edges(n)
    cells = va_cell_ids(X_fit, edges=edges, n=n)
    reps: list[dict[str, Any]] = []
    for cid in range(n * n):
        local = np.where(cells == cid)[0]
        if len(local) == 0:
            continue
        iy, ix = divmod(cid, n)
        cx = 0.5 * (edges[ix] + edges[ix + 1])
        cy = 0.5 * (edges[iy] + edges[iy + 1])
        d = np.linalg.norm(X_fit[local] - np.array([cx, cy]), axis=1)
        order = np.argsort(d)[: min(n_per_cell, len(local))]
        for j in order:
            li = int(local[j])
            reps.append({
                "cell_id": int(cid),
                "fit_local": li,
                "image_idx": int(fit_idx[li]),
                "x": float(X_fit[li, 0]),
                "y": float(X_fit[li, 1]),
            })
    return reps


def run_z4_coverage_vs_understanding(
    df: pd.DataFrame,
    split: dict[str, Any],
    src_cols: list[str],
    tgt_cols: list[str],
    *,
    n_draws: int = Z4_N_DRAWS,
    draw_seed: int = RANDOM_SEED + Z4_DRAW_SEED_OFFSET,
    ridge_alpha: float = Z4_RIDGE_ALPHA,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fit_idx = split["fit_idx"]
    test_idx = split["test_idx"]
    X_all = df[src_cols].to_numpy(float)
    Y_all = df[tgt_cols].to_numpy(float)
    X_fit, Y_fit = X_all[fit_idx], Y_all[fit_idx]
    X_te, Y_te = X_all[test_idx], Y_all[test_idx]
    ceiling = check_z3_ceiling(X_fit, Y_fit, X_te, Y_te)
    ceil_r2 = float(ceiling["ceiling_R2_mean"])

    rng = np.random.default_rng(draw_seed)
    rows: list[dict[str, Any]] = []
    for k in Z4_K_LIST:
        if k > len(fit_idx) or k < 3:
            continue
        for d in range(n_draws):
            S = select_anchors_random(fit_idx, k, rng)
            cov = coverage_stats(X_all[S])
            phi, map_type = fit_shared_map(
                X_all[S], Y_all[S], k, mode="ridge", ridge_alpha=ridge_alpha
            )
            und = eval_understanding(phi, X_te, Y_te)
            rows.append({
                "k": int(k),
                "draw": int(d),
                "coverage_count": cov["coverage_count"],
                "coverage_rate": cov["coverage_rate"],
                "R2_mean": float(und["R2_mean"]),
                "R2_valence": float(und["R2_valence"]),
                "R2_arousal": float(und["R2_arousal"]),
                "residual_l2": float(und["residual_l2"]),
                "map_type": map_type,
                "ceiling_R2_mean": ceil_r2,
                "frac_of_ceiling": float(und["R2_mean"] / ceil_r2) if ceil_r2 > 1e-12 else float("nan"),
            })

    cdf = pd.DataFrame(rows)
    within: dict[str, Any] = {}
    n_pass = 0
    for k in Z4_UPGRADE_K_SET:
        sub = cdf[cdf["k"] == k]
        if len(sub) < 8:
            within[str(k)] = {"n": int(len(sub)), "rho": None, "p": None, "pass": False}
            continue
        rho, p = spearmanr(sub["coverage_rate"], sub["R2_mean"])
        ok = bool(np.isfinite(rho) and rho >= Z4_UPGRADE_RHO and p < Z4_UPGRADE_P)
        if ok:
            n_pass += 1
        within[str(k)] = {
            "n": int(len(sub)),
            "rho": float(rho),
            "p": float(p),
            "pass": ok,
            "coverage_rate_mean": float(sub["coverage_rate"].mean()),
            "R2_mean_median": float(sub["R2_mean"].median()),
        }

    # residualize R2 on k (categorical dummies) then Spearman with coverage
    pooled_rho, pooled_p = float("nan"), float("nan")
    if len(cdf) >= 20:
        # simple residual: R2 - mean_R2(k)
        mu = cdf.groupby("k")["R2_mean"].transform("mean")
        resid = cdf["R2_mean"] - mu
        pooled_rho, pooled_p = spearmanr(cdf["coverage_rate"], resid)
        pooled_rho, pooled_p = float(pooled_rho), float(pooled_p)

    summary = {
        "ceiling": ceiling,
        "within_k_spearman": within,
        "n_upgrade_k_pass": int(n_pass),
        "pooled_residual_spearman": {"rho": pooled_rho, "p": pooled_p},
        "criterion_z4_1_pass": bool(n_pass >= Z4_UPGRADE_MIN_K_PASS),
        "upgrade_thresholds": {
            "rho": Z4_UPGRADE_RHO,
            "p": Z4_UPGRADE_P,
            "min_k_pass": Z4_UPGRADE_MIN_K_PASS,
            "k_set": list(Z4_UPGRADE_K_SET),
        },
    }
    return cdf, summary


def run_z4_minimal_coverage(
    df: pd.DataFrame,
    split: dict[str, Any],
    src_cols: list[str],
    tgt_cols: list[str],
    *,
    ridge_alpha: float = Z4_RIDGE_ALPHA,
) -> dict[str, Any]:
    fit_idx = split["fit_idx"]
    test_idx = split["test_idx"]
    X_all = df[src_cols].to_numpy(float)
    Y_all = df[tgt_cols].to_numpy(float)
    X_fit = X_all[fit_idx]
    Y_fit = Y_all[fit_idx]
    X_te, Y_te = X_all[test_idx], Y_all[test_idx]
    ceiling = check_z3_ceiling(X_fit, Y_fit, X_te, Y_te)
    ceil_r2 = float(ceiling["ceiling_R2_mean"])
    target = Z4_CEILING_FRAC * ceil_r2

    reps = candidate_reps_per_cell(fit_idx, X_fit)
    unused = list(range(len(reps)))
    S: list[int] = []
    path: list[dict[str, Any]] = []
    occupied: set[int] = set()

    def eval_S(s_idx: list[int]) -> float:
        if len(s_idx) < 3:
            # need ≥3 for ridge; use translation for incomplete early steps reporting only
            if len(s_idx) == 0:
                return float("nan")
            phi, _ = fit_shared_map(X_all[s_idx], Y_all[s_idx], len(s_idx), mode="ridge", ridge_alpha=ridge_alpha)
            return float(eval_understanding(phi, X_te, Y_te)["R2_mean"])
        phi, _ = fit_shared_map(X_all[s_idx], Y_all[s_idx], len(s_idx), mode="ridge", ridge_alpha=ridge_alpha)
        return float(eval_understanding(phi, X_te, Y_te)["R2_mean"])

    while len(S) < Z4_GREEDY_MAX_K and unused:
        best_u = None
        best_r2 = -np.inf
        best_new_cell = False
        for u in unused:
            cand = S + [reps[u]["image_idx"]]
            # skip singleton translation noise: allow growth, but only score with ≥1
            r2 = eval_S(cand)
            if not np.isfinite(r2):
                continue
            opens = reps[u]["cell_id"] not in occupied
            # prefer higher R2; tie-break: open new cell
            better = r2 > best_r2 + 1e-12 or (
                abs(r2 - best_r2) <= 1e-12 and opens and not best_new_cell
            )
            if better:
                best_r2 = r2
                best_u = u
                best_new_cell = opens
        if best_u is None:
            break
        S.append(reps[best_u]["image_idx"])
        occupied.add(int(reps[best_u]["cell_id"]))
        unused.remove(best_u)
        cov = coverage_stats(X_all[np.asarray(S, int)])
        path.append({
            "step": len(S),
            "added_image_idx": int(reps[best_u]["image_idx"]),
            "added_cell_id": int(reps[best_u]["cell_id"]),
            "R2_mean": float(best_r2),
            "coverage_count": cov["coverage_count"],
            "coverage_rate": cov["coverage_rate"],
            "frac_of_ceiling": float(best_r2 / ceil_r2) if ceil_r2 > 1e-12 else float("nan"),
            "hit_target": bool(best_r2 >= target),
        })
        if best_r2 >= target and len(S) >= 3:
            break

    final_r2 = path[-1]["R2_mean"] if path else float("nan")
    final_cov = path[-1]["coverage_count"] if path else 0
    hit = bool(final_r2 >= target and len(S) >= 3)
    criterion_pass = bool(hit and final_cov <= Z4_MINIMAL_MAX_CELLS)

    return {
        "ceiling_R2_mean": ceil_r2,
        "target_R2": float(target),
        "ceiling_frac": Z4_CEILING_FRAC,
        "n_candidate_reps": int(len(reps)),
        "n_nonempty_cells_in_fit": int(len({r["cell_id"] for r in reps})),
        "path": path,
        "final": {
            "k": int(len(S)),
            "image_idx": [int(i) for i in S],
            "R2_mean": float(final_r2),
            "coverage_count": int(final_cov),
            "coverage_rate": float(final_cov / (Z4_GRID_N ** 2)),
            "frac_of_ceiling": float(final_r2 / ceil_r2) if ceil_r2 > 1e-12 else float("nan"),
            "hit_90pct_ceiling": hit,
            "cell_ids": sorted(occupied),
        },
        "criterion_z4_3_pass": criterion_pass,
        "upgrade_max_cells": Z4_MINIMAL_MAX_CELLS,
    }


def plot_z4_coverage(cdf: pd.DataFrame, minimal: dict[str, Any], out_path: Path, *, title_suffix: str = "") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    ax = axes[0]
    ks = sorted(cdf["k"].unique())
    cmap = plt.cm.viridis
    for i, k in enumerate(ks):
        sub = cdf[cdf["k"] == k]
        col = cmap(0.2 + 0.7 * i / max(len(ks) - 1, 1))
        ax.scatter(
            sub["coverage_rate"], sub["R2_mean"], s=18, alpha=0.45, color=col,
            edgecolors="none", label=f"k={k}",
        )
    ax.set_xlabel("VA coverage rate (#cells / 25)")
    ax.set_ylabel("Held-out understanding R²")
    ax.set_title("Z4-1: coverage vs understanding (same k)")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")

    ax = axes[1]
    n = Z4_GRID_N
    grid = np.zeros((n, n))
    for cid in minimal["final"]["cell_ids"]:
        iy, ix = divmod(int(cid), n)
        grid[iy, ix] = 1.0
    im = ax.imshow(
        grid, origin="lower", cmap="YlOrRd", vmin=0, vmax=1,
        extent=[Z4_VA_LIM[0], Z4_VA_LIM[1], Z4_VA_LIM[0], Z4_VA_LIM[1]],
        alpha=0.85,
    )
    # path points
    if minimal["final"]["image_idx"]:
        # scatter added order by cell centers from path
        for step in minimal["path"]:
            cid = int(step["added_cell_id"])
            iy, ix = divmod(cid, n)
            edges = z4_grid_edges()
            cx = 0.5 * (edges[ix] + edges[ix + 1])
            cy = 0.5 * (edges[iy] + edges[iy + 1])
            ax.scatter([cx], [cy], s=40, c="#1565c0", edgecolors="white", linewidths=0.6, zorder=3)
            ax.text(cx, cy, str(step["step"]), ha="center", va="center", fontsize=6, color="white")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    fin = minimal["final"]
    ax.set_title(
        f"Z4-3 minimal cover: k={fin['k']}, cells={fin['coverage_count']}, "
        f"R²={fin['R2_mean']:.3f} ({fin['frac_of_ceiling']:.0%}·ceil)"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="occupied cell")

    fig.suptitle(
        f"Z4 VA coverage analysis{title_suffix}",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_z4_all(
    df: pd.DataFrame,
    split: dict[str, Any],
    specs: list[tuple[str, list[str], list[str], str]],
    out_dir: Path,
    *,
    n_draws: int = Z4_N_DRAWS,
    draw_seed: int = RANDOM_SEED + Z4_DRAW_SEED_OFFSET,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bridges: dict[str, Any] = {}
    all_rows: list[pd.DataFrame] = []

    for bridge, src, tgt, short in specs:
        print(f"[Z4] {bridge} coverage×understanding ...", flush=True)
        cdf, sum1 = run_z4_coverage_vs_understanding(
            df, split, src, tgt, n_draws=n_draws, draw_seed=draw_seed
        )
        cdf = cdf.copy()
        cdf["bridge"] = bridge
        all_rows.append(cdf)
        print(f"[Z4] {bridge} minimal coverage greedy ...", flush=True)
        minimal = run_z4_minimal_coverage(df, split, src, tgt)
        upgrade = bool(sum1["criterion_z4_1_pass"] and minimal["criterion_z4_3_pass"])
        bridges[bridge] = {
            "short_name": short,
            "z4_1": sum1,
            "z4_3": minimal,
            "upgrade_to_maintext": upgrade,
            "upgrade_decision": (
                "MAINTEXT_CANDIDATE" if upgrade else "SUPPLEMENT_OR_PENDING"
            ),
        }
        plot_z4_coverage(
            cdf, minimal, out_dir / f"Fig_coverage_analysis_{short}.png",
            title_suffix=f" ({short})",
        )
        # primary alias for MtoF
        if short == "MtoF":
            plot_z4_coverage(cdf, minimal, out_dir / "Fig_coverage_analysis.png", title_suffix=" (M→F)")
            (out_dir / "z4_minimal_coverage.json").write_text(
                json.dumps(minimal, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    cov_csv = pd.concat(all_rows, ignore_index=True)
    cov_csv.to_csv(out_dir / "z4_coverage_understanding.csv", index=False)

    # primary M→F summary
    primary = bridges.get("gender_M_to_F") or next(iter(bridges.values()))
    summary = {
        "spec_version": "z4_va_coverage_v1",
        "preregistered": True,
        "grid_n": Z4_GRID_N,
        "va_lim": list(Z4_VA_LIM),
        "n_draws": n_draws,
        "bridges": {
            name: {
                "upgrade_to_maintext": br["upgrade_to_maintext"],
                "upgrade_decision": br["upgrade_decision"],
                "z4_1_pass": br["z4_1"]["criterion_z4_1_pass"],
                "z4_3_pass": br["z4_3"]["criterion_z4_3_pass"],
                "within_k_spearman": br["z4_1"]["within_k_spearman"],
                "pooled_residual_spearman": br["z4_1"]["pooled_residual_spearman"],
                "minimal_final": br["z4_3"]["final"],
            }
            for name, br in bridges.items()
        },
        "primary_bridge": "gender_M_to_F" if "gender_M_to_F" in bridges else next(iter(bridges)),
        "primary_upgrade": primary["upgrade_to_maintext"],
    }
    (out_dir / "z4_coverage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # full dump
    (out_dir / "z4_full_payload.json").write_text(
        json.dumps({"summary": summary, "bridges": bridges}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


# ---------------------------------------------------------------------------
# Z5: minimal-coverage anchor identification & characterization
# ---------------------------------------------------------------------------

def z5_cell_center(cell_id: int, edges: np.ndarray | None = None, n: int = Z5_GRID_N) -> tuple[float, float]:
    edges = z4_grid_edges(n) if edges is None else edges
    iy, ix = divmod(int(cell_id), n)
    v = 0.5 * (edges[ix] + edges[ix + 1])
    a = 0.5 * (edges[iy] + edges[iy + 1])
    return float(v), float(a)


def z5_quadrant_label(v: float, a: float, origin_v: float, origin_a: float) -> str:
    hv = "highV" if v >= origin_v else "lowV"
    ha = "highA" if a >= origin_a else "lowA"
    return f"{hv}_{ha}"


def z5_load_reference_points() -> dict[str, Any]:
    if Z5_CENTROIDS_JSON.exists():
        data = json.loads(Z5_CENTROIDS_JSON.read_text(encoding="utf-8"))
    else:
        data = {
            "fixed_points": {
                "gender": {"valence": CANON_FP_M_TO_F[0], "arousal": CANON_FP_M_TO_F[1]},
                "culture": {"valence": 4.33, "arousal": 3.98},
            },
            "centroids": {
                "oasis_900_overall": {"valence": 4.33, "arousal": 3.67},
            },
        }
    fp_g = data["fixed_points"]["gender"]
    fp_c = data["fixed_points"]["culture"]
    cen = data["centroids"]["oasis_900_overall"]
    return {
        "gender_fp": (float(fp_g["valence"]), float(fp_g["arousal"])),
        "culture_fp": (float(fp_c["valence"]), float(fp_c["arousal"])),
        "oasis_centroid": (float(cen["valence"]), float(cen["arousal"])),
    }


def z5_jaccard(a: set[int] | frozenset[int], b: set[int] | frozenset[int]) -> float:
    ua = set(a)
    ub = set(b)
    if not ua and not ub:
        return 1.0
    union = ua | ub
    if not union:
        return float("nan")
    return float(len(ua & ub) / len(union))


def z5_median_pairwise_jaccard(cell_sets: list[frozenset[int]]) -> float:
    if len(cell_sets) < 2:
        return 1.0
    vals = []
    for i in range(len(cell_sets)):
        for j in range(i + 1, len(cell_sets)):
            vals.append(z5_jaccard(cell_sets[i], cell_sets[j]))
    return float(np.median(vals)) if vals else float("nan")


def z5_eval_anchor_set(
    image_idx: list[int],
    X_all: np.ndarray,
    Y_all: np.ndarray,
    X_te: np.ndarray,
    Y_te: np.ndarray,
    *,
    ridge_alpha: float = Z5_RIDGE_ALPHA,
) -> float:
    if len(image_idx) == 0:
        return float("nan")
    s = np.asarray(image_idx, int)
    phi, _ = fit_shared_map(X_all[s], Y_all[s], len(s), mode="ridge", ridge_alpha=ridge_alpha)
    return float(eval_understanding(phi, X_te, Y_te)["R2_mean"])


def greedy_minimal_coverage(
    df: pd.DataFrame,
    split: dict[str, Any],
    src_cols: list[str],
    tgt_cols: list[str],
    *,
    ridge_alpha: float = Z5_RIDGE_ALPHA,
    restart_seed: int = 0,
    shuffle_reps: bool = True,
    train_category: str | None = None,
    fixed_k: int | None = None,
) -> dict[str, Any]:
    """Z4-3 greedy minimal coverage with optional candidate shuffle (Z5 stability)."""
    fit_idx_all = np.asarray(split["fit_idx"], int)
    test_idx = split["test_idx"]
    X_all = df[src_cols].to_numpy(float)
    Y_all = df[tgt_cols].to_numpy(float)
    if train_category is not None:
        if "category" not in df.columns:
            raise ValueError("category column required for train_category filter")
        cat_mask = df.iloc[fit_idx_all]["category"].eq(train_category).to_numpy()
        fit_idx = fit_idx_all[cat_mask]
    else:
        fit_idx = fit_idx_all
    X_fit = X_all[fit_idx]
    Y_fit = Y_all[fit_idx]
    X_te, Y_te = X_all[test_idx], Y_all[test_idx]
    ceiling = check_z3_ceiling(X_all[fit_idx_all], Y_all[fit_idx_all], X_te, Y_te)
    ceil_r2 = float(ceiling["ceiling_R2_mean"])
    target = Z5_CEILING_FRAC * ceil_r2

    reps = candidate_reps_per_cell(fit_idx, X_fit, n_per_cell=Z5_GREEDY_CAND_PER_CELL, n=Z5_GRID_N)
    unused = list(range(len(reps)))
    if shuffle_reps and len(unused) > 1:
        rng = np.random.default_rng(int(restart_seed))
        rng.shuffle(unused)

    S: list[int] = []
    path: list[dict[str, Any]] = []
    occupied: set[int] = set()
    rng_tie = np.random.default_rng(int(restart_seed) + 9001)

    def eval_S(s_idx: list[int]) -> float:
        if len(s_idx) == 0:
            return float("nan")
        return z5_eval_anchor_set(s_idx, X_all, Y_all, X_te, Y_te, ridge_alpha=ridge_alpha)

    k_limit = int(fixed_k) if fixed_k is not None else Z5_GREEDY_MAX_K
    while len(S) < k_limit and unused:
        best_u: int | None = None
        best_r2 = -np.inf
        best_new_cell = False
        tied: list[int] = []
        for u in unused:
            cand = S + [reps[u]["image_idx"]]
            r2 = eval_S(cand)
            if not np.isfinite(r2):
                continue
            opens = reps[u]["cell_id"] not in occupied
            if r2 > best_r2 + 1e-12:
                best_r2 = r2
                best_u = u
                best_new_cell = opens
                tied = [u]
            elif abs(r2 - best_r2) <= 1e-12:
                tied.append(u)
                if opens and not best_new_cell:
                    best_u = u
                    best_new_cell = True
        if best_u is None and tied:
            best_u = int(rng_tie.choice(tied))
            best_r2 = eval_S(S + [reps[tied[0]]["image_idx"]])
        if best_u is None:
            break
        S.append(reps[best_u]["image_idx"])
        occupied.add(int(reps[best_u]["cell_id"]))
        unused.remove(best_u)
        cov = coverage_stats(X_all[np.asarray(S, int)], n=Z5_GRID_N)
        path.append({
            "step": len(S),
            "added_image_idx": int(reps[best_u]["image_idx"]),
            "added_cell_id": int(reps[best_u]["cell_id"]),
            "R2_mean": float(best_r2),
            "coverage_count": cov["coverage_count"],
            "coverage_rate": cov["coverage_rate"],
            "frac_of_ceiling": float(best_r2 / ceil_r2) if ceil_r2 > 1e-12 else float("nan"),
            "hit_target": bool(best_r2 >= target),
        })
        if fixed_k is not None:
            if len(S) >= fixed_k:
                break
        elif best_r2 >= target and len(S) >= 3:
            break

    final_r2 = path[-1]["R2_mean"] if path else float("nan")
    final_cov = path[-1]["coverage_count"] if path else 0
    hit = bool(final_r2 >= target and (len(S) >= 3 if fixed_k is None else len(S) >= fixed_k))
    rep_by_cell: dict[int, int] = {}
    for step in path:
        rep_by_cell[int(step["added_cell_id"])] = int(step["added_image_idx"])

    return {
        "restart_seed": int(restart_seed),
        "ceiling_R2_mean": ceil_r2,
        "target_R2": float(target),
        "ceiling_frac": Z5_CEILING_FRAC,
        "path": path,
        "final": {
            "k": int(len(S)),
            "image_idx": [int(i) for i in S],
            "R2_mean": float(final_r2),
            "coverage_count": int(final_cov),
            "coverage_rate": float(final_cov / (Z5_GRID_N ** 2)),
            "frac_of_ceiling": float(final_r2 / ceil_r2) if ceil_r2 > 1e-12 else float("nan"),
            "hit_90pct_ceiling": hit,
            "cell_ids": sorted(occupied),
        },
        "rep_image_by_cell": rep_by_cell,
        "n_nonempty_cells_in_fit": int(len({r["cell_id"] for r in reps})),
        "n_candidate_reps": int(len(reps)),
        "train_category": train_category,
        "n_fit_images": int(len(fit_idx)),
        "fixed_k": fixed_k,
    }


def eval_r2_by_test_category(
    df: pd.DataFrame,
    split: dict[str, Any],
    src_cols: list[str],
    tgt_cols: list[str],
    image_idx: list[int],
    *,
    categories: tuple[str, ...] = Z5_CATEGORY_ORDER,
    ridge_alpha: float = Z5_RIDGE_ALPHA,
) -> dict[str, Any]:
    """Held-out R² overall and per semantic test category."""
    test_idx = np.asarray(split["test_idx"], int)
    X_all = df[src_cols].to_numpy(float)
    Y_all = df[tgt_cols].to_numpy(float)
    X_te, Y_te = X_all[test_idx], Y_all[test_idx]
    test_cats = df.iloc[test_idx]["category"].astype(str)
    if len(image_idx) == 0:
        return {"overall": {"R2_mean": float("nan"), "n": int(len(test_idx))}, "by_category": {}}
    phi, _ = fit_shared_map(
        X_all[np.asarray(image_idx, int)],
        Y_all[np.asarray(image_idx, int)],
        len(image_idx),
        mode="ridge",
        ridge_alpha=ridge_alpha,
    )
    overall = eval_understanding(phi, X_te, Y_te)
    by_cat: dict[str, Any] = {}
    for cat in categories:
        mask = test_cats.eq(cat).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        ev = eval_understanding(phi, X_te[mask], Y_te[mask])
        by_cat[cat] = {
            "R2_mean": float(ev["R2_mean"]),
            "R2_valence": float(ev["R2_valence"]),
            "R2_arousal": float(ev["R2_arousal"]),
            "n": n,
        }
    return {
        "overall": {
            "R2_mean": float(overall["R2_mean"]),
            "R2_valence": float(overall["R2_valence"]),
            "R2_arousal": float(overall["R2_arousal"]),
            "n": int(len(test_idx)),
        },
        "by_category": by_cat,
    }


def _anchor_category_summary(df: pd.DataFrame, image_idx: list[int]) -> dict[str, Any]:
    cats = df.iloc[image_idx]["category"].astype(str).tolist() if len(image_idx) else []
    counts: dict[str, int] = {}
    for c in cats:
        counts[c] = counts.get(c, 0) + 1
    return {
        "categories": cats,
        "category_counts": counts,
        "all_same_category": len(set(cats)) <= 1 if cats else False,
        "dominant_category": max(counts, key=counts.get) if counts else None,
    }








def stability_multi_restart(
    df: pd.DataFrame,
    split: dict[str, Any],
    src_cols: list[str],
    tgt_cols: list[str],
    *,
    n_restarts: int = Z5_N_GREEDY_RESTARTS,
    base_seed: int = RANDOM_SEED,
    ridge_alpha: float = Z5_RIDGE_ALPHA,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    cell_sets: list[frozenset[int]] = []
    for r in range(n_restarts):
        out = greedy_minimal_coverage(
            df, split, src_cols, tgt_cols,
            ridge_alpha=ridge_alpha,
            restart_seed=base_seed + r * 17,
            shuffle_reps=True,
        )
        runs.append(out)
        cell_sets.append(frozenset(out["final"]["cell_ids"]))

    freq: dict[int, int] = {}
    set_counts: dict[tuple[int, ...], int] = {}
    for cs in cell_sets:
        for c in cs:
            freq[int(c)] = freq.get(int(c), 0) + 1
        key = tuple(sorted(cs))
        set_counts[key] = set_counts.get(key, 0) + 1

    n_runs = len(runs)
    cell_freq = {str(k): float(v / n_runs) for k, v in sorted(freq.items())}
    jaccard_median = z5_median_pairwise_jaccard(cell_sets)
    tag = "STABLE_SET" if jaccard_median >= Z5_STABILITY_JACCARD_PASS else "UNSTABLE"

    best_key = max(set_counts, key=lambda k: set_counts[k])
    most_common_set = list(best_key)
    most_common_frac = float(set_counts[best_key] / n_runs)

    return {
        "n_restarts": n_runs,
        "jaccard_median": float(jaccard_median),
        "stability_tag": tag,
        "cell_freq": cell_freq,
        "most_common_cell_set": most_common_set,
        "most_common_set_frac": most_common_frac,
        "runs": runs,
        "cell_sets": [sorted(list(cs)) for cs in cell_sets],
    }


def random_minimal_search(
    df: pd.DataFrame,
    split: dict[str, Any],
    src_cols: list[str],
    tgt_cols: list[str],
    *,
    k: int,
    target_r2: float,
    n_trials: int = Z5_N_RANDOM_MINIMAL_TRIALS,
    seed: int = RANDOM_SEED + 5000,
    ridge_alpha: float = Z5_RIDGE_ALPHA,
    reference_cell_set: set[int] | None = None,
) -> dict[str, Any]:
    fit_idx = np.asarray(split["fit_idx"], int)
    test_idx = split["test_idx"]
    X_all = df[src_cols].to_numpy(float)
    Y_all = df[tgt_cols].to_numpy(float)
    X_te, Y_te = X_all[test_idx], Y_all[test_idx]
    rng = np.random.default_rng(seed)

    best: dict[str, Any] | None = None
    match_ref = 0
    hits = 0
    for t in range(n_trials):
        if len(fit_idx) < k:
            break
        S = rng.choice(fit_idx, size=k, replace=False)
        r2 = z5_eval_anchor_set(S.tolist(), X_all, Y_all, X_te, Y_te, ridge_alpha=ridge_alpha)
        if not np.isfinite(r2) or r2 < target_r2:
            continue
        hits += 1
        cov = coverage_stats(X_all[S], n=Z5_GRID_N)
        cells = set(int(c) for c in cov["cell_ids"])
        if reference_cell_set is not None and cells == reference_cell_set:
            match_ref += 1
        cand = {
            "trial": int(t),
            "k": int(k),
            "R2_mean": float(r2),
            "coverage_count": int(cov["coverage_count"]),
            "cell_ids": sorted(cells),
            "image_idx": [int(i) for i in S],
        }
        if best is None or cand["coverage_count"] < best["coverage_count"]:
            best = cand
        elif best is not None and cand["coverage_count"] == best["coverage_count"] and cand["R2_mean"] > best["R2_mean"]:
            best = cand

    return {
        "n_trials": int(n_trials),
        "k": int(k),
        "target_R2": float(target_r2),
        "n_hit_target": int(hits),
        "best_trial": best,
        "match_reference_cell_set_frac": (
            float(match_ref / hits) if hits > 0 and reference_cell_set is not None else float("nan")
        ),
    }


def exhaustive_minimal_cells(
    df: pd.DataFrame,
    split: dict[str, Any],
    src_cols: list[str],
    tgt_cols: list[str],
    *,
    k: int,
    target_r2: float,
    ridge_alpha: float = Z5_RIDGE_ALPHA,
) -> dict[str, Any] | None:
    fit_idx = split["fit_idx"]
    test_idx = split["test_idx"]
    X_all = df[src_cols].to_numpy(float)
    Y_all = df[tgt_cols].to_numpy(float)
    X_fit = X_all[fit_idx]
    X_te, Y_te = X_all[test_idx], Y_all[test_idx]

    reps = candidate_reps_per_cell(fit_idx, X_fit, n_per_cell=1, n=Z5_GRID_N)
    cell_to_image: dict[int, int] = {int(r["cell_id"]): int(r["image_idx"]) for r in reps}
    cells = sorted(cell_to_image.keys())
    if len(cells) > Z5_EXHAUSTIVE_MAX_CELLS or len(cells) < k:
        return None

    best: dict[str, Any] | None = None
    n_eval = 0
    for combo in combinations(cells, k):
        S = [cell_to_image[c] for c in combo]
        r2 = z5_eval_anchor_set(S, X_all, Y_all, X_te, Y_te, ridge_alpha=ridge_alpha)
        n_eval += 1
        if not np.isfinite(r2) or r2 < target_r2:
            continue
        cand = {
            "cell_ids": sorted(combo),
            "image_idx": S,
            "R2_mean": float(r2),
            "coverage_count": int(k),
        }
        if best is None or cand["R2_mean"] > best["R2_mean"]:
            best = cand

    return {
        "n_nonempty_cells": int(len(cells)),
        "n_combinations_evaluated": int(n_eval),
        "k": int(k),
        "best": best,
    }


def z5_count_images_in_cell(
    fit_idx: np.ndarray,
    X_fit: np.ndarray,
    cell_id: int,
    n: int = Z5_GRID_N,
) -> int:
    cells = va_cell_ids(X_fit, n=n)
    return int(np.sum(cells == cell_id))


def z5_pick_representative_cells(stability: dict[str, Any]) -> tuple[list[int], str]:
    """Return representative cell ids and selection rule label."""
    runs = stability["runs"]
    k_ref = int(runs[0]["final"]["k"])
    tag = stability["stability_tag"]
    if tag == "STABLE_SET" and stability["most_common_set_frac"] >= 0.5:
        return list(stability["most_common_cell_set"]), "most_common_exact_set"
    freq = {int(k): float(v) for k, v in stability["cell_freq"].items()}
    ranked = sorted(freq, key=lambda c: (-freq[c], c))
    core = [c for c in ranked if freq[c] >= Z5_CELL_FREQ_REPORT_MIN]
    if len(core) >= k_ref:
        return sorted(core[:k_ref]), "top_freq_core_cells"
    return sorted(ranked[:k_ref]), "top_freq_cells"


def characterize_cells(
    cell_ids: list[int],
    rep_image_by_cell: dict[int, int],
    fit_idx: np.ndarray,
    X_fit: np.ndarray,
    refs: dict[str, Any],
    df: pd.DataFrame,
    *,
    agreement: dict[int, dict[str, float]] | None = None,
) -> list[dict[str, Any]]:
    edges = z4_grid_edges(Z5_GRID_N)
    ov, oa = refs["oasis_centroid"]
    gv, ga = refs["gender_fp"]
    cv, ca = refs["culture_fp"]
    profiles: list[dict[str, Any]] = []
    for cid in sorted(cell_ids):
        v, a = z5_cell_center(cid, edges)
        pt = np.array([v, a])
        prof: dict[str, Any] = {
            "cell_id": int(cid),
            "v": float(v),
            "a": float(a),
            "quadrant": z5_quadrant_label(v, a, ov, oa),
            "dist_centroid": float(np.linalg.norm(pt - np.array([ov, oa]))),
            "dist_gender_fp": float(np.linalg.norm(pt - np.array([gv, ga]))),
            "dist_culture_fp": float(np.linalg.norm(pt - np.array([cv, ca]))),
            "n_images_in_cell_fit": z5_count_images_in_cell(fit_idx, X_fit, cid),
            "image_id": int(rep_image_by_cell.get(cid, -1)),
        }
        if agreement and prof["image_id"] >= 0 and prof["image_id"] in agreement:
            prof.update(agreement[prof["image_id"]])
        profiles.append(prof)
    return profiles


def overlap_metrics(cells_a: list[int], cells_b: list[int]) -> dict[str, Any]:
    sa, sb = set(cells_a), set(cells_b)
    shared = sorted(sa & sb)
    only_a = sorted(sa - sb)
    only_b = sorted(sb - sa)
    j = z5_jaccard(sa, sb)
    if j >= Z5_OVERLAP_JACCARD_HIGH:
        label = "largely_shared_anchors"
    elif j < Z5_OVERLAP_JACCARD_LOW:
        label = "direction_specific_anchors"
    else:
        label = "partial_overlap"
    return {
        "jaccard": float(j),
        "interpretation": label,
        "shared_cells": shared,
        "unique_a": only_a,
        "unique_b": only_b,
        "symmetric_diff": sorted(sa ^ sb),
    }


def agreement_proxy_sd(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Image-level group-rater SD proxy (gender bridge; not individual joint VA)."""
    cols = [
        "valence_male_sd", "arousal_male_sd",
        "valence_female_sd", "arousal_female_sd",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing SD columns for agreement proxy: {missing}")

    out = df.copy()
    out["sd_m"] = out[["valence_male_sd", "arousal_male_sd"]].mean(axis=1)
    out["sd_f"] = out[["valence_female_sd", "arousal_female_sd"]].mean(axis=1)
    out["sd_pair"] = out[["sd_m", "sd_f"]].mean(axis=1)
    global_median = float(out["sd_pair"].median())
    out["sd_pair_percentile"] = out["sd_pair"].rank(pct=True, method="average")

    summary = {
        "global_median_sd_pair": global_median,
        "global_mean_sd_pair": float(out["sd_pair"].mean()),
        "n_images": int(len(out)),
        "metric": "group_sd_mean",
        "note": "Population-level rater SD proxy; not individual (V,A) joint agreement.",
    }
    keep = ["image_id", "sd_m", "sd_f", "sd_pair", "sd_pair_percentile"]
    if "image_filename" in out.columns:
        keep.insert(1, "image_filename")
    return out[keep], summary


def run_z5_one_direction(
    df: pd.DataFrame,
    split: dict[str, Any],
    src_cols: list[str],
    tgt_cols: list[str],
    *,
    direction_short: str,
    base_seed: int = RANDOM_SEED,
    stability_only: bool = False,
    include_agreement: bool = False,
) -> dict[str, Any]:
    refs = z5_load_reference_points()
    fit_idx = split["fit_idx"]
    X_fit = df[src_cols].to_numpy(float)[fit_idx]

    print(f"[Z5] {direction_short} stability ({Z5_N_GREEDY_RESTARTS} greedy restarts) ...", flush=True)
    stability = stability_multi_restart(
        df, split, src_cols, tgt_cols, n_restarts=Z5_N_GREEDY_RESTARTS, base_seed=base_seed
    )
    rep_cells, rep_rule = z5_pick_representative_cells(stability)
    primary_run = stability["runs"][0]
    rep_by_cell = primary_run["rep_image_by_cell"]
    for run in stability["runs"]:
        if frozenset(run["final"]["cell_ids"]) == frozenset(rep_cells):
            rep_by_cell = run["rep_image_by_cell"]
            break

    profile: dict[str, Any] = {
        "direction": direction_short,
        "stability": {
            "tag": stability["stability_tag"],
            "jaccard_median": stability["jaccard_median"],
            "cell_freq": stability["cell_freq"],
            "most_common_cell_set": stability["most_common_cell_set"],
            "most_common_set_frac": stability["most_common_set_frac"],
            "pass_threshold": Z5_STABILITY_JACCARD_PASS,
        },
        "representative_selection": {
            "rule": rep_rule,
            "cell_ids": rep_cells,
            "k": int(len(rep_cells)),
        },
        "greedy_primary": primary_run["final"],
    }

    if stability_only:
        profile["representative_cells"] = characterize_cells(
            rep_cells, rep_by_cell, fit_idx, X_fit, refs, df
        )
        return profile

    target_r2 = float(primary_run["target_R2"])
    k_final = int(primary_run["final"]["k"])
    print(f"[Z5] {direction_short} random minimal search (k={k_final}) ...", flush=True)
    random_search = random_minimal_search(
        df, split, src_cols, tgt_cols,
        k=k_final,
        target_r2=target_r2,
        n_trials=Z5_N_RANDOM_MINIMAL_TRIALS,
        seed=base_seed + 5000,
        reference_cell_set=set(primary_run["final"]["cell_ids"]),
    )
    profile["random_minimal_search"] = random_search

    exhaustive = exhaustive_minimal_cells(
        df, split, src_cols, tgt_cols, k=k_final, target_r2=target_r2
    )
    if exhaustive is not None:
        print(f"[Z5] {direction_short} exhaustive cell combos ...", flush=True)
        profile["exhaustive_minimal"] = exhaustive

    agreement_map: dict[int, dict[str, float]] | None = None
    agreement_stats: dict[str, Any] | None = None
    if include_agreement:
        sd_df, sd_summary = agreement_proxy_sd(df)
        agreement_map = {
            int(idx): {
                "image_id_label": str(r["image_id"]),
                "sd_m": float(r["sd_m"]),
                "sd_f": float(r["sd_f"]),
                "sd_pair": float(r["sd_pair"]),
                "sd_pair_percentile": float(r["sd_pair_percentile"]),
            }
            for idx, r in sd_df.iterrows()
        }
        rep_row_idx = [rep_by_cell[c] for c in rep_cells if c in rep_by_cell]
        rep_sd = sd_df.loc[rep_row_idx, "sd_pair"].to_numpy(float)
        rest_mask = ~sd_df.index.isin(rep_row_idx)
        rest_sd = sd_df.loc[rest_mask, "sd_pair"].to_numpy(float)
        mw_p = float("nan")
        if len(rep_sd) >= 1 and len(rest_sd) >= 1:
            try:
                _, mw_p = mannwhitneyu(rep_sd, rest_sd, alternative="two-sided")
            except ValueError:
                mw_p = float("nan")
        agreement_stats = {
            **sd_summary,
            "minimal_median_sd_pair": float(np.median(rep_sd)) if len(rep_sd) else float("nan"),
            "rest_median_sd_pair": float(np.median(rest_sd)) if len(rest_sd) else float("nan"),
            "mannwhitney_p": float(mw_p),
            "interpretation": (
                "consensus_proxy" if np.nanmedian(rep_sd) < sd_summary["global_median_sd_pair"]
                else "contested_proxy"
            ),
        }
        profile["agreement"] = agreement_stats

    profile["representative_cells"] = characterize_cells(
        rep_cells, rep_by_cell, fit_idx, X_fit, refs, df, agreement=agreement_map
    )
    profile["_stability_runs"] = [
        {
            "run": i,
            "restart_seed": run["restart_seed"],
            "cell_ids": run["final"]["cell_ids"],
            "k": run["final"]["k"],
            "R2_mean": run["final"]["R2_mean"],
            "coverage_count": run["final"]["coverage_count"],
            "frac_of_ceiling": run["final"]["frac_of_ceiling"],
        }
        for i, run in enumerate(stability["runs"])
    ]
    return profile


def plot_z5_profile(
    profiles: dict[str, dict[str, Any]],
    overlap: dict[str, Any],
    out_path: Path,
    *,
    include_agreement: bool = False,
    sd_df: pd.DataFrame | None = None,
) -> None:
    refs = z5_load_reference_points()
    ov, oa = refs["oasis_centroid"]
    gv, ga = refs["gender_fp"]
    n_panels = 3 if include_agreement and sd_df is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.8))
    if n_panels == 1:
        axes = [axes]

    colors = {"MtoF": "#1565c0", "FtoM": "#c62828"}
    lim = Z5_VA_LIM

    ax = axes[0]
    ax.axhline(oa, color="#888", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(ov, color="#888", ls="--", lw=0.8, alpha=0.6)
    ax.scatter([gv], [ga], s=80, marker="*", c="#2e7d32", edgecolors="white", zorder=5, label="Gender FP")
    ax.scatter([ov], [oa], s=60, marker="D", c="#616161", edgecolors="white", zorder=5, label="OASIS centroid")
    primary = profiles.get("MtoF") or next(iter(profiles.values()))
    for cell in primary["representative_cells"]:
        ax.scatter(
            [cell["v"]], [cell["a"]], s=120, c=colors.get(primary["direction"], "#1565c0"),
            edgecolors="white", linewidths=0.8, zorder=4,
        )
        ax.text(cell["v"], cell["a"], str(cell["cell_id"]), ha="center", va="center", fontsize=7, color="white")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    stab = primary["stability"]["tag"]
    ax.set_title(f"Z5-1: minimal anchors ({primary['direction']}, {stab})")

    ax = axes[1]
    ax.axhline(oa, color="#888", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(ov, color="#888", ls="--", lw=0.8, alpha=0.6)
    for dshort, prof in profiles.items():
        col = colors.get(dshort, "#333")
        for cell in prof["representative_cells"]:
            ax.scatter(
                [cell["v"]], [cell["a"]], s=100, c=col, alpha=0.85,
                edgecolors="white", linewidths=0.6, label=dshort if cell == prof["representative_cells"][0] else "",
            )
            ax.text(cell["v"], cell["a"], str(cell["cell_id"]), ha="center", va="center", fontsize=6, color="white")
    j = overlap.get("jaccard", float("nan"))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(f"Z5-2: direction overlap (Jaccard={j:.2f})")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8)

    if n_panels >= 3 and sd_df is not None:
        ax = axes[2]
        rep_row_idx: set[int] = set()
        for prof in profiles.values():
            for cell in prof["representative_cells"]:
                if cell.get("image_id", -1) >= 0:
                    rep_row_idx.add(int(cell["image_id"]))
        minimal = sd_df.loc[list(rep_row_idx), "sd_pair"] if rep_row_idx else sd_df.iloc[0:0]["sd_pair"]
        rest = sd_df.loc[~sd_df.index.isin(rep_row_idx), "sd_pair"]
        parts = ax.violinplot([rest, minimal], positions=[0, 1], showmeans=True, showmedians=True)
        for b in parts["bodies"]:
            b.set_alpha(0.65)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Other images", "Minimal anchors"])
        ax.set_ylabel("Group SD proxy (sd_pair)")
        ax.set_title("Z5-3: agreement proxy")

    fig.suptitle("Z5 minimal-coverage anchor profile", y=1.02, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_z5_all(
    df: pd.DataFrame,
    split: dict[str, Any],
    specs: list[tuple[str, list[str], list[str], str]],
    out_dir: Path,
    *,
    base_seed: int = RANDOM_SEED,
    stability_only: bool = False,
    include_agreement: bool = False,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    profiles: dict[str, dict[str, Any]] = {}
    stability_rows: list[dict[str, Any]] = []
    freq_rows: list[dict[str, Any]] = []

    for bridge, src, tgt, short in specs:
        prof = run_z5_one_direction(
            df, split, src, tgt,
            direction_short=short,
            base_seed=base_seed + (0 if short == "MtoF" else 100),
            stability_only=stability_only,
            include_agreement=include_agreement,
        )
        prof["bridge"] = bridge
        profiles[short] = prof

        for row in prof.get("_stability_runs", []):
            stability_rows.append({
                "bridge": bridge,
                "direction": short,
                **row,
                "cell_ids_str": ",".join(str(c) for c in row["cell_ids"]),
            })
        for cid, freq in prof["stability"]["cell_freq"].items():
            freq_rows.append({
                "bridge": bridge,
                "direction": short,
                "cell_id": int(cid),
                "selection_freq": float(freq),
                "is_core": float(freq) >= Z5_CELL_FREQ_REPORT_MIN,
            })

    overlap: dict[str, Any] = {}
    if "MtoF" in profiles and "FtoM" in profiles:
        c_m = profiles["MtoF"]["representative_selection"]["cell_ids"]
        c_f = profiles["FtoM"]["representative_selection"]["cell_ids"]
        overlap = overlap_metrics(c_m, c_f)
        overlap["MtoF_k"] = profiles["MtoF"]["representative_selection"]["k"]
        overlap["FtoM_k"] = profiles["FtoM"]["representative_selection"]["k"]
        profiles["MtoF"]["overlap_with_opposite"] = {
            "jaccard": overlap["jaccard"],
            "shared_cells": overlap["shared_cells"],
            f"unique_{profiles['MtoF']['direction']}": overlap["unique_a"],
            f"unique_{profiles['FtoM']['direction']}": overlap["unique_b"],
            "interpretation": overlap["interpretation"],
        }
        profiles["FtoM"]["overlap_with_opposite"] = {
            "jaccard": overlap["jaccard"],
            "shared_cells": overlap["shared_cells"],
            f"unique_{profiles['MtoF']['direction']}": overlap["unique_a"],
            f"unique_{profiles['FtoM']['direction']}": overlap["unique_b"],
            "interpretation": overlap["interpretation"],
        }

    sd_df: pd.DataFrame | None = None
    if include_agreement:
        sd_df, _ = agreement_proxy_sd(df)
        sd_df.to_csv(out_dir / "z5_agreement_proxy.csv", index=False)

    if stability_rows:
        pd.DataFrame(stability_rows).to_csv(out_dir / "z5_stability_runs.csv", index=False)
    if freq_rows:
        pd.DataFrame(freq_rows).to_csv(out_dir / "z5_cell_selection_frequency.csv", index=False)

    export_profiles = {}
    for short, prof in profiles.items():
        export_prof = {k: v for k, v in prof.items() if not k.startswith("_")}
        export_profiles[short] = export_prof

    payload = {
        "spec_version": "z5_minimal_anchor_profile_v1",
        "preregistered": True,
        "stability_only": stability_only,
        "include_agreement": include_agreement,
        "constants": {
            "n_greedy_restarts": Z5_N_GREEDY_RESTARTS,
            "stability_jaccard_pass": Z5_STABILITY_JACCARD_PASS,
            "cell_freq_report_min": Z5_CELL_FREQ_REPORT_MIN,
            "n_random_trials": Z5_N_RANDOM_MINIMAL_TRIALS,
        },
        "profiles": export_profiles,
        "overlap": overlap,
    }
    (out_dir / "z5_minimal_anchor_profile.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if overlap:
        (out_dir / "z5_overlap.json").write_text(
            json.dumps(overlap, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if not stability_only:
        plot_z5_profile(
            profiles,
            overlap,
            out_dir / "Fig_minimal_anchor_profile.png",
            include_agreement=include_agreement,
            sd_df=sd_df,
        )

    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Anchor-structure Z1/Z2/Z3/Z4/Z5/Z6"
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="z3",
        choices=["z1", "z2", "z3", "z4", "z5", "z6", "both"],
    )
    ap.add_argument("--split-seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--kmeans-seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--n-perm", type=int, default=N_PERM_DEFAULT)
    ap.add_argument("--k-max", type=int, default=6)
    ap.add_argument("--n-boot", type=int, default=Z1_N_BOOT_DEFAULT)
    ap.add_argument("--z3-draws", type=int, default=Z3_N_RANDOM_DRAWS)
    ap.add_argument(
        "--z3-affine-mode",
        type=str,
        default=Z3_AFFINE_MODE_DEFAULT,
        choices=["ridge", "naive"],
        help="k>=3 shared-anchor fit: ridge (main) or naive LS (diagnostic)",
    )
    ap.add_argument("--z3-ridge-alpha", type=float, default=Z3_RIDGE_ALPHA)
    ap.add_argument("--z4-draws", type=int, default=Z4_N_DRAWS)
    ap.add_argument(
        "--z5-stability-only",
        action="store_true",
        help="Z5: run stability check only (skip random search, figure)",
    )
    ap.add_argument(
        "--z5-agreement",
        action="store_true",
        help="Z5: include group-SD agreement proxy (5-3)",
    )
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    ap.add_argument(
        "--directions",
        type=str,
        default="both",
        choices=["both", "M_to_F", "F_to_M"],
        help="Gender bridge directions (default: both; maps are asymmetric)",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k_list = tuple(range(1, int(args.k_max) + 1))

    df = load_gender_va()
    split = make_theme_split(df, args.split_seed)

    z2_specs: list[tuple[str, list[str], list[str]]] = []
    z1_specs: list[tuple[str, list[str], list[str], str]] = []
    z3_specs: list[tuple[str, list[str], list[str], str]] = []
    z4_specs: list[tuple[str, list[str], list[str], str]] = []
    z5_specs: list[tuple[str, list[str], list[str], str]] = []
    if args.directions in ("both", "M_to_F"):
        z2_specs.append(("gender_M_to_F", SRC_M, SRC_F))
        z1_specs.append(("gender_M_to_F", SRC_M, SRC_F, "MtoF"))
        z3_specs.append(("gender_M_to_F", SRC_M, SRC_F, "MtoF"))
        z4_specs.append(("gender_M_to_F", SRC_M, SRC_F, "MtoF"))
        z5_specs.append(("gender_M_to_F", SRC_M, SRC_F, "MtoF"))
    if args.directions in ("both", "F_to_M"):
        z2_specs.append(("gender_F_to_M", SRC_F, SRC_M))
        z1_specs.append(("gender_F_to_M", SRC_F, SRC_M, "FtoM"))
        z3_specs.append(("gender_F_to_M", SRC_F, SRC_M, "FtoM"))
        z4_specs.append(("gender_F_to_M", SRC_F, SRC_M, "FtoM"))
        z5_specs.append(("gender_F_to_M", SRC_F, SRC_M, "FtoM"))

    results_by_bridge: dict[str, dict] = {}
    z1_payload = None
    z3_payload = None
    z4_payload = None

    if args.mode in ("z2", "both"):
        results_by_bridge = run_z2_pipeline(
            df, split, z2_specs, out_dir, k_list, args.kmeans_seed, args.n_perm
        )
    elif (out_dir / "z2_ksweep_gender.json").exists():
        prev = json.loads((out_dir / "z2_ksweep_gender.json").read_text(encoding="utf-8"))
        win_csv = out_dir / "z2_win_table.csv"
        if win_csv.exists():
            wdf = pd.read_csv(win_csv)
            for name, br in prev.get("bridges", {}).items():
                sub = wdf[wdf["bridge"] == name].to_dict(orient="records")
                for r in sub:
                    if r["k"] != "ot":
                        try:
                            r["k"] = int(r["k"])
                        except Exception:
                            pass
                    r["win_vs_k1"] = (
                        bool(r["win_vs_k1"])
                        if not isinstance(r["win_vs_k1"], bool)
                        else r["win_vs_k1"]
                    )
                results_by_bridge[name] = {
                    "decision": br["decision"],
                    "monotonicity": br["monotonicity"],
                    "stability": br.get("stability", {}),
                    "win_rows": sub,
                }

    if args.mode in ("z1", "both"):
        z1_out = run_z1_all(
            df,
            split,
            z1_specs,
            out_dir,
            n_boot=args.n_boot,
            boot_seed=args.split_seed + 777,
        )
        z1_payload = z1_out["payload"]

    if args.mode in ("z3", "both"):
        z3_payload = run_z3_all(
            df,
            split,
            z3_specs,
            out_dir,
            n_draws=args.z3_draws,
            draw_seed=args.split_seed + Z3_DRAW_SEED_OFFSET,
            affine_mode=args.z3_affine_mode,
            ridge_alpha=args.z3_ridge_alpha,
        )

    if args.mode == "z4":
        z4_payload = run_z4_all(
            df,
            split,
            z4_specs,
            out_dir,
            n_draws=args.z4_draws,
            draw_seed=args.split_seed + Z4_DRAW_SEED_OFFSET,
        )
        print(json.dumps({
            "primary_upgrade": z4_payload.get("primary_upgrade"),
            "bridges": {
                k: {
                    "decision": v["upgrade_decision"],
                    "z4_1": v["z4_1_pass"],
                    "z4_3": v["z4_3_pass"],
                    "minimal": v["minimal_final"],
                }
                for k, v in z4_payload["bridges"].items()
            },
        }, indent=2))
        print(f"Saved outputs under {out_dir}")
        return


    if args.mode == "z5":
        z5_payload = run_z5_all(
            df,
            split,
            z5_specs,
            out_dir,
            base_seed=args.split_seed,
            stability_only=args.z5_stability_only,
            include_agreement=args.z5_agreement,
        )
        summary = {
            d: {
                "stability": p["stability"]["tag"],
                "jaccard_median": p["stability"]["jaccard_median"],
                "representative_cells": p["representative_selection"]["cell_ids"],
                "k": p["representative_selection"]["k"],
                "R2": p["greedy_primary"]["R2_mean"],
                "frac_ceiling": p["greedy_primary"]["frac_of_ceiling"],
            }
            for d, p in z5_payload["profiles"].items()
        }
        if z5_payload.get("overlap"):
            summary["overlap"] = z5_payload["overlap"]
        print(json.dumps(summary, indent=2))
        print(f"Saved outputs under {out_dir}")
        return

    if args.mode == "z6":
        from analysis_z6_polygon_orientation import run_z6

        z6_payload = run_z6(out_dir)
        g = z6_payload["gate"]
        print(json.dumps({
            "gate_tag": g["gate_tag"],
            "gate_detail": g["detail"],
            "directions": {
                d: {
                    "cells": p["cell_ids"],
                    "det_A": p["phi"]["det"],
                    "L_cell": {
                        "T0": p["L_cell"].get("T0_phi", {}).get("orientation"),
                        "T1": p["L_cell"].get("T1_truth", {}).get("orientation"),
                        "T2": p["L_cell"].get("T2_phi_to_truth", {}).get("orientation"),
                    },
                    "L_image": {
                        "T0": p["L_image"].get("T0_phi", {}).get("orientation"),
                        "T1": p["L_image"].get("T1_truth", {}).get("orientation"),
                        "T2": p["L_image"].get("T2_phi_to_truth", {}).get("orientation"),
                    },
                    "mean_excess_twist": p["mean_excess_twist"],
                }
                for d, p in z6_payload["directions"].items()
            },
        }, indent=2))
        print(f"Saved outputs under {out_dir}")
        return

    write_summary(
        out_dir,
        split,
        results_by_bridge,
        k_list,
        z1_payload=z1_payload,
        z3_payload=z3_payload,
    )
    print(f"Saved outputs under {out_dir}")


if __name__ == "__main__":
    main()
