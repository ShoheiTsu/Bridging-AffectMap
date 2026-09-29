#!/usr/bin/env python3
"""
Feasibility: predictive case translation without peeking at the target rating.

Female-target estimand (primary M→F direction):
  For held-out image i, predict translation partner j without using y_f(i),
  then estimate y_f(i) ≈ pred_m(j).

Baselines on the same held-out queries:
  1. oracle_1nn   — current method (uses y_f(i) to pick j; upper bound / unfair)
  2. clip_nn_xfer — nearest train image in CLIP → reuse its oracle j'
  3. clip_nn_mean — nearest-k train images → majority/mean of their j' (k=5)
  4. phi_same     — Φ_mf(y_m(i)) with Φ fit on train fold only
  5. raw_same     — pred_m(i) (male decoder on same image; no translation)

Outputs JSON under results/equivalence_nontriviality/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from analysis_emotion_equivalent_pairs import (  # noqa: E402
    TARGET_F,
    TARGET_M,
    build_1nn_rows,
    load_common_preds,
    theme_kfold_indices,
)
from analysis_population_bridge_suite import fit_affine  # noqa: E402
from config import CVAE_CROSS_GENDER_DIR, OASIS_SCORES_CSV, RANDOM_SEED, RESULTS_STEP1  # noqa: E402
from dataset import add_theme_base, load_oasis_meta  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "equivalence_nontriviality"


def _l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=1)


def build_oracle_map(
    df: pd.DataFrame,
    *,
    query_idx: np.ndarray,
    pool_idx: np.ndarray,
    y_f: np.ndarray,
    p_m: np.ndarray,
) -> dict[int, int]:
    """Female-target oracle: i -> j minimizing ||y_f(i) - pred_m(j)||."""
    rows = build_1nn_rows(
        df,
        te_idx=query_idx,
        pool_idx=pool_idx,
        y_f=y_f,
        p_m_all=p_m,
        p_f_all=p_m,  # unused
    )
    return {int(r["i_idx"]): int(r["j_idx"]) for r in rows}


def build_oracle_map_male_target(
    df: pd.DataFrame,
    *,
    query_idx: np.ndarray,
    pool_idx: np.ndarray,
    y_m: np.ndarray,
    p_f: np.ndarray,
) -> dict[int, int]:
    """Male-target oracle: i -> j minimizing ||y_m(i) - pred_f(j)||."""
    y_q = y_m[query_idx]
    p_pool = p_f[pool_idx]
    dist = np.sqrt(((y_q[:, None, :] - p_pool[None, :, :]) ** 2).sum(axis=2))
    pool_pos = {idx: pos for pos, idx in enumerate(pool_idx.tolist())}
    for r, i_idx in enumerate(query_idx.tolist()):
        if i_idx in pool_pos:
            dist[r, pool_pos[i_idx]] = np.inf
    best_pos = np.argmin(dist, axis=1)
    best_j = pool_idx[best_pos]
    return {int(query_idx[k]): int(best_j[k]) for k in range(len(query_idx))}


def clip_nn_j(
    x_query: np.ndarray,
    x_train: np.ndarray,
    train_idx: np.ndarray,
    oracle_j: dict[int, int],
    *,
    k: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted j indices and CLIP distances for each query row."""
    d = np.linalg.norm(x_query[:, None, :] - x_train[None, :, :], axis=2)
    k_eff = min(k, d.shape[1])
    part = np.argpartition(d, kth=k_eff - 1, axis=1)[:, :k_eff]
    j_pred = np.empty(len(x_query), dtype=int)
    clip_d = np.empty(len(x_query), dtype=float)
    for r in range(len(x_query)):
        neigh = part[r][np.argsort(d[r, part[r]])]
        if k == 1:
            i_ref = int(train_idx[int(neigh[0])])
            j_pred[r] = int(oracle_j[i_ref])
            clip_d[r] = float(d[r, int(neigh[0])])
        else:
            refs = [int(train_idx[int(p)]) for p in neigh]
            js = [oracle_j[ref] for ref in refs]
            # mode of j; tie-break by smallest CLIP distance among tied j
            vals, counts = np.unique(js, return_counts=True)
            top = vals[counts == counts.max()]
            if len(top) == 1:
                j_pred[r] = int(top[0])
            else:
                best = top[0]
                best_d = np.inf
                for ref, j in zip(refs, js):
                    if j in top:
                        pos = int(np.where(train_idx == ref)[0][0])
                        if d[r, pos] < best_d:
                            best_d = d[r, pos]
                            best = j
                j_pred[r] = int(best)
            clip_d[r] = float(d[r, int(neigh[0])])
    return j_pred, clip_d


def summarize_errors(name: str, err: np.ndarray) -> dict:
    return {
        "method": name,
        "n": int(len(err)),
        "mean_l2": float(np.mean(err)),
        "median_l2": float(np.median(err)),
    }


def run_feasibility(*, n_folds: int = 5, seed: int = RANDOM_SEED, k: int = 5) -> dict:
    ckpt = CVAE_CROSS_GENDER_DIR / "weights.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    X_clip = np.load(RESULTS_STEP1 / "features_clip.npy").astype(np.float32)
    df = load_oasis_meta(OASIS_SCORES_CSV)
    valid = df[TARGET_M + TARGET_F].notna().all(axis=1).values
    df = df.loc[valid].reset_index(drop=True)
    X_clip = X_clip[valid]
    y_f = df[TARGET_F].to_numpy(np.float32)
    y_m = df[TARGET_M].to_numpy(np.float32)
    df = add_theme_base(df)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = theme_kfold_indices(df, n_folds=n_folds, seed=seed)

    methods = ["oracle_1nn", "clip_nn_xfer", f"clip_nn_top{k}", "phi_same", "raw_same"]
    all_err: dict[str, list[float]] = {m: [] for m in methods}
    all_y: list[np.ndarray] = []
    all_pred: dict[str, list[np.ndarray]] = {m: [] for m in methods}

    for tr_idx, te_idx in folds:
        scaler = StandardScaler().fit(X_clip[tr_idx])
        Xs = scaler.transform(X_clip).astype(np.float32)
        p_m, _p_f = load_common_preds(Xs, ckpt, device)

        pool_idx = tr_idx
        oracle_train = build_oracle_map(
            df, query_idx=tr_idx, pool_idx=pool_idx, y_f=y_f, p_m=p_m,
        )
        oracle_test_rows = build_1nn_rows(
            df, te_idx=te_idx, pool_idx=pool_idx, y_f=y_f, p_m_all=p_m, p_f_all=p_m,
        )

        phi = fit_affine(y_m[tr_idx], y_f[tr_idx])
        phi_pred = phi.apply(y_m[te_idx])

        x_tr = Xs[tr_idx]
        x_te = Xs[te_idx]
        j_xfer, _ = clip_nn_j(x_te, x_tr, tr_idx, oracle_train, k=1)
        j_topk, _ = clip_nn_j(x_te, x_tr, tr_idx, oracle_train, k=k)

        y_true = y_f[te_idx]
        all_y.append(y_true)

        err_oracle = np.array([r["distance_l2"] for r in oracle_test_rows], dtype=float)
        err_xfer = _l2(y_true, p_m[j_xfer])
        err_topk = _l2(y_true, p_m[j_topk])
        err_phi = _l2(y_true, phi_pred)
        err_raw = _l2(y_true, p_m[te_idx])

        all_err["oracle_1nn"].extend(err_oracle.tolist())
        all_err["clip_nn_xfer"].extend(err_xfer.tolist())
        all_err[f"clip_nn_top{k}"].extend(err_topk.tolist())
        all_err["phi_same"].extend(err_phi.tolist())
        all_err["raw_same"].extend(err_raw.tolist())

        all_pred["oracle_1nn"].append(p_m[[int(r["j_idx"]) for r in oracle_test_rows]])
        all_pred["clip_nn_xfer"].append(p_m[j_xfer])
        all_pred[f"clip_nn_top{k}"].append(p_m[j_topk])
        all_pred["phi_same"].append(phi_pred)
        all_pred["raw_same"].append(p_m[te_idx])

    y_all = np.vstack(all_y)
    summary = []
    beat_phi: dict[str, float] = {}
    for m in methods:
        err = np.asarray(all_err[m], dtype=float)
        pred = np.vstack(all_pred[m])
        summary.append(summarize_errors(m, err))
        summary[-1]["r2_valence"] = float(r2_score(y_all[:, 0], pred[:, 0]))
        summary[-1]["r2_arousal"] = float(r2_score(y_all[:, 1], pred[:, 1]))
        summary[-1]["r2_mean"] = float(
            (summary[-1]["r2_valence"] + summary[-1]["r2_arousal"]) / 2
        )
        beat_phi[m] = float(np.mean(err < np.asarray(all_err["phi_same"], dtype=float)))

    oracle = np.asarray(all_err["oracle_1nn"], dtype=float)
    phi_e = np.asarray(all_err["phi_same"], dtype=float)
    xfer = np.asarray(all_err["clip_nn_xfer"], dtype=float)

    return {
        "task": "predictive_translation_feasibility",
        "direction": "female_target_MtoF",
        "cv": {"n_folds": n_folds, "seed": seed, "pool": "train_only"},
        "n_queries": int(len(oracle)),
        "methods": summary,
        "ratios_median_l2": {
            "oracle_over_phi": float(np.median(oracle) / np.median(phi_e)),
            "xfer_over_phi": float(np.median(xfer) / np.median(phi_e)),
            "xfer_over_oracle": float(np.median(xfer) / np.median(oracle)),
        },
        "beat_phi_rate": beat_phi,
        "interpretation_hint": (
            "If clip_nn_xfer median L2 ≈ phi_same or lower, predictive translation may be viable. "
            "If ≫ phi_same, reframing as 'equivalence exists but is not predictable from CLIP' is supported."
        ),
    }


def main() -> None:
    out = run_feasibility()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "predictive_translation_feasibility.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
