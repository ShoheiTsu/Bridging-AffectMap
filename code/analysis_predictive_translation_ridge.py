#!/usr/bin/env python3
"""
Predictive case translation via Ridge regression on frozen CLIP features.

Train (per outer fold, pool = train only):
  X = CLIP(i'),  Y = pred_m(j*_oracle(i'))   with j* = argmin ||y_f(i') - pred_m(j)||

Test (held-out i, no y_f(i) used for partner choice):
  ŷ = Ridge(X_i)  compared to y_f(i), Φ_mf(y_m(i)), CLIP-1NN transfer, pred_m(i).

Stratification (primary = raw_gap_l2 = ||y_m - y_f|| pre-Φ; see doc).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from analysis_paper2_emotion_equivalent_pairs import (  # noqa: E402
    TARGET_F,
    TARGET_M,
    build_1nn_rows,
    load_common_preds,
    theme_kfold_indices,
)
from analysis_population_bridge_suite import fit_affine  # noqa: E402
from analysis_predictive_translation_feasibility import (  # noqa: E402
    OUT_DIR,
    build_oracle_map,
    build_oracle_map_male_target,
    clip_nn_j,
    summarize_errors,
    _l2,
)
from config import (  # noqa: E402
    ALPHA_SEARCH_HIGH,
    ALPHA_SEARCH_LOW,
    CVAE_CROSS_GENDER_DIR,
    OASIS_SCORES_CSV,
    RANDOM_SEED,
    RESULTS_STEP1,
    VALENCE_AROUSAL_SCALE_MAX,
    VALENCE_AROUSAL_SCALE_MIN,
)
from dataset import add_theme_base, load_oasis_meta, train_val_split_by_theme  # noqa: E402

DECISION_RULES = {
    "median_beats_phi": "High raw-gap tercile median L2 < Phi median → claim 'surpasses'",
    "median_competes": "Does not beat Phi but median gap vs 1-NN shrinks materially → 'competes'",
    "median_no_gain": "No meaningful change vs CLIP 1-NN → same conclusion as 1-NN",
}


def _normalize_y(y: np.ndarray) -> np.ndarray:
    return (y - VALENCE_AROUSAL_SCALE_MIN) / (VALENCE_AROUSAL_SCALE_MAX - VALENCE_AROUSAL_SCALE_MIN)


def _denormalize_y(y: np.ndarray) -> np.ndarray:
    return y * (VALENCE_AROUSAL_SCALE_MAX - VALENCE_AROUSAL_SCALE_MIN) + VALENCE_AROUSAL_SCALE_MIN


def select_ridge_alpha(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    df: pd.DataFrame,
    tr_idx: np.ndarray,
    *,
    n_alphas: int = 24,
    seed: int = RANDOM_SEED,
) -> float:
    """Theme-blocked inner split on outer-train indices; maximize mean R² on inner val."""
    sub = df.iloc[tr_idx].reset_index(drop=True)
    _, _, inner_tr_rel, inner_va_rel = train_val_split_by_theme(
        sub, train_ratio=0.8, random_state=seed, return_indices=True,
    )
    if len(inner_va_rel) < 5:
        return 1.0

    scl = StandardScaler().fit(X_tr[inner_tr_rel])
    X_in_tr = scl.transform(X_tr[inner_tr_rel])
    X_in_va = scl.transform(X_tr[inner_va_rel])
    y_in_tr = _normalize_y(y_tr[inner_tr_rel])
    y_in_va = y_tr[inner_va_rel]

    alphas = np.logspace(np.log10(ALPHA_SEARCH_LOW), np.log10(ALPHA_SEARCH_HIGH), n_alphas)
    best_alpha = float(alphas[0])
    best_score = -np.inf
    for alpha in alphas:
        preds = np.zeros((len(inner_va_rel), 2), dtype=float)
        for j in range(2):
            m = Ridge(alpha=float(alpha), fit_intercept=True).fit(X_in_tr, y_in_tr[:, j])
            preds[:, j] = _denormalize_y(m.predict(X_in_va))
        score = (r2_score(y_in_va[:, 0], preds[:, 0]) + r2_score(y_in_va[:, 1], preds[:, 1])) / 2
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def fit_ridge_va(X_tr: np.ndarray, y_tr: np.ndarray, alpha: float) -> tuple[StandardScaler, list[Ridge]]:
    scl = StandardScaler().fit(X_tr)
    Xn = scl.transform(X_tr)
    yn = _normalize_y(y_tr)
    models = [
        Ridge(alpha=alpha, fit_intercept=True).fit(Xn, yn[:, j])
        for j in range(2)
    ]
    return scl, models


def predict_ridge_va(scl: StandardScaler, models: list[Ridge], X: np.ndarray) -> np.ndarray:
    Xn = scl.transform(X)
    return np.column_stack([_denormalize_y(m.predict(Xn)) for m in models])


def stratify_tercile(rec: pd.DataFrame, col: str) -> dict:
    sub = rec.dropna(subset=[col]).copy()
    sub["tercile"] = pd.qcut(sub[col], 3, labels=["low", "mid", "high"])
    out: dict = {"stratifier": col, "terciles": {}, "spearman_stratifier_vs_beat": {}, "chi2_tercile_vs_beat_p": {}}
    for method in ["clip_ridge_xfer", "clip_nn_xfer", "phi_same"]:
        if f"beat_phi_{method}" not in sub.columns:
            continue
        beat_col = f"beat_phi_{method}"
        rho, _ = stats.spearmanr(sub[col], sub[beat_col].astype(int))
        chi2, pchi, _, _ = stats.chi2_contingency(pd.crosstab(sub["tercile"], sub[beat_col]))
        out["spearman_stratifier_vs_beat"][method] = round(float(rho), 4)
        out["chi2_tercile_vs_beat_p"][method] = float(pchi)

    rows = {}
    for t in ["low", "mid", "high"]:
        s = sub[sub.tercile == t]
        row: dict = {"n": int(len(s))}
        for method, err_col in [
            ("clip_ridge_xfer", "ridge_l2"),
            ("clip_nn_xfer", "xfer_l2"),
            ("phi_same", "phi_l2"),
        ]:
            if err_col not in s.columns:
                continue
            beat_col = f"beat_phi_{method}"
            delta = s[err_col] - s["phi_l2"]
            w_p = float(stats.wilcoxon(s[err_col], s["phi_l2"]).pvalue) if len(s) else float("nan")
            row[method] = {
                "median_l2": round(float(s[err_col].median()), 4),
                "beat_phi_rate": round(float(s[beat_col].mean()), 4) if beat_col in s.columns else None,
                "median_delta_minus_phi": round(float(delta.median()), 4),
                "wilcoxon_two_sided_p": w_p,
            }
        rows[t] = row
    out["terciles"] = rows
    return out


def collect_predictive_translation_records(
    *,
    direction: str = "MtoF",
    n_folds: int = 5,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, list[float], dict[str, list[float]], dict[str, list[np.ndarray]], list[np.ndarray]]:
    if direction not in {"MtoF", "FtoM"}:
        raise ValueError(f"direction must be MtoF or FtoM, got {direction!r}")
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
    raw_gap = np.linalg.norm(y_m - y_f, axis=1)

    tw = pd.read_csv(PROJECT_ROOT / "results/relational_cross_within_twist/relational_twist_per_image.csv")
    tw["excess_twist"] = tw["geo_umap_disp"] - tw["model_twist_pm_pf"]

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = theme_kfold_indices(df, n_folds=n_folds, seed=seed)

    records: list[dict] = []
    alphas_used: list[float] = []
    all_err: dict[str, list[float]] = {
        "clip_ridge_xfer": [],
        "clip_nn_xfer": [],
        "phi_same": [],
        "raw_same": [],
    }
    all_y: list[np.ndarray] = []
    all_pred: dict[str, list[np.ndarray]] = {k: [] for k in all_err}

    for tr_idx, te_idx in folds:
        scaler = StandardScaler().fit(X_clip[tr_idx])
        Xs = scaler.transform(X_clip).astype(np.float32)
        p_m, p_f = load_common_preds(Xs, ckpt, device)

        if direction == "MtoF":
            oracle_train = build_oracle_map(
                df, query_idx=tr_idx, pool_idx=tr_idx, y_f=y_f, p_m=p_m,
            )
            y_train_targets = np.array(
                [p_m[oracle_train[int(i)]] for i in tr_idx], dtype=np.float32,
            )
            phi = fit_affine(y_m[tr_idx], y_f[tr_idx])
            phi_pred = phi.apply(y_m[te_idx])
            j_xfer, _ = clip_nn_j(Xs[te_idx], Xs[tr_idx], tr_idx, oracle_train, k=1)
            y_true = y_f[te_idx]
            xfer_pred = p_m[j_xfer]
            raw_pred = p_m[te_idx]
        else:
            oracle_train = build_oracle_map_male_target(
                df, query_idx=tr_idx, pool_idx=tr_idx, y_m=y_m, p_f=p_f,
            )
            y_train_targets = np.array(
                [p_f[oracle_train[int(i)]] for i in tr_idx], dtype=np.float32,
            )
            phi = fit_affine(y_f[tr_idx], y_m[tr_idx])
            phi_pred = phi.apply(y_f[te_idx])
            j_xfer, _ = clip_nn_j(Xs[te_idx], Xs[tr_idx], tr_idx, oracle_train, k=1)
            y_true = y_m[te_idx]
            xfer_pred = p_f[j_xfer]
            raw_pred = p_f[te_idx]

        alpha = select_ridge_alpha(Xs[tr_idx], y_train_targets, df, tr_idx, seed=seed + int(tr_idx[0] % 997))
        alphas_used.append(alpha)
        scl_r, models = fit_ridge_va(Xs[tr_idx], y_train_targets, alpha)
        ridge_pred = predict_ridge_va(scl_r, models, Xs[te_idx])

        all_y.append(y_true)
        err_ridge = _l2(y_true, ridge_pred)
        err_xfer = _l2(y_true, xfer_pred)
        err_phi = _l2(y_true, phi_pred)
        err_raw = _l2(y_true, raw_pred)

        all_err["clip_ridge_xfer"].extend(err_ridge.tolist())
        all_err["clip_nn_xfer"].extend(err_xfer.tolist())
        all_err["phi_same"].extend(err_phi.tolist())
        all_err["raw_same"].extend(err_raw.tolist())
        all_pred["clip_ridge_xfer"].append(ridge_pred)
        all_pred["clip_nn_xfer"].append(xfer_pred)
        all_pred["phi_same"].append(phi_pred)
        all_pred["raw_same"].append(raw_pred)

        for k, i in enumerate(te_idx):
            records.append(
                {
                    "image_id": df.loc[i, "image_id"],
                    "direction": direction,
                    "y_f_i_valence": float(y_f[i, 0]),
                    "y_f_i_arousal": float(y_f[i, 1]),
                    "y_m_i_valence": float(y_m[i, 0]),
                    "y_m_i_arousal": float(y_m[i, 1]),
                    "phi_l2": float(err_phi[k]),
                    "ridge_l2": float(err_ridge[k]),
                    "xfer_l2": float(err_xfer[k]),
                    "raw_l2": float(err_raw[k]),
                    "raw_gap_l2": float(raw_gap[i]),
                    "beat_phi_ridge": bool(err_ridge[k] < err_phi[k]),
                    "beat_phi_xfer": bool(err_xfer[k] < err_phi[k]),
                    "beat_phi_ridge_vs_xfer": bool(err_ridge[k] < err_xfer[k]),
                }
            )

    rec = pd.DataFrame(records)
    rec = rec.merge(tw[["image_id", "excess_twist"]], on="image_id", how="left")
    return rec, alphas_used, all_err, all_pred, all_y


def _pack_ridge_translation_result(
    rec: pd.DataFrame,
    alphas_used: list[float],
    all_err: dict[str, list[float]],
    all_pred: dict[str, list[np.ndarray]],
    all_y: list[np.ndarray],
    *,
    direction: str,
    n_folds: int,
    seed: int,
) -> dict:
    y_all = np.vstack(all_y)
    summary = []
    beat_phi: dict[str, float] = {}
    for method in all_err:
        err = np.asarray(all_err[method], dtype=float)
        pred = np.vstack(all_pred[method])
        summary.append(summarize_errors(method, err))
        summary[-1]["r2_valence"] = float(r2_score(y_all[:, 0], pred[:, 0]))
        summary[-1]["r2_arousal"] = float(r2_score(y_all[:, 1], pred[:, 1]))
        summary[-1]["r2_mean"] = float(
            (summary[-1]["r2_valence"] + summary[-1]["r2_arousal"]) / 2
        )
        beat_phi[method] = float(np.mean(err < np.asarray(all_err["phi_same"], dtype=float)))

    rec["beat_phi_clip_ridge_xfer"] = rec["beat_phi_ridge"]
    rec["beat_phi_clip_nn_xfer"] = rec["beat_phi_xfer"]

    stratifiers = {
        "raw_gap_l2": stratify_tercile(rec, "raw_gap_l2"),
        "phi_l2": stratify_tercile(rec, "phi_l2"),
        "excess_twist": stratify_tercile(rec, "excess_twist"),
    }

    high_raw = stratifiers["raw_gap_l2"]["terciles"]["high"]["clip_ridge_xfer"]
    high_raw_nn = stratifiers["raw_gap_l2"]["terciles"]["high"]["clip_nn_xfer"]
    high_raw_phi = stratifiers["raw_gap_l2"]["terciles"]["high"]["phi_same"]

    ridge_med = high_raw["median_l2"]
    phi_med = high_raw_phi["median_l2"]
    nn_med = high_raw_nn["median_l2"]
    if ridge_med < phi_med:
        verdict = "surpasses_phi_in_high_raw_gap_tercile"
    elif ridge_med < nn_med - 0.03:
        verdict = "competes_improved_vs_1nn_not_phi"
    else:
        verdict = "same_as_1nn_competes_not_surpasses"

    rho_raw_phi, _ = stats.spearmanr(rec["raw_gap_l2"], rec["phi_l2"])

    direction_label = "female_target_MtoF" if direction == "MtoF" else "male_target_FtoM"

    return {
        "task": "predictive_translation_ridge",
        "direction": direction_label,
        "direction_code": direction,
        "cv": {"n_folds": n_folds, "seed": seed, "pool": "train_only"},
        "n_queries": int(len(rec)),
        "ridge_alpha_per_fold": alphas_used,
        "ridge_alpha_median": float(np.median(alphas_used)),
        "methods": summary,
        "beat_phi_rate": beat_phi,
        "decision_rules": DECISION_RULES,
        "verdict": verdict,
        "high_raw_gap_tercile": {
            "ridge_median_l2": ridge_med,
            "clip_nn_median_l2": nn_med,
            "phi_median_l2": phi_med,
            "ridge_beat_phi_rate": high_raw["beat_phi_rate"],
            "clip_nn_beat_phi_rate": high_raw_nn["beat_phi_rate"],
            "ridge_wilcoxon_vs_phi_p": high_raw["wilcoxon_two_sided_p"],
        },
        "stratifiers": stratifiers,
        "stratifier_correlations": {
            "spearman_raw_gap_vs_fold_phi_l2": round(float(rho_raw_phi), 4),
        },
    }


def run_ridge_translation(
    *,
    direction: str = "MtoF",
    n_folds: int = 5,
    seed: int = RANDOM_SEED,
) -> dict:
    rec, alphas_used, all_err, all_pred, all_y = collect_predictive_translation_records(
        direction=direction, n_folds=n_folds, seed=seed,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_name = (
        "predictive_translation_per_image.csv"
        if direction == "MtoF"
        else "predictive_translation_per_image_FtoM.csv"
    )
    rec.to_csv(OUT_DIR / csv_name, index=False)
    return _pack_ridge_translation_result(
        rec, alphas_used, all_err, all_pred, all_y,
        direction=direction, n_folds=n_folds, seed=seed,
    )


def run_ridge_translation_both(*, n_folds: int = 5, seed: int = RANDOM_SEED) -> dict[str, dict]:
    return {
        "MtoF": run_ridge_translation(direction="MtoF", n_folds=n_folds, seed=seed),
        "FtoM": run_ridge_translation(direction="FtoM", n_folds=n_folds, seed=seed),
    }


def main() -> None:
    results = run_ridge_translation_both()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for direction, out in results.items():
        json_name = (
            "predictive_translation_ridge.json"
            if direction == "MtoF"
            else "predictive_translation_ridge_FtoM.json"
        )
        path = OUT_DIR / json_name
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(out, indent=2))
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
