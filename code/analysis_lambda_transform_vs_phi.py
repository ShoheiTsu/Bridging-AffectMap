#!/usr/bin/env python3
"""
Compare readout-coefficient (λ) transforms vs output-coordinate Φ on theme-held-out CLIP Ridge.

Methods (M→F primary; F→M optional):
  none        — predict female VA with male Ridge readout λ_m
  phi         — 2×2 affine on male Ridge predictions (same as Fig. 2A linear_shift on readout preds)
  lambda_diag — per-dimension scale of λ_m coefficients (512 params / output)
  lambda_rank_r — rank-r residual correction: y ≈ λ_m readout + (X @ U_r) w
  oracle      — female readout λ_f

Evaluation: nested LOTO by theme_base (same protocol as Δλ maps).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from analysis_population_bridge_suite import fit_affine  # noqa: E402
from config import (  # noqa: E402
    OASIS_SCORES_CSV,
    RANDOM_SEED,
    RESULTS_GENDER,
    RESULTS_STEP1,
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

# Pooled R² over all LOTO test images: singleton themes (n_test=1) are valid and
# should be included. (Global MIN_TEST_LOTO=2 remains for CVAE LOTO training.)
MIN_TEST_LOTO = 1

OUT_DIR = RESULTS_GENDER / "lambda_transform_vs_phi"
VA_SCALE = VALENCE_AROUSAL_SCALE_MAX - VALENCE_AROUSAL_SCALE_MIN


def _denorm(y_norm: np.ndarray) -> np.ndarray:
    return y_norm * VA_SCALE + VALENCE_AROUSAL_SCALE_MIN


def _r2_pair(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    rv = float(r2_score(y[:, 0], p[:, 0]))
    ra = float(r2_score(y[:, 1], p[:, 1]))
    return rv, ra, float((rv + ra) / 2)


def _l2(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.linalg.norm(y - p, axis=1)


def _load_alpha_map(gender: str) -> dict[str, float]:
    path = RESULTS_GENDER / "loto_nested" / gender / "summary.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: float(v["best_alpha_clip"]) for k, v in data["by_fold"].items()}


def _fit_ridge_readout(
    X_tr_n: np.ndarray, y_tr: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    y_n = normalize_y(y_tr, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX)
    mv = Ridge(alpha=alpha, random_state=RANDOM_SEED).fit(X_tr_n, y_n[:, 0])
    ma = Ridge(alpha=alpha, random_state=RANDOM_SEED).fit(X_tr_n, y_n[:, 1])
    return mv.coef_, ma.coef_, float(mv.intercept_), float(ma.intercept_)


def _predict_readout(
    X_n: np.ndarray,
    coef_v: np.ndarray,
    coef_a: np.ndarray,
    ic_v: float,
    ic_a: float,
) -> np.ndarray:
    pv = _denorm(X_n @ coef_v + ic_v)
    pa = _denorm(X_n @ coef_a + ic_a)
    return np.column_stack([pv, pa])


def _select_blend_weight(
    X_va_n: np.ndarray,
    coef_m_v: np.ndarray,
    coef_m_a: np.ndarray,
    coef_f_v: np.ndarray,
    coef_f_a: np.ndarray,
    ic_m_v: float,
    ic_m_a: float,
    ic_f_v: float,
    ic_f_a: float,
    y_va: np.ndarray,
) -> float:
    best_w, best_score = 0.0, -np.inf
    for w in np.linspace(0.0, 1.0, 21):
        cv = (1.0 - w) * coef_m_v + w * coef_f_v
        ca = (1.0 - w) * coef_m_a + w * coef_f_a
        icv = (1.0 - w) * ic_m_v + w * ic_f_v
        ica = (1.0 - w) * ic_m_a + w * ic_f_a
        pred = _predict_readout(X_va_n, cv, ca, icv, ica)
        score = (r2_score(y_va[:, 0], pred[:, 0]) + r2_score(y_va[:, 1], pred[:, 1])) / 2
        if score > best_score:
            best_score = score
            best_w = float(w)
    return best_w


def _apply_blend_weight(
    w: float,
    X_n: np.ndarray,
    coef_m_v: np.ndarray,
    coef_m_a: np.ndarray,
    coef_f_v: np.ndarray,
    coef_f_a: np.ndarray,
    ic_m_v: float,
    ic_m_a: float,
    ic_f_v: float,
    ic_f_a: float,
) -> np.ndarray:
    cv = (1.0 - w) * coef_m_v + w * coef_f_v
    ca = (1.0 - w) * coef_m_a + w * coef_f_a
    icv = (1.0 - w) * ic_m_v + w * ic_f_v
    ica = (1.0 - w) * ic_m_a + w * ic_f_a
    return _predict_readout(X_n, cv, ca, icv, ica)


def _fit_diagonal_scale_coef(
    coef_m: np.ndarray,
    coef_f: np.ndarray,
    *,
    alpha: float | None = None,
) -> np.ndarray:
    """Shrinkage element-wise ratio in coefficient space: s ≈ coef_f / coef_m."""
    if alpha is None:
        alpha = float(np.mean(coef_m * coef_m) * 0.1 + 1e-6)
    denom = coef_m * coef_m + alpha
    return (coef_m * coef_f + alpha) / denom


def _fit_diagonal_scale(
    X_n: np.ndarray,
    coef: np.ndarray,
    ic: float,
    y: np.ndarray,
) -> tuple[np.ndarray, float]:
    """y ≈ b + X @ (s * coef); linear in (b, s). (High DOF — can overfit.)"""
    design = X_n * coef[None, :]
    n = len(y)
    A = np.column_stack([np.ones(n), design])
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return sol[1:], float(sol[0])


def _predict_diagonal(
    X_n: np.ndarray,
    coef: np.ndarray,
    s: np.ndarray,
    b: float,
) -> np.ndarray:
    return b + X_n @ (s * coef)


def _fit_rank_correction(
    X_n: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    rank: int,
    *,
    ridge_alpha: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """y ≈ base + (X @ U) @ w with U = top-r PCA axes; ridge on w."""
    if rank <= 0:
        return np.zeros((X_n.shape[1], 0)), np.zeros(0)
    r = min(rank, X_n.shape[0] - 1, X_n.shape[1])
    pca = PCA(n_components=r, random_state=RANDOM_SEED)
    pca.fit(X_n)
    U = pca.components_.T  # 512 x r
    F = X_n @ U
    resid = y - base
    FtF = F.T @ F + ridge_alpha * np.eye(r)
    w = np.linalg.solve(FtF, F.T @ resid)
    return U, w


def _predict_rank(
    X_n: np.ndarray,
    base: np.ndarray,
    U: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    if U.size == 0:
        return base.copy()
    return base + (X_n @ U) @ w


def loto_indices(df: pd.DataFrame) -> list[int]:
    idx: list[int] = []
    for left_out in df["theme_base"].unique():
        m = (df["theme_base"] == left_out).values
        if int(m.sum()) >= MIN_TEST_LOTO:
            idx.extend(np.where(m)[0].tolist())
    return idx


def run_loto_analysis(
    *,
    direction: str = "MtoF",
    ranks: list[int],
    n_trials: int = 30,
    use_cached_alpha: bool = True,
) -> tuple[pd.DataFrame, dict]:
    if direction not in {"MtoF", "FtoM"}:
        raise ValueError(direction)

    src_targets = TARGET_MALE if direction == "MtoF" else TARGET_FEMALE
    tgt_targets = TARGET_FEMALE if direction == "MtoF" else TARGET_MALE
    src_gender = "male" if direction == "MtoF" else "female"
    tgt_gender = "female" if direction == "MtoF" else "male"

    oasis = load_oasis_meta(OASIS_SCORES_CSV)
    cols = [
        "valence",
        "arousal",
        *TARGET_MALE,
        *TARGET_FEMALE,
    ]
    full = oasis[oasis[cols].notna().all(axis=1)].reset_index(drop=True)
    full = add_theme_base(full)

    X_clip = np.load(RESULTS_STEP1 / "features_clip.npy").astype(float)
    valid = oasis[cols].notna().all(axis=1).values
    X_clip = X_clip[valid]

    y_src = full[src_targets].to_numpy(float)
    y_tgt = full[tgt_targets].to_numpy(float)

    alpha_src = _load_alpha_map(src_gender) if use_cached_alpha else {}
    alpha_tgt = _load_alpha_map(tgt_gender) if use_cached_alpha else {}

    rank_keys = [f"lambda_rank_{r}" for r in ranks]
    method_preds: dict[str, list[np.ndarray]] = {
        "none": [],
        "phi": [],
        "lambda_diag_coef": [],
        "lambda_diag_pred": [],
        "lambda_blend": [],
        "oracle": [],
        **{k: [] for k in rank_keys},
    }

    records: list[dict] = []
    y_chunks: list[np.ndarray] = []
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
        y_tr_src = y_src[train_mask]
        y_tr_tgt = y_tgt[train_mask]
        y_te_tgt = y_tgt[test_mask]

        X_tr_i = X_clip[train_idx_global]
        X_val = X_clip[val_idx_global]
        y_tr_i_src = y_src[train_idx_global]
        y_val_src = y_src[val_idx_global]
        y_tr_i_tgt = y_tgt[train_idx_global]
        y_val_tgt = y_tgt[val_idx_global]

        scl = StandardScaler().fit(X_tr)
        X_tr_n = scl.transform(X_tr)
        X_te_n = scl.transform(X_te)
        X_tr_i_n = scl.transform(X_tr_i)
        X_val_n = scl.transform(X_val)

        if use_cached_alpha and left_out in alpha_src and left_out in alpha_tgt:
            a_src = alpha_src[left_out]
            a_tgt = alpha_tgt[left_out]
        else:
            a_src, _ = optimize_alpha_nested(
                X_tr_i_n, X_val_n, y_tr_i_src, y_val_src, True, n_trials
            )
            a_tgt, _ = optimize_alpha_nested(
                X_tr_i_n, X_val_n, y_tr_i_tgt, y_val_tgt, True, n_trials
            )

        cv, ca, icv, ica = _fit_ridge_readout(X_tr_n, y_tr_src, a_src)
        fv, fa, ifv, ifa = _fit_ridge_readout(X_tr_n, y_tr_tgt, a_tgt)

        pred_src_tr = _predict_readout(X_tr_n, cv, ca, icv, ica)
        pred_src_te = _predict_readout(X_te_n, cv, ca, icv, ica)
        pred_oracle_te = _predict_readout(X_te_n, fv, fa, ifv, ifa)

        phi = fit_affine(pred_src_tr, y_tr_tgt)
        pred_phi_te = phi.apply(pred_src_te)

        sv_coef = _fit_diagonal_scale_coef(cv, fv)
        sa_coef = _fit_diagonal_scale_coef(ca, fa)
        pred_diag_coef_te = np.column_stack(
            [
                _predict_diagonal(X_te_n, cv, sv_coef, ifv),
                _predict_diagonal(X_te_n, ca, sa_coef, ifa),
            ]
        )

        sv_pred, bv = _fit_diagonal_scale(X_tr_n, cv, icv, y_tr_tgt[:, 0])
        sa_pred, ba = _fit_diagonal_scale(X_tr_n, ca, ica, y_tr_tgt[:, 1])
        pred_diag_pred_te = np.column_stack(
            [
                _predict_diagonal(X_te_n, cv, sv_pred, bv),
                _predict_diagonal(X_te_n, ca, sa_pred, ba),
            ]
        )

        X_val_n = scl.transform(X_val)
        y_val_tgt = y_tgt[val_idx_global]
        blend_w = _select_blend_weight(
            X_val_n, cv, ca, fv, fa, icv, ica, ifv, ifa, y_val_tgt
        )
        pred_blend_te = _apply_blend_weight(
            blend_w, X_te_n, cv, ca, fv, fa, icv, ica, ifv, ifa
        )

        rank_preds_te: dict[int, np.ndarray] = {}
        base_tr_v = _predict_diagonal(X_tr_n, cv, np.ones_like(cv), icv)
        base_tr_a = _predict_diagonal(X_tr_n, ca, np.ones_like(ca), ica)
        base_te_v = pred_src_te[:, 0]
        base_te_a = pred_src_te[:, 1]
        for r in ranks:
            Uv, wv = _fit_rank_correction(X_tr_n, base_tr_v, y_tr_tgt[:, 0], r)
            Ua, wa = _fit_rank_correction(X_tr_n, base_tr_a, y_tr_tgt[:, 1], r)
            rank_preds_te[r] = np.column_stack(
                [
                    _predict_rank(X_te_n, base_te_v, Uv, wv),
                    _predict_rank(X_te_n, base_te_a, Ua, wa),
                ]
            )

        sub = full[test_mask].reset_index(drop=True)
        for local_i in range(n_test):
            y_true = y_te_tgt[local_i]
            err = {
                "none": float(_l2(y_true[None, :], pred_src_te[local_i : local_i + 1])[0]),
                "phi": float(_l2(y_true[None, :], pred_phi_te[local_i : local_i + 1])[0]),
                "lambda_diag_coef": float(
                    _l2(y_true[None, :], pred_diag_coef_te[local_i : local_i + 1])[0]
                ),
                "lambda_diag_pred": float(
                    _l2(y_true[None, :], pred_diag_pred_te[local_i : local_i + 1])[0]
                ),
                "lambda_blend": float(
                    _l2(y_true[None, :], pred_blend_te[local_i : local_i + 1])[0]
                ),
                "oracle": float(
                    _l2(y_true[None, :], pred_oracle_te[local_i : local_i + 1])[0]
                ),
            }
            for r in ranks:
                key = f"lambda_rank_{r}"
                err[key] = float(
                    _l2(y_true[None, :], rank_preds_te[r][local_i : local_i + 1])[0]
                )

            rec = {
                "direction": direction,
                "theme_base": left_out,
                "image_id": str(sub.loc[local_i, "image_id"]),
                "valence_common": float(sub.loc[local_i, "valence"]),
                "arousal_common": float(sub.loc[local_i, "arousal"]),
                **{f"{k}_l2": v for k, v in err.items()},
                **{
                    f"beat_phi_{k}": int(err[k] < err["phi"])
                    for k in ["none", "lambda_diag_coef", "lambda_diag_pred", "lambda_blend", *rank_keys, "oracle"]
                },
            }
            records.append(rec)

        y_chunks.append(y_te_tgt)
        method_preds["none"].append(pred_src_te)
        method_preds["phi"].append(pred_phi_te)
        method_preds["lambda_diag_coef"].append(pred_diag_coef_te)
        method_preds["lambda_diag_pred"].append(pred_diag_pred_te)
        method_preds["lambda_blend"].append(pred_blend_te)
        method_preds["oracle"].append(pred_oracle_te)
        for r in ranks:
            method_preds[f"lambda_rank_{r}"].append(rank_preds_te[r])

        if fold_idx % 40 == 0:
            print(f"  fold {fold_idx + 1}/{len(bases)}: {left_out} (n_test={n_test})")

    df_rec = pd.DataFrame(records)
    y_all = np.vstack(y_chunks)

    summary: dict = {
        "direction": direction,
        "protocol": "loto_theme_held_out_clip_ridge",
        "n_images": int(len(df_rec)),
        "ranks": ranks,
        "r2": {},
        "beat_phi_rate": {},
        "median_l2": {},
    }

    for name, chunks in method_preds.items():
        pred_all = np.vstack(chunks)
        rv, ra, rm = _r2_pair(y_all, pred_all)
        summary["r2"][name] = {
            "R2_valence": rv,
            "R2_arousal": ra,
            "R2_mean": rm,
        }
        col = f"{name}_l2"
        if col in df_rec.columns:
            summary["median_l2"][name] = float(df_rec[col].median())
            if name != "phi" and f"beat_phi_{name}" in df_rec.columns:
                summary["beat_phi_rate"][name] = float(df_rec[f"beat_phi_{name}"].mean())

    # Spatial correlations (M→F primary stratifiers)
    if len(df_rec) > 10:
        lam_path = RESULTS_GENDER / "lambda_gender_diff_per_image_loto.csv"
        if lam_path.exists():
            lam = pd.read_csv(lam_path)[["image_id", "delta_lambda_v", "delta_lambda_a"]]
            lam["abs_delta_lambda"] = np.hypot(lam["delta_lambda_v"], lam["delta_lambda_a"])
            merged = df_rec.merge(lam, on="image_id", how="left")
            merged["phi_residual_l2"] = merged["phi_l2"] - merged["oracle_l2"]
            for method in ["lambda_diag_coef", "lambda_diag_pred", "lambda_blend", *[f"lambda_rank_{r}" for r in ranks]]:
                beat_col = f"beat_phi_{method}"
                if beat_col not in merged.columns:
                    continue
                block = merged.dropna(subset=["abs_delta_lambda", "phi_residual_l2"])
                if len(block) < 20:
                    continue
                summary.setdefault("spatial_correlations", {})[method] = {
                    "spearman_beat_vs_abs_delta_lambda": float(
                        stats.spearmanr(block[beat_col], block["abs_delta_lambda"]).statistic
                    ),
                    "spearman_beat_vs_phi_residual": float(
                        stats.spearmanr(block[beat_col], block["phi_residual_l2"]).statistic
                    ),
                    "spearman_delta_l2_vs_abs_delta_lambda": float(
                        stats.spearmanr(
                            block[f"{method}_l2"] - block["phi_l2"],
                            block["abs_delta_lambda"],
                        ).statistic
                    ),
                }

    return df_rec, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="λ transform vs Φ (theme-held-out CLIP Ridge).")
    ap.add_argument("--direction", choices=["MtoF", "FtoM", "both"], default="both")
    ap.add_argument("--ranks", type=str, default="0,1,2,4,8,16,32")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--recompute-alpha", action="store_true")
    args = ap.parse_args()

    ranks = [int(x.strip()) for x in args.ranks.split(",") if x.strip()]
    directions = ["MtoF", "FtoM"] if args.direction == "both" else [args.direction]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summary: dict = {"ranks": ranks, "directions": {}}

    for direction in directions:
        print(f"\n=== {direction} ===")
        df_rec, summary = run_loto_analysis(
            direction=direction,
            ranks=ranks,
            n_trials=args.n_trials,
            use_cached_alpha=not args.recompute_alpha,
        )
        tag = direction.lower()
        csv_path = OUT_DIR / f"per_image_{tag}.csv"
        json_path = OUT_DIR / f"summary_{tag}.json"
        df_rec.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        all_summary["directions"][direction] = summary
        print(json.dumps(summary, indent=2))
        print(f"Saved {csv_path}\nSaved {json_path}")

    combined_path = OUT_DIR / "summary.json"
    combined_path.write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    print(f"Saved {combined_path}")


if __name__ == "__main__":
    main()
