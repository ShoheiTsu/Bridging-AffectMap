#!/usr/bin/env python3
"""
男女別ターゲットで Nested LOTO と Nested LOCO を実行する。
- 男性: valence_male, arousal_male / 女性: valence_female, arousal_female
- step1_loto_nested, step1_loco_nested と同一プロトコル（内側で alpha 最適化、テーマ/カテゴリ単位で leave-out）
- 結果: results_gender/loto_nested/{male,female}/, results_gender/loco_nested/{male,female}/
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler

from config import (
    OASIS_SCORES_CSV,
    RESULTS_STEP1,
    RESULTS_GENDER,
    RANDOM_SEED,
    VALENCE_AROUSAL_SCALE_MIN,
    VALENCE_AROUSAL_SCALE_MAX,
    ALPHA_SEARCH_LOW,
    ALPHA_SEARCH_HIGH,
    ALPHA_SEARCH_LOW_VIT_LOTO,
    LOTO_VIT_CONSERVATIVE_ALPHA_RATIO,
    ALPHA_SEARCH_HIGH_LOCO,
    CATEGORY_COLUMN,
)
from dataset import load_oasis_meta, add_theme_base, train_val_split_by_theme

TARGET_MALE = ["valence_male", "arousal_male"]
TARGET_FEMALE = ["valence_female", "arousal_female"]
MIN_TEST_LOTO = 2  # default; override via --min-test-loto (pooled R² / SI cross−within may use 1)
MIN_TEST_LOCO = 5


def normalize_y(y, scale_min=1.0, scale_max=7.0):
    return (y - scale_min) / (scale_max - scale_min)


def denormalize_y(y_norm, scale_min=1.0, scale_max=7.0):
    return y_norm * (scale_max - scale_min) + scale_min


def optimize_alpha_nested(X_train, X_val, y_train, y_val, normalize_target, n_trials=30,
                          alpha_low=None, alpha_high=None, conservative_ratio=None):
    low = alpha_low if alpha_low is not None else ALPHA_SEARCH_LOW
    high = alpha_high if alpha_high is not None else ALPHA_SEARCH_HIGH
    scl = StandardScaler().fit(X_train)
    X_tr_n = scl.transform(X_train)
    X_val_n = scl.transform(X_val)
    if normalize_target:
        y_tr_n = normalize_y(y_train, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX)
        y_val_n = normalize_y(y_val, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX)
    else:
        y_tr_n, y_val_n = y_train, y_val

    def objective(trial):
        alpha = trial.suggest_float("alpha", low, high, log=True)
        r2_sum = 0.0
        for j in range(2):
            m = Ridge(alpha=alpha, random_state=RANDOM_SEED).fit(X_tr_n, y_tr_n[:, j])
            p = m.predict(X_val_n)
            if normalize_target:
                p = denormalize_y(p, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX)
            r2_sum += r2_score(y_val[:, j], p)
        return r2_sum / 2

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    if conservative_ratio is not None and study.best_trial is not None:
        best_val = study.best_trial.value
        threshold = conservative_ratio * best_val
        candidates = [t for t in study.trials if t.value is not None and t.value >= threshold]
        if candidates:
            chosen = max(candidates, key=lambda t: t.params["alpha"])
            return chosen.params["alpha"], scl
    return study.best_params["alpha"], scl


def optimize_alpha_fusion(X_clip_tr, X_clip_val, X_vit_tr, X_vit_val, y_tr, y_val, normalize_target, n_trials=30, alpha_high=None):
    high = alpha_high if alpha_high is not None else ALPHA_SEARCH_HIGH
    X_tr = np.hstack([X_clip_tr, X_vit_tr])
    X_val = np.hstack([X_clip_val, X_vit_val])
    return optimize_alpha_nested(X_tr, X_val, y_tr, y_val, normalize_target, n_trials=n_trials, alpha_high=high)


def _infer_word_column(df_words: pd.DataFrame) -> str:
    candidates = [
        "word", "token", "label", "top_word", "clip_word", "pred_word", "keyword", "class_name"
    ]
    cols = set(df_words.columns)
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"Word column not found. Expected one of {candidates}, got {list(df_words.columns)}")


def _build_background_mask_from_words(df_ref: pd.DataFrame, words_csv: Path) -> np.ndarray:
    """
    Build boolean mask aligned to df_ref rows.
    True if CLIP word extraction indicates background for that image_id.
    """
    df_words = pd.read_csv(words_csv)
    if "image_id" not in df_words.columns:
        raise ValueError(f"{words_csv} must contain 'image_id' column.")

    ids_bg: set[str] = set()
    lower_cols = {c.lower(): c for c in df_words.columns}
    if "is_background" in lower_cols:
        c = lower_cols["is_background"]
        sub = df_words[df_words[c].astype(str).str.lower().isin(["1", "true", "yes"])]
        ids_bg = set(sub["image_id"].astype(str).tolist())
    elif "background_score" in lower_cols:
        c = lower_cols["background_score"]
        sub = df_words[pd.to_numeric(df_words[c], errors="coerce").fillna(0.0) > 0.0]
        ids_bg = set(sub["image_id"].astype(str).tolist())
    else:
        wc = _infer_word_column(df_words)
        sub = df_words[df_words[wc].astype(str).str.lower().str.strip() == "background"]
        ids_bg = set(sub["image_id"].astype(str).tolist())

    mask = df_ref["image_id"].astype(str).isin(ids_bg).values
    return mask.astype(bool)


def _apply_background_vit_gate(X_vit: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    """
    Keep ViT branch active only for images flagged as background.
    Non-background rows become zeros in ViT branch.
    """
    return X_vit * bg_mask[:, None].astype(X_vit.dtype)


def run_nested_loto(
    gender,
    y,
    df,
    X_clip,
    X_vit,
    X_vit_fusion: np.ndarray | None = None,
    normalize_target=True,
    n_trials=30,
    out_root: Path | None = None,
):
    """X_vit: ViT-only ridge. X_vit_fusion: ViT branch in fusion hstack (defaults to X_vit)."""
    X_vf = X_vit if X_vit_fusion is None else X_vit_fusion
    bases = df["theme_base"].unique().tolist()
    root = out_root if out_root is not None else RESULTS_GENDER
    out_dir = root / "loto_nested" / gender
    out_dir.mkdir(parents=True, exist_ok=True)

    y_all_true = []
    y_all_pred_clip = []
    y_all_pred_vit = []
    y_all_pred_fusion = []
    fold_results = {}
    rows = []

    for fold_idx, left_out in enumerate(bases):
        train_mask = (df["theme_base"] != left_out).values
        test_mask = (df["theme_base"] == left_out).values
        n_train, n_test = train_mask.sum(), test_mask.sum()
        if n_test < MIN_TEST_LOTO:
            continue
        if fold_idx % 30 == 0:
            print(f"  LOTO {gender} fold {fold_idx+1}/{len(bases)}: {left_out}")

        df_train = df[train_mask].reset_index(drop=True)
        _, _, train_inner_idx, val_inner_idx = train_val_split_by_theme(
            df_train, train_ratio=0.8, random_state=RANDOM_SEED, return_indices=True
        )
        train_idx_global = np.where(train_mask)[0][train_inner_idx]
        val_idx_global = np.where(train_mask)[0][val_inner_idx]

        X_tr_c = X_clip[train_mask]; X_te_c = X_clip[test_mask]
        X_tr_v = X_vit[train_mask]; X_te_v = X_vit[test_mask]
        X_tr_vf = X_vf[train_mask]; X_te_vf = X_vf[test_mask]
        y_tr = y[train_mask]; y_te = y[test_mask]
        X_tr_c_i = X_clip[train_idx_global]; X_val_c = X_clip[val_idx_global]
        X_tr_v_i = X_vit[train_idx_global]; X_val_v = X_vit[val_idx_global]
        X_tr_vf_i = X_vf[train_idx_global]; X_val_vf = X_vf[val_idx_global]
        y_tr_i = y[train_idx_global]; y_val_i = y[val_idx_global]

        best_a_c, scl_c = optimize_alpha_nested(X_tr_c_i, X_val_c, y_tr_i, y_val_i, normalize_target, n_trials)
        best_a_v, scl_v = optimize_alpha_nested(
            X_tr_v_i, X_val_v, y_tr_i, y_val_i, normalize_target, n_trials,
            alpha_low=ALPHA_SEARCH_LOW_VIT_LOTO, conservative_ratio=LOTO_VIT_CONSERVATIVE_ALPHA_RATIO
        )
        scl_c.fit(X_tr_c); scl_v.fit(X_tr_v); scl_vf = StandardScaler().fit(X_tr_vf)
        X_tr_c_n = scl_c.transform(X_tr_c); X_te_c_n = scl_c.transform(X_te_c)
        X_tr_v_n = scl_v.transform(X_tr_v); X_te_v_n = scl_v.transform(X_te_v)
        X_tr_vf_n = scl_vf.transform(X_tr_vf); X_te_vf_n = scl_vf.transform(X_te_vf)
        X_tr_c_i_n = scl_c.transform(X_tr_c_i); X_val_c_n = scl_c.transform(X_val_c)
        X_tr_vf_i_n = scl_vf.transform(X_tr_vf_i); X_val_vf_n = scl_vf.transform(X_val_vf)
        best_a_f, scl_f = optimize_alpha_fusion(
            X_tr_c_i_n, X_val_c_n, X_tr_vf_i_n, X_val_vf_n, y_tr_i, y_val_i, normalize_target, n_trials
        )
        X_tr_f = np.hstack([X_tr_c_n, X_tr_vf_n]); X_te_f = np.hstack([X_te_c_n, X_te_vf_n])
        scl_f.fit(X_tr_f); X_tr_f_n = scl_f.transform(X_tr_f); X_te_f_n = scl_f.transform(X_te_f)

        if normalize_target:
            y_tr_n = normalize_y(y_tr, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX)
        else:
            y_tr_n = y_tr

        pred_c = np.zeros_like(y_te)
        pred_v = np.zeros_like(y_te)
        pred_f = np.zeros_like(y_te)
        for j in range(2):
            m = Ridge(alpha=best_a_c, random_state=RANDOM_SEED).fit(X_tr_c_n, y_tr_n[:, j])
            p = m.predict(X_te_c_n)
            pred_c[:, j] = denormalize_y(p, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX) if normalize_target else p
            m = Ridge(alpha=best_a_v, random_state=RANDOM_SEED).fit(X_tr_v_n, y_tr_n[:, j])
            p = m.predict(X_te_v_n)
            pred_v[:, j] = denormalize_y(p, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX) if normalize_target else p
            m = Ridge(alpha=best_a_f, random_state=RANDOM_SEED).fit(X_tr_f_n, y_tr_n[:, j])
            p = m.predict(X_te_f_n)
            pred_f[:, j] = denormalize_y(p, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX) if normalize_target else p

        r2_c_v = r2_score(y_te[:, 0], pred_c[:, 0]); r2_c_a = r2_score(y_te[:, 1], pred_c[:, 1]); r2_c_m = (r2_c_v + r2_c_a) / 2
        r2_v_v = r2_score(y_te[:, 0], pred_v[:, 0]); r2_v_a = r2_score(y_te[:, 1], pred_v[:, 1]); r2_v_m = (r2_v_v + r2_v_a) / 2
        r2_f_v = r2_score(y_te[:, 0], pred_f[:, 0]); r2_f_a = r2_score(y_te[:, 1], pred_f[:, 1]); r2_f_m = (r2_f_v + r2_f_a) / 2

        fold_results[left_out] = {"fold_idx": fold_idx, "n_train": int(n_train), "n_test": int(n_test),
            "best_alpha_clip": float(best_a_c), "best_alpha_vit": float(best_a_v), "best_alpha_fusion": float(best_a_f),
            "R2_clip_valence": float(r2_c_v), "R2_clip_arousal": float(r2_c_a), "R2_clip_mean": float(r2_c_m),
            "R2_vit_valence": float(r2_v_v), "R2_vit_arousal": float(r2_v_a), "R2_vit_mean": float(r2_v_m), "R2_fusion_mean": float(r2_f_m)}
        rows.append({"left_out_theme": left_out, "n_train": n_train, "n_test": n_test,
            "R2_clip_valence": round(r2_c_v, 4), "R2_clip_arousal": round(r2_c_a, 4), "R2_clip_mean": round(r2_c_m, 4),
            "R2_vit_valence": round(r2_v_v, 4), "R2_vit_arousal": round(r2_v_a, 4), "R2_vit_mean": round(r2_v_m, 4), "R2_fusion_mean": round(r2_f_m, 4)})
        y_all_true.append(y_te); y_all_pred_clip.append(pred_c); y_all_pred_vit.append(pred_v); y_all_pred_fusion.append(pred_f)

    if len(y_all_true) == 0:
        return None
    y_all_true = np.vstack(y_all_true)
    y_all_pred_clip = np.vstack(y_all_pred_clip)
    y_all_pred_vit = np.vstack(y_all_pred_vit)
    y_all_pred_fusion = np.vstack(y_all_pred_fusion)
    # 列順: 第0列=Valence, 第1列=Arousal（target_cols = TARGET_MALE / TARGET_FEMALE の順）
    np.save(out_dir / "y_all_true.npy", y_all_true)
    np.save(out_dir / "y_all_pred_clip.npy", y_all_pred_clip)
    np.save(out_dir / "y_all_pred_vit.npy", y_all_pred_vit)
    np.save(out_dir / "y_all_pred_fusion.npy", y_all_pred_fusion)
    r2_c_v = r2_score(y_all_true[:, 0], y_all_pred_clip[:, 0]); r2_c_a = r2_score(y_all_true[:, 1], y_all_pred_clip[:, 1])
    r2_v_v = r2_score(y_all_true[:, 0], y_all_pred_vit[:, 0]); r2_v_a = r2_score(y_all_true[:, 1], y_all_pred_vit[:, 1])
    r2_f_v = r2_score(y_all_true[:, 0], y_all_pred_fusion[:, 0]); r2_f_a = r2_score(y_all_true[:, 1], y_all_pred_fusion[:, 1])
    summary = {
        "normalize_target": normalize_target,
        "evaluation": "Nested_LOTO",
        "target_gender": gender,
        "n_themes_evaluated": len(fold_results),
        "overall": {
            "R2_clip_valence": float(r2_c_v), "R2_clip_arousal": float(r2_c_a), "R2_clip_mean": float((r2_c_v + r2_c_a) / 2),
            "R2_vit_valence": float(r2_v_v), "R2_vit_arousal": float(r2_v_a), "R2_vit_mean": float((r2_v_v + r2_v_a) / 2),
            "R2_fusion_valence": float(r2_f_v), "R2_fusion_arousal": float(r2_f_a), "R2_fusion_mean": float((r2_f_v + r2_f_a) / 2),
        },
        "by_fold": fold_results,
    }
    pd.DataFrame(rows).to_csv(out_dir / "loto_nested.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_nested_loco(
    gender,
    y,
    df,
    X_clip,
    X_vit,
    X_vit_fusion: np.ndarray | None = None,
    normalize_target=True,
    n_trials=50,
    out_root: Path | None = None,
):
    """X_vit: ViT-only ridge. X_vit_fusion: ViT branch in fusion hstack (defaults to X_vit)."""
    X_vf = X_vit if X_vit_fusion is None else X_vit_fusion
    categories = df[CATEGORY_COLUMN].unique().tolist()
    root = out_root if out_root is not None else RESULTS_GENDER
    out_dir = root / "loco_nested" / gender
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_high = ALPHA_SEARCH_HIGH_LOCO if ALPHA_SEARCH_HIGH_LOCO is not None else ALPHA_SEARCH_HIGH

    y_all_true = []
    y_all_pred_clip = []
    y_all_pred_vit = []
    y_all_pred_fusion = []
    fold_results = {}
    rows = []

    for fold_idx, left_out in enumerate(categories):
        train_mask = (df[CATEGORY_COLUMN] != left_out).values
        test_mask = (df[CATEGORY_COLUMN] == left_out).values
        n_train, n_test = train_mask.sum(), test_mask.sum()
        if n_test < MIN_TEST_LOCO:
            continue
        print(f"  LOCO {gender} fold {fold_idx+1}/{len(categories)}: {left_out}")

        df_train = df[train_mask].reset_index(drop=True)
        _, _, train_inner_idx, val_inner_idx = train_val_split_by_theme(
            df_train, train_ratio=0.8, random_state=RANDOM_SEED, return_indices=True
        )
        train_idx_global = np.where(train_mask)[0][train_inner_idx]
        val_idx_global = np.where(train_mask)[0][val_inner_idx]

        X_tr_c = X_clip[train_mask]; X_te_c = X_clip[test_mask]
        X_tr_v = X_vit[train_mask]; X_te_v = X_vit[test_mask]
        X_tr_vf = X_vf[train_mask]; X_te_vf = X_vf[test_mask]
        y_tr = y[train_mask]; y_te = y[test_mask]
        X_tr_c_i = X_clip[train_idx_global]; X_val_c = X_clip[val_idx_global]
        X_tr_v_i = X_vit[train_idx_global]; X_val_v = X_vit[val_idx_global]
        X_tr_vf_i = X_vf[train_idx_global]; X_val_vf = X_vf[val_idx_global]
        y_tr_i = y[train_idx_global]; y_val_i = y[val_idx_global]

        best_a_c, scl_c = optimize_alpha_nested(X_tr_c_i, X_val_c, y_tr_i, y_val_i, normalize_target, n_trials, alpha_high=alpha_high)
        best_a_v, scl_v = optimize_alpha_nested(X_tr_v_i, X_val_v, y_tr_i, y_val_i, normalize_target, n_trials, alpha_high=alpha_high)
        scl_c.fit(X_tr_c); scl_v.fit(X_tr_v); scl_vf = StandardScaler().fit(X_tr_vf)
        X_tr_c_n = scl_c.transform(X_tr_c); X_te_c_n = scl_c.transform(X_te_c)
        X_tr_v_n = scl_v.transform(X_tr_v); X_te_v_n = scl_v.transform(X_te_v)
        X_tr_vf_n = scl_vf.transform(X_tr_vf); X_te_vf_n = scl_vf.transform(X_te_vf)
        X_tr_c_i_n = scl_c.transform(X_tr_c_i); X_val_c_n = scl_c.transform(X_val_c)
        X_tr_vf_i_n = scl_vf.transform(X_tr_vf_i); X_val_vf_n = scl_vf.transform(X_val_vf)
        best_a_f, scl_f = optimize_alpha_fusion(
            X_tr_c_i_n, X_val_c_n, X_tr_vf_i_n, X_val_vf_n, y_tr_i, y_val_i, normalize_target, n_trials, alpha_high=alpha_high
        )
        X_tr_f = np.hstack([X_tr_c_n, X_tr_vf_n]); X_te_f = np.hstack([X_te_c_n, X_te_vf_n])
        scl_f.fit(X_tr_f); X_tr_f_n = scl_f.transform(X_tr_f); X_te_f_n = scl_f.transform(X_te_f)

        if normalize_target:
            y_tr_n = normalize_y(y_tr, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX)
        else:
            y_tr_n = y_tr

        pred_c = np.zeros_like(y_te)
        pred_v = np.zeros_like(y_te)
        pred_f = np.zeros_like(y_te)
        for j in range(2):
            m = Ridge(alpha=best_a_c, random_state=RANDOM_SEED).fit(X_tr_c_n, y_tr_n[:, j])
            p = m.predict(X_te_c_n)
            pred_c[:, j] = denormalize_y(p, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX) if normalize_target else p
            m = Ridge(alpha=best_a_v, random_state=RANDOM_SEED).fit(X_tr_v_n, y_tr_n[:, j])
            p = m.predict(X_te_v_n)
            pred_v[:, j] = denormalize_y(p, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX) if normalize_target else p
            m = Ridge(alpha=best_a_f, random_state=RANDOM_SEED).fit(X_tr_f_n, y_tr_n[:, j])
            p = m.predict(X_te_f_n)
            pred_f[:, j] = denormalize_y(p, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX) if normalize_target else p

        r2_c_v = r2_score(y_te[:, 0], pred_c[:, 0]); r2_c_a = r2_score(y_te[:, 1], pred_c[:, 1]); r2_c_m = (r2_c_v + r2_c_a) / 2
        r2_v_v = r2_score(y_te[:, 0], pred_v[:, 0]); r2_v_a = r2_score(y_te[:, 1], pred_v[:, 1]); r2_v_m = (r2_v_v + r2_v_a) / 2
        r2_f_m = (r2_score(y_te[:, 0], pred_f[:, 0]) + r2_score(y_te[:, 1], pred_f[:, 1])) / 2

        fold_results[left_out] = {"fold_idx": fold_idx, "n_train": int(n_train), "n_test": int(n_test),
            "best_alpha_clip": float(best_a_c), "best_alpha_vit": float(best_a_v), "best_alpha_fusion": float(best_a_f),
            "R2_clip_valence": float(r2_c_v), "R2_clip_arousal": float(r2_c_a), "R2_clip_mean": float(r2_c_m),
            "R2_vit_valence": float(r2_v_v), "R2_vit_arousal": float(r2_v_a), "R2_vit_mean": float(r2_v_m), "R2_fusion_mean": float(r2_f_m)}
        rows.append({"left_out_category": left_out, "n_train": n_train, "n_test": n_test,
            "R2_clip_valence": round(r2_c_v, 4), "R2_clip_arousal": round(r2_c_a, 4), "R2_clip_mean": round(r2_c_m, 4),
            "R2_vit_valence": round(r2_v_v, 4), "R2_vit_arousal": round(r2_v_a, 4), "R2_vit_mean": round(r2_v_m, 4), "R2_fusion_mean": round(r2_f_m, 4)})
        y_all_true.append(y_te); y_all_pred_clip.append(pred_c); y_all_pred_vit.append(pred_v); y_all_pred_fusion.append(pred_f)
        # Fig7 カテゴリ別 MSE 用: fold（=left_out カテゴリ）ごとに予測を保存
        fold_dir = out_dir / f"fold_{left_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.save(fold_dir / "y_true.npy", y_te)
        np.save(fold_dir / "y_pred_clip.npy", pred_c)
        np.save(fold_dir / "y_pred_vit.npy", pred_v)
        np.save(fold_dir / "y_pred_fusion.npy", pred_f)

    if len(y_all_true) == 0:
        return None
    y_all_true = np.vstack(y_all_true)
    y_all_pred_clip = np.vstack(y_all_pred_clip)
    y_all_pred_vit = np.vstack(y_all_pred_vit)
    y_all_pred_fusion = np.vstack(y_all_pred_fusion)
    # 列順: 第0列=Valence, 第1列=Arousal（target_cols の順）
    np.save(out_dir / "y_all_true.npy", y_all_true)
    np.save(out_dir / "y_all_pred_clip.npy", y_all_pred_clip)
    np.save(out_dir / "y_all_pred_vit.npy", y_all_pred_vit)
    np.save(out_dir / "y_all_pred_fusion.npy", y_all_pred_fusion)
    r2_c_v = r2_score(y_all_true[:, 0], y_all_pred_clip[:, 0]); r2_c_a = r2_score(y_all_true[:, 1], y_all_pred_clip[:, 1])
    r2_v_v = r2_score(y_all_true[:, 0], y_all_pred_vit[:, 0]); r2_v_a = r2_score(y_all_true[:, 1], y_all_pred_vit[:, 1])
    r2_f_v = r2_score(y_all_true[:, 0], y_all_pred_fusion[:, 0]); r2_f_a = r2_score(y_all_true[:, 1], y_all_pred_fusion[:, 1])
    summary = {
        "normalize_target": normalize_target,
        "evaluation": "Nested_LOCO",
        "target_gender": gender,
        "n_categories_evaluated": len(fold_results),
        "overall": {
            "R2_clip_valence": float(r2_c_v), "R2_clip_arousal": float(r2_c_a), "R2_clip_mean": float((r2_c_v + r2_c_a) / 2),
            "R2_vit_valence": float(r2_v_v), "R2_vit_arousal": float(r2_v_a), "R2_vit_mean": float((r2_v_v + r2_v_a) / 2),
            "R2_fusion_valence": float(r2_f_v), "R2_fusion_arousal": float(r2_f_a), "R2_fusion_mean": float((r2_f_v + r2_f_a) / 2),
        },
        "by_fold": fold_results,
    }
    pd.DataFrame(rows).to_csv(out_dir / "loco_nested.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(
    loco_only=False,
    *,
    background_words_csv: Path | None = None,
    output_root: Path | None = None,
    min_test_loto: int | None = None,
    loto_only: bool = False,
):
    global MIN_TEST_LOTO
    if min_test_loto is not None:
        MIN_TEST_LOTO = int(min_test_loto)
        print(f"MIN_TEST_LOTO overridden to {MIN_TEST_LOTO}")
    out_root = output_root if output_root is not None else RESULTS_GENDER
    out_root.mkdir(parents=True, exist_ok=True)
    X_clip = np.load(RESULTS_STEP1 / "features_clip.npy")
    X_vit = np.load(RESULTS_STEP1 / "features_vit.npy")
    df = load_oasis_meta(OASIS_SCORES_CSV)
    df = add_theme_base(df)
    cols = TARGET_MALE + TARGET_FEMALE
    if not all(c in df.columns for c in cols):
        print("Missing gender columns. Run prepare_oasis_data.py with OASIS_bygender.csv.")
        return
    valid = df[cols].notna().all(axis=1).values
    df = df.loc[valid].reset_index(drop=True)
    X_clip = X_clip[valid]
    X_vit = X_vit[valid]
    X_vit_fusion = None
    if background_words_csv is not None:
        bg_mask = _build_background_mask_from_words(df, background_words_csv)
        X_vit_fusion = _apply_background_vit_gate(X_vit, bg_mask)
        print(
            f"Fusion-only background→ViT gate from {background_words_csv} "
            f"(background rows: {int(bg_mask.sum())}/{len(bg_mask)}). "
            "ViT-only branch uses full features."
        )
    print(f"Using {len(df)} samples with valid male/female scores.\n")

    normalize_target = True
    n_trials_loto = 30
    n_trials_loco = 50

    for gender, target_cols in [("male", TARGET_MALE), ("female", TARGET_FEMALE)]:
        y = df[target_cols].values.astype(np.float32)
        print(f"=== {gender.upper()} target ===")
        if not loco_only:
            print("Nested LOTO...")
            summary_loto = run_nested_loto(
                gender,
                y,
                df,
                X_clip,
                X_vit,
                X_vit_fusion,
                normalize_target,
                n_trials_loto,
                out_root=out_root,
            )
            if summary_loto:
                o = summary_loto["overall"]
                print(f"  Overall R² CLIP={o['R2_clip_mean']:.4f}, ViT={o['R2_vit_mean']:.4f}, Fusion={o['R2_fusion_mean']:.4f}")
        if not loto_only:
            print("Nested LOCO...")
            summary_loco = run_nested_loco(
                gender,
                y,
                df,
                X_clip,
                X_vit,
                X_vit_fusion,
                normalize_target,
                n_trials_loco,
                out_root=out_root,
            )
            if summary_loco:
                o = summary_loco["overall"]
                print(f"  Overall R² CLIP={o['R2_clip_mean']:.4f}, ViT={o['R2_vit_mean']:.4f}, Fusion={o['R2_fusion_mean']:.4f}")
        print()

    print(f"Done. Results: {out_root}/loto_nested/{{male,female}}/, {out_root}/loco_nested/{{male,female}}/")
    print("LOCO の summary.json と loco_nested.csv にカテゴリ別 R2_clip_valence/arousal, R2_vit_valence/arousal を保存済み。")
    print("Run python code/export_fig_gender_doc.py and python code/export_gender_r2_tables.py to update fig_gender_doc.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nested LOTO/LOCO for gender targets.")
    parser.add_argument("--loco-only", action="store_true", help="Run only Nested LOCO (skip LOTO). Use to refresh LOCO with valence/arousal per category.")
    parser.add_argument("--loto-only", action="store_true", help="Run only Nested LOTO (skip LOCO).")
    parser.add_argument(
        "--min-test-loto",
        type=int,
        default=None,
        help="Override MIN_TEST_LOTO (default 2). Use 1 to include singleton themes (n=900 pooled).",
    )
    parser.add_argument(
        "--background-words-csv",
        type=Path,
        default=None,
        help="CLIP word extraction CSV for method-2 background判定 (requires image_id + word/token/label or is_background/background_score).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root (default: results/). Use results/fusion_background_vit for gated-ViT fusion.",
    )
    args = parser.parse_args()
    main(
        loco_only=args.loco_only,
        background_words_csv=args.background_words_csv,
        output_root=args.output_root,
        min_test_loto=args.min_test_loto,
        loto_only=args.loto_only,
    )
