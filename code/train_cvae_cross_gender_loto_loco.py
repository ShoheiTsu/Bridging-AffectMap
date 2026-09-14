#!/usr/bin/env python3
"""
クロス性別 cVAE を LOTO / LOCO で評価。

各 fold で:
  Phase 1: Encoder + Decoder_m を男性ターゲットで学習（train を theme で 80/20 に分割し val で監視）
  Phase 2: Encoder 凍結、Decoder_f を (z_train, y_female) で学習
  Test: 男性 within R²、女性 cross R² を fold ごとに記録し、全 fold を集約して overall R² を算出。

要: PyTorch, EmotionPro1 の features_clip.npy と oasis_scores.csv（男女別スコア・theme・category）
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from config import (
    OASIS_SCORES_CSV,
    RESULTS_STEP1,
    CVAE_CROSS_GENDER_DIR,
    RANDOM_SEED,
    VALENCE_AROUSAL_SCALE_MIN,
    VALENCE_AROUSAL_SCALE_MAX,
    CATEGORY_COLUMN,
)
from dataset import load_oasis_meta, add_theme_base, train_val_split_by_theme

TARGET_MALE = ["valence_male", "arousal_male"]
TARGET_FEMALE = ["valence_female", "arousal_female"]

INPUT_DIM = 512
LATENT_DIM = 64
HIDDEN = 128
DROPOUT = 0.1
BATCH_SIZE = 64
EPOCHS_PHASE1 = 120
EPOCHS_PHASE2 = 80
LR = 1e-3
SCALE_VA = (VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX)
MIN_TEST_LOTO = 2
MIN_TEST_LOCO = 5
PATIENCE = 15       # 早期打ち切り（val loss が改善しない epoch 数）
VAE_BETA = 0.01     # VAE 時の KL 重み（use_vae=True のとき）

# 簡略 nested: 内側グリッドと内側用 epoch（短めで高速化）
NESTED_LR_GRID = [1e-3, 3e-4]
NESTED_LATENT_DIM_GRID = [32, 64]
EPOCHS_PHASE1_INNER = 60   # 内側探索用（本番より短い）
EPOCHS_PHASE2_INNER = 40


def normalize_y(y, scale_min=1.0, scale_max=7.0):
    return (y - scale_min) / (scale_max - scale_min)


def denormalize_y(y_norm, scale_min=1.0, scale_max=7.0):
    return y_norm * (scale_max - scale_min) + scale_min


def _build_and_train_one_fold(
    X_scaled,
    y_m_norm,
    y_f_norm,
    train_idx,
    test_idx,
    df,
    device,
    seed,
    direction="male_first",
    epochs_p1=EPOCHS_PHASE1,
    epochs_p2=EPOCHS_PHASE2,
    patience=PATIENCE,
    use_vae=False,
    vae_beta=VAE_BETA,
    verbose=False,
    y_f_norm_phase2_override=None,
    lr=LR,
    latent_dim=LATENT_DIM,
):
    """
    Single-phase joint training: Encoder + Dec_m + Dec_f.
    loss = MSE(Dec_m(z), y_male) + MSE(Dec_f(z), y_female) (equal weights).
    direction is ignored (kept for call-site compatibility).
    y_f_norm_phase2_override: optional (n_samples, 2) female targets aligned to df rows
 (e.g. permuted labels); used for the female MSE branch during training only.
    """
    _ = direction
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    y_f_tgt = y_f_norm if y_f_norm_phase2_override is None else y_f_norm_phase2_override

    df_train = df.iloc[train_idx].reset_index(drop=True)
    _, _, train_inner_idx, val_inner_idx = train_val_split_by_theme(
        df_train, train_ratio=0.8, random_state=seed, return_indices=True
    )
    train_inner = train_idx[train_inner_idx]
    val_inner = train_idx[val_inner_idx]
    n_tr = len(train_inner)
    n_val = len(val_inner)
    n_te = len(test_idx)
    if n_te == 0:
        return None

    def to_t(a):
        return torch.from_numpy(a).float().to(device)

    class Encoder(nn.Module):
        def __init__(self, vae=False, lat_dim=latent_dim):
            super().__init__()
            self.vae = vae
            self.net = nn.Sequential(
                nn.Linear(INPUT_DIM, HIDDEN),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, HIDDEN),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
            )
            self.fc_mu = nn.Linear(HIDDEN, lat_dim)
            self.fc_logvar = nn.Linear(HIDDEN, lat_dim) if vae else None
            if not vae:
                self.fc_out = nn.Linear(HIDDEN, lat_dim)

        def forward(self, x):
            h = self.net(x)
            if self.vae:
                mu = self.fc_mu(h)
                logvar = self.fc_logvar(h)
                return mu, logvar
            return self.fc_out(h)

    class Decoder(nn.Module):
        def __init__(self, lat_dim=latent_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(lat_dim, HIDDEN),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, HIDDEN),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, 2),
            )

        def forward(self, z):
            return self.net(z)

    encoder = Encoder(vae=use_vae, lat_dim=latent_dim).to(device)
    decoder_m = Decoder(lat_dim=latent_dim).to(device)
    decoder_f = Decoder(lat_dim=latent_dim).to(device)
    mse = nn.MSELoss()

    def get_z(enc, x, sample=False):
        if use_vae:
            mu, logvar = enc(x)
            if sample and enc.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std, device=x.device)
                return mu + eps * std, mu, logvar
            return mu, mu, logvar
        return enc(x), None, None

    def kl_loss(mu, logvar):
        if mu is None:
            return torch.tensor(0.0, device=device)
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    opt = torch.optim.Adam(
        list(encoder.parameters())
        + list(decoder_m.parameters())
        + list(decoder_f.parameters()),
        lr=lr,
    )
    X_tr = to_t(X_scaled[train_inner])
    y_m_tr = to_t(y_m_norm[train_inner])
    y_f_tr = to_t(y_f_tgt[train_inner])
    X_val = to_t(X_scaled[val_inner]) if n_val > 0 else None
    y_m_val = to_t(y_m_norm[val_inner]) if n_val > 0 else None
    y_f_val = to_t(y_f_tgt[val_inner]) if n_val > 0 else None

    n_epochs = epochs_p1 + epochs_p2
    best_val = float("inf")
    no_improve = 0
    for epoch in range(n_epochs):
        encoder.train()
        decoder_m.train()
        decoder_f.train()
        perm = np.random.permutation(n_tr)
        for start in range(0, n_tr, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_tr)
            idx = perm[start:end]
            z, mu, logvar = get_z(encoder, X_tr[idx], sample=use_vae)
            pred_m = decoder_m(z)
            pred_f = decoder_f(z)
            loss = mse(pred_m, y_m_tr[idx]) + mse(pred_f, y_f_tr[idx])
            if use_vae and mu is not None:
                loss = loss + vae_beta * kl_loss(mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if X_val is not None:
            encoder.eval()
            decoder_m.eval()
            decoder_f.eval()
            with torch.no_grad():
                z_v, mu_v, _ = get_z(encoder, X_val, sample=False)
                v = mse(decoder_m(z_v), y_m_val).item() + mse(decoder_f(z_v), y_f_val).item()
            if v < best_val:
                best_val = v
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"    Joint early stop epoch {epoch+1}")
                break

    encoder.eval()
    decoder_m.eval()
    decoder_f.eval()
    with torch.no_grad():
        X_te = to_t(X_scaled[test_idx])
        z_te, _, _ = get_z(encoder, X_te, sample=False)
        pred_m = decoder_m(z_te).cpu().numpy()
        pred_f = decoder_f(z_te).cpu().numpy()

    return {
        "pred_male_te": denormalize_y(pred_m, *SCALE_VA),
        "pred_female_te": denormalize_y(pred_f, *SCALE_VA),
    }



def _split_inner_train_val(train_idx, df, seed):
    """外側訓練を theme_base で 80/20 に分割し、内側訓練・内側 val のインデックスを返す。"""
    df_train = df.iloc[train_idx].reset_index(drop=True)
    _, _, train_inner_idx, val_inner_idx = train_val_split_by_theme(
        df_train, train_ratio=0.8, random_state=seed, return_indices=True
    )
    return train_idx[train_inner_idx], train_idx[val_inner_idx]


def _overall_joint_from_stacked(y_m, p_m, y_f, p_f):
    r2_m_v = r2_score(y_m[:, 0], p_m[:, 0])
    r2_m_a = r2_score(y_m[:, 1], p_m[:, 1])
    r2_f_v = r2_score(y_f[:, 0], p_f[:, 0])
    r2_f_a = r2_score(y_f[:, 1], p_f[:, 1])
    return {
        "R2_male_valence": float(r2_m_v),
        "R2_male_arousal": float(r2_m_a),
        "R2_male_mean": float((r2_m_v + r2_m_a) / 2),
        "R2_female_valence": float(r2_f_v),
        "R2_female_arousal": float(r2_f_a),
        "R2_female_mean": float((r2_f_v + r2_f_a) / 2),
    }


def _dual_direction_summaries(base_summary, oj):
    """oj: R2_male_* / R2_female_* from joint training. Adds legacy within/cross keys."""
    meta = {k: v for k, v in base_summary.items() if k != "overall"}
    mf = {
        **meta,
        "direction": "male_first",
        "overall": {
            "R2_within_valence": oj["R2_male_valence"],
            "R2_within_arousal": oj["R2_male_arousal"],
            "R2_within_mean": oj["R2_male_mean"],
            "R2_cross_valence": oj["R2_female_valence"],
            "R2_cross_arousal": oj["R2_female_arousal"],
            "R2_cross_mean": oj["R2_female_mean"],
        },
    }
    ff = {
        **meta,
        "direction": "female_first",
        "overall": {
            "R2_within_valence": oj["R2_female_valence"],
            "R2_within_arousal": oj["R2_female_arousal"],
            "R2_within_mean": oj["R2_female_mean"],
            "R2_cross_valence": oj["R2_male_valence"],
            "R2_cross_arousal": oj["R2_male_arousal"],
            "R2_cross_mean": oj["R2_male_mean"],
        },
    }
    return mf, ff


def _expand_fold_rows_for_legacy(rows):
    """Each row must have R2_male_* and R2_female_*; emit two CSV rows (within/cross swap)."""
    out = []
    for r in rows:
        for direction in ("male_first", "female_first"):
            row = dict(r)
            row["direction"] = direction
            if direction == "male_first":
                row["R2_within_valence"] = r["R2_male_valence"]
                row["R2_within_arousal"] = r["R2_male_arousal"]
                row["R2_within_mean"] = r["R2_male_mean"]
                row["R2_cross_valence"] = r["R2_female_valence"]
                row["R2_cross_arousal"] = r["R2_female_arousal"]
                row["R2_cross_mean"] = r["R2_female_mean"]
            else:
                row["R2_within_valence"] = r["R2_female_valence"]
                row["R2_within_arousal"] = r["R2_female_arousal"]
                row["R2_within_mean"] = r["R2_female_mean"]
                row["R2_cross_valence"] = r["R2_male_valence"]
                row["R2_cross_arousal"] = r["R2_male_arousal"]
                row["R2_cross_mean"] = r["R2_male_mean"]
            out.append(row)
    return out


def run_loto(X, y_male, y_female, df, device, seed=RANDOM_SEED, direction="male_first",
             use_vae=False, patience=PATIENCE, verbose=True, use_nested=False, loto_by_theme=False,
             all_images=False):
    """direction: male_first → within=男性, cross=女性. female_first → within=女性, cross=男性.
    use_nested: True なら内側 80/20 で LR と LATENT_DIM をグリッド探索し、最良で外側訓練全体を学習。
    loto_by_theme: False なら theme_base（例: Acorns）で 1 fold = 約 248。True なら theme（例: Acorns 1）で 1 fold = 391。
    all_images: True なら全 fold を実行（スキップなし）し、df 行順で全画像の予測を返す。
    False なら n_test < MIN_TEST_LOTO の fold はスキップ。
    Joint training（direction 引数は互換のため残すが無視）。
    """
    _ = direction
    import itertools
    df = add_theme_base(df)
    n_df = len(df)
    min_test = 0 if all_images else MIN_TEST_LOTO
    if loto_by_theme and "theme" in df.columns:
        leave_out_values = df["theme"].unique().tolist()
        leave_out_col = "theme"
    else:
        leave_out_values = df["theme_base"].unique().tolist()
        leave_out_col = "theme_base"
    y_m_norm = normalize_y(y_male, *SCALE_VA)
    y_f_norm = normalize_y(y_female, *SCALE_VA)
    all_y_m = []
    all_p_m = []
    all_y_f = []
    all_p_f = []
    fold_rows = []
    pred_male_all = np.full((n_df, 2), np.nan, dtype=np.float32) if all_images else None
    pred_female_all = np.full((n_df, 2), np.nan, dtype=np.float32) if all_images else None
    for fold_idx, left_out in enumerate(leave_out_values):
        train_mask = (df[leave_out_col] != left_out).values
        test_mask = (df[leave_out_col] == left_out).values
        n_test = test_mask.sum()
        if n_test < min_test:
            continue
        if verbose and fold_idx % 30 == 0:
            print(f"  LOTO [joint] fold {fold_idx+1}/{len(leave_out_values)}: {left_out} (n_test={n_test})"
                  + (" [nested]" if use_nested else "") + (" [by theme]" if loto_by_theme else ""))
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        scaler = StandardScaler().fit(X[train_idx])
        X_scaled = scaler.transform(X)

        if use_nested:
            inner_train_idx, inner_val_idx = _split_inner_train_val(train_idx, df, seed + fold_idx)
            if len(inner_val_idx) < 2:
                res = _build_and_train_one_fold(
                    X_scaled, y_m_norm, y_f_norm, train_idx, test_idx, df, device, seed + fold_idx,
                    use_vae=use_vae, patience=patience, verbose=False,
                )
            else:
                best_score = -np.inf
                best_lr, best_latent = LR, LATENT_DIM
                for lr, lat in itertools.product(NESTED_LR_GRID, NESTED_LATENT_DIM_GRID):
                    res_in = _build_and_train_one_fold(
                        X_scaled, y_m_norm, y_f_norm, inner_train_idx, inner_val_idx, df, device,
                        seed + fold_idx + 1000, use_vae=use_vae, patience=patience,
                        verbose=False, lr=lr, latent_dim=lat,
                        epochs_p1=EPOCHS_PHASE1_INNER, epochs_p2=EPOCHS_PHASE2_INNER,
                    )
                    if res_in is None:
                        continue
                    y_m_val = y_male[inner_val_idx]
                    y_f_val = y_female[inner_val_idx]
                    p_m = res_in["pred_male_te"]
                    p_f = res_in["pred_female_te"]
                    r2_m = (r2_score(y_m_val[:, 0], p_m[:, 0]) + r2_score(y_m_val[:, 1], p_m[:, 1])) / 2
                    r2_f = (r2_score(y_f_val[:, 0], p_f[:, 0]) + r2_score(y_f_val[:, 1], p_f[:, 1])) / 2
                    score = (r2_m + r2_f) / 2
                    if score > best_score:
                        best_score = score
                        best_lr, best_latent = lr, lat
                res = _build_and_train_one_fold(
                    X_scaled, y_m_norm, y_f_norm, train_idx, test_idx, df, device, seed + fold_idx,
                    use_vae=use_vae, patience=patience, verbose=False,
                    lr=best_lr, latent_dim=best_latent,
                )
        else:
            res = _build_and_train_one_fold(
                X_scaled, y_m_norm, y_f_norm, train_idx, test_idx, df, device, seed + fold_idx,
                use_vae=use_vae, patience=patience, verbose=False,
            )
        if res is None:
            continue
        y_m_te = y_male[test_idx]
        y_f_te = y_female[test_idx]
        p_m = res["pred_male_te"]
        p_f = res["pred_female_te"]
        if all_images:
            pred_male_all[test_idx] = p_m
            pred_female_all[test_idx] = p_f
        # R² は「2 サンプル以上」の fold のみで定義。全体 R² はこれらの fold を集約した試行で算出
        if n_test >= 2:
            all_y_m.append(y_m_te)
            all_p_m.append(p_m)
            all_y_f.append(y_f_te)
            all_p_f.append(p_f)
            r2_m_v = r2_score(y_m_te[:, 0], p_m[:, 0])
            r2_m_a = r2_score(y_m_te[:, 1], p_m[:, 1])
            r2_f_v = r2_score(y_f_te[:, 0], p_f[:, 0])
            r2_f_a = r2_score(y_f_te[:, 1], p_f[:, 1])
            fold_rows.append({
                "left_out": left_out,
                "n_train": len(train_idx),
                "n_test": n_test,
                "R2_male_valence": round(r2_m_v, 4),
                "R2_male_arousal": round(r2_m_a, 4),
                "R2_male_mean": round((r2_m_v + r2_m_a) / 2, 4),
                "R2_female_valence": round(r2_f_v, 4),
                "R2_female_arousal": round(r2_f_a, 4),
                "R2_female_mean": round((r2_f_v + r2_f_a) / 2, 4),
            })
    if len(all_y_m) == 0:
        return None, None, None, None
    y_m_st = np.vstack(all_y_m)
    p_m_st = np.vstack(all_p_m)
    y_f_st = np.vstack(all_y_f)
    p_f_st = np.vstack(all_p_f)
    oj = _overall_joint_from_stacked(y_m_st, p_m_st, y_f_st, p_f_st)
    summary = {
        "evaluation": "LOTO",
        "direction": "joint",
        "nested": use_nested,
        "all_images": all_images,
        "n_folds": len(all_y_m),
        "overall": oj,
    }
    return summary, fold_rows, pred_male_all, pred_female_all


def run_loco(X, y_male, y_female, df, device, seed=RANDOM_SEED, direction="male_first",
             use_vae=False, patience=PATIENCE, verbose=True, use_nested=False, all_images=False):
    """Joint training (same z). direction ignored (API compat)."""
    _ = direction
    import itertools
    n_df = len(df)
    min_test = 0 if all_images else MIN_TEST_LOCO
    categories = df[CATEGORY_COLUMN].unique().tolist()
    y_m_norm = normalize_y(y_male, *SCALE_VA)
    y_f_norm = normalize_y(y_female, *SCALE_VA)
    all_y_m = []
    all_p_m = []
    all_y_f = []
    all_p_f = []
    fold_rows = []
    pred_male_all = np.full((n_df, 2), np.nan, dtype=np.float32) if all_images else None
    pred_female_all = np.full((n_df, 2), np.nan, dtype=np.float32) if all_images else None
    for fold_idx, left_out in enumerate(categories):
        train_mask = (df[CATEGORY_COLUMN] != left_out).values
        test_mask = (df[CATEGORY_COLUMN] == left_out).values
        n_test = test_mask.sum()
        if n_test < min_test:
            continue
        if verbose:
            print(f"  LOCO [joint] fold {fold_idx+1}/{len(categories)}: {left_out} (n_test={n_test})"
                  + (" [nested]" if use_nested else ""))
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]
        scaler = StandardScaler().fit(X[train_idx])
        X_scaled = scaler.transform(X)
        if use_nested:
            inner_train_idx, inner_val_idx = _split_inner_train_val(train_idx, df, seed + fold_idx)
            if len(inner_val_idx) < 2:
                res = _build_and_train_one_fold(
                    X_scaled, y_m_norm, y_f_norm, train_idx, test_idx, df, device, seed + fold_idx,
                    use_vae=use_vae, patience=patience, verbose=False,
                )
            else:
                best_score = -np.inf
                best_lr, best_latent = LR, LATENT_DIM
                for lr, lat in itertools.product(NESTED_LR_GRID, NESTED_LATENT_DIM_GRID):
                    res_in = _build_and_train_one_fold(
                        X_scaled, y_m_norm, y_f_norm, inner_train_idx, inner_val_idx, df, device,
                        seed + fold_idx + 1000, use_vae=use_vae, patience=patience,
                        verbose=False, lr=lr, latent_dim=lat,
                        epochs_p1=EPOCHS_PHASE1_INNER, epochs_p2=EPOCHS_PHASE2_INNER,
                    )
                    if res_in is None:
                        continue
                    y_m_val = y_male[inner_val_idx]
                    y_f_val = y_female[inner_val_idx]
                    p_m = res_in["pred_male_te"]
                    p_f = res_in["pred_female_te"]
                    r2_m = (r2_score(y_m_val[:, 0], p_m[:, 0]) + r2_score(y_m_val[:, 1], p_m[:, 1])) / 2
                    r2_f = (r2_score(y_f_val[:, 0], p_f[:, 0]) + r2_score(y_f_val[:, 1], p_f[:, 1])) / 2
                    score = (r2_m + r2_f) / 2
                    if score > best_score:
                        best_score = score
                        best_lr, best_latent = lr, lat
                res = _build_and_train_one_fold(
                    X_scaled, y_m_norm, y_f_norm, train_idx, test_idx, df, device, seed + fold_idx,
                    use_vae=use_vae, patience=patience, verbose=False,
                    lr=best_lr, latent_dim=best_latent,
                )
        else:
            res = _build_and_train_one_fold(
                X_scaled, y_m_norm, y_f_norm, train_idx, test_idx, df, device, seed + fold_idx,
                use_vae=use_vae, patience=patience, verbose=False,
            )
        if res is None:
            continue
        y_m_te = y_male[test_idx]
        y_f_te = y_female[test_idx]
        p_m, p_f = res["pred_male_te"], res["pred_female_te"]
        if all_images:
            pred_male_all[test_idx] = p_m
            pred_female_all[test_idx] = p_f
        if n_test >= 2:
            all_y_m.append(y_m_te)
            all_p_m.append(p_m)
            all_y_f.append(y_f_te)
            all_p_f.append(p_f)
            r2_m_v = r2_score(y_m_te[:, 0], p_m[:, 0])
            r2_m_a = r2_score(y_m_te[:, 1], p_m[:, 1])
            r2_f_v = r2_score(y_f_te[:, 0], p_f[:, 0])
            r2_f_a = r2_score(y_f_te[:, 1], p_f[:, 1])
            fold_rows.append({
                "left_out": left_out,
                "n_train": len(train_idx),
                "n_test": n_test,
                "R2_male_valence": round(r2_m_v, 4),
                "R2_male_arousal": round(r2_m_a, 4),
                "R2_male_mean": round((r2_m_v + r2_m_a) / 2, 4),
                "R2_female_valence": round(r2_f_v, 4),
                "R2_female_arousal": round(r2_f_a, 4),
                "R2_female_mean": round((r2_f_v + r2_f_a) / 2, 4),
            })
    if len(all_y_m) == 0:
        return None, None, None, None
    y_m_st = np.vstack(all_y_m)
    p_m_st = np.vstack(all_p_m)
    y_f_st = np.vstack(all_y_f)
    p_f_st = np.vstack(all_p_f)
    oj = _overall_joint_from_stacked(y_m_st, p_m_st, y_f_st, p_f_st)
    summary = {
        "evaluation": "LOCO",
        "direction": "joint",
        "nested": use_nested,
        "all_images": all_images,
        "n_folds": len(all_y_m),
        "overall": oj,
    }
    return summary, fold_rows, pred_male_all, pred_female_all



def main():
    import argparse
    parser = argparse.ArgumentParser(description="cVAE cross-gender LOTO/LOCO (both directions)")
    parser.add_argument("--vae", action="store_true", help="Use VAE (KL) instead of deterministic encoder")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience (default %d)" % PATIENCE)
    parser.add_argument("--loto-only", action="store_true", help="Run only LOTO")
    parser.add_argument("--loco-only", action="store_true", help="Run only LOCO")
    parser.add_argument("--nested", action="store_true", help="Simplified nested: inner 80/20 grid over LR and LATENT_DIM, then train on full outer train with best")
    parser.add_argument("--loto-by-theme", action="store_true", help="LOTO by theme (391 folds). Default: by theme_base (248 folds)")
    parser.add_argument("--all-images", action="store_true", help="Run all folds (no skip), save per-image predictions in df order (y_all_*.npy, pred_*_*.npy)")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("This script requires PyTorch. Install with: pip install torch")
        return

    CVAE_CROSS_GENDER_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_nested" if args.nested else ""
    out_loto = CVAE_CROSS_GENDER_DIR / ("loto" + suffix)
    out_loco = CVAE_CROSS_GENDER_DIR / ("loco" + suffix)
    out_loto.mkdir(exist_ok=True)
    out_loco.mkdir(exist_ok=True)

    X_clip = np.load(RESULTS_STEP1 / "features_clip.npy").astype(np.float32)
    df = load_oasis_meta(OASIS_SCORES_CSV)
    df = add_theme_base(df)
    cols = TARGET_MALE + TARGET_FEMALE
    if not all(c in df.columns for c in cols):
        print("Missing valence_male, arousal_male, valence_female, arousal_female in CSV.")
        return
    valid = df[cols].notna().all(axis=1).values
    df = df.loc[valid].reset_index(drop=True)
    X_clip = X_clip[valid]
    y_male = df[TARGET_MALE].values.astype(np.float32)
    y_female = df[TARGET_FEMALE].values.astype(np.float32)
    print(f"Samples: {len(df)}  use_vae={args.vae}  patience={args.patience}  nested={args.nested}  all_images={args.all_images}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _print_joint(label, oj):
        print(
            f"  {label} R² male:   V={oj['R2_male_valence']:.4f} A={oj['R2_male_arousal']:.4f} "
            f"mean={oj['R2_male_mean']:.4f}"
        )
        print(
            f"  {label} R² female: V={oj['R2_female_valence']:.4f} A={oj['R2_female_arousal']:.4f} "
            f"mean={oj['R2_female_mean']:.4f}"
        )

    def _print_overall(label, o):
        print(f"  {label} R² within:  V={o['R2_within_valence']:.4f} A={o['R2_within_arousal']:.4f} mean={o['R2_within_mean']:.4f}")
        print(f"  {label} R² cross:   V={o['R2_cross_valence']:.4f} A={o['R2_cross_arousal']:.4f} mean={o['R2_cross_mean']:.4f}")

    summary_loto_combined = {}
    all_rows_loto = []

    if not args.loco_only:
        print("=== LOTO (Leave-One-Theme-Out), joint training ===")
        summary, rows, pred_m, pred_f = run_loto(
            X_clip, y_male, y_female, df, device,
            use_vae=args.vae, patience=args.patience, use_nested=args.nested,
            loto_by_theme=args.loto_by_theme, all_images=args.all_images,
        )
        if summary:
            oj = summary["overall"]
            mf, ff = _dual_direction_summaries(summary, oj)
            summary_loto_combined = {"joint": summary, "male_first": mf, "female_first": ff}
            _print_joint("LOTO", oj)
            _print_overall("LOTO (male_first 表記)", mf["overall"])
            _print_overall("LOTO (female_first 表記)", ff["overall"])
            all_rows_loto = _expand_fold_rows_for_legacy(rows)
            with open(out_loto / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_loto_combined, f, indent=2)
            pd.DataFrame(all_rows_loto).to_csv(out_loto / "cvae_loto.csv", index=False)
            print(f"  Saved {out_loto / 'summary.json'}, {out_loto / 'cvae_loto.csv'}")
        if args.all_images and pred_m is not None:
            np.save(out_loto / "y_all_true_male.npy", y_male)
            np.save(out_loto / "y_all_true_female.npy", y_female)
            for direction in ("male_first", "female_first"):
                np.save(out_loto / f"y_all_pred_male_{direction}.npy", pred_m)
                np.save(out_loto / f"y_all_pred_female_{direction}.npy", pred_f)
            print(f"  Saved per-image LOTO: y_all_true_*.npy, y_all_pred_*_*.npy (df order, n={len(df)})")

    summary_loco_combined = {}
    all_rows_loco = []

    if not args.loto_only:
        print("\n=== LOCO (Leave-One-Category-Out), joint training ===")
        summary, rows, pred_m, pred_f = run_loco(
            X_clip, y_male, y_female, df, device,
            use_vae=args.vae, patience=args.patience, use_nested=args.nested,
            all_images=args.all_images,
        )
        if summary:
            oj = summary["overall"]
            mf, ff = _dual_direction_summaries(summary, oj)
            summary_loco_combined = {"joint": summary, "male_first": mf, "female_first": ff}
            _print_joint("LOCO", oj)
            _print_overall("LOCO (male_first 表記)", mf["overall"])
            _print_overall("LOCO (female_first 表記)", ff["overall"])
            all_rows_loco = _expand_fold_rows_for_legacy(rows)
            with open(out_loco / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_loco_combined, f, indent=2)
            pd.DataFrame(all_rows_loco).to_csv(out_loco / "cvae_loco.csv", index=False)
            print(f"  Saved {out_loco / 'summary.json'}, {out_loco / 'cvae_loco.csv'}")
        if args.all_images and pred_m is not None:
            np.save(out_loco / "y_all_true_male.npy", y_male)
            np.save(out_loco / "y_all_true_female.npy", y_female)
            for direction in ("male_first", "female_first"):
                np.save(out_loco / f"y_all_pred_male_{direction}.npy", pred_m)
                np.save(out_loco / f"y_all_pred_female_{direction}.npy", pred_f)
            print(f"  Saved per-image LOCO: y_all_true_*.npy, y_all_pred_*_*.npy (df order, n={len(df)})")

    print("\nDone.")


if __name__ == "__main__":
    main()
