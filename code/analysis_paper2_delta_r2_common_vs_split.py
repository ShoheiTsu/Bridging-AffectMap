#!/usr/bin/env python3
"""
Paper2: 共通Encoder vs 分離Encoder の ΔR² 検定（同一テスト分割）。

現時点の実装は single split（theme-based train/val/test）で、
共通モデル `weights.pt` と分離モデル `split_encoders/*.pt` を同じ test に適用して比較する。
"""
import sys
from pathlib import Path
import json
import argparse

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from config import (
    OASIS_SCORES_CSV,
    RESULTS_STEP1,
    CVAE_CROSS_GENDER_DIR,
    RANDOM_SEED,
    VALENCE_AROUSAL_SCALE_MIN,
    VALENCE_AROUSAL_SCALE_MAX,
)
from dataset import load_oasis_meta, add_theme_base

TARGET_M = ["valence_male", "arousal_male"]
TARGET_F = ["valence_female", "arousal_female"]
INPUT_DIM = 512
HIDDEN = 128
DROPOUT = 0.1
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2


def denormalize_y(y_norm):
    return y_norm * (VALENCE_AROUSAL_SCALE_MAX - VALENCE_AROUSAL_SCALE_MIN) + VALENCE_AROUSAL_SCALE_MIN


def train_val_test_split_by_theme(df, train_ratio=0.6, val_ratio=0.2, random_state=42):
    df = add_theme_base(df)
    bases = df["theme_base"].unique()
    n = len(bases)
    order = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(order)
    n_train = max(1, int(n * train_ratio))
    n_val = max(0, int(n * val_ratio))
    n_test = n - n_train - n_val
    if n_test < 1:
        n_val = max(0, n_val - 1)
        n_test = n - n_train - n_val
    train_bases = set(bases[order[:n_train]])
    val_bases = set(bases[order[n_train:n_train + n_val]]) if n_val else set()
    test_bases = set(bases[order[n_train + n_val:]]) if n_test else set()
    train_idx = np.where(df["theme_base"].isin(train_bases).values)[0]
    val_idx = np.where(df["theme_base"].isin(val_bases).values)[0] if val_bases else np.array([], dtype=int)
    test_idx = np.where(df["theme_base"].isin(test_bases).values)[0] if test_bases else np.array([], dtype=int)
    return train_idx, val_idx, test_idx


def build_encoder_decoder(torch, latent_dim):
    import torch.nn as nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(INPUT_DIM, HIDDEN),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, HIDDEN),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, latent_dim),
            )

        def forward(self, x):
            return self.net(x)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, HIDDEN),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, HIDDEN),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, 2),
            )

        def forward(self, z):
            return self.net(z)

    return Encoder, Decoder


def load_common_predictions(X_test_scaled, ckpt_path, device):
    import torch

    ckpt = torch.load(ckpt_path, map_location=device)
    latent_dim = int(ckpt["config"]["latent_dim"])
    Encoder, Decoder = build_encoder_decoder(torch, latent_dim)
    enc = Encoder().to(device)
    dec_m = Decoder().to(device)
    dec_f = Decoder().to(device)
    enc.load_state_dict(ckpt["encoder"])
    dec_m.load_state_dict(ckpt["decoder_m"])
    dec_f.load_state_dict(ckpt["decoder_f"])
    enc.eval()
    dec_m.eval()
    dec_f.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_test_scaled).float().to(device)
        z = enc(xt)
        pm = dec_m(z).cpu().numpy()
        pf = dec_f(z).cpu().numpy()
    return denormalize_y(pm), denormalize_y(pf)


def load_split_predictions(X_test_scaled, ckpt_m_path, ckpt_f_path, device):
    import torch

    ckpt_m = torch.load(ckpt_m_path, map_location=device)
    ckpt_f = torch.load(ckpt_f_path, map_location=device)
    latent_dim = int(ckpt_m["config"]["latent_dim"])
    Encoder, Decoder = build_encoder_decoder(torch, latent_dim)

    enc_m = Encoder().to(device)
    enc_f = Encoder().to(device)
    dec_m = Decoder().to(device)
    dec_f = Decoder().to(device)
    enc_m.load_state_dict(ckpt_m["encoder"])
    dec_m.load_state_dict(ckpt_m["decoder"])
    enc_f.load_state_dict(ckpt_f["encoder"])
    dec_f.load_state_dict(ckpt_f["decoder"])
    enc_m.eval()
    dec_m.eval()
    enc_f.eval()
    dec_f.eval()

    with torch.no_grad():
        xt = torch.from_numpy(X_test_scaled).float().to(device)
        pm = dec_m(enc_m(xt)).cpu().numpy()
        pf = dec_f(enc_f(xt)).cpu().numpy()
    return denormalize_y(pm), denormalize_y(pf)


def compute_r2_summary(y_m, y_f, p_m, p_f):
    rmv = r2_score(y_m[:, 0], p_m[:, 0])
    rma = r2_score(y_m[:, 1], p_m[:, 1])
    rfv = r2_score(y_f[:, 0], p_f[:, 0])
    rfa = r2_score(y_f[:, 1], p_f[:, 1])
    return {
        "male_valence": float(rmv),
        "male_arousal": float(rma),
        "male_mean": float((rmv + rma) / 2),
        "female_valence": float(rfv),
        "female_arousal": float(rfa),
        "female_mean": float((rfv + rfa) / 2),
        "overall_mean": float((rmv + rma + rfv + rfa) / 4),
    }


def permutation_delta_r2(y_m, y_f, p_m_a, p_f_a, p_m_b, p_f_b, n_perm=2000, seed=RANDOM_SEED):
    """
    Paired permutation by sample-level model swap:
    A=common, B=split. Returns p-value for (B-A) >= observed.
    """
    rng = np.random.default_rng(seed)
    obs_a = compute_r2_summary(y_m, y_f, p_m_a, p_f_a)["overall_mean"]
    obs_b = compute_r2_summary(y_m, y_f, p_m_b, p_f_b)["overall_mean"]
    obs_delta = obs_b - obs_a

    n = len(y_m)
    null = []
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        pm_a = np.where(swap[:, None], p_m_b, p_m_a)
        pm_b = np.where(swap[:, None], p_m_a, p_m_b)
        pf_a = np.where(swap[:, None], p_f_b, p_f_a)
        pf_b = np.where(swap[:, None], p_f_a, p_f_b)
        da = compute_r2_summary(y_m, y_f, pm_a, pf_a)["overall_mean"]
        db = compute_r2_summary(y_m, y_f, pm_b, pf_b)["overall_mean"]
        null.append(db - da)
    null = np.array(null, dtype=float)
    p = (1 + np.sum(null >= obs_delta)) / (1 + len(null))
    return float(obs_delta), float(p), float(np.mean(null)), float(np.std(null))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=500)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--split-seed", type=int, default=RANDOM_SEED, help="theme split seed used for evaluation split")
    parser.add_argument("--split-tag", default="", help="subdir tag under split_encoders (e.g., seed42)")
    args = parser.parse_args()

    common_ckpt = CVAE_CROSS_GENDER_DIR / "weights.pt"
    split_dir = CVAE_CROSS_GENDER_DIR / "split_encoders"
    if args.split_tag:
        split_dir = split_dir / args.split_tag
    split_m = split_dir / "encoder_m.pt"
    split_f = split_dir / "encoder_f.pt"
    if not common_ckpt.exists() or not split_m.exists() or not split_f.exists():
        raise SystemExit("Required checkpoints not found (weights.pt, split_encoders/encoder_m.pt, encoder_f.pt).")

    X = np.load(RESULTS_STEP1 / "features_clip.npy").astype(np.float32)
    df = load_oasis_meta(OASIS_SCORES_CSV)
    cols = TARGET_M + TARGET_F
    valid = df[cols].notna().all(axis=1).values
    df = df.loc[valid].reset_index(drop=True)
    X = X[valid]
    y_m = df[TARGET_M].to_numpy(np.float32)
    y_f = df[TARGET_F].to_numpy(np.float32)

    train_idx, val_idx, test_idx = train_val_test_split_by_theme(
        df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, random_state=args.split_seed
    )
    scaler = StandardScaler().fit(X[train_idx])
    Xs = scaler.transform(X).astype(np.float32)
    Xte = Xs[test_idx]
    y_m_te = y_m[test_idx]
    y_f_te = y_f[test_idx]

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_m_common, p_f_common = load_common_predictions(Xte, common_ckpt, device)
    p_m_split, p_f_split = load_split_predictions(Xte, split_m, split_f, device)

    r_common = compute_r2_summary(y_m_te, y_f_te, p_m_common, p_f_common)
    r_split = compute_r2_summary(y_m_te, y_f_te, p_m_split, p_f_split)
    delta, pval, null_mean, null_std = permutation_delta_r2(
        y_m_te, y_f_te, p_m_common, p_f_common, p_m_split, p_f_split,
        n_perm=args.n_perm, seed=args.seed
    )

    out = {
        "evaluation": "single_split_theme_based",
        "n_test": int(len(test_idx)),
        "n_perm": int(args.n_perm),
        "split_seed": int(args.split_seed),
        "split_tag": args.split_tag,
        "r2_common": r_common,
        "r2_split": r_split,
        "delta_r2_overall_mean_split_minus_common": float(delta),
        "p_value_one_tailed_split_gt_common": float(pval),
        "null_mean": float(null_mean),
        "null_std": float(null_std),
    }
    out_path = CVAE_CROSS_GENDER_DIR / "paper2_delta_r2_common_vs_split.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

