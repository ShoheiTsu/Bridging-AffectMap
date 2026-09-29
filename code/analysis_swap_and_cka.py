#!/usr/bin/env python3
"""
Falsifiers B/C:
- B: encoder-swap degradation (single split)
- C: linear CKA between Encoder_m and Encoder_f
"""
import sys
from pathlib import Path
import argparse
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from config import OASIS_SCORES_CSV, RESULTS_STEP1, CVAE_CROSS_GENDER_DIR, RANDOM_SEED
from dataset import load_oasis_meta, add_theme_base

TARGET_M = ["valence_male", "arousal_male"]
TARGET_F = ["valence_female", "arousal_female"]
INPUT_DIM = 512
HIDDEN = 128
DROPOUT = 0.1


def split_by_theme(df, seed, tr=0.6, va=0.2):
    df = add_theme_base(df)
    bases = df["theme_base"].unique()
    order = np.arange(len(bases))
    np.random.seed(seed)
    np.random.shuffle(order)
    n_tr = max(1, int(len(bases) * tr))
    n_va = max(0, int(len(bases) * va))
    tr_b = set(bases[order[:n_tr]])
    va_b = set(bases[order[n_tr:n_tr + n_va]])
    te_b = set(bases[order[n_tr + n_va:]])
    tr_idx = np.where(df["theme_base"].isin(tr_b).values)[0]
    va_idx = np.where(df["theme_base"].isin(va_b).values)[0] if va_b else np.array([], dtype=int)
    te_idx = np.where(df["theme_base"].isin(te_b).values)[0] if te_b else np.array([], dtype=int)
    return tr_idx, va_idx, te_idx


def denorm(y):
    return y * 6.0 + 1.0


def build_modules(torch, latent_dim):
    import torch.nn as nn
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(INPUT_DIM, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, latent_dim),
            )
        def forward(self, x): return self.net(x)
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, 2),
            )
        def forward(self, z): return self.net(z)
    return Encoder, Decoder


def linear_cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    dotxy = X.T @ Y
    hsic_xy = np.sum(dotxy ** 2)
    hsic_xx = np.sum((X.T @ X) ** 2)
    hsic_yy = np.sum((Y.T @ Y) ** 2)
    denom = np.sqrt(max(hsic_xx, 1e-12) * max(hsic_yy, 1e-12))
    return float(hsic_xy / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-tag", required=True)
    ap.add_argument("--split-seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    split_dir = CVAE_CROSS_GENDER_DIR / "split_encoders" / args.split_tag
    wm = split_dir / "encoder_m.pt"
    wf = split_dir / "encoder_f.pt"
    if not wm.exists() or not wf.exists():
        raise SystemExit(f"not found: {wm} / {wf}")

    X = np.load(RESULTS_STEP1 / "features_clip.npy").astype(np.float32)
    df = load_oasis_meta(OASIS_SCORES_CSV)
    valid = df[TARGET_M + TARGET_F].notna().all(axis=1).values
    df = df.loc[valid].reset_index(drop=True)
    X = X[valid]
    y_m = df[TARGET_M].to_numpy(np.float32)
    y_f = df[TARGET_F].to_numpy(np.float32)
    tr_idx, _, te_idx = split_by_theme(df, args.split_seed)
    scaler = StandardScaler().fit(X[tr_idx])
    Xs = scaler.transform(X).astype(np.float32)
    Xte = Xs[te_idx]
    y_m_te = y_m[te_idx]
    y_f_te = y_f[te_idx]

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_m = torch.load(wm, map_location=device)
    ck_f = torch.load(wf, map_location=device)
    lat = int(ck_m["config"]["latent_dim"])
    Encoder, Decoder = build_modules(torch, lat)
    enc_m, enc_f = Encoder().to(device), Encoder().to(device)
    dec_m, dec_f = Decoder().to(device), Decoder().to(device)
    enc_m.load_state_dict(ck_m["encoder"]); dec_m.load_state_dict(ck_m["decoder"])
    enc_f.load_state_dict(ck_f["encoder"]); dec_f.load_state_dict(ck_f["decoder"])
    for m in (enc_m, enc_f, dec_m, dec_f): m.eval()

    with torch.no_grad():
        xt = torch.from_numpy(Xte).float().to(device)
        z_m = enc_m(xt).cpu().numpy()
        z_f = enc_f(xt).cpu().numpy()
        p_mm = denorm(dec_m(enc_m(xt)).cpu().numpy())  # native male
        p_ff = denorm(dec_f(enc_f(xt)).cpu().numpy())  # native female
        p_fm = denorm(dec_m(enc_f(xt)).cpu().numpy())  # swap F->M
        p_mf = denorm(dec_f(enc_m(xt)).cpu().numpy())  # swap M->F

    # swap degradation (mean of valence+arousal R2)
    r2_m_native = (r2_score(y_m_te[:, 0], p_mm[:, 0]) + r2_score(y_m_te[:, 1], p_mm[:, 1])) / 2
    r2_m_swap = (r2_score(y_m_te[:, 0], p_fm[:, 0]) + r2_score(y_m_te[:, 1], p_fm[:, 1])) / 2
    r2_f_native = (r2_score(y_f_te[:, 0], p_ff[:, 0]) + r2_score(y_f_te[:, 1], p_ff[:, 1])) / 2
    r2_f_swap = (r2_score(y_f_te[:, 0], p_mf[:, 0]) + r2_score(y_f_te[:, 1], p_mf[:, 1])) / 2

    cka = linear_cka(z_m, z_f)
    out = {
        "split_tag": args.split_tag,
        "split_seed": int(args.split_seed),
        "n_test": int(len(te_idx)),
        "swap": {
            "male_native_r2_mean": float(r2_m_native),
            "male_swap_r2_mean": float(r2_m_swap),
            "male_degradation": float(r2_m_native - r2_m_swap),
            "female_native_r2_mean": float(r2_f_native),
            "female_swap_r2_mean": float(r2_f_swap),
            "female_degradation": float(r2_f_native - r2_f_swap),
        },
        "cka": {
            "linear_cka_zm_zf": float(cka)
        }
    }
    out_path = CVAE_CROSS_GENDER_DIR / f"swap_cka_{args.split_tag}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

