#!/usr/bin/env python3
"""
Follow-up to the swap analysis:
Swap recovery with a linear map T (separating co-adaptation from a true encoder gap).

Protocol (fixed split):
1) Get z_m, z_f from split encoders
2) Fit T_f2m: z_f -> z_m and T_m2f: z_m -> z_f on train (Ridge)
3) Compare on test:
   - native: Dec_m(z_m), Dec_f(z_f)
   - raw swap: Dec_m(z_f), Dec_f(z_m)
   - corrected swap: Dec_m(T_f2m(z_f)), Dec_f(T_m2f(z_m))
4) Compute recovery ratios
"""
import sys
from pathlib import Path
import argparse
import json

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from config import OASIS_SCORES_CSV, RESULTS_STEP1, CVAE_CROSS_GENDER_DIR, RANDOM_SEED
from dataset import load_oasis_meta, add_theme_base

TARGET_M = ["valence_male", "arousal_male"]
TARGET_F = ["valence_female", "arousal_female"]
INPUT_DIM, HIDDEN, DROPOUT = 512, 128, 0.1


def split_by_theme(df, seed=42, tr=0.6, va=0.2):
    df = add_theme_base(df)
    bases = df["theme_base"].unique()
    order = np.arange(len(bases))
    np.random.seed(seed); np.random.shuffle(order)
    ntr = max(1, int(len(bases) * tr)); nva = max(0, int(len(bases) * va))
    tr_b = set(bases[order[:ntr]]); te_b = set(bases[order[ntr + nva:]])
    tr_idx = np.where(df["theme_base"].isin(tr_b).values)[0]
    te_idx = np.where(df["theme_base"].isin(te_b).values)[0]
    return tr_idx, te_idx


def denorm(y): return y * 6.0 + 1.0


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


def load_split_models(split_tag, device):
    import torch
    d = CVAE_CROSS_GENDER_DIR / "split_encoders" / split_tag
    pm = d / "encoder_m.pt"
    pf = d / "encoder_f.pt"
    if not pm.exists() or not pf.exists():
        raise SystemExit(f"not found: {pm} / {pf}")
    ck_m = torch.load(pm, map_location=device)
    ck_f = torch.load(pf, map_location=device)
    lat = int(ck_m["config"]["latent_dim"])
    Encoder, Decoder = build_modules(torch, lat)
    enc_m, enc_f = Encoder().to(device), Encoder().to(device)
    dec_m, dec_f = Decoder().to(device), Decoder().to(device)
    enc_m.load_state_dict(ck_m["encoder"]); dec_m.load_state_dict(ck_m["decoder"])
    enc_f.load_state_dict(ck_f["encoder"]); dec_f.load_state_dict(ck_f["decoder"])
    for m in (enc_m, enc_f, dec_m, dec_f): m.eval()
    scaler_mean = np.array(ck_m["scaler_x_mean"], dtype=np.float32)
    scaler_scale = np.array(ck_m["scaler_x_scale"], dtype=np.float32)
    return enc_m, enc_f, dec_m, dec_f, scaler_mean, scaler_scale


def r2_mean(y, p):
    return float((r2_score(y[:, 0], p[:, 0]) + r2_score(y[:, 1], p[:, 1])) / 2)


def recovery_ratio(native, raw_swap, corrected):
    denom = native - raw_swap
    if abs(denom) < 1e-12:
        return None
    return float((corrected - raw_swap) / denom)


def cond_number(A):
    s = np.linalg.svd(A, compute_uv=False)
    s = s[s > 1e-12]
    if len(s) == 0:
        return np.inf
    return float(np.max(s) / np.min(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-tag", required=True, help="e.g., seed42")
    ap.add_argument("--split-seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha for latent linear map")
    args = ap.parse_args()

    X = np.load(RESULTS_STEP1 / "features_clip.npy").astype(np.float32)
    df = load_oasis_meta(OASIS_SCORES_CSV)
    valid = df[TARGET_M + TARGET_F].notna().all(axis=1).values
    df = df.loc[valid].reset_index(drop=True)
    X = X[valid]
    y_m = df[TARGET_M].to_numpy(np.float32)
    y_f = df[TARGET_F].to_numpy(np.float32)
    tr_idx, te_idx = split_by_theme(df, seed=args.split_seed)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc_m, enc_f, dec_m, dec_f, mean, scale = load_split_models(args.split_tag, device)
    Xs = ((X - mean) / scale).astype(np.float32)
    Xtr = torch.from_numpy(Xs[tr_idx]).float().to(device)
    Xte = torch.from_numpy(Xs[te_idx]).float().to(device)

    with torch.no_grad():
        z_m_tr = enc_m(Xtr).cpu().numpy()
        z_f_tr = enc_f(Xtr).cpu().numpy()
        z_m_te = enc_m(Xte).cpu().numpy()
        z_f_te = enc_f(Xte).cpu().numpy()

    # Learn linear maps in latent space
    T_f2m = Ridge(alpha=args.alpha, fit_intercept=True).fit(z_f_tr, z_m_tr)
    T_m2f = Ridge(alpha=args.alpha, fit_intercept=True).fit(z_m_tr, z_f_tr)

    z_f_te_to_m = T_f2m.predict(z_f_te)
    z_m_te_to_f = T_m2f.predict(z_m_te)

    # Decode
    with torch.no_grad():
        zm_te_t = torch.from_numpy(z_m_te).float().to(device)
        zf_te_t = torch.from_numpy(z_f_te).float().to(device)
        zf2m_t = torch.from_numpy(z_f_te_to_m.astype(np.float32)).float().to(device)
        zm2f_t = torch.from_numpy(z_m_te_to_f.astype(np.float32)).float().to(device)

        # male side
        p_m_native = denorm(dec_m(zm_te_t).cpu().numpy())
        p_m_swap = denorm(dec_m(zf_te_t).cpu().numpy())
        p_m_corr = denorm(dec_m(zf2m_t).cpu().numpy())
        # female side
        p_f_native = denorm(dec_f(zf_te_t).cpu().numpy())
        p_f_swap = denorm(dec_f(zm_te_t).cpu().numpy())
        p_f_corr = denorm(dec_f(zm2f_t).cpu().numpy())

    y_m_te = y_m[te_idx]
    y_f_te = y_f[te_idx]

    m_native = r2_mean(y_m_te, p_m_native)
    m_swap = r2_mean(y_m_te, p_m_swap)
    m_corr = r2_mean(y_m_te, p_m_corr)
    f_native = r2_mean(y_f_te, p_f_native)
    f_swap = r2_mean(y_f_te, p_f_swap)
    f_corr = r2_mean(y_f_te, p_f_corr)

    out = {
        "split_tag": args.split_tag,
        "split_seed": int(args.split_seed),
        "alpha": float(args.alpha),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "male_side": {
            "native_r2_mean": float(m_native),
            "raw_swap_r2_mean": float(m_swap),
            "corrected_swap_r2_mean": float(m_corr),
            "degradation_native_minus_swap": float(m_native - m_swap),
            "recovery_corrected_minus_swap": float(m_corr - m_swap),
            "recovery_ratio": recovery_ratio(m_native, m_swap, m_corr),
        },
        "female_side": {
            "native_r2_mean": float(f_native),
            "raw_swap_r2_mean": float(f_swap),
            "corrected_swap_r2_mean": float(f_corr),
            "degradation_native_minus_swap": float(f_native - f_swap),
            "recovery_corrected_minus_swap": float(f_corr - f_swap),
            "recovery_ratio": recovery_ratio(f_native, f_swap, f_corr),
        },
        "latent_map_diagnostics": {
            "f2m_r2_train": float(T_f2m.score(z_f_tr, z_m_tr)),
            "m2f_r2_train": float(T_m2f.score(z_m_tr, z_f_tr)),
            "f2m_coef_condition_number": cond_number(T_f2m.coef_),
            "m2f_coef_condition_number": cond_number(T_m2f.coef_),
        },
    }

    out_path = CVAE_CROSS_GENDER_DIR / f"swap_linear_map_recovery_{args.split_tag}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

