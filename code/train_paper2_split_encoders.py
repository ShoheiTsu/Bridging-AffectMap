#!/usr/bin/env python3
"""
Train split encoders from scratch for Paper2.

- Encoder_m + Decoder_m on male scores
- Encoder_f + Decoder_f on female scores

Outputs:
- results/cvae_cross_gender/split_encoders/encoder_m.pt
- results/cvae_cross_gender/split_encoders/encoder_f.pt
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
LATENT_DIM = 64
DROPOUT = 0.1


def normalize_y(y, scale_min=1.0, scale_max=7.0):
    return (y - scale_min) / (scale_max - scale_min)


def denormalize_y(y_norm, scale_min=1.0, scale_max=7.0):
    return y_norm * (scale_max - scale_min) + scale_min


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


def build_modules(torch):
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
                nn.Linear(HIDDEN, LATENT_DIM),
            )

        def forward(self, x):
            return self.net(x)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(LATENT_DIM, HIDDEN),
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


def train_one(X, y_norm, train_idx, val_idx, test_idx, epochs, batch_size, lr, seed):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Encoder, Decoder = build_modules(torch)
    enc = Encoder().to(device)
    dec = Decoder().to(device)
    mse = nn.MSELoss()
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)

    def to_t(a):
        return torch.from_numpy(a).float().to(device)

    X_tr = to_t(X[train_idx])
    y_tr = to_t(y_norm[train_idx])
    X_val = to_t(X[val_idx]) if len(val_idx) > 0 else None
    y_val = to_t(y_norm[val_idx]) if len(val_idx) > 0 else None

    best_val = float("inf")
    best_state = None
    n_tr = len(train_idx)
    for ep in range(epochs):
        enc.train()
        dec.train()
        perm = np.random.permutation(n_tr)
        for st in range(0, n_tr, batch_size):
            ed = min(st + batch_size, n_tr)
            idx = perm[st:ed]
            z = enc(X_tr[idx])
            p = dec(z)
            loss = mse(p, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if X_val is not None:
            enc.eval()
            dec.eval()
            with torch.no_grad():
                v = mse(dec(enc(X_val)), y_val).item()
            if v < best_val:
                best_val = v
                best_state = {
                    "encoder": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                    "decoder": {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()},
                }

    if best_state is not None:
        enc.load_state_dict(best_state["encoder"])
        dec.load_state_dict(best_state["decoder"])

    enc.eval()
    dec.eval()
    with torch.no_grad():
        X_te = to_t(X[test_idx])
        pred = dec(enc(X_te)).cpu().numpy()
    return enc, dec, pred


def main():
    parser = argparse.ArgumentParser(description="Train split encoders for Paper2")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="backward compat: equals train-seed when set")
    parser.add_argument("--train-seed", type=int, default=None, help="random seed for init/shuffle")
    parser.add_argument("--split-seed", type=int, default=RANDOM_SEED, help="fixed theme split seed")
    parser.add_argument("--out-tag", default="", help="subdir suffix under split_encoders (e.g., seed42)")
    args = parser.parse_args()

    train_seed = args.train_seed if args.train_seed is not None else args.seed
    out_dir = CVAE_CROSS_GENDER_DIR / "split_encoders"
    if args.out_tag:
        out_dir = out_dir / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(RESULTS_STEP1 / "features_clip.npy").astype(np.float32)
    df = load_oasis_meta(OASIS_SCORES_CSV)
    cols = TARGET_M + TARGET_F
    valid = df[cols].notna().all(axis=1).values
    df = df.loc[valid].reset_index(drop=True)
    X = X[valid]
    y_m = df[TARGET_M].to_numpy(np.float32)
    y_f = df[TARGET_F].to_numpy(np.float32)

    train_idx, val_idx, test_idx = train_val_test_split_by_theme(df, random_state=args.split_seed)
    scaler = StandardScaler().fit(X[train_idx])
    Xs = scaler.transform(X).astype(np.float32)
    y_m_n = normalize_y(y_m, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX).astype(np.float32)
    y_f_n = normalize_y(y_f, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX).astype(np.float32)

    enc_m, dec_m, pred_m_n = train_one(
        Xs, y_m_n, train_idx, val_idx, test_idx,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=train_seed
    )
    enc_f, dec_f, pred_f_n = train_one(
        Xs, y_f_n, train_idx, val_idx, test_idx,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=train_seed + 1
    )

    pred_m = denormalize_y(pred_m_n, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX)
    pred_f = denormalize_y(pred_f_n, VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX)
    y_m_te = y_m[test_idx]
    y_f_te = y_f[test_idx]
    r2_m_v = r2_score(y_m_te[:, 0], pred_m[:, 0])
    r2_m_a = r2_score(y_m_te[:, 1], pred_m[:, 1])
    r2_f_v = r2_score(y_f_te[:, 0], pred_f[:, 0])
    r2_f_a = r2_score(y_f_te[:, 1], pred_f[:, 1])

    import torch

    common_payload = {
        "scaler_x_mean": scaler.mean_.tolist(),
        "scaler_x_scale": scaler.scale_.tolist(),
        "config": {"input_dim": INPUT_DIM, "hidden": HIDDEN, "latent_dim": LATENT_DIM},
    }
    torch.save(
        {"encoder": enc_m.state_dict(), "decoder": dec_m.state_dict(), **common_payload},
        out_dir / "encoder_m.pt",
    )
    torch.save(
        {"encoder": enc_f.state_dict(), "decoder": dec_f.state_dict(), **common_payload},
        out_dir / "encoder_f.pt",
    )

    report = {
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "r2_male_valence": float(r2_m_v),
        "r2_male_arousal": float(r2_m_a),
        "r2_female_valence": float(r2_f_v),
        "r2_female_arousal": float(r2_f_a),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "train_seed": int(train_seed),
        "split_seed": int(args.split_seed),
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved {out_dir / 'encoder_m.pt'}")
    print(f"Saved {out_dir / 'encoder_f.pt'}")


if __name__ == "__main__":
    main()

