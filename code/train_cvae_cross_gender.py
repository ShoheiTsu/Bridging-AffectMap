#!/usr/bin/env python3
"""
Cross-gender cVAE: shared Encoder to z, Decoder_m and Decoder_f.

Single-phase training: MSE(Dec_m(z), y_male) + MSE(Dec_f(z), y_female), equal weights.
Evaluation: R^2 for male and female scores from the same z.

Requires: PyTorch, features_clip.npy, oasis_scores.csv (male/female score columns).
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
)
from dataset import load_oasis_meta, add_theme_base

TARGET_MALE = ["valence_male", "arousal_male"]
TARGET_FEMALE = ["valence_female", "arousal_female"]

# モデル構造
INPUT_DIM = 512   # CLIP feature dim
LATENT_DIM = 64
HIDDEN = 128
DROPOUT = 0.1
# training hyperparameters
BATCH_SIZE = 64
EPOCHS_PHASE1 = 200   # 互換のため残す: joint 総 epoch = EPOCHS_PHASE1 + EPOCHS_PHASE2
EPOCHS_PHASE2 = 150
LR = 1e-3
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2      # test = 1 - train - val
SCALE_VA = (VALENCE_AROUSAL_SCALE_MIN, VALENCE_AROUSAL_SCALE_MAX)


def train_val_test_split_by_theme(df, train_ratio=0.6, val_ratio=0.2, random_state=42):
    """Split by theme_base into train / val / test (indices)."""
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


def normalize_y(y, scale_min=1.0, scale_max=7.0):
    return (y - scale_min) / (scale_max - scale_min)


def denormalize_y(y_norm, scale_min=1.0, scale_max=7.0):
    return y_norm * (scale_max - scale_min) + scale_min


def main():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("This script requires PyTorch. Install with: pip install torch")
        return

    CVAE_CROSS_GENDER_DIR.mkdir(parents=True, exist_ok=True)

    # データ
    X_clip = np.load(RESULTS_STEP1 / "features_clip.npy").astype(np.float32)
    df = load_oasis_meta(OASIS_SCORES_CSV)
    cols = TARGET_MALE + TARGET_FEMALE
    if not all(c in df.columns for c in cols):
        print("Missing valence_male, arousal_male, valence_female, arousal_female in CSV.")
        return
    valid = df[cols].notna().all(axis=1).values
    df = df.loc[valid].reset_index(drop=True)
    X_clip = X_clip[valid]
    y_male = df[TARGET_MALE].values.astype(np.float32)
    y_female = df[TARGET_FEMALE].values.astype(np.float32)
    n_samples = len(df)
    print(f"Samples: {n_samples}")

    train_idx, val_idx, test_idx = train_val_test_split_by_theme(
        df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, random_state=RANDOM_SEED
    )
    if len(test_idx) < 10:
        print("Too few test samples; adjust TRAIN_RATIO / VAL_RATIO.")
        return
    print(f"Train {len(train_idx)} / Val {len(val_idx)} / Test {len(test_idx)}")

    scaler_x = StandardScaler().fit(X_clip[train_idx])
    X = scaler_x.transform(X_clip)
    y_m_norm = normalize_y(y_male, *SCALE_VA)
    y_f_norm = normalize_y(y_female, *SCALE_VA)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)

    # モデル定義
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

    encoder = Encoder().to(device)
    decoder_m = Decoder().to(device)
    decoder_f = Decoder().to(device)
    mse = nn.MSELoss()
    n_epochs = EPOCHS_PHASE1 + EPOCHS_PHASE2
    opt = torch.optim.Adam(
        list(encoder.parameters())
        + list(decoder_m.parameters())
        + list(decoder_f.parameters()),
        lr=LR,
    )

    def to_tensor(a, device):
        return torch.from_numpy(a).float().to(device)

    # Joint: Encoder + Dec_m + Dec_f (both targets, same z)
    X_tr = to_tensor(X[train_idx], device)
    y_m_tr = to_tensor(y_m_norm[train_idx], device)
    y_f_tr = to_tensor(y_f_norm[train_idx], device)
    X_val = to_tensor(X[val_idx], device) if len(val_idx) > 0 else None
    y_m_val = to_tensor(y_m_norm[val_idx], device) if len(val_idx) > 0 else None
    y_f_val = to_tensor(y_f_norm[val_idx], device) if len(val_idx) > 0 else None

    n_tr = len(train_idx)
    for epoch in range(n_epochs):
        perm = np.random.permutation(n_tr)
        encoder.train()
        decoder_m.train()
        decoder_f.train()
        for start in range(0, n_tr, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_tr)
            idx = perm[start:end]
            xb = X_tr[idx]
            z = encoder(xb)
            pred_m = decoder_m(z)
            pred_f = decoder_f(z)
            loss = mse(pred_m, y_m_tr[idx]) + mse(pred_f, y_f_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (epoch + 1) % 50 == 0 and X_val is not None:
            encoder.eval()
            decoder_m.eval()
            decoder_f.eval()
            with torch.no_grad():
                z_v = encoder(X_val)
                v_loss = mse(decoder_m(z_v), y_m_val).item() + mse(
                    decoder_f(z_v), y_f_val
                ).item()
            print(f"  Joint epoch {epoch+1}/{n_epochs} val_loss={v_loss:.4f}")

    # Evaluate on held-out test themes
    encoder.eval()
    decoder_m.eval()
    decoder_f.eval()
    with torch.no_grad():
        X_te = to_tensor(X[test_idx], device)
        z_te = encoder(X_te)
        pred_m = decoder_m(z_te).cpu().numpy()
        pred_f = decoder_f(z_te).cpu().numpy()
    y_m_te = y_male[test_idx]
    y_f_te = y_female[test_idx]
    pred_m_scale = denormalize_y(pred_m, *SCALE_VA)
    pred_f_scale = denormalize_y(pred_f, *SCALE_VA)

    r2_male_v = r2_score(y_m_te[:, 0], pred_m_scale[:, 0])
    r2_male_a = r2_score(y_m_te[:, 1], pred_m_scale[:, 1])
    r2_male_mean = (r2_male_v + r2_male_a) / 2
    r2_female_v = r2_score(y_f_te[:, 0], pred_f_scale[:, 0])
    r2_female_a = r2_score(y_f_te[:, 1], pred_f_scale[:, 1])
    r2_female_mean = (r2_female_v + r2_female_a) / 2

    print("\n--- Test (theme-held-out) ---")
    print(f"Male:   R² V={r2_male_v:.4f} A={r2_male_a:.4f} mean={r2_male_mean:.4f}")
    print(f"Female: R² V={r2_female_v:.4f} A={r2_female_a:.4f} mean={r2_female_mean:.4f}")

    report = {
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": len(test_idx),
        "latent_dim": LATENT_DIM,
        "epochs_joint": n_epochs,
        "R2_male_valence": float(r2_male_v),
        "R2_male_arousal": float(r2_male_a),
        "R2_male_mean": float(r2_male_mean),
        "R2_female_valence": float(r2_female_v),
        "R2_female_arousal": float(r2_female_a),
        "R2_female_mean": float(r2_female_mean),
        # 下位互換（旧「女性 cross」列名）
        "R2_female_cross_valence": float(r2_female_v),
        "R2_female_cross_arousal": float(r2_female_a),
        "R2_female_cross_mean": float(r2_female_mean),
    }
    with open(CVAE_CROSS_GENDER_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {CVAE_CROSS_GENDER_DIR / 'report.json'}")

    # 重み保存（後で可視化などに利用可能）
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder_m": decoder_m.state_dict(),
        "decoder_f": decoder_f.state_dict(),
        "scaler_x_mean": scaler_x.mean_.tolist(),
        "scaler_x_scale": scaler_x.scale_.tolist(),
        "config": {"input_dim": INPUT_DIM, "latent_dim": LATENT_DIM, "hidden": HIDDEN},
    }, CVAE_CROSS_GENDER_DIR / "weights.pt")
    print(f"Weights saved to {CVAE_CROSS_GENDER_DIR / 'weights.pt'}")
    print("Done.")


if __name__ == "__main__":
    main()
