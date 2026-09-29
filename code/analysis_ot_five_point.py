#!/usr/bin/env python3
"""
Layer-2 mapping analysis: five-point mapping comparison (No/Global/Linear/OT/Oracle) on a fixed split.
"""
import sys
from pathlib import Path
import json
import argparse

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import ot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from config import OASIS_SCORES_CSV, RESULTS_STEP1, CVAE_CROSS_GENDER_DIR, RANDOM_SEED
from dataset import load_oasis_meta, add_theme_base

TARGET_M = ["valence_male", "arousal_male"]
TARGET_F = ["valence_female", "arousal_female"]
INPUT_DIM = 512
HIDDEN = 128
DROPOUT = 0.1


def split_by_theme(df, seed=42, tr=0.6, va=0.2):
    df = add_theme_base(df)
    bases = df["theme_base"].unique()
    order = np.arange(len(bases))
    np.random.seed(seed)
    np.random.shuffle(order)
    ntr = max(1, int(len(bases) * tr))
    nva = max(0, int(len(bases) * va))
    tr_b = set(bases[order[:ntr]])
    va_b = set(bases[order[ntr:ntr + nva]])
    te_b = set(bases[order[ntr + nva:]])
    tr_idx = np.where(df["theme_base"].isin(tr_b).values)[0]
    va_idx = np.where(df["theme_base"].isin(va_b).values)[0] if len(va_b) else np.array([], dtype=int)
    te_idx = np.where(df["theme_base"].isin(te_b).values)[0] if len(te_b) else np.array([], dtype=int)
    return tr_idx, va_idx, te_idx


def split_inner_train_val_by_theme(df, train_idx, seed=42, tr=0.8):
    """Outer train indices -> inner train/val split by theme_base."""
    df_tr = df.iloc[train_idx].reset_index(drop=True)
    df_tr = add_theme_base(df_tr)
    bases = df_tr["theme_base"].unique()
    order = np.arange(len(bases))
    np.random.seed(seed)
    np.random.shuffle(order)
    ntr = max(1, int(len(bases) * tr))
    tr_b = set(bases[order[:ntr]])
    va_b = set(bases[order[ntr:]])
    tr_local = np.where(df_tr["theme_base"].isin(tr_b).values)[0]
    va_local = np.where(df_tr["theme_base"].isin(va_b).values)[0]
    return train_idx[tr_local], train_idx[va_local]


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


def load_common_preds(Xs, ckpt_path, device):
    import torch
    ck = torch.load(ckpt_path, map_location=device)
    lat = int(ck["config"]["latent_dim"])
    Encoder, Decoder = build_modules(torch, lat)
    enc = Encoder().to(device); dm = Decoder().to(device); df = Decoder().to(device)
    enc.load_state_dict(ck["encoder"]); dm.load_state_dict(ck["decoder_m"]); df.load_state_dict(ck["decoder_f"])
    enc.eval(); dm.eval(); df.eval()
    with torch.no_grad():
        xt = torch.from_numpy(Xs).float().to(device)
        z = enc(xt)
        pm = denorm(dm(z).cpu().numpy())
        pf = denorm(df(z).cpu().numpy())
    return pm, pf


def r2_mean(y, p):
    rv = r2_score(y[:, 0], p[:, 0])
    ra = r2_score(y[:, 1], p[:, 1])
    return float(rv), float(ra), float((rv + ra) / 2)


def ot_barycentric_map(Xs, Xt, Xq, reg):
    # uniform weights
    a = np.full(len(Xs), 1.0 / len(Xs))
    b = np.full(len(Xt), 1.0 / len(Xt))
    M = ot.dist(Xs, Xt, metric="euclidean") ** 2
    G = ot.sinkhorn(a, b, M, reg=reg)
    # barycentric mapping from source support to target space
    mapped_src = (G @ Xt) / np.clip(G.sum(axis=1, keepdims=True), 1e-12, None)
    # map query by nearest source point
    Dq = ot.dist(Xq, Xs, metric="euclidean")
    nn = np.argmin(Dq, axis=1)
    return mapped_src[nn]


def bootstrap_r2_ci(y_true, pred, n_boot=2000, seed=42):
    """Paired bootstrap over test items -> R2_mean/V/A CI."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rm, rv, ra = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v, a, m = r2_mean(y_true[idx], pred[idx])
        rm.append(m); rv.append(v); ra.append(a)
    def ci(arr):
        arr = np.asarray(arr, float)
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
    return {
        "R2_mean_ci": ci(rm), "R2_valence_ci": ci(rv), "R2_arousal_ci": ci(ra),
        "R2_mean_boot_mean": float(np.mean(rm)),
        "R2_mean_boot": [float(v) for v in rm],
    }


def perm_test_delta_r2(y_true, p_a, p_b, n_perm=500, seed=42):
    rng = np.random.default_rng(seed)
    obs = r2_mean(y_true, p_b)[2] - r2_mean(y_true, p_a)[2]
    null = []
    n = len(y_true)
    for _ in range(n_perm):
        sw = rng.random(n) < 0.5
        pa = np.where(sw[:, None], p_b, p_a)
        pb = np.where(sw[:, None], p_a, p_b)
        null.append(r2_mean(y_true, pb)[2] - r2_mean(y_true, pa)[2])
    null = np.array(null, dtype=float)
    p = (1 + np.sum(null >= obs)) / (1 + len(null))
    return float(obs), float(p), float(np.mean(null)), float(np.std(null))


def evaluate_five_point_direction(
    df,
    tr_idx,
    te_idx,
    pred_src_all,
    pred_tgt_all,
    y_tgt_all,
    *,
    split_seed: int,
    n_perm: int,
    ot_reg: float,
    auto_reg: bool,
    ot_reg_grid: list[float],
):
    """Run the five-point comparison for one source→target direction."""
    pred_src_tr, pred_src_te = pred_src_all[tr_idx], pred_src_all[te_idx]
    pred_tgt_te = pred_tgt_all[te_idx]
    y_tgt_tr, y_tgt_te = y_tgt_all[tr_idx], y_tgt_all[te_idx]

    pred_no = pred_src_te.copy()
    delta = (y_tgt_tr - pred_src_tr).mean(axis=0, keepdims=True)
    pred_global = pred_src_te + delta

    lin = LinearRegression()
    lin.fit(pred_src_tr, y_tgt_tr)
    pred_linear = lin.predict(pred_src_te)

    selected_reg = float(ot_reg)
    reg_selection = None
    if auto_reg:
        in_tr_idx, in_va_idx = split_inner_train_val_by_theme(
            df, tr_idx, seed=split_seed + 1000, tr=0.8
        )
        pred_src_in_tr = pred_src_all[in_tr_idx]
        y_tgt_in_tr = y_tgt_all[in_tr_idx]
        pred_src_in_va = pred_src_all[in_va_idx]
        y_tgt_in_va = y_tgt_all[in_va_idx]
        reg_rows = []
        best_score = -np.inf
        for rg in ot_reg_grid:
            try:
                pred_va = ot_barycentric_map(
                    pred_src_in_tr, y_tgt_in_tr, pred_src_in_va, reg=rg
                )
                score = r2_mean(y_tgt_in_va, pred_va)[2]
            except Exception:
                score = -np.inf
            reg_rows.append(
                {"reg": float(rg), "val_r2_mean": float(score) if np.isfinite(score) else None}
            )
            if np.isfinite(score) and score > best_score:
                best_score = score
                selected_reg = float(rg)
        reg_selection = {
            "inner_train_n": int(len(in_tr_idx)),
            "inner_val_n": int(len(in_va_idx)),
            "grid_results": reg_rows,
            "selected_reg": float(selected_reg),
            "selected_val_r2_mean": float(best_score) if np.isfinite(best_score) else None,
        }

    pred_ot = ot_barycentric_map(pred_src_tr, y_tgt_tr, pred_src_te, reg=selected_reg)
    pred_oracle = pred_tgt_te.copy()

    methods = {
        "no_transport": pred_no,
        "global_shift": pred_global,
        "linear_shift": pred_linear,
        "ot": pred_ot,
        "oracle": pred_oracle,
    }
    scores = {}
    for k, p in methods.items():
        rv, ra, rm = r2_mean(y_tgt_te, p)
        boot = bootstrap_r2_ci(y_tgt_te, p, n_boot=2000, seed=split_seed)
        scores[k] = {"R2_valence": rv, "R2_arousal": ra, "R2_mean": rm, **boot}

    d_ot_vs_linear, p_ot_vs_linear, nmean, nstd = perm_test_delta_r2(
        y_tgt_te, pred_linear, pred_ot, n_perm=n_perm, seed=split_seed
    )
    comparison = {
        "delta_r2_mean": float(d_ot_vs_linear),
        "p_value_one_tailed_ot_gt_linear": float(p_ot_vs_linear),
        "null_mean": float(nmean),
        "null_std": float(nstd),
        "n_perm": int(n_perm),
    }
    return scores, comparison, float(selected_reg), reg_selection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--ot-reg", type=float, default=0.5)
    ap.add_argument("--auto-reg", action="store_true", help="select OT reg by inner validation on outer-train")
    ap.add_argument("--ot-reg-grid", type=str, default="0.01,0.05,0.1,0.2,0.5")
    args = ap.parse_args()

    common_ckpt = CVAE_CROSS_GENDER_DIR / "weights.pt"
    if not common_ckpt.exists():
        raise SystemExit(f"not found: {common_ckpt}")

    X = np.load(RESULTS_STEP1 / "features_clip.npy").astype(np.float32)
    df = load_oasis_meta(OASIS_SCORES_CSV)
    valid = df[TARGET_M + TARGET_F].notna().all(axis=1).values
    df = df.loc[valid].reset_index(drop=True)
    X = X[valid]
    y_m = df[TARGET_M].to_numpy(np.float32)
    y_f = df[TARGET_F].to_numpy(np.float32)
    tr_idx, _, te_idx = split_by_theme(df, seed=args.split_seed)
    scaler = StandardScaler().fit(X[tr_idx])
    Xs = scaler.transform(X).astype(np.float32)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_m_all, p_f_all = load_common_preds(Xs, common_ckpt, device)
    reg_grid = [float(x) for x in args.ot_reg_grid.split(",") if x.strip()]
    scores_f, comp_f, reg_f, regsel_f = evaluate_five_point_direction(
        df, tr_idx, te_idx, p_m_all, p_f_all, y_f,
        split_seed=args.split_seed,
        n_perm=args.n_perm,
        ot_reg=args.ot_reg,
        auto_reg=args.auto_reg,
        ot_reg_grid=reg_grid,
    )
    scores_m, comp_m, reg_m, regsel_m = evaluate_five_point_direction(
        df, tr_idx, te_idx, p_f_all, p_m_all, y_m,
        split_seed=args.split_seed,
        n_perm=args.n_perm,
        ot_reg=args.ot_reg,
        auto_reg=args.auto_reg,
        ot_reg_grid=reg_grid,
    )

    out = {
        "evaluation": "single_split_fixed",
        "split_seed": int(args.split_seed),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "ot_reg": float(reg_f),
        "ot_reg_selection": regsel_f,
        "scores_female_target": scores_f,
        "scores_male_target": scores_m,
        "primary_comparison_ot_vs_linear": comp_f,
        "primary_comparison_ot_vs_linear_male_target": comp_m,
        "ot_reg_male_target": float(reg_m),
        "ot_reg_selection_male_target": regsel_m,
    }
    out_path = CVAE_CROSS_GENDER_DIR / "ot_five_point_fixedsplit.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

