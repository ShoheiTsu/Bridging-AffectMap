#!/usr/bin/env python3
"""
Fig.2 Layer1 distribution outputs (bootstrap / label-shuffle null / CKA null / λ cluster).

Spec: doc/FIG2_LAYER1_DISTRIBUTIONS_SPEC.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from config import (  # noqa: E402
    CVAE_CROSS_GENDER_DIR,
    OASIS_SCORES_CSV,
    RANDOM_SEED,
    RESULTS_GENDER,
    RESULTS_STEP1,
    VALENCE_AROUSAL_SCALE_MAX,
    VALENCE_AROUSAL_SCALE_MIN,
)
from dataset import add_theme_base, load_oasis_meta  # noqa: E402

CVAE = CVAE_CROSS_GENDER_DIR
LAMBDA_CLUSTER_DIR = RESULTS_GENDER / "lambda_gender_diff_cluster_analysis"
LAMBDA_NPY = RESULTS_GENDER / "lambda_gender_diff_loto_clip.npy"
FIG_DIR = ROOT / "code" / "paper2_viz" / "figures"

FIG2_TRAIN_SEEDS = tuple(range(42, 52))
FIG2_SPLIT_SEED = 42
FIG2_SESOI = 0.05
FIG2_N_BOOT = 500
FIG2_LABEL_SHUFFLE_REF_SEED = 42
FIG2_N_LABEL_SHUFFLE_DEFAULT = 25
FIG2_TRAIN_EPOCHS = 250
FIG2_CKA_N_PERM = 2000

TARGET_M = ["valence_male", "arousal_male"]
TARGET_F = ["valence_female", "arousal_female"]
INPUT_DIM, HIDDEN, LATENT_DIM = 512, 128, 64
DROPOUT = 0.1
TRAIN_RATIO, VAL_RATIO = 0.6, 0.2

OUT_BOOT = CVAE / "paper2_layer1_fig2_criterion_a_boot.csv"
OUT_LABEL_NULL = CVAE / "paper2_layer1_fig2_criterion_a_label_shuffle_null.json"
OUT_LABEL_AC_NULL = CVAE / "paper2_layer1_fig2_label_shuffle_ac_null.json"
OUT_CKA_NPZ = CVAE / "paper2_layer1_fig2_criterion_c_cka_null_arrays.npz"
OUT_CKA_LABEL_NULL = CVAE / "paper2_layer1_fig2_criterion_c_label_shuffle_null.npz"
OUT_CRIT_B_TEST = CVAE / "paper2_layer1_fig2_criterion_b_per_image_test.csv"
OUT_LAMBDA_NPZ = LAMBDA_CLUSTER_DIR / "lambda_gender_diff_common_ref_fig2.npz"
OUT_LAMBDA_JSON = LAMBDA_CLUSTER_DIR / "lambda_gender_diff_common_ref_fig2.json"


def normalize_y(y: np.ndarray) -> np.ndarray:
    return (y - VALENCE_AROUSAL_SCALE_MIN) / (VALENCE_AROUSAL_SCALE_MAX - VALENCE_AROUSAL_SCALE_MIN)


def denormalize_y(y_norm: np.ndarray) -> np.ndarray:
    return y_norm * (VALENCE_AROUSAL_SCALE_MAX - VALENCE_AROUSAL_SCALE_MIN) + VALENCE_AROUSAL_SCALE_MIN


def theme_split(df, split_seed: int = FIG2_SPLIT_SEED):
    df = add_theme_base(df)
    bases = df["theme_base"].unique()
    order = np.arange(len(bases))
    np.random.seed(split_seed)
    np.random.shuffle(order)
    n_train = max(1, int(len(bases) * TRAIN_RATIO))
    n_val = max(0, int(len(bases) * VAL_RATIO))
    n_test = len(bases) - n_train - n_val
    if n_test < 1:
        n_val = max(0, n_val - 1)
        n_test = len(bases) - n_train - n_val
    tr_b = set(bases[order[:n_train]])
    va_b = set(bases[order[n_train:n_train + n_val]]) if n_val else set()
    te_b = set(bases[order[n_train + n_val:]]) if n_test else set()
    tr = np.where(df["theme_base"].isin(tr_b).values)[0]
    va = np.where(df["theme_base"].isin(va_b).values)[0] if va_b else np.array([], int)
    te = np.where(df["theme_base"].isin(te_b).values)[0] if te_b else np.array([], int)
    return tr, va, te


def compute_r2_summary(y_m, y_f, p_m, p_f) -> dict[str, float]:
    rmv = r2_score(y_m[:, 0], p_m[:, 0])
    rma = r2_score(y_m[:, 1], p_m[:, 1])
    rfv = r2_score(y_f[:, 0], p_f[:, 0])
    rfa = r2_score(y_f[:, 1], p_f[:, 1])
    return {
        "male_mean": float((rmv + rma) / 2),
        "female_mean": float((rfv + rfa) / 2),
        "overall_mean": float((rmv + rma + rfv + rfa) / 4),
    }


def load_oasis_pack(split_seed: int = FIG2_SPLIT_SEED):
    X = np.load(RESULTS_STEP1 / "features_clip.npy").astype(np.float32)
    df = load_oasis_meta(OASIS_SCORES_CSV)
    valid = df[TARGET_M + TARGET_F].notna().all(axis=1).values
    df = df.loc[valid].reset_index(drop=True)
    X = X[valid]
    y_m = df[TARGET_M].to_numpy(np.float32)
    y_f = df[TARGET_F].to_numpy(np.float32)
    tr, va, te = theme_split(df, split_seed)
    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X).astype(np.float32)
    return {
        "X": X, "Xs": Xs, "y_m": y_m, "y_f": y_f,
        "train_idx": tr, "val_idx": va, "test_idx": te,
        "scaler": scaler,
    }


def build_modules(torch):
    import torch.nn as nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(INPUT_DIM, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, LATENT_DIM),
            )

        def forward(self, x):
            return self.net(x)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(LATENT_DIM, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, 2),
            )

        def forward(self, z):
            return self.net(z)

    return Encoder, Decoder


def load_common_preds(X_te: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    ckpt = torch.load(CVAE / "weights.pt", map_location=device)
    lat = int(ckpt["config"]["latent_dim"])
    Encoder, Decoder = build_modules(torch)
    enc = Encoder().to(device)
    dm = Decoder().to(device)
    df = Decoder().to(device)
    enc.load_state_dict(ckpt["encoder"])
    dm.load_state_dict(ckpt["decoder_m"])
    df.load_state_dict(ckpt["decoder_f"])
    for m in (enc, dm, df):
        m.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_te).float().to(device)
        z = enc(xt)
        pm = denormalize_y(dm(z).cpu().numpy())
        pf = denormalize_y(df(z).cpu().numpy())
    return pm, pf


def load_split_preds(X_te: np.ndarray, tag: str, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    d = CVAE / "split_encoders" / tag
    ck_m = torch.load(d / "encoder_m.pt", map_location=device)
    ck_f = torch.load(d / "encoder_f.pt", map_location=device)
    lat = int(ck_m["config"]["latent_dim"])
    Encoder, Decoder = build_modules(torch)
    em, ef = Encoder().to(device), Encoder().to(device)
    dm, df = Decoder().to(device), Decoder().to(device)
    em.load_state_dict(ck_m["encoder"])
    dm.load_state_dict(ck_m["decoder"])
    ef.load_state_dict(ck_f["encoder"])
    df.load_state_dict(ck_f["decoder"])
    for m in (em, ef, dm, df):
        m.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_te).float().to(device)
        pm = denormalize_y(dm(em(xt)).cpu().numpy())
        pf = denormalize_y(df(ef(xt)).cpu().numpy())
    return pm, pf


def run_criterion_a_boot(seeds: tuple[int, ...], n_boot: int) -> pd.DataFrame:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = load_oasis_pack()
    te = pack["test_idx"]
    X_te = pack["Xs"][te]
    y_m = pack["y_m"][te]
    y_f = pack["y_f"][te]
    rows: list[dict] = []
    for seed in seeds:
        tag = f"seed{seed}"
        pm_c, pf_c = load_common_preds(X_te, device)
        pm_s, pf_s = load_split_preds(X_te, tag, device)
        rng = np.random.default_rng(FIG2_SPLIT_SEED + seed)
        n = len(te)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            r_c = compute_r2_summary(y_m[idx], y_f[idx], pm_c[idx], pf_c[idx])
            r_s = compute_r2_summary(y_m[idx], y_f[idx], pm_s[idx], pf_s[idx])
            rows.append({
                "train_seed": int(seed),
                "boot_idx": int(b),
                "delta_r2": float(r_s["overall_mean"] - r_c["overall_mean"]),
                "r2_common": float(r_c["overall_mean"]),
                "r2_split": float(r_s["overall_mean"]),
            })
        print(f"[A boot] seed {seed} done ({n_boot} draws)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_BOOT, index=False)
    return df


def train_split_pair(
    Xs: np.ndarray,
    y_m: np.ndarray,
    y_f: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    train_seed: int,
    epochs: int,
    batch_size: int = 64,
    lr: float = 1e-3,
    return_latents: bool = False,
):
    """Train split encoders; return test preds (denormalized), optionally (z_m, z_f)."""
    import torch
    import torch.nn as nn

    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Encoder, Decoder = build_modules(torch)
    y_m_n = normalize_y(y_m).astype(np.float32)
    y_f_n = normalize_y(y_f).astype(np.float32)

    def _train_one(y_norm):
        enc = Encoder().to(device)
        dec = Decoder().to(device)
        mse = nn.MSELoss()
        opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
        X_tr = torch.from_numpy(Xs[train_idx]).float().to(device)
        y_tr = torch.from_numpy(y_norm[train_idx]).float().to(device)
        X_va = torch.from_numpy(Xs[val_idx]).float().to(device) if len(val_idx) else None
        y_va = torch.from_numpy(y_norm[val_idx]).float().to(device) if len(val_idx) else None
        best, best_val = None, float("inf")
        n_tr = len(train_idx)
        for _ in range(epochs):
            enc.train()
            dec.train()
            perm = np.random.permutation(n_tr)
            for st in range(0, n_tr, batch_size):
                sl = perm[st:st + batch_size]
                z = enc(X_tr[sl])
                loss = mse(dec(z), y_tr[sl])
                opt.zero_grad()
                loss.backward()
                opt.step()
            if X_va is not None:
                enc.eval()
                dec.eval()
                with torch.no_grad():
                    v = mse(dec(enc(X_va)), y_va).item()
                if v < best_val:
                    best_val = v
                    best = {
                        "enc": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                        "dec": {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()},
                    }
        if best is not None:
            enc.load_state_dict(best["enc"])
            dec.load_state_dict(best["dec"])
        enc.eval()
        dec.eval()
        with torch.no_grad():
            xt = torch.from_numpy(Xs[test_idx]).float().to(device)
            z = enc(xt)
            pred = denormalize_y(dec(z).cpu().numpy())
            z_np = z.cpu().numpy()
        return pred, z_np

    pm, zm = _train_one(y_m_n)
    pf, zf = _train_one(y_f_n)
    if return_latents:
        return pm, pf, zm, zf
    return pm, pf


def load_split_latents(X_te: np.ndarray, tag: str, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    d = CVAE / "split_encoders" / tag
    ck_m = torch.load(d / "encoder_m.pt", map_location=device)
    ck_f = torch.load(d / "encoder_f.pt", map_location=device)
    Encoder, _ = build_modules(torch)
    em, ef = Encoder().to(device), Encoder().to(device)
    em.load_state_dict(ck_m["encoder"])
    ef.load_state_dict(ck_f["encoder"])
    em.eval()
    ef.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X_te).float().to(device)
        zm = em(xt).cpu().numpy()
        zf = ef(xt).cpu().numpy()
    return zm, zf


def run_label_shuffle_ac_null(
    *,
    ref_seed: int = FIG2_LABEL_SHUFFLE_REF_SEED,
    n_perm: int,
    epochs: int,
) -> dict:
    """Joint Criterion A (ΔR²) + Criterion C (CKA) label-shuffle null."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = load_oasis_pack()
    te = pack["test_idx"]
    X_te = pack["Xs"][te]
    y_m_te = pack["y_m"][te]
    y_f_te = pack["y_f"][te]
    pm_c, pf_c = load_common_preds(X_te, device)
    r_common = compute_r2_summary(y_m_te, y_f_te, pm_c, pf_c)["overall_mean"]

    tag = f"seed{ref_seed}"
    pm_obs, pf_obs = load_split_preds(X_te, tag, device)
    delta_obs = compute_r2_summary(y_m_te, y_f_te, pm_obs, pf_obs)["overall_mean"] - r_common
    zm_obs, zf_obs = load_split_latents(X_te, tag, device)
    cka_obs = linear_cka(zm_obs, zf_obs)

    null_deltas: list[float] = []
    null_ckas: list[float] = []
    tr = pack["train_idx"]
    for p in range(n_perm):
        rng = np.random.default_rng(FIG2_SPLIT_SEED + 9000 + p)
        y_m_sh = pack["y_m"].copy()
        y_f_sh = pack["y_f"].copy()
        perm_m = rng.permutation(len(tr))
        perm_f = rng.permutation(len(tr))
        y_m_sh[tr] = pack["y_m"][tr][perm_m]
        y_f_sh[tr] = pack["y_f"][tr][perm_f]
        pm_s, pf_s, zm_s, zf_s = train_split_pair(
            pack["Xs"], y_m_sh, y_f_sh, tr, pack["val_idx"], te,
            train_seed=ref_seed + p, epochs=epochs, return_latents=True,
        )
        delta = compute_r2_summary(y_m_te, y_f_te, pm_s, pf_s)["overall_mean"] - r_common
        cka = linear_cka(zm_s, zf_s)
        null_deltas.append(float(delta))
        null_ckas.append(float(cka))
        print(
            f"[A/C label-null] perm {p + 1}/{n_perm}: ΔR²={delta:+.4f}, CKA={cka:.4f}",
            flush=True,
        )

    arr = np.asarray(null_deltas, float)
    cka_arr = np.asarray(null_ckas, float)
    # For C: obs high → p = fraction of null as large as obs
    p_cka = float((1.0 + np.sum(cka_arr >= cka_obs)) / (1.0 + len(cka_arr)))

    payload = {
        "estimand": "train_label_shuffle_split_encoders",
        "description": (
            "Independently permute male/female training labels on train_idx, "
            "retrain split encoders; record ΔR² (vs common) and linear CKA(z_m,z_f) "
            "on held-out test. Shared null for Criterion A and C."
        ),
        "train_seed_reference": int(ref_seed),
        "split_seed": int(FIG2_SPLIT_SEED),
        "n_perm": int(n_perm),
        "epochs": int(epochs),
        "r2_common_overall": float(r_common),
        "delta_r2_observed": float(delta_obs),
        "null_delta_r2": arr.tolist(),
        "null_summary": {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "q025": float(np.quantile(arr, 0.025)),
            "q975": float(np.quantile(arr, 0.975)),
            "median": float(np.median(arr)),
        },
        "p_obs_le_null": float(np.mean(arr >= delta_obs)),
        "cka_observed": float(cka_obs),
        "null_cka": cka_arr.tolist(),
        "cka_null_summary": {
            "mean": float(np.mean(cka_arr)),
            "std": float(np.std(cka_arr, ddof=1)) if len(cka_arr) > 1 else 0.0,
            "q025": float(np.quantile(cka_arr, 0.025)),
            "q975": float(np.quantile(cka_arr, 0.975)),
            "median": float(np.median(cka_arr)),
        },
        "p_obs_cka_gt_null": float(p_cka),
    }
    # Backward-compatible A-only JSON (+ CKA fields)
    OUT_LABEL_NULL.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_LABEL_AC_NULL.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez(
        OUT_CKA_LABEL_NULL,
        cka_obs=np.array([cka_obs], float),
        null_cka=cka_arr,
        n_perm=np.array([n_perm]),
        train_seed_reference=np.array([ref_seed]),
    )
    print(f"[A/C label-null] saved {OUT_LABEL_AC_NULL.name}", flush=True)
    return payload


def run_criterion_a_label_shuffle_null(
    *,
    ref_seed: int = FIG2_LABEL_SHUFFLE_REF_SEED,
    n_perm: int,
    epochs: int,
) -> dict:
    """Backward-compatible alias → joint A/C label-shuffle null."""
    return run_label_shuffle_ac_null(ref_seed=ref_seed, n_perm=n_perm, epochs=epochs)

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.sum((X.T @ Y) ** 2)
    hsic_xx = np.sum((X.T @ X) ** 2)
    hsic_yy = np.sum((Y.T @ Y) ** 2)
    return float(hsic_xy / np.sqrt(max(hsic_xx, 1e-12) * max(hsic_yy, 1e-12)))


def run_criterion_c_null_arrays(seeds: tuple[int, ...], n_perm: int) -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_list, null_blocks = [], []
    for seed in seeds:
        tag = f"seed{seed}"
        d = CVAE / "split_encoders" / tag
        ck_m = torch.load(d / "encoder_m.pt", map_location=device)
        ck_f = torch.load(d / "encoder_f.pt", map_location=device)
        pack = load_oasis_pack()
        te = pack["test_idx"]
        X_te = pack["Xs"][te]
        Encoder, _ = build_modules(torch)
        em, ef = Encoder().to(device), Encoder().to(device)
        em.load_state_dict(ck_m["encoder"])
        ef.load_state_dict(ck_f["encoder"])
        em.eval()
        ef.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X_te).float().to(device)
            z_m = em(xt).cpu().numpy()
            z_f = ef(xt).cpu().numpy()
        obs = linear_cka(z_m, z_f)
        rng = np.random.default_rng(FIG2_SPLIT_SEED + seed)
        null = np.empty(n_perm, float)
        for i in range(n_perm):
            null[i] = linear_cka(z_m, z_f[rng.permutation(len(z_f))])
        obs_list.append(obs)
        null_blocks.append(null)
        print(f"[C CKA] seed {seed}: obs={obs:.4f}, null_mean={null.mean():.4f}", flush=True)

    obs_a = np.asarray(obs_list, float)
    null_a = np.asarray(null_blocks, float)
    np.savez(
        OUT_CKA_NPZ,
        seeds=np.asarray(seeds, int),
        cka_obs=obs_a,
        cka_null=null_a,
        null_mean=null_a.mean(axis=1),
        null_q025=np.quantile(null_a, 0.025, axis=1),
        null_q975=np.quantile(null_a, 0.975, axis=1),
        n_perm=int(n_perm),
    )


def _load_cluster_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_cross_within_bias_clusters",
        FIG_DIR / "analyze_cross_within_bias_clusters.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_cross_within_bias_clusters"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def local_mean_grid(v_ref, a_ref, values, g, radius, min_n):
    n_g = len(g)
    out = np.full((n_g, n_g), np.nan)
    r2 = radius * radius
    for i in range(n_g):
        for j in range(n_g):
            vc, ac = g[j], g[i]
            m = (v_ref - vc) ** 2 + (a_ref - ac) ** 2 <= r2
            if int(m.sum()) < min_n:
                continue
            out[i, j] = float(np.mean(values[m]))
    return out


def run_lambda_common_ref(
    *,
    grid_step: float = 0.1,
    radius: float = 0.5,
    min_n: int = 5,
    n_perm: int = 2000,
    t_threshold: float = 2.0,
    alpha: float = 0.05,
) -> dict:
    cluster = _load_cluster_module()
    data = np.load(LAMBDA_NPY, allow_pickle=True).item()
    g = np.arange(VALENCE_AROUSAL_SCALE_MIN + grid_step / 2, VALENCE_AROUSAL_SCALE_MAX, grid_step)
    v_ref = data["v_common"]
    a_ref = data["a_common"]
    extent = [float(g[0] - grid_step / 2), float(g[-1] + grid_step / 2)] * 2
    rng = np.random.default_rng(RANDOM_SEED)

    out_npz: dict[str, np.ndarray] = {
        "grid": g.astype(float),
        "grid_step": np.array([grid_step]),
        "extent": np.asarray(extent, float),
        "n_images": np.array([int(data["n"])]),
    }
    summary: dict = {"n_images": int(data["n"]), "panels": []}

    for dim_key, delta_key in [("valence", "delta_v"), ("arousal", "delta_a")]:
        delta = np.asarray(data[delta_key], float)
        masks, n_map = cluster.build_cell_masks(v_ref, a_ref, g, radius, min_n)
        res = cluster.permutation_cluster_correction(
            delta, masks, n_map, g, min_n, rng,
            n_perm=n_perm, t_threshold=t_threshold, alpha=alpha,
        )
        grid_mean = local_mean_grid(v_ref, a_ref, delta, g, radius, min_n)
        out_npz[f"mean_map_{dim_key}"] = grid_mean
        if res.get("ok"):
            out_npz[f"labels_sig_{dim_key}"] = res["labels_sig"].astype(np.int8)
            sig_clusters = [c for c in res["clusters"] if c.get("significant")]
            summary["panels"].append({
                "dimension": dim_key,
                "ref": "common",
                "p_cluster_global": float(res["p_cluster"]),
                "n_sig_clusters": int(res["n_sig_clusters"]),
                "significant_clusters": [
                    {
                        "cluster_id": int(c["cluster_id"]),
                        "n_cells": int(c["n_cells"]),
                        "centroid_valence": float(c["centroid_valence"]),
                        "centroid_arousal": float(c["centroid_arousal"]),
                        "mean_delta": float(c["mean_delta_l2"]),
                    }
                    for c in sig_clusters
                ],
            })
        else:
            out_npz[f"labels_sig_{dim_key}"] = np.zeros_like(grid_mean, dtype=np.int8)
            summary["panels"].append({"dimension": dim_key, "ref": "common", "ok": False})

    LAMBDA_CLUSTER_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_LAMBDA_NPZ, **out_npz)
    OUT_LAMBDA_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[D λ] saved {OUT_LAMBDA_NPZ.name}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Fig.2 Layer1 distribution outputs")
    ap.add_argument(
        "--part",
        choices=["all", "a_boot", "a_null", "ac_null", "c_null", "lambda"],
        default="all",
    )
    ap.add_argument("--seeds", type=int, nargs="*", default=list(FIG2_TRAIN_SEEDS))
    ap.add_argument("--n-boot", type=int, default=FIG2_N_BOOT)
    ap.add_argument("--n-label-shuffle", type=int, default=FIG2_N_LABEL_SHUFFLE_DEFAULT)
    ap.add_argument("--epochs-null", type=int, default=FIG2_TRAIN_EPOCHS)
    ap.add_argument("--n-cka-perm", type=int, default=FIG2_CKA_N_PERM)
    args = ap.parse_args()
    seeds = tuple(args.seeds)

    if args.part in ("all", "a_boot"):
        run_criterion_a_boot(seeds, args.n_boot)
    if args.part in ("all", "a_null", "ac_null"):
        run_label_shuffle_ac_null(
            n_perm=args.n_label_shuffle, epochs=args.epochs_null,
        )
    if args.part in ("all", "c_null"):
        run_criterion_c_null_arrays(seeds, args.n_cka_perm)
    if args.part in ("all", "lambda"):
        run_lambda_common_ref()

    print(json.dumps({"status": "ok", "part": args.part, "out_dir": str(CVAE)}, indent=2))


if __name__ == "__main__":
    main()
