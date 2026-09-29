#!/usr/bin/env python3
"""
Layer1 Criteria A/B/C — reference baselines for "compared to what".

Criterion A (ΔR² ≈ 0):
  vs preregistered SESOI=0.05 encoder-falsifier threshold.
  Report: |ΔR²| ≪ SESOI; n seeds meeting (ΔR²>SESOI & p<0.05); relative to common R².

Criterion B (swap large):
  vs native paired R², and vs linear-latent-map recovery of the swap.
  "large" = collapse toward/below chance vs native; "coordinate-ish" if recovery_ratio ≈ 1.

Criterion C (CKA high):
  vs row-permutation null CKA(z_m, permute(z_f)).

Outputs:
  results/cvae_cross_gender/paper2_layer1_criteria_references.json
  (also merges `references` block into paper2_layer1_fixedsplit_seed_sweep.json)

Example:
  python3 code/analysis_paper2_layer1_criteria_references.py
  python3 code/analysis_paper2_layer1_criteria_references.py --seeds 42 43 44 45 46 47 48 49 50 51
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from config import (  # noqa: E402
    OASIS_SCORES_CSV,
    RESULTS_STEP1,
    CVAE_CROSS_GENDER_DIR,
    RANDOM_SEED,
)
from dataset import load_oasis_meta, add_theme_base  # noqa: E402

CVAE = CVAE_CROSS_GENDER_DIR
SWEEP = CVAE / "paper2_layer1_fixedsplit_seed_sweep.json"
OUT = CVAE / "paper2_layer1_criteria_references.json"
TARGET_M = ["valence_male", "arousal_male"]
TARGET_F = ["valence_female", "arousal_female"]
INPUT_DIM, HIDDEN, DROPOUT = 512, 128, 0.1
SESOI = 0.05  # preregistered encoder-falsifier threshold


def split_by_theme(df, seed=42, tr=0.6, va=0.2):
    df = add_theme_base(df)
    bases = df["theme_base"].unique()
    order = np.arange(len(bases))
    np.random.seed(seed)
    np.random.shuffle(order)
    ntr = max(1, int(len(bases) * tr))
    nva = max(0, int(len(bases) * va))
    tr_b = set(bases[order[:ntr]])
    te_b = set(bases[order[ntr + nva:]])
    tr_idx = np.where(df["theme_base"].isin(tr_b).values)[0]
    te_idx = np.where(df["theme_base"].isin(te_b).values)[0]
    return tr_idx, te_idx


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

        def forward(self, x):
            return self.net(x)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(latent_dim, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN, 2),
            )

        def forward(self, z):
            return self.net(z)

    return Encoder, Decoder


def linear_cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.sum((X.T @ Y) ** 2)
    hsic_xx = np.sum((X.T @ X) ** 2)
    hsic_yy = np.sum((Y.T @ Y) ** 2)
    denom = np.sqrt(max(hsic_xx, 1e-12) * max(hsic_yy, 1e-12))
    return float(hsic_xy / denom)


def load_z_and_preds(split_tag: str, split_seed: int, device):
    import torch

    X = np.load(RESULTS_STEP1 / "features_clip.npy").astype(np.float32)
    df = load_oasis_meta(OASIS_SCORES_CSV)
    valid = df[TARGET_M + TARGET_F].notna().all(axis=1).values
    df = df.loc[valid].reset_index(drop=True)
    X = X[valid]
    y_m = df[TARGET_M].to_numpy(np.float32)
    y_f = df[TARGET_F].to_numpy(np.float32)
    tr_idx, te_idx = split_by_theme(df, seed=split_seed)

    d = CVAE / "split_encoders" / split_tag
    ck_m = torch.load(d / "encoder_m.pt", map_location=device)
    ck_f = torch.load(d / "encoder_f.pt", map_location=device)
    lat = int(ck_m["config"]["latent_dim"])
    Encoder, Decoder = build_modules(torch, lat)
    enc_m, enc_f = Encoder().to(device), Encoder().to(device)
    dec_m, dec_f = Decoder().to(device), Decoder().to(device)
    enc_m.load_state_dict(ck_m["encoder"])
    dec_m.load_state_dict(ck_m["decoder"])
    enc_f.load_state_dict(ck_f["encoder"])
    dec_f.load_state_dict(ck_f["decoder"])
    for m in (enc_m, enc_f, dec_m, dec_f):
        m.eval()

    mean = np.asarray(ck_m["scaler_x_mean"], np.float32)
    scale = np.asarray(ck_m["scaler_x_scale"], np.float32)
    Xs = ((X - mean) / scale).astype(np.float32)

    with torch.no_grad():
        xt = torch.from_numpy(Xs[te_idx]).float().to(device)
        z_m = enc_m(xt).cpu().numpy()
        z_f = enc_f(xt).cpu().numpy()
        p_mm = denorm(dec_m(enc_m(xt)).cpu().numpy())
        p_ff = denorm(dec_f(enc_f(xt)).cpu().numpy())
        p_fm = denorm(dec_m(enc_f(xt)).cpu().numpy())
        p_mf = denorm(dec_f(enc_m(xt)).cpu().numpy())

    return {
        "y_m": y_m[te_idx],
        "y_f": y_f[te_idx],
        "z_m": z_m,
        "z_f": z_f,
        "p_mm": p_mm,
        "p_ff": p_ff,
        "p_fm": p_fm,
        "p_mf": p_mf,
        "tr_idx": tr_idx,
        "te_idx": te_idx,
        "Xs": Xs,
        "enc_m": enc_m,
        "enc_f": enc_f,
        "dec_m": dec_m,
        "dec_f": dec_f,
        "device": device,
    }


def r2_mean(y, p):
    return float((r2_score(y[:, 0], p[:, 0]) + r2_score(y[:, 1], p[:, 1])) / 2)


def cka_null(z_m, z_f, n_perm=2000, seed=0):
    rng = np.random.default_rng(seed)
    obs = linear_cka(z_m, z_f)
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        perm = rng.permutation(len(z_f))
        null[i] = linear_cka(z_m, z_f[perm])
    p = (1.0 + np.sum(null >= obs)) / (1.0 + n_perm)
    return {
        "observed": float(obs),
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null)),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q975": float(np.quantile(null, 0.975)),
        "p_obs_gt_null": float(p),
        "excess_over_null": float(obs - np.mean(null)),
        "n_perm": int(n_perm),
    }


def ensure_recovery(seed: int, *, alpha: float, split_seed: int) -> dict:
    tag = f"seed{seed}"
    path = CVAE / f"paper2_swap_linear_map_recovery_{tag}.json"
    need = True
    if path.exists():
        prev = json.loads(path.read_text(encoding="utf-8"))
        if abs(float(prev.get("alpha", -1)) - alpha) < 1e-12:
            need = False
    if need:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "code" / "analysis_paper2_swap_linear_map_recovery.py"),
                "--split-tag", tag,
                "--split-seed", str(split_seed),
                "--alpha", str(alpha),
            ],
            check=True,
            cwd=str(ROOT),
        )
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_seed(seed: int, sweep_row: dict, *, alpha: float, split_seed: int, n_perm_cka: int) -> dict:
    import torch

    tag = f"seed{seed}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = load_z_and_preds(tag, split_seed, device)
    cka = cka_null(pack["z_m"], pack["z_f"], n_perm=n_perm_cka, seed=split_seed + seed)
    rec = ensure_recovery(seed, alpha=alpha, split_seed=split_seed)

    m = rec["male_side"]
    f = rec["female_side"]
    delta = float(sweep_row["delta_r2"])
    p_a = float(sweep_row["p_split_gt_common"])

    # Common R² from per-seed delta snapshot if present
    dpath = CVAE / f"paper2_delta_r2_{tag}.json"
    common_r2 = None
    split_r2 = None
    if dpath.exists():
        djson = json.loads(dpath.read_text(encoding="utf-8"))
        common_r2 = float(djson["r2_common"]["overall_mean"])
        split_r2 = float(djson["r2_split"]["overall_mean"])

    return {
        "train_seed": int(seed),
        "criterion_a": {
            "delta_r2": delta,
            "p_split_gt_common": p_a,
            "sesoi": SESOI,
            "exceeds_sesoi": bool(delta > SESOI),
            "falsifies_encoder_null": bool(delta > SESOI and p_a < 0.05),
            "within_sesoi_abs": bool(abs(delta) < SESOI),
            "common_r2_overall": common_r2,
            "split_r2_overall": split_r2,
            "delta_over_common_r2": (None if common_r2 in (None, 0) else float(delta / common_r2)),
        },
        "criterion_b": {
            "male": {
                "native": float(m["native_r2_mean"]),
                "swap": float(m["raw_swap_r2_mean"]),
                "corrected": float(m["corrected_swap_r2_mean"]),
                "degradation": float(m["degradation_native_minus_swap"]),
                "recovery_ratio": float(m["recovery_ratio"]) if m["recovery_ratio"] is not None else None,
                "residual_gap_native_minus_corrected": float(m["native_r2_mean"] - m["corrected_swap_r2_mean"]),
            },
            "female": {
                "native": float(f["native_r2_mean"]),
                "swap": float(f["raw_swap_r2_mean"]),
                "corrected": float(f["corrected_swap_r2_mean"]),
                "degradation": float(f["degradation_native_minus_swap"]),
                "recovery_ratio": float(f["recovery_ratio"]) if f["recovery_ratio"] is not None else None,
                "residual_gap_native_minus_corrected": float(f["native_r2_mean"] - f["corrected_swap_r2_mean"]),
            },
            "ridge_alpha": float(alpha),
        },
        "criterion_c": cka,
    }


def summarize(rows: list[dict]) -> dict:
    d = np.asarray([r["criterion_a"]["delta_r2"] for r in rows], float)
    n = len(rows)
    falsify = sum(1 for r in rows if r["criterion_a"]["falsifies_encoder_null"])
    within = sum(1 for r in rows if r["criterion_a"]["within_sesoi_abs"])

    def side(key):
        nat = np.asarray([r["criterion_b"][key]["native"] for r in rows], float)
        sw = np.asarray([r["criterion_b"][key]["swap"] for r in rows], float)
        cor = np.asarray([r["criterion_b"][key]["corrected"] for r in rows], float)
        deg = nat - sw
        rr = np.asarray([r["criterion_b"][key]["recovery_ratio"] for r in rows], float)
        resid = nat - cor
        return {
            "native_mean": float(np.mean(nat)),
            "swap_mean": float(np.mean(sw)),
            "corrected_mean": float(np.mean(cor)),
            "degradation_mean": float(np.mean(deg)),
            "recovery_ratio_mean": float(np.mean(rr)),
            "recovery_ratio_std": float(np.std(rr, ddof=1)) if n > 1 else 0.0,
            "residual_gap_mean": float(np.mean(resid)),
            "native_values": nat.tolist(),
            "swap_values": sw.tolist(),
            "corrected_values": cor.tolist(),
            "recovery_ratio_values": rr.tolist(),
        }

    cka_obs = np.asarray([r["criterion_c"]["observed"] for r in rows], float)
    cka_null = np.asarray([r["criterion_c"]["null_mean"] for r in rows], float)
    cka_p = np.asarray([r["criterion_c"]["p_obs_gt_null"] for r in rows], float)

    return {
        "sesoi_delta_r2": SESOI,
        "criterion_a": {
            "reference": (
                f"Preregistered SESOI: encoder-locus falsifier requires "
                f"ΔR² > {SESOI} AND p<0.05 (shared vs split). "
                f"'Sufficiently no difference' = fails this falsifier / |ΔR²| < SESOI."
            ),
            "delta_r2_mean": float(np.mean(d)),
            "delta_r2_sem": float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "n_within_sesoi_abs": int(within),
            "n_falsifies_encoder_null": int(falsify),
            "frac_within_sesoi_abs": float(within / n),
            "frac_falsifies": float(falsify / n),
            "mean_abs_delta_over_sesoi": float(np.mean(np.abs(d)) / SESOI),
            "passes_near_zero_vs_sesoi": bool(falsify == 0 and within == n),
        },
        "criterion_b": {
            "reference": (
                "Compared to native paired R² (same gender decoder+encoder). "
                "Swap collapse is large iff native − swap ≫ 0. "
                "Coordinate mismatch (not wholly distinct geometry) if linear recovery "
                "restores most of the drop (recovery_ratio near 1)."
            ),
            "male": side("male"),
            "female": side("female"),
            "recovery_ratio_pooled_mean": float(np.mean([
                *[r["criterion_b"]["male"]["recovery_ratio"] for r in rows],
                *[r["criterion_b"]["female"]["recovery_ratio"] for r in rows],
            ])),
        },
        "criterion_c": {
            "reference": (
                "Compared to row-permutation null CKA(z_m, permute(z_f)). "
                "'High similarity' = observed ≫ null (and p_obs>null ≈ 0)."
            ),
            "cka_obs_mean": float(np.mean(cka_obs)),
            "cka_obs_min": float(np.min(cka_obs)),
            "cka_obs_max": float(np.max(cka_obs)),
            "cka_null_mean_of_means": float(np.mean(cka_null)),
            "cka_null_q025_mean": float(np.mean([r["criterion_c"]["null_q025"] for r in rows])),
            "cka_null_q975_mean": float(np.mean([r["criterion_c"]["null_q975"] for r in rows])),
            "n_sig_gt_null_p05": int(np.sum(cka_p < 0.05)),
            "excess_obs_minus_null_mean": float(np.mean(cka_obs - cka_null)),
            "obs_values": cka_obs.tolist(),
            "null_mean_values": cka_null.tolist(),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--split-seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--alpha", type=float, default=1.0, help="Ridge α for latent swap recovery")
    ap.add_argument("--n-perm-cka", type=int, default=2000)
    args = ap.parse_args()

    sweep = json.loads(SWEEP.read_text(encoding="utf-8"))
    by_seed = {int(r["train_seed"]): r for r in sweep["rows"]}
    seeds = args.seeds or sorted(by_seed.keys())

    rows = []
    for seed in seeds:
        if seed not in by_seed:
            raise SystemExit(f"seed {seed} missing from {SWEEP.name}")
        print(f"Analyzing reference baselines for seed {seed}...", flush=True)
        rows.append(
            analyze_seed(
                seed, by_seed[seed],
                alpha=args.alpha, split_seed=args.split_seed, n_perm_cka=args.n_perm_cka,
            )
        )

    summary = summarize(rows)
    payload = {
        "spec_version": "layer1_criteria_references_v1",
        "n_seeds": len(rows),
        "train_seeds": [int(r["train_seed"]) for r in rows],
        "split_seed": int(args.split_seed),
        "sesoi_delta_r2": SESOI,
        "ridge_alpha_recovery": float(args.alpha),
        "summary": summary,
        "rows": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Merge compact references into sweep for figure code
    sweep["references"] = {
        "sesoi_delta_r2": SESOI,
        "criterion_a": summary["criterion_a"],
        "criterion_b": {
            "reference": summary["criterion_b"]["reference"],
            "male_recovery_ratio_mean": summary["criterion_b"]["male"]["recovery_ratio_mean"],
            "female_recovery_ratio_mean": summary["criterion_b"]["female"]["recovery_ratio_mean"],
            "male_native_mean": summary["criterion_b"]["male"]["native_mean"],
            "male_swap_mean": summary["criterion_b"]["male"]["swap_mean"],
            "male_corrected_mean": summary["criterion_b"]["male"]["corrected_mean"],
            "female_native_mean": summary["criterion_b"]["female"]["native_mean"],
            "female_swap_mean": summary["criterion_b"]["female"]["swap_mean"],
            "female_corrected_mean": summary["criterion_b"]["female"]["corrected_mean"],
            "recovery_ratio_pooled_mean": summary["criterion_b"]["recovery_ratio_pooled_mean"],
        },
        "criterion_c": summary["criterion_c"],
        "source": str(OUT.relative_to(ROOT)),
    }
    SWEEP.write_text(json.dumps(sweep, indent=2), encoding="utf-8")

    print(json.dumps({
        "saved": str(OUT),
        "criterion_a": {
            "mean_dR2": summary["criterion_a"]["delta_r2_mean"],
            "frac_within_sesoi": summary["criterion_a"]["frac_within_sesoi_abs"],
            "n_falsifies": summary["criterion_a"]["n_falsifies_encoder_null"],
        },
        "criterion_b": {
            "recovery_ratio_pooled": summary["criterion_b"]["recovery_ratio_pooled_mean"],
            "male_deg": summary["criterion_b"]["male"]["degradation_mean"],
            "female_deg": summary["criterion_b"]["female"]["degradation_mean"],
        },
        "criterion_c": {
            "cka_obs": [summary["criterion_c"]["cka_obs_min"], summary["criterion_c"]["cka_obs_max"]],
            "cka_null_mean": summary["criterion_c"]["cka_null_mean_of_means"],
            "excess": summary["criterion_c"]["excess_obs_minus_null_mean"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
