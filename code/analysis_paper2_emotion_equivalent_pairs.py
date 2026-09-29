#!/usr/bin/env python3
"""
Paper2: search for emotion-equivalent image pairs.

Female-target definition:
  For each test image i with female truth y_f(i), find j* whose male-decoder
  prediction Dec_m(z_j) is closest among candidates.

Outputs:
- results/cvae_cross_gender/paper2_emotion_equivalent_pairs_fixedsplit.csv
- results/cvae_cross_gender/paper2_emotion_equivalent_pairs_fixedsplit_topN.csv
"""
import sys
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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
    np.random.seed(seed)
    np.random.shuffle(order)
    ntr = max(1, int(len(bases) * tr))
    nva = max(0, int(len(bases) * va))
    tr_b = set(bases[order[:ntr]])
    te_b = set(bases[order[ntr + nva:]])
    tr_idx = np.where(df["theme_base"].isin(tr_b).values)[0]
    te_idx = np.where(df["theme_base"].isin(te_b).values)[0]
    return tr_idx, te_idx


def theme_kfold_indices(df: pd.DataFrame, *, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Theme-blocked K-fold: each image appears in exactly one test fold."""
    df = add_theme_base(df)
    bases = np.asarray(sorted(df["theme_base"].unique()))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bases))
    bases = bases[order]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for f in range(n_folds):
        te_b = set(bases[f::n_folds].tolist())
        te_idx = np.where(df["theme_base"].isin(te_b).values)[0]
        tr_idx = np.where(~df["theme_base"].isin(te_b).values)[0]
        if len(te_idx) == 0 or len(tr_idx) == 0:
            continue
        folds.append((tr_idx, te_idx))
    return folds


def build_1nn_rows(
    df: pd.DataFrame,
    *,
    te_idx: np.ndarray,
    pool_idx: np.ndarray,
    y_f: np.ndarray,
    p_m_all: np.ndarray,
    p_f_all: np.ndarray,
    fold: int | None = None,
) -> list[dict]:
    """Canonical female-target 1-NN rows (same schema as fixedsplit CSV)."""
    y_f_te = y_f[te_idx]
    P_pool = p_m_all[pool_idx]
    D = np.sqrt(((y_f_te[:, None, :] - P_pool[None, :, :]) ** 2).sum(axis=2))
    pool_pos = {idx: pos for pos, idx in enumerate(pool_idx.tolist())}
    for r, i_idx in enumerate(te_idx.tolist()):
        if i_idx in pool_pos:
            D[r, pool_pos[i_idx]] = np.inf
    best_pos = np.argmin(D, axis=1)
    best_dist = D[np.arange(len(te_idx)), best_pos]
    best_idx = pool_idx[best_pos]

    rows: list[dict] = []
    for k, i_idx in enumerate(te_idx.tolist()):
        j_idx = int(best_idx[k])
        row = {
            "i_idx": int(i_idx),
            "j_idx": j_idx,
            "distance_l2": float(best_dist[k]),
            "i_image_id": str(df.loc[i_idx, "image_id"]) if "image_id" in df.columns else "",
            "j_image_id": str(df.loc[j_idx, "image_id"]) if "image_id" in df.columns else "",
            "i_filename": str(df.loc[i_idx, "image_filename"]) if "image_filename" in df.columns else "",
            "j_filename": str(df.loc[j_idx, "image_filename"]) if "image_filename" in df.columns else "",
            "i_theme": str(df.loc[i_idx, "theme"]) if "theme" in df.columns else "",
            "j_theme": str(df.loc[j_idx, "theme"]) if "theme" in df.columns else "",
            "i_category": str(df.loc[i_idx, "category"]) if "category" in df.columns else "",
            "j_category": str(df.loc[j_idx, "category"]) if "category" in df.columns else "",
            "y_f_i_valence": float(y_f[i_idx, 0]),
            "y_f_i_arousal": float(y_f[i_idx, 1]),
            "pred_m_j_valence": float(p_m_all[j_idx, 0]),
            "pred_m_j_arousal": float(p_m_all[j_idx, 1]),
            "pred_f_i_valence": float(p_f_all[i_idx, 0]),
            "pred_f_i_arousal": float(p_f_all[i_idx, 1]),
            "same_category": bool(
                ("category" in df.columns) and (df.loc[i_idx, "category"] == df.loc[j_idx, "category"])
            ),
            "same_theme_base": bool(
                ("theme_base" in df.columns) and (df.loc[i_idx, "theme_base"] == df.loc[j_idx, "theme_base"])
            ),
        }
        if fold is not None:
            row["fold"] = int(fold)
        rows.append(row)
    return rows


def same_category_permutation_p(
    i_cats: np.ndarray,
    j_cats: np.ndarray,
    *,
    n_perm: int = 5000,
    seed: int = 0,
) -> dict:
    """Permutation null for same-category rate (shuffle target labels)."""
    obs_same = float(np.mean(i_cats == j_cats))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        null[k] = float(np.mean(i_cats == rng.permutation(j_cats)))
    # one-sided: is same-category rate above chance?
    p_above = float((np.sum(null >= obs_same) + 1) / (n_perm + 1))
    return {
        "observed_same_category_rate": obs_same,
        "observed_cross_category_rate": 1.0 - obs_same,
        "permutation_null_same_category_mean": float(null.mean()),
        "permutation_null_same_category_std": float(null.std()),
        "permutation_p_same_category_above_chance": p_above,
        "n_perm": int(n_perm),
    }


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
    enc = Encoder().to(device)
    dm = Decoder().to(device)
    df = Decoder().to(device)
    enc.load_state_dict(ck["encoder"])
    dm.load_state_dict(ck["decoder_m"])
    df.load_state_dict(ck["decoder_f"])
    enc.eval(); dm.eval(); df.eval()
    with torch.no_grad():
        xt = torch.from_numpy(Xs).float().to(device)
        z = enc(xt)
        pm = denorm(dm(z).cpu().numpy())
        pf = denorm(df(z).cpu().numpy())
    return pm, pf


def safe_col(df, name, fallback=""):
    return df[name] if name in df.columns else pd.Series([fallback] * len(df))


def build_knn_rows(
    df: pd.DataFrame,
    *,
    query_idx: np.ndarray,
    pool_idx: np.ndarray,
    y_query: np.ndarray,
    p_pool: np.ndarray,
    direction: str,
    k: int,
    pool: str,
) -> pd.DataFrame:
    """For each query image, attach top-k pool matches by L2 in VA prediction space.

    FtoM: hub = female true y_f; match ranked by ||Dec_m(z)-y_f||
    MtoF: hub = male true y_m; match ranked by ||Dec_f(z)-y_m||
    """
    D = np.sqrt(((y_query[:, None, :] - p_pool[None, :, :]) ** 2).sum(axis=2))
    if pool == "all":
        pool_pos = {idx: pos for pos, idx in enumerate(pool_idx.tolist())}
        for r, q_idx in enumerate(query_idx.tolist()):
            if q_idx in pool_pos:
                D[r, pool_pos[q_idx]] = np.inf

    k_eff = min(k, D.shape[1])
    part = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]
    rows: list[dict] = []
    for r, q_idx in enumerate(query_idx.tolist()):
        order = part[r][np.argsort(D[r, part[r]])]
        for rank, pos in enumerate(order, start=1):
            m_idx = int(pool_idx[int(pos)])
            q = int(q_idx)
            if direction == "FtoM":
                i_idx, j_idx = q, m_idx  # i=female hub, j=male match
            else:
                i_idx, j_idx = m_idx, q  # i=female match, j=male hub
            rows.append({
                "direction": direction,
                "rank": int(rank),
                "hub_idx": q,
                "match_idx": m_idx,
                "distance_l2": float(D[r, int(pos)]),
                "i_idx": i_idx,
                "j_idx": j_idx,
                "i_image_id": str(df.loc[i_idx, "image_id"]),
                "j_image_id": str(df.loc[j_idx, "image_id"]),
                "hub_image_id": str(df.loc[q, "image_id"]),
                "match_image_id": str(df.loc[m_idx, "image_id"]),
                "hub_theme": str(df.loc[q, "theme"]) if "theme" in df.columns else "",
                "match_theme": str(df.loc[m_idx, "theme"]) if "theme" in df.columns else "",
                "hub_category": str(df.loc[q, "category"]) if "category" in df.columns else "",
                "match_category": str(df.loc[m_idx, "category"]) if "category" in df.columns else "",
                "i_theme": str(df.loc[i_idx, "theme"]) if "theme" in df.columns else "",
                "j_theme": str(df.loc[j_idx, "theme"]) if "theme" in df.columns else "",
                "i_category": str(df.loc[i_idx, "category"]) if "category" in df.columns else "",
                "j_category": str(df.loc[j_idx, "category"]) if "category" in df.columns else "",
                "hub_valence": float(y_query[r, 0]),
                "hub_arousal": float(y_query[r, 1]),
                "match_pred_valence": float(p_pool[int(pos), 0]),
                "match_pred_arousal": float(p_pool[int(pos), 1]),
                "y_f_i_valence": float(df.loc[i_idx, "valence_female"]),
                "y_f_i_arousal": float(df.loc[i_idx, "arousal_female"]),
                "y_m_j_valence": float(df.loc[j_idx, "valence_male"]),
                "y_m_j_arousal": float(df.loc[j_idx, "arousal_male"]),
                "same_category": bool(df.loc[q, "category"] == df.loc[m_idx, "category"]) if "category" in df.columns else False,
                "same_theme_base": bool(
                    ("theme_base" in df.columns)
                    and (df.loc[q, "theme_base"] == df.loc[m_idx, "theme_base"])
                ),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixedsplit", "themecv"], default="fixedsplit")
    ap.add_argument("--split-seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--n-folds", type=int, default=5, help="theme-blocked CV folds (--mode themecv)")
    ap.add_argument("--pool", choices=["train", "all"], default="all", help="candidate j pool (fixedsplit)")
    ap.add_argument("--cv-pool", choices=["train", "all"], default="train",
                    help="match pool in themecv: train=other folds only (default), all=all except self")
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--knn-k", type=int, default=5, help="top-k neighbors per hub (both directions)")
    ap.add_argument("--n-perm", type=int, default=5000, help="permutation null for same-category rate")
    args = ap.parse_args()

    ckpt = CVAE_CROSS_GENDER_DIR / "weights.pt"
    if not ckpt.exists():
        raise SystemExit(f"not found: {ckpt}")

    X = np.load(RESULTS_STEP1 / "features_clip.npy").astype(np.float32)
    df = load_oasis_meta(OASIS_SCORES_CSV)
    valid = df[TARGET_M + TARGET_F].notna().all(axis=1).values
    df = df.loc[valid].reset_index(drop=True)
    X = X[valid]
    y_f = df[TARGET_F].to_numpy(np.float32)
    y_m = df[TARGET_M].to_numpy(np.float32)
    df = add_theme_base(df)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = CVAE_CROSS_GENDER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "themecv":
        folds = theme_kfold_indices(df, n_folds=args.n_folds, seed=args.split_seed)
        all_rows: list[dict] = []
        knn_parts: list[pd.DataFrame] = []
        fold_sizes: list[dict] = []
        for f, (tr_idx, te_idx) in enumerate(folds):
            scaler = StandardScaler().fit(X[tr_idx])
            Xs = scaler.transform(X).astype(np.float32)
            p_m_all, p_f_all = load_common_preds(Xs, ckpt, device)
            pool_idx = tr_idx if args.cv_pool == "train" else np.arange(len(df))
            rows = build_1nn_rows(
                df, te_idx=te_idx, pool_idx=pool_idx,
                y_f=y_f, p_m_all=p_m_all, p_f_all=p_f_all, fold=f,
            )
            all_rows.extend(rows)
            knn_ftom = build_knn_rows(
                df,
                query_idx=te_idx,
                pool_idx=pool_idx,
                y_query=y_f[te_idx],
                p_pool=p_m_all[pool_idx],
                direction="FtoM",
                k=args.knn_k,
                pool=args.cv_pool,
            )
            knn_mtof = build_knn_rows(
                df,
                query_idx=te_idx,
                pool_idx=pool_idx,
                y_query=y_m[te_idx],
                p_pool=p_f_all[pool_idx],
                direction="MtoF",
                k=args.knn_k,
                pool=args.cv_pool,
            )
            knn_ftom = knn_ftom.assign(fold=f)
            knn_mtof = knn_mtof.assign(fold=f)
            knn_parts.extend([knn_ftom, knn_mtof])
            fold_sizes.append({"fold": f, "n_train": int(len(tr_idx)), "n_test": int(len(te_idx))})

        out_df = pd.DataFrame(all_rows).sort_values("distance_l2", ascending=True).reset_index(drop=True)
        # each image once as source i
        assert len(out_df) == len(df), f"expected n={len(df)}, got {len(out_df)}"
        knn_all = pd.concat(knn_parts, ignore_index=True)

        perm = same_category_permutation_p(
            out_df["i_category"].to_numpy(),
            out_df["j_category"].to_numpy(),
            n_perm=args.n_perm,
            seed=args.split_seed,
        )
        out_csv = out_dir / "paper2_emotion_equivalent_pairs_themecv.csv"
        out_top = out_dir / "paper2_emotion_equivalent_pairs_themecv_topN.csv"
        out_meta = out_dir / "paper2_emotion_equivalent_pairs_themecv_meta.json"
        knn_csv = out_dir / "paper2_emotion_equivalent_pairs_themecv_knn.csv"
        out_df.to_csv(out_csv, index=False)
        out_df.head(args.top_n).to_csv(out_top, index=False)
        knn_all.to_csv(knn_csv, index=False)
        meta = {
            "mode": "themecv",
            "split_seed": int(args.split_seed),
            "n_folds": int(args.n_folds),
            "cv_pool": args.cv_pool,
            "n_pairs": int(len(out_df)),
            "n_images": int(len(df)),
            "knn_k": int(args.knn_k),
            "knn_csv": str(knn_csv),
            "fold_sizes": fold_sizes,
            "mean_distance": float(out_df["distance_l2"].mean()),
            "median_distance": float(out_df["distance_l2"].median()),
            "same_category_rate": float(out_df["same_category"].mean()),
            "same_theme_base_rate": float(out_df["same_theme_base"].mean()),
            "category_permutation": perm,
            "directions": {
                "FtoM": "hub=female true y_f(i); matches=top-k j by ||Dec_m(z_j)-y_f(i)||",
                "MtoF": "hub=male true y_m(j); matches=top-k i by ||Dec_f(z_i)-y_m(j)||",
            },
            "note": (
                "Theme-blocked K-fold: each OASIS image is a query exactly once. "
                "Decoder weights are shared (weights.pt); StandardScaler is fit on train fold only. "
                "cv_pool=train matches only from other folds (recommended)."
            ),
        }
        out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(json.dumps(meta, indent=2))
        print(f"Saved {out_csv}")
        print(f"Saved {out_top}")
        print(f"Saved {knn_csv} (n={len(knn_all)})")
        return

    # ---- fixedsplit (legacy; n≈153 test queries) ----
    tr_idx, te_idx = split_by_theme(df, seed=args.split_seed)
    pool_idx = tr_idx if args.pool == "train" else np.arange(len(df))

    scaler = StandardScaler().fit(X[tr_idx])
    Xs = scaler.transform(X).astype(np.float32)
    p_m_all, p_f_all = load_common_preds(Xs, ckpt, device)

    rows = build_1nn_rows(
        df, te_idx=te_idx, pool_idx=pool_idx,
        y_f=y_f, p_m_all=p_m_all, p_f_all=p_f_all,
    )
    out_df = pd.DataFrame(rows).sort_values("distance_l2", ascending=True).reset_index(drop=True)
    out_csv = out_dir / "paper2_emotion_equivalent_pairs_fixedsplit.csv"
    out_top = out_dir / "paper2_emotion_equivalent_pairs_fixedsplit_topN.csv"
    out_meta = out_dir / "paper2_emotion_equivalent_pairs_fixedsplit_meta.json"
    out_df.to_csv(out_csv, index=False)
    out_df.head(args.top_n).to_csv(out_top, index=False)

    knn_ftom = build_knn_rows(
        df,
        query_idx=te_idx,
        pool_idx=pool_idx,
        y_query=y_f[te_idx],
        p_pool=p_m_all[pool_idx],
        direction="FtoM",
        k=args.knn_k,
        pool=args.pool,
    )
    knn_mtof = build_knn_rows(
        df,
        query_idx=te_idx,
        pool_idx=pool_idx,
        y_query=y_m[te_idx],
        p_pool=p_f_all[pool_idx],
        direction="MtoF",
        k=args.knn_k,
        pool=args.pool,
    )

    knn_all = pd.concat([knn_ftom, knn_mtof], ignore_index=True)
    knn_csv = out_dir / "paper2_emotion_equivalent_pairs_fixedsplit_knn.csv"
    knn_all.to_csv(knn_csv, index=False)

    perm = same_category_permutation_p(
        out_df["i_category"].to_numpy(),
        out_df["j_category"].to_numpy(),
        n_perm=args.n_perm,
        seed=args.split_seed,
    )
    meta = {
        "mode": "fixedsplit",
        "split_seed": int(args.split_seed),
        "pool": args.pool,
        "n_test": int(len(te_idx)),
        "n_pool": int(len(pool_idx)),
        "knn_k": int(args.knn_k),
        "mean_distance": float(out_df["distance_l2"].mean()),
        "median_distance": float(out_df["distance_l2"].median()),
        "same_category_rate": float(out_df["same_category"].mean()) if "same_category" in out_df.columns else None,
        "same_theme_base_rate": float(out_df["same_theme_base"].mean()) if "same_theme_base" in out_df.columns else None,
        "category_permutation": perm,
        "knn_csv": str(knn_csv),
        "directions": {
            "FtoM": "hub=female true y_f(i); matches=top-k j by ||Dec_m(z_j)-y_f(i)||",
            "MtoF": "hub=male true y_m(j); matches=top-k i by ||Dec_f(z_i)-y_m(j)||",
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))
    print(f"Saved {out_csv}")
    print(f"Saved {out_top}")
    print(f"Saved {knn_csv} (n={len(knn_all)})")
    print(f"Saved {out_meta}")


if __name__ == "__main__":
    main()

