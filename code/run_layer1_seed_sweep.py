#!/usr/bin/env python3
"""
Layer-1 seed sweep driver (Criteria A/B/C).

For each train_seed:
  1) train split encoders (split_seed fixed)
  2) ΔR²(split − common) + permutation p
  3) swap degradation + linear CKA

Aggregates -> results/cvae_cross_gender/layer1_fixedsplit_seed_sweep.json

Example:
  python3 code/run_layer1_seed_sweep.py --seeds 47 48 49 50 51
  python3 code/run_layer1_seed_sweep.py --seeds 42 43 44 45 46 47 48 49 50 51 --skip-train-existing
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
CVAE = ROOT / "results" / "cvae_cross_gender"
OUT_SWEEP = CVAE / "layer1_fixedsplit_seed_sweep.json"
PY = sys.executable


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def train_one(seed: int, *, epochs: int, split_seed: int) -> Path:
    tag = f"seed{seed}"
    out = CVAE / "split_encoders" / tag
    _run([
        PY, str(CODE / "train_split_encoders.py"),
        "--train-seed", str(seed),
        "--split-seed", str(split_seed),
        "--out-tag", tag,
        "--epochs", str(epochs),
    ])
    return out


def delta_r2_one(seed: int, *, split_seed: int, n_perm: int) -> dict:
    tag = f"seed{seed}"
    # analysis writes a shared path; we immediately snapshot per-seed
    _run([
        PY, str(CODE / "analysis_delta_r2_common_vs_split.py"),
        "--split-tag", tag,
        "--split-seed", str(split_seed),
        "--n-perm", str(n_perm),
        "--seed", str(split_seed),
    ])
    src = CVAE / "delta_r2_common_vs_split.json"
    payload = json.loads(src.read_text(encoding="utf-8"))
    snap = CVAE / f"delta_r2_{tag}.json"
    snap.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def swap_cka_one(seed: int, *, split_seed: int) -> dict:
    tag = f"seed{seed}"
    _run([
        PY, str(CODE / "analysis_swap_and_cka.py"),
        "--split-tag", tag,
        "--split-seed", str(split_seed),
    ])
    return json.loads((CVAE / f"swap_cka_{tag}.json").read_text(encoding="utf-8"))


def row_from(seed: int, delta: dict, swap: dict) -> dict:
    return {
        "train_seed": int(seed),
        "delta_r2": float(delta["delta_r2_overall_mean_split_minus_common"]),
        "p_split_gt_common": float(delta["p_value_one_tailed_split_gt_common"]),
        "split_female_arousal": float(delta["r2_split"]["female_arousal"]),
        "common_female_arousal": float(delta["r2_common"]["female_arousal"]),
        "male_swap_degradation": float(swap["swap"]["male_degradation"]),
        "female_swap_degradation": float(swap["swap"]["female_degradation"]),
        "linear_cka": float(swap["cka"]["linear_cka_zm_zf"]),
    }


def load_existing_row(seed: int) -> dict | None:
    """Prefer already-aggregated sweep row; else reconstruct from per-seed JSONs."""
    if OUT_SWEEP.exists():
        prev = json.loads(OUT_SWEEP.read_text(encoding="utf-8"))
        for r in prev.get("rows", []):
            if int(r["train_seed"]) == int(seed):
                return r
    dpath = CVAE / f"delta_r2_seed{seed}.json"
    spath = CVAE / f"swap_cka_seed{seed}.json"
    if dpath.exists() and spath.exists():
        return row_from(seed, json.loads(dpath.read_text()), json.loads(spath.read_text()))
    return None


def aggregate(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: int(r["train_seed"]))
    d = np.asarray([r["delta_r2"] for r in rows], float)
    cka = np.asarray([r["linear_cka"] for r in rows], float)
    n = len(d)
    return {
        "n_seeds": int(n),
        "train_seeds": [int(r["train_seed"]) for r in rows],
        "split_seed": 42,
        "rows": rows,
        "delta_r2_mean": float(np.mean(d)),
        "delta_r2_std": float(np.std(d, ddof=1)) if n > 1 else 0.0,
        "delta_r2_sem": float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "cka_min": float(np.min(cka)),
        "cka_max": float(np.max(cka)),
        "cka_mean": float(np.mean(cka)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[47, 48, 49, 50, 51])
    ap.add_argument("--merge-existing-seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46],
                    help="include these seeds from existing sweep/json without retraining")
    ap.add_argument("--skip-train-existing", action="store_true",
                    help="skip train if encoder_m.pt already exists for a seed")
    ap.add_argument("--reanalyze-existing", action="store_true",
                    help="re-run delta/swap for merge-existing seeds (no retrain)")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--n-perm", type=int, default=500)
    args = ap.parse_args()

    rows: dict[int, dict] = {}

    # Keep prior rows for merge seeds unless reanalyze requested
    for s in args.merge_existing_seeds:
        if args.reanalyze_existing:
            continue
        ex = load_existing_row(s)
        if ex is not None:
            rows[int(s)] = ex
            print(f"[keep] seed {s}: delta_r2={ex['delta_r2']:+.4f}, CKA={ex['linear_cka']:.3f}")

    for seed in args.seeds:
        tag = f"seed{seed}"
        enc = CVAE / "split_encoders" / tag / "encoder_m.pt"
        need_train = not (args.skip_train_existing and enc.exists())
        if need_train:
            train_one(seed, epochs=args.epochs, split_seed=args.split_seed)
        elif not enc.exists():
            raise SystemExit(f"missing checkpoints for {tag}")
        else:
            print(f"[skip-train] {tag}")

        delta = delta_r2_one(seed, split_seed=args.split_seed, n_perm=args.n_perm)
        swap = swap_cka_one(seed, split_seed=args.split_seed)
        rows[int(seed)] = row_from(seed, delta, swap)
        r = rows[int(seed)]
        print(f"[done] seed {seed}: ΔR²={r['delta_r2']:+.4f}, CKA={r['linear_cka']:.3f}, "
              f"swapM={r['male_swap_degradation']:.2f}, swapF={r['female_swap_degradation']:.2f}")

    if args.reanalyze_existing:
        for seed in args.merge_existing_seeds:
            delta = delta_r2_one(seed, split_seed=args.split_seed, n_perm=args.n_perm)
            swap = swap_cka_one(seed, split_seed=args.split_seed)
            rows[int(seed)] = row_from(seed, delta, swap)

    payload = aggregate(list(rows.values()))
    OUT_SWEEP.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "n_seeds": payload["n_seeds"],
        "train_seeds": payload["train_seeds"],
        "delta_r2_mean": payload["delta_r2_mean"],
        "delta_r2_sem": payload["delta_r2_sem"],
        "cka_range": [payload["cka_min"], payload["cka_max"]],
        "saved": str(OUT_SWEEP),
    }, indent=2))


if __name__ == "__main__":
    main()
