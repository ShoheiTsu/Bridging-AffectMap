#!/usr/bin/env python3
"""
2D affine calibration for external-subject generalization.

Map public OASIS (valence, arousal) to new-subject means:
  y = A x + b
where x,y are 2D vectors [V, A].

Outputs:
- results/cvae_cross_gender/subject_generalization_affine2d.json
- results/cvae_cross_gender/subject_generalization_affine2d_table.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_CSV = PROJECT_ROOT / "results" / "cvae_cross_gender" / "subject_scores_oasis_alignment_1to7.csv"
OUT_DIR = PROJECT_ROOT / "results" / "cvae_cross_gender"


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    xv = x - x.mean()
    yv = y - y.mean()
    den = np.sqrt(np.sum(xv * xv) * np.sum(yv * yv))
    if den <= 0:
        return float("nan")
    return float(np.sum(xv * yv) / den)


def metrics_1d(y_true: np.ndarray, y_pred: np.ndarray):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Pearson_r": pearsonr_np(y_true, y_pred),
    }


def metrics_2d(y_true: np.ndarray, y_pred: np.ndarray):
    mv = metrics_1d(y_true[:, 0], y_pred[:, 0])
    ma = metrics_1d(y_true[:, 1], y_pred[:, 1])
    return {
        "valence": mv,
        "arousal": ma,
        "overall_mean": {
            "R2_mean": float((mv["R2"] + ma["R2"]) / 2),
            "MAE_mean": float((mv["MAE"] + ma["MAE"]) / 2),
            "Pearson_r_mean": float((mv["Pearson_r"] + ma["Pearson_r"]) / 2),
        },
    }


def evaluate_predictor(name: str, x: np.ndarray, y: np.ndarray):
    # raw
    raw = metrics_2d(y, x)
    # affine
    reg = LinearRegression().fit(x, y)
    yhat = reg.predict(x)
    aff = metrics_2d(y, yhat)
    return {
        "predictor": name,
        "raw": raw,
        "affine2d": aff,
        "affine_params": {
            "A_2x2": reg.coef_.tolist(),
            "b_2": reg.intercept_.tolist(),
        },
    }


def main():
    if not IN_CSV.exists():
        raise SystemExit(f"Input not found: {IN_CSV}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)

    req = [
        "valence_mean_1to7", "arousal_mean_1to7",
        "valence", "arousal",
        "valence_male", "arousal_male",
        "valence_female", "arousal_female",
    ]
    use = df.dropna(subset=req).copy()
    if len(use) < 10:
        raise SystemExit(f"Too few aligned rows: {len(use)}")

    y = use[["valence_mean_1to7", "arousal_mean_1to7"]].to_numpy(np.float64)
    x_overall = use[["valence", "arousal"]].to_numpy(np.float64)
    x_male = use[["valence_male", "arousal_male"]].to_numpy(np.float64)
    x_female = use[["valence_female", "arousal_female"]].to_numpy(np.float64)

    results = [
        evaluate_predictor("oasis_overall", x_overall, y),
        evaluate_predictor("oasis_male", x_male, y),
        evaluate_predictor("oasis_female", x_female, y),
    ]

    # final summary ranking by affine R2_mean
    ranking = sorted(
        [
            {
                "predictor": r["predictor"],
                "raw_R2_mean": r["raw"]["overall_mean"]["R2_mean"],
                "affine2d_R2_mean": r["affine2d"]["overall_mean"]["R2_mean"],
                "affine2d_MAE_mean": r["affine2d"]["overall_mean"]["MAE_mean"],
                "affine2d_Pearson_r_mean": r["affine2d"]["overall_mean"]["Pearson_r_mean"],
            }
            for r in results
        ],
        key=lambda z: z["affine2d_R2_mean"],
        reverse=True,
    )

    out = {
        "task": "external_subject_generalization_affine2d",
        "n_images": int(len(use)),
        "input_csv": str(IN_CSV),
        "results": results,
        "ranking_by_affine2d_R2_mean": ranking,
        "final_recommended_predictor": ranking[0]["predictor"] if ranking else None,
    }

    out_json = OUT_DIR / "subject_generalization_affine2d.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    rows = []
    for r in results:
        rows.append({
            "predictor": r["predictor"],
            "raw_R2_mean": r["raw"]["overall_mean"]["R2_mean"],
            "raw_MAE_mean": r["raw"]["overall_mean"]["MAE_mean"],
            "raw_r_mean": r["raw"]["overall_mean"]["Pearson_r_mean"],
            "affine2d_R2_mean": r["affine2d"]["overall_mean"]["R2_mean"],
            "affine2d_MAE_mean": r["affine2d"]["overall_mean"]["MAE_mean"],
            "affine2d_r_mean": r["affine2d"]["overall_mean"]["Pearson_r_mean"],
            "affine2d_R2_valence": r["affine2d"]["valence"]["R2"],
            "affine2d_R2_arousal": r["affine2d"]["arousal"]["R2"],
        })
    out_csv = OUT_DIR / "subject_generalization_affine2d_table.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(json.dumps(out, indent=2))
    print(f"Saved {out_json}")
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()

