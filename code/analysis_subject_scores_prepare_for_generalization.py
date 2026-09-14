#!/usr/bin/env python3
"""
Prepare external subject VA scores for generalization analysis.

Inputs:
- stimulus_image_list.csv: image_id, file_name, absolute_path
- subject_image_va_scores.csv: row_id, subject_id, image_id, valence, arousal  (trial-level, -1..1)

Outputs:
- results/cvae_cross_gender/subject_scores_image_means_1to7.csv
- results/cvae_cross_gender/subject_scores_oasis_alignment_1to7.csv
- results/cvae_cross_gender/subject_scores_prepare_summary.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STIM_CSV = PROJECT_ROOT / "stimulus_image_list.csv"
SUBJ_CSV = PROJECT_ROOT / "subject_image_va_scores.csv"
OUT_DIR = PROJECT_ROOT / "results" / "cvae_cross_gender"
OASIS_CSV = PROJECT_ROOT.parent / "EmotionPro1" / "data" / "oasis_scores.csv"


def minus1_1_to_1_7(x: pd.Series) -> pd.Series:
    # linear map: -1 -> 1, +1 -> 7
    return 3.0 * x + 4.0


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stim = pd.read_csv(STIM_CSV)
    subj = pd.read_csv(SUBJ_CSV)

    required_stim = {"image_id", "file_name", "absolute_path"}
    required_subj = {"row_id", "subject_id", "image_id", "valence", "arousal"}
    if not required_stim.issubset(stim.columns):
        raise SystemExit(f"Missing columns in stimulus_image_list.csv: {required_stim - set(stim.columns)}")
    if not required_subj.issubset(subj.columns):
        raise SystemExit(f"Missing columns in subject_image_va_scores.csv: {required_subj - set(subj.columns)}")

    # trial-level sanity range
    vmin, vmax = float(subj["valence"].min()), float(subj["valence"].max())
    amin, amax = float(subj["arousal"].min()), float(subj["arousal"].max())

    # image-level mean in -1..1
    agg = (
        subj.groupby("image_id", as_index=False)
        .agg(
            n_trials=("row_id", "count"),
            n_subjects=("subject_id", "nunique"),
            valence_mean_m11=("valence", "mean"),
            arousal_mean_m11=("arousal", "mean"),
            valence_sd_m11=("valence", "std"),
            arousal_sd_m11=("arousal", "std"),
        )
    )
    agg["valence_mean_1to7"] = minus1_1_to_1_7(agg["valence_mean_m11"])
    agg["arousal_mean_1to7"] = minus1_1_to_1_7(agg["arousal_mean_m11"])

    merged = stim.merge(agg, on="image_id", how="left")
    out_img = OUT_DIR / "subject_scores_image_means_1to7.csv"
    merged.to_csv(out_img, index=False)

    # align with OASIS public scores (1..7) for generalization evaluation
    oasis = pd.read_csv(OASIS_CSV)
    keep = ["image_filename", "valence", "arousal", "valence_male", "arousal_male", "valence_female", "arousal_female"]
    oasis = oasis[keep].rename(columns={"image_filename": "file_name"})
    align = merged.merge(oasis, on="file_name", how="left", validate="one_to_one")
    out_align = OUT_DIR / "subject_scores_oasis_alignment_1to7.csv"
    align.to_csv(out_align, index=False)

    summary = {
        "n_trials_total": int(len(subj)),
        "n_images_in_stimulus_list": int(len(stim)),
        "n_images_with_subject_scores": int(agg["image_id"].nunique()),
        "trial_scale_minus1_1_range": {
            "valence_min": vmin, "valence_max": vmax,
            "arousal_min": amin, "arousal_max": amax,
        },
        "image_mean_scale_1to7_range": {
            "valence_mean_min": float(agg["valence_mean_1to7"].min()),
            "valence_mean_max": float(agg["valence_mean_1to7"].max()),
            "arousal_mean_min": float(agg["arousal_mean_1to7"].min()),
            "arousal_mean_max": float(agg["arousal_mean_1to7"].max()),
        },
        "n_aligned_with_oasis_by_file_name": int(align["valence"].notna().sum()),
        "outputs": {
            "image_means_csv": str(out_img),
            "aligned_csv": str(out_align),
        },
    }
    out_summary = OUT_DIR / "subject_scores_prepare_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {out_img}")
    print(f"Saved {out_align}")
    print(f"Saved {out_summary}")


if __name__ == "__main__":
    main()

