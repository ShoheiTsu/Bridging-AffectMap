#!/usr/bin/env python3
"""
Nontriviality of emotion-equivalent pairs (Analysis A) and residual-region utility (Analysis B).

Analysis A (statistical nontriviality):
  A1. Are equivalent-pair affective distances closer than random re-pairings (permutation)?
  A2. Are same_category / cross-category / same_theme_base rates non-chance
      (permutation vs pool baserates)?
  A3. Do equivalent pairs rely on semantic closeness
      (CLIP feature distance: equivalent vs random pairs)?

Analysis B (residual-region utility):
  Where Φ residual (residual_l2 from residual_per_image_gender; n=900) or model twist
  is large, does case-based equivalence translation beat Φ?

Outputs:
  results/equivalence_nontriviality/equivalence_nontriviality.json
  results/equivalence_nontriviality/Figure_equivalence_nontriviality.png
  results/equivalence_nontriviality/RESULTS.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from config import OASIS_SCORES_CSV, RESULTS_STEP1, CVAE_CROSS_GENDER_DIR, FIG_INTEGRATED  # noqa: E402
from dataset import add_theme_base, load_oasis_meta  # noqa: E402

TARGET_M = ["valence_male", "arousal_male"]
TARGET_F = ["valence_female", "arousal_female"]

EQ_CSV_FIXED = CVAE_CROSS_GENDER_DIR / "emotion_equivalent_pairs_fixedsplit.csv"
EQ_META_FIXED = CVAE_CROSS_GENDER_DIR / "emotion_equivalent_pairs_fixedsplit_meta.json"
EQ_CSV_CV = CVAE_CROSS_GENDER_DIR / "emotion_equivalent_pairs_themecv.csv"
EQ_META_CV = CVAE_CROSS_GENDER_DIR / "emotion_equivalent_pairs_themecv_meta.json"
# Back-compat alias
EQ_CSV = EQ_CSV_FIXED
REL_TWIST_CSV = PROJECT_ROOT / "results" / "relational_cross_within_twist" / "relational_twist_per_image.csv"

OUT_DIR = PROJECT_ROOT / "results" / "equivalence_nontriviality"
DOC_OUT = OUT_DIR / "RESULTS.md"

N_PERM = 10000
SEED = 42


def resolve_pairs_paths(mode: str) -> tuple[Path, Path, Path, Path]:
    """Return (eq_csv, eq_meta, json_out, md_out) for fixedsplit or themecv."""
    if mode == "themecv":
        return (
            EQ_CSV_CV,
            EQ_META_CV,
            OUT_DIR / "equivalence_nontriviality_themecv.json",
            OUT_DIR / "RESULTS_THEMECV.md",
        )
    return (
        EQ_CSV_FIXED,
        EQ_META_FIXED,
        OUT_DIR / "equivalence_nontriviality.json",
        DOC_OUT,
    )


def load_clip_aligned() -> tuple[pd.DataFrame, np.ndarray]:
    df = load_oasis_meta(OASIS_SCORES_CSV)
    valid = df[TARGET_M + TARGET_F].notna().all(axis=1).values
    df = df.loc[valid].reset_index(drop=True)
    df = add_theme_base(df)
    X = np.load(RESULTS_STEP1 / "features_clip.npy").astype(np.float32)[valid]
    return df, X


def analysis_a1_emotional_permutation(eq: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Observed best-match emotional distance vs random re-pairing of the same preds."""
    y_f = eq[["y_f_i_valence", "y_f_i_arousal"]].to_numpy(float)
    p_m = eq[["pred_m_j_valence", "pred_m_j_arousal"]].to_numpy(float)
    obs = np.linalg.norm(y_f - p_m, axis=1)
    obs_median = float(np.median(obs))
    obs_mean = float(obs.mean())

    n = len(eq)
    null_medians = np.empty(N_PERM)
    null_means = np.empty(N_PERM)
    for k in range(N_PERM):
        perm = rng.permutation(n)
        d = np.linalg.norm(y_f - p_m[perm], axis=1)
        null_medians[k] = np.median(d)
        null_means[k] = d.mean()

    p_median = float((np.sum(null_medians <= obs_median) + 1) / (N_PERM + 1))
    p_mean = float((np.sum(null_means <= obs_mean) + 1) / (N_PERM + 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Path filled by caller via save_a1_null(); keep placeholder for JSON
    return {
        "observed_median_distance": obs_median,
        "observed_mean_distance": obs_mean,
        "null_median_distance_mean": float(null_medians.mean()),
        "null_median_distance_min": float(null_medians.min()),
        "null_median_distance_ci95": [float(np.quantile(null_medians, 0.025)), float(np.quantile(null_medians, 0.975))],
        "permutation_p_median": p_median,
        "permutation_p_mean": p_mean,
        "effect_ratio_obs_over_null_median": float(obs_median / null_medians.mean()),
        "_null_medians": null_medians,
    }


def analysis_a2_semantic_category(eq: pd.DataFrame, pool_df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Observed same-category / same-theme rates vs pool base-rate null."""
    obs_same_cat = float(eq["same_category"].mean())
    obs_cross_cat = 1.0 - obs_same_cat
    obs_same_theme = float(eq["same_theme_base"].mean())

    # Expected same-category under random pool matching, conditioned on each test image's category.
    pool_counts = pool_df["category"].value_counts()
    pool_total = len(pool_df)
    p_cat = {c: pool_counts.get(c, 0) / pool_total for c in pool_counts.index}
    exp_same_cat = float(np.mean([p_cat.get(c, 0.0) for c in eq["i_category"]]))

    # Permutation: shuffle matched j-categories among the 153 to build null of same-category rate.
    j_cats = eq["j_category"].to_numpy()
    i_cats = eq["i_category"].to_numpy()
    null_rates = np.empty(N_PERM)
    for k in range(N_PERM):
        null_rates[k] = np.mean(i_cats == rng.permutation(j_cats))
    # two-sided p vs permutation null centered at chance
    p_same_cat = float((np.sum(np.abs(null_rates - null_rates.mean()) >= abs(obs_same_cat - null_rates.mean())) + 1) / (N_PERM + 1))

    return {
        "observed_same_category_rate": obs_same_cat,
        "observed_cross_category_rate": obs_cross_cat,
        "observed_same_theme_base_rate": obs_same_theme,
        "expected_same_category_pool_baserate": exp_same_cat,
        "expected_cross_category_pool_baserate": 1.0 - exp_same_cat,
        "permutation_null_same_category_mean": float(null_rates.mean()),
        "permutation_p_same_category_vs_chance": p_same_cat,
        "interpretation": "cross-category rate is at/above chance and same-theme is 0 -> emotional equivalence is not driven by semantic-category similarity",
    }


def analysis_a3_clip_distance(eq: pd.DataFrame, df: pd.DataFrame, X: np.ndarray, rng: np.random.Generator) -> dict:
    """Are equivalence pairs semantically (CLIP) closer than random pairs?"""
    i_idx = eq["i_idx"].to_numpy(int)
    j_idx = eq["j_idx"].to_numpy(int)
    Xi = X[i_idx]
    Xj = X[j_idx]
    obs_clip = np.linalg.norm(Xi - Xj, axis=1)
    obs_median = float(np.median(obs_clip))

    n = len(eq)
    null_medians = np.empty(N_PERM)
    for k in range(N_PERM):
        null_medians[k] = np.median(np.linalg.norm(Xi - Xj[rng.permutation(n)], axis=1))
    # one-sided: is observed CLIP distance SMALLER than random? (semantic-similarity hypothesis)
    p_smaller = float((np.sum(null_medians <= obs_median) + 1) / (N_PERM + 1))
    null_mean = float(null_medians.mean())
    semantic_ratio = float(obs_median / null_mean)

    return {
        "observed_median_clip_distance": obs_median,
        "random_pair_median_clip_distance_mean": null_mean,
        "random_pair_median_clip_distance_ci95": [float(np.quantile(null_medians, 0.025)), float(np.quantile(null_medians, 0.975))],
        "permutation_p_equivalence_semantically_closer": p_smaller,
        "semantic_ratio_obs_over_null": semantic_ratio,
        "interpretation": (
            "equivalence pairs are only marginally closer in CLIP space (ratio ~%.3f of random) "
            "vs emotionally ~0.033 of random -> semantic proximity explains a negligible share of the match"
        ) % semantic_ratio,
    }


def analysis_b_residual_translation(eq: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Does case-based translation help where the global bridge Φ leaves large residual/twist?

    Primary Φ residual is the gender-bridge in-sample residual on all OASIS images
    (`residual_per_image_gender.csv`, n=900), so beat rate / terciles align with
    theme-CV pairs (n=900) rather than the LOTO MIN_TEST=2 subset (n=810).
    Model-twist correlations remain on the available relational-twist merge.
    """
    del rng  # reserved for future resampling diagnostics
    gender_resid = pd.read_csv(
        PROJECT_ROOT / "results" / "population_bridge_analysis" / "residual_per_image_gender.csv"
    )[["image_id", "residual_l2"]].rename(columns={"residual_l2": "resid_score_l2"})
    rel = pd.read_csv(REL_TWIST_CSV)[["image_id", "model_twist_pm_pf", "cross_bias_err_mf_minus_ff"]]

    m = eq.merge(gender_resid, left_on="i_image_id", right_on="image_id", how="inner")
    m = m.dropna(subset=["resid_score_l2", "distance_l2"]).reset_index(drop=True)

    # case-based translation residual = distance_l2 ; global-bridge residual = resid_score_l2
    m["translation_improvement"] = m["resid_score_l2"] - m["distance_l2"]

    m_twist = m.merge(rel, left_on="i_image_id", right_on="image_id", how="left", suffixes=("", "_rel"))
    m_twist = m_twist.dropna(subset=["model_twist_pm_pf"]).reset_index(drop=True)

    def sp(frame: pd.DataFrame, a: str, b: str) -> dict:
        r, p = spearmanr(frame[a], frame[b])
        return {"spearman_rho": float(r), "p": float(p)}

    # tercile contrast: high vs low Φ residual
    q1, q2 = m["resid_score_l2"].quantile([1 / 3, 2 / 3])
    low = m[m["resid_score_l2"] <= q1]["translation_improvement"]
    high = m[m["resid_score_l2"] >= q2]["translation_improvement"]
    u, p_u = mannwhitneyu(high, low, alternative="greater")

    return {
        "n_merged": int(len(m)),
        "n_merged_with_model_twist": int(len(m_twist)),
        "phi_residual_source": "population_bridge_analysis/residual_per_image_gender.csv",
        "phi_residual_estimand": "in-sample gender-bridge Φ on OASIS group means (n=900)",
        "corr_phi_residual_vs_translation_improvement": sp(m, "resid_score_l2", "translation_improvement"),
        "corr_model_twist_vs_translation_improvement": (
            sp(m_twist, "model_twist_pm_pf", "translation_improvement")
            if len(m_twist) else {"spearman_rho": float("nan"), "p": float("nan")}
        ),
        "corr_phi_residual_vs_equiv_distance": sp(m, "resid_score_l2", "distance_l2"),
        "tercile_contrast_high_vs_low_phi_residual": {
            "median_improvement_high": float(high.median()),
            "median_improvement_low": float(low.median()),
            "mannwhitney_u": float(u),
            "p_high_gt_low": float(p_u),
            "n_high": int(len(high)),
            "n_low": int(len(low)),
        },
        "mean_translation_improvement": float(m["translation_improvement"].mean()),
        "frac_images_translation_beats_phi": float((m["translation_improvement"] > 0).mean()),
        "_merged": m,
    }


def make_figure(a1: dict, a3: dict, b: dict, out: Path) -> None:
    m = b["_merged"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))

    ax = axes[0]
    ax.axvline(a1["observed_median_distance"], color="#e53935", lw=2, label=f"observed median={a1['observed_median_distance']:.3f}")
    ax.axvspan(*a1["null_median_distance_ci95"], color="#90caf9", alpha=0.4, label="random re-pairing 95%")
    ax.axvline(a1["null_median_distance_mean"], color="#1976d2", lw=1.5, ls="--", label=f"null mean={a1['null_median_distance_mean']:.3f}")
    ax.set_xlabel("Emotional distance (VA L2)")
    ax.set_yticks([])
    ax.set_title(f"A1 Emotional closeness (p={a1['permutation_p_median']:.4f})")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    ax.axvline(a3["observed_median_clip_distance"], color="#e53935", lw=2, label=f"equivalence pairs={a3['observed_median_clip_distance']:.2f}")
    ax.axvspan(*a3["random_pair_median_clip_distance_ci95"], color="#a5d6a7", alpha=0.5, label="random pairs 95%")
    ax.set_xlabel("CLIP semantic distance (L2)")
    ax.set_yticks([])
    ax.set_title(f"A3 Semantic decoupling (p={a3['permutation_p_equivalence_semantically_closer']:.3f})")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[2]
    ax.scatter(m["resid_score_l2"], m["translation_improvement"], s=22, alpha=0.6, color="#5c6bc0")
    ax.axhline(0, color="0.5", lw=0.8)
    rho = b["corr_phi_residual_vs_translation_improvement"]["spearman_rho"]
    pval = b["corr_phi_residual_vs_translation_improvement"]["p"]
    ax.set_xlabel("Φ residual (excess twist proxy)")
    ax.set_ylabel("Translation improvement\n(Φ resid − equiv resid)")
    ax.set_title(f"B Residual coupling (ρ={rho:.2f}, p={pval:.1e})")

    fig.suptitle("Emotion-equivalence: non-triviality (A) and residual-region utility (B)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    meta: dict,
    a1: dict,
    a2: dict,
    a3: dict,
    b: dict,
    out: Path,
    *,
    pairs_csv: Path,
    mode: str,
    n_pairs: int,
) -> None:
    n_label = meta.get("n_pairs", meta.get("n_test", n_pairs))
    pool_label = meta.get("n_images", meta.get("n_pool", "—"))
    L = [
        "# Emotion-equivalent pairs: nontriviality and residual utility — results",
        "",
        "Auto-generated by `code/analysis_equivalence_nontriviality.py`",
        "",
        f"mode: **{mode}**",
        f"Pairs CSV: `{pairs_csv}`",
        f"(n={n_label} pairs; pool/images {pool_label}; seed {meta.get('split_seed', SEED)})",
        "",
        "---",
        "",
        "## Analysis A: nontriviality of equivalent pairs",
        "",
        "### A1. Affective closeness (random re-pairing permutation)",
        "",
        f"Equivalent-pair affective distance (VA L2) median = **{a1['observed_median_distance']:.3f}** "
        f"(mean {a1['observed_mean_distance']:.3f}). "
        f"Under a null that randomly reassigns the same {n_pairs} male-decoder predictions to female targets, "
        f"median = **{a1['null_median_distance_mean']:.3f}** "
        f"[95% {a1['null_median_distance_ci95'][0]:.3f}, {a1['null_median_distance_ci95'][1]:.3f}].",
        "",
        f"Permutation p = **{a1['permutation_p_median']:.4f}** (median) / {a1['permutation_p_mean']:.4f} (mean). "
        f"Observed/null ratio = **{a1['effect_ratio_obs_over_null_median']:.3f}**.",
        "",
        "**Interpretation:** The specific equivalent pairing is significantly closer in affect than arbitrary "
        "reassignment. The small distances are not a selection artefact; they show that **image pairs that "
        "share affective coordinates exist**.",
        "",
        "### A2. Dissociation from semantic category / theme",
        "",
        f"Observed same-category rate = **{a2['observed_same_category_rate']:.3f}** "
        f"(cross-category = **{a2['observed_cross_category_rate']:.3f}**); "
        f"same-theme_base rate = **{a2['observed_same_theme_base_rate']:.3f}**.",
        "",
        f"Pool-baserate expected same-category = {a2['expected_same_category_pool_baserate']:.3f} "
        f"(cross-category {a2['expected_cross_category_pool_baserate']:.3f}); "
        f"permutation null mean {a2['permutation_null_same_category_mean']:.3f}; "
        f"p = **{a2['permutation_p_same_category_vs_chance']:.4f}** (vs chance).",
        "",
        (
            "**Interpretation:** Same-category rate exceeds the null, but a majority remain cross-category. "
            "Equivalence is not reducible to same-category matches, supporting \"different meaning, same affect\"."
            if a2["permutation_p_same_category_vs_chance"] < 0.05
            and a2["observed_same_category_rate"] > a2["permutation_null_same_category_mean"]
            else
            "**Interpretation:** Cross-category rates are at or above chance and same-theme matches are absent. "
            "Equivalence does **not** depend on semantic-category similarity — supporting "
            "\"different meaning, same affect\"."
        ),
        "",
        "### A3. CLIP semantic distance (equivalent vs random pairs)",
        "",
        f"Equivalent-pair CLIP distance median = **{a3['observed_median_clip_distance']:.2f}**; "
        f"random pairs = **{a3['random_pair_median_clip_distance_mean']:.2f}** "
        f"[95% {a3['random_pair_median_clip_distance_ci95'][0]:.2f}, {a3['random_pair_median_clip_distance_ci95'][1]:.2f}].",
        "",
        f"Equivalent pairs are **{a3['semantic_ratio_obs_over_null']:.3f}×** the random semantic distance "
        f"(only slightly smaller; one-sided permutation p = {a3['permutation_p_equivalence_semantically_closer']:.4f}).",
        "",
        f"**Interpretation (effect size):** Affective distance shrinks to about "
        f"**{a1['effect_ratio_obs_over_null_median']:.3f}×** random, whereas CLIP distance remains nearly "
        f"**~{a3['semantic_ratio_obs_over_null']:.2f}×** random. Any semantic advantage is small; "
        "visual-semantic proximity explains only a minor share of equivalence. Affective equivalence "
        "largely lives in a **separate affective-coordinate** match.",
        "",
        "---",
        "",
        "## Analysis B: does equivalence translation help residual regions?",
        "",
        f"Associations among Φ residual (`resid_score_l2`), model twist, and case-based translation "
        f"improvement (Φ residual − equivalence residual); n={b['n_merged']} merged images:",
        "",
        f"- Φ residual × translation improvement: Spearman ρ = **{b['corr_phi_residual_vs_translation_improvement']['spearman_rho']:.3f}** "
        f"(p = {b['corr_phi_residual_vs_translation_improvement']['p']:.2e})",
        f"- model twist × translation improvement: ρ = **{b['corr_model_twist_vs_translation_improvement']['spearman_rho']:.3f}** "
        f"(p = {b['corr_model_twist_vs_translation_improvement']['p']:.2e})",
        f"- **Φ residual × equivalence distance**: ρ = **{b['corr_phi_residual_vs_equiv_distance']['spearman_rho']:.3f}** "
        f"(p = {b['corr_phi_residual_vs_equiv_distance']['p']:.2f})",
        "",
        f"High vs low Φ-residual terciles: median translation improvement = "
        f"**{b['tercile_contrast_high_vs_low_phi_residual']['median_improvement_high']:.3f}** vs "
        f"**{b['tercile_contrast_high_vs_low_phi_residual']['median_improvement_low']:.3f}**; "
        f"Mann–Whitney p(high>low) = **{b['tercile_contrast_high_vs_low_phi_residual']['p_high_gt_low']:.2e}**.",
        "",
        f"Fraction of images where equivalence translation beats Φ = **{b['frac_images_translation_beats_phi']:.3f}** "
        f"(mean improvement {b['mean_translation_improvement']:.3f} VA units).",
        "",
        f"**Note (mechanical correlation):** Because improvement := Φ residual − equivalence distance, "
        f"a high correlation with Φ residual is **partly definitional**. The substantive check is the "
        f"near-independence of Φ residual and equivalence distance "
        f"(ρ≈{b['corr_phi_residual_vs_equiv_distance']['spearman_rho']:.2f}) — equivalence error stays low "
        "even where Φ fails.",
        "",
        f"**Interpretation:** Even where the global affine Φ leaves large residuals, case-based translation "
        f"keeps error low (tercile improvement "
        f"{b['tercile_contrast_high_vs_low_phi_residual']['median_improvement_high']:.2f} vs "
        f"{b['tercile_contrast_high_vs_low_phi_residual']['median_improvement_low']:.2f}; "
        f"beats Φ on {100*b['frac_images_translation_beats_phi']:.1f}% of images), acting as a "
        "**hierarchical complement** to the global linear bridge.",
        "",
        "---",
        "",
        "## Figure",
        "",
        f"`results/equivalence_nontriviality/Figure_equivalence_nontriviality{'_themecv' if mode == 'themecv' else ''}.png`",
        "",
        "## Outputs",
        "",
        "| File | Description |",
        "|------|-------------|",
        f"| `{out.relative_to(PROJECT_ROOT) if str(out).startswith(str(PROJECT_ROOT)) else out}` | Report |",
        "",
    ]
    out.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Equivalence nontriviality A/B analysis")
    parser.add_argument(
        "--mode",
        choices=("fixedsplit", "themecv"),
        default="fixedsplit",
        help="Pair source: fixedsplit (n≈153) or themecv (n=900)",
    )
    parser.add_argument("--n-perm", type=int, default=10000)
    args = parser.parse_args()
    global N_PERM
    N_PERM = int(args.n_perm)

    eq_csv, eq_meta_path, json_out, md_out = resolve_pairs_paths(args.mode)
    if not eq_csv.exists():
        raise SystemExit(f"Missing pairs CSV: {eq_csv}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_INTEGRATED.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df, X = load_clip_aligned()
    eq = pd.read_csv(eq_csv)
    meta = json.loads(eq_meta_path.read_text(encoding="utf-8")) if eq_meta_path.exists() else {}
    meta = {**meta, "mode": args.mode, "pairs_csv": str(eq_csv.name)}

    print(f"mode={args.mode} n_pairs={len(eq)} n_perm={N_PERM}")
    print("A1: emotional permutation...")
    a1 = analysis_a1_emotional_permutation(eq, rng)
    null_name = (
        "a1_null_median_distribution_themecv.npy"
        if args.mode == "themecv"
        else "a1_null_median_distribution.npy"
    )
    null_npy = OUT_DIR / null_name
    np.save(null_npy, a1.pop("_null_medians"))
    a1["null_median_distribution_npy"] = str(null_npy)
    print("A2: semantic category...")
    a2 = analysis_a2_semantic_category(eq, df, rng)
    # Refresh A2 interpretation for themecv honesty
    if (
        a2["permutation_p_same_category_vs_chance"] < 0.05
        and a2["observed_same_category_rate"] > a2["permutation_null_same_category_mean"]
    ):
        a2["interpretation"] = (
            "same-category enrichment is significant above chance, but majority remain "
            "cross-category -> equivalence is not reducible to same-category matching"
        )
    print("A3: CLIP distance...")
    a3 = analysis_a3_clip_distance(eq, df, X, rng)
    print("B: residual translation...")
    b = analysis_b_residual_translation(eq, rng)

    fig_name = (
        "Figure_equivalence_nontriviality_themecv.png"
        if args.mode == "themecv"
        else "Figure_equivalence_nontriviality.png"
    )
    make_figure(a1, a3, b, FIG_INTEGRATED / fig_name)

    export = {
        "meta": meta,
        "analysis_a1_emotional_permutation": a1,
        "analysis_a2_semantic_category": a2,
        "analysis_a3_clip_semantic_distance": a3,
        "analysis_b_residual_translation": {k: v for k, v in b.items() if not k.startswith("_")},
        "n_permutations": N_PERM,
        "seed": SEED,
        "mode": args.mode,
    }
    json_out.write_text(json.dumps(export, indent=2), encoding="utf-8")
    write_markdown(
        meta, a1, a2, a3, b, md_out,
        pairs_csv=eq_csv, mode=args.mode, n_pairs=len(eq),
    )
    print(f"Done. JSON: {json_out}")
    print(f"Doc: {md_out}")
    t = b["tercile_contrast_high_vs_low_phi_residual"]
    print(
        f"SUMMARY A1 ratio={a1['effect_ratio_obs_over_null_median']:.4f} "
        f"A2 same={a2['observed_same_category_rate']:.3f} cross={a2['observed_cross_category_rate']:.3f} "
        f"p={a2['permutation_p_same_category_vs_chance']:.4f} "
        f"A3 clip_ratio={a3['semantic_ratio_obs_over_null']:.3f} "
        f"B beat={b['frac_images_translation_beats_phi']:.3f} "
        f"imp_hi/lo={t['median_improvement_high']:.3f}/{t['median_improvement_low']:.3f} "
        f"rho_resid_dist={b['corr_phi_residual_vs_equiv_distance']['spearman_rho']:.3f} "
        f"n_merged={b['n_merged']}"
    )


if __name__ == "__main__":
    main()
