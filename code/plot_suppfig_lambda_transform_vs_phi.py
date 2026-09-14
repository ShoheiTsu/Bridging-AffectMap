#!/usr/bin/env python3
"""Supplementary figure: λ transform vs Φ (theme-held-out CLIP Ridge)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "lambda_transform_vs_phi" / "summary_mtof.json"
OUT = ROOT / "Paper_fig" / "Paper_SuppFig_lambda_transform_vs_phi"


def main() -> None:
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    r2 = s["r2"]
    beat = s["beat_phi_rate"]

    # Primary comparison (exclude failed diagonal variants from main bars)
    order = [
        ("none", "λ_m\n(none)"),
        ("phi", "Φ\n(output)"),
        ("lambda_rank_1", "λ rank-1"),
        ("lambda_rank_2", "λ rank-2"),
        ("lambda_rank_4", "λ rank-4"),
        ("oracle", "λ_f\n(oracle)"),
    ]
    labels = [lab for _, lab in order]
    means = [r2[k]["R2_mean"] for k, _ in order]
    colors = ["#8a8a8a", "#2c6eaa", "#c47a2c", "#c47a2c", "#c47a2c", "#2a9d6a"]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color=colors, width=0.72, edgecolor="white", linewidth=0.6)
    ax.axhline(means[1], color="#2c6eaa", ls="--", lw=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Mean R² (V+A)/2")
    ax.set_ylim(0.4, 0.75)
    ax.set_title("A  Overall performance (M→F, LOTO Ridge)")
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    # Rank sweep
    ax = axes[1]
    ranks = [0, 1, 2, 4, 8, 16, 32]
    y = [r2[f"lambda_rank_{r}"]["R2_mean"] for r in ranks]
    ax.plot(ranks, y, "o-", color="#c47a2c", lw=1.5, ms=5, label="λ rank-r")
    ax.axhline(r2["phi"]["R2_mean"], color="#2c6eaa", ls="--", lw=1.2, label="Φ")
    ax.axhline(r2["none"]["R2_mean"], color="#8a8a8a", ls=":", lw=1.2, label="λ_m")
    ax.axhline(r2["oracle"]["R2_mean"], color="#2a9d6a", ls="-.", lw=1.0, label="oracle")
    ax.set_xlabel("Rank r")
    ax.set_ylabel("Mean R²")
    ax.set_title("B  Rank sweep (overfitting)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    ax.set_ylim(0.4, 0.72)

    note = (
        f"Φ beat rates: none {beat['none']:.0%}, rank-1 {beat['lambda_rank_1']:.0%}, "
        f"oracle {beat['oracle']:.0%}. "
        "Diagonal coefficient scaling failed (not shown). "
        "Convex blend (1−w)λ_m+wλ_f selected w=1 on every fold."
    )
    fig.supxlabel(note, fontsize=7.5, x=0.5)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".svg"):
        fig.savefig(OUT.with_suffix(ext), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT}.png / .svg")


if __name__ == "__main__":
    main()
