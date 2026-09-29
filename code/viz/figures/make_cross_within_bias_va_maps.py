#!/usr/bin/env python3
"""LOTO cross-model bias VA maps: (cross error vector) − (within-native error vector), and adjust deltas."""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from config import (  # noqa: E402
    FIG_INTEGRATED,
    OASIS_SCORES_CSV,
    RESULTS_GENDER,
    VALENCE_AROUSAL_SCALE_MAX,
    VALENCE_AROUSAL_SCALE_MIN,
)
from dataset import add_theme_base, load_oasis_meta  # noqa: E402

VA_MIN = VALENCE_AROUSAL_SCALE_MIN
VA_MAX = VALENCE_AROUSAL_SCALE_MAX
MIN_TEST_LOTO = 1  # pooled R² / SI maps include singleton themes (n=900)
HEAT_VMAX_FIXED = 1.6


@dataclass(frozen=True)
class MapSpec:
    key: str
    title: str
    ref_gender: str  # anchor VA grid


MAP_SPECS: tuple[MapSpec, ...] = (
    MapSpec(
        "cross_bias_female_target",
        "Male→female error minus female→female error",
        "female",
    ),
    MapSpec(
        "cross_bias_male_target",
        "Female→male error minus male→male error",
        "male",
    ),
    MapSpec(
        "adjust_delta_female_target",
        "Male adjust→female error minus male→female error",
        "female",
    ),
    MapSpec(
        "adjust_delta_male_target",
        "Female adjust→male error minus female→male error",
        "male",
    ),
)

REF_LABELS = {"male": "male score (ref)", "female": "female score (ref)"}


def make_grid(step: float) -> np.ndarray:
    return np.arange(VA_MIN + step / 2, VA_MAX, step)


def loto_indices(df) -> list[int]:
    idx: list[int] = []
    for left_out in df["theme_base"].unique():
        m = (df["theme_base"] == left_out).values
        if int(m.sum()) >= MIN_TEST_LOTO:
            idx.extend(np.where(m)[0].tolist())
    return idx


def err_vec(pred: np.ndarray, true_v: np.ndarray, true_a: np.ndarray) -> np.ndarray:
    return pred - np.column_stack([true_v, true_a])


def load_pool(*, model: str, category: str | None) -> dict:
    oasis = load_oasis_meta(OASIS_SCORES_CSV)
    cols = ["valence_male", "arousal_male", "valence_female", "arousal_female", "category"]
    full = oasis[oasis[cols].notna().all(axis=1)].reset_index(drop=True)
    full = add_theme_base(full)
    loto_idx = loto_indices(full)
    loto_sub = full.iloc[loto_idx].reset_index(drop=True)
    if category is not None:
        loto_sub = loto_sub[loto_sub["category"] == category].reset_index(drop=True)
        arr_idx = np.where(full.iloc[loto_idx]["category"].values == category)[0]
    else:
        arr_idx = np.arange(len(loto_sub))

    base = RESULTS_GENDER / "loto_nested"
    adj = RESULTS_GENDER / "fusion_background_vit" / "loto_nested"
    return {
        "v_m": loto_sub["valence_male"].to_numpy(float),
        "a_m": loto_sub["arousal_male"].to_numpy(float),
        "v_f": loto_sub["valence_female"].to_numpy(float),
        "a_f": loto_sub["arousal_female"].to_numpy(float),
        "pm": np.load(base / "male" / f"y_all_pred_{model}.npy")[arr_idx],
        "pf": np.load(base / "female" / f"y_all_pred_{model}.npy")[arr_idx],
        "pm_adj": np.load(adj / "male" / "y_all_pred_fusion.npy")[arr_idx],
        "pf_adj": np.load(adj / "female" / "y_all_pred_fusion.npy")[arr_idx],
        "n": len(loto_sub),
    }


def vec_for_spec(data: dict, spec: MapSpec) -> np.ndarray:
    v_m, a_m = data["v_m"], data["a_m"]
    v_f, a_f = data["v_f"], data["a_f"]
    pm, pf = data["pm"], data["pf"]
    pm_adj, pf_adj = data["pm_adj"], data["pf_adj"]

    if spec.key == "cross_bias_female_target":
        return err_vec(pm, v_f, a_f) - err_vec(pf, v_f, a_f)
    if spec.key == "cross_bias_male_target":
        return err_vec(pf, v_m, a_m) - err_vec(pm, v_m, a_m)
    if spec.key == "adjust_delta_female_target":
        return err_vec(pm_adj, v_f, a_f) - err_vec(pm, v_f, a_f)
    if spec.key == "adjust_delta_male_target":
        return err_vec(pf_adj, v_m, a_m) - err_vec(pf, v_m, a_m)
    raise KeyError(spec.key)


def _local(v_ref, a_ref, vec, g, radius, min_n):
    n_g = len(g)
    dx = np.full((n_g, n_g), np.nan)
    dy = np.full((n_g, n_g), np.nan)
    cnt = np.zeros((n_g, n_g), dtype=int)
    mag = np.full((n_g, n_g), np.nan)
    r2 = radius * radius
    norms = np.linalg.norm(vec, axis=1)
    for i in range(n_g):
        for j in range(n_g):
            vc, ac = g[j], g[i]
            m = (v_ref - vc) ** 2 + (a_ref - ac) ** 2 <= r2
            n = int(m.sum())
            cnt[i, j] = n
            if n < min_n:
                continue
            dx[i, j] = float(np.mean(vec[m, 0]))
            dy[i, j] = float(np.mean(vec[m, 1]))
            mag[i, j] = float(np.mean(norms[m]))
    return dx, dy, mag, cnt


def emit_1x2(
    vec: np.ndarray,
    v_anchor: np.ndarray,
    a_anchor: np.ndarray,
    spec: MapSpec,
    *,
    g: np.ndarray,
    step: float,
    radius: float,
    min_n: int,
    subset_tag: str,
    model: str,
    out_dir: Path,
    heat_vmax: float | None,
    name_suffix: str,
) -> None:
    ref_label = REF_LABELS[spec.ref_gender]
    dx, dy, mag, cnt = _local(v_anchor, a_anchor, vec, g, radius, min_n)
    if heat_vmax is not None:
        vmax = heat_vmax
    else:
        finite = mag[np.isfinite(mag)]
        vmax = max(float(np.nanpercentile(finite, 95)) if finite.size else 1.0, 1e-6)

    xg, yg = np.meshgrid(g, g, indexing="xy")
    qscale = 1.0 / (step * 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    mask = cnt >= min_n
    axes[0].quiver(
        xg[mask], yg[mask],
        np.nan_to_num(dx[mask], nan=0.0), np.nan_to_num(dy[mask], nan=0.0),
        angles="xy", scale_units="xy", scale=qscale, width=0.004,
        headwidth=3, headlength=4, color="0.2",
    )
    axes[0].set_xlim(VA_MIN, VA_MAX)
    axes[0].set_ylim(VA_MIN, VA_MAX)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel(f"Valence ({ref_label})")
    axes[0].set_ylabel(f"Arousal ({ref_label})")
    axes[0].set_title(f"{spec.title}\n(bias / delta vector)", fontsize=9)
    axes[0].grid(True, alpha=0.2)

    ext = [g[0] - step / 2, g[-1] + step / 2, g[0] - step / 2, g[-1] + step / 2]
    im = axes[1].imshow(mag, origin="lower", extent=ext, aspect="equal", cmap="viridis", vmin=0.0, vmax=vmax)
    axes[1].set_xlim(VA_MIN, VA_MAX)
    axes[1].set_ylim(VA_MIN, VA_MAX)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel(f"Valence ({ref_label})")
    axes[1].set_ylabel(f"Arousal ({ref_label})")
    axes[1].set_title(f"{spec.title}\n||vector||", fontsize=9)

    title = f"LOTO {subset_tag}: {spec.title} ({model})"
    if heat_vmax is not None:
        title += f", cbar max={heat_vmax:g}"
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    fig.colorbar(im, ax=axes[1], shrink=0.85, label="Mean ||vector|| (V-A units)")
    out = out_dir / f"Figure_{subset_tag}_{spec.key}_{model}{name_suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def emit_overview_2x2(
    fields: list[tuple[MapSpec, np.ndarray, np.ndarray, np.ndarray]],
    *,
    g: np.ndarray,
    step: float,
    radius: float,
    min_n: int,
    subset_tag: str,
    model: str,
    out_dir: Path,
    heat_vmax: float | None,
    name_suffix: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    last_im = None
    for col, (spec, vec, v_ref, a_ref) in enumerate(fields):
        dx, dy, mag, cnt = _local(v_ref, a_ref, vec, g, radius, min_n)
        ref_label = REF_LABELS[spec.ref_gender]
        xg, yg = np.meshgrid(g, g, indexing="xy")
        qscale = 1.0 / (step * 2.5)
        mask = cnt >= min_n
        ax0 = axes[0, col]
        ax0.quiver(
            xg[mask], yg[mask],
            np.nan_to_num(dx[mask], nan=0.0), np.nan_to_num(dy[mask], nan=0.0),
            angles="xy", scale_units="xy", scale=qscale, width=0.003,
            headwidth=3, headlength=4, color="0.2",
        )
        ax0.set_xlim(VA_MIN, VA_MAX)
        ax0.set_ylim(VA_MIN, VA_MAX)
        ax0.set_aspect("equal")
        ax0.set_title(f"{spec.title}\nvector", fontsize=8)
        ax0.set_xlabel(f"Valence ({ref_label})", fontsize=8)
        ax0.set_ylabel(f"Arousal ({ref_label})", fontsize=8)
        ax0.grid(True, alpha=0.2)

        if heat_vmax is not None:
            vmax = heat_vmax
        else:
            finite = mag[np.isfinite(mag)]
            vmax = max(float(np.nanpercentile(finite, 95)) if finite.size else 1.0, 1e-6)
        ext = [g[0] - step / 2, g[-1] + step / 2, g[0] - step / 2, g[-1] + step / 2]
        ax1 = axes[1, col]
        last_im = ax1.imshow(mag, origin="lower", extent=ext, aspect="equal", cmap="viridis", vmin=0.0, vmax=vmax)
        ax1.set_xlim(VA_MIN, VA_MAX)
        ax1.set_ylim(VA_MIN, VA_MAX)
        ax1.set_aspect("equal")
        ax1.set_title("||vector||", fontsize=8)
        ax1.set_xlabel(f"Valence ({ref_label})", fontsize=8)
        ax1.set_ylabel(f"Arousal ({ref_label})", fontsize=8)

    row_labels = ["Vector", "Magnitude"]
    for r, lab in enumerate(row_labels):
        axes[r, 0].annotate(
            lab, xy=(-0.18, 0.5), xycoords="axes fraction", rotation=90,
            va="center", ha="center", fontsize=10, fontweight="bold",
        )
    title = f"LOTO cross/within bias & adjust deltas ({subset_tag}, {model})"
    if heat_vmax is not None:
        title += f", cbar max={heat_vmax:g}"
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout(rect=(0.04, 0, 0.94, 0.94))
    if last_im is not None:
        fig.colorbar(last_im, ax=axes[1, :].ravel().tolist(), shrink=0.85, label="Mean ||vector||")
    out = out_dir / f"Figure_{subset_tag}_cross_within_bias_adjust_overview_{model}{name_suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def run_subset(
    *,
    data: dict,
    subset_tag: str,
    model: str,
    g: np.ndarray,
    args: argparse.Namespace,
) -> None:
    fields: list[tuple[MapSpec, np.ndarray, np.ndarray, np.ndarray]] = []
    for spec in MAP_SPECS:
        vec = vec_for_spec(data, spec)
        v_ref = data["v_m"] if spec.ref_gender == "male" else data["v_f"]
        a_ref = data["a_m"] if spec.ref_gender == "male" else data["a_f"]
        fields.append((spec, vec, v_ref, a_ref))
        for heat_vmax, suffix in ((None, ""), (HEAT_VMAX_FIXED, f"_cbarmax{HEAT_VMAX_FIXED:g}")):
            emit_1x2(
                vec, v_ref, a_ref, spec,
                g=g, step=args.grid_step, radius=args.radius, min_n=args.min_n,
                subset_tag=subset_tag, model=model, out_dir=args.out_dir,
                heat_vmax=heat_vmax, name_suffix=suffix,
            )
    for heat_vmax, suffix in ((None, ""), (HEAT_VMAX_FIXED, f"_cbarmax{HEAT_VMAX_FIXED:g}")):
        emit_overview_2x2(
            fields, g=g, step=args.grid_step, radius=args.radius, min_n=args.min_n,
            subset_tag=subset_tag, model=model, out_dir=args.out_dir,
            heat_vmax=heat_vmax, name_suffix=suffix,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("fusion", "clip", "vit"), default="fusion")
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--grid-step", type=float, default=0.1)
    ap.add_argument("--min-n", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=FIG_INTEGRATED)
    args = ap.parse_args()

    g = make_grid(args.grid_step)
    for subset_tag, category in (("all_loto", None), ("person_loto", "Person")):
        data = load_pool(model=args.model, category=category)
        print(f"Subset {subset_tag}: n={data['n']}")
        run_subset(data=data, subset_tag=subset_tag, model=args.model, g=g, args=args)


if __name__ == "__main__":
    main()
