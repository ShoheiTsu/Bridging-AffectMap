#!/usr/bin/env python3
"""
Supp: residual gender-bridge vector-field topology (M→F and F→M).

Estimands
  M→F: Δ = y_f − Φ_mf(y_m), field anchored at male VA
  F→M: Δ = y_m − Φ_fm(y_f), field anchored at female VA
  Φ fit on OASIS-900 (same class as residual_per_image_gender).

Outputs under results/.../residual_field_topology/ (optional review copies under testfig/).
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
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from analysis_population_bridge_suite import AffineMap, fit_affine  # noqa: E402
from config import (  # noqa: E402
    OASIS_SCORES_CSV,
    RESULTS_GENDER,
    VALENCE_AROUSAL_SCALE_MAX,
    VALENCE_AROUSAL_SCALE_MIN,
)
from dataset import load_oasis_meta  # noqa: E402

VA_MIN = VALENCE_AROUSAL_SCALE_MIN
VA_MAX = VALENCE_AROUSAL_SCALE_MAX
OUT_DIR = RESULTS_GENDER / "population_bridge_analysis" / "residual_field_topology"
TESTFIG = PROJECT_ROOT / "testfig"

# direction → (source cols, residual Δ cols relative labels in df)
DIR_SPECS = {
    "MtoF": {
        "label": "M→F",
        "resid_formula": "y_f − Φ_mf(y_m)",
        "src_v": "valence_male",
        "src_a": "arousal_male",
        "src_axis": "male / source",
        "res_dv": "res_mf_dv",
        "res_da": "res_mf_da",
        "resid_l2": "resid_mf_l2",
        "raw_dv": "raw_mf_dv",
        "raw_da": "raw_mf_da",
        "color": "#1565c0",
    },
    "FtoM": {
        "label": "F→M",
        "resid_formula": "y_m − Φ_fm(y_f)",
        "src_v": "valence_female",
        "src_a": "arousal_female",
        "src_axis": "female / source",
        "res_dv": "res_fm_dv",
        "res_da": "res_fm_da",
        "resid_l2": "resid_fm_l2",
        "raw_dv": "raw_fm_dv",
        "raw_da": "raw_fm_da",
        "color": "#c62828",
    },
}


def make_grid(step: float) -> np.ndarray:
    return np.arange(VA_MIN + step / 2, VA_MAX, step)


def local_vec(
    v_ref: np.ndarray,
    a_ref: np.ndarray,
    vec: np.ndarray,
    g: np.ndarray,
    *,
    radius: float,
    min_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_g = len(g)
    dx = np.full((n_g, n_g), np.nan)
    dy = np.full((n_g, n_g), np.nan)
    cnt = np.zeros((n_g, n_g), dtype=int)
    r2 = radius * radius
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
    return dx, dy, cnt


def field_curl(dx: np.ndarray, dy: np.ndarray, step: float) -> np.ndarray:
    dux_dy, dux_dx = np.gradient(dx, step, step)
    duy_dy, duy_dx = np.gradient(dy, step, step)
    curl = duy_dx - dux_dy
    curl[~np.isfinite(dx) | ~np.isfinite(dy)] = np.nan
    return curl


def prepare_stream(
    dx: np.ndarray, dy: np.ndarray, *, sigma: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(dx) & np.isfinite(dy)
    dx0 = np.where(valid, dx, 0.0)
    dy0 = np.where(valid, dy, 0.0)
    w = valid.astype(float)
    w_s = gaussian_filter(w, sigma=sigma, mode="nearest")
    dx_s = gaussian_filter(dx0, sigma=sigma, mode="nearest") / np.maximum(w_s, 1e-6)
    dy_s = gaussian_filter(dy0, sigma=sigma, mode="nearest") / np.maximum(w_s, 1e-6)
    dx_s[w_s < 0.15] = 0.0
    dy_s[w_s < 0.15] = 0.0
    return dx_s, dy_s, valid


def local_maxima_mask(abs_curl: np.ndarray, *, footprint: int = 3) -> np.ndarray:
    fill = np.nan_to_num(abs_curl, nan=-np.inf)
    neigh = maximum_filter(fill, size=footprint, mode="nearest")
    return np.isfinite(abs_curl) & (fill == neigh) & (fill > 0)


def build_frame() -> tuple[pd.DataFrame, dict]:
    oasis = load_oasis_meta(OASIS_SCORES_CSV)
    cols = ["valence_male", "arousal_male", "valence_female", "arousal_female"]
    df = oasis[oasis[cols].notna().all(axis=1)].copy().reset_index(drop=True)
    Xm = df[["valence_male", "arousal_male"]].to_numpy(float)
    Xf = df[["valence_female", "arousal_female"]].to_numpy(float)

    phi_mf: AffineMap = fit_affine(Xm, Xf)
    phi_fm: AffineMap = fit_affine(Xf, Xm)
    pred_f = phi_mf.apply(Xm)   # Φ_mf(y_m) ≈ y_f
    pred_m = phi_fm.apply(Xf)   # Φ_fm(y_f) ≈ y_m

    raw_mf = Xf - Xm
    raw_fm = Xm - Xf
    res_mf = Xf - pred_f
    res_fm = Xm - pred_m

    df = df.assign(
        raw_mf_dv=raw_mf[:, 0],
        raw_mf_da=raw_mf[:, 1],
        raw_fm_dv=raw_fm[:, 0],
        raw_fm_da=raw_fm[:, 1],
        res_mf_dv=res_mf[:, 0],
        res_mf_da=res_mf[:, 1],
        res_fm_dv=res_fm[:, 0],
        res_fm_da=res_fm[:, 1],
        resid_mf_l2=np.linalg.norm(res_mf, axis=1),
        resid_fm_l2=np.linalg.norm(res_fm, axis=1),
        gap_mf_l2=np.linalg.norm(raw_mf, axis=1),
        gap_fm_l2=np.linalg.norm(raw_fm, axis=1),
    )
    meta = {
        "phi_mf": phi_mf.to_dict(),
        "phi_fm": phi_fm.to_dict(),
        "n": int(len(df)),
    }
    return df, meta


def analyze_direction(
    df: pd.DataFrame,
    g: np.ndarray,
    direction: str,
    *,
    grid_step: float,
    radius: float,
    min_n: int,
    top_peaks: int,
    per_peak: int,
    curl_q: float,
) -> dict:
    spec = DIR_SPECS[direction]
    v_ref = df[spec["src_v"]].to_numpy(float)
    a_ref = df[spec["src_a"]].to_numpy(float)
    vec = df[[spec["res_dv"], spec["res_da"]]].to_numpy(float)
    raw = df[[spec["raw_dv"], spec["raw_da"]]].to_numpy(float)

    dx, dy, cnt = local_vec(v_ref, a_ref, vec, g, radius=radius, min_n=min_n)
    dx_r, dy_r, cnt_r = local_vec(v_ref, a_ref, raw, g, radius=radius, min_n=min_n)
    curl = field_curl(dx, dy, grid_step)
    curl_r = field_curl(dx_r, dy_r, grid_step)
    abs_curl = np.abs(curl)
    peak_mask = local_maxima_mask(abs_curl, footprint=3)
    thr = float(np.nanquantile(abs_curl, curl_q))
    peak_ijs = np.argwhere(peak_mask & (abs_curl >= thr))
    if peak_ijs.size == 0:
        peak_ijs = np.argwhere(peak_mask)
    sc = np.array([abs_curl[i, j] for i, j in peak_ijs]) if len(peak_ijs) else np.array([])
    if len(sc):
        keep = np.argsort(sc)[::-1][:top_peaks]
        peak_ijs = peak_ijs[keep]
        peaks_xy = np.column_stack([g[peak_ijs[:, 1]], g[peak_ijs[:, 0]]])
    else:
        peaks_xy = np.zeros((0, 2))

    # candidates near peaks (distance in source VA)
    rows: list[dict] = []
    used: set[str] = set()
    for rank, (i, j) in enumerate(peak_ijs, start=1):
        vc, ac = float(g[j]), float(g[i])
        cval = float(abs_curl[i, j])
        d2 = (df[spec["src_v"]] - vc) ** 2 + (df[spec["src_a"]] - ac) ** 2
        near = df.loc[d2 <= radius * radius].copy()
        near = near.assign(
            direction=direction,
            dist_to_peak=np.sqrt(d2.loc[near.index].to_numpy()),
            peak_rank=rank,
            peak_v=vc,
            peak_a=ac,
            peak_abs_curl=cval,
            resid_l2=near[spec["resid_l2"]],
        )
        near = near.sort_values(["resid_l2", "dist_to_peak"], ascending=[False, True])
        for _, r in near.head(per_peak).iterrows():
            iid = str(r["image_id"])
            if iid in used:
                continue
            used.add(iid)
            rows.append(r.to_dict())
    cands = pd.DataFrame(rows)
    if not cands.empty:
        cands = cands.sort_values(["peak_rank", "resid_l2"], ascending=[True, False]).reset_index(drop=True)

    return {
        "direction": direction,
        "spec": spec,
        "dx": dx,
        "dy": dy,
        "cnt": cnt,
        "abs_curl": abs_curl,
        "raw_dx": dx_r,
        "raw_dy": dy_r,
        "raw_abs_curl": np.abs(curl_r),
        "peaks_xy": peaks_xy,
        "peak_ijs": peak_ijs,
        "cands": cands,
        "curl_stats": {
            "median": float(np.nanmedian(abs_curl)),
            "q90": float(np.nanquantile(abs_curl, 0.9)),
            "max": float(np.nanmax(abs_curl)),
        },
        "raw_curl_stats": {
            "median": float(np.nanmedian(np.abs(curl_r))),
            "q90": float(np.nanquantile(np.abs(curl_r), 0.9)),
            "max": float(np.nanmax(np.abs(curl_r))),
        },
    }


def _draw_field_panel(
    ax,
    g: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    abs_curl: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    peak_xy: np.ndarray | None = None,
    vmax: float | None = None,
) -> object:
    xg, yg = np.meshgrid(g, g, indexing="xy")
    dx_s, dy_s, valid = prepare_stream(dx, dy)
    if vmax is None:
        vmax = float(np.nanquantile(abs_curl, 0.98)) if np.any(np.isfinite(abs_curl)) else 0.1
        vmax = max(vmax, 1e-4)
    im = ax.pcolormesh(
        xg, yg, abs_curl, shading="auto", cmap="magma", vmin=0, vmax=vmax, zorder=1,
    )
    step_q = max(1, len(g) // 12)
    sl = (slice(None, None, step_q), slice(None, None, step_q))
    m = valid[sl]
    ax.quiver(
        xg[sl][m], yg[sl][m], dx[sl][m], dy[sl][m],
        color="white", alpha=0.85, angles="xy", scale_units="xy", scale=2.2,
        width=0.0035, zorder=3,
    )
    try:
        ax.streamplot(
            g, g, dx_s, dy_s,
            color="#80cbc4", density=1.15, linewidth=0.9,
            arrowsize=0.7, zorder=2, broken_streamlines=True,
        )
    except Exception as exc:  # noqa: BLE001
        ax.text(0.02, 0.98, f"streamplot failed: {exc}", transform=ax.transAxes,
                va="top", fontsize=7, color="w")
    if peak_xy is not None and len(peak_xy):
        ax.scatter(
            peak_xy[:, 0], peak_xy[:, 1], s=55, facecolors="none",
            edgecolors="#00e5ff", linewidths=1.4, zorder=5, label="rotational peaks",
        )
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_xlim(VA_MIN, VA_MAX)
    ax.set_ylim(VA_MIN, VA_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    return im


def _draw_cand_strip(ax, cands: pd.DataFrame, title: str, n: int = 8) -> None:
    ax.set_axis_off()
    if cands is None or cands.empty:
        ax.text(0.5, 0.5, "No candidates", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        return
    n = min(n, len(cands))
    inner = ax.inset_axes([0.02, 0.08, 0.96, 0.84])
    inner.set_axis_off()
    for k in range(n):
        r = cands.iloc[k]
        path = Path(str(r["image_path"]))
        ax_i = inner.inset_axes([k / n + 0.008, 0.22, 0.92 / n, 0.72])
        if path.exists():
            ax_i.imshow(Image.open(path).convert("RGB"))
        ax_i.axis("off")
        stem = Path(str(r.get("image_filename", path.name))).stem
        ax_i.set_title(
            f"{r['image_id']} {stem[:12]}\n"
            f"#{int(r['peak_rank'])} curl={r['peak_abs_curl']:.2f}\n"
            f"resid={r['resid_l2']:.2f}",
            fontsize=5.8,
        )
    ax.set_title(title, fontsize=9)


def plot_bidirectional(
    g: np.ndarray,
    packs: dict[str, dict],
    out: Path,
    *,
    dpi: int = 200,
) -> None:
    """A/B residual fields both directions; C/D candidate strips."""
    fig = plt.figure(figsize=(12.4, 10.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.95], hspace=0.30, wspace=0.26)

    # shared color scale for residual |curl|
    vmax = max(
        float(np.nanquantile(packs["MtoF"]["abs_curl"], 0.98)),
        float(np.nanquantile(packs["FtoM"]["abs_curl"], 0.98)),
        1e-4,
    )

    for col, key, panel in [(0, "MtoF", "A"), (1, "FtoM", "B")]:
        p = packs[key]
        sp = p["spec"]
        ax = fig.add_subplot(gs[0, col])
        im = _draw_field_panel(
            ax, g, p["dx"], p["dy"], p["abs_curl"],
            title=(
                f"{panel}  Residual field {sp['label']}: {sp['resid_formula']}\n"
                f"|curl| q90={p['curl_stats']['q90']:.2f}; peaks={len(p['peaks_xy'])}"
            ),
            xlabel=f"Valence ({sp['src_axis']})",
            ylabel=f"Arousal ({sp['src_axis']})",
            peak_xy=p["peaks_xy"],
            vmax=vmax,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).set_label("|curl|")

    for col, key, panel in [(0, "MtoF", "C"), (1, "FtoM", "D")]:
        p = packs[key]
        ax = fig.add_subplot(gs[1, col])
        _draw_cand_strip(
            ax, p["cands"],
            f"{panel}  {p['spec']['label']} candidates near high-|curl| peaks",
            n=8,
        )

    fig.suptitle(
        "Supp | Residual VA vector-field topology — both transfer directions\n"
        "Φ_mf / Φ_fm = OASIS-900 affine; fields anchored at source VA (grid=0.1, Fig.2-D matched)",
        fontsize=11, y=0.98,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def plot_single_direction_legacy(
    g: np.ndarray,
    pack: dict,
    out: Path,
    *,
    dpi: int = 200,
) -> None:
    """Keep M→F-only layout as Fig_residual_field_topology_prototype.png."""
    fig = plt.figure(figsize=(12.2, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.95], hspace=0.32, wspace=0.28)
    sp = pack["spec"]

    ax0 = fig.add_subplot(gs[0, 0])
    im0 = _draw_field_panel(
        ax0, g, pack["raw_dx"], pack["raw_dy"], pack["raw_abs_curl"],
        title=f"A  Raw gap ({sp['label']}) on source VA\n(mostly global affine)",
        xlabel=f"Valence ({sp['src_axis']})",
        ylabel=f"Arousal ({sp['src_axis']})",
    )
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02).set_label("|curl|")

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = _draw_field_panel(
        ax1, g, pack["dx"], pack["dy"], pack["abs_curl"],
        title=f"B  Residual {sp['resid_formula']}\nstreamlines + |curl| peaks",
        xlabel=f"Valence ({sp['src_axis']})",
        ylabel=f"Arousal ({sp['src_axis']})",
        peak_xy=pack["peaks_xy"],
    )
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02).set_label("|curl|")

    axc = fig.add_subplot(gs[1, :])
    _draw_cand_strip(
        axc, pack["cands"],
        f"C  {sp['label']} candidates near high-|curl| peaks",
        n=10,
    )
    fig.suptitle(
        f"Supp | Residual VA vector-field topology ({sp['label']})\n"
        f"Φ = OASIS-900 affine; anchored at {sp['src_axis']}; grid=0.1",
        fontsize=11, y=0.98,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-step", type=float, default=0.1)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--min-n", type=int, default=6)
    ap.add_argument("--top-peaks", type=int, default=6)
    ap.add_argument("--per-peak", type=int, default=4)
    ap.add_argument("--curl-q", type=float, default=0.85)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out-dir", type=Path, default=TESTFIG)
    ap.add_argument(
        "--directions",
        type=str,
        default="both",
        choices=["both", "MtoF", "FtoM"],
    )
    args = ap.parse_args()

    df, meta = build_frame()
    g = make_grid(args.grid_step)
    dirs = ["MtoF", "FtoM"] if args.directions == "both" else [args.directions]

    packs: dict[str, dict] = {}
    for d in dirs:
        packs[d] = analyze_direction(
            df, g, d,
            grid_step=args.grid_step,
            radius=args.radius,
            min_n=args.min_n,
            top_peaks=args.top_peaks,
            per_peak=args.per_peak,
            curl_q=args.curl_q,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # per-direction candidate CSVs + combined
    all_cands = []
    for d, p in packs.items():
        c = p["cands"]
        keep = [
            "direction", "image_id", "image_filename", "image_path", "category",
            "valence_male", "arousal_male", "valence_female", "arousal_female",
            "resid_l2", "peak_rank", "peak_v", "peak_a", "peak_abs_curl", "dist_to_peak",
        ]
        for col in keep:
            if col not in c.columns:
                c[col] = np.nan
        path = OUT_DIR / f"residual_curl_candidate_images_{d}.csv"
        c[keep].to_csv(path, index=False)
        print(f"Saved {path} (n={len(c)})")
        all_cands.append(c[keep])
    if all_cands:
        comb = pd.concat(all_cands, axis=0, ignore_index=True)
        comb_path = OUT_DIR / "residual_curl_candidate_images_both.csv"
        comb.to_csv(comb_path, index=False)
        # backward-compatible M→F default name
        if "MtoF" in packs:
            packs["MtoF"]["cands"][keep].to_csv(
                OUT_DIR / "residual_curl_candidate_images.csv", index=False,
            )

    summary = {
        "estimand": "residual_vector_field_topology_v2_bidirectional",
        "note": (
            "Residual fields for both affine transfers on OASIS-900. "
            "M→F anchored at male VA; F→M at female VA. Exploratory Supp."
        ),
        "n_images": meta["n"],
        "grid_step": args.grid_step,
        "radius": args.radius,
        "min_n": args.min_n,
        "phi_mf": meta["phi_mf"],
        "phi_fm": meta["phi_fm"],
        "directions": {},
    }
    npz_kw: dict = {"grid": g}
    for d, p in packs.items():
        summary["directions"][d] = {
            "resid_formula": p["spec"]["resid_formula"],
            "source_axis": p["spec"]["src_axis"],
            "phi_abs_curl": p["curl_stats"],
            "raw_abs_curl": p["raw_curl_stats"],
            "n_peaks": int(len(p["peaks_xy"])),
            "n_candidates": int(len(p["cands"])),
            "peak_locations": [
                {
                    "v": float(v),
                    "a": float(a),
                    "abs_curl": float(p["abs_curl"][i, j]),
                }
                for (i, j), (v, a) in zip(p["peak_ijs"], p["peaks_xy"])
            ],
        }
        npz_kw[f"{d}_dx"] = p["dx"]
        npz_kw[f"{d}_dy"] = p["dy"]
        npz_kw[f"{d}_abs_curl"] = p["abs_curl"]
        npz_kw[f"{d}_cnt"] = p["cnt"]
        # legacy keys for M→F
        if d == "MtoF":
            npz_kw["phi_dx"] = p["dx"]
            npz_kw["phi_dy"] = p["dy"]
            npz_kw["phi_abs_curl"] = p["abs_curl"]
            npz_kw["phi_cnt"] = p["cnt"]
            npz_kw["raw_dx"] = p["raw_dx"]
            npz_kw["raw_dy"] = p["raw_dy"]
            npz_kw["raw_abs_curl"] = p["raw_abs_curl"]

    (OUT_DIR / "residual_field_topology_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    np.savez_compressed(OUT_DIR / "residual_field_topology_grids.npz", **npz_kw)

    if args.directions == "both" and set(packs) >= {"MtoF", "FtoM"}:
        plot_bidirectional(
            g, packs,
            args.out_dir / "Fig_residual_field_topology_both.png",
            dpi=args.dpi,
        )
        plot_bidirectional(
            g, packs,
            OUT_DIR / "Fig_residual_field_topology_both.png",
            dpi=args.dpi,
        )

    # also refresh per-direction prototypes
    for d, p in packs.items():
        stem = (
            "Fig_residual_field_topology_prototype.png"
            if d == "MtoF"
            else f"Fig_residual_field_topology_{d}.png"
        )
        plot_single_direction_legacy(g, p, args.out_dir / stem, dpi=args.dpi)
        plot_single_direction_legacy(g, p, OUT_DIR / stem, dpi=args.dpi)
        print(
            f"[{d}] |curl| resid q90={p['curl_stats']['q90']:.3f}  "
            f"peaks={len(p['peaks_xy'])} cands={len(p['cands'])}"
        )


if __name__ == "__main__":
    main()
