#!/usr/bin/env python3
"""Visualize Φ prediction asymmetry: not a transpose of M↔F flows.

Shows (1) which VA-cell transitions break transpose symmetry,
(2) Φ displacement fields M→F vs F→M on VA,
(3) round-trip residuals (Φ_fm∘Φ_mf ≠ Id),
(4) concrete OASIS images on the most asymmetric flow edges.

Output: testfig/Fig_phi_prediction_asymmetry_va_images.png (+ .svg, .json)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from analysis_population_bridge_suite import fit_affine  # noqa: E402
from config import OASIS_SCORES_CSV  # noqa: E402
from dataset import load_oasis_meta  # noqa: E402

OUT = PROJECT_ROOT / "testfig"
NBINS = 3
EDGES = np.linspace(1.0, 7.0, NBINS + 1)
LAB1 = ["L", "M", "H"]
COL_MF = "#1565C0"
COL_FM = "#C62828"


def cell_labels() -> list[str]:
    return [f"V{LAB1[iv]}A{LAB1[ia]}" for iv in range(NBINS) for ia in range(NBINS)]


def cell_id(v: np.ndarray, a: np.ndarray) -> np.ndarray:
    vi = np.clip(np.digitize(v, EDGES) - 1, 0, NBINS - 1)
    ai = np.clip(np.digitize(a, EDGES) - 1, 0, NBINS - 1)
    return vi * NBINS + ai


def cell_center(cid: int) -> tuple[float, float]:
    iv, ia = divmod(int(cid), NBINS)
    vc = 0.5 * (EDGES[iv] + EDGES[iv + 1])
    ac = 0.5 * (EDGES[ia] + EDGES[ia + 1])
    return vc, ac


def transition_counts(src: np.ndarray, tgt: np.ndarray, n: int) -> np.ndarray:
    mat = np.zeros((n, n), float)
    for s, t in zip(src, tgt):
        mat[int(s), int(t)] += 1
    return mat


def _thumb(ax, path: Path | None, caption: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("0.7")
    if path is not None and path.exists():
        try:
            ax.imshow(mpimg.imread(str(path)))
        except Exception:
            ax.set_facecolor("0.92")
            ax.text(0.5, 0.5, "(unreadable)", ha="center", va="center", fontsize=7, transform=ax.transAxes)
    else:
        ax.set_facecolor("0.92")
        ax.text(0.5, 0.5, "(missing)", ha="center", va="center", fontsize=7, transform=ax.transAxes)
    ax.set_title(caption, fontsize=6.2, pad=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=OUT)
    ap.add_argument("--top-edges", type=int, default=3)
    ap.add_argument("--imgs-per-edge", type=int, default=2)
    ap.add_argument(
        "--exemplar-mode",
        choices=["all", "non_person", "both"],
        default="both",
        help="all=allow Person/Nude; non_person=exclude; both=write two figure versions",
    )
    args = ap.parse_args()

    oasis = load_oasis_meta(OASIS_SCORES_CSV)
    cols = ["valence_male", "arousal_male", "valence_female", "arousal_female"]
    df = oasis[oasis[cols].notna().all(axis=1)].copy().reset_index(drop=True)
    ym = df[["valence_male", "arousal_male"]].to_numpy(float)
    yf = df[["valence_female", "arousal_female"]].to_numpy(float)

    phi_mf = fit_affine(ym, yf)
    phi_fm = fit_affine(yf, ym)
    pred_f = phi_mf.apply(ym)   # M→F prediction
    pred_m = phi_fm.apply(yf)   # F→M prediction
    # round-trips
    rt_m = phi_fm.apply(pred_f)  # Φ_fm(Φ_mf(y_m))
    rt_f = phi_mf.apply(pred_m)  # Φ_mf(Φ_fm(y_f))
    err_rt_m = np.linalg.norm(rt_m - ym, axis=1)
    err_rt_f = np.linalg.norm(rt_f - yf, axis=1)
    err_mf = np.linalg.norm(pred_f - yf, axis=1)
    err_fm = np.linalg.norm(pred_m - ym, axis=1)

    n_cells = NBINS * NBINS
    labels = cell_labels()
    cm = cell_id(ym[:, 0], ym[:, 1])
    cf = cell_id(yf[:, 0], yf[:, 1])
    cpf = cell_id(pred_f[:, 0], pred_f[:, 1])
    cpm = cell_id(pred_m[:, 0], pred_m[:, 1])

    P_mf = transition_counts(cm, cpf, n_cells)
    P_fm = transition_counts(cf, cpm, n_cells)
    # Asymmetry of directed edge i→j: count under M→F minus transpose partner under F→M
    A = P_mf - P_fm.T  # positive => M→F over-represents i→j relative to F→M's j→i

    # Top absolute asymmetric edges (off-diagonal preferred, but diagonal ok too)
    edges = []
    for i in range(n_cells):
        for j in range(n_cells):
            d = float(A[i, j])
            if abs(d) < 1:
                continue
            edges.append(
                {
                    "i": i,
                    "j": j,
                    "src": labels[i],
                    "tgt": labels[j],
                    "P_mf_ij": int(P_mf[i, j]),
                    "P_fm_ji": int(P_fm[j, i]),
                    "delta": d,
                    "abs_delta": abs(d),
                }
            )
    edges.sort(key=lambda e: -e["abs_delta"])
    top = edges[: args.top_edges]

    # Collect example images for each top edge
    def pick(idxs, prefer_mf: bool, mode: str):
        if len(idxs) == 0:
            return []
        safe, risky = [], []
        for k in idxs:
            cat = str(df.loc[k, "category"])
            theme = str(df.loc[k, "theme"])
            is_risky = (
                cat == "Person"
                or bool(re.search(r"nude|face|pose", theme, flags=re.I))
            )
            (risky if is_risky else safe).append(k)
        if mode == "non_person":
            pool = np.asarray(safe, dtype=int)
            if pool.size == 0:
                return []
        elif mode == "person_focus":
            # prefer Person/Nude/face/pose when available
            pool = np.asarray(risky if risky else safe, dtype=int)
        else:  # all: residual-ranked, no category filter
            pool = np.asarray(idxs, dtype=int)
        if pool.size == 0:
            return []
        if prefer_mf:
            order = np.argsort(-(err_mf[pool]))
        else:
            order = np.argsort(-(err_fm[pool]))
        chosen = pool[order[: args.imgs_per_edge]]
        out = []
        for k in chosen:
            out.append(
                {
                    "row": int(k),
                    "theme": str(df.loc[k, "theme"]),
                    "category": str(df.loc[k, "category"]),
                    "path": str(df.loc[k, "image_path"]),
                    "ym": ym[k].tolist(),
                    "yf": yf[k].tolist(),
                    "pred_f": pred_f[k].tolist(),
                    "pred_m": pred_m[k].tolist(),
                    "err_mf": float(err_mf[k]),
                    "err_fm": float(err_fm[k]),
                    "err_rt_m": float(err_rt_m[k]),
                    "safe_non_person": str(df.loc[k, "category"]) != "Person"
                    and not bool(re.search(r"nude|face|pose", str(df.loc[k, "theme"]), flags=re.I)),
                }
            )
        return out

    def build_exemplars(mode: str) -> list[dict]:
        out = []
        for e in top:
            i, j = e["i"], e["j"]
            idx_mf = np.where((cm == i) & (cpf == j))[0]
            idx_fm = np.where((cf == j) & (cpm == i))[0]
            out.append(
                {
                    **e,
                    "examples_MtoF_edge": pick(idx_mf, prefer_mf=True, mode=mode),
                    "examples_FtoM_edge": pick(idx_fm, prefer_mf=False, mode=mode),
                }
            )
        return out

    if args.exemplar_mode == "both":
        modes = ["all", "non_person", "person_focus"]
    else:
        modes = [args.exemplar_mode]

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem_map = {
        "all": "Fig_phi_prediction_asymmetry_va_images_all",
        "non_person": "Fig_phi_prediction_asymmetry_va_images_non_person",
        "person_focus": "Fig_phi_prediction_asymmetry_va_images_person_nude",
    }
    mode_title = {
        "all": "exemplars: all categories (incl. Person/Nude)",
        "non_person": "exemplars: non-Person only (no Nude/face/pose)",
        "person_focus": "exemplars: Person / Nude / face / pose focused",
    }

    for mode in modes:
        exemplars = build_exemplars(mode)
        stem = stem_map[mode]

        # -------- figure --------
        fig = plt.figure(figsize=(14.5, 11.2))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1.05, 1.05, 1.15], hspace=0.38, wspace=0.28)

        # A: asymmetry matrix
        ax = fig.add_subplot(gs[0, 0])
        vmax = max(np.abs(A).max(), 1)
        im = ax.imshow(A, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(n_cells), labels, rotation=65, ha="right", fontsize=6)
        ax.set_yticks(range(n_cells), labels, fontsize=6)
        ax.set_xlabel("target cell (j)")
        ax.set_ylabel("source cell (i)")
        ax.set_title(
            "A  Φ flow asymmetry\n"
            r"$A_{ij}=P^{\mathrm{mf}}_{ij}-P^{\mathrm{fm}}_{ji}$",
            fontsize=9,
        )
        for e in top:
            ax.plot(e["j"], e["i"], "s", ms=14, mfc="none", mec="k", mew=1.3)
            ax.text(e["j"], e["i"], f'{e["delta"]:+.0f}', ha="center", va="center",
                    fontsize=7, fontweight="bold", color="k")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("count excess (M→F vs F→Mᵀ)")

        # B: dual Φ fields on male / female VA
        ax = fig.add_subplot(gs[0, 1])
        ax.scatter(ym[:, 0], ym[:, 1], s=6, c="#90CAF9", alpha=0.35, edgecolors="none", label="y_m")
        rng = np.random.default_rng(0)
        take = rng.choice(len(df), size=min(120, len(df)), replace=False)
        d = pred_f - ym
        ax.quiver(
            ym[take, 0], ym[take, 1], d[take, 0], d[take, 1],
            angles="xy", scale_units="xy", scale=1.0, width=0.003,
            color=COL_MF, alpha=0.75, label=r"$\Phi_{mf}(y_m)-y_m$",
        )
        ax.set_xlim(1, 7)
        ax.set_ylim(1, 7)
        ax.set_aspect("equal")
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_title("B  Φ_mf displacement on male VA", fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="lower right")

        ax = fig.add_subplot(gs[0, 2])
        ax.scatter(yf[:, 0], yf[:, 1], s=6, c="#EF9A9A", alpha=0.35, edgecolors="none", label="y_f")
        d2 = pred_m - yf
        ax.quiver(
            yf[take, 0], yf[take, 1], d2[take, 0], d2[take, 1],
            angles="xy", scale_units="xy", scale=1.0, width=0.003,
            color=COL_FM, alpha=0.75, label=r"$\Phi_{fm}(y_f)-y_f$",
        )
        ax.set_xlim(1, 7)
        ax.set_ylim(1, 7)
        ax.set_aspect("equal")
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_title("C  Φ_fm displacement on female VA", fontsize=9)
        ax.legend(frameon=False, fontsize=7, loc="lower right")

        # D: round-trip residual geography (male space)
        ax = fig.add_subplot(gs[1, 0])
        sc = ax.scatter(
            ym[:, 0], ym[:, 1], c=err_rt_m, s=12, cmap="magma",
            vmin=0, vmax=np.quantile(err_rt_m, 0.98), edgecolors="none", alpha=0.85,
        )
        ax.set_xlim(1, 7)
        ax.set_ylim(1, 7)
        ax.set_aspect("equal")
        ax.set_xlabel("Valence (male)")
        ax.set_ylabel("Arousal (male)")
        ax.set_title(
            "D  Round-trip residual on male VA\n"
            r"$\|\Phi_{fm}(\Phi_{mf}(y_m))-y_m\|$"
            f"  mean={err_rt_m.mean():.2f}",
            fontsize=9,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03).set_label("round-trip L2")

        # E: directional residual imbalance
        ax = fig.add_subplot(gs[1, 1])
        delta_err = err_mf - err_fm
        lim = float(np.quantile(np.abs(delta_err), 0.98))
        sc = ax.scatter(
            0.5 * (ym[:, 0] + yf[:, 0]),
            0.5 * (ym[:, 1] + yf[:, 1]),
            c=delta_err, s=12, cmap="coolwarm", vmin=-lim, vmax=lim,
            edgecolors="none", alpha=0.85,
        )
        ax.set_xlim(1, 7)
        ax.set_ylim(1, 7)
        ax.set_aspect("equal")
        ax.set_xlabel("Valence (midpoint)")
        ax.set_ylabel("Arousal (midpoint)")
        ax.set_title(
            "E  Directional residual imbalance\n"
            r"$\|y_f-\Phi_{mf}(y_m)\| - \|y_m-\Phi_{fm}(y_f)\|$",
            fontsize=9,
        )
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03).set_label("Δ residual (M→F − F→M)")

        # F: asymmetric edges as arrows between cell centers
        ax = fig.add_subplot(gs[1, 2])
        for eedge in EDGES:
            ax.axvline(eedge, color="0.88", lw=0.7)
            ax.axhline(eedge, color="0.88", lw=0.7)
        ax.scatter(ym[:, 0], ym[:, 1], s=4, c="0.75", alpha=0.25, edgecolors="none")
        for e in top:
            x0, y0 = cell_center(e["i"])
            x1, y1 = cell_center(e["j"])
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=COL_MF if e["delta"] > 0 else COL_FM,
                    lw=1.8,
                    connectionstyle="arc3,rad=0.15",
                    alpha=0.9,
                ),
            )
            ax.text(
                0.5 * (x0 + x1), 0.5 * (y0 + y1),
                f'{e["src"]}→{e["tgt"]}\nΔ={e["delta"]:+.0f}',
                ha="center", va="center", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", alpha=0.9),
            )
        ax.set_xlim(1, 7)
        ax.set_ylim(1, 7)
        ax.set_aspect("equal")
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_title("F  Top asymmetric cell transitions\n(blue excess M→F / red excess F→Mᵀ)", fontsize=9)

        # G: image strips for top edges
        n_edges = len(exemplars)
        n_img = args.imgs_per_edge
        cols_g = 1 + n_img + n_img
        gs_g = gs[2, :].subgridspec(
            n_edges, cols_g, wspace=0.08, hspace=0.35,
            width_ratios=[0.9] + [1] * (cols_g - 1),
        )

        for r, ex in enumerate(exemplars):
            ax_lab = fig.add_subplot(gs_g[r, 0])
            ax_lab.axis("off")
            ax_lab.text(
                0.0, 0.5,
                f"Edge {r+1}\n{ex['src']} → {ex['tgt']}\n"
                f"P_mf={ex['P_mf_ij']}  P_fmᵀ={ex['P_fm_ji']}\n"
                f"Δ={ex['delta']:+.0f}",
                transform=ax_lab.transAxes, ha="left", va="center", fontsize=7.5,
                fontweight="bold",
                color=COL_MF if ex["delta"] > 0 else COL_FM,
            )
            for c, iminfo in enumerate(ex["examples_MtoF_edge"]):
                ax_i = fig.add_subplot(gs_g[r, 1 + c])
                cap = (
                    f"M→F · {iminfo['category']}\n{iminfo['theme'][:18]}\n"
                    f"ε_mf={iminfo['err_mf']:.2f}"
                )
                _thumb(ax_i, Path(iminfo["path"]), cap)
            for c in range(len(ex["examples_MtoF_edge"]), n_img):
                ax_i = fig.add_subplot(gs_g[r, 1 + c])
                ax_i.axis("off")
            for c, iminfo in enumerate(ex["examples_FtoM_edge"]):
                ax_i = fig.add_subplot(gs_g[r, 1 + n_img + c])
                cap = (
                    f"F→M · {iminfo['category']}\n{iminfo['theme'][:18]}\n"
                    f"ε_fm={iminfo['err_fm']:.2f}"
                )
                _thumb(ax_i, Path(iminfo["path"]), cap)
            for c in range(len(ex["examples_FtoM_edge"]), n_img):
                ax_i = fig.add_subplot(gs_g[r, 1 + n_img + c])
                ax_i.axis("off")

        fig.suptitle(
            "Φ prediction asymmetry is real — and localized in VA\n"
            r"Raw pair flows are exact transposes; $\Phi_{mf}$ vs $\Phi_{fm}$ cell flows are not "
            rf"($\Phi_{{fm}}\neq\Phi_{{mf}}^{{-1}}$).  [{mode_title[mode]}]",
            fontsize=11, y=0.995,
        )

        png = args.outdir / f"{stem}.png"
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(args.outdir / f"{stem}.svg", format="svg", bbox_inches="tight")
        plt.close(fig)

        if mode == "non_person":
            alias = args.outdir / "Fig_phi_prediction_asymmetry_va_images.png"
            shutil.copy2(png, alias)
            shutil.copy2(
                args.outdir / f"{stem}.svg",
                args.outdir / "Fig_phi_prediction_asymmetry_va_images.svg",
            )

        summary = {
            "exemplar_mode": mode,
            "n_images": int(len(df)),
            "nbins": NBINS,
            "phi_mf": phi_mf.to_dict(),
            "phi_fm": phi_fm.to_dict(),
            "roundtrip_mean_male": float(err_rt_m.mean()),
            "roundtrip_mean_female": float(err_rt_f.mean()),
            "residual_mean_mf": float(err_mf.mean()),
            "residual_mean_fm": float(err_fm.mean()),
            "asymmetry_frobenius_rel": float(np.linalg.norm(A) / (np.linalg.norm(P_mf) + 1e-12)),
            "top_asymmetric_edges": exemplars,
            "note": (
                "A_ij = P_mf[i,j] - P_fm[j,i]. Positive => M→F over-represents transition i→j "
                "relative to the transpose partner under F→M."
            ),
        }
        (args.outdir / f"{stem}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"[ok] {png}  ({mode_title[mode]})")
        for e in exemplars:
            print(f"  {e['src']}→{e['tgt']} Δ={e['delta']:+.0f}")
            for side in ("examples_MtoF_edge", "examples_FtoM_edge"):
                for im in e[side]:
                    print(f"    {side}: {im['category']:8s} {im['theme']}")

    print(f"round-trip mean (male space): {err_rt_m.mean():.3f}")
    print(f"residual mean M→F / F→M: {err_mf.mean():.3f} / {err_fm.mean():.3f}")


if __name__ == "__main__":
    main()
