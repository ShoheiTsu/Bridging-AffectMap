#!/usr/bin/env python3
"""
Fig.6 discovery panels: (A) non-Person equivalence exemplars,
(B) VA spatial co-localization of Φ residual vs translation efficacy,
(C) semantic-category flow among equivalent pairs (4×4 transition).

Do not plot/claim scalar residual×improvement ρ≈0.90.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from config import OASIS_SCORES_CSV  # noqa: E402
from dataset import load_oasis_meta  # noqa: E402

CVAE = ROOT / "results" / "cvae_cross_gender"
PBA = ROOT / "results" / "population_bridge_analysis"
OUT_RES = ROOT / "results" / "equivalence_nontriviality"
KNN_CSV_FIXED = CVAE / "emotion_equivalent_pairs_fixedsplit_knn.csv"
KNN_CSV_CV = CVAE / "emotion_equivalent_pairs_themecv_knn.csv"
PAIRS_FIXED = CVAE / "emotion_equivalent_pairs_fixedsplit.csv"
PAIRS_CV = CVAE / "emotion_equivalent_pairs_themecv.csv"
PAIRS_CV_META = CVAE / "emotion_equivalent_pairs_themecv_meta.json"
# Prefer theme-CV for all Fig.6 panels (A/B/C).
KNN_CSV = KNN_CSV_CV if KNN_CSV_CV.exists() else KNN_CSV_FIXED
PAIRS_PRIMARY = PAIRS_CV if PAIRS_CV.exists() else PAIRS_FIXED
VA_LIM = (1.5, 7.0)  # legacy binning (unused by main Fig.6 B)
VA_LIM_A = (1.0, 7.0)  # Panel A/B VA maps: scale 1–7, aspect 1:1
CAT_ORDER = ["Scene", "Person", "Object", "Animal"]

RISK_RE = re.compile(r"nude|face|pose", re.I)
PREFERRED_CAT = {"Object", "Scene", "Animal"}
N_MATCH_PER_HUB = 3
N_HUBS_PER_DIR = 1  # one illustrative hub per direction (F→M and M→F)
# Source gender colours (user convention): female = red, male = blue
COL_FEMALE_SRC = "#c62828"
COL_MALE_SRC = "#1565c0"

# Preferred shared images for left VA (same image → different F vs M matches)
PREFERRED_SHARED_SOURCES = [
    "Flowers 3",
    "Bird 4",
    "Alcohol 8",
    "Lamb 1",
    "Zebra 1",
    "Solar panel 1",
]

# Spread sources for Fig.6-A left (VA-diverse; source thumbnails only)
PREFERRED_SPREAD_SOURCES = [
    ("FtoM", "Flood 3"),   # low-V / mid-A
    ("FtoM", "Snow 2"),    # mid-V / low-A
    ("FtoM", "Bird 4"),    # high-V / mid-A
    ("MtoF", "Crow 2"),    # mid-V / low-A (male source)
]

# Extra 1→3 exemplar hubs (paper candidates; risk-filtered at export)
PREFERRED_1TO3_EXEMPLARS: dict[str, list[str]] = {
    "FtoM": [
        "Bird 4", "Flood 3", "Snow 2", "Dessert 4", "Horse racing 1",
        "Parasailing 3", "Lion 1", "Dog 22", "Skyscraper 1", "Spider 1",
    ],
    "MtoF": [
        "Crow 2", "Bridge 1", "Cold 8", "Bird 2", "Dog 12",
        "Weapon 1", "Funeral 1", "Wedding 11", "Shark 11", "Bridge 2",
    ],
}


# Preferred theme bases for diversity (order = preference)
SEED_PAIRS = [
    ("Seal 1", "Bar 2"),
    ("Money 1", "Street 3"),
    ("Gun 7", "Destruction 5"),  # avoid face-like Statue exemplars
    ("Skyscraper 2", "Statue 2"),
    ("Pigeon 4", "Wedding 4"),
]

# Named alternative sets for manual layout trials (theme pairs)
EXEMPLAR_SETS: dict[str, list[tuple[str, str]]] = {
    "default": [
        ("Seal 1", "Bar 2"),
        ("Money 1", "Street 3"),
        ("Gun 7", "Destruction 5"),
    ],
    "alt_statue": [
        ("Seal 1", "Bar 2"),
        ("Money 1", "Street 3"),
        ("Skyscraper 2", "Statue 2"),
    ],
    "alt_pigeon_wedding": [
        ("Seal 1", "Bar 2"),
        ("Money 1", "Street 3"),
        ("Pigeon 4", "Wedding 4"),
    ],
    "pleasant_neutral": [
        ("Road 1", "Present 1"),
        ("Flowers 10", "Dock 1"),
        ("Paperclips 2", "Grass 5"),
    ],
    "unpleasant": [
        ("Gun 7", "Destruction 5"),
        ("Car accident 3", "Explosion 6"),
        ("Snake 1", "Soldiers 10"),
    ],
    "mixed_spread": [
        ("Bird 4", "Beach 4"),
        ("Snow 2", "Bark 3"),
        ("Flood 3", "Plane crash 4"),
    ],
    "object_animal": [
        ("Alcohol 5", "Penguins 1"),
        ("Pigeon 4", "Wedding 4"),
        ("Seal 1", "Bar 2"),
    ],
}


def _risk_row(r: pd.Series) -> bool:
    if r["i_category"] == "Person" or r["j_category"] == "Person":
        return True
    return bool(RISK_RE.search(str(r["i_theme"])) or RISK_RE.search(str(r["j_theme"])))


def build_merged() -> pd.DataFrame:
    """Merge equivalence pairs with Φ residual for panel B (prefer theme-CV)."""
    pairs = pd.read_csv(PAIRS_PRIMARY)
    resid = pd.read_csv(PBA / "residual_per_image_gender.csv").set_index("image_id")["residual_l2"]
    m = pairs.copy()
    m["phi_residual"] = m["i_image_id"].map(resid)
    m["translation_improvement"] = m["phi_residual"] - m["distance_l2"]
    m["beats_phi"] = m["translation_improvement"] > 0
    m["risky"] = m.apply(_risk_row, axis=1)
    return m.dropna(subset=["phi_residual"]).reset_index(drop=True)


def load_category_flow_pairs() -> tuple[pd.DataFrame, dict]:
    """Prefer theme-CV n=900 pairs for panel C; fall back to fixedsplit."""
    if PAIRS_CV.exists():
        m = pd.read_csv(PAIRS_CV)
        meta = {}
        if PAIRS_CV_META.exists():
            meta = json.loads(PAIRS_CV_META.read_text(encoding="utf-8"))
        return m, meta
    return pd.read_csv(PAIRS_FIXED), {"mode": "fixedsplit", "note": "CV CSV missing; using fixedsplit"}


def load_knn() -> pd.DataFrame:
    path = KNN_CSV_CV if KNN_CSV_CV.exists() else KNN_CSV_FIXED
    if not path.exists():
        raise FileNotFoundError(
            f"Missing knn CSV ({path.name}). Run:\n"
            "  python3 code/analysis_emotion_equivalent_pairs.py --mode themecv --knn-k 5"
        )
    return pd.read_csv(path)


def select_exemplars(
    m: pd.DataFrame,
    n: int = 3,
    seed_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Prefer curated theme pairs; fall back to safest cross-category by distance."""
    cross = m[(~m["same_category"]) & (~m["risky"])].copy()
    picks: list[pd.Series] = []
    used_bases: set[str] = set()
    seeds = seed_pairs if seed_pairs is not None else SEED_PAIRS

    def base(theme: str) -> str:
        parts = str(theme).rsplit(" ", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else str(theme)

    for ti, tj in seeds:
        if len(picks) >= n:
            break
        hit = cross[(cross["i_theme"] == ti) & (cross["j_theme"] == tj)]
        if hit.empty:
            hit = cross[(cross["i_theme"] == tj) & (cross["j_theme"] == ti)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        bi, bj = base(row["i_theme"]), base(row["j_theme"])
        if bi in used_bases or bj in used_bases:
            continue
        picks.append(row)
        used_bases.update({bi, bj})

    if len(picks) < n:
        rest = cross.sort_values("distance_l2")
        for _, row in rest.iterrows():
            if len(picks) >= n:
                break
            bi, bj = base(row["i_theme"]), base(row["j_theme"])
            if bi in used_bases or bj in used_bases:
                continue
            if row["i_category"] not in PREFERRED_CAT or row["j_category"] not in PREFERRED_CAT:
                continue
            picks.append(row)
            used_bases.update({bi, bj})

    return pd.DataFrame(picks)


def _thumb(ax, path: Path | None, title: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if path is not None and path.exists():
        ax.imshow(mpimg.imread(path))
    else:
        ax.set_facecolor("#eceff1")
        ax.text(0.5, 0.5, "(image n/a)", ha="center", va="center", fontsize=8, transform=ax.transAxes)
    ax.set_title(title, fontsize=7.5, pad=2)


def _load_rgb(path: Path | None, size: int = 56) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    img = mpimg.imread(str(path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    h, w = img.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    crop = img[y0:y0 + side, x0:x0 + side]
    yy = (np.linspace(0, side - 1, size)).astype(int)
    xx = (np.linspace(0, side - 1, size)).astype(int)
    return crop[yy][:, xx]


def _risk_theme(theme: str, category: str) -> bool:
    if category == "Person":
        return True
    return bool(RISK_RE.search(str(theme)))


def select_hub_bundles(
    knn: pd.DataFrame,
    *,
    direction: str,
    n_hubs: int = N_HUBS_PER_DIR,
    n_match: int = N_MATCH_PER_HUB,
    preferred_hubs: list[str] | None = None,
) -> list[dict]:
    """Pick hubs with several safe cross-category matches (illustrative)."""
    d = knn[knn["direction"] == direction].copy()
    if d.empty:
        return []

    bundles: list[dict] = []
    used_hub: set[str] = set()
    used_match: set[str] = set()
    used_bases: set[str] = set()

    def base(theme: str) -> str:
        parts = str(theme).rsplit(" ", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else str(theme)

    # Prefer curated hub themes, then closest rank-1 hubs
    hubs = d[d["rank"] == 1].copy()
    pref = preferred_hubs or []
    hubs["_pref"] = hubs["hub_theme"].astype(str).map(
        lambda t: pref.index(t) if t in pref else len(pref)
    )
    hubs = hubs.sort_values(["_pref", "distance_l2"]).reset_index(drop=True)
    for _, hub_row in hubs.iterrows():
        if len(bundles) >= n_hubs:
            break
        hid = str(hub_row["hub_image_id"])
        if hid in used_hub:
            continue
        if _risk_theme(hub_row["hub_theme"], hub_row["hub_category"]):
            continue
        hb = base(hub_row["hub_theme"])
        if hb in used_bases:
            continue

        cand = d[d["hub_image_id"] == hid].sort_values(["rank", "distance_l2"])
        matches: list[pd.Series] = []
        for _, mrow in cand.iterrows():
            mid = str(mrow["match_image_id"])
            if mid == hid or mid in used_match:
                continue
            if _risk_theme(mrow["match_theme"], mrow["match_category"]):
                continue
            if bool(mrow["same_category"]):
                continue  # prefer meaning-crossing matches for illustration
            mb = base(mrow["match_theme"])
            if mb in used_bases or mb == hb:
                continue
            matches.append(mrow)
            if len(matches) >= n_match:
                break
        if len(matches) < max(2, n_match - 1):
            continue

        used_hub.add(hid)
        used_bases.add(hb)
        for mrow in matches:
            used_match.add(str(mrow["match_image_id"]))
            used_bases.add(base(mrow["match_theme"]))
        bundles.append({
            "direction": direction,
            "hub": hub_row,
            "matches": matches[:n_match],
        })
    return bundles


def select_spread_source_hubs(
    knn: pd.DataFrame,
    *,
    preferred: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """Pick direction×theme hubs that land in well-separated VA regions."""
    pref = preferred or PREFERRED_SPREAD_SOURCES
    r1 = knn[knn["rank"] == 1]
    out: list[dict] = []
    used: set[str] = set()
    for direction, theme in pref:
        hit = r1[(r1["direction"] == direction) & (r1["hub_theme"] == theme)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        hid = str(row["hub_image_id"])
        if hid in used or _risk_theme(row["hub_theme"], row["hub_category"]):
            continue
        used.add(hid)
        out.append({
            "direction": direction,
            "hub": row,
            "color": COL_FEMALE_SRC if direction == "FtoM" else COL_MALE_SRC,
            "gender_tag": "♀" if direction == "FtoM" else "♂",
        })
    return out


def select_shared_source_cases(
    knn: pd.DataFrame,
    *,
    n: int = 4,
    preferred: list[str] | None = None,
) -> list[dict]:
    """Same image as hub in both directions, with divergent F vs M matches."""
    r1 = knn[knn["rank"] == 1].copy()
    f = r1[r1["direction"] == "FtoM"].set_index("hub_image_id")
    m = r1[r1["direction"] == "MtoF"].set_index("hub_image_id")
    pref = preferred or PREFERRED_SHARED_SOURCES
    scored: list[dict] = []
    for hid in sorted(set(f.index) & set(m.index)):
        a, b = f.loc[hid], m.loc[hid]
        if isinstance(a, pd.DataFrame):
            a = a.iloc[0]
        if isinstance(b, pd.DataFrame):
            b = b.iloc[0]
        if _risk_theme(a["hub_theme"], a["hub_category"]):
            continue
        if _risk_theme(a["match_theme"], a["match_category"]) or _risk_theme(
            b["match_theme"], b["match_category"]
        ):
            continue
        if str(a["match_image_id"]) == str(b["match_image_id"]):
            continue
        d_match = float(np.hypot(
            float(a["match_pred_valence"]) - float(b["match_pred_valence"]),
            float(a["match_pred_arousal"]) - float(b["match_pred_arousal"]),
        ))
        va_shift = float(np.hypot(
            float(a["hub_valence"]) - float(b["hub_valence"]),
            float(a["hub_arousal"]) - float(b["hub_arousal"]),
        ))
        theme = str(a["hub_theme"])
        scored.append({
            "hub_image_id": str(hid),
            "hub_theme": theme,
            "hub_category": str(a["hub_category"]),
            "f_row": a,
            "m_row": b,
            "d_match_pred": d_match,
            "va_shift": va_shift,
            "pref": pref.index(theme) if theme in pref else len(pref),
        })
    scored.sort(key=lambda r: (r["pref"], -r["d_match_pred"], -r["va_shift"]))
    out: list[dict] = []
    used_bases: set[str] = set()
    for row in scored:
        base = str(row["hub_theme"]).rsplit(" ", 1)[0]
        if base in used_bases:
            continue
        used_bases.add(base)
        out.append(row)
        if len(out) >= n:
            break
    return out


def _place_thumb(
    ax,
    *,
    xy: tuple[float, float],
    rgb,
    label: str,
    color: str,
    zoom: float,
    fontsize: float = 5.2,
    lw: float = 1.3,
    zorder: int = 4,
) -> None:
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker

    if rgb is None:
        return
    ab = AnnotationBbox(
        VPacker(
            children=[
                OffsetImage(rgb, zoom=zoom),
                TextArea(label, textprops=dict(fontsize=fontsize, ha="center", color=color, fontweight="bold")),
            ],
            align="center", pad=0, sep=1,
        ),
        xy,
        frameon=True,
        bboxprops=dict(boxstyle="round,pad=0.14", fc="white", ec=color, lw=lw),
        pad=0.03,
        zorder=zorder,
    )
    ax.add_artist(ab)


def draw_source_va_panel(
    fig,
    outer_ax,
    spread_sources: list[dict],
    meta: pd.DataFrame,
    *,
    zoom: float = 0.28,
) -> None:
    """Left A: source-only thumbnails on VA (♀ red / ♂ blue), mixed-spread loci."""
    ax = fig.add_subplot(outer_ax.get_subplotspec())
    outer_ax.set_axis_off()
    path_map = meta.set_index("image_id")["image_path"]
    ax.set_xlim(*VA_LIM_A)
    ax.set_ylim(*VA_LIM_A)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(VA_LIM_A[0], VA_LIM_A[1] + 0.01, 1.0))
    ax.set_yticks(np.arange(VA_LIM_A[0], VA_LIM_A[1] + 0.01, 1.0))
    ax.set_xlabel("Valence (source rating)")
    ax.set_ylabel("Arousal (source rating)")
    ax.set_title(
        "Source images on VA (spread loci)\n"
        "♀ red = female-rated source · ♂ blue = male-rated source",
        fontsize=7.8, pad=5,
    )
    ax.grid(color="0.90", lw=0.6, zorder=0)
    if {"valence", "arousal"}.issubset(meta.columns):
        ax.scatter(
            meta["valence"], meta["arousal"],
            s=5, c="0.85", alpha=0.32, edgecolors="none", zorder=1,
        )

    for src in spread_sources:
        hub = src["hub"]
        hv = float(hub["hub_valence"])
        ha = float(hub["hub_arousal"])
        # slight clip only at edges so spread loci stay truthful
        hv_p = float(np.clip(hv, VA_LIM_A[0] + 0.45, VA_LIM_A[1] - 0.45))
        ha_p = float(np.clip(ha, VA_LIM_A[0] + 0.45, VA_LIM_A[1] - 0.45))
        path = Path(str(path_map.get(str(hub["hub_image_id"]), "")))
        rgb = _load_rgb(path if path.exists() else None, size=72)
        ax.scatter([hv], [ha], s=18, c=src["color"], zorder=3, edgecolors="white", linewidths=0.4)
        _place_thumb(
            ax, xy=(hv_p, ha_p), rgb=rgb,
            label=f"{src['gender_tag']} {hub['hub_theme']}\n({hub['hub_category']})",
            color=src["color"], zoom=zoom, fontsize=5.6, lw=1.45,
        )


def draw_hub_match_strip(
    fig,
    outer_ax,
    hub_bundles: list[dict],
    meta: pd.DataFrame,
) -> None:
    """Right A: arrange 1 hub → 3 matches as image strips (both directions)."""
    outer_ax.set_axis_off()
    n_rows = max(1, len(hub_bundles))
    # cols: hub | ≈ | m1 | m2 | m3 | caption
    gs = outer_ax.get_subplotspec().subgridspec(
        n_rows, 6,
        width_ratios=[1.15, 0.22, 1.0, 1.0, 1.0, 1.35],
        wspace=0.10, hspace=0.38,
    )
    path_map = meta.set_index("image_id")["image_path"]

    for r, bundle in enumerate(hub_bundles):
        hub = bundle["hub"]
        direction = bundle["direction"]
        col_c = COL_FEMALE_SRC if direction == "FtoM" else COL_MALE_SRC
        tag = "♀→♂" if direction == "FtoM" else "♂→♀"
        matches = bundle["matches"]

        ax_h = fig.add_subplot(gs[r, 0])
        ax_eq = fig.add_subplot(gs[r, 1])
        ax_eq.set_axis_off()
        ax_eq.text(0.5, 0.55, "→", ha="center", va="center", fontsize=18, color=col_c, fontweight="bold")
        ax_eq.text(0.5, 0.15, "1∶3", ha="center", va="center", fontsize=7, color=col_c)

        hpath = Path(str(path_map.get(str(hub["hub_image_id"]), "")))
        _thumb(
            ax_h, hpath if hpath.exists() else None,
            f"HUB {hub['hub_theme']}\n({hub['hub_category']})",
        )
        for spine in ax_h.spines.values():
            spine.set_visible(True)
            spine.set_color(col_c)
            spine.set_linewidth(2.0)

        for mi in range(3):
            ax_m = fig.add_subplot(gs[r, 2 + mi])
            if mi < len(matches):
                mrow = matches[mi]
                mpath = Path(str(path_map.get(str(mrow["match_image_id"]), "")))
                _thumb(
                    ax_m, mpath if mpath.exists() else None,
                    f"#{int(mrow['rank'])} {mrow['match_theme']}\n"
                    f"d={float(mrow['distance_l2']):.3f}",
                )
                for spine in ax_m.spines.values():
                    spine.set_visible(True)
                    spine.set_color(col_c)
                    spine.set_linewidth(1.1)
            else:
                ax_m.set_axis_off()

        ax_cap = fig.add_subplot(gs[r, 5])
        ax_cap.set_axis_off()
        ax_cap.text(
            0.0, 0.55,
            f"{tag}\n"
            f"VA≈({float(hub['hub_valence']):.1f}, {float(hub['hub_arousal']):.1f})\n"
            f"{hub['hub_category']} → cross-cat matches",
            ha="left", va="center", fontsize=7.5, color=col_c,
            transform=ax_cap.transAxes,
        )


def hub_bundles_from_spread(
    knn: pd.DataFrame,
    spread_sources: list[dict],
    *,
    n_match: int = N_MATCH_PER_HUB,
) -> list[dict]:
    """Build 1→k match bundles for the same hubs shown on the left VA map."""
    bundles: list[dict] = []
    used_match: set[str] = set()

    def base(theme: str) -> str:
        parts = str(theme).rsplit(" ", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else str(theme)

    def collect(cand: pd.DataFrame, hid: str, hb: str, *, cross_only: bool, unique_base: bool) -> list[pd.Series]:
        matches: list[pd.Series] = []
        used_bases = {hb}
        for _, mrow in cand.iterrows():
            mid = str(mrow["match_image_id"])
            if mid == hid or mid in used_match:
                continue
            if _risk_theme(mrow["match_theme"], mrow["match_category"]):
                continue
            if cross_only and bool(mrow["same_category"]):
                continue
            mb = base(mrow["match_theme"])
            if unique_base and mb in used_bases:
                continue
            matches.append(mrow)
            used_bases.add(mb)
            if len(matches) >= n_match:
                break
        return matches

    for src in spread_sources:
        hub = src["hub"]
        hid = str(hub["hub_image_id"])
        direction = src["direction"]
        cand = knn[(knn["direction"] == direction) & (knn["hub_image_id"] == hid)].sort_values(
            ["rank", "distance_l2"]
        )
        hb = base(hub["hub_theme"])
        matches = collect(cand, hid, hb, cross_only=True, unique_base=True)
        if len(matches) < n_match:
            # relax base uniqueness (e.g. Beach 4 + Beach 6) to fill k
            extra = collect(cand, hid, hb, cross_only=True, unique_base=False)
            seen = {str(m["match_image_id"]) for m in matches}
            for mrow in extra:
                mid = str(mrow["match_image_id"])
                if mid in seen:
                    continue
                matches.append(mrow)
                seen.add(mid)
                if len(matches) >= n_match:
                    break
        if len(matches) < 2:
            matches = collect(cand, hid, hb, cross_only=False, unique_base=False)
        for mrow in matches:
            used_match.add(str(mrow["match_image_id"]))
        if len(matches) >= 2:
            bundles.append({"direction": direction, "hub": hub, "matches": matches[:n_match]})
    return bundles


def draw_exemplar_panel(
    fig,
    outer_ax,
    hub_bundles: list[dict],
    meta: pd.DataFrame,
    spread_sources: list[dict] | None = None,
    source_cases: list[dict] | None = None,  # legacy unused
    *,
    zoom_hub: float = 0.30,
    zoom_match: float = 0.18,
) -> None:
    """Panel A: left = source-only VA (spread); right = 1→3 image strips."""
    del zoom_hub, zoom_match, source_cases
    outer_ax.set_axis_off()
    gs = outer_ax.get_subplotspec().subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.18)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    srcs = spread_sources or []
    draw_source_va_panel(fig, ax_left, srcs, meta, zoom=0.26)
    draw_hub_match_strip(fig, ax_right, hub_bundles, meta)


def draw_exemplar_panel_pairs_va(
    fig,
    outer_ax,
    exemplars: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    zoom: float = 0.28,
) -> None:
    """Legacy 1:1 pair placement on VA (kept for alternative exports)."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker

    outer_ax.set_xlim(*VA_LIM_A)
    outer_ax.set_ylim(*VA_LIM_A)
    outer_ax.set_aspect("equal", adjustable="box")
    outer_ax.set_xticks(np.arange(VA_LIM_A[0], VA_LIM_A[1] + 0.01, 1.0))
    outer_ax.set_yticks(np.arange(VA_LIM_A[0], VA_LIM_A[1] + 0.01, 1.0))
    outer_ax.set_xlabel("Valence (female target)")
    outer_ax.set_ylabel("Arousal (female target)")
    outer_ax.grid(color="0.90", lw=0.6, zorder=0)
    if {"valence", "arousal"}.issubset(meta.columns):
        outer_ax.scatter(
            meta["valence"], meta["arousal"],
            s=6, c="0.82", alpha=0.35, edgecolors="none", zorder=1,
        )

    path_map = meta.set_index("image_id")["image_path"]
    pair_offsets = [
        (-0.42, 0.28, 0.42, 0.28),
        (-0.42, -0.18, 0.42, -0.18),
        (-0.28, 0.45, 0.28, -0.40),
        (0.35, 0.40, -0.35, -0.35),
    ]
    colors = ["#1565c0", "#c62828", "#2e7d32", "#6a1b9a"]

    for r, (_, row) in enumerate(exemplars.iterrows()):
        v = float(row["y_f_i_valence"])
        a = float(row["y_f_i_arousal"])
        oxi, oyi, oxj, oyj = pair_offsets[r % len(pair_offsets)]
        for _ in range(2):
            if not (VA_LIM_A[0] + 0.55 < v + oxi < VA_LIM_A[1] - 0.55):
                oxi *= -1
            if not (VA_LIM_A[0] + 0.55 < v + oxj < VA_LIM_A[1] - 0.55):
                oxj *= -1
            if not (VA_LIM_A[0] + 0.55 < a + oyi < VA_LIM_A[1] - 0.55):
                oyi *= -1
            if not (VA_LIM_A[0] + 0.55 < a + oyj < VA_LIM_A[1] - 0.55):
                oyj *= -1
        xi, yi = v + oxi, a + oyi
        xj, yj = v + oxj, a + oyj
        col = colors[r % len(colors)]
        outer_ax.plot([xi, xj], [yi, yj], color=col, lw=1.4, alpha=0.85, zorder=2)
        pi = Path(str(path_map.get(row["i_image_id"], "")))
        pj = Path(str(path_map.get(row["j_image_id"], "")))
        for path, xy, theme, cat in (
            (pi if pi.exists() else None, (xi, yi), row["i_theme"], row["i_category"]),
            (pj if pj.exists() else None, (xj, yj), row["j_theme"], row["j_category"]),
        ):
            rgb = _load_rgb(path, size=64)
            if rgb is None:
                continue
            im = OffsetImage(rgb, zoom=zoom)
            lab = TextArea(
                f"{theme}\n({cat})",
                textprops=dict(fontsize=5.5, ha="center", color="0.2"),
            )
            packed = VPacker(children=[im, lab], align="center", pad=0, sep=1)
            ab = AnnotationBbox(
                packed, xy, xycoords="data", frameon=True,
                bboxprops=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=1.0),
                pad=0.05, zorder=3,
            )
            outer_ax.add_artist(ab)


def draw_exemplar_panel_strip(fig, outer_ax, exemplars: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Legacy strip layout (kept for side-by-side comparison exports)."""
    outer_ax.set_axis_off()
    n = len(exemplars)
    gs = outer_ax.get_subplotspec().subgridspec(
        n, 5, width_ratios=[1.0, 0.18, 1.0, 0.05, 1.15], wspace=0.08, hspace=0.35,
    )
    path_map = meta.set_index("image_id")["image_path"]

    for r, (_, row) in enumerate(exemplars.iterrows()):
        ax_i = fig.add_subplot(gs[r, 0])
        ax_ar = fig.add_subplot(gs[r, 1])
        ax_j = fig.add_subplot(gs[r, 2])
        ax_txt = fig.add_subplot(gs[r, 4])
        ax_ar.set_axis_off()
        ax_ar.text(0.5, 0.5, "≈", ha="center", va="center", fontsize=16, color="#455a64")
        pi = Path(str(path_map.get(row["i_image_id"], "")))
        pj = Path(str(path_map.get(row["j_image_id"], "")))
        _thumb(ax_i, pi if pi.exists() else None, f"{row['i_theme']}\n({row['i_category']})")
        _thumb(ax_j, pj if pj.exists() else None, f"{row['j_theme']}\n({row['j_category']})")
        ax_txt.set_axis_off()
        ax_txt.text(
            0.0, 0.55,
            f"VA≈({row['y_f_i_valence']:.1f}, {row['y_f_i_arousal']:.1f})\n"
            f"equiv. dist={row['distance_l2']:.3f}\n"
            f"{row['i_category']} ↔ {row['j_category']}",
            fontsize=7.2, va="center", family="sans-serif",
        )


def va_bin_edges(lim: tuple[float, float], grid: float) -> np.ndarray:
    """VA bin edges on [lim[0], lim[1]] with step ``grid`` (inclusive end)."""
    if grid <= 0:
        raise ValueError(f"grid must be > 0, got {grid}")
    edges = np.arange(lim[0], lim[1] + grid * 0.5, grid, dtype=float)
    if not np.isclose(edges[-1], lim[1]):
        edges = np.append(edges, float(lim[1]))
    else:
        edges[-1] = float(lim[1])
    return edges


def va_bin_stats(
    m: pd.DataFrame,
    nbins: int | None = None,
    min_n: int = 3,
    lim: tuple[float, float] = VA_LIM_A,
    grid: float | None = None,
):
    if grid is not None:
        edges = va_bin_edges(lim, float(grid))
        nbins = int(len(edges) - 1)
    elif nbins is None:
        # default: unit grid on VA_LIM_A → 6×6 for (1,7)
        edges = va_bin_edges(lim, 1.0)
        nbins = int(len(edges) - 1)
        grid = 1.0
    else:
        edges = np.linspace(lim[0], lim[1], nbins + 1)
        grid = float((lim[1] - lim[0]) / nbins)
    resid = np.full((nbins, nbins), np.nan)
    equiv_dist = np.full((nbins, nbins), np.nan)
    counts = np.zeros((nbins, nbins), int)
    v = m["y_f_i_valence"].to_numpy(float)
    a = m["y_f_i_arousal"].to_numpy(float)
    iv = np.clip(np.digitize(v, edges) - 1, 0, nbins - 1)
    ia = np.clip(np.digitize(a, edges) - 1, 0, nbins - 1)
    for i in range(nbins):
        for j in range(nbins):
            mask = (iv == j) & (ia == i)  # rows=arousal, cols=valence
            counts[i, j] = int(mask.sum())
            if counts[i, j] < min_n:
                continue
            resid[i, j] = float(m.loc[mask, "phi_residual"].mean())
            equiv_dist[i, j] = float(m.loc[mask, "distance_l2"].mean())
    return edges, resid, equiv_dist, counts, float(grid)


def _spatial_cohigh_mask(
    resid: np.ndarray, equiv_dist: np.ndarray, *, q_high: float = 0.65, q_low: float = 0.35,
) -> tuple[np.ndarray, float, float]:
    """Cells with high Φ residual and low equivalence distance (existence co-localization)."""
    r_ok = resid[np.isfinite(resid)]
    d_ok = equiv_dist[np.isfinite(equiv_dist)]
    if len(r_ok) and len(d_ok):
        r_thr = float(np.nanquantile(resid, q_high))
        d_thr = float(np.nanquantile(equiv_dist, q_low))
        both = (resid >= r_thr) & (equiv_dist <= d_thr)
    else:
        r_thr = float("nan")
        d_thr = float("nan")
        both = np.zeros_like(resid, dtype=bool)
    return both, r_thr, d_thr


def _spatial_axis_style(ax, *, grid_used: float, outline_lw: float) -> None:
    ticks_major = np.arange(VA_LIM_A[0], VA_LIM_A[1] + 0.01, 1.0)
    ticks_minor = (
        np.arange(VA_LIM_A[0], VA_LIM_A[1] + 0.01, grid_used)
        if grid_used < 1.0 - 1e-9
        else None
    )
    ax.set_xlim(*VA_LIM_A)
    ax.set_ylim(*VA_LIM_A)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(ticks_major)
    ax.set_yticks(ticks_major)
    if ticks_minor is not None:
        ax.set_xticks(ticks_minor, minor=True)
        ax.set_yticks(ticks_minor, minor=True)
        ax.grid(which="minor", color="0.88", lw=0.4, zorder=0)
    ax.grid(which="major", color="0.82", lw=0.55, zorder=0)
    ax.set_xlabel("Valence (female target)")
    ax.set_ylabel("Arousal (female target)")
    _ = outline_lw


def _draw_cohigh_outlines(ax, edges: np.ndarray, both: np.ndarray, *, lw: float) -> None:
    yy, xx = np.where(both)
    for yi, xi in zip(yy, xx):
        x0, x1 = edges[xi], edges[xi + 1]
        y0, y1 = edges[yi], edges[yi + 1]
        ax.plot(
            [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            color="#1565c0", lw=lw, zorder=3,
        )


def draw_spatial_panel(
    fig, outer_ax, m: pd.DataFrame, *, grid: float = 1.0, min_n: int = 3,
) -> dict:
    outer_ax.set_axis_off()
    gs = outer_ax.get_subplotspec().subgridspec(1, 2, wspace=0.28)
    edges, resid, equiv_dist, counts, grid_used = va_bin_stats(
        m, min_n=min_n, lim=VA_LIM_A, grid=float(grid),
    )
    extent = [edges[0], edges[-1], edges[0], edges[-1]]
    both, r_thr, d_thr = _spatial_cohigh_mask(resid, equiv_dist)
    outline_lw = 1.2 if grid_used <= 0.5 else 1.6
    n_cell = int(np.sum(counts >= min_n))
    foot = f"n={len(m)}; cells n≥{min_n}: {n_cell}  |  {counts.shape[0]}×{counts.shape[1]}"

    for col, field, cmap, lab, title, vmin, vmax in [
        (0, resid, "YlOrRd", "mean Φ residual", "Where Φ residual is large", None, None),
        (
            1, equiv_dist, "YlGn_r", "mean equiv. distance L2",
            "Where equivalent partners are close", None, None,
        ),
    ]:
        ax = fig.add_subplot(gs[0, col])
        vmin_use = vmin if vmin is not None else (np.nanmin(field) if np.isfinite(field).any() else 0)
        vmax_use = vmax if vmax is not None else (np.nanmax(field) if np.isfinite(field).any() else 1)
        im = ax.imshow(
            field, origin="lower", extent=extent, cmap=cmap, aspect="equal",
            vmin=vmin_use, vmax=vmax_use, interpolation="nearest",
        )
        _draw_cohigh_outlines(ax, edges, both, lw=outline_lw)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(lab, fontsize=7)
        _spatial_axis_style(ax, grid_used=grid_used, outline_lw=outline_lw)
        ax.set_title(f"{title}\n(grid={grid_used:g})", fontsize=8.5)
        ax.text(0.02, 0.02, foot, transform=ax.transAxes, fontsize=6.5, color="0.3")
    return {
        "grid": grid_used,
        "nbins": int(counts.shape[0]),
        "min_n": int(min_n),
        "n_pairs": int(len(m)),
        "n_cells_ge_min": n_cell,
        "n_cells_cohigh": int(np.sum(both)),
        "residual_q65": r_thr,
        "equiv_dist_q35": d_thr,
        "display_mode": "residual_vs_equiv_distance",
    }


def draw_spatial_single_panel(
    fig,
    ax,
    m: pd.DataFrame,
    *,
    field: str,
    grid: float = 1.0,
    min_n: int = 3,
    cohigh: bool = True,
) -> dict:
    """Standalone Fig.5B (residual) or Fig.5C (equiv distance) panel."""
    edges, resid, equiv_dist, counts, grid_used = va_bin_stats(
        m, min_n=min_n, lim=VA_LIM_A, grid=float(grid),
    )
    extent = [edges[0], edges[-1], edges[0], edges[-1]]
    both, r_thr, d_thr = _spatial_cohigh_mask(resid, equiv_dist)
    outline_lw = 1.2 if grid_used <= 0.5 else 1.6

    if field == "residual":
        data = resid
        cmap, lab, title = "YlOrRd", "mean Φ residual", "Φ residual on female-target VA"
    elif field == "equiv_dist":
        data = equiv_dist
        cmap, lab, title = "YlGn_r", "mean equiv. distance L2", "Low equivalence distance (oracle partners)"
    else:
        raise ValueError(f"unknown field={field!r}")

    im = ax.imshow(
        data, origin="lower", extent=extent, cmap=cmap, aspect="equal",
        vmin=np.nanmin(data) if np.isfinite(data).any() else 0,
        vmax=np.nanmax(data) if np.isfinite(data).any() else 1,
        interpolation="nearest",
    )
    if cohigh:
        _draw_cohigh_outlines(ax, edges, both, lw=outline_lw)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(lab, fontsize=7)
    _spatial_axis_style(ax, grid_used=grid_used, outline_lw=outline_lw)
    ax.set_title(f"{title}\n(grid={grid_used:g}; co-localization outlines)", fontsize=9)
    n_cell = int(np.sum(counts >= min_n))
    ax.text(
        0.02, 0.02,
        f"n={len(m)}; cells n≥{min_n}: {n_cell}",
        transform=ax.transAxes, fontsize=7, color="0.35",
    )
    return {
        "field": field,
        "grid": grid_used,
        "n_cells_cohigh": int(np.sum(both)),
        "residual_q65": r_thr,
        "equiv_dist_q35": d_thr,
    }


def category_transition_matrix(m: pd.DataFrame) -> tuple[np.ndarray, float, int]:
    """i_category → j_category counts; return matrix, cross-category rate, n."""
    ct = (
        pd.crosstab(m["i_category"], m["j_category"])
        .reindex(index=CAT_ORDER, columns=CAT_ORDER, fill_value=0)
    )
    mat = ct.to_numpy(dtype=float)
    n = int(len(m))
    cross = float((~m["same_category"]).mean()) if n else float("nan")
    return mat, cross, n


def draw_category_flow_panel(ax, m: pd.DataFrame, *, subtitle: str | None = None) -> None:
    """4×4 category transition among emotion-equivalent pairs (preferred over Sankey)."""
    mat, cross, n = category_transition_matrix(m)
    im = ax.imshow(mat, cmap="inferno", vmin=0, aspect="equal")
    ax.set_xticks(range(len(CAT_ORDER)), CAT_ORDER, rotation=30, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(CAT_ORDER)), CAT_ORDER, fontsize=7.5)
    ax.set_xlabel("Target category (j)")
    ax.set_ylabel("Source category (i)")
    vmax = float(mat.max()) if mat.size else 1.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = int(mat[i, j])
            if v <= 0:
                continue
            # inferno: dark at low, bright at high
            color = "white" if mat[i, j] < vmax * 0.55 else "0.05"
            weight = "bold" if i != j else "normal"
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8.5,
                    color=color, fontweight=weight)
    # light grid to separate cells
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    # off-diagonal outline cue
    for i in range(4):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#90a4ae", lw=1.4, zorder=3))
    fig = ax.figure
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("pair count", fontsize=7)
    extra = f"\n{subtitle}" if subtitle else ""
    ax.set_title(
        f"Semantic category flow (n={n})\n"
        f"{cross:.0%} cross-category — off-diagonal mass{extra}",
        fontsize=8.2,
    )
    ax.text(
        0.02, -0.18,
        "Diagonal = same category; off-diagonal = meaning-crossing pairs.",
        transform=ax.transAxes, fontsize=6.5, color="0.4",
    )


def make_fig6_discovery(save_png: Path, save_svg: Path | None = None) -> tuple[list[dict], dict]:
    m = build_merged()
    knn = load_knn()
    pairs_mode = "themecv" if PAIRS_CV.exists() and PAIRS_PRIMARY == PAIRS_CV else "fixedsplit"
    # Left: VA-spread source hubs (Flood / Snow / Bird / Crow — mixed_spread style)
    spread_sources = select_spread_source_hubs(knn)
    # Right strips: one ♀ hub + one ♂ hub (keep readable); prefer Bird + Crow
    strip_srcs = [
        s for s in spread_sources
        if str(s["hub"]["hub_theme"]) in {"Bird 4", "Crow 2", "Flood 3", "Snow 2"}
    ]
    # Prefer exactly one FtoM and one MtoF for the strip panel
    f_src = next((s for s in strip_srcs if s["direction"] == "FtoM" and s["hub"]["hub_theme"] == "Bird 4"), None)
    if f_src is None:
        f_src = next((s for s in strip_srcs if s["direction"] == "FtoM"), None)
    m_src = next((s for s in strip_srcs if s["direction"] == "MtoF"), None)
    strip_pick = [s for s in (f_src, m_src) if s is not None]
    hub_bundles = hub_bundles_from_spread(knn, strip_pick, n_match=N_MATCH_PER_HUB)
    meta = load_oasis_meta(OASIS_SCORES_CSV)
    flow_m, flow_meta = load_category_flow_pairs()
    mat, cross, n_pairs = category_transition_matrix(flow_m)
    perm = flow_meta.get("category_permutation", {})
    same_rate = 1.0 - cross
    null_same = perm.get("permutation_null_same_category_mean")
    p_same = perm.get("permutation_p_same_category_above_chance")
    if null_same is not None and p_same is not None:
        p_txt = "p<0.001" if float(p_same) < 0.001 else f"p={float(p_same):.2f}"
        c_note = f"theme-CV; same-cat {same_rate:.0%} vs null≈{null_same:.0%} ({p_txt})"
    else:
        c_note = f"mode={flow_meta.get('mode', 'unknown')}"

    # Full figure: A is wide (source VA + 1-to-3 strips)
    fig = plt.figure(figsize=(18.5, 6.8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.85, 0.95, 0.75], wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_a.text(-0.02, 1.02, "A", transform=ax_a.transAxes, fontsize=13, fontweight="bold", va="bottom")
    ax_b.text(-0.02, 1.02, "B", transform=ax_b.transAxes, fontsize=13, fontweight="bold", va="bottom")
    ax_c.text(-0.02, 1.02, "C", transform=ax_c.transAxes, fontsize=13, fontweight="bold", va="bottom")
    ax_a.set_title(
        "Sources on VA (spread)  |  1 hub → 3 matches (image strip)",
        fontsize=8.6, pad=6,
    )
    draw_exemplar_panel(fig, ax_a, hub_bundles, meta, spread_sources=spread_sources)
    # Primary composite uses grid=1; grid=0.5 exported as separate full-fig + panel B below
    spatial_stats_primary = draw_spatial_panel(fig, ax_b, m, grid=1.0)
    draw_category_flow_panel(ax_c, flow_m, subtitle=c_note)

    fig.suptitle(
        "Fig. 6 | Case translation: what it is, where it helps, and that meaning is crossed\n"
        "Spatial co-localization: high Φ residual ↔ close equivalent partners (not beat rate); "
        "category flow over all pairs",
        y=0.98, fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    save_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_png, dpi=300, bbox_inches="tight", facecolor="white")
    if save_svg is not None:
        fig.savefig(save_svg, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    paperish = "Paper_fig" in str(save_png)
    stem_prefix = "Paper_Fig6" if paperish else "Fig6"

    # Full-fig + standalone panel B for grid=1 and grid=0.5
    spatial_by_grid: dict[str, dict] = {"1": spatial_stats_primary}
    for g in (1.0, 0.5):
        gtag = "1" if g == 1.0 else "0p5"
        fig_g = plt.figure(figsize=(18.5, 6.8))
        gs_g = GridSpec(1, 3, figure=fig_g, width_ratios=[1.85, 0.95, 0.75], wspace=0.24)
        ax_a_g = fig_g.add_subplot(gs_g[0, 0])
        ax_b_g = fig_g.add_subplot(gs_g[0, 1])
        ax_c_g = fig_g.add_subplot(gs_g[0, 2])
        ax_a_g.text(-0.02, 1.02, "A", transform=ax_a_g.transAxes, fontsize=13, fontweight="bold", va="bottom")
        ax_b_g.text(-0.02, 1.02, "B", transform=ax_b_g.transAxes, fontsize=13, fontweight="bold", va="bottom")
        ax_c_g.text(-0.02, 1.02, "C", transform=ax_c_g.transAxes, fontsize=13, fontweight="bold", va="bottom")
        ax_a_g.set_title(
            "Sources on VA (spread)  |  1 hub → 3 matches (image strip)",
            fontsize=8.6, pad=6,
        )
        draw_exemplar_panel(fig_g, ax_a_g, hub_bundles, meta, spread_sources=spread_sources)
        st = draw_spatial_panel(fig_g, ax_b_g, m, grid=g)
        spatial_by_grid[gtag] = st
        draw_category_flow_panel(ax_c_g, flow_m, subtitle=c_note)
        fig_g.suptitle(
            f"Fig. 6 | Case translation (panel B grid={g:g})\n"
            "Spatial co-localization; category flow over all pairs",
            y=0.98, fontsize=11,
        )
        fig_g.tight_layout(rect=(0, 0.04, 1, 0.90))
        stem_g = save_png.with_name(f"{stem_prefix}_case_translation_grid{gtag}")
        fig_g.savefig(stem_g.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig_g.savefig(stem_g.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
        plt.close(fig_g)

        fig_b, ax_b_only = plt.subplots(figsize=(8.4, 4.2))
        ax_b_only.text(
            -0.02, 1.04, "B", transform=ax_b_only.transAxes,
            fontsize=13, fontweight="bold", va="bottom",
        )
        draw_spatial_panel(fig_b, ax_b_only, m, grid=g)
        fig_b.suptitle(f"Fig.6-B | VA co-localization (grid={g:g})", fontsize=10, y=1.02)
        fig_b.tight_layout()
        stem_b = save_png.with_name(f"{stem_prefix}_panelB_spatial_grid{gtag}")
        fig_b.savefig(stem_b.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig_b.savefig(stem_b.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
        plt.close(fig_b)

    # Standalone panel A (PNG + SVG) — taller for strip rows
    fig_a, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_title(
        "Fig.6-A | Left: source-only VA (spread loci)  ·  Right: 1→3 image strips",
        fontsize=10,
    )
    draw_exemplar_panel(fig_a, ax, hub_bundles, meta, spread_sources=spread_sources)
    fig_a.tight_layout()
    stem_a = save_png.with_name(f"{stem_prefix}_panelA_hub_matches")
    fig_a.savefig(stem_a.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig_a.savefig(stem_a.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig_a)

    # Standalone panel C (theme-CV n=900)
    fig_c, axc = plt.subplots(figsize=(5.2, 4.8))
    draw_category_flow_panel(axc, flow_m, subtitle=c_note)
    fig_c.suptitle("Fig.6-C | Semantic category flow (theme-blocked CV)", fontsize=10, y=1.02)
    fig_c.tight_layout()
    stem_c = save_png.with_name(f"{stem_prefix}_panelC_category_flow")
    fig_c.savefig(stem_c.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig_c.savefig(stem_c.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig_c)

    off_diag = float(mat.sum() - np.trace(mat))
    manifest = {
        "circular_rule": "Do not claim scalar residual×improvement ρ; show VA co-localization only.",
        "pairs_mode": pairs_mode,
        "pairs_csv": str(PAIRS_PRIMARY.name),
        "knn_csv": str((KNN_CSV_CV if KNN_CSV_CV.exists() else KNN_CSV_FIXED).name),
        "panel_a_mode": "spread_source_va_plus_1to3_strip",
        "n_match_per_hub": N_MATCH_PER_HUB,
        "spread_sources": [
            {
                "direction": s["direction"],
                "hub_image_id": str(s["hub"]["hub_image_id"]),
                "hub_theme": str(s["hub"]["hub_theme"]),
                "hub_category": str(s["hub"]["hub_category"]),
                "hub_va": [float(s["hub"]["hub_valence"]), float(s["hub"]["hub_arousal"])],
                "gender_tag": s["gender_tag"],
            }
            for s in spread_sources
        ],
        "hub_bundles": [
            {
                "direction": b["direction"],
                "hub_image_id": str(b["hub"]["hub_image_id"]),
                "hub_theme": str(b["hub"]["hub_theme"]),
                "hub_category": str(b["hub"]["hub_category"]),
                "hub_va": [float(b["hub"]["hub_valence"]), float(b["hub"]["hub_arousal"])],
                "matches": [
                    {
                        "match_image_id": str(mrow["match_image_id"]),
                        "match_theme": str(mrow["match_theme"]),
                        "match_category": str(mrow["match_category"]),
                        "rank": int(mrow["rank"]),
                        "distance_l2": float(mrow["distance_l2"]),
                    }
                    for mrow in b["matches"]
                ],
            }
            for b in hub_bundles
        ],
        "spatial": {
            "primary_grid": 1.0,
            "min_n": 3,
            "by_grid": spatial_by_grid,
            "n_pairs": int(len(m)),
            "frac_beats_oracle": float(m["beats_phi"].mean()),
            "mean_equiv_distance": float(m["distance_l2"].mean()),
        },
        "category_flow": {
            "source": str(PAIRS_CV.name if PAIRS_CV.exists() else PAIRS_FIXED.name),
            "mode": flow_meta.get("mode"),
            "n_pairs": n_pairs,
            "cross_category_rate": cross,
            "categories": CAT_ORDER,
            "counts_i_by_j": mat.astype(int).tolist(),
            "n_same_category": int(np.trace(mat)),
            "n_cross_category": int(off_diag),
            "category_permutation": perm,
            "note": "4×4 transition; prefer theme-CV n=900 over fixedsplit n≈153.",
        },
        "oasis_citation": "Kurdi, Lozano & Banaji (2017), Behavior Research Methods — open for research use.",
    }
    OUT_RES.mkdir(parents=True, exist_ok=True)
    (OUT_RES / "fig6_discovery_exemplar_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return hub_bundles, manifest


def export_exemplar_alternatives(
    out_dir: Path,
    *,
    also_paper_panel_a: bool = True,
) -> list[Path]:
    """Write VA-placed Panel-A variants for each named exemplar set (+ candidate strip)."""
    m = build_merged()
    meta = load_oasis_meta(OASIS_SCORES_CSV)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Per-set VA panels
    for name, seeds in EXEMPLAR_SETS.items():
        exemplars = select_exemplars(m, n=3, seed_pairs=seeds)
        if exemplars.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.2, 6.0))
        ax.set_title(
            f"Fig.6-A candidate | set={name}\n"
            "Emotion-equivalent pairs on VA (illustrative)",
            fontsize=9,
        )
        draw_exemplar_panel_pairs_va(fig, ax, exemplars, meta, zoom=0.30)
        fig.tight_layout()
        for stem_dir, stem in (
            (out_dir, f"Fig6_panelA_va_exemplars_{name}"),
            (ROOT / "Paper_fig", f"Paper_Fig6_panelA_va_exemplars_{name}") if also_paper_panel_a else (None, None),
        ):
            if stem_dir is None:
                continue
            stem_dir.mkdir(parents=True, exist_ok=True)
            png = stem_dir / f"{stem}.png"
            svg = stem_dir / f"{stem}.svg"
            fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
            fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white")
            written.append(png)
        plt.close(fig)

        # also legacy strip for the same set (easy visual compare)
        fig2 = plt.figure(figsize=(7.2, 5.2))
        ax2 = fig2.add_subplot(111)
        ax2.set_title(f"strip layout | set={name}", fontsize=9)
        draw_exemplar_panel_strip(fig2, ax2, exemplars, meta)
        fig2.tight_layout()
        png2 = out_dir / f"Fig6_panelA_strip_exemplars_{name}.png"
        fig2.savefig(png2, dpi=250, bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        written.append(png2)

    # Contact sheet of top safe cross-category candidates (pairs as mini strips)
    cross = m[(~m["same_category"]) & (~m["risky"])].copy()
    cross = cross[
        cross["i_category"].isin(PREFERRED_CAT) & cross["j_category"].isin(PREFERRED_CAT)
    ].sort_values("distance_l2")
    used: set[str] = set()
    rows: list[pd.Series] = []

    def _base(theme: str) -> str:
        parts = str(theme).rsplit(" ", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else str(theme)

    for _, row in cross.iterrows():
        bi, bj = _base(row["i_theme"]), _base(row["j_theme"])
        if bi in used or bj in used:
            continue
        used.update({bi, bj})
        rows.append(row)
        if len(rows) >= 12:
            break
    cand = pd.DataFrame(rows)
    path_map = meta.set_index("image_id")["image_path"]
    n = len(cand)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.5, 3.4 * nrows))
    axes = np.atleast_2d(axes)
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        ax.set_axis_off()
        if idx >= n:
            continue
        row = cand.iloc[idx]
        gs = ax.get_subplotspec().subgridspec(1, 3, width_ratios=[1, 0.12, 1], wspace=0.05)
        ai = fig.add_subplot(gs[0, 0])
        am = fig.add_subplot(gs[0, 1])
        aj = fig.add_subplot(gs[0, 2])
        am.set_axis_off()
        am.text(0.5, 0.55, "≈", ha="center", va="center", fontsize=14, color="#455a64")
        am.text(
            0.5, 0.15,
            f"({row['y_f_i_valence']:.1f},{row['y_f_i_arousal']:.1f})\nd={row['distance_l2']:.3f}",
            ha="center", va="center", fontsize=5.5, color="0.35",
        )
        pi = Path(str(path_map.get(row["i_image_id"], "")))
        pj = Path(str(path_map.get(row["j_image_id"], "")))
        _thumb(ai, pi if pi.exists() else None, f"{row['i_theme']}\n({row['i_category']})")
        _thumb(aj, pj if pj.exists() else None, f"{row['j_theme']}\n({row['j_category']})")
    fig.suptitle(
        "Fig.6-A exemplar pool (safe cross-category, theme-diverse, sorted by equiv. distance)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pool_png = out_dir / "Fig6_panelA_exemplar_pool_12.png"
    fig.savefig(pool_png, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    written.append(pool_png)

    # Manifest of sets + pool
    manifest = {
        "sets": {
            name: [{"i": a, "j": b} for a, b in seeds]
            for name, seeds in EXEMPLAR_SETS.items()
        },
        "pool_top12": [
            {
                "i_theme": str(r["i_theme"]),
                "j_theme": str(r["j_theme"]),
                "i_category": str(r["i_category"]),
                "j_category": str(r["j_category"]),
                "distance_l2": float(r["distance_l2"]),
                "va": [float(r["y_f_i_valence"]), float(r["y_f_i_arousal"])],
            }
            for _, r in cand.iterrows()
        ],
    }
    (out_dir / "Fig6_panelA_exemplar_alternatives.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return written


def select_1to3_hubs_for_direction(
    knn: pd.DataFrame,
    direction: str,
    *,
    n: int = 6,
    preferred_themes: list[str] | None = None,
) -> list[dict]:
    """Diverse safe hubs for one direction (FtoM or MtoF), preferring curated themes."""
    pref = preferred_themes or PREFERRED_1TO3_EXEMPLARS.get(direction, [])
    r1 = knn[(knn["direction"] == direction) & (knn["rank"] == 1)].copy()
    out: list[dict] = []
    used_ids: set[str] = set()
    used_bases: set[str] = set()

    def base(theme: str) -> str:
        parts = str(theme).rsplit(" ", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else str(theme)

    def try_add(row: pd.Series) -> bool:
        hid = str(row["hub_image_id"])
        hb = base(row["hub_theme"])
        if hid in used_ids or hb in used_bases:
            return False
        if _risk_theme(row["hub_theme"], row["hub_category"]):
            return False
        src = {
            "direction": direction,
            "hub": row,
            "color": COL_FEMALE_SRC if direction == "FtoM" else COL_MALE_SRC,
            "gender_tag": "♀" if direction == "FtoM" else "♂",
        }
        bundles = hub_bundles_from_spread(knn, [src], n_match=N_MATCH_PER_HUB)
        if not bundles or len(bundles[0]["matches"]) < N_MATCH_PER_HUB:
            return False
        # reject if any match is risky (belt-and-suspenders)
        for mrow in bundles[0]["matches"]:
            if _risk_theme(mrow["match_theme"], mrow["match_category"]):
                return False
        used_ids.add(hid)
        used_bases.add(hb)
        out.append(src)
        return True

    # 1) curated themes
    for theme in pref:
        hit = r1[r1["hub_theme"] == theme]
        if hit.empty:
            continue
        try_add(hit.iloc[0])
        if len(out) >= n:
            return out

    # 2) fill with diverse VA / cross-category hubs
    cand = r1[~r1["same_category"]].copy()
    cand = cand.assign(
        _quad=(
            (cand["hub_valence"] >= 4.0).astype(int) * 2
            + (cand["hub_arousal"] >= 3.5).astype(int)
        )
    )
    # round-robin by VA quadrant for spread
    for _, grp in cand.groupby("_quad", sort=True):
        for _, row in grp.sort_values("distance_l2").iterrows():
            try_add(row)
            if len(out) >= n:
                return out
    for _, row in cand.sort_values("distance_l2").iterrows():
        try_add(row)
        if len(out) >= n:
            break
    return out


def export_fig6_1to3_exemplars(
    *,
    out_dir: Path | None = None,
    n_per_dir: int = 6,
) -> list[Path]:
    """Export individual 1→3 strips (+ direction sheets) for F→M and M→F."""
    out_dir = out_dir or (ROOT / "Paper_fig")
    out_dir.mkdir(parents=True, exist_ok=True)
    knn = load_knn()
    meta = load_oasis_meta(OASIS_SCORES_CSV)
    written: list[Path] = []
    index_rows: list[dict] = []

    for direction, tag in (("FtoM", "FtoM_female_to_male"), ("MtoF", "MtoF_male_to_female")):
        hubs = select_1to3_hubs_for_direction(knn, direction, n=n_per_dir)
        # Build each hub alone so sheet rows match individual exports
        bundles: list[dict] = []
        for src in hubs:
            one = hub_bundles_from_spread(knn, [src], n_match=N_MATCH_PER_HUB)
            if one:
                bundles.append(one[0])
        # Direction sheet (all rows)
        if bundles:
            fig = plt.figure(figsize=(11.5, 1.55 * len(bundles) + 0.8))
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            draw_hub_match_strip(fig, ax, bundles, meta)
            arrow = "♀→♂" if direction == "FtoM" else "♂→♀"
            fig.suptitle(
                f"Fig.6 | 1 hub → 3 matches ({arrow}; theme-CV)\n"
                f"{len(bundles)} exemplars — not 1∶1; ranks #1–#3 by emotional distance",
                fontsize=10, y=0.995,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            stem = out_dir / f"Paper_Fig6_1to3_sheet_{tag}"
            fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
            fig.savefig(stem.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
            plt.close(fig)
            written.extend([stem.with_suffix(".png"), stem.with_suffix(".svg")])

        # Individual 1→3 figures
        for b in bundles:
            hub = b["hub"]
            theme_slug = re.sub(r"[^A-Za-z0-9]+", "_", str(hub["hub_theme"])).strip("_")
            fig = plt.figure(figsize=(11.2, 2.4))
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            draw_hub_match_strip(fig, ax, [b], meta)
            arrow = "♀→♂" if direction == "FtoM" else "♂→♀"
            matches = ", ".join(
                f"{m['match_theme']}(#{int(m['rank'])})" for m in b["matches"]
            )
            fig.suptitle(
                f"Fig.6 exemplar | {arrow} 1→3  ·  {hub['hub_theme']} ({hub['hub_category']})\n"
                f"→ {matches}",
                fontsize=9.5, y=1.02,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.88))
            stem = out_dir / f"Paper_Fig6_1to3_{direction}_{theme_slug}"
            fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
            fig.savefig(stem.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
            plt.close(fig)
            written.extend([stem.with_suffix(".png"), stem.with_suffix(".svg")])
            index_rows.append({
                "direction": direction,
                "hub_theme": str(hub["hub_theme"]),
                "hub_category": str(hub["hub_category"]),
                "hub_image_id": str(hub["hub_image_id"]),
                "hub_va": [float(hub["hub_valence"]), float(hub["hub_arousal"])],
                "matches": [
                    {
                        "rank": int(m["rank"]),
                        "theme": str(m["match_theme"]),
                        "category": str(m["match_category"]),
                        "image_id": str(m["match_image_id"]),
                        "distance_l2": float(m["distance_l2"]),
                        "same_category": bool(m["same_category"]),
                    }
                    for m in b["matches"]
                ],
                "png": stem.with_suffix(".png").name,
            })

    # Also dump a testfig copy of sheets
    test_dir = ROOT / "testfig"
    test_dir.mkdir(parents=True, exist_ok=True)
    for p in list(written):
        if p.suffix == ".png" and "sheet_" in p.name:
            dest = test_dir / p.name.replace("Paper_", "")
            dest.write_bytes(p.read_bytes())

    (out_dir / "Paper_Fig6_1to3_exemplar_index.json").write_text(
        json.dumps({"n_match": N_MATCH_PER_HUB, "exemplars": index_rows}, indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return written


COL_MTOF_PINK = "#ec407a"   # M→F ribbons
COL_FTOM_CYAN = "#00bcd4"   # F→M ribbons

MUTUAL_SANKEY_RIBBON_LAYOUTS: dict[str, dict[str, float]] = {
    "default": {"y_m": 0.82, "y_f": 0.18, "y_top": 0.66, "y_bot": 0.34, "fig_h": 7.6},
    "compact": {"y_m": 0.74, "y_f": 0.26, "y_top": 0.57, "y_bot": 0.43, "fig_h": 5.6},
}


def _mutual_sankey_layout(name: str) -> dict[str, float]:
    return MUTUAL_SANKEY_RIBBON_LAYOUTS.get(name, MUTUAL_SANKEY_RIBBON_LAYOUTS["default"])


def _theme_base(theme: str) -> str:
    parts = str(theme).rsplit(" ", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else str(theme)


def find_reciprocal_edges(knn: pd.DataFrame, *, k: int = 3) -> pd.DataFrame:
    """Edges where M→F (male hub→female match) and F→M (female hub→male match) both appear in top-k."""
    mt = knn[(knn["direction"] == "MtoF") & (knn["rank"] <= k)]
    ft = knn[(knn["direction"] == "FtoM") & (knn["rank"] <= k)]
    ft_key = {(str(r.hub_image_id), str(r.match_image_id)): r for _, r in ft.iterrows()}
    rows: list[dict] = []
    for _, r in mt.iterrows():
        mid, fid = str(r["hub_image_id"]), str(r["match_image_id"])
        back = ft_key.get((fid, mid))
        if back is None:
            continue
        if _risk_theme(r["hub_theme"], r["hub_category"]) or _risk_theme(
            r["match_theme"], r["match_category"]
        ):
            continue
        rows.append({
            "m_id": mid,
            "f_id": fid,
            "m_theme": str(r["hub_theme"]),
            "f_theme": str(r["match_theme"]),
            "m_category": str(r["hub_category"]),
            "f_category": str(r["match_category"]),
            "d_mf": float(r["distance_l2"]),
            "d_fm": float(back["distance_l2"]),
            "rank_mf": int(r["rank"]),
            "rank_fm": int(back["rank"]),
            "score": float(r["distance_l2"]) + float(back["distance_l2"]),
        })
    return pd.DataFrame(rows).sort_values(["rank_mf", "rank_fm", "score"]).reset_index(drop=True)


def _barycentric_order(
    males: list[dict],
    females: list[dict],
    edges: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """Reorder nodes to reduce ribbon crossings (iterative barycenter)."""
    m_ids = [m["id"] for m in males]
    f_ids = [f["id"] for f in females]
    m_idx = {i: k for k, i in enumerate(m_ids)}
    f_idx = {i: k for k, i in enumerate(f_ids)}
    links = []
    for _, e in edges.iterrows():
        if e["m_id"] not in m_idx or e["f_id"] not in f_idx:
            continue
        sc = e["score"] if "score" in e.index else np.nan
        if not np.isfinite(sc):
            parts = []
            if np.isfinite(e.get("d_mf", np.nan)):
                parts.append(float(e["d_mf"]))
            if np.isfinite(e.get("d_fm", np.nan)):
                parts.append(float(e["d_fm"]))
            sc = float(np.mean(parts)) if parts else 1.0
        links.append((str(e["m_id"]), str(e["f_id"]), float(sc)))
    if not links:
        return males, females

    def sort_by_bary(nodes: list[dict], other_pos: dict[str, float], as_src: bool) -> list[dict]:
        scores: list[tuple[float, int, dict]] = []
        for i, n in enumerate(nodes):
            nbrs = []
            for a, b, s in links:
                if as_src and a == n["id"] and b in other_pos:
                    nbrs.append((other_pos[b], 1.0 / (s + 1e-3)))
                if (not as_src) and b == n["id"] and a in other_pos:
                    nbrs.append((other_pos[a], 1.0 / (s + 1e-3)))
            if nbrs:
                wsum = sum(w for _, w in nbrs)
                bary = sum(p * w for p, w in nbrs) / wsum
            else:
                bary = float(i)
            scores.append((bary, i, n))
        scores.sort(key=lambda t: (t[0], t[1]))
        return [t[2] for t in scores]

    for _ in range(6):
        f_pos = {f["id"]: float(i) for i, f in enumerate(females)}
        males = sort_by_bary(males, f_pos, as_src=True)
        m_pos = {m["id"]: float(i) for i, m in enumerate(males)}
        females = sort_by_bary(females, m_pos, as_src=False)
    return males, females


def _ribbon_weighted_span_cost(
    males: list[dict],
    females: list[dict],
    edges: pd.DataFrame,
) -> float:
    """Weighted sum of horizontal |Δx| over displayed oracle ribbons."""
    if len(edges) == 0:
        return 0.0
    if len(males) == 1:
        mx = np.array([0.5])
    else:
        mx = np.linspace(0.14, 0.86, len(males))
    if len(females) == 1:
        fx = np.array([0.5])
    else:
        fx = np.linspace(0.14, 0.86, len(females))
    m_pos = {m["id"]: float(x) for m, x in zip(males, mx)}
    f_pos = {f["id"]: float(x) for f, x in zip(females, fx)}
    cost = 0.0
    for _, e in edges.iterrows():
        x0, x1 = m_pos.get(str(e["m_id"])), f_pos.get(str(e["f_id"]))
        if x0 is None or x1 is None:
            continue
        w = 0.0
        if np.isfinite(e.get("d_mf", np.nan)):
            w += 1.0 / (float(e["d_mf"]) + 1e-3)
        if np.isfinite(e.get("d_fm", np.nan)):
            w += 1.0 / (float(e["d_fm"]) + 1e-3)
        cost += w * abs(x0 - x1)
    return float(cost)


def _order_min_link_span(
    males: list[dict],
    females: list[dict],
    edges: pd.DataFrame,
    *,
    swap_passes: int = 24,
) -> tuple[list[dict], list[dict]]:
    """Place nodes so connected pairs sit near the same x (min weighted |Δx|)."""
    males = list(males)
    females = list(females)
    if len(edges) == 0:
        return males, females
    males, females = _barycentric_order(males, females, edges)
    for _ in range(4):
        males, females = _barycentric_order(males, females, edges)
    best = _ribbon_weighted_span_cost(males, females, edges)
    for _ in range(swap_passes):
        improved = False
        for i in range(len(males) - 1):
            trial = males.copy()
            trial[i], trial[i + 1] = trial[i + 1], trial[i]
            c = _ribbon_weighted_span_cost(trial, females, edges)
            if c + 1e-9 < best:
                males, best = trial, c
                improved = True
        for i in range(len(females) - 1):
            trial = females.copy()
            trial[i], trial[i + 1] = trial[i + 1], trial[i]
            c = _ribbon_weighted_span_cost(males, trial, edges)
            if c + 1e-9 < best:
                females, best = trial, c
                improved = True
        if not improved:
            break
    return males, females


def _reciprocal_id_pairs(knn: pd.DataFrame, *, k: int = 3) -> set[tuple[str, str]]:
    """(m_id, f_id) pairs that are reciprocal in top-k (used to prefer asymmetry)."""
    mt = knn[(knn["direction"] == "MtoF") & (knn["rank"] <= k)]
    ft = knn[(knn["direction"] == "FtoM") & (knn["rank"] <= k)]
    back = {(str(r.hub_image_id), str(r.match_image_id)) for _, r in ft.iterrows()}
    out: set[tuple[str, str]] = set()
    for _, r in mt.iterrows():
        mid, fid = str(r["hub_image_id"]), str(r["match_image_id"])
        if (fid, mid) in back:
            out.add((mid, fid))
    return out


def select_asymmetric_sankey_columns(
    knn: pd.DataFrame,
    *,
    n_cols: int = 2,
    n_per_side: int | None = None,
    k: int = 3,
    n_hubs_mf: int = 2,
    n_hubs_fm: int = 2,
    n_match: int = 3,
    n_bridge: int = 1,
) -> list[dict]:
    """Build columns with partial M→F / F→M image sharing.

    Intended motif:
      M→F: A → B, C, D
      F→M: B → C, F, G
    where B is an M→F target reused as F→M hub, and C is an image id that
    appears both as an M→F female target and an F→M male target.
    """
    if n_per_side is None:
        n_per_side = max(8, n_hubs_mf + n_hubs_fm * n_match)
    kk = max(k, n_match)
    mt_all = knn[(knn["direction"] == "MtoF") & (knn["rank"] <= kk)].copy()
    ft_all = knn[(knn["direction"] == "FtoM") & (knn["rank"] <= kk)].copy()

    from collections import defaultdict

    mf_matches: dict[str, list[pd.Series]] = defaultdict(list)
    fm_matches: dict[str, list[pd.Series]] = defaultdict(list)
    for _, r in mt_all.iterrows():
        mf_matches[str(r["hub_image_id"])].append(r)
    for _, r in ft_all.iterrows():
        fm_matches[str(r["hub_image_id"])].append(r)
    for d in (mf_matches, fm_matches):
        for hid in d:
            d[hid] = sorted(d[hid], key=lambda r: (int(r["rank"]), float(r["distance_l2"])))

    def safe_hub(r: pd.Series) -> bool:
        return not _risk_theme(r["hub_theme"], r["hub_category"])

    def safe_match(r: pd.Series) -> bool:
        return not _risk_theme(r["match_theme"], r["match_category"])

    # Discover chain seeds: A→…B…C… and B→…C…
    seeds: list[dict] = []
    for a_id, a_rows in mf_matches.items():
        if len(a_rows) < n_match or not safe_hub(a_rows[0]):
            continue
        a_rows = [r for r in a_rows if safe_match(r)][:n_match]
        if len(a_rows) < n_match:
            continue
        f_ids = [str(r["match_image_id"]) for r in a_rows]
        f_set = set(f_ids)
        for b_id in f_ids:
            b_rows = fm_matches.get(b_id, [])
            if len(b_rows) < n_match:
                continue
            if not safe_hub(b_rows[0]):
                continue
            b_rows = [r for r in b_rows if safe_match(r)]
            if len(b_rows) < n_match:
                continue
            # prefer ordering that puts shared ids first among B's targets
            shared = [r for r in b_rows if str(r["match_image_id"]) in f_set and str(r["match_image_id"]) != b_id]
            other = [r for r in b_rows if str(r["match_image_id"]) not in f_set]
            ordered = shared + other
            # dedupe by match id
            seen: set[str] = set()
            picks_b: list[pd.Series] = []
            for r in ordered:
                mid = str(r["match_image_id"])
                if mid == b_id or mid in seen:
                    continue
                seen.add(mid)
                picks_b.append(r)
                if len(picks_b) >= n_match:
                    break
            if len(picks_b) < n_match:
                continue
            n_shared = sum(1 for r in picks_b if str(r["match_image_id"]) in f_set)
            if n_shared < 1:
                continue
            mean_d = float(np.mean(
                [float(r["distance_l2"]) for r in a_rows]
                + [float(r["distance_l2"]) for r in picks_b]
            ))
            seeds.append({
                "a_id": a_id,
                "b_id": b_id,
                "a_rows": a_rows,
                "b_rows": picks_b,
                "n_shared": n_shared,
                "mean_d": mean_d,
                "shared_ids": [str(r["match_image_id"]) for r in picks_b if str(r["match_image_id"]) in f_set],
            })
    seeds.sort(key=lambda s: (-s["n_shared"], s["mean_d"]))

    used_ids: set[str] = set()
    columns: list[dict] = []

    def pick_extra_mf(exclude: set[str]) -> tuple[str, list[pd.Series]] | None:
        for hid, rows in sorted(
            mf_matches.items(),
            key=lambda kv: float(kv[1][0]["distance_l2"]) if kv[1] else 9e9,
        ):
            if hid in used_ids or hid in exclude or not rows:
                continue
            if not safe_hub(rows[0]):
                continue
            if _theme_base(rows[0]["hub_theme"]) in {
                _theme_base(x) for x in exclude
            }:
                pass
            picks = []
            seen_f: set[str] = set()
            for r in rows:
                if not safe_match(r):
                    continue
                fid = str(r["match_image_id"])
                if fid in used_ids or fid in exclude or fid in seen_f:
                    continue
                picks.append(r)
                seen_f.add(fid)
                if len(picks) >= n_match:
                    break
            if len(picks) >= n_match:
                return hid, picks
        return None

    def pick_extra_fm(exclude: set[str], prefer_share_with: set[str]) -> tuple[str, list[pd.Series]] | None:
        ranked = []
        for hid, rows in fm_matches.items():
            if hid in used_ids or hid in exclude or not rows:
                continue
            if not safe_hub(rows[0]):
                continue
            picks = []
            seen_m: set[str] = set()
            # shared first
            for pool in (
                [r for r in rows if str(r["match_image_id"]) in prefer_share_with],
                rows,
            ):
                for r in pool:
                    if not safe_match(r):
                        continue
                    mid = str(r["match_image_id"])
                    if mid in used_ids or mid in exclude or mid in seen_m or mid == hid:
                        continue
                    picks.append(r)
                    seen_m.add(mid)
                    if len(picks) >= n_match:
                        break
                if len(picks) >= n_match:
                    break
            if len(picks) < n_match:
                continue
            n_sh = sum(1 for r in picks if str(r["match_image_id"]) in prefer_share_with)
            ranked.append((-n_sh, float(np.mean([float(r["distance_l2"]) for r in picks])), hid, picks))
        ranked.sort()
        return (ranked[0][2], ranked[0][3]) if ranked else None

    for seed in seeds:
        if len(columns) >= n_cols:
            break
        a_id, b_id = seed["a_id"], seed["b_id"]
        if a_id in used_ids or b_id in used_ids:
            continue
        if any(str(r["match_image_id"]) in used_ids for r in seed["a_rows"]):
            continue
        if any(str(r["match_image_id"]) in used_ids for r in seed["b_rows"]):
            continue

        males: list[dict] = []
        females: list[dict] = []
        m_ids: set[str] = set()
        f_ids: set[str] = set()
        edge_rows: list[dict] = []
        mf_hubs: set[str] = set()
        fm_hubs: set[str] = set()
        bridges: list[str] = []
        shared_targets: list[str] = []

        def add_male(node: dict) -> None:
            if node["id"] in m_ids or len(males) >= n_per_side:
                return
            males.append(node)
            m_ids.add(node["id"])

        def add_female(node: dict) -> None:
            if node["id"] in f_ids or len(females) >= n_per_side:
                return
            females.append(node)
            f_ids.add(node["id"])

        a0 = seed["a_rows"][0]
        add_male({
            "id": a_id,
            "theme": str(a0["hub_theme"]),
            "category": str(a0["hub_category"]),
            "role": "mf_hub",
        })
        mf_hubs.add(a_id)
        for r in seed["a_rows"]:
            fid = str(r["match_image_id"])
            role = "mf_match+fm_hub" if fid == b_id else "mf_match"
            if fid in seed["shared_ids"]:
                role += "+shared"
            add_female({
                "id": fid,
                "theme": str(r["match_theme"]),
                "category": str(r["match_category"]),
                "role": role,
            })
            edge_rows.append({
                "m_id": a_id, "f_id": fid,
                "m_theme": str(a0["hub_theme"]), "f_theme": str(r["match_theme"]),
                "m_category": str(a0["hub_category"]), "f_category": str(r["match_category"]),
                "d_mf": float(r["distance_l2"]), "d_fm": np.nan,
                "rank_mf": int(r["rank"]), "rank_fm": -1,
                "kind": "MtoF", "score": float(r["distance_l2"]),
            })

        b0 = seed["b_rows"][0]
        # B already on female row as mf_match
        bridges.append(b_id)
        fm_hubs.add(b_id)
        for r in seed["b_rows"]:
            mid = str(r["match_image_id"])
            role = "fm_match"
            if mid in seed["shared_ids"]:
                role += "+shared"
                shared_targets.append(mid)
            add_male({
                "id": mid,
                "theme": str(r["match_theme"]),
                "category": str(r["match_category"]),
                "role": role,
            })
            edge_rows.append({
                "m_id": mid, "f_id": b_id,
                "m_theme": str(r["match_theme"]), "f_theme": str(b0["hub_theme"]),
                "m_category": str(r["match_category"]), "f_category": str(b0["hub_category"]),
                "d_mf": np.nan, "d_fm": float(r["distance_l2"]),
                "rank_mf": -1, "rank_fm": int(r["rank"]),
                "kind": "FtoM", "score": float(r["distance_l2"]),
            })

        local_excl = set(m_ids) | set(f_ids)

        # Extra M→F hub if requested
        while len(mf_hubs) < n_hubs_mf:
            extra = pick_extra_mf(local_excl | used_ids)
            if extra is None:
                break
            hid, picks = extra
            h0 = picks[0]
            add_male({
                "id": hid,
                "theme": str(h0["hub_theme"]),
                "category": str(h0["hub_category"]),
                "role": "mf_hub",
            })
            mf_hubs.add(hid)
            local_excl.add(hid)
            for r in picks:
                fid = str(r["match_image_id"])
                add_female({
                    "id": fid,
                    "theme": str(r["match_theme"]),
                    "category": str(r["match_category"]),
                    "role": "mf_match",
                })
                local_excl.add(fid)
                edge_rows.append({
                    "m_id": hid, "f_id": fid,
                    "m_theme": str(h0["hub_theme"]), "f_theme": str(r["match_theme"]),
                    "m_category": str(h0["hub_category"]), "f_category": str(r["match_category"]),
                    "d_mf": float(r["distance_l2"]), "d_fm": np.nan,
                    "rank_mf": int(r["rank"]), "rank_fm": -1,
                    "kind": "MtoF", "score": float(r["distance_l2"]),
                })

        # Extra F→M hub if requested
        while len(fm_hubs) < n_hubs_fm:
            prefer = {f["id"] for f in females if "mf_match" in str(f.get("role", ""))}
            extra = pick_extra_fm(local_excl | used_ids | mf_hubs, prefer)
            if extra is None:
                break
            hid, picks = extra
            h0 = picks[0]
            add_female({
                "id": hid,
                "theme": str(h0["hub_theme"]),
                "category": str(h0["hub_category"]),
                "role": "fm_hub",
            })
            fm_hubs.add(hid)
            local_excl.add(hid)
            for r in picks:
                mid = str(r["match_image_id"])
                add_male({
                    "id": mid,
                    "theme": str(r["match_theme"]),
                    "category": str(r["match_category"]),
                    "role": "fm_match",
                })
                local_excl.add(mid)
                edge_rows.append({
                    "m_id": mid, "f_id": hid,
                    "m_theme": str(r["match_theme"]), "f_theme": str(h0["hub_theme"]),
                    "m_category": str(r["match_category"]), "f_category": str(h0["hub_category"]),
                    "d_mf": np.nan, "d_fm": float(r["distance_l2"]),
                    "rank_mf": -1, "rank_fm": int(r["rank"]),
                    "kind": "FtoM", "score": float(r["distance_l2"]),
                })

        edges = pd.DataFrame(edge_rows).sort_values("score").reset_index(drop=True)
        males, females = _barycentric_order(males, females, edges)
        used_ids |= m_ids | f_ids
        columns.append({
            "males": males,
            "females": females,
            "edges": edges,
            "mf_hubs": sorted(mf_hubs),
            "fm_hubs": sorted(fm_hubs),
            "bridges": bridges,
            "shared_target_ids": shared_targets,
            "n_match": n_match,
        })

    return columns



def _sankey_ribbon(
    ax,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: float,
    *,
    color: str,
    alpha: float = 0.38,
    zorder: int = 2,
) -> None:
    """Smooth cubic Bezier band between two anchor points (top→bottom)."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath

    half = max(float(width) / 2.0, 0.008)
    cy0 = y0 - 0.28 * (y0 - y1)
    cy1 = y1 + 0.28 * (y0 - y1)
    # left edge top→bottom, then right edge bottom→top
    verts = [
        (x0 - half, y0),
        (x0 - half, cy0),
        (x1 - half, cy1),
        (x1 - half, y1),
        (x1 + half, y1),
        (x1 + half, cy1),
        (x0 + half, cy0),
        (x0 + half, y0),
        (x0 - half, y0),
    ]
    codes = [
        MPath.MOVETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.LINETO,
        MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
        MPath.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(
            MPath(verts, codes),
            facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder, lw=0, clip_on=False,
        )
    )


def _edge_direction_weights(edges: pd.DataFrame) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Per-node pink (M→F) / cyan (F→M) connection mass for males and females."""
    m_pink: dict[str, float] = {}
    m_cyan: dict[str, float] = {}
    f_pink: dict[str, float] = {}
    f_cyan: dict[str, float] = {}
    if edges is None or len(edges) == 0:
        return m_pink, m_cyan, f_pink, f_cyan
    for _, e in edges.iterrows():
        mid, fid = str(e["m_id"]), str(e["f_id"])
        kind = str(e["kind"]) if "kind" in e.index else ""
        if kind == "MtoF" or (np.isfinite(e.get("d_mf", np.nan)) and not np.isfinite(e.get("d_fm", np.nan))):
            w = 1.0 / (float(e["d_mf"]) + 1e-3)
            m_pink[mid] = m_pink.get(mid, 0.0) + w
            f_pink[fid] = f_pink.get(fid, 0.0) + w
        elif kind == "FtoM" or (np.isfinite(e.get("d_fm", np.nan)) and not np.isfinite(e.get("d_mf", np.nan))):
            w = 1.0 / (float(e["d_fm"]) + 1e-3)
            m_cyan[mid] = m_cyan.get(mid, 0.0) + w
            f_cyan[fid] = f_cyan.get(fid, 0.0) + w
        else:
            # mutual / both present
            if np.isfinite(e.get("d_mf", np.nan)):
                w = 1.0 / (float(e["d_mf"]) + 1e-3)
                m_pink[mid] = m_pink.get(mid, 0.0) + w
                f_pink[fid] = f_pink.get(fid, 0.0) + w
            if np.isfinite(e.get("d_fm", np.nan)):
                w = 1.0 / (float(e["d_fm"]) + 1e-3)
                m_cyan[mid] = m_cyan.get(mid, 0.0) + w
                f_cyan[fid] = f_cyan.get(fid, 0.0) + w
    return m_pink, m_cyan, f_pink, f_cyan


def _order_by_band_pattern(
    column: dict,
    *,
    pattern: int = 1,
) -> tuple[list[dict], list[dict]]:
    """Order nodes so cyan (F→M) and pink (M→F) separate left/right.

    Pattern 1 — top-row view:
      left = cyan-heavy males, right = pink-heavy males;
      females follow the same cyan←left / pink→right logic.

    Pattern 2 — dual view:
      top: cyan-heavy males on the left;
      bottom: pink-heavy females on the right
      (female sort prioritizes pink-on-right over cyan-on-left).
    """
    edges = column.get("edges")
    m_pink, m_cyan, f_pink, f_cyan = _edge_direction_weights(edges)

    def male_key(n: dict) -> tuple:
        mid = n["id"]
        cyan = m_cyan.get(mid, 0.0)
        pink = m_pink.get(mid, 0.0)
        if pattern == 2:
            # left ← cyan-heavy (F→M into ♂); right → pink-heavy (M→F out of ♂)
            return (-(cyan - pink), -cyan, mid)
        return (-cyan, pink, mid)

    def female_key_p1(n: dict) -> tuple:
        fid = n["id"]
        return (-f_cyan.get(fid, 0.0), f_pink.get(fid, 0.0), fid)

    def female_key_p2(n: dict) -> tuple:
        fid = n["id"]
        cyan = f_cyan.get(fid, 0.0)
        pink = f_pink.get(fid, 0.0)
        # left ← cyan-heavy (F→M out of ♀); right → pink-heavy (M→F into ♀)
        return (pink - cyan, -cyan, fid)

    males = sorted(column["males"], key=male_key)
    if pattern == 2:
        females = sorted(column["females"], key=female_key_p2)
    else:
        females = sorted(column["females"], key=female_key_p1)
    return males, females


def _node_share_rank(node: dict, *, bridges: set[str], shared_ids: set[str]) -> tuple:
    """Lower = more shared / leftward. Bridge & shared first, then hubs, then unique."""
    nid = str(node["id"])
    role = str(node.get("role", ""))
    parts = set(role.split("+")) if role else set()
    if nid in bridges or ("fm_hub" in parts and "mf_match" in parts):
        return (0, nid)
    if nid in shared_ids or "shared" in parts:
        return (1, nid)
    if "mf_hub" in parts or parts == {"fm_hub"}:
        return (2, nid)
    return (3, nid)


def _order_shared_to_unique(column: dict) -> tuple[list[dict], list[dict]]:
    """Left←shared / bridge … unique→right within a column."""
    bridges = set(column.get("bridges") or [])
    shared_ids = set(column.get("shared_target_ids") or [])
    m_ids = {m["id"] for m in column["males"]}
    f_ids = {f["id"] for f in column["females"]}
    shared_ids |= (m_ids & f_ids)
    males = sorted(
        column["males"],
        key=lambda n: _node_share_rank(n, bridges=bridges, shared_ids=shared_ids),
    )
    females = sorted(
        column["females"],
        key=lambda n: _node_share_rank(n, bridges=bridges, shared_ids=shared_ids),
    )
    return males, females


def _va_for_node(meta: pd.DataFrame, image_id: str, *, gender: str) -> tuple[float, float] | None:
    """Return (V, A) for thumbnail caption: male row→male scores, female row→female scores."""
    hit = meta[meta["image_id"].astype(str) == str(image_id)]
    if hit.empty:
        return None
    row = hit.iloc[0]
    if gender == "male":
        v, a = row.get("valence_male"), row.get("arousal_male")
    else:
        v, a = row.get("valence_female"), row.get("arousal_female")
    if pd.isna(v) or pd.isna(a):
        v, a = row.get("valence"), row.get("arousal")
    if pd.isna(v) or pd.isna(a):
        return None
    return float(v), float(a)


def complete_sankey_edges_among_nodes(
    males: list[dict],
    females: list[dict],
    knn: pd.DataFrame,
    *,
    k: int = 3,
) -> pd.DataFrame:
    """All top-k M→F / F→M knn links between displayed male and female nodes.

    One row per (m_id, f_id); fills both ``d_mf`` and ``d_fm`` when reciprocal.
    Used for drawing after node order is fixed from selection edges.
    """
    m_meta = {str(m["id"]): m for m in males}
    f_meta = {str(f["id"]): f for f in females}
    m_ids, f_ids = set(m_meta), set(f_meta)
    pairs: dict[tuple[str, str], dict] = {}

    def _ensure(mid: str, fid: str) -> dict:
        key = (mid, fid)
        if key not in pairs:
            pairs[key] = {
                "m_id": mid,
                "f_id": fid,
                "m_theme": str(m_meta[mid]["theme"]),
                "f_theme": str(f_meta[fid]["theme"]),
                "m_category": str(m_meta[mid].get("category", "")),
                "f_category": str(f_meta[fid].get("category", "")),
                "d_mf": np.nan,
                "d_fm": np.nan,
                "rank_mf": -1,
                "rank_fm": -1,
                "kind": "",
                "score": np.nan,
            }
        return pairs[key]

    mt = knn[(knn["direction"] == "MtoF") & (knn["rank"] <= k)]
    ft = knn[(knn["direction"] == "FtoM") & (knn["rank"] <= k)]
    for _, r in mt.iterrows():
        mid, fid = str(r["hub_image_id"]), str(r["match_image_id"])
        if mid not in m_ids or fid not in f_ids:
            continue
        row = _ensure(mid, fid)
        d = float(r["distance_l2"])
        # keep closest if multiple ranks somehow collide
        if not np.isfinite(row["d_mf"]) or d < float(row["d_mf"]):
            row["d_mf"] = d
            row["rank_mf"] = int(r["rank"])
            row["m_theme"] = str(r["hub_theme"])
            row["f_theme"] = str(r["match_theme"])
            row["m_category"] = str(r["hub_category"])
            row["f_category"] = str(r["match_category"])
    for _, r in ft.iterrows():
        fid, mid = str(r["hub_image_id"]), str(r["match_image_id"])
        if mid not in m_ids or fid not in f_ids:
            continue
        row = _ensure(mid, fid)
        d = float(r["distance_l2"])
        if not np.isfinite(row["d_fm"]) or d < float(row["d_fm"]):
            row["d_fm"] = d
            row["rank_fm"] = int(r["rank"])
            row["m_theme"] = str(r["match_theme"])
            row["f_theme"] = str(r["hub_theme"])
            row["m_category"] = str(r["match_category"])
            row["f_category"] = str(r["hub_category"])

    rows = []
    for row in pairs.values():
        has_mf = np.isfinite(row["d_mf"])
        has_fm = np.isfinite(row["d_fm"])
        if has_mf and has_fm:
            row["kind"] = "both"
            row["score"] = 0.5 * (float(row["d_mf"]) + float(row["d_fm"]))
        elif has_mf:
            row["kind"] = "MtoF"
            row["score"] = float(row["d_mf"])
        else:
            row["kind"] = "FtoM"
            row["score"] = float(row["d_fm"])
        rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=[
                "m_id", "f_id", "m_theme", "f_theme", "m_category", "f_category",
                "d_mf", "d_fm", "rank_mf", "rank_fm", "kind", "score",
            ]
        )
    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True)


def draw_mutual_sankey_column(
    ax,
    column: dict,
    meta: pd.DataFrame,
    *,
    title: str,
    order_mode: str = "pattern1",
    knn: pd.DataFrame | None = None,
    k: int = 3,
    complete_edges: bool = True,
    ribbon_layout: str = "default",
) -> None:
    """One column: male images on top, female on bottom, pink M→F / cyan F→M ribbons.

    order_mode:
      - "pattern1": cyan←left / pink→right on both rows (top-row view)
      - "pattern2": top cyan←left; bottom pink→right
      - "shared": shared/bridge ← left
      - "barycentric": minimize crossings
      - "min_span": minimize weighted horizontal ribbon span |Δx|

    If ``complete_edges`` and ``knn`` are set, node order is computed from the
    selection edges first (preserving band layout), then ribbons are drawn for
    every top-k correspondence among the displayed nodes.
    """
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, pad=4)

    path_map = meta.set_index("image_id")["image_path"]
    edges_for_order: pd.DataFrame = column["edges"]
    male_list = list(column["males"])
    female_list = list(column["females"])

    if complete_edges and knn is not None:
        # Provisional order, then re-sort using completed knn edges so cyan←left / pink→right
        # works even when column["edges"] is empty (merged exploratory panels).
        if order_mode == "pattern1":
            males, females = _order_by_band_pattern(column, pattern=1)
        elif order_mode == "pattern2":
            males, females = _order_by_band_pattern(column, pattern=2)
        elif order_mode == "min_span":
            males, females = male_list, female_list
        elif order_mode == "shared":
            males, females = _order_shared_to_unique(column)
        else:
            males, females = _barycentric_order(male_list, female_list, edges_for_order)
        edges = complete_sankey_edges_among_nodes(males, females, knn, k=k)
        if order_mode == "min_span" and len(edges) > 0:
            males, females = _order_min_link_span(males, females, edges)
        elif order_mode in ("pattern1", "pattern2") and len(edges) > 0:
            col_edges = {**column, "males": males, "females": females, "edges": edges}
            males, females = _order_by_band_pattern(
                col_edges, pattern=1 if order_mode == "pattern1" else 2,
            )
    else:
        if order_mode == "pattern1":
            males, females = _order_by_band_pattern(column, pattern=1)
        elif order_mode == "pattern2":
            males, females = _order_by_band_pattern(column, pattern=2)
        elif order_mode == "min_span":
            males, females = _order_min_link_span(male_list, female_list, edges_for_order)
        elif order_mode == "shared":
            males, females = _order_shared_to_unique(column)
        else:
            males, females = _barycentric_order(male_list, female_list, edges_for_order)
        edges = edges_for_order

    def xs(n: int) -> np.ndarray:
        if n == 1:
            return np.array([0.5])
        return np.linspace(0.14, 0.86, n)

    mx = xs(len(males))
    fx = xs(len(females))
    lay = _mutual_sankey_layout(ribbon_layout)
    y_m, y_f = lay["y_m"], lay["y_f"]
    y_top, y_bot = lay["y_top"], lay["y_bot"]
    m_pos = {m["id"]: float(x) for m, x in zip(males, mx)}
    f_pos = {f["id"]: float(x) for f, x in zip(females, fx)}

    inv_vals = []
    for _, e in edges.iterrows():
        if np.isfinite(e.get("d_mf", np.nan)):
            inv_vals.append(1.0 / (float(e["d_mf"]) + 1e-3))
        if np.isfinite(e.get("d_fm", np.nan)):
            inv_vals.append(1.0 / (float(e["d_fm"]) + 1e-3))
    inv_max = float(max(inv_vals)) if inv_vals else 1.0

    def width_from_d(d: float, *, kind: str = "MtoF") -> float:
        # Cap thickness; pink (M→F) kept thinner than cyan (F→M)
        amp = 0.032 if kind == "MtoF" else 0.042
        return 0.008 + amp * ((1.0 / (d + 1e-3)) / inv_max)

    for _, e in edges.sort_values("score", ascending=False).iterrows():
        x0, x1 = m_pos.get(e["m_id"]), f_pos.get(e["f_id"])
        if x0 is None or x1 is None:
            continue
        has_mf = np.isfinite(e.get("d_mf", np.nan))
        has_fm = np.isfinite(e.get("d_fm", np.nan))
        if has_mf and has_fm:
            w_mf = width_from_d(float(e["d_mf"]), kind="MtoF")
            w_fm = width_from_d(float(e["d_fm"]), kind="FtoM")
            sep = 0.5 * max(w_mf, w_fm) + 0.008
            _sankey_ribbon(
                ax, x0 - sep, y_top, x1 - sep, y_bot, w_mf,
                color=COL_MTOF_PINK, alpha=0.45, zorder=2,
            )
            _sankey_ribbon(
                ax, x0 + sep, y_top, x1 + sep, y_bot, w_fm,
                color=COL_FTOM_CYAN, alpha=0.45, zorder=2,
            )
        elif has_mf:
            _sankey_ribbon(
                ax, x0, y_top, x1, y_bot, width_from_d(float(e["d_mf"]), kind="MtoF"),
                color=COL_MTOF_PINK, alpha=0.40, zorder=2,
            )
        elif has_fm:
            _sankey_ribbon(
                ax, x0, y_top, x1, y_bot, width_from_d(float(e["d_fm"]), kind="FtoM"),
                color=COL_FTOM_CYAN, alpha=0.40, zorder=2,
            )

    ax.text(0.02, 0.94, "♂ male", fontsize=8, color="#1565c0", fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.04, "♀ female", fontsize=8, color="#c62828", fontweight="bold", transform=ax.transAxes)

    def place(node: dict, x: float, y: float, border: str, *, gender: str) -> None:
        path = Path(str(path_map.get(node["id"], "")))
        n_nodes = max(len(males), len(females))
        if n_nodes <= 3:
            zoom, sz, fs = 0.58, 72, 5.0
        elif n_nodes <= 5:
            zoom, sz, fs = 0.45, 60, 4.8
        elif n_nodes <= 8:
            zoom, sz, fs = 0.32, 50, 4.2
        elif n_nodes <= 12:
            zoom, sz, fs = 0.24, 42, 3.6
        else:
            zoom, sz, fs = 0.20, 36, 3.2
        rgb = _load_rgb(path if path.exists() else None, size=sz)
        if rgb is None:
            return
        va = _va_for_node(meta, node["id"], gender=gender)
        va_txt = f"V={va[0]:.1f}  A={va[1]:.1f}" if va else "V=–  A=–"
        pack = VPacker(
            children=[
                OffsetImage(rgb, zoom=zoom),
                TextArea(
                    f"{node['theme']}\n({node['category']})\n{va_txt}",
                    textprops=dict(fontsize=fs, ha="center", color=border),
                ),
            ],
            align="center", pad=0, sep=1,
        )
        ab = AnnotationBbox(
            pack, (x, y), frameon=True,
            bboxprops=dict(boxstyle="round,pad=0.12", fc="white", ec=border, lw=1.4),
            pad=0.02, zorder=5,
        )
        ax.add_artist(ab)

    for m, x in zip(males, mx):
        place(m, float(x), y_m, "#1565c0", gender="male")
    for f, x in zip(females, fx):
        place(f, float(x), y_f, "#c62828", gender="female")


def _merge_sankey_columns(columns: list[dict]) -> dict:
    """Concatenate communities into one continuous row (shared communities first)."""
    males: list[dict] = []
    females: list[dict] = []
    seen_m: set[str] = set()
    seen_f: set[str] = set()
    bridges: list[str] = []
    shared: list[str] = []
    edge_parts: list[pd.DataFrame] = []
    mf_hubs: list[str] = []
    fm_hubs: list[str] = []

    for c in columns:
        for m in c["males"]:
            if m["id"] not in seen_m:
                males.append(m)
                seen_m.add(m["id"])
        for f in c["females"]:
            if f["id"] not in seen_f:
                females.append(f)
                seen_f.add(f["id"])
        bridges.extend(c.get("bridges") or [])
        shared.extend(c.get("shared_target_ids") or [])
        mf_hubs.extend(c.get("mf_hubs") or [])
        fm_hubs.extend(c.get("fm_hubs") or [])
        edge_parts.append(c["edges"])

    edges = pd.concat(edge_parts, ignore_index=True) if edge_parts else pd.DataFrame()
    merged = {
        "males": males,
        "females": females,
        "edges": edges,
        "bridges": list(dict.fromkeys(bridges)),
        "shared_target_ids": list(dict.fromkeys(shared)),
        "mf_hubs": list(dict.fromkeys(mf_hubs)),
        "fm_hubs": list(dict.fromkeys(fm_hubs)),
    }
    return merged


PERSON_THEME_BLOCK_RE = re.compile(
    r"nude|face|pose|miserable|angry pose|bored pose|neutral face|nude woman|nude man",
    re.I,
)

EXPLORATORY_HUB_THEMES = frozenset({
    "Nude woman 6", "Nude woman 9", "Nude woman 11", "Nude woman 14",
    "Nude man 22", "Nude man 7",
    "Nude couple 1", "Nude couple 2", "Nude couple 3", "Nude couple 4",
    "Nude couple 12", "Nude couple 13",
    "Wedding 1",
})

EXPLORATORY_LINKED_PATTERN_IDS = frozenset({
    "linked_dancing_8",
    "linked_gazing_5",
})

EXPLORATORY_PERSON_BLOCK_RE = re.compile(
    r"miserable|angry pose|bored pose|neutral face",
    re.I,
)


def _person_theme_ok(theme: str) -> bool:
    """Allow Person hubs/matches for Fig.5A when theme is not high-risk."""
    return not bool(PERSON_THEME_BLOCK_RE.search(str(theme)))


def _exploratory_person_theme_ok(theme: str) -> bool:
    """Relaxed Person filter for approved exploratory exemplars."""
    t = str(theme)
    if EXPLORATORY_PERSON_BLOCK_RE.search(t):
        return False
    if t in EXPLORATORY_HUB_THEMES:
        return True
    if re.match(r"^Nude ", t, re.I):
        return True
    return True


def _exploratory_match_display_ok(theme: str, category: str) -> bool:
    if category == "Person":
        return _exploratory_person_theme_ok(theme)
    return not bool(RISK_RE.search(str(theme)))


def _match_display_ok(theme: str, category: str) -> bool:
    if category == "Person":
        return _person_theme_ok(theme)
    return not bool(RISK_RE.search(str(theme)))


def _node_from_hub_row(r: pd.Series) -> dict:
    return {
        "id": str(r["hub_image_id"]),
        "theme": str(r["hub_theme"]),
        "category": str(r["hub_category"]),
    }


def _node_from_match_row(r: pd.Series) -> dict:
    return {
        "id": str(r["match_image_id"]),
        "theme": str(r["match_theme"]),
        "category": str(r["match_category"]),
    }


def _top_knn_matches(
    knn: pd.DataFrame,
    *,
    direction: str,
    hub_id: str,
    k: int = 3,
) -> list[pd.Series]:
    sub = knn[
        (knn["direction"] == direction)
        & (knn["hub_image_id"].astype(str) == str(hub_id))
        & (knn["rank"] <= k)
    ].sort_values(["rank", "distance_l2"])
    return [sub.iloc[i] for i in range(min(k, len(sub)))]


def _forward_person_scenario(
    knn: pd.DataFrame,
    hub_id: str,
    *,
    pattern_id: str,
    pattern_label: str,
) -> dict | None:
    rows = _top_knn_matches(knn, direction="FtoM", hub_id=hub_id, k=3)
    if len(rows) < 3:
        return None
    hub_row = rows[0]
    if hub_row["hub_category"] != "Person" or not _person_theme_ok(hub_row["hub_theme"]):
        return None
    if not all(_match_display_ok(r["match_theme"], r["match_category"]) for r in rows):
        return None
    matches = [_node_from_match_row(r) for r in rows]
    dists = [float(r["distance_l2"]) for r in rows]
    jcats = [m["category"] for m in matches]
    reverse = None
    person_match = next((m for m in matches if m["category"] == "Person"), None)
    if person_match is not None:
        rev_rows = _top_knn_matches(knn, direction="MtoF", hub_id=person_match["id"], k=3)
        if len(rev_rows) >= 3 and all(
            _match_display_ok(r["match_theme"], r["match_category"]) for r in rev_rows
        ):
            rev_matches = [_node_from_match_row(r) for r in rev_rows]
            reverse = {
                "direction": "MtoF",
                "hub": person_match,
                "matches": rev_matches,
                "distances": [float(r["distance_l2"]) for r in rev_rows],
                "match_categories": [m["category"] for m in rev_matches],
            }
    return {
        "pattern_id": pattern_id,
        "pattern_label": pattern_label,
        "forward": {
            "direction": "FtoM",
            "hub": _node_from_hub_row(hub_row),
            "matches": matches,
            "distances": dists,
            "match_categories": jcats,
        },
        "reverse": reverse,
    }


def enumerate_person_sankey_scenarios(knn: pd.DataFrame) -> list[dict]:
    """Curated Fig.5A Person oracle exemplars (female Person hub → top-3 male matches)."""
    ft_hubs = knn[(knn["direction"] == "FtoM") & (knn["rank"] == 1)].copy()
    ft_hubs = ft_hubs[ft_hubs["hub_category"] == "Person"]
    ft_hubs = ft_hubs[ft_hubs["hub_theme"].map(_person_theme_ok)]

    def pick_hub(theme: str) -> str | None:
        hit = ft_hubs[ft_hubs["hub_theme"] == theme]
        if hit.empty:
            return None
        return str(hit.iloc[0]["hub_image_id"])

    specs: list[tuple[str, str, str | None, callable | None]] = [
        (
            "cross3_celebration",
            "Forward only · 3 cross-category targets (no Person)",
            "Celebration 1",
            lambda s: "Person" not in s["forward"]["match_categories"],
        ),
        (
            "cross3_cheerleader2",
            "Forward only · Scene / Animal / Object",
            "Cheerleader 2",
            lambda s: "Person" not in s["forward"]["match_categories"],
        ),
        (
            "cross3_doctor1",
            "Forward only · Scene + Animal mix",
            "Doctor 1",
            lambda s: "Person" not in s["forward"]["match_categories"],
        ),
        (
            "incl_person_school1",
            "Forward · Person included among top-3",
            "School 1",
            lambda s: "Person" in s["forward"]["match_categories"],
        ),
        (
            "incl_person_doctor4",
            "Forward · Person + Scene + Animal",
            "Doctor 4",
            lambda s: "Person" in s["forward"]["match_categories"],
        ),
        (
            "incl_person_police3",
            "Forward · Person + Scene + Animal",
            "Police 3",
            lambda s: "Person" in s["forward"]["match_categories"],
        ),
        (
            "linked_football1",
            "Forward incl. Person match + reverse M→F (rev incl. Person)",
            "Football player 1",
            lambda s: s["reverse"] is not None and "Person" in s["reverse"]["match_categories"],
        ),
        (
            "linked_toast1",
            "Forward incl. Person match + reverse M→F (rev incl. Person)",
            "Toast 1",
            lambda s: s["reverse"] is not None and "Person" in s["reverse"]["match_categories"],
        ),
        (
            "linked_biking1",
            "Forward incl. Person match + reverse M→F (rev incl. Person)",
            "Biking 1",
            lambda s: s["reverse"] is not None and "Person" in s["reverse"]["match_categories"],
        ),
    ]

    scenarios: list[dict] = []
    for pid, label, theme, pred in specs:
        if theme is None:
            continue
        hid = pick_hub(theme)
        if hid is None:
            continue
        sc = _forward_person_scenario(knn, hid, pattern_id=pid, pattern_label=label)
        if sc is None:
            continue
        if pred is not None and not pred(sc):
            continue
        if not pid.startswith("linked_"):
            sc["reverse"] = None
        scenarios.append(sc)
    return scenarios


LINKED_PERSON_SENSITIVE_RE = re.compile(
    r"kkk|dead bod|severed|tumor|war \d|injury",
    re.I,
)


def _slug_from_theme(theme: str) -> str:
    s = re.sub(r"[^\w\s-]", "", str(theme).lower())
    s = re.sub(r"\s+", "_", s.strip())
    return s or "theme"


def _linked_person_sort_key(sc: dict) -> tuple:
    rev_n = sc["reverse"]["match_categories"].count("Person") if sc.get("reverse") else 0
    fwd_n = sc["forward"]["match_categories"].count("Person")
    return (-rev_n, -fwd_n, sc["forward"]["hub"]["theme"].lower())


def enumerate_linked_person_sankey_scenarios(
    knn: pd.DataFrame,
    *,
    exclude_sensitive: bool = False,
) -> list[dict]:
    """All F→M Person hubs whose top-3 includes Person with bidirectional M→F Person in rev top-3."""
    ft_hubs = knn[(knn["direction"] == "FtoM") & (knn["rank"] == 1)].copy()
    ft_hubs = ft_hubs[ft_hubs["hub_category"] == "Person"]
    ft_hubs = ft_hubs[ft_hubs["hub_theme"].map(_person_theme_ok)]

    scenarios: list[dict] = []
    seen_ids: set[str] = set()
    for _, row in ft_hubs.iterrows():
        hid = str(row["hub_image_id"])
        if hid in seen_ids:
            continue
        theme = str(row["hub_theme"])
        if exclude_sensitive and LINKED_PERSON_SENSITIVE_RE.search(theme):
            continue
        slug = _slug_from_theme(theme)
        sc = _forward_person_scenario(
            knn,
            hid,
            pattern_id=f"linked_{slug}",
            pattern_label="Bidirectional Person↔Person chain (F→M + M→F)",
        )
        if sc is None or sc.get("reverse") is None:
            continue
        if "Person" not in sc["forward"]["match_categories"]:
            continue
        if "Person" not in sc["reverse"]["match_categories"]:
            continue
        rev_hub = sc["reverse"]["hub"]
        if exclude_sensitive and LINKED_PERSON_SENSITIVE_RE.search(rev_hub["theme"]):
            continue
        sc["sensitive"] = bool(
            LINKED_PERSON_SENSITIVE_RE.search(theme)
            or LINKED_PERSON_SENSITIVE_RE.search(rev_hub["theme"])
        )
        seen_ids.add(hid)
        scenarios.append(sc)

    scenarios.sort(key=_linked_person_sort_key)
    return scenarios


def _save_person_sankey_figure(
    sc: dict,
    meta: pd.DataFrame,
    stem: Path,
    *,
    show_va: bool = False,
    panel_tag: str = "A",
    subtitle: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.8 if sc.get("reverse") else 4.6))
    ax.text(-0.02, 1.03, panel_tag, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
    draw_person_oracle_sankey_panel(
        ax, sc, meta, show_va=show_va, hub_thumb=False, match_thumb=True,
    )
    hub = sc["forward"]["hub"]
    cats = sc["forward"]["match_categories"]
    rev = sc.get("reverse")
    rev_note = ""
    if rev is not None:
        rev_n = rev["match_categories"].count("Person")
        rev_note = f" · rev Person×{rev_n}"
    fig.suptitle(
        subtitle
        or (
            f"{hub['theme']} ({hub['category']}) → {', '.join(cats)}{rev_note}"
            f"  ·  theme-CV oracle top-3"
        ),
        fontsize=10,
        y=0.98,
    )
    fig.text(
        0.5, 0.01,
        "Person query: theme label only. Match thumbnails: translation targets. "
        "Illustrative pairs; not a statistical summary.",
        ha="center", fontsize=7, color="0.45",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    for ext in (".png", ".svg"):
        p = stem.with_suffix(ext)
        kw = dict(bbox_inches="tight", facecolor="white")
        if ext == ".png":
            kw["dpi"] = 300
        else:
            kw["format"] = "svg"
        fig.savefig(p, **kw)
    plt.close(fig)


def _save_linked_person_sankey_overview_sheet(
    *,
    png_paths: list[Path],
    manifest: list[dict],
    out_path: Path,
    n_cols: int = 4,
) -> Path:
    """Contact sheet of linked Person sankey PNGs for quick browsing."""
    from matplotlib.image import imread

    n = len(png_paths)
    if n == 0:
        raise ValueError("No PNG paths for overview sheet")
    meta_by_id = {m["pattern_id"]: m for m in manifest}
    n_rows = int(np.ceil(n / n_cols))
    fig_w = 4.2 * n_cols
    fig_h = 3.4 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    axes_flat = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes_flat):
        ax.set_axis_off()
        if i >= n:
            continue
        png = png_paths[i]
        pid = png.stem
        m = meta_by_id.get(pid)
        img = imread(png)
        ax.imshow(img)
        if m:
            hub = m["forward_hub"]["theme"]
            pm = m["person_bridge"]["theme"]
            rev_n = sum(1 for c in m.get("reverse_match_categories", []) if c == "Person")
            tag = " [sensitive]" if m.get("sensitive") else ""
            ax.set_title(f"{hub} → {pm}\nrev Person×{rev_n}{tag}", fontsize=8)
        else:
            ax.set_title(png.stem, fontsize=8)

    fig.suptitle(
        "Fig. 5A linked Person↔Person candidates (F→M row + M→F reverse row)",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def export_fig5_panel_a_linked_person_sankey_patterns(
    *,
    out_dirs: tuple[Path, Path] | None = None,
    show_va: bool = False,
    exclude_sensitive: bool = False,
    make_overview_sheet: bool = True,
) -> dict[str, Path]:
    """Export all bidirectional Person↔Person linked sankey candidates."""
    if out_dirs is None:
        out_dirs = (ROOT / "Paper_fig", ROOT / "testfig")
    knn = load_knn()
    meta = load_oasis_meta(OASIS_SCORES_CSV)
    scenarios = enumerate_linked_person_sankey_scenarios(knn, exclude_sensitive=exclude_sensitive)
    if not scenarios:
        raise RuntimeError("No linked Person sankey scenarios found")

    written: dict[str, Path] = {}
    manifest: list[dict] = []

    for sc in scenarios:
        pid = sc["pattern_id"]
        for out_dir in out_dirs:
            sub = out_dir / f"{'Paper_Fig5' if out_dir.name == 'Paper_fig' else 'Fig5'}_panelA_person_sankey_linked"
            sub.mkdir(parents=True, exist_ok=True)
            stem = sub / pid
            _save_person_sankey_figure(
                sc,
                meta,
                stem,
                show_va=show_va,
                subtitle=(
                    f"linked | {sc['forward']['hub']['theme']} → "
                    f"{sc['reverse']['hub']['theme']} (Person bridge)"
                ),
            )
            written[pid] = stem.with_suffix(".png")
        manifest.append({
            "pattern_id": pid,
            "pattern_label": sc["pattern_label"],
            "sensitive": sc.get("sensitive", False),
            "forward_hub": sc["forward"]["hub"],
            "forward_match_categories": sc["forward"]["match_categories"],
            "forward_matches": sc["forward"]["matches"],
            "person_bridge": sc["reverse"]["hub"],
            "reverse_match_categories": sc["reverse"]["match_categories"],
            "reverse_matches": sc["reverse"]["matches"],
            "reverse": sc["reverse"],
        })

    for out_dir in out_dirs:
        prefix = "Paper_Fig5" if out_dir.name == "Paper_fig" else "Fig5"
        sub = out_dir / f"{prefix}_panelA_person_sankey_linked"
        idx_path = sub / f"{prefix}_panelA_person_sankey_linked_index.json"
        idx_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if make_overview_sheet:
            pngs = sorted(sub.glob("linked_*.png"))
            sheet = sub / f"{prefix}_panelA_person_sankey_linked_overview.png"
            _save_linked_person_sankey_overview_sheet(
                png_paths=pngs,
                manifest=manifest,
                out_path=sheet,
            )
            written[f"{prefix}_overview"] = sheet

    return written


def linked_person_scenario_to_column(sc: dict) -> dict:
    """Convert one linked Person scenario into a mutual-sankey column dict."""
    bridge = sc["reverse"]["hub"]
    males = [dict(m) for m in sc["forward"]["matches"]]
    fem_map: dict[str, dict] = {sc["forward"]["hub"]["id"]: dict(sc["forward"]["hub"])}
    for m in sc["reverse"]["matches"]:
        fem_map[m["id"]] = dict(m)
    females = list(fem_map.values())
    return {
        "males": males,
        "females": females,
        "edges": pd.DataFrame(),
        "bridges": [bridge["id"]],
        "shared_target_ids": [sc["forward"]["hub"]["id"]],
        "mf_hubs": [bridge["id"]],
        "fm_hubs": [sc["forward"]["hub"]["id"]],
        "pattern_id": sc.get("pattern_id", ""),
        "forward_hub_theme": sc["forward"]["hub"]["theme"],
        "person_bridge_theme": bridge["theme"],
    }


def _longest_linked_person_chain(scenarios: list[dict]) -> list[dict]:
    """Follow rev female match → next forward hub links; return longest chain."""
    by_id = {sc["pattern_id"]: sc for sc in scenarios}
    by_fhub = {sc["forward"]["hub"]["id"]: sc["pattern_id"] for sc in scenarios}

    def _next_ids(sc: dict) -> list[str]:
        out: list[str] = []
        for m in sc["reverse"]["matches"]:
            pid = by_fhub.get(m["id"])
            if pid and pid != sc["pattern_id"]:
                out.append(pid)
        return out

    def _walk(start_id: str) -> list[str]:
        best = [start_id]
        seen = {start_id}

        def rec(cur: str, path: list[str]) -> None:
            nonlocal best
            if len(path) > len(best):
                best = list(path)
            for nxt in _next_ids(by_id[cur]):
                if nxt in seen:
                    continue
                seen.add(nxt)
                rec(nxt, path + [nxt])
                seen.remove(nxt)

        rec(start_id, [start_id])
        return best

    best_path: list[str] = []
    for sc in scenarios:
        path = _walk(sc["pattern_id"])
        if len(path) > len(best_path):
            best_path = path
    return [by_id[pid] for pid in best_path]


def export_linked_person_mutual_sankey_chain(
    *,
    out_dir: Path | None = None,
    exclude_sensitive: bool = True,
    order_mode: str = "pattern2",
    k: int = 3,
    export_individual: bool = True,
    export_merged_chain: bool = True,
    top_n_individual: int = 8,
) -> dict[str, Path]:
    """Trial: Fig.6 mutual-sankey pattern2 layout for linked Person↔Person chains."""
    out_dir = out_dir or (ROOT / "testfig" / "Fig5_panelA_person_sankey_linked_safe")
    out_dir.mkdir(parents=True, exist_ok=True)
    knn = load_knn()
    meta = load_oasis_meta(OASIS_SCORES_CSV)
    scenarios = enumerate_linked_person_sankey_scenarios(knn, exclude_sensitive=exclude_sensitive)
    if not scenarios:
        raise RuntimeError("No linked Person scenarios for mutual sankey export")

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=COL_MTOF_PINK, edgecolor="none", alpha=0.7, label="M→F (pink)"),
        Patch(facecolor=COL_FTOM_CYAN, edgecolor="none", alpha=0.7, label="F→M (cyan)"),
    ]
    written: dict[str, Path] = {}

    def _save_column(stem: Path, col: dict, *, title: str, subtitle: str) -> None:
        n_m, n_f = len(col["males"]), len(col["females"])
        fig_w = max(9.0, 0.85 * max(n_m, n_f) + 3.5)
        fig, ax = plt.subplots(figsize=(fig_w, 7.6))
        draw_mutual_sankey_column(
            ax, col, meta, title=title, order_mode=order_mode,
            knn=knn, k=k, complete_edges=True,
        )
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)
        fig.suptitle(subtitle, fontsize=10.5, y=0.98)
        fig.text(
            0.5, 0.01,
            "Linked Person chain in mutual-sankey layout (pattern2). "
            "Illustrative oracle top-3 links among displayed nodes.",
            ha="center", fontsize=7, color="0.45",
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.90))
        for ext in (".png", ".svg"):
            p = stem.with_suffix(ext)
            kw = dict(bbox_inches="tight", facecolor="white")
            if ext == ".png":
                kw["dpi"] = 300
            else:
                kw["format"] = "svg"
            fig.savefig(p, **kw)
        plt.close(fig)
        written[stem.name] = stem.with_suffix(".png")

    if export_individual:
        for sc in scenarios[:top_n_individual]:
            col = linked_person_scenario_to_column(sc)
            males, females = _order_by_band_pattern(col, pattern=2 if order_mode == "pattern2" else 1)
            edges = complete_sankey_edges_among_nodes(males, females, knn, k=k)
            n_mf = int(np.isfinite(edges["d_mf"]).sum()) if len(edges) else 0
            n_fm = int(np.isfinite(edges["d_fm"]).sum()) if len(edges) else 0
            n_both = int((np.isfinite(edges["d_mf"]) & np.isfinite(edges["d_fm"])).sum()) if len(edges) else 0
            pid = sc["pattern_id"]
            hub = sc["forward"]["hub"]["theme"]
            bridge = sc["reverse"]["hub"]["theme"]
            stem = out_dir / f"Fig5_panelA_person_mutual_{pid}_{order_mode}"
            _save_column(
                stem,
                col,
                title=(
                    f"{hub} → {bridge} · {n_mf} M→F + {n_fm} F→M ({n_both} both)"
                ),
                subtitle=(
                    f"Fig. 5A trial | linked Person chain · {order_mode}\n"
                    f"{hub} (♀ query) ↔ {bridge} (♂ bridge) · theme-CV oracle top-{k}"
                ),
            )

    if export_merged_chain:
        chain = _longest_linked_person_chain(scenarios)
        if len(chain) >= 2:
            cols = [linked_person_scenario_to_column(sc) for sc in chain]
            merged = _merge_sankey_columns(cols)
            chain_label = " → ".join(
                f"{sc['forward']['hub']['theme']}→{sc['reverse']['hub']['theme']}" for sc in chain
            )
            stem = out_dir / f"Fig5_panelA_person_mutual_chain{len(chain)}_{order_mode}"
            _save_column(
                stem,
                merged,
                title=f"Merged chain ×{len(chain)} · {len(merged['males'])}♂ · {len(merged['females'])}♀",
                subtitle=(
                    f"Fig. 5A trial | merged Person chain ({len(chain)} hops) · {order_mode}\n"
                    f"{chain_label}"
                ),
            )

    manifest = {
        "order_mode": order_mode,
        "exclude_sensitive": exclude_sensitive,
        "n_scenarios": len(scenarios),
        "outputs": {k: str(v) for k, v in written.items()},
        "longest_chain": [
            {
                "pattern_id": sc["pattern_id"],
                "forward_hub": sc["forward"]["hub"]["theme"],
                "person_bridge": sc["reverse"]["hub"]["theme"],
            }
            for sc in _longest_linked_person_chain(scenarios)
        ],
    }
    (out_dir / "Fig5_panelA_person_mutual_sankey_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return written


def _forward_person_scenario_exploratory(
    knn: pd.DataFrame,
    hub_id: str,
    *,
    pattern_id: str,
    pattern_label: str,
) -> dict | None:
    """Like _forward_person_scenario but with exploratory nude allowlist."""
    rows = _top_knn_matches(knn, direction="FtoM", hub_id=hub_id, k=3)
    if len(rows) < 3:
        return None
    hub_row = rows[0]
    if hub_row["hub_category"] != "Person" or not _exploratory_person_theme_ok(hub_row["hub_theme"]):
        return None
    if not all(_exploratory_match_display_ok(r["match_theme"], r["match_category"]) for r in rows):
        return None
    matches = [_node_from_match_row(r) for r in rows]
    dists = [float(r["distance_l2"]) for r in rows]
    jcats = [m["category"] for m in matches]
    reverse = None
    person_match = next((m for m in matches if m["category"] == "Person"), None)
    if person_match is not None:
        rev_rows = _top_knn_matches(knn, direction="MtoF", hub_id=person_match["id"], k=3)
        if len(rev_rows) >= 3 and all(
            _exploratory_match_display_ok(r["match_theme"], r["match_category"]) for r in rev_rows
        ):
            rev_matches = [_node_from_match_row(r) for r in rev_rows]
            reverse = {
                "direction": "MtoF",
                "hub": person_match,
                "matches": rev_matches,
                "distances": [float(r["distance_l2"]) for r in rev_rows],
                "match_categories": [m["category"] for m in rev_matches],
            }
    return {
        "pattern_id": pattern_id,
        "pattern_label": pattern_label,
        "forward": {
            "direction": "FtoM",
            "hub": _node_from_hub_row(hub_row),
            "matches": matches,
            "distances": dists,
            "match_categories": jcats,
        },
        "reverse": reverse,
    }


def _hub_theme_to_id(knn: pd.DataFrame, theme: str, *, direction: str) -> str | None:
    hit = knn[
        (knn["direction"] == direction)
        & (knn["rank"] == 1)
        & (knn["hub_theme"] == theme)
    ]
    if hit.empty:
        return None
    return str(hit.iloc[0]["hub_image_id"])


def _oracle_hub_column(
    knn: pd.DataFrame,
    *,
    direction: str,
    theme: str,
    k: int = 3,
) -> dict | None:
    """Single-hub mutual-sankey column (♂ top / ♀ bottom)."""
    hid = _hub_theme_to_id(knn, theme, direction=direction)
    if hid is None:
        return None
    rows = _top_knn_matches(knn, direction=direction, hub_id=hid, k=k)
    if len(rows) < min(k, 3):
        return None
    if not all(_exploratory_match_display_ok(r["match_theme"], r["match_category"]) for r in rows):
        return None
    hub = _node_from_hub_row(rows[0])
    matches = [_node_from_match_row(r) for r in rows]
    if direction == "FtoM":
        males, females = matches, [hub]
    else:
        males, females = [hub], matches
    return {
        "males": males,
        "females": females,
        "edges": pd.DataFrame(),
        "bridges": [m["id"] for m in matches if m["category"] == "Person"][:1],
        "shared_target_ids": [hub["id"]],
        "mf_hubs": [m["id"] for m in matches if m["category"] == "Person"][:1] if direction == "FtoM" else [hub["id"]],
        "fm_hubs": [hub["id"]] if direction == "FtoM" else [m["id"] for m in matches if m["category"] == "Person"][:1],
        "hub_theme": theme,
        "direction": direction,
    }


def _save_mutual_sankey_column_figure(
    stem: Path,
    col: dict,
    meta: pd.DataFrame,
    knn: pd.DataFrame,
    *,
    order_mode: str = "pattern2",
    k: int = 3,
    title: str,
    subtitle: str,
    footer: str | None = None,
    legend_handles: list | None = None,
    ribbon_layout: str = "default",
    fig_height: float | None = None,
) -> Path:
    from matplotlib.patches import Patch

    if legend_handles is None:
        legend_handles = [
            Patch(facecolor=COL_MTOF_PINK, edgecolor="none", alpha=0.7, label="M→F (pink)"),
            Patch(facecolor=COL_FTOM_CYAN, edgecolor="none", alpha=0.7, label="F→M (cyan)"),
        ]
    n_m, n_f = len(col["males"]), len(col["females"])
    lay = _mutual_sankey_layout(ribbon_layout)
    fig_h = fig_height if fig_height is not None else lay["fig_h"]
    fig_w = max(9.0, 0.85 * max(n_m, n_f) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    draw_mutual_sankey_column(
        ax, col, meta, title=title, order_mode=order_mode,
        knn=knn, k=k, complete_edges=True, ribbon_layout=ribbon_layout,
    )
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)
    fig.suptitle(subtitle, fontsize=10.5, y=0.98)
    fig.text(
        0.5, 0.01,
        footer or (
            "Exploratory oracle top-3 (approved themes). "
            "Illustrative; not a statistical summary."
        ),
        ha="center", fontsize=7, color="0.45",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    for ext in (".png", ".svg"):
        p = stem.with_suffix(ext)
        kw = dict(bbox_inches="tight", facecolor="white")
        if ext == ".png":
            kw["dpi"] = 300
        else:
            kw["format"] = "svg"
        fig.savefig(p, **kw)
    plt.close(fig)
    return stem.with_suffix(".png")


def export_fig5_exploratory_mutual_sankey(
    *,
    out_dir: Path | None = None,
    order_mode: str = "pattern2",
    k: int = 3,
) -> dict[str, Path]:
    """Exploratory mutual-sankey panels for approved nude/wedding themes."""
    out_dir = out_dir or (ROOT / "testfig" / "Fig5_panelA_exploratory_mutual")
    out_dir.mkdir(parents=True, exist_ok=True)
    knn = load_knn()
    meta = load_oasis_meta(OASIS_SCORES_CSV)
    written: dict[str, Path] = {}
    manifest: list[dict] = []

    nude_woman = ["Nude woman 6", "Nude woman 9", "Nude woman 11", "Nude woman 14"]
    nude_man = ["Nude man 22", "Nude man 7"]
    nude_couple = [
        "Nude couple 1", "Nude couple 2", "Nude couple 3", "Nude couple 4",
        "Nude couple 12", "Nude couple 13",
    ]

    def _edge_stats(col: dict) -> tuple[int, int, int]:
        males, females = _order_by_band_pattern(col, pattern=2 if order_mode == "pattern2" else 1)
        edges = complete_sankey_edges_among_nodes(males, females, knn, k=k)
        if len(edges) == 0:
            return 0, 0, 0
        n_mf = int(np.isfinite(edges["d_mf"]).sum())
        n_fm = int(np.isfinite(edges["d_fm"]).sum())
        n_both = int((np.isfinite(edges["d_mf"]) & np.isfinite(edges["d_fm"])).sum())
        return n_mf, n_fm, n_both

    def _slug(theme: str) -> str:
        return _slug_from_theme(theme)

    for theme in nude_woman:
        col = _oracle_hub_column(knn, direction="FtoM", theme=theme, k=k)
        if col is None:
            continue
        n_mf, n_fm, n_both = _edge_stats(col)
        slug = _slug(theme)
        stem = out_dir / f"Fig5_exploratory_ftom_{slug}_{order_mode}"
        note = ""
        sc = _forward_person_scenario_exploratory(
            knn, _hub_theme_to_id(knn, theme, direction="FtoM") or "",
            pattern_id=f"exploratory_{slug}", pattern_label="exploratory",
        )
        if sc and sc.get("reverse"):
            rev_cats = sc["reverse"]["match_categories"]
            rev_themes = [m["theme"] for m in sc["reverse"]["matches"]]
            nude_rev = [t for t in rev_themes if re.match(r"^Nude man", t, re.I)]
            if nude_rev:
                note = f" · bridge {sc['reverse']['hub']['theme']} → {', '.join(nude_rev)}"
        png = _save_mutual_sankey_column_figure(
            stem, col, meta, knn, order_mode=order_mode, k=k,
            title=f"♀ {theme} → top-3 ♂ · {n_mf} M→F + {n_fm} F→M ({n_both} both){note}",
            subtitle=(
                f"Fig. 5A exploratory | F→M Person query · {order_mode}\n"
                f"{theme} (♀) → male-side oracle top-{k}{note}"
            ),
        )
        written[stem.name] = png
        manifest.append({"kind": "ftom_hub", "theme": theme, "note": note, "png": png.name})
        if sc and sc.get("reverse"):
            lcol = linked_person_scenario_to_column(sc)
            lstem = out_dir / f"Fig5_exploratory_linked_{slug}_{order_mode}"
            ln_mf, ln_fm, ln_both = _edge_stats(lcol)
            bridge = sc["reverse"]["hub"]["theme"]
            rev_nude = [m["theme"] for m in sc["reverse"]["matches"] if re.match(r"^Nude", m["theme"], re.I)]
            lnote = f" · rev nude: {', '.join(rev_nude)}" if rev_nude else ""
            lpng = _save_mutual_sankey_column_figure(
                lstem, lcol, meta, knn, order_mode=order_mode, k=k,
                title=f"linked ♀{theme} → ♂{bridge} · {ln_mf} M→F + {ln_fm} F→M{lnote}",
                subtitle=(
                    f"Fig. 5A exploratory | Person bridge · {order_mode}\n"
                    f"{theme} → {bridge} (♂) → female top-{k}{lnote}"
                ),
            )
            written[lstem.name] = lpng
            manifest.append({
                "kind": "linked", "theme": theme, "bridge": bridge,
                "reverse_nude": rev_nude, "png": lpng.name,
            })

    for theme in nude_man:
        for direction, prefix, arrow in [
            ("MtoF", "mtof", "♂ query → ♀ top-3"),
            ("FtoM", "ftom_rev", "♂ as F→M match hub (reverse view)"),
        ]:
            if direction == "MtoF":
                col = _oracle_hub_column(knn, direction="MtoF", theme=theme, k=k)
            else:
                col = _oracle_hub_column(knn, direction="FtoM", theme=theme, k=k)
            if col is None:
                continue
            n_mf, n_fm, n_both = _edge_stats(col)
            slug = _slug(theme)
            stem = out_dir / f"Fig5_exploratory_{prefix}_{slug}_{order_mode}"
            gender = "♂" if direction == "MtoF" else "♂ hub in F→M"
            png = _save_mutual_sankey_column_figure(
                stem, col, meta, knn, order_mode=order_mode, k=k,
                title=f"{gender} {theme} · {n_mf} M→F + {n_fm} F→M ({n_both} both)",
                subtitle=f"Fig. 5A exploratory | {arrow} · {order_mode}\n{theme}",
            )
            written[stem.name] = png
            manifest.append({"kind": prefix, "theme": theme, "direction": direction, "png": png.name})

    for theme in nude_couple:
        col = _oracle_hub_column(knn, direction="FtoM", theme=theme, k=k)
        if col is None:
            continue
        slug = _slug(theme)
        n_mf, n_fm, n_both = _edge_stats(col)
        stem = out_dir / f"Fig5_exploratory_couple_{slug}_{order_mode}"
        png = _save_mutual_sankey_column_figure(
            stem, col, meta, knn, order_mode=order_mode, k=k,
            title=f"♀ {theme} → top-3 ♂ · {n_mf} M→F + {n_fm} F→M ({n_both} both)",
            subtitle=f"Fig. 5A exploratory | Nude couple F→M · {order_mode}\n{theme}",
        )
        written[stem.name] = png
        manifest.append({"kind": "nude_couple", "theme": theme, "png": png.name})
        hid = _hub_theme_to_id(knn, theme, direction="FtoM")
        if hid:
            sc = _forward_person_scenario_exploratory(
                knn, hid, pattern_id=f"exploratory_{slug}", pattern_label="exploratory",
            )
            if sc and sc.get("reverse"):
                lcol = linked_person_scenario_to_column(sc)
                lstem = out_dir / f"Fig5_exploratory_couple_linked_{slug}_{order_mode}"
                bridge = sc["reverse"]["hub"]["theme"]
                rev_nude = [m["theme"] for m in sc["reverse"]["matches"] if re.match(r"^Nude", m["theme"], re.I)]
                lpng = _save_mutual_sankey_column_figure(
                    lstem, lcol, meta, knn, order_mode=order_mode, k=k,
                    title=f"linked {theme} → {bridge}",
                    subtitle=(
                        f"Fig. 5A exploratory | Nude couple bridge · {order_mode}\n"
                        f"{theme} → {bridge}"
                        + (f" · rev nude: {', '.join(rev_nude)}" if rev_nude else "")
                    ),
                )
                written[lstem.name] = lpng
                manifest.append({"kind": "couple_linked", "theme": theme, "bridge": bridge, "png": lpng.name})

    col_w1 = _oracle_hub_column(knn, direction="FtoM", theme="Wedding 1", k=k)
    if col_w1 is not None:
        n_mf, n_fm, n_both = _edge_stats(col_w1)
        stem = out_dir / f"Fig5_exploratory_wedding_1_ftom_{order_mode}"
        png = _save_mutual_sankey_column_figure(
            stem, col_w1, meta, knn, order_mode=order_mode, k=k,
            title=f"♀ Wedding 1 → Animal/Scene top-3 · {n_mf} M→F + {n_fm} F→M",
            subtitle=(
                f"Fig. 5A exploratory | Wedding 1 hub (no Person in top-{k}) · {order_mode}\n"
                "Cross-category oracle exemplar"
            ),
        )
        written[stem.name] = png
        manifest.append({"kind": "wedding_1", "png": png.name})

    all_linked = enumerate_linked_person_sankey_scenarios(knn, exclude_sensitive=False)
    for sc in all_linked:
        if sc["pattern_id"] not in EXPLORATORY_LINKED_PATTERN_IDS:
            continue
        col = linked_person_scenario_to_column(sc)
        pid = sc["pattern_id"]
        hub = sc["forward"]["hub"]["theme"]
        bridge = sc["reverse"]["hub"]["theme"]
        rev_themes = [m["theme"] for m in sc["reverse"]["matches"]]
        w9 = "Wedding 9" in rev_themes
        stem = out_dir / f"Fig5_exploratory_{pid}_wedding9_{order_mode}" if w9 else out_dir / f"Fig5_exploratory_{pid}_{order_mode}"
        n_mf, n_fm, n_both = _edge_stats(col)
        rev_note = " · rev includes Wedding 9" if w9 else ""
        png = _save_mutual_sankey_column_figure(
            stem, col, meta, knn, order_mode=order_mode, k=k,
            title=f"{hub} → {bridge} · {n_mf} M→F + {n_fm} F→M{rev_note}",
            subtitle=(
                f"Fig. 5A exploratory | Wedding 9 chain · {order_mode}\n"
                f"{hub} → {bridge} → {', '.join(rev_themes)}"
            ),
            footer="Wedding 9 highlighted chain. Illustrative oracle top-3; not a statistical summary.",
        )
        written[stem.name] = png
        manifest.append({
            "kind": "wedding9_linked", "pattern_id": pid,
            "forward_hub": hub, "bridge": bridge, "reverse": rev_themes, "png": png.name,
        })

    (out_dir / "Fig5_panelA_exploratory_mutual_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return written


EXPLORATORY_MERGED_NUDE_COUPLE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("ftom_rev", "FtoM", "Nude man 7"),
    ("couple_linked", "FtoM", "Nude couple 1"),
    ("couple_linked", "FtoM", "Nude couple 12"),
    ("couple_linked", "FtoM", "Nude couple 13"),
    ("couple", "FtoM", "Nude couple 1"),
    ("couple", "FtoM", "Nude couple 2"),
)


def _exploratory_column_from_spec(
    knn: pd.DataFrame,
    spec: tuple[str, str, str],
) -> dict | None:
    kind, direction, theme = spec
    if kind == "couple_linked":
        hid = _hub_theme_to_id(knn, theme, direction="FtoM")
        if hid is None:
            return None
        sc = _forward_person_scenario_exploratory(
            knn, hid, pattern_id=f"merged_{_slug_from_theme(theme)}", pattern_label="merged",
        )
        if sc is None or sc.get("reverse") is None:
            return None
        return linked_person_scenario_to_column(sc)
    return _oracle_hub_column(knn, direction=direction, theme=theme)


def export_fig5_exploratory_merged_nude_couple_sankey(
    *,
    out_dir: Path | None = None,
    specs: tuple[tuple[str, str, str], ...] | None = None,
    order_mode: str = "pattern2",
    k: int = 3,
    stem_name: str = "Fig5_exploratory_merged_nude_couple_selection",
) -> Path:
    """Merge selected exploratory mutual-sankey columns into one pattern2 panel."""
    out_dir = out_dir or (ROOT / "testfig" / "Fig5_panelA_exploratory_mutual")
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = specs or EXPLORATORY_MERGED_NUDE_COUPLE_SPECS
    knn = load_knn()
    meta = load_oasis_meta(OASIS_SCORES_CSV)

    columns: list[dict] = []
    labels: list[str] = []
    for spec in specs:
        col = _exploratory_column_from_spec(knn, spec)
        if col is None:
            raise RuntimeError(f"Could not build column for {spec}")
        columns.append(col)
        kind, _, theme = spec
        labels.append(f"{kind}:{theme}")

    merged = _merge_sankey_columns(columns)
    males, females = _order_by_band_pattern(merged, pattern=2 if order_mode == "pattern2" else 1)
    edges = complete_sankey_edges_among_nodes(males, females, knn, k=k)
    n_mf = int(np.isfinite(edges["d_mf"]).sum()) if len(edges) else 0
    n_fm = int(np.isfinite(edges["d_fm"]).sum()) if len(edges) else 0
    n_both = int((np.isfinite(edges["d_mf"]) & np.isfinite(edges["d_fm"])).sum()) if len(edges) else 0

    chain_bits = [
        "Nude man 7 (F→M hub)",
        "Nude couple 1 / 2 (F→M)",
        "linked couple 1 / 12 / 13 (Person bridges)",
    ]
    stem = out_dir / f"{stem_name}_{order_mode}"
    png = _save_mutual_sankey_column_figure(
        stem,
        merged,
        meta,
        knn,
        order_mode=order_mode,
        k=k,
        ribbon_layout="default",
        title=(
            f"Merged exploratory selection · {len(merged['males'])}♂ · {len(merged['females'])}♀ · "
            f"{n_mf} M→F + {n_fm} F→M ({n_both} both)"
        ),
        subtitle=(
            f"Fig. 5A exploratory | merged nude/couple selection · {order_mode}\n"
            + " · ".join(chain_bits)
        ),
        footer=(
            "Merged from: ftom_rev Nude man 7; couple 1–2; linked couple 1, 12, 13. "
            "Illustrative oracle top-3 among displayed nodes."
        ),
    )

    stem_compact = out_dir / f"{stem_name}_{order_mode}_compact_ribbons"
    png_compact = _save_mutual_sankey_column_figure(
        stem_compact,
        merged,
        meta,
        knn,
        order_mode=order_mode,
        k=k,
        ribbon_layout="compact",
        title=(
            f"Merged exploratory selection (compact ribbons) · "
            f"{len(merged['males'])}♂ · {len(merged['females'])}♀"
        ),
        subtitle=(
            f"Fig. 5A exploratory | merged nude/couple · {order_mode} · short ribbon span\n"
            + " · ".join(chain_bits)
        ),
        footer=(
            "Same node order as default layout; ribbon vertical span reduced. "
            "Illustrative oracle top-3 among displayed nodes."
        ),
    )

    m_span, f_span = _order_min_link_span(list(merged["males"]), list(merged["females"]), edges)
    span_cost = _ribbon_weighted_span_cost(m_span, f_span, edges)
    stem_min = out_dir / f"{stem_name}_min_span"
    png_min = _save_mutual_sankey_column_figure(
        stem_min,
        merged,
        meta,
        knn,
        order_mode="min_span",
        k=k,
        ribbon_layout="default",
        title=(
            f"Merged exploratory selection (min |Δx|) · "
            f"{len(merged['males'])}♂ · {len(merged['females'])}♀ · cost={span_cost:.1f}"
        ),
        subtitle=(
            "Fig. 5A exploratory | merged nude/couple · min horizontal ribbon span\n"
            + " · ".join(chain_bits)
        ),
        footer=(
            "Node order minimizes weighted sum of horizontal ribbon lengths (1/d oracle weight). "
            "Illustrative oracle top-3 among displayed nodes."
        ),
    )

    stem_min_compact = out_dir / f"{stem_name}_min_span_compact_ribbons"
    png_min_compact = _save_mutual_sankey_column_figure(
        stem_min_compact,
        merged,
        meta,
        knn,
        order_mode="min_span",
        k=k,
        ribbon_layout="compact",
        title=(
            f"Merged exploratory selection (min |Δx|, compact) · "
            f"{len(merged['males'])}♂ · {len(merged['females'])}♀"
        ),
        subtitle=(
            "Fig. 5A exploratory | min horizontal span + short vertical ribbons\n"
            + " · ".join(chain_bits)
        ),
        footer=(
            "Same min-|Δx| node order; ribbon vertical span reduced. "
            "Illustrative oracle top-3 among displayed nodes."
        ),
    )

    manifest = {
        "order_mode": order_mode,
        "specs": [{"kind": s[0], "direction": s[1], "theme": s[2]} for s in specs],
        "n_males": len(merged["males"]),
        "n_females": len(merged["females"]),
        "n_edges": len(edges),
        "n_mf": n_mf,
        "n_fm": n_fm,
        "n_both": n_both,
        "male_themes": [m["theme"] for m in merged["males"]],
        "female_themes": [f["theme"] for f in merged["females"]],
        "png": png.name,
        "png_compact_ribbons": png_compact.name,
        "png_min_span": png_min.name,
        "png_min_span_compact_ribbons": png_min_compact.name,
        "ribbon_weighted_span_cost_min_span": span_cost,
        "ribbon_layouts": MUTUAL_SANKEY_RIBBON_LAYOUTS,
    }
    (out_dir / f"{stem_name}_{order_mode}_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return png


def _draw_person_sankey_row(
    ax,
    *,
    hub: dict,
    matches: list[dict],
    distances: list[float],
    meta: pd.DataFrame,
    y_hub: float,
    y_match: float,
    ribbon_color: str,
    hub_gender: str,
    match_gender: str,
    hub_thumb: bool,
    match_thumb: bool,
    show_va: bool,
    row_label: str,
) -> None:
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker

    path_map = meta.set_index("image_id")["image_path"]
    n = len(matches)
    xs = np.linspace(0.16, 0.84, n) if n > 1 else np.array([0.5])
    x_hub = 0.5

    inv = [1.0 / (d + 1e-3) for d in distances]
    inv_max = max(inv) if inv else 1.0

    def ribbon_width(d: float) -> float:
        return 0.010 + 0.034 * ((1.0 / (d + 1e-3)) / inv_max)

    for x, m, d in zip(xs, matches, distances):
        _sankey_ribbon(
            ax, x_hub, y_hub, float(x), y_match, ribbon_width(d),
            color=ribbon_color, alpha=0.42, zorder=2,
        )

    def place_node(node: dict, x: float, y: float, *, thumb: bool, gender: str, border: str) -> None:
        lines = [f"{node['theme']}", f"({node['category']})"]
        if show_va:
            va = _va_for_node(meta, node["id"], gender=gender)
            if va:
                lines.append(f"V={va[0]:.1f}  A={va[1]:.1f}")
        children = [TextArea("\n".join(lines), textprops=dict(fontsize=8.5, ha="center", color=border))]
        if thumb:
            path = Path(str(path_map.get(node["id"], "")))
            rgb = _load_rgb(path if path.exists() else None, size=56)
            if rgb is not None:
                children = [OffsetImage(rgb, zoom=0.42)] + children
        pack = VPacker(children=children, align="center", pad=0, sep=2)
        ab = AnnotationBbox(
            pack, (x, y), frameon=True, zorder=5,
            bboxprops=dict(boxstyle="round,pad=0.12", fc="white", ec=border, lw=1.3),
            pad=0.02,
        )
        ax.add_artist(ab)

    hub_border = "#c62828" if hub_gender == "female" else "#1565c0"
    match_border = "#1565c0" if match_gender == "male" else "#c62828"
    place_node(hub, x_hub, y_hub, thumb=hub_thumb, gender=hub_gender, border=hub_border)
    for x, m in zip(xs, matches):
        place_node(m, float(x), y_match, thumb=match_thumb, gender=match_gender, border=match_border)
    ax.text(
        0.01, y_hub + 0.06, row_label, transform=ax.transAxes,
        fontsize=8, color="0.35", ha="left", va="bottom",
    )


def draw_person_oracle_sankey_panel(
    ax,
    scenario: dict,
    meta: pd.DataFrame,
    *,
    show_va: bool = False,
    hub_thumb: bool = False,
    match_thumb: bool = True,
) -> None:
    """Single-direction Person oracle sankey; optional reverse row below."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fwd = scenario["forward"]
    rev = scenario.get("reverse")
    if rev is None:
        _draw_person_sankey_row(
            ax,
            hub=fwd["hub"],
            matches=fwd["matches"],
            distances=fwd["distances"],
            meta=meta,
            y_hub=0.78,
            y_match=0.22,
            ribbon_color=COL_FTOM_CYAN,
            hub_gender="female",
            match_gender="male",
            hub_thumb=hub_thumb,
            match_thumb=match_thumb,
            show_va=show_va,
            row_label="♀ Person query → male-side oracle matches (F→M)",
        )
    else:
        _draw_person_sankey_row(
            ax,
            hub=fwd["hub"],
            matches=fwd["matches"],
            distances=fwd["distances"],
            meta=meta,
            y_hub=0.88,
            y_match=0.58,
            ribbon_color=COL_FTOM_CYAN,
            hub_gender="female",
            match_gender="male",
            hub_thumb=hub_thumb,
            match_thumb=match_thumb,
            show_va=show_va,
            row_label="♀ Person query → male-side matches",
        )
        _draw_person_sankey_row(
            ax,
            hub=rev["hub"],
            matches=rev["matches"],
            distances=rev["distances"],
            meta=meta,
            y_hub=0.42,
            y_match=0.08,
            ribbon_color=COL_MTOF_PINK,
            hub_gender="male",
            match_gender="female",
            hub_thumb=False,
            match_thumb=match_thumb,
            show_va=show_va,
            row_label="♂ Person match as query → female-side matches (M→F)",
        )
    ax.set_title(scenario.get("pattern_label", ""), fontsize=9, pad=6)


def export_fig5_panel_a_person_sankey_patterns(
    *,
    out_dirs: tuple[Path, Path] | None = None,
    show_va: bool = False,
) -> dict[str, Path]:
    """Export several Fig.5A candidate panels: Person hub → top-3 oracle matches."""
    if out_dirs is None:
        out_dirs = (ROOT / "Paper_fig", ROOT / "testfig")
    knn = load_knn()
    meta = load_oasis_meta(OASIS_SCORES_CSV)
    scenarios = enumerate_person_sankey_scenarios(knn)
    if not scenarios:
        raise RuntimeError("No Person sankey scenarios found")

    written: dict[str, Path] = {}
    manifest: list[dict] = []

    for sc in scenarios:
        pid = sc["pattern_id"]
        for out_dir in out_dirs:
            out_dir.mkdir(parents=True, exist_ok=True)
            prefix = "Paper_Fig5" if out_dir.name == "Paper_fig" else "Fig5"
            stem = out_dir / f"{prefix}_panelA_person_sankey_{pid}"
            _save_person_sankey_figure(
                sc,
                meta,
                stem,
                show_va=show_va,
                subtitle=(
                    f"Fig. 5A candidate | {sc['forward']['hub']['theme']} "
                    f"({sc['forward']['hub']['category']}) → "
                    f"{', '.join(sc['forward']['match_categories'])}"
                ),
            )
            written[pid] = stem.with_suffix(".png")
        manifest.append({
            "pattern_id": pid,
            "pattern_label": sc["pattern_label"],
            "forward_hub": sc["forward"]["hub"],
            "forward_match_categories": sc["forward"]["match_categories"],
            "forward_matches": sc["forward"]["matches"],
            "reverse": sc.get("reverse"),
        })

    for out_dir in out_dirs:
        prefix = "Paper_Fig5" if out_dir.name == "Paper_fig" else "Fig5"
        idx_path = out_dir / f"{prefix}_panelA_person_sankey_index.json"
        idx_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return written


def export_fig6_mutual_sankey(
    *,
    out_dir: Path | None = None,
    n_cols: int = 2,
    n_groups: int = 5,
    n_per_side: int = 3,
    n_fill: int = 5,
    k: int = 3,
) -> list[Path]:
    """Continuous horizontal sankey variants; pink=M→F, cyan=F→M.

    Pattern 1: cyan←left / pink→right on both rows.
    Pattern 2: top cyan←left, bottom pink→right.
    Exports ~n_groups distinct image communities × both patterns, plus merged mains.
    """
    out_dir = out_dir or (ROOT / "Paper_fig")
    out_dir.mkdir(parents=True, exist_ok=True)
    knn = load_knn()
    meta = load_oasis_meta(OASIS_SCORES_CSV)
    n_want = max(n_cols, n_groups)
    columns = select_asymmetric_sankey_columns(
        knn, n_cols=n_want, n_per_side=None, k=k,
        n_hubs_mf=2, n_hubs_fm=2, n_match=3, n_bridge=1,
    )
    if not columns:
        raise RuntimeError("No asymmetric M→F / F→M communities found for sankey export")

    def _col_share_score(c: dict) -> tuple:
        return (len(c.get("bridges") or []), len(c.get("shared_target_ids") or []))

    columns = sorted(columns, key=_col_share_score, reverse=True)
    groups = columns[:n_groups]

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=COL_MTOF_PINK, edgecolor="none", alpha=0.7, label="M→F (pink)"),
        Patch(facecolor=COL_FTOM_CYAN, edgecolor="none", alpha=0.7, label="F→M (cyan)"),
    ]

    def _save_strip(
        stem: Path,
        col: dict,
        *,
        order_mode: str,
        subtitle: str,
        panel_title: str,
    ) -> list[Path]:
        n_m, n_f = len(col["males"]), len(col["females"])
        fig_w = max(9.0, 0.85 * max(n_m, n_f) + 3.5)
        fig, ax = plt.subplots(figsize=(fig_w, 7.6))
        draw_mutual_sankey_column(
            ax, col, meta, title=panel_title, order_mode=order_mode,
            knn=knn, k=k, complete_edges=True,
        )
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)
        fig.suptitle(subtitle, fontsize=10.5, y=0.98)
        fig.tight_layout(rect=(0, 0.02, 1, 0.90))
        out = []
        for ext in (".png", ".svg"):
            p = stem.with_suffix(ext)
            kw = dict(bbox_inches="tight", facecolor="white")
            if ext == ".png":
                kw["dpi"] = 300
            else:
                kw["format"] = "svg"
            fig.savefig(p, **kw)
            out.append(p)
        plt.close(fig)
        return out

    def _edge_counts(col: dict) -> tuple[int, int, int]:
        """(n_mf, n_fm, n_both) after completing knn links among displayed nodes."""
        males, females = _order_by_band_pattern(col, pattern=1)
        edges = complete_sankey_edges_among_nodes(males, females, knn, k=k)
        if len(edges) == 0:
            return 0, 0, 0
        n_mf = int(np.isfinite(edges["d_mf"]).sum())
        n_fm = int(np.isfinite(edges["d_fm"]).sum())
        n_both = int((np.isfinite(edges["d_mf"]) & np.isfinite(edges["d_fm"])).sum())
        return n_mf, n_fm, n_both

    written: list[Path] = []

    # Per-group pattern galleries (~5 each)
    group_manifest = []
    for gi, col in enumerate(groups, start=1):
        nmf, nfm, nboth = _edge_counts(col)
        bridge_themes = []
        id2t = {n["id"]: n["theme"] for n in col["males"] + col["females"]}
        for bid in col.get("bridges") or []:
            bridge_themes.append(id2t.get(bid, bid))
        tag = "+".join(bridge_themes[:2]) if bridge_themes else f"g{gi}"
        tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)[:40]

        written += _save_strip(
            out_dir / f"Paper_Fig6_mutual_sankey_pattern1_g{gi}_{tag}",
            col,
            order_mode="pattern1",
            panel_title=(
                f"Pattern 1 · group {gi}/{len(groups)} · "
                f"{nmf} M→F + {nfm} F→M ({nboth} both) · bridge={tag}"
            ),
            subtitle=(
                f"Fig. 6 (extra) | Pattern 1 · image group {gi}\n"
                "Top view: cyan F→M ←left · pink M→F →right (same on bottom)  ·  V/A under thumbs"
            ),
        )
        written += _save_strip(
            out_dir / f"Paper_Fig6_mutual_sankey_pattern2_g{gi}_{tag}",
            col,
            order_mode="pattern2",
            panel_title=(
                f"Pattern 2 · group {gi}/{len(groups)} · "
                f"{nmf} M→F + {nfm} F→M ({nboth} both) · bridge={tag}"
            ),
            subtitle=(
                f"Fig. 6 (extra) | Pattern 2 · image group {gi}\n"
                "Top: cyan F→M ←left  ·  Bottom: pink M→F →right  ·  V/A under thumbs"
            ),
        )
        group_manifest.append({
            "group": gi,
            "tag": tag,
            "bridges": col.get("bridges", []),
            "bridge_themes": bridge_themes,
            "shared_target_ids": col.get("shared_target_ids", []),
            "n_males": len(col["males"]),
            "n_females": len(col["females"]),
            "n_mf": nmf,
            "n_fm": nfm,
            "n_both": nboth,
            "files": {
                "pattern1": f"Paper_Fig6_mutual_sankey_pattern1_g{gi}_{tag}.png",
                "pattern2": f"Paper_Fig6_mutual_sankey_pattern2_g{gi}_{tag}.png",
            },
        })

    # Merged of first 2 communities → canonical 2col / pattern1 / pattern2 / linked
    merged = _merge_sankey_columns(groups[: max(2, min(2, len(groups)))])
    n_m, n_f = len(merged["males"]), len(merged["females"])
    n_mf, n_fm, n_both = _edge_counts(merged)
    # Persist completed draw edges for the index (order still from selection edges)
    m_ord, f_ord = _order_by_band_pattern(merged, pattern=1)
    merged["edges_drawn"] = complete_sankey_edges_among_nodes(m_ord, f_ord, knn, k=k)

    written += _save_strip(
        out_dir / "Paper_Fig6_mutual_sankey_2col",
        merged,
        order_mode="pattern1",
        panel_title=(
            f"Pattern 1 (merged)  ·  {n_mf} M→F + {n_fm} F→M "
            f"({n_both} both)  ·  ♂{n_m} / ♀{n_f}"
        ),
        subtitle=(
            "Fig. 6 (extra) | Pattern 1 — continuous strip (♂ top · ♀ bottom)\n"
            "All top-k correspondences among shown images  ·  V/A under thumbs"
        ),
    )
    written += _save_strip(
        out_dir / "Paper_Fig6_mutual_sankey_pattern1",
        merged,
        order_mode="pattern1",
        panel_title="Pattern 1 · cyan←left / pink→right (both rows)",
        subtitle=(
            "Fig. 6 (extra) | Pattern 1\n"
            "Top view: cyan F→M ←left · pink M→F →right (same logic on bottom)"
        ),
    )
    written += _save_strip(
        out_dir / "Paper_Fig6_mutual_sankey_pattern2",
        merged,
        order_mode="pattern2",
        panel_title="Pattern 2 · top cyan←left · bottom pink→right",
        subtitle=(
            "Fig. 6 (extra) | Pattern 2\n"
            "Top: cyan F→M denser on the left  ·  Bottom: pink M→F denser on the right"
        ),
    )
    written += _save_strip(
        out_dir / "Paper_Fig6_mutual_sankey_linked",
        merged,
        order_mode="pattern1",
        panel_title=(
            f"Linked continuous (= pattern 1)  ·  {n_mf} M→F + {n_fm} F→M ({n_both} both)"
        ),
        subtitle=(
            "Fig. 6 (extra) | Continuous sankey (pattern 1 ordering)\n"
            "All top-k M→F / F→M links among displayed images (order fixed)"
        ),
    )

    def _titles(col: dict, i: int) -> str:
        nmf, nfm, nboth = _edge_counts(col)
        return (
            f"Community {i + 1}  ·  {nmf} M→F + {nfm} F→M ({nboth} both)"
            f"  ·  bridge={len(col.get('bridges') or [])}, "
            f"shared={len(col.get('shared_target_ids') or [])}"
        )

    for i, col in enumerate(groups[:2]):
        written += _save_strip(
            out_dir / f"Paper_Fig6_mutual_sankey_col{i + 1}",
            col,
            order_mode="pattern1",
            panel_title=_titles(col, i),
            subtitle="♂ top · ♀ bottom  ·  pattern 1 ordering  ·  V/A under thumbs",
        )

    def _node_out(n: dict, gender: str) -> dict:
        va = _va_for_node(meta, n["id"], gender=gender)
        out = dict(n)
        if va:
            out["valence"] = va[0]
            out["arousal"] = va[1]
        return out

    manifest = {
        "k": k,
        "n_groups": len(groups),
        "n_fill": n_fill,
        "selection": "asymmetric_bridge_band_patterns",
        "draw_edges": "complete all top-k knn links among displayed nodes after fixing pattern order",
        "order": "pattern1: cyan left/pink right both rows; pattern2: top cyan left, bottom pink right",
        "va_scores": "male row = valence_male/arousal_male; female row = valence_female/arousal_female",
        "colors": {"MtoF": COL_MTOF_PINK, "FtoM": COL_FTOM_CYAN},
        "merged_edge_counts": {"n_mf": n_mf, "n_fm": n_fm, "n_both": n_both},
        "files": {
            "pattern1_main": "Paper_Fig6_mutual_sankey_2col.svg",
            "pattern1": "Paper_Fig6_mutual_sankey_pattern1.svg",
            "pattern2": "Paper_Fig6_mutual_sankey_pattern2.svg",
            "linked": "Paper_Fig6_mutual_sankey_linked.svg",
            "groups": group_manifest,
        },
        "merged": {
            "males": [_node_out(n, "male") for n in merged["males"]],
            "females": [_node_out(n, "female") for n in merged["females"]],
            "bridges": merged.get("bridges", []),
            "shared_target_ids": merged.get("shared_target_ids", []),
            "edges_drawn": [
                {
                    "m_id": str(e["m_id"]),
                    "f_id": str(e["f_id"]),
                    "m_theme": str(e["m_theme"]),
                    "f_theme": str(e["f_theme"]),
                    "d_mf": None if not np.isfinite(e.get("d_mf", np.nan)) else float(e["d_mf"]),
                    "d_fm": None if not np.isfinite(e.get("d_fm", np.nan)) else float(e["d_fm"]),
                    "rank_mf": int(e["rank_mf"]),
                    "rank_fm": int(e["rank_fm"]),
                    "kind": str(e["kind"]),
                }
                for _, e in merged.get("edges_drawn", pd.DataFrame()).iterrows()
            ],
        },
        "columns": [
            {
                "males": [_node_out(n, "male") for n in c["males"]],
                "females": [_node_out(n, "female") for n in c["females"]],
                "mf_hubs": c.get("mf_hubs", []),
                "fm_hubs": c.get("fm_hubs", []),
                "bridges": c.get("bridges", []),
                "shared_target_ids": c.get("shared_target_ids", []),
                "edges_selection": [
                    {
                        "m_theme": str(e["m_theme"]),
                        "f_theme": str(e["f_theme"]),
                        "d_mf": None if not np.isfinite(e.get("d_mf", np.nan)) else float(e["d_mf"]),
                        "d_fm": None if not np.isfinite(e.get("d_fm", np.nan)) else float(e["d_fm"]),
                        "rank_mf": int(e["rank_mf"]),
                        "rank_fm": int(e["rank_fm"]),
                        "kind": str(e["kind"]) if "kind" in e.index else "MtoF",
                    }
                    for _, e in c["edges"].iterrows()
                ],
                "edges_drawn": [
                    {
                        "m_theme": str(e["m_theme"]),
                        "f_theme": str(e["f_theme"]),
                        "d_mf": None if not np.isfinite(e.get("d_mf", np.nan)) else float(e["d_mf"]),
                        "d_fm": None if not np.isfinite(e.get("d_fm", np.nan)) else float(e["d_fm"]),
                        "rank_mf": int(e["rank_mf"]),
                        "rank_fm": int(e["rank_fm"]),
                        "kind": str(e["kind"]),
                    }
                    for _, e in complete_sankey_edges_among_nodes(
                        c["males"], c["females"], knn, k=k,
                    ).iterrows()
                ],
            }
            for c in groups
        ],
    }
    (out_dir / "Paper_Fig6_mutual_sankey_index.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return written


def _case_translation_panel_context() -> tuple[list[dict], list[dict], pd.DataFrame, pd.DataFrame, str, object]:
    """Shared data for Fig.5/6 case-translation panels A and D."""
    knn = load_knn()
    spread_sources = select_spread_source_hubs(knn)
    strip_srcs = [
        s for s in spread_sources
        if str(s["hub"]["hub_theme"]) in {"Bird 4", "Crow 2", "Flood 3", "Snow 2"}
    ]
    f_src = next((s for s in strip_srcs if s["direction"] == "FtoM" and s["hub"]["hub_theme"] == "Bird 4"), None)
    if f_src is None:
        f_src = next((s for s in strip_srcs if s["direction"] == "FtoM"), None)
    m_src = next((s for s in strip_srcs if s["direction"] == "MtoF"), None)
    strip_pick = [s for s in (f_src, m_src) if s is not None]
    hub_bundles = hub_bundles_from_spread(knn, strip_pick, n_match=N_MATCH_PER_HUB)
    meta = load_oasis_meta(OASIS_SCORES_CSV)
    flow_m, flow_meta = load_category_flow_pairs()
    _, cross, _ = category_transition_matrix(flow_m)
    perm = flow_meta.get("category_permutation", {})
    same_rate = 1.0 - cross
    null_same = perm.get("permutation_null_same_category_mean")
    p_same = perm.get("permutation_p_same_category_above_chance")
    if null_same is not None and p_same is not None:
        p_txt = "p<0.001" if float(p_same) < 0.001 else f"p={float(p_same):.2f}"
        c_note = f"theme-CV; same-cat {same_rate:.0%} vs null≈{null_same:.0%} ({p_txt})"
    else:
        c_note = f"mode={flow_meta.get('mode', 'unknown')}"
    return hub_bundles, spread_sources, flow_m, meta, c_note, flow_meta


def export_fig5_case_translation_ad(
    *,
    out_dirs: tuple[Path, Path] | None = None,
) -> dict[str, Path]:
    """Export Fig.5A (exemplars) and Fig.5D (category flow) for Paper."""
    if out_dirs is None:
        out_dirs = (ROOT / "Paper_fig", ROOT / "testfig")
    hub_bundles, spread_sources, flow_m, meta, c_note, _ = _case_translation_panel_context()
    written: dict[str, Path] = {}

    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = "Paper_Fig5" if out_dir.name == "Paper_fig" else "Fig5"

        fig_a, ax_a = plt.subplots(figsize=(13.5, 6.2))
        ax_a.text(-0.02, 1.04, "A", transform=ax_a.transAxes, fontsize=13, fontweight="bold", va="bottom")
        ax_a.set_title(
            "Sources on VA (spread)  |  1 hub → 3 matches (image strip)",
            fontsize=9, pad=6,
        )
        draw_exemplar_panel(fig_a, ax_a, hub_bundles, meta, spread_sources=spread_sources)
        fig_a.tight_layout()
        stem_a = out_dir / f"{prefix}_panelA_hub_matches"
        fig_a.savefig(stem_a.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig_a.savefig(stem_a.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
        plt.close(fig_a)
        written[f"{prefix}_panelA"] = stem_a.with_suffix(".png")

        fig_d, ax_d = plt.subplots(figsize=(5.4, 4.9))
        ax_d.text(-0.08, 1.04, "D", transform=ax_d.transAxes, fontsize=13, fontweight="bold", va="bottom")
        draw_category_flow_panel(ax_d, flow_m, subtitle=c_note)
        fig_d.suptitle("Semantic category flow (theme-CV n=900)", fontsize=10, y=1.02)
        fig_d.tight_layout()
        stem_d = out_dir / f"{prefix}_panelD_category_flow"
        fig_d.savefig(stem_d.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
        fig_d.savefig(stem_d.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
        plt.close(fig_d)
        written[f"{prefix}_panelD"] = stem_d.with_suffix(".png")

    return written


def export_fig5_spatial_panels(
    m: pd.DataFrame | None = None,
    *,
    grid: float = 1.0,
    out_dirs: tuple[Path, Path] | None = None,
) -> dict[str, Path]:
    """Export Fig.5B (Φ residual) and Fig.5C (equiv distance) as standalone panels."""
    if m is None:
        m = build_merged()
    if out_dirs is None:
        out_dirs = (ROOT / "Paper_fig", ROOT / "testfig")
    written: dict[str, Path] = {}
    specs = [
        ("panelB_phi_residual_va", "residual", "B"),
        ("panelC_equiv_partner_va", "equiv_dist", "C"),
    ]
    for stem, field, letter in specs:
        for out_dir in out_dirs:
            out_dir.mkdir(parents=True, exist_ok=True)
            prefix = "Paper_Fig5" if out_dir.name == "Paper_fig" else "Fig5"
            fig, ax = plt.subplots(figsize=(5.4, 4.8))
            ax.text(-0.08, 1.04, letter, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
            draw_spatial_single_panel(fig, ax, m, field=field, grid=grid)
            fig.tight_layout()
            png = out_dir / f"{prefix}_{stem}.png"
            svg = out_dir / f"{prefix}_{stem}.svg"
            fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
            fig.savefig(svg, format="svg", bbox_inches="tight", facecolor="white")
            plt.close(fig)
            written[f"{prefix}_{stem}"] = png
    return written


def main() -> None:
    test = ROOT / "testfig" / "Fig6_case_translation_discovery.png"
    paper_png = ROOT / "Paper_fig" / "Paper_Fig6_case_translation.png"
    paper_svg = ROOT / "Paper_fig" / "Paper_Fig6_case_translation.svg"
    hub_bundles, manifest = make_fig6_discovery(test, None)
    print("Prototype:", test)
    print("Spread sources (left VA):")
    for s in manifest.get("spread_sources", []):
        print(
            f"  [{s['direction']}] {s['gender_tag']} {s['hub_theme']} "
            f"VA=({s['hub_va'][0]:.2f},{s['hub_va'][1]:.2f})"
        )
    print("Hub bundles (right strips):")
    for b in manifest["hub_bundles"]:
        matches = ", ".join(
            f"{m['match_theme']}(#{m['rank']},d={m['distance_l2']:.3f})"
            for m in b["matches"]
        )
        print(f"  [{b['direction']}] {b['hub_theme']} ({b['hub_category']}) → {matches}")
    make_fig6_discovery(paper_png, paper_svg)
    print("Paper:", paper_png)
    print("SVG:", paper_svg)
    fig5_paths = export_fig5_spatial_panels()
    fig5_ad = export_fig5_case_translation_ad()
    from plot_predictive_translation_prototype import export_fig5_panel_e_predictive_category
    fig5_e = export_fig5_panel_e_predictive_category()
    print("Fig.5 B/C panels:")
    for k, p in fig5_paths.items():
        print(f"  {k}: {p}")
    print("Fig.5 A/D panels:")
    for k, p in fig5_ad.items():
        print(f"  {k}: {p}")
    print("Fig.5 E panel:")
    for k, p in fig5_e.items():
        print(f"  {k}: {p}")
    print("n hubs drawn:", len(hub_bundles))
    extra = export_fig6_1to3_exemplars(n_per_dir=6)
    print(f"1→3 exemplars written: {len(extra)} files")
    for p in extra:
        if p.suffix == ".png":
            print(" ", p.name)
    sankey = export_fig6_mutual_sankey(n_cols=2, n_groups=5, k=3)
    print(f"Mutual sankey written: {len(sankey)} files")
    for p in sankey:
        if p.suffix == ".png":
            print(" ", p.name)
    person_a = export_fig5_panel_a_person_sankey_patterns()
    print(f"Fig.5A Person sankey patterns: {len(person_a)}")
    for pid, p in person_a.items():
        print(f"  {pid}: {p.name}")
    linked_a = export_fig5_panel_a_linked_person_sankey_patterns()
    n_linked = sum(1 for k in linked_a if k.startswith("linked_"))
    print(f"Fig.5A linked Person↔Person patterns: {n_linked}")
    for pid, p in sorted(linked_a.items()):
        if pid.startswith("linked_"):
            print(f"  {pid}: {p.name}")
    if "Fig5_overview" in linked_a:
        print(f"  overview sheet: {linked_a['Fig5_overview'].name}")


if __name__ == "__main__":
    main()
