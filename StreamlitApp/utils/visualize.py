# Visualize page: 3D (py3Dmol / Plotly), helical wheel, known-AMP similarity, map HTML, shape blurbs.
from __future__ import annotations

import csv
import math
import pathlib
from typing import Any, List, Optional, Tuple

import numpy as np

# Fallback if `Data/ampData.csv` is missing (e.g. local dev without Data/).
_FALLBACK_KNOWN_AMPS: Tuple[str, ...] = (
    "KWKLFKKIGAVLKVL",
    "GIGKFLHSAKKFGKAFVGEIMNS",
    "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLV",
    "KLFKKILKYL",
    "FLPLLAGLAANFLPKIFCKITRKC",
)

def _amp_data_csv_path() -> pathlib.Path:
    # `Data/ampData.csv`: label=1 rows become KNOWN_AMPS for “similar AMP” lookup.
    # StreamlitApp/utils/visualize.py -> repo root is parents[2]
    return pathlib.Path(__file__).resolve().parents[2] / "Data" / "ampData.csv"


def _load_known_amps_from_csv() -> List[str]:
    # Load unique AMP-labeled sequences from CSV and normalize to uppercase.
    path = _amp_data_csv_path()
    if not path.exists():
        return list(_FALLBACK_KNOWN_AMPS)

    seen: set[str] = set()
    amps: List[str] = []
    try:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "sequence" not in reader.fieldnames:
                return list(_FALLBACK_KNOWN_AMPS)
            for row in reader:
                label = str(row.get("label", "")).strip()
                if label != "1":
                    continue
                raw = (row.get("sequence") or "").strip()
                if not raw:
                    continue
                seq = raw.upper()
                if seq in seen:
                    continue
                seen.add(seq)
                amps.append(seq)
    except Exception:
        return list(_FALLBACK_KNOWN_AMPS)

    return amps if amps else list(_FALLBACK_KNOWN_AMPS)


# Known AMP pool for similarity search (from ampData.csv label=1, or fallback list).
KNOWN_AMPS: List[str] = _load_known_amps_from_csv()

# py3Dmol viewer: skip very long sequences (labels + sticks scale with length).
MAX_3D_SEQUENCE_LENGTH: int = 60

STRUCTURE_3D_LEGEND_MARKDOWN: str = """
**Color legend**
- **Blue:** Positively charged residues (K, R, H)  
- **Red:** Negatively charged residues (D, E)  
- **Green:** Hydrophobic residues (A, V, I, L, M, F, W, Y)  
- **Gray:** Other / polar or unclassified residues  
"""

STRUCTURE_3D_INTERPRETATION_MARKDOWN: str = """
**Structural interpretation (approximation only)**

This is a **simplified helical CA trace** used to visualize how residue chemistry is arranged in 3D space, **not** an experimentally determined fold.

- **Clusters of green** often correspond to membrane-facing / hydrophobic patches.  
- **Blue regions** highlight cationic residues that can promote binding to anionic bacterial surfaces.  
- **Spatial separation** between hydrophobic and charged segments can suggest **amphipathic** character, common among many AMPs.  

Together, these cues help discuss whether a sequence has motifs frequently associated with antimicrobial peptides, **wet-lab validation is still required**.
"""

# One-letter -> three-letter (for minimal PDB lines for py3Dmol).
_ONE_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


def sequence_similarity(seq1: str, seq2: str) -> float:
    # Compute simple position-wise match score normalized by the longer sequence.
    if not seq1 or not seq2:
        return 0.0
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / max(len(seq1), len(seq2))


def find_most_similar(sequence: str) -> Tuple[Optional[str], float]:
    # Return the closest known AMP and its simple position-match similarity score.
    if not sequence or not KNOWN_AMPS:
        return None, 0.0
    seq = "".join(c for c in sequence.upper() if not c.isspace())
    if not seq:
        return None, 0.0
    best_seq = KNOWN_AMPS[0]
    best_score = sequence_similarity(seq, KNOWN_AMPS[0])
    for amp in KNOWN_AMPS[1:]:
        score = sequence_similarity(seq, amp)
        if score > best_score:
            best_score = score
            best_seq = amp
    return best_seq, best_score


def get_residue_color(aa: str) -> str:
    # Map one-letter residue codes to py3Dmol color categories.
    ch = aa.upper() if aa else ""
    positive = ["K", "R", "H"]
    negative = ["D", "E"]
    hydrophobic = ["A", "V", "I", "L", "M", "F", "W", "Y"]
    if ch in positive:
        return "blue"
    if ch in negative:
        return "red"
    if ch in hydrophobic:
        return "green"
    return "gray"


def residue_color_mpl(aa: str) -> str:
    # Return high-contrast Matplotlib colors that mirror the 3D residue categories.
    cat = get_residue_color(aa)
    return {
        "blue": "#1D4ED8",
        "red": "#DC2626",
        "green": "#16A34A",
        "gray": "#57534E",
    }.get(cat, "#57534E")


HELIX_WHEEL_LEGEND_MARKDOWN: str = """
**Helical wheel readout**
- **Blue wedge:** cationic (K, R, H), often important for initial membrane association.  
- **Red wedge:** anionic (D, E).  
- **Green wedge:** hydrophobic, often grouped on one face in amphipathic helices (membrane-facing).  
- **Gray:** polar / other, may participate in solubility or hydrogen bonding.  

Residues are placed using a **100° step** per position (common α-helical wheel convention). This is a **2D projection**, not a solved 3D structure.
"""

# Short blurbs for compact UI expanders (Visualize Peptide page)
COMPACT_3D_LEGEND: str = """
**How to read this 3D view**
- **Plotly:** thick gray **backbone line** + colored residue markers (interactive rotation).
- **3Dmol:** gray **cylinder backbone** between Cα positions + colored spheres (same chemistry colors).
- **Blue:** positively charged residues (K, R, H)
- **Red:** negatively charged residues (D, E)
- **Green:** hydrophobic residues (A, V, I, L, M, F, W, Y)
- **Gray:** other / polar residues
- Geometry is a **helix-like approximation**, not an experimental structure.
"""
COMPACT_WHEEL_LEGEND: str = """
**How to read this helical wheel**
- **Radial spokes:** residue positions around the helix (100 degrees per residue)
- **Black connectors:** sequence order (`i -> i+1`) across the wheel
- **Colored circles:** residue chemistry classes
- Color mapping matches the 3D view (**blue / red / green / gray**)
"""
COMPACT_MAP_LEGEND: str = """
**How to read this sequence map**
- Uses the same residue color mapping as 3D and helical wheel
- Highlights where charged vs hydrophobic residues cluster along the sequence
- Useful for quick amphipathic pattern checks
"""


def plot_helical_wheel(sequence: str, figsize: Tuple[float, float] = (6.2, 6.2)) -> Any:
    # Polar wheel: 100°/residue, same phase as `helix_coordinates` / 3D trace (not a solved structure).
    import matplotlib.pyplot as plt
    from matplotlib import patheffects as pe

    # Normalize user input to whitespace-free uppercase sequence.
    clean = "".join(c for c in (sequence or "").upper() if not c.isspace())
    n = len(clean)
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("white")
    if n == 0:
        ax.set_facecolor("#ffffff")
        ax.set_title("Helical wheel (empty sequence)", pad=12)
        return fig

    ax.set_facecolor("#ffffff")

    angles_deg = np.array([i * 100.0 for i in range(n)], dtype=float) % 360.0
    angles_rad = np.deg2rad(angles_deg)
    r_inner, r_ring = 0.06, 0.88
    fs = max(7, min(11, int(220 / max(n, 1))))
    pt_size = float(np.clip(8000.0 / max(n, 1), 130.0, 420.0))

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Radial spokes (residue positions)
    for i in range(n):
        th = angles_rad[i]
        ax.plot(
            [th, th],
            [r_inner, r_ring],
            color="#1a1a1a",
            linewidth=0.65,
            alpha=0.45,
            zorder=1,
        )

    # Sequence-order connections (straight chords in the plane, classic wheel “star”)
    for i in range(n - 1):
        ax.plot(
            [angles_rad[i], angles_rad[i + 1]],
            [r_ring, r_ring],
            color="#0a0a0a",
            linewidth=1.05,
            solid_capstyle="round",
            zorder=2,
        )

    # Draw residue nodes after spokes/connectors so labels stay readable.
    colors = [residue_color_mpl(aa) for aa in clean]
    ax.scatter(
        angles_rad,
        np.full(n, r_ring),
        s=pt_size,
        c=colors,
        edgecolors="#111111",
        linewidths=1.2,
        zorder=4,
    )

    for i, aa in enumerate(clean):
        # Put residue letters on the wheel so users can visually match positions.
        t = ax.text(
            angles_rad[i],
            r_ring,
            aa,
            ha="center",
            va="center",
            fontsize=fs,
            color="#0a0a0a",
            fontweight="bold",
            zorder=5,
        )
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])

    ax.set_ylim(0, 1.0)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    ax.set_title(
        "Helical wheel (α-helix, 100°/residue), spokes + sequence connectors",
        pad=14,
        fontsize=11,
        color="#111111",
    )
    return fig


def get_residue_style(aa: str) -> str:
    # Return inline CSS style for sequence-map residue coloring.
    positive = ["K", "R", "H"]
    negative = ["D", "E"]
    hydrophobic = ["A", "V", "I", "L", "M", "F", "W", "Y"]
    if aa in positive:
        return "background-color: #1D4ED8; color: #ffffff; padding: 2px 3px; border-radius: 2px;"
    if aa in negative:
        return "background-color: #DC2626; color: #ffffff; padding: 2px 3px; border-radius: 2px;"
    if aa in hydrophobic:
        return "background-color: #16A34A; color: #ffffff; padding: 2px 3px; border-radius: 2px;"
    return "background-color: #57534E; color: #ffffff; padding: 2px 3px; border-radius: 2px;"


def build_importance_map_html(sequence: str) -> str:
    # Build safe HTML spans for residue-by-residue chemical highlighting.
    import html as html_mod

    # Emit one colored <span> per residue for inline sequence highlighting.
    parts: List[str] = []
    for ch in sequence:
        if ch.isspace():
            continue
        aa = ch.upper()
        style = get_residue_style(aa)
        parts.append(f'<span style="{style}">{html_mod.escape(aa)}</span>')
    return "".join(parts)


def helix_coordinates(sequence: str, *, smooth: bool = False) -> np.ndarray:
    # Shared CA trace used by PDB, Plotly, and py3Dmol (same geometry as the helical wheel).
    clean = "".join(c for c in (sequence or "").upper() if not c.isspace())
    n = len(clean)
    if n == 0:
        return np.zeros((0, 3), dtype=float)

    theta_step = 100.0 * math.pi / 180.0  # ~α-helix angular step on the wheel
    rise = 1.45
    coords: List[Tuple[float, float, float]] = []
    for i in range(n):
        angle = i * theta_step
        r = 5.0 + 0.12 * math.sin(i * 0.4)
        x = math.cos(angle) * r
        y = math.sin(angle) * r
        z = i * rise
        coords.append((x, y, z))

    if smooth and n >= 3:
        # Light smoothing makes the 3D backbone look less jagged.
        xs = np.array([c[0] for c in coords], dtype=float)
        ys = np.array([c[1] for c in coords], dtype=float)
        zs = np.array([c[2] for c in coords], dtype=float)
        k = np.array([0.2, 0.6, 0.2])
        for _ in range(2):
            xs = np.convolve(xs, k, mode="same")
            ys = np.convolve(ys, k, mode="same")
            zs = np.convolve(zs, k, mode="same")
        xs[0], xs[-1] = coords[0][0], coords[-1][0]
        ys[0], ys[-1] = coords[0][1], coords[-1][1]
        zs[0], zs[-1] = coords[0][2], coords[-1][2]
        coords = list(zip(xs.tolist(), ys.tolist(), zs.tolist()))

    return np.array(coords, dtype=float)


def generate_helix_pdb(sequence: str, smooth: bool = False) -> str:
    # Minimal CA-only helix-like PDB for py3Dmol (coordinates only; bonds drawn via cylinders).
    pdb_lines: List[str] = []
    atom_index = 1
    clean = "".join(c for c in sequence.upper() if not c.isspace())
    n = len(clean)
    if n == 0:
        return ""

    coords = helix_coordinates(clean, smooth=smooth)
    for i, aa in enumerate(clean):
        res_name = _ONE_TO_THREE.get(aa, "UNK")
        x, y, z = float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])
        res_num = i + 1
        pdb_lines.append(
            f"ATOM  {atom_index:5d}  CA  {res_name:3s} A{res_num:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        atom_index += 1
    return "\n".join(pdb_lines)


def residue_shape_label(aa: str) -> str:
    # Short chemistry label for hovers and shape summary text.
    cat = get_residue_color(aa)
    return {
        "blue": "cationic",
        "red": "anionic",
        "green": "hydrophobic",
        "gray": "polar / other",
    }.get(cat, "polar / other")


def _helical_wheel_resultant(indices: List[int]) -> float:
    # Circular mean length in [0, 1]: high values mean residues cluster on one face of the wheel.
    if len(indices) < 2:
        return 0.0
    angles = [math.radians((i * 100.0) % 360.0) for i in indices]
    vx = sum(math.cos(a) for a in angles) / len(angles)
    vy = sum(math.sin(a) for a in angles) / len(angles)
    return float(math.hypot(vx, vy))


# Heuristic bullets from wheel geometry + residue classes; not a second classifier.
def build_shape_visual_summary(
    sequence: str,
    *,
    amp_label: Optional[str] = None,
    amp_prob: Optional[float] = None,
) -> List[str]:
    # Short bullets tying the helix/wheel geometry to AMP-relevant “shape chemistry” (heuristic).
    clean = "".join(c for c in (sequence or "").upper() if not c.isspace())
    n = len(clean)
    lines: List[str] = []
    if n == 0:
        return lines

    lines.append(
        "This view places residues on a **helix-like CA trace** (same geometry as the wheel). "
    )

    pos_i = [i for i, aa in enumerate(clean) if get_residue_color(aa) == "blue"]
    neg_i = [i for i, aa in enumerate(clean) if get_residue_color(aa) == "red"]
    hyd_i = [i for i, aa in enumerate(clean) if get_residue_color(aa) == "green"]
    pol_i = [i for i, aa in enumerate(clean) if get_residue_color(aa) == "gray"]

    # Fractions and resultant scores describe how residues are distributed on the helix face.
    f_h = len(hyd_i) / n
    f_p = len(pol_i) / n
    f_pos = len(pos_i) / n

    R_h = _helical_wheel_resultant(hyd_i)
    R_k = _helical_wheel_resultant(pos_i)

    if f_h >= 0.18 and f_p >= 0.12:
        lines.append(
            "You can point to **both** a **hydrophobic** (green) and **polar / other** (gray) presence along the trace,"
            "a common ingredient for **interface** behavior (aqueous vs lipid-facing), which many AMP mechanisms exploit."
        )
    elif f_h >= 0.25 and f_p < 0.1:
        lines.append(
            "The trace is **dominated by hydrophobic** (green) positions; without much polar (gray) or cationic (blue) balance, "
            "membrane engagement can be less like classic cationic AMP helices (still sequence-context dependent)."
        )
    elif f_p >= 0.35 and f_h < 0.15:
        lines.append(
            "The trace is **rich in polar / other** (gray) and light on hydrophobic (green) packing, often more soluble, "
            "but less like a compact amphipathic helix unless charge or hydrophobic content appears elsewhere."
        )

    if len(hyd_i) >= 3 and R_h >= 0.52:
        lines.append(
            "**Hydrophobic residues cluster on one side** of the helical wheel (tight arc), consistent with an **amphipathic** "
            "helix face that could sit at the **membrane interface**."
        )
    elif len(hyd_i) >= 2 and R_h < 0.35:
        lines.append(
            "**Hydrophobic** (green) positions are **spread** around the wheel, less of a single membrane-facing stripe; "
            "some AMPs still look like this, but classic amphipathic faces are easier to see when green groups on one arc."
        )

    if len(pos_i) >= 2 and R_k >= 0.5:
        lines.append(
            "**Cationic** (blue) residues group in angular space, helpful for a **localized positive patch** toward anionic lipids, "
            "a pattern often discussed for membrane-targeting peptides."
        )

    if amp_label is not None and amp_prob is not None:
        p = float(amp_prob)
        pred_conf = round(p * 100, 1) if amp_label == "AMP" else round((1.0 - p) * 100, 1)
        if amp_label == "AMP" and pred_conf >= 65:
            lines.append(
                f"**Model:** AMP at **{pred_conf}%** confidence on this sequence, combined with the spatial pattern above, "
                "use the plot to argue **where** positive charge and hydrophobic bulk sit relative to each other."
            )
        elif amp_label == "Non-AMP" and pred_conf >= 65:
            lines.append(
                f"**Model:** Non-AMP at **{pred_conf}%** confidence, if the trace still **looks** amphipathic, treat that as "
                "**chemistry vs. classifier** tension worth testing in the lab, not proof of activity."
            )
        else:
            lines.append(
                f"**Model:** **{amp_label}** (about **{pred_conf}%** on that call), read the **shape** bullets as physical intuition; "
                "they do not override the model or experiments."
            )

    # De-duplicate, cap length.
    out: List[str] = []
    seen: set[str] = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out[:12]


def render_3d_plotly(
    sequence: str,
    *,
    height: int = 460,
) -> bool:
    # Plotly: CA helix trace + residue markers (same geometry as wheel / 3Dmol).
    try:
        import plotly.graph_objects as go
        import streamlit as st
    except Exception:
        return False

    clean = "".join(c for c in (sequence or "").upper() if not c.isspace())
    if not clean:
        return False
    if len(clean) > MAX_3D_SEQUENCE_LENGTH:
        return False

    coords = helix_coordinates(clean, smooth=True)
    if coords.shape[0] == 0:
        return False

    colors = [residue_color_mpl(aa) for aa in clean]
    labels = [residue_shape_label(aa) for aa in clean]
    hover = [f"{i + 1} {aa} · {labels[i]}" for i, aa in enumerate(clean)]

    msize = float(np.clip(900.0 / max(len(clean), 1), 3.5, 11.0))
    show_text = len(clean) <= 36
    text_pos = "top center" if len(clean) <= 24 else "middle center"

    fig = go.Figure()
    # Backbone line trace follows the helix-like CA coordinates.
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="lines",
            line=dict(color="rgba(110,110,118,0.92)", width=12),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Markers trace shows residue chemistry colors (and letters for shorter sequences).
    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode="markers+text" if show_text else "markers",
            marker=dict(
                size=msize,
                color=colors,
                line=dict(color="#1a1a1a", width=0.8),
            ),
            text=list(clean) if show_text else None,
            textposition=text_pos,
            textfont=dict(size=max(9, min(12, int(220 / max(len(clean), 1)))), color="#111111"),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            name="Residues",
        )
    )

    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=36, b=0),
        paper_bgcolor="#fafafa",
        title=dict(
            text="Helix-like CA trace (approximation) · drag to rotate",
            font=dict(size=13, color="#333333"),
            x=0.5,
            xanchor="center",
        ),
        scene=dict(
            aspectmode="data",
            bgcolor="#f3f4f6",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)
    return True

# 3Dmol viewer: CA-only structure with category coloring and optional enhanced styling/spin.
def render_3d_structure(
    sequence: str,
    width: int = 500,
    height: int = 400,
    iframe_height: int = 420,
    *,
    enhanced: bool = False,
    spin: bool = False,
) -> bool:
    # Render CA-only py3Dmol structure with category coloring and optional enhanced styling/spin.
    import streamlit.components.v1 as components

    # Input sanitization keeps renderer stable across pasted FASTA/text snippets.
    clean = "".join(c for c in (sequence or "").upper() if not c.isspace())
    if not clean:
        return False
    if len(clean) > MAX_3D_SEQUENCE_LENGTH:
        return False
    try:
        import py3Dmol  # type: ignore
    except Exception:
        return False

    try:
        coords = helix_coordinates(clean, smooth=enhanced)
        pdb_data = generate_helix_pdb(clean, smooth=enhanced)
        view = py3Dmol.view(width=width, height=height)
        view.addModel(pdb_data, "pdb")

        try:
            view.setBackgroundColor("#0f0f12" if enhanced else "#1e1e1e")
        except Exception:
            pass

        cyl_r = 0.34 if enhanced else 0.28
        # Backbone cylinders connect consecutive residue positions.
        for i in range(len(coords) - 1):
            p0 = coords[i]
            p1 = coords[i + 1]
            cyl: dict = {
                "start": {"x": float(p0[0]), "y": float(p0[1]), "z": float(p0[2])},
                "end": {"x": float(p1[0]), "y": float(p1[1]), "z": float(p1[2])},
                "radius": cyl_r,
                "color": "#7a7a82",
                "fromCap": 1,
                "toCap": 1,
            }
            try:
                view.addCylinder(cyl)
            except Exception:
                try:
                    view.addCylinder(
                        {
                            "start": {"x": float(p0[0]), "y": float(p0[1]), "z": float(p0[2])},
                            "end": {"x": float(p1[0]), "y": float(p1[1]), "z": float(p1[2])},
                            "radius": cyl_r,
                            "color": "#7a7a82",
                        }
                    )
                except Exception:
                    pass

        sphere_radius = 0.36 if enhanced else 0.32
        # Residue spheres are colored by chemistry class (blue/red/green/gray).
        for i, aa in enumerate(clean):
            color = get_residue_color(aa)
            sel = {"resi": i + 1}
            sphere_style = {"sphere": {"radius": sphere_radius, "color": color}}
            view.setStyle(sel, sphere_style)

        max_labels = 60 if enhanced else 40
        label_every = max(1, (len(clean) + max_labels - 1) // max_labels)
        fs = 10 if enhanced else 9
        # Add labels sparsely to keep the viewer readable on longer peptides.
        for i, aa in enumerate(clean):
            if i % label_every != 0:
                continue
            try:
                view.addLabel(
                    aa,
                    {
                        "position": {"resi": i + 1, "atom": "CA"},
                        "backgroundColor": "#1a1a1a",
                        "fontColor": "#ffffff",
                        "fontSize": fs,
                    },
                )
            except Exception:
                pass

        view.zoomTo()

        if spin:
            try:
                view.spin(True)
            except Exception:
                try:
                    sp = getattr(view, "spin", None)
                    if callable(sp):
                        sp()
                except Exception:
                    pass

        if hasattr(view, "_make_html"):
            html = view._make_html()
        else:
            html = view.write()
        components.html(html, height=iframe_height)
        return True
    except Exception:
        return False
