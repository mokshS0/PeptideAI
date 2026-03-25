# Shared UI: formatting, tables, analysis bullets, exports: used on several sidebar pages
import html as _html
from typing import Dict, List, Tuple, Optional

from utils.analyze import compute_properties

def predicted_confidence(row: Dict) -> Optional[float]:
    # Convert AMP probability into confidence of the predicted class.
    if not row:
        return None
    pred = row.get("Prediction")
    p_amp = row.get("Confidence")
    if p_amp is None:
        return None
    try:
        p_amp = float(p_amp)
    except (TypeError, ValueError):
        return None
    if pred == "AMP":
        return p_amp
    # Convert AMP probability into confidence for the predicted class.
    return 1.0 - p_amp


def format_conf_percent(conf_prob: float, digits: int = 1) -> str:
    return f"{round(conf_prob * 100, digits)}%"


def heuristic_reason_for_profile(charge: float, hydro_fraction: float) -> str:
    if charge > 2:
        return "High positive charge supports membrane disruption"
    if 0.3 <= hydro_fraction <= 0.6:
        return "Balanced hydrophobicity"
    return "Favorable predicted profile"


def choose_top_candidate(predictions: List[Dict]) -> Optional[Dict]:
    # Select best candidate row and attach a short profile-based reason.
    if not predictions:
        return None

    # Prefer AMP rows first, then fall back to highest-confidence overall row.
    amp_rows = [r for r in predictions if r.get("Prediction") == "AMP"]
    rows = amp_rows if amp_rows else predictions

    best_row = None
    best_conf = -1.0
    for r in rows:
        c = predicted_confidence(r)
        if c is None:
            continue
        if c > best_conf:
            best_conf = c
            best_row = r

    if best_row is None:
        return None

    seq = best_row.get("Sequence", "")
    if not seq:
        return None

    props = compute_properties(seq)
    charge = props.get("Net Charge (approx.)", 0)
    hydro = props.get("Hydrophobic Fraction", 0)

    return {
        "Sequence": seq,
        "Prediction": best_row.get("Prediction"),
        "predicted_confidence": best_conf,
        "Reason": heuristic_reason_for_profile(charge, hydro),
        "Charge": charge,
        "Hydrophobic Fraction": hydro,
    }


def mutation_heatmap_html(original: str, final: str) -> str:
    # Highlight per-position residue changes between original and final sequences.
    orig = original or ""
    fin = final or ""
    max_len = max(len(orig), len(fin))

    # Use monospace layout so per-position residue changes align visually.
    out: List[str] = [
        "<div style='font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, \"Liberation Mono\", monospace; white-space: pre-wrap;'>"
    ]
    for i in range(max_len):
        o = orig[i] if i < len(orig) else ""
        f = fin[i] if i < len(fin) else ""
        residue = f if f else o
        changed = (o != f)
        residue_escaped = _html.escape(residue)
        if changed and residue:
            out.append(f"<span style='color:#d62728; font-weight:700;'>{residue_escaped}</span>")
        else:
            out.append(residue_escaped if residue else "&nbsp;")
    out.append("</div>")
    return "".join(out)


def mutation_diff_table(original: str, final: str) -> List[Dict]:
    orig = original or ""
    fin = final or ""
    max_len = max(len(orig), len(fin))
    rows: List[Dict] = []
    for i in range(max_len):
        o = orig[i] if i < len(orig) else ""
        f = fin[i] if i < len(fin) else ""
        rows.append(
            {
                "Position": i + 1,
                "Original": o,
                "Final": f,
                "Changed": "Yes" if o != f else "No",
            }
        )
    return rows


def _ideal_distance_to_interval(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 0.0
    if value < low:
        return low - value
    return value - high


def optimization_summary(orig_seq: str, orig_conf: float, final_seq: str, final_conf: float) -> Dict:
    # Compute confidence and property deltas for the Optimize summary panel.
    orig_seq = orig_seq or ""
    final_seq = final_seq or ""

    # Property deltas drive the compact "what changed" summary panel.
    props_orig = compute_properties(orig_seq) if orig_seq else {}
    props_final = compute_properties(final_seq) if final_seq else {}

    charge_orig = props_orig.get("Net Charge (approx.)", props_orig.get("Net charge", 0))
    charge_final = props_final.get("Net Charge (approx.)", props_final.get("Net charge", 0))

    hydro_orig = props_orig.get("Hydrophobic Fraction", 0)
    hydro_final = props_final.get("Hydrophobic Fraction", 0)

    delta_conf_pct = (float(final_conf) - float(orig_conf)) * 100.0

    if charge_final > charge_orig:
        charge_change = "Increased"
    elif charge_final < charge_orig:
        charge_change = "Decreased"
    else:
        charge_change = "Same"

    ideal_low, ideal_high = 0.4, 0.5
    dist_orig = _ideal_distance_to_interval(float(hydro_orig), ideal_low, ideal_high)
    dist_final = _ideal_distance_to_interval(float(hydro_final), ideal_low, ideal_high)

    if dist_final < dist_orig:
        hydro_change = "Improved balance"
    elif dist_final > dist_orig:
        hydro_change = "Less optimal"
    else:
        hydro_change = "Same"

    return {
        "delta_conf_pct": delta_conf_pct,
        "charge_orig": charge_orig,
        "charge_final": charge_final,
        "charge_change": charge_change,
        "hydro_orig": hydro_orig,
        "hydro_final": hydro_final,
        "hydro_change": hydro_change,
    }


def sequence_length_warning(seq: str) -> Optional[str]:
    if not seq:
        return None
    n = len(seq)
    if n < 8:
        return "Too short for typical AMP"
    if n > 50:
        return "Unusually long sequence"
    return None


def sequence_health_label(conf_prob: float, charge: float, hydro_fraction: float) -> Tuple[str, str]:
    # Return a short quality label plus color for Analyze page status display.
    # Very high model confidence is treated as strong even outside ideal property ranges.
    if conf_prob >= 0.9:
        return "Strong AMP candidate", "#2ca02c"
    if conf_prob > 0.75 and charge >= 2 and 0.3 <= hydro_fraction <= 0.6:
        return "Strong AMP candidate", "#2ca02c"
    if conf_prob > 0.5:
        return "Moderate potential", "#ff9800"
    return "Unlikely AMP", "#d62728"


# Plain-language bullets for Analyze — rules of thumb, not a second model.
def build_analysis_insights(
    label: str,
    conf: float,
    comp: Dict[str, float],
    length: int,
    hydro: float,
    charge: float,
) -> List[str]:
    # Short, mechanism-oriented bullets for the Analyze page (heuristics, not lab truth).
    lines: List[str] = []
    p_amp = float(conf)
    conf_pct = round(p_amp * 100, 1)
    pred_conf = conf_pct if label == "AMP" else round((1 - p_amp) * 100, 1)

    if label == "AMP":
        if pred_conf >= 80:
            lines.append(
                f"Model: **AMP** with high confidence ({pred_conf}% on this prediction)—profile below explains typical mechanisms."
            )
        elif pred_conf >= 60:
            lines.append(
                f"Model: **AMP** with moderate confidence ({pred_conf}%); cross-check chemistry bullets before treating it as a strong hit."
            )
        else:
            lines.append(
                f"Model: **AMP** but low confidence ({pred_conf}%); the mechanistic notes below matter more than the label alone."
            )
    else:
        if pred_conf >= 80:
            lines.append(
                f"Model: **Non-AMP** with high confidence ({pred_conf}% on this prediction)—below are common reasons a sequence may not behave like a classic AMP."
            )
        elif pred_conf >= 60:
            lines.append(
                f"Model: **Non-AMP** with moderate confidence ({pred_conf}%); reasons below are typical but not exhaustive."
            )
        else:
            lines.append(
                f"Model: **Non-AMP** with low confidence ({pred_conf}%); treat the label as tentative and read the property-based notes."
            )

    polar_frac = sum(float(comp.get(aa, 0.0)) for aa in "STNQYC")
    basic_frac = sum(float(comp.get(aa, 0.0)) for aa in "KRH")

    explain_weak = (label == "Non-AMP") or (label == "AMP" and pred_conf < 65)

    if explain_weak:
        if charge <= 0:
            lines.append(
                "Weak or absent **positive net charge**: many AMPs rely on cationic residues to bind **anionic bacterial surfaces** (e.g. LPS, teichoic acids); near-neutral or negative peptides often lack that first electrostatic hook."
            )
        if hydro < 0.28:
            lines.append(
                "Low **hydrophobic** content: membrane insertion, pore formation, or lipid disruption is harder without a hydrophobic face or core to partition into the bilayer."
            )
        if hydro > 0.65:
            lines.append(
                "Very high **hydrophobic** content: risk of aggregation or poor **aqueous solubility** before the peptide can reach bacteria—delivery and effective concentration suffer."
            )
        if polar_frac < 0.12:
            lines.append(
                "Few **polar / H-bonding** residues (S, T, N, Q, Y, C): weaker interfacial interactions with lipids and water at the membrane—many AMP mechanisms benefit from polar positioning at the interface."
            )
        if basic_frac < 0.06 and charge < 2:
            lines.append(
                "Sparse **basic** residues (K, R, H): a hallmark of many AMPs is concentrated positive charge for initial **bacterial association**; this sequence is thin on that axis."
            )
        if length < 8:
            lines.append(
                "Very **short** length: may be too small to form a stable membrane-active structure or to span a bilayer meaningfully."
            )
        elif length > 50:
            lines.append(
                "Unusually **long** chain: folding, proteolysis, and synthesis cost can diverge from small cationic AMP archetypes."
            )

        if label == "Non-AMP" and charge >= 2 and 0.28 <= hydro <= 0.58:
            lines.append(
                "**Note:** Charge and hydrophobic balance still look somewhat AMP-like—the model says Non-AMP, so treat this as a **disagreement** worth validating experimentally, not proof either way."
            )

    if label == "AMP" and pred_conf >= 65:
        if charge >= 2 and 0.28 <= hydro <= 0.58:
            lines.append(
                "**Positive charge** plus **moderate hydrophobic fraction** aligns with membrane-targeting motifs common in AMP literature."
            )
        if polar_frac >= 0.12:
            lines.append(
                "Adequate **polar** residues can help **interfacial** placement and H-bonding at the membrane."
            )

    if (comp.get("K", 0) + comp.get("R", 0) + comp.get("H", 0)) >= 0.18:
        lines.append(
            "Higher **basic** residue fraction supports **electrostatic** attraction to anionic bacterial components."
        )
    if (comp.get("C", 0) + comp.get("W", 0)) >= 0.08:
        lines.append(
            "**Cysteine / tryptophan** can contribute to membrane insertion, stacking, or oxidative chemistry depending on context."
        )

    # De-duplicate while preserving order.
    out: List[str] = []
    seen = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out[:14]


def build_analysis_summary_text(
    sequence: str,
    prediction: str,
    confidence_display: str,
    props: Dict,
    analysis_lines: List[str],
) -> str:
    length = props.get("Length", len(sequence))
    charge = props.get("Net Charge (approx.)", props.get("Net charge", 0))
    hydro = props.get("Hydrophobic Fraction", props.get("Hydrophobic", 0))
    analysis_block = "\n".join(f"- {line}" for line in (analysis_lines or []))
    return (
        f"Sequence: {sequence}\n"
        f"Prediction: {prediction}\n"
        f"Confidence: {confidence_display}\n"
        f"Length: {length}\n"
        f"Net Charge (approx.): {charge}\n"
        f"Hydrophobic Fraction: {hydro}\n\n"
        f"Summary:\n{analysis_block}\n"
    )

