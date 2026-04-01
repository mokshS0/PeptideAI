# Main Streamlit entrypoint: one file, several “pages” chosen from the sidebar.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import plotly.express as px
import html as _html
from sklearn.manifold import TSNE

# Utils map to sidebar pages: predict / analyze / optimize / visualize / tsne, plus shared_ui.
from utils.predict import load_model, predict_amp, encode_sequence, get_embedding_extractor
from utils.analyze import aa_composition, compute_properties
from utils.optimize import optimize_sequence
from utils.shared_ui import (
    choose_top_candidate,
    format_conf_percent,
    mutation_heatmap_html,
    mutation_diff_table,
    optimization_summary,
    sequence_length_warning,
    sequence_health_label,
    build_analysis_insights,
    build_analysis_summary_text,
)
from utils.rate_limit import RateLimiter
from utils.visualize import (
    KNOWN_AMPS,
    MAX_3D_SEQUENCE_LENGTH,
    COMPACT_3D_LEGEND,
    COMPACT_MAP_LEGEND,
    COMPACT_WHEEL_LEGEND,
    build_shape_visual_summary,
    find_most_similar,
    build_importance_map_html,
    plot_helical_wheel,
    render_3d_plotly,
    render_3d_structure,
)

try:
    import pyperclip 
except Exception:
    pyperclip = None


def _tooltip_label(label: str, tooltip_text: str) -> None:
    # Render a label with a lightweight HTML hover tooltip.
    safe = _html.escape(tooltip_text, quote=True)
    st.markdown(f"{label} <span title='{safe}' style='cursor:help;color:#666'>(i)</span>", unsafe_allow_html=True)


def _session_rate_limiter(state_key: str, max_calls: int, period_seconds: float) -> RateLimiter:
    # One limiter object per browser session (Streamlit reruns keep the same session_state).
    if state_key not in st.session_state:
        st.session_state[state_key] = RateLimiter(max_calls, period_seconds)
    return st.session_state[state_key]


def _rate_limit_ok(state_key: str, max_calls: int, period_seconds: float, action_label: str) -> bool:
    rl = _session_rate_limiter(state_key, max_calls, period_seconds)
    if rl.allow():
        return True
    wait = max(1.0, rl.time_until_next())
    st.warning(
        f"Rate limit: please wait **~{int(wait)}s** before another {action_label}. "
        "(Light throttle on shared hosting.)"
    )
    return False


def _try_copy_to_clipboard(text: str) -> None:
    # Best-effort server-side clipboard copy (browser copy is intentionally avoided).
    if pyperclip is not None:
        try:
            pyperclip.copy(text)
        except Exception:
            pass


# Widget keys are cleared when a page is not rendered; these copy text into plain session keys.
def _sync_predict_input_saved():
    st.session_state.predict_input_saved = st.session_state.get("predict_input_widget", "")


def _sync_analyze_draft():
    st.session_state.analyze_draft = st.session_state.get("analyze_input_widget", "")


def _sync_optimize_input():
    st.session_state.optimize_input = st.session_state.get("optimize_input_widget", "")


def _sync_visualize_peptide_input():
    st.session_state.visualize_peptide_input = st.session_state.get("visualize_peptide_input_widget", "")


# Configure global app layout once before rendering widgets.
st.set_page_config(page_title="PeptideAI", layout="wide")

# Global title shown above all pages.
st.title("PeptideAI")
st.write("Antimicrobial Peptide Predictor and Optimizer")
st.divider()

# Initialize session keys so navigation keeps user state across pages.
if "predictions" not in st.session_state:
    st.session_state.predictions = []             # list of dicts
if "predict_ran" not in st.session_state:
    st.session_state.predict_ran = False
# predict_input_saved: survives navigation when Streamlit strips widget keys.
if "predict_input_saved" not in st.session_state:
    st.session_state.predict_input_saved = ""
if "analyze_input" not in st.session_state:
    st.session_state.analyze_input = ""           # last analyze input
if "analyze_draft" not in st.session_state:
    st.session_state.analyze_draft = ""            # typed analyze sequence (persists across pages)
if "analyze_output" not in st.session_state:
    st.session_state.analyze_output = None        # (label, conf_display, comp, props, analysis)
if "optimize_input" not in st.session_state:
    st.session_state.optimize_input = ""          # last optimize sequence (persisted draft)
if "optimize_output" not in st.session_state:
    st.session_state.optimize_output = None       # (orig_seq, orig_conf, improved_seq, improved_conf, history)
if "optimize_last_ran_input" not in st.session_state:
    st.session_state.optimize_last_ran_input = ""
if "visualize_sequences" not in st.session_state:
    st.session_state.visualize_sequences = None
if "visualize_df" not in st.session_state:
    st.session_state.visualize_df = None
if "visualize_peptide_input" not in st.session_state:
    st.session_state.visualize_peptide_input = ""

# Sidebar route selector drives top-level page rendering.
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Predict",
        "Analyze",
        "Optimize",
        "Visualize",
        "t-SNE",
        "About",
    ],
)
st.sidebar.caption("Light per-session rate limits apply on expensive model runs.")

if st.sidebar.button("Clear All Fields"):
    # Reset only app-owned state keys, then rerun to refresh all widgets.
    keys = [
        "predictions",
        "predict_ran",
        "predict_input_widget",
        "predict_input_saved",
        "analyze_input",
        "analyze_draft",
        "analyze_input_widget",
        "analyze_output",
        "optimize_input",
        "optimize_input_widget",
        "optimize_output",
        "optimize_last_ran_input",
        "visualize_sequences",
        "visualize_df",
        "visualize_peptide_input",
        "visualize_peptide_input_widget",
    ]
    for rk in ("_rl_predict", "_rl_analyze", "_rl_optimize", "_rl_tsne"):
        keys.append(rk)
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    st.sidebar.success("Cleared app state.")
    # Support both old and new Streamlit rerun APIs.
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn is not None:
        rerun_fn()
    else:
        st.stop()


# Load weights once; every page shares this same model instance.
model = load_model()

# Shared style tweak keeps expander spacing consistent across pages.
st.markdown(
    """<style>
    div[data-testid="stExpander"] details > summary {
        padding-top: 0.3rem !important;
        padding-bottom: 0.3rem !important;
        min-height: 2rem !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

# Predict page: batch inference from text area and optional upload.
if page == "Predict":
    st.header("AMP Predictor")

    preset_cols = st.columns(2)
    with preset_cols[0]:
        if st.button("Use strong AMP example"):
            ex = "RGGRLCYCRGWICFCVGR"
            st.session_state.predict_input_widget = ex
            st.session_state.predict_input_saved = ex
            st.rerun()
    with preset_cols[1]:
        if st.button("Use weak sequence example"):
            ex = "KAEEEVEKNKEEAEEKAEKKIAE"
            st.session_state.predict_input_widget = ex
            st.session_state.predict_input_saved = ex
            st.rerun()

    # Restore textarea after navigating away (widget key may have been dropped).
    if "predict_input_widget" not in st.session_state:
        st.session_state.predict_input_widget = st.session_state.predict_input_saved

    seq_input = st.text_area(
        "Enter peptide sequences (one per line):",
        height=150,
        key="predict_input_widget",
        on_change=_sync_predict_input_saved,
    )
    _sync_predict_input_saved()
    uploaded_file = st.file_uploader("Or upload a FASTA/text file", type=["txt", "fasta"])

    # Show quick length guidance before running the model.
    preview_sequences = [s.strip() for s in (seq_input or "").splitlines() if s.strip()]
    if preview_sequences:
        short_cnt = sum(1 for s in preview_sequences if len(s) < 8)
        long_cnt = sum(1 for s in preview_sequences if len(s) > 50)
        if short_cnt:
            st.caption(f"Warning: {short_cnt} sequence(s) too short for typical AMP (< 8 aa).")
        if long_cnt:
            st.caption(f"Warning: {long_cnt} sequence(s) unusually long (> 50 aa).")

    run = st.button("Run Prediction")

    if run:

        # Merge direct text input and uploaded FASTA/plain-text entries.
        sequences = []
        if seq_input:
            sequences += [s.strip() for s in seq_input.splitlines() if s.strip()]
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")
            sequences += [l.strip() for l in text.splitlines() if not l.startswith(">") and l.strip()]

        if not sequences:
            st.warning("Please input or upload sequences first.")
        elif not _rate_limit_ok("_rl_predict", 40, 60.0, "batch prediction"):
            pass
        else:
            progress = st.progress(0.0)
            with st.spinner("Running prediction..."):
                results = []
                # Predict each sequence one-by-one so progress updates are accurate.
                for i, seq in enumerate(sequences):
                    label, conf = predict_amp(seq, model)
                    conf_display = round(conf * 100, 1) if label == "AMP" else round((1 - conf) * 100, 1)
                    results.append({
                        "Sequence": seq,
                        "Prediction": label,
                        "Confidence": conf,
                        "Description": f"{label} with {conf_display}% confidence"
                    })
                    progress.progress((i + 1) / max(1, len(sequences)), text=f"Predicted {i + 1}/{len(sequences)}")
            progress.progress(1.0)

            # Persist results so users can switch pages without losing output.
            st.session_state.predictions = results
            st.session_state.predict_ran = True
            st.success("Prediction complete.")

    # Always show latest saved prediction set for continuity across navigation.
    if st.session_state.predictions and not (run and st.session_state.predict_ran is False):
        st.divider()

        top_candidate = choose_top_candidate(st.session_state.predictions)
        if top_candidate:
            with st.container():

                st.write("**Top AMP Predicted Candidate**")
                seq = top_candidate.get("Sequence", "")
                cc = st.columns([9, 1])
                with cc[0]:
                    st.code(seq, language="text")
                with cc[1]:
                    if st.button("Copy", key="copy_top_candidate"):
                        _try_copy_to_clipboard(seq)
                        toast_fn = getattr(st, "toast", None)
                        if toast_fn is not None:
                            toast_fn("Copied, or select the sequence above (Ctrl+C)")
                        else:
                            st.success("Copied, or select the sequence above (Ctrl+C)")
                label = top_candidate.get("Prediction", "")
                conf_str = format_conf_percent(top_candidate["predicted_confidence"], digits=1)
                st.write(f"**{label} with {conf_str} confidence**")
                st.write(f"Reason: {top_candidate['Reason']}")

        st.divider()
        # Full table + CSV export preserve the complete prediction batch.
        st.dataframe(pd.DataFrame(st.session_state.predictions), use_container_width=True)
        csv = pd.DataFrame(st.session_state.predictions).to_csv(index=False)
        st.download_button("Download predictions as CSV", csv, "predictions.csv", "text/csv")

# Analyze page: single-sequence diagnostics and report export.
elif page == "Analyze":
    st.header("Peptide Analyzer")

    # Match optimizer-like boxed input style for consistent UI spacing.
    with st.container(border=True):
        if "analyze_input_widget" not in st.session_state:
            init = st.session_state.analyze_draft or st.session_state.analyze_input
            st.session_state.analyze_input_widget = init
        st.text_input(
            "Enter a peptide sequence to analyze:",
            key="analyze_input_widget",
            on_change=_sync_analyze_draft,
        )
        _sync_analyze_draft()
    seq = st.session_state.analyze_draft

    warn = sequence_length_warning(seq)
    if warn:
        st.caption(f"Warning: {warn}")

    # Recompute only when sequence changes to avoid redundant work on reruns.
    if seq and seq != st.session_state.get("analyze_input", ""):
        if not _rate_limit_ok("_rl_analyze", 35, 60.0, "analysis run"):
            pass
        else:
            with st.spinner("Running analysis..."):
                label, conf = predict_amp(seq, model)
                conf_pct = round(conf * 100, 1)
                conf_display = conf_pct if label == "AMP" else 100 - conf_pct

                comp = aa_composition(seq)
                props = compute_properties(seq)

                # Normalize property key variants returned by helper functions.
                net_charge = props.get("Net Charge (approx.)",
                                       props.get("Net charge", props.get("NetCharge", 0)))

                length = props.get("Length", len(seq))
                hydro = props.get("Hydrophobic Fraction", props.get("Hydrophobic", 0))
                charge = net_charge
                mw = props.get("Molecular Weight (Da)", props.get("MolecularWeight", 0))

                analysis = build_analysis_insights(label, conf, comp, length, float(hydro), float(charge))

                # Save computed payload for display + report exports below.
                st.session_state.analyze_input = seq
                st.session_state.analyze_draft = seq
                st.session_state.analyze_output = (label, conf, conf_display, comp, props, analysis)

    # Render last computed analysis block.
    if st.session_state.analyze_output:
        label, conf, conf_display, comp, props, analysis = st.session_state.analyze_output

        st.subheader("AMP Prediction")
        display_conf = round(conf * 100, 1) if label == "AMP" else round((1 - conf) * 100, 1)
        st.write(f"Prediction: **{label}** with **{display_conf}%** confidence")

        # Health badge blends model confidence with simple chemistry heuristics.
        hydro = props.get("Hydrophobic Fraction", 0)
        charge = props.get("Net Charge (approx.)", props.get("Net charge", 0))
        health_label, color = sequence_health_label(float(conf), float(charge), float(hydro))
        st.markdown(
            f"<span style='color:{color}; font-weight:800;'>{health_label}</span>",
            unsafe_allow_html=True,
        )

        st.subheader("Amino Acid Composition")
        comp_df = pd.DataFrame(list(comp.items()), columns=["Amino Acid", "Frequency"]).set_index("Amino Acid")
        st.bar_chart(comp_df)

        st.subheader("Physicochemical Properties and Favorability")

        # Pull fields defensively in case key names vary.
        length = props.get("Length", len(st.session_state.analyze_input))
        hydro = props.get("Hydrophobic Fraction", 0)
        charge = props.get("Net Charge (approx.)", props.get("Net charge", 0))
        mw = props.get("Molecular Weight (Da)", 0)

        favorability = {
            "Length": "Good" if 10 <= length <= 50 else "Too short" if length < 10 else "Too long",
            "Hydrophobic Fraction": "Good" if 0.4 <= hydro <= 0.6 else "Low" if hydro < 0.4 else "High",
            "Net Charge": "Favorable" if charge > 0 else "Neutral" if charge == 0 else "Unfavorable",
            "Molecular Weight": "Acceptable" if 500 <= mw <= 5000 else "Extreme"
        }
        def _info_icon(tooltip_text: str) -> str:
            safe = _html.escape(tooltip_text, quote=False)
            return (
                "<span "
                "class='amp-i' "
                f"data-tooltip='{safe}' "
                "style=\"display:inline-flex; align-items:center; justify-content:center; "
                "margin-left:6px; width:16px; height:16px; border-radius:50%; "
                "background:#f2f2f2; border:1px solid #d9d9d9; color:#333; "
                "font-size:12px; font-weight:700; cursor:help;\">(i)</span>"
            )

        # Use HTML table for custom inline "(i)" tooltips.
        hydro_label = f"Hydrophobic Fraction{_info_icon('Fraction of residues that prefer non-aqueous environments')}"
        charge_label = f"Net Charge{_info_icon('Positive charge helps peptides bind bacterial membranes')}"
        table_html = (
            "<style>"
            ".amp-i{position:relative; display:inline-flex;}"
            ".amp-i::after{"
            "content:attr(data-tooltip);"
            "position:absolute;"
            "left:50%;"
            "top:125%;"
            "transform:translateX(-50%);"
            "max-width:1080px;"
            "white-space:normal;"
            "padding:8px 10px;"
            "background:rgba(30,30,30,0.95);"
            "color:#fff;"
            "border-radius:8px;"
            "font-size:12px;"
            "line-height:1.25;"
            "box-shadow:0 8px 30px rgba(0,0,0,0.25);"
            "opacity:0;"
            "pointer-events:none;"
            "z-index:9999;"
            "}"
            ".amp-i:hover::after{opacity:1;}"
            "</style>"
            "<table style='width:100%; border-collapse:collapse;'>"
            "<thead>"
            "<tr>"
            "<th style='text-align:left; padding:8px; border-bottom:1px solid #e6e6e6;'>Property</th>"
            "<th style='text-align:right; padding:8px; border-bottom:1px solid #e6e6e6;'>Value</th>"
            "<th style='text-align:left; padding:8px; border-bottom:1px solid #e6e6e6;'>Favorability</th>"
            "</tr>"
            "</thead>"
            "<tbody>"
            f"<tr><td style='padding:8px;'>{_html.escape('Length')}{_info_icon('Peptides with ~10-50 aa often balance membrane insertion and solubility.')}</td><td style='padding:8px; text-align:right;'>{_html.escape(str(length))}</td><td style='padding:8px;'>{_html.escape(favorability['Length'])}</td></tr>"
            f"<tr><td style='padding:8px;'>{hydro_label}</td><td style='padding:8px; text-align:right;'>{_html.escape(str(hydro))}</td><td style='padding:8px;'>{_html.escape(favorability['Hydrophobic Fraction'])}</td></tr>"
            f"<tr><td style='padding:8px;'>{charge_label}</td><td style='padding:8px; text-align:right;'>{_html.escape(str(charge))}</td><td style='padding:8px;'>{_html.escape(favorability['Net Charge'])}</td></tr>"
            f"<tr><td style='padding:8px;'>{_html.escape('Molecular Weight')}{_info_icon('Moderate molecular weight can help stability and binding; extremes may hurt performance.')}</td><td style='padding:8px; text-align:right;'>{_html.escape(str(mw))}</td><td style='padding:8px;'>{_html.escape(favorability['Molecular Weight'])}</td></tr>"
            "</tbody>"
            "</table>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

        st.divider()
        st.subheader("Property Radar Chart")
        categories = ["Length", "Hydrophobic Fraction", "Net Charge", "Molecular Weight"]
        values = [min(length / 50, 1), min(hydro, 1), 1 if charge > 0 else 0, min(mw / 5000, 1)]
        values += values[:1]
        ideal_min = [10/50, 0.4, 1/6, 500/5000] + [10/50]
        ideal_max = [50/50, 0.6, 6/6, 5000/5000] + [50/50]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        # Compact radar chart compares sequence values against an "ideal AMP" band.
        fig, ax = plt.subplots(figsize=(2.8, 3.2), subplot_kw=dict(polar=True)) 
        fig.patch.set_facecolor("white")
        ax.fill_between(angles, ideal_min, ideal_max, color='#457a00', alpha=0.15, label="Ideal AMP range")
        ax.plot(angles, values, 'o-', color='#457a00', linewidth=2, label="Sequence")
        ax.fill(angles, values, color='#457a00', alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=8)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='y', labelsize=7)
        ax.legend(loc='lower center', bbox_to_anchor=(0.85, 1.15), ncol=2, fontsize=7)
        st.pyplot(fig, use_container_width=False)

        st.divider()
        st.subheader("Most similar known AMP")
        st.caption(
            f"Compared to **{len(KNOWN_AMPS)}** unique AMP sequences (label = 1 in `Data/ampData.csv`)."
        )
        seq_sim = str(st.session_state.analyze_input or "").strip()
        seq_clean_sim = "".join(c for c in seq_sim.upper() if not c.isspace())
        if seq_clean_sim:
            match_seq, sim_score = find_most_similar(seq_clean_sim)
            if match_seq is not None:
                st.write(f"**Best match:** `{match_seq}`")
                st.write(f"**Similarity score:** **{sim_score:.3f}** (position match / max length)")
                if sim_score > 0.6:
                    st.success("High similarity to a known AMP in the reference set.")
                elif sim_score > 0.3:
                    st.warning("Moderate similarity, interpret with care.")
                else:
                    st.error("Low similarity, sequence is distant from reference AMPs.")
            else:
                st.warning("Could not compute similarity.")
        else:
            st.caption("Run analysis with a sequence to compare against known AMPs.")

        st.divider()
        # Summarize key findings as plain-language bullets.
        st.subheader("Analysis Summary")
        for line in analysis:
            st.write(f"- {line}")

        # Export section packages current analysis in CSV or TXT format.
        st.divider()
        st.subheader("Export Analysis Report")
        export_format = st.radio("Format", ["CSV", "TXT"], horizontal=True)

        confidence_display_str = f"{round(conf_display, 1)}%"
        summary_text = build_analysis_summary_text(
            sequence=st.session_state.analyze_input,
            prediction=label,
            confidence_display=confidence_display_str,
            props=props,
            analysis_lines=analysis,
        )

        csv_df = pd.DataFrame(
            [
                {
                    "Sequence": st.session_state.analyze_input,
                    "Prediction": label,
                    "Confidence": confidence_display_str,
                    "Length": props.get("Length", len(st.session_state.analyze_input)),
                    "Charge": charge,
                    "Hydrophobic fraction": hydro,
                    "Summary": "\n".join(analysis or []),
                }
            ]
        )

        if export_format == "CSV":
            csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV report",
                csv_bytes,
                file_name="analysis_report.csv",
                mime="text/csv",
            )
        else:
            st.download_button(
                "Download TXT report",
                summary_text.encode("utf-8"),
                file_name="analysis_report.txt",
                mime="text/plain",
            )

# Optimize page: Mutation search with per-step diagnostics.
elif page == "Optimize":
    st.header("Peptide Optimizer")

    with st.container(border=True):
        if "optimize_input_widget" not in st.session_state:
            st.session_state.optimize_input_widget = st.session_state.optimize_input
        st.text_input(
            "Enter a peptide sequence to optimize:",
            key="optimize_input_widget",
            on_change=_sync_optimize_input,
        )
        _sync_optimize_input()
    seq = st.session_state.optimize_input

    warn_opt = sequence_length_warning(seq) if seq else None
    if warn_opt:
        st.caption(f"Warning: {warn_opt}")

    # Re-run optimization when the entered sequence changes.
    if seq and str(seq).strip() and str(seq).strip() != st.session_state.get("optimize_last_ran_input", ""):
        seq = str(seq).strip()
        if not _rate_limit_ok("_rl_optimize", 12, 60.0, "optimization run"):
            pass
        else:
            st.session_state.optimize_last_ran_input = seq
            progress = st.progress(0.0, text="Optimizing...")
            with st.spinner("Optimizing sequence..."):
                improved_seq, improved_conf, history = optimize_sequence(seq, model)
                _ol, orig_conf = predict_amp(seq, model)
                st.session_state.optimize_output = (seq, orig_conf, improved_seq, improved_conf, history)
            progress.progress(1.0, text="Optimization complete")
            st.success("Optimization finished.")

    # Render latest optimization artifacts from session state.
    if st.session_state.optimize_output:
        orig_seq, orig_conf, improved_seq, improved_conf, history = st.session_state.optimize_output
        summary = optimization_summary(orig_seq, orig_conf, improved_seq, improved_conf)
        delta_str = f"{summary['delta_conf_pct']:+.2f}%"

        col_results, col_opt_summary = st.columns(2)
        with col_results:
            st.subheader("Results")
            st.write(f"**Original Sequence:** {orig_seq}, Confidence: {round(orig_conf*100,1)}%")
            st.write(
                f"**Optimized Sequence:** {improved_seq}, Confidence: {round(improved_conf*100,1)}%"
            )
        with col_opt_summary:
            st.subheader("Optimization Summary")
            st.write(f"Confidence: **{delta_str}** (final - original)")
            st.write(
                f"Charge: **{summary['charge_change']}** (orig {summary['charge_orig']}, final {summary['charge_final']})"
            )
            st.write(
                f"Hydrophobicity: **{summary['hydro_change']}** (orig {summary['hydro_orig']}, final {summary['hydro_final']})"
            )

        st.divider()
        # Heatmap + table make residue-level edits easy to inspect.
        st.subheader("Mutation Heatmap (Changed Residues Highlighted)")
        st.markdown(mutation_heatmap_html(orig_seq, improved_seq), unsafe_allow_html=True)
        with st.expander("Mutation Details (table)"):
            diff_rows = mutation_diff_table(orig_seq, improved_seq)
            st.dataframe(pd.DataFrame(diff_rows), use_container_width=True)

        if len(history) > 1:
            df_steps = pd.DataFrame([{
                "Step": i,
                "Change": change,
                "Old Type": old_type,
                "New Type": new_type,
                "Reason for Improvement": reason,
                "New Confidence (%)": round(conf * 100, 2)
            } for i, (seq_after, conf, change, old_type, new_type, reason) in enumerate(history[1:], start=1)])
            st.subheader("Mutation Steps")
            st.dataframe(df_steps, use_container_width=True)

            # Step 0 = original peptide AMP probability; steps 1+ match the table after each mutation.
            step_nums = [0] + df_steps["Step"].tolist()
            conf_values = [round(float(orig_conf) * 100, 2)] + df_steps["New Confidence (%)"].tolist()
            df_graph = pd.DataFrame({"Step": step_nums, "Confidence (%)": conf_values})
            fig = px.line(df_graph, x="Step", y="Confidence (%)", markers=True, color_discrete_sequence=["#457a00"])
            fig.update_layout(
                yaxis=dict(range=[0, 100]),
                title="AMP model confidence (%) — step 0 = original, then each accepted change",
            )
            st.plotly_chart(fig, use_container_width=True)

# Visualize page: structural/sequence interpretation for one peptide.
elif page == "Visualize":
    st.header("Peptide Visualizer")
    with st.container(border=True):
        if "visualize_peptide_input_widget" not in st.session_state:
            st.session_state.visualize_peptide_input_widget = st.session_state.visualize_peptide_input
        st.text_input(
            "Enter a peptide sequence to visualize:",
            key="visualize_peptide_input_widget",
            on_change=_sync_visualize_peptide_input,
        )
        _sync_visualize_peptide_input()
    seq_viz = (st.session_state.get("visualize_peptide_input") or "").strip()
    clean_viz = "".join(c for c in seq_viz.upper() if not c.isspace())
    if clean_viz:
        with st.spinner("Building 3D view and helical wheel..."):
            warn_len = sequence_length_warning(clean_viz)
            if warn_len:
                st.warning(warn_len)
            if len(clean_viz) > MAX_3D_SEQUENCE_LENGTH:
                st.warning(
                    f"Sequence longer than **{MAX_3D_SEQUENCE_LENGTH}** aa: **3D model is disabled**; "
                    "helical wheel and functional map still render."
                )

            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("3D structural approximation")
                st.caption(
                    "**Plotly** = backbone line + colored residues; **3Dmol** = cylinder backbone + spheres. "
                    "Same helix geometry as the wheel (approximation only)."
                )
                if len(clean_viz) <= MAX_3D_SEQUENCE_LENGTH:
                    # Render the same geometry two ways (interactive Plotly vs py3Dmol).
                    tab_pl, tab_mol = st.tabs(["Plotly 3D", "3Dmol viewer"])
                    with tab_pl:
                        if not render_3d_plotly(clean_viz):
                            st.warning("Plotly 3D could not be rendered.")
                    with tab_mol:
                        if not render_3d_structure(clean_viz, enhanced=True, spin=False):
                            st.info("3Dmol view unavailable (install **py3dmol** in your environment).")
                    with st.expander("3D · legend", expanded=False):
                        st.markdown(COMPACT_3D_LEGEND)
                else:
                    st.info("3D visualization is limited to **60 residues** for performance.")

            with col_r:
                st.subheader("Helical wheel")
                st.caption(
                    "Radial spokes per residue, black connectors along the sequence, colored disks (same scheme as 3D)."
                )
                # The wheel uses the same 100-degree/step geometry as the 3D view.
                fig_wheel = plot_helical_wheel(clean_viz)
                st.pyplot(fig_wheel, use_container_width=True)
                plt.close(fig_wheel)
                with st.expander("Wheel · legend", expanded=False):
                    st.markdown(COMPACT_WHEEL_LEGEND)

            st.divider()
            st.subheader("Functional region map")
            st.caption("Residue-level chemistry; colors align with the 3D view and wheel.")
            # Inline map shows residue classes (blue/red/green/gray) letter-by-letter.
            st.markdown(build_importance_map_html(clean_viz), unsafe_allow_html=True)
            with st.expander("Map · legend", expanded=False):
                st.markdown(COMPACT_MAP_LEGEND)

            st.divider()
            st.subheader("How this visualization helps (shape & AMP context)")
            st.caption(
                "Heuristic readout from the **helix wheel geometry** and residue classes. Use it with the classifier, not instead of experiments."
            )
            v_label, v_conf = predict_amp(clean_viz, model)
            for line in build_shape_visual_summary(clean_viz, amp_label=v_label, amp_prob=v_conf):
                st.markdown(f"- {line}")

# t-SNE page: embedding projection for multi-sequence exploration.
# --- t-SNE on first-layer activations ---
elif page == "t-SNE":
    st.header("t-SNE Visualizer")
    st.write("Upload peptide sequences (FASTA or plain list) to embed sequences and explore clusters with t-SNE.")

    uploaded_file = st.file_uploader("Upload FASTA or text file", type=["txt", "fasta"])

    # Parse upload and replace previous sequence set.
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        sequences = [l.strip() for l in text.splitlines() if not l.startswith(">") and l.strip()]
        st.session_state.visualize_sequences = sequences

        # Invalidate previous embedding projection after new upload.
        st.session_state.visualize_df = None

    # Compute embeddings once and cache the projected dataframe in session.
    if st.session_state.visualize_sequences and st.session_state.visualize_df is None:
        sequences = st.session_state.visualize_sequences
        if len(sequences) < 2:
            st.warning("Need at least 2 sequences for t-SNE visualization.")
        elif not _rate_limit_ok("_rl_tsne", 10, 120.0, "t-SNE embedding run"):
            pass
        else:
            progress = st.progress(0.0, text="Generating embedding...")
            with st.spinner("Generating embedding..."):
                embeddings_list, labels, confs, lengths, hydros, charges = [], [], [], [], [], []

                # Use penultimate model representation as embedding features.
                embedding_extractor = get_embedding_extractor(model)

                    # Build embeddings, then predict label/conf for each sequence (for hover + coloring).
                for i, s in enumerate(sequences):
                    x = torch.tensor(encode_sequence(s, model), dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        emb = embedding_extractor(x).squeeze().numpy()
                    embeddings_list.append(emb)
                    label, conf = predict_amp(s, model)
                    labels.append(label)
                    confs.append(conf)
                    props = compute_properties(s)
                    lengths.append(props.get("Length", len(s)))
                    hydros.append(props.get("Hydrophobic Fraction", 0))
                    charges.append(props.get("Net Charge (approx.)", props.get("Net charge", 0)))
                    progress.progress((i + 1) / max(1, len(sequences)), text=f"Encoding {i + 1}/{len(sequences)}")

                embeddings_array = np.stack(embeddings_list)
                perplexity = min(30, max(2, len(sequences) - 1))
                # TSNE turns the high-dimensional embedding into a 2D map for exploration.
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                reduced = tsne.fit_transform(embeddings_array)

                df = pd.DataFrame(reduced, columns=["x", "y"])
                df["Sequence"] = sequences
                df["Label"] = labels
                df["Confidence"] = confs
                df["Length"] = lengths
                df["Hydrophobic Fraction"] = hydros
                df["Net Charge"] = charges

                st.session_state.visualize_df = df
                progress.progress(1.0, text="Embedding ready")

    # Render interactive scatter + filters once a projected dataframe exists.
    if st.session_state.visualize_df is not None:
        df = st.session_state.visualize_df
        st.subheader("t-SNE plot")

        st.sidebar.subheader("Filter Sequences")
        min_len, max_len = int(df["Length"].min()), int(df["Length"].max())
        if min_len == max_len:
            st.sidebar.write(f"All sequences have length {min_len}")
            length_range = (min_len, max_len)
        else:
            length_range = st.sidebar.slider("Sequence length", min_len, max_len, (min_len, max_len))

        label_options = st.sidebar.multiselect("Label", ["AMP", "Non-AMP"], default=["AMP", "Non-AMP"])
        filtered_df = df[(df["Length"].between(length_range[0], length_range[1])) & (df["Label"].isin(label_options))]
        color_by = st.sidebar.selectbox("Color points by", ["Label", "Confidence", "Hydrophobic Fraction", "Net Charge", "Length"])

        color_map = {"AMP": "#2ca02c", "Non-AMP": "#d62728"}
        fig = px.scatter(
            filtered_df,
            x="x", y="y",
            color=color_by if color_by != "Label" else "Label",
            color_discrete_map=color_map if color_by == "Label" else None,
            hover_data={"Sequence": True, "Label": True, "Confidence": True, "Length": True, "Hydrophobic Fraction": True, "Net Charge": True},
            title="t-SNE Visualization of Model Embeddings"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("t-SNE Analysis")
        st.markdown("""
• Each point represents a peptide sequence.  
• Sequences close together have similar internal representations in the model.  
• AMP and Non-AMP clusters indicate strong model separation.  
• Coloring by properties reveals biochemical trends.
""")

# About Page
elif page == "About":
    st.header("About the Project")
    st.markdown("""
PeptideAI is a lightweight Streamlit app for exploring antimicrobial peptide (AMP) sequences.

It uses a trained neural network to estimate whether a peptide is likely to be antimicrobial, then helps you interpret and improve candidates:
- **AMP Predictor**: batch predictions from multi-line or FASTA input, length warnings, persisted results, top-candidate highlight, and CSV export.
- **Peptide Analyzer**: single-sequence numerical and textual analysis, AMP prediction, composition, physicochemical table + radar, similarity to known AMPs, and report export.
- **Peptide Optimizer**: guided sequence optimization with Enter-to-run input, mutation heatmap, step table, and confidence-vs-step trend.
- **Visualize**: Plotly 3D backbone + optional 3Dmol view, helical wheel, functional map, and shape-focused AMP context summary.
- **t-SNE**: upload many sequences, embed with the model, run t-SNE, and explore clusters with filters and hover metadata.
- **About**: this overview and disclaimer.

**Disclaimer:** Predictions are model-based heuristics and are **not** a substitute for wet-lab validation or regulatory use.
""")
