import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import plotly.express as px
from sklearn.manifold import TSNE

# modular imports
from utils.predict import load_model, predict_amp, encode_sequence
from utils.analyze import aa_composition, compute_properties
from utils.optimize import optimize_sequence

# APP CONFIG
st.set_page_config(page_title="AMP Predictor", layout="wide")

# App title 
st.title("PeptideAI: Antimicrobial Peptide Predictor and Optimizer")
st.write("Use the sidebar to navigate between prediction, analysis, optimization, and visualization tools.")
st.markdown("---")  

# SESSION STATE KEYS (one-time init)
if "predictions" not in st.session_state:
    st.session_state.predictions = []               # list of dicts
if "predict_ran" not in st.session_state:
    st.session_state.predict_ran = False
if "analyze_input" not in st.session_state:
    st.session_state.analyze_input = ""             # last analyze input
if "analyze_output" not in st.session_state:
    st.session_state.analyze_output = None         # (label, conf_display, comp, props, analysis)
if "optimize_input" not in st.session_state:
    st.session_state.optimize_input = ""           # last optimize input
if "optimize_output" not in st.session_state:
    st.session_state.optimize_output = None       # (orig_seq, orig_conf, improved_seq, improved_conf, history)
if "visualize_sequences" not in st.session_state:
    st.session_state.visualize_sequences = None
if "visualize_df" not in st.session_state:
    st.session_state.visualize_df = None

# SIDEBAR: navigation + global clear 
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Predict", "Analyze", "Optimize", "Visualize", "About"])

if st.sidebar.button("Clear All Fields"):

    # clear only our known keys
    keys = ["predictions", "predict_ran",
            "analyze_input", "analyze_output",
            "optimize_input", "optimize_output",
            "visualize_sequences", "visualize_df"]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    st.sidebar.success("Cleared app state.")
    st.experimental_rerun()

# Load model once 
model = load_model()

#  PREDICT PAGE
if page == "Predict":
    st.header("AMP Prediction")

    seq_input = st.text_area("Enter peptide sequences (one per line):",
                             value="", height=150)
    uploaded_file = st.file_uploader("Or upload a FASTA/text file", type=["txt", "fasta"])

    run = st.button("Run Prediction")

    if run:

        # Gather sequences
        sequences = []
        if seq_input:
            sequences += [s.strip() for s in seq_input.splitlines() if s.strip()]
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")
            sequences += [l.strip() for l in text.splitlines() if not l.startswith(">") and l.strip()]

        if not sequences:
            st.warning("Please input or upload sequences first.")
        else:
            with st.spinner("Predicting..."):
                results = []
                for seq in sequences:
                    label, conf = predict_amp(seq, model)
                    conf_display = round(conf * 100, 1) if label == "AMP" else round((1 - conf) * 100, 1)
                    results.append({
                        "Sequence": seq,
                        "Prediction": label,
                        "Confidence": conf,
                        "Description": f"{label} with {conf_display}% confidence"
                    })

            # Persist new predictions and mark that we ran
            st.session_state.predictions = results
            st.session_state.predict_ran = True
            st.success("Prediction complete.")

    # If user hasn't just run predictions, show the last saved results (if any)
    if st.session_state.predictions and not (run and st.session_state.predict_ran is False):
        st.subheader("Predictions (last run)")
        st.dataframe(pd.DataFrame(st.session_state.predictions), use_container_width=True)
        csv = pd.DataFrame(st.session_state.predictions).to_csv(index=False)
        st.download_button("Download predictions as CSV", csv, "predictions.csv", "text/csv")

#  ANALYZE PAGE
elif page == "Analyze":
    st.header("Sequence Analysis")

    # show the last saved analyze output if user navigated back
    last_seq = st.session_state.analyze_input
    seq = st.text_input("Enter a peptide sequence to analyze:",
                        value=last_seq)

    # only run analysis when input changed from last saved input
    if seq and seq != st.session_state.get("analyze_input", ""):
        with st.spinner("Running analysis..."):
            label, conf = predict_amp(seq, model)
            conf_pct = round(conf * 100, 1)
            conf_display = conf_pct if label == "AMP" else 100 - conf_pct

            comp = aa_composition(seq)
            props = compute_properties(seq)

            # normalize property key names if necessary
            net_charge = props.get("Net Charge (approx.)",
                                   props.get("Net charge", props.get("NetCharge", 0)))
            
            # build analysis summary (same rules as before)
            length = props.get("Length", len(seq))
            hydro = props.get("Hydrophobic Fraction", props.get("Hydrophobic", 0))
            charge = net_charge
            mw = props.get("Molecular Weight (Da)", props.get("MolecularWeight", 0))

            analysis = []
            if (conf_pct if label == "AMP" else (100 - conf_pct)) >= 80:
                analysis.append(f"Highly likely to be {label}.")
            elif (conf_pct if label == "AMP" else (100 - conf_pct)) >= 60:
                analysis.append(f"Moderately likely to be {label}.")
            else:
                analysis.append(f"Low likelihood to be {label}.")

            if hydro < 0.4:
                analysis.append("Low hydrophobicity may reduce membrane interaction.")
            elif hydro > 0.6:
                analysis.append("High hydrophobicity may reduce solubility.")

            if charge <= 0:
                analysis.append("Low or negative charge may limit antimicrobial activity.")

            if length < 10:
                analysis.append("Short sequence may reduce efficacy.")
            elif length > 50:
                analysis.append("Long sequence may affect stability.")

            if comp.get("K", 0) + comp.get("R", 0) + comp.get("H", 0) >= 3:
                analysis.append("High basic residue content enhances membrane binding.")
            if comp.get("C", 0) + comp.get("W", 0) >= 2:
                analysis.append("Multiple cysteine/tryptophan residues may improve activity.")

            # Save to session state
            st.session_state.analyze_input = seq
            st.session_state.analyze_output = (label, conf, conf_display, comp, props, analysis)

    # If we have stored output, display it
    if st.session_state.analyze_output:
        label, conf, conf_display, comp, props, analysis = st.session_state.analyze_output

        st.subheader("AMP Prediction")
        display_conf = round(conf * 100, 1) if label == "AMP" else round((1 - conf) * 100, 1)
        st.write(f"Prediction: **{label}** with **{display_conf}%** confidence")

        st.subheader("Amino Acid Composition")
        comp_df = pd.DataFrame(list(comp.items()), columns=["Amino Acid", "Frequency"]).set_index("Amino Acid")
        st.bar_chart(comp_df)

        st.subheader("Physicochemical Properties and Favorability")

        # pull properties safely
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
        st.table(pd.DataFrame([
            {"Property": "Length", "Value": length, "Favorability": favorability["Length"]},
            {"Property": "Hydrophobic Fraction", "Value": hydro, "Favorability": favorability["Hydrophobic Fraction"]},
            {"Property": "Net Charge", "Value": charge, "Favorability": favorability["Net Charge"]},
            {"Property": "Molecular Weight", "Value": mw, "Favorability": favorability["Molecular Weight"]}
        ]))

        st.subheader("Property Radar Chart")
        categories = ["Length", "Hydrophobic Fraction", "Net Charge", "Molecular Weight"]
        values = [min(length / 50, 1), min(hydro, 1), 1 if charge > 0 else 0, min(mw / 5000, 1)]
        values += values[:1]
        ideal_min = [10/50, 0.4, 1/6, 500/5000] + [10/50]
        ideal_max = [50/50, 0.6, 6/6, 5000/5000] + [50/50]
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        # Adjusted figsize for better vertical space
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

        # Analysis Summary
        st.subheader("Analysis Summary")
        for line in analysis:
            st.write(f"- {line}")

#  OPTIMIZE PAGE
elif page == "Optimize":
    st.header("AMP Sequence Optimizer")

    # Single entry point: text input retained across navigation
    seq = st.text_input("Enter a peptide sequence to optimize:",
                       value=st.session_state.get("optimize_input", ""))

    # Run optimization when user changes input and clicks button
    if seq and st.button("Run Optimization"):
        st.session_state.optimize_input = seq
        with st.spinner("Optimizing sequence..."):
            improved_seq, improved_conf, history = optimize_sequence(seq, model)
            orig_label, orig_conf = predict_amp(seq, model)
            st.session_state.optimize_output = (seq, orig_conf, improved_seq, improved_conf, history)
        st.success("Optimization finished.")

    # If there is saved output show it
    if st.session_state.optimize_output:
        orig_seq, orig_conf, improved_seq, improved_conf, history = st.session_state.optimize_output
        st.subheader("Results")
        st.write(f"**Original Sequence:** {orig_seq} — Confidence: {round(orig_conf*100,1)}%")
        st.write(f"**Optimized Sequence:** {improved_seq} — Confidence: {round(improved_conf*100,1)}%")

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

            # Confidence improvement plot
            step_nums = df_steps["Step"].tolist()
            conf_values = df_steps["New Confidence (%)"].tolist()
            df_graph = pd.DataFrame({"Step": step_nums, "Confidence (%)": conf_values})
            fig = px.line(df_graph, x="Step", y="Confidence (%)", markers=True, color_discrete_sequence=["#457a00"])
            fig.update_layout(yaxis=dict(range=[0, 100]), title="Confidence Improvement Over Steps")
            st.plotly_chart(fig, use_container_width=True)

#  VISUALIZE PAGE
elif page == "Visualize":
    st.header("Sequence Embedding Visualization")
    st.write("Upload peptide sequences (FASTA or plain list) to visualize embeddings with t-SNE.")

    uploaded_file = st.file_uploader("Upload FASTA or text file", type=["txt", "fasta"])

    # If file uploaded, set session sequences (replacing previous)
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
        sequences = [l.strip() for l in text.splitlines() if not l.startswith(">") and l.strip()]
        st.session_state.visualize_sequences = sequences

        # Clear any previous df so we recompute
        st.session_state.visualize_df = None

    # If we have sequences stored, compute embeddings and t-SNE if no df present
    if st.session_state.visualize_sequences and st.session_state.visualize_df is None:
        sequences = st.session_state.visualize_sequences
        if len(sequences) < 2:
            st.warning("Need at least 2 sequences for t-SNE visualization.")
        else:
            with st.spinner("Generating embeddings and running t-SNE..."):
                embeddings_list, labels, confs, lengths, hydros, charges = [], [], [], [], [], []

                # Use model internals for embeddings; keep same approach as your module
                embedding_extractor = torch.nn.Sequential(*list(model.layers)[:-1])

                for s in sequences:
                    x = torch.tensor(encode_sequence(s), dtype=torch.float32).unsqueeze(0)
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

                embeddings_array = np.stack(embeddings_list)
                perplexity = min(30, max(2, len(sequences) - 1))
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

    # If we have a t-SNE dataframe, show plot and sidebar filters
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

#  ABOUT PAGE
elif page == "About":
    st.header("About the Project")
    st.markdown("""
**Problem:** Antimicrobial resistance is a global health threat. Traditional peptide screening is slow and costly.  
**Solution:** This tool predicts antimicrobial activity directly from sequence using deep learning, speeding up AMP discovery.
""")
