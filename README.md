---
title: PeptideAI
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.41.1"
python_version: "3.13"
app_file: StreamlitApp/StreamlitApp.py
pinned: false
short_description: AMP peptide scoring, composition, wheel & 3D views.
---

# PeptideAI

**Live app:** [huggingface.co/spaces/m0ksh/PeptideAI](https://huggingface.co/spaces/m0ksh/PeptideAI)

PeptideAI is a Streamlit app for working with short peptide sequences. It estimates whether a sequence might behave like an antimicrobial peptide (AMP) using a small neural network, and adds views for composition, rough physicochemical numbers, optional mutation search, and helix-style visualization.

## What you can do

- Get an AMP vs non-AMP prediction with a confidence-style score  
- See amino acid composition and simple properties (length, charge, hydrophobic fraction, mass)  
- Run a greedy “optimize” pass that tries mutations the model likes more  
- Visualize a helix-like trace and helical wheel (approximation, not a solved structure)  
- Run t-SNE on embeddings when you have several sequences  

## Run it on your machine

```bash
pip install -r requirements.txt
streamlit run StreamlitApp/StreamlitApp.py
```