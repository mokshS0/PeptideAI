# t-SNE page: optional helper embedding + scatter (StreamlitApp also runs t-SNE inline with Plotly).
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import streamlit as st
import torch
import numpy as np
from utils.predict import encode_sequence, get_embedding_extractor

def tsne_visualization(sequences, model):
    # Project model embeddings into 2D and render a quick scatter plot.
    st.info("Generating embeddings... this may take a moment.")
    embeddings = []
    embedding_extractor = get_embedding_extractor(model)
    for seq in sequences:
        x = torch.tensor(encode_sequence(seq, model), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            # Use an early hidden layer as a compact learned representation.
            emb = embedding_extractor(x)
        embeddings.append(emb.numpy().flatten())

    embeddings = np.vstack(embeddings)

    perplexity = min(30, len(sequences) - 1)
    if perplexity < 2:
        st.warning("Need at least 2 sequences for visualization.")
        return

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced = tsne.fit_transform(embeddings)
    df = pd.DataFrame(reduced, columns=["x", "y"])

    st.success("t-SNE visualization complete.")
    st.scatter_chart(df)