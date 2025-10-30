import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import streamlit as st
import torch
import numpy as np
from utils.predict import encode_sequence

# t-SNE Visualization
def tsne_visualization(sequences, model):
    st.info("Generating embeddings... this may take a moment.")
    embeddings = []
    for seq in sequences:
        x = torch.tensor(encode_sequence(seq), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            emb = model.layers[0](x)  # Grab first layer embedding
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