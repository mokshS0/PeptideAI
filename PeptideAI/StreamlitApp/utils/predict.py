import os
import pathlib
import requests
import numpy as np
import torch
import streamlit as st
from torch import nn

# Model Definition
class FastMLP(nn.Module):
    def __init__(self, input_dim=1024):
        super(FastMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single output for binary classification
        )

    def forward(self, x):
        return self.layers(x)

# Utility: download file from URL to local path (streaming)
def _download_file(url: str, dest_path: str):
    dest = pathlib.Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# Model Loader
@st.cache_resource
def load_model():
    model_path = pathlib.Path("./models/ampMLModel.pt")

    # If the model file doesn't exist, try to download it from a configured URL
    if not model_path.exists():

        # Try Streamlit secrets first, then environment variable
        model_url = None
        try:
            model_url = st.secrets.get("MODEL_URL")
        except Exception:
            model_url = None

        if not model_url:
            model_url = os.environ.get("MODEL_URL")

        if model_url:
            try:
                _download_file(model_url, str(model_path))
            except Exception as e:
                st.error(f"Failed to download model from MODEL_URL: {e}")
                raise
        else:
            raise FileNotFoundError(
                "Model file './models/ampMLModel.pt' not found.\n"
            )

    # Build model and load weights
    model = FastMLP(input_dim=1024)
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()
    return model

# Sequence Encoder
def encode_sequence(seq, max_len=51):
    """
    Converts amino acid sequence to flattened one-hot vector
    padded/truncated to match model input_dim (1024)
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

    one_hot = np.zeros((max_len, len(amino_acids)))  # max_len x 20
    for i, aa in enumerate(seq[:max_len]):
        if aa in aa_to_idx:
            one_hot[i, aa_to_idx[aa]] = 1

    flat = one_hot.flatten()  # length = max_len*20 = 1020

    if len(flat) < 1024:
        flat = np.pad(flat, (0, 1024 - len(flat)))

    return flat

# Prediction Function
def predict_amp(sequence, model):
    """
    Takes an amino acid sequence string and the loaded model,
    returns ("AMP"/"Non-AMP") and probability
    """
    x = torch.tensor(encode_sequence(sequence), dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    label = "AMP" if prob >= 0.5 else "Non-AMP"
    return label, round(prob, 3)
