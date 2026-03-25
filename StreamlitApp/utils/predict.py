# Predict page (and shared): load AMP model, one-hot encode, run predict_amp.
import pathlib
import numpy as np
import torch
import streamlit as st
from torch import nn

# Lightweight MLP used for AMP binary classification.
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

@st.cache_resource
def load_model():
    # Load model weights once per Streamlit process.
    # Always resolve relative to the StreamlitApp folder, not the process CWD.
    streamlitapp_dir = pathlib.Path(__file__).resolve().parent.parent
    repo_root = streamlitapp_dir.parent

    candidates = [
        repo_root / "MLModels" / "ampMLModel.pt",
        repo_root / "models" / "ampMLModel.pt",
        streamlitapp_dir / "models" / "ampMLModel.pt",
    ]
    model_path = next((p for p in candidates if p.exists()), candidates[0])

    if not model_path.exists():
        raise FileNotFoundError(
            "Model file 'ampMLModel.pt' not found in any of:\n"
            f"- {repo_root / 'MLModels' / 'ampMLModel.pt'}\n"
            f"- {repo_root / 'models' / 'ampMLModel.pt'}\n"
            f"- {streamlitapp_dir / 'models' / 'ampMLModel.pt'}\n"
        )

    # Instantiate architecture and hydrate weights from disk.
    model = FastMLP(input_dim=1024)
    model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
    model.eval()
    return model

def encode_sequence(seq, max_len=51):
    # Convert sequence to a padded/truncated flattened one-hot vector (1024 dims).
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

    # Encode each residue as a one-hot row, then flatten to vector features.
    one_hot = np.zeros((max_len, len(amino_acids)))
    for i, aa in enumerate(seq[:max_len]):
        if aa in aa_to_idx:
            one_hot[i, aa_to_idx[aa]] = 1

    flat = one_hot.flatten()

    if len(flat) < 1024:
        flat = np.pad(flat, (0, 1024 - len(flat)))

    return flat

def predict_amp(sequence, model):
    # Run AMP inference and return predicted label plus AMP probability.
    x = torch.tensor(encode_sequence(sequence), dtype=torch.float32).unsqueeze(0)

    # Sigmoid(logit) gives AMP probability in [0, 1].
    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()

    label = "AMP" if prob >= 0.5 else "Non-AMP"
    return label, round(prob, 3)