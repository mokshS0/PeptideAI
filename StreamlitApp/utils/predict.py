# Predict page (and shared): ProtBERT embedding + MLP classifier inference.
import pathlib
import numpy as np
import torch
import streamlit as st
from torch import nn
from transformers import BertModel, BertTokenizer

MODEL_INPUT_DIM = 1024  # ProtBERT pooled embedding size; MLP first layer must match.
MODEL_ARCH = "FastMLP"
PROTBERT_MODEL_NAME = "Rostlab/prot_bert"  # HF id for tokenizer + encoder weights.

class FastMLP(nn.Module):
    # Small classifier head on top of frozen ProtBERT embeddings at inference.
    def __init__(self, input_dim=MODEL_INPUT_DIM):
        super(FastMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Single output logit for binary classification
        )

    def forward(self, x):
        return self.layers(x)


def _load_checkpoint(path: pathlib.Path):
    # Accept either raw state_dict (legacy) or structured checkpoint dict.
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj.get("meta", {})
    if isinstance(obj, dict):
        return obj, {}
    raise ValueError(
        f"Unsupported model checkpoint format at '{path}'. "
        "Expected a PyTorch state_dict or {'state_dict': ..., 'meta': ...}."
    )


def _infer_first_layer_input_dim(state_dict: dict) -> int | None:
    # Infer MLP input dim from Linear weight shape (out_features, in_features).
    w = state_dict.get("layers.0.weight")
    if w is None:
        return None
    if hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[1])
    return None


def _normalize_sequence(sequence: str) -> str:
    # Uppercase + strip whitespace so tokenization matches training conventions.
    return "".join(c for c in str(sequence).upper() if not c.isspace())


@st.cache_resource
def load_model():
    # Load AMP classifier weights + ProtBERT encoder once per Streamlit process.
    streamlitapp_dir = pathlib.Path(__file__).resolve().parent.parent
    repo_root = streamlitapp_dir.parent

    candidates = [
        repo_root / "MLModels" / "ampMLModel.pt",
        repo_root / "MLModels" / "fast_mlp_amp.pt",
        repo_root / "models" / "ampMLModel.pt",
        streamlitapp_dir / "models" / "ampMLModel.pt",
    ]
    # Prefer first existing path so local / HF layouts both work.
    model_path = next((p for p in candidates if p.exists()), candidates[0])

    if not model_path.exists():
        raise FileNotFoundError(
            "Classifier checkpoint not found in any of:\n"
            f"- {repo_root / 'MLModels' / 'ampMLModel.pt'}\n"
            f"- {repo_root / 'MLModels' / 'fast_mlp_amp.pt'}\n"
            f"- {repo_root / 'models' / 'ampMLModel.pt'}\n"
            f"- {streamlitapp_dir / 'models' / 'ampMLModel.pt'}\n"
        )

    state_dict, _meta = _load_checkpoint(model_path)
    inferred_input_dim = _infer_first_layer_input_dim(state_dict)
    if inferred_input_dim != MODEL_INPUT_DIM:
        raise ValueError(
            "Model/input mismatch. Loaded classifier expects "
            f"{inferred_input_dim} input features; ProtBERT pooled embeddings are {MODEL_INPUT_DIM}-dim."
        )

    classifier = FastMLP(input_dim=MODEL_INPUT_DIM)
    classifier.load_state_dict(state_dict)
    classifier.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use an explicit slow tokenizer to avoid fast-backend conversion issues on Spaces.
    tokenizer = BertTokenizer.from_pretrained(PROTBERT_MODEL_NAME, do_lower_case=False)

    # Use explicit BERT class to avoid AutoModel config auto-detection issues.
    encoder = BertModel.from_pretrained(PROTBERT_MODEL_NAME).to(device)
    encoder.eval()

    return {
        "classifier": classifier,
        "tokenizer": tokenizer,
        "encoder": encoder,
        "device": device,
        "classifier_path": str(model_path),
    }


def encode_sequence(seq, model_bundle):
    # Convert peptide sequence to ProtBERT mean-pooled embedding (1024 dims).
    clean = _normalize_sequence(seq)
    spaced = " ".join(list(clean))
    tokenizer = model_bundle["tokenizer"]
    encoder = model_bundle["encoder"]
    device = model_bundle["device"]

    tokens = tokenizer(
        spaced,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(device)
    with torch.no_grad():
        outputs = encoder(**tokens)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy()
    return emb.astype(np.float32)


def get_embedding_extractor(model_bundle):
    # Penultimate MLP activations for t-SNE (same depth as training-time “embedding” use).
    classifier = model_bundle["classifier"]
    extractor = torch.nn.Sequential(*list(classifier.layers)[:-1])
    extractor.eval()
    return extractor


def predict_amp(sequence, model_bundle):
    # Run AMP inference and return predicted label plus AMP probability.
    x = torch.tensor(encode_sequence(sequence, model_bundle), dtype=torch.float32).unsqueeze(0)
    classifier = model_bundle["classifier"]

    with torch.no_grad():
        logits = classifier(x)
        prob = torch.sigmoid(logits).item()

    label = "AMP" if prob >= 0.5 else "Non-AMP"
    return label, round(prob, 3)