# Train AMP classifier on ProtBERT embeddings; export MLP weights for Streamlit `load_model`.
# Inference in Streamlit uses BertTokenizer + BertModel + same mean-pool as here.

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "Rostlab/prot_bert"  # Same HF id as Streamlit `PROTBERT_MODEL_NAME`.
INPUT_DIM = 1024  # Mean-pooled ProtBERT last_hidden_state dim.


class FastMLP(nn.Module):
    # Binary head; Streamlit loads these weights only (encoder pulled separately).
    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _normalize(seq: str) -> str:
    # Match app-side `_normalize_sequence` (uppercase, no spaces).
    return "".join(c for c in str(seq).upper() if not c.isspace())


def get_embedding(sequence: str, tokenizer, encoder, device) -> np.ndarray:
    # Space-separated residues: ProtBERT tokenizer expects word-piece style input.
    spaced = " ".join(list(_normalize(sequence)))
    tokens = tokenizer(spaced, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = encoder(**tokens)
    emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy()
    return emb.astype(np.float32)


def load_dataset(csv_path: pathlib.Path) -> pd.DataFrame:
    # Expected columns: sequence (str), label (0/1).
    df = pd.read_csv(csv_path)
    if "sequence" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must include 'sequence' and 'label' columns.")
    df = df.dropna(subset=["sequence", "label"]).copy()
    df["sequence"] = df["sequence"].astype(str).map(_normalize)
    df["label"] = df["label"].astype(int)
    return df


def main():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train ProtBERT + MLP AMP model.")
    parser.add_argument("--csv", type=pathlib.Path, default=repo_root / "Data" / "ampData.csv")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent / "ampMLModel.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_dataset(args.csv)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = AutoModel.from_pretrained(MODEL_NAME).to(device)
    encoder.eval()

    print("Extracting ProtBERT embeddings...")
    x = np.stack([get_embedding(s, tokenizer, encoder, device) for s in df["sequence"]])
    y = df["label"].to_numpy(dtype=np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)

    model = FastMLP(input_dim=INPUT_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()  # logits; sigmoid applied only for metrics / deployment.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(x_train_t.size(0), device=device)
        total_loss = 0.0
        for i in range(0, x_train_t.size(0), args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb, yb = x_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {total_loss:.4f}")

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x_test_t)).cpu().numpy().flatten()
    pred_labels = (probs >= 0.5).astype(int)
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
    print(f"PR-AUC:  {average_precision_score(y_test, probs):.4f}")
    print("\nClassification report:\n")
    print(classification_report(y_test.astype(int), pred_labels, digits=4))

    # Classifier-only checkpoint; ProtBERT weights are always loaded from HF at inference.
    checkpoint = {
        "state_dict": model.cpu().state_dict(),
        "meta": {
            "arch": "FastMLP",
            "input_dim": INPUT_DIM,
            "encoding": "protbert_mean_pool",
            "uses_protbert": True,
            "protbert_model_name": MODEL_NAME,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    print(f"Saved checkpoint: {args.out}")


if __name__ == "__main__":
    main()
