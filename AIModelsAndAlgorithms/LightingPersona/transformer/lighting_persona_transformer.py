"""Transformer model for lighting persona prediction.

This script trains a sequence model on room-day lighting behavior. Each
example is one room on one day, represented as 288 five-minute time steps.
At every time step the model sees per-lamp levels plus activity/context
features, then predicts the dominant ``lightning_persona`` for that room-day.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Data"
OUT = Path(__file__).resolve().parent

SEQ_LEN = 288
SEED = 42

DEFAULT_LAMPS = [
    "bed_left",
    "bed_right",
    "cabinet",
    "closet",
    "corridor_left",
    "corridor_right",
    "dinner_table",
    "hidden_top",
    "shower",
    "sink",
    "table",
]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dominant_value(series: pd.Series):
    modes = series.dropna().mode()
    return modes.iat[0] if len(modes) else np.nan


def build_room_day_sequences(max_rows: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    usecols = [
        "timestamp",
        "room_number",
        "lamp_location",
        "Value",
        "reservation_active",
        "pir_motion",
        "n_occupants",
        "active_actors",
        "hurry_morning",
        "lazy_day",
        "forgetful",
        "lightning_persona",
    ]
    df = pd.read_csv(
        DATA / "lightningData.csv",
        usecols=usecols,
        parse_dates=["timestamp"],
        nrows=max_rows,
    )
    df = df[df["reservation_active"].eq("Yes")].copy()
    df = df.dropna(subset=["timestamp", "lightning_persona"])
    df["date"] = df["timestamp"].dt.date
    df["sample_id"] = df["room_number"].astype(str) + "|" + df["date"].astype(str)
    df["slot"] = df["timestamp"].dt.hour * 12 + df["timestamp"].dt.minute // 5
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0).clip(0, 80) / 80.0

    for col in ["pir_motion", "n_occupants", "active_actors", "hurry_morning", "lazy_day", "forgetful"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    target = df.groupby("sample_id")["lightning_persona"].agg(dominant_value).dropna()
    sample_ids = target.index.tolist()
    slot_index = pd.MultiIndex.from_product(
        [sample_ids, range(SEQ_LEN)],
        names=["sample_id", "slot"],
    )

    lamp_df = df[df["lamp_location"].ne("none")].copy()
    lamp_profile = (
        lamp_df.groupby(["sample_id", "slot", "lamp_location"])["Value"]
        .mean()
        .unstack(fill_value=0.0)
        .reindex(columns=DEFAULT_LAMPS, fill_value=0.0)
    )
    lamp_profile = lamp_profile.reindex(slot_index, fill_value=0.0)

    context_cols = ["pir_motion", "n_occupants", "active_actors", "hurry_morning", "lazy_day", "forgetful"]
    context_profile = df.groupby(["sample_id", "slot"])[context_cols].mean()
    context_profile = context_profile.reindex(slot_index, fill_value=0.0)

    slots = np.tile(np.arange(SEQ_LEN), len(sample_ids))
    time_profile = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * (slots // 12) / 24),
            "hour_cos": np.cos(2 * np.pi * (slots // 12) / 24),
        },
        index=slot_index,
    )

    features = pd.concat([lamp_profile, context_profile, time_profile], axis=1)
    feature_names = list(features.columns)
    x = features.to_numpy(dtype=np.float32).reshape(len(sample_ids), SEQ_LEN, len(feature_names))
    y = target.to_numpy()
    return x, y, feature_names


class LightingPersonaTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        seq_len: int = SEQ_LEN,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 192,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x) + self.pos_embed
        hidden = self.encoder(hidden)
        hidden = self.norm(hidden)
        pooled = hidden.mean(dim=1)
        return self.head(pooled)


def chronological_split(x: np.ndarray, y: np.ndarray, train_fraction: float = 0.8):
    split_idx = max(1, min(len(x) - 1, int(len(x) * train_fraction)))
    return x[:split_idx], x[split_idx:], y[:split_idx], y[split_idx:]


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y_str, feature_names = build_room_day_sequences(max_rows=args.max_rows)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    classes = list(label_encoder.classes_)

    x_train, x_test, y_train, y_test = chronological_split(x, y, train_fraction=args.train_fraction)
    counts = np.bincount(y_train, minlength=len(classes)).clip(min=1)
    weights = torch.tensor(len(y_train) / (len(classes) * counts), dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    model = LightingPersonaTransformer(
        input_dim=x.shape[-1],
        n_classes=len(classes),
        seq_len=SEQ_LEN,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).long()),
        batch_size=args.batch_size,
        shuffle=True,
    )
    x_test_tensor = torch.from_numpy(x_test).to(device)

    best_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)

        model.eval()
        with torch.no_grad():
            pred = model(x_test_tensor).argmax(dim=1).cpu().numpy()
        acc = float((pred == y_test).mean())
        if acc > best_acc:
            best_acc = acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d} train_loss={total_loss / len(x_train):.4f} test_acc={acc:.3f}")

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_tensor).argmax(dim=1).cpu().numpy()

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=3,
        zero_division=0,
    )
    confusion = confusion_matrix(y_test, y_pred, labels=range(len(classes)))
    cm_df = pd.DataFrame(confusion, index=classes, columns=classes)
    cm_df.index.name = "true"

    report_path = output_dir / "lighting_persona_transformer_report.txt"
    cm_path = output_dir / "lighting_persona_transformer_confusion_matrix.csv"
    model_path = output_dir / "lighting_persona_transformer.pt"
    metadata_path = output_dir / "lighting_persona_transformer_metadata.json"

    report_text = "\n".join(
        [
            f"samples: {len(x):,}",
            f"train samples: {len(x_train):,}",
            f"test samples: {len(x_test):,}",
            f"sequence length: {SEQ_LEN}",
            f"input features: {len(feature_names)}",
            f"best test accuracy: {best_acc:.3f}",
            "",
            report,
        ]
    )
    report_path.write_text(report_text)
    cm_df.to_csv(cm_path)
    torch.save(
        {
            "model_state_dict": best_state,
            "classes": classes,
            "feature_names": feature_names,
            "seq_len": SEQ_LEN,
            "input_dim": x.shape[-1],
            "config": {
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
            },
        },
        model_path,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "classes": classes,
                "feature_names": feature_names,
                "seq_len": SEQ_LEN,
                "input_dim": x.shape[-1],
                "model_file": model_path.name,
                "report_file": report_path.name,
                "confusion_matrix_file": cm_path.name,
            },
            indent=2,
        )
    )

    print(report_text)
    print(f"saved {model_path}")
    print(f"saved {metadata_path}")
    print(f"saved {report_path}")
    print(f"saved {cm_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit for quick experiments.")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="Artifact directory. Defaults to the app's lighting persona Transformer folder.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
