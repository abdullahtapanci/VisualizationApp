"""Transformer persona classifier.

Same task as the LSTM script (predict persona from a stay's 24h
PIR motion profile) but with a Transformer encoder. Each 5-min slot
is projected to d_model dims, summed with a learned positional
embedding, run through stacked self-attention layers, mean-pooled,
then sent to a linear head with class-weighted cross-entropy.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA = Path(__file__).resolve().parent.parent.parent / "Data"
OUT = Path(__file__).resolve().parent

SEQ_LEN = 288       # 24h * 12 five-min slots
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
DIM_FF = 128
DROPOUT = 0.2
BATCH = 32
EPOCHS = 40
LR = 3e-4
WEIGHT_DECAY = 1e-2
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)


def build_daily_profiles():
    pir = pd.read_csv(DATA / "PIRSensorData.csv", parse_dates=["timestamp"])
    pir = pir.dropna(subset=["guest_id"]).copy()
    pir["guest_id"] = pir["guest_id"].astype(int)
    pir["slot"] = pir["timestamp"].dt.hour * 12 + pir["timestamp"].dt.minute // 5

    profile = (
        pir.groupby(["guest_id", "slot"])["pir_motion"].mean()
        .unstack(fill_value=0.0)
        .reindex(columns=range(SEQ_LEN), fill_value=0.0)
        .sort_index()
    )
    persona = (
        pir.groupby("guest_id")["persona"].agg(lambda s: s.mode().iat[0])
        .reindex(profile.index)
    )
    return profile.values.astype(np.float32), persona.values


class PersonaTransformer(nn.Module):
    def __init__(self, n_classes, seq_len=SEQ_LEN, d_model=D_MODEL,
                 nhead=N_HEADS, n_layers=N_LAYERS, dim_ff=DIM_FF, dropout=DROPOUT):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        # learned positional embedding (fixed-length input -> simpler than sinusoidal)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,    # pre-LN: more stable for small datasets
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):                     # x: (B, T, 1)
        h = self.input_proj(x) + self.pos_embed
        h = self.encoder(h)                   # (B, T, d_model)
        h = self.norm(h)
        pooled = h.mean(dim=1)                # (B, d_model)
        return self.head(pooled)              # (B, n_classes)


def main():
    X, y_str = build_daily_profiles()
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    classes = list(le.classes_)
    print(f"stays: {len(X)}, classes: {classes}")
    print(f"class counts: {dict(zip(classes, np.bincount(y)))}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    counts = np.bincount(y_tr)
    weights = torch.tensor(len(y_tr) / (len(counts) * counts), dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PersonaTransformer(n_classes=len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    x_tr = torch.from_numpy(X_tr).unsqueeze(-1)
    y_tr_t = torch.from_numpy(y_tr).long()
    x_te = torch.from_numpy(X_te).unsqueeze(-1).to(device)

    loader = DataLoader(TensorDataset(x_tr, y_tr_t), batch_size=BATCH, shuffle=True)

    best_acc, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # transformers are gradient-prone
            opt.step()
            total += loss.item() * xb.size(0)

        model.eval()
        with torch.no_grad():
            acc = (model(x_te).argmax(1).cpu().numpy() == y_te).mean()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(f"  epoch {epoch:3d}  train_loss={total/len(x_tr):.4f}  test_acc={acc:.3f}")

    print(f"\nbest test_acc during training: {best_acc:.3f}")
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred = model(x_te).argmax(1).cpu().numpy()

    print("\nclassification report:")
    print(classification_report(y_te, y_pred, target_names=classes, digits=3))

    cm = confusion_matrix(y_te, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name = "true"
    cm_df.to_csv(OUT / "transformer_confusion_matrix.csv")
    print("confusion matrix (rows=true, cols=pred):")
    print(cm_df)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        fig.savefig(OUT / "transformer_confusion_matrix.png", dpi=120)
        print(f"saved {OUT / 'transformer_confusion_matrix.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
