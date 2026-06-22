"""LSTM persona classifier.

Supervised counterpart to the CNN autoencoder: given a stay's 24h
PIR motion profile, predict the persona label (Explorer / Napper /
NightOwl / Standard). Uses a bidirectional LSTM with a small MLP
head, class-weighted cross-entropy for imbalance, and a stratified
80/20 hold-out for honest evaluation.
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

SEQ_LEN = 288        # 24h * 12 five-min slots
HIDDEN = 64
N_LAYERS = 2
BATCH = 32
EPOCHS = 40
LR = 1e-3
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)


def build_daily_profiles():
    """Return (X, persona) where X is (N, 288) in [0,1]."""
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


class LSTMClassifier(nn.Module):
    def __init__(self, n_classes, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):                     # x: (B, T, 1)
        out, _ = self.lstm(x)                 # (B, T, 2*H)
        pooled = out.mean(dim=1)              # mean-pool across time
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

    # class weights = inverse frequency, normalized
    counts = np.bincount(y_tr)
    weights = torch.tensor(len(y_tr) / (len(counts) * counts), dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMClassifier(n_classes=len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    x_tr = torch.from_numpy(X_tr).unsqueeze(-1)    # (N, T, 1)
    y_tr_t = torch.from_numpy(y_tr).long()
    x_te = torch.from_numpy(X_te).unsqueeze(-1).to(device)

    loader = DataLoader(TensorDataset(x_tr, y_tr_t), batch_size=BATCH, shuffle=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        if epoch == 1 or epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(x_te).argmax(1).cpu().numpy() == y_te).mean()
            print(f"  epoch {epoch:3d}  train_loss={total/len(x_tr):.4f}  test_acc={acc:.3f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(x_te).argmax(1).cpu().numpy()

    print("\nclassification report:")
    print(classification_report(y_te, y_pred, target_names=classes, digits=3))

    cm = confusion_matrix(y_te, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name = "true"
    cm_df.to_csv(OUT / "lstm_confusion_matrix.csv")
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
        fig.savefig(OUT / "lstm_confusion_matrix.png", dpi=120)
        print(f"saved {OUT / 'lstm_confusion_matrix.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
