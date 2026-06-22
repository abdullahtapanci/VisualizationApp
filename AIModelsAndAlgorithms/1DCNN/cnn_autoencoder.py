"""1D CNN autoencoder for per-stay motion sequences.

Each stay -> fixed-length 24h motion profile (288 = 24*12 five-minute
slots, averaged across all days of the stay). A small 1D conv encoder
compresses the profile to a low-dimensional embedding; the decoder
reconstructs it. Embeddings are saved for downstream k-means / HDBSCAN.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA = Path(__file__).resolve().parent.parent / "Data"
OUT = Path(__file__).resolve().parent

SEQ_LEN = 288          # 24h * 12 slots/h
EMBED_DIM = 8
BATCH = 64
EPOCHS = 150
LR = 1e-3
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)


def build_daily_profiles():
    """Return (X, guest_ids, persona) where X is (N, 288) in [0,1]."""
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
    return profile.values.astype(np.float32), profile.index.values, persona.values


class ConvAE(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        # encoder: 288 -> 144 -> 72 -> 36
        self.enc = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.flat_dim = 64 * (SEQ_LEN // 8)
        self.to_embed = nn.Linear(self.flat_dim, embed_dim)
        self.from_embed = nn.Linear(embed_dim, self.flat_dim)
        # decoder: 36 -> 72 -> 144 -> 288
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(64, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(32, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(16, 1, 7, padding=3),
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.to_embed(h)

    def decode(self, z):
        h = self.from_embed(z).view(-1, 64, SEQ_LEN // 8)
        return self.dec(h)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def main():
    X, guest_ids, persona = build_daily_profiles()
    print(f"stays: {len(X)}, seq_len: {X.shape[1]}")

    x = torch.from_numpy(X).unsqueeze(1)              # (N, 1, 288)
    loader = DataLoader(TensorDataset(x), batch_size=BATCH, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * batch.size(0)
        if epoch == 1 or epoch % 10 == 0:
            print(f"  epoch {epoch:3d}  loss={total / len(x):.4f}")

    model.eval()
    with torch.no_grad():
        z = model.encode(x.to(device)).cpu().numpy()

    emb = pd.DataFrame(z, index=guest_ids, columns=[f"z{i:02d}" for i in range(EMBED_DIM)])
    emb.index.name = "guest_id"
    emb["true_persona"] = persona
    emb.to_csv(OUT / "cnn_embeddings.csv")
    print(f"saved {OUT / 'cnn_embeddings.csv'}  shape={emb.shape}")

    # quick sanity-check plot: 4 reconstructions
    try:
        import matplotlib.pyplot as plt
        with torch.no_grad():
            sample = x[:4].to(device)
            recon = torch.sigmoid(model(sample)[0]).cpu().numpy().squeeze(1)
        fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(X[i], label="input", lw=1)
            ax.plot(recon[i], label="recon", lw=1)
            ax.set_ylabel(f"stay {guest_ids[i]}\n{persona[i]}")
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("5-min slot of day (0..287)")
        plt.tight_layout()
        fig.savefig(OUT / "cnn_reconstructions.png", dpi=120)
        print(f"saved {OUT / 'cnn_reconstructions.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
