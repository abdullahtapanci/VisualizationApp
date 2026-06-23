from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AIModelsAndAlgorithms.TempretureRecomendation.energy_saving_targets import (  # noqa: E402
    BASE_NUMERIC_COLS,
    CATEGORICAL_COLS,
    SETPOINT_MAX,
    SETPOINT_MIN,
    build_feature_row,
    prepare_temperature_frame,
)
from backend.hvac_energy import _estimate_power  # noqa: E402


DATA_CSV = ROOT / "Data" / "temperatureData.csv"
OUT = ROOT / "AIModelsAndAlgorithms" / "TempretureRecomendation" / "transformer"


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TemperatureRecommendationTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 192,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        hidden = self.input_projection(x)
        hidden = self.position(hidden)
        hidden = self.encoder(hidden)
        return torch.sigmoid(self.head(hidden[:, -1, :])).squeeze(-1)


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]], list[str]]:
    rows = [build_feature_row(row, pd.Timestamp(row["timestamp"])) for _, row in df.iterrows()]
    features = pd.DataFrame(rows)
    category_values = {}
    feature_names = list(BASE_NUMERIC_COLS)
    parts = [features[BASE_NUMERIC_COLS].astype("float32").reset_index(drop=True)]
    for col in CATEGORICAL_COLS:
        values = sorted(features[col].fillna("Unknown").astype(str).unique().tolist())
        if "Unknown" not in values:
            values.append("Unknown")
        category_values[col] = values
        encoded = pd.DataFrame(
            {
                f"{col}={value}": (features[col].fillna("Unknown").astype(str) == value).astype("float32")
                for value in values
            }
        )
        feature_names.extend(encoded.columns.tolist())
        parts.append(encoded.reset_index(drop=True))
    return pd.concat(parts, axis=1), category_values, feature_names


def build_sequences(
    frame: pd.DataFrame,
    encoded_features: np.ndarray,
    targets_scaled: np.ndarray,
    seq_len: int,
    stride: int,
    max_sequences: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    end_indices = []
    for _, group in frame.groupby("room_number", sort=False):
        idx = group.index.to_numpy()
        if len(idx) < seq_len:
            continue
        for end in range(seq_len - 1, len(idx), stride):
            window_idx = idx[end - seq_len + 1 : end + 1]
            xs.append(encoded_features[window_idx])
            ys.append(targets_scaled[idx[end]])
            end_indices.append(idx[end])
            if max_sequences and len(xs) >= max_sequences:
                break
        if max_sequences and len(xs) >= max_sequences:
            break
    if not xs:
        raise ValueError("No sequences were built. Increase max rows or reduce sequence length.")
    return np.stack(xs).astype("float32"), np.asarray(ys, dtype="float32"), np.asarray(end_indices)


def split_sequences(x: np.ndarray, y: np.ndarray, end_indices: np.ndarray):
    n = len(x)
    train_end = max(1, int(n * 0.8))
    val_end = max(train_end + 1, int(n * 0.9))
    val_end = min(val_end, n - 1)
    return (
        x[:train_end],
        y[:train_end],
        x[train_end:val_end],
        y[train_end:val_end],
        x[val_end:],
        y[val_end:],
        end_indices[val_end:],
    )


def estimate_model_power(row: pd.Series, setpoint: float) -> float:
    candidate = row.copy()
    current_mode = str(row.get("current_energy_mode") or row.get("hvac_mode") or "off")
    room_temp = float(row.get("room_temp") or 0.0)
    if current_mode == "heating":
        mode = "heating" if room_temp < setpoint - 0.4 else "off"
    elif current_mode == "cooling":
        mode = "cooling" if room_temp > setpoint + 0.4 else "off"
    else:
        mode = "heating" if room_temp < setpoint - 0.6 else ("cooling" if room_temp > setpoint + 0.6 else "off")
    candidate["setpoint"] = setpoint
    candidate["hvac_mode"] = mode
    return float(_estimate_power(candidate))


def train(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = prepare_temperature_frame(DATA_CSV, max_rows=args.max_rows)
    df = df.dropna(subset=["recommended_setpoint"]).sort_values(["timestamp", "room_number"]).reset_index(drop=True)
    encoded, category_values, feature_names = build_feature_matrix(df)
    targets_scaled = ((df["recommended_setpoint"].astype(float) - SETPOINT_MIN) / (SETPOINT_MAX - SETPOINT_MIN)).clip(0, 1).to_numpy("float32")
    x, y, end_indices = build_sequences(
        df,
        encoded.to_numpy("float32"),
        targets_scaled,
        args.sequence_length,
        args.stride,
        args.max_sequences,
    )
    x_train, y_train, x_val, y_val, x_test, y_test, test_end_indices = split_sequences(x, y, end_indices)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"

    model = TemperatureRecommendationTransformer(
        input_dim=x.shape[-1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_tensor = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * xb.size(0)
            count += xb.size(0)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_tensor), val_y).item()) if len(x_val) else total / max(count, 1)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d} train_loss={total / max(count, 1):.5f} val_loss={val_loss:.5f}")

    if best_state is None:
        raise RuntimeError("Training did not produce a model state.")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    test_loader = DataLoader(torch.from_numpy(x_test), batch_size=args.batch_size, shuffle=False)
    preds_scaled = []
    with torch.no_grad():
        for xb in test_loader:
            preds_scaled.append(model(xb.to(device)).cpu().numpy())
    pred_scaled = np.concatenate(preds_scaled) if preds_scaled else np.array([], dtype="float32")
    pred = np.round(np.clip(pred_scaled * (SETPOINT_MAX - SETPOINT_MIN) + SETPOINT_MIN, SETPOINT_MIN, SETPOINT_MAX) * 2) / 2
    target = np.round((y_test * (SETPOINT_MAX - SETPOINT_MIN) + SETPOINT_MIN) * 2) / 2

    mae = mean_absolute_error(target, pred)
    rmse = mean_squared_error(target, pred) ** 0.5
    r2 = r2_score(target, pred)

    test_rows = df.iloc[test_end_indices].copy().reset_index(drop=True)
    test_rows["recommended_target"] = target
    test_rows["model_prediction"] = pred
    test_rows["model_power_w"] = [
        estimate_model_power(row, setpoint) for (_, row), setpoint in zip(test_rows.iterrows(), pred)
    ]
    current_energy = float(test_rows["current_power_w"].sum())
    target_energy = float(test_rows["target_power_w"].sum())
    model_energy = float(test_rows["model_power_w"].sum())
    target_saving_pct = 100 * (current_energy - target_energy) / current_energy if current_energy else 0.0
    model_saving_pct = 100 * (current_energy - model_energy) / current_energy if current_energy else 0.0
    comfort_gap = float((test_rows["room_temp"] - test_rows["ideal_temp"]).abs().mean())

    sample = test_rows.rename(
        columns={
            "room_state": "occupancy_prediction",
            "ac_persona": "temperature_persona_prediction",
        }
    )
    sample_cols = [
        "timestamp",
        "room_number",
        "hvac_mode",
        "occupancy_prediction",
        "temperature_persona_prediction",
        "room_temp",
        "outside_temp",
        "setpoint",
        "ideal_temp",
        "recommended_target",
        "model_prediction",
        "current_power_w",
        "target_power_w",
        "model_power_w",
        "target_mode",
    ]
    sample_path = output_dir / "tempreture_recomendation_transformer_sample_predictions.csv"
    sample[sample_cols].head(args.sample_rows).to_csv(sample_path, index=False)

    metadata = {
        "task": "tempreture_recomendation",
        "target": "recommended_setpoint",
        "target_strategy": "learned_energy_saving_comfort_bounded",
        "setpoint_min": SETPOINT_MIN,
        "setpoint_max": SETPOINT_MAX,
        "sequence_length": args.sequence_length,
        "input_dim": x.shape[-1],
        "base_numeric_cols": BASE_NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "category_values": category_values,
        "feature_names": feature_names,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "max_rows": args.max_rows,
        "max_sequences": args.max_sequences,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "target_saving_pct": target_saving_pct,
            "model_saving_pct": model_saving_pct,
            "target_mean_comfort_gap_c": comfort_gap,
            "model_mean_comfort_gap_c": comfort_gap,
        },
    }
    model_path = output_dir / "tempreture_recomendation_transformer.pt"
    torch.save(
        {
            "state_dict": best_state,
            "metadata": metadata,
        },
        model_path,
    )
    metadata_path = output_dir / "tempreture_recomendation_transformer_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    report_lines = [
        f"rows: {len(df):,}",
        f"sequences: {len(x):,}",
        f"train sequences: {len(x_train):,}",
        f"val sequences: {len(x_val):,}",
        f"test sequences: {len(x_test):,}",
        "algorithm: TransformerEncoder",
        "target strategy: learned energy-saving comfort-bounded setpoint",
        f"MAE: {mae:.3f}",
        f"RMSE: {rmse:.3f}",
        f"R2: {r2:.3f}",
        f"current energy proxy W-sum: {current_energy:.0f}",
        f"target energy proxy W-sum: {target_energy:.0f}",
        f"model energy proxy W-sum: {model_energy:.0f}",
        f"target saving vs current: {target_saving_pct:.2f}%",
        f"model saving vs current: {model_saving_pct:.2f}%",
        f"target mean comfort gap C: {comfort_gap:.3f}",
        f"model mean comfort gap C: {comfort_gap:.3f}",
    ]
    report_text = "\n".join(report_lines)
    report_path = output_dir / "tempreture_recomendation_transformer_report.txt"
    report_path.write_text(report_text + "\n")

    print(report_text)
    print(f"saved {model_path}")
    print(f"saved {metadata_path}")
    print(f"saved {report_path}")
    print(f"saved {sample_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--sample-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
