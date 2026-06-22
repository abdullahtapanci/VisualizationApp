#!/usr/bin/env python3
"""Train the energy-aware Transformer HVAC setpoint recommender.

This script writes the same artifact names used by the Flask app:

* tempreture_recomendation_transformer.pt
* tempreture_recomendation_transformer_metadata.json
* tempreture_recomendation_transformer_report.txt
* tempreture_recomendation_transformer_sample_predictions.csv
"""

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


THIS_DIR = Path(__file__).resolve().parent
MODEL_DIR = THIS_DIR.parent
ROOT = THIS_DIR.parents[2]
DATA = ROOT / "Data"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from energy_aware_target import (  # noqa: E402
    SETPOINT_MAX,
    SETPOINT_MIN,
    energy_aware_recommended_setpoints,
    target_diagnostics,
)


SEED = 42

USECOLS = [
    "timestamp",
    "room_number",
    "floor",
    "facade",
    "room_type",
    "size_m2",
    "outside_temp",
    "room_temp",
    "setpoint",
    "ideal_temp",
    "hvac_mode",
    "ac_persona",
    "occupant_state",
    "pir_persona",
    "room_state",
    "pir_motion",
    "guest_id",
]

BASE_NUMERIC_COLS = [
    "floor_scaled",
    "size_scaled",
    "outside_temp_scaled",
    "room_temp_scaled",
    "setpoint_scaled",
    "ideal_temp_scaled",
    "temp_error_scaled",
    "comfort_error_scaled",
    "pir_motion",
    "has_guest",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

CATEGORICAL_COLS = [
    "facade",
    "room_type",
    "hvac_mode",
    "occupant_state",
    "pir_persona",
    "occupancy_prediction",
    "temperature_persona_prediction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=500_000)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=THIS_DIR,
        help="Artifact directory. Defaults to the app's Transformer model folder.",
    )
    return parser.parse_args()


def cyclic(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    radians = 2 * np.pi * series / period
    return np.sin(radians), np.cos(radians)


def scale_temp(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0).clip(-20, 50)
    return (values + 20) / 70


def build_frame(max_rows: int | None) -> pd.DataFrame:
    df = pd.read_csv(
        DATA / "temperatureData.csv",
        usecols=USECOLS,
        parse_dates=["timestamp"],
        nrows=max_rows,
    )
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values(["room_number", "timestamp"]).reset_index(drop=True)

    for col in ["floor", "size_m2", "outside_temp", "room_temp", "setpoint", "ideal_temp", "pir_motion"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["guest_id"] = pd.to_numeric(df["guest_id"], errors="coerce")

    df["occupancy_prediction"] = df["room_state"].fillna("Unknown").astype(str)
    df["temperature_persona_prediction"] = df["ac_persona"].fillna("Unknown").astype(str)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("Unknown").astype(str)

    df["floor_scaled"] = df["floor"].clip(0, 30) / 30.0
    df["size_scaled"] = df["size_m2"].clip(0, 120) / 120.0
    df["outside_temp_scaled"] = scale_temp(df["outside_temp"])
    df["room_temp_scaled"] = scale_temp(df["room_temp"])
    df["setpoint_scaled"] = scale_temp(df["setpoint"])
    df["ideal_temp_scaled"] = scale_temp(df["ideal_temp"])
    df["temp_error_scaled"] = ((df["room_temp"] - df["setpoint"]).clip(-20, 20) + 20) / 40.0
    df["comfort_error_scaled"] = ((df["room_temp"] - df["ideal_temp"]).clip(-20, 20) + 20) / 40.0
    df["pir_motion"] = df["pir_motion"].clip(0, 1).astype("float32")
    df["has_guest"] = df["guest_id"].notna().astype("float32")
    df["hour_sin"], df["hour_cos"] = cyclic(df["timestamp"].dt.hour, 24)
    df["dow_sin"], df["dow_cos"] = cyclic(df["timestamp"].dt.dayofweek, 7)
    for col in BASE_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")

    df["recommended_setpoint"] = energy_aware_recommended_setpoints(df)
    df["target_scaled"] = (
        (df["recommended_setpoint"] - SETPOINT_MIN) / (SETPOINT_MAX - SETPOINT_MIN)
    ).astype("float32")
    return df.reset_index(drop=True)


def encode_features(df: pd.DataFrame) -> tuple[np.ndarray, dict[str, list[str]]]:
    numeric = df[BASE_NUMERIC_COLS].to_numpy(dtype=np.float32)
    category_values = {
        col: sorted(set(df[col].astype(str).tolist()) | {"Unknown"})
        for col in CATEGORICAL_COLS
    }
    encoded_parts = [numeric]
    for col in CATEGORICAL_COLS:
        categories = category_values[col]
        mapping = {value: idx for idx, value in enumerate(categories)}
        values = df[col].astype(str).map(lambda x: x if x in mapping else "Unknown").map(mapping).to_numpy()
        one_hot = np.zeros((len(df), len(categories)), dtype=np.float32)
        one_hot[np.arange(len(df)), values] = 1.0
        encoded_parts.append(one_hot)
    return np.concatenate(encoded_parts, axis=1), category_values


def build_sequences(
    df: pd.DataFrame,
    encoded: np.ndarray,
    seq_len: int,
    stride: int,
    max_sequences: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, end_indices = [], [], []
    for _, group in df.groupby("room_number", sort=False):
        idx = group.index.to_numpy()
        if len(idx) < seq_len:
            continue
        for end in range(seq_len - 1, len(idx), stride):
            window = idx[end - seq_len + 1:end + 1]
            x.append(encoded[window])
            y.append(float(df.at[idx[end], "target_scaled"]))
            end_indices.append(idx[end])
            if max_sequences is not None and len(x) >= max_sequences:
                return (
                    np.asarray(x, dtype=np.float32),
                    np.asarray(y, dtype=np.float32),
                    np.asarray(end_indices, dtype=np.int64),
                )
    if not x:
        raise ValueError("No sequences were built. Reduce --sequence-length or check the data.")
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(end_indices, dtype=np.int64),
    )


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


class TempretureRecomendationTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
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


def run_epoch(model, loader, criterion, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    total_rows = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = criterion(pred, yb)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(xb)
        total_rows += len(xb)
    return total_loss / max(total_rows, 1)


def add_energy_diagnostics(frame: pd.DataFrame, recommended: np.ndarray) -> pd.DataFrame:
    rows = []
    for (_, row), rec in zip(frame.iterrows(), recommended):
        rows.append(target_diagnostics(row, float(rec)))
    return pd.DataFrame(rows, index=frame.index)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_file = args.output_dir / "tempreture_recomendation_transformer.pt"
    metadata_file = args.output_dir / "tempreture_recomendation_transformer_metadata.json"
    report_file = args.output_dir / "tempreture_recomendation_transformer_report.txt"
    predictions_file = args.output_dir / "tempreture_recomendation_transformer_sample_predictions.csv"

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    df = build_frame(args.max_rows)
    encoded, category_values = encode_features(df)
    x, y, end_indices = build_sequences(
        df,
        encoded,
        seq_len=args.sequence_length,
        stride=args.stride,
        max_sequences=args.max_sequences,
    )

    n = len(x)
    train_end = int(n * 0.80)
    val_end = int(n * 0.90)
    x_train, y_train = x[:train_end], y[:train_end]
    x_val, y_val = x[train_end:val_end], y[train_end:val_end]
    x_test, y_test = x[val_end:], y[val_end:]
    test_indices = end_indices[val_end:]
    test_rows = df.loc[test_indices].copy().reset_index(drop=True)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TempretureRecomendationTransformer(
        input_dim=x_train.shape[-1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history = []
    best_state = None
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:02d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    predictions = []
    model.eval()
    with torch.no_grad():
        for xb, _ in test_loader:
            pred = model(xb.to(device)).detach().cpu().numpy()
            predictions.extend(pred.tolist())

    pred_scaled = np.asarray(predictions, dtype=np.float32)
    pred_setpoints = np.clip(pred_scaled * (SETPOINT_MAX - SETPOINT_MIN) + SETPOINT_MIN, SETPOINT_MIN, SETPOINT_MAX)
    pred_setpoints = np.round(pred_setpoints * 2) / 2
    target_setpoints = y_test * (SETPOINT_MAX - SETPOINT_MIN) + SETPOINT_MIN
    target_setpoints = np.round(target_setpoints * 2) / 2

    mae = mean_absolute_error(target_setpoints, pred_setpoints)
    rmse = float(np.sqrt(mean_squared_error(target_setpoints, pred_setpoints)))
    r2 = r2_score(target_setpoints, pred_setpoints)
    target_diag = add_energy_diagnostics(test_rows, target_setpoints)
    model_diag = add_energy_diagnostics(test_rows, pred_setpoints)
    current_energy_proxy = float(target_diag["current_power_w"].sum())
    target_energy_proxy = float(target_diag["target_power_w"].sum())
    model_energy_proxy = float(model_diag["target_power_w"].sum())
    target_saving_pct = 100 * (1 - target_energy_proxy / current_energy_proxy) if current_energy_proxy else 0.0
    model_saving_pct = 100 * (1 - model_energy_proxy / current_energy_proxy) if current_energy_proxy else 0.0
    target_comfort_gap = float(target_diag["target_comfort_gap_c"].mean())
    model_comfort_gap = float(model_diag["target_comfort_gap_c"].mean())

    sample = test_rows[
        [
            "timestamp",
            "room_number",
            "room_temp",
            "setpoint",
            "ideal_temp",
            "outside_temp",
            "hvac_mode",
            "occupancy_prediction",
            "temperature_persona_prediction",
        ]
    ].copy()
    sample["recommended_target"] = target_setpoints
    sample["model_prediction"] = pred_setpoints
    sample["target_mode"] = target_diag["target_mode"].to_numpy()
    sample["current_power_w"] = target_diag["current_power_w"].to_numpy()
    sample["target_power_w"] = target_diag["target_power_w"].to_numpy()
    sample["model_power_w"] = model_diag["target_power_w"].to_numpy()
    sample.head(5000).to_csv(predictions_file, index=False)

    metadata = {
        "model_type": "transformer_encoder",
        "task": "tempreture_recomendation",
        "target": "recommended_setpoint",
        "target_strategy": "energy_aware_comfort_constrained",
        "setpoint_min": SETPOINT_MIN,
        "setpoint_max": SETPOINT_MAX,
        "sequence_length": args.sequence_length,
        "input_dim": int(x_train.shape[-1]),
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "base_numeric_cols": BASE_NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "category_values": category_values,
        "history": history,
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "target_saving_pct": float(target_saving_pct),
            "model_saving_pct": float(model_saving_pct),
            "target_mean_comfort_gap_c": float(target_comfort_gap),
            "model_mean_comfort_gap_c": float(model_comfort_gap),
        },
        "max_rows": args.max_rows,
        "max_sequences": args.max_sequences,
        "stride": args.stride,
        "seed": SEED,
    }

    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, model_file)
    metadata_file.write_text(json.dumps(metadata, indent=2))

    report = "\n".join(
        [
            f"rows: {len(df):,}",
            f"sequences: {len(x):,}",
            f"train sequences: {len(x_train):,}",
            f"val sequences: {len(x_val):,}",
            f"test sequences: {len(x_test):,}",
            "algorithm: TransformerEncoder",
            "target strategy: energy-aware comfort-constrained setpoint",
            f"MAE: {mae:.3f}",
            f"RMSE: {rmse:.3f}",
            f"R2: {r2:.3f}",
            f"current energy proxy W-sum: {current_energy_proxy:.0f}",
            f"target energy proxy W-sum: {target_energy_proxy:.0f}",
            f"model energy proxy W-sum: {model_energy_proxy:.0f}",
            f"target saving vs current: {target_saving_pct:.2f}%",
            f"model saving vs current: {model_saving_pct:.2f}%",
            f"target mean comfort gap C: {target_comfort_gap:.3f}",
            f"model mean comfort gap C: {model_comfort_gap:.3f}",
        ]
    )
    report_file.write_text(report)
    print(report)
    print(f"\nsaved {report_file}")
    print(f"saved {predictions_file}")
    print(f"saved {model_file}")
    print(f"saved {metadata_file}")


if __name__ == "__main__":
    main()
