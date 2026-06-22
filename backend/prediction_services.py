"""Prediction helpers used by the visualization app.

These are lightweight serving-time predictors built from the same feature ideas
as the offline model scripts. They use recent room history from SQLite and
return class probabilities for the UI.
"""

from __future__ import annotations

from datetime import timedelta
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from backend.data_loader import get_db_connection
from backend.hvac_energy import HOURS_PER_SAMPLE as HVAC_HOURS_PER_SAMPLE
from backend.hvac_energy import _estimate_power as estimate_hvac_power


OCCUPANCY_CLASSES = ["Occupied", "Vacant", "Cleaning"]
PERSONA_CLASSES = ["Balanced", "Routine", "StaticBright", "StaticDim", "NightFocused", "Housekeeping", "Unknown"]
BASE_DIR = Path(__file__).resolve().parent.parent
LIGHTING_PERSONA_MODEL_FILE = (
    BASE_DIR / "AIModelsAndAlgorithms" / "LightingPersona" / "lighting_persona_model.joblib"
)
LIGHTING_PERSONA_TRANSFORMER_FILE = (
    BASE_DIR / "AIModelsAndAlgorithms" / "LightingPersona" / "transformer" / "lighting_persona_transformer.pt"
)
LIGHTING_PERSONA_TRANSFORMER_METADATA_FILE = (
    BASE_DIR / "AIModelsAndAlgorithms" / "LightingPersona" / "transformer" / "lighting_persona_transformer_metadata.json"
)
OCCUPANCY_MODEL_FILE = (
    BASE_DIR / "AIModelsAndAlgorithms" / "OccupancyPrediction" / "occupancy_model.joblib"
)
OCCUPANCY_TRANSFORMER_FILE = (
    BASE_DIR / "AIModelsAndAlgorithms" / "OccupancyPrediction" / "trandformer" / "occupancy_transformer.pt"
)
OCCUPANCY_TRANSFORMER_METADATA_FILE = (
    BASE_DIR / "AIModelsAndAlgorithms" / "OccupancyPrediction" / "trandformer" / "occupancy_transformer_metadata.json"
)
TEMPRETURE_PERSONA_TRANSFORMER_FILE = (
    BASE_DIR
    / "AIModelsAndAlgorithms"
    / "TempreturePersona"
    / "transformer"
    / "tempreture_persona_transformer.pt"
)
TEMPRETURE_PERSONA_TRANSFORMER_METADATA_FILE = (
    BASE_DIR
    / "AIModelsAndAlgorithms"
    / "TempreturePersona"
    / "transformer"
    / "tempreture_persona_transformer_metadata.json"
)
TEMPRETURE_RECOMENDATION_TRANSFORMER_FILE = (
    BASE_DIR
    / "AIModelsAndAlgorithms"
    / "TempretureRecomendation"
    / "transformer"
    / "tempreture_recomendation_transformer.pt"
)
TEMPRETURE_RECOMENDATION_TRANSFORMER_METADATA_FILE = (
    BASE_DIR
    / "AIModelsAndAlgorithms"
    / "TempretureRecomendation"
    / "transformer"
    / "tempreture_recomendation_transformer_metadata.json"
)
TEMPRETURE_RECOMENDATION_HGB_FILE = (
    BASE_DIR
    / "AIModelsAndAlgorithms"
    / "TempretureRecomendation"
    / "tempreture_recomendation_hgb_model.joblib"
)
_LIGHTING_PERSONA_MODEL_CACHE: dict | None = None
_LIGHTING_PERSONA_MODEL_MTIME: float | None = None
_LIGHTING_PERSONA_TRANSFORMER_CACHE: dict | None = None
_LIGHTING_PERSONA_TRANSFORMER_MTIME: float | None = None
_OCCUPANCY_MODEL_CACHE: dict | None = None
_OCCUPANCY_MODEL_MTIME: float | None = None
_OCCUPANCY_TRANSFORMER_CACHE: dict | None = None
_OCCUPANCY_TRANSFORMER_MTIME: float | None = None
_TEMPRETURE_PERSONA_TRANSFORMER_CACHE: dict | None = None
_TEMPRETURE_PERSONA_TRANSFORMER_MTIME: float | None = None
_TEMPRETURE_RECOMENDATION_TRANSFORMER_CACHE: dict | None = None
_TEMPRETURE_RECOMENDATION_TRANSFORMER_MTIME: float | None = None
_TEMPRETURE_RECOMENDATION_HGB_CACHE: dict | None = None
_TEMPRETURE_RECOMENDATION_HGB_MTIME: float | None = None


def _normalise_scores(scores: dict[str, float], classes: list[str]) -> dict[str, float]:
    total = sum(max(float(scores.get(c, 0.0)), 0.0) for c in classes)
    if total <= 0:
        return {c: 1.0 / len(classes) for c in classes}
    return {c: max(float(scores.get(c, 0.0)), 0.0) / total for c in classes}


def _top_class(scores: dict[str, float]) -> tuple[str, float]:
    label, prob = max(scores.items(), key=lambda item: item[1])
    return label, float(prob)


def _load_lighting_persona_model() -> dict | None:
    global _LIGHTING_PERSONA_MODEL_CACHE, _LIGHTING_PERSONA_MODEL_MTIME

    if not LIGHTING_PERSONA_MODEL_FILE.exists():
        return None

    model_mtime = LIGHTING_PERSONA_MODEL_FILE.stat().st_mtime
    if _LIGHTING_PERSONA_MODEL_CACHE is None or _LIGHTING_PERSONA_MODEL_MTIME != model_mtime:
        _LIGHTING_PERSONA_MODEL_CACHE = joblib.load(LIGHTING_PERSONA_MODEL_FILE)
        _LIGHTING_PERSONA_MODEL_MTIME = model_mtime
    return _LIGHTING_PERSONA_MODEL_CACHE


def _load_lighting_persona_transformer() -> dict | None:
    global _LIGHTING_PERSONA_TRANSFORMER_CACHE, _LIGHTING_PERSONA_TRANSFORMER_MTIME

    if not LIGHTING_PERSONA_TRANSFORMER_FILE.exists():
        return None
    if not LIGHTING_PERSONA_TRANSFORMER_METADATA_FILE.exists():
        return None

    model_mtime = max(
        LIGHTING_PERSONA_TRANSFORMER_FILE.stat().st_mtime,
        LIGHTING_PERSONA_TRANSFORMER_METADATA_FILE.stat().st_mtime,
    )
    if (
        _LIGHTING_PERSONA_TRANSFORMER_CACHE is not None
        and _LIGHTING_PERSONA_TRANSFORMER_MTIME == model_mtime
    ):
        return _LIGHTING_PERSONA_TRANSFORMER_CACHE

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to use the Transformer persona model.") from exc

    class LightingPersonaTransformer(nn.Module):
        def __init__(
            self,
            input_dim: int,
            n_classes: int,
            seq_len: int,
            d_model: int,
            n_heads: int,
            n_layers: int,
            dim_feedforward: int,
            dropout: float,
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

        def forward(self, x):
            hidden = self.input_proj(x) + self.pos_embed
            hidden = self.encoder(hidden)
            hidden = self.norm(hidden)
            pooled = hidden.mean(dim=1)
            return self.head(pooled)

    metadata = json.loads(LIGHTING_PERSONA_TRANSFORMER_METADATA_FILE.read_text())
    checkpoint = torch.load(LIGHTING_PERSONA_TRANSFORMER_FILE, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or metadata.get("config") or {}
    classes = checkpoint.get("classes") or metadata["classes"]
    feature_names = checkpoint.get("feature_names") or metadata["feature_names"]
    seq_len = int(checkpoint.get("seq_len") or metadata["seq_len"])
    input_dim = int(checkpoint.get("input_dim") or metadata["input_dim"])
    model = LightingPersonaTransformer(
        input_dim=input_dim,
        n_classes=len(classes),
        seq_len=seq_len,
        d_model=int(config.get("d_model", 96)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 3)),
        dim_feedforward=int(config.get("dim_feedforward", 192)),
        dropout=float(config.get("dropout", 0.15)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _LIGHTING_PERSONA_TRANSFORMER_CACHE = {
        "model": model,
        "classes": [str(c) for c in classes],
        "feature_names": list(feature_names),
        "seq_len": seq_len,
        "torch": torch,
    }
    _LIGHTING_PERSONA_TRANSFORMER_MTIME = model_mtime
    return _LIGHTING_PERSONA_TRANSFORMER_CACHE


def _resolve_persona_model_type(payload: dict) -> str:
    requested = (
        payload.get("lighting_persona_model_type")
        or payload.get("persona_model_type")
        or os.environ.get("LIGHTING_PERSONA_MODEL_TYPE")
        or "random_forest"
    )
    requested = str(requested).strip().lower().replace("-", "_")
    if requested in {"rf", "randomforest", "random_forest"}:
        return "random_forest"
    if requested in {"transformer", "torch"}:
        return "transformer"
    if requested == "auto":
        return "transformer" if LIGHTING_PERSONA_TRANSFORMER_FILE.exists() else "random_forest"
    return "random_forest"


def _load_occupancy_model() -> dict | None:
    global _OCCUPANCY_MODEL_CACHE, _OCCUPANCY_MODEL_MTIME

    if not OCCUPANCY_MODEL_FILE.exists():
        return None

    model_mtime = OCCUPANCY_MODEL_FILE.stat().st_mtime
    if _OCCUPANCY_MODEL_CACHE is None or _OCCUPANCY_MODEL_MTIME != model_mtime:
        _OCCUPANCY_MODEL_CACHE = joblib.load(OCCUPANCY_MODEL_FILE)
        _OCCUPANCY_MODEL_MTIME = model_mtime
    return _OCCUPANCY_MODEL_CACHE


def _load_occupancy_transformer() -> dict | None:
    global _OCCUPANCY_TRANSFORMER_CACHE, _OCCUPANCY_TRANSFORMER_MTIME

    if not OCCUPANCY_TRANSFORMER_FILE.exists():
        return None
    if not OCCUPANCY_TRANSFORMER_METADATA_FILE.exists():
        return None

    model_mtime = max(
        OCCUPANCY_TRANSFORMER_FILE.stat().st_mtime,
        OCCUPANCY_TRANSFORMER_METADATA_FILE.stat().st_mtime,
    )
    if _OCCUPANCY_TRANSFORMER_CACHE is not None and _OCCUPANCY_TRANSFORMER_MTIME == model_mtime:
        return _OCCUPANCY_TRANSFORMER_CACHE

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to use the Transformer occupancy model.") from exc

    class OccupancyTransformer(nn.Module):
        def __init__(
            self,
            input_dim: int,
            n_classes: int,
            seq_len: int,
            d_model: int,
            n_heads: int,
            n_layers: int,
            dim_feedforward: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_dim, d_model)
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
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
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )

        def forward(self, x):
            hidden = self.input_proj(x) + self.pos_embed
            hidden = self.encoder(hidden)
            hidden = self.norm(hidden)
            pooled = hidden.mean(dim=1)
            return self.head(pooled)

    metadata = json.loads(OCCUPANCY_TRANSFORMER_METADATA_FILE.read_text())
    checkpoint = torch.load(OCCUPANCY_TRANSFORMER_FILE, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or metadata.get("config") or {}
    classes = checkpoint.get("classes") or metadata["classes"]
    feature_columns = checkpoint.get("feature_columns") or metadata["feature_columns"]
    seq_len = int(checkpoint.get("seq_len") or metadata["seq_len"])
    input_dim = int(checkpoint.get("input_dim") or metadata["input_dim"])
    model = OccupancyTransformer(
        input_dim=input_dim,
        n_classes=len(classes),
        seq_len=seq_len,
        d_model=int(config.get("d_model", 64)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 2)),
        dim_feedforward=int(config.get("dim_feedforward", 128)),
        dropout=float(config.get("dropout", 0.15)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _OCCUPANCY_TRANSFORMER_CACHE = {
        "model": model,
        "classes": [str(c) for c in classes],
        "feature_columns": list(feature_columns),
        "seq_len": seq_len,
        "horizon_minutes": int(checkpoint.get("horizon_minutes") or metadata.get("horizon_minutes", 60)),
        "room_types": list(checkpoint.get("room_types") or metadata.get("room_types", [])),
        "torch": torch,
    }
    _OCCUPANCY_TRANSFORMER_MTIME = model_mtime
    return _OCCUPANCY_TRANSFORMER_CACHE


def _resolve_occupancy_model_type(payload: dict) -> str:
    requested = (
        payload.get("occupancy_model_type")
        or os.environ.get("OCCUPANCY_MODEL_TYPE")
        or "random_forest"
    )
    requested = str(requested).strip().lower().replace("-", "_")
    if requested in {"rf", "randomforest", "random_forest"}:
        return "random_forest"
    if requested in {"transformer", "torch"}:
        return "transformer"
    if requested == "auto":
        return "transformer" if OCCUPANCY_TRANSFORMER_FILE.exists() else "random_forest"
    return "random_forest"


def _load_tempreture_persona_transformer() -> dict | None:
    global _TEMPRETURE_PERSONA_TRANSFORMER_CACHE, _TEMPRETURE_PERSONA_TRANSFORMER_MTIME

    if not TEMPRETURE_PERSONA_TRANSFORMER_FILE.exists():
        return None
    if not TEMPRETURE_PERSONA_TRANSFORMER_METADATA_FILE.exists():
        return None

    model_mtime = max(
        TEMPRETURE_PERSONA_TRANSFORMER_FILE.stat().st_mtime,
        TEMPRETURE_PERSONA_TRANSFORMER_METADATA_FILE.stat().st_mtime,
    )
    if (
        _TEMPRETURE_PERSONA_TRANSFORMER_CACHE is not None
        and _TEMPRETURE_PERSONA_TRANSFORMER_MTIME == model_mtime
    ):
        return _TEMPRETURE_PERSONA_TRANSFORMER_CACHE

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to use the Transformer temperature persona model.") from exc

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

    class TempreturePersonaTransformer(nn.Module):
        def __init__(
            self,
            input_dim: int,
            n_classes: int,
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
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )

        def forward(self, x):
            hidden = self.input_projection(x)
            hidden = self.position(hidden)
            hidden = self.encoder(hidden)
            pooled = hidden.mean(dim=1)
            return self.head(pooled)

    metadata = json.loads(TEMPRETURE_PERSONA_TRANSFORMER_METADATA_FILE.read_text())
    checkpoint = torch.load(TEMPRETURE_PERSONA_TRANSFORMER_FILE, map_location="cpu", weights_only=False)
    checkpoint_metadata = checkpoint.get("metadata") or {}
    config = {**metadata, **checkpoint_metadata}
    classes = [str(c) for c in config["classes"]]
    model = TempreturePersonaTransformer(
        input_dim=int(config["input_dim"]),
        n_classes=len(classes),
        d_model=int(config.get("d_model", 96)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 2)),
        dim_feedforward=int(config.get("dim_feedforward", 192)),
        dropout=float(config.get("dropout", 0.15)),
    )
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError("Temperature persona Transformer checkpoint does not contain a state_dict.")
    model.load_state_dict(state_dict)
    model.eval()

    _TEMPRETURE_PERSONA_TRANSFORMER_CACHE = {
        "model": model,
        "torch": torch,
        "classes": classes,
        "seq_len": int(config.get("sequence_length", 24)),
        "feature_names": list(config["feature_names"]),
        "base_numeric_cols": list(config["base_numeric_cols"]),
        "categorical_cols": list(config["categorical_cols"]),
        "category_values": dict(config["category_values"]),
        "metadata": config,
    }
    _TEMPRETURE_PERSONA_TRANSFORMER_MTIME = model_mtime
    return _TEMPRETURE_PERSONA_TRANSFORMER_CACHE


def _load_tempreture_recomendation_transformer() -> dict | None:
    global _TEMPRETURE_RECOMENDATION_TRANSFORMER_CACHE, _TEMPRETURE_RECOMENDATION_TRANSFORMER_MTIME

    if not TEMPRETURE_RECOMENDATION_TRANSFORMER_FILE.exists():
        return None
    if not TEMPRETURE_RECOMENDATION_TRANSFORMER_METADATA_FILE.exists():
        return None

    model_mtime = max(
        TEMPRETURE_RECOMENDATION_TRANSFORMER_FILE.stat().st_mtime,
        TEMPRETURE_RECOMENDATION_TRANSFORMER_METADATA_FILE.stat().st_mtime,
    )
    if (
        _TEMPRETURE_RECOMENDATION_TRANSFORMER_CACHE is not None
        and _TEMPRETURE_RECOMENDATION_TRANSFORMER_MTIME == model_mtime
    ):
        return _TEMPRETURE_RECOMENDATION_TRANSFORMER_CACHE

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to use the Transformer temperature recommendation model.") from exc

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
            last_token = hidden[:, -1, :]
            return torch.sigmoid(self.head(last_token)).squeeze(-1)

    metadata = json.loads(TEMPRETURE_RECOMENDATION_TRANSFORMER_METADATA_FILE.read_text())
    checkpoint = torch.load(
        TEMPRETURE_RECOMENDATION_TRANSFORMER_FILE,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_metadata = checkpoint.get("metadata") or {}
    config = {**metadata, **checkpoint_metadata}
    model = TempretureRecomendationTransformer(
        input_dim=int(config["input_dim"]),
        d_model=int(config.get("d_model", 96)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 2)),
        dim_feedforward=int(config.get("dim_feedforward", 192)),
        dropout=float(config.get("dropout", 0.15)),
    )
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError("Temperature recommendation Transformer checkpoint does not contain a state_dict.")
    model.load_state_dict(state_dict)
    model.eval()

    _TEMPRETURE_RECOMENDATION_TRANSFORMER_CACHE = {
        "model": model,
        "torch": torch,
        "seq_len": int(config.get("sequence_length", 24)),
        "setpoint_min": float(config.get("setpoint_min", 16.0)),
        "setpoint_max": float(config.get("setpoint_max", 28.0)),
        "base_numeric_cols": list(config["base_numeric_cols"]),
        "categorical_cols": list(config["categorical_cols"]),
        "category_values": dict(config["category_values"]),
        "metadata": config,
    }
    _TEMPRETURE_RECOMENDATION_TRANSFORMER_MTIME = model_mtime
    return _TEMPRETURE_RECOMENDATION_TRANSFORMER_CACHE


def _load_tempreture_recomendation_hgb() -> dict | None:
    global _TEMPRETURE_RECOMENDATION_HGB_CACHE, _TEMPRETURE_RECOMENDATION_HGB_MTIME

    if not TEMPRETURE_RECOMENDATION_HGB_FILE.exists():
        return None

    model_mtime = TEMPRETURE_RECOMENDATION_HGB_FILE.stat().st_mtime
    if (
        _TEMPRETURE_RECOMENDATION_HGB_CACHE is None
        or _TEMPRETURE_RECOMENDATION_HGB_MTIME != model_mtime
    ):
        _TEMPRETURE_RECOMENDATION_HGB_CACHE = joblib.load(TEMPRETURE_RECOMENDATION_HGB_FILE)
        _TEMPRETURE_RECOMENDATION_HGB_MTIME = model_mtime
    return _TEMPRETURE_RECOMENDATION_HGB_CACHE


def _resolve_tempreture_recomendation_model_type(payload: dict) -> str:
    requested = (
        payload.get("temperature_recommendation_model_type")
        or payload.get("tempreture_recomendation_model_type")
        or payload.get("temperature_recomendation_model_type")
        or os.environ.get("TEMPRETURE_RECOMENDATION_MODEL_TYPE")
        or "transformer"
    )
    requested = str(requested).strip().lower().replace("-", "_")
    if requested in {"hgb", "hist_gradient_boosting", "histgradientboosting", "hist_gradient_boosting_regressor"}:
        return "hist_gradient_boosting"
    if requested in {"transformer", "torch"}:
        return "transformer"
    if requested == "auto":
        return "hist_gradient_boosting" if TEMPRETURE_RECOMENDATION_HGB_FILE.exists() else "transformer"
    return "transformer"


def _cyclic_value(value: int, period: int) -> tuple[float, float]:
    radians = 2 * math.pi * value / period
    return math.sin(radians), math.cos(radians)


def _reservation_features(guest_id) -> dict:
    if pd.isna(guest_id):
        return {
            "Nationality": "No active reservation",
            "room_type": "Unknown",
            "Total Nights": 0.0,
            "Total Amount": 0.0,
            "Stay_Duration": 0.0,
            "price_per_night": 0.0,
        }

    conn = get_db_connection()
    try:
        reservation = pd.read_sql_query(
            """
            SELECT nationality AS "Nationality", room_type,
                   total_nights AS "Total Nights",
                   total_amount AS "Total Amount",
                   stay_duration AS "Stay_Duration"
            FROM hotel_reservations
            WHERE guest_id = ?
            LIMIT 1
            """,
            conn,
            params=(int(guest_id),),
        )
    finally:
        conn.close()

    if reservation.empty:
        return {
            "Nationality": "No active reservation",
            "room_type": "Unknown",
            "Total Nights": 0.0,
            "Total Amount": 0.0,
            "Stay_Duration": 0.0,
            "price_per_night": 0.0,
        }

    row = reservation.iloc[0]
    stay_duration = float(row.get("Stay_Duration") or 0.0)
    total_amount = float(row.get("Total Amount") or 0.0)
    return {
        "Nationality": row.get("Nationality") or "No active reservation",
        "room_type": row.get("room_type") or "Unknown",
        "Total Nights": float(row.get("Total Nights") or 0.0),
        "Total Amount": total_amount,
        "Stay_Duration": stay_duration,
        "price_per_night": total_amount / stay_duration if stay_duration else 0.0,
    }


def _build_occupancy_feature_row(df: pd.DataFrame, room_number: int) -> tuple[dict, dict]:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["pir_motion"] = pd.to_numeric(df["pir_motion"], errors="coerce").fillna(0)
    df["adults"] = pd.to_numeric(df["adults"], errors="coerce").fillna(0)
    df["children"] = pd.to_numeric(df["children"], errors="coerce").fillna(0)
    df["guest_id"] = pd.to_numeric(df["guest_id"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    latest = df.iloc[-1]
    recent = df.tail(12)
    hour = int(latest["timestamp"].hour)
    dayofweek = int(latest["timestamp"].dayofweek)
    hour_sin, hour_cos = _cyclic_value(hour, 24)
    dow_sin, dow_cos = _cyclic_value(dayofweek, 7)
    adults = float(latest.get("adults") or 0.0)
    children = float(latest.get("children") or 0.0)
    guest_id = latest.get("guest_id")
    active_guest = int(pd.notna(guest_id))
    reservation = _reservation_features(guest_id)

    feature_row = {
        "room_number": room_number,
        "pir_motion": float(latest.get("pir_motion") or 0.0),
        "motion_mean_1h": float(recent["pir_motion"].mean()) if not recent.empty else 0.0,
        "motion_sum_1h": float(recent["pir_motion"].sum()) if not recent.empty else 0.0,
        "adults": adults,
        "children": children,
        "guest_count": adults + children,
        "active_guest": active_guest,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        **reservation,
    }
    diagnostics = {
        "motion_rate": feature_row["motion_mean_1h"],
        "motion_count": int(feature_row["motion_sum_1h"]),
        "active_guest": active_guest,
        "guest_count": feature_row["guest_count"],
        "latest_state": str(latest.get("room_state") or "Unknown"),
    }
    return feature_row, diagnostics


def _build_occupancy_transformer_sequence(
    df: pd.DataFrame,
    room_number: int,
    feature_columns: list[str],
    seq_len: int,
    room_types: list[str],
) -> tuple[np.ndarray, dict]:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["pir_motion"] = pd.to_numeric(df["pir_motion"], errors="coerce").fillna(0)
    df["adults"] = pd.to_numeric(df["adults"], errors="coerce").fillna(0)
    df["children"] = pd.to_numeric(df["children"], errors="coerce").fillna(0)
    df["guest_id"] = pd.to_numeric(df["guest_id"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").tail(seq_len)

    if df.empty:
        return np.zeros((seq_len, len(feature_columns)), dtype=np.float32), {
            "motion_rate": 0.0,
            "motion_count": 0,
            "active_guest": 0,
            "guest_count": 0.0,
            "latest_state": "Unknown",
        }

    rows = []
    for _, row in df.iterrows():
        hour_sin, hour_cos = _cyclic_value(int(row["timestamp"].hour), 24)
        dow_sin, dow_cos = _cyclic_value(int(row["timestamp"].dayofweek), 7)
        adults = float(row.get("adults") or 0.0)
        children = float(row.get("children") or 0.0)
        active_guest = int(pd.notna(row.get("guest_id")))
        reservation = _reservation_features(row.get("guest_id"))
        room_type = reservation.get("room_type") or "Unknown"

        feature_row = {
            "pir_motion": float(row.get("pir_motion") or 0.0),
            "adults": adults,
            "children": children,
            "guest_count": adults + children,
            "active_guest": active_guest,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "Total Nights": float(reservation.get("Total Nights") or 0.0) / 30.0,
            "Total Amount": float(reservation.get("Total Amount") or 0.0) / 2000.0,
            "Stay_Duration": float(reservation.get("Stay_Duration") or 0.0) / 30.0,
            "price_per_night": float(reservation.get("price_per_night") or 0.0) / 500.0,
        }
        for known_room_type in room_types:
            feature_row[f"room_type_{known_room_type}"] = 1.0 if room_type == known_room_type else 0.0
        rows.append(feature_row)

    matrix = np.zeros((seq_len, len(feature_columns)), dtype=np.float32)
    offset = seq_len - len(rows)
    for idx, feature_row in enumerate(rows):
        matrix[offset + idx] = [float(feature_row.get(col, 0.0)) for col in feature_columns]

    latest = df.iloc[-1]
    diagnostics = {
        "motion_rate": float(df["pir_motion"].mean()),
        "motion_count": int(df["pir_motion"].sum()),
        "active_guest": int(pd.notna(latest.get("guest_id"))),
        "guest_count": float((latest.get("adults") or 0) + (latest.get("children") or 0)),
        "latest_state": str(latest.get("room_state") or "Unknown"),
    }
    return matrix, diagnostics


def _build_lighting_persona_feature_row(df: pd.DataFrame, room_number: int) -> dict:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0).clip(0, 80)
    df["pir_motion"] = pd.to_numeric(df["pir_motion"], errors="coerce").fillna(0)
    df["n_occupants"] = pd.to_numeric(df["n_occupants"], errors="coerce").fillna(0)
    df["active_actors"] = pd.to_numeric(df["active_actors"], errors="coerce").fillna(0)
    df["floor"] = pd.to_numeric(df["floor"], errors="coerce").fillna(0)
    df = df.dropna(subset=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["is_on"] = df["value"].gt(0).astype(int)

    on_df = df[df["is_on"].eq(1)]
    lamp_counts = on_df["lamp_location"].value_counts(normalize=True).to_dict()
    hourly_means = df.groupby("hour")["value"].mean().to_dict()

    feature_row = {
        "room_number": room_number,
        "floor": float(df["floor"].iloc[0]) if not df.empty else 0.0,
        "samples": int(len(df)),
        "value_mean": float(df["value"].mean()) if not df.empty else 0.0,
        "value_std": float(df["value"].std()) if len(df) > 1 else 0.0,
        "value_max": float(df["value"].max()) if not df.empty else 0.0,
        "lit_ratio": float(df["is_on"].mean()) if not df.empty else 0.0,
        "pir_motion_rate": float(df["pir_motion"].mean()) if not df.empty else 0.0,
        "occupants_mean": float(df["n_occupants"].mean()) if not df.empty else 0.0,
        "occupants_max": float(df["n_occupants"].max()) if not df.empty else 0.0,
        "active_actors_mean": float(df["active_actors"].mean()) if not df.empty else 0.0,
    }
    for hour in range(24):
        feature_row[f"hour_{hour:02d}_value_mean"] = float(hourly_means.get(hour, 0.0))
    for lamp, share in lamp_counts.items():
        feature_row[f"lamp_ratio_{lamp}"] = float(share)
    return feature_row


def _build_lighting_persona_transformer_sequence(
    df: pd.DataFrame,
    timestamp: pd.Timestamp,
    feature_names: list[str],
    seq_len: int,
):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["slot"] = df["timestamp"].dt.hour * 12 + df["timestamp"].dt.minute // 5
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0).clip(0, 80) / 80.0
    for col in ["pir_motion", "n_occupants", "active_actors", "hurry_morning", "lazy_day", "forgetful"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Use the selected calendar day up to the selected timestamp. Future slots
    # stay zero so prediction does not read future behavior.
    day_start = timestamp.normalize()
    df = df[(df["timestamp"] >= day_start) & (df["timestamp"] <= timestamp)]

    matrix = np.zeros((seq_len, len(feature_names)), dtype=np.float32)
    feature_index = {name: idx for idx, name in enumerate(feature_names)}
    lamp_df = df[df["lamp_location"].ne("none")]
    if not lamp_df.empty:
        lamp_profile = lamp_df.groupby(["slot", "lamp_location"])["value"].mean()
        for (slot, lamp), value in lamp_profile.items():
            if 0 <= int(slot) < seq_len and lamp in feature_index:
                matrix[int(slot), feature_index[lamp]] = float(value)

    context_cols = ["pir_motion", "n_occupants", "active_actors", "hurry_morning", "lazy_day", "forgetful"]
    if not df.empty:
        context_profile = df.groupby("slot")[context_cols].mean()
        for slot, row in context_profile.iterrows():
            if 0 <= int(slot) < seq_len:
                for col in context_cols:
                    if col in feature_index:
                        matrix[int(slot), feature_index[col]] = float(row[col])

    slots = np.arange(seq_len)
    if "hour_sin" in feature_index:
        matrix[:, feature_index["hour_sin"]] = np.sin(2 * np.pi * (slots // 12) / 24)
    if "hour_cos" in feature_index:
        matrix[:, feature_index["hour_cos"]] = np.cos(2 * np.pi * (slots // 12) / 24)
    return matrix


def _scale_temp(value) -> float:
    numeric = float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0])
    numeric = min(max(numeric, -20.0), 50.0)
    return (numeric + 20.0) / 70.0


def _build_tempreture_persona_step(row: pd.Series, timestamp: pd.Timestamp) -> dict:
    room_temp = float(row.get("room_temp") or 0.0)
    setpoint = float(row.get("setpoint") or 0.0)
    ideal_temp = float(row.get("ideal_temp") or 0.0)
    hour_sin, hour_cos = _cyclic_value(int(timestamp.hour), 24)
    dow_sin, dow_cos = _cyclic_value(int(timestamp.dayofweek), 7)
    return {
        "floor_scaled": min(max(float(row.get("floor") or 0.0), 0.0), 30.0) / 30.0,
        "size_scaled": min(max(float(row.get("size_m2") or 0.0), 0.0), 120.0) / 120.0,
        "outside_temp_scaled": _scale_temp(row.get("outside_temp")),
        "room_temp_scaled": _scale_temp(room_temp),
        "setpoint_scaled": _scale_temp(setpoint),
        "ideal_temp_scaled": _scale_temp(ideal_temp),
        "temp_error_scaled": (min(max(room_temp - setpoint, -20.0), 20.0) + 20.0) / 40.0,
        "comfort_error_scaled": (min(max(room_temp - ideal_temp, -20.0), 20.0) + 20.0) / 40.0,
        "pir_motion": min(max(float(row.get("pir_motion") or 0.0), 0.0), 1.0),
        "has_guest": 1.0 if pd.notna(row.get("guest_id")) else 0.0,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "facade": str(row.get("facade") or "Unknown"),
        "room_type": str(row.get("room_type") or "Unknown"),
        "hvac_mode": str(row.get("hvac_mode") or "Unknown"),
        "occupant_state": str(row.get("occupant_state") or "Unknown"),
        "pir_persona": str(row.get("pir_persona") or "Unknown"),
        "room_state": str(row.get("room_state") or "Unknown"),
    }


def _encode_tempreture_persona_step(step: dict, transformer_bundle: dict) -> list[float]:
    values = [float(step.get(col, 0.0)) for col in transformer_bundle["base_numeric_cols"]]
    for col in transformer_bundle["categorical_cols"]:
        categories = transformer_bundle["category_values"].get(col, [])
        raw_value = str(step.get(col) or "Unknown")
        if raw_value not in categories:
            raw_value = "Unknown" if "Unknown" in categories else (categories[0] if categories else "")
        values.extend(1.0 if category == raw_value else 0.0 for category in categories)
    return values


def _build_tempreture_persona_transformer_sequence(
    df: pd.DataFrame,
    timestamp: pd.Timestamp,
    transformer_bundle: dict,
) -> tuple[np.ndarray, dict]:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ["floor", "size_m2", "outside_temp", "room_temp", "setpoint", "ideal_temp", "pir_motion"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["guest_id"] = pd.to_numeric(df["guest_id"], errors="coerce")

    seq_len = transformer_bundle["seq_len"]
    history = df.tail(seq_len)
    if history.empty:
        synthetic = pd.Series({"timestamp": timestamp})
        history_rows = [synthetic] * seq_len
    else:
        history_rows = [row for _, row in history.iterrows()]
        if len(history_rows) < seq_len:
            history_rows = [history_rows[0]] * (seq_len - len(history_rows)) + history_rows

    encoded_steps = []
    for row in history_rows:
        row_timestamp = pd.Timestamp(row.get("timestamp")) if pd.notna(row.get("timestamp")) else timestamp
        encoded_steps.append(
            _encode_tempreture_persona_step(
                _build_tempreture_persona_step(row, row_timestamp),
                transformer_bundle,
            )
        )

    latest = df.iloc[-1] if not df.empty else pd.Series(dtype=object)
    room_temp = float(latest.get("room_temp") or 0.0)
    setpoint = float(latest.get("setpoint") or 0.0)
    ideal_temp = float(latest.get("ideal_temp") or 0.0)
    diagnostics = {
        "room_temp": room_temp,
        "setpoint": setpoint,
        "ideal_temp": ideal_temp,
        "outside_temp": float(latest.get("outside_temp") or 0.0),
        "temp_error": room_temp - setpoint,
        "comfort_error": room_temp - ideal_temp,
        "hvac_mode": str(latest.get("hvac_mode") or "Unknown"),
        "room_state": str(latest.get("room_state") or "Unknown"),
        "occupant_state": str(latest.get("occupant_state") or "Unknown"),
        "motion_rate": float(df["pir_motion"].mean()) if not df.empty else 0.0,
        "has_guest": int(pd.notna(latest.get("guest_id"))) if not latest.empty else 0,
        "samples": int(len(df)),
    }
    return np.array(encoded_steps, dtype=np.float32), diagnostics


def _build_tempreture_recomendation_step(
    row: pd.Series,
    timestamp: pd.Timestamp,
    occupancy: str,
    persona: str,
) -> dict:
    step = _build_tempreture_persona_step(row, timestamp)
    step["occupancy_prediction"] = occupancy
    step["temperature_persona_prediction"] = persona
    return step


def _encode_tempreture_recomendation_step(step: dict, transformer_bundle: dict) -> list[float]:
    values = [float(step.get(col, 0.0)) for col in transformer_bundle["base_numeric_cols"]]
    for col in transformer_bundle["categorical_cols"]:
        categories = transformer_bundle["category_values"].get(col, [])
        raw_value = str(step.get(col) or "Unknown")
        if raw_value not in categories:
            raw_value = "Unknown" if "Unknown" in categories else (categories[0] if categories else "")
        values.extend(1.0 if category == raw_value else 0.0 for category in categories)
    return values


def _build_tempreture_recomendation_transformer_sequence(
    df: pd.DataFrame,
    timestamp: pd.Timestamp,
    transformer_bundle: dict,
    occupancy: str,
    persona: str,
) -> tuple[np.ndarray, dict]:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ["floor", "size_m2", "outside_temp", "room_temp", "setpoint", "ideal_temp", "pir_motion"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["guest_id"] = pd.to_numeric(df["guest_id"], errors="coerce")

    seq_len = transformer_bundle["seq_len"]
    history = df.tail(seq_len)
    if history.empty:
        synthetic = pd.Series({"timestamp": timestamp})
        history_rows = [synthetic] * seq_len
    else:
        history_rows = [row for _, row in history.iterrows()]
        if len(history_rows) < seq_len:
            history_rows = [history_rows[0]] * (seq_len - len(history_rows)) + history_rows

    encoded_steps = []
    for row in history_rows:
        row_timestamp = pd.Timestamp(row.get("timestamp")) if pd.notna(row.get("timestamp")) else timestamp
        encoded_steps.append(
            _encode_tempreture_recomendation_step(
                _build_tempreture_recomendation_step(row, row_timestamp, occupancy, persona),
                transformer_bundle,
            )
        )

    latest = df.iloc[-1] if not df.empty else pd.Series(dtype=object)
    room_temp = float(latest.get("room_temp") or 0.0)
    setpoint = float(latest.get("setpoint") or 0.0)
    ideal_temp = float(latest.get("ideal_temp") or 0.0)
    outside_temp = float(latest.get("outside_temp") or 0.0)
    hvac_mode = str(latest.get("hvac_mode") or "Unknown")
    diagnostics = {
        "room_temp": room_temp,
        "setpoint": setpoint,
        "ideal_temp": ideal_temp,
        "outside_temp": outside_temp,
        "temp_error": room_temp - setpoint,
        "comfort_error": room_temp - ideal_temp,
        "hvac_mode": hvac_mode,
        "room_state": str(latest.get("room_state") or "Unknown"),
        "occupant_state": str(latest.get("occupant_state") or "Unknown"),
        "motion_rate": float(df["pir_motion"].mean()) if not df.empty else 0.0,
        "has_guest": int(pd.notna(latest.get("guest_id"))) if not latest.empty else 0,
        "samples": int(len(df)),
    }
    return np.array(encoded_steps, dtype=np.float32), diagnostics


def _build_tempreture_recomendation_hgb_frame(
    latest_row: pd.Series,
    timestamp: pd.Timestamp,
    model_bundle: dict,
    occupancy: str,
    persona: str,
) -> pd.DataFrame:
    step = _build_tempreture_recomendation_step(latest_row, timestamp, occupancy, persona)
    feature_columns = list(model_bundle.get("feature_columns") or [])
    return pd.DataFrame([{col: step.get(col, 0.0) for col in feature_columns}])


def _infer_hvac_recomendation_mode(room_temp: float, setpoint: float, outside_temp: float, hvac_mode: str) -> str:
    mode = (hvac_mode or "").strip().lower()
    if mode in {"cooling", "heating"}:
        return mode
    if room_temp > setpoint + 0.4 or outside_temp >= 27:
        return "cooling"
    if room_temp < setpoint - 0.4 or outside_temp <= 12:
        return "heating"
    return "idle"


def predict_occupancy(payload: dict) -> dict:
    room_number = int(payload.get("room_number"))
    timestamp = pd.Timestamp(payload.get("timestamp"))
    lookback_hours = int(payload.get("lookback_hours") or 1)
    horizon_minutes = int(payload.get("horizon_minutes") or 60)
    lookback_start = timestamp - timedelta(hours=lookback_hours)
    model_type = _resolve_occupancy_model_type(payload)

    conn = get_db_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp, pir_motion, room_state, adults, children, guest_id
            FROM pir_sensor_data
            WHERE room_number = ?
              AND timestamp >= ?
              AND timestamp <= ?
            ORDER BY timestamp
            """,
            conn,
            params=(
                room_number,
                lookback_start.strftime("%Y-%m-%d %H:%M:%S"),
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
    finally:
        conn.close()

    if df.empty:
        scores = {"Occupied": 0.30, "Vacant": 0.65, "Cleaning": 0.05}
        prediction, confidence = _top_class(scores)
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": scores,
            "features": {"motion_rate": 0.0, "motion_count": 0, "active_guest": 0, "latest_state": "Unknown"},
            "message": "No recent PIR history found; returned conservative default.",
        }

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["pir_motion"] = pd.to_numeric(df["pir_motion"], errors="coerce").fillna(0)
    df = df.dropna(subset=["timestamp"])
    latest = df.iloc[-1]
    motion_rate = float(df["pir_motion"].mean())
    motion_count = int(df["pir_motion"].sum())
    active_guest = int(pd.notna(latest.get("guest_id")))
    guest_count = float((latest.get("adults") or 0) + (latest.get("children") or 0))
    latest_state = str(latest.get("room_state") or "Unknown")

    if model_type == "transformer":
        transformer_bundle = _load_occupancy_transformer()
        if transformer_bundle is None:
            raise RuntimeError("Transformer occupancy model files were not found.")
        torch = transformer_bundle["torch"]
        sequence, diagnostics = _build_occupancy_transformer_sequence(
            df,
            room_number,
            transformer_bundle["feature_columns"],
            transformer_bundle["seq_len"],
            transformer_bundle["room_types"],
        )
        with torch.no_grad():
            logits = transformer_bundle["model"](torch.from_numpy(sequence).unsqueeze(0))
            proba = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        probabilities = {label: 0.0 for label in OCCUPANCY_CLASSES}
        for label, value in zip(transformer_bundle["classes"], proba):
            probabilities[label] = float(value)
        probabilities = _normalise_scores(probabilities, OCCUPANCY_CLASSES)
        prediction, confidence = _top_class(probabilities)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "features": {
                **diagnostics,
                "lookback_hours": lookback_hours,
                "horizon_minutes": transformer_bundle.get("horizon_minutes", horizon_minutes),
                "model": "TransformerEncoder",
                "model_type": "transformer",
            },
        }

    model_bundle = _load_occupancy_model()
    if model_bundle:
        feature_columns = model_bundle["feature_columns"]
        feature_row, diagnostics = _build_occupancy_feature_row(df, room_number)
        features_df = pd.DataFrame([{col: feature_row.get(col, 0.0) for col in feature_columns}])
        model = model_bundle["model"]
        predicted = str(model.predict(features_df)[0])

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_df)[0]
            model_classes = [str(c) for c in model.classes_]
            probabilities = {label: 0.0 for label in OCCUPANCY_CLASSES}
            for label, value in zip(model_classes, proba):
                probabilities[label] = float(value)
            probabilities = _normalise_scores(probabilities, OCCUPANCY_CLASSES)
            prediction, confidence = _top_class(probabilities)
        else:
            probabilities = _normalise_scores({predicted: 1.0}, OCCUPANCY_CLASSES)
            prediction, confidence = predicted, probabilities.get(predicted, 0.0)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "features": {
                **diagnostics,
                "lookback_hours": lookback_hours,
                "horizon_minutes": model_bundle.get("horizon_minutes", horizon_minutes),
                "model": "RandomForestClassifier",
                "model_type": "random_forest",
            },
        }

    scores = {"Occupied": 0.20, "Vacant": 0.20, "Cleaning": 0.03}
    if latest_state in OCCUPANCY_CLASSES:
        scores[latest_state] += 0.42
    if active_guest or guest_count > 0:
        scores["Occupied"] += 0.28
        scores["Vacant"] -= 0.08
    if motion_rate > 0.12:
        scores["Occupied"] += 0.24
    elif motion_rate > 0.02:
        scores["Occupied"] += 0.12
    else:
        scores["Vacant"] += 0.22
    if latest_state == "Cleaning" or (motion_count >= 3 and not active_guest):
        scores["Cleaning"] += 0.22
    if latest_state == "Vacant" and motion_count == 0:
        scores["Vacant"] += 0.30

    probabilities = _normalise_scores(scores, OCCUPANCY_CLASSES)
    prediction, confidence = _top_class(probabilities)
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "features": {
            "motion_rate": motion_rate,
            "motion_count": motion_count,
            "active_guest": active_guest,
            "guest_count": guest_count,
            "latest_state": latest_state,
            "lookback_hours": lookback_hours,
            "horizon_minutes": horizon_minutes,
        },
    }


def predict_lighting_persona(payload: dict) -> dict:
    room_number = int(payload.get("room_number"))
    timestamp = pd.Timestamp(payload.get("timestamp"))
    lookback_hours = int(payload.get("lookback_hours") or 24)
    lookback_start = timestamp - timedelta(hours=lookback_hours)
    model_type = _resolve_persona_model_type(payload)

    conn = get_db_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp, room_number, floor, lamp_location, "Value" AS value, pir_motion,
                   n_occupants, active_actors, lightning_persona,
                   hurry_morning, lazy_day, forgetful
            FROM lightning_data
            WHERE room_number = ?
              AND timestamp >= ?
              AND timestamp <= ?
            ORDER BY timestamp
            """,
            conn,
            params=(
                room_number,
                lookback_start.strftime("%Y-%m-%d %H:%M:%S"),
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
    finally:
        conn.close()

    if df.empty:
        probabilities = _normalise_scores({"Unknown": 1.0}, PERSONA_CLASSES)
        prediction, confidence = _top_class(probabilities)
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "features": {"mean_on_level": 0.0, "lit_ratio": 0.0, "night_share": 0.0},
            "message": "No recent lighting history found.",
        }

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0).clip(0, 80)
    df = df.dropna(subset=["timestamp"])
    lamp_df = df[df["lamp_location"].ne("none")].copy()
    lamp_df["is_on"] = lamp_df["value"].gt(0)
    on_df = lamp_df[lamp_df["is_on"]]
    mean_on_level = float(on_df["value"].mean()) if not on_df.empty else 0.0
    lit_ratio = float(lamp_df["is_on"].mean()) if not lamp_df.empty else 0.0
    night_df = lamp_df[(lamp_df["timestamp"].dt.hour >= 22) | (lamp_df["timestamp"].dt.hour < 6)]
    day_df = lamp_df[(lamp_df["timestamp"].dt.hour >= 6) & (lamp_df["timestamp"].dt.hour < 22)]
    night_level = float(night_df["value"].mean()) if not night_df.empty else 0.0
    day_level = float(day_df["value"].mean()) if not day_df.empty else 0.0
    night_share = night_level / (night_level + day_level) if (night_level + day_level) else 0.0
    active_actors_mean = float(pd.to_numeric(df["active_actors"], errors="coerce").fillna(0).mean())

    if model_type == "transformer":
        transformer_bundle = _load_lighting_persona_transformer()
        if transformer_bundle is None:
            raise RuntimeError("Transformer persona model files were not found.")
        torch = transformer_bundle["torch"]
        sequence = _build_lighting_persona_transformer_sequence(
            df,
            timestamp,
            transformer_bundle["feature_names"],
            transformer_bundle["seq_len"],
        )
        with torch.no_grad():
            logits = transformer_bundle["model"](torch.from_numpy(sequence).unsqueeze(0))
            proba = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        probabilities = {label: 0.0 for label in PERSONA_CLASSES}
        for label, value in zip(transformer_bundle["classes"], proba):
            probabilities[label] = float(value)
        probabilities = _normalise_scores(probabilities, PERSONA_CLASSES)
        prediction, confidence = _top_class(probabilities)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "features": {
                "mean_on_level": mean_on_level,
                "lit_ratio": lit_ratio,
                "night_share": night_share,
                "night_level": night_level,
                "day_level": day_level,
                "active_actors_mean": active_actors_mean,
                "lookback_hours": lookback_hours,
                "model": "TransformerEncoder",
                "model_type": "transformer",
            },
        }

    model_bundle = _load_lighting_persona_model()
    if model_bundle:
        feature_columns = model_bundle["feature_columns"]
        feature_row = _build_lighting_persona_feature_row(lamp_df, room_number)
        features_df = pd.DataFrame([{col: feature_row.get(col, 0.0) for col in feature_columns}])
        model = model_bundle["model"]
        predicted = str(model.predict(features_df)[0])

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_df)[0]
            model_classes = [str(c) for c in model.classes_]
            probabilities = {label: 0.0 for label in PERSONA_CLASSES}
            for label, value in zip(model_classes, proba):
                probabilities[label] = float(value)
            probabilities = _normalise_scores(probabilities, PERSONA_CLASSES)
            prediction, confidence = _top_class(probabilities)
        else:
            probabilities = _normalise_scores({predicted: 1.0}, PERSONA_CLASSES)
            prediction, confidence = predicted, probabilities.get(predicted, 0.0)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "features": {
                "mean_on_level": mean_on_level,
                "lit_ratio": lit_ratio,
                "night_share": night_share,
                "night_level": night_level,
                "day_level": day_level,
                "active_actors_mean": active_actors_mean,
                "lookback_hours": lookback_hours,
                "model": "RandomForestClassifier",
                "model_type": "random_forest",
            },
        }

    persona_counts = lamp_df["lightning_persona"].dropna().value_counts(normalize=True).to_dict()

    scores = {c: 0.02 for c in PERSONA_CLASSES}
    for persona, share in persona_counts.items():
        if persona in scores:
            scores[persona] += float(share) * 0.35

    if mean_on_level >= 55:
        scores["StaticBright"] += 0.32
    elif mean_on_level <= 28 and lit_ratio > 0.05:
        scores["StaticDim"] += 0.30
    else:
        scores["Balanced"] += 0.18
        scores["Routine"] += 0.14

    if night_share > 0.58 and mean_on_level <= 50:
        scores["NightFocused"] += 0.30
    if 0.20 <= lit_ratio <= 0.75 and 25 <= mean_on_level <= 60:
        scores["Balanced"] += 0.20
    if active_actors_mean > 1.5 and lit_ratio > 0.35:
        scores["Routine"] += 0.12
    if mean_on_level >= 75 and lit_ratio > 0.40:
        scores["Housekeeping"] += 0.10

    probabilities = _normalise_scores(scores, PERSONA_CLASSES)
    prediction, confidence = _top_class(probabilities)
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "features": {
            "mean_on_level": mean_on_level,
            "lit_ratio": lit_ratio,
            "night_share": night_share,
            "night_level": night_level,
            "day_level": day_level,
            "active_actors_mean": active_actors_mean,
            "lookback_hours": lookback_hours,
        },
    }


def predict_tempreture_persona(payload: dict) -> dict:
    room_number = int(payload.get("room_number"))
    timestamp = pd.Timestamp(payload.get("timestamp"))
    lookback_hours = int(payload.get("lookback_hours") or 2)
    lookback_start = timestamp - timedelta(hours=lookback_hours)

    transformer_bundle = _load_tempreture_persona_transformer()
    if transformer_bundle is None:
        raise RuntimeError("Transformer temperature persona model files were not found.")

    conn = get_db_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp, room_number, floor, facade, room_type, size_m2,
                   outside_temp, room_temp, setpoint, ideal_temp, hvac_mode,
                   ac_persona, occupant_state, pir_persona, room_state,
                   pir_motion, guest_id
            FROM temperature_data
            WHERE room_number = ?
              AND timestamp >= ?
              AND timestamp <= ?
            ORDER BY timestamp
            """,
            conn,
            params=(
                room_number,
                lookback_start.strftime("%Y-%m-%d %H:%M:%S"),
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
    finally:
        conn.close()

    if df.empty:
        classes = transformer_bundle["classes"]
        probabilities = _normalise_scores({label: 1.0 for label in classes}, classes)
        prediction, confidence = _top_class(probabilities)
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "features": {
                "lookback_hours": lookback_hours,
                "samples": 0,
                "model": "TransformerEncoder",
                "model_type": "transformer",
            },
            "message": "No recent temperature history found.",
        }

    sequence, diagnostics = _build_tempreture_persona_transformer_sequence(
        df,
        timestamp,
        transformer_bundle,
    )
    torch = transformer_bundle["torch"]
    with torch.no_grad():
        logits = transformer_bundle["model"](torch.from_numpy(sequence).unsqueeze(0))
        proba = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    probabilities = {label: float(value) for label, value in zip(transformer_bundle["classes"], proba)}
    probabilities = _normalise_scores(probabilities, transformer_bundle["classes"])
    prediction, confidence = _top_class(probabilities)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "features": {
            **diagnostics,
            "lookback_hours": lookback_hours,
            "sequence_length": transformer_bundle["seq_len"],
            "model": "TransformerEncoder",
            "model_type": "transformer",
        },
    }


def predict_tempreture_recomendation(payload: dict) -> dict:
    room_number = int(payload.get("room_number"))
    timestamp = pd.Timestamp(payload.get("timestamp"))
    lookback_hours = int(payload.get("lookback_hours") or 2)
    lookback_start = timestamp - timedelta(hours=lookback_hours)
    occupancy = str(payload.get("occupancy_prediction") or "Unknown")
    persona = str(payload.get("temperature_persona_prediction") or "Unknown")
    model_type = _resolve_tempreture_recomendation_model_type(payload)

    conn = get_db_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp, room_number, floor, facade, room_type, size_m2,
                   outside_temp, room_temp, setpoint, ideal_temp, hvac_mode,
                   ac_persona, occupant_state, pir_persona, room_state,
                   pir_motion, guest_id
            FROM temperature_data
            WHERE room_number = ?
              AND timestamp >= ?
              AND timestamp <= ?
            ORDER BY timestamp
            """,
            conn,
            params=(
                room_number,
                lookback_start.strftime("%Y-%m-%d %H:%M:%S"),
                timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
    finally:
        conn.close()

    if df.empty:
        return {
            "empty": True,
            "message": "No recent temperature history found.",
        }

    if model_type == "hist_gradient_boosting":
        model_bundle = _load_tempreture_recomendation_hgb()
        if model_bundle is None:
            raise RuntimeError("HistGradientBoosting temperature recommendation model file was not found.")

        df_for_latest = df.copy()
        df_for_latest["timestamp"] = pd.to_datetime(df_for_latest["timestamp"], errors="coerce")
        df_for_latest = df_for_latest.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        for col in ["floor", "size_m2", "outside_temp", "room_temp", "setpoint", "ideal_temp", "pir_motion"]:
            df_for_latest[col] = pd.to_numeric(df_for_latest[col], errors="coerce").fillna(0)
        df_for_latest["guest_id"] = pd.to_numeric(df_for_latest["guest_id"], errors="coerce")
        latest_row = df_for_latest.iloc[-1]
        hgb_frame = _build_tempreture_recomendation_hgb_frame(
            latest_row,
            pd.Timestamp(latest_row.get("timestamp")) if pd.notna(latest_row.get("timestamp")) else timestamp,
            model_bundle,
            occupancy,
            persona,
        )
        metadata = model_bundle.get("metadata") or {}
        setpoint_min = float(metadata.get("setpoint_min", 16.0))
        setpoint_max = float(metadata.get("setpoint_max", 28.0))
        recommended_setpoint = float(model_bundle["model"].predict(hgb_frame)[0])
        recommended_setpoint = round(min(max(recommended_setpoint, setpoint_min), setpoint_max) * 2) / 2

        room_temp = float(latest_row.get("room_temp") or 0.0)
        setpoint = float(latest_row.get("setpoint") or 0.0)
        ideal_temp = float(latest_row.get("ideal_temp") or 0.0)
        diagnostics = {
            "room_temp": room_temp,
            "setpoint": setpoint,
            "ideal_temp": ideal_temp,
            "outside_temp": float(latest_row.get("outside_temp") or 0.0),
            "temp_error": room_temp - setpoint,
            "comfort_error": room_temp - ideal_temp,
            "hvac_mode": str(latest_row.get("hvac_mode") or "Unknown"),
            "room_state": str(latest_row.get("room_state") or "Unknown"),
            "occupant_state": str(latest_row.get("occupant_state") or "Unknown"),
            "motion_rate": float(df_for_latest["pir_motion"].mean()) if not df_for_latest.empty else 0.0,
            "has_guest": int(pd.notna(latest_row.get("guest_id"))),
            "samples": int(len(df_for_latest)),
        }
        served_model = "HistGradientBoostingRegressor"
        served_model_type = "hist_gradient_boosting"
        reason = (
            "HistGradientBoostingRegressor setpoint recommendation from current tabular room, "
            f"occupancy prediction ({occupancy}), and temperature persona ({persona})."
        )
        sequence_length = None
    else:
        transformer_bundle = _load_tempreture_recomendation_transformer()
        if transformer_bundle is None:
            raise RuntimeError("Transformer temperature recommendation model files were not found.")

        sequence, diagnostics = _build_tempreture_recomendation_transformer_sequence(
            df,
            timestamp,
            transformer_bundle,
            occupancy,
            persona,
        )
        torch = transformer_bundle["torch"]
        setpoint_min = transformer_bundle["setpoint_min"]
        setpoint_max = transformer_bundle["setpoint_max"]
        with torch.no_grad():
            scaled = float(transformer_bundle["model"](torch.from_numpy(sequence).unsqueeze(0)).cpu().numpy()[0])
        recommended_setpoint = scaled * (setpoint_max - setpoint_min) + setpoint_min
        recommended_setpoint = round(min(max(recommended_setpoint, setpoint_min), setpoint_max) * 2) / 2
        served_model = "TransformerEncoder"
        served_model_type = "transformer"
        reason = (
            "Transformer setpoint recommendation from recent temperature sequence, "
            f"occupancy prediction ({occupancy}), and temperature persona ({persona})."
        )
        sequence_length = transformer_bundle["seq_len"]

    current_setpoint = diagnostics["setpoint"]
    room_temp = diagnostics["room_temp"]
    outside_temp = diagnostics["outside_temp"]
    target_mode = _infer_hvac_recomendation_mode(
        room_temp,
        current_setpoint,
        outside_temp,
        diagnostics["hvac_mode"],
    )
    delta = recommended_setpoint - current_setpoint
    if abs(delta) < 0.25:
        action = "keep"
    elif delta > 0:
        action = "raise"
    else:
        action = "lower"

    latest_row = df.iloc[-1].copy()
    current_power_w = float(estimate_hvac_power(latest_row))
    recommended_row = latest_row.copy()
    recommended_row["setpoint"] = recommended_setpoint
    recommended_power_w = float(estimate_hvac_power(recommended_row))
    current_energy_wh = current_power_w * HVAC_HOURS_PER_SAMPLE
    recommended_energy_wh = recommended_power_w * HVAC_HOURS_PER_SAMPLE
    saved_energy_wh = current_energy_wh - recommended_energy_wh
    saved_pct = (100.0 * saved_energy_wh / current_energy_wh) if current_energy_wh else 0.0

    if occupancy == "Vacant":
        reason += " Vacant rooms can use a more energy-saving setpoint."
    elif persona == "EnergySaver":
        reason += " EnergySaver persona allows a wider comfort band."
    elif persona == "AlwaysOnComfort":
        reason += " AlwaysOnComfort persona prioritizes comfort."

    return {
        "empty": False,
        "prediction": recommended_setpoint,
        "recommended_setpoint": recommended_setpoint,
        "current_setpoint": current_setpoint,
        "setpoint_delta": delta,
        "action": action,
        "target_mode": target_mode,
        "model": served_model,
        "model_type": served_model_type,
        "reason": reason,
        "energy": {
            "current_power_w": current_power_w,
            "recommended_power_w": recommended_power_w,
            "current_wh": current_energy_wh,
            "recommended_wh": recommended_energy_wh,
            "saved_wh": saved_energy_wh,
            "saved_pct": saved_pct,
            "sample_minutes": int(round(HVAC_HOURS_PER_SAMPLE * 60)),
        },
        "input": {
            "room_number": room_number,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "lookback_hours": lookback_hours,
            "occupancy_prediction": occupancy,
            "temperature_persona_prediction": persona,
            "model_predictions": payload.get("model_predictions"),
        },
        "features": {
            **diagnostics,
            "setpoint_min": setpoint_min,
            "setpoint_max": setpoint_max,
            **({"sequence_length": sequence_length} if sequence_length is not None else {}),
        },
    }


def compute_tempreture_recomendation_energy_for_room(
    room_number: int,
    start_timestamp: str,
    end_timestamp: str,
    lookback_hours: int = 2,
) -> dict:
    transformer_bundle = _load_tempreture_recomendation_transformer()
    if transformer_bundle is None:
        raise RuntimeError("Transformer temperature recommendation model files were not found.")

    start_ts = pd.Timestamp(start_timestamp)
    end_ts = pd.Timestamp(end_timestamp)
    history_start = start_ts - timedelta(hours=lookback_hours)

    conn = get_db_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp, room_number, floor, facade, room_type, size_m2,
                   outside_temp, room_temp, setpoint, ideal_temp, hvac_mode,
                   ac_persona, occupant_state, pir_persona, room_state,
                   pir_motion, guest_id
            FROM temperature_data
            WHERE room_number = ?
              AND timestamp >= ?
              AND timestamp <= ?
            ORDER BY timestamp
            """,
            conn,
            params=(
                room_number,
                history_start.strftime("%Y-%m-%d %H:%M:%S"),
                end_ts.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
    finally:
        conn.close()

    if df.empty:
        return {"empty": True, "message": "No temperature history found."}

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ["floor", "size_m2", "outside_temp", "room_temp", "setpoint", "ideal_temp", "pir_motion"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["guest_id"] = pd.to_numeric(df["guest_id"], errors="coerce")

    target_mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
    target_indices = df.index[target_mask].to_numpy(dtype=np.int64)
    if len(target_indices) == 0:
        return {"empty": True, "message": "No temperature rows found in the selected interval."}

    seq_len = transformer_bundle["seq_len"]
    encoded_sequences = []
    target_rows = []
    for idx in target_indices:
        history = df.iloc[: idx + 1].tail(seq_len)
        history_rows = [row for _, row in history.iterrows()]
        if len(history_rows) < seq_len:
            history_rows = [history_rows[0]] * (seq_len - len(history_rows)) + history_rows

        current = df.iloc[idx]
        occupancy = str(current.get("room_state") or "Unknown")
        persona = str(current.get("ac_persona") or "Unknown")
        encoded_steps = []
        for row in history_rows:
            row_timestamp = pd.Timestamp(row.get("timestamp")) if pd.notna(row.get("timestamp")) else pd.Timestamp(current["timestamp"])
            encoded_steps.append(
                _encode_tempreture_recomendation_step(
                    _build_tempreture_recomendation_step(row, row_timestamp, occupancy, persona),
                    transformer_bundle,
                )
            )
        encoded_sequences.append(encoded_steps)
        target_rows.append(current.copy())

    torch = transformer_bundle["torch"]
    setpoint_min = transformer_bundle["setpoint_min"]
    setpoint_max = transformer_bundle["setpoint_max"]
    tensor = torch.from_numpy(np.array(encoded_sequences, dtype=np.float32))
    predictions = []
    with torch.no_grad():
        for start in range(0, len(tensor), 512):
            scaled = transformer_bundle["model"](tensor[start:start + 512]).cpu().numpy()
            predictions.extend(scaled.tolist())

    records = []
    for row, scaled in zip(target_rows, predictions):
        recommended_setpoint = float(scaled) * (setpoint_max - setpoint_min) + setpoint_min
        recommended_setpoint = round(min(max(recommended_setpoint, setpoint_min), setpoint_max) * 2) / 2
        current_power_w = float(estimate_hvac_power(row))
        recommended_row = row.copy()
        recommended_row["setpoint"] = recommended_setpoint
        recommended_power_w = float(estimate_hvac_power(recommended_row))
        current_wh = current_power_w * HVAC_HOURS_PER_SAMPLE
        recommended_wh = recommended_power_w * HVAC_HOURS_PER_SAMPLE
        saved_wh = current_wh - recommended_wh
        records.append(
            {
                "timestamp": pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                "hvac_mode": str(row.get("hvac_mode") or "Unknown"),
                "room_state": str(row.get("room_state") or "Unknown"),
                "temperature_persona": str(row.get("ac_persona") or "Unknown"),
                "room_temp": float(row.get("room_temp") or 0.0),
                "outside_temp": float(row.get("outside_temp") or 0.0),
                "current_setpoint": float(row.get("setpoint") or 0.0),
                "recommended_setpoint": recommended_setpoint,
                "current_power_w": current_power_w,
                "recommended_power_w": recommended_power_w,
                "current_wh": current_wh,
                "recommended_wh": recommended_wh,
                "saved_wh": saved_wh,
                "energy_change_wh": saved_wh,
            }
        )

    current_total = sum(r["current_wh"] for r in records)
    recommended_total = sum(r["recommended_wh"] for r in records)
    saved_total = current_total - recommended_total
    saved_pct = (100.0 * saved_total / current_total) if current_total else 0.0
    return {
        "empty": False,
        "summary": {
            "current_wh": current_total,
            "recommended_wh": recommended_total,
            "saved_wh": saved_total,
            "saved_pct": saved_pct,
            # Retained for compatibility with clients using the previous names.
            "energy_change_wh": saved_total,
            "energy_change_pct": saved_pct,
            "sample_minutes": int(round(HVAC_HOURS_PER_SAMPLE * 60)),
            "n_rows": len(records),
            "model": "TransformerEncoder",
            "model_type": "transformer",
        },
        "records": records[:5000],
    }
