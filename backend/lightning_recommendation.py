"""Lighting recommendation helpers for the visualization API."""

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
from backend.energy_calc import (
    DIMMABLE_LAMPS,
    NON_DIMMABLE_LAMPS,
    HOURS_PER_SAMPLE,
    _row_powers,
)


PERSONA_POLICY = {
    "StaticBright": {"factor": 0.90, "cap": 75, "min_on": 35},
    "Balanced": {"factor": 0.78, "cap": 60, "min_on": 25},
    "Routine": {"factor": 0.75, "cap": 55, "min_on": 25},
    "NightFocused": {"factor": 0.68, "cap": 45, "min_on": 15},
    "StaticDim": {"factor": 0.62, "cap": 35, "min_on": 10},
    "Housekeeping": {"factor": 1.00, "cap": 80, "min_on": 80},
    "Unknown": {"factor": 0.75, "cap": 55, "min_on": 20},
}

KNOWN_LAMPS = sorted(DIMMABLE_LAMPS | NON_DIMMABLE_LAMPS)
BASE_DIR = Path(__file__).resolve().parent.parent
LIGHTING_RECOMMENDATION_MODEL_FILE = (
    BASE_DIR / "AIModelsAndAlgorithms" / "LightingRecommendation" / "lighting_recommendation_model.joblib"
)
LIGHTING_RECOMMENDATION_TRANSFORMER_FILE = (
    BASE_DIR
    / "AIModelsAndAlgorithms"
    / "LightingRecommendation"
    / "transformer"
    / "lighting_recommendation_transformer.pt"
)
LIGHTING_RECOMMENDATION_TRANSFORMER_METADATA_FILE = (
    BASE_DIR
    / "AIModelsAndAlgorithms"
    / "LightingRecommendation"
    / "transformer"
    / "lighting_recommendation_transformer_metadata.json"
)
_LIGHTING_RECOMMENDATION_MODEL_CACHE: dict | None = None
_LIGHTING_RECOMMENDATION_MODEL_MTIME: float | None = None
_LIGHTING_RECOMMENDATION_TRANSFORMER_CACHE: dict | None = None
_LIGHTING_RECOMMENDATION_TRANSFORMER_MTIME: float | None = None


def _load_lighting_recommendation_model() -> dict | None:
    global _LIGHTING_RECOMMENDATION_MODEL_CACHE, _LIGHTING_RECOMMENDATION_MODEL_MTIME

    if not LIGHTING_RECOMMENDATION_MODEL_FILE.exists():
        return None

    model_mtime = LIGHTING_RECOMMENDATION_MODEL_FILE.stat().st_mtime
    if (
        _LIGHTING_RECOMMENDATION_MODEL_CACHE is None
        or _LIGHTING_RECOMMENDATION_MODEL_MTIME != model_mtime
    ):
        _LIGHTING_RECOMMENDATION_MODEL_CACHE = joblib.load(LIGHTING_RECOMMENDATION_MODEL_FILE)
        _LIGHTING_RECOMMENDATION_MODEL_MTIME = model_mtime
    return _LIGHTING_RECOMMENDATION_MODEL_CACHE


def _load_lighting_recommendation_transformer() -> dict | None:
    global _LIGHTING_RECOMMENDATION_TRANSFORMER_CACHE, _LIGHTING_RECOMMENDATION_TRANSFORMER_MTIME

    if not LIGHTING_RECOMMENDATION_TRANSFORMER_FILE.exists():
        return None
    if not LIGHTING_RECOMMENDATION_TRANSFORMER_METADATA_FILE.exists():
        return None

    model_mtime = max(
        LIGHTING_RECOMMENDATION_TRANSFORMER_FILE.stat().st_mtime,
        LIGHTING_RECOMMENDATION_TRANSFORMER_METADATA_FILE.stat().st_mtime,
    )
    if (
        _LIGHTING_RECOMMENDATION_TRANSFORMER_CACHE is not None
        and _LIGHTING_RECOMMENDATION_TRANSFORMER_MTIME == model_mtime
    ):
        return _LIGHTING_RECOMMENDATION_TRANSFORMER_CACHE

    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to use the Transformer lighting recommendation model.") from exc

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

    class LightingRecommendationTransformer(nn.Module):
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
            x = self.input_projection(x)
            x = self.position(x)
            x = self.encoder(x)
            last_token = x[:, -1, :]
            return torch.sigmoid(self.head(last_token)).squeeze(-1)

    metadata = json.loads(LIGHTING_RECOMMENDATION_TRANSFORMER_METADATA_FILE.read_text())
    checkpoint = torch.load(
        LIGHTING_RECOMMENDATION_TRANSFORMER_FILE,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_metadata = checkpoint.get("metadata") or {}
    config = {**metadata, **checkpoint_metadata}
    input_dim = int(config["input_dim"])
    model = LightingRecommendationTransformer(
        input_dim=input_dim,
        d_model=int(config.get("d_model", 96)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 2)),
        dim_feedforward=int(config.get("dim_feedforward", 192)),
        dropout=float(config.get("dropout", 0.15)),
    )
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError("Transformer checkpoint does not contain a model state_dict.")
    model.load_state_dict(state_dict)
    model.eval()

    _LIGHTING_RECOMMENDATION_TRANSFORMER_CACHE = {
        "model": model,
        "metadata": config,
        "torch": torch,
        "seq_len": int(config.get("sequence_length", 12)),
        "level_min": float(config.get("level_min", 0)),
        "level_max": float(config.get("level_max", 80)),
        "base_numeric_cols": list(config["base_numeric_cols"]),
        "categorical_cols": list(config["categorical_cols"]),
        "category_values": dict(config["category_values"]),
    }
    _LIGHTING_RECOMMENDATION_TRANSFORMER_MTIME = model_mtime
    return _LIGHTING_RECOMMENDATION_TRANSFORMER_CACHE


def _resolve_recommendation_model_type(payload: dict) -> str:
    requested = (
        payload.get("lighting_recommendation_model_type")
        or payload.get("recommendation_model_type")
        or os.environ.get("LIGHTING_RECOMMENDATION_MODEL_TYPE")
        or "hist_gradient_boosting"
    )
    requested = str(requested).strip().lower().replace("-", "_")
    if requested in {"hgb", "histgradientboosting", "hist_gradient_boosting", "hist_gradient_boosting_regressor"}:
        return "hist_gradient_boosting"
    if requested in {"transformer", "torch"}:
        return "transformer"
    if requested == "auto":
        return (
            "transformer"
            if LIGHTING_RECOMMENDATION_TRANSFORMER_FILE.exists()
            and LIGHTING_RECOMMENDATION_TRANSFORMER_METADATA_FILE.exists()
            else "hist_gradient_boosting"
        )
    return "hist_gradient_boosting"


def _finite_number(value: float, fallback: float = 0.0) -> float:
    """Return a JSON-safe float for values that may be empty or NaN."""
    if pd.isna(value):
        return fallback
    return float(value)


def _normalise_persona(persona: str | None) -> str:
    persona = (persona or "Unknown").strip()
    return persona if persona in PERSONA_POLICY else "Unknown"


def _normalise_occupancy(occupancy: str | None) -> str:
    occupancy = (occupancy or "Occupied").strip()
    if occupancy not in {"Occupied", "Vacant", "Cleaning"}:
        return "Occupied"
    return occupancy


def _energy_wh(lamp: str, level: float) -> float:
    if level <= 0 or lamp == "none":
        return 0.0
    actual_w, _ = _row_powers(lamp, int(round(level)))
    if pd.isna(actual_w):
        return 0.0
    return float(actual_w) * HOURS_PER_SAMPLE


def _past_mean(df: pd.DataFrame, column: str, since: pd.Timestamp | None = None) -> float:
    if since is not None:
        df = df[df["timestamp"] >= since]
    if df.empty:
        return 0.0
    return _finite_number(pd.to_numeric(df[column], errors="coerce").mean())


def _build_model_feature_row(
    lamp: str,
    room_number: int,
    floor: float,
    current_level: float,
    latest_row: pd.Series | None,
    lamp_df: pd.DataFrame,
    room_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    occupancy: str,
    persona: str,
) -> dict:
    one_hour = timestamp - timedelta(hours=1)
    three_hours = timestamp - timedelta(hours=3)
    day_start = timestamp - timedelta(hours=24)

    # The training script uses past-only rolling features. At serving time the
    # latest selected sample is the current state, so history features exclude it.
    if latest_row is not None:
        lamp_history = lamp_df[lamp_df["timestamp"] < latest_row["timestamp"]]
        room_history = room_df[room_df["timestamp"] < latest_row["timestamp"]]
        reservation_active = latest_row.get("reservation_active") or "No"
        pir_motion = _finite_number(latest_row.get("pir_motion"))
        n_occupants = _finite_number(latest_row.get("n_occupants"))
        active_actors = _finite_number(latest_row.get("active_actors"))
        hurry_morning = _finite_number(latest_row.get("hurry_morning"))
        lazy_day = _finite_number(latest_row.get("lazy_day"))
        forgetful = _finite_number(latest_row.get("forgetful"))
    else:
        lamp_history = lamp_df
        room_history = room_df
        latest_room = room_df.iloc[-1] if not room_df.empty else None
        reservation_active = latest_room.get("reservation_active") if latest_room is not None else "No"
        pir_motion = _finite_number(latest_room.get("pir_motion")) if latest_room is not None else 0.0
        n_occupants = _finite_number(latest_room.get("n_occupants")) if latest_room is not None else 0.0
        active_actors = _finite_number(latest_room.get("active_actors")) if latest_room is not None else 0.0
        hurry_morning = _finite_number(latest_room.get("hurry_morning")) if latest_room is not None else 0.0
        lazy_day = _finite_number(latest_room.get("lazy_day")) if latest_room is not None else 0.0
        forgetful = _finite_number(latest_room.get("forgetful")) if latest_room is not None else 0.0

    lamp_on_history = lamp_history.assign(is_on=lamp_history["value"].gt(0).astype(int))
    hour = timestamp.hour
    dayofweek = timestamp.dayofweek
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    dow_sin = math.sin(2 * math.pi * dayofweek / 7)
    dow_cos = math.cos(2 * math.pi * dayofweek / 7)

    return {
        "room_number": room_number,
        "floor": floor,
        "Value": current_level,
        "pir_motion": pir_motion,
        "n_occupants": n_occupants,
        "active_actors": active_actors,
        "hurry_morning": hurry_morning,
        "lazy_day": lazy_day,
        "forgetful": forgetful,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "lamp_value_mean_1h": _past_mean(lamp_history, "value", one_hour),
        "lamp_value_mean_3h": _past_mean(lamp_history, "value", three_hours),
        "lamp_value_mean_24h": _past_mean(lamp_history, "value", day_start),
        "lamp_on_rate_3h": _past_mean(lamp_on_history, "is_on", three_hours),
        "room_value_mean_1h": _past_mean(room_history, "value", one_hour),
        "room_value_mean_24h": _past_mean(room_history, "value", day_start),
        "room_motion_rate_1h": _past_mean(room_history, "pir_motion", one_hour),
        "lamp_location": lamp,
        "reservation_active": reservation_active or "No",
        "occupancy_prediction": occupancy,
        "lighting_persona_prediction": persona,
    }


def _predict_model_level(model_bundle: dict, feature_row: dict) -> float:
    feature_columns = model_bundle["feature_columns"]
    features_df = pd.DataFrame([{col: feature_row.get(col, 0.0) for col in feature_columns}])
    predicted = float(model_bundle["model"].predict(features_df)[0])
    return float(min(max(round(predicted), model_bundle.get("level_min", 0)), model_bundle.get("level_max", 80)))


def _build_transformer_step(row: pd.Series, lamp: str, timestamp: pd.Timestamp, occupancy: str, persona: str) -> dict:
    value = _finite_number(row.get("value"))
    hour = timestamp.hour
    dayofweek = timestamp.dayofweek
    return {
        "value_scaled": min(max(value, 0.0), 80.0) / 80.0,
        "pir_motion": min(max(_finite_number(row.get("pir_motion")), 0.0), 1.0),
        "n_occupants_scaled": min(max(_finite_number(row.get("n_occupants")), 0.0), 8.0) / 8.0,
        "active_actors_scaled": min(max(_finite_number(row.get("active_actors")), 0.0), 8.0) / 8.0,
        "hurry_morning": _finite_number(row.get("hurry_morning")),
        "lazy_day": _finite_number(row.get("lazy_day")),
        "forgetful": _finite_number(row.get("forgetful")),
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "dow_sin": math.sin(2 * math.pi * dayofweek / 7),
        "dow_cos": math.cos(2 * math.pi * dayofweek / 7),
        "lamp_location": lamp,
        "reservation_active": str(row.get("reservation_active") or "Unknown"),
        "occupancy_prediction": occupancy,
        "lighting_persona_prediction": persona,
    }


def _encode_transformer_step(step: dict, transformer_bundle: dict) -> list[float]:
    values = [float(step.get(col, 0.0)) for col in transformer_bundle["base_numeric_cols"]]
    for col in transformer_bundle["categorical_cols"]:
        categories = transformer_bundle["category_values"].get(col, [])
        raw_value = str(step.get(col) or "Unknown")
        if raw_value not in categories:
            raw_value = "Unknown" if "Unknown" in categories else (categories[0] if categories else "")
        values.extend(1.0 if category == raw_value else 0.0 for category in categories)
    return values


def _predict_transformer_level(
    transformer_bundle: dict,
    lamp: str,
    lamp_df: pd.DataFrame,
    room_df: pd.DataFrame,
    timestamp: pd.Timestamp,
    occupancy: str,
    persona: str,
) -> float:
    torch = transformer_bundle["torch"]
    seq_len = transformer_bundle["seq_len"]
    level_min = transformer_bundle["level_min"]
    level_max = transformer_bundle["level_max"]

    if lamp_df.empty:
        latest_room = room_df.iloc[-1] if not room_df.empty else pd.Series(dtype=object)
        synthetic = latest_room.copy()
        synthetic["value"] = 0.0
        synthetic["reservation_active"] = synthetic.get("reservation_active") or "Unknown"
        history_rows = [synthetic] * seq_len
        history_times = [timestamp] * seq_len
    else:
        history = lamp_df.sort_values("timestamp").tail(seq_len)
        history_rows = [row for _, row in history.iterrows()]
        history_times = [pd.Timestamp(row.get("timestamp")) for row in history_rows]
        if len(history_rows) < seq_len:
            first_row = history_rows[0]
            first_time = history_times[0]
            pad_count = seq_len - len(history_rows)
            history_rows = [first_row] * pad_count + history_rows
            history_times = [first_time] * pad_count + history_times

    encoded_steps = [
        _encode_transformer_step(
            _build_transformer_step(row, lamp, step_time, occupancy, persona),
            transformer_bundle,
        )
        for row, step_time in zip(history_rows, history_times)
    ]
    tensor = torch.tensor(np.array([encoded_steps], dtype=np.float32))
    with torch.no_grad():
        scaled = float(transformer_bundle["model"](tensor).detach().cpu().numpy()[0])
    predicted = round(scaled * level_max)
    return float(min(max(predicted, level_min), level_max))


def _recommend_level(lamp: str,
                     current_level: float,
                     mean_1h: float,
                     mean_3h: float,
                     mean_24h: float,
                     occupancy: str,
                     persona: str) -> tuple[float, str]:
    if occupancy == "Vacant":
        return 0.0, "Vacant prediction: turn lamp off."

    if occupancy == "Cleaning" or persona == "Housekeeping":
        if lamp in NON_DIMMABLE_LAMPS:
            return 80.0 if current_level > 0 else 0.0, "Cleaning mode keeps active non-dimmable lamps on."
        return 80.0 if current_level > 0 else 0.0, "Cleaning mode keeps active dimmable lamps bright."

    if current_level <= 0:
        return 0.0, "Lamp is already off in the latest room history."

    if lamp in NON_DIMMABLE_LAMPS:
        return 80.0, "Non-dimmable lamp: keep on or turn off only."

    policy = PERSONA_POLICY[_normalise_persona(persona)]
    history_values = [v for v in [mean_1h, mean_3h, mean_24h] if pd.notna(v) and v > 0]
    recent_preference = float(pd.Series(history_values).median()) if history_values else current_level
    recommended = min(current_level, recent_preference, policy["cap"]) * policy["factor"]
    recommended = max(recommended, policy["min_on"])
    recommended = min(recommended, current_level)
    return float(round(recommended)), (
        f"{persona} policy: cap {policy['cap']}, "
        f"factor {policy['factor']:.2f}, recent preference {recent_preference:.1f}."
    )


def compute_lightning_recommendation(payload: dict) -> dict:
    room_number = int(payload.get("room_number"))
    timestamp = pd.Timestamp(payload.get("timestamp"))
    occupancy = _normalise_occupancy(payload.get("occupancy_prediction"))
    persona = _normalise_persona(payload.get("lighting_persona_prediction"))
    lookback_hours = int(payload.get("lookback_hours") or 24)
    lookback_start = timestamp - timedelta(hours=lookback_hours)

    conn = get_db_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp, floor, lamp_location, "Value" AS value, pir_motion,
                   n_occupants, active_actors, reservation_active,
                   hurry_morning, lazy_day, forgetful
            FROM lightning_data
            WHERE room_number = ?
              AND timestamp >= ?
              AND timestamp <= ?
              AND lamp_location <> 'none'
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
            "message": "No lighting history found for this room and timestamp.",
        }

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df = df.dropna(subset=["timestamp"])
    for column in [
        "floor",
        "pir_motion",
        "n_occupants",
        "active_actors",
        "hurry_morning",
        "lazy_day",
        "forgetful",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    one_hour = timestamp - timedelta(hours=1)
    three_hours = timestamp - timedelta(hours=3)
    room_df = df.sort_values("timestamp")
    floor = _finite_number(room_df["floor"].iloc[-1]) if not room_df.empty else 0.0
    requested_model_type = _resolve_recommendation_model_type(payload)
    model_bundle = None
    transformer_bundle = None
    if requested_model_type == "transformer":
        transformer_bundle = _load_lighting_recommendation_transformer()
    else:
        model_bundle = _load_lighting_recommendation_model()

    if transformer_bundle:
        recommendation_model = "TransformerRegressor"
    elif model_bundle:
        recommendation_model = "HistGradientBoostingRegressor"
    else:
        recommendation_model = "RuleBasedFallback"

    recommendations = []
    for lamp in KNOWN_LAMPS:
        lamp_df = df[df["lamp_location"] == lamp].sort_values("timestamp")
        if lamp_df.empty:
            current_level = mean_1h = mean_3h = mean_24h = 0.0
            last_seen = None
            latest_row = None
        else:
            latest = lamp_df.iloc[-1]
            current_level = _finite_number(latest["value"])
            last_seen = latest["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            mean_1h = _finite_number(lamp_df[lamp_df["timestamp"] >= one_hour]["value"].mean())
            mean_3h = _finite_number(lamp_df[lamp_df["timestamp"] >= three_hours]["value"].mean())
            mean_24h = _finite_number(lamp_df["value"].mean())
            latest_row = latest

        model_level = None
        if transformer_bundle:
            model_level = _predict_transformer_level(
                transformer_bundle,
                lamp,
                lamp_df,
                room_df,
                timestamp,
                occupancy,
                persona,
            )
            if current_level <= 0:
                recommended_level = 0.0
                constraint = " Lamp is currently off, so the served recommendation keeps it off."
            elif lamp in NON_DIMMABLE_LAMPS:
                recommended_level = 80.0 if model_level > 0 else 0.0
                constraint = " Non-dimmable hardware constraint snaps the model output to on/off."
            else:
                recommended_level = model_level
                constraint = ""
            reason = (
                "AI model prediction: TransformerRegressor estimated "
                "the next 5-minute recommended level from recent per-lamp sequence history, "
                "occupancy, and lighting persona."
                f"{constraint}"
            )
        elif model_bundle:
            feature_row = _build_model_feature_row(
                lamp,
                room_number,
                floor,
                current_level,
                latest_row,
                lamp_df,
                room_df,
                timestamp,
                occupancy,
                persona,
            )
            model_level = _predict_model_level(model_bundle, feature_row)
            if current_level <= 0:
                recommended_level = 0.0
                constraint = " Lamp is currently off, so the served recommendation keeps it off."
            elif lamp in NON_DIMMABLE_LAMPS:
                recommended_level = 80.0 if model_level > 0 else 0.0
                constraint = " Non-dimmable hardware constraint snaps the model output to on/off."
            else:
                recommended_level = model_level
                constraint = ""
            reason = (
                "AI model prediction: HistGradientBoostingRegressor estimated "
                "the next 5-minute recommended level from current lamp state, "
                "recent lighting history, occupancy, and lighting persona."
                f"{constraint}"
            )
        else:
            recommended_level, reason = _recommend_level(
                lamp,
                current_level,
                mean_1h,
                mean_3h,
                mean_24h,
                occupancy,
                persona,
            )
        actual_wh = _energy_wh(lamp, current_level)
        recommended_wh = _energy_wh(lamp, recommended_level)
        is_dimmable = lamp in DIMMABLE_LAMPS
        recommendations.append({
            "lamp": lamp,
            "lamp_type": "Dimmable LED" if is_dimmable else "Non-dimmable bulb",
            "current_level": current_level,
            "recommended_level": recommended_level,
            "mean_1h": mean_1h,
            "mean_3h": mean_3h,
            "mean_24h": mean_24h,
            "actual_wh": actual_wh,
            "recommended_wh": recommended_wh,
            "saved_wh": actual_wh - recommended_wh if is_dimmable else 0.0,
            "saving_counted": is_dimmable,
            "last_seen": last_seen,
            "reason": reason,
            "model": recommendation_model,
            "model_predicted_level": model_level,
        })

    dimmable_recommendations = [r for r in recommendations if r["saving_counted"]]
    actual_total = sum(r["actual_wh"] for r in dimmable_recommendations)
    recommended_total = sum(r["recommended_wh"] for r in dimmable_recommendations)
    # Summary savings must be the net difference between the two displayed
    # dimmable-energy totals. Summing only positive per-lamp reductions would
    # overstate savings when another dimmable lamp increases its consumption.
    saved_total = actual_total - recommended_total
    saved_pct = (100.0 * saved_total / actual_total) if actual_total else 0.0
    max_dimmable_total = sum(_energy_wh(lamp, 80.0) for lamp in DIMMABLE_LAMPS)
    max_baseline_saved_total = max(max_dimmable_total - recommended_total, 0.0)
    max_baseline_saved_pct = (
        100.0 * max_baseline_saved_total / max_dimmable_total
    ) if max_dimmable_total else 0.0
    all_actual_total = sum(r["actual_wh"] for r in recommendations)
    all_recommended_total = sum(r["recommended_wh"] for r in recommendations)

    guest = payload.get("guest") or {}
    return {
        "empty": False,
        "input": {
            "room_number": room_number,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "lookback_hours": lookback_hours,
            "occupancy_prediction": occupancy,
            "lighting_persona_prediction": persona,
            "guest": guest,
            "model_predictions": payload.get("model_predictions"),
            "recommendation_model": recommendation_model,
        },
        "summary": {
            "actual_wh": actual_total,
            "recommended_wh": recommended_total,
            "saved_wh": saved_total,
            "saved_pct": saved_pct,
            "max_dimmable_wh": max_dimmable_total,
            "max_baseline_saved_wh": max_baseline_saved_total,
            "max_baseline_saved_pct": max_baseline_saved_pct,
            "savings_scope": "dimmable_lamps_only",
            "all_lamps_actual_wh": all_actual_total,
            "all_lamps_recommended_wh": all_recommended_total,
            "active_lamps": sum(1 for r in recommendations if r["current_level"] > 0),
            "recommended_active_lamps": sum(1 for r in recommendations if r["recommended_level"] > 0),
            "recommended_active_dimmable_lamps": sum(
                1 for r in dimmable_recommendations if r["recommended_level"] > 0
            ),
            "recommendation_model": recommendation_model,
        },
        "recommendations": recommendations,
    }


def compute_lightning_recommendation_energy_for_room(
    room_number: int,
    start_timestamp: str,
    end_timestamp: str,
    lookback_hours: int = 24,
    model_type: str = "auto",
) -> dict:
    start_ts = pd.Timestamp(start_timestamp)
    end_ts = pd.Timestamp(end_timestamp)
    lookback_start = start_ts - timedelta(hours=lookback_hours)

    conn = get_db_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp, floor, lamp_location, "Value" AS value, pir_motion,
                   n_occupants, active_actors, reservation_active,
                   hurry_morning, lazy_day, forgetful, room_state,
                   lightning_persona
            FROM lightning_data
            WHERE room_number = ?
              AND timestamp >= ?
              AND timestamp <= ?
              AND lamp_location <> 'none'
            ORDER BY timestamp
            """,
            conn,
            params=(
                room_number,
                lookback_start.strftime("%Y-%m-%d %H:%M:%S"),
                end_ts.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
    finally:
        conn.close()

    if df.empty:
        return {"empty": True, "message": "No lighting history found."}

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for column in [
        "floor",
        "pir_motion",
        "n_occupants",
        "active_actors",
        "hurry_morning",
        "lazy_day",
        "forgetful",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    requested_model_type = _resolve_recommendation_model_type(
        {"lighting_recommendation_model_type": model_type}
    )
    model_bundle = None
    transformer_bundle = None
    if requested_model_type == "transformer":
        transformer_bundle = _load_lighting_recommendation_transformer()
    else:
        model_bundle = _load_lighting_recommendation_model()

    if transformer_bundle:
        recommendation_model = "TransformerRegressor"
    elif model_bundle:
        recommendation_model = "HistGradientBoostingRegressor"
    else:
        recommendation_model = "RuleBasedFallback"

    target_times = (
        df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]["timestamp"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not target_times:
        return {"empty": True, "message": "No lighting rows found in the selected interval."}

    records = []
    for timestamp in target_times:
        history_df = df[df["timestamp"] <= timestamp].copy()
        current_df = history_df[history_df["timestamp"] == timestamp]
        if current_df.empty:
            continue

        latest_context = current_df.iloc[-1]
        occupancy = _normalise_occupancy(latest_context.get("room_state"))
        persona = _normalise_persona(latest_context.get("lightning_persona"))
        room_df = history_df.sort_values("timestamp")
        floor = _finite_number(room_df["floor"].iloc[-1]) if not room_df.empty else 0.0

        for lamp in DIMMABLE_LAMPS:
            lamp_df = history_df[history_df["lamp_location"] == lamp].sort_values("timestamp")
            current_lamp_rows = current_df[current_df["lamp_location"] == lamp]
            if current_lamp_rows.empty:
                current_level = 0.0
                latest_row = None
            else:
                latest_row = current_lamp_rows.iloc[-1]
                current_level = _finite_number(latest_row["value"])

            if transformer_bundle:
                model_level = _predict_transformer_level(
                    transformer_bundle,
                    lamp,
                    lamp_df,
                    room_df,
                    timestamp,
                    occupancy,
                    persona,
                )
                recommended_level = 0.0 if current_level <= 0 else model_level
            elif model_bundle:
                feature_row = _build_model_feature_row(
                    lamp,
                    room_number,
                    floor,
                    current_level,
                    latest_row,
                    lamp_df,
                    room_df,
                    timestamp,
                    occupancy,
                    persona,
                )
                model_level = _predict_model_level(model_bundle, feature_row)
                recommended_level = 0.0 if current_level <= 0 else model_level
            else:
                one_hour = timestamp - timedelta(hours=1)
                three_hours = timestamp - timedelta(hours=3)
                mean_1h = _finite_number(lamp_df[lamp_df["timestamp"] >= one_hour]["value"].mean())
                mean_3h = _finite_number(lamp_df[lamp_df["timestamp"] >= three_hours]["value"].mean())
                mean_24h = _finite_number(lamp_df["value"].mean())
                recommended_level, _ = _recommend_level(
                    lamp,
                    current_level,
                    mean_1h,
                    mean_3h,
                    mean_24h,
                    occupancy,
                    persona,
                )

            current_wh = _energy_wh(lamp, current_level)
            recommended_wh = _energy_wh(lamp, recommended_level)
            full_brightness_wh = _energy_wh(lamp, 80.0)
            records.append(
                {
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "lamp": lamp,
                    "occupancy_prediction": occupancy,
                    "lighting_persona_prediction": persona,
                    "current_level": current_level,
                    "recommended_level": recommended_level,
                    "current_wh": current_wh,
                    "full_brightness_wh": full_brightness_wh,
                    "recommended_wh": recommended_wh,
                    "energy_change_wh": full_brightness_wh - recommended_wh,
                }
            )

    current_total = sum(r["current_wh"] for r in records)
    full_brightness_total = sum(r["full_brightness_wh"] for r in records)
    recommended_total = sum(r["recommended_wh"] for r in records)
    energy_change = full_brightness_total - recommended_total
    energy_change_pct = (100.0 * energy_change / full_brightness_total) if full_brightness_total else 0.0
    return {
        "empty": False,
        "summary": {
            "current_wh": current_total,
            "full_brightness_wh": full_brightness_total,
            "recommended_wh": recommended_total,
            "energy_change_wh": energy_change,
            "energy_change_pct": energy_change_pct,
            "n_rows": len(records),
            "n_timestamps": len(target_times),
            "sample_minutes": int(round(HOURS_PER_SAMPLE * 60)),
            "model": recommendation_model,
            "savings_scope": "dimmable_lamps_only",
        },
        "records": records[:5000],
    }
