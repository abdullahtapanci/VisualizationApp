from __future__ import annotations

import math

import numpy as np
import pandas as pd

from backend.hvac_energy import _estimate_power


SETPOINT_MIN = 14.0
SETPOINT_MAX = 28.0
SETPOINT_STEP = 0.5

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

CATEGORICAL_COLS = ["occupancy_prediction", "temperature_persona_prediction"]


def scale_temp(value) -> float:
    numeric = float(pd.to_numeric(pd.Series([value]), errors="coerce").fillna(0).iloc[0])
    numeric = min(max(numeric, -20.0), 50.0)
    return (numeric + 20.0) / 70.0


def cyclic_value(value: int, period: int) -> tuple[float, float]:
    radians = 2 * math.pi * value / period
    return math.sin(radians), math.cos(radians)


def infer_energy_mode(room_temp: float, setpoint: float, outside_temp: float, hvac_mode: str) -> str:
    mode = (hvac_mode or "").strip().lower()
    if mode in {"cooling", "heating"}:
        return mode
    if room_temp > setpoint + 0.4 or outside_temp >= 27:
        return "cooling"
    if room_temp < setpoint - 0.4 or outside_temp <= 12:
        return "heating"
    return "off"


def candidate_mode(room_temp: float, setpoint: float, current_mode: str) -> str:
    mode = (current_mode or "").strip().lower()
    if mode == "heating":
        return "heating" if room_temp < setpoint - 0.4 else "off"
    if mode == "cooling":
        return "cooling" if room_temp > setpoint + 0.4 else "off"
    if room_temp < setpoint - 0.6:
        return "heating"
    if room_temp > setpoint + 0.6:
        return "cooling"
    return "off"


def comfort_band(occupancy: str, persona: str) -> float:
    occupancy_norm = str(occupancy or "").strip().lower()
    persona_norm = str(persona or "").strip().lower().replace("_", "")
    if occupancy_norm == "vacant":
        return 5.0
    if persona_norm == "energysaver":
        return 3.5
    if persona_norm == "alwaysoncomfort":
        return 1.0
    if persona_norm in {"preconditioning", "housekeeping"}:
        return 1.5
    return 2.5


def estimate_candidate_power(row: pd.Series, setpoint: float, current_mode: str) -> tuple[float, str]:
    room_temp = float(row.get("room_temp") or 0.0)
    candidate = row.copy()
    mode = candidate_mode(room_temp, setpoint, current_mode)
    candidate["setpoint"] = setpoint
    candidate["hvac_mode"] = mode
    return float(_estimate_power(candidate)), mode


def choose_energy_saving_setpoint(row: pd.Series) -> dict:
    room_temp = float(row.get("room_temp") or 0.0)
    outside_temp = float(row.get("outside_temp") or 0.0)
    current_setpoint = float(row.get("setpoint") or 0.0)
    ideal_temp = float(row.get("ideal_temp") or current_setpoint or 22.0)
    occupancy = str(row.get("room_state") or "Unknown")
    persona = str(row.get("ac_persona") or "Unknown")

    current_mode = infer_energy_mode(
        room_temp,
        current_setpoint,
        outside_temp,
        str(row.get("hvac_mode") or "Unknown"),
    )
    current_row = row.copy()
    current_row["hvac_mode"] = current_mode
    current_power = float(_estimate_power(current_row))

    band = comfort_band(occupancy, persona)
    lower = max(SETPOINT_MIN, ideal_temp - band)
    upper = min(SETPOINT_MAX, ideal_temp + band)
    if lower > upper:
        lower, upper = SETPOINT_MIN, SETPOINT_MAX

    grid = np.arange(SETPOINT_MIN, SETPOINT_MAX + 0.001, SETPOINT_STEP)
    candidates = [float(value) for value in grid if lower - 1e-9 <= value <= upper + 1e-9]
    candidates.append(round(min(max(current_setpoint, SETPOINT_MIN), SETPOINT_MAX) * 2) / 2)
    if not candidates:
        candidates = [round(min(max(current_setpoint, SETPOINT_MIN), SETPOINT_MAX) * 2) / 2]

    evaluated = []
    for setpoint in candidates:
        power, mode = estimate_candidate_power(row, setpoint, current_mode)
        comfort_gap = abs(room_temp - ideal_temp)
        setpoint_gap = abs(setpoint - ideal_temp)
        evaluated.append((power, setpoint_gap, setpoint, mode, comfort_gap))

    target_power, _, target_setpoint, target_mode, target_comfort_gap = min(
        evaluated,
        key=lambda item: (item[0], item[1]),
    )
    if target_power > current_power + 1e-6:
        target_setpoint = round(min(max(current_setpoint, SETPOINT_MIN), SETPOINT_MAX) * 2) / 2
        target_power, target_mode = estimate_candidate_power(row, target_setpoint, current_mode)

    return {
        "recommended_setpoint": round(target_setpoint * 2) / 2,
        "target_power_w": target_power,
        "target_mode": target_mode,
        "current_power_w": current_power,
        "current_energy_mode": current_mode,
        "target_comfort_gap_c": target_comfort_gap,
        "comfort_band_c": band,
    }


def build_feature_row(row: pd.Series, timestamp: pd.Timestamp) -> dict:
    room_temp = float(row.get("room_temp") or 0.0)
    setpoint = float(row.get("setpoint") or 0.0)
    ideal_temp = float(row.get("ideal_temp") or 0.0)
    hour_sin, hour_cos = cyclic_value(int(timestamp.hour), 24)
    dow_sin, dow_cos = cyclic_value(int(timestamp.dayofweek), 7)
    return {
        "floor_scaled": min(max(float(row.get("floor") or 0.0), 0.0), 30.0) / 30.0,
        "size_scaled": min(max(float(row.get("size_m2") or 0.0), 0.0), 120.0) / 120.0,
        "outside_temp_scaled": scale_temp(row.get("outside_temp")),
        "room_temp_scaled": scale_temp(room_temp),
        "setpoint_scaled": scale_temp(setpoint),
        "ideal_temp_scaled": scale_temp(ideal_temp),
        "temp_error_scaled": (min(max(room_temp - setpoint, -20.0), 20.0) + 20.0) / 40.0,
        "comfort_error_scaled": (min(max(room_temp - ideal_temp, -20.0), 20.0) + 20.0) / 40.0,
        "pir_motion": min(max(float(row.get("pir_motion") or 0.0), 0.0), 1.0),
        "has_guest": 1.0 if pd.notna(row.get("guest_id")) else 0.0,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "occupancy_prediction": str(row.get("room_state") or "Unknown"),
        "temperature_persona_prediction": str(row.get("ac_persona") or "Unknown"),
    }


def prepare_temperature_frame(data_csv, max_rows: int | None = None) -> pd.DataFrame:
    usecols = [
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
    df = pd.read_csv(data_csv, usecols=usecols, parse_dates=["timestamp"], nrows=max_rows)
    df = df.dropna(subset=["timestamp"]).sort_values(["timestamp", "room_number"]).reset_index(drop=True)
    for col in ["floor", "size_m2", "outside_temp", "room_temp", "setpoint", "ideal_temp", "pir_motion"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["guest_id"] = pd.to_numeric(df["guest_id"], errors="coerce")
    target = df.apply(choose_energy_saving_setpoint, axis=1, result_type="expand")
    df = pd.concat([df, target], axis=1)
    return df
