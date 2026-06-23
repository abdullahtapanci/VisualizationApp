from __future__ import annotations

import math

import pandas as pd

from backend.hvac_energy import _estimate_power


SETPOINT_MIN = 14.0
SETPOINT_MAX = 28.0

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


def energy_saving_delta(occupancy: str, persona: str) -> float:
    occupancy_norm = str(occupancy or "").strip().lower()
    persona_norm = str(persona or "").strip().lower().replace("_", "")
    if occupancy_norm == "vacant":
        return 2.0
    if persona_norm == "energysaver":
        return 1.5
    if persona_norm == "alwaysoncomfort":
        return 0.5
    if persona_norm in {"preconditioning", "housekeeping"}:
        return 0.5
    return 1.0


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

    delta = energy_saving_delta(occupancy, persona)
    if current_mode == "heating":
        target_setpoint = current_setpoint - delta
    elif current_mode == "cooling":
        target_setpoint = current_setpoint + delta
    else:
        target_setpoint = current_setpoint
    target_setpoint = round(min(max(target_setpoint, SETPOINT_MIN), SETPOINT_MAX) * 2) / 2
    target_power, target_mode = estimate_candidate_power(row, target_setpoint, current_mode)
    if target_power > current_power + 1e-6:
        target_setpoint = round(min(max(current_setpoint, SETPOINT_MIN), SETPOINT_MAX) * 2) / 2
        target_power, target_mode = estimate_candidate_power(row, target_setpoint, current_mode)
    target_comfort_gap = abs(room_temp - float(row.get("ideal_temp") or current_setpoint or 22.0))

    return {
        "recommended_setpoint": round(target_setpoint * 2) / 2,
        "target_power_w": target_power,
        "target_mode": target_mode,
        "current_power_w": current_power,
        "current_energy_mode": current_mode,
        "target_comfort_gap_c": target_comfort_gap,
        "energy_saving_delta_c": delta,
    }


def build_energy_saving_targets(df: pd.DataFrame) -> pd.DataFrame:
    room_temp = df["room_temp"].astype(float)
    outside_temp = df["outside_temp"].astype(float)
    current_setpoint = df["setpoint"].astype(float)
    hvac_mode = df["hvac_mode"].fillna("Unknown").astype(str).str.lower()
    occupancy = df["room_state"].fillna("Unknown").astype(str).str.lower()
    persona = df["ac_persona"].fillna("Unknown").astype(str).str.lower().str.replace("_", "", regex=False)

    current_mode = hvac_mode.where(hvac_mode.isin(["heating", "cooling"]), "off")
    current_mode = current_mode.mask(
        ~hvac_mode.isin(["heating", "cooling"]) & ((room_temp > current_setpoint + 0.4) | (outside_temp >= 27)),
        "cooling",
    )
    current_mode = current_mode.mask(
        ~hvac_mode.isin(["heating", "cooling"]) & ((room_temp < current_setpoint - 0.4) | (outside_temp <= 12)),
        "heating",
    )

    delta = pd.Series(1.0, index=df.index)
    delta = delta.mask(occupancy.eq("vacant"), 2.0)
    delta = delta.mask(persona.eq("energysaver"), 1.5)
    delta = delta.mask(persona.eq("alwaysoncomfort"), 0.5)
    delta = delta.mask(persona.isin(["preconditioning", "housekeeping"]), 0.5)

    target = current_setpoint.copy()
    target = target.mask(current_mode.eq("heating"), current_setpoint - delta)
    target = target.mask(current_mode.eq("cooling"), current_setpoint + delta)
    target = (target.clip(SETPOINT_MIN, SETPOINT_MAX) * 2).round() / 2

    target_mode = current_mode.copy()
    target_mode = target_mode.mask(current_mode.eq("heating") & ~(room_temp < target - 0.4), "off")
    target_mode = target_mode.mask(current_mode.eq("cooling") & ~(room_temp > target + 0.4), "off")

    current_df = df.copy()
    current_df["hvac_mode"] = current_mode
    target_df = df.copy()
    target_df["setpoint"] = target
    target_df["hvac_mode"] = target_mode
    current_power = current_df.apply(_estimate_power, axis=1)
    target_power = target_df.apply(_estimate_power, axis=1)

    increase_mask = target_power > current_power
    target = target.mask(increase_mask, current_setpoint.clip(SETPOINT_MIN, SETPOINT_MAX))
    target = (target * 2).round() / 2
    target_mode = target_mode.mask(increase_mask, current_mode)
    target_power = target_power.mask(increase_mask, current_power)

    ideal_temp = df["ideal_temp"].astype(float).where(df["ideal_temp"].notna(), current_setpoint)
    return pd.DataFrame(
        {
            "recommended_setpoint": target,
            "target_power_w": target_power.astype(float),
            "target_mode": target_mode,
            "current_power_w": current_power.astype(float),
            "current_energy_mode": current_mode,
            "target_comfort_gap_c": (room_temp - ideal_temp).abs(),
            "energy_saving_delta_c": delta,
        }
    )


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
    target = build_energy_saving_targets(df)
    df = pd.concat([df, target], axis=1)
    return df
