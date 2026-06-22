"""HVAC energy-consumption helpers.

The temperature_data table records HVAC state once every 5 minutes. We do
not have direct meter readings, so active HVAC rows are estimated with a
small thermal-demand model:

* ``hvac_mode`` decides whether heating, cooling, or no HVAC load applies.
* ``room_temp`` and ``setpoint`` determine how far the room is from target.
* ``outside_temp`` adds weather stress.
* ``size_m2`` scales the load for larger or smaller rooms.
* ``room_state`` and ``pir_motion`` adjust the load for occupancy/activity.

Each row contributes ``estimated_power_w * (5/60)`` Wh.
"""

from __future__ import annotations

import pandas as pd

from backend.data_loader import get_db_connection


SAMPLE_MINUTES = 5
HOURS_PER_SAMPLE = SAMPLE_MINUTES / 60.0
REFERENCE_ROOM_SIZE_M2 = 35.0

# Rated/max electrical power per HVAC mode, in watts.
RATED_POWER_W = {
    "heating": 2000.0,
    "cooling": 1500.0,
    "off": 0.0,
}

MODE_POWER_MODEL = {
    "heating": {
        "base_w": 420.0,
        "room_gap_w_per_c": 260.0,
        "weather_gap_w_per_c": 32.0,
        "max_w": RATED_POWER_W["heating"],
    },
    "cooling": {
        "base_w": 380.0,
        "room_gap_w_per_c": 230.0,
        "weather_gap_w_per_c": 30.0,
        "max_w": RATED_POWER_W["cooling"],
    },
}


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _occupancy_factor(room_state, pir_motion) -> float:
    """Return load multiplier from room state and PIR activity."""
    state = str(room_state or "").strip().lower()
    try:
        motion = bool(int(float(pir_motion))) if pd.notna(pir_motion) else False
    except (TypeError, ValueError):
        motion = False

    if state == "vacant":
        return 0.55
    if state == "cleaning":
        return 0.75
    if state == "occupied":
        return 1.10 if motion else 0.95
    return 1.0 if motion else 0.85


def _estimate_power(row) -> float:
    """Estimate electrical power draw for one 5-minute HVAC sample."""
    mode = str(row.get("hvac_mode") or "").strip().lower()
    if mode not in MODE_POWER_MODEL:
        return 0.0

    room_temp = row.get("room_temp")
    setpoint = row.get("setpoint")
    outside_temp = row.get("outside_temp")
    size_m2 = row.get("size_m2")

    if pd.isna(room_temp) or pd.isna(setpoint) or pd.isna(outside_temp):
        return 0.0

    if mode == "heating":
        room_gap = max(float(setpoint) - float(room_temp), 0.0)
        weather_gap = max(float(setpoint) - float(outside_temp), 0.0)
    else:
        room_gap = max(float(room_temp) - float(setpoint), 0.0)
        weather_gap = max(float(outside_temp) - float(setpoint), 0.0)

    config = MODE_POWER_MODEL[mode]
    size = REFERENCE_ROOM_SIZE_M2 if pd.isna(size_m2) else float(size_m2)
    size_factor = _clip(size / REFERENCE_ROOM_SIZE_M2, 0.70, 1.45)
    occupancy_factor = _occupancy_factor(row.get("room_state"),
                                         row.get("pir_motion"))

    raw_power = (
        config["base_w"]
        + config["room_gap_w_per_c"] * room_gap
        + config["weather_gap_w_per_c"] * weather_gap
    )
    adjusted_power = raw_power * size_factor * occupancy_factor
    return _clip(adjusted_power, 0.0, config["max_w"])


def compute_hvac_energy_for_room(room_number: int,
                                 start_timestamp: str,
                                 end_timestamp: str) -> dict:
    """Compute HVAC energy use for a room within an interval.

    Returns a dict with the per-row dataframe plus aggregate totals and a
    per-mode breakdown so the API layer can render charts.
    """
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp, hvac_mode, room_temp, setpoint, outside_temp,
                   size_m2, room_state, pir_motion
            FROM temperature_data
            WHERE room_number = ?
              AND timestamp >= ?
              AND timestamp <  ?
            """,
            conn,
            params=(room_number, start_timestamp, end_timestamp),
        )
    finally:
        conn.close()

    if df.empty:
        return {"empty": True, "df": df}

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df["hvac_mode"] = df["hvac_mode"].fillna("off").astype(str).str.lower()

    df["power_w"] = df.apply(_estimate_power, axis=1)
    df["energy_wh"] = df["power_w"] * HOURS_PER_SAMPLE

    by_mode = (df.groupby("hvac_mode")
                 .agg(samples=("hvac_mode", "size"),
                      energy_wh=("energy_wh", "sum"),
                      avg_power_w=("power_w", "mean"))
                 .reset_index())
    by_mode["minutes"] = by_mode["samples"] * SAMPLE_MINUTES
    by_mode["rated_w"] = by_mode["hvac_mode"].map(RATED_POWER_W).fillna(0.0)

    total_energy = float(df["energy_wh"].sum())
    heating_energy = float(df.loc[df["hvac_mode"] == "heating", "energy_wh"].sum())
    cooling_energy = float(df.loc[df["hvac_mode"] == "cooling", "energy_wh"].sum())
    active_minutes = int((df["hvac_mode"].isin(["heating", "cooling"])).sum() * SAMPLE_MINUTES)
    total_minutes = int(len(df) * SAMPLE_MINUTES)

    # Max baseline: HVAC running continuously at the dominant active mode's
    # rated power for the entire interval (the "no smart control" scenario).
    # Dominant mode = whichever mode (heating/cooling) accumulated more
    # samples in the actual data; falls back to heating if neither was used.
    heating_samples = int((df["hvac_mode"] == "heating").sum())
    cooling_samples = int((df["hvac_mode"] == "cooling").sum())
    if heating_samples == 0 and cooling_samples == 0:
        dominant_mode = "heating"  # default fallback
    elif heating_samples >= cooling_samples:
        dominant_mode = "heating"
    else:
        dominant_mode = "cooling"
    max_power_w = RATED_POWER_W[dominant_mode]
    max_energy = float(max_power_w * len(df) * HOURS_PER_SAMPLE)
    saved_energy = max_energy - total_energy
    saved_pct = (100.0 * saved_energy / max_energy) if max_energy else 0.0

    by_mode_records = [
        {
            "mode": r["hvac_mode"],
            "samples": int(r["samples"]),
            "minutes": int(r["minutes"]),
            "rated_w": float(r["rated_w"]),
            "avg_power_w": float(r["avg_power_w"]),
            "energy_wh": float(r["energy_wh"]),
        }
        for _, r in by_mode.iterrows()
    ]

    return {
        "empty": False,
        "df": df,
        "by_mode": by_mode,
        "summary": {
            "total_wh": total_energy,
            "heating_wh": heating_energy,
            "cooling_wh": cooling_energy,
            "max_wh": max_energy,
            "saved_wh": saved_energy,
            "saved_pct": saved_pct,
            "dominant_mode": dominant_mode,
            "max_power_w": float(max_power_w),
            "active_minutes": active_minutes,
            "total_minutes": total_minutes,
            "n_rows": int(len(df)),
        },
        "by_mode_records": by_mode_records,
        "rated_power_w": dict(RATED_POWER_W),
        "model": {
            "reference_room_size_m2": REFERENCE_ROOM_SIZE_M2,
            "mode_power_model": MODE_POWER_MODEL,
            "occupancy_factors": {
                "vacant": 0.55,
                "cleaning": 0.75,
                "occupied_no_motion": 0.95,
                "occupied_with_motion": 1.10,
                "unknown_no_motion": 0.85,
                "unknown_with_motion": 1.0,
            },
        },
    }
