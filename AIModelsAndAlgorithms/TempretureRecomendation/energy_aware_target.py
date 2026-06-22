"""Energy-aware HVAC setpoint target shared by temperature recommenders."""

from __future__ import annotations

import numpy as np
import pandas as pd


SETPOINT_MIN = 16.0
SETPOINT_MAX = 28.0
SETPOINT_STEP = 0.5
REFERENCE_ROOM_SIZE_M2 = 35.0

MODE_POWER_MODEL = {
    "heating": {
        "base_w": 420.0,
        "room_gap_w_per_c": 260.0,
        "weather_gap_w_per_c": 32.0,
        "max_w": 2000.0,
    },
    "cooling": {
        "base_w": 380.0,
        "room_gap_w_per_c": 230.0,
        "weather_gap_w_per_c": 30.0,
        "max_w": 1500.0,
    },
}

PERSONA_COMFORT = {
    "AlwaysOnComfort": {"band": 0.4, "comfort_weight": 9.0, "change_weight": 0.10},
    "Reactive": {"band": 0.8, "comfort_weight": 5.5, "change_weight": 0.08},
    "Preconditioning": {"band": 0.8, "comfort_weight": 5.0, "change_weight": 0.08},
    "EnergySaver": {"band": 1.5, "comfort_weight": 2.6, "change_weight": 0.05},
    "Housekeeping": {"band": 1.0, "comfort_weight": 4.0, "change_weight": 0.06},
    "Unknown": {"band": 1.0, "comfort_weight": 4.0, "change_weight": 0.07},
}

OCCUPANCY_COMFORT = {
    "Occupied": {"band_adjust": 0.0, "comfort_multiplier": 1.2},
    "Vacant": {"band_adjust": 2.5, "comfort_multiplier": 0.35},
    "Cleaning": {"band_adjust": 0.8, "comfort_multiplier": 0.7},
    "Unknown": {"band_adjust": 0.8, "comfort_multiplier": 0.8},
}


def round_half_degree(value: float) -> float:
    return float(np.clip(round(float(value) * 2) / 2, SETPOINT_MIN, SETPOINT_MAX))


def infer_target_mode(row: pd.Series) -> str:
    mode = str(row.get("hvac_mode") or "").strip().lower()
    room_temp = _num(row.get("room_temp"), 22.0)
    setpoint = _num(row.get("setpoint"), 22.0)
    ideal_temp = _num(row.get("ideal_temp"), 22.0)
    outside_temp = _num(row.get("outside_temp"), ideal_temp)

    if mode in {"cooling", "heating"}:
        return mode
    if room_temp > setpoint + 0.4 or room_temp >= ideal_temp + 0.7 or outside_temp >= 27:
        return "cooling"
    if room_temp < setpoint - 0.4 or room_temp <= ideal_temp - 0.7 or outside_temp <= 12:
        return "heating"
    return "idle"


def estimate_power_for_setpoint(row: pd.Series, setpoint: float, target_mode: str | None = None) -> float:
    mode = (target_mode or str(row.get("hvac_mode") or "")).strip().lower()
    if mode not in MODE_POWER_MODEL:
        return 0.0

    room_temp = _num(row.get("room_temp"), np.nan)
    outside_temp = _num(row.get("outside_temp"), np.nan)
    if pd.isna(room_temp) or pd.isna(outside_temp):
        return 0.0

    if mode == "heating":
        room_gap = max(float(setpoint) - room_temp, 0.0)
        weather_gap = max(float(setpoint) - outside_temp, 0.0)
    else:
        room_gap = max(room_temp - float(setpoint), 0.0)
        weather_gap = max(outside_temp - float(setpoint), 0.0)

    config = MODE_POWER_MODEL[mode]
    size = _num(row.get("size_m2"), REFERENCE_ROOM_SIZE_M2)
    size_factor = float(np.clip(size / REFERENCE_ROOM_SIZE_M2, 0.70, 1.45))
    occupancy_factor = _occupancy_factor(row.get("occupancy_prediction") or row.get("room_state"), row.get("pir_motion"))

    raw_power = (
        config["base_w"]
        + config["room_gap_w_per_c"] * room_gap
        + config["weather_gap_w_per_c"] * weather_gap
    )
    return float(np.clip(raw_power * size_factor * occupancy_factor, 0.0, config["max_w"]))


def energy_aware_recommended_setpoint(row: pd.Series) -> float:
    """Choose the lowest-score setpoint from comfort and HVAC-energy costs.

    The objective intentionally rewards lower estimated HVAC power, but only
    inside a comfort envelope that depends on occupancy and temperature persona.
    This creates a supervised target both HGB and Transformer can learn.
    """
    target_mode = infer_target_mode(row)
    current_setpoint = round_half_degree(_num(row.get("setpoint"), 22.0))
    ideal_temp = _num(row.get("ideal_temp"), 22.0)
    persona = _normalise_persona(row.get("temperature_persona_prediction") or row.get("ac_persona"))
    occupancy = _normalise_occupancy(row.get("occupancy_prediction") or row.get("room_state"))

    if occupancy == "Vacant":
        comfort_center = 28.0 if target_mode == "cooling" else 16.5 if target_mode == "heating" else 25.0
    elif occupancy == "Cleaning" or persona == "Housekeeping":
        comfort_center = 24.0 if target_mode == "cooling" else 20.0 if target_mode == "heating" else 23.0
    else:
        comfort_center = ideal_temp

    persona_cfg = PERSONA_COMFORT.get(persona, PERSONA_COMFORT["Unknown"])
    occ_cfg = OCCUPANCY_COMFORT.get(occupancy, OCCUPANCY_COMFORT["Unknown"])
    comfort_band = persona_cfg["band"] + occ_cfg["band_adjust"]
    comfort_weight = persona_cfg["comfort_weight"] * occ_cfg["comfort_multiplier"]
    change_weight = persona_cfg["change_weight"]

    candidates = np.arange(SETPOINT_MIN, SETPOINT_MAX + 0.001, SETPOINT_STEP)
    current_power = estimate_power_for_setpoint(row, current_setpoint, target_mode)
    max_power = max(MODE_POWER_MODEL.get(target_mode, {}).get("max_w", 1.0), 1.0)

    best_score = float("inf")
    best_setpoint = current_setpoint
    for candidate in candidates:
        power = estimate_power_for_setpoint(row, candidate, target_mode)
        energy_cost = power / max_power

        comfort_gap = max(abs(float(candidate) - comfort_center) - comfort_band, 0.0)
        comfort_cost = comfort_weight * (comfort_gap ** 2)
        change_cost = change_weight * abs(float(candidate) - current_setpoint)

        # If the current setting is already near comfort and energy-neutral,
        # avoid unnecessary movement.
        if target_mode == "idle":
            energy_cost = 0.0
            change_cost *= 2.0

        score = energy_cost + comfort_cost + change_cost
        if score < best_score:
            best_score = score
            best_setpoint = float(candidate)

    # Practical HVAC controls generally accept 0.5 C setpoint steps.
    return round_half_degree(best_setpoint)


def energy_aware_recommended_setpoints(df: pd.DataFrame) -> np.ndarray:
    """Vectorized version of :func:`energy_aware_recommended_setpoint`."""
    n_rows = len(df)
    if n_rows == 0:
        return np.array([], dtype=np.float32)

    room_temp = _array_num(df.get("room_temp"), 22.0)
    setpoint = np.clip(np.round(_array_num(df.get("setpoint"), 22.0) * 2) / 2, SETPOINT_MIN, SETPOINT_MAX)
    ideal_temp = _array_num(df.get("ideal_temp"), 22.0)
    outside_temp = _array_num(df.get("outside_temp"), ideal_temp)
    size_m2 = _array_num(df.get("size_m2"), REFERENCE_ROOM_SIZE_M2)
    pir_motion = _array_num(df.get("pir_motion"), 0.0)

    hvac_mode = _string_array(df.get("hvac_mode"), "unknown")
    target_mode = hvac_mode.copy()
    known_mode = np.isin(target_mode, ["cooling", "heating"])
    cooling_inferred = (
        (room_temp > setpoint + 0.4)
        | (room_temp >= ideal_temp + 0.7)
        | (outside_temp >= 27)
    )
    heating_inferred = (
        (room_temp < setpoint - 0.4)
        | (room_temp <= ideal_temp - 0.7)
        | (outside_temp <= 12)
    )
    target_mode[~known_mode & cooling_inferred] = "cooling"
    target_mode[~known_mode & ~cooling_inferred & heating_inferred] = "heating"
    target_mode[~np.isin(target_mode, ["cooling", "heating"])] = "idle"

    occupancy = _string_array(
        df.get("occupancy_prediction", df.get("room_state")),
        "Unknown",
        preserve_case=True,
    )
    persona = _string_array(
        df.get("temperature_persona_prediction", df.get("ac_persona")),
        "Unknown",
        preserve_case=True,
    )
    persona = np.where(np.isin(persona, list(PERSONA_COMFORT)), persona, "Unknown")
    occupancy = np.where(np.isin(occupancy, list(OCCUPANCY_COMFORT)), occupancy, "Unknown")

    comfort_center = ideal_temp.copy()
    vacant = occupancy == "Vacant"
    cleaning = (occupancy == "Cleaning") | (persona == "Housekeeping")
    comfort_center[vacant & (target_mode == "cooling")] = 28.0
    comfort_center[vacant & (target_mode == "heating")] = 16.5
    comfort_center[vacant & (target_mode == "idle")] = 25.0
    comfort_center[cleaning & (target_mode == "cooling")] = 24.0
    comfort_center[cleaning & (target_mode == "heating")] = 20.0
    comfort_center[cleaning & (target_mode == "idle")] = 23.0

    comfort_band = np.zeros(n_rows, dtype=np.float32)
    comfort_weight = np.zeros(n_rows, dtype=np.float32)
    change_weight = np.zeros(n_rows, dtype=np.float32)
    for key, cfg in PERSONA_COMFORT.items():
        mask = persona == key
        comfort_band[mask] = cfg["band"]
        comfort_weight[mask] = cfg["comfort_weight"]
        change_weight[mask] = cfg["change_weight"]
    for key, cfg in OCCUPANCY_COMFORT.items():
        mask = occupancy == key
        comfort_band[mask] += cfg["band_adjust"]
        comfort_weight[mask] *= cfg["comfort_multiplier"]

    occupancy_factor = _occupancy_factor_array(occupancy, pir_motion)
    size_factor = np.clip(size_m2 / REFERENCE_ROOM_SIZE_M2, 0.70, 1.45)
    max_power = np.ones(n_rows, dtype=np.float32)
    max_power[target_mode == "heating"] = MODE_POWER_MODEL["heating"]["max_w"]
    max_power[target_mode == "cooling"] = MODE_POWER_MODEL["cooling"]["max_w"]

    candidates = np.arange(SETPOINT_MIN, SETPOINT_MAX + 0.001, SETPOINT_STEP)
    best_score = np.full(n_rows, np.inf, dtype=np.float32)
    best_setpoint = setpoint.copy()
    for candidate in candidates:
        power = _estimate_power_array(
            target_mode,
            room_temp,
            outside_temp,
            size_factor,
            occupancy_factor,
            candidate,
        )
        energy_cost = power / np.maximum(max_power, 1.0)
        energy_cost[target_mode == "idle"] = 0.0

        comfort_gap = np.maximum(np.abs(candidate - comfort_center) - comfort_band, 0.0)
        comfort_cost = comfort_weight * (comfort_gap ** 2)
        movement = change_weight * np.abs(candidate - setpoint)
        movement[target_mode == "idle"] *= 2.0
        score = energy_cost + comfort_cost + movement
        replace = score < best_score
        best_score[replace] = score[replace]
        best_setpoint[replace] = candidate

    return np.clip(np.round(best_setpoint * 2) / 2, SETPOINT_MIN, SETPOINT_MAX).astype("float32")


def target_diagnostics(row: pd.Series, recommended_setpoint: float) -> dict[str, float | str]:
    target_mode = infer_target_mode(row)
    current_setpoint = round_half_degree(_num(row.get("setpoint"), 22.0))
    current_power = estimate_power_for_setpoint(row, current_setpoint, target_mode)
    recommended_power = estimate_power_for_setpoint(row, recommended_setpoint, target_mode)
    ideal_temp = _num(row.get("ideal_temp"), 22.0)
    return {
        "target_mode": target_mode,
        "current_power_w": current_power,
        "target_power_w": recommended_power,
        "target_power_saving_w": current_power - recommended_power,
        "current_comfort_gap_c": abs(current_setpoint - ideal_temp),
        "target_comfort_gap_c": abs(float(recommended_setpoint) - ideal_temp),
    }


def _normalise_persona(value) -> str:
    persona = str(value or "Unknown")
    return persona if persona in PERSONA_COMFORT else "Unknown"


def _normalise_occupancy(value) -> str:
    occupancy = str(value or "Unknown")
    return occupancy if occupancy in OCCUPANCY_COMFORT else "Unknown"


def _occupancy_factor(room_state, pir_motion) -> float:
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


def _num(value, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return float(fallback if pd.isna(numeric) else numeric)


def _array_num(values, fallback) -> np.ndarray:
    if values is None:
        return np.asarray(fallback, dtype=np.float32)
    series = pd.to_numeric(values, errors="coerce")
    if np.isscalar(fallback):
        series = series.fillna(float(fallback))
    else:
        series = series.fillna(pd.Series(fallback, index=series.index))
    return series.to_numpy(dtype=np.float32)


def _string_array(values, fallback: str, preserve_case: bool = False) -> np.ndarray:
    if values is None:
        arr = np.array([fallback], dtype=object)
    else:
        arr = pd.Series(values).fillna(fallback).astype(str).to_numpy(dtype=object)
    if preserve_case:
        return arr
    return np.char.lower(arr.astype(str))


def _occupancy_factor_array(occupancy: np.ndarray, pir_motion: np.ndarray) -> np.ndarray:
    state = np.char.lower(occupancy.astype(str))
    motion = pir_motion > 0
    factor = np.where(motion, 1.0, 0.85).astype(np.float32)
    factor[state == "vacant"] = 0.55
    factor[state == "cleaning"] = 0.75
    occupied = state == "occupied"
    factor[occupied] = np.where(motion[occupied], 1.10, 0.95)
    return factor


def _estimate_power_array(
    target_mode: np.ndarray,
    room_temp: np.ndarray,
    outside_temp: np.ndarray,
    size_factor: np.ndarray,
    occupancy_factor: np.ndarray,
    setpoint: float,
) -> np.ndarray:
    power = np.zeros_like(room_temp, dtype=np.float32)

    heating = target_mode == "heating"
    if heating.any():
        config = MODE_POWER_MODEL["heating"]
        room_gap = np.maximum(setpoint - room_temp[heating], 0.0)
        weather_gap = np.maximum(setpoint - outside_temp[heating], 0.0)
        raw = config["base_w"] + config["room_gap_w_per_c"] * room_gap + config["weather_gap_w_per_c"] * weather_gap
        power[heating] = np.clip(raw * size_factor[heating] * occupancy_factor[heating], 0.0, config["max_w"])

    cooling = target_mode == "cooling"
    if cooling.any():
        config = MODE_POWER_MODEL["cooling"]
        room_gap = np.maximum(room_temp[cooling] - setpoint, 0.0)
        weather_gap = np.maximum(outside_temp[cooling] - setpoint, 0.0)
        raw = config["base_w"] + config["room_gap_w_per_c"] * room_gap + config["weather_gap_w_per_c"] * weather_gap
        power[cooling] = np.clip(raw * size_factor[cooling] * occupancy_factor[cooling], 0.0, config["max_w"])

    return power
