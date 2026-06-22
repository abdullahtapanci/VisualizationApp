"""Energy-consumption calculation helpers.

Reads DimmerGraphs.xlsx once at import time, then exposes a single
function ``compute_energy_for_room(...)`` that the Flask layer can call.

Lamp-type rules (matched to the recorded ``Value`` column in lightning_data):

  - Dimmable LED  : hidden_top, dinner_table, table, bed_left, bed_right
                    Power is taken from the LED columns of the dimmer table
                    using the recorded dimmer level as the lookup key.
  - Non-dimmable
    bulb          : closet, corridor_left, corridor_right, shower, cabinet,
                    sink. The recorded value is binary (0 or 80), so the
                    bulb runs at full power (level 80) whenever it is on.

Each row in lightning_data represents a 5-minute sample, so each on-row
contributes ``power * (5 / 60)`` Wh of energy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.data_loader import get_db_connection


BASE_DIR = Path(__file__).resolve().parent.parent
DIMMER_FILE = BASE_DIR / "EnergyConsumption" / "DimmerGraphs.xlsx"

SAMPLE_MINUTES = 5
HOURS_PER_SAMPLE = SAMPLE_MINUTES / 60.0

DIMMABLE_LAMPS = {"hidden_top", "dinner_table", "table",
                  "bed_left", "bed_right"}
NON_DIMMABLE_LAMPS = {"closet", "corridor_left", "corridor_right",
                      "shower", "cabinet", "sink"}

# The application records dimmer levels on a 0..100 scale, but the IoT
# device (and DimmerGraphs.xlsx) uses the native 0..254 range. Convert
# before looking up power.
APP_SCALE_MAX = 100
DEVICE_SCALE_MAX = 254
MAX_LEVEL = 80   # operational max level recorded in lightning_data (app scale)


def _app_to_device_level(level: int) -> int:
    """Convert a 0..100 application level to the 0..254 IoT/device range."""
    return int(round(level * DEVICE_SCALE_MAX / APP_SCALE_MAX))


def _load_dimmer_table(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None)
    # The numeric block starts at row index 7, columns 2..8.
    data = raw.iloc[7:, 2:9].copy()
    data.columns = ["Level", "Lux_LED", "V_LED", "I_LED_uA",
                    "Lux_Bulp", "V_Bulp", "I_Bulp_mA"]
    data = data.dropna(subset=["Level"]).reset_index(drop=True)
    data["Level"] = data["Level"].astype(int)
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    # LED current is microamps, bulb current is milliamps.
    data["P_LED_W"] = data["V_LED"] * data["I_LED_uA"] / 1_000_000.0
    data["P_Bulp_W"] = data["V_Bulp"] * data["I_Bulp_mA"] / 1_000.0
    return data


_DIMMER_TABLE = _load_dimmer_table(DIMMER_FILE)
_LED_POWER = dict(zip(_DIMMER_TABLE["Level"].astype(int),
                      _DIMMER_TABLE["P_LED_W"]))
_BULB_POWER = dict(zip(_DIMMER_TABLE["Level"].astype(int),
                       _DIMMER_TABLE["P_Bulp_W"]))
_LED_MAX = _LED_POWER[_app_to_device_level(MAX_LEVEL)]
_BULB_MAX = _BULB_POWER[_app_to_device_level(MAX_LEVEL)]


def _row_powers(lamp: str, level: int) -> tuple[float, float]:
    """Return (actual_W, max_W) for a single lamp/level row.

    ``level`` arrives on the application's 0..100 scale and is converted to
    the device's native 0..254 range before being used as a lookup key into
    the dimmer table.
    """
    if lamp in DIMMABLE_LAMPS:
        device_level = _app_to_device_level(level)
        return _LED_POWER.get(device_level, np.nan), _LED_MAX
    if lamp in NON_DIMMABLE_LAMPS:
        # Non-dimmable lamps are on/off only; full bulb power when on.
        return _BULB_MAX, _BULB_MAX
    return np.nan, np.nan


def compute_energy_for_room(room_number: int,
                            start_timestamp: str,
                            end_timestamp: str) -> dict:
    """Compute actual and max-level energy use for a room within an interval.

    Returns a dict with the per-row dataframe plus aggregate totals and a
    per-lamp breakdown so the API layer can render charts on top of it.
    """
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT timestamp, lamp_location, "Value" AS value
            FROM lightning_data
            WHERE room_number = ?
              AND timestamp >= ?
              AND timestamp <  ?
              AND lamp_location <> 'none'
              AND "Value" > 0
            """,
            conn,
            params=(room_number, start_timestamp, end_timestamp),
        )
    finally:
        conn.close()

    if df.empty:
        return {"empty": True, "df": df}

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"])
    df["level"] = df["value"].astype(int)

    powers = df.apply(lambda r: _row_powers(r["lamp_location"], r["level"]),
                      axis=1, result_type="expand")
    powers.columns = ["P_actual_W", "P_max_W"]
    df = pd.concat([df, powers], axis=1)
    df["E_actual_Wh"] = df["P_actual_W"] * HOURS_PER_SAMPLE
    df["E_max_Wh"] = df["P_max_W"] * HOURS_PER_SAMPLE

    total_actual = float(df["E_actual_Wh"].sum())
    total_max = float(df["E_max_Wh"].sum())
    saved = total_max - total_actual
    saved_pct = (100.0 * saved / total_max) if total_max else 0.0

    by_lamp = (df.groupby("lamp_location")[["E_actual_Wh", "E_max_Wh"]]
                 .sum()
                 .sort_values("E_max_Wh", ascending=False))

    # Dimmable-only aggregates (savings live entirely on dimmable lamps —
    # non-dimmable bulbs are on/off so their actual == max).
    dimmable_only = by_lamp[by_lamp.index.isin(DIMMABLE_LAMPS)]
    dim_actual = float(dimmable_only["E_actual_Wh"].sum())
    dim_max = float(dimmable_only["E_max_Wh"].sum())
    dim_saved = dim_max - dim_actual
    dim_saved_pct = (100.0 * dim_saved / dim_max) if dim_max else 0.0
    dimmable_summary = {
        "actual_wh": dim_actual,
        "max_wh": dim_max,
        "saved_wh": dim_saved,
        "saved_pct": dim_saved_pct,
        "n_lamps": int(len(dimmable_only)),
    }
    by_lamp_records = [
        {
            "lamp": idx,
            "actual_wh": float(r["E_actual_Wh"]),
            "max_wh": float(r["E_max_Wh"]),
            "saved_pct": (100.0 * (r["E_max_Wh"] - r["E_actual_Wh"]) /
                          r["E_max_Wh"]) if r["E_max_Wh"] else 0.0,
        }
        for idx, r in by_lamp.iterrows()
    ]

    return {
        "empty": False,
        "df": df,
        "by_lamp": by_lamp,
        "summary": {
            "actual_wh": total_actual,
            "max_wh": total_max,
            "saved_wh": saved,
            "saved_pct": saved_pct,
            "n_rows": int(len(df)),
        },
        "dimmable_summary": dimmable_summary,
        "by_lamp_records": by_lamp_records,
    }
