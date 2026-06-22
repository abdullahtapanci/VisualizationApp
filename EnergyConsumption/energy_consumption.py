"""
Energy consumption calculator for the hotel-environment lighting data.

The script asks the user for a room number and a time interval, then computes
the energy used by every lamp in that room during the interval under two
scenarios:

  1. Actual    - power is looked up from DimmerGraphs.xlsx using the recorded
                 dimmer level (the `Value` column of lightningData.csv).
  2. Max-level - assumes every lamp that is on is running at full level
                 (level 80, the operational max in the dataset). This shows
                 the consumption that would occur without any dimming.

Lamp classification
-------------------
  Dimmable LEDs  : hidden_top, dinner_table, table, bed_left, bed_right
  Non-dimmable
  bulbs          : closet, corridor_left, corridor_right, shower, cabinet, sink

The lightning log records one sample every 5 minutes, so each row contributes
power * (5 / 60) Wh of energy.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DIMMER_FILE = BASE_DIR / "EnergyConsumption" / "DimmerGraphs.xlsx"
LIGHTNING_FILE = BASE_DIR / "Data" / "lightningData.csv"

SAMPLE_MINUTES = 5
HOURS_PER_SAMPLE = SAMPLE_MINUTES / 60.0

DIMMABLE_LAMPS = {"hidden_top", "dinner_table", "table", "bed_left", "bed_right"}
NON_DIMMABLE_LAMPS = {"closet", "corridor_left", "corridor_right",
                      "shower", "cabinet", "sink"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dimmer_table(path: Path) -> pd.DataFrame:
    """Read the dimmer/bulb measurement table and compute power per level."""
    raw = pd.read_excel(path, sheet_name=0, header=None)
    # The numeric block starts at row index 7, columns 2..8.
    data = raw.iloc[7:, 2:9].copy()
    data.columns = ["Level", "Lux_LED", "V_LED", "I_LED_uA",
                    "Lux_Bulp", "V_Bulp", "I_Bulp_mA"]
    data = data.dropna(subset=["Level"]).reset_index(drop=True)
    data["Level"] = data["Level"].astype(int)
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Power in watts. LED current is in uA, bulb current is in mA.
    data["P_LED_W"] = data["V_LED"] * data["I_LED_uA"] / 1_000_000.0
    data["P_Bulp_W"] = data["V_Bulp"] * data["I_Bulp_mA"] / 1_000.0
    return data


def power_lookup(dimmer: pd.DataFrame) -> tuple[dict[int, float], dict[int, float]]:
    """Return {level: power_W} dictionaries for LED and bulb."""
    led = dict(zip(dimmer["Level"].astype(int), dimmer["P_LED_W"]))
    bulb = dict(zip(dimmer["Level"].astype(int), dimmer["P_Bulp_W"]))
    return led, bulb


def load_lightning(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "room_number",
                                    "lamp_location", "Value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------
def prompt_inputs(rooms_available: list[int],
                  ts_min: pd.Timestamp,
                  ts_max: pd.Timestamp) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    print("=" * 62)
    print(" Hotel lighting energy-consumption calculator")
    print("=" * 62)
    print(f"Rooms available  : {rooms_available[0]} .. {rooms_available[-1]}")
    print(f"Data covers      : {ts_min}  ->  {ts_max}")
    print()

    while True:
        try:
            room = int(input("Room number: ").strip())
            if room in rooms_available:
                break
            print(f"  ! room {room} not in dataset. Try again.")
        except ValueError:
            print("  ! please enter an integer.")

    fmt_help = "format YYYY-MM-DD HH:MM   (HH:MM optional)"
    start = _ask_datetime(f"Start time ({fmt_help}): ", ts_min, ts_max)
    end = _ask_datetime(f"End   time ({fmt_help}): ", ts_min, ts_max)
    if end <= start:
        print("  ! end <= start, swapping.")
        start, end = end, start
    return room, start, end


def _ask_datetime(prompt: str, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.Timestamp:
    while True:
        raw = input(prompt).strip()
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                ts = pd.Timestamp(datetime.strptime(raw, fmt))
                if ts < lo or ts > hi:
                    print(f"  ! outside data range [{lo} .. {hi}]")
                    break
                return ts
            except ValueError:
                continue
        else:
            print("  ! could not parse; try e.g. 2022-01-22 08:00")


# ---------------------------------------------------------------------------
# Calculation
# ---------------------------------------------------------------------------
def compute_energy(df: pd.DataFrame,
                   led_power: dict[int, float],
                   bulb_power: dict[int, float]) -> pd.DataFrame:
    """Add per-row power and energy (Wh) columns for both scenarios."""
    df = df.copy()
    df = df[df["lamp_location"] != "none"]            # rows with no lamp
    df = df[df["Value"] > 0]                           # lamp must be on

    led_max = led_power[80]
    bulb_max = bulb_power[80]

    def row_power(row):
        lamp = row["lamp_location"]
        level = int(row["Value"])
        if lamp in DIMMABLE_LAMPS:
            return led_power.get(level, np.nan), led_max
        if lamp in NON_DIMMABLE_LAMPS:
            # Non-dimmable: only on/off behaviour, full bulb power when on.
            return bulb_max, bulb_max
        return np.nan, np.nan

    powers = df.apply(row_power, axis=1, result_type="expand")
    powers.columns = ["P_actual_W", "P_max_W"]
    df = pd.concat([df, powers], axis=1)
    df["E_actual_Wh"] = df["P_actual_W"] * HOURS_PER_SAMPLE
    df["E_max_Wh"] = df["P_max_W"] * HOURS_PER_SAMPLE
    return df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(df: pd.DataFrame, room: int,
                 start: pd.Timestamp, end: pd.Timestamp) -> dict:
    total_actual = df["E_actual_Wh"].sum()
    total_max = df["E_max_Wh"].sum()
    saved = total_max - total_actual
    pct = 100 * saved / total_max if total_max else 0.0

    print()
    print("=" * 62)
    print(f" Report - Room {room}")
    print(f" Interval     : {start}  ->  {end}")
    print(f" Duration     : {(end - start)}")
    print("=" * 62)
    print(f" Actual energy used     : {total_actual:10.2f} Wh "
          f"({total_actual/1000:.3f} kWh)")
    print(f" Energy at max-level    : {total_max:10.2f} Wh "
          f"({total_max/1000:.3f} kWh)")
    print(f" Savings from dimming   : {saved:10.2f} Wh  ({pct:5.1f} %)")
    print()
    print(" Breakdown by lamp:")
    by_lamp = df.groupby("lamp_location")[["E_actual_Wh", "E_max_Wh"]].sum()
    by_lamp = by_lamp.sort_values("E_max_Wh", ascending=False)
    if by_lamp.empty:
        print("   (no lamp activity in the chosen interval)")
    else:
        print(f"   {'Lamp':<16} {'Actual (Wh)':>13} {'Max (Wh)':>13} "
              f"{'Saved %':>9}")
        for lamp, r in by_lamp.iterrows():
            sv = (r['E_max_Wh'] - r['E_actual_Wh']) / r['E_max_Wh'] * 100 \
                if r['E_max_Wh'] else 0.0
            print(f"   {lamp:<16} {r['E_actual_Wh']:>13.2f} "
                  f"{r['E_max_Wh']:>13.2f} {sv:>8.1f}%")
    return {"actual": total_actual, "max": total_max,
            "by_lamp": by_lamp}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(df: pd.DataFrame, room: int,
               start: pd.Timestamp, end: pd.Timestamp,
               by_lamp: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    title_suffix = f"Room {room}  |  {start}  ->  {end}"

    # 1. Total actual vs max ------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    totals = [df["E_actual_Wh"].sum(), df["E_max_Wh"].sum()]
    ax.bar(["Actual", "Max-level"], totals,
           color=["#2c7fb8", "#e36b6b"])
    for i, v in enumerate(totals):
        ax.text(i, v, f"{v:.1f} Wh", ha="center", va="bottom")
    ax.set_ylabel("Energy (Wh)")
    ax.set_title("Total energy: actual vs max-level\n" + title_suffix)
    fig.tight_layout()
    fig.savefig(out_dir / "01_total_energy.png", dpi=110)
    plt.close(fig)

    # 2. Per-lamp comparison ------------------------------------------------
    if not by_lamp.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(by_lamp))
        w = 0.4
        ax.bar(x - w / 2, by_lamp["E_actual_Wh"], w,
               label="Actual", color="#2c7fb8")
        ax.bar(x + w / 2, by_lamp["E_max_Wh"], w,
               label="Max-level", color="#e36b6b")
        ax.set_xticks(x)
        ax.set_xticklabels(by_lamp.index, rotation=30, ha="right")
        ax.set_ylabel("Energy (Wh)")
        ax.set_title("Per-lamp energy: actual vs max-level\n" + title_suffix)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "02_per_lamp.png", dpi=110)
        plt.close(fig)

    # 3. Hourly time series -------------------------------------------------
    if not df.empty:
        ts = df.set_index("timestamp")[["E_actual_Wh", "E_max_Wh"]]
        # Pick bucket size that gives a readable line.
        span_hours = (end - start).total_seconds() / 3600.0
        rule = "H" if span_hours <= 72 else "D"
        bucket = ts.resample(rule).sum()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(bucket.index, bucket["E_actual_Wh"],
                marker="o", label="Actual", color="#2c7fb8")
        ax.plot(bucket.index, bucket["E_max_Wh"],
                marker="s", label="Max-level", color="#e36b6b")
        ax.fill_between(bucket.index, bucket["E_actual_Wh"],
                        bucket["E_max_Wh"], alpha=0.15, color="#e36b6b",
                        label="Savings from dimming")
        ax.set_ylabel(f"Energy per {'hour' if rule=='H' else 'day'} (Wh)")
        ax.set_title(f"Energy over time ({'hourly' if rule=='H' else 'daily'})\n"
                     + title_suffix)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / "03_timeseries.png", dpi=110)
        plt.close(fig)

    # 4. Dimmable vs non-dimmable share ------------------------------------
    df_typed = df.copy()
    df_typed["lamp_type"] = np.where(
        df_typed["lamp_location"].isin(DIMMABLE_LAMPS),
        "Dimmable LED", "Non-dimmable bulb")
    grp = df_typed.groupby("lamp_type")[["E_actual_Wh", "E_max_Wh"]].sum()
    if not grp.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(grp))
        w = 0.4
        ax.bar(x - w / 2, grp["E_actual_Wh"], w,
               label="Actual", color="#2c7fb8")
        ax.bar(x + w / 2, grp["E_max_Wh"], w,
               label="Max-level", color="#e36b6b")
        ax.set_xticks(x)
        ax.set_xticklabels(grp.index)
        ax.set_ylabel("Energy (Wh)")
        ax.set_title("LED vs bulb contribution\n" + title_suffix)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "04_lamp_type.png", dpi=110)
        plt.close(fig)

    print(f"\n Graphs written to: {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(" Loading data ...")
    dimmer = load_dimmer_table(DIMMER_FILE)
    led_power, bulb_power = power_lookup(dimmer)
    light = load_lightning(LIGHTNING_FILE)

    rooms = sorted(light["room_number"].unique().tolist())
    ts_min, ts_max = light["timestamp"].min(), light["timestamp"].max()

    room, start, end = prompt_inputs(rooms, ts_min, ts_max)

    mask = ((light["room_number"] == room)
            & (light["timestamp"] >= start)
            & (light["timestamp"] < end))
    sub = light.loc[mask].copy()

    if sub.empty:
        print("\n No samples for that room/interval - nothing to compute.")
        sys.exit(0)

    energy = compute_energy(sub, led_power, bulb_power)
    summary = print_report(energy, room, start, end)

    out_dir = Path(__file__).resolve().parent / "output" / f"room{room}"
    make_plots(energy, room, start, end, summary["by_lamp"], out_dir)
    plt.show()


if __name__ == "__main__":
    main()
