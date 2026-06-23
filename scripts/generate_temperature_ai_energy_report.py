#!/usr/bin/env python3
"""Generate HVAC energy comparisons with current, AI, and max baseline.

The output mirrors the lighting energy report: one selected day, one selected
month, and all available local database days for a room. The AI usage is based
on the selected temperature recommendation model.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "temperature_ai_energy_report"
SAMPLE_MINUTES = 5


def configure_plot_cache(output_dir: Path) -> None:
    cache_dir = output_dir / ".plot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot current vs AI vs max-baseline HVAC energy.",
    )
    parser.add_argument("--room", type=int, default=1, help="Room number to analyze.")
    parser.add_argument("--day", default="2022-01-22", help="Day to plot, YYYY-MM-DD.")
    parser.add_argument("--month", default="2022-01", help="Month to plot, YYYY-MM.")
    parser.add_argument(
        "--model-type",
        default="hist_gradient_boosting",
        choices=["auto", "hist_gradient_boosting", "transformer"],
        help="Temperature recommendation model to use for AI energy.",
    )
    parser.add_argument("--lookback-hours", type=int, default=2)
    parser.add_argument(
        "--scope",
        default="day,month,all",
        help="Comma-separated scopes to compute: day, month, all.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only-plots", action="store_true")
    return parser.parse_args()


def safe_pct(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def day_bounds(day: str):
    import pandas as pd

    start = pd.Timestamp(day)
    end_inclusive = start + pd.Timedelta(days=1) - pd.Timedelta(minutes=SAMPLE_MINUTES)
    return start, end_inclusive


def month_days(month: str) -> list[str]:
    import pandas as pd

    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end - pd.Timedelta(days=1), freq="D")]


def available_data_days(room: int) -> list[str]:
    import pandas as pd

    from backend.data_loader import get_db_connection

    conn = get_db_connection()
    try:
        rows = pd.read_sql_query(
            """
            SELECT DISTINCT date(timestamp) AS day
            FROM temperature_data
            WHERE room_number = ?
            ORDER BY day
            """,
            conn,
            params=(room,),
        )
    finally:
        conn.close()
    return rows["day"].dropna().astype(str).tolist()


def raw_records_path(output_dir: Path, room: int, day: str, model_type: str) -> Path:
    model_slug = model_type.replace("_", "-")
    return output_dir / "raw_daily_records" / f"room_{room}_{day}_{model_slug}.csv"


def _prepare_temperature_frame(room: int, start_ts, end_ts, lookback_hours: int):
    import pandas as pd

    from backend.data_loader import get_db_connection

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
                room,
                history_start.strftime("%Y-%m-%d %H:%M:%S"),
                end_ts.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
    finally:
        conn.close()

    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    for col in ["floor", "size_m2", "outside_temp", "room_temp", "setpoint", "ideal_temp", "pir_motion"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["guest_id"] = pd.to_numeric(df["guest_id"], errors="coerce")
    return df


def _predict_hgb_setpoint(row, model_bundle, occupancy: str, persona: str) -> float:
    import pandas as pd

    from backend.prediction_services import _build_tempreture_recomendation_hgb_frame

    timestamp = pd.Timestamp(row.get("timestamp"))
    frame = _build_tempreture_recomendation_hgb_frame(row, timestamp, model_bundle, occupancy, persona)
    metadata = model_bundle.get("metadata") or {}
    setpoint_min = float(metadata.get("setpoint_min", 16.0))
    setpoint_max = float(metadata.get("setpoint_max", 28.0))
    recommended = float(model_bundle["model"].predict(frame)[0])
    return round(min(max(recommended, setpoint_min), setpoint_max) * 2) / 2


def _predict_transformer_setpoints(df, target_indices, transformer_bundle):
    import numpy as np
    import pandas as pd

    from backend.prediction_services import (
        _build_tempreture_recomendation_step,
        _encode_tempreture_recomendation_step,
    )

    seq_len = transformer_bundle["seq_len"]
    encoded_sequences = []
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
            row_timestamp = pd.Timestamp(row.get("timestamp"))
            step = _build_tempreture_recomendation_step(row, row_timestamp, occupancy, persona)
            encoded_steps.append(_encode_tempreture_recomendation_step(step, transformer_bundle))
        encoded_sequences.append(encoded_steps)

    torch = transformer_bundle["torch"]
    tensor = torch.from_numpy(np.array(encoded_sequences, dtype=np.float32))
    predictions = []
    with torch.no_grad():
        for start in range(0, len(tensor), 512):
            scaled = transformer_bundle["model"](tensor[start:start + 512]).cpu().numpy()
            predictions.extend(scaled.tolist())

    setpoint_min = transformer_bundle["setpoint_min"]
    setpoint_max = transformer_bundle["setpoint_max"]
    return [
        round(min(max(float(value) * (setpoint_max - setpoint_min) + setpoint_min, setpoint_min), setpoint_max) * 2) / 2
        for value in predictions
    ]


def _max_wh_for_group(group) -> float:
    from backend.hvac_energy import HOURS_PER_SAMPLE, RATED_POWER_W

    heating = int((group["current_energy_mode"] == "heating").sum())
    cooling = int((group["current_energy_mode"] == "cooling").sum())
    if heating == 0 and cooling == 0:
        dominant_mode = "heating"
    elif heating >= cooling:
        dominant_mode = "heating"
    else:
        dominant_mode = "cooling"
    return float(RATED_POWER_W[dominant_mode] * len(group) * HOURS_PER_SAMPLE)


def compute_one_day(
    room: int,
    day: str,
    model_type: str,
    lookback_hours: int,
    output_dir: Path,
    force: bool,
):
    import pandas as pd

    from backend.hvac_energy import HOURS_PER_SAMPLE
    from backend.prediction_services import (
        _energy_estimation_mode,
        _load_tempreture_recomendation_hgb,
        _load_tempreture_recomendation_transformer,
    )
    from backend.hvac_energy import _estimate_power

    out_path = raw_records_path(output_dir, room, day, model_type)
    if out_path.exists() and not force:
        print(f"[skip] {day}: using saved {out_path}", flush=True)
        return pd.read_csv(out_path)

    start_ts, end_ts = day_bounds(day)
    df = _prepare_temperature_frame(room, start_ts, end_ts, lookback_hours)
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(out_path, index=False)
        return pd.DataFrame()

    target_mask = (df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)
    target_indices = df.index[target_mask].to_numpy()
    if len(target_indices) == 0:
        return pd.DataFrame()

    resolved_model = model_type
    if resolved_model == "auto":
        resolved_model = "hist_gradient_boosting"

    if resolved_model == "hist_gradient_boosting":
        model_bundle = _load_tempreture_recomendation_hgb()
        if model_bundle is None:
            raise RuntimeError("Temperature HGB model was not found.")
        recommended_setpoints = []
        for idx in target_indices:
            row = df.iloc[idx]
            recommended_setpoints.append(
                _predict_hgb_setpoint(
                    row,
                    model_bundle,
                    str(row.get("room_state") or "Unknown"),
                    str(row.get("ac_persona") or "Unknown"),
                )
            )
        served_model = "HistGradientBoostingRegressor"
    else:
        transformer_bundle = _load_tempreture_recomendation_transformer()
        if transformer_bundle is None:
            raise RuntimeError("Temperature Transformer model was not found.")
        recommended_setpoints = _predict_transformer_setpoints(df, target_indices, transformer_bundle)
        served_model = "TransformerRegressor"

    records = []
    for idx, recommended_setpoint in zip(target_indices, recommended_setpoints):
        row = df.iloc[idx].copy()
        room_temp = float(row.get("room_temp") or 0.0)
        outside_temp = float(row.get("outside_temp") or 0.0)
        current_setpoint = float(row.get("setpoint") or 0.0)
        current_energy_mode = _energy_estimation_mode(
            room_temp,
            current_setpoint,
            outside_temp,
            str(row.get("hvac_mode") or "Unknown"),
        )
        current_row = row.copy()
        current_row["hvac_mode"] = current_energy_mode
        current_power_w = float(_estimate_power(current_row))

        recommended_row = row.copy()
        recommended_row["setpoint"] = recommended_setpoint
        recommended_energy_mode = _energy_estimation_mode(
            room_temp,
            recommended_setpoint,
            outside_temp,
            current_energy_mode,
        )
        recommended_row["hvac_mode"] = recommended_energy_mode
        recommended_power_w = float(_estimate_power(recommended_row))

        records.append(
            {
                "timestamp": pd.Timestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                "hvac_mode": str(row.get("hvac_mode") or "Unknown"),
                "room_state": str(row.get("room_state") or "Unknown"),
                "temperature_persona": str(row.get("ac_persona") or "Unknown"),
                "room_temp": room_temp,
                "outside_temp": outside_temp,
                "current_setpoint": current_setpoint,
                "recommended_setpoint": recommended_setpoint,
                "current_energy_mode": current_energy_mode,
                "recommended_energy_mode": recommended_energy_mode,
                "current_power_w": current_power_w,
                "recommended_power_w": recommended_power_w,
                "current_wh": current_power_w * HOURS_PER_SAMPLE,
                "recommended_wh": recommended_power_w * HOURS_PER_SAMPLE,
                "model": served_model,
            }
        )

    out = pd.DataFrame(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[saved] {out_path} ({len(out)} rows)", flush=True)
    return out


def summarize_records(records, period: str) -> dict:
    import pandas as pd

    if records is None or len(records) == 0:
        return {
            "period": period,
            "current_wh": 0.0,
            "ai_wh": 0.0,
            "max_wh": 0.0,
            "current_saved_vs_max_wh": 0.0,
            "ai_saved_vs_max_wh": 0.0,
            "ai_saving_vs_current_wh": 0.0,
            "current_saved_vs_max_pct": 0.0,
            "ai_saved_vs_max_pct": 0.0,
            "ai_saving_vs_current_pct": 0.0,
            "n_rows": 0,
        }

    df = pd.DataFrame(records)
    current = float(df["current_wh"].sum())
    ai = float(df["recommended_wh"].sum())
    max_wh = _max_wh_for_group(df)
    current_saved = max_wh - current
    ai_saved = max_wh - ai
    ai_saving_vs_current = current - ai
    return {
        "period": period,
        "current_wh": current,
        "ai_wh": ai,
        "max_wh": max_wh,
        "current_saved_vs_max_wh": current_saved,
        "ai_saved_vs_max_wh": ai_saved,
        "ai_saving_vs_current_wh": ai_saving_vs_current,
        "current_saved_vs_max_pct": safe_pct(current_saved, max_wh),
        "ai_saved_vs_max_pct": safe_pct(ai_saved, max_wh),
        "ai_saving_vs_current_pct": safe_pct(ai_saving_vs_current, current),
        "n_rows": int(len(df)),
    }


def add_hour_column(df):
    import pandas as pd

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["hour"] = out["timestamp"].dt.floor("h")
    return out


def aggregate_hourly(df):
    import pandas as pd

    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [summarize_records(group, pd.Timestamp(hour).strftime("%Y-%m-%d %H:%M:%S")) for hour, group in add_hour_column(df).groupby("hour")]
    )


def aggregate_days(daily_frames: dict[str, object]):
    import pandas as pd

    return pd.DataFrame([summarize_records(frame, day) for day, frame in daily_frames.items()])


def save_summary_csv(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[saved] {path}", flush=True)


def plot_grouped_energy(summary_df, title: str, path: Path, x_col: str = "period") -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if summary_df.empty:
        return
    labels = summary_df[x_col].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.26
    fig_width = max(10, min(22, len(labels) * 0.55))
    fig, ax = plt.subplots(figsize=(fig_width, 7), dpi=180)
    ax.bar(x - width, summary_df["current_wh"], width, label="Current usage", color="#2563eb")
    ax.bar(x, summary_df["ai_wh"], width, label="AI recommendation", color="#16a34a")
    ax.bar(x + width, summary_df["max_wh"], width, label="Max baseline", color="#9ca3af")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_ylabel("Energy used (Wh)")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.legend(ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}", flush=True)


def plot_savings(summary_df, title: str, path: Path, x_col: str = "period") -> None:
    import matplotlib.pyplot as plt

    if summary_df.empty:
        return
    labels = summary_df[x_col].astype(str).tolist()
    fig_width = max(10, min(22, len(labels) * 0.55))
    fig, ax1 = plt.subplots(figsize=(fig_width, 7), dpi=180)
    ax1.bar(labels, summary_df["ai_saving_vs_current_wh"], color="#16a34a", alpha=0.85)
    ax1.axhline(0, color="#374151", linewidth=1)
    ax1.set_ylabel("AI saving vs current (Wh)")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(labels, summary_df["ai_saving_vs_current_pct"], color="#dc2626", marker="o", linewidth=2, label="AI saving vs current (%)")
    ax2.plot(labels, summary_df["ai_saved_vs_max_pct"], color="#f59e0b", marker="o", linewidth=2, label="AI saving vs max baseline (%)")
    ax2.set_ylabel("Savings (%)")
    lines, line_labels = ax2.get_legend_handles_labels()
    ax2.legend(lines, line_labels, loc="upper right")
    ax1.set_title(title, fontsize=14, weight="bold")
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}", flush=True)


def plot_single_period(summary: dict, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["Current usage", "AI recommendation", "Max baseline"]
    values = [summary["current_wh"], summary["ai_wh"], summary["max_wh"]]
    colors = ["#2563eb", "#16a34a", "#9ca3af"]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_ylabel("Energy used (Wh)")
    ax.set_ylim(0, max(values) * 1.25 if max(values) else 1)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + (max(values) * 0.025 if max(values) else 0.02), f"{value:.1f} Wh", ha="center", va="bottom", fontsize=11, weight="bold")
    ax.text(
        0.5,
        max(values) * 1.12 if max(values) else 0.9,
        f"AI vs current: {summary['ai_saving_vs_current_wh']:.1f} Wh ({summary['ai_saving_vs_current_pct']:.1f}%)",
        ha="center",
        color="#16a34a" if summary["ai_saving_vs_current_wh"] >= 0 else "#dc2626",
        fontsize=11,
        weight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}", flush=True)


def load_saved_day(output_dir: Path, room: int, day: str, model_type: str):
    import pandas as pd

    path = raw_records_path(output_dir, room, day, model_type)
    if not path.exists():
        print(f"[missing] {path}", flush=True)
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_cache(args.output_dir)

    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    scopes = {item.strip().lower() for item in args.scope.split(",") if item.strip()}
    selected_days: set[str] = set()
    if "day" in scopes:
        selected_days.add(args.day)
    if "month" in scopes:
        selected_days.update(month_days(args.month))
    if "all" in scopes:
        selected_days.update(available_data_days(args.room))

    daily_frames = {}
    for day in sorted(selected_days):
        if args.only_plots:
            df = load_saved_day(args.output_dir, args.room, day, args.model_type)
        else:
            df = compute_one_day(args.room, day, args.model_type, args.lookback_hours, args.output_dir, args.force)
        if not df.empty:
            daily_frames[day] = df

    if not daily_frames:
        raise SystemExit("No raw AI energy records are available.")

    daily_summary = aggregate_days(daily_frames)
    save_summary_csv(daily_summary, args.output_dir / f"room_{args.room}_daily_current_ai_max_summary.csv")

    all_records = pd.concat(daily_frames.values())
    all_summary = pd.DataFrame([summarize_records(all_records, "all_available_days")])
    save_summary_csv(all_summary, args.output_dir / f"room_{args.room}_all_current_ai_max_summary.csv")

    if "day" in scopes and args.day in daily_frames:
        day_df = daily_frames[args.day]
        day_summary = summarize_records(day_df, args.day)
        save_summary_csv(pd.DataFrame([day_summary]), args.output_dir / f"room_{args.room}_{args.day}_current_ai_max_summary.csv")
        hourly = aggregate_hourly(day_df)
        save_summary_csv(hourly, args.output_dir / f"room_{args.room}_{args.day}_hourly_current_ai_max_summary.csv")
        plot_single_period(day_summary, f"Room {args.room} HVAC Energy Used: Current vs AI vs Max Baseline\n{args.day}", args.output_dir / f"room_{args.room}_{args.day}_daily_current_ai_max.png")
        plot_grouped_energy(hourly, f"Room {args.room} Hourly HVAC Energy Used: Current vs AI vs Max Baseline\n{args.day}", args.output_dir / f"room_{args.room}_{args.day}_hourly_current_ai_max.png")
        plot_savings(hourly, f"Room {args.room} Hourly HVAC Savings\n{args.day}", args.output_dir / f"room_{args.room}_{args.day}_hourly_ai_savings.png")

    if "month" in scopes:
        month_prefix = f"{args.month}-"
        month_summary = daily_summary[daily_summary["period"].astype(str).str.startswith(month_prefix)].copy()
        if not month_summary.empty:
            save_summary_csv(month_summary, args.output_dir / f"room_{args.room}_{args.month}_daily_current_ai_max_summary.csv")
            month_total = pd.DataFrame([summarize_records(pd.concat([daily_frames[d] for d in month_summary["period"] if d in daily_frames]), args.month)])
            save_summary_csv(month_total, args.output_dir / f"room_{args.room}_{args.month}_current_ai_max_summary.csv")
            plot_grouped_energy(month_summary, f"Room {args.room} Daily HVAC Energy Used: Current vs AI vs Max Baseline\n{args.month}", args.output_dir / f"room_{args.room}_{args.month}_daily_current_ai_max.png")
            plot_savings(month_summary, f"Room {args.room} Daily HVAC Savings\n{args.month}", args.output_dir / f"room_{args.room}_{args.month}_daily_ai_savings.png")
            plot_single_period(month_total.iloc[0].to_dict(), f"Room {args.room} Monthly HVAC Energy Used: Current vs AI vs Max Baseline\n{args.month}", args.output_dir / f"room_{args.room}_{args.month}_monthly_current_ai_max.png")

    if "all" in scopes:
        monthly = daily_summary.copy()
        monthly["month"] = monthly["period"].astype(str).str.slice(0, 7)
        monthly_rows = []
        for month, group in monthly.groupby("month"):
            monthly_rows.append({
                "period": month,
                "current_wh": float(group["current_wh"].sum()),
                "ai_wh": float(group["ai_wh"].sum()),
                "max_wh": float(group["max_wh"].sum()),
                "current_saved_vs_max_wh": float(group["current_saved_vs_max_wh"].sum()),
                "ai_saved_vs_max_wh": float(group["ai_saved_vs_max_wh"].sum()),
                "ai_saving_vs_current_wh": float(group["ai_saving_vs_current_wh"].sum()),
            })
        monthly_summary = pd.DataFrame(monthly_rows)
        if not monthly_summary.empty:
            monthly_summary["current_saved_vs_max_pct"] = monthly_summary.apply(lambda r: safe_pct(r["current_saved_vs_max_wh"], r["max_wh"]), axis=1)
            monthly_summary["ai_saved_vs_max_pct"] = monthly_summary.apply(lambda r: safe_pct(r["ai_saved_vs_max_wh"], r["max_wh"]), axis=1)
            monthly_summary["ai_saving_vs_current_pct"] = monthly_summary.apply(lambda r: safe_pct(r["ai_saving_vs_current_wh"], r["current_wh"]), axis=1)
            save_summary_csv(monthly_summary, args.output_dir / f"room_{args.room}_monthly_current_ai_max_summary.csv")
            plot_grouped_energy(monthly_summary, f"Room {args.room} Monthly HVAC Energy Used: Current vs AI vs Max Baseline", args.output_dir / f"room_{args.room}_monthly_current_ai_max.png")
            plot_savings(monthly_summary, f"Room {args.room} Monthly HVAC Savings", args.output_dir / f"room_{args.room}_monthly_ai_savings.png")
            plot_single_period(all_summary.iloc[0].to_dict(), f"Room {args.room} All Available HVAC Energy Used: Current vs AI vs Max Baseline", args.output_dir / f"room_{args.room}_all_current_ai_max.png")

    print("\nDone. Results are in:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
