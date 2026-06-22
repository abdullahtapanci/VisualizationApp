#!/usr/bin/env python3
"""Generate lighting energy comparisons with actual, AI, and max baseline.

The calculation uses the backend lighting recommendation energy helper. It is
intentionally resumable: each day is saved as a raw CSV before aggregated plots
are created, so long month/all-month runs can continue from prior results.
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

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "lighting_ai_energy_report"
SAMPLE_MINUTES = 5


def configure_plot_cache(output_dir: Path) -> None:
    cache_dir = output_dir / ".plot_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir / "xdg"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot actual vs AI vs max-baseline lighting energy.",
    )
    parser.add_argument("--room", type=int, default=1, help="Room number to analyze.")
    parser.add_argument("--day", default="2022-01-22", help="Day to plot, YYYY-MM-DD.")
    parser.add_argument("--month", default="2022-01", help="Month to plot, YYYY-MM.")
    parser.add_argument(
        "--model-type",
        default="hist_gradient_boosting",
        choices=["auto", "hist_gradient_boosting", "transformer"],
        help="Lighting recommendation model to use for AI energy.",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="History window used by the recommendation model.",
    )
    parser.add_argument(
        "--scope",
        default="day,month,all",
        help="Comma-separated scopes to compute: day, month, all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where CSV and PNG results are written.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute daily raw CSV files even if they already exist.",
    )
    parser.add_argument(
        "--only-plots",
        action="store_true",
        help="Do not recompute AI results; only rebuild plots from saved CSVs.",
    )
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
            FROM lightning_data
            WHERE room_number = ?
              AND lamp_location <> 'none'
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


def compute_one_day(
    room: int,
    day: str,
    model_type: str,
    lookback_hours: int,
    output_dir: Path,
    force: bool,
):
    import pandas as pd

    from backend.lightning_recommendation import compute_lightning_recommendation_energy_for_room

    out_path = raw_records_path(output_dir, room, day, model_type)
    if out_path.exists() and not force:
        print(f"[skip] {day}: using saved {out_path}", flush=True)
        return pd.read_csv(out_path)

    start, end_inclusive = day_bounds(day)
    print(
        f"[compute] room {room} {day} with {model_type} "
        f"({start} to {end_inclusive})",
        flush=True,
    )
    result = compute_lightning_recommendation_energy_for_room(
        room_number=room,
        start_timestamp=start.strftime("%Y-%m-%d %H:%M:%S"),
        end_timestamp=end_inclusive.strftime("%Y-%m-%d %H:%M:%S"),
        lookback_hours=lookback_hours,
        model_type=model_type,
    )
    if result.get("empty"):
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(result["records"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[saved] {out_path} ({len(df)} rows)", flush=True)
    return df


def summarize_records(records, period: str) -> dict:
    import pandas as pd

    if records is None or len(records) == 0:
        return {
            "period": period,
            "actual_wh": 0.0,
            "ai_wh": 0.0,
            "max_wh": 0.0,
            "actual_saved_vs_max_wh": 0.0,
            "ai_saved_vs_max_wh": 0.0,
            "ai_saving_vs_actual_wh": 0.0,
            "actual_saved_vs_max_pct": 0.0,
            "ai_saved_vs_max_pct": 0.0,
            "ai_saving_vs_actual_pct": 0.0,
            "n_rows": 0,
        }

    df = pd.DataFrame(records)
    actual = float(df["current_wh"].sum())
    ai = float(df["recommended_wh"].sum())
    max_wh = float(df["full_brightness_wh"].sum())
    actual_saved = max_wh - actual
    ai_saved = max_wh - ai
    ai_saving_vs_actual = actual - ai
    return {
        "period": period,
        "actual_wh": actual,
        "ai_wh": ai,
        "max_wh": max_wh,
        "actual_saved_vs_max_wh": actual_saved,
        "ai_saved_vs_max_wh": ai_saved,
        "ai_saving_vs_actual_wh": ai_saving_vs_actual,
        "actual_saved_vs_max_pct": safe_pct(actual_saved, max_wh),
        "ai_saved_vs_max_pct": safe_pct(ai_saved, max_wh),
        "ai_saving_vs_actual_pct": safe_pct(ai_saving_vs_actual, actual),
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
    rows = []
    for hour, group in add_hour_column(df).groupby("hour"):
        rows.append(summarize_records(group, pd.Timestamp(hour).strftime("%Y-%m-%d %H:%M:%S")))
    return pd.DataFrame(rows)


def aggregate_days(daily_frames: dict[str, object]):
    import pandas as pd

    rows = []
    for day, frame in daily_frames.items():
        rows.append(summarize_records(frame, day))
    return pd.DataFrame(rows)


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
    ax.bar(x - width, summary_df["actual_wh"], width, label="Actual usage", color="#2563eb")
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
    ax1.bar(labels, summary_df["ai_saving_vs_actual_wh"], color="#16a34a", alpha=0.85)
    ax1.axhline(0, color="#374151", linewidth=1)
    ax1.set_ylabel("AI saving vs actual (Wh)")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(
        labels,
        summary_df["ai_saving_vs_actual_pct"],
        color="#dc2626",
        marker="o",
        linewidth=2,
        label="AI saving vs actual (%)",
    )
    ax2.plot(
        labels,
        summary_df["ai_saved_vs_max_pct"],
        color="#f59e0b",
        marker="o",
        linewidth=2,
        label="AI saving vs max baseline (%)",
    )
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

    labels = ["Actual usage", "AI recommendation", "Max baseline"]
    values = [summary["actual_wh"], summary["ai_wh"], summary["max_wh"]]
    colors = ["#2563eb", "#16a34a", "#9ca3af"]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_ylabel("Energy used (Wh)")
    ax.set_ylim(0, max(values) * 1.25 if max(values) else 1)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (max(values) * 0.025 if max(values) else 0.02),
            f"{value:.1f} Wh",
            ha="center",
            va="bottom",
            fontsize=11,
            weight="bold",
        )
    ax.text(
        0.5,
        max(values) * 1.12 if max(values) else 0.9,
        f"AI vs actual: {summary['ai_saving_vs_actual_wh']:.1f} Wh "
        f"({summary['ai_saving_vs_actual_pct']:.1f}%)",
        ha="center",
        color="#16a34a" if summary["ai_saving_vs_actual_wh"] >= 0 else "#dc2626",
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

    if not selected_days:
        raise SystemExit("No days selected. Use --scope day,month,all.")

    daily_frames = {}
    for day in sorted(selected_days):
        if args.only_plots:
            df = load_saved_day(args.output_dir, args.room, day, args.model_type)
        else:
            df = compute_one_day(
                room=args.room,
                day=day,
                model_type=args.model_type,
                lookback_hours=args.lookback_hours,
                output_dir=args.output_dir,
                force=args.force,
            )
        if not df.empty:
            daily_frames[day] = df

    if not daily_frames:
        raise SystemExit("No raw AI energy records are available.")

    daily_summary = aggregate_days(daily_frames)
    save_summary_csv(
        daily_summary,
        args.output_dir / f"room_{args.room}_daily_actual_ai_max_summary.csv",
    )

    all_summary = pd.DataFrame([summarize_records(pd.concat(daily_frames.values()), "all_available_days")])
    save_summary_csv(
        all_summary,
        args.output_dir / f"room_{args.room}_all_actual_ai_max_summary.csv",
    )

    if "day" in scopes and args.day in daily_frames:
        day_df = daily_frames[args.day]
        day_summary = summarize_records(day_df, args.day)
        save_summary_csv(
            pd.DataFrame([day_summary]),
            args.output_dir / f"room_{args.room}_{args.day}_actual_ai_max_summary.csv",
        )
        hourly = aggregate_hourly(day_df)
        save_summary_csv(
            hourly,
            args.output_dir / f"room_{args.room}_{args.day}_hourly_actual_ai_max_summary.csv",
        )
        plot_single_period(
            day_summary,
            f"Room {args.room} Lighting Energy Used: Actual vs AI vs Max Baseline\n"
            f"{args.day}, dimmable lamps only",
            args.output_dir / f"room_{args.room}_{args.day}_daily_actual_ai_max.png",
        )
        plot_grouped_energy(
            hourly,
            f"Room {args.room} Hourly Lighting Energy Used: Actual vs AI vs Max Baseline\n"
            f"{args.day}, dimmable lamps only",
            args.output_dir / f"room_{args.room}_{args.day}_hourly_actual_ai_max.png",
        )
        plot_savings(
            hourly,
            f"Room {args.room} Hourly Lighting Savings\n{args.day}, dimmable lamps only",
            args.output_dir / f"room_{args.room}_{args.day}_hourly_ai_savings.png",
        )

    if "month" in scopes:
        month_prefix = f"{args.month}-"
        month_summary = daily_summary[daily_summary["period"].astype(str).str.startswith(month_prefix)].copy()
        if not month_summary.empty:
            save_summary_csv(
                month_summary,
                args.output_dir / f"room_{args.room}_{args.month}_daily_actual_ai_max_summary.csv",
            )
            month_total = pd.DataFrame([summarize_records(
                pd.concat([daily_frames[d] for d in month_summary["period"] if d in daily_frames]),
                args.month,
            )])
            save_summary_csv(
                month_total,
                args.output_dir / f"room_{args.room}_{args.month}_actual_ai_max_summary.csv",
            )
            plot_grouped_energy(
                month_summary,
                f"Room {args.room} Daily Lighting Energy Used: Actual vs AI vs Max Baseline\n"
                f"{args.month}, dimmable lamps only",
                args.output_dir / f"room_{args.room}_{args.month}_daily_actual_ai_max.png",
            )
            plot_savings(
                month_summary,
                f"Room {args.room} Daily Lighting Savings\n{args.month}, dimmable lamps only",
                args.output_dir / f"room_{args.room}_{args.month}_daily_ai_savings.png",
            )
            plot_single_period(
                month_total.iloc[0].to_dict(),
                f"Room {args.room} Monthly Lighting Energy Used: Actual vs AI vs Max Baseline\n"
                f"{args.month}, dimmable lamps only",
                args.output_dir / f"room_{args.room}_{args.month}_monthly_actual_ai_max.png",
            )

    if "all" in scopes:
        monthly = daily_summary.copy()
        monthly["month"] = monthly["period"].astype(str).str.slice(0, 7)
        monthly_rows = []
        for month, group in monthly.groupby("month"):
            monthly_rows.append({
                "period": month,
                "actual_wh": float(group["actual_wh"].sum()),
                "ai_wh": float(group["ai_wh"].sum()),
                "max_wh": float(group["max_wh"].sum()),
                "actual_saved_vs_max_wh": float(group["actual_saved_vs_max_wh"].sum()),
                "ai_saved_vs_max_wh": float(group["ai_saved_vs_max_wh"].sum()),
                "ai_saving_vs_actual_wh": float(group["ai_saving_vs_actual_wh"].sum()),
            })
        monthly_summary = pd.DataFrame(monthly_rows)
        if not monthly_summary.empty:
            monthly_summary["actual_saved_vs_max_pct"] = monthly_summary.apply(
                lambda r: safe_pct(r["actual_saved_vs_max_wh"], r["max_wh"]), axis=1
            )
            monthly_summary["ai_saved_vs_max_pct"] = monthly_summary.apply(
                lambda r: safe_pct(r["ai_saved_vs_max_wh"], r["max_wh"]), axis=1
            )
            monthly_summary["ai_saving_vs_actual_pct"] = monthly_summary.apply(
                lambda r: safe_pct(r["ai_saving_vs_actual_wh"], r["actual_wh"]), axis=1
            )
            save_summary_csv(
                monthly_summary,
                args.output_dir / f"room_{args.room}_monthly_actual_ai_max_summary.csv",
            )
            plot_grouped_energy(
                monthly_summary,
                f"Room {args.room} Monthly Lighting Energy Used: Actual vs AI vs Max Baseline\n"
                "Dimmable lamps only",
                args.output_dir / f"room_{args.room}_monthly_actual_ai_max.png",
            )
            plot_savings(
                monthly_summary,
                f"Room {args.room} Monthly Lighting Savings\nDimmable lamps only",
                args.output_dir / f"room_{args.room}_monthly_ai_savings.png",
            )
            plot_single_period(
                all_summary.iloc[0].to_dict(),
                f"Room {args.room} All Available Lighting Energy Used: Actual vs AI vs Max Baseline\n"
                "Dimmable lamps only",
                args.output_dir / f"room_{args.room}_all_actual_ai_max.png",
            )

    print("\nDone. Results are in:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
