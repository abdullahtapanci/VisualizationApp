"""Train a lighting persona classifier from sliding behavior windows.

The original model used one sample per room-day, which often produced tiny
train/test sets. This version creates one sample per room/time-window, for
example one sample every 30-60 minutes using the previous 4 hours of lighting
behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
OUT = Path(__file__).resolve().parent
SEED = 42

USECOLS = [
    "timestamp",
    "room_number",
    "floor",
    "lamp_location",
    "Value",
    "reservation_active",
    "pir_motion",
    "n_occupants",
    "active_actors",
    "lightning_persona",
]


def _dominant_value(series: pd.Series):
    modes = series.dropna().astype(str).mode()
    return modes.iat[0] if len(modes) else np.nan


def _feature_row(window: pd.DataFrame, room_number: int, window_end: pd.Timestamp) -> dict:
    lamps = window[window["lamp_location"].ne("none")].copy()
    lamps["is_on"] = lamps["Value"].gt(0).astype(int)
    on_df = lamps[lamps["is_on"].eq(1)]
    lamp_counts = on_df["lamp_location"].value_counts(normalize=True).to_dict()
    hourly_means = lamps.groupby(lamps["timestamp"].dt.hour)["Value"].mean().to_dict()

    row = {
        "room_number": float(room_number),
        "floor": float(window["floor"].iloc[-1]) if len(window) else 0.0,
        "samples": float(len(window)),
        "value_mean": float(lamps["Value"].mean()) if len(lamps) else 0.0,
        "value_std": float(lamps["Value"].std()) if len(lamps) > 1 else 0.0,
        "value_max": float(lamps["Value"].max()) if len(lamps) else 0.0,
        "lit_ratio": float(lamps["is_on"].mean()) if len(lamps) else 0.0,
        "pir_motion_rate": float(window["pir_motion"].mean()) if len(window) else 0.0,
        "occupants_mean": float(window["n_occupants"].mean()) if len(window) else 0.0,
        "occupants_max": float(window["n_occupants"].max()) if len(window) else 0.0,
        "active_actors_mean": float(window["active_actors"].mean()) if len(window) else 0.0,
        "window_end_hour_sin": float(np.sin(2 * np.pi * window_end.hour / 24)),
        "window_end_hour_cos": float(np.cos(2 * np.pi * window_end.hour / 24)),
        "window_end_dow_sin": float(np.sin(2 * np.pi * window_end.dayofweek / 7)),
        "window_end_dow_cos": float(np.cos(2 * np.pi * window_end.dayofweek / 7)),
    }
    for hour in range(24):
        row[f"hour_{hour:02d}_value_mean"] = float(hourly_means.get(hour, 0.0))
    for lamp, share in lamp_counts.items():
        row[f"lamp_ratio_{lamp}"] = float(share)
    return row


def build_window_features(
    max_rows: int | None = None,
    window_hours: int = 4,
    stride_minutes: int = 60,
    min_samples_per_window: int = 12,
) -> pd.DataFrame:
    df = pd.read_csv(
        DATA / "lightningData.csv",
        usecols=USECOLS,
        parse_dates=["timestamp"],
        nrows=max_rows,
    )
    df = df[df["reservation_active"].eq("Yes")].copy()
    df = df.dropna(subset=["timestamp", "lightning_persona"])
    if df.empty:
        raise ValueError("No rows left after filtering reservation_active=Yes and lightning_persona.")

    for col in ["Value", "pir_motion", "n_occupants", "active_actors", "floor"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["Value"] = df["Value"].clip(0, 80)
    df["lamp_location"] = df["lamp_location"].fillna("none").astype(str)
    df["lightning_persona"] = df["lightning_persona"].fillna("Unknown").astype(str)
    df = df.sort_values(["room_number", "timestamp"]).reset_index(drop=True)

    window = pd.Timedelta(hours=window_hours)
    stride = pd.Timedelta(minutes=stride_minutes)
    rows = []
    for room_number, group in df.groupby("room_number", sort=False):
        group = group.sort_values("timestamp")
        start = group["timestamp"].min() + window
        end = group["timestamp"].max()
        current = start
        while current <= end:
            window_df = group[(group["timestamp"] > current - window) & (group["timestamp"] <= current)]
            if len(window_df) >= min_samples_per_window:
                persona = _dominant_value(window_df["lightning_persona"])
                if pd.notna(persona) and str(persona).strip().lower() not in {"unknown", "none", "nan", ""}:
                    row = _feature_row(window_df, int(room_number), current)
                    row["persona"] = str(persona)
                    row["window_end"] = current
                    rows.append(row)
            current += stride

    features = pd.DataFrame(rows).fillna(0)
    if features.empty:
        raise ValueError("No training windows were built. Reduce min_samples_per_window or check labels.")
    return features


def train(
    max_rows: int | None = None,
    output_dir: Path = OUT,
    window_hours: int = 4,
    stride_minutes: int = 60,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = build_window_features(
        max_rows=max_rows,
        window_hours=window_hours,
        stride_minutes=stride_minutes,
    )
    feature_cols = [c for c in df.columns if c not in {"persona", "window_end"}]
    x = df[feature_cols]
    y = df["persona"].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y if y.value_counts().min() >= 2 else None,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            feature_cols,
                        )
                    ]
                ),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=250,
                    learning_rate=0.06,
                    max_leaf_nodes=31,
                    l2_regularization=0.05,
                    early_stopping=True,
                    random_state=SEED,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    labels = sorted(y.unique())

    report = classification_report(y_test, y_pred, labels=labels, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "true"

    metadata = {
        "model_type": "hist_gradient_boosting_classifier",
        "task": "lighting_persona",
        "sample_unit": f"{window_hours}h_sliding_window",
        "stride_minutes": stride_minutes,
        "classes": labels,
        "feature_columns": feature_cols,
        "max_rows": max_rows,
        "rows": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }

    report_text = "\n".join(
        [
            f"rows: {len(df):,}",
            f"train rows: {len(x_train):,}",
            f"test rows: {len(x_test):,}",
            f"window hours: {window_hours}",
            f"stride minutes: {stride_minutes}",
            "",
            report,
        ]
    )
    (output_dir / "lighting_persona_classification_report.txt").write_text(report_text)
    cm_df.to_csv(output_dir / "lighting_persona_confusion_matrix.csv")
    (output_dir / "lighting_persona_metadata.json").write_text(json.dumps(metadata, indent=2))
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "classes": labels,
            "metadata": metadata,
        },
        output_dir / "lighting_persona_model.joblib",
    )

    print(report_text)
    print("\npersona distribution:")
    print(y.value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--window-hours", type=int, default=4)
    parser.add_argument("--stride-minutes", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    train(
        max_rows=args.max_rows,
        output_dir=args.output_dir,
        window_hours=args.window_hours,
        stride_minutes=args.stride_minutes,
    )


if __name__ == "__main__":
    main()
