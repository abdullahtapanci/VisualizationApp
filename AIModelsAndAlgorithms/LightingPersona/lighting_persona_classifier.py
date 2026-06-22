"""Lighting persona classifier from room-day lighting behavior.

This model predicts the dominant ``lightning_persona`` for a room-day using
only behavior-derived lighting features:

* brightness statistics
* percentage of time lights are on
* hourly brightness profile
* lamp-location usage mix
* occupant/activity counts

The output persona can be used as an input to a later lighting recommendation
model that chooses the next light level or preferred lamp group.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
OUT = Path(__file__).resolve().parent
MODEL_FILE = OUT / "lighting_persona_model.joblib"
SEED = 42


def _dominant_value(series: pd.Series):
    modes = series.dropna().mode()
    return modes.iat[0] if len(modes) else np.nan


def build_room_day_features(max_rows: int | None = None) -> pd.DataFrame:
    """Return one feature row per room/day with the dominant persona target."""
    usecols = [
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
    df = pd.read_csv(
        DATA / "lightningData.csv",
        usecols=usecols,
        parse_dates=["timestamp"],
        nrows=max_rows,
    )
    df = df[df["reservation_active"].eq("Yes")].copy()
    df = df.dropna(subset=["lightning_persona"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["is_on"] = df["Value"].gt(0).astype(int)

    group_keys = ["room_number", "date"]
    base = (
        df.groupby(group_keys)
        .agg(
            floor=("floor", "first"),
            samples=("Value", "size"),
            value_mean=("Value", "mean"),
            value_std=("Value", "std"),
            value_max=("Value", "max"),
            lit_ratio=("is_on", "mean"),
            pir_motion_rate=("pir_motion", "mean"),
            occupants_mean=("n_occupants", "mean"),
            occupants_max=("n_occupants", "max"),
            active_actors_mean=("active_actors", "mean"),
            persona=("lightning_persona", _dominant_value),
        )
        .reset_index()
    )

    hourly = (
        df.groupby(group_keys + ["hour"])["Value"]
        .mean()
        .unstack(fill_value=0)
        .reindex(columns=range(24), fill_value=0)
    )
    hourly.columns = [f"hour_{h:02d}_value_mean" for h in hourly.columns]

    lamp_on = df[df["is_on"].eq(1)].copy()
    lamp_mix = (
        lamp_on.groupby(group_keys + ["lamp_location"])
        .size()
        .unstack(fill_value=0)
    )
    lamp_mix = lamp_mix.div(lamp_mix.sum(axis=1).replace(0, 1), axis=0)
    lamp_mix.columns = [f"lamp_ratio_{c}" for c in lamp_mix.columns]

    features = (
        base.set_index(group_keys)
        .join(hourly, how="left")
        .join(lamp_mix, how="left")
        .fillna(0)
        .reset_index()
    )
    return features


def train(max_rows: int | None = None, output_dir: Path = OUT) -> None:
    df = build_room_day_features(max_rows=max_rows)
    df = df[df["persona"].ne(0)].copy()

    drop_cols = {"date", "persona"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    numeric_cols = feature_cols

    x = df[feature_cols]
    y = df["persona"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
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
                            "num",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            numeric_cols,
                        )
                    ]
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    labels = sorted(y.unique())

    report = classification_report(y_test, y_pred, labels=labels, digits=3)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "true"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / "lighting_persona_model.joblib"
    report_file = output_dir / "lighting_persona_classification_report.txt"
    cm_file = output_dir / "lighting_persona_confusion_matrix.csv"
    cm_df.to_csv(cm_file)
    report_file.write_text(report)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "classes": labels,
        },
        model_file,
    )

    print(f"room-day rows: {len(df):,}")
    print(f"train rows: {len(x_train):,}")
    print(f"test rows: {len(x_test):,}")
    print("\npersona distribution:")
    print(y.value_counts(normalize=True).round(3).to_string())
    print("\nclassification report:")
    print(report)
    print("confusion matrix (rows=true, columns=predicted):")
    print(cm_df)
    print(f"\nsaved {report_file}")
    print(f"saved {cm_file}")
    print(f"saved {model_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for quick experiments. Omit for full data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUT,
        help="Artifact directory. Defaults to the app's lighting persona model folder.",
    )
    args = parser.parse_args()
    train(max_rows=args.max_rows, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
