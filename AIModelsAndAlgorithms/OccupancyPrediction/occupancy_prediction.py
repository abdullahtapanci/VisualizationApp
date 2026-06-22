"""Next-hour occupancy prediction from PIR and reservation data.

The target is the room_state 60 minutes in the future for the same room.
Features use only information available at the current timestamp:

* current PIR motion
* recent PIR motion over the previous hour
* time of day / day of week
* room number
* active guest/adult/child counts from PIR data
* reservation attributes joined by guest_id when available

This is a strong baseline before moving to LSTM/Transformer sequence models.
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
OUT = Path(__file__).resolve().parent
MODEL_FILE = OUT / "occupancy_model.joblib"

FIVE_MIN_SAMPLES_PER_HOUR = 12
SEED = 42


def _cyclic(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    radians = 2 * np.pi * series / period
    return np.sin(radians), np.cos(radians)


def load_dataset(max_rows: int | None = None) -> pd.DataFrame:
    """Build one modeling table from PIRSensorData.csv and reservations."""
    pir = pd.read_csv(
        DATA / "PIRSensorData.csv",
        usecols=[
            "timestamp",
            "room_number",
            "pir_motion",
            "room_state",
            "adults",
            "children",
            "guest_id",
        ],
        parse_dates=["timestamp"],
        nrows=max_rows,
    )
    pir = pir.sort_values(["room_number", "timestamp"]).reset_index(drop=True)

    pir["guest_id"] = pd.to_numeric(pir["guest_id"], errors="coerce")
    pir["active_guest"] = pir["guest_id"].notna().astype(int)
    pir["guest_count"] = pir["adults"].fillna(0) + pir["children"].fillna(0)

    by_room = pir.groupby("room_number", sort=False)
    pir["motion_mean_1h"] = (
        by_room["pir_motion"]
        .rolling(FIVE_MIN_SAMPLES_PER_HOUR, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    pir["motion_sum_1h"] = (
        by_room["pir_motion"]
        .rolling(FIVE_MIN_SAMPLES_PER_HOUR, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    pir["target_room_state_1h"] = by_room["room_state"].shift(-FIVE_MIN_SAMPLES_PER_HOUR)

    pir["hour"] = pir["timestamp"].dt.hour
    pir["dayofweek"] = pir["timestamp"].dt.dayofweek
    pir["hour_sin"], pir["hour_cos"] = _cyclic(pir["hour"], 24)
    pir["dow_sin"], pir["dow_cos"] = _cyclic(pir["dayofweek"], 7)

    reservations = pd.read_csv(
        DATA / "hotelReservationData.csv",
        usecols=[
            "Guest ID",
            "Nationality",
            "Room Type",
            "Total Nights",
            "Total Amount",
            "Stay_Duration",
        ],
    ).rename(columns={"Guest ID": "guest_id", "Room Type": "room_type"})
    reservations["guest_id"] = pd.to_numeric(reservations["guest_id"], errors="coerce")
    reservations["price_per_night"] = (
        reservations["Total Amount"]
        / reservations["Stay_Duration"].replace(0, np.nan)
    )

    df = pir.merge(reservations, on="guest_id", how="left")
    df = df.dropna(subset=["target_room_state_1h"]).copy()
    df["Nationality"] = df["Nationality"].fillna("No active reservation")
    df["room_type"] = df["room_type"].fillna("Unknown")
    return df


def train(max_rows: int | None = None, output_dir: Path = OUT) -> None:
    df = load_dataset(max_rows=max_rows)

    # Chronological split prevents training on future rows and testing on past rows.
    split_time = df["timestamp"].quantile(0.80)
    train_df = df[df["timestamp"] <= split_time].copy()
    test_df = df[df["timestamp"] > split_time].copy()

    target = "target_room_state_1h"
    numeric_features = [
        "room_number",
        "pir_motion",
        "motion_mean_1h",
        "motion_sum_1h",
        "adults",
        "children",
        "guest_count",
        "active_guest",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "Total Nights",
        "Total Amount",
        "Stay_Duration",
        "price_per_night",
    ]
    categorical_features = ["Nationality", "room_type"]

    x_train = train_df[numeric_features + categorical_features]
    y_train = train_df[target]
    x_test = test_df[numeric_features + categorical_features]
    y_test = test_df[target]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=30)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=160,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=SEED,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    labels = sorted(y_train.unique())

    report = classification_report(y_test, y_pred, labels=labels, digits=3)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "true"

    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / "occupancy_model.joblib"
    report_file = output_dir / "occupancy_classification_report.txt"
    cm_file = output_dir / "occupancy_confusion_matrix.csv"
    cm_df.to_csv(cm_file)
    report_file.write_text(report)
    joblib.dump(
        {
            "model": model,
            "feature_columns": numeric_features + categorical_features,
            "classes": labels,
            "horizon_minutes": 60,
        },
        model_file,
    )

    print(f"rows: {len(df):,}")
    print(f"train rows: {len(train_df):,}")
    print(f"test rows: {len(test_df):,}")
    print(f"split time: {split_time}")
    print("\nclass distribution in target:")
    print(df[target].value_counts(normalize=True).round(3).to_string())
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
        help="Artifact directory. Defaults to the app's occupancy model folder.",
    )
    args = parser.parse_args()
    train(max_rows=args.max_rows, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
