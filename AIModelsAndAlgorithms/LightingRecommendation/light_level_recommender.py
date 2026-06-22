"""Energy-aware lighting level recommendation model.

This is the model that sits after:

1. occupancy prediction
2. lighting persona prediction
3. recent guest lighting-history feature extraction

During offline training, the script uses known ``room_state`` and
``lightning_persona`` as stand-ins for the upstream model outputs. In live use,
replace the ``occupancy_prediction`` and ``lighting_persona_prediction`` feature
columns with outputs from your real models.

The target is not the raw historical light level. Instead, it is an efficient
recommended level derived from policy rules plus recent behavior. This gives
the model a learnable target that preserves the guest's pattern while reducing
unnecessary brightness.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
OUT = Path(__file__).resolve().parent
MODEL_FILE = OUT / "lighting_recommendation_model.joblib"
SEED = 42

SAMPLES_1H = 12
SAMPLES_3H = 36
SAMPLES_24H = 288

PERSONA_POLICY = {
    "StaticBright": {"factor": 0.90, "cap": 75, "min_on": 35},
    "Balanced": {"factor": 0.78, "cap": 60, "min_on": 25},
    "Routine": {"factor": 0.75, "cap": 55, "min_on": 25},
    "NightFocused": {"factor": 0.68, "cap": 45, "min_on": 15},
    "StaticDim": {"factor": 0.62, "cap": 35, "min_on": 10},
    "Housekeeping": {"factor": 1.00, "cap": 80, "min_on": 80},
    "Unknown": {"factor": 0.75, "cap": 55, "min_on": 20},
}


def _cyclic(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    radians = 2 * np.pi * series / period
    return np.sin(radians), np.cos(radians)


def _rolling_history(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    window: int,
    output_col: str,
) -> pd.Series:
    """Past-only rolling mean for a sorted dataframe."""
    return (
        df.groupby(group_cols, sort=False)[value_col]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .rename(output_col)
    )


def _efficient_target(row: pd.Series) -> float:
    """Create an energy-saving target level from behavior and policy.

    This target intentionally lowers avoidable brightness. It keeps lights off
    when historical data says they were off, turns vacant rooms off, leaves
    cleaning bright, and otherwise caps/reduces brightness by persona.
    """
    actual = float(row["Value"])
    occupancy = str(row["occupancy_prediction"])
    persona = str(row["lighting_persona_prediction"])
    policy = PERSONA_POLICY.get(persona, PERSONA_POLICY["Unknown"])

    if occupancy == "Vacant" or row["reservation_active"] == "No":
        return 0.0
    if occupancy == "Cleaning" or persona == "Housekeeping":
        return 80.0 if actual > 0 else 0.0
    if actual <= 0:
        return 0.0

    recent_candidates = [
        row.get("lamp_value_mean_1h", np.nan),
        row.get("lamp_value_mean_3h", np.nan),
        row.get("room_value_mean_24h", np.nan),
    ]
    valid_recent = [v for v in recent_candidates if pd.notna(v)]
    recent = np.median(valid_recent) if valid_recent else np.nan
    if pd.isna(recent) or recent <= 0:
        recent = actual

    recommended = min(actual, recent, policy["cap"]) * policy["factor"]
    recommended = max(recommended, policy["min_on"])
    return float(np.clip(round(recommended), 0, 80))


def build_features(max_rows: int | None = None) -> pd.DataFrame:
    usecols = [
        "timestamp",
        "room_number",
        "floor",
        "lamp_location",
        "Value",
        "room_state",
        "reservation_active",
        "pir_motion",
        "lightning_persona",
        "n_occupants",
        "active_actors",
        "hurry_morning",
        "lazy_day",
        "forgetful",
    ]
    df = pd.read_csv(
        DATA / "lightningData.csv",
        usecols=usecols,
        parse_dates=["timestamp"],
        nrows=max_rows,
    )
    df = df.sort_values(["room_number", "lamp_location", "timestamp"]).reset_index(drop=True)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0).clip(0, 80)
    df["is_on"] = df["Value"].gt(0).astype(int)

    # These two columns are where upstream model outputs plug in.
    df["occupancy_prediction"] = df["room_state"].fillna("Unknown")
    df["lighting_persona_prediction"] = df["lightning_persona"].fillna("Unknown")

    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["hour_sin"], df["hour_cos"] = _cyclic(df["hour"], 24)
    df["dow_sin"], df["dow_cos"] = _cyclic(df["dayofweek"], 7)

    df["lamp_value_mean_1h"] = _rolling_history(
        df, ["room_number", "lamp_location"], "Value", SAMPLES_1H, "lamp_value_mean_1h"
    )
    df["lamp_value_mean_3h"] = _rolling_history(
        df, ["room_number", "lamp_location"], "Value", SAMPLES_3H, "lamp_value_mean_3h"
    )
    df["lamp_value_mean_24h"] = _rolling_history(
        df, ["room_number", "lamp_location"], "Value", SAMPLES_24H, "lamp_value_mean_24h"
    )
    df["lamp_on_rate_3h"] = _rolling_history(
        df, ["room_number", "lamp_location"], "is_on", SAMPLES_3H, "lamp_on_rate_3h"
    )
    df["room_value_mean_1h"] = _rolling_history(
        df, ["room_number"], "Value", SAMPLES_1H, "room_value_mean_1h"
    )
    df["room_value_mean_24h"] = _rolling_history(
        df, ["room_number"], "Value", SAMPLES_24H, "room_value_mean_24h"
    )
    df["room_motion_rate_1h"] = _rolling_history(
        df, ["room_number"], "pir_motion", SAMPLES_1H, "room_motion_rate_1h"
    )

    df["recommended_level"] = df.apply(_efficient_target, axis=1)
    return df.dropna(subset=["recommended_level"]).copy()


def train(max_rows: int | None = None, output_dir: Path = OUT) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / "lighting_recommendation_model.joblib"
    report_file = output_dir / "lighting_recommendation_report.txt"
    predictions_file = output_dir / "lighting_recommendation_sample_predictions.csv"

    df = build_features(max_rows=max_rows)

    # Chronological split: train on earlier behavior, test on later behavior.
    split_time = df["timestamp"].quantile(0.80)
    train_df = df[df["timestamp"] <= split_time].copy()
    test_df = df[df["timestamp"] > split_time].copy()

    target = "recommended_level"
    numeric_features = [
        "room_number",
        "floor",
        "Value",
        "pir_motion",
        "n_occupants",
        "active_actors",
        "hurry_morning",
        "lazy_day",
        "forgetful",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "lamp_value_mean_1h",
        "lamp_value_mean_3h",
        "lamp_value_mean_24h",
        "lamp_on_rate_3h",
        "room_value_mean_1h",
        "room_value_mean_24h",
        "room_motion_rate_1h",
    ]
    categorical_features = [
        "lamp_location",
        "reservation_active",
        "occupancy_prediction",
        "lighting_persona_prediction",
    ]

    x_train = train_df[numeric_features + categorical_features]
    y_train = train_df[target]
    x_test = test_df[numeric_features + categorical_features]
    y_test = test_df[target]

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
                            numeric_features,
                        ),
                        (
                            "cat",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                                ]
                            ),
                            categorical_features,
                        ),
                    ]
                ),
            ),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    max_iter=180,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    l2_regularization=0.01,
                    random_state=SEED,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    pred = np.clip(np.rint(model.predict(x_test)), 0, 80)

    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)

    actual_energy_proxy = float(test_df["Value"].sum())
    policy_energy_proxy = float(y_test.sum())
    model_energy_proxy = float(pred.sum())
    policy_saving_pct = 100 * (1 - policy_energy_proxy / actual_energy_proxy) if actual_energy_proxy else 0
    model_saving_pct = 100 * (1 - model_energy_proxy / actual_energy_proxy) if actual_energy_proxy else 0

    comparison = test_df[
        [
            "timestamp",
            "room_number",
            "lamp_location",
            "occupancy_prediction",
            "lighting_persona_prediction",
            "Value",
        ]
    ].copy()
    comparison["recommended_target"] = y_test.to_numpy()
    comparison["model_prediction"] = pred
    comparison.head(5000).to_csv(predictions_file, index=False)
    joblib.dump(
        {
            "model": model,
            "feature_columns": numeric_features + categorical_features,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "target": target,
            "prediction_horizon_minutes": 5,
            "level_min": 0,
            "level_max": 80,
        },
        model_file,
    )

    report = "\n".join(
        [
            f"rows: {len(df):,}",
            f"train rows: {len(train_df):,}",
            f"test rows: {len(test_df):,}",
            f"split time: {split_time}",
            f"MAE: {mae:.3f}",
            f"RMSE: {rmse:.3f}",
            f"R2: {r2:.3f}",
            f"actual level-sum proxy: {actual_energy_proxy:.0f}",
            f"efficient target level-sum proxy: {policy_energy_proxy:.0f}",
            f"model level-sum proxy: {model_energy_proxy:.0f}",
            f"target saving vs actual: {policy_saving_pct:.2f}%",
            f"model saving vs actual: {model_saving_pct:.2f}%",
        ]
    )
    report_file.write_text(report)

    print(report)
    print(f"\nsaved {report_file}")
    print(f"saved {predictions_file}")
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
        help="Artifact directory. Defaults to the app's lighting recommendation model folder.",
    )
    args = parser.parse_args()
    train(max_rows=args.max_rows, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
