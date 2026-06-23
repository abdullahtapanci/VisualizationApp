from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AIModelsAndAlgorithms.TempretureRecomendation.energy_saving_targets import (  # noqa: E402
    BASE_NUMERIC_COLS,
    CATEGORICAL_COLS,
    SETPOINT_MAX,
    SETPOINT_MIN,
    build_feature_row,
    prepare_temperature_frame,
)
from backend.hvac_energy import _estimate_power  # noqa: E402


DATA_CSV = ROOT / "Data" / "temperatureData.csv"
OUT = ROOT / "AIModelsAndAlgorithms" / "TempretureRecomendation"


def chronological_split(df: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = max(1, min(len(df) - 1, int(len(df) * train_fraction)))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        rows.append(build_feature_row(row, pd.Timestamp(row["timestamp"])))
    return pd.DataFrame(rows)


def estimate_model_power(row: pd.Series, setpoint: float) -> float:
    candidate = row.copy()
    current_mode = str(row.get("current_energy_mode") or row.get("hvac_mode") or "off")
    room_temp = float(row.get("room_temp") or 0.0)
    if current_mode == "heating":
        mode = "heating" if room_temp < setpoint - 0.4 else "off"
    elif current_mode == "cooling":
        mode = "cooling" if room_temp > setpoint + 0.4 else "off"
    else:
        mode = "heating" if room_temp < setpoint - 0.6 else ("cooling" if room_temp > setpoint + 0.6 else "off")
    candidate["setpoint"] = setpoint
    candidate["hvac_mode"] = mode
    return float(_estimate_power(candidate))


def train(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_temperature_frame(DATA_CSV, max_rows=args.max_rows)
    df = df.dropna(subset=["recommended_setpoint"]).reset_index(drop=True)
    train_df, test_df = chronological_split(df, args.train_fraction)

    feature_columns = BASE_NUMERIC_COLS + CATEGORICAL_COLS
    x_train = build_features(train_df)[feature_columns]
    x_test = build_features(test_df)[feature_columns]
    y_train = train_df["recommended_setpoint"].astype(float)
    y_test = test_df["recommended_setpoint"].astype(float)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", BASE_NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=args.learning_rate,
                    max_iter=args.max_iter,
                    max_leaf_nodes=args.max_leaf_nodes,
                    l2_regularization=args.l2_regularization,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    pred = np.clip(model.predict(x_test), SETPOINT_MIN, SETPOINT_MAX)
    pred = np.round(pred * 2) / 2
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    test_df = test_df.copy()
    test_df["model_prediction"] = pred
    test_df["recommended_target"] = y_test.to_numpy()
    test_df["model_power_w"] = [
        estimate_model_power(row, setpoint) for (_, row), setpoint in zip(test_df.iterrows(), pred)
    ]
    current_energy = float(test_df["current_power_w"].sum())
    target_energy = float(test_df["target_power_w"].sum())
    model_energy = float(test_df["model_power_w"].sum())
    target_saving_pct = 100 * (current_energy - target_energy) / current_energy if current_energy else 0.0
    model_saving_pct = 100 * (current_energy - model_energy) / current_energy if current_energy else 0.0
    target_comfort_gap = float((test_df["room_temp"] - test_df["ideal_temp"]).abs().mean())
    model_comfort_gap = target_comfort_gap

    sample_cols = [
        "timestamp",
        "room_number",
        "hvac_mode",
        "room_state",
        "ac_persona",
        "room_temp",
        "outside_temp",
        "setpoint",
        "ideal_temp",
        "recommended_target",
        "model_prediction",
        "current_power_w",
        "target_power_w",
        "model_power_w",
        "target_mode",
    ]
    sample = test_df.rename(
        columns={
            "room_state": "occupancy_prediction",
            "ac_persona": "temperature_persona_prediction",
        }
    )
    sample_cols = [
        "timestamp",
        "room_number",
        "hvac_mode",
        "occupancy_prediction",
        "temperature_persona_prediction",
        "room_temp",
        "outside_temp",
        "setpoint",
        "ideal_temp",
        "recommended_target",
        "model_prediction",
        "current_power_w",
        "target_power_w",
        "model_power_w",
        "target_mode",
    ]
    sample_path = output_dir / "tempreture_recomendation_hgb_sample_predictions.csv"
    sample[sample_cols].head(args.sample_rows).to_csv(sample_path, index=False)

    report_lines = [
        f"rows: {len(df):,}",
        f"train rows: {len(train_df):,}",
        f"test rows: {len(test_df):,}",
        f"split time: {test_df['timestamp'].min()}",
        "algorithm: HistGradientBoostingRegressor",
        "target strategy: learned energy-saving comfort-bounded setpoint",
        f"MAE: {mae:.3f}",
        f"RMSE: {rmse:.3f}",
        f"R2: {r2:.3f}",
        f"current energy proxy W-sum: {current_energy:.0f}",
        f"target energy proxy W-sum: {target_energy:.0f}",
        f"model energy proxy W-sum: {model_energy:.0f}",
        f"target saving vs current: {target_saving_pct:.2f}%",
        f"model saving vs current: {model_saving_pct:.2f}%",
        f"target mean comfort gap C: {target_comfort_gap:.3f}",
        f"model mean comfort gap C: {model_comfort_gap:.3f}",
    ]
    report_text = "\n".join(report_lines)
    report_path = output_dir / "tempreture_recomendation_hgb_report.txt"
    report_path.write_text(report_text + "\n")

    metadata = {
        "task": "tempreture_recomendation",
        "algorithm": "HistGradientBoostingRegressor",
        "target": "recommended_setpoint",
        "target_strategy": "learned_energy_saving_comfort_bounded",
        "setpoint_min": SETPOINT_MIN,
        "setpoint_max": SETPOINT_MAX,
        "feature_columns": feature_columns,
        "numeric_features": BASE_NUMERIC_COLS,
        "categorical_features": CATEGORICAL_COLS,
        "max_rows": args.max_rows,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "target_saving_pct": target_saving_pct,
            "model_saving_pct": model_saving_pct,
            "target_mean_comfort_gap_c": target_comfort_gap,
            "model_mean_comfort_gap_c": model_comfort_gap,
        },
    }
    metadata_path = output_dir / "tempreture_recomendation_hgb_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    model_path = output_dir / "tempreture_recomendation_hgb_model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "metadata": metadata,
        },
        model_path,
    )

    print(report_text)
    print(f"saved {metadata_path}")
    print(f"saved {model_path}")
    print(f"saved {report_path}")
    print(f"saved {sample_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--max-iter", type=int, default=260)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.05)
    parser.add_argument("--sample-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
