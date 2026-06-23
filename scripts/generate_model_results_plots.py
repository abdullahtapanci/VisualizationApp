from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/visualizationapp-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT / "AIModelsAndAlgorithms"
OUT_ROOT = ROOT / "ModelResults"


MODEL_SPECS = [
    {
        "name": "occupancy_random_forest",
        "title": "Occupancy Random Forest",
        "kind": "classifier",
        "report": MODELS_ROOT / "OccupancyPrediction" / "occupancy_classification_report.txt",
        "confusion": MODELS_ROOT / "OccupancyPrediction" / "occupancy_confusion_matrix.csv",
    },
    {
        "name": "occupancy_transformer",
        "title": "Occupancy Transformer",
        "kind": "classifier",
        "report": MODELS_ROOT / "OccupancyPrediction" / "trandformer" / "occupancy_transformer_report.txt",
        "confusion": MODELS_ROOT / "OccupancyPrediction" / "trandformer" / "occupancy_transformer_confusion_matrix.csv",
        "metadata": MODELS_ROOT / "OccupancyPrediction" / "trandformer" / "occupancy_transformer_metadata.json",
    },
    {
        "name": "lighting_persona_random_forest",
        "title": "Lighting Persona Random Forest",
        "kind": "classifier",
        "report": MODELS_ROOT / "LightingPersona" / "lighting_persona_classification_report.txt",
        "confusion": MODELS_ROOT / "LightingPersona" / "lighting_persona_confusion_matrix.csv",
    },
    {
        "name": "lighting_persona_transformer",
        "title": "Lighting Persona Transformer",
        "kind": "classifier",
        "report": MODELS_ROOT / "LightingPersona" / "transformer" / "lighting_persona_transformer_report.txt",
        "confusion": MODELS_ROOT / "LightingPersona" / "transformer" / "lighting_persona_transformer_confusion_matrix.csv",
        "metadata": MODELS_ROOT / "LightingPersona" / "transformer" / "lighting_persona_transformer_metadata.json",
    },
    {
        "name": "lighting_recommendation_hgb",
        "title": "Lighting Recommendation HistGradientBoostingRegressor",
        "kind": "regressor",
        "report": MODELS_ROOT / "LightingRecommendation" / "lighting_recommendation_report.txt",
        "predictions": MODELS_ROOT / "LightingRecommendation" / "lighting_recommendation_sample_predictions.csv",
        "target_col": "recommended_target",
        "prediction_col": "model_prediction",
        "actual_col": "Value",
        "group_cols": ["lamp_location", "occupancy_prediction", "lighting_persona_prediction"],
    },
    {
        "name": "lighting_recommendation_transformer",
        "title": "Lighting Recommendation Transformer",
        "kind": "regressor",
        "report": MODELS_ROOT / "LightingRecommendation" / "transformer" / "lighting_recommendation_transformer_report.txt",
        "predictions": MODELS_ROOT / "LightingRecommendation" / "transformer" / "lighting_recommendation_transformer_sample_predictions.csv",
        "metadata": MODELS_ROOT / "LightingRecommendation" / "transformer" / "lighting_recommendation_transformer_metadata.json",
        "target_col": "recommended_target",
        "prediction_col": "model_prediction",
        "actual_col": "Value",
        "group_cols": ["lamp_location", "occupancy_prediction", "lighting_persona_prediction", "reservation_active"],
    },
    {
        "name": "temperature_persona_hgb",
        "title": "Temperature Persona HistGradientBoostingClassifier",
        "kind": "classifier",
        "report": MODELS_ROOT / "TempreturePersona" / "tempreture_persona_hgb_report.txt",
        "confusion": MODELS_ROOT / "TempreturePersona" / "tempreture_persona_hgb_confusion_matrix.csv",
        "predictions": MODELS_ROOT / "TempreturePersona" / "tempreture_persona_hgb_sample_predictions.csv",
        "metadata": MODELS_ROOT / "TempreturePersona" / "tempreture_persona_hgb_metadata.json",
        "true_col": "ac_persona",
        "prediction_col": "model_prediction",
        "confidence_col": "model_confidence",
    },
    {
        "name": "temperature_persona_transformer",
        "title": "Temperature Persona Transformer",
        "kind": "classifier",
        "report": MODELS_ROOT / "TempreturePersona" / "transformer" / "tempreture_persona_transformer_report.txt",
        "confusion": MODELS_ROOT / "TempreturePersona" / "transformer" / "tempreture_persona_transformer_confusion_matrix.csv",
        "metadata": MODELS_ROOT / "TempreturePersona" / "transformer" / "tempreture_persona_transformer_metadata.json",
    },
    {
        "name": "temperature_recommendation_hgb",
        "title": "Temperature Recommendation HistGradientBoostingRegressor",
        "kind": "regressor",
        "report": MODELS_ROOT / "TempretureRecomendation" / "tempreture_recomendation_hgb_report.txt",
        "predictions": MODELS_ROOT / "TempretureRecomendation" / "tempreture_recomendation_hgb_sample_predictions.csv",
        "metadata": MODELS_ROOT / "TempretureRecomendation" / "tempreture_recomendation_hgb_metadata.json",
        "target_col": "recommended_target",
        "prediction_col": "model_prediction",
        "actual_col": "setpoint",
        "group_cols": ["hvac_mode", "occupancy_prediction", "temperature_persona_prediction", "target_mode"],
        "energy_cols": ("current_power_w", "target_power_w", "model_power_w"),
    },
    {
        "name": "temperature_recommendation_transformer",
        "title": "Temperature Recommendation Transformer",
        "kind": "regressor",
        "report": MODELS_ROOT / "TempretureRecomendation" / "transformer" / "tempreture_recomendation_transformer_report.txt",
        "predictions": MODELS_ROOT / "TempretureRecomendation" / "transformer" / "tempreture_recomendation_transformer_sample_predictions.csv",
        "metadata": MODELS_ROOT / "TempretureRecomendation" / "transformer" / "tempreture_recomendation_transformer_metadata.json",
        "target_col": "recommended_target",
        "prediction_col": "model_prediction",
        "actual_col": "setpoint",
        "group_cols": ["hvac_mode", "occupancy_prediction", "temperature_persona_prediction", "target_mode"],
        "energy_cols": ("current_power_w", "target_power_w", "model_power_w"),
    },
]


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def read_report(path: Path | None) -> str:
    if path and path.exists():
        return path.read_text(errors="ignore")
    return ""


def extract_metrics(report_text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    patterns = {
        "accuracy": r"(?:best test accuracy|accuracy):\s*([+-]?\d+(?:\.\d+)?)",
        "mae": r"MAE:\s*([+-]?\d+(?:\.\d+)?)",
        "rmse": r"RMSE:\s*([+-]?\d+(?:\.\d+)?)",
        "r2": r"R2:\s*([+-]?\d+(?:\.\d+)?)",
        "target_saving_pct": r"target saving vs (?:actual|current):\s*([+-]?\d+(?:\.\d+)?)%",
        "model_saving_pct": r"model saving vs (?:actual|current):\s*([+-]?\d+(?:\.\d+)?)%",
        "target_mean_comfort_gap_c": r"target mean comfort gap C:\s*([+-]?\d+(?:\.\d+)?)",
        "model_mean_comfort_gap_c": r"model mean comfort gap C:\s*([+-]?\d+(?:\.\d+)?)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, report_text, flags=re.IGNORECASE)
        if match:
            metrics[key] = float(match.group(1))
    return metrics


def plot_metric_summary(metrics: dict[str, float], title: str, out_dir: Path) -> None:
    if not metrics:
        return
    labels = list(metrics)
    values = [metrics[k] for k in labels]
    plt.figure(figsize=(9, 4.8))
    colors = ["#2563eb" if "saving" not in label else "#059669" for label in labels]
    bars = plt.bar([label.replace("_", "\n") for label in labels], values, color=colors)
    plt.title(f"{title}: Metric Summary")
    plt.ylabel("Value")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3g}", ha="center", va="bottom", fontsize=9)
    savefig(out_dir / "01_metric_summary.png")


def load_confusion(path: Path | None) -> pd.DataFrame | None:
    if not path or not path.exists():
        return None
    cm = pd.read_csv(path, index_col=0)
    cm.index = cm.index.astype(str)
    cm.columns = cm.columns.astype(str)
    return cm.apply(pd.to_numeric, errors="coerce").fillna(0)


def plot_heatmap(matrix: pd.DataFrame, title: str, path: Path, percent: bool = False) -> None:
    data = matrix.copy()
    if percent:
        denom = data.sum(axis=1).replace(0, np.nan)
        data = data.div(denom, axis=0).fillna(0) * 100
    plt.figure(figsize=(max(7, 0.8 * len(data.columns) + 3), max(5, 0.55 * len(data.index) + 2)))
    image = plt.imshow(data.values, cmap="Blues")
    plt.colorbar(image, fraction=0.046, pad=0.04, label="%" if percent else "Count")
    plt.xticks(range(len(data.columns)), data.columns, rotation=45, ha="right")
    plt.yticks(range(len(data.index)), data.index)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data.iat[row, col]
            label = f"{value:.1f}%" if percent else f"{int(value)}"
            plt.text(col, row, label, ha="center", va="center", fontsize=8, color="#111827")
    savefig(path)


def classifier_metrics_from_cm(cm: pd.DataFrame) -> pd.DataFrame:
    labels = list(cm.index)
    values = cm.reindex(index=labels, columns=labels, fill_value=0).values.astype(float)
    tp = np.diag(values)
    support = values.sum(axis=1)
    predicted = values.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        },
        index=labels,
    )


def plot_class_metrics(cm: pd.DataFrame, title: str, out_dir: Path) -> None:
    metrics = classifier_metrics_from_cm(cm)
    ax = metrics[["precision", "recall", "f1"]].plot(kind="bar", figsize=(10, 5), color=["#2563eb", "#059669", "#f59e0b"])
    ax.set_title(f"{title}: Per-Class Scores")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    savefig(out_dir / "04_per_class_precision_recall_f1.png")

    support = metrics["support"].sort_values(ascending=False)
    plt.figure(figsize=(9, 4.8))
    plt.bar(support.index, support.values, color="#64748b")
    plt.title(f"{title}: Test Support by Class")
    plt.ylabel("Rows")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.25)
    savefig(out_dir / "05_class_support.png")


def plot_classifier_predictions(spec: dict, out_dir: Path) -> None:
    path = spec.get("predictions")
    if not path or not path.exists():
        return
    df = pd.read_csv(path)
    true_col = spec.get("true_col")
    pred_col = spec.get("prediction_col")
    conf_col = spec.get("confidence_col")
    if true_col in df.columns and pred_col in df.columns:
        counts = pd.crosstab(df[true_col].astype(str), df[pred_col].astype(str))
        plot_heatmap(counts, f"{spec['title']}: Sample Prediction Crosstab", out_dir / "06_sample_prediction_crosstab.png")
    if conf_col in df.columns:
        plt.figure(figsize=(8, 4.8))
        plt.hist(pd.to_numeric(df[conf_col], errors="coerce").dropna(), bins=30, color="#2563eb", alpha=0.85)
        plt.title(f"{spec['title']}: Prediction Confidence")
        plt.xlabel("Confidence")
        plt.ylabel("Rows")
        plt.grid(axis="y", alpha=0.25)
        savefig(out_dir / "07_confidence_distribution.png")


def plot_regression_predictions(spec: dict, report_metrics: dict[str, float], out_dir: Path) -> None:
    path = spec.get("predictions")
    if not path or not path.exists():
        return
    df = pd.read_csv(path)
    target_col = spec["target_col"]
    pred_col = spec["prediction_col"]
    actual_col = spec.get("actual_col")
    if target_col not in df.columns or pred_col not in df.columns:
        return
    target = pd.to_numeric(df[target_col], errors="coerce")
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    valid = target.notna() & pred.notna()
    target = target[valid]
    pred = pred[valid]
    residual = pred - target

    sample_idx = np.linspace(0, len(target) - 1, min(len(target), 6000), dtype=int) if len(target) else []
    plt.figure(figsize=(6.5, 6))
    plt.scatter(target.iloc[sample_idx], pred.iloc[sample_idx], s=10, alpha=0.35, color="#2563eb")
    lower = min(target.min(), pred.min())
    upper = max(target.max(), pred.max())
    plt.plot([lower, upper], [lower, upper], color="#111827", linestyle="--", linewidth=1)
    plt.title(f"{spec['title']}: Prediction vs Target")
    plt.xlabel("Target")
    plt.ylabel("Model prediction")
    plt.grid(alpha=0.25)
    savefig(out_dir / "02_prediction_vs_target.png")

    plt.figure(figsize=(8, 4.8))
    plt.hist(residual.dropna(), bins=45, color="#7c3aed", alpha=0.85)
    plt.axvline(0, color="#111827", linestyle="--", linewidth=1)
    plt.title(f"{spec['title']}: Residual Distribution")
    plt.xlabel("Prediction - target")
    plt.ylabel("Rows")
    plt.grid(axis="y", alpha=0.25)
    savefig(out_dir / "03_residual_distribution.png")

    if actual_col in df.columns:
        actual = pd.to_numeric(df.loc[valid, actual_col], errors="coerce")
        means = pd.Series(
            {
                "actual/current": actual.mean(),
                "target": target.mean(),
                "model": pred.mean(),
            }
        )
        plt.figure(figsize=(7, 4.8))
        bars = plt.bar(means.index, means.values, color=["#64748b", "#059669", "#2563eb"])
        plt.title(f"{spec['title']}: Mean Output Level")
        plt.ylabel("Setpoint or light level")
        plt.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, means.values):
            plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
        savefig(out_dir / "04_actual_target_model_mean.png")

    for group_col in spec.get("group_cols", []):
        if group_col not in df.columns:
            continue
        work = df.loc[valid, [group_col]].copy()
        work["absolute_error"] = residual.abs().values
        grouped = work.groupby(group_col)["absolute_error"].mean().sort_values(ascending=False).head(12)
        if grouped.empty:
            continue
        plt.figure(figsize=(9, 4.8))
        plt.bar(grouped.index.astype(str), grouped.values, color="#f59e0b")
        plt.title(f"{spec['title']}: Mean Absolute Error by {group_col}")
        plt.ylabel("MAE")
        plt.xticks(rotation=35, ha="right")
        plt.grid(axis="y", alpha=0.25)
        savefig(out_dir / f"05_mae_by_{group_col}.png")

    energy_cols = spec.get("energy_cols")
    if energy_cols and all(col in df.columns for col in energy_cols):
        current_col, target_power_col, model_power_col = energy_cols
        totals = pd.Series(
            {
                "current": pd.to_numeric(df[current_col], errors="coerce").sum(),
                "target": pd.to_numeric(df[target_power_col], errors="coerce").sum(),
                "model": pd.to_numeric(df[model_power_col], errors="coerce").sum(),
            }
        )
        savings = pd.Series(
            {
                "target saving": 100 * (totals["current"] - totals["target"]) / totals["current"] if totals["current"] else 0,
                "model saving": 100 * (totals["current"] - totals["model"]) / totals["current"] if totals["current"] else 0,
            }
        )
        plt.figure(figsize=(7.5, 4.8))
        plt.bar(totals.index, totals.values, color=["#64748b", "#059669", "#2563eb"])
        plt.title(f"{spec['title']}: Energy Proxy Totals")
        plt.ylabel("Power sum proxy")
        plt.grid(axis="y", alpha=0.25)
        savefig(out_dir / "06_energy_proxy_totals.png")

        plt.figure(figsize=(7.5, 4.8))
        bars = plt.bar(savings.index, savings.values, color=["#10b981", "#2563eb"])
        plt.axhline(0, color="#111827", linewidth=1)
        plt.title(f"{spec['title']}: Energy Saving vs Current")
        plt.ylabel("Saving (%)")
        plt.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, savings.values):
            plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom" if value >= 0 else "top")
        savefig(out_dir / "07_energy_saving_percent.png")


def write_summary(spec: dict, metrics: dict[str, float], out_dir: Path) -> None:
    lines = [spec["title"], "=" * len(spec["title"]), ""]
    if metrics:
        lines.append("Parsed metrics:")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    for key in ["report", "confusion", "predictions", "metadata"]:
        path = spec.get(key)
        if path and path.exists():
            lines.append(f"{key}: {path.relative_to(ROOT)}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    for spec in MODEL_SPECS:
        out_dir = OUT_ROOT / spec["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        report_text = read_report(spec.get("report"))
        metrics = extract_metrics(report_text)
        plot_metric_summary(metrics, spec["title"], out_dir)

        if spec["kind"] == "classifier":
            cm = load_confusion(spec.get("confusion"))
            if cm is not None:
                plot_heatmap(cm, f"{spec['title']}: Confusion Matrix", out_dir / "02_confusion_matrix_counts.png")
                plot_heatmap(cm, f"{spec['title']}: Normalized Confusion Matrix", out_dir / "03_confusion_matrix_percent.png", percent=True)
                plot_class_metrics(cm, spec["title"], out_dir)
            plot_classifier_predictions(spec, out_dir)
        else:
            plot_regression_predictions(spec, metrics, out_dir)

        write_summary(spec, metrics, out_dir)
        generated.extend(sorted(out_dir.glob("*.png")))

    index_lines = ["# Model Results", ""]
    for spec in MODEL_SPECS:
        out_dir = OUT_ROOT / spec["name"]
        plots = sorted(out_dir.glob("*.png"))
        index_lines.append(f"## {spec['title']}")
        index_lines.append(f"Folder: `{out_dir.relative_to(ROOT)}`")
        for plot in plots:
            index_lines.append(f"- `{plot.name}`")
        index_lines.append("")
    (OUT_ROOT / "README.md").write_text("\n".join(index_lines))
    print(f"Generated {len(generated)} plots in {OUT_ROOT}")
    for path in generated:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
