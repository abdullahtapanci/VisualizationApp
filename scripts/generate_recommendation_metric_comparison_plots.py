from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/visualizationapp-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = ROOT / "AIModelsAndAlgorithms"
OUT_DIR = ROOT / "ModelResults" / "recommendation_metric_comparisons"


PLOTS = [
    {
        "title": "Lighting Recommendation Model Metrics",
        "filename": "lighting_recommendation_model_metrics_comparison.png",
        "reports": {
            "Hist Gradient Boosting Regressor": MODELS_ROOT
            / "LightingRecommendation"
            / "lighting_recommendation_report.txt",
            "Transformer Regressor": MODELS_ROOT
            / "LightingRecommendation"
            / "transformer"
            / "lighting_recommendation_transformer_report.txt",
        },
    },
    {
        "title": "Temperature Recommendation Model Metrics",
        "filename": "temperature_recommendation_model_metrics_comparison.png",
        "reports": {
            "Hist Gradient Boosting Regressor": MODELS_ROOT
            / "TempretureRecomendation"
            / "tempreture_recomendation_hgb_report.txt",
            "Transformer Regressor": MODELS_ROOT
            / "TempretureRecomendation"
            / "transformer"
            / "tempreture_recomendation_transformer_report.txt",
        },
    },
]


def extract_metric(report_path: Path, metric: str) -> float:
    text = report_path.read_text(errors="ignore")
    match = re.search(rf"^{metric}:\s*([+-]?\d+(?:\.\d+)?)", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find {metric} in {report_path}")
    return float(match.group(1))


def collect_metrics(reports: dict[str, Path]) -> dict[str, dict[str, float]]:
    return {
        model_name: {
            "MAE": extract_metric(report_path, "MAE"),
            "RMSE": extract_metric(report_path, "RMSE"),
            "R2": extract_metric(report_path, "R2"),
        }
        for model_name, report_path in reports.items()
    }


def draw_plot(title: str, metrics: dict[str, dict[str, float]], output_path: Path) -> None:
    model_names = list(metrics)
    metric_specs = [
        ("MAE", "MAE lower is better", "MAE"),
        ("RMSE", "RMSE lower is better", "RMSE"),
        ("R2", "R2 higher is better", "R2"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    fig.suptitle(title, fontsize=14, y=1.03)
    bar_color = "#72a6e6"

    for ax, (metric_key, subplot_title, y_label) in zip(axes, metric_specs):
        values = [metrics[model][metric_key] for model in model_names]
        bars = ax.bar(model_names, values, color=bar_color)
        ax.set_title(subplot_title, fontsize=12)
        ax.set_ylabel(y_label)
        ax.grid(axis="y", alpha=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=18)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")

        if metric_key == "R2":
            ax.set_ylim(0, max(1.05, max(values) * 1.08))
        else:
            ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    generated = []
    for plot in PLOTS:
        metrics = collect_metrics(plot["reports"])
        output_path = OUT_DIR / plot["filename"]
        draw_plot(plot["title"], metrics, output_path)
        generated.append(output_path)

    for path in generated:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
