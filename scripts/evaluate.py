from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score

from scripts.train import load_dataset, split_dataset
from src.predictor import BASELINE, TRANSFORMER, PredictorService


CONFIDENCE_THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.80)


def confidence_threshold_analysis(
    expected: list[str], predicted: list[str], confidences: list[float]
) -> dict[str, dict[str, float | None]]:
    """Describe the accuracy/coverage trade-off at common review thresholds."""
    total = len(expected)
    analysis: dict[str, dict[str, float | None]] = {}
    for threshold in CONFIDENCE_THRESHOLDS:
        accepted = [index for index, confidence in enumerate(confidences) if confidence >= threshold]
        accepted_count = len(accepted)
        coverage = accepted_count / total if total else 0.0
        accepted_accuracy = (
            sum(expected[index] == predicted[index] for index in accepted) / accepted_count
            if accepted_count
            else None
        )
        analysis[f"{threshold:.2f}"] = {
            "accepted_coverage": float(coverage),
            "accepted_accuracy": (
                float(accepted_accuracy) if accepted_accuracy is not None else None
            ),
            "manual_review_rate": float(1.0 - coverage),
        }
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved classifiers on the shared test split.")
    parser.add_argument("--data", type=Path, default=Path("data/sample_requests.csv"))
    parser.add_argument("--text-column", default="message")
    parser.add_argument("--label-column", default="category")
    parser.add_argument("--artifact-dir", type=Path, default=Path("model_artifacts"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model", choices=["available", "baseline", "transformer"], default="available")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_dataset(args.data, args.text_column, args.label_column)
    _, _, test = split_dataset(frame, args.label_column, args.seed)
    service = PredictorService(args.artifact_dir)
    models = service.available_models()
    if args.model != "available":
        requested = BASELINE if args.model == "baseline" else TRANSFORMER
        models = [requested]
    if not models:
        raise RuntimeError("No trained models were found. Run scripts/train.py first.")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    expected = test[args.label_column].tolist()
    labels = sorted(test[args.label_column].unique().tolist())

    for model_name in models:
        predictions = [
            service.predict(text, model_name, explain=False)
            for text in test[args.text_column].tolist()
        ]
        predicted = [prediction.category for prediction in predictions]
        confidences = [prediction.confidence for prediction in predictions]
        results[model_name] = {
            "accuracy": float(accuracy_score(expected, predicted)),
            "macro_f1": float(f1_score(expected, predicted, average="macro")),
            "classification_report": classification_report(
                expected, predicted, output_dict=True, zero_division=0
            ),
            "confidence_thresholds": confidence_threshold_analysis(
                expected, predicted, confidences
            ),
        }
        display = ConfusionMatrixDisplay.from_predictions(
            expected,
            predicted,
            labels=labels,
            xticks_rotation=45,
            cmap="Blues",
            colorbar=False,
        )
        display.figure_.tight_layout()
        display.figure_.savefig(args.report_dir / f"confusion_{model_name}.png", dpi=150)
        plt.close(display.figure_)

    (args.report_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved evaluation reports to {args.report_dir}")


if __name__ == "__main__":
    main()
