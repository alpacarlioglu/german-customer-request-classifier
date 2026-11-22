from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score

from scripts.train import load_dataset, split_dataset
from src.predictor import BASELINE, TRANSFORMER, PredictorService


CONFIDENCE_THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.80)
SCAN_THRESHOLDS = tuple(round(0.30 + 0.05 * step, 2) for step in range(14))  # 0.30 .. 0.95
DEFAULT_MANUAL_TEST = Path("data/eval/german_test_set.csv")
DEFAULT_OOS = Path("data/eval/out_of_scope.csv")
STRESS_OOS = Path("data/processed/stress_test_messages.csv")


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


def predict_texts(
    service: PredictorService, model_name: str, texts: list[str]
) -> tuple[list[str], list[float]]:
    """Predict categories and confidences for a list of messages."""
    predictions = [service.predict(text, model_name, explain=False) for text in texts]
    return [item.category for item in predictions], [item.confidence for item in predictions]


def labeled_metrics(
    expected: list[str], predicted: list[str], confidences: list[float]
) -> dict[str, object]:
    """Accuracy, macro F1, per-class report and threshold analysis for one set."""
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(f1_score(expected, predicted, average="macro")),
        "classification_report": classification_report(
            expected, predicted, output_dict=True, zero_division=0
        ),
        "confidence_thresholds": confidence_threshold_analysis(expected, predicted, confidences),
    }


def per_source_metrics(
    sources: list[str], expected: list[str], predicted: list[str]
) -> dict[str, dict[str, float]]:
    """Accuracy and macro F1 for each value of the source column."""
    result: dict[str, dict[str, float]] = {}
    for source in sorted(set(sources)):
        indices = [index for index, value in enumerate(sources) if value == source]
        subset_expected = [expected[index] for index in indices]
        subset_predicted = [predicted[index] for index in indices]
        result[source] = {
            "n": len(indices),
            "accuracy": float(accuracy_score(subset_expected, subset_predicted)),
            "macro_f1": float(f1_score(subset_expected, subset_predicted, average="macro")),
        }
    return result


def out_of_scope_metrics(confidences: list[float]) -> dict[str, object]:
    """Mean confidence and review-flag rate per threshold for out-of-scope messages."""
    count = len(confidences)
    return {
        "n": count,
        "mean_confidence": float(sum(confidences) / count) if count else None,
        "review_flag_rates": {
            f"{threshold:.2f}": (
                float(sum(value < threshold for value in confidences) / count) if count else None
            )
            for threshold in CONFIDENCE_THRESHOLDS
        },
    }


def recommend_threshold(
    expected: list[str],
    predicted: list[str],
    confidences: list[float],
    ood_confidences: list[float] | None,
    target_accuracy: float,
    basis: str = "test_split",
) -> dict[str, object]:
    """Lowest threshold in 0.30..0.95 whose accepted accuracy reaches the target.

    This only reports a value; it never changes the configured review threshold.
    """
    total = len(expected)
    result: dict[str, object] = {
        "threshold": None,
        "accepted_coverage": None,
        "accepted_accuracy": None,
        "ood_flag_rate": None,
        "target_accuracy": float(target_accuracy),
        "basis": basis,
    }
    if total == 0:
        return result
    for threshold in SCAN_THRESHOLDS:
        accepted = [index for index, value in enumerate(confidences) if value >= threshold]
        if not accepted:
            continue
        accuracy = sum(expected[index] == predicted[index] for index in accepted) / len(accepted)
        if accuracy >= target_accuracy:
            result.update(
                threshold=threshold,
                accepted_coverage=float(len(accepted) / total),
                accepted_accuracy=float(accuracy),
                ood_flag_rate=(
                    float(sum(value < threshold for value in ood_confidences) / len(ood_confidences))
                    if ood_confidences
                    else None
                ),
            )
            break
    return result


def load_manual_test(path: Path) -> pd.DataFrame:
    """Load the independent hand-written test set (columns message, category)."""
    frame = pd.read_csv(path)
    missing = {"message", "category"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    frame = frame.dropna(subset=["message", "category"]).copy()
    frame["message"] = frame["message"].astype(str)
    frame["category"] = frame["category"].astype(str).str.strip()
    return frame[(frame["message"].str.strip() != "") & (frame["category"] != "")].reset_index(
        drop=True
    )


def load_ood_messages(path: Path) -> list[str]:
    """Load out-of-scope messages from the message column (also used by the stress file)."""
    frame = pd.read_csv(path)
    if "message" not in frame.columns:
        raise ValueError(f"{path} needs a message column.")
    texts = frame["message"].dropna().astype(str)
    return [text for text in texts if text.strip()]


def resolve_optional_path(explicit: Path | None, candidates: tuple[Path, ...]) -> Path | None:
    """Use the explicit path, else the first existing default, else None."""
    if explicit is not None:
        return explicit
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _percent(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def build_results_markdown(
    results: dict[str, dict[str, object]], datasets: dict[str, tuple[str, int] | None]
) -> str:
    """Render a compact markdown summary using only numbers from the run."""
    used = [f"{name}: {info[0]} ({info[1]} rows)" for name, info in datasets.items() if info]
    skipped = [name for name, info in datasets.items() if not info]
    lines = ["# Evaluation results", "", "Datasets: " + "; ".join(used) + "."]
    if skipped:
        lines.append("Skipped (not available): " + ", ".join(skipped) + ".")
    for model_name, model in results.items():
        lines += ["", f"## {model_name}", ""]
        lines += ["| Dataset | Rows | Accuracy | Macro F1 |", "| --- | --- | --- | --- |"]
        main_rows = datasets["test split"][1] if datasets.get("test split") else 0
        lines.append(
            f"| test split | {main_rows} | {_percent(model['accuracy'])} | "
            f"{_percent(model['macro_f1'])} |"
        )
        manual = model.get("manual_test")
        if manual:
            lines.append(
                f"| manual_test | {datasets['manual_test'][1]} | "
                f"{_percent(manual['accuracy'])} | {_percent(manual['macro_f1'])} |"
            )
        lines += [
            "",
            "Accepted coverage / accepted accuracy by threshold:",
            "",
            "| Threshold | Split coverage | Split accuracy | Manual coverage | Manual accuracy |",
            "| --- | --- | --- | --- | --- |",
        ]
        split_analysis = model["confidence_thresholds"]
        manual_analysis = manual["confidence_thresholds"] if manual else {}
        for key, row in split_analysis.items():
            other = manual_analysis.get(key, {})
            lines.append(
                f"| {key} | {_percent(row['accepted_coverage'])} | "
                f"{_percent(row['accepted_accuracy'])} | "
                f"{_percent(other.get('accepted_coverage'))} | "
                f"{_percent(other.get('accepted_accuracy'))} |"
            )
        oos = model.get("out_of_scope")
        if oos:
            lines += [
                "",
                f"Out-of-scope: n={oos['n']}, mean confidence {oos['mean_confidence']:.3f}. "
                "Share flagged for review:",
                "",
                "| Threshold | Flagged |",
                "| --- | --- |",
            ]
            lines += [
                f"| {key} | {_percent(value)} |" for key, value in oos["review_flag_rates"].items()
            ]
        rec = model["recommended_threshold"]
        lines.append("")
        if rec["threshold"] is None:
            lines.append(
                f"Recommended threshold: none reaches accepted accuracy "
                f"{rec['target_accuracy']:.2f} on {rec['basis']}."
            )
        else:
            lines.append(
                f"Recommended threshold: {rec['threshold']:.2f} (basis {rec['basis']}, target "
                f"accuracy {rec['target_accuracy']:.2f}): accepted coverage "
                f"{_percent(rec['accepted_coverage'])}, accepted accuracy "
                f"{_percent(rec['accepted_accuracy'])}, out-of-scope flag rate "
                f"{_percent(rec['ood_flag_rate'])}."
            )
    return "\n".join(lines) + "\n"


def save_confusion_matrix(
    expected: list[str], predicted: list[str], labels: list[str], path: Path
) -> None:
    display = ConfusionMatrixDisplay.from_predictions(
        expected,
        predicted,
        labels=labels,
        xticks_rotation=45,
        cmap="Blues",
        colorbar=False,
    )
    display.figure_.tight_layout()
    display.figure_.savefig(path, dpi=150)
    plt.close(display.figure_)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved classifiers on the shared test split.")
    parser.add_argument("--data", type=Path, default=Path("data/sample_requests.csv"))
    parser.add_argument("--text-column", default="message")
    parser.add_argument("--label-column", default="category")
    parser.add_argument("--artifact-dir", type=Path, default=Path("model_artifacts"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--model", choices=["available", "baseline", "transformer"], default="available")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manual-test", type=Path, default=None,
                        help="Hand-written test CSV (message,category). Default: data/eval/german_test_set.csv if present.")
    parser.add_argument("--ood-data", type=Path, default=None,
                        help="Out-of-scope CSV with a message column. Default: data/eval/out_of_scope.csv, "
                        "else data/processed/stress_test_messages.csv, if present.")
    parser.add_argument("--target-accuracy", type=float, default=0.95,
                        help="Accepted-accuracy target used to recommend a review threshold.")
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

    manual_path = resolve_optional_path(args.manual_test, (DEFAULT_MANUAL_TEST,))
    manual = load_manual_test(manual_path) if manual_path else None
    ood_path = resolve_optional_path(args.ood_data, (DEFAULT_OOS, STRESS_OOS))
    ood_texts = load_ood_messages(ood_path) if ood_path else []
    datasets: dict[str, tuple[str, int] | None] = {
        "test split": (str(args.data), len(test)),
        "manual_test": (str(manual_path), len(manual)) if manual is not None else None,
        "out_of_scope": (str(ood_path), len(ood_texts)) if ood_texts else None,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, object]] = {}
    expected = test[args.label_column].tolist()
    labels = sorted(test[args.label_column].unique().tolist())

    for model_name in models:
        predicted, confidences = predict_texts(service, model_name, test[args.text_column].tolist())
        result = labeled_metrics(expected, predicted, confidences)
        if "source" in test.columns:
            result["per_source"] = per_source_metrics(
                test["source"].fillna("unknown").astype(str).tolist(), expected, predicted
            )
        basis_args = (expected, predicted, confidences, "test_split")
        if manual is not None and len(manual):
            manual_expected = manual["category"].tolist()
            manual_predicted, manual_confidences = predict_texts(
                service, model_name, manual["message"].tolist()
            )
            result["manual_test"] = labeled_metrics(
                manual_expected, manual_predicted, manual_confidences
            )
            basis_args = (manual_expected, manual_predicted, manual_confidences, "manual_test")
        ood_confidences: list[float] | None = None
        if ood_texts:
            _, ood_confidences = predict_texts(service, model_name, ood_texts)
            result["out_of_scope"] = out_of_scope_metrics(ood_confidences)
        result["recommended_threshold"] = recommend_threshold(
            basis_args[0], basis_args[1], basis_args[2], ood_confidences,
            args.target_accuracy, basis_args[3],
        )
        results[model_name] = result
        save_confusion_matrix(
            expected, predicted, labels, args.report_dir / f"confusion_{model_name}.png"
        )

    (args.report_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.report_dir / "results.md").write_text(
        build_results_markdown(results, datasets), encoding="utf-8"
    )
    print(f"Saved evaluation reports to {args.report_dir}")


if __name__ == "__main__":
    main()
