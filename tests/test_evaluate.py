import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import evaluate, train
from scripts.evaluate import (
    confidence_threshold_analysis,
    out_of_scope_metrics,
    per_source_metrics,
    recommend_threshold,
)


def test_confidence_threshold_analysis() -> None:
    analysis = confidence_threshold_analysis(["a", "b", "a"], ["a", "a", "a"], [0.9, 0.55, 0.4])

    assert analysis["0.50"]["accepted_coverage"] == pytest.approx(2 / 3)
    assert analysis["0.50"]["accepted_accuracy"] == pytest.approx(0.5)
    assert analysis["0.80"]["accepted_accuracy"] == 1.0
    assert analysis["0.80"]["manual_review_rate"] == pytest.approx(2 / 3)


def test_confidence_threshold_analysis_empty() -> None:
    analysis = confidence_threshold_analysis([], [], [])

    assert analysis["0.50"]["accepted_accuracy"] is None
    assert analysis["0.50"]["accepted_coverage"] == 0.0


def test_recommend_threshold_picks_lowest_meeting_target() -> None:
    expected = ["a"] * 4
    predicted = ["a", "a", "b", "b"]
    confidences = [0.95, 0.9, 0.5, 0.4]

    result = recommend_threshold(expected, predicted, confidences, [0.4, 0.6, 0.9], 0.95)

    assert result["threshold"] == 0.55
    assert result["accepted_accuracy"] == 1.0
    assert result["accepted_coverage"] == 0.5
    assert result["ood_flag_rate"] == pytest.approx(1 / 3)
    assert result["basis"] == "test_split"


def test_recommend_threshold_when_none_meets_target() -> None:
    result = recommend_threshold(["a", "a"], ["b", "b"], [0.9, 0.9], None, 0.95, "manual_test")

    assert result["threshold"] is None
    assert result["ood_flag_rate"] is None
    assert result["basis"] == "manual_test"


def test_recommend_threshold_without_ood_and_empty_inputs() -> None:
    assert recommend_threshold(["a"], ["a"], [0.9], None, 0.95)["ood_flag_rate"] is None
    assert recommend_threshold([], [], [], [], 0.95)["threshold"] is None


def test_out_of_scope_metrics() -> None:
    result = out_of_scope_metrics([0.4, 0.55, 0.9, 0.7])

    assert result["n"] == 4
    assert result["mean_confidence"] == pytest.approx(0.6375)
    assert result["review_flag_rates"]["0.50"] == pytest.approx(0.25)
    assert result["review_flag_rates"]["0.80"] == pytest.approx(0.75)
    assert out_of_scope_metrics([])["mean_confidence"] is None


def test_per_source_metrics() -> None:
    result = per_source_metrics(["x", "x", "y", "y"], ["a", "b", "a", "b"], ["a", "a", "a", "b"])

    assert result["x"]["accuracy"] == 0.5
    assert result["y"]["accuracy"] == 1.0
    assert result["y"]["n"] == 2


def make_dataset(path: Path) -> None:
    topics = {
        "Rechnung": "Meine Rechnung ist falsch Betrag {i}",
        "Kündigung": "Ich möchte meinen Vertrag kündigen Nummer {i}",
        "Adresse": "Meine neue Adresse lautet Straße {i}",
    }
    rows = [
        {"message": template.format(i=i), "category": category, "source": "a" if i % 2 else "b"}
        for category, template in topics.items()
        for i in range(12)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_main_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data.csv"
    make_dataset(data)
    manual = tmp_path / "manual.csv"
    pd.DataFrame(
        {"message": ["Rechnung falsch", "Vertrag kündigen"], "category": ["Rechnung", "Kündigung"]}
    ).to_csv(manual, index=False)
    ood = tmp_path / "ood.csv"
    pd.DataFrame({"message": ["Wie wird das Wetter morgen", "Spiel Musik"]}).to_csv(ood, index=False)
    artifacts, reports = tmp_path / "artifacts", tmp_path / "reports"

    monkeypatch.setattr(sys, "argv", ["train", "--data", str(data), "--output-dir", str(artifacts)])
    train.main()
    metadata = json.loads((artifacts / "baseline" / "metadata.json").read_text(encoding="utf-8"))
    assert "temperature" in metadata and "calibration" in metadata

    monkeypatch.chdir(tmp_path)  # no default eval files exist here
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate", "--data", str(data), "--artifact-dir", str(artifacts),
         "--report-dir", str(reports), "--manual-test", str(manual), "--ood-data", str(ood),
         "--target-accuracy", "0.9"],
    )
    evaluate.main()

    metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))["tfidf_logreg"]
    for key in ("accuracy", "macro_f1", "classification_report", "confidence_thresholds",
                "manual_test", "out_of_scope", "per_source", "recommended_threshold"):
        assert key in metrics
    assert metrics["out_of_scope"]["n"] == 2
    assert metrics["recommended_threshold"]["basis"] == "manual_test"
    assert (reports / "confusion_tfidf_logreg.png").exists()
    markdown = (reports / "results.md").read_text(encoding="utf-8")
    assert "tfidf_logreg" in markdown and "manual_test" in markdown and "Recommended threshold" in markdown


def test_main_without_optional_sets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data.csv"
    make_dataset(data)
    artifacts, reports = tmp_path / "artifacts", tmp_path / "reports"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(data), "--output-dir", str(artifacts)])
    train.main()
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate", "--data", str(data), "--artifact-dir", str(artifacts), "--report-dir", str(reports)],
    )
    evaluate.main()

    metrics = json.loads((reports / "metrics.json").read_text(encoding="utf-8"))["tfidf_logreg"]
    assert "manual_test" not in metrics and "out_of_scope" not in metrics
    assert metrics["recommended_threshold"]["basis"] == "test_split"
    assert "Skipped" in (reports / "results.md").read_text(encoding="utf-8")
