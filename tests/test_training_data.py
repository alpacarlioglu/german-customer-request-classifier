from pathlib import Path

import pandas as pd

from scripts.evaluate import confidence_threshold_analysis
from scripts.train import dataset_metadata, load_dataset, split_dataset


def test_load_dataset_preserves_preparation_columns(tmp_path: Path) -> None:
    path = tmp_path / "prepared.csv"
    rows = []
    for category in ("A", "B"):
        for index in range(3):
            rows.append(
                {
                    "message": f"Nachricht {category} {index}",
                    "category": category,
                    "source": "fixture",
                    "source_label": "original",
                    "license": "test",
                    "review_status": "reviewed",
                    "split": ("train", "validation", "test")[index],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)

    frame = load_dataset(path, "message", "category")

    assert {"source", "source_label", "license", "review_status", "split"}.issubset(
        frame.columns
    )


def test_split_dataset_uses_prepared_assignments_exactly() -> None:
    frame = pd.DataFrame(
        {
            "message": ["one", "two", "three", "four", "five", "six"],
            "category": ["A", "B", "A", "B", "A", "B"],
            "split": ["train", "train", "validation", "validation", "test", "test"],
        }
    )

    train, validation, test = split_dataset(frame, "category", seed=42)

    assert train["message"].tolist() == ["one", "two"]
    assert validation["message"].tolist() == ["three", "four"]
    assert test["message"].tolist() == ["five", "six"]


def test_dataset_metadata_counts_rows_categories_and_sources() -> None:
    frame = pd.DataFrame(
        {
            "category": ["A", "A", "B"],
            "source": ["one", "two", "one"],
        }
    )

    assert dataset_metadata(frame, "category") == {
        "row_count": 3,
        "category_counts": {"A": 2, "B": 1},
        "source_counts": {"one": 2, "two": 1},
    }


def test_confidence_threshold_analysis_reports_accuracy_and_coverage() -> None:
    result = confidence_threshold_analysis(
        expected=["A", "B", "A", "B"],
        predicted=["A", "A", "A", "B"],
        confidences=[0.90, 0.75, 0.55, 0.40],
    )

    assert result["0.50"] == {
        "accepted_coverage": 0.75,
        "accepted_accuracy": 2 / 3,
        "manual_review_rate": 0.25,
    }
    assert result["0.80"] == {
        "accepted_coverage": 0.25,
        "accepted_accuracy": 1.0,
        "manual_review_rate": 0.75,
    }
    assert result["0.65"] == {
        "accepted_coverage": 0.5,
        "accepted_accuracy": 0.5,
        "manual_review_rate": 0.5,
    }
