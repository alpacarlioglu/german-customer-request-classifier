from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.database import Database


@pytest.fixture
def database(tmp_path: Path):
    url = os.getenv("TEST_DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    db = Database(url)
    db.drop_schema()
    db.create_schema()
    yield db
    db.drop_schema()
    db.engine.dispose()


def test_insert_and_confirm_prediction(database: Database) -> None:
    record_id = database.add_prediction(
        "Meine E-Mail ist <EMAIL>", "Rechnung", 0.82, "tfidf_logreg"
    )
    database.update_feedback(record_id, confirmed=True)

    records = database.list_predictions(feedback=True)

    assert len(records) == 1
    assert records[0]["id"] == record_id
    assert records[0]["user_feedback"] is True
    assert records[0]["corrected_category"] is None


def test_correction_and_filters(database: Database) -> None:
    record_id = database.add_prediction(
        "Mein Anschluss funktioniert nicht", "Rechnung", 0.42, "tfidf_logreg"
    )
    database.update_feedback(record_id, confirmed=False, corrected_category="Technischer Support")

    records = database.list_predictions(
        category="Rechnung", min_confidence=0.4, max_confidence=0.5, feedback=False
    )

    assert len(records) == 1
    assert records[0]["corrected_category"] == "Technischer Support"


def test_batch_insert_and_unreviewed_filter(database: Database) -> None:
    ids = database.add_predictions(
        [
            {
                "message": "Nachricht eins",
                "predicted_category": "Rechnung",
                "confidence": 0.7,
                "selected_model": "tfidf_logreg",
            },
            {
                "message": "Nachricht zwei",
                "predicted_category": "Kündigung",
                "confidence": 0.6,
                "selected_model": "tfidf_logreg",
            },
        ]
    )

    assert len(ids) == 2
    assert len(database.list_predictions(feedback="unreviewed")) == 2


def test_rejected_prediction_requires_correction(database: Database) -> None:
    record_id = database.add_prediction("Text", "Rechnung", 0.5, "tfidf_logreg")
    with pytest.raises(ValueError):
        database.update_feedback(record_id, confirmed=False)

