from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.predictor import BASELINE, PredictorService


def create_test_model(artifact_dir: Path) -> None:
    messages = [
        "Meine Rechnung ist falsch",
        "Die Rechnung ist zu hoch",
        "Bitte meinen Vertrag kündigen",
        "Ich möchte sofort kündigen",
    ]
    labels = ["Rechnung", "Rechnung", "Kündigung", "Kündigung"]
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    pipeline.fit(messages, labels)
    model_dir = artifact_dir / "baseline"
    model_dir.mkdir(parents=True)
    joblib.dump(pipeline, model_dir / "model.joblib")


def test_predicts_with_confidence_and_review_flag(tmp_path: Path) -> None:
    create_test_model(tmp_path)
    service = PredictorService(tmp_path, review_threshold=0.99)

    prediction = service.predict("Meine Rechnung stimmt nicht", BASELINE)

    assert prediction.category in {"Rechnung", "Kündigung"}
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.needs_review is True
    assert prediction.model_name == BASELINE
    assert service.categories(BASELINE) == ["Kündigung", "Rechnung"]


def test_masks_message_before_prediction(tmp_path: Path) -> None:
    create_test_model(tmp_path)
    service = PredictorService(tmp_path)

    prediction = service.predict("Rechnung an anna@example.de", BASELINE, explain=False)

    assert "anna@example.de" not in prediction.masked_message
    assert "<EMAIL>" in prediction.masked_message

