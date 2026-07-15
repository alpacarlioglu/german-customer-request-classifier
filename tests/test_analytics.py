import pandas as pd

from src.analytics import confusion_pairs, model_feedback_accuracy, summary_metrics


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "confidence": 0.9,
                "user_feedback": True,
                "selected_model": "tfidf_logreg",
                "predicted_category": "Rechnung",
                "corrected_category": None,
            },
            {
                "confidence": 0.4,
                "user_feedback": False,
                "selected_model": "tfidf_logreg",
                "predicted_category": "Rechnung",
                "corrected_category": "Kündigung",
            },
        ]
    )


def test_feedback_metrics_and_confusions() -> None:
    frame = sample_frame()

    metrics = summary_metrics(frame, review_threshold=0.65)
    pairs = confusion_pairs(frame)
    accuracy = model_feedback_accuracy(frame)

    assert metrics["feedback_accuracy"] == 0.5
    assert metrics["manual_review_rate"] == 0.5
    assert pairs.iloc[0]["corrected_category"] == "Kündigung"
    assert accuracy.iloc[0]["accuracy"] == 0.5

