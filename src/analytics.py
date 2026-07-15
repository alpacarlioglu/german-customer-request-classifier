from __future__ import annotations

from typing import Any

import pandas as pd


COLUMNS = [
    "id",
    "message",
    "predicted_category",
    "confidence",
    "selected_model",
    "corrected_category",
    "user_feedback",
    "created_at",
]


def records_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(records, columns=COLUMNS)
    if not frame.empty:
        frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True)
    return frame


def summary_metrics(frame: pd.DataFrame, review_threshold: float) -> dict[str, float | int]:
    if frame.empty:
        return {"total": 0, "reviewed": 0, "feedback_accuracy": 0.0, "manual_review_rate": 0.0}

    reviewed = frame[frame["user_feedback"].notna()]
    confirmed = reviewed[reviewed["user_feedback"] == True]  # noqa: E712
    return {
        "total": len(frame),
        "reviewed": len(reviewed),
        "feedback_accuracy": len(confirmed) / len(reviewed) if len(reviewed) else 0.0,
        "manual_review_rate": float((frame["confidence"] < review_threshold).mean()),
    }


def confusion_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["predicted_category", "corrected_category", "count"])
    corrected = frame[frame["user_feedback"] == False]  # noqa: E712
    if corrected.empty:
        return pd.DataFrame(columns=["predicted_category", "corrected_category", "count"])
    return (
        corrected.groupby(["predicted_category", "corrected_category"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def model_feedback_accuracy(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["selected_model", "accuracy", "reviewed"])
    reviewed = frame[frame["user_feedback"].notna()].copy()
    if reviewed.empty:
        return pd.DataFrame(columns=["selected_model", "accuracy", "reviewed"])
    result = reviewed.groupby("selected_model")["user_feedback"].agg(["mean", "count"]).reset_index()
    return result.rename(columns={"mean": "accuracy", "count": "reviewed"})

