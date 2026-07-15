from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analytics import (
    confusion_pairs,
    model_feedback_accuracy,
    records_frame,
    summary_metrics,
)
from src.app_services import get_database, get_settings


st.title("Analytics")

try:
    records = get_database().list_predictions()
    settings = get_settings()
except Exception as exc:
    st.error(f"Could not load analytics: {exc}")
    st.stop()

frame = records_frame(records)
if frame.empty:
    st.info("Analytics will appear after predictions have been stored.")
    st.stop()

metrics = summary_metrics(frame, settings.review_threshold)
total_column, reviewed_column, accuracy_column, review_column = st.columns(4)
total_column.metric("Requests", metrics["total"])
reviewed_column.metric("Reviewed", metrics["reviewed"])
accuracy_column.metric("Feedback accuracy", f'{metrics["feedback_accuracy"]:.1%}')
review_column.metric("Manual-review rate", f'{metrics["manual_review_rate"]:.1%}')

category_chart, confidence_chart = st.columns(2)
with category_chart:
    st.subheader("Requests per category")
    category_counts = frame["predicted_category"].value_counts().sort_values(ascending=True)
    st.bar_chart(category_counts)

with confidence_chart:
    st.subheader("Confidence distribution")
    bins = pd.cut(frame["confidence"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    confidence_counts = bins.value_counts(sort=False)
    confidence_counts.index = confidence_counts.index.astype(str)
    st.bar_chart(confidence_counts)

st.subheader("Most frequent category confusions")
pairs = confusion_pairs(frame)
if pairs.empty:
    st.info("No corrected predictions are available yet.")
else:
    display_pairs = pairs.copy()
    display_pairs["confusion"] = (
        display_pairs["predicted_category"] + " → " + display_pairs["corrected_category"]
    )
    st.bar_chart(display_pairs.set_index("confusion")["count"])
    st.dataframe(pairs, use_container_width=True, hide_index=True)

st.subheader("Model accuracy from user feedback")
accuracy = model_feedback_accuracy(frame)
if accuracy.empty:
    st.info("No reviewed predictions are available yet.")
else:
    chart = accuracy.set_index("selected_model")["accuracy"]
    st.bar_chart(chart)
    formatted = accuracy.copy()
    formatted["accuracy"] = formatted["accuracy"].map(lambda value: f"{value:.1%}")
    st.dataframe(formatted, use_container_width=True, hide_index=True)

