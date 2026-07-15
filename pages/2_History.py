from __future__ import annotations

import streamlit as st

from src.analytics import records_frame
from src.app_services import get_database


st.title("History")

try:
    database = get_database()
    all_records = database.list_predictions()
except Exception as exc:
    st.error(f"Could not load prediction history: {exc}")
    st.stop()

if not all_records:
    st.info("No predictions have been stored yet.")
    st.stop()

all_frame = records_frame(all_records)
categories = sorted(all_frame["predicted_category"].dropna().unique().tolist())
models = sorted(all_frame["selected_model"].dropna().unique().tolist())

category_column, confidence_column = st.columns(2)
category = category_column.selectbox("Category", ["All"] + categories)
confidence_range = confidence_column.slider(
    "Confidence range", min_value=0.0, max_value=1.0, value=(0.0, 1.0), step=0.05
)
feedback_column, model_column = st.columns(2)
feedback_label = feedback_column.selectbox(
    "Correctness", ["All", "Confirmed", "Corrected", "Unreviewed"]
)
selected_model = model_column.selectbox("Model", ["All"] + models)

feedback_value: bool | str
if feedback_label == "Confirmed":
    feedback_value = True
elif feedback_label == "Corrected":
    feedback_value = False
elif feedback_label == "Unreviewed":
    feedback_value = "unreviewed"
else:
    feedback_value = "all"

try:
    filtered = database.list_predictions(
        category=None if category == "All" else category,
        min_confidence=confidence_range[0],
        max_confidence=confidence_range[1],
        feedback=feedback_value,
        selected_model=None if selected_model == "All" else selected_model,
    )
    frame = records_frame(filtered)
except Exception as exc:
    st.error(f"Could not apply filters: {exc}")
    st.stop()

st.caption(f"{len(frame)} matching predictions")
if frame.empty:
    st.info("No records match the selected filters.")
else:
    display = frame.copy()
    display["correctness"] = display["user_feedback"].map(
        {True: "confirmed", False: "corrected"}
    ).fillna("unreviewed")
    display["confidence"] = display["confidence"].map(lambda value: f"{value:.1%}")
    columns = [
        "created_at",
        "message",
        "predicted_category",
        "confidence",
        "selected_model",
        "corrected_category",
        "correctness",
    ]
    st.dataframe(display[columns], use_container_width=True, hide_index=True)
    st.download_button(
        "Export filtered history",
        frame.to_csv(index=False).encode("utf-8"),
        file_name="prediction_history.csv",
        mime="text/csv",
    )

