from __future__ import annotations

import streamlit as st

from src.app_services import get_database, get_predictor, get_settings


st.set_page_config(
    page_title="German Customer Request Classification",
    page_icon="💬",
    layout="wide",
)

st.title("German Customer Request Classification")
st.write(
    "Classify German customer messages, collect corrections, and inspect feedback-based analytics."
)

settings = get_settings()
predictor = get_predictor()
models = predictor.available_models()

left, middle, right = st.columns(3)
left.metric("Trained models", len(models))
middle.metric("Manual-review threshold", f"{settings.review_threshold:.0%}")

try:
    database = get_database()
    database.ping()
    right.metric("PostgreSQL", "Connected")
except Exception as exc:
    right.metric("PostgreSQL", "Not connected")
    st.warning(f"Database setup is required: {exc}")

if not models:
    st.info(
        "No model artifacts were found. Train the baseline with "
        "`python -m scripts.train --model baseline`, then reload the app."
    )
else:
    st.success("Available models: " + ", ".join(models))

st.subheader("Workflow")
st.markdown(
    """
1. Open **Classify** to predict one message or upload a CSV batch.
2. Confirm or correct individual predictions.
3. Use **History** to filter and export stored results.
4. Use **Analytics** to inspect confidence and feedback quality.
"""
)

st.caption(
    "Messages are masked for common email addresses, phone numbers, and account numbers before "
    "classification and storage. Review MODEL_CARD.md before using the predictions operationally."
)
