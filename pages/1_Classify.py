from __future__ import annotations

import html
import re

import pandas as pd
import streamlit as st

from src.app_services import get_database, get_predictor


def highlighted_message(message: str, importance: list[tuple[str, float]]) -> str:
    term_scores: dict[str, float] = {}
    for term, score in importance:
        for word in term.lower().split():
            term_scores[word] = max(score, term_scores.get(word, 0.0))

    parts: list[str] = []
    for part in re.split(r"(\s+)", message):
        cleaned = re.sub(r"[^\wÄÖÜäöüß<>]", "", part).lower()
        score = term_scores.get(cleaned, 0.0)
        safe_part = html.escape(part)
        if score > 0:
            alpha = 0.15 + (0.55 * score)
            parts.append(
                f'<span style="background: rgba(255, 193, 7, {alpha:.2f}); '
                f'padding: 0.08rem 0.12rem; border-radius: 0.2rem">{safe_part}</span>'
            )
        else:
            parts.append(safe_part)
    return "".join(parts)


st.title("Classify")

try:
    database = get_database()
    predictor = get_predictor()
    models = predictor.available_models()
except Exception as exc:
    st.error(f"The application is not ready: {exc}")
    st.stop()

if not models:
    st.warning("Train at least one model before classifying messages.")
    st.stop()

single_tab, batch_tab = st.tabs(["Single message", "CSV batch"])

with single_tab:
    model_name = st.selectbox("Model", models, key="single_model")
    with st.form("classification_form"):
        message = st.text_area(
            "German customer message",
            height=140,
            placeholder="Meine Rechnung ist zu hoch. Bitte prüfen Sie den Betrag.",
        )
        classify = st.form_submit_button("Classify", type="primary")

    if classify:
        try:
            prediction = predictor.predict(message, model_name, explain=True)
            record_id = database.add_prediction(
                prediction.masked_message,
                prediction.category,
                prediction.confidence,
                prediction.model_name,
            )
            st.session_state["last_prediction"] = prediction
            st.session_state["last_record_id"] = record_id
            st.session_state.pop("feedback_saved", None)
        except Exception as exc:
            st.error(f"Classification failed: {exc}")

    prediction = st.session_state.get("last_prediction")
    record_id = st.session_state.get("last_record_id")
    if prediction and record_id:
        category_column, confidence_column, review_column = st.columns(3)
        category_column.metric("Predicted category", prediction.category)
        confidence_column.metric("Confidence", f"{prediction.confidence:.1%}")
        review_column.metric("Manual review", "Required" if prediction.needs_review else "No")

        if prediction.needs_review:
            st.warning("Confidence is below the configured threshold. Please review this prediction.")
        st.markdown("**Masked message and influential words**")
        st.markdown(
            highlighted_message(prediction.masked_message, prediction.word_importance),
            unsafe_allow_html=True,
        )
        if not prediction.word_importance:
            st.caption("No positive word contributions were available for this prediction.")

        categories = predictor.categories(prediction.model_name)
        with st.form("feedback_form"):
            feedback_choice = st.radio(
                "Is the prediction correct?", ["Confirm", "Correct"], horizontal=True
            )
            corrected_category = st.selectbox(
                "Correct category",
                categories,
                index=categories.index(prediction.category),
                help="This value is used only when `Correct` is selected.",
            )
            save_feedback = st.form_submit_button("Save feedback")
        if save_feedback:
            try:
                confirmed = feedback_choice == "Confirm"
                if not confirmed and corrected_category == prediction.category:
                    st.error("Choose a category different from the prediction.")
                else:
                    database.update_feedback(
                        record_id,
                        confirmed=confirmed,
                        corrected_category=None if confirmed else corrected_category,
                    )
                    st.session_state["feedback_saved"] = True
                    st.success("Feedback saved.")
            except Exception as exc:
                st.error(f"Could not save feedback: {exc}")

with batch_tab:
    batch_model = st.selectbox("Model", models, key="batch_model")
    uploaded = st.file_uploader("CSV file with a `message` column", type="csv")
    if uploaded is not None and st.button("Classify batch", type="primary"):
        try:
            source = pd.read_csv(uploaded)
            if "message" not in source.columns:
                raise ValueError("The CSV must contain a `message` column.")
            if len(source) > 500:
                raise ValueError("A batch can contain at most 500 rows.")
            if source.empty:
                raise ValueError("The CSV contains no rows.")

            output_rows: list[dict[str, object]] = []
            database_rows: list[dict[str, object]] = []
            progress = st.progress(0.0)
            for position, raw_message in enumerate(source["message"].fillna("").astype(str)):
                prediction = predictor.predict(raw_message, batch_model, explain=False)
                output_rows.append(
                    {
                        "message": prediction.masked_message,
                        "predicted_category": prediction.category,
                        "confidence": prediction.confidence,
                        "needs_review": prediction.needs_review,
                        "selected_model": prediction.model_name,
                    }
                )
                database_rows.append(
                    {
                        "message": prediction.masked_message,
                        "predicted_category": prediction.category,
                        "confidence": prediction.confidence,
                        "selected_model": prediction.model_name,
                    }
                )
                progress.progress((position + 1) / len(source))
            database.add_predictions(database_rows)
            st.session_state["batch_results"] = pd.DataFrame(output_rows)
            st.success(f"Classified and stored {len(output_rows)} messages.")
        except Exception as exc:
            st.error(f"Batch classification failed: {exc}")

    batch_results = st.session_state.get("batch_results")
    if batch_results is not None:
        st.dataframe(batch_results, use_container_width=True, hide_index=True)
        st.download_button(
            "Download batch results",
            batch_results.to_csv(index=False).encode("utf-8"),
            file_name="classified_requests.csv",
            mime="text/csv",
        )
