from __future__ import annotations

import streamlit as st

from src.config import Settings, load_settings
from src.database import Database
from src.predictor import PredictorService


@st.cache_resource
def get_settings() -> Settings:
    return load_settings()


@st.cache_resource
def get_database() -> Database:
    settings = get_settings()
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is not configured. Copy .env.example to .env.")
    database = Database(settings.database_url)
    database.create_schema()
    return database


@st.cache_resource
def get_predictor() -> PredictorService:
    settings = get_settings()
    return PredictorService(settings.artifact_dir, settings.review_threshold)

