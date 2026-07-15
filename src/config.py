from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    database_url: str | None
    review_threshold: float
    artifact_dir: Path


def load_settings() -> Settings:
    """Load application settings from the environment and an optional .env file."""
    load_dotenv(PROJECT_ROOT / ".env")
    threshold = float(os.getenv("REVIEW_THRESHOLD", "0.65"))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("REVIEW_THRESHOLD must be between 0 and 1.")

    artifact_dir = Path(os.getenv("MODEL_ARTIFACT_DIR", "model_artifacts"))
    if not artifact_dir.is_absolute():
        artifact_dir = PROJECT_ROOT / artifact_dir

    return Settings(
        database_url=os.getenv("DATABASE_URL"),
        review_threshold=threshold,
        artifact_dir=artifact_dir,
    )

