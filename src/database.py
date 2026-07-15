from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    selected_model: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    corrected_category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    user_feedback: Mapped[bool | None] = mapped_column(Boolean, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )


class Database:
    def __init__(self, database_url: str):
        if not database_url:
            raise ValueError("DATABASE_URL is required.")
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def drop_schema(self) -> None:
        Base.metadata.drop_all(self.engine)

    def ping(self) -> None:
        with self.engine.connect() as connection:
            connection.execute(select(1))

    def add_prediction(
        self, message: str, predicted_category: str, confidence: float, selected_model: str
    ) -> int:
        record = PredictionRecord(
            message=message,
            predicted_category=predicted_category,
            confidence=confidence,
            selected_model=selected_model,
        )
        with self.sessions.begin() as session:
            session.add(record)
            session.flush()
            return record.id

    def add_predictions(self, predictions: Iterable[dict[str, Any]]) -> list[int]:
        records = [PredictionRecord(**item) for item in predictions]
        with self.sessions.begin() as session:
            session.add_all(records)
            session.flush()
            return [record.id for record in records]

    def update_feedback(
        self, record_id: int, confirmed: bool, corrected_category: str | None = None
    ) -> None:
        if not confirmed and not corrected_category:
            raise ValueError("A corrected category is required when the prediction is rejected.")

        with self.sessions.begin() as session:
            record = session.get(PredictionRecord, record_id)
            if record is None:
                raise LookupError(f"Prediction {record_id} was not found.")
            record.user_feedback = confirmed
            record.corrected_category = None if confirmed else corrected_category

    def list_predictions(
        self,
        *,
        category: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
        feedback: bool | None | str = "all",
        selected_model: str | None = None,
        limit: int = 10_000,
    ) -> list[dict[str, Any]]:
        statement = select(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit)
        if category:
            statement = statement.where(PredictionRecord.predicted_category == category)
        if min_confidence is not None:
            statement = statement.where(PredictionRecord.confidence >= min_confidence)
        if max_confidence is not None:
            statement = statement.where(PredictionRecord.confidence <= max_confidence)
        if feedback == "unreviewed":
            statement = statement.where(PredictionRecord.user_feedback.is_(None))
        elif feedback is True or feedback is False:
            statement = statement.where(PredictionRecord.user_feedback.is_(feedback))
        if selected_model:
            statement = statement.where(PredictionRecord.selected_model == selected_model)

        with Session(self.engine) as session:
            records = session.scalars(statement).all()
            return [self._as_dict(record) for record in records]

    @staticmethod
    def _as_dict(record: PredictionRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "message": record.message,
            "predicted_category": record.predicted_category,
            "confidence": record.confidence,
            "selected_model": record.selected_model,
            "corrected_category": record.corrected_category,
            "user_feedback": record.user_feedback,
            "created_at": record.created_at,
        }

