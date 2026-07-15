from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.preprocessing import preprocess_text


BASELINE = "tfidf_logreg"
TRANSFORMER = "transformer"


@dataclass(frozen=True)
class Prediction:
    masked_message: str
    category: str
    confidence: float
    model_name: str
    needs_review: bool
    word_importance: list[tuple[str, float]]


class PredictorService:
    """Load saved classifiers and expose one small prediction interface."""

    def __init__(self, artifact_dir: str | Path, review_threshold: float = 0.65):
        self.artifact_dir = Path(artifact_dir)
        self.review_threshold = review_threshold
        self._loaded: dict[str, Any] = {}

    def available_models(self) -> list[str]:
        models: list[str] = []
        if (self.artifact_dir / "baseline" / "model.joblib").exists():
            models.append(BASELINE)
        transformer_dir = self.artifact_dir / "transformer"
        if (transformer_dir / "config.json").exists() and (
            transformer_dir / "metadata.json"
        ).exists():
            models.append(TRANSFORMER)
        return models

    def categories(self, model_name: str) -> list[str]:
        """Return the labels supported by a trained model."""
        if model_name == BASELINE:
            return [str(label) for label in self._load_baseline().classes_]
        if model_name == TRANSFORMER:
            _, _, labels, _, _ = self._load_transformer()
            return [str(label) for label in labels]
        raise ValueError(f"Unknown model: {model_name}")

    def predict(self, text: str, model_name: str, explain: bool = True) -> Prediction:
        masked = preprocess_text(text)
        if not masked:
            raise ValueError("Please enter a non-empty message.")

        if model_name == BASELINE:
            category, confidence, importance = self._predict_baseline(masked, explain)
        elif model_name == TRANSFORMER:
            category, confidence, importance = self._predict_transformer(masked, explain)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        return Prediction(
            masked_message=masked,
            category=category,
            confidence=confidence,
            model_name=model_name,
            needs_review=confidence < self.review_threshold,
            word_importance=importance,
        )

    def _load_baseline(self):
        if BASELINE not in self._loaded:
            path = self.artifact_dir / "baseline" / "model.joblib"
            if not path.exists():
                raise FileNotFoundError("Baseline model not found. Run scripts/train.py first.")
            self._loaded[BASELINE] = joblib.load(path)
        return self._loaded[BASELINE]

    def _predict_baseline(
        self, text: str, explain: bool
    ) -> tuple[str, float, list[tuple[str, float]]]:
        pipeline = self._load_baseline()
        probabilities = pipeline.predict_proba([text])[0]
        class_index = int(np.argmax(probabilities))
        category = str(pipeline.classes_[class_index])
        importance = self._baseline_importance(pipeline, text, class_index) if explain else []
        return category, float(probabilities[class_index]), importance

    @staticmethod
    def _baseline_importance(pipeline, text: str, class_index: int) -> list[tuple[str, float]]:
        vectorizer = pipeline.named_steps["tfidf"]
        classifier = pipeline.named_steps["classifier"]
        row = vectorizer.transform([text]).tocsr()
        if not row.nnz:
            return []

        coefficients = classifier.coef_
        if coefficients.shape[0] == 1:
            class_coefficients = coefficients[0] if class_index == 1 else -coefficients[0]
        else:
            class_coefficients = coefficients[class_index]

        contributions = row.data * class_coefficients[row.indices]
        names = vectorizer.get_feature_names_out()
        ranked = sorted(
            ((str(names[index]), float(score)) for index, score in zip(row.indices, contributions)),
            key=lambda item: item[1],
            reverse=True,
        )
        positive = [(word, score) for word, score in ranked if score > 0][:10]
        if not positive:
            return []
        largest = max(score for _, score in positive)
        return [(word, score / largest) for word, score in positive]

    def _load_transformer(self):
        if TRANSFORMER in self._loaded:
            return self._loaded[TRANSFORMER]

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformer dependencies are missing. Install requirements-transformer.txt."
            ) from exc

        path = self.artifact_dir / "transformer"
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path)
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        model.eval()
        self._loaded[TRANSFORMER] = (tokenizer, model, metadata["labels"], device, torch)
        return self._loaded[TRANSFORMER]

    def _transformer_probabilities(self, text: str) -> np.ndarray:
        tokenizer, model, _, device, torch = self._load_transformer()
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        return probabilities

    def _predict_transformer(
        self, text: str, explain: bool
    ) -> tuple[str, float, list[tuple[str, float]]]:
        _, _, labels, _, _ = self._load_transformer()
        probabilities = self._transformer_probabilities(text)
        class_index = int(np.argmax(probabilities))
        importance = (
            self._occlusion_importance(text, class_index, float(probabilities[class_index]))
            if explain
            else []
        )
        return str(labels[class_index]), float(probabilities[class_index]), importance

    def _occlusion_importance(
        self, text: str, class_index: int, original_probability: float
    ) -> list[tuple[str, float]]:
        words = text.split()
        scores: list[tuple[str, float]] = []
        for index, word in enumerate(words[:30]):
            shortened = " ".join(words[:index] + words[index + 1 :])
            if not shortened:
                continue
            probability = float(self._transformer_probabilities(shortened)[class_index])
            scores.append((word, max(0.0, original_probability - probability)))

        scores.sort(key=lambda item: item[1], reverse=True)
        positive = [(word, score) for word, score in scores if score > 0][:10]
        if not positive:
            return []
        largest = max(score for _, score in positive)
        return [(word, score / largest) for word, score in positive]
