from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return softmax probabilities of logits / temperature (numerically stable)."""
    scaled = np.asarray(logits, dtype=float) / float(temperature)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def negative_log_likelihood(
    logits: np.ndarray, label_indices: np.ndarray, temperature: float
) -> float:
    """Mean negative log-likelihood of the true labels after temperature scaling."""
    scaled = np.asarray(logits, dtype=float) / float(temperature)
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    log_probabilities = scaled - np.log(np.exp(scaled).sum(axis=-1, keepdims=True))
    rows = np.arange(len(label_indices))
    return float(-log_probabilities[rows, np.asarray(label_indices, dtype=int)].mean())


def fit_temperature(logits: np.ndarray, label_indices: np.ndarray) -> float:
    """Fit one temperature by minimising the negative log-likelihood over log T."""
    logits = np.asarray(logits, dtype=float)
    label_indices = np.asarray(label_indices, dtype=int)
    if logits.ndim != 2 or len(logits) == 0 or len(logits) != len(label_indices):
        raise ValueError("logits must be 2-D and match label_indices in length.")

    def objective(log_temperature: float) -> float:
        return negative_log_likelihood(logits, label_indices, float(np.exp(log_temperature)))

    result = minimize_scalar(
        objective,
        bounds=(float(np.log(MIN_TEMPERATURE)), float(np.log(MAX_TEMPERATURE))),
        method="bounded",
    )
    return float(np.clip(np.exp(result.x), MIN_TEMPERATURE, MAX_TEMPERATURE))
