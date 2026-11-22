import numpy as np

from src.calibration import apply_temperature, fit_temperature


def sample_logits(true_scale: float, rows: int = 3000, classes: int = 4, seed: int = 0):
    """Draw labels from softmax(base) and return logits scaled by true_scale."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1.5, size=(rows, classes))
    probabilities = apply_temperature(base, 1.0)
    labels = np.array([rng.choice(classes, p=row) for row in probabilities])
    return base * true_scale, labels


def test_recovers_temperature_above_one_for_overconfident_logits() -> None:
    logits, labels = sample_logits(true_scale=3.0)

    temperature = fit_temperature(logits, labels)

    assert 2.5 < temperature < 3.5


def test_temperature_near_one_for_calibrated_logits() -> None:
    logits, labels = sample_logits(true_scale=1.0)

    temperature = fit_temperature(logits, labels)

    assert 0.9 < temperature < 1.1


def test_softmax_is_stable_with_large_logits() -> None:
    logits = np.array([[1000.0, 999.0, -1000.0], [-5000.0, -5001.0, -4000.0]])

    probabilities = apply_temperature(logits, 1.0)

    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[0].argmax() == 0


def test_apply_temperature_rows_sum_to_one_and_flatten() -> None:
    logits = np.array([[2.0, 0.0, -1.0], [0.5, 0.4, 0.3]])

    sharp = apply_temperature(logits, 0.5)
    flat = apply_temperature(logits, 5.0)

    assert np.allclose(sharp.sum(axis=1), 1.0)
    assert np.allclose(flat.sum(axis=1), 1.0)
    assert flat[0].max() < sharp[0].max()


def test_temperature_is_clipped_to_bounds() -> None:
    logits = np.array([[50.0, 0.0], [0.0, 50.0]] * 10)
    labels = np.array([1, 0] * 10)  # always wrong, wants a huge temperature

    assert fit_temperature(logits, labels) <= 20.0
