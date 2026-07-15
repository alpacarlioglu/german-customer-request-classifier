import pytest

from src.preprocessing import mask_pii, normalize_text, preprocess_text


def test_masks_email_phone_iban_and_account_number() -> None:
    message = (
        "Kontakt: max@example.de, +49 170 1234567, "
        "DE89 3704 0044 0532 0130 00 oder Konto 1234567890."
    )

    masked = mask_pii(message)

    assert "max@example.de" not in masked
    assert "+49 170 1234567" not in masked
    assert "DE89" not in masked
    assert "1234567890" not in masked
    assert "<EMAIL>" in masked
    assert "<PHONE>" in masked
    assert masked.count("<ACCOUNT>") == 2


def test_normalizes_whitespace_without_removing_german_characters() -> None:
    assert normalize_text("  Überhöhte\n\tGebühr  ") == "Überhöhte Gebühr"


def test_preprocessing_rejects_non_string_values() -> None:
    with pytest.raises(TypeError):
        preprocess_text(None)  # type: ignore[arg-type]

