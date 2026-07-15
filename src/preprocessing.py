from __future__ import annotations

import re


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
IBAN_RE = re.compile(r"\b[A-Z]{2}\s?\d{2}(?:\s?[A-Z0-9]){11,30}\b", re.IGNORECASE)
PHONE_RE = re.compile(
    r"(?<!\w)(?:(?:\+|00)49[\s./-]*|0)\d(?:[\s./-]*\d){6,12}(?!\w)"
)
ACCOUNT_RE = re.compile(r"(?<!\w)\d{8,20}(?!\w)")
WHITESPACE_RE = re.compile(r"\s+")


def mask_pii(text: str) -> str:
    """Replace common contact and account identifiers with stable placeholders."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    masked = EMAIL_RE.sub("<EMAIL>", text)
    masked = IBAN_RE.sub("<ACCOUNT>", masked)
    masked = PHONE_RE.sub("<PHONE>", masked)
    masked = ACCOUNT_RE.sub("<ACCOUNT>", masked)
    return masked


def normalize_text(text: str) -> str:
    """Trim text and collapse repeated whitespace without changing German characters."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return WHITESPACE_RE.sub(" ", text).strip()


def preprocess_text(text: str) -> str:
    return normalize_text(mask_pii(text))

