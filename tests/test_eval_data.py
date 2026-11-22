"""Guards the hand-authored evaluation datasets in data/eval/."""
import re
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing import preprocess_text

ROOT = Path(__file__).resolve().parents[1]
TEST_SET = ROOT / "data" / "eval" / "german_test_set.csv"
OOS_SET = ROOT / "data" / "eval" / "out_of_scope.csv"
SAMPLE = ROOT / "data" / "sample_requests.csv"
CATEGORIES = {"Rechnung", "Technischer Support", "Adressänderung", "Kündigung"}


def normalize(text: str) -> str:
    text = re.sub(r"\W+", " ", str(text).casefold())
    return re.sub(r"\s+", " ", text).strip()


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(TEST_SET, encoding="utf-8")


@pytest.fixture(scope="module")
def oos_df():
    return pd.read_csv(OOS_SET, encoding="utf-8")


@pytest.fixture(scope="module")
def sample_df():
    return pd.read_csv(SAMPLE, encoding="utf-8")


def test_columns(test_df, oos_df):
    assert list(test_df.columns) == ["message", "category", "source", "review_status"]
    assert list(oos_df.columns) == ["message", "source", "review_status"]


def test_row_counts(test_df, oos_df):
    assert len(test_df) == 120
    assert (test_df["category"].value_counts() == 30).all()
    assert len(oos_df) == 60


def test_categories_exact(test_df):
    assert set(test_df["category"]) == CATEGORIES


def test_no_empty_messages(test_df, oos_df, sample_df):
    for df in (test_df, oos_df, sample_df):
        assert df["message"].notna().all()
        assert (df["message"].astype(str).str.strip() != "").all()


def test_no_duplicates_within_or_across(test_df, oos_df, sample_df):
    groups = {"test": test_df, "oos": oos_df, "sample": sample_df}
    seen = {}
    for name, df in groups.items():
        for message in df["message"]:
            key = normalize(message)
            assert key not in seen, f"duplicate: {message!r} ({name} vs {seen[key]})"
            seen[key] = name


def test_preprocess_non_empty(test_df):
    for message in test_df["message"]:
        result = preprocess_text(message)
        assert isinstance(result, str) and result.strip(), message
