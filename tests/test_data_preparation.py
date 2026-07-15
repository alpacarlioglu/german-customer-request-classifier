import pandas as pd
import pytest

from scripts.prepare_external_data import (
    assign_splits,
    cap_categories,
    deduplicate,
    prepare_manual,
    prepare_massive_stress,
    prepare_minds14,
    prepare_support_tickets,
)


def test_support_tickets_use_only_defensible_german_mappings() -> None:
    frame = pd.DataFrame(
        [
            {
                "subject": "Falsche Rechnung",
                "body": "Der Betrag wurde zweimal berechnet.",
                "type": "Request",
                "queue": "Billing and Payments",
                "language": "de",
            },
            {
                "subject": "Server nicht erreichbar",
                "body": "Seit heute gibt es einen Fehler.",
                "type": "Incident",
                "queue": "Technical Support",
                "language": "de",
            },
            {
                "subject": "Product question",
                "body": "Please help.",
                "type": "Problem",
                "queue": "Product Support",
                "language": "en",
            },
            {
                "subject": "Allgemeine Produktfrage",
                "body": "Welche Farbe ist verfügbar?",
                "type": "Problem",
                "queue": "Product Support",
                "language": "de",
            },
        ]
    )

    prepared, _ = prepare_support_tickets(frame)

    assert prepared["category"].tolist() == ["Rechnung", "Technischer Support"]
    assert set(prepared["source_label"]) == {"Billing and Payments", "Technical Support"}


def test_minds14_maps_only_address_intent_and_masks_pii() -> None:
    frame = pd.DataFrame(
        {
            "transcription": [
                "Ändern Sie die Adresse für test@example.com",
                "Wie hoch ist mein Kontostand?",
            ],
            "intent_class": [1, 7],
        }
    )

    prepared = prepare_minds14(frame)

    assert len(prepared) == 1
    assert prepared.iloc[0]["category"] == "Adressänderung"
    assert "<EMAIL>" in prepared.iloc[0]["message"]


def test_manual_data_includes_only_approved_rows(tmp_path) -> None:
    path = tmp_path / "manual.csv"
    pd.DataFrame(
        [
            {
                "message": "Bitte Vertrag kündigen",
                "category": "Kündigung",
                "review_status": "approved",
                "split": "test",
            },
            {
                "message": "Vielleicht Rechnung",
                "category": "Rechnung",
                "review_status": "pending",
                "split": "",
            },
        ]
    ).to_csv(path, index=False)

    prepared = prepare_manual(path)

    assert prepared["message"].tolist() == ["Bitte Vertrag kündigen"]
    assert prepared["split"].tolist() == ["test"]


def test_manual_data_rejects_unknown_category(tmp_path) -> None:
    path = tmp_path / "manual.csv"
    pd.DataFrame(
        [{"message": "Hallo", "category": "Andere", "review_status": "approved"}]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="unsupported categories"):
        prepare_manual(path)


def test_deduplication_removes_conflicting_labels() -> None:
    frame = pd.DataFrame(
        {
            "message": ["Gleicher Text!", "gleicher text", "Einzigartig"],
            "category": ["Rechnung", "Kündigung", "Rechnung"],
            "source": ["a", "b", "a"],
        }
    )

    deduplicated, stats = deduplicate(frame)

    assert deduplicated["message"].tolist() == ["Einzigartig"]
    assert stats["conflicting_rows_removed"] == 2


def test_category_cap_never_discards_manually_reviewed_rows() -> None:
    frame = pd.DataFrame(
        {
            "message": ["reviewed one", "reviewed two", "public one", "public two"],
            "category": ["Rechnung"] * 4,
            "source": ["manual_reviewed", "manual_reviewed", "public", "public"],
        }
    )

    capped = cap_categories(frame, target=2, seed=42)

    assert capped["message"].tolist() == ["reviewed one", "reviewed two"]


def test_split_assignment_is_deterministic_and_keeps_manual_test_row() -> None:
    rows = []
    for category in ("Rechnung", "Kündigung"):
        for index in range(10):
            rows.append(
                {
                    "message": f"{category} {index}",
                    "category": category,
                    "source": "fixture",
                    "split": "test" if index == 0 else "",
                }
            )
    frame = pd.DataFrame(rows)

    first = assign_splits(frame, seed=42)
    second = assign_splits(frame, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["split"]) == {"train", "validation", "test"}
    assert first.loc[first["message"].str.endswith(" 0"), "split"].eq("test").all()
    assert not first.duplicated("message").any()


def test_massive_is_deduplicated_and_kept_without_project_category() -> None:
    frame = pd.DataFrame(
        {"text": ["Wie spät ist es?", "wie spät ist es", "Spiele Musik"], "label": [1, 1, 2]}
    )

    prepared = prepare_massive_stress([frame], size=10, seed=42)

    assert len(prepared) == 2
    assert "category" not in prepared.columns
