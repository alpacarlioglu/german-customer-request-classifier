from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.preprocessing import preprocess_text


CATEGORIES = ("Rechnung", "Technischer Support", "Adressänderung", "Kündigung")
OUTPUT_COLUMNS = (
    "message",
    "category",
    "source",
    "source_label",
    "license",
    "review_status",
    "split",
)

SOURCES = {
    "tobi_tickets": {
        "filename": "tobi_customer_support_tickets.csv",
        "url": (
            "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/"
            "ddf1c81a5475992c4fa6752bf1e8b4e31f07bbeb/"
            "aa_dataset-tickets-multi-lang-5-2-50-version.csv"
        ),
        "license": "CC BY-NC 4.0",
    },
    "minds14_de": {
        "filename": "minds14_de_train.parquet",
        "url": (
            "https://huggingface.co/datasets/PolyAI/minds14/resolve/"
            "refs%2Fconvert%2Fparquet/de-DE/train/0000.parquet"
        ),
        "license": "CC BY 4.0",
    },
    "massive_de_train": {
        "filename": "massive_de_train.parquet",
        "url": (
            "https://huggingface.co/datasets/mteb/amazon_massive_intent/resolve/"
            "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740/de/train-00000-of-00001.parquet"
        ),
        "license": "CC BY 4.0",
    },
    "massive_de_validation": {
        "filename": "massive_de_validation.parquet",
        "url": (
            "https://huggingface.co/datasets/mteb/amazon_massive_intent/resolve/"
            "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740/de/validation-00000-of-00001.parquet"
        ),
        "license": "CC BY 4.0",
    },
    "massive_de_test": {
        "filename": "massive_de_test.parquet",
        "url": (
            "https://huggingface.co/datasets/mteb/amazon_massive_intent/resolve/"
            "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740/de/test-00000-of-00001.parquet"
        ),
        "license": "CC BY 4.0",
    },
}

BILLING_RE = re.compile(
    r"\b(?:rechnung\w*|abrechnung\w*|gebühr\w*|zahlung\w*|betrag\w*|"
    r"kosten\w*|bezahlt\w*|gutschrift\w*|lastschrift\w*)\b",
    re.IGNORECASE,
)
CANCELLATION_RE = re.compile(
    r"\b(?:kündig\w*|vertragsende\w*|abo\w*\s+(?:beenden|stornieren)|"
    r"vertrag\w*\s+(?:beenden|auflösen))\b",
    re.IGNORECASE,
)
ADDRESS_RE = re.compile(
    r"\b(?:adressänder\w*|anschrift\w*\s+(?:ändern|aktualisieren)|"
    r"adresse\w*\s+(?:ändern|aktualisieren)|umgezogen|umzug)\b",
    re.IGNORECASE,
)


def download_file(url: str, destination: Path, offline: bool = False) -> Path:
    if destination.exists():
        return destination
    if offline:
        raise FileNotFoundError(f"Missing cached source in offline mode: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name} ...")
    urllib.request.urlretrieve(url, partial)
    partial.replace(destination)
    return destination


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_message(*parts: object) -> str:
    text_parts = [str(part).strip() for part in parts if pd.notna(part) and str(part).strip()]
    return preprocess_text(" ".join(text_parts))[:2000]


def normalized_duplicate_key(message: str) -> str:
    return re.sub(r"[\W_]+", " ", message.casefold(), flags=re.UNICODE).strip()


def _prepared_rows(
    frame: pd.DataFrame,
    category: str,
    source: str,
    source_label: pd.Series,
    license_name: str,
    review_status: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "message": frame["message"],
            "category": category,
            "source": source,
            "source_label": source_label.astype(str),
            "license": license_name,
            "review_status": review_status,
            "split": "",
        }
    )


def prepare_support_tickets(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"subject", "body", "type", "queue", "language"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Support-ticket source is missing: {', '.join(sorted(missing))}")

    german = frame[frame["language"].astype(str).str.lower().eq("de")].copy()
    german["message"] = [
        clean_message(subject, body) for subject, body in zip(german["subject"], german["body"])
    ]
    german = german[german["message"].ne("")]

    billing = german[
        german["queue"].eq("Billing and Payments")
        & german["message"].str.contains(BILLING_RE, na=False)
        & ~german["message"].str.contains(CANCELLATION_RE, na=False)
        & ~german["message"].str.contains(ADDRESS_RE, na=False)
    ].copy()
    technical_queues = {
        "Technical Support",
        "IT Support",
        "Service Outages and Maintenance",
    }
    technical = german[
        german["queue"].isin(technical_queues)
        & german["type"].isin({"Incident", "Problem"})
        & ~german["message"].str.contains(CANCELLATION_RE, na=False)
        & ~german["message"].str.contains(ADDRESS_RE, na=False)
    ].copy()

    prepared = pd.concat(
        [
            _prepared_rows(
                billing,
                "Rechnung",
                "tobi_customer_support_tickets",
                billing["queue"],
                "CC BY-NC 4.0",
                "automatic_conservative_mapping",
            ),
            _prepared_rows(
                technical,
                "Technischer Support",
                "tobi_customer_support_tickets",
                technical["queue"],
                "CC BY-NC 4.0",
                "automatic_conservative_mapping",
            ),
        ],
        ignore_index=True,
    )

    candidate_parts: list[pd.DataFrame] = []
    for pattern, suggested in ((CANCELLATION_RE, "Kündigung"), (ADDRESS_RE, "Adressänderung")):
        candidates = german[german["message"].str.contains(pattern, na=False)].copy()
        if not candidates.empty:
            candidate_parts.append(
                pd.DataFrame(
                    {
                        "message": candidates["message"],
                        "suggested_category": suggested,
                        "source": "tobi_customer_support_tickets",
                        "source_label": candidates["queue"].astype(str),
                        "review_status": "pending",
                        "notes": "Review meaning before moving to reviewed_requests.csv",
                    }
                )
            )
    review_candidates = (
        pd.concat(candidate_parts, ignore_index=True)
        if candidate_parts
        else pd.DataFrame(
            columns=[
                "message",
                "suggested_category",
                "source",
                "source_label",
                "review_status",
                "notes",
            ]
        )
    )
    return prepared, review_candidates


def prepare_minds14(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"transcription", "intent_class"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MInDS-14 source is missing: {', '.join(sorted(missing))}")

    address = frame[frame["intent_class"].astype(str).eq("1")].copy()
    address["message"] = address["transcription"].map(clean_message)
    address = address[address["message"].ne("")]
    return _prepared_rows(
        address,
        "Adressänderung",
        "minds14_de",
        pd.Series("address", index=address.index),
        "CC BY 4.0",
        "automatic_exact_label_mapping",
    )


def prepare_seed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"message", "category"}
    if not required.issubset(frame.columns):
        raise ValueError("Seed data must contain message and category columns.")
    frame = frame[frame["category"].isin(CATEGORIES)].copy()
    frame["message"] = frame["message"].map(clean_message)
    frame["source"] = "project_synthetic_seed"
    frame["source_label"] = frame["category"]
    frame["license"] = "project"
    frame["review_status"] = "synthetic_demo"
    frame["split"] = ""
    return frame[list(OUTPUT_COLUMNS)]


def prepare_manual(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    frame = pd.read_csv(path).fillna("")
    required = {"message", "category", "review_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manual data is missing: {', '.join(sorted(missing))}")

    invalid = sorted(set(frame.loc[frame["category"].ne(""), "category"]) - set(CATEGORIES))
    if invalid:
        raise ValueError(f"Manual data has unsupported categories: {', '.join(invalid)}")
    approved = frame[frame["review_status"].str.lower().eq("approved")].copy()
    approved["message"] = approved["message"].map(clean_message)
    approved = approved[approved["message"].ne("")]
    approved["source"] = "manual_reviewed"
    approved["source_label"] = approved["category"]
    approved["license"] = "project"
    if "split" not in approved:
        approved["split"] = ""
    approved["split"] = approved["split"].astype(str).str.lower()
    invalid_splits = sorted(set(approved["split"]) - {"", "train", "validation", "test"})
    if invalid_splits:
        raise ValueError(f"Manual data has unsupported splits: {', '.join(invalid_splits)}")
    return approved[list(OUTPUT_COLUMNS)]


def deduplicate(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    working = frame.copy()
    working["duplicate_key"] = working["message"].map(normalized_duplicate_key)
    working = working[working["duplicate_key"].ne("")]
    conflicting_keys = set(
        working.groupby("duplicate_key")["category"].nunique().loc[lambda values: values > 1].index
    )
    conflict_rows = int(working["duplicate_key"].isin(conflicting_keys).sum())
    working = working[~working["duplicate_key"].isin(conflicting_keys)].copy()
    priority = {
        "manual_reviewed": 0,
        "project_synthetic_seed": 1,
        "minds14_de": 2,
        "tobi_customer_support_tickets": 3,
    }
    working["source_priority"] = working["source"].map(priority).fillna(99)
    before = len(working)
    working = working.sort_values(["source_priority", "source"]).drop_duplicates(
        "duplicate_key", keep="first"
    )
    duplicate_rows = before - len(working)
    return (
        working.drop(columns=["duplicate_key", "source_priority"]).reset_index(drop=True),
        {"duplicate_rows_removed": int(duplicate_rows), "conflicting_rows_removed": conflict_rows},
    )


def cap_categories(frame: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    parts = []
    for offset, category in enumerate(CATEGORIES):
        category_rows = frame[frame["category"].eq(category)]
        reviewed = category_rows[category_rows["source"].eq("manual_reviewed")]
        other = category_rows[~category_rows["source"].eq("manual_reviewed")]
        available_slots = max(0, target - len(reviewed))
        if len(other) > available_slots:
            other = other.sample(n=available_slots, random_state=seed + offset)
        parts.append(pd.concat([reviewed, other], ignore_index=True))
    return pd.concat(parts, ignore_index=True)


def assign_splits(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = frame.copy()
    result["split"] = result["split"].fillna("").astype(str).str.lower()
    for offset, category in enumerate(CATEGORIES):
        indices = result.index[result["category"].eq(category)].tolist()
        if not indices:
            continue
        category_size = len(indices)
        validation_target = max(1, round(category_size * 0.15)) if category_size >= 3 else 0
        test_target = max(1, round(category_size * 0.15)) if category_size >= 3 else 0
        unassigned = result.loc[indices]
        unassigned = unassigned[unassigned["split"].eq("")].sample(
            frac=1, random_state=seed + offset
        )
        remaining = unassigned.index.tolist()
        for split_name, target in (("test", test_target), ("validation", validation_target)):
            already = int((result.loc[indices, "split"] == split_name).sum())
            take = min(max(0, target - already), len(remaining))
            result.loc[remaining[:take], "split"] = split_name
            remaining = remaining[take:]
        result.loc[remaining, "split"] = "train"
    return result.sort_values(["split", "category", "source", "message"]).reset_index(drop=True)


def prepare_massive_stress(frames: list[pd.DataFrame], size: int, seed: int) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    required = {"text", "label"}
    if not required.issubset(combined.columns):
        raise ValueError("MASSIVE source must contain text and label columns.")
    output = pd.DataFrame(
        {
            "message": combined["text"].map(clean_message),
            "source": "massive_de",
            "source_label": combined["label"].astype(str),
            "license": "CC BY 4.0",
        }
    )
    output["duplicate_key"] = output["message"].map(normalized_duplicate_key)
    output = output[output["duplicate_key"].ne("")].drop_duplicates("duplicate_key")
    output = output.drop(columns="duplicate_key")
    if len(output) > size:
        output = output.sample(n=size, random_state=seed)
    return output.sort_values("message").reset_index(drop=True)


def counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(name): int(value)
        for name, value in frame[column].value_counts().sort_index().items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare provenance-aware German intent data.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--manual-file", type=Path, default=Path("data/manual/reviewed_requests.csv"))
    parser.add_argument("--seed-file", type=Path, default=Path("data/sample_requests.csv"))
    parser.add_argument("--target-per-category", type=int, default=400)
    parser.add_argument("--stress-test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_per_category < 1 or args.stress_test_size < 1:
        raise ValueError("Target sizes must be positive integers.")
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for name, source in SOURCES.items():
        paths[name] = download_file(
            str(source["url"]), args.raw_dir / str(source["filename"]), args.offline
        )

    tickets = pd.read_csv(paths["tobi_tickets"])
    ticket_rows, review_candidates = prepare_support_tickets(tickets)
    minds = pd.read_parquet(
        paths["minds14_de"], columns=["transcription", "intent_class"]
    )
    minds_rows = prepare_minds14(minds)
    seed_rows = prepare_seed(args.seed_file)
    manual_rows = prepare_manual(args.manual_file)

    combined = pd.concat([manual_rows, seed_rows, minds_rows, ticket_rows], ignore_index=True)
    combined, dedupe_stats = deduplicate(combined)
    combined = cap_categories(combined, args.target_per_category, args.seed)
    combined = assign_splits(combined, args.seed)
    combined.to_csv(args.processed_dir / "customer_requests.csv", index=False)
    review_candidates.to_csv(args.processed_dir / "review_candidates.csv", index=False)

    massive_frames = [
        pd.read_parquet(paths[name], columns=["text", "label"])
        for name in ("massive_de_train", "massive_de_validation", "massive_de_test")
    ]
    stress = prepare_massive_stress(massive_frames, args.stress_test_size, args.seed)
    stress.to_csv(args.processed_dir / "stress_test_messages.csv", index=False)

    category_counts = counts(combined, "category")
    shortfalls = {
        category: args.target_per_category - category_counts.get(category, 0)
        for category in CATEGORIES
        if category_counts.get(category, 0) < args.target_per_category
    }
    warnings = []
    if shortfalls:
        warnings.append(
            "Category targets are incomplete; add approved manual examples before quality claims."
        )
    if category_counts.get("Kündigung", 0) <= 10:
        warnings.append("Kündigung currently relies only on the small synthetic demo set.")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "target_per_category": args.target_per_category,
        "row_count": int(len(combined)),
        "category_counts": category_counts,
        "source_counts": counts(combined, "source"),
        "split_counts": counts(combined, "split"),
        "shortfalls": shortfalls,
        "review_candidate_count": int(len(review_candidates)),
        "stress_test_count": int(len(stress)),
        "deduplication": dedupe_stats,
        "source_files": {
            name: {
                "filename": path.name,
                "sha256": file_sha256(path),
                "url": SOURCES[name]["url"],
                "license": SOURCES[name]["license"],
            }
            for name, path in paths.items()
        },
        "warnings": warnings,
    }
    (args.processed_dir / "data_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Prepared {len(combined)} training/evaluation rows.")
    print(f"Category counts: {category_counts}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if args.strict and shortfalls:
        raise RuntimeError(f"Category targets were not met: {shortfalls}")


if __name__ == "__main__":
    main()
