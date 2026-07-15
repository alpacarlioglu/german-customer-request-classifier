from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import preprocess_text


DEFAULT_TRANSFORMER = "distilbert/distilbert-base-multilingual-cased"


def load_dataset(path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {text_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")

    # Keep provenance and prepared-split columns when they are present. Training
    # still uses only the configured text and label columns.
    frame = frame.dropna(subset=[text_column, label_column]).copy()
    frame[text_column] = frame[text_column].astype(str).map(preprocess_text)
    frame[label_column] = frame[label_column].astype(str).str.strip()
    frame = frame[(frame[text_column] != "") & (frame[label_column] != "")]
    if frame[label_column].nunique() < 2:
        raise ValueError("At least two categories are required.")
    prepared_splits = (
        "split" in frame.columns
        and {"train", "validation", "test"}.issubset(
            set(frame["split"].astype(str).str.strip().str.lower())
        )
    )
    if not prepared_splits and frame[label_column].value_counts().min() < 7:
        raise ValueError("Each category needs at least seven examples for stratified splitting.")
    return frame.reset_index(drop=True)


def split_dataset(
    frame: pd.DataFrame, label_column: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" in frame.columns:
        split_names = frame["split"].astype(str).str.strip().str.lower()
        required = {"train", "validation", "test"}
        if required.issubset(set(split_names)):
            unexpected = set(split_names) - required
            if unexpected:
                raise ValueError(
                    f"Prepared dataset contains unsupported split values: "
                    f"{', '.join(sorted(unexpected))}"
                )
            train = frame[split_names == "train"]
            validation = frame[split_names == "validation"]
            test = frame[split_names == "test"]
            return (
                train.reset_index(drop=True),
                validation.reset_index(drop=True),
                test.reset_index(drop=True),
            )

    train, remainder = train_test_split(
        frame,
        test_size=0.30,
        random_state=seed,
        stratify=frame[label_column],
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        stratify=remainder[label_column],
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def dataset_metadata(frame: pd.DataFrame, label_column: str) -> dict[str, object]:
    """Return compact provenance information to save beside a trained model."""
    metadata: dict[str, object] = {
        "row_count": int(len(frame)),
        "category_counts": {
            str(category): int(count)
            for category, count in frame[label_column].value_counts().sort_index().items()
        },
    }
    if "source" in frame.columns:
        sources = frame["source"].fillna("unknown").astype(str)
        metadata["source_counts"] = {
            str(source): int(count)
            for source, count in sources.value_counts().sort_index().items()
        }
    else:
        metadata["source_counts"] = {}
    return metadata


def basic_metrics(labels: list[str], predictions: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def train_baseline(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir: Path,
    data_metadata: dict[str, object] | None = None,
) -> None:
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=42),
            ),
        ]
    )
    pipeline.fit(train[text_column], train[label_column])
    predictions = pipeline.predict(validation[text_column]).tolist()

    model_dir = output_dir / "baseline"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "model.joblib")
    metadata = {
        "model_name": "tfidf_logreg",
        "labels": pipeline.classes_.tolist(),
        "validation_metrics": basic_metrics(validation[label_column].tolist(), predictions),
        "dataset": data_metadata or dataset_metadata(
            pd.concat([train, validation], ignore_index=True), label_column
        ),
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved baseline model to {model_dir}")


def train_transformer(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    text_column: str,
    label_column: str,
    output_dir: Path,
    checkpoint: str,
    epochs: int,
    batch_size: int,
    seed: int,
    data_metadata: dict[str, object] | None = None,
) -> None:
    try:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-transformer.txt before training the Transformer."
        ) from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    labels = sorted(train[label_column].unique().tolist())
    label_to_id = {label: index for index, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    class TextDataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.texts = frame[text_column].tolist()
            self.labels = [label_to_id[label] for label in frame[label_column]]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int):
            encoded = tokenizer(
                self.texts[index],
                truncation=True,
                max_length=256,
                padding="max_length",
                return_tensors="pt",
            )
            item = {key: value.squeeze(0) for key, value in encoded.items()}
            item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
            return item

    train_loader = DataLoader(TextDataset(train), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(TextDataset(validation), batch_size=batch_size)
    id_to_label = {index: label for label, index in label_to_id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f"Finished Transformer epoch {epoch + 1}/{epochs}")

    model.eval()
    predicted_ids: list[int] = []
    with torch.no_grad():
        for batch in validation_loader:
            expected = batch.pop("labels")
            batch = {key: value.to(device) for key, value in batch.items()}
            predicted_ids.extend(model(**batch).logits.argmax(dim=-1).cpu().tolist())
    predictions = [id_to_label[index] for index in predicted_ids]

    model_dir = output_dir / "transformer"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    metadata = {
        "model_name": "transformer",
        "checkpoint": checkpoint,
        "labels": labels,
        "validation_metrics": basic_metrics(validation[label_column].tolist(), predictions),
        "dataset": data_metadata or dataset_metadata(
            pd.concat([train, validation], ignore_index=True), label_column
        ),
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved Transformer model to {model_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train customer-request classifiers.")
    parser.add_argument("--data", type=Path, default=Path("data/sample_requests.csv"))
    parser.add_argument("--model", choices=["baseline", "transformer", "both"], default="baseline")
    parser.add_argument("--text-column", default="message")
    parser.add_argument("--label-column", default="category")
    parser.add_argument("--output-dir", type=Path, default=Path("model_artifacts"))
    parser.add_argument("--transformer-checkpoint", default=DEFAULT_TRANSFORMER)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_dataset(args.data, args.text_column, args.label_column)
    train, validation, _ = split_dataset(frame, args.label_column, args.seed)
    data_metadata = dataset_metadata(frame, args.label_column)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model in {"baseline", "both"}:
        train_baseline(
            train,
            validation,
            args.text_column,
            args.label_column,
            args.output_dir,
            data_metadata,
        )
    if args.model in {"transformer", "both"}:
        train_transformer(
            train,
            validation,
            args.text_column,
            args.label_column,
            args.output_dir,
            args.transformer_checkpoint,
            args.epochs,
            args.batch_size,
            args.seed,
            data_metadata,
        )


if __name__ == "__main__":
    main()
