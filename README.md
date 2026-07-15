# German Customer Request Classification

A focused NLP portfolio project that classifies German customer messages, stores masked predictions and feedback in PostgreSQL, and presents history and analytics in Streamlit.

## Features

- TF-IDF and Logistic Regression baseline
- Optional multilingual Transformer comparison
- Accuracy, macro F1, classification report, and confusion matrices
- Configurable confidence threshold for manual review
- Influential-term highlighting
- PII masking before prediction and storage
- Single-message and CSV batch classification
- PostgreSQL prediction history and feedback
- History filtering and CSV export
- Feedback-based analytics
- Pytest suite and one GitHub Actions workflow

## Project structure

```text
streamlit_app.py          Streamlit entry page
pages/                    Classify, History, and Analytics pages
src/preprocessing.py      PII masking and text normalization
src/predictor.py          Shared model loading, prediction, and explanation API
src/database.py           SQLAlchemy table and database operations
src/analytics.py          Feedback aggregations
scripts/train.py          Baseline and Transformer training
scripts/evaluate.py       Shared test-set evaluation
scripts/prepare_external_data.py  External-data preparation and provenance
```

## 1. Create the environment

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

For Transformer training and inference, install the optional dependencies:

```bash
pip install -r requirements-transformer.txt
```

## 2. Configure PostgreSQL

Create an empty database using your local PostgreSQL installation or an existing PostgreSQL server:

```bash
createdb customer_requests
```

Copy the example configuration and edit the credentials if necessary:

```bash
cp .env.example .env
```

```dotenv
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/customer_requests
REVIEW_THRESHOLD=0.65
MODEL_ARTIFACT_DIR=model_artifacts
```

The application creates its single `predictions` table on startup. Alembic is intentionally omitted for this first fixed schema.

## 3. Prepare external data (optional)

The external-data workflow creates a provenance-aware dataset from narrowly mapped public sources plus manually reviewed examples:

```bash
python -m scripts.prepare_external_data
```

To request a different per-category target while keeping the run reproducible:

```bash
python -m scripts.prepare_external_data --target-per-category 400 --seed 42
```

The generated training data is written to `data/processed/customer_requests.csv`; out-of-scope German MASSIVE examples are written separately for stress testing. Raw downloads and generated datasets are ignored by Git to avoid accidental redistribution. See [DATA_SOURCES.md](DATA_SOURCES.md) for mappings, licenses, provenance fields, review requirements, and limitations.

The preparation command also writes `review_candidates.csv` and `data_report.json`. The report records category/source/split counts, source checksums, deduplication results, and explicit shortfalls. Add only human-approved rows to `data/manual/reviewed_requests.csv`; pending rows are never used for training.

Combining datasets is intentionally conservative: ambiguous labels are dropped, the overlapping bilingual ticket dataset is not included by default, and voice-assistant data is never used as normal training data. Add reviewed German examples in `data/manual/reviewed_requests.csv`, particularly for cancellation and address-change requests.

## 4. Train models

Train the baseline on the included synthetic sample:

```bash
python -m scripts.train --model baseline
```

Train the optional Transformer:

```bash
python -m scripts.train --model transformer --epochs 2
```

Train both models on your own CSV:

```bash
python -m scripts.train --data data/requests.csv --model both
```

The CSV must contain `message` and `category` columns, with at least seven examples in each category. Use substantially more data for any meaningful evaluation.

To train on the prepared external dataset:

```bash
python -m scripts.train --data data/processed/customer_requests.csv --model baseline
```

## 5. Evaluate

Evaluate every available model on the same deterministic test split:

```bash
python -m scripts.evaluate
```

The command writes `reports/metrics.json` and one confusion-matrix image per model.
It also reports accuracy/coverage trade-offs at several confidence thresholds, including the application's default `0.65` manual-review threshold.

## 6. Run Streamlit

```bash
streamlit run streamlit_app.py
```

The application provides three pages:

1. **Classify** predicts, highlights influential words, accepts feedback, and processes CSV files of up to 500 rows.
2. **History** filters stored predictions and exports the current results.
3. **Analytics** shows category volume, confidence, confusion pairs, and feedback accuracy.

## Run tests

```bash
pytest -q
```

Tests use a temporary SQLite database locally for speed and isolation. GitHub Actions supplies PostgreSQL 16 through a service container so the database layer is also tested against the production dialect.

## Data and privacy notes

Only the masked message is stored. The application does not retain the original raw text. Regular-expression masking reduces accidental PII storage but is not a privacy guarantee; review the rules against real data. External records retain provenance during preparation, and licensing must be checked before generated data is published. See [DATA_SOURCES.md](DATA_SOURCES.md) for data governance and [MODEL_CARD.md](MODEL_CARD.md) for intended use and limitations.
