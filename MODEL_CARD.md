# Model Card: German Customer Request Classification

## Purpose

This project demonstrates intent classification for short German customer-service messages. It compares a TF-IDF and Logistic Regression baseline with an optional compact multilingual Transformer. The predictions are intended to support routing and prioritization, not to make decisions that materially affect customers.

## Intended use

- Explore common customer-request categories.
- Route messages for human review.
- Compare a transparent classical baseline with a Transformer.
- Collect confirmation and correction feedback in PostgreSQL.

Do not use the models to deny service, determine eligibility, assess creditworthiness, or replace human review in consequential workflows.

## Training data

The included `data/sample_requests.csv` is a small synthetic demonstration dataset. It is suitable for verifying the software, but it is not large or representative enough for production performance claims. Replace it with appropriately licensed, reviewed, and de-identified domain data.

The optional external-data pipeline can combine narrowly selected records from multiple sources. The main candidates are synthetic German customer-support tickets for billing and technical support, German MInDS-14 address-intent transcripts, and manually reviewed German examples. Bilingual augmented tickets are excluded by default because they overlap the primary ticket family. MASSIVE German voice-assistant utterances are reserved for out-of-scope stress testing and are not normal training data.

Prepared records retain `source`, `source_label`, `license`, `review_status`, and fixed `split` fields. Public labels are mapped only when their meaning clearly matches a project intent. See [DATA_SOURCES.md](DATA_SOURCES.md) for source-specific licenses, attribution, mapping policy, and redistribution constraints.

## Metrics

Run `python -m scripts.evaluate` to produce accuracy, macro F1, per-class metrics, and confusion matrices on the deterministic test split. Metrics from the sample dataset should not be interpreted as real-world quality estimates.

## Confidence and manual review

The application marks predictions below `REVIEW_THRESHOLD` for manual review. Logistic Regression probabilities and Transformer softmax scores may be poorly calibrated. A score of 0.90 does not guarantee that 90% of comparable predictions are correct.

## Explainability

- The baseline highlights positive TF-IDF coefficient contributions.
- The Transformer removes one word at a time and measures the drop in predicted probability.

These explanations show model sensitivity, not causal reasoning. Transformer explanations are capped at 30 words for responsiveness.

## Limitations and bias

- The synthetic data covers only four categories and limited writing styles.
- Merged sources differ in domain and format. The model may learn source-specific wording, message length, or transcription style instead of customer intent.
- Public datasets do not provide clean coverage of all four project categories; cancellation and address-change data require substantial manual review.
- Some candidate datasets are synthetic or augmented and may contain unrealistic patterns or near duplicates.
- CC BY-NC 4.0 sources restrict commercial use; the generated dataset must not be assumed to share the repository code's license.
- Dialects, spelling mistakes, code-switching, long messages, and unseen intents may reduce accuracy.
- Category imbalance can produce uneven quality despite macro F1 reporting.
- PII masking uses regular expressions and cannot guarantee removal of every identifier.
- Masking may remove information that is useful for classification.
- User feedback may be sparse, inconsistent, or biased toward difficult cases.
- Exact deduplication cannot guarantee removal of translated or lightly paraphrased leakage.

## Privacy

Common email addresses, German phone numbers, IBAN-like values, and long account numbers are masked before classification and database storage. Raw messages should not be logged. The masking rules must be reviewed against actual data before operational use.

## Recommended monitoring

Review feedback accuracy, manual-review rate, confusion pairs, and performance by category. Retrain only after corrected labels have been validated and the evaluation split has been kept separate.

For merged training data, also review source-by-category counts and performance on a separately authored, manually reviewed German test set. Do not report real-world performance from the included sample or from unreviewed synthetic/augmented records.
