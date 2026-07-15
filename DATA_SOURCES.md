# Data Sources and Provenance

The external-data pipeline combines only records whose labels can be mapped defensibly to this project's four intents. It keeps provenance with every row so that source composition, licensing, and source-specific bias remain visible.

External data is optional. The small synthetic `data/sample_requests.csv` remains available for exercising the application without downloading third-party data.

## Output schema

The prepared training file uses these columns:

| Column | Meaning |
|---|---|
| `message` | Cleaned, PII-masked German message used by the model |
| `category` | Project label: `Rechnung`, `Technischer Support`, `Adressänderung`, or `Kündigung` |
| `source` | Stable identifier for the source dataset or manual collection |
| `source_label` | Original label before mapping, when available |
| `license` | License associated with the source record |
| `review_status` | Whether the mapping/text has been manually reviewed |
| `split` | Fixed `train`, `validation`, or `test` assignment |

Training consumes `message` and `category`; the remaining fields exist for auditability and leakage checks. A source-by-category summary should be reviewed before training because a category tied to one source can cause the model to learn dataset style instead of intent.

## Sources

### German Customer Support Tickets

- Repository: [Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
- License: CC BY-NC 4.0
- Intended use here: candidate German billing and technical-support records
- Important caveat: source fields and label values must be validated against the downloaded version. The pipeline must not assume that every queue name or record is suitable solely from the dataset title.

Only labels with a clear semantic match are retained. Ambiguous queues and records are rejected instead of being forced into a project category. This dataset is synthetic, so results derived from it must not be presented as performance on real customer conversations.

### MInDS-14 German

- Repository: [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14/tree/main/de-DE)
- License: CC BY 4.0
- Intended use here: German examples with the explicit `address` intent as candidates for `Adressänderung`
- Important caveat: MInDS-14 contains spoken banking queries. Its wording and domain differ from written customer-support tickets.

The pipeline uses German metadata/transcripts only; audio is not needed for text classification. Candidate mappings still require review because `address` utterances may not always express a request to change an address.

### Bilingual Ticket Classification

- Repository: [ale-dp/bilingual-ticket-classification](https://huggingface.co/datasets/ale-dp/bilingual-ticket-classification)
- License: CC BY-NC 4.0
- Intended use here: fallback candidate source for German billing or technical-support messages
- Important caveat: this dataset overlaps the Tobi-Bueck customer-support-ticket family and includes augmented data.

It is excluded from the default merged training set to reduce duplicate and near-duplicate leakage. If it is enabled as a fallback, provenance must remain intact and overlap must be checked before splitting.

### MASSIVE German derivative

- Source project: [AmazonScience/massive](https://github.com/alexa/massive)
- German derivative used by the preparation workflow: the corresponding MTEB/Hugging Face German data artifact selected by the script
- License: CC BY 4.0
- Intended use here: out-of-scope stress testing only
- Important caveat: its voice-assistant intents do not align cleanly with this project's four customer-service categories.

MASSIVE records are not merged into normal training or evaluation. They are used to check whether unrelated German requests receive low confidence or are routed to manual review.

## Manual review data

`data/manual/reviewed_requests.csv` supplements categories that lack a trustworthy public mapping, especially `Kündigung` and `Adressänderung`. It should also include reviewed examples for `Rechnung` and `Technischer Support`, so that manual writing style is not itself a category signal.

Every manually collected row should:

- be written or reviewed by a fluent German speaker;
- contain no real personal or customer information;
- record its origin and review status;
- have permission for use and redistribution;
- avoid paraphrased variants crossing split boundaries.

Automatically mapping `cancel_order`, `cancel_transfer`, or account-closure intents to contract cancellation is not acceptable. These describe different user goals.

## Preparation

Run the reproducible preparation command from the repository root:

```bash
python -m scripts.prepare_external_data
```

The target size and random seed can be changed explicitly:

```bash
python -m scripts.prepare_external_data --target-per-category 400 --seed 42
```

Expected generated artifacts are:

```text
data/processed/customer_requests.csv
data/processed/stress_test_messages.csv
data/processed/review_candidates.csv
data/processed/data_report.json
```

The pipeline downloads or reads source data, validates the available schema, selects defensible labels, masks PII, normalizes text, removes duplicates, merges reviewed records, balances categories, and creates fixed splits. Exact normalized messages must not occur in more than one split.

Use `--offline` to rebuild from files already cached in `data/raw`. Use `--strict` in a controlled dataset-release workflow when incomplete category targets should fail the command. The normal command writes an honest report with warnings and shortfalls so that the software remains usable while manual review is still in progress.

## Repository and redistribution policy

Raw downloads and generated processed files are ignored by Git. This avoids accidentally committing large artifacts or redistributing third-party records without checking their terms. The preparation code, source references, mapping decisions, and attribution remain versioned.

Before publishing any generated dataset:

1. Recheck each upstream license and required attribution.
2. Preserve record-level provenance.
3. Confirm that the intended use is permitted, particularly for CC BY-NC 4.0 sources.
4. Confirm that manual examples may be redistributed.
5. Review the output for personal information and licensing conflicts.

The repository's software license does not replace or override the licenses of upstream datasets.

## Known limitations

- The public sources differ in domain and style: synthetic email-like tickets, spoken banking queries, and voice-assistant commands.
- A balanced label count does not guarantee balanced linguistic variety.
- Regex-based PII masking is incomplete and must not be treated as anonymization assurance.
- Exact deduplication may miss translated, augmented, or lightly paraphrased copies.
- Manually created examples can introduce author and template bias.
- A fixed manually reviewed German test set is required before making meaningful model-quality claims.
