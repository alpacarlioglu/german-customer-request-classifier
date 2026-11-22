# Evaluation results

Datasets: test split: data/processed/customer_requests.csv (131 rows); manual_test: data/eval/german_test_set.csv (120 rows); out_of_scope: data/eval/out_of_scope.csv (60 rows).

## tfidf_logreg

| Dataset | Rows | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| test split | 131 | 88.5% | 84.4% |
| manual_test | 120 | 49.2% | 46.7% |

Accepted coverage / accepted accuracy by threshold:

| Threshold | Split coverage | Split accuracy | Manual coverage | Manual accuracy |
| --- | --- | --- | --- | --- |
| 0.50 | 100.0% | 88.5% | 89.2% | 51.4% |
| 0.60 | 92.4% | 91.7% | 66.7% | 58.8% |
| 0.65 | 89.3% | 93.2% | 57.5% | 62.3% |
| 0.70 | 87.0% | 93.0% | 47.5% | 68.4% |
| 0.80 | 77.1% | 96.0% | 31.7% | 65.8% |

Out-of-scope: n=60, mean confidence 0.689. Share flagged for review:

| Threshold | Flagged |
| --- | --- |
| 0.50 | 1.7% |
| 0.60 | 28.3% |
| 0.65 | 41.7% |
| 0.70 | 60.0% |
| 0.80 | 80.0% |

Recommended threshold: none reaches accepted accuracy 0.95 on manual_test.
