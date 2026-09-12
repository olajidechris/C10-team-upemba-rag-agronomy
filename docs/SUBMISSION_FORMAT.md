# Submission Format

## Supported Script Tracks

This repository supports two submission-generation tracks:

1. **All-local track**: `scripts/all_local.py`
2. **External-API reranking track**: `scripts/external_reranking_api.py`

Both tracks must produce the same final submission format.

## Required Output File

- File name: `submission.csv`
- Columns (exact order): `QueryId,DocumentId`
- Shape: 5 rows per test query (top-5 ranked documents)

## Input Data Location

Use files in `data/input/`:

- `documents.csv`
- `train_queries.csv`
- `qrels_train.csv`
- `test_queries.csv`
- `sample_submission.csv`

## Notes

- The two script tracks are alternative implementations of the same retrieval task.
- Either track is valid as long as the output matches the required submission schema.
