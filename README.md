# TriAi Team Upemba Project: Optimizing RAG Document Retrieval for Agronomic Advice

This repository contains Team Upemba's final submission for the TRI AI Saturdays Cohort 10 Project 9 Challenge. The project focuses on a CPU-efficient hybrid retrieval pipeline designed to surface high-value agricultural extension advice for African smallholder farmers from natural language queries.

## Project Summary

Our approach combines lexical retrieval, dense semantic retrieval, and reranking to bridge the gap between formal agronomic terminology and the localized language farmers use in practice. The repository also preserves the original working artifacts while presenting a cleaner submission-ready structure.

## Dataset

The project uses a self-contained knowledge base of **695** short agricultural extension factsheets focused on Sub-Saharan African agriculture.

- Topics include crop diseases, pests, soil management, nutrient deficiencies, and climate adaptation.
- Source materials include grounded open-access CC-BY research documents and synthetic public-domain extension-style documents.
- Training data includes **308** queries and **4,194** graded relevance judgments in `data/input/qrels_train.csv`.
- Relevance grades follow a 4-tier scale: 3 (Perfect), 2 (Relevant), 1 (Marginal/Wrong Crop), and 0 (Irrelevant/Hard Negative).

## Retrieval and Reranking Pipeline

The submission centers on a hybrid retrieval strategy and a reranking stage tailored to agronomic search:

1. **Hybrid search** combines BM25-style lexical matching with dense semantic embeddings.
2. **Cross-encoder reranking** improves top-rank precision using graded relevance supervision.
3. **Crop-aware handling** reduces wrong-crop matches that appear semantically similar.
4. **CPU-conscious design** keeps the final retrieval workflow suitable for low-resource environments.

## Evaluation

The system is evaluated with **nDCG@5**, the challenge's rank-sensitive retrieval metric.

- The challenge baseline TF-IDF score was approximately **0.551 nDCG@5**.
- Team Upemba's reported validation performance reached **0.8755 nDCG@5** on the labeled training-query evaluation used during project development.
- The checked-in `scripts/all_local.py` script evaluates retrieval quality against `data/input/train_queries.csv` and `data/input/qrels_train.csv` before generating test-set predictions.

## Reproduction

This repository contains two submission-generation script tracks:

- `scripts/all_local.py` - all-local hybrid retrieval and reranking pipeline.
- `scripts/external_reranking_api.py` - hybrid retrieval pipeline with external API reranking.

Run the scripts from the repository root:

```bash
python scripts/all_local.py
python scripts/external_reranking_api.py
```

Optional environment variables:

- `AGRONOMY_INPUT_PATH` - overrides the default input directory (`data/input/` locally).
- `AGRONOMY_OUTPUT_PATH` - overrides the default output directory (`data/output/` locally).
- `VOYAGE_API_KEY` - required for `scripts/external_reranking_api.py` when Kaggle secrets are unavailable.

For the all-local pipeline, the intended workflow is:

1. Run `scripts/all_local.py` in an environment that can prepare the required model and retrieval caches.
2. Reuse the generated cache artifacts from the runtime-created `data/output/pipeline_cache/` directory (or the configured output directory) for subsequent CPU-only runs.
3. Generate the final `submission.csv` with `QueryId,DocumentId` pairs for the top-5 ranked results per test query.

Input files are expected in `data/input/`, including:

- `documents.csv`
- `train_queries.csv`
- `qrels_train.csv`
- `test_queries.csv`
- `sample_submission.csv`

## Checked-In Repository Map

- `README.md` - top-level project overview and navigation.
- `PROJECT_STRUCTURE.md` - simplified high-level repository layout.
- `data/input/` - competition input files, labels, and sample outputs.
- `data/submission/` - checked-in final submission payloads.
- `scripts/` - executable script versions of the repository's retrieval pipelines.
- `docs/` - submission documents, structure notes, and format guidance.
- `resources/` - archived notebooks, challenge responses, legacy artifacts, and cached result bundles.

For the maintained high-level tree, refer to `docs/PROJECT_STRUCTURE.md`.

When the scripts run locally, they also create `data/output/` for generated artifacts such as `pipeline_cache/` and `submission.csv`.

## Documentation

- `docs/PROBLEM_STATEMENT.md` - challenge problem statement.
- `docs/DATA_CARD.md` - dataset description and usage context.
- `docs/IMPACT_STATEMENT.md` - intended benefits, risks, and mitigations.
- `docs/STAKEHOLDER_ENGAGEMENT_PLAN.md` - stakeholder plan for farmer and extension-worker feedback.

## Team Upemba Members

- Christopher Aziki Olajide (Team Lead)
- Ezekiel Olaniyi-Omolofi
- Chisaneme Aloni
- Elizabeth Sanyaolu
- Sumayah Adegbite
- Oma Anosike Ihuoma

## Mentors

- Oluwaseun Ajayi
- Samuel Taiwo
- Adnan Adetunji
