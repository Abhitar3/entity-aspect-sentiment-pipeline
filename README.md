# Entity-Aspect-Sentiment Research Pipeline

This repository contains an NLP/LLM research pipeline for converting developer discussion posts into structured sentence-level Entity-Aspect-Sentiment triples.

The current system is designed for software engineering research using Stack Overflow-style posts. It extracts software entities, identifies explicit mentions and referring expressions, classifies Aspect A and Aspect B with sentiment, resolves aspect-relevant coreference, and produces traceable final triples with optional debug reports.

## Current Pipeline

At a high level, the pipeline does the following:

1. Accepts one or more full developer posts as input.
2. Cleans the text and splits posts into sentences.
3. Extracts software entities at the full-post level.
4. Detects explicit entity mentions and referring expressions in each sentence.
5. Creates tagged mention-level sentence views using `<E>...</E>`.
6. Uses LLM calls to classify Aspect A, Aspect B, and sentiment for each tagged mention.
7. Runs coreference resolution only for aspect-positive referring expressions.
8. Produces final Entity-Aspect-Sentiment triples.
9. Optionally saves debug reports for intermediate pipeline inspection.

## Main Files

- `pipeline_langchain.py` - Main post-level pipeline orchestration.
- `pipeline_langchain_new.py` - Sentence-level/newer experimental pipeline variant.
- `entity_extraction_langchain.py` - Full-post software entity extraction.
- `aspect_langchain.py` - Aspect A/B and sentiment classification.
- `coref_langchain.py` - Referring expression detection and coreference resolution.
- `text_utils.py` and `html_sentence_splitter.py` - HTML stripping and sentence splitting.
- `run_pipeline_with_debug_reports.py` - Batch runner that saves per-post debug reports.
- `export_coref_evaluation.py` - Utility for creating coreference evaluation spreadsheets.
- `api_app.py` - FastAPI wrapper around the pipeline for API testing/demo use.
- `aspect_b_subaspect_langchain.py` - Separate Aspect B item-code classifier for usability subtype analysis.
- `run_aspect_b_subaspect_pipeline.py` - Runner for the Aspect B item-code experiment.

## Project Layout

The code files are intentionally kept at the repository root for now so the existing imports and commands keep working.

```text
data/raw/              Original CSV/XLSX/TXT research datasets
data/processed/        JSON inputs prepared for pipeline runs
data/evaluation/       Evaluation workbooks or derived evaluation files
outputs/results/       Generated pipeline outputs
outputs/debug_reports/ Intermediate per-post debug reports
inputs/                Small local sample JSON files
docs/                  Pipeline, API, evaluation, and project notes
```

The `data/` and `outputs/` folders are ignored by Git except for `.gitkeep` placeholders. This keeps the repository lightweight while preserving the project structure.

## API Usage

Run the local FastAPI app:

```powershell
python -m uvicorn api_app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Example request body for `/analyze`:

```json
{
  "posts": [
    "BeautifulSoup is the de-facto standard library for parsing web pages in Python. It's great for server-rendered or static content."
  ],
  "include_debug": true
}
```

The API also supports raw-text analysis for demo use through `/analyze-text`, where the user can provide a single post string instead of a prebuilt JSON file.

## Documentation

- `docs/pipeline_flow.md` - Step-by-step pipeline flow and LLM call locations.
- `docs/evaluation_summary.md` - Current evaluation and benchmarking summary.
- `docs/api_usage.md` - FastAPI request/response usage.
- `docs/data_versioning_plan.md` - Notes on data organization and future versioning options.
- `docs/project_structure.md` - Current and planned project organization.
- `docs/file_inventory.md` - Current file responsibilities without moving code.

## Notes

This is an active research codebase. The current organization preserves the working pipeline files in place to avoid breaking imports or changing behavior. Future cleanup can move code into `src/` and scripts into `scripts/` after a stable baseline commit.

Because datasets and generated outputs now live under ignored folders, commands should use paths such as `data/processed/posts.json` and `outputs/results/posts_result.json` instead of root-level JSON or CSV paths.
