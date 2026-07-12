# File Inventory

This file documents the current codebase layout without changing file locations. The working pipeline files currently remain at the repository root to preserve imports and avoid behavior changes.

## Core Pipeline

- `pipeline_langchain.py` - Main post-level pipeline orchestration.
- `entity_extraction_langchain.py` - Full-post software entity extraction.
- `aspect_langchain.py` - Aspect A/B classification and sentiment labeling.
- `coref_langchain.py` - Referring expression detection and aspect-positive coreference resolution.
- `text_utils.py` - Text preprocessing and sentence splitting interface.
- `html_sentence_splitter.py` - HTML stripping and sentence splitting implementation.

## API Layer

- `api_app.py` - FastAPI wrapper exposing the pipeline through local API endpoints.

## LLM / JSON Utilities

- `lc_llm.py` - Central LangChain/OpenAI model client configuration.
- `lc_json.py` - Safe JSON parsing helper for LLM outputs.
- `entity_focus.py` - Helper for extracting the tagged target mention from `<E>...</E>` views.
- `canonicalize.py` - Triple/entity canonicalization utility.

## Input Conversion Scripts

- `make_sample_json.py` - Creates sample JSON input for quick pipeline checks.
- `csv_to_posts_json.py` - Converts CSV sentence/post data into JSON input.
- `excel_to_posts_json.py` - Converts spreadsheet data into JSON input.
- `create_posts_json.py` - Utility for creating post JSON files.

## Evaluation and Debugging

- `run_pipeline_with_debug_reports.py` - Runs the pipeline over posts and saves per-post debug reports.
- `export_coref_evaluation.py` - Exports coreference evaluation rows from debug reports.
- `compare_triples.py` - Compares system triples against evaluation inputs.
- `run_pipeline_with_timer.py` - Runs the pipeline with elapsed-time reporting.

## Active Data and Result Files

The repository currently contains local CSV, XLSX, JSON, and result files from research experiments. These should be reviewed before committing or moved under `data/` and `outputs/` later.

Recommended future placement:

- Raw datasets: `data/raw/`
- Converted JSON inputs: `data/processed/`
- Evaluation spreadsheets: `data/evaluation/`
- Pipeline outputs: `outputs/results/`
- Debug reports: `outputs/debug_reports/`

## Future Refactor

A later refactor can move files into `src/`, `api/`, and `scripts/`. That should be done on a separate branch after the current working baseline is committed and verified.
