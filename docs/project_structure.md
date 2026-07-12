# Project Structure

The current project keeps working code files at the repository root to avoid changing imports or pipeline behavior.

## Current Important Files

```text
project/
├── api_app.py
├── pipeline_langchain.py
├── entity_extraction_langchain.py
├── aspect_langchain.py
├── coref_langchain.py
├── text_utils.py
├── html_sentence_splitter.py
├── run_pipeline_with_debug_reports.py
├── export_coref_evaluation.py
├── inputs/
├── docs/
├── data/
└── outputs/
```

## New Organization Folders

```text
docs/
```

Stores project documentation.

```text
data/raw/
data/processed/
data/evaluation/
```

Reserved for datasets and evaluation files.

```text
outputs/results/
outputs/debug_reports/
```

Reserved for generated pipeline outputs and debug reports.

## Future Refactor Option

After a stable Git baseline, the project can be refactored into a cleaner package layout:

```text
src/
api/
scripts/
tests/
configs/
```

That refactor should be done later because moving files now would require import updates and could affect the working pipeline.
