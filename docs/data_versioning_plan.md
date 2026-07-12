# Data Versioning Plan

The project currently contains many local CSV, JSON, XLSX, result, and debug files. These files are useful for research, but not all of them should be committed directly to Git.

## Git Should Track

Git should track source code, documentation, small examples, and configuration templates.

Examples:

- Python source files.
- `README.md`.
- Documentation under `docs/`.
- Small sample inputs under `inputs/`.
- `.gitignore`.
- Future `.env.example`.

## Git Should Not Track

Git should not track secrets, environments, caches, or large/generated outputs.

Examples:

- `.env`
- `venv/`
- `__pycache__/`
- Large debug report folders.
- Generated result files unless intentionally selected as examples.

## DVC Candidate Files

DVC can be used later for datasets and generated outputs that need versioning but are too large or too changeable for Git.

Candidate folders:

- `data/raw/`
- `data/processed/`
- `data/evaluation/`
- `outputs/results/`
- `outputs/debug_reports/`

## Current Safe Approach

For now, the working pipeline remains unchanged. The project will first establish a clean Git baseline with code and documentation. DVC can be added after deciding which datasets and outputs should be reproducible research artifacts.
