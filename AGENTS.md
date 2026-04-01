# Agents

## Python Environment

All Python-related tasks must use the `uv` package manager with the local `.venv` environment.

- Use `uv run` to execute Python scripts (e.g., `uv run python train_dpo.py`).
- Use `uv pip install` to install packages into the local `.venv`.
- Use `uv sync` to synchronize dependencies.
- Do not use `pip` or `conda` directly.