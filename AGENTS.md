# Repository Guidelines

This is a concise, coding‑agent–friendly guide for contributing and extending the llm-jp-eval-mm evaluation framework.

## Project Structure

- `src/eval_mm/`: Core library
  - `tasks/`: Task loaders/adapters; register in `task_registry.py`
  - `metrics/`: Scorers and aggregation utilities; register in `scorer_registry.py`
  - `utils/`: Helpers (e.g., Azure/OpenAI client)
- `examples/`: Legacy in-process vLLM backend (`base_vllm.py`, `vllm_registry.py`), kept until the server-based path has been run on the clusters
- `scripts/`: Leaderboard, Streamlit browser, dataset prep
- `assets/`, `data/`, `dataset/`: Static assets and datasets (not committed)
- `result/`, `outputs/`: Evaluation artifacts written by runs

## Key Commands

- Setup: `uv sync --group dev`
- Run sample eval: `uv run eval-mm run --base_url http://localhost:8000/v1 --model_id <id> --task_id <task> --metrics <metric> --max_dataset_len 10` against a running `vllm serve`
- Serve and evaluate a pool model: `uv run eval-mm run-model --model_id <id> --tasks <task> --max_dataset_len 10 --inference_only`
- Tests: `bash test.sh` (tasks/metrics)
- Lint/format: `uv run ruff format src && uv run ruff check --fix src`
- Type check: `uv run mypy src`
- Browse predictions: `uv run streamlit run scripts/browse_prediction.py -- --task_id <id> --result_dir result --model_list <model>`
- Leaderboard: `python scripts/make_leaderboard.py --result_dir result`

## Development Playbook (for Agents)

- Add a task: implement `Task` in `src/eval_mm/tasks/<name>.py`; import it in `src/eval_mm/tasks/__init__.py`; register with `@register_task` in `task_registry.py`.
- Add a scorer: implement in `src/eval_mm/metrics/<name>_scorer.py`; import in `metrics/__init__.py`; register in `scorer_registry.py`.
- Add a model: add an entry to `src/eval_mm/model_pool.toml` (`status = "pending"` until it has been served and checked on a few samples). Models are evaluated through an OpenAI-compatible server; do not add per-model wrappers.
- Import pattern: `from eval_mm import TaskRegistry, ScorerRegistry` (avoid `src.` prefixes).
- Tests: include `def test_*` near tasks/metrics; prefer `bash test.sh` (tasks/metrics). For a single file, you may optionally run `uv run --group dev pytest <path> -v`, but CI expects the scripts.

## Plan-First Workflow

- Before any change, prepare a short checklist: objective, source of truth, inventory, diff policy, implementation steps, and acceptance criteria.
- After alignment, implement the minimum needed for the agreed scope.
- Example (naming unification):
  - Source of truth: treat `scripts/nvlink/config.sh` entries (e.g., task IDs and metric map) as canonical.
  - Inventory: compare identifiers used across code and configuration, and list discrepancies.
  - Implementation: adopt the canonical identifiers in public-facing interfaces; keep backward-compatible aliases only if necessary.
  - Validation: run `uv run python scripts/validate_config_consistency.py` and `bash test.sh`.

## Coding Style & Conventions

- Python ≥ 3.12, 4‑space indentation, type hints required
- Names: packages/modules `lower_snake_case`; classes `CamelCase`; functions/vars `lower_snake_case`
- Keep functions focused; prefer dataclasses/typed types for structured data
- Use Ruff + pre-commit; follow existing import order and ignore rules

## Commit & PR Guidelines

- Prefix commits with `feat:`, `fix:`, `chore:`, `docs:` (see `git log`)
- PRs include: clear description, linked issues, repro commands, sample outputs (e.g., `result/<task>/<model>/evaluation.jsonl`); CI must pass

## Security & Config

- LLM‑as‑a‑Judge: set `.env` with `AZURE_OPENAI_ENDPOINT`/`AZURE_OPENAI_KEY` or `OPENAI_API_KEY`
- Do not commit secrets or large datasets; use `.env.sample`
- Add model deps via `uv` groups and update conflicts in `pyproject.toml`

## Temporary Validation (_tmp_ Policy)

- Name temporary files/dirs with `_tmp_` (e.g., `result/<task>/<model>_tmp_/<run>`).
- Keep them under `result/`, `outputs/`, or `artifact/` and remove after validation.
- Avoid committing `_tmp_` artifacts; they are ignored by `.gitignore`.
