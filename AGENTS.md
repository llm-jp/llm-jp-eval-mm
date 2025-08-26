# Repository Guidelines

This is a concise, coding‑agent–friendly guide for contributing and extending the llm-jp-eval-mm evaluation framework.

## Project Structure
- `src/eval_mm/`: Core library
  - `tasks/`: Task loaders/adapters; register in `task_registry.py`
  - `metrics/`: Scorers and aggregation utilities; register in `scorer_registry.py`
  - `utils/`: Helpers (e.g., Azure/OpenAI client)
- `examples/`: Reference VLM wrappers and runnable samples
- `scripts/`: Leaderboard, Streamlit browser, dataset prep
- `assets/`, `data/`, `dataset/`: Static assets and datasets (not committed)
- `result/`, `outputs/`: Evaluation artifacts written by runs

## Key Commands
- Setup: `uv sync` (model deps via groups, e.g., `uv sync --group normal`)
- Run sample eval: `uv run --group normal python examples/sample.py ...`
- Tests: `bash test.sh` (tasks/metrics), `bash test_model.sh` (model smoke)
- Lint/format: `uv run ruff format src && uv run ruff check --fix src`
- Type check: `uv run mypy src`
- Browse predictions: `uv run streamlit run scripts/browse_prediction.py -- --task_id <id> --result_dir result --model_list <model>`
- Leaderboard: `python scripts/make_leaderboard.py --result_dir result`

## Development Playbook (for Agents)
- Add a task: implement `Task` in `src/eval_mm/tasks/<name>.py`; import it in `src/eval_mm/tasks/__init__.py`; register with `@register_task` in `task_registry.py`.
- Add a scorer: implement in `src/eval_mm/metrics/<name>_scorer.py`; import in `metrics/__init__.py`; register in `scorer_registry.py`.
- Add a model: wrap in `examples/` (see existing VLM wrappers) and map via `examples/model_table.py`.
- Import pattern: `from eval_mm import TaskRegistry, ScorerRegistry` (avoid `src.` prefixes).
- Tests: include `def test_*` near tasks/metrics; run `uv run pytest src/eval_mm/tasks/<file>.py -v`.

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
