# Repository Guidelines

## Project Structure & Modules
- `pvg/`: Core Python package. Key areas: `components/` (orchestrators, managers), `orchestrator/` (training loop), `trainers/`, `inference/` (vLLM client/server), `data/` (datasets, prompts), `zero/` (DeepSpeed configs).
- `non-slurm/` and `standalone/`: Launch scripts for local/cluster runs; see `spawn-vast*.sh` and `standalone/generator.sh`.
- `docs/`: Architecture notes and optimization guides.
- `theory/`: Paper material and references.
- Root scripts: `startup.sh`, `emergency_cleanup.sh`, `benchmark_evaluator.py`.

## Build, Test, and Development Commands
- Environment (uv):
  - `uv venv --python 3.11` then `uv sync`
  - Install hooks: `make hooks` (pre-commit, ruff, black)
- Lint/format:
  - `uv run ruff . --fix`
  - `uv run black .`
- Run CLI:
  - Quick help: `uv run python -m pvg.main --help`
  - Example (single-node accelerate via script): see `non-slurm/spawn-vast*.sh` for full training pipelines.

## Coding Style & Naming
- Python 3.11; line length 120; prefer type hints where practical.
- Naming: `snake_case` for modules/functions, `PascalCase` for classes, `SCREAMING_SNAKE_CASE` for constants.
- Imports: sorted by ruff-isort; third-party packages (e.g., `wandb`) treated as third-party.
- Tools: ruff (lint/fix), black (format). Ensure pre-commit passes before pushing.

## Testing Guidelines
- No centralized test suite yet. When adding tests:
  - Use `pytest`; place files under `tests/` with `test_*.py` names.
  - Add dev dep: `uv add --dev pytest` and run `uv run pytest -q`.
  - Favor small, isolated unit tests for `pvg/components/*` and `pvg/processors/*`.
  - For performance checks, see `benchmark_evaluator.py` as an example harness.

## Commit & Pull Request Guidelines
- Commits: concise, imperative mood (e.g., “add verifier caching”, “fix vLLM ports”). Group related changes.
- Branches: use topical names like `feat/verifier-regressor` or `fix/eval-timeouts`.
- PRs: include description, rationale, minimal repro (commands/paths), and impact on configs (`zero/*.json`, `.env`, scripts). Link issues when applicable and add logs/screenshots for runtime changes.

## Security & Configuration Tips
- Large models: prefer local caches (`local_models/...`) referenced by scripts; avoid committing weights.
- GPU/Distributed: review `NCCL_*`, `CUDA_VISIBLE_DEVICES`, and DeepSpeed configs in `pvg/zero/` before runs.
- Secrets: configure Weights & Biases via env; do not commit tokens.
