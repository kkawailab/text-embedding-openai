# Repository Guidelines

## Project Structure & Module Organization
- `main.py` contains the executable workflow for generating OpenAI text embeddings, including helper utilities such as `get_embeddings` and `cosine_similarity`.
- Project metadata lives in `pyproject.toml` and `uv.lock`; update dependencies there so contributors can recreate the environment with `uv`.
- Keep documentation and onboarding notes in `README.md`; add design docs within `docs/` if a feature merits deeper explanation.
- Place new tests under `tests/` mirroring the module hierarchy (e.g., `tests/test_embeddings.py`) and co-locate sample inputs/outputs under `tests/fixtures/`.

## Build, Test, and Development Commands
```bash
uv run python main.py        # run the demo script end to end
uv run pytest                # execute the full automated test suite
uv add --dev pytest          # install missing dev tooling before the first test run
```
Use `UV_PYTHON=python3.13` (or higher) to ensure the interpreter matches the `requires-python` constraint.

## Coding Style & Naming Conventions
- Follow standard PEP 8 with 4-space indentation; prefer descriptive snake_case for functions/variables and UpperCamelCase only for classes.
- Keep pure functions like `cosine_similarity` side-effect free; isolate API calls inside dedicated helpers so they can be mocked.
- When adding modules, export a single public entry point via clear `__all__` definitions and document expected shapes for numpy arrays.
- Secure config (API keys) through environment variables such as `OPENAI_API_KEY`; never hard-code secrets or commit `.env` files.

## Testing Guidelines
- Write pytest tests (`test_*` functions inside `Test*` classes when grouping related behavior) and aim for coverage on all numeric helpers and API adapters.
- Use fixtures to stub OpenAI responses; store canned payloads under `tests/fixtures/` and keep them minimal to speed execution.
- Run `uv run pytest -q` locally before opening a pull request; add regression tests whenever fixing a bug in similarity scoring or embedding parsing.

## Commit & Pull Request Guidelines
- Follow the existing imperative style (`Add initial project files and configurations`); keep subject lines under ~72 characters and explain reasoning in the body when needed.
- Reference related issues in the PR description, summarize functional changes, list test evidence (`uv run pytest` output), and attach screenshots only if the change affects user-visible behavior (CLI logs suffice otherwise).
