# Run mypy type checker
uv run mypy src/

# Run ruff linter
uv run ruff check src/

# Run ruff formatter check
uv run ruff format --check src/

# Run all tests
uv run pytest tests/ -v
