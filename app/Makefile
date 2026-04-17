.PHONY: lint lint-fix type-check check

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix
	uv run ruff format .

type-check:
	uv run mypy .

check: lint lint-fix type-check
