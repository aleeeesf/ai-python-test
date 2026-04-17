# AGENTS

## Scope
These instructions apply to the whole repository.

## Project Layout
- `app/`: Main FastAPI notification service.
- `provider/`: External provider simulator (rate limit + random failures).
- `platform/k6/`: Load-test and provider health-check scripts.
- `platform/grafana/` and `platform/influxdb/`: dashboards/metrics wiring for k6 runs.

## Challenge Goal
Build a resilient notification service that passes pressure tests while honoring the API contract.

The service receives a notification request, stores it, and later processes delivery against
the external provider with retry/error handling.

## Runtime Topology
- The API under test is exposed on port `5000`.
- External provider runs on port `3001`.
- Load tests target `http://provider:5000` in Docker Compose.
- Provider requires header `X-API-Key: test-dev-2026`.

## Python Import Style
When editing files inside `app/`, use imports relative to the `app` runtime root, for example:
- `from infrastructure.http.v1.router import v1_router`
- `from application.dtos import CreateRequestDTO`
- `from domain.entities.request import NotificationRequest`

Do not use `from app....` imports in application code.

## Architecture Guidelines
Use a layered/ports-and-adapters approach:

- `domain/`
  - Entities, enums, domain-level exceptions.
  - Port interfaces (`domain/ports`) only; no framework code.
- `application/`
  - Use cases and DTOs for app orchestration.
  - No FastAPI/HTTP details inside use cases.
- `infrastructure/`
  - HTTP routes, repository implementations, provider adapters, worker/dispatchers.
- `core/`
  - Configuration and dependency wiring.

Keep HTTP handlers thin. Put business flow in use cases.

## Current Process Flow (Important)
`POST /v1/requests/{id}/process` is split into:
- `start(...)` in use case:
  - validates existence,
  - checks status (`sent`, `processing`, etc.),
  - transitions to `processing` when applicable.
- `deliver(...)` in background:
  - calls provider,
  - retries transient errors,
  - writes final status (`sent` or `failed`).

This separation is intentional so process endpoint can return fast (`202`) under load.

## Provider Behavior (Used for Resilience Logic)
Provider endpoint: `POST /v1/notify`
- `200`: success with `provider_id`
- `401`: invalid API key (non-retryable)
- `429`: rate limit (retryable)
- `500`: random external error (retryable)
- network/timeout: retryable

Use retries with bounded attempts and backoff. Always end in terminal state (`sent`/`failed`).

## Local Checks
Run checks from repository root unless noted:

```bash
app/.venv/bin/ruff check --config app/pyproject.toml app
cd app && MYPYPATH=. .venv/bin/mypy .
```

Optional formatting:

```bash
app/.venv/bin/ruff format --config app/pyproject.toml app
```

## Running Locally
Service only:

```bash
cd app
uv run uvicorn main:app --host 0.0.0.0 --port 5000
```

Full stack and load test:

```bash
docker-compose up -d provider influxdb grafana
docker-compose up -d --build app
docker-compose run --rm load-test
```

## Pre-commit
Pre-commit config is at `app/.pre-commit-config.yaml`.
Hooks run Ruff and mypy for files under `app/`.

## CI
GitHub Actions workflow:
- `.github/workflows/ci.yml`

It runs on:
- `main`
- `feature/**`

And executes:
- Ruff lint
- mypy type checking

## API Contract (Challenge)
Required endpoints on port `5000`:
- `POST /v1/requests` -> `201` with `{"id":"..."}`
- `POST /v1/requests/{id}/process` -> `200` or `202`
- `GET /v1/requests/{id}` -> `200` with `{"id":"...","status":"queued|processing|sent|failed"}`

Status semantics:
- `queued`: created, not yet processing
- `processing`: accepted and currently being delivered
- `sent`: provider accepted successfully
- `failed`: retries exhausted or non-retryable failure

## Implementation Notes
- Keep routes thin; orchestration belongs in `application/use_cases/`.
- Ports/interfaces belong in `domain/ports/`.
- Adapters/implementations belong in `infrastructure/`.
- For this challenge, in-memory repository and embedded background worker are acceptable.
- Catch unexpected exceptions in background delivery and persist `failed` state to avoid hanging in `processing`.
- Avoid introducing blocking I/O in request handlers.
