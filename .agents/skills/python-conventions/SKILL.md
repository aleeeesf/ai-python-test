---
name: python-conventions
description: Python conventions and best practices for writing clear, idiomatic, PEP-aligned code with explicit typing and documented public APIs. Use this skill when writing, reviewing, or refactoring Python code.
version: 1.0.0
provider: fastpaip
---

# Python Conventions

Use this skill to produce Python code that is readable, maintainable, and aligned with common Python conventions and relevant PEPs.

## Core Principles

1. Prefer clear, explicit code over clever code.
2. Follow `PEP 8` for style and formatting.
3. Follow `PEP 257` for docstrings.
4. Use `mypy` as the typing reference for the repository.
5. Keep functions and classes focused on one responsibility.
6. Prefer standard library tools before adding dependencies.
7. Preserve existing project conventions when they are already established.

## Naming

- Use `snake_case` for functions, variables, modules, and files.
- Use `PascalCase` for classes and exceptions.
- Use `UPPER_SNAKE_CASE` for constants.
- Name booleans with prefixes like `is_`, `has_`, `can_`, or `should_`.
- Avoid ambiguous abbreviations and overly terse names.
- Use `self` and `cls` for instance and class methods.

## Typing

- Type all public functions, methods, and data structures.
- Prefer precise types over `Any`.
- Use `Any` only when the value is genuinely dynamic and cannot be modeled cleanly.
- Prefer `list[T]`, `dict[K, V]`, `tuple[...]`, `Optional[T]`, and `|` unions when they express the contract clearly.
- Use `TypedDict`, `dataclass`, or small model classes when they improve clarity.
- Do not silence type errors unless there is a concrete reason and a local justification.

## Docstrings

- Document public functions, classes, and modules when the purpose is not obvious from the code.
- Keep docstrings concise and factual.
- Use one line per parameter in `Args`.
- Do not repeat the parameter type in `Args` when it is already in the function signature.
- In `Returns`, describe what the return value contains or represents.
- Omit `Returns` when the function returns `None`.
- In `Raises`, document only exceptions that callers are expected to handle.
- Class docstrings should explain responsibility and behavior, not restate the class name.

### Docstring Pattern

```python
def fetch_policies(self, page: int, page_size: int) -> list[dict[str, Any]]:
    """Fetch a single page of policies from the Codeoscopic API.

    Args:
        page: Zero-based page number.
        page_size: Number of results per page (max 100).

    Returns:
        List of raw policy dicts from the API response.

    Raises:
        aiohttp.ClientError: On network failure after all retries are exhausted.
    """
```

```python
class CodeoscopicAuthClient:
    """Manages OAuth2 client_credentials auth for the Codeoscopic API.

    Caches the token in memory and re-authenticates transparently before expiry.
    """
```

## Structure

- Keep functions small and easy to scan.
- Extract helpers when a function starts mixing validation, transformation, and I/O.
- Prefer composition over inheritance unless inheritance is clearly justified.
- Use `dataclass` for simple structured data.
- Prefer `pathlib`, `with`, comprehensions, and iterator helpers when they improve clarity.

## Errors

- Raise specific exceptions.
- Catch only exceptions you can handle.
- Avoid broad `except Exception` blocks unless the code clearly needs them.
- Do not swallow errors silently.

## Imports

- Group imports as standard library, third-party, then local imports.
- Avoid `from module import *`.
- Remove unused imports.
- Prefer absolute imports when the project layout supports them.

## Refactoring Rules

- Preserve behavior unless a change is explicitly requested.
- Make the smallest correct change that improves readability or correctness.
- Prefer simple, direct code over over-abstracted helpers.
- When a tradeoff exists, choose the option that is easiest to maintain.

## Review Checklist

- Does the code satisfy `PEP 8` and `PEP 257`?
- Are names explicit and consistent?
- Are public APIs typed clearly enough for `mypy`?
- Are docstrings present and useful where needed?
- Is the implementation minimal, readable, and maintainable?

