# Testing Conventions Skill

Agent skill for FastPaip testing standards. Provides guidelines, scripts, and patterns for writing tests following the test pyramid approach.

## Structure

```
testing-conventions.skill/
├── SKILL.md                    # Main skill instructions (loaded by Claude)
├── scripts/
│   ├── generate_test.py       # Generate test file scaffolds
│   └── validate_tests.py      # Validate tests against conventions
└── references/
    ├── patterns.md            # Detailed test pattern examples
    └── fixtures.md            # Fixture patterns and best practices
```

## Quick Start

### 1. Generate a Test

```bash
cd scripts/

# Unit test
python generate_test.py \
  --type unit \
  --module core \
  --subject ArticleService \
  --output ../../../packages/core/tests/unit/domain/test_article_service.py

# Integration test
python generate_test.py \
  --type integration \
  --module core \
  --subject ArticleRepository \
  --output ../../../packages/core/tests/integration/infrastructure/test_article_repository.py

# E2E test
python generate_test.py \
  --type e2e \
  --module core \
  --subject ArticleFlow \
  --output ../../../packages/core/tests/e2e/test_article_flow.py
```

### 2. Validate Tests

```bash
cd scripts/

# Validate single file
python validate_tests.py ../../../packages/core/tests/unit/domain/test_article_service.py

# Validate directory
python validate_tests.py ../../../packages/core/tests/

# Strict mode (warnings become errors)
python validate_tests.py --strict ../../../packages/core/tests/
```

## What This Skill Provides

### SKILL.md
- Quick reference for test structure and naming
- Test pyramid guidance (when to use unit/integration/e2e)
- Common patterns and anti-patterns
- Running tests commands

### Scripts

#### generate_test.py
Generates test scaffolds with:
- Correct naming conventions
- Proper structure (Arrange-Act-Assert)
- Type-appropriate patterns (unit/integration/e2e)
- TODO markers for completion

#### validate_tests.py
Checks tests for:
- File naming conventions
- Directory structure
- Function naming patterns
- Docstrings
- Common anti-patterns (time.sleep, testing privates, etc.)
- Fixture naming conventions

### References

#### patterns.md
Comprehensive examples:
- Unit test patterns (domain services, orchestration, error handling)
- Integration test patterns (repositories, queries, transactions)
- E2E test patterns (complete flows, error recovery)
- Fixture patterns (mocks, factories, composition)
- Common scenarios (event handlers, API routes, async code)
- Anti-patterns to avoid

#### fixtures.md
Detailed fixture guide:
- Fixture scopes (function, class, module, session)
- Mock fixtures (basic, pre-configured, with side effects)
- Database fixtures (sessions, seeded data, transactions)
- Factory fixtures (basic, with session, sequences)
- Composed fixtures (layered, combined)
- Parametrized fixtures
- Async fixtures
- Cleanup patterns
- Best practices and troubleshooting

## Usage in Claude

When Claude identifies you're working with tests, this skill provides:
1. **Quick guidance** via SKILL.md (always loaded)
2. **Detailed examples** via references (loaded on demand)
3. **Executable scripts** for generation and validation

Claude can:
- Suggest appropriate test type (unit/integration/e2e)
- Generate test scaffolds using the script
- Reference detailed patterns from references/
- Validate tests against conventions
- Guide fixture design and usage

## Benefits

1. **Consistency**: All tests follow the same conventions
2. **Discoverability**: Patterns and examples readily available
3. **Automation**: Scripts handle repetitive scaffolding
4. **Validation**: Catch convention violations early
5. **Documentation**: References provide deep examples

## Testing Philosophy

- **Test behavior, not implementation**
- **Keep tests isolated and independent**
- **Follow the test pyramid** (many unit, some integration, few e2e)
- **Use AAA pattern** (Arrange-Act-Assert)
- **Mock external dependencies** in unit tests
- **Use real database** in integration tests
- **Test complete flows** in e2e tests

## Fixture Convention

FastPaip uses a specific fixture naming pattern:

```python
@pytest.fixture(name="fixture_name")
def _fixture_name_fixture():
    """Docstring describing the fixture."""
    pass
```

This provides:
- **Clean test signatures**: Use `fixture_name` in tests
- **Private implementation**: The `_` prefix marks implementation details
- **Searchability**: All fixtures end with `_fixture` suffix
- **Organization**: Group with section comment headers

See [references/fixtures.md](references/fixtures.md) for comprehensive examples.

## Integration with FastPaip

This skill follows FastPaip's:
- Bounded Context architecture
- Domain-Driven Design principles
- Ports and Adapters pattern
- Event-driven communication

Tests are organized by BC and test type:
```
packages/<bc>/tests/
├── unit/           # Domain and application logic
├── integration/    # Infrastructure and persistence
└── e2e/           # Complete user flows
```

## Maintenance

To update conventions:
1. Update SKILL.md for high-level changes
2. Update references/ for detailed examples
3. Update scripts/ for new generation patterns
4. Keep original docs/testing-conventions.md in sync
