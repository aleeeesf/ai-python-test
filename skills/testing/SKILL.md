---
name: testing-conventions
description: FastPaip testing standards for unit, integration, and e2e tests using pytest. Use this skill when writing or reviewing tests, setting up test fixtures, or validating test structure.
version: 1.0.0
provider: fastpaip
---

# Testing Conventions

FastPaip testing standards following the test pyramid: many fast unit tests, some integration tests, few e2e tests.

## Quick Reference

### Test Structure
```
packages/<bc>/tests/
├── unit/           # Fast, no I/O, mocked dependencies
├── integration/    # Real DB, repository tests
└── e2e/           # Complete flows
```

### Naming Pattern
```python
def test_<action>_<condition>():
    """Clear description of what's being tested."""
```

### Test Pyramid Priority
1. **Unit tests** (most) - Domain/application logic, fully mocked
2. **Integration tests** (some) - Repositories, infrastructure with real DB
3. **E2E tests** (few) - Complete user flows

## When to Use Each Type

**Unit**: Testing business logic, domain models, application services
- Fast (< 1s each)
- No external dependencies
- Mock all collaborators

**Integration**: Testing persistence, queries, infrastructure adapters
- Real database (test instance)
- Verify DB operations
- Test component interactions

**E2E**: Testing complete user scenarios
- Full system behavior
- Realistic/real services
- Run less frequently

## Fixtures Pattern

```python
# Use name parameter with _fixture suffix
# Organize with section comment headers

# ==============================================================================
# Mock Fixtures
# ==============================================================================

@pytest.fixture(name="mock_classifier")
def _mock_classifier_fixture() -> MagicMock:
    """Mock LLM classifier."""
    return MagicMock(spec=LLMClassifier)

# ==============================================================================
# Database Fixtures
# ==============================================================================

@pytest.fixture(name="session")
def _session_fixture(engine):
    """Database session."""
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s

# ==============================================================================
# Factory Fixtures
# ==============================================================================

@pytest.fixture(name="article_factory")
def _article_factory_fixture():
    """Factory for creating articles."""
    def create(**kwargs):
        return Article(**kwargs)
    return create
```

## Factory-Boy Patterns for Repositories

### When to Use Factory-Boy vs Simple Factories

**Use factory-boy for:**
- Database model testing (SQLAlchemy/SQLModel)
- Integration tests with relationships
- Complex domain models with multiple variations
- Batch test data creation

**Use simple factories for:**
- Unit tests (no database)
- Simple DTOs without persistence logic

### Factory Class Structure

Create TWO factory classes per domain model:

**1. DB Model Factory (SQLAlchemyModelFactory)**
- Persists to database
- Use in integration tests for test data setup
- Use `.create()` or `.create_batch(n)`

**2. DTO Factory (factory.Factory)**
- No persistence
- Use in tests for input data
- Use `.build()` or `.build_batch(n)`

```python
import factory
from factory.alchemy import SQLAlchemyModelFactory

# DB Model Factory (persists to database)
class DBCallFactory(SQLAlchemyModelFactory):
    """Factory for creating DBCall database model instances."""

    class Meta:
        model = DBCall
        sqlalchemy_session_persistence = "flush"

    call_location = factory.Sequence(
        lambda n: f"s3://test-bucket/recordings/call-{n:04d}.wav"
    )

# DTO Factory (no persistence)
class CallDTOFactory(factory.Factory):
    """Factory for creating Call DTO instances."""

    class Meta:
        model = Call

    id = None  # DTOs for create operations don't have IDs yet
    call_location = factory.Sequence(
        lambda n: f"s3://test-bucket/recordings/call-{n:04d}.wav"
    )
```

### Factory Session Configuration

Factory-boy factories need to be configured with the test session. Create an `autouse` fixture in conftest:

```python
# src/tests/integration/domains/<domain>/infrastructure/repositories/conftest.py

import pytest
from tests.integration.domains.<domain>.infrastructure.repositories.factories import (
    DBCallFactory,
)

@pytest.fixture(scope="function", autouse=True)
def _configure_factories(session):
    """Automatically configure factory-boy factories with the test session."""
    DBCallFactory._meta.sqlalchemy_session = session
```

## Repository Integration Test Patterns

### Pattern 1: Testing Create Operations (DTO → DB)

Use DTO factories to create input, test repository's create method:

```python
from tests.integration.domains.call_acknowledge.infrastructure.repositories.factories import (
    CallDTOFactory,
)

class TestRepositoryCreate:
    """Tests for the create method DTO-to-DB mapping."""

    def test_creates_and_returns_with_id(self, session: Session):
        # Arrange - Use DTO factory (no DB yet)
        repository = CallAcknowledgementRepository(session)
        input_data = CallDTOFactory.build()

        # Act
        result = repository.create(data=input_data)

        # Assert
        assert result.id is not None
        assert result.call_location == input_data.call_location
```

### Pattern 2: Testing Read Operations (DB → DTO)

Use DB factories to set up database state, test repository's get methods:

```python
from tests.integration.domains.call_acknowledge.infrastructure.repositories.factories import (
    DBCallFactory,
)

class TestRepositoryGetById:
    """Tests for the get_by_id method DB-to-DTO mapping."""

    def test_retrieves_by_id(self, session: Session):
        # Arrange - Use DB factory (persists to DB)
        repository = CallAcknowledgementRepository(session)
        db_record = DBCallFactory.create()
        session.flush()  # CRITICAL: Generate ID

        # Act
        result = repository.get_by_id(db_record.id)

        # Assert
        assert result.id == db_record.id
        assert result.call_location == db_record.call_location
```

### Pattern 3: Testing Query Operations

Use DB factories with `.create_batch()` for multiple records:

```python
class TestRepositoryQuery:
    """Tests for query methods."""

    def test_filters_correctly(self, session: Session):
        # Arrange - Create multiple records
        repository = CallAcknowledgementRepository(session)
        db_calls = DBCallFactory.create_batch(3)
        session.flush()

        # Act
        results = repository.find_all()

        # Assert
        assert len(results) == 3
```

## Quick Decision Trees

### Which Factory to Use?

```
Are you testing a repository CREATE method?
├─ Yes → Use DTOFactory.build()
│         (Creates input data, not in DB yet)
└─ No → Are you testing READ/UPDATE/DELETE?
         └─ Yes → Use DBFactory.create() + session.flush()
                  (Sets up DB state for testing)

Are you writing a unit test?
├─ Yes → Use simple sample data fixtures
│         (No database, no factory-boy)
└─ No → Use factory-boy patterns above
```

### When to Call session.flush()?

```
Did you use DBFactory.create()?
├─ Yes → Call session.flush() immediately after
│         (Generates IDs and persists to DB)
└─ No → You used DTOFactory.build()
         └─ Don't call flush (no DB interaction yet)
```

### Factory Method Choice

| Scenario | Factory Type | Method | Example |
|----------|--------------|--------|---------|
| Create input for repo.create() | DTO Factory | `.build()` | `CallDTOFactory.build()` |
| Setup DB for repo.get_by_id() | DB Factory | `.create()` | `DBCallFactory.create()` |
| Create multiple test records | DB Factory | `.create_batch(n)` | `DBCallFactory.create_batch(3)` |
| Create multiple input DTOs | DTO Factory | `.build_batch(n)` | `CallDTOFactory.build_batch(3)` |
| Unit test with no DB | DTO Factory | `.build()` | `CallDTOFactory.build()` |

## AAA Pattern (Arrange-Act-Assert)

```python
def test_classifies_article():
    # Arrange
    article = Article(content="test")
    mock_classifier = MagicMock()

    # Act
    result = classify(article, mock_classifier)

    # Assert
    assert result.category is not None
    mock_classifier.classify.assert_called_once()
```

## Database Schema Validation Tests

Validate database schema matches application expectations after migrations:

**Location**: `tests/integration/infrastructure/persistence/test_migrations.py`

**Purpose**: Catch schema mismatches early - verify all tables, columns, and constraints exist as code expects.

### Pattern

```python
def test_table_name_exists_with_correct_schema(self, db_engine, tables):
    """Verify table_name has all required columns and constraints."""
    inspector = inspect(db_engine)

    assert inspector.has_table("table_name")

    # Verify columns
    columns = {col["name"] for col in inspector.get_columns("table_name")}
    assert "required_column" in columns
    assert "another_column" in columns

    # Verify unique constraint
    unique_constraints = inspector.get_unique_constraints("table_name")
    has_constraint = any(
        "column_name" in constraint["column_names"]
        for constraint in unique_constraints
    )
    assert has_constraint
```

### Setup

Models must be imported in `tests/integration/conftest.py`:
```python
from transcriptions_system.domains.context.domain.models.db_model import DBModel
```

This ensures SQLModel.metadata knows about all tables for schema creation.

## Mocking Real Models - CRITICAL Best Practice

### ❌ DON'T Mock Pydantic Models or Dataclasses

Never create fake mock objects for real domain models like `Event`, `Call`, DTOs, or other Pydantic models:

```python
# ❌ BAD - Creates incomplete mock that lacks methods
mock_event = type("Event", (), {"type": EventType.NEW_CALL, "data": None})()
result = is_new_call_event(mock_event)  # Causes AttributeError: 'Event' has no 'get_data_as'

# ❌ BAD - Missing required methods
mock_call = type("Call", (), {"call_location": None, "ani": "+123"})()
data = mock_call.get_data_as(Call)  # AttributeError!
```

**Why it's wrong:**
- Mock objects created with `type()` only have the attributes in the dict
- They lack **all methods** from the real class (`get_data_as()`, model validators, etc.)
- Production code that calls these methods will fail with `AttributeError`
- Tests don't mirror real behavior

### ✅ DO Use Real Models Even with Invalid Data

Always use the actual Pydantic/dataclass models. They accept `Any` data in their fields:

```python
# ✅ GOOD - Use real Event class
event = Event(type=EventType.NEW_CALL, data=None)
result = is_new_call_event(event)  # Works! Has all methods

# ✅ GOOD - Use dict to simulate invalid data
invalid_event = Event(
    type=EventType.NEW_CALL,
    data={"call_location": None, "ani": "+123"}  # Invalid: ani is None
)
result = is_new_call_event(invalid_event)

# ✅ GOOD - Use real models in collections
events = [
    Event(type=EventType.NEW_CALL, data=Call(...)),
    Event(type=EventType.NEW_CALL, data=None),  # Test None case
    Event(type=EventType.NEW_CALL, data={"call_location": None, ...}),  # Test invalid
]
```

### Pattern: Testing Edge Cases with Real Models

```python
def test_guard_handles_none_data():
    """Guard should return False for None event data."""
    # ✅ Use real Event with None
    event = Event(type=EventType.NEW_CALL, data=None)
    assert is_new_call_event(event) is False

def test_guard_handles_missing_fields():
    """Guard should return False when required fields are None."""
    # ✅ Use dict to simulate invalid model data
    invalid_data = {
        "call_location": "s3://...",
        "ani": None  # Missing required field
    }
    event = Event(type=EventType.NEW_CALL, data=invalid_data)
    assert is_new_call_event(event) is False

def test_guard_handles_wrong_type():
    """Guard should return False for wrong event type."""
    # ✅ Use real model with correct type
    call = Call(call_location="s3://...", ani="+123")
    event = Event(type=EventType.NEW_CALL_ACKNOWLEDGEMENT, data=call)
    assert is_new_call_event(event) is False
```

### When You Really Need a Mock

If you need to mock dependencies (repositories, services), use `MagicMock` with `spec`:

```python
# ✅ GOOD - Mock external service, NOT domain models
mock_repository = MagicMock(spec=CallRepository)
mock_repository.get_by_id.return_value = Call(...)

# ✅ GOOD - Use real Event, mock the repository it depends on
event = Event(type=EventType.NEW_CALL, data=call)
result = process_event(event, mock_repository)
```

## Common Mistakes to Avoid

- ❌ Testing framework behavior
- ❌ Testing private methods directly
- ❌ Testing trivial code (getters/setters)
- ❌ Fixtures with too many dependencies
- ❌ Tests that depend on execution order
- ❌ Shared mutable state between tests
- ❌ **Mocking real domain models with `type()` factory** - Use real models instead!

## Running Tests

```bash
# All tests
uv run pytest

# By type
uv run pytest packages/*/tests/unit/
uv run pytest packages/*/tests/integration/

# Specific BC
uv run pytest packages/core/tests/

# With coverage
uv run pytest --cov=packages --cov-report=term-missing

# Fast feedback
uv run pytest -x  # Stop on first failure
```

## Using the Scripts

### Generate Test Scaffold
```bash
python scripts/generate_test.py \
  --type unit \
  --module core \
  --subject ArticleService \
  --output packages/core/tests/unit/domain/test_article_service.py
```

### Validate Test Structure
```bash
python scripts/validate_tests.py packages/core/tests/
```

## Auto-Generating Repository Tests

When you ask me to create repository tests, I will automatically:

1. **Discover domain models** in `src/transcriptions_system/domains/<domain>/domain/models/`
2. **Generate factories.py** with:
   - DB factory for each database model (SQLAlchemyModelFactory)
   - DTO factory for each DTO (factory.Factory)
   - Appropriate Sequence/Iterator/Faker fields based on field types
3. **Generate/update conftest.py** with:
   - autouse fixture for factory session configuration
   - Automatic import and configuration of all DB factories
4. **Create test file skeleton** with:
   - Test classes for each repository method
   - Proper factory usage (DTO for create, DB for get/query)
   - AAA pattern with inline comments
   - Edge case tests (None returns, empty results, filtering)

### How to Trigger Auto-Generation

```bash
# Example user requests that trigger auto-generation:
"Create integration tests for CallClassificationRepository"
"Generate factories for call_acknowledge domain"
"Set up repository tests for the contact_acknowledge module"
```

### What Gets Generated

**factories.py structure:**
```python
import factory
from factory.alchemy import SQLAlchemyModelFactory

# DB Factory (persists to database)
class DB<Model>Factory(SQLAlchemyModelFactory):
    class Meta:
        model = DB<Model>
        sqlalchemy_session_persistence = "flush"
    # Auto-detected fields with Sequence/Iterator/Faker

# DTO Factory (in-memory only)
class <DTO>Factory(factory.Factory):
    class Meta:
        model = <DTO>
    id = None
    # Same fields as DB factory
```

**conftest.py structure:**
```python
import pytest
from tests.integration.domains.<domain>.infrastructure.repositories.factories import (
    DB<Model>Factory,  # All DB factories imported
)

@pytest.fixture(scope="function", autouse=True)
def _configure_factories(session):
    """Auto-configure factories with session."""
    DB<Model>Factory._meta.sqlalchemy_session = session
    # One line per DB factory
```

**test file structure:**
```python
class Test<Repository>Create:
    """Tests for create method."""
    def test_<scenario>(self, session):
        # Arrange - Use DTOFactory.build()
        # Act - Call repository.create()
        # Assert - Verify mapping
        pass

class Test<Repository>GetById:
    """Tests for get_by_id method."""
    def test_<scenario>(self, session):
        # Arrange - Use DBFactory.create() + session.flush()
        # Act - Call repository.get_by_id()
        # Assert - Verify retrieval
        pass
```

## Detailed Patterns

For comprehensive examples and patterns, see:
- `references/factory-boy.md` - Complete factory-boy integration guide
- `references/repository-tests.md` - Repository testing patterns and philosophy
- `references/patterns.md` - Detailed test patterns with examples
- `references/fixtures.md` - Fixture patterns and best practices
- `references/workflows.md` - Step-by-step workflow examples
- Original: `/docs/testing-conventions.md`

### Template Files

The skill uses these templates for code generation:
- `templates/factories_template.py` - Factory generation template
- `templates/conftest_template.py` - Conftest generation template

## Key Principle

**Test behavior, not implementation.** Focus on what the code does (outcomes), not how it does it (internals).
