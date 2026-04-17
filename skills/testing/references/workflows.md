# Testing Conventions - Example Workflows

Real-world examples of using the testing-conventions skill.

## Workflow 1: Writing a New Feature with Tests

### Scenario
You're adding a new `ArticleSummaryService` to the `core` BC that generates summaries using an LLM.

### Step 1: Generate Unit Test Scaffold

```bash
python scripts/generate_test.py \
  --type unit \
  --module core \
  --subject ArticleSummaryService \
  --output ../../../packages/core/tests/unit/application/test_article_summary_service.py
```

### Step 2: Fill in Test Cases

```python
"""
Unit tests for ArticleSummaryService.

Tests domain/application logic with mocked dependencies.
"""

import pytest
from unittest.mock import MagicMock

from core.application.services import ArticleSummaryService
from core.domain.models import Article, Summary


class TestArticleSummaryService:
    """Test suite for ArticleSummaryService."""

    def test_generates_summary_when_valid_article(
        self,
        mock_llm_client: MagicMock,
        sample_article: Article,
    ):
        """Test that service generates summary for valid article."""
        # Arrange
        mock_llm_client.generate.return_value = "This is a summary"
        service = ArticleSummaryService(llm_client=mock_llm_client)
        
        # Act
        summary = service.generate_summary(sample_article)
        
        # Assert
        assert summary.text == "This is a summary"
        assert summary.article_id == sample_article.id
        mock_llm_client.generate.assert_called_once()

    def test_raises_error_when_invalid_article(
        self,
        mock_llm_client: MagicMock,
    ):
        """Test that service validates article input."""
        # Arrange
        service = ArticleSummaryService(llm_client=mock_llm_client)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Article content required"):
            service.generate_summary(None)


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLM client."""
    return MagicMock()


@pytest.fixture
def sample_article():
    """Sample article for testing."""
    return Article(
        id=1,
        content="Long article content here...",
        title="Test Article"
    )
```

### Step 3: Generate Integration Test

```bash
python scripts/generate_test.py \
  --type integration \
  --module core \
  --subject SummaryRepository \
  --output ../../../packages/core/tests/integration/infrastructure/test_summary_repository.py
```

### Step 4: Implement and Test

```bash
# Run the specific test
uv run pytest packages/core/tests/unit/application/test_article_summary_service.py -v

# Run with coverage
uv run pytest packages/core/tests/unit/application/test_article_summary_service.py --cov=core.application.services
```

---

## Workflow 2: Validating Existing Tests

### Scenario
You've inherited some tests and want to ensure they follow conventions.

### Step 1: Validate Directory

```bash
python scripts/validate_tests.py packages/core/tests/
```

### Step 2: Review Issues

```
⚠️  2 warning(s) found:

  packages/core/tests/unit/test_helper.py:15
    Test function missing docstring: test_calculates_score

  packages/core/tests/unit/test_service.py:42
    Avoid time.sleep in tests (found in test_processes_async). Use proper synchronization.

Summary: 0 errors, 2 warnings
```

### Step 3: Fix Issues

```python
# Before
def test_calculates_score():
    assert calculate_score(5, 10) == 0.5

# After
def test_calculates_score():
    """Test score calculation with valid inputs."""
    assert calculate_score(5, 10) == 0.5
```

### Step 4: Re-validate (Strict Mode)

```bash
python scripts/validate_tests.py --strict packages/core/tests/
```

---

## Workflow 3: Test-Driven Development (TDD)

### Scenario
Building a new `ArticleFilter` class using TDD.

### Step 1: Generate Test Scaffold

```bash
python scripts/generate_test.py \
  --type unit \
  --module core \
  --subject ArticleFilter \
  --output ../../../packages/core/tests/unit/domain/test_article_filter.py
```

### Step 2: Write First Test (Red)

```python
def test_filters_by_category_when_category_matches():
    """Test filter returns articles matching category."""
    # Arrange
    articles = [
        Article(id=1, category="TECH"),
        Article(id=2, category="SCIENCE"),
        Article(id=3, category="TECH"),
    ]
    filter = ArticleFilter(category="TECH")
    
    # Act
    result = filter.apply(articles)
    
    # Assert
    assert len(result) == 2
    assert all(a.category == "TECH" for a in result)
```

### Step 3: Run Test (Fails)

```bash
uv run pytest packages/core/tests/unit/domain/test_article_filter.py -v
# FAILED - ImportError: cannot import name 'ArticleFilter'
```

### Step 4: Implement Minimal Code (Green)

```python
# packages/core/src/core/domain/filters.py
class ArticleFilter:
    def __init__(self, category: str):
        self.category = category
    
    def apply(self, articles: list) -> list:
        return [a for a in articles if a.category == self.category]
```

### Step 5: Run Test (Passes)

```bash
uv run pytest packages/core/tests/unit/domain/test_article_filter.py -v
# PASSED
```

### Step 6: Add More Tests and Refactor

```python
def test_filters_by_date_range_when_dates_match():
    """Test filter returns articles within date range."""
    # Implementation...

def test_filters_return_empty_when_no_matches():
    """Test filter returns empty list when no articles match."""
    # Implementation...
```

---

## Workflow 4: Refactoring with Test Safety

### Scenario
Refactoring `ClassificationService` with existing test coverage.

### Step 1: Run Existing Tests

```bash
uv run pytest packages/classifications/tests/unit/ -v
# All tests pass
```

### Step 2: Validate Test Quality

```bash
python scripts/validate_tests.py packages/classifications/tests/
# ✅ All tests pass validation!
```

### Step 3: Check Coverage

```bash
uv run pytest packages/classifications/tests/ --cov=classifications --cov-report=term-missing
```

```
Name                                    Stmts   Miss  Cover   Missing
---------------------------------------------------------------------
classifications/application/service.py     42      2    95%   78-79
classifications/domain/models.py           15      0   100%
classifications/infrastructure/repo.py     28      3    89%   45-47
---------------------------------------------------------------------
TOTAL                                      85      5    94%
```

### Step 4: Add Missing Tests

Based on coverage report, lines 78-79 in service.py are not covered:

```python
def test_handles_network_timeout():
    """Test service handles network timeout gracefully."""
    # Test the missing error handling path
```

### Step 5: Refactor with Confidence

Now with full coverage, refactor knowing tests will catch regressions:

```bash
# Make changes to implementation
# Run tests after each change
uv run pytest packages/classifications/tests/unit/ -x  # Stop on first failure
```

---

## Workflow 5: Testing Event-Driven Flow (E2E)

### Scenario
Testing complete flow: Article → Acknowledgement → Classification → Summary

### Step 1: Generate E2E Test

```bash
python scripts/generate_test.py \
  --type e2e \
  --module core \
  --subject ArticleProcessingFlow \
  --output ../../../packages/core/tests/e2e/test_article_processing_flow.py
```

### Step 2: Implement Complete Flow Test

```python
def test_complete_article_processing_flow(
    api_client: TestClient,
    worker: BackgroundWorker,
    db_session: Session,
):
    """Test article flows through entire pipeline."""
    # Arrange - Submit article
    response = api_client.post("/articles", json={
        "raw_article": "AI breakthrough in healthcare",
        "url": "https://example.com/article"
    })
    assert response.status_code == 201
    article_id = response.json()["id"]
    
    # Act - Process acknowledgement event
    worker.process_event_type("ACKNOWLEDGEMENT")
    
    # Assert - Article acknowledged
    article = db_session.get(Article, article_id)
    assert article.status == "acknowledged"
    
    # Act - Process classification event
    worker.process_event_type("CLASSIFICATION")
    
    # Assert - Article classified
    db_session.refresh(article)
    assert article.classification is not None
    assert article.classification.category in VALID_CATEGORIES
    
    # Act - Process summary event
    worker.process_event_type("SUMMARY")
    
    # Assert - Summary generated
    db_session.refresh(article)
    assert article.summary is not None
    assert len(article.summary.text) > 0
```

### Step 3: Run E2E Suite

```bash
# Run all e2e tests
uv run pytest packages/*/tests/e2e/ -v --tb=short

# Run with slower timeout for external services
uv run pytest packages/*/tests/e2e/ -v --timeout=30
```

---

## Workflow 6: Creating Repository Integration Tests with Factory-Boy

### Scenario
You're implementing a new repository (`SQLCallClassificationRepository`) and need to create comprehensive integration tests using factory-boy.

### Step 1: Identify Domain Models

First, identify the domain models you need to test:

```bash
# List domain models
ls src/transcriptions_system/domains/call_classification/domain/models/
# Output: db_call_classification.py, enums.py
```

### Step 2: Create Factories

Create `factories.py` with both DB and DTO factories:

```python
# src/tests/integration/domains/call_classification/infrastructure/repositories/factories.py

"""
Test data factories for call_classification domain integration tests.

Uses factory-boy to create database objects and DTOs for testing.
Provides sensible defaults while allowing test-specific overrides.
"""

import factory
from factory.alchemy import SQLAlchemyModelFactory

from transcriptions_system.domains.call_classification.domain.models.db_call_classification import (
    DBCallClassification,
)
from transcriptions_system.domains.call_classification.domain.models.enums import (
    CallCategoryEnum,
)
from transcriptions_system.domains.call_classification.application.dtos import (
    CallClassification,
)


class DBCallClassificationFactory(SQLAlchemyModelFactory):
    """Factory for creating DBCallClassification database model instances.

    Usage:
        # Create with defaults
        db_classification = DBCallClassificationFactory.create()

        # Override specific fields
        db_classification = DBCallClassificationFactory.create(
            call_category=CallCategoryEnum.RECLAMO
        )

        # Create multiple
        db_classifications = DBCallClassificationFactory.create_batch(3)
    """

    class Meta:
        model = DBCallClassification
        sqlalchemy_session_persistence = "flush"

    call_id = factory.Sequence(lambda n: n)
    call_category = factory.Iterator(CallCategoryEnum)  # Cycle through enum values


class CallClassificationDTOFactory(factory.Factory):
    """Factory for creating CallClassification DTO instances.

    Usage:
        # Create DTO without ID (for create operations)
        dto = CallClassificationDTOFactory.build()

        # Create DTO with ID (simulating retrieved data)
        dto = CallClassificationDTOFactory.build(id=123)

        # Override category
        dto = CallClassificationDTOFactory.build(
            call_category=CallCategoryEnum.INFORMACION
        )
    """

    class Meta:
        model = CallClassification

    id = None  # DTOs for create operations don't have IDs yet
    call_id = factory.Sequence(lambda n: n)
    call_category = CallCategoryEnum.INFORMACION
```

### Step 3: Create Conftest with Factory Configuration

Create `conftest.py` to auto-configure factories with the session:

```python
# src/tests/integration/domains/call_classification/infrastructure/repositories/conftest.py

"""Integration test fixtures for call_classification domain.

Provides database fixtures with transaction isolation for testing
infrastructure adapters against a real PostgreSQL database.
"""

import pytest
from tests.integration.domains.call_classification.infrastructure.repositories.factories import (
    DBCallClassificationFactory,
)


@pytest.fixture(scope="function", autouse=True)
def _configure_factories(session):
    """
    Automatically configure factory-boy factories with the test session.

    This fixture runs automatically for all tests in this directory,
    ensuring factories use the correct test database session.
    """
    DBCallClassificationFactory._meta.sqlalchemy_session = session
```

### Step 4: Create Repository Tests

Create test file with proper factory usage patterns:

```python
# src/tests/integration/domains/call_classification/infrastructure/repositories/test_sql_call_classification_repository.py

"""Integration tests for SQLCallClassificationRepository.

Tests repository operations against a real PostgreSQL database instance.
Uses transaction rollback for test isolation.

Focus: Tests verify that OUR repository correctly:
1. Maps DTOs to database models and back
2. Implements the port contract correctly
3. Handles database-specific scenarios (ENUMs, edge cases)
"""

import pytest
from sqlmodel import Session

from transcriptions_system.domains.call_classification.application.dtos import CallClassification
from transcriptions_system.domains.call_classification.domain.models.enums import CallCategoryEnum
from transcriptions_system.domains.call_classification.infrastructure.repositories.sql_call_classification_repository import (
    SQLCallClassificationRepository,
)
from tests.integration.domains.call_classification.infrastructure.repositories.factories import (
    CallClassificationDTOFactory,
    DBCallClassificationFactory,
)


class TestSQLCallClassificationRepositoryCreate:
    """Tests for the create method DTO-to-DB mapping."""

    def test_maps_dto_to_database_and_returns_with_id(self, session: Session):
        """Should correctly map CallClassification DTO to DB model."""
        # Arrange - Use DTO factory for input
        repository = SQLCallClassificationRepository(session)
        classification_data = CallClassificationDTOFactory.build()

        # Act
        created = repository.create(data=classification_data)

        # Assert - Verify DTO mapping
        assert isinstance(created, CallClassification)
        assert created.id is not None
        assert created.call_category == classification_data.call_category

    def test_handles_enum_values_correctly(self, session: Session):
        """Should correctly persist and retrieve enum values."""
        # Arrange
        repository = SQLCallClassificationRepository(session)
        classification_data = CallClassificationDTOFactory.build(
            call_category=CallCategoryEnum.RECLAMO
        )

        # Act
        created = repository.create(data=classification_data)

        # Assert - Enum preserved
        assert created.call_category == CallCategoryEnum.RECLAMO


class TestSQLCallClassificationRepositoryGetById:
    """Tests for the get_by_id method DB-to-DTO mapping."""

    def test_returns_none_for_nonexistent_id(self, session: Session):
        """Should return None when classification does not exist."""
        # Arrange
        repository = SQLCallClassificationRepository(session)

        # Act
        result = repository.get_by_id(id=99999)

        # Assert
        assert result is None

    def test_maps_database_record_to_dto_correctly(self, session: Session):
        """Should correctly map DB model to DTO."""
        # Arrange - Use DB factory to set up test data
        repository = SQLCallClassificationRepository(session)
        db_classification = DBCallClassificationFactory.create(
            call_category=CallCategoryEnum.INFORMACION
        )
        session.flush()  # CRITICAL: Generate ID

        # Act - Test ONLY get_by_id
        retrieved = repository.get_by_id(db_classification.id)

        # Assert
        assert isinstance(retrieved, CallClassification)
        assert retrieved.id == db_classification.id
        assert retrieved.call_category == CallCategoryEnum.INFORMACION


class TestSQLCallClassificationRepositoryFindByCategory:
    """Tests for the find_by_category query method."""

    def test_returns_empty_list_when_no_matches(self, session: Session):
        """Should return empty list when no calls match the category."""
        # Arrange
        repository = SQLCallClassificationRepository(session)
        DBCallClassificationFactory.create(call_category=CallCategoryEnum.INFORMACION)
        session.flush()

        # Act
        results = repository.find_by_category(CallCategoryEnum.RECLAMO)

        # Assert
        assert results == []

    def test_filters_by_category_correctly(self, session: Session):
        """Should return only calls matching the specified category."""
        # Arrange - Create test data with different categories
        repository = SQLCallClassificationRepository(session)
        info_classification = DBCallClassificationFactory.create(
            call_category=CallCategoryEnum.INFORMACION
        )
        reclamo_classification = DBCallClassificationFactory.create(
            call_category=CallCategoryEnum.RECLAMO
        )
        session.flush()

        # Act
        results = repository.find_by_category(CallCategoryEnum.INFORMACION)

        # Assert
        assert len(results) == 1
        assert results[0].id == info_classification.id
        assert results[0].call_category == CallCategoryEnum.INFORMACION

    def test_returns_all_matches_when_multiple_exist(self, session: Session):
        """Should return all calls matching the category."""
        # Arrange - Use create_batch for multiple records
        repository = SQLCallClassificationRepository(session)
        info_classifications = DBCallClassificationFactory.create_batch(
            3,
            call_category=CallCategoryEnum.INFORMACION
        )
        session.flush()

        # Act
        results = repository.find_by_category(CallCategoryEnum.INFORMACION)

        # Assert
        assert len(results) == 3
        result_ids = {r.id for r in results}
        expected_ids = {c.id for c in info_classifications}
        assert result_ids == expected_ids
```

### Step 5: Run Tests

```bash
# Ensure PostgreSQL is running
docker-compose up -d postgres

# Run the repository tests
pytest src/tests/integration/domains/call_classification/infrastructure/repositories/ -v

# Run with coverage
pytest src/tests/integration/domains/call_classification/ --cov=transcriptions_system.domains.call_classification.infrastructure
```

### Step 6: Verify Test Patterns

Check that your tests follow best practices:

```bash
# Validate test structure
python scripts/validate_tests.py src/tests/integration/domains/call_classification/

# Check for common issues:
# ✅ Factory session configured in conftest
# ✅ DTO factories used for create tests
# ✅ DB factories used for read tests
# ✅ session.flush() called after DB factory
# ✅ AAA pattern with comments
# ✅ One test class per repository method
```

### Key Patterns in This Workflow

1. **Two Factory Types**:
   - `DBCallClassificationFactory` for setting up test data
   - `CallClassificationDTOFactory` for input data

2. **Auto-Configure Session**:
   - `autouse=True` fixture in conftest
   - No manual factory configuration in tests

3. **Factory Selection**:
   - Testing `create()` → Use DTO factory
   - Testing `get_by_id()` → Use DB factory + flush
   - Testing queries → Use DB factory + flush

4. **Test Organization**:
   - One class per repository method
   - Clear docstrings
   - AAA pattern with comments

5. **Enum Handling**:
   - Use `factory.Iterator` for cycling through enum values
   - Override with specific enum when testing edge cases

### Common Mistakes to Avoid

```python
# ❌ Wrong: Using DB factory for create test input
def test_create(self, session):
    data = DBCallClassificationFactory.create()  # Already in DB!
    result = repository.create(data)

# ✅ Correct: Using DTO factory for create test input
def test_create(self, session):
    data = CallClassificationDTOFactory.build()  # Not in DB
    result = repository.create(data)

# ❌ Wrong: Forgetting session.flush()
def test_get_by_id(self, session):
    db_classification = DBCallClassificationFactory.create()
    result = repository.get_by_id(db_classification.id)  # ID is None!

# ✅ Correct: Always flush after DB factory
def test_get_by_id(self, session):
    db_classification = DBCallClassificationFactory.create()
    session.flush()  # Generate ID
    result = repository.get_by_id(db_classification.id)

# ❌ Wrong: Not configuring factory in conftest
# Tests will fail with "Session not bound" error

# ✅ Correct: Add factory to conftest autouse fixture
@pytest.fixture(scope="function", autouse=True)
def _configure_factories(session):
    DBCallClassificationFactory._meta.sqlalchemy_session = session
```

---

## Quick Tips

### Generate Multiple Test Files at Once

```bash
#!/bin/bash
# generate_test_suite.sh

TESTS=(
  "unit:core:ArticleService:packages/core/tests/unit/domain/test_article_service.py"
  "unit:core:ArticleRepository:packages/core/tests/unit/infrastructure/test_article_repository.py"
  "integration:core:ArticleRepository:packages/core/tests/integration/test_article_repository.py"
  "e2e:core:ArticleFlow:packages/core/tests/e2e/test_article_flow.py"
)

for test in "${TESTS[@]}"; do
  IFS=: read -r type module subject output <<< "$test"
  python scripts/generate_test.py \
    --type "$type" \
    --module "$module" \
    --subject "$subject" \
    --output "../../../$output"
done
```

### Pre-commit Hook for Test Validation

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Validate only staged test files
for file in $(git diff --cached --name-only | grep "test_.*\.py$"); do
  python .claude/skills/testing-conventions.skill/scripts/validate_tests.py "$file"
  if [ $? -ne 0 ]; then
    echo "❌ Test validation failed for $file"
    exit 1
  fi
done
```

### Watch Mode for TDD

```bash
# Install pytest-watch
uv pip install pytest-watch

# Run tests on file changes
uv run ptw packages/core/tests/unit/ -- -v --tb=short
```

### Coverage-Driven Testing

```bash
# Generate HTML coverage report
uv run pytest --cov=packages --cov-report=html

# Open in browser
open htmlcov/index.html

# Focus on untested code
uv run pytest --cov=packages --cov-report=term-missing | grep "0%"
```
