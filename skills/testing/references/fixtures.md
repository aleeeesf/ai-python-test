# Fixture Patterns Reference

Comprehensive guide to fixture patterns in FastPaip tests. Load this when working extensively with pytest fixtures.

## FastPaip Fixture Convention

**FastPaip uses a specific naming pattern for fixtures:**

```python
@pytest.fixture(name="fixture_name")
def _fixture_name_fixture():
    """Docstring describing the fixture."""
    pass
```

**Benefits of this pattern:**
- **Clean test signatures**: Use `fixture_name` in tests, not `_fixture_name_fixture`
- **Private implementation**: The `_` prefix indicates implementation detail
- **Searchable**: All fixtures end with `_fixture` suffix
- **Organized**: Group with section comment headers

**Example from the codebase:**

```python
# ==============================================================================
# Repository Fixtures
# ==============================================================================

@pytest.fixture(name="classification_repo")
def _classification_repo_fixture() -> MagicMock:
    """
    Provides a mock ClassificationRepository.
    
    Configured to return an ArticleClassificationEntity when create() is called.
    """
    mock = MagicMock()
    mock.create.return_value = ArticleClassificationEntity(
        id=1,
        article_id=1,
        article_category=ArticleCategoryEnum.OTHER,
    )
    return mock


# ==============================================================================
# Sample Data Fixtures
# ==============================================================================

@pytest.fixture(name="sample_acknowledged_article")
def _sample_acknowledged_article_fixture() -> Article:
    """Provides a sample acknowledged Article (has ID, no classification)."""
    return Article(
        id=1,
        raw_article="This is an acknowledged article awaiting classification.",
    )
```

## Fixture Scopes

```python
# Function scope (default) - new instance per test
@pytest.fixture(name="session")
def _session_fixture():
    """New session for each test."""
    return Session()

# Class scope - shared within test class
@pytest.fixture(name="shared_resource", scope="class")
def _shared_resource_fixture():
    """Shared within TestClass."""
    return ExpensiveResource()

# Module scope - shared within module
@pytest.fixture(name="module_resource", scope="module")
def _module_resource_fixture():
    """Shared for all tests in file."""
    setup()
    yield resource
    teardown()

# Session scope - shared across entire test run
@pytest.fixture(name="engine", scope="session")
def _engine_fixture():
    """One engine for entire test session."""
    return create_engine("sqlite:///:memory:")
```

## Mock Fixtures

### Basic Mock

```python
@pytest.fixture(name="mock_classifier")
def _mock_classifier_fixture() -> MagicMock:
    """Mock LLM classifier."""
    return MagicMock(spec=LLMClassifier)
```

### Pre-configured Mock

```python
@pytest.fixture(name="mock_classifier")
def _mock_classifier_fixture() -> MagicMock:
    """Mock classifier with default behavior."""
    mock = MagicMock(spec=LLMClassifier)
    mock.classify.return_value = "TECHNOLOGY"
    mock.get_confidence.return_value = 0.95
    return mock
```

### Mock with Side Effects

```python
@pytest.fixture(name="mock_api_client")
def _mock_api_client_fixture() -> MagicMock:
    """Mock API client with different responses."""
    mock = MagicMock()
    mock.fetch.side_effect = [
        {"status": "success", "data": "first"},
        {"status": "success", "data": "second"},
        {"status": "error", "message": "failed"},
    ]
    return mock
```

## Database Fixtures

### Basic Session

```python
@pytest.fixture
def session(engine):
    """Database session with automatic rollback."""
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
        session.rollback()
```

### Session with Seeded Data

```python
@pytest.fixture
def session_with_articles(session):
    """Session pre-populated with test articles."""
    articles = [
        Article(title="Article 1", category="TECH"),
        Article(title="Article 2", category="SCIENCE"),
        Article(title="Article 3", category="TECH"),
    ]
    for article in articles:
        session.add(article)
    session.commit()
    return session
```

### Transaction Fixture

```python
@pytest.fixture
def transaction(session):
    """Nested transaction for test isolation."""
    trans = session.begin_nested()
    yield session
    trans.rollback()
```

## Factory Fixtures

### Basic Factory

```python
@pytest.fixture
def article_factory():
    """Factory for creating articles."""
    def create(**kwargs):
        defaults = {
            "title": "Test Article",
            "content": "Test content",
            "category": "TECHNOLOGY",
        }
        defaults.update(kwargs)
        return Article(**defaults)
    return create


def test_with_custom_article(article_factory):
    tech_article = article_factory(category="TECHNOLOGY")
    science_article = article_factory(category="SCIENCE")
    # Use articles in test...
```

### Factory with Session

```python
@pytest.fixture
def article_factory(session):
    """Factory that persists articles to database."""
    def create(**kwargs):
        article = Article(**kwargs)
        session.add(article)
        session.commit()
        session.refresh(article)
        return article
    return create


def test_with_persisted_articles(article_factory):
    article = article_factory(title="Saved Article")
    assert article.id is not None
```

### Sequence Factory

```python
@pytest.fixture
def article_factory():
    """Factory with auto-incrementing values."""
    counter = {"value": 0}
    
    def create(**kwargs):
        counter["value"] += 1
        defaults = {
            "title": f"Article {counter['value']}",
            "url": f"https://example.com/{counter['value']}"
        }
        defaults.update(kwargs)
        return Article(**defaults)
    
    return create
```

## Composed Fixtures

### Layered Fixtures

```python
@pytest.fixture
def engine():
    """Base: database engine."""
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def session(engine):
    """Layer 1: session from engine."""
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def repository(session):
    """Layer 2: repository using session."""
    return SQLArticleRepository(session)


@pytest.fixture
def service(repository, mock_classifier):
    """Layer 3: service using repository and classifier."""
    return ArticleService(
        repository=repository,
        classifier=mock_classifier,
    )
```

### Combined Fixture

```python
@pytest.fixture
def classification_pipeline(mock_classifier, mock_repository, mock_queue):
    """Pre-wired classification pipeline."""
    return ClassificationPipeline(
        classifier=mock_classifier,
        repository=mock_repository,
        queue=mock_queue,
    )
```

## Parametrized Fixtures

### Simple Parametrization

```python
@pytest.fixture(params=["TECHNOLOGY", "SCIENCE", "HEALTH"])
def category(request):
    """Run test with each category."""
    return request.param


def test_handles_all_categories(category):
    """Test runs 3 times with different categories."""
    article = Article(category=category)
    assert article.category in ["TECHNOLOGY", "SCIENCE", "HEALTH"]
```

### Complex Parametrization

```python
@pytest.fixture(params=[
    ("TECHNOLOGY", 0.95),
    ("SCIENCE", 0.87),
    ("HEALTH", 0.92),
])
def classification_data(request):
    """Category and confidence pairs."""
    category, confidence = request.param
    return Classification(category=category, confidence=confidence)
```

## Async Fixtures

```python
@pytest.fixture
async def async_client():
    """Async HTTP client."""
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture
async def async_session():
    """Async database session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with AsyncSession(engine) as session:
        yield session
```

## Cleanup Fixtures

### Yield Fixture

```python
@pytest.fixture
def temp_file():
    """Create and cleanup temporary file."""
    file_path = Path("/tmp/test_file.txt")
    file_path.write_text("test content")
    
    yield file_path
    
    # Cleanup
    if file_path.exists():
        file_path.unlink()
```

### Context Manager Fixture

```python
@pytest.fixture
def mock_environment():
    """Temporarily modify environment."""
    old_value = os.environ.get("API_KEY")
    os.environ["API_KEY"] = "test-key"
    
    yield
    
    # Restore
    if old_value:
        os.environ["API_KEY"] = old_value
    else:
        del os.environ["API_KEY"]
```

## Conditional Fixtures

### Skip if Missing

```python
@pytest.fixture
def openai_client():
    """OpenAI client - skip if no API key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)
```

### Platform-Specific

```python
@pytest.fixture
def platform_specific_tool():
    """Tool available only on certain platforms."""
    if sys.platform != "linux":
        pytest.skip("Linux only")
    return LinuxTool()
```

## Fixture Dependencies

### Multiple Dependencies

```python
@pytest.fixture
def article(session):
    """Article stored in database."""
    article = Article(title="Test")
    session.add(article)
    session.commit()
    return article


@pytest.fixture
def classification(session, article):
    """Classification for article."""
    classification = Classification(
        article_id=article.id,
        category="TECHNOLOGY"
    )
    session.add(classification)
    session.commit()
    return classification


def test_article_with_classification(article, classification):
    """Test uses both fixtures."""
    assert classification.article_id == article.id
```

## Conftest Organization

### Package-Level conftest.py

```python
# packages/core/tests/conftest.py
"""Shared fixtures for core package tests."""

import pytest

@pytest.fixture
def mock_core_service():
    """Mock core service for all core tests."""
    return MagicMock()
```

### Type-Level conftest.py

```python
# packages/core/tests/unit/conftest.py
"""Fixtures specific to unit tests."""

import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_repository():
    """Mock repository (unit tests only)."""
    return MagicMock()


# packages/core/tests/integration/conftest.py
"""Fixtures specific to integration tests."""

import pytest
from sqlmodel import Session, create_engine

@pytest.fixture
def session():
    """Real database session (integration tests only)."""
    engine = create_engine("sqlite:///:memory:")
    with Session(engine) as session:
        yield session
```

## Best Practices

### ✅ Do This

```python
# Single responsibility
@pytest.fixture
def mock_classifier():
    return MagicMock(spec=LLMClassifier)

# Clear names
@pytest.fixture
def article_with_classification():
    return create_article_with_classification()

# Minimal scope
@pytest.fixture  # Function scope by default
def session():
    return Session()
```

### ❌ Don't Do This

```python
# Too many responsibilities
@pytest.fixture
def everything():
    # Sets up database
    # Creates mocks
    # Configures environment
    # Returns complex tuple
    return (db, mock1, mock2, mock3, config)

# Unclear names
@pytest.fixture
def data():  # What data?
    return something

# Overly broad scope
@pytest.fixture(scope="session")  # Rarely needed
def shared_state():
    return MutableState()  # Dangerous!
```

## Troubleshooting

### Fixture Not Found

```python
# Problem: fixture defined in wrong conftest.py
# Solution: Move to appropriate conftest.py or import explicitly

from .conftest import my_fixture  # Explicit import
```

### Fixture Ordering Issues

```python
# Problem: fixtures run in wrong order
# Solution: Use explicit dependencies

@pytest.fixture
def dependent(prerequisite):  # prerequisite runs first
    return use(prerequisite)
```

### Fixture Scope Issues

```python
# Problem: function fixture can't use session fixture
# Solution: Match or broaden scope

@pytest.fixture(scope="session")
def narrow():  # Can't use session-scoped fixtures
    return value

# Fix:
@pytest.fixture(scope="session")
def broader():
    return value
```

## Quick Reference

| Pattern | Use Case |
|---------|----------|
| Mock fixture | External dependencies in unit tests |
| Session fixture | Database access in integration tests |
| Factory fixture | Need variations of test data |
| Parametrized fixture | Run test with multiple inputs |
| Composed fixture | Complex setup from simpler pieces |
| Yield fixture | Need cleanup after test |
| Module scope | Expensive setup (database, files) |
| Session scope | Very expensive setup (rarely needed) |
