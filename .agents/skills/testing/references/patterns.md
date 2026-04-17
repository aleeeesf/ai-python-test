# Test Patterns Reference

Detailed examples of testing patterns used in FastPaip. Load this when you need specific examples or deep guidance on test implementation.

## Table of Contents

- [Unit Test Patterns](#unit-test-patterns)
- [Integration Test Patterns](#integration-test-patterns)
- [E2E Test Patterns](#e2e-test-patterns)
- [Fixture Patterns](#fixture-patterns)
- [Common Scenarios](#common-scenarios)

---

## Unit Test Patterns

### Domain Service Testing

Test business logic with all dependencies mocked:

```python
def test_classifies_article_when_acknowledgement_event(
    mock_classifier: MagicMock,
    mock_repository: MagicMock,
    sample_acknowledgement_event: Event,
):
    """Domain service correctly processes acknowledgement events."""
    # Arrange
    mock_classifier.classify.return_value = "TECHNOLOGY"
    mock_repository.create.return_value = Classification(
        article_id=123,
        category="TECHNOLOGY",
        confidence=0.95
    )
    
    # Act
    result = classification_service(
        sample_acknowledgement_event,
        classifier=mock_classifier,
        repository=mock_repository,
    )
    
    # Assert
    assert result.type == EventType.NEW_CLASSIFICATION
    assert result.data.category == "TECHNOLOGY"
    mock_classifier.classify.assert_called_once_with(
        sample_acknowledgement_event.data.raw_article
    )
    mock_repository.create.assert_called_once()
```

### Orchestration Testing

Test coordination of multiple dependencies:

```python
def test_coordinates_classification_pipeline(
    mock_fetcher: MagicMock,
    mock_classifier: MagicMock,
    mock_persister: MagicMock,
):
    """Application service orchestrates all steps correctly."""
    # Arrange
    mock_fetcher.fetch.return_value = Article(content="test")
    mock_classifier.classify.return_value = "SCIENCE"
    mock_persister.save.return_value = 123
    
    # Act
    article_id = orchestrate_classification(
        article_id=1,
        fetcher=mock_fetcher,
        classifier=mock_classifier,
        persister=mock_persister,
    )
    
    # Assert - verify order and data flow
    assert article_id == 123
    mock_fetcher.fetch.assert_called_once_with(1)
    mock_classifier.classify.assert_called_once()
    call_args = mock_classifier.classify.call_args[0][0]
    assert call_args.content == "test"
    mock_persister.save.assert_called_once()
```

### Pass-Through Behavior

Test that unhandled events are passed through unchanged:

```python
def test_passes_through_non_matching_events(
    mock_classifier: MagicMock,
    sample_different_event: Event,
):
    """Service ignores events it doesn't handle."""
    # Act
    result = classification_service(
        sample_different_event,
        classifier=mock_classifier,
        repository=MagicMock(),
    )
    
    # Assert - event unchanged, no calls made
    assert result == sample_different_event
    mock_classifier.classify.assert_not_called()
```

### Error Handling

Test that errors are handled appropriately:

```python
def test_raises_error_when_invalid_article(
    mock_classifier: MagicMock,
):
    """Service validates input and raises on invalid data."""
    # Arrange
    invalid_event = Event(type=EventType.NEW_ARTICLE, data=None)
    
    # Act & Assert
    with pytest.raises(ValueError, match="Article data required"):
        classification_service(invalid_event, mock_classifier, MagicMock())
```

---

## Integration Test Patterns

### Repository Create/Read

Test persistence operations with real database:

```python
def test_persists_article_to_database(session: Session):
    """Repository correctly saves and retrieves articles."""
    # Arrange
    repo = SQLArticleRepository(session)
    article = Article(
        raw_article="test content",
        url="https://example.com/test"
    )
    
    # Act
    created = repo.create(article)
    session.expunge_all()  # Clear session cache
    retrieved = repo.get_by_id(created.id)
    
    # Assert
    assert retrieved is not None
    assert retrieved.id == created.id
    assert retrieved.raw_article == "test content"
    assert retrieved.url == "https://example.com/test"
```

### Query Operations

Test complex queries and filters:

```python
def test_filters_articles_by_category(session: Session):
    """Repository correctly filters by category."""
    # Arrange
    repo = SQLArticleRepository(session)
    article1 = repo.create(Article(category="TECH"))
    article2 = repo.create(Article(category="SCIENCE"))
    article3 = repo.create(Article(category="TECH"))
    
    # Act
    tech_articles = repo.find_by_category("TECH")
    
    # Assert
    assert len(tech_articles) == 2
    assert all(a.category == "TECH" for a in tech_articles)
    assert article1.id in [a.id for a in tech_articles]
    assert article3.id in [a.id for a in tech_articles]
```

### Queue Operations

Test event queue round-trip:

```python
def test_enqueue_dequeue_round_trip(session: Session):
    """Queue correctly stores and retrieves events in order."""
    # Arrange
    queue = SQLEventQueue(session)
    event1 = Event(type=EventType.NEW_ARTICLE, data={"id": 1})
    event2 = Event(type=EventType.NEW_ARTICLE, data={"id": 2})
    
    # Act
    queue.enqueue(event1)
    queue.enqueue(event2)
    dequeued1 = queue.dequeue()
    dequeued2 = queue.dequeue()
    
    # Assert
    assert dequeued1.data["id"] == 1
    assert dequeued2.data["id"] == 2
    assert queue.dequeue() is None  # Empty queue
```

### Transaction Rollback

Test that rollback works correctly:

```python
def test_rolls_back_on_error(session: Session):
    """Transaction rolls back on error, leaving DB unchanged."""
    # Arrange
    repo = SQLArticleRepository(session)
    initial_count = session.query(Article).count()
    
    # Act
    try:
        article = repo.create(Article(raw_article="test"))
        # Simulate error after creation
        raise ValueError("Simulated error")
    except ValueError:
        session.rollback()
    
    # Assert
    final_count = session.query(Article).count()
    assert final_count == initial_count
```

---

## E2E Test Patterns

### Complete Flow

Test entire user journey:

```python
def test_article_flows_through_pipeline(
    api_client: TestClient,
    queue_worker: BackgroundWorker,
    db_session: Session,
):
    """Complete article processing from submission to classification."""
    # Arrange - initial state
    initial_count = db_session.query(Article).count()
    
    # Act - submit article via API
    response = api_client.post("/articles", json={
        "raw_article": "AI breakthrough in healthcare",
        "url": "https://example.com/article"
    })
    assert response.status_code == 201
    article_id = response.json()["id"]
    
    # Act - process in background
    queue_worker.process_all()
    
    # Assert - verify final state
    article = db_session.get(Article, article_id)
    assert article is not None
    assert article.classification is not None
    assert article.classification.category in ["TECHNOLOGY", "HEALTHCARE"]
    assert article.classification.confidence > 0.5
    
    final_count = db_session.query(Article).count()
    assert final_count == initial_count + 1
```

### Error Recovery

Test system handles errors gracefully:

```python
def test_recovers_from_classification_failure(
    api_client: TestClient,
    queue_worker: BackgroundWorker,
    mock_llm_service: MagicMock,
):
    """System retries failed classifications."""
    # Arrange
    mock_llm_service.classify.side_effect = [
        Exception("API timeout"),  # First attempt fails
        "TECHNOLOGY"               # Second attempt succeeds
    ]
    
    # Act
    response = api_client.post("/articles", json={
        "raw_article": "test content"
    })
    article_id = response.json()["id"]
    
    # Process - first attempt fails
    queue_worker.process_once()
    
    # Process - retry succeeds
    queue_worker.process_once()
    
    # Assert - eventually succeeds
    article = api_client.get(f"/articles/{article_id}").json()
    assert article["classification"]["category"] == "TECHNOLOGY"
```

---

## Fixture Patterns

### Mock Fixtures

```python
@pytest.fixture(name="mock_classifier")
def _mock_classifier_fixture() -> MagicMock:
    """Mock LLM classifier with spec."""
    mock = MagicMock(spec=LLMClassifier)
    mock.classify.return_value = "TECHNOLOGY"
    return mock


@pytest.fixture(name="mock_repository")
def _mock_repository_fixture() -> MagicMock:
    """Mock repository with common operations."""
    mock = MagicMock(spec=ClassificationRepository)
    mock.create.return_value = Classification(
        id=1,
        article_id=123,
        category="TECHNOLOGY",
        confidence=0.95
    )
    mock.get_by_id.return_value = None
    return mock
```

### Database Fixtures

```python
@pytest.fixture(name="engine", scope="session")
def _engine_fixture():
    """Test database engine (session-scoped)."""
    return create_engine("sqlite:///:memory:")


@pytest.fixture(name="session")
def _session_fixture(engine):
    """Database session with tables created."""
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
        session.rollback()
```

### Factory Fixtures

```python
@pytest.fixture(name="article_factory")
def _article_factory_fixture():
    """Factory for creating article instances."""
    def create(
        raw_article: str = "default content",
        url: str = "https://example.com",
        **kwargs
    ):
        return Article(
            raw_article=raw_article,
            url=url,
            **kwargs
        )
    return create


def test_handles_long_article(article_factory):
    """Test with article variations."""
    long_article = article_factory(raw_article="x" * 10000)
    short_article = article_factory(raw_article="test")
    # Use in test...
```

### Composed Fixtures

```python
@pytest.fixture(name="classification_service_with_mocks")
def _classification_service_with_mocks_fixture(
    mock_classifier,
    mock_repository
):
    """Pre-configured service for testing."""
    return lambda event: service(
        event,
        classifier=mock_classifier,
        repository=mock_repository,
    )


def test_something(classification_service_with_mocks):
    """Test using composed fixture."""
    result = classification_service_with_mocks(sample_event)
    # Assert...
```

---

## Common Scenarios

### Testing Event Handlers

```python
def test_handles_article_created_event(
    mock_classifier: MagicMock,
    mock_queue: MagicMock,
):
    """Event handler processes article created events."""
    # Arrange
    event = ArticleCreatedEvent(article_id=123)
    mock_classifier.classify.return_value = "TECHNOLOGY"
    
    # Act
    handle_article_created(
        event,
        classifier=mock_classifier,
        queue=mock_queue,
    )
    
    # Assert
    mock_classifier.classify.assert_called_once()
    mock_queue.enqueue.assert_called_once()
    enqueued_event = mock_queue.enqueue.call_args[0][0]
    assert enqueued_event.type == EventType.CLASSIFICATION_COMPLETE
```

### Testing API Routes

```python
def test_post_article_returns_201(api_client: TestClient):
    """POST /articles creates article and returns 201."""
    # Act
    response = api_client.post("/articles", json={
        "raw_article": "test content",
        "url": "https://example.com/test"
    })
    
    # Assert
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["raw_article"] == "test content"


def test_get_article_returns_404_when_not_found(api_client: TestClient):
    """GET /articles/{id} returns 404 for non-existent article."""
    # Act
    response = api_client.get("/articles/999999")
    
    # Assert
    assert response.status_code == 404
```

### Testing Async Code

```python
@pytest.mark.asyncio
async def test_async_classification(mock_async_client):
    """Test asynchronous classification service."""
    # Arrange
    mock_async_client.classify.return_value = "TECHNOLOGY"
    article = Article(content="test")
    
    # Act
    result = await classify_async(article, mock_async_client)
    
    # Assert
    assert result.category == "TECHNOLOGY"
    mock_async_client.classify.assert_awaited_once()
```

### Parametrized Tests

```python
@pytest.mark.parametrize("category,confidence,expected", [
    ("TECHNOLOGY", 0.95, True),
    ("TECHNOLOGY", 0.75, True),
    ("TECHNOLOGY", 0.45, False),
    ("UNKNOWN", 0.95, False),
])
def test_is_high_confidence_classification(
    category: str,
    confidence: float,
    expected: bool
):
    """Test confidence threshold logic."""
    classification = Classification(
        category=category,
        confidence=confidence
    )
    assert classification.is_high_confidence() == expected
```

---

## Anti-Patterns to Avoid

### Don't Test Implementation Details

```python
# ❌ Bad - tests internal implementation
def test_uses_specific_algorithm():
    service = MyService()
    assert service._internal_method() == "specific value"

# ✅ Good - tests behavior
def test_produces_correct_result():
    service = MyService()
    result = service.process(input_data)
    assert result.is_valid()
```

### Don't Share Mutable State

```python
# ❌ Bad - shared state between tests
class TestMyService:
    shared_data = []  # Mutable state!
    
    def test_first(self):
        self.shared_data.append(1)
        assert len(self.shared_data) == 1
    
    def test_second(self):
        # Fails if test_first runs first!
        assert len(self.shared_data) == 0

# ✅ Good - isolated state per test
class TestMyService:
    def test_first(self, sample_data):
        sample_data.append(1)
        assert len(sample_data) == 1
    
    def test_second(self, sample_data):
        assert len(sample_data) == 0
```

### Don't Test Multiple Things

```python
# ❌ Bad - tests too many things
def test_everything():
    # Creates article
    article = create_article()
    assert article.id is not None
    
    # Classifies it
    classification = classify(article)
    assert classification.category == "TECH"
    
    # Stores it
    stored = save(classification)
    assert stored == True
    
    # Retrieves it
    retrieved = get(article.id)
    assert retrieved is not None

# ✅ Good - one behavior per test
def test_creates_article_with_id():
    article = create_article()
    assert article.id is not None

def test_classifies_article():
    article = create_article()
    classification = classify(article)
    assert classification.category == "TECH"
```

---

## Quick Decision Tree

**Writing a new test? Ask:**

1. **Does it test business logic with no I/O?** → Unit test
2. **Does it test database operations?** → Integration test
3. **Does it test a complete user flow?** → E2E test

**Need a fixture? Ask:**

1. **Is it an external dependency?** → Mock fixture (`mock_*`)
2. **Is it a real instance for integration?** → Real fixture (`session`, `engine`)
3. **Need variations of test data?** → Factory fixture

**Test failing? Ask:**

1. **Is it testing behavior or implementation?** → Should test behavior
2. **Does it depend on execution order?** → Should be independent
3. **Is it testing framework code?** → Don't test that
