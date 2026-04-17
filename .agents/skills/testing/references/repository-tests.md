# Repository Integration Testing Guide

Comprehensive guide for testing repository implementations in the transcriptions-system project.

## Table of Contents

1. [Testing Philosophy](#testing-philosophy)
2. [Test Organization](#test-organization)
3. [Factory Selection](#factory-selection)
4. [Session Management](#session-management)
5. [Testing CRUD Operations](#testing-crud-operations)
6. [Testing Queries](#testing-queries)
7. [Common Pitfalls](#common-pitfalls)

## Testing Philosophy

### What to Test in Repository Integration Tests

✅ **DO test:**
- **DTO ↔ DB model mapping** - Does repository correctly convert between layers?
- **Port contract compliance** - Does repository implement the interface correctly?
- **Query correctness** - Do filters and conditions return expected results?
- **Null/edge case handling** - How does repository handle missing data?
- **Database-specific behavior** - ENUM conversions, JSON fields, etc.

❌ **DON'T test:**
- **SQLAlchemy framework behavior** - Transaction commits, connection pooling
- **Database constraints** - Primary keys, foreign keys (trust the framework)
- **ID auto-generation** - The database handles this
- **Basic SQL operations** - INSERT, UPDATE work as expected

### Example: What Not to Test

```python
# ❌ Bad - Testing framework behavior
def test_generates_unique_ids(self, session):
    """This tests SQLAlchemy, not our repository."""
    call1 = repository.create(CallDTOFactory.build())
    call2 = repository.create(CallDTOFactory.build())
    assert call1.id != call2.id  # SQLAlchemy guarantees this

# ✅ Good - Testing our mapping logic
def test_maps_dto_to_database_and_returns_with_id(self, session):
    """Tests OUR repository's DTO → DB → DTO conversion."""
    input_dto = CallDTOFactory.build()
    result = repository.create(input_dto)

    assert isinstance(result, Call)  # Returns correct type
    assert result.id is not None  # ID was populated
    assert result.call_location == input_dto.call_location  # Data preserved
```

## Test Organization

### File Structure

```
src/tests/integration/domains/<domain>/infrastructure/repositories/
├── conftest.py                              # Factory configuration
├── factories.py                             # DB and DTO factories
└── test_sql_<repository_name>.py           # Repository tests
```

### Test Class Structure

**One class per repository method** - Group tests by the method being tested:

```python
class TestSQLCallAcknowledgementRepositoryCreate:
    """Tests for the create method DTO-to-DB mapping."""

    def test_maps_dto_to_database_and_returns_with_id(self, session):
        pass

    def test_preserves_s3_uri_format_with_special_characters(self, session):
        pass


class TestSQLCallAcknowledgementRepositoryGetById:
    """Tests for the get_by_id method DB-to-DTO mapping."""

    def test_returns_none_for_nonexistent_id(self, session):
        pass

    def test_maps_database_record_to_dto_correctly(self, session):
        pass
```

**Benefits:**
- Clear test organization
- Easy to find tests for specific methods
- Docstrings explain what's being tested
- Parallel test execution per class

### Test Naming Convention

```python
def test_<action>_<condition>():
    """<Clear description of expected behavior>."""
```

**Examples:**
```python
def test_returns_none_for_nonexistent_id(self, session):
    """Should return None when call does not exist, per port contract."""

def test_maps_database_record_to_dto_correctly(self, session):
    """Should correctly map DBCall to Call DTO with all fields."""

def test_distinguishes_between_different_call_records(self, session):
    """Should retrieve correct record when multiple exist (tests ID filtering)."""
```

## Factory Selection

### Decision Tree

```
Which repository method are you testing?

CREATE (DTO → DB)
└─ Use: CallDTOFactory.build()
   Why: Testing repository's ability to persist input DTO
   Example:
       input_data = CallDTOFactory.build()
       result = repository.create(data=input_data)

GET/FIND (DB → DTO)
└─ Use: DBCallFactory.create() + session.flush()
   Why: Setting up test data in database
   Example:
       db_call = DBCallFactory.create()
       session.flush()
       result = repository.get_by_id(db_call.id)

UPDATE (DTO → DB, then verify)
└─ Use: DBCallFactory.create() for setup
   Then: CallDTOFactory.build() for update data
   Example:
       db_call = DBCallFactory.create()
       session.flush()
       update_data = CallDTOFactory.build(id=db_call.id)
       result = repository.update(update_data)

DELETE (DB → void)
└─ Use: DBCallFactory.create()
   Why: Need existing record to delete
   Example:
       db_call = DBCallFactory.create()
       session.flush()
       repository.delete(db_call.id)
```

### Pattern: Create Operations

```python
from tests.integration.domains.call_acknowledge.infrastructure.repositories.factories import (
    CallDTOFactory,  # ← DTO factory for input
)

class TestRepositoryCreate:
    """Tests for create method - DTO → DB mapping."""

    def test_creates_and_returns_with_id(self, session: Session):
        # Arrange - Use DTO factory (not in DB yet)
        repository = CallAcknowledgementRepository(session)
        input_data = CallDTOFactory.build()  # ← .build(), not .create()

        # Act - Repository persists to DB
        result = repository.create(data=input_data)

        # Assert - Verify mapping
        assert isinstance(result, Call)
        assert result.id is not None  # ID was generated
        assert result.call_location == input_data.call_location
```

### Pattern: Read Operations

```python
from tests.integration.domains.call_acknowledge.infrastructure.repositories.factories import (
    DBCallFactory,  # ← DB factory for test data
)

class TestRepositoryGetById:
    """Tests for get_by_id method - DB → DTO mapping."""

    def test_retrieves_by_id(self, session: Session):
        # Arrange - Use DB factory (persist test data)
        repository = CallAcknowledgementRepository(session)
        db_call = DBCallFactory.create()  # ← .create(), not .build()
        session.flush()  # ← CRITICAL: Generate ID

        # Act - Repository reads from DB
        result = repository.get_by_id(db_call.id)

        # Assert - Verify mapping
        assert isinstance(result, Call)
        assert result.id == db_call.id
        assert result.call_location == db_call.call_location
```

## Session Management

### The Session Lifecycle in Tests

```python
# 1. Fixture provides session (from integration/conftest.py)
@pytest.fixture(scope="function", name="session")
def _session_fixture(db_engine, tables) -> Session:
    connection = db_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    try:
        yield session  # ← Your test uses this
    finally:
        session.close()
        transaction.rollback()  # ← Rolls back all changes
        connection.close()

# 2. Test runs with clean session
def test_something(session):
    # Session is fresh, empty database
    repository = MyRepository(session)
    # ... test code ...
    # Test ends, session is rolled back automatically
```

### When to Call session.flush()

**Rule:** Always call `session.flush()` immediately after `DBFactory.create()`

```python
# ✅ Correct
db_call = DBCallFactory.create()
session.flush()  # Generates ID, makes data queryable
assert db_call.id is not None  # Works

# ❌ Wrong
db_call = DBCallFactory.create()
assert db_call.id is not None  # Fails - ID not generated yet
```

**Why flush?**
- Executes INSERT statement
- Generates auto-increment IDs
- Makes data available for queries
- Stays within transaction (rollback still works)

### When to Call session.expunge_all()

**Use case:** Testing that repository queries from DB, not from session cache

```python
def test_retrieves_fresh_data_from_database(self, session: Session):
    # Arrange
    repository = CallAcknowledgementRepository(session)
    db_call = DBCallFactory.create()
    session.flush()

    # Clear session cache
    session.expunge_all()  # ← Forces query to hit database

    # Act
    result = repository.get_by_id(db_call.id)

    # Assert - Verifies repository queries DB, not cache
    assert result is not None
```

**When NOT needed:** Most tests don't need this. Session caching is fine.

## Testing CRUD Operations

### Create Operation Pattern

```python
class TestRepositoryCreate:
    """Tests for create method."""

    def test_maps_dto_to_database_and_returns_with_id(self, session: Session):
        """Should correctly map Call DTO to DBCall and return with generated ID."""
        # Arrange
        repository = SQLCallAcknowledgementRepository(session)
        call_data = CallDTOFactory.build()

        # Act
        created_call = repository.create(data=call_data)

        # Assert - Verify DTO mapping
        assert isinstance(created_call, Call), "Should return Call DTO"
        assert created_call.id is not None, "Should populate ID from database"
        assert created_call.call_location == call_data.call_location

    def test_preserves_complex_field_values(self, session: Session):
        """Should correctly handle edge cases in field values."""
        # Arrange
        repository = SQLCallAcknowledgementRepository(session)
        complex_location = "s3://my-bucket/path/to/deep/folder/call-with-special_chars-123.wav"
        call_data = CallDTOFactory.build(call_location=complex_location)

        # Act
        created_call = repository.create(data=call_data)

        # Assert - Verify no corruption of complex strings
        assert created_call.call_location == complex_location
```

### Read Operation Patterns

#### Get by ID

```python
class TestRepositoryGetById:
    """Tests for get_by_id method."""

    def test_returns_none_for_nonexistent_id(self, session: Session):
        """Should return None when call does not exist, per port contract."""
        # Arrange
        repository = SQLCallAcknowledgementRepository(session)

        # Act
        result = repository.get_by_id(id=99999)

        # Assert - Verify contract compliance
        assert result is None, "Port contract requires None for missing entities"

    def test_maps_database_record_to_dto_correctly(self, session: Session):
        """Should correctly map DBCall to Call DTO with all fields."""
        # Arrange - Setup DB state directly (not testing create method)
        repository = SQLCallAcknowledgementRepository(session)
        db_call = DBCallFactory.create()
        session.flush()  # Ensure ID is generated

        # Act - Test ONLY get_by_id (DB → DTO mapping)
        retrieved_call = repository.get_by_id(db_call.id)

        # Assert - Verify mapping correctness
        assert isinstance(retrieved_call, Call)
        assert retrieved_call.id == db_call.id
        assert retrieved_call.call_location == db_call.call_location

    def test_distinguishes_between_different_records(self, session: Session):
        """Should retrieve correct record when multiple exist (tests ID filtering)."""
        # Arrange - Setup multiple records
        repository = SQLCallAcknowledgementRepository(session)
        db_calls = DBCallFactory.create_batch(3)
        session.flush()

        # Act - Request specific ID (middle record)
        retrieved_call = repository.get_by_id(db_calls[1].id)

        # Assert - Verify correct filtering logic
        assert retrieved_call.id == db_calls[1].id
        assert retrieved_call.call_location == db_calls[1].call_location
```

#### Find/Query Operations

```python
class TestRepositoryFindByCategory:
    """Tests for find_by_category query method."""

    def test_returns_empty_list_when_no_matches(self, session: Session):
        """Should return empty list when no calls match the category."""
        # Arrange
        repository = CallClassificationRepository(session)
        DBCallClassificationFactory.create(call_category=CallCategoryEnum.INFORMACION)
        session.flush()

        # Act
        results = repository.find_by_category(CallCategoryEnum.RECLAMO)

        # Assert
        assert results == []

    def test_filters_by_category_correctly(self, session: Session):
        """Should return only calls matching the specified category."""
        # Arrange
        repository = CallClassificationRepository(session)
        info_call = DBCallClassificationFactory.create(
            call_category=CallCategoryEnum.INFORMACION
        )
        reclamo_call = DBCallClassificationFactory.create(
            call_category=CallCategoryEnum.RECLAMO
        )
        session.flush()

        # Act
        results = repository.find_by_category(CallCategoryEnum.INFORMACION)

        # Assert
        assert len(results) == 1
        assert results[0].id == info_call.id
        assert results[0].call_category == CallCategoryEnum.INFORMACION

    def test_returns_all_matches_when_multiple_exist(self, session: Session):
        """Should return all calls matching the category."""
        # Arrange
        repository = CallClassificationRepository(session)
        info_calls = DBCallClassificationFactory.create_batch(
            3,
            call_category=CallCategoryEnum.INFORMACION
        )
        session.flush()

        # Act
        results = repository.find_by_category(CallCategoryEnum.INFORMACION)

        # Assert
        assert len(results) == 3
        result_ids = {r.id for r in results}
        expected_ids = {c.id for c in info_calls}
        assert result_ids == expected_ids
```

### Update Operation Pattern

```python
class TestRepositoryUpdate:
    """Tests for update method."""

    def test_updates_fields_and_returns_updated_dto(self, session: Session):
        """Should update database record and return updated DTO."""
        # Arrange - Create existing record
        repository = CallAcknowledgementRepository(session)
        db_call = DBCallFactory.create(call_location="s3://old-location.wav")
        session.flush()

        # Create update DTO
        update_data = CallDTOFactory.build(
            id=db_call.id,
            call_location="s3://new-location.wav"
        )

        # Act
        result = repository.update(update_data)

        # Assert
        assert result.id == db_call.id
        assert result.call_location == "s3://new-location.wav"

        # Verify persistence
        session.expunge_all()
        persisted = repository.get_by_id(db_call.id)
        assert persisted.call_location == "s3://new-location.wav"
```

### Delete Operation Pattern

```python
class TestRepositoryDelete:
    """Tests for delete method."""

    def test_deletes_record_successfully(self, session: Session):
        """Should remove record from database."""
        # Arrange
        repository = CallAcknowledgementRepository(session)
        db_call = DBCallFactory.create()
        session.flush()
        call_id = db_call.id

        # Act
        repository.delete(call_id)
        session.flush()  # Commit deletion

        # Assert - Record no longer exists
        result = repository.get_by_id(call_id)
        assert result is None

    def test_delete_nonexistent_id_handles_gracefully(self, session: Session):
        """Should handle deletion of non-existent ID without error."""
        # Arrange
        repository = CallAcknowledgementRepository(session)

        # Act & Assert - Should not raise exception
        repository.delete(99999)  # May be no-op or raise custom exception based on contract
```

## Testing Queries

### Pattern: Filter by Single Criterion

```python
def test_finds_calls_by_location_pattern(self, session: Session):
    """Should find calls matching S3 location pattern."""
    # Arrange
    repository = CallAcknowledgementRepository(session)
    matching_call = DBCallFactory.create(call_location="s3://prod-bucket/call.wav")
    non_matching_call = DBCallFactory.create(call_location="s3://dev-bucket/call.wav")
    session.flush()

    # Act
    results = repository.find_by_location_pattern("prod-bucket")

    # Assert
    assert len(results) == 1
    assert results[0].id == matching_call.id
```

### Pattern: Complex Queries

```python
def test_finds_recent_calls_with_category(self, session: Session):
    """Should find calls from last 7 days with specific category."""
    # Arrange
    repository = CallRepository(session)
    recent_info = DBCallFactory.create(
        created_at=datetime.now() - timedelta(days=2),
        category=CallCategoryEnum.INFORMACION
    )
    old_info = DBCallFactory.create(
        created_at=datetime.now() - timedelta(days=10),
        category=CallCategoryEnum.INFORMACION
    )
    recent_reclamo = DBCallFactory.create(
        created_at=datetime.now() - timedelta(days=2),
        category=CallCategoryEnum.RECLAMO
    )
    session.flush()

    # Act
    results = repository.find_recent_by_category(
        category=CallCategoryEnum.INFORMACION,
        days=7
    )

    # Assert
    assert len(results) == 1
    assert results[0].id == recent_info.id
```

## Common Pitfalls

### Pitfall 1: Forgetting session.flush()

```python
# ❌ Bad
db_call = DBCallFactory.create()
assert db_call.id is not None  # FAILS - ID not generated

# ✅ Good
db_call = DBCallFactory.create()
session.flush()  # Generate ID
assert db_call.id is not None  # PASSES
```

### Pitfall 2: Testing Framework Behavior

```python
# ❌ Bad - Testing SQLAlchemy
def test_commit_persists_to_database(self, session):
    """This tests SQLAlchemy, not our code."""
    db_call = DBCallFactory.create()
    session.commit()
    # Testing framework behavior

# ✅ Good - Testing our repository
def test_repository_persists_call_correctly(self, session):
    """Tests our repository's create implementation."""
    input_dto = CallDTOFactory.build()
    result = repository.create(input_dto)
    assert result.id is not None  # Our repository populated the ID
```

### Pitfall 3: Wrong Factory Type

```python
# ❌ Bad - Using DB factory for input
def test_create(self, session):
    call_data = DBCallFactory.create()  # Already in DB!
    result = repository.create(call_data)  # Duplicate insertion

# ✅ Good - Using DTO factory for input
def test_create(self, session):
    call_data = CallDTOFactory.build()  # Not in DB
    result = repository.create(call_data)  # Fresh insertion
```

### Pitfall 4: Testing Multiple Methods in One Test

```python
# ❌ Bad - Testing both create and get_by_id
def test_create_and_retrieve(self, session):
    """Tests both create and get_by_id - which one failed if test fails?"""
    call_data = CallDTOFactory.build()
    created = repository.create(call_data)  # Test create
    retrieved = repository.get_by_id(created.id)  # Test get_by_id
    assert retrieved.id == created.id

# ✅ Good - Separate tests
def test_create_returns_with_id(self, session):
    """Tests ONLY create."""
    call_data = CallDTOFactory.build()
    created = repository.create(call_data)
    assert created.id is not None

def test_get_by_id_retrieves_correctly(self, session):
    """Tests ONLY get_by_id."""
    db_call = DBCallFactory.create()
    session.flush()
    retrieved = repository.get_by_id(db_call.id)
    assert retrieved.id == db_call.id
```

### Pitfall 5: Not Testing Edge Cases

```python
# ❌ Bad - Only happy path
class TestRepositoryGetById:
    def test_retrieves_call(self, session):
        db_call = DBCallFactory.create()
        session.flush()
        result = repository.get_by_id(db_call.id)
        assert result.id == db_call.id

# ✅ Good - Include edge cases
class TestRepositoryGetById:
    def test_retrieves_call_when_exists(self, session):
        """Happy path."""
        db_call = DBCallFactory.create()
        session.flush()
        result = repository.get_by_id(db_call.id)
        assert result.id == db_call.id

    def test_returns_none_when_not_found(self, session):
        """Edge case: missing record."""
        result = repository.get_by_id(99999)
        assert result is None

    def test_distinguishes_between_multiple_records(self, session):
        """Edge case: correct filtering."""
        db_calls = DBCallFactory.create_batch(3)
        session.flush()
        result = repository.get_by_id(db_calls[1].id)
        assert result.id == db_calls[1].id
```

## Best Practices Summary

1. **One test class per repository method** - Clear organization
2. **Use DTO factories for input** - Testing create operations
3. **Use DB factories for setup** - Testing read/update/delete operations
4. **Always flush after DB factory** - Generate IDs immediately
5. **Test behavior, not implementation** - Focus on contracts
6. **Include edge cases** - None returns, empty lists, filtering
7. **Clear test names** - Describe what's being tested
8. **AAA pattern** - Arrange, Act, Assert with comments
9. **Verify mapping** - Check DTO ↔ DB conversions
10. **Isolate tests** - One repository method per test
