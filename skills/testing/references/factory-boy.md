# Factory-Boy Integration Guide

Comprehensive guide for using factory-boy in the transcriptions-system project for creating test data with SQLAlchemy/SQLModel.

## Table of Contents

1. [Introduction](#introduction)
2. [Factory Class Anatomy](#factory-class-anatomy)
3. [Session Management](#session-management)
4. [Common Patterns](#common-patterns)
5. [Decision Trees](#decision-trees)
6. [Complex Models](#complex-models)
7. [Troubleshooting](#troubleshooting)

## Introduction

Factory-boy provides a flexible way to create test data for integration tests. In this project, we use two types of factories:

- **SQLAlchemyModelFactory**: For database models (persists to DB)
- **factory.Factory**: For DTOs (in-memory objects)

### Why Two Factory Types?

```python
# DB Factory - Persists to database
db_call = DBCallFactory.create()  # INSERT executed
session.flush()  # ID generated

# DTO Factory - No persistence
call_dto = CallDTOFactory.build()  # No database interaction
```

**Use case mapping:**
- Testing `repository.create()` → Use DTO factory (input data)
- Testing `repository.get_by_id()` → Use DB factory (test data setup)

## Factory Class Anatomy

### SQLAlchemyModelFactory (DB Models)

```python
import factory
from factory.alchemy import SQLAlchemyModelFactory
from transcriptions_system.domains.call_acknowledge.domain.models.db_call import DBCall

class DBCallFactory(SQLAlchemyModelFactory):
    """Factory for creating DBCall database model instances.

    This factory PERSISTS data to the database when you call .create().
    """

    # Meta configuration
    class Meta:
        model = DBCall  # The SQLModel/SQLAlchemy model
        sqlalchemy_session_persistence = "flush"  # CRITICAL: Use flush, not commit

    # Field definitions
    call_location = factory.Sequence(
        lambda n: f"s3://test-bucket/recordings/call-{n:04d}.wav"
    )
```

**Key Meta Options:**
- `model`: The database model class
- `sqlalchemy_session_persistence = "flush"`: Use flush (not commit) for test isolation
- `sqlalchemy_session`: Set via conftest fixture (see Session Management)

### factory.Factory (DTOs)

```python
from transcriptions_system.domains.call_acknowledge.application.dtos import Call

class CallDTOFactory(factory.Factory):
    """Factory for creating Call DTO instances.

    This factory does NOT persist to database - use for input data.
    """

    class Meta:
        model = Call  # The Pydantic/dataclass model

    id = None  # DTOs for create operations don't have IDs yet
    call_location = factory.Sequence(
        lambda n: f"s3://test-bucket/recordings/call-{n:04d}.wav"
    )
```

**Key Differences from DB Factory:**
- No `sqlalchemy_session_persistence`
- No session configuration needed
- Use `.build()` instead of `.create()`

## Session Management

### The Problem

Factory-boy factories need access to the SQLAlchemy session to persist data. In tests, each test gets its own session via fixtures.

### The Solution: Auto-Configure Factories

Create a conftest.py at the repository test level:

```python
# src/tests/integration/domains/<domain>/infrastructure/repositories/conftest.py

import pytest
from tests.integration.domains.<domain>.infrastructure.repositories.factories import (
    DBCallFactory,  # Import all DB factories
)

@pytest.fixture(scope="function", autouse=True)
def _configure_factories(session):
    """
    Automatically configure factory-boy factories with the test session.

    This fixture runs automatically for all tests in this directory,
    ensuring factories use the correct test database session.

    IMPORTANT:
    - Add a line for each new DB factory you create
    - Only DB factories need configuration (not DTO factories)
    - This runs per-test (scope="function") for proper isolation
    """
    DBCallFactory._meta.sqlalchemy_session = session
    # Add more factories here as needed:
    # DBContactFactory._meta.sqlalchemy_session = session
```

### Why autouse=True?

- Runs automatically for every test in the directory
- No need to explicitly request the fixture
- Ensures factories are always configured correctly
- Prevents cryptic "session not found" errors

## Common Patterns

### Pattern 1: Unique Sequential Values

Use `factory.Sequence` for fields that need unique values:

```python
class DBCallFactory(SQLAlchemyModelFactory):
    # Sequential integers
    call_id = factory.Sequence(lambda n: n)

    # Sequential strings with formatting
    call_location = factory.Sequence(
        lambda n: f"s3://bucket/call-{n:04d}.wav"  # call-0001.wav, call-0002.wav
    )

    # Sequential with timestamp
    created_at = factory.Sequence(
        lambda n: datetime(2024, 1, 1) + timedelta(days=n)
    )
```

### Pattern 2: Realistic Fake Data

Use `factory.Faker` for realistic test data:

```python
from factory import Faker

class DBContactFactory(SQLAlchemyModelFactory):
    first_name = Faker("first_name")  # "John", "Jane", etc.
    last_name = Faker("last_name")
    email = Faker("email")
    phone = Faker("phone_number")
    address = Faker("address")
```

### Pattern 3: Enum Fields

Use `factory.Iterator` or direct assignment for enum fields:

```python
from transcriptions_system.domains.call_classification.domain.models.enums import CallCategoryEnum

class DBCallClassificationFactory(SQLAlchemyModelFactory):
    # Option 1: Cycle through enum values
    call_category = factory.Iterator(CallCategoryEnum)

    # Option 2: Always use specific value
    call_category = CallCategoryEnum.INFORMACION

    # Option 3: Randomize
    call_category = factory.LazyFunction(
        lambda: random.choice(list(CallCategoryEnum))
    )
```

### Pattern 4: Conditional Fields

Use `factory.LazyAttribute` for computed fields:

```python
class DBCallFactory(SQLAlchemyModelFactory):
    call_location = factory.Sequence(lambda n: f"s3://bucket/call-{n:04d}.wav")

    # Derive transcript location from call location
    transcript_location = factory.LazyAttribute(
        lambda obj: obj.call_location.replace(".wav", ".json")
    )
```

### Pattern 5: Batch Creation

Create multiple instances efficiently:

```python
# Create 3 calls with sequential IDs
db_calls = DBCallFactory.create_batch(3)
# Results in: call-0001.wav, call-0002.wav, call-0003.wav

# Override specific fields
db_calls = DBCallFactory.create_batch(
    3,
    call_location="s3://custom-bucket/call.wav"  # All use same location
)

# Mix of defaults and overrides
calls = [
    DBCallFactory.create(call_location="s3://bucket/important.wav"),
    *DBCallFactory.create_batch(2),  # Use defaults for these
]
```

## Decision Trees

### Which Factory Type?

```
What are you creating test data for?
├─ Database model (DBCall, DBContact, etc.)
│  └─ Use SQLAlchemyModelFactory
│     - Extends factory.alchemy.SQLAlchemyModelFactory
│     - Needs session configuration
│     - Use .create() to persist
│
└─ DTO / Pydantic model (Call, Contact, etc.)
   └─ Use factory.Factory
      - Extends factory.Factory
      - No session needed
      - Use .build() for in-memory objects
```

### Which Factory Method?

```
What do you need?
├─ Single instance in database
│  └─ DBFactory.create()
│     - Persists to DB
│     - Followed by session.flush()
│
├─ Multiple instances in database
│  └─ DBFactory.create_batch(n)
│     - Creates n instances
│     - Followed by session.flush()
│
├─ Single DTO (no DB)
│  └─ DTOFactory.build()
│     - In-memory only
│     - For input data
│
└─ Multiple DTOs (no DB)
   └─ DTOFactory.build_batch(n)
      - Creates n DTOs
      - For batch operations
```

### When to Flush?

```
Did you call DBFactory.create()?
├─ Yes
│  └─ Call session.flush() immediately after
│     WHY: Generates auto-incremented IDs
│     WHY: Makes data available for queries
│     EXAMPLE:
│         db_call = DBCallFactory.create()
│         session.flush()  # ← REQUIRED
│         assert db_call.id is not None  # Now ID exists
│
└─ No (used DTOFactory.build())
   └─ Don't call flush
      WHY: No database interaction happened
```

## Complex Models

### Handling Foreign Keys

**Scenario**: DBCallClassification has a foreign key to DBCall

```python
class DBCallClassificationFactory(SQLAlchemyModelFactory):
    class Meta:
        model = DBCallClassification
        sqlalchemy_session_persistence = "flush"

    # Option 1: Create related object inline
    call_id = factory.LazyAttribute(
        lambda obj: DBCallFactory.create().id
    )

    # Option 2: Use SubFactory (preferred)
    call = factory.SubFactory(DBCallFactory)
    # Factory-boy automatically extracts call.id for call_id field

    call_category = CallCategoryEnum.INFORMACION
```

**Usage:**
```python
# Creates both DBCall and DBCallClassification
classification = DBCallClassificationFactory.create()
session.flush()

# Access related call
assert classification.call is not None
```

### Handling Complex Enums

For ENUMs stored as VARCHAR with custom types:

```python
from sqlalchemy import Column
from transcriptions_system.infrastructure.db.custom_types import SQLAEnum

# Model definition
class DBCallClassification(SQLModel, table=True):
    call_category: CallCategoryEnum = Field(
        sa_column=Column(SQLAEnum(CallCategoryEnum), nullable=False)
    )

# Factory handles this automatically
class DBCallClassificationFactory(SQLAlchemyModelFactory):
    call_category = CallCategoryEnum.INFORMACION  # Just assign enum value
```

### Handling JSON Fields

For models with JSON columns:

```python
class DBCallFactory(SQLAlchemyModelFactory):
    # Simple JSON
    metadata = {"source": "aws_connect", "version": "1.0"}

    # Dynamic JSON
    metadata = factory.LazyFunction(
        lambda: {
            "source": "aws_connect",
            "timestamp": datetime.now().isoformat(),
        }
    )
```

### Handling Nullable Fields

```python
class DBContactFactory(SQLAlchemyModelFactory):
    name = factory.Sequence(lambda n: f"Contact {n}")

    # Always None
    email = None

    # Sometimes None (50% of the time)
    phone = factory.Maybe(
        factory.Faker("boolean"),
        yes_declaration=factory.Faker("phone_number"),
        no_declaration=None,
    )
```

## Troubleshooting

### Error: "Session not bound"

**Symptom:**
```
sqlalchemy.exc.UnboundExecutionError: This session is not bound to a Connection
```

**Cause:** Factory not configured with session

**Solution:** Add factory to conftest.py
```python
@pytest.fixture(scope="function", autouse=True)
def _configure_factories(session):
    DBCallFactory._meta.sqlalchemy_session = session  # ← Add this line
```

### Error: "ID is None after create()"

**Symptom:**
```python
db_call = DBCallFactory.create()
assert db_call.id is not None  # AssertionError
```

**Cause:** Forgot to call `session.flush()`

**Solution:**
```python
db_call = DBCallFactory.create()
session.flush()  # ← Add this
assert db_call.id is not None  # Now works
```

### Error: "IntegrityError: duplicate key value"

**Symptom:**
```
IntegrityError: (psycopg2.errors.UniqueViolation) duplicate key value violates unique constraint
```

**Cause:** Using static values for unique fields

**Solution:** Use `factory.Sequence`
```python
# ❌ Bad - all instances have same ID
class DBCallFactory(SQLAlchemyModelFactory):
    external_id = "call-123"  # Duplicate!

# ✅ Good - each instance gets unique ID
class DBCallFactory(SQLAlchemyModelFactory):
    external_id = factory.Sequence(lambda n: f"call-{n}")
```

### Issue: Factories not isolated between tests

**Symptom:** Sequence numbers continue across tests (call-0001, call-0002 in test 1, then call-0003 in test 2)

**This is NORMAL:** factory-boy sequences are global by design

**Not a problem** because:
- Tests use transaction rollback (data doesn't persist)
- Unique values are still unique within each test
- If you need to reset: `DBCallFactory.reset_sequence()`

### Issue: Test data not visible in queries

**Symptom:**
```python
db_call = DBCallFactory.create()
session.flush()
result = session.query(DBCall).all()
assert len(result) == 1  # AssertionError: len(result) == 0
```

**Cause:** Query happened in different transaction or session was expired

**Solution:** Ensure using same session
```python
# ✅ Use repository with same session
repository = CallRepository(session)  # Same session as factory
result = repository.find_all()
```

## Best Practices

1. **Always call session.flush() after DBFactory.create()**
   ```python
   db_call = DBCallFactory.create()
   session.flush()  # Don't forget!
   ```

2. **Use Sequence for unique fields**
   ```python
   call_location = factory.Sequence(lambda n: f"s3://bucket/call-{n:04d}.wav")
   ```

3. **Keep DTO and DB factories in sync**
   ```python
   # Both should have same fields (except id/Meta)
   class DBCallFactory(SQLAlchemyModelFactory):
       call_location = factory.Sequence(lambda n: f"s3://bucket/call-{n:04d}.wav")

   class CallDTOFactory(factory.Factory):
       call_location = factory.Sequence(lambda n: f"s3://bucket/call-{n:04d}.wav")
   ```

4. **One conftest per domain repository tests**
   ```
   src/tests/integration/domains/call_acknowledge/infrastructure/repositories/
   ├── conftest.py          ← Configure factories here
   ├── factories.py
   └── test_*.py
   ```

5. **Group factory configuration in conftest**
   ```python
   @pytest.fixture(scope="function", autouse=True)
   def _configure_factories(session):
       # All domain factories in one place
       DBCallFactory._meta.sqlalchemy_session = session
       DBContactFactory._meta.sqlalchemy_session = session
       DBCallClassificationFactory._meta.sqlalchemy_session = session
   ```

## Summary

- **Two factory types**: SQLAlchemyModelFactory (DB) and factory.Factory (DTO)
- **Session configuration**: Required for DB factories via conftest autouse fixture
- **Always flush**: After DBFactory.create() to generate IDs
- **Use Sequence**: For unique field values
- **DTO vs DB**: DTO for input data, DB for test setup
- **Batch operations**: Use create_batch(n) or build_batch(n)
- **Complex models**: Use SubFactory for relationships
