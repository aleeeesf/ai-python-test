"""
Factory Template for Repository Integration Tests

This template is used by the testing-conventions skill to generate factories.py files
for new domains. It creates both DB and DTO factories based on discovered domain models.

Template Variables (replaced by skill):
- {domain_name}: Name of the domain (e.g., "call_acknowledge", "call_classification")
- {db_model_name}: Name of the DB model class (e.g., "DBCall", "DBCallClassification")
- {db_model_import}: Import path for DB model
- {dto_name}: Name of the DTO class (e.g., "Call", "CallClassification")
- {dto_import}: Import path for DTO
- {db_factory_name}: Name of DB factory class (e.g., "DBCallFactory")
- {dto_factory_name}: Name of DTO factory class (e.g., "CallDTOFactory")
- {field_definitions_db}: Factory field definitions for DB factory
- {field_definitions_dto}: Factory field definitions for DTO factory
- {extra_imports}: Any additional imports needed (enums, etc.)

Example Output:
    factories.py with DBCallFactory and CallDTOFactory
"""

# Template starts here:

"""
Test data factories for {domain_name} domain integration tests.

Uses factory-boy to create database objects and DTOs for testing.
Provides sensible defaults while allowing test-specific overrides.
"""

import factory
from factory.alchemy import SQLAlchemyModelFactory

{extra_imports}
from {db_model_import} import {db_model_name}
from {dto_import} import {dto_name}


class {db_factory_name}(SQLAlchemyModelFactory):
    """Factory for creating {db_model_name} database model instances.

    Usage:
        # Create with defaults
        db_obj = {db_factory_name}.create()

        # Override specific fields
        db_obj = {db_factory_name}.create(field_name="custom_value")

        # Create multiple
        db_objs = {db_factory_name}.create_batch(3)
    """

    class Meta:
        model = {db_model_name}
        sqlalchemy_session_persistence = "flush"

{field_definitions_db}


class {dto_factory_name}(factory.Factory):
    """Factory for creating {dto_name} DTO instances (no database persistence).

    Usage:
        # Create DTO without ID (for create operations)
        dto = {dto_factory_name}.build()

        # Create DTO with ID (simulating retrieved data)
        dto = {dto_factory_name}.build(id=123)

        # Override fields
        dto = {dto_factory_name}.build(field_name="custom_value")

        # Create multiple
        dtos = {dto_factory_name}.build_batch(3)
    """

    class Meta:
        model = {dto_name}

    id = None  # DTOs for create operations don't have IDs yet
{field_definitions_dto}


# Field Generation Rules (used by skill when generating this file):
#
# String fields:
#   field_name = factory.Sequence(lambda n: f"value-{n:04d}")
#
# S3 URIs:
#   s3_uri = factory.Sequence(lambda n: f"s3://test-bucket/path-{n:04d}.ext")
#
# Integer fields (non-ID):
#   count = factory.Sequence(lambda n: n)
#
# Enum fields:
#   category = factory.Iterator(MyEnum)  # Cycles through values
#   # OR
#   category = MyEnum.DEFAULT_VALUE  # Fixed value
#
# Boolean fields:
#   is_active = True  # Fixed value
#   # OR
#   is_active = factory.Faker("boolean")  # Random
#
# DateTime fields:
#   created_at = factory.LazyFunction(datetime.now)
#
# Foreign Key fields:
#   related_id = factory.Sequence(lambda n: n)
#   # OR with SubFactory:
#   related = factory.SubFactory(RelatedFactory)
#
# JSON fields:
#   metadata = {"key": "value"}  # Fixed
#   # OR
#   metadata = factory.LazyFunction(lambda: {"timestamp": datetime.now().isoformat()})
#
# Nullable fields:
#   optional_field = None  # Always None
#   # OR
#   optional_field = factory.Maybe(
#       factory.Faker("boolean"),
#       yes_declaration=factory.Faker("word"),
#       no_declaration=None,
#   )
