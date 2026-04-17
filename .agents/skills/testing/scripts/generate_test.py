#!/usr/bin/env python3
"""
FastPaip Test Generator

Generates test file scaffolding following FastPaip testing conventions.
Ensures consistent structure, naming, and patterns.
"""

import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Literal


TestType = Literal["unit", "integration", "e2e"]


@dataclass
class TestConfig:
    """Test generation configuration."""

    type: TestType
    module: str  # BC name (e.g., "core", "acknowledge")
    subject: str  # Class/function being tested
    output: Path


UNIT_TEMPLATE = '''"""
Unit tests for {subject}.

Tests domain/application logic with mocked dependencies.
"""

import pytest
from unittest.mock import MagicMock

from {module}.domain.models import {subject}


class Test{subject}:
    """Test suite for {subject}."""

    def test_{snake_subject}_performs_action_when_condition(self):
        """Test that {subject} performs expected action under specific condition."""
        # Arrange
        # TODO: Set up test data and mocks

        # Act
        # TODO: Execute the code under test

        # Assert
        # TODO: Verify expected outcomes
        pass

    def test_{snake_subject}_raises_error_when_invalid_input(self):
        """Test that {subject} handles invalid input appropriately."""
        # Arrange
        # TODO: Set up invalid input scenario

        # Act & Assert
        with pytest.raises(ValueError):
            # TODO: Call code that should raise
            pass


# Fixtures

@pytest.fixture(name="mock_dependency")
def _mock_dependency_fixture() -> MagicMock:
    """Mock for external dependency."""
    return MagicMock()


@pytest.fixture(name="sample_{snake_subject}")
def _sample_{snake_subject}_fixture():
    """Sample {subject} instance for testing."""
    return {subject}(
        # TODO: Add required fields
    )
'''

INTEGRATION_TEMPLATE = '''"""
Integration tests for {subject}.

Tests infrastructure components with real database.
"""

import pytest
from sqlmodel import Session

from {module}.infrastructure.db_models import {subject}


class Test{subject}Integration:
    """Integration test suite for {subject}."""

    def test_persists_{snake_subject}_to_database(self, session: Session):
        """Test that {subject} correctly saves to database."""
        # Arrange
        instance = {subject}(
            # TODO: Add required fields
        )

        # Act
        session.add(instance)
        session.commit()
        session.refresh(instance)

        # Assert
        assert instance.id is not None
        # TODO: Add more assertions

    def test_retrieves_{snake_subject}_by_id(self, session: Session):
        """Test that {subject} can be retrieved by ID."""
        # Arrange
        instance = {subject}(
            # TODO: Add required fields
        )
        session.add(instance)
        session.commit()
        instance_id = instance.id
        session.expunge_all()

        # Act
        retrieved = session.get({subject}, instance_id)

        # Assert
        assert retrieved is not None
        assert retrieved.id == instance_id
        # TODO: Add more assertions


# Fixtures

@pytest.fixture(name="session")
def _session_fixture(engine):
    """Database session for integration tests."""
    from sqlmodel import SQLModel
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
        session.rollback()


@pytest.fixture(name="engine")
def _engine_fixture():
    """Test database engine."""
    from sqlmodel import create_engine
    return create_engine("sqlite:///:memory:")
'''

E2E_TEMPLATE = '''"""
End-to-end tests for {subject} flow.

Tests complete user scenarios through the full system.
"""

import pytest


class Test{subject}E2E:
    """E2E test suite for {subject} flow."""

    def test_complete_{snake_subject}_flow(self):
        """Test complete flow from start to finish."""
        # TODO: Implement full scenario test
        # 1. Set up initial state
        # 2. Execute user actions
        # 3. Verify final system state
        pass

    def test_{snake_subject}_handles_error_scenario(self):
        """Test that system handles error scenarios gracefully."""
        # TODO: Test error handling in complete flow
        pass
'''


def to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case."""
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def generate_test(config: TestConfig) -> str:
    """Generate test file content based on configuration."""
    snake_subject = to_snake_case(config.subject)

    templates = {
        "unit": UNIT_TEMPLATE,
        "integration": INTEGRATION_TEMPLATE,
        "e2e": E2E_TEMPLATE,
    }

    template = templates[config.type]

    return template.format(
        module=config.module,
        subject=config.subject,
        snake_subject=snake_subject,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate test file scaffold following FastPaip conventions"
    )
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "e2e"],
        required=True,
        help="Type of test to generate",
    )
    parser.add_argument(
        "--module",
        required=True,
        help="Bounded context module name (e.g., core, acknowledge)",
    )
    parser.add_argument(
        "--subject",
        required=True,
        help="Class or function being tested (e.g., ArticleService)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output file path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file",
    )

    args = parser.parse_args()

    config = TestConfig(
        type=args.type,
        module=args.module,
        subject=args.subject,
        output=args.output,
    )

    # Check if file exists
    if config.output.exists() and not args.force:
        print(f"Error: File already exists: {config.output}", file=sys.stderr)
        print("Use --force to overwrite", file=sys.stderr)
        sys.exit(1)

    # Generate content
    content = generate_test(config)

    # Create parent directories
    config.output.parent.mkdir(parents=True, exist_ok=True)

    # Write file
    config.output.write_text(content)

    print(f"✅ Generated {config.type} test: {config.output}")
    print("\nNext steps:")
    print("1. Fill in TODO sections")
    print("2. Add necessary imports")
    print("3. Implement test logic")
    print(f"4. Run: uv run pytest {config.output}")


if __name__ == "__main__":
    main()
