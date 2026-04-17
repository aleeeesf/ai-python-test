"""
Conftest Template for Repository Integration Tests

This template is used by the testing-conventions skill to generate conftest.py files
that automatically configure factory-boy factories with the test session.

Template Variables (replaced by skill):
- {domain_name}: Name of the domain (e.g., "call_acknowledge", "call_classification")
- {factory_imports}: Import statements for all factories
- {factory_configurations}: Lines that configure each factory with session

Example Output:
    conftest.py with autouse fixture that configures DBCallFactory, etc.
"""

# Template starts here:

"""Integration test fixtures for {domain_name} domain.

Provides database fixtures with transaction isolation for testing
infrastructure adapters against a real PostgreSQL database.
"""

import pytest

{factory_imports}


@pytest.fixture(scope="function", autouse=True)
def _configure_factories(session):
    """
    Automatically configure factory-boy factories with the test session.

    This fixture runs automatically for all tests in this directory,
    ensuring factories use the correct test database session.

    IMPORTANT: When you add new DB factories to factories.py, add them here:
    1. Import the factory at the top of this file
    2. Add a configuration line below
    """


{factory_configurations}


# Example of how to add more factories:
#
# from tests.integration.domains.{domain_name}.infrastructure.repositories.factories import (
#     DBNewModelFactory,  # Add import
# )
#
# @pytest.fixture(scope="function", autouse=True)
# def _configure_factories(session):
#     DBCallFactory._meta.sqlalchemy_session = session
#     DBNewModelFactory._meta.sqlalchemy_session = session  # Add configuration
