"""Unit tests for InMemoryRequestsRepository."""

import pytest

from domain.entities.request import NotificationStatus
from infrastructure.repositories.in_memory_requests_repository import (
    InMemoryRequestsRepository,
)


class TestInMemoryRequestsRepository:
    """Test suite for InMemoryRequestsRepository."""

    @pytest.mark.asyncio
    async def test_saves_and_retrieves_request(self, queued_request):
        """Repository returns a saved request by ID."""
        # Arrange
        repository = InMemoryRequestsRepository()

        # Act
        await repository.save(queued_request)
        stored_request = await repository.get_by_id(queued_request.id)

        # Assert
        assert stored_request is not None
        assert stored_request.id == queued_request.id
        assert stored_request.status == queued_request.status

    @pytest.mark.asyncio
    async def test_returns_none_when_request_does_not_exist(self):
        """Repository returns None for unknown request IDs."""
        # Arrange
        repository = InMemoryRequestsRepository()

        # Act
        result = await repository.get_by_id("missing-request")

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_updates_existing_request(self, queued_request):
        """Repository overwrites request fields on update."""
        # Arrange
        repository = InMemoryRequestsRepository()
        await repository.save(queued_request)
        queued_request.status = NotificationStatus.PROCESSING
        queued_request.error = None

        # Act
        await repository.update(queued_request)
        updated_request = await repository.get_by_id(queued_request.id)

        # Assert
        assert updated_request is not None
        assert updated_request.status == NotificationStatus.PROCESSING
