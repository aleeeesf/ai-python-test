"""Unit tests for create_request function."""

import pytest

from application.dtos import CreateRequestDTO
from application.use_cases.create_request import create_request
from domain.models.request import NotificationStatus


class TestCreateRequest:
    """Test suite for create_request function."""

    @pytest.mark.asyncio
    async def test_creates_request_when_valid_input(self, requests_repository):
        """Create function persists a queued request with user input and returns its ID."""
        # Arrange
        user_input = "Manda un mail a user@example.com diciendo Test notification"
        create_request_dto = CreateRequestDTO(user_input=user_input)

        # Act
        request_id = await create_request(create_request_dto, requests_repository)
        stored_request = await requests_repository.get_by_id(request_id)

        # Assert
        assert request_id
        assert stored_request is not None
        assert stored_request.id == request_id
        assert stored_request.user_input == user_input
        assert stored_request.to is None
        assert stored_request.message is None
        assert stored_request.type is None
        assert stored_request.status == NotificationStatus.QUEUED
