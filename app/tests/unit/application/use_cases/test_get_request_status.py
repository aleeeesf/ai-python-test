"""Unit tests for get_request_status function."""

import pytest

from application.use_cases.get_request_status import get_request_status


class TestGetRequestStatus:
    """Test suite for get_request_status function."""

    @pytest.mark.asyncio
    async def test_returns_none_when_request_does_not_exist(self, requests_repository):
        """Status function returns None for unknown request IDs."""
        # Act
        result = await get_request_status("missing-request", requests_repository)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_status_when_request_exists(
        self,
        requests_repository,
        queued_request,
    ):
        """Status function returns request ID and status when request exists."""
        # Arrange
        await requests_repository.save(queued_request)

        # Act
        result = await get_request_status(queued_request.id, requests_repository)

        # Assert
        assert result is not None
        assert result.id == queued_request.id
        assert result.status == queued_request.status
