"""Unit tests for ProcessWorker."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from domain.ports.notification_provider import ProviderResult
from infrastructure.workers.process_worker import ProcessWorker


class TestProcessWorker:
    """Test suite for ProcessWorker."""

    @pytest.mark.asyncio
    async def test_dispatch_creates_and_cleans_up_task(self):
        """Worker dispatches a background task and removes it when done."""
        # Arrange
        mock_repo = AsyncMock()
        mock_provider = AsyncMock()
        mock_provider.send.return_value = ProviderResult(
            provider_id="p-123", status="delivered"
        )
        worker = ProcessWorker(
            requests_repository=mock_repo,
            notification_provider=mock_provider,
        )

        # Act
        worker.dispatch("request-123")
        assert len(worker._tasks) == 1
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        # Assert - task cleaned up after execution
        assert len(worker._tasks) == 0

    @pytest.mark.asyncio
    async def test_dispatch_handles_deliver_exception(self):
        """Worker catches exceptions from deliver and cleans up task."""
        # Arrange
        mock_repo = AsyncMock()
        mock_provider = AsyncMock()
        mock_provider.send.side_effect = RuntimeError(
            "Unexpected error during delivery"
        )
        worker = ProcessWorker(
            requests_repository=mock_repo,
            notification_provider=mock_provider,
        )

        # Act - dispatch should not raise
        worker.dispatch("request-123")
        assert len(worker._tasks) == 1

        # Give task time to execute and fail
        await asyncio.sleep(0.01)
        await asyncio.sleep(0)

        # Assert - task still cleaned up despite exception
        assert len(worker._tasks) == 0

    @pytest.mark.asyncio
    async def test_dispatch_multiple_concurrent_requests(self):
        """Worker handles multiple concurrent dispatch calls."""
        # Arrange
        mock_repo = AsyncMock()
        mock_provider = AsyncMock()
        mock_provider.send.return_value = ProviderResult(
            provider_id="p-123", status="delivered"
        )
        worker = ProcessWorker(
            requests_repository=mock_repo,
            notification_provider=mock_provider,
        )

        # Act
        worker.dispatch("request-1")
        worker.dispatch("request-2")
        worker.dispatch("request-3")

        assert len(worker._tasks) == 3

        # Wait for all to complete
        await asyncio.sleep(0.01)
        await asyncio.sleep(0)

        # Assert - all tasks cleaned up
        assert len(worker._tasks) == 0
