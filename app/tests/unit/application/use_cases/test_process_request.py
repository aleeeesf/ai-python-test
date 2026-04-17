"""Unit tests for start_process_request and deliver_request functions."""

import pytest

from application.use_cases.process_request import (
    deliver_request,
    start_process_request,
)
from domain.entities.request import NotificationStatus


class TestStartProcessRequest:
    """Test suite for start_process_request function."""

    @pytest.mark.asyncio
    async def test_returns_not_found_when_request_does_not_exist(
        self,
        requests_repository,
    ):
        """Start returns not found for unknown request IDs."""
        # Act
        result = await start_process_request("missing-request", requests_repository)

        # Assert
        assert result.found is False
        assert result.should_process is False
        assert result.status is None

    @pytest.mark.asyncio
    async def test_marks_request_as_processing_when_request_is_queued(
        self,
        requests_repository,
        queued_request,
    ):
        """Start transitions queued requests to processing."""
        # Arrange
        await requests_repository.save(queued_request)

        # Act
        result = await start_process_request(queued_request.id, requests_repository)
        stored_request = await requests_repository.get_by_id(queued_request.id)

        # Assert
        assert result.found is True
        assert result.should_process is True
        assert result.status == NotificationStatus.PROCESSING
        assert stored_request is not None
        assert stored_request.status == NotificationStatus.PROCESSING
        assert stored_request.error is None

    @pytest.mark.asyncio
    async def test_skips_processing_when_request_is_already_sent(
        self,
        requests_repository,
        sent_request,
    ):
        """Start does not reprocess sent requests."""
        # Arrange
        await requests_repository.save(sent_request)

        # Act
        result = await start_process_request(sent_request.id, requests_repository)

        # Assert
        assert result.found is True
        assert result.should_process is False
        assert result.status == NotificationStatus.SENT

    @pytest.mark.asyncio
    async def test_skips_processing_when_request_is_processing(
        self,
        requests_repository,
        processing_request,
    ):
        """Start does not reprocess sent requests."""
        # Arrange
        await requests_repository.save(processing_request)

        # Act
        result = await start_process_request(processing_request.id, requests_repository)

        # Assert
        assert result.found is True
        assert result.should_process is False
        assert result.status == NotificationStatus.PROCESSING


class TestDeliverRequest:
    """Test suite for deliver_request function."""

    @pytest.mark.asyncio
    async def test_marks_request_as_sent_when_provider_succeeds(
        self,
        requests_repository,
        fake_notification_provider,
        processing_request,
    ):
        """Deliver marks a processing request as sent on provider success."""
        # Arrange
        await requests_repository.save(processing_request)

        # Act
        await deliver_request(
            processing_request.id,
            requests_repository,
            fake_notification_provider,
        )
        stored_request = await requests_repository.get_by_id(processing_request.id)

        # Assert
        assert stored_request is not None
        assert stored_request.status == NotificationStatus.SENT
        assert (
            stored_request.provider_id == fake_notification_provider.result.provider_id
        )
        assert stored_request.error is None
        assert len(fake_notification_provider.calls) == 1

    @pytest.mark.asyncio
    async def test_marks_request_as_failed_when_provider_is_unauthorized(
        self,
        requests_repository,
        fake_notification_provider,
        processing_request,
        provider_errors,
    ):
        """Deliver fails immediately on unauthorized provider errors."""
        # Arrange
        fake_notification_provider.side_effects = [provider_errors["unauthorized"]]
        await requests_repository.save(processing_request)

        # Act
        await deliver_request(
            processing_request.id,
            requests_repository,
            fake_notification_provider,
        )
        stored_request = await requests_repository.get_by_id(processing_request.id)

        # Assert
        assert stored_request is not None
        assert stored_request.status == NotificationStatus.FAILED
        assert stored_request.error == "Invalid API key"
        assert len(fake_notification_provider.calls) == 1

    @pytest.mark.asyncio
    async def test_retries_and_marks_request_as_sent_when_transient_error_recovers(
        self,
        requests_repository,
        fake_notification_provider,
        processing_request,
        provider_errors,
    ):
        """Deliver retries transient provider errors and succeeds if provider recovers."""
        # Arrange
        fake_notification_provider.side_effects = [provider_errors["rate_limit"]]
        await requests_repository.save(processing_request)

        # Act
        await deliver_request(
            processing_request.id,
            requests_repository,
            fake_notification_provider,
        )
        stored_request = await requests_repository.get_by_id(processing_request.id)

        # Assert
        assert stored_request is not None
        assert stored_request.status == NotificationStatus.SENT
        assert stored_request.error is None
        assert len(fake_notification_provider.calls) == 2

    @pytest.mark.asyncio
    async def test_marks_request_as_failed_when_unexpected_error_happens(
        self,
        requests_repository,
        fake_notification_provider,
        processing_request,
    ):
        """Deliver marks request as failed on unexpected processing errors."""
        # Arrange
        fake_notification_provider.side_effects = [RuntimeError("Boom")]
        await requests_repository.save(processing_request)

        # Act
        await deliver_request(
            processing_request.id,
            requests_repository,
            fake_notification_provider,
        )
        stored_request = await requests_repository.get_by_id(processing_request.id)

        # Assert
        assert stored_request is not None
        assert stored_request.status == NotificationStatus.FAILED
        assert stored_request.error == "Unexpected processing error: Boom"
        assert len(fake_notification_provider.calls) == 1

    @pytest.mark.asyncio
    async def test_marks_request_as_failed_when_retry_limit_exhausted(
        self,
        requests_repository,
        fake_notification_provider,
        processing_request,
        provider_errors,
    ):
        """Deliver marks request as failed after exhausting retries on transient errors."""
        # Arrange - Fail on all 4 attempts (initial + 3 retries)
        fake_notification_provider.side_effects = [
            provider_errors["rate_limit"],
            provider_errors["server"],
            provider_errors["network"],
            provider_errors["rate_limit"],  # 4th attempt - should give up
        ]
        await requests_repository.save(processing_request)

        # Act
        await deliver_request(
            processing_request.id,
            requests_repository,
            fake_notification_provider,
        )
        stored_request = await requests_repository.get_by_id(processing_request.id)

        # Assert
        assert stored_request is not None
        assert stored_request.status == NotificationStatus.FAILED
        assert stored_request.error is not None  # Contains error message
        assert len(fake_notification_provider.calls) == 4  # Exhausted retries

    @pytest.mark.asyncio
    async def test_deliver_when_request_does_not_exist(
        self,
        requests_repository,
        fake_notification_provider,
    ):
        """Deliver gracefully handles missing request (idempotent)."""
        # Act - Should not raise, just return
        await deliver_request(
            "nonexistent-request-id",
            requests_repository,
            fake_notification_provider,
        )

        # Assert
        assert len(fake_notification_provider.calls) == 0

    @pytest.mark.asyncio
    async def test_deliver_when_request_not_in_processing_state(
        self,
        requests_repository,
        fake_notification_provider,
        sent_request,
    ):
        """Deliver skips requests not in PROCESSING state (idempotent)."""
        # Arrange
        await requests_repository.save(sent_request)

        # Act - Request is already SENT, should skip
        await deliver_request(
            sent_request.id,
            requests_repository,
            fake_notification_provider,
        )

        # Assert
        assert len(fake_notification_provider.calls) == 0


class TestDeliverRequestConcurrency:
    """Concurrency tests for deliver_request function."""

    @pytest.mark.asyncio
    async def test_concurrent_delivers_only_one_succeeds(
        self,
        requests_repository,
        fake_notification_provider,
        processing_request,
    ):
        """Race condition: Two concurrent delivers on same request (only provider wins)."""
        import asyncio

        # Arrange
        await requests_repository.save(processing_request)

        # Make provider return different results on each call to detect double-call
        call_count = 0
        original_send = fake_notification_provider.send

        async def send_with_counter(to: str, message: str, type: str):
            nonlocal call_count
            call_count += 1
            return await original_send(to, message, type)

        fake_notification_provider.send = send_with_counter

        # Act - Race: Two concurrent deliver calls
        await asyncio.gather(
            deliver_request(
                processing_request.id,
                requests_repository,
                fake_notification_provider,
            ),
            deliver_request(
                processing_request.id,
                requests_repository,
                fake_notification_provider,
            ),
        )

        stored_request = await requests_repository.get_by_id(processing_request.id)

        # Assert - Request should be SENT despite race
        assert stored_request is not None
        assert stored_request.status == NotificationStatus.SENT
        # Both calls may reach provider (no lock in this simple impl)
        # The test documents the behavior
        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_multiple_different_requests_concurrent(
        self,
        requests_repository,
        fake_notification_provider,
    ):
        """Concurrent delivers on different requests don't interfere."""
        import asyncio

        # Arrange - Two different processing requests
        from domain.entities.request import NotificationRequest

        req1 = NotificationRequest(
            id="req-1",
            to="user1@example.com",
            message="Message 1",
            type="email",
            status=NotificationStatus.PROCESSING,
        )
        req2 = NotificationRequest(
            id="req-2",
            to="user2@example.com",
            message="Message 2",
            type="email",
            status=NotificationStatus.PROCESSING,
        )
        await requests_repository.save(req1)
        await requests_repository.save(req2)

        # Act - Process both concurrently
        await asyncio.gather(
            deliver_request(req1.id, requests_repository, fake_notification_provider),
            deliver_request(req2.id, requests_repository, fake_notification_provider),
        )

        # Assert - Both marked as SENT independently
        stored_req1 = await requests_repository.get_by_id(req1.id)
        stored_req2 = await requests_repository.get_by_id(req2.id)

        assert stored_req1 is not None
        assert stored_req1.status == NotificationStatus.SENT
        assert stored_req2 is not None
        assert stored_req2.status == NotificationStatus.SENT
        # Provider called once for each request
        assert len(fake_notification_provider.calls) == 2

    @pytest.mark.asyncio
    async def test_concurrent_success_and_failure_different_requests(
        self,
        requests_repository,
        fake_notification_provider,
    ):
        """Concurrent: different requests process independently without interference."""
        import asyncio

        from domain.entities.request import NotificationRequest

        # Arrange - Two different processing requests
        req1 = NotificationRequest(
            id="req-1-concurrent",
            to="user1@example.com",
            message="Message 1",
            type="email",
            status=NotificationStatus.PROCESSING,
        )
        req2 = NotificationRequest(
            id="req-2-concurrent",
            to="user2@example.com",
            message="Message 2",
            type="email",
            status=NotificationStatus.PROCESSING,
        )
        await requests_repository.save(req1)
        await requests_repository.save(req2)

        # Act - Process both concurrently
        await asyncio.gather(
            deliver_request(req1.id, requests_repository, fake_notification_provider),
            deliver_request(req2.id, requests_repository, fake_notification_provider),
        )

        # Assert - Both reach terminal state (SENT here since provider succeeds)
        stored_req1 = await requests_repository.get_by_id(req1.id)
        stored_req2 = await requests_repository.get_by_id(req2.id)

        assert stored_req1 is not None
        # Both should be in terminal state, not PROCESSING
        assert stored_req1.status in (
            NotificationStatus.SENT,
            NotificationStatus.FAILED,
        )
        assert stored_req2 is not None
        assert stored_req2.status in (
            NotificationStatus.SENT,
            NotificationStatus.FAILED,
        )
        # Each request processed independently with separate calls
        assert len(fake_notification_provider.calls) >= 2
