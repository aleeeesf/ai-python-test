"""Unit tests for start_process_request and deliver_request functions."""

import pytest
from application.use_cases.process_request import (
    deliver_request,
    start_process_request,
)
from domain.models.request import NotificationStatus


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
        """Start does not reprocess requests already being processed."""
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
    async def test_extracts_info_and_marks_request_as_sent_on_success(
        self,
        requests_repository,
        fake_ai_extractor,
        fake_notification_provider,
        processing_request,
    ):
        """Deliver extracts AI data, sends notification, and marks request as sent."""
        # Arrange
        await requests_repository.save(processing_request)

        # Act
        await deliver_request(
            processing_request.id,
            requests_repository,
            fake_ai_extractor,
            fake_notification_provider,
        )
        stored_request = await requests_repository.get_by_id(processing_request.id)

        # Assert
        assert stored_request is not None
        assert stored_request.status == NotificationStatus.SENT
        assert stored_request.to == fake_ai_extractor.result.to
        assert stored_request.message == fake_ai_extractor.result.message
        assert stored_request.type == fake_ai_extractor.result.type
        assert (
            stored_request.provider_id == fake_notification_provider.result.provider_id
        )
        assert stored_request.error is None
        assert len(fake_ai_extractor.calls) == 1
        assert len(fake_notification_provider.calls) == 1

    @pytest.mark.asyncio
    async def test_marks_request_as_failed_when_ai_extraction_fails(
        self,
        requests_repository,
        fake_ai_extractor,
        fake_notification_provider,
        processing_request,
        ai_errors,
    ):
        """Deliver marks request as failed when AI extraction fails."""
        # Arrange
        fake_ai_extractor.side_effects = [ai_errors["validation"]]
        await requests_repository.save(processing_request)

        # Act
        await deliver_request(
            processing_request.id,
            requests_repository,
            fake_ai_extractor,
            fake_notification_provider,
        )
        stored_request = await requests_repository.get_by_id(processing_request.id)

        # Assert
        assert stored_request is not None
        assert stored_request.status == NotificationStatus.FAILED
        assert "AI extraction failed" in stored_request.error
        assert len(fake_ai_extractor.calls) == 1
        assert len(fake_notification_provider.calls) == 0  # Never reached provider

    @pytest.mark.asyncio
    async def test_marks_request_as_failed_when_provider_is_unauthorized(
        self,
        requests_repository,
        fake_ai_extractor,
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
            fake_ai_extractor,
            fake_notification_provider,
        )
        stored_request = await requests_repository.get_by_id(processing_request.id)

        # Assert
        assert stored_request is not None
        assert stored_request.status == NotificationStatus.FAILED
        assert stored_request.error == "Invalid API key"
        assert len(fake_ai_extractor.calls) == 1
        assert len(fake_notification_provider.calls) == 1

    @pytest.mark.asyncio
    async def test_retries_and_marks_request_as_sent_when_transient_error_recovers(
        self,
        requests_repository,
        fake_ai_extractor,
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
            fake_ai_extractor,
            fake_notification_provider,
        )
        stored_request = await requests_repository.get_by_id(processing_request.id)

        # Assert
        assert stored_request is not None
        assert stored_request.status == NotificationStatus.SENT
        assert stored_request.error is None
        assert len(fake_ai_extractor.calls) == 1
        assert len(fake_notification_provider.calls) == 2

    @pytest.mark.asyncio
    async def test_marks_request_as_failed_when_unexpected_error_happens(
        self,
        requests_repository,
        fake_ai_extractor,
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
            fake_ai_extractor,
            fake_notification_provider,
        )
        stored_request = await requests_repository.get_by_id(processing_request.id)

        # Assert
        assert stored_request is not None
        assert stored_request.status == NotificationStatus.FAILED
        assert "Unexpected provider error" in stored_request.error
        assert len(fake_ai_extractor.calls) == 1
        assert len(fake_notification_provider.calls) == 1

    @pytest.mark.asyncio
    async def test_marks_request_as_failed_when_retry_limit_exhausted(
        self,
        requests_repository,
        fake_ai_extractor,
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
            fake_ai_extractor,
            fake_notification_provider,
        )
        stored_request = await requests_repository.get_by_id(processing_request.id)

        # Assert
        assert stored_request is not None
        assert stored_request.status == NotificationStatus.FAILED
        assert stored_request.error is not None  # Contains error message
        assert len(fake_ai_extractor.calls) == 1
        assert len(fake_notification_provider.calls) == 4  # Exhausted retries

    @pytest.mark.asyncio
    async def test_deliver_when_request_does_not_exist(
        self,
        requests_repository,
        fake_ai_extractor,
        fake_notification_provider,
    ):
        """Deliver gracefully handles missing request (idempotent)."""
        # Act - Should not raise, just return
        await deliver_request(
            "nonexistent-request-id",
            requests_repository,
            fake_ai_extractor,
            fake_notification_provider,
        )

        # Assert
        assert len(fake_ai_extractor.calls) == 0
        assert len(fake_notification_provider.calls) == 0

    @pytest.mark.asyncio
    async def test_deliver_when_request_not_in_processing_state(
        self,
        requests_repository,
        fake_ai_extractor,
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
            fake_ai_extractor,
            fake_notification_provider,
        )

        # Assert
        assert len(fake_ai_extractor.calls) == 0
        assert len(fake_notification_provider.calls) == 0
