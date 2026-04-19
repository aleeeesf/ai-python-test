"""Use case for processing notification requests."""

import asyncio
from datetime import datetime

from application.dtos import StartProcessResultDTO
from core.logging import get_logger
from domain.exceptions.ai_extractor import AIExtractionError
from domain.exceptions.notification_provider import (
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderUnauthorizedError,
)
from domain.models.request import NotificationRequest, NotificationStatus
from domain.ports.ai_extractor import AIExtractor
from domain.ports.notification_provider import (
    NotificationProvider,
    NotificationProviderResult,
)
from domain.ports.requests_repository import RequestsRepository

logger = get_logger(__name__)


async def start_process_request(
    request_id: str,
    requests_repository: RequestsRepository,
) -> StartProcessResultDTO:
    """
    Start processing a notification request.

    Validates request existence and status, then transitions to PROCESSING.

    Args:
        request_id: The ID of the request to start processing.
        requests_repository: The repository for persisting requests.

    Returns:
        StartProcessResultDTO with processing result and new status.
    """
    request = await requests_repository.get_by_id(request_id)
    if request is None:
        logger.warning(f"Start failed | request_id={request_id} | reason=not_found")
        return StartProcessResultDTO(
            found=False,
            should_process=False,
            status=None,
        )

    if (
        request.status == NotificationStatus.SENT
        or request.status == NotificationStatus.PROCESSING
    ):
        logger.info(
            f"Start skipped | request_id={request_id} | current_status={request.status.value} | reason=already_processed"
        )
        return StartProcessResultDTO(
            found=True,
            should_process=False,
            status=request.status,
        )

    request.status = NotificationStatus.PROCESSING
    request.updated_at = datetime.now()
    request.error = None
    await requests_repository.update(request)
    logger.info(
        f"Processing started | request_id={request_id} | user_input_len={len(request.user_input)}"
    )
    return StartProcessResultDTO(
        found=True,
        should_process=True,
        status=request.status,
    )


async def deliver_request(
    request_id: str,
    requests_repository: RequestsRepository,
    ai_extractor: AIExtractor,
    notification_provider: NotificationProvider,
) -> None:
    """
    Deliver a notification request to the external provider.

    Orchestrates AI extraction, validation, and provider delivery with retry logic.

    Args:
        request_id: The ID of the request to deliver.
        requests_repository: The repository for persisting requests.
        ai_extractor: The AI service for extracting info from user input.
        notification_provider: The external provider client.

    Returns:
        None. Updates request status as side effect.
    """
    request = await requests_repository.get_by_id(request_id)
    if request is None:
        logger.warning(f"Deliver skipped | request_id={request_id} | reason=not_found")
        return
    if request.status != NotificationStatus.PROCESSING:
        logger.debug(
            f"Deliver skipped | request_id={request_id} | current_status={request.status.value} | reason=not_processing"
        )
        return

    # Step 1: Extract information from natural language using AI
    try:
        extracted = await ai_extractor.extract(request.user_input)
        request.to = extracted.to
        request.message = extracted.message
        request.type = extracted.type
        request.updated_at = datetime.now()
        await requests_repository.update(request)
        logger.info(
            f"AI extraction succeeded | request_id={request_id} | to={request.to} | type={request.type}"
        )
    except AIExtractionError as error:
        request.status = NotificationStatus.FAILED
        request.error = f"AI extraction failed: {error!s}"
        request.updated_at = datetime.now()
        await requests_repository.update(request)
        logger.error(
            f"AI extraction failed | request_id={request_id} | error={error!s}"
        )
        return
    except Exception as error:
        request.status = NotificationStatus.FAILED
        request.error = f"Unexpected AI extraction error: {error}"
        request.updated_at = datetime.now()
        await requests_repository.update(request)
        logger.error(
            f"AI extraction failed (unexpected) | request_id={request_id} | error={error!s}"
        )
        return

    # Step 2: Send notification to external provider with retries
    try:
        provider_result = await _send_with_retries(request, notification_provider)
    except (
        ProviderUnauthorizedError,
        ProviderResponseError,
        ProviderNetworkError,
        ProviderRateLimitError,
        ProviderServerError,
    ) as error:
        request.status = NotificationStatus.FAILED
        request.error = str(error)
        request.updated_at = datetime.now()
        await requests_repository.update(request)
        logger.error(
            f"Provider delivery failed | request_id={request_id} | to={request.to} | error={error!s}"
        )
        return
    except Exception as error:
        request.status = NotificationStatus.FAILED
        request.error = f"Unexpected provider error: {error}"
        request.updated_at = datetime.now()
        await requests_repository.update(request)
        logger.error(
            f"Provider delivery failed (unexpected) | request_id={request_id} | to={request.to} | error={error!s}"
        )
        return

    request.status = NotificationStatus.SENT
    request.provider_id = provider_result.provider_id
    request.error = None
    request.updated_at = datetime.now()
    await requests_repository.update(request)
    logger.info(
        f"Delivery succeeded | request_id={request_id} | to={request.to} | provider_id={provider_result.provider_id}"
    )


async def _send_with_retries(
    request: NotificationRequest,
    notification_provider: NotificationProvider,
) -> NotificationProviderResult:
    """
    Send a notification request to the external provider with retries.

    Retryable errors: rate limit, server error, network error.
    Non-retryable errors: unauthorized, response error.

    Args:
        request: The request to send.
        notification_provider: The external provider client.

    Returns:
        The result of the send operation.

    Raises:
        ProviderError: If all retries are exhausted or non-retryable error occurs.
    """
    retry_delays_seconds = (0.2, 0.5, 1.0)
    attempts = len(retry_delays_seconds) + 1

    # Ensure extracted fields are present before attempting delivery
    if not request.to or not request.message or not request.type:
        raise AIExtractionError("Missing extracted data for notification delivery")

    for attempt in range(attempts):
        try:
            result = await notification_provider.send(
                to=request.to,
                message=request.message,
                type=request.type,
            )
            logger.debug(
                f"Provider call succeeded | request_id={request.id} | attempt={attempt + 1}"
            )
            return result
        except (
            ProviderRateLimitError,
            ProviderServerError,
            ProviderNetworkError,
        ) as error:  # retryable errors
            if attempt >= len(retry_delays_seconds):
                logger.warning(
                    f"Retry exhausted | request_id={request.id} | attempts={attempt + 1} | error={error!s}"
                )
                raise
            logger.warning(
                f"Retryable error | request_id={request.id} | attempt={attempt + 1} | error={error!s} | retry_in={retry_delays_seconds[attempt]}s"
            )
            await asyncio.sleep(retry_delays_seconds[attempt])
        except (
            ProviderUnauthorizedError,
            ProviderResponseError,
        ) as error:  # non-retryable errors
            logger.error(
                f"Non-retryable error | request_id={request.id} | attempt={attempt + 1} | error={error!s}"
            )
            raise

    raise RuntimeError("Retry loop exited without returning a provider result")
