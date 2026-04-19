"""Shared fixtures for unit and integration tests.

IMPORTANT: Fixtures reset state between tests to prevent pollution.
If you add new attributes to Fake* classes, MUST add cleanup logic here.

Patterns:
- Fakes: Deterministic, testable versions of domain objects
- Stubs: Simple implementations recording calls without side effects
- InMemory: Real implementations using in-process storage (not mocks)
"""

import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure tests resolve imports from the app runtime root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.dependencies import (
    get_ai_extractor,
    get_notification_provider,
    get_process_dispatcher,
    get_requests_repository,
)
from domain.exceptions.ai_extractor import AIExtractionError
from domain.exceptions.notification_provider import (
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderUnauthorizedError,
)
from domain.models.request import NotificationRequest, NotificationStatus
from domain.ports.ai_extractor import AIExtractedInfo
from domain.ports.notification_provider import NotificationProviderResult
from domain.ports.process_dispatcher import ProcessDispatcher
from infrastructure.repositories.in_memory_requests_repository import (
    InMemoryRequestsRepository,
)
from main import app as fastapi_app


class FakeAIExtractor:
    """Deterministic fake AI extractor for testing.

    Allows configuring extraction success/failure scenarios per test.
    Records all extract() calls for verification.
    """

    def __init__(self) -> None:
        self.result: AIExtractedInfo = AIExtractedInfo(
            to="extracted@example.com",
            message="Extracted message",
            type="email",
        )
        self.side_effects: list[Exception] = []
        self.calls: list[str] = []

    def reset(self) -> None:
        """Reset state between tests."""
        self.side_effects.clear()
        self.calls.clear()

    async def extract(self, user_input: str) -> AIExtractedInfo:
        """Record the call and return the configured response."""
        self.calls.append(user_input)
        if self.side_effects:
            raise self.side_effects.pop(0)
        return self.result


class FakeNotificationProvider:
    """Deterministic fake provider for testing.

    Allows configuring success/failure scenarios per test.
    Records all send() calls for verification.
    """

    def __init__(self) -> None:
        self.result = NotificationProviderResult(
            provider_id="p-1234", status="delivered"
        )
        self.side_effects: list[Exception] = []
        self.calls: list[dict[str, str]] = []

    def reset(self) -> None:
        """Reset state between tests."""
        self.side_effects.clear()
        self.calls.clear()

    async def send(
        self, to: str, message: str, type: str
    ) -> NotificationProviderResult:
        """Record the call and return the configured response."""
        self.calls.append(
            {
                "to": to,
                "message": message,
                "type": type,
            }
        )
        if self.side_effects:
            raise self.side_effects.pop(0)
        return self.result


class StubProcessDispatcher(ProcessDispatcher):
    """Stub dispatcher recording requests without executing background work.

    Used for integration tests to avoid async background noise.
    """

    def __init__(self) -> None:
        self.dispatched_request_ids: list[str] = []

    def reset(self) -> None:
        """Reset state between tests."""
        self.dispatched_request_ids.clear()

    def dispatch(self, request_id: str) -> None:
        """Record dispatched request IDs."""
        self.dispatched_request_ids.append(request_id)


# ==============================================================================
# Repository Fixtures
# ==============================================================================


@pytest.fixture(name="requests_repository")
def _requests_repository_fixture() -> InMemoryRequestsRepository:
    """In-memory repository used by tests."""
    return InMemoryRequestsRepository()


# ==============================================================================
# Provider Fixtures
# ==============================================================================


@pytest.fixture(name="fake_ai_extractor")
def _fake_ai_extractor_fixture() -> Generator[FakeAIExtractor, None, None]:
    """Fake AI extractor.

    Resets state after each test to prevent state pollution.
    """
    extractor = FakeAIExtractor()
    yield extractor
    extractor.reset()


@pytest.fixture(name="fake_notification_provider")
def _fake_notification_provider_fixture() -> Generator[
    FakeNotificationProvider, None, None
]:
    """Fake external notification provider.

    Resets state after each test to prevent state pollution.
    """
    provider = FakeNotificationProvider()
    yield provider
    provider.reset()


@pytest.fixture(name="stub_process_dispatcher")
def _stub_process_dispatcher_fixture() -> Generator[StubProcessDispatcher, None, None]:
    """Dispatcher stub for route integration tests.

    Resets state after each test to prevent state pollution.
    """
    dispatcher = StubProcessDispatcher()
    yield dispatcher
    dispatcher.reset()


# ==============================================================================
# Data Fixtures
# ==============================================================================


@pytest.fixture(name="queued_request")
def _queued_request_fixture() -> NotificationRequest:
    """Queued request ready to be processed."""
    return NotificationRequest(
        id="request-queued",
        user_input="Manda un mail a user@example.com diciendo Test notification",
        to=None,
        message=None,
        type=None,
        status=NotificationStatus.QUEUED,
    )


@pytest.fixture(name="processing_request")
def _processing_request_fixture() -> NotificationRequest:
    """Request already in processing state."""
    return NotificationRequest(
        id="request-processing",
        user_input="Manda un mail a user@example.com diciendo Test notification",
        to="user@example.com",
        message="Test notification",
        type="email",
        status=NotificationStatus.PROCESSING,
    )


@pytest.fixture(name="sent_request")
def _sent_request_fixture() -> NotificationRequest:
    """Request already delivered successfully."""
    return NotificationRequest(
        id="request-sent",
        user_input="Manda un mail a user@example.com diciendo Test notification",
        to="user@example.com",
        message="Test notification",
        type="email",
        status=NotificationStatus.SENT,
        provider_id="p-9999",
    )


@pytest.fixture(name="failed_request")
def _failed_request_fixture() -> NotificationRequest:
    """Request that failed previously."""
    return NotificationRequest(
        id="request-failed",
        user_input="Manda un mail a user@example.com diciendo Test notification",
        to="user@example.com",
        message="Test notification",
        type="email",
        status=NotificationStatus.FAILED,
        error="Previous failure",
    )


@pytest.fixture(name="ai_errors")
def _ai_errors_fixture() -> dict[str, Exception]:
    """AI extraction exception catalog used by tests."""
    return {
        "validation": AIExtractionError("Invalid extraction response"),
    }


@pytest.fixture(name="provider_errors")
def _provider_errors_fixture() -> dict[str, Exception]:
    """Provider exception catalog used by tests."""
    return {
        "unauthorized": ProviderUnauthorizedError("Invalid API key"),
        "rate_limit": ProviderRateLimitError("Rate limit exceeded"),
        "server": ProviderServerError("External server error"),
        "network": ProviderNetworkError("Provider timeout"),
        "response": ProviderResponseError("Invalid provider response payload"),
    }


# ==============================================================================
# Integration Fixtures
# ==============================================================================


@pytest.fixture(name="client")
def _client_fixture(
    requests_repository: InMemoryRequestsRepository,
    fake_ai_extractor: FakeAIExtractor,
    fake_notification_provider: FakeNotificationProvider,
    stub_process_dispatcher: StubProcessDispatcher,
) -> Generator[TestClient, None, None]:
    """FastAPI test client with dependency overrides.

    Wires fake/stub implementations into FastAPI dependency injection.
    Clears overrides after yielding to prevent cross-contamination.
    """
    # Override repository, provider, and dispatcher dependencies
    fastapi_app.dependency_overrides[get_requests_repository] = lambda: (
        requests_repository
    )
    fastapi_app.dependency_overrides[get_ai_extractor] = lambda: fake_ai_extractor
    fastapi_app.dependency_overrides[get_notification_provider] = lambda: (
        fake_notification_provider
    )
    fastapi_app.dependency_overrides[get_process_dispatcher] = lambda: (
        stub_process_dispatcher
    )

    with TestClient(fastapi_app) as test_client:
        yield test_client

    fastapi_app.dependency_overrides.clear()
