"""Unit tests for ExternalNotificationProvider."""

import httpx
import pytest
import respx

from domain.exceptions.notification_provider import (
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderUnauthorizedError,
)
from infrastructure.providers.external_notification_provider import (
    ExternalNotificationProvider,
)


class TestExternalNotificationProvider:
    """Test suite for ExternalNotificationProvider."""

    @pytest.mark.asyncio
    async def test_returns_provider_result_when_provider_succeeds(self):
        """Provider adapter returns parsed provider result on HTTP 200."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/notify").mock(
                return_value=httpx.Response(
                    200,
                    json={"provider_id": "p-1234", "status": "delivered"},
                )
            )
            provider = ExternalNotificationProvider(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act
            result = await provider.send(
                to="user@example.com",
                message="Test notification",
                type="email",
            )

            # Assert
            assert result.provider_id == "p-1234"
            assert result.status == "delivered"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status_code", "payload", "expected_exception"),
        [
            (401, {"detail": "Invalid API key"}, ProviderUnauthorizedError),
            (429, {"detail": "Rate limit exceeded"}, ProviderRateLimitError),
            (500, {"detail": "External server error"}, ProviderServerError),
            (418, {"detail": "Unexpected"}, ProviderResponseError),
        ],
    )
    async def test_raises_expected_error_for_http_status(
        self,
        status_code,
        payload,
        expected_exception,
    ):
        """Provider adapter maps non-200 responses to domain exceptions."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/notify").mock(
                return_value=httpx.Response(status_code, json=payload)
            )
            provider = ExternalNotificationProvider(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act / Assert
            with pytest.raises(expected_exception):
                await provider.send(
                    to="user@example.com",
                    message="Test notification",
                    type="email",
                )

    @pytest.mark.asyncio
    async def test_raises_response_error_when_payload_is_invalid(self):
        """Provider adapter rejects malformed success payloads."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/notify").mock(
                return_value=httpx.Response(200, json={"status": "delivered"})
            )
            provider = ExternalNotificationProvider(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act / Assert
            with pytest.raises(ProviderResponseError):
                await provider.send(
                    to="user@example.com",
                    message="Test notification",
                    type="email",
                )

    @pytest.mark.asyncio
    async def test_raises_network_error_when_provider_times_out(self):
        """Provider adapter maps timeout exceptions to ProviderNetworkError."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/notify").mock(
                side_effect=httpx.TimeoutException("timeout")
            )
            provider = ExternalNotificationProvider(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act / Assert
            with pytest.raises(ProviderNetworkError):
                await provider.send(
                    to="user@example.com",
                    message="Test notification",
                    type="email",
                )

    @pytest.mark.asyncio
    async def test_raises_network_error_when_connection_refused(self):
        """Provider adapter maps connection errors to ProviderNetworkError."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/notify").mock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            provider = ExternalNotificationProvider(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act / Assert
            with pytest.raises(ProviderNetworkError):
                await provider.send(
                    to="user@example.com",
                    message="Test notification",
                    type="email",
                )

    @pytest.mark.asyncio
    async def test_raises_network_error_when_read_timeout(self):
        """Provider adapter maps read timeout to ProviderNetworkError."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/notify").mock(
                side_effect=httpx.ReadTimeout("Read timeout")
            )
            provider = ExternalNotificationProvider(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act / Assert
            with pytest.raises(ProviderNetworkError):
                await provider.send(
                    to="user@example.com",
                    message="Test notification",
                    type="email",
                )

    @pytest.mark.asyncio
    async def test_raises_response_error_when_json_decode_fails(self):
        """Provider adapter handles malformed JSON response from provider."""
        # Arrange
        async with respx.mock:
            respx.post("http://provider:3001/v1/notify").mock(
                return_value=httpx.Response(
                    200, content=b"invalid", extensions={"http_version": b"HTTP/1.1"}
                )
            )
            provider = ExternalNotificationProvider(
                api_url="http://provider:3001",
                api_key="test-dev-2026",
            )

            # Act / Assert
            with pytest.raises(ProviderResponseError):
                await provider.send(
                    to="user@example.com",
                    message="Test notification",
                    type="email",
                )
