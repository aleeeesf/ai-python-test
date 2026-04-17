import httpx

from domain.entities.request import NotificationType
from domain.exceptions.provider import (
    ProviderNetworkError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderUnauthorizedError,
)
from domain.ports.notification_provider import NotificationProvider, ProviderResult


class ExternalNotificationProvider(NotificationProvider):
    def __init__(self, api_url: str, api_key: str) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key

    async def send(
        self,
        to: str,
        message: str,
        type: NotificationType,
    ) -> ProviderResult:
        """
        Send a notification to a recipient using the external provider.

        Args:
            to: The recipient of the notification.
            message: The message to send.
            type: The notification type.

        Returns:
            ProviderResult: The result of the notification send operation.

        Raises:
            ProviderNetworkError: If there is a network error when connecting to the provider.
            ProviderUnauthorizedError: If the API key is invalid.
            ProviderRateLimitError: If the provider rate limit is exceeded.
            ProviderServerError: If the provider returns a server error.
            ProviderResponseError: If the provider returns an unexpected response.
        """
        headers = {
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "to": to,
            "message": message,
            "type": type,
        }
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.post(
                    f"{self._api_url}/v1/notify",
                    json=payload,
                    headers=headers,
                )
        except httpx.TimeoutException as error:
            raise ProviderNetworkError("Provider timeout") from error
        except httpx.HTTPError as error:
            raise ProviderNetworkError("Provider connection error") from error

        if response.status_code == 200:
            try:
                data = response.json()
            except (ValueError, KeyError) as error:
                raise ProviderResponseError(
                    "Invalid JSON response from provider"
                ) from error
            provider_id = data.get("provider_id")
            status = data.get("status")
            if not isinstance(provider_id, str) or not isinstance(status, str):
                raise ProviderResponseError("Invalid provider response payload")
            return ProviderResult(provider_id=provider_id, status=status)

        if response.status_code == 401:
            raise ProviderUnauthorizedError("Invalid API key for external provider")
        if response.status_code == 429:
            raise ProviderRateLimitError("External provider rate limit exceeded")
        if response.status_code >= 500:
            raise ProviderServerError("External provider server error")
        raise ProviderResponseError(
            f"Unexpected provider status code: {response.status_code}"
        )
