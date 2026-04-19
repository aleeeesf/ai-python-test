from typing import Protocol

from pydantic import BaseModel

from domain.models.request import NotificationType


class NotificationProviderResult(BaseModel):
    """Result of a notification send operation."""

    provider_id: str
    status: str


class NotificationProvider(Protocol):
    async def send(
        self,
        to: str,
        message: str,
        type: NotificationType,
    ) -> NotificationProviderResult:
        """Send a notification to a recipient.

        Args:
            to: The recipient of the notification.
            message: The message to send.
            type: The notification type.

        Returns:
            The provider response data.
        """
        ...
