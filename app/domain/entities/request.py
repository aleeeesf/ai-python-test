from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

NotificationType = Literal["email", "sms", "push"]


class NotificationStatus(Enum):
    """Status of a notification request."""

    QUEUED = "queued"
    PROCESSING = "processing"
    SENT = "sent"
    FAILED = "failed"


class NotificationRequest(BaseModel):
    """Notification request persisted and processed by the service."""

    id: str = Field(
        ...,
        min_length=1,
        examples=["123e4567-e89b-12d3-a456-426614174000"],
        description="The ID of the notification request.",
    )
    to: str = Field(
        ...,
        min_length=1,
        examples=["user@example.com"],
        description="The recipient of the notification.",
    )
    message: str = Field(
        ...,
        min_length=1,
        examples=["Your verification code is 1234"],
        description="The message to send.",
    )
    type: NotificationType = Field(
        ..., examples=["email"], description="The type of notification."
    )
    status: NotificationStatus = Field(
        ...,
        examples=[NotificationStatus.QUEUED],
        description="The status of the notification.",
    )
    created_at: datetime = Field(
        ...,
        default_factory=datetime.now,
        description="The creation timestamp of the notification.",
    )
    updated_at: datetime = Field(
        ...,
        default_factory=datetime.now,
        description="The last update timestamp of the notification.",
    )
    provider_id: str | None = Field(
        default=None,
        examples=["123e4567-e89b-12d3-a456-426614174000"],
        description="The ID of the provider that sent the notification.",
    )
    error: str | None = Field(
        default=None,
        examples=[None],
        description="The error message if the notification failed.",
    )
