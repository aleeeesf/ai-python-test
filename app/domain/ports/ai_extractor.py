"""Port interface for AI-based information extraction."""

from typing import Literal, Protocol

from pydantic import AliasChoices, BaseModel, Field, field_validator

from domain.models.request import NotificationType


class ChatMessage(BaseModel):
    """Message sent to or received from the AI provider."""

    role: Literal["system", "user", "assistant"] = Field(..., examples=["user"])
    content: str = Field(..., min_length=1)

    @field_validator("content", mode="before")
    @classmethod
    def strip_content(cls, value: str) -> str:
        """Trim whitespace from message content."""
        if isinstance(value, str):
            return value.strip()
        return value


class ChatChoice(BaseModel):
    """Single choice returned by the AI provider."""

    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class AIResponse(BaseModel):
    """Structured response returned by the AI provider."""

    id: str = Field(..., min_length=1)
    object: str = Field(..., min_length=1)
    created: int = Field(..., ge=0)
    model: str = Field(..., min_length=1)
    choices: list[ChatChoice] = Field(..., min_length=1)


class AIExtractedInfo(BaseModel):
    """Structured notification data extracted from user input."""

    to: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices(
            "to", "To", "recipient", "Recipient", "destination"
        ),
    )
    message: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("message", "Message", "body", "text"),
    )
    type: NotificationType = Field(
        ...,
        validation_alias=AliasChoices("type", "Type", "channel", "method"),
    )

    @field_validator("to", "message", mode="before")
    @classmethod
    def strip_fields(cls, value: str) -> str:
        """Trim whitespace from extracted text fields."""
        if isinstance(value, str):
            return value.strip()
        return value


class AIExtractor(Protocol):
    """Port for extracting structured information from natural language input."""

    async def extract(self, user_input: str) -> AIExtractedInfo:
        """Extract structured info from natural language input."""
        ...

    async def request_extraction(self, messages: list[ChatMessage]) -> AIResponse:
        """Call the AI provider and return its typed response."""
        ...
