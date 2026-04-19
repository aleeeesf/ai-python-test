"""Exceptions for AI extraction functionality."""


class AIExtractionError(Exception):
    """Base exception for AI extraction failures."""


class AIProviderError(AIExtractionError):
    """Raised when AI provider returns an error response (HTTP non-200)."""


class AIResponseValidationError(AIExtractionError):
    """Raised when AI response content cannot be parsed or validated."""


class AINetworkError(AIExtractionError):
    """Raised when AI service is unreachable or times out."""
