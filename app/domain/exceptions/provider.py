class ProviderUnauthorizedError(Exception):
    """Raised when provider credentials are invalid."""


class ProviderRateLimitError(Exception):
    """Raised when provider rate limit is exceeded."""


class ProviderServerError(Exception):
    """Raised when provider returns a 5xx error."""


class ProviderNetworkError(Exception):
    """Raised when provider is unreachable or times out."""


class ProviderResponseError(Exception):
    """Raised when provider returns an unexpected response payload."""
