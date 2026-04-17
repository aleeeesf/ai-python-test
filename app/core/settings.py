from pydantic import Field
from pydantic_settings import BaseSettings


class ExternalProviderSettings(BaseSettings):
    """Settings for the external notification provider."""

    api_url: str = Field(
        default="http://localhost:3001",
        description="External provider API URL",
    )  # Only for this challenge setted as default, in production it should be loaded from environment variables
    api_key: str = Field(
        default="test-dev-2026",
        description="External provider API key",
    )  # Only for this challenge setted as default, in production it should be loaded from environment variables


class Settings(BaseSettings):
    """Global application settings."""

    external_provider: ExternalProviderSettings = ExternalProviderSettings()

    # Logging configuration
    logging_enabled: bool = False  # Disabled by default (tests shouldn't log)
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL


settings = Settings()
