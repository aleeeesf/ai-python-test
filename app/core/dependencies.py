from functools import lru_cache

from core.settings import Settings, settings
from domain.ports.notification_provider import NotificationProvider
from domain.ports.process_dispatcher import ProcessDispatcher
from infrastructure.providers.external_notification_provider import (
    ExternalNotificationProvider,
)
from infrastructure.repositories.in_memory_requests_repository import (
    InMemoryRequestsRepository,
)
from infrastructure.workers.process_worker import ProcessWorker


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the application settings instance."""
    return settings


@lru_cache(maxsize=1)
def get_requests_repository() -> InMemoryRequestsRepository:
    """Get the requests repository instance."""
    return InMemoryRequestsRepository()


@lru_cache(maxsize=1)
def get_notification_provider() -> NotificationProvider:
    """Get the notification provider instance."""
    app_settings = get_settings()
    provider_settings = app_settings.external_provider
    return ExternalNotificationProvider(
        api_url=provider_settings.api_url,
        api_key=provider_settings.api_key,
    )


@lru_cache(maxsize=1)
def get_process_dispatcher() -> ProcessDispatcher:
    """Get the process dispatcher instance."""
    return ProcessWorker(
        requests_repository=get_requests_repository(),
        notification_provider=get_notification_provider(),
    )
