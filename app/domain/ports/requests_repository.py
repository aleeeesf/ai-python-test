from typing import Protocol

from domain.entities.request import NotificationRequest


class RequestsRepository(Protocol):
    async def save(self, request: NotificationRequest) -> None: ...

    async def get_by_id(self, request_id: str) -> NotificationRequest | None: ...

    async def update(self, request: NotificationRequest) -> None: ...
