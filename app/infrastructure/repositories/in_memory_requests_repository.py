from threading import Lock

from domain.models.request import NotificationRequest
from domain.ports.requests_repository import RequestsRepository


class InMemoryRequestsRepository(RequestsRepository):
    """
    In-memory implementation of RequestsRepository.
    """

    def __init__(self) -> None:
        """
        Initialize the in-memory requests repository.

        Returns:
            None
        """
        self._requests: dict[str, NotificationRequest] = {}
        self._lock = Lock()

    async def save(self, request: NotificationRequest) -> None:
        """
        Save a request.

        Args:
            request: NotificationRequest - The request to save.

        Returns:
            None
        """
        with self._lock:
            self._requests[request.id] = request

    async def get_by_id(self, request_id: str) -> NotificationRequest | None:
        """
        Get a request by ID.

        Args:
            request_id: The ID of the request to get.

        Returns:
            NotificationRequest | None: The request with the given ID, or None if not found.
        """
        with self._lock:
            return self._requests.get(request_id)

    async def update(self, request: NotificationRequest) -> None:
        """
        Update a request.

        Args:
            request: NotificationRequest - The request to update.
        """
        with self._lock:
            self._requests[request.id] = request
