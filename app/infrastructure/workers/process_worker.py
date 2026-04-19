import asyncio

from application.use_cases.process_request import deliver_request
from domain.ports.ai_extractor import AIExtractor
from domain.ports.notification_provider import NotificationProvider
from domain.ports.process_dispatcher import ProcessDispatcher
from domain.ports.requests_repository import RequestsRepository


class ProcessWorker(ProcessDispatcher):
    """
    Worker that processes requests in the background.
    """

    def __init__(
        self,
        requests_repository: RequestsRepository,
        ai_extractor: AIExtractor,
        notification_provider: NotificationProvider,
    ) -> None:
        self.requests_repository = requests_repository
        self.ai_extractor = ai_extractor
        self.notification_provider = notification_provider
        self._tasks: set[asyncio.Task[None]] = set()

    def dispatch(self, request_id: str) -> None:
        """
        Creates a background task to process the request.

        Args:
            request_id: The ID of the request to process.

        Returns:
            None
        """
        task = asyncio.create_task(
            deliver_request(
                request_id,
                self.requests_repository,
                self.ai_extractor,
                self.notification_provider,
            )
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
