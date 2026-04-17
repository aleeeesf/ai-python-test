from typing import Protocol


class ProcessDispatcher(Protocol):
    def dispatch(self, request_id: str) -> None: ...
