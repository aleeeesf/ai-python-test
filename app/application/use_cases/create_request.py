"""Use case for creating a new notification request."""

from uuid import uuid4

from application.dtos import CreateRequestDTO
from domain.models.request import NotificationRequest, NotificationStatus
from domain.ports.requests_repository import RequestsRepository


async def create_request(
    create_dto: CreateRequestDTO,
    requests_repository: RequestsRepository,
) -> str:
    """
    Create a new notification request with natural language input.

    Args:
        create_dto: The request data to create (contains user_input).
        requests_repository: The repository for persisting requests.

    Returns:
        The ID of the created request.
    """
    request_id = str(uuid4())
    new_request = NotificationRequest(
        id=request_id,
        user_input=create_dto.user_input,
        to=None,
        message=None,
        type=None,
        status=NotificationStatus.QUEUED,
    )
    await requests_repository.save(new_request)
    return request_id
