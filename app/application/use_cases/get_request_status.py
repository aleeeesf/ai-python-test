"""Use case for retrieving the status of a notification request."""

from application.dtos import StatusResponseDTO
from domain.ports.requests_repository import RequestsRepository


async def get_request_status(
    request_id: str,
    requests_repository: RequestsRepository,
) -> StatusResponseDTO | None:
    """
    Get the status of a notification request.

    Args:
        request_id: The ID of the request to retrieve.
        requests_repository: The repository for accessing requests.

    Returns:
        The status of the request, or None if the request is not found.
    """
    request = await requests_repository.get_by_id(request_id)
    if request is None:
        return None

    return StatusResponseDTO(id=request.id, status=request.status)
