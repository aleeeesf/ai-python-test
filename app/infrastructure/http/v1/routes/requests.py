from fastapi import APIRouter, Depends, HTTPException, Response, status

from application.dtos import CreateRequestDTO, CreateResponseDTO, StatusResponseDTO
from application.use_cases.create_request import create_request
from application.use_cases.get_request_status import get_request_status
from application.use_cases.process_request import start_process_request
from core.dependencies import (
    get_process_dispatcher,
    get_requests_repository,
)
from domain.entities.request import NotificationStatus
from domain.ports.process_dispatcher import ProcessDispatcher
from domain.ports.requests_repository import RequestsRepository

router = APIRouter(prefix="/requests", tags=["requests"])


@router.post("", status_code=201, response_model=CreateResponseDTO)
async def create_request_handler(
    request: CreateRequestDTO,
    requests_repository: RequestsRepository = Depends(get_requests_repository),
) -> CreateResponseDTO:
    """
    Create a new request.

    Args:
        request: CreateRequestDTO - The request to create.
        requests_repository: RequestsRepository - The repository to store the request.

    Returns:
        CreateResponseDTO: The created request.
    """
    request_id = await create_request(request, requests_repository)
    return CreateResponseDTO(id=request_id)


@router.post("/{request_id}/process")
async def process_request_handler(
    request_id: str,
    requests_repository: RequestsRepository = Depends(get_requests_repository),
    process_dispatcher: ProcessDispatcher = Depends(get_process_dispatcher),
) -> Response:
    """
    Process a request.

    Args:
        request_id: The ID of the request to process.
        requests_repository: RequestsRepository - The repository to retrieve the request.
        process_dispatcher: ProcessDispatcher - The dispatcher to process the request in the background.

    Returns:
        Response: FastAPI Response with status code 200 or 202.
    """
    process_result = await start_process_request(request_id, requests_repository)
    if not process_result.found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Request not found"
        )

    if process_result.status == NotificationStatus.SENT:
        return Response(status_code=status.HTTP_200_OK)

    if process_result.should_process:
        process_dispatcher.dispatch(request_id=request_id)  # process in background
        return Response(status_code=status.HTTP_202_ACCEPTED)

    if process_result.status == NotificationStatus.PROCESSING:
        return Response(status_code=status.HTTP_202_ACCEPTED)

    return Response(status_code=status.HTTP_202_ACCEPTED)


@router.get("/{request_id}", response_model=StatusResponseDTO)
async def get_request_status_handler(
    request_id: str,
    requests_repository: RequestsRepository = Depends(get_requests_repository),
) -> StatusResponseDTO:
    """
    Get the status of a request.

    Args:
        request_id: The ID of the request to get the status of.
        requests_repository: RequestsRepository - The repository to retrieve the request.

    Returns:
        StatusResponseDTO: The status of the request.
    """
    status_response = await get_request_status(request_id, requests_repository)
    if status_response is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Request not found",
        )
    return status_response
