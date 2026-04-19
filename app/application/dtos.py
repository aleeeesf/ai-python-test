from pydantic import BaseModel, Field

from domain.models.request import NotificationStatus


class CreateRequestDTO(BaseModel):
    user_input: str = Field(
        ..., min_length=1, examples=["Manda un mail a feda@test.com diciendo hola"]
    )


class CreateResponseDTO(BaseModel):
    id: str = Field(..., examples=["123e4567-e89b-12d3-a456-426614174000"])


class StatusResponseDTO(BaseModel):
    id: str = Field(..., examples=["123e4567-e89b-12d3-a456-426614174000"])
    status: NotificationStatus = Field(..., examples=["queued"])


class StartProcessResultDTO(BaseModel):
    found: bool
    should_process: bool
    status: NotificationStatus | None = None
