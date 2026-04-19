from fastapi import APIRouter

from infrastructure.http.v1.routes.requests import router as requests_router

v1_router = APIRouter(prefix="/v1")

v1_router.include_router(requests_router)
