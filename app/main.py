from fastapi import FastAPI
from infrastructure.http.v1.router import v1_router

app = FastAPI(title="Notification Service (Technical Test)")

app.include_router(v1_router)
