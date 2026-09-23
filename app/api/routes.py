from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.core.auth import require_auth
from app.schemas import AddRequest, AddResponse, HealthResponse, SearchRequest, SearchResponse
from app.services.container import AppServices


router = APIRouter()


def get_services(request: Request) -> AppServices:
    return request.app.state.services


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.post("/add", response_model=AddResponse)
async def add(
    request: AddRequest,
    _: None = Depends(require_auth),
    services: AppServices = Depends(get_services),
) -> AddResponse:
    return await services.add.handle(request)


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    _: None = Depends(require_auth),
    services: AppServices = Depends(get_services),
) -> SearchResponse:
    return await services.search.handle(request)
