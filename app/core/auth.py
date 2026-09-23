from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from secrets import compare_digest


async def require_auth(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header(alias="X-Api-Key")] = None,
) -> None:
    """Validate the Memory System Key according to the public contract."""

    settings = request.app.state.services.settings
    if settings.auth_mode == "none":
        return

    provided = ""
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() in {"bearer", "token"}:
            provided = value.strip()
    if not provided and x_api_key:
        provided = x_api_key.strip()

    if not provided or not compare_digest(provided, settings.memory_system_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"reason": "invalid or missing authentication credentials"},
        )
