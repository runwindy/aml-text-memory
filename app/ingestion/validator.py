from __future__ import annotations

from app.schemas import AddRequest


def validate_request(request: AddRequest) -> None:
    if not request.messages:
        raise ValueError("messages must not be empty")
    for message in request.messages:
        if not message.role or not message.content:
            raise ValueError("every message must have role and content")
