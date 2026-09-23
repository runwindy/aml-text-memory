from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ImageURL(BaseModel):
    url: str = Field(min_length=1)


class ContentPart(BaseModel):
    """Multimodal content part.

    Text track uses a plain string. This model exists so the framework can be
    extended to the multimodal track without changing the API envelope.
    """

    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURL | None = None

    @model_validator(mode="after")
    def validate_part(self) -> "ContentPart":
        if self.type == "text":
            if not self.text or not self.text.strip():
                raise ValueError("text part requires a non-empty text field")
        elif self.type == "image_url":
            if self.image_url is None:
                raise ValueError("image_url part requires image_url.url")
        return self


def _validate_content(value: object) -> str | list[ContentPart]:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("content must not be empty")
        return value

    if isinstance(value, list):
        if not value:
            raise ValueError("content parts must not be empty")
        parts: list[ContentPart] = []
        for item in value:
            parts.append(item if isinstance(item, ContentPart) else ContentPart.model_validate(item))
        return parts

    raise ValueError("content must be a string or an ordered ContentPart[] array")


class AddMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[ContentPart]
    timestamp: int | None = None

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, value: object) -> str | list[ContentPart]:
        return _validate_content(value)


class AddRequest(BaseModel):
    request_id: str = Field(min_length=1)
    messages: list[AddMessage] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class AddResponse(BaseModel):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str | list[ContentPart]
    options: list[str] | None = None
    user_id: str = Field(min_length=1)
    top_k: int = Field(gt=0)

    @field_validator("query", mode="before")
    @classmethod
    def validate_query(cls, value: object) -> str | list[ContentPart]:
        return _validate_content(value)

    @field_validator("options")
    @classmethod
    def validate_options(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("options must not be empty when provided")
        if any(not str(item).strip() for item in value):
            raise ValueError("options must not contain empty strings")
        return value


class SearchResponseItem(BaseModel):
    id: str
    content: str | list[ContentPart]
    score: float | None = None
    created_at: datetime | None = None


class SearchResponse(BaseModel):
    data: list[SearchResponseItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
