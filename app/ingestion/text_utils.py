from __future__ import annotations

from app.schemas import ContentPart


def content_to_text(content: str | list[ContentPart]) -> str:
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for part in content:
        if part.type == "text":
            parts.append(part.text or "")
        elif part.type == "image_url" and part.image_url is not None:
            parts.append(f"[image:{part.image_url.url[:64]}]")
    return "\n".join(parts)
