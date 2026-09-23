from __future__ import annotations

from dataclasses import dataclass

from app.memory.extractor import content_to_text
from app.schemas import SearchRequest


@dataclass(slots=True)
class QueryPlan:
    query_text: str
    retrieval_text: str
    query_type: str
    has_options: bool


def analyze_query(request: SearchRequest) -> QueryPlan:
    query_text = content_to_text(request.query)
    retrieval_parts = [query_text]
    if request.options:
        retrieval_parts.append("Options:")
        retrieval_parts.extend(request.options)
    retrieval_text = "\n".join(retrieval_parts)

    lowered = query_text.lower()
    if request.options:
        query_type = "choice"
    elif any(word in lowered for word in ("when", "date", "year", "month", "day", "先后", "时间", "日期")):
        query_type = "temporal"
    elif any(word in lowered for word in ("prefer", "favorite", "like", "love", "偏好", "喜欢")):
        query_type = "preference"
    elif any(word in lowered for word in ("and", "both", "relation", "关系", "以及")):
        query_type = "multi_hop"
    else:
        query_type = "fact"

    return QueryPlan(
        query_text=query_text,
        retrieval_text=retrieval_text,
        query_type=query_type,
        has_options=bool(request.options),
    )
