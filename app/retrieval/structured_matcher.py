from __future__ import annotations

from app.memory.models import MemoryRecord
from app.retrieval.hybrid import tokenize
from app.retrieval.query_keywords import QueryKeywords


def _normalize(value: str | None) -> str:
    return str(value or "").strip().lower()


def _token_set(value: str | None) -> set[str]:
    return set(tokenize(str(value or "")))


def _entity_score(record: MemoryRecord, keywords: QueryKeywords) -> float:
    if not keywords.entities:
        return 0.0
    record_entities = {_normalize(entity) for entity in record.entities}
    record_entities.update({_normalize(record.subject), _normalize(record.object_value)})
    query_entities = {_normalize(entity) for entity in keywords.entities}
    if not query_entities:
        return 0.0
    return len(query_entities & record_entities) / len(query_entities)


def _predicate_score(record: MemoryRecord, keywords: QueryKeywords) -> float:
    if not keywords.predicates:
        return 0.0
    predicate = _normalize(record.predicate)
    for expected in keywords.predicates:
        if expected == predicate or expected in predicate or predicate in expected:
            return 1.0
    return 0.0


def _time_score(record: MemoryRecord, keywords: QueryKeywords) -> float:
    if not keywords.time_terms:
        return 0.0
    current_terms = {"now", "current", "currently", "today", "latest", "present", "现在", "目前", "当前", "最新"}
    past_terms = {"before", "after", "past", "previous", "以前", "过去", "之前", "之后"}
    terms = set(keywords.time_terms)

    if terms & current_terms:
        if record.status == "active" and record.valid_to is None:
            return 1.0
        if record.status == "superseded" or record.valid_to is not None:
            return 0.2
        return 0.6
    if terms & past_terms:
        if record.status == "superseded" or record.valid_to is not None:
            return 1.0
        return 0.4

    for term in terms:
        if term in _normalize(record.content) or term in _normalize(record.valid_from):
            return 0.8
    return 0.0


def _type_score(record: MemoryRecord, keywords: QueryKeywords) -> float:
    if keywords.question_type == "preference" and record.memory_type in {"preference", "profile"}:
        return 1.0
    if keywords.question_type == "when" and record.memory_type == "event":
        return 1.0
    if keywords.question_type == "rule" and record.memory_type == "rule":
        return 1.0
    if keywords.relation_hints and record.memory_type == "relation":
        return 1.0
    if keywords.question_type in {"who", "where", "what", "fact"} and record.memory_type in {"fact", "profile"}:
        return 1.0
    return 0.2


def _relation_score(record: MemoryRecord, keywords: QueryKeywords) -> float:
    if not keywords.relation_hints:
        return 0.0
    predicate = _normalize(record.predicate)
    return 1.0 if predicate in keywords.relation_hints else 0.0


def _content_score(record: MemoryRecord, keywords: QueryKeywords) -> float:
    query_tokens = set(keywords.tokens)
    if not query_tokens:
        return 0.0
    record_tokens = _token_set(record.content)
    if not record_tokens:
        return 0.0
    return len(query_tokens & record_tokens) / len(query_tokens)


def structured_scores(
    records: list[MemoryRecord],
    keywords: QueryKeywords,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for record in records:
        score = (
            0.35 * _entity_score(record, keywords)
            + 0.20 * _predicate_score(record, keywords)
            + 0.15 * _time_score(record, keywords)
            + 0.15 * _type_score(record, keywords)
            + 0.10 * _relation_score(record, keywords)
            + 0.05 * _content_score(record, keywords)
        )
        scores[record.id] = max(0.0, min(score, 1.0))
    return scores
