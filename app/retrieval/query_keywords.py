from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.retrieval.hybrid import tokenize

_ENTITY_RE = re.compile(r"[A-Z][A-Za-z'\-]+|[\u4e00-\u9fff]{2,4}")
_STOP_ENTITIES = {
    "what",
    "where",
    "when",
    "who",
    "why",
    "how",
    "which",
    "does",
    "did",
    "is",
    "are",
    "was",
    "were",
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
}

_PREDICATE_MAP = {
    "live": "live_in",
    "lives": "live_in",
    "living": "live_in",
    "reside": "live_in",
    "resides": "live_in",
    "住在": "live_in",
    "居住": "live_in",
    "work": "work_at",
    "works": "work_at",
    "working": "work_at",
    "工作": "work_at",
    "like": "likes",
    "likes": "likes",
    "love": "likes",
    "loves": "likes",
    "prefer": "prefers",
    "prefers": "prefers",
    "喜欢": "likes",
    "偏好": "prefers",
    "dislike": "dislikes",
    "dislikes": "dislikes",
    "hate": "dislikes",
    "hates": "dislikes",
    "讨厌": "dislikes",
    "不喜欢": "dislikes",
    "move": "moved_to",
    "moved": "moved_to",
    "搬到": "moved_to",
    "visit": "visited",
    "visited": "visited",
    "去了": "visited",
}

_RELATION_MAP = {
    "sister": "sister_of",
    "brother": "brother_of",
    "mother": "mother_of",
    "father": "father_of",
    "husband": "husband_of",
    "wife": "wife_of",
    "friend": "friend_of",
    "colleague": "colleague_of",
    "姐姐": "sister_of",
    "妹妹": "sister_of",
    "哥哥": "brother_of",
    "弟弟": "brother_of",
    "妈妈": "mother_of",
    "爸爸": "father_of",
    "丈夫": "husband_of",
    "妻子": "wife_of",
    "朋友": "friend_of",
    "同事": "colleague_of",
}

_TIME_TERMS = {
    "now",
    "current",
    "currently",
    "today",
    "latest",
    "present",
    "before",
    "after",
    "past",
    "previous",
    "when",
    "date",
    "year",
    "month",
    "day",
    "现在",
    "目前",
    "当前",
    "最新",
    "以前",
    "过去",
    "之前",
    "之后",
    "时间",
    "日期",
}


@dataclass(slots=True)
class QueryKeywords:
    tokens: list[str]
    entities: list[str] = field(default_factory=list)
    predicates: list[str] = field(default_factory=list)
    time_terms: list[str] = field(default_factory=list)
    relation_hints: list[str] = field(default_factory=list)
    question_type: str = "fact"


def _question_type(tokens: set[str], text: str) -> str:
    lowered = text.lower()
    if any(token in tokens for token in ("who", "谁")):
        return "who"
    if any(token in tokens for token in ("where", "哪里", "哪儿")):
        return "where"
    if any(token in tokens for token in ("when", "什么时候", "何时")):
        return "when"
    if any(token in tokens for token in ("how", "多少", "几个")):
        return "how_many"
    if any(token in tokens for token in ("why", "为什么")):
        return "why"
    if any(token in tokens for token in ("prefer", "favorite", "like", "喜欢", "偏好")):
        return "preference"
    if any(token in tokens for token in ("must", "should", "rule", "规则", "流程")):
        return "rule"
    return "fact"


def extract_query_keywords(text: str) -> QueryKeywords:
    tokens = [token.lower() for token in tokenize(text)]
    token_set = set(tokens)

    entities: list[str] = []
    for match in _ENTITY_RE.finditer(text):
        value = match.group(0).strip()
        value = re.sub(r"['\u2019]s$", "", value)
        normalized = value.lower()
        if normalized in _STOP_ENTITIES:
            continue
        if value and value not in entities:
            entities.append(value)

    predicates: list[str] = []
    for token in tokens:
        predicate = _PREDICATE_MAP.get(token)
        if predicate and predicate not in predicates:
            predicates.append(predicate)

    if any(term in text for term in ("喜欢", "偏好", "讨厌", "不喜欢")):
        for term, predicate in (("喜欢", "likes"), ("偏好", "prefers"), ("讨厌", "dislikes"), ("不喜欢", "dislikes")):
            if term in text and predicate not in predicates:
                predicates.append(predicate)

    time_terms = [token for token in tokens if token in _TIME_TERMS]
    relation_hints: list[str] = []
    for token in tokens:
        relation = _RELATION_MAP.get(token)
        if relation and relation not in relation_hints:
            relation_hints.append(relation)
    for term, relation in _RELATION_MAP.items():
        if term in text and relation not in relation_hints:
            relation_hints.append(relation)

    return QueryKeywords(
        tokens=tokens,
        entities=entities,
        predicates=predicates,
        time_terms=time_terms,
        relation_hints=relation_hints,
        question_type=_question_type(token_set, text),
    )
