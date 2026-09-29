from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.memory.models import MemoryRecord
from app.schemas import AddRequest

_ENTITY_NORMALIZE_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)


def normalize_entity_name(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"'s$", "", text)
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = _ENTITY_NORMALIZE_RE.sub(" ", text)
    return " ".join(text.split())


def entity_id_for(user_id: str, canonical_name: str) -> str:
    raw = f"{user_id}\x1f{canonical_name}"
    return "ent_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass(slots=True)
class EntityNode:
    entity_id: str
    user_id: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    entity_type: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EntityEdge:
    edge_id: str
    user_id: str
    source_entity_id: str
    target_entity_id: str
    edge_type: str
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MemoryEntityLink:
    memory_id: str
    entity_id: str
    role: str
    confidence: float | None = None


def _mentions(record: MemoryRecord) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    if record.subject:
        result.append((record.subject, "subject"))
    if record.object_value:
        result.append((record.object_value, "object"))
    for entity in record.entities or []:
        result.append((str(entity), "entity"))
    return result


def build_entity_graph(
    *,
    request: AddRequest,
    records: Sequence[MemoryRecord],
) -> tuple[list[EntityNode], list[EntityEdge], list[MemoryEntityLink]]:
    """Build lightweight entity nodes, co-occurrence edges, and memory links.

    Disambiguation is intentionally conservative: names are normalized and
    matched by normalized canonical name.  It is deterministic, cheap, and
    easy to replace with embedding-based resolution later.
    """

    nodes: dict[str, EntityNode] = {}
    links: list[MemoryEntityLink] = []
    edges: list[EntityEdge] = []

    for record in records:
        if record.memory_type == "raw":
            continue
        record_entity_ids: list[str] = []
        for mention, role in _mentions(record):
            canonical = normalize_entity_name(mention)
            if not canonical:
                continue
            entity_id = entity_id_for(request.user_id, canonical)
            node = nodes.get(entity_id)
            if node is None:
                node = EntityNode(
                    entity_id=entity_id,
                    user_id=request.user_id,
                    canonical_name=canonical,
                    aliases=[str(mention).strip()],
                    confidence=record.confidence,
                )
                nodes[entity_id] = node
            else:
                if str(mention).strip() not in node.aliases:
                    node.aliases.append(str(mention).strip())
            links.append(
                MemoryEntityLink(
                    memory_id=record.id,
                    entity_id=entity_id,
                    role=role,
                    confidence=record.confidence,
                )
            )
            if entity_id not in record_entity_ids:
                record_entity_ids.append(entity_id)

        for index, source_id in enumerate(record_entity_ids):
            for target_id in record_entity_ids[index + 1 :]:
                edge_id = "eedge_" + hashlib.sha256(
                    f"{request.user_id}\x1f{source_id}\x1f{target_id}\x1fco_occurs".encode("utf-8")
                ).hexdigest()[:32]
                edges.append(
                    EntityEdge(
                        edge_id=edge_id,
                        user_id=request.user_id,
                        source_entity_id=source_id,
                        target_entity_id=target_id,
                        edge_type="co_occurs",
                        confidence=float(record.confidence or 0.6),
                        evidence_ids=list(record.source_message_ids),
                    )
                )

    return list(nodes.values()), edges, links
