from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.ingestion.models import CanonicalMessage
from app.memory.dialogue_tree import DialogueNode
from app.memory.models import MemoryRecord
from app.schemas import AddRequest


def _stable_id(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return "edge_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass(slots=True)
class MemoryEdge:
    edge_id: str
    user_id: str
    source_id: str
    target_id: str
    edge_type: str
    confidence: float
    valid_from: str | None
    valid_to: str | None
    evidence_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def build_memory_edges(
    *,
    request: AddRequest,
    messages: Sequence[CanonicalMessage],
    records: Sequence[MemoryRecord],
    dialogue_nodes: Sequence[DialogueNode],
) -> list[MemoryEdge]:
    """Build explicit and assistant-derived graph edges.

    This is the persistent skeleton of the temporal knowledge graph.  It stores
    relation records and assistant-to-user support edges; Graph-JEPA predicted
    edges are computed at Search time and are intentionally not persisted here.
    """

    edges: list[MemoryEdge] = []

    # 1. Explicit relation records extracted by the decomposer.
    for record in records:
        if record.memory_type != "relation":
            continue
        if not record.subject or not record.object_value:
            continue
        edges.append(
            MemoryEdge(
                edge_id=_stable_id("relation", request.request_id, record.id),
                user_id=request.user_id,
                source_id=record.subject,
                target_id=record.object_value,
                edge_type=record.predicate or "relation",
                confidence=float(record.confidence or 0.7),
                valid_from=record.valid_from,
                valid_to=record.valid_to,
                evidence_ids=list(record.source_message_ids),
                metadata={"source_memory_id": record.id},
            )
        )

    # 2. Assistant-derived edges: assistant responds to / supports user nodes.
    message_node_by_message_id = {
        str(node.metadata.get("message_id")): node
        for node in dialogue_nodes
        if node.node_type == "message" and node.metadata.get("message_id")
    }
    nodes_by_parent: dict[str, list[DialogueNode]] = {}
    for node in dialogue_nodes:
        if node.parent_id:
            nodes_by_parent.setdefault(node.parent_id, []).append(node)

    for message in messages:
        if message.role != "assistant":
            continue
        assistant_node = message_node_by_message_id.get(message.message_id)
        if assistant_node is None or assistant_node.parent_id is None:
            continue
        siblings = nodes_by_parent.get(assistant_node.parent_id, [])
        signals = set((assistant_node.metadata or {}).get("assistant_signals") or [])
        for sibling in siblings:
            if sibling.role != "user":
                continue
            edges.append(
                MemoryEdge(
                    edge_id=_stable_id("responds_to", assistant_node.node_id, sibling.node_id),
                    user_id=request.user_id,
                    source_id=assistant_node.node_id,
                    target_id=sibling.node_id,
                    edge_type="responds_to",
                    confidence=0.8,
                    valid_from=None,
                    valid_to=None,
                    evidence_ids=[message.message_id, str(sibling.metadata.get("message_id", ""))],
                    metadata={"assistant_signals": sorted(signals)},
                )
            )
            if {"confirmation", "summary", "elaboration"} & signals:
                edge_type = "confirms" if "confirmation" in signals else "supports"
                edges.append(
                    MemoryEdge(
                        edge_id=_stable_id(edge_type, assistant_node.node_id, sibling.node_id),
                        user_id=request.user_id,
                        source_id=assistant_node.node_id,
                        target_id=sibling.node_id,
                        edge_type=edge_type,
                        confidence=0.7,
                        valid_from=None,
                        valid_to=None,
                        evidence_ids=[message.message_id, str(sibling.metadata.get("message_id", ""))],
                        metadata={"assistant_signals": sorted(signals)},
                    )
                )

    return edges
