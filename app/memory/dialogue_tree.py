from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.ingestion.models import CanonicalMessage
from app.memory.assistant import detect_assistant_signals
from app.schemas import AddRequest


def _stable_id(*parts: object) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return "node_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


@dataclass(slots=True)
class DialogueNode:
    node_id: str
    user_id: str
    session_id: str
    request_id: str
    node_type: str
    role: str | None
    sequence_no: int | None
    turn_index: int | None
    parent_id: str | None
    prev_id: str | None
    next_id: str | None
    content: str
    timestamp: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_dialogue_tree(
    request: AddRequest,
    messages: Sequence[CanonicalMessage],
) -> list[DialogueNode]:
    """Build an ordered session tree.

    Session
      └── Turn
            ├── User message
            └── Assistant message
    """

    by_session: dict[str, list[CanonicalMessage]] = {}
    for message in messages:
        by_session.setdefault(message.session_id, []).append(message)

    nodes: list[DialogueNode] = []
    for session_id, session_messages in by_session.items():
        session_messages = sorted(session_messages, key=lambda item: item.sequence_no)
        session_node_id = _stable_id("session", request.user_id, session_id)
        nodes.append(
            DialogueNode(
                node_id=session_node_id,
                user_id=request.user_id,
                session_id=session_id,
                request_id=request.request_id,
                node_type="session",
                role=None,
                sequence_no=None,
                turn_index=None,
                parent_id=None,
                prev_id=None,
                next_id=None,
                content="",
                timestamp=None,
                metadata={"source": "dialogue_tree"},
            )
        )

        turn_index = 0
        current_turn_id: str | None = None
        previous_turn_id: str | None = None
        child_nodes: list[DialogueNode] = []

        def start_turn() -> str:
            nonlocal turn_index, current_turn_id, previous_turn_id, child_nodes
            turn_index += 1
            turn_id = _stable_id("turn", request.user_id, session_id, turn_index)
            nodes.append(
                DialogueNode(
                    node_id=turn_id,
                    user_id=request.user_id,
                    session_id=session_id,
                    request_id=request.request_id,
                    node_type="turn",
                    role=None,
                    sequence_no=None,
                    turn_index=turn_index,
                    parent_id=session_node_id,
                    prev_id=previous_turn_id,
                    next_id=None,
                    content="",
                    timestamp=None,
                    metadata={},
                )
            )
            if previous_turn_id is not None:
                for node in nodes:
                    if node.node_id == previous_turn_id:
                        node.next_id = turn_id
                        break
            previous_turn_id = turn_id
            current_turn_id = turn_id
            child_nodes = []
            return turn_id

        for message in session_messages:
            if message.role == "user" or current_turn_id is None:
                start_turn()
            assert current_turn_id is not None
            message_node_id = _stable_id(
                "message",
                request.user_id,
                session_id,
                message.sequence_no,
                message.role,
                message.content_hash,
            )
            message_node = DialogueNode(
                node_id=message_node_id,
                user_id=request.user_id,
                session_id=session_id,
                request_id=request.request_id,
                node_type="message",
                role=message.role,
                sequence_no=message.sequence_no,
                turn_index=turn_index,
                parent_id=current_turn_id,
                prev_id=child_nodes[-1].node_id if child_nodes else None,
                next_id=None,
                content=message.raw_content,
                timestamp=message.timestamp_ms,
                metadata={
                    "message_id": message.message_id,
                    "normalized_content": message.normalized_content,
                    "source_roles": [message.role],
                    "assistant_signals": (
                        detect_assistant_signals(message.raw_content)
                        if message.role == "assistant"
                        else []
                    ),
                },
            )
            if child_nodes:
                child_nodes[-1].next_id = message_node_id
            child_nodes.append(message_node)
            nodes.append(message_node)

    return nodes
