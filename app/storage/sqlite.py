from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from app.core.errors import RequestConflictError
from app.ingestion.models import CanonicalMessage, RawAddRequest
from app.memory.dialogue_tree import DialogueNode, build_dialogue_tree
from app.memory.entities import EntityEdge, EntityNode, MemoryEntityLink, build_entity_graph
from app.memory.knowledge_graph import MemoryEdge, build_memory_edges
from app.retrieval.graph_jepa import build_predicted_adjacency
from app.memory.models import MemoryRecord
from app.schemas import AddRequest, AddResponse


class SQLiteMemoryStore:
    """Persistent Bronze/Silver/Gold baseline store.

    Tables:
    - add_requests: idempotency and response cache
    - raw_add_requests: Bronze raw requests
    - messages: Silver canonical messages
    - memory_items: Gold memory units + embeddings
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.database_path), timeout=60)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=60000")
        return connection

    @staticmethod
    def _ensure_column(
        connection: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def _init_sync(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS add_requests (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    payload_hash TEXT,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS raw_add_requests (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    api_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    cleaner_version TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    raw_content TEXT NOT NULL,
                    normalized_content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    timestamp_ms INTEGER,
                    timestamp_inferred INTEGER NOT NULL,
                    time_granularity TEXT,
                    language TEXT NOT NULL,
                    pii_flags_json TEXT NOT NULL,
                    quality_flags_json TEXT NOT NULL,
                    safety_flags_json TEXT NOT NULL,
                    ingestion_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_user
                    ON messages(user_id);
                CREATE INDEX IF NOT EXISTS idx_messages_user_session
                    ON messages(user_id, session_id);

                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    timestamp INTEGER,
                    created_at TEXT NOT NULL,
                    embedding_json TEXT,
                    metadata_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_memory_items_user
                    ON memory_items(user_id);
                CREATE INDEX IF NOT EXISTS idx_memory_items_user_session
                    ON memory_items(user_id, session_id);

                CREATE TABLE IF NOT EXISTS dialogue_nodes (
                    node_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    role TEXT,
                    sequence_no INTEGER,
                    turn_index INTEGER,
                    parent_id TEXT,
                    prev_id TEXT,
                    next_id TEXT,
                    content TEXT NOT NULL,
                    timestamp INTEGER,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_dialogue_nodes_user
                    ON dialogue_nodes(user_id);
                CREATE INDEX IF NOT EXISTS idx_dialogue_nodes_user_session
                    ON dialogue_nodes(user_id, session_id);
                CREATE INDEX IF NOT EXISTS idx_dialogue_nodes_parent
                    ON dialogue_nodes(parent_id);

                CREATE TABLE IF NOT EXISTS memory_edges (
                    edge_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    confidence REAL,
                    valid_from TEXT,
                    valid_to TEXT,
                    evidence_ids_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_edges_user
                    ON memory_edges(user_id);
                CREATE INDEX IF NOT EXISTS idx_memory_edges_source
                    ON memory_edges(source_id);
                CREATE INDEX IF NOT EXISTS idx_memory_edges_target
                    ON memory_edges(target_id);

                CREATE TABLE IF NOT EXISTS entity_nodes (
                    entity_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    aliases_json TEXT NOT NULL,
                    entity_type TEXT,
                    confidence REAL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_entity_nodes_user
                    ON entity_nodes(user_id);
                CREATE INDEX IF NOT EXISTS idx_entity_nodes_user_name
                    ON entity_nodes(user_id, canonical_name);

                CREATE TABLE IF NOT EXISTS entity_edges (
                    edge_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_entity_id TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    confidence REAL,
                    valid_from TEXT,
                    valid_to TEXT,
                    evidence_ids_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_entity_edges_user
                    ON entity_edges(user_id);
                CREATE INDEX IF NOT EXISTS idx_entity_edges_source
                    ON entity_edges(source_entity_id);
                CREATE INDEX IF NOT EXISTS idx_entity_edges_target
                    ON entity_edges(target_entity_id);

                CREATE TABLE IF NOT EXISTS memory_entity_links (
                    memory_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    confidence REAL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (memory_id, entity_id, role)
                );
                CREATE INDEX IF NOT EXISTS idx_memory_entity_links_entity
                    ON memory_entity_links(entity_id);

                CREATE TABLE IF NOT EXISTS async_jobs (
                    job_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error_message TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_async_jobs_status
                    ON async_jobs(status, created_at);
                CREATE INDEX IF NOT EXISTS idx_async_jobs_request
                    ON async_jobs(request_id);

                CREATE TABLE IF NOT EXISTS dirty_entities (
                    dirty_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    processed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_dirty_entities_unprocessed
                    ON dirty_entities(processed_at, created_at);
                """
            )
            # Migration for databases created before payload_hash existed.
            self._ensure_column(connection, "add_requests", "payload_hash", "TEXT")

            # Migrations for structured Gold fields.
            self._ensure_column(connection, "memory_items", "subject", "TEXT")
            self._ensure_column(connection, "memory_items", "predicate", "TEXT")
            self._ensure_column(connection, "memory_items", "object_value", "TEXT")
            self._ensure_column(connection, "memory_items", "qualifiers_json", "TEXT")
            self._ensure_column(connection, "memory_items", "entities_json", "TEXT")
            self._ensure_column(connection, "memory_items", "source_message_ids_json", "TEXT")
            self._ensure_column(connection, "memory_items", "valid_from", "TEXT")
            self._ensure_column(connection, "memory_items", "valid_to", "TEXT")
            self._ensure_column(connection, "memory_items", "confidence", "REAL")
            self._ensure_column(connection, "memory_items", "importance", "REAL")
            self._ensure_column(connection, "memory_items", "status", "TEXT")

            # Request lifecycle and extra Silver views.
            self._ensure_column(connection, "add_requests", "status", "TEXT")
            self._ensure_column(connection, "add_requests", "error_message", "TEXT")
            self._ensure_column(connection, "messages", "raw_content_hash", "TEXT")
            self._ensure_column(connection, "messages", "time_mentions_json", "TEXT")
            connection.execute(
                "UPDATE add_requests SET status='succeeded' WHERE status IS NULL"
            )
            connection.commit()
        finally:
            connection.close()

    def _get_add_response_sync(self, request_id: str) -> dict | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT response_json FROM add_requests "
                "WHERE request_id = ? AND status = 'succeeded'",
                (request_id,),
            ).fetchone()
            if row is None:
                return None
            return json.loads(row["response_json"])
        finally:
            connection.close()

    def _get_add_metadata_sync(self, request_id: str) -> dict | None:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT user_id, session_id, payload_hash, status, error_message
                FROM add_requests
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "user_id": row["user_id"],
                "session_id": row["session_id"],
                "payload_hash": row["payload_hash"],
                "status": row["status"],
                "error_message": row["error_message"],
            }
        finally:
            connection.close()

    def _claim_add_sync(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        payload_hash: str,
    ) -> dict | None:
        """Atomically reserve a request_id.

        Returns None when this caller created the pending row; otherwise it
        returns the existing row so the caller can decide between idempotent
        replay, retry, or conflict.
        """

        now = datetime.now(timezone.utc).isoformat()
        connection = self._connect()
        try:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO add_requests(
                        request_id, user_id, session_id, payload_hash,
                        response_json, created_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (request_id, user_id, session_id, payload_hash, "{}", now, "pending"),
                )
                if cursor.rowcount == 1:
                    return None

                row = connection.execute(
                    """
                    SELECT request_id, user_id, session_id, payload_hash,
                           response_json, status, error_message
                    FROM add_requests
                    WHERE request_id = ?
                    """,
                    (request_id,),
                ).fetchone()
                return dict(row) if row is not None else None
        finally:
            connection.close()

    def _get_add_record_sync(self, request_id: str) -> dict | None:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT request_id, user_id, session_id, payload_hash,
                       response_json, status, error_message, created_at
                FROM add_requests
                WHERE request_id = ?
                """,
                (request_id,),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            connection.close()

    def _reset_add_for_retry_sync(self, request_id: str) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    UPDATE add_requests
                    SET status = 'pending', error_message = NULL
                    WHERE request_id = ?
                    """,
                    (request_id,),
                )
        finally:
            connection.close()

    def _mark_add_failed_sync(self, request_id: str, error_message: str) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    UPDATE add_requests
                    SET status = 'failed', error_message = ?
                    WHERE request_id = ?
                    """,
                    (error_message[:2000], request_id),
                )
        finally:
            connection.close()

    def _save_raw_request_sync(self, raw: RawAddRequest, *, status: str = "pending") -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO raw_add_requests(
                        request_id, user_id, session_id, payload_json, payload_hash,
                        received_at, api_version, status, cleaner_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        raw.request_id,
                        raw.user_id,
                        raw.session_id,
                        raw.payload_json,
                        raw.payload_hash,
                        raw.received_at,
                        raw.api_version,
                        status,
                        raw.cleaner_version,
                    ),
                )
                connection.execute(
                    """
                    UPDATE add_requests
                    SET status = ?, error_message = NULL
                    WHERE request_id = ?
                    """,
                    (status, raw.request_id),
                )
        finally:
            connection.close()

    def _save_silver_gold_sync(
        self,
        request: AddRequest,
        messages: Sequence[CanonicalMessage],
        records: Sequence[MemoryRecord],
        payload_hash: str,
    ) -> dict:
        response = AddResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        ).model_dump()
        response_json = json.dumps(response, ensure_ascii=False)

        connection = self._connect()
        try:
            with connection:
                existing = connection.execute(
                    """
                    SELECT status, user_id, session_id, payload_hash
                    FROM add_requests
                    WHERE request_id = ?
                    """,
                    (request.request_id,),
                ).fetchone()

                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO add_requests(
                            request_id, user_id, session_id, payload_hash,
                            response_json, created_at, status
                        ) VALUES (?, ?, ?, ?, ?, ?, 'succeeded')
                        """,
                        (
                            request.request_id,
                            request.user_id,
                            request.session_id,
                            payload_hash,
                            response_json,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                else:
                    if (
                        existing["user_id"] != request.user_id
                        or existing["session_id"] != request.session_id
                        or (
                            existing["payload_hash"] is not None
                            and existing["payload_hash"] != payload_hash
                        )
                    ):
                        raise RequestConflictError(
                            "request_id was already used with a different payload"
                        )

                if messages:
                    connection.executemany(
                        """
                        INSERT OR IGNORE INTO messages(
                            message_id, request_id, user_id, session_id, sequence_no, role,
                            raw_content, raw_content_hash, normalized_content, content_hash,
                            timestamp_ms, timestamp_inferred, time_granularity,
                            time_mentions_json, language, pii_flags_json,
                            quality_flags_json, safety_flags_json,
                            ingestion_version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                message.message_id,
                                message.request_id,
                                message.user_id,
                                message.session_id,
                                message.sequence_no,
                                message.role,
                                message.raw_content,
                                message.raw_content_hash,
                                message.normalized_content,
                                message.content_hash,
                                message.timestamp_ms,
                                1 if message.timestamp_inferred else 0,
                                message.time_granularity,
                                json.dumps(message.time_mentions, ensure_ascii=False),
                                message.language,
                                json.dumps(message.pii_flags, ensure_ascii=False),
                                json.dumps(message.quality_flags, ensure_ascii=False),
                                json.dumps(message.safety_flags, ensure_ascii=False),
                                message.ingestion_version,
                                message.created_at,
                            )
                            for message in messages
                        ],
                    )

                if records:
                    connection.executemany(
                        """
                        INSERT OR IGNORE INTO memory_items(
                            id, user_id, session_id, request_id, content, memory_type,
                            timestamp, created_at, embedding_json, metadata_json,
                            subject, predicate, object_value, qualifiers_json,
                            entities_json, source_message_ids_json,
                            valid_from, valid_to, confidence, importance, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                record.id,
                                record.user_id,
                                record.session_id,
                                record.request_id,
                                record.content,
                                record.memory_type,
                                record.timestamp,
                                record.created_at,
                                json.dumps(record.embedding) if record.embedding is not None else None,
                                json.dumps(record.metadata, ensure_ascii=False),
                                record.subject,
                                record.predicate,
                                record.object_value,
                                json.dumps(record.qualifiers, ensure_ascii=False),
                                json.dumps(record.entities, ensure_ascii=False),
                                json.dumps(record.source_message_ids, ensure_ascii=False),
                                record.valid_from,
                                record.valid_to,
                                record.confidence,
                                record.importance,
                                record.status,
                            )
                            for record in records
                        ],
                    )

                if messages:
                    dialogue_nodes = build_dialogue_tree(request, list(messages))
                    memory_edges = build_memory_edges(
                        request=request,
                        messages=list(messages),
                        records=list(records),
                        dialogue_nodes=dialogue_nodes,
                    )
                    memory_edges.extend(
                        self._build_predicted_memory_edges(
                            user_id=request.user_id,
                            records=list(records),
                        )
                    )
                    self._save_graph_sync(
                        connection,
                        user_id=request.user_id,
                        dialogue_nodes=dialogue_nodes,
                        memory_edges=memory_edges,
                    )
                    entity_nodes, entity_edges, entity_links = build_entity_graph(
                        request=request,
                        records=list(records),
                    )
                    self._save_entity_graph_sync(
                        connection,
                        user_id=request.user_id,
                        entity_nodes=entity_nodes,
                        entity_edges=entity_edges,
                        entity_links=entity_links,
                    )

                self._govern_temporal_memories_sync(connection, request.user_id)
                connection.execute(
                    """
                    UPDATE add_requests
                    SET status = 'succeeded', response_json = ?, error_message = NULL
                    WHERE request_id = ?
                    """,
                    (response_json, request.request_id),
                )
            return response
        finally:
            connection.close()

    def _save_raw_ready_sync(
        self,
        request: AddRequest,
        messages: Sequence[CanonicalMessage],
        records: Sequence[MemoryRecord],
        payload_hash: str,
    ) -> dict:
        response = self._save_silver_gold_sync(
            request,
            list(messages),
            list(records),
            payload_hash,
        )
        connection = self._connect()
        try:
            now = datetime.now(timezone.utc).isoformat()
            job_id = "job_" + hashlib.sha256(
                f"{request.request_id}\x1fstructured_extraction".encode("utf-8")
            ).hexdigest()[:32]
            with connection:
                connection.execute(
                    """
                    UPDATE add_requests
                    SET status = 'raw_ready'
                    WHERE request_id = ?
                    """,
                    (request.request_id,),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO async_jobs(
                        job_id, request_id, user_id, stage, status, payload_json,
                        created_at, updated_at, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        job_id,
                        request.request_id,
                        request.user_id,
                        "structured_extraction",
                        "pending",
                        request.model_dump_json(),
                        now,
                        now,
                    ),
                )
            return response
        finally:
            connection.close()

    def _claim_next_async_job_sync(self) -> dict | None:
        connection = self._connect()
        try:
            now = datetime.now(timezone.utc).isoformat()
            with connection:
                row = connection.execute(
                    """
                    SELECT job_id, request_id, user_id, stage, status, payload_json,
                           created_at, updated_at, error_message
                    FROM async_jobs
                    WHERE status = 'pending'
                    ORDER BY created_at
                    LIMIT 1
                    """,
                ).fetchone()
                if row is None:
                    return None
                connection.execute(
                    """
                    UPDATE async_jobs
                    SET status = 'processing', updated_at = ?
                    WHERE job_id = ?
                    """,
                    (now, row["job_id"]),
                )
            item = dict(row)
            item["status"] = "processing"
            return item
        finally:
            connection.close()

    def _complete_async_job_sync(self, job_id: str) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    UPDATE async_jobs
                    SET status = 'completed', updated_at = ?, error_message = NULL
                    WHERE job_id = ?
                    """,
                    (datetime.now(timezone.utc).isoformat(), job_id),
                )
        finally:
            connection.close()

    def _fail_async_job_sync(self, job_id: str, error_message: str) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    UPDATE async_jobs
                    SET status = 'failed', updated_at = ?, error_message = ?
                    WHERE job_id = ?
                    """,
                    (datetime.now(timezone.utc).isoformat(), error_message[:2000], job_id),
                )
        finally:
            connection.close()

    def _mark_dirty_entities_processed_sync(self, user_id: str) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    """
                    UPDATE dirty_entities
                    SET processed_at = ?
                    WHERE user_id = ? AND processed_at IS NULL
                    """,
                    (datetime.now(timezone.utc).isoformat(), user_id),
                )
        finally:
            connection.close()

    @staticmethod
    def _save_graph_sync(
        connection: sqlite3.Connection,
        *,
        user_id: str,
        dialogue_nodes: Sequence[DialogueNode],
        memory_edges: Sequence[MemoryEdge],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if dialogue_nodes:
            connection.executemany(
                """
                INSERT OR IGNORE INTO dialogue_nodes(
                    node_id, user_id, session_id, request_id, node_type, role,
                    sequence_no, turn_index, parent_id, prev_id, next_id,
                    content, timestamp, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        node.node_id,
                        node.user_id,
                        node.session_id,
                        node.request_id,
                        node.node_type,
                        node.role,
                        node.sequence_no,
                        node.turn_index,
                        node.parent_id,
                        node.prev_id,
                        node.next_id,
                        node.content,
                        node.timestamp,
                        json.dumps(node.metadata, ensure_ascii=False),
                        now,
                    )
                    for node in dialogue_nodes
                ],
            )
        if memory_edges:
            connection.executemany(
                """
                INSERT OR IGNORE INTO memory_edges(
                    edge_id, user_id, source_id, target_id, edge_type,
                    confidence, valid_from, valid_to, evidence_ids_json,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        edge.edge_id,
                        edge.user_id,
                        edge.source_id,
                        edge.target_id,
                        edge.edge_type,
                        edge.confidence,
                        edge.valid_from,
                        edge.valid_to,
                        json.dumps(edge.evidence_ids, ensure_ascii=False),
                        json.dumps(edge.metadata, ensure_ascii=False),
                        now,
                    )
                    for edge in memory_edges
                ],
            )

    @staticmethod
    def _build_predicted_memory_edges(
        *,
        user_id: str,
        records: Sequence[MemoryRecord],
    ) -> list[MemoryEdge]:
        adjacency = build_predicted_adjacency(
            records,
            query_vector=None,
            top_k=8,
            threshold=0.55,
            weight=0.5,
            max_nodes=200,
        )
        edges: list[MemoryEdge] = []
        for source_id, targets in adjacency.items():
            for target_id, weight in targets:
                if source_id >= target_id:
                    continue
                edge_id = "edge_" + hashlib.sha256(
                    f"predicted\x1f{user_id}\x1f{source_id}\x1f{target_id}".encode("utf-8")
                ).hexdigest()[:32]
                edges.append(
                    MemoryEdge(
                        edge_id=edge_id,
                        user_id=user_id,
                        source_id=source_id,
                        target_id=target_id,
                        edge_type="predicted_related",
                        confidence=max(0.0, min(1.0, float(weight))),
                        valid_from=None,
                        valid_to=None,
                        evidence_ids=[],
                        metadata={"source": "graph_jepa"},
                    )
                )
        return edges

    @staticmethod
    def _save_entity_graph_sync(
        connection: sqlite3.Connection,
        *,
        user_id: str,
        entity_nodes: Sequence[EntityNode],
        entity_edges: Sequence[EntityEdge],
        entity_links: Sequence[MemoryEntityLink],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if entity_nodes:
            connection.executemany(
                """
                INSERT OR IGNORE INTO entity_nodes(
                    entity_id, user_id, canonical_name, aliases_json,
                    entity_type, confidence, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        node.entity_id,
                        node.user_id,
                        node.canonical_name,
                        json.dumps(node.aliases, ensure_ascii=False),
                        node.entity_type,
                        node.confidence,
                        json.dumps(node.metadata, ensure_ascii=False),
                        now,
                        now,
                    )
                    for node in entity_nodes
                ],
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO dirty_entities(
                    user_id, entity_id, reason, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                [(user_id, node.entity_id, "new_memory", now) for node in entity_nodes],
            )
        if entity_edges:
            connection.executemany(
                """
                INSERT OR IGNORE INTO entity_edges(
                    edge_id, user_id, source_entity_id, target_entity_id,
                    edge_type, confidence, valid_from, valid_to,
                    evidence_ids_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        edge.edge_id,
                        edge.user_id,
                        edge.source_entity_id,
                        edge.target_entity_id,
                        edge.edge_type,
                        edge.confidence,
                        None,
                        None,
                        json.dumps(edge.evidence_ids, ensure_ascii=False),
                        json.dumps(edge.metadata, ensure_ascii=False),
                        now,
                    )
                    for edge in entity_edges
                ],
            )
        if entity_links:
            connection.executemany(
                """
                INSERT OR IGNORE INTO memory_entity_links(
                    memory_id, entity_id, role, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        link.memory_id,
                        link.entity_id,
                        link.role,
                        link.confidence,
                        now,
                    )
                    for link in entity_links
                ],
            )

    def _govern_temporal_memories_sync(
        self,
        connection: sqlite3.Connection,
        user_id: str,
    ) -> None:
        """Maintain valid_from / valid_to for structured memories.

        History is preserved. The newest record for a subject/predicate pair
        becomes active; older records are marked superseded and get valid_to.
        """

        rows = connection.execute(
            """
            SELECT id, subject, predicate, timestamp, valid_from, memory_type
            FROM memory_items
            WHERE user_id = ?
              AND subject IS NOT NULL
              AND predicate IS NOT NULL
              AND memory_type IN ('fact', 'profile', 'preference')
            ORDER BY subject, predicate, timestamp DESC
            """,
            (user_id,),
        ).fetchall()

        groups: dict[tuple[str, str], list[sqlite3.Row]] = {}
        for row in rows:
            groups.setdefault((row["subject"], row["predicate"]), []).append(row)

        for group in groups.values():
            latest = group[0]
            latest_valid_from = latest["valid_from"]
            if not latest_valid_from and latest["timestamp"] is not None:
                try:
                    latest_valid_from = datetime.fromtimestamp(
                        latest["timestamp"] / 1000,
                        tz=timezone.utc,
                    ).date().isoformat()
                except (OverflowError, OSError, ValueError):
                    latest_valid_from = None

            for row in group[1:]:
                connection.execute(
                    """
                    UPDATE memory_items
                    SET valid_to = ?, status = 'superseded'
                    WHERE id = ?
                    """,
                    (latest_valid_from, row["id"]),
                )
                edge_id = "edge_" + hashlib.sha256(
                    f"supersedes\x1f{latest['id']}\x1f{row['id']}".encode("utf-8")
                ).hexdigest()[:32]
                connection.execute(
                    """
                    INSERT OR IGNORE INTO memory_edges(
                        edge_id, user_id, source_id, target_id, edge_type,
                        confidence, valid_from, valid_to, evidence_ids_json,
                        metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge_id,
                        user_id,
                        latest["id"],
                        row["id"],
                        "supersedes",
                        0.8,
                        latest_valid_from,
                        None,
                        json.dumps([], ensure_ascii=False),
                        json.dumps({"governance": "temporal"}, ensure_ascii=False),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

            connection.execute(
                """
                UPDATE memory_items
                SET valid_to = NULL, status = 'active'
                WHERE id = ?
                """,
                (latest["id"],),
            )

    def _save_ingestion_sync(
        self,
        request: AddRequest,
        raw: RawAddRequest,
        messages: Sequence[CanonicalMessage],
        records: Sequence[MemoryRecord],
    ) -> dict:
        response = AddResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        ).model_dump()
        response_json = json.dumps(response, ensure_ascii=False)

        connection = self._connect()
        try:
            existing = connection.execute(
                """
                SELECT user_id, session_id, payload_hash, response_json
                FROM add_requests
                WHERE request_id = ?
                """,
                (request.request_id,),
            ).fetchone()

            if existing is not None:
                if (
                    existing["user_id"] != request.user_id
                    or existing["session_id"] != request.session_id
                    or (
                        existing["payload_hash"] is not None
                        and existing["payload_hash"] != raw.payload_hash
                    )
                ):
                    raise RequestConflictError(
                        "request_id was already used with a different payload"
                    )
                return json.loads(existing["response_json"])

            with connection:
                connection.execute(
                    """
                    INSERT INTO add_requests(
                        request_id, user_id, session_id, payload_hash,
                        response_json, created_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, 'succeeded')
                    """,
                    (
                        request.request_id,
                        request.user_id,
                        request.session_id,
                        raw.payload_hash,
                        response_json,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO raw_add_requests(
                        request_id, user_id, session_id, payload_json, payload_hash,
                        received_at, api_version, status, cleaner_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        raw.request_id,
                        raw.user_id,
                        raw.session_id,
                        raw.payload_json,
                        raw.payload_hash,
                        raw.received_at,
                        raw.api_version,
                        "indexed",
                        raw.cleaner_version,
                    ),
                )
                if messages:
                    connection.executemany(
                        """
                        INSERT INTO messages(
                            message_id, request_id, user_id, session_id, sequence_no, role,
                            raw_content, normalized_content, content_hash, timestamp_ms,
                            timestamp_inferred, time_granularity, language,
                            pii_flags_json, quality_flags_json, safety_flags_json,
                            ingestion_version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                message.message_id,
                                message.request_id,
                                message.user_id,
                                message.session_id,
                                message.sequence_no,
                                message.role,
                                message.raw_content,
                                message.normalized_content,
                                message.content_hash,
                                message.timestamp_ms,
                                1 if message.timestamp_inferred else 0,
                                message.time_granularity,
                                message.language,
                                json.dumps(message.pii_flags, ensure_ascii=False),
                                json.dumps(message.quality_flags, ensure_ascii=False),
                                json.dumps(message.safety_flags, ensure_ascii=False),
                                message.ingestion_version,
                                message.created_at,
                            )
                            for message in messages
                        ],
                    )
                if records:
                    connection.executemany(
                        """
                        INSERT INTO memory_items(
                            id, user_id, session_id, request_id, content, memory_type,
                            timestamp, created_at, embedding_json, metadata_json,
                            subject, predicate, object_value, qualifiers_json,
                            entities_json, source_message_ids_json,
                            valid_from, valid_to, confidence, importance, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            (
                                record.id,
                                record.user_id,
                                record.session_id,
                                record.request_id,
                                record.content,
                                record.memory_type,
                                record.timestamp,
                                record.created_at,
                                json.dumps(record.embedding) if record.embedding is not None else None,
                                json.dumps(record.metadata, ensure_ascii=False),
                                record.subject,
                                record.predicate,
                                record.object_value,
                                json.dumps(record.qualifiers, ensure_ascii=False),
                                json.dumps(record.entities, ensure_ascii=False),
                                json.dumps(record.source_message_ids, ensure_ascii=False),
                                record.valid_from,
                                record.valid_to,
                                record.confidence,
                                record.importance,
                                record.status,
                            )
                            for record in records
                        ],
                    )
                self._govern_temporal_memories_sync(connection, request.user_id)
            return response
        except sqlite3.IntegrityError:
            existing = connection.execute(
                """
                SELECT user_id, session_id, payload_hash, response_json
                FROM add_requests
                WHERE request_id = ?
                """,
                (request.request_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["user_id"] != request.user_id
                    or existing["session_id"] != request.session_id
                    or (
                        existing["payload_hash"] is not None
                        and existing["payload_hash"] != raw.payload_hash
                    )
                ):
                    raise RequestConflictError(
                        "request_id was already used with a different payload"
                    )
                return json.loads(existing["response_json"])
            raise
        finally:
            connection.close()

    def _save_add_sync(self, request: AddRequest, records: Sequence[MemoryRecord]) -> dict:
        payload_json = request.model_dump_json()
        raw = RawAddRequest(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
            payload_json=payload_json,
            payload_hash=hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
            received_at=datetime.now(timezone.utc).isoformat(),
        )
        return self._save_ingestion_sync(request, raw, [], list(records))

    def _fetch_dialogue_nodes_sync(self, user_id: str) -> list[dict]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT node_id, user_id, session_id, request_id, node_type, role,
                       sequence_no, turn_index, parent_id, prev_id, next_id,
                       content, timestamp, metadata_json, created_at
                FROM dialogue_nodes
                WHERE user_id = ?
                ORDER BY session_id, COALESCE(sequence_no, 0), COALESCE(turn_index, 0)
                """,
                (user_id,),
            ).fetchall()
            result: list[dict] = []
            for row in rows:
                item = dict(row)
                item["metadata"] = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                result.append(item)
            return result
        finally:
            connection.close()

    def _fetch_memory_edges_sync(self, user_id: str) -> list[dict]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT edge_id, user_id, source_id, target_id, edge_type,
                       confidence, valid_from, valid_to, evidence_ids_json,
                       metadata_json, created_at
                FROM memory_edges
                WHERE user_id = ?
                ORDER BY created_at
                """,
                (user_id,),
            ).fetchall()
            result: list[dict] = []
            for row in rows:
                item = dict(row)
                item["evidence_ids"] = json.loads(row["evidence_ids_json"]) if row["evidence_ids_json"] else []
                item["metadata"] = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                result.append(item)
            return result
        finally:
            connection.close()

    def _fetch_for_user_sync(self, user_id: str, limit: int | None) -> list[MemoryRecord]:
        connection = self._connect()
        try:
            sql = """
                SELECT id, user_id, session_id, request_id, content, memory_type,
                       timestamp, created_at, embedding_json, metadata_json,
                       subject, predicate, object_value, qualifiers_json,
                       entities_json, source_message_ids_json,
                       valid_from, valid_to, confidence, importance, status
                FROM memory_items
                WHERE user_id = ?
                ORDER BY created_at DESC
            """
            params: list[object] = [user_id]
            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)

            rows = connection.execute(sql, params).fetchall()
            records: list[MemoryRecord] = []
            for row in rows:
                embedding = None
                if row["embedding_json"]:
                    embedding = json.loads(row["embedding_json"])
                metadata = {}
                if row["metadata_json"]:
                    metadata = json.loads(row["metadata_json"])
                records.append(
                    MemoryRecord(
                        id=row["id"],
                        user_id=row["user_id"],
                        session_id=row["session_id"],
                        request_id=row["request_id"],
                        content=row["content"],
                        memory_type=row["memory_type"],
                        timestamp=row["timestamp"],
                        created_at=row["created_at"],
                        embedding=embedding,
                        metadata=metadata,
                        subject=row["subject"],
                        predicate=row["predicate"],
                        object_value=row["object_value"],
                        qualifiers=json.loads(row["qualifiers_json"]) if row["qualifiers_json"] else {},
                        entities=json.loads(row["entities_json"]) if row["entities_json"] else [],
                        source_message_ids=json.loads(row["source_message_ids_json"]) if row["source_message_ids_json"] else [],
                        valid_from=row["valid_from"],
                        valid_to=row["valid_to"],
                        confidence=row["confidence"],
                        importance=row["importance"],
                        status=row["status"] or "active",
                    )
                )
            return records
        finally:
            connection.close()

    async def init(self) -> None:
        await asyncio.to_thread(self._init_sync)

    async def get_add_response(self, request_id: str) -> dict | None:
        return await asyncio.to_thread(self._get_add_response_sync, request_id)

    async def get_add_metadata(self, request_id: str) -> dict | None:
        return await asyncio.to_thread(self._get_add_metadata_sync, request_id)

    async def claim_add(
        self,
        *,
        request_id: str,
        user_id: str,
        session_id: str,
        payload_hash: str,
    ) -> dict | None:
        return await asyncio.to_thread(
            self._claim_add_sync,
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            payload_hash=payload_hash,
        )

    async def get_add_record(self, request_id: str) -> dict | None:
        return await asyncio.to_thread(self._get_add_record_sync, request_id)

    async def wait_for_add_response(
        self,
        request_id: str,
        *,
        timeout: float = 60.0,
        interval: float = 0.1,
    ) -> dict | None:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            record = await self.get_add_record(request_id)
            if record is not None:
                status = record.get("status")
                if status == "succeeded":
                    response_json = record.get("response_json")
                    if isinstance(response_json, str):
                        return json.loads(response_json)
                    return response_json
                if status == "failed":
                    return None
            if asyncio.get_running_loop().time() >= deadline:
                return None
            await asyncio.sleep(interval)

    async def reset_add_for_retry(self, request_id: str) -> None:
        await asyncio.to_thread(self._reset_add_for_retry_sync, request_id)

    async def mark_add_failed(self, request_id: str, error_message: str) -> None:
        await asyncio.to_thread(self._mark_add_failed_sync, request_id, error_message)

    async def save_raw_request(self, raw: RawAddRequest, *, status: str = "pending") -> None:
        await asyncio.to_thread(self._save_raw_request_sync, raw, status=status)

    async def save_add(self, request: AddRequest, records: Sequence[MemoryRecord]) -> dict:
        return await asyncio.to_thread(self._save_add_sync, request, list(records))

    async def save_ingestion_result(
        self,
        *,
        request: AddRequest,
        raw: RawAddRequest,
        messages: Sequence[CanonicalMessage],
        records: Sequence[MemoryRecord],
    ) -> dict:
        return await asyncio.to_thread(
            self._save_ingestion_sync,
            request,
            raw,
            list(messages),
            list(records),
        )

    async def save_silver_gold(
        self,
        *,
        request: AddRequest,
        messages: Sequence[CanonicalMessage],
        records: Sequence[MemoryRecord],
        payload_hash: str,
    ) -> dict:
        return await asyncio.to_thread(
            self._save_silver_gold_sync,
            request,
            list(messages),
            list(records),
            payload_hash,
        )

    async def fetch_for_user(self, user_id: str, limit: int | None = None) -> list[MemoryRecord]:
        return await asyncio.to_thread(self._fetch_for_user_sync, user_id, limit)

    async def fetch_dialogue_nodes(self, user_id: str) -> list[dict]:
        return await asyncio.to_thread(self._fetch_dialogue_nodes_sync, user_id)

    async def fetch_memory_edges(self, user_id: str) -> list[dict]:
        return await asyncio.to_thread(self._fetch_memory_edges_sync, user_id)

    async def save_raw_ready_result(
        self,
        *,
        request: AddRequest,
        messages: Sequence[CanonicalMessage],
        records: Sequence[MemoryRecord],
        payload_hash: str,
    ) -> dict:
        return await asyncio.to_thread(
            self._save_raw_ready_sync,
            request,
            list(messages),
            list(records),
            payload_hash,
        )

    async def claim_next_async_job(self) -> dict | None:
        return await asyncio.to_thread(self._claim_next_async_job_sync)

    async def complete_async_job(self, job_id: str) -> None:
        await asyncio.to_thread(self._complete_async_job_sync, job_id)

    async def fail_async_job(self, job_id: str, error_message: str) -> None:
        await asyncio.to_thread(self._fail_async_job_sync, job_id, error_message)

    async def mark_dirty_entities_processed(self, user_id: str) -> None:
        await asyncio.to_thread(self._mark_dirty_entities_processed_sync, user_id)



