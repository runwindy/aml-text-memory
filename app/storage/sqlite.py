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
            connection.commit()
        finally:
            connection.close()

    def _get_add_response_sync(self, request_id: str) -> dict | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT response_json FROM add_requests WHERE request_id = ?",
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
                SELECT user_id, session_id, payload_hash
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
            }
        finally:
            connection.close()

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
              AND memory_type IN ('fact', 'profile', 'preference', 'event')
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
                        request_id, user_id, session_id, payload_hash, response_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
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

    async def fetch_for_user(self, user_id: str, limit: int | None = None) -> list[MemoryRecord]:
        return await asyncio.to_thread(self._fetch_for_user_sync, user_id, limit)



