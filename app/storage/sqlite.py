from __future__ import annotations

import json
import sqlite3
import asyncio
from pathlib import Path
from typing import Sequence

from app.memory.models import MemoryRecord
from app.schemas import AddRequest, AddResponse


class SQLiteMemoryStore:
    """A small persistent baseline store.

    It is intentionally easy to replace. For the full evaluation scale, keep
    this interface and swap the implementation for Postgres + Qdrant, Weaviate,
    pgvector, or another production store.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.database_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _init_sync(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS add_requests (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

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

    def _save_add_sync(self, request: AddRequest, records: Sequence[MemoryRecord]) -> dict:
        response = AddResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        ).model_dump()
        response_json = json.dumps(response, ensure_ascii=False)

        connection = self._connect()
        try:
            existing = connection.execute(
                "SELECT response_json FROM add_requests WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()
            if existing is not None:
                return json.loads(existing["response_json"])

            with connection:
                connection.execute(
                    """
                    INSERT INTO add_requests(request_id, user_id, session_id, response_json, created_at)
                    VALUES (?, ?, ?, ?, datetime('now'))
                    """,
                    (
                        request.request_id,
                        request.user_id,
                        request.session_id,
                        response_json,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO memory_items(
                        id, user_id, session_id, request_id, content, memory_type,
                        timestamp, created_at, embedding_json, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        )
                        for record in records
                    ],
                )
            return response
        except sqlite3.IntegrityError:
            existing = connection.execute(
                "SELECT response_json FROM add_requests WHERE request_id = ?",
                (request.request_id,),
            ).fetchone()
            if existing is not None:
                return json.loads(existing["response_json"])
            raise
        finally:
            connection.close()

    def _fetch_for_user_sync(self, user_id: str, limit: int | None) -> list[MemoryRecord]:
        connection = self._connect()
        try:
            sql = """
                SELECT id, user_id, session_id, request_id, content, memory_type,
                       timestamp, created_at, embedding_json, metadata_json
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
                    )
                )
            return records
        finally:
            connection.close()

    async def init(self) -> None:
        await asyncio.to_thread(self._init_sync)

    async def get_add_response(self, request_id: str) -> dict | None:
        return await asyncio.to_thread(self._get_add_response_sync, request_id)

    async def save_add(self, request: AddRequest, records: Sequence[MemoryRecord]) -> dict:
        return await asyncio.to_thread(self._save_add_sync, request, list(records))

    async def fetch_for_user(self, user_id: str, limit: int | None = None) -> list[MemoryRecord]:
        return await asyncio.to_thread(self._fetch_for_user_sync, user_id, limit)

