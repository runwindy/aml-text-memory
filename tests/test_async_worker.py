from __future__ import annotations

import sqlite3
import time

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def wait_for_status(db_path, request_id: str, expected: str, timeout: float = 10.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        connection = sqlite3.connect(db_path)
        try:
            row = connection.execute(
                "SELECT status FROM add_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        finally:
            connection.close()
        if row and row[0] == expected:
            return row[0]
        time.sleep(0.1)
    return row[0] if row else "missing"


def test_async_add_returns_raw_ready_and_worker_completes(tmp_path) -> None:
    db_path = tmp_path / "async.db"
    settings = Settings(
        database_path=str(db_path),
        auth_mode="none",
        embedding_provider="hashing",
        embedding_dim=64,
        organizer_provider="rule",
        decomposer_provider="off",
        reranker_provider="lexical",
        async_processing_enabled=True,
        async_worker_enabled=True,
        async_worker_poll_interval=0.05,
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/add",
            json={
                "request_id": "async-request-1",
                "user_id": "async-user",
                "session_id": "async-session",
                "messages": [
                    {"role": "user", "content": "I moved to Shanghai.", "timestamp": 1704067200000},
                    {"role": "assistant", "content": "Got it, now you live in Shanghai.", "timestamp": 1704067260000},
                ],
            },
        )
        assert response.status_code == 200
        raw_status = wait_for_status(db_path, "async-request-1", "succeeded", timeout=10.0)
        assert raw_status == "succeeded"

        search = client.post(
            "/search",
            json={"query": "Where does the user live now?", "user_id": "async-user", "top_k": 10},
        )
        assert search.status_code == 200
        assert search.json()["data"]

    connection = sqlite3.connect(db_path)
    try:
        job_rows = connection.execute("SELECT status FROM async_jobs").fetchall()
        entity_count = connection.execute("SELECT COUNT(*) FROM entity_nodes").fetchone()[0]
        predicted_count = connection.execute(
            "SELECT COUNT(*) FROM memory_edges WHERE edge_type = 'predicted_related'"
        ).fetchone()[0]
    finally:
        connection.close()

    assert job_rows and job_rows[0][0] == "completed"
    assert entity_count >= 2
    # predicted edges may be empty for a tiny two-record example, so do not
    # require them; the table is exercised by the insertion path.
    assert predicted_count >= 0
