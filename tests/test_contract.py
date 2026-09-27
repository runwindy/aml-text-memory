from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def make_client(tmp_path, **overrides) -> TestClient:
    values = {
        "database_path": str(tmp_path / "test.db"),
        "auth_mode": "none",
        "embedding_provider": "hashing",
        "embedding_dim": 128,
    }
    values.update(overrides)
    settings = Settings(**values)
    return TestClient(create_app(settings))


def test_health(tmp_path):
    with make_client(tmp_path) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_add_and_search(tmp_path):
    with make_client(tmp_path) as client:
        add_response = client.post(
            "/add",
            json={
                "request_id": "req-1",
                "messages": [
                    {"role": "user", "content": "Alice lives in Shanghai.", "timestamp": 1704067200000},
                    {"role": "assistant", "content": "Alice likes hiking.", "timestamp": 1704067260000},
                ],
                "user_id": "user-1",
                "session_id": "session-1",
            },
        )
        assert add_response.status_code == 200
        assert add_response.json()["success"] is True
        assert add_response.json()["request_id"] == "req-1"

        search_response = client.post(
            "/search",
            json={"query": "Where does Alice live?", "user_id": "user-1", "top_k": 10},
        )
        assert search_response.status_code == 200
        payload = search_response.json()
        assert isinstance(payload["data"], list)
        assert len(payload["data"]) <= 10
        assert any("Shanghai" in item["content"] for item in payload["data"])


def test_user_isolation(tmp_path):
    with make_client(tmp_path) as client:
        client.post(
            "/add",
            json={
                "request_id": "req-2",
                "messages": [{"role": "user", "content": "Bob lives in Beijing."}],
                "user_id": "user-bob",
                "session_id": "session-bob",
            },
        )
        search_response = client.post(
            "/search",
            json={"query": "Where does Bob live?", "user_id": "user-alice", "top_k": 10},
        )
        assert search_response.status_code == 200
        assert search_response.json()["data"] == []


def test_idempotent_add(tmp_path):
    with make_client(tmp_path) as client:
        payload = {
            "request_id": "req-3",
            "messages": [{"role": "user", "content": "Carol likes tea."}],
            "user_id": "user-carol",
            "session_id": "session-carol",
        }
        first = client.post("/add", json=payload)
        second = client.post("/add", json=payload)
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json() == second.json()

        search_response = client.post(
            "/search",
            json={"query": "What does Carol like?", "user_id": "user-carol", "top_k": 10},
        )
        contents = [item["content"] for item in search_response.json()["data"]]
        assert len(contents) == len(set(contents))


def test_auth_required(tmp_path):
    with make_client(tmp_path, auth_mode="bearer", memory_system_key="secret") as client:
        payload = {
            "request_id": "req-4",
            "messages": [{"role": "user", "content": "Dave likes coffee."}],
            "user_id": "user-dave",
            "session_id": "session-dave",
        }
        assert client.post("/add", json=payload).status_code == 401
        authorized = client.post(
            "/add",
            json=payload,
            headers={"Authorization": "Bearer secret"},
        )
        assert authorized.status_code == 200

def test_request_id_conflict(tmp_path):
    with make_client(tmp_path) as client:
        base = {
            "request_id": "conflict-request-1",
            "messages": [{"role": "user", "content": "Alice likes tea."}],
            "user_id": "user-conflict",
            "session_id": "session-conflict",
        }
        first = client.post("/add", json=base)
        assert first.status_code == 200

        changed = {
            "request_id": "conflict-request-1",
            "messages": [{"role": "user", "content": "Alice likes coffee."}],
            "user_id": "user-conflict",
            "session_id": "session-conflict",
        }
        second = client.post("/add", json=changed)
        assert second.status_code == 409
        assert "different payload" in second.json()["detail"]["reason"]

def test_structured_memory_extraction(tmp_path):
    import sqlite3

    from fastapi.testclient import TestClient

    db_path = tmp_path / "structured.db"
    client = TestClient(
        create_app(
            Settings(
                database_path=str(db_path),
                auth_mode="none",
                embedding_provider="hashing",
                embedding_dim=128,
            )
        )
    )
    with client:
        response = client.post(
            "/add",
            json={
                "request_id": "structured-request-1",
                "messages": [
                    {
                        "role": "user",
                        "content": "My name is Alice and I live in Shanghai.",
                        "timestamp": 1704067200000,
                    },
                    {
                        "role": "assistant",
                        "content": "Alice likes tea.",
                        "timestamp": 1704067260000,
                    },
                ],
                "user_id": "structured-user",
                "session_id": "structured-session",
            },
        )
        assert response.status_code == 200

        search = client.post(
            "/search",
            json={
                "query": "Where does Alice live?",
                "user_id": "structured-user",
                "top_k": 10,
            },
        )
        assert search.status_code == 200
        assert search.json()["data"]

    connection = sqlite3.connect(db_path)
    try:
        memory_types = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT memory_type FROM memory_items"
            ).fetchall()
        }
    finally:
        connection.close()

    assert "raw" in memory_types
    assert "profile" in memory_types
    assert "fact" in memory_types
    assert "event" in memory_types or "preference" in memory_types
