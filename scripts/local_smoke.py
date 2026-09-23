from __future__ import annotations

import argparse
import json

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--key", default="")
    args = parser.parse_args()

    headers = {}
    if args.key:
        headers["X-Api-Key"] = args.key

    user_id = "demo-user"
    session_id = "demo-session"

    add_payload = {
        "request_id": "demo-request-0",
        "messages": [
            {"role": "user", "content": "Alice lives in Shanghai and loves hiking.", "timestamp": 1704067200000},
            {"role": "assistant", "content": "Noted. Alice likes the Blue Ridge Trail.", "timestamp": 1704067260000},
        ],
        "user_id": user_id,
        "session_id": session_id,
    }
    search_payload = {
        "query": "Where does Alice live and what does she like?",
        "user_id": user_id,
        "top_k": 10,
    }

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        print("health:", client.get("/health", headers=headers).text)
        add_response = client.post("/add", headers=headers, json=add_payload)
        print("add:", add_response.status_code, add_response.text)
        add_response.raise_for_status()

        search_response = client.post("/search", headers=headers, json=search_payload)
        print("search:", search_response.status_code, search_response.text)
        search_response.raise_for_status()
        print(json.dumps(search_response.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
