from __future__ import annotations

from app.memory.models import MemoryRecord
from app.retrieval.graph_expander import expand_graph
from app.retrieval.query_keywords import extract_query_keywords


def test_query_keywords():
    keywords = extract_query_keywords("Where does Alice's sister live now?")
    assert "alice" in {value.lower() for value in keywords.entities}
    assert "live_in" in keywords.predicates
    assert "sister_of" in keywords.relation_hints
    assert "now" in keywords.time_terms
    assert keywords.question_type == "where"


def test_graph_expander():
    fact_alice = MemoryRecord(
        id="mem-1",
        user_id="u1",
        session_id="s1",
        content="Alice's sister is Emma.",
        memory_type="fact",
        subject="Alice",
        predicate="sister_of",
        object_value="Emma",
        entities=["Alice", "Emma"],
    )
    relation = MemoryRecord(
        id="mem-rel-1",
        user_id="u1",
        session_id="s1",
        content="[relation] mem-1 --sister_of--> mem-2",
        memory_type="relation",
        subject="mem-1",
        predicate="sister_of",
        object_value="mem-2",
        entities=["mem-1", "mem-2"],
    )
    fact_emma = MemoryRecord(
        id="mem-2",
        user_id="u1",
        session_id="s1",
        content="Emma lives in Tokyo.",
        memory_type="fact",
        subject="Emma",
        predicate="live_in",
        object_value="Tokyo",
        entities=["Emma", "Tokyo"],
    )

    scores = expand_graph(
        [fact_alice, relation, fact_emma],
        ["mem-1"],
        extract_query_keywords("Where does Alice's sister live?"),
        max_hops=2,
    )

    assert "mem-2" in scores
    assert "mem-rel-1" in scores
