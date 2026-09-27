from __future__ import annotations

from pydantic import BaseModel, Field


class Proposition(BaseModel):
    proposition_id: str = ""
    memory_type: str = "fact"
    subject: str
    predicate: str
    object_value: str
    negated: bool = False
    modality: str | None = None
    condition: str | None = None
    time_text: str | None = None
    timestamp_ms: int | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float = 0.7
    source_message_indices: list[int] = Field(default_factory=list)


class Event(BaseModel):
    event_id: str = ""
    event_type: str
    subject: str
    object_value: str | None = None
    from_value: str | None = None
    to_value: str | None = None
    time_text: str | None = None
    timestamp_ms: int | None = None
    confidence: float = 0.7
    source_message_indices: list[int] = Field(default_factory=list)


class Relation(BaseModel):
    source: str
    target: str
    relation_type: str
    confidence: float = 0.7
    source_message_indices: list[int] = Field(default_factory=list)


class DecompositionResult(BaseModel):
    propositions: list[Proposition] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class BatchedDecomposition(BaseModel):
    window_id: str
    propositions: list[Proposition] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class BatchDecompositionResult(BaseModel):
    windows: list[BatchedDecomposition] = Field(default_factory=list)
