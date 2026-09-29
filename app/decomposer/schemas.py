from __future__ import annotations

from pydantic import BaseModel, Field


class Proposition(BaseModel):
    proposition_id: str | None = ""
    memory_type: str | None = "fact"
    subject: str | None = ""
    predicate: str | None = ""
    object_value: str | None = ""
    negated: bool = False
    modality: str | None = None
    condition: str | None = None
    time_text: str | None = None
    timestamp_ms: int | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float | None = 0.7
    source_message_indices: list[int] | None = None


class Event(BaseModel):
    event_id: str | None = ""
    event_type: str | None = ""
    subject: str | None = ""
    object_value: str | None = None
    from_value: str | None = None
    to_value: str | None = None
    time_text: str | None = None
    timestamp_ms: int | None = None
    confidence: float | None = 0.7
    source_message_indices: list[int] | None = None


class Relation(BaseModel):
    source: str | None = ""
    target: str | None = ""
    relation_type: str | None = ""
    confidence: float | None = 0.7
    source_message_indices: list[int] | None = None


class DecompositionResult(BaseModel):
    propositions: list[Proposition] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class BatchedDecomposition(BaseModel):
    window_id: str | None = ""
    propositions: list[Proposition] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


class BatchDecompositionResult(BaseModel):
    windows: list[BatchedDecomposition] = Field(default_factory=list)
