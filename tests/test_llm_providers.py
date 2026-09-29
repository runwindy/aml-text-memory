from __future__ import annotations

from app.config import Settings
from app.decomposer.llm_decomposer import LLMDecomposer
from app.organizer.composite import build_decomposer, build_organizer
from app.organizer.llm import LLMOrganizer


def test_deepseek_is_openai_compatible() -> None:
    settings = Settings(
        auth_mode="none",
        embedding_provider="hashing",
        organizer_provider="deepseek",
        organizer_model="deepseek-v4.1-flash",
        organizer_api_base="https://example.invalid/v1",
        organizer_api_key="test-key",
        decomposer_provider="deepseek",
        decomposer_model="deepseek-v4.1-flash",
        decomposer_api_base="https://example.invalid/v1",
        decomposer_api_key="test-key",
    )

    organizer = build_organizer(settings)
    decomposer = build_decomposer(settings)

    assert isinstance(organizer, LLMOrganizer)
    assert organizer.provider == "deepseek"
    assert organizer.model == "deepseek-v4.1-flash"
    assert organizer.disable_thinking is True

    assert isinstance(decomposer, LLMDecomposer)
    assert decomposer.provider == "deepseek"
    assert decomposer.model == "deepseek-v4.1-flash"
    assert decomposer.disable_thinking is True


def test_rule_provider_still_falls_back() -> None:
    settings = Settings(
        auth_mode="none",
        embedding_provider="hashing",
        organizer_provider="rule",
        decomposer_provider="off",
    )
    organizer = build_organizer(settings)
    assert not isinstance(organizer, LLMOrganizer)
    assert build_decomposer(settings) is None
