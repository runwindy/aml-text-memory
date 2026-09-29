from __future__ import annotations

import json
import logging
import re
from typing import Sequence

import httpx

from app.decomposer.prompt import BATCH_DECOMPOSITION_PROMPT, DECOMPOSITION_PROMPT
from app.decomposer.schemas import BatchDecompositionResult, DecompositionResult
from app.ingestion.models import CanonicalMessage
from app.memory.window_extractor import DialogueWindow

logger = logging.getLogger(__name__)


def _extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("decomposer response does not contain a JSON object")
    return json.loads(match.group(0))


class LLMDecomposer:
    """OpenAI-compatible complex-sentence decomposer."""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 2048,
        timeout: float = 120.0,
        provider: str = "openai",
        disable_thinking: bool = False,
    ) -> None:
        if not api_base:
            raise ValueError("decomposer_api_base is required")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.provider = provider
        self.disable_thinking = disable_thinking

    async def _call_model(self, prompt: str) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if self.disable_thinking:
            # DeepSeek reasoning models accept this switch.  Without it,
            # reasoning tokens may consume the whole completion budget and
            # leave no final JSON content.
            body["thinking"] = {"type": "disabled"}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            payload = response.json()
        return _extract_json_object(payload["choices"][0]["message"]["content"])

    async def decompose(
        self,
        window: DialogueWindow,
    ) -> DecompositionResult:
        results = await self.decompose_many([window])
        return results.get(window.window_id, DecompositionResult())

    async def decompose_many(
        self,
        windows: list[DialogueWindow],
    ) -> dict[str, DecompositionResult]:
        if not windows:
            return {}

        blocks: list[str] = []
        for window in windows:
            lines = [
                f"[{index}] [{message.role}] {message.raw_content}"
                for index, message in enumerate(window.messages)
            ]
            blocks.append(
                f'<window id="{window.window_id}">\n' + "\n".join(lines) + "\n</window>"
            )

        prompt = BATCH_DECOMPOSITION_PROMPT.replace(
            "{dialogue_windows}",
            "\n\n".join(blocks),
        )
        try:
            payload = await self._call_model(prompt)
            batch = BatchDecompositionResult.model_validate(payload)
            if not batch.windows:
                logger.warning("decomposer returned zero windows for %d input windows", len(windows))
            return {
                item.window_id: DecompositionResult(
                    propositions=item.propositions,
                    events=item.events,
                    relations=item.relations,
                )
                for item in batch.windows
            }
        except Exception:
            logger.warning(
                "decomposer call failed; returning empty decomposition",
                exc_info=True,
            )
            return {}
