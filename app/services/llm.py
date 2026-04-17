from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

from app.utils.logging import get_logger


class LLMBackend(Protocol):
    def complete(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        ...


@dataclass
class LiteLLMBackend:
    """Small provider adapter that can be replaced by LiteLLM without changing call sites."""

    def complete(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        last_user = next((message["content"] for message in reversed(messages) if message["role"] == "user"), "")
        return json.dumps(
            {
                "decision": "answer" if len(last_user) < 180 else "search_docs",
                "confidence": 0.74 if len(last_user) < 180 else 0.55,
                "answer": f"Stubbed model response for: {last_user[:120]}",
                "tool": None,
            }
        )


class LLMClient:
    def __init__(self, backend: LLMBackend, model: str, retry_count: int = 2, timeout_s: float = 12.0) -> None:
        self.backend = backend
        self.model = model
        self.retry_count = retry_count
        self.timeout_s = timeout_s
        self.logger = get_logger(__name__)

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.0) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_s
        attempt = 0
        last_error: Exception | None = None
        while attempt <= self.retry_count:
            attempt += 1
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("LLM request timed out")
                start = time.monotonic()
                raw = self.backend.complete(self.model, messages, temperature=temperature)
                latency_ms = round((time.monotonic() - start) * 1000, 2)
                self.logger.info("llm_completion", extra={"event": "llm_completion", "latency_ms": latency_ms})
                return self._parse(raw)
            except Exception as exc:  # pragma: no cover - safety fallback
                last_error = exc
                if attempt > self.retry_count:
                    break
                time.sleep(min(0.2 * attempt, 0.5))
        raise RuntimeError("LLM completion failed") from last_error

    def _parse(self, raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"decision": "answer", "confidence": 0.5, "answer": raw, "tool": None}
