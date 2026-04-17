from __future__ import annotations

import ast
import importlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from app.utils.logging import get_logger


class LLMBackend(Protocol):
    def complete(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        ...


@dataclass
class LiteLLMBackend:
    """Provider adapter with an optional LiteLLM integration and deterministic mock fallback."""

    def __init__(self, provider: str = "openai") -> None:
        self.provider = provider

    def complete(self, model: str, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        if model.startswith("mock://") or self.provider == "mock":
            return self._mock_completion(messages)

        litellm_module = importlib.util.find_spec("litellm")
        if litellm_module is None:
            return self._mock_completion(messages)

        from litellm import completion

        response = completion(model=model, messages=messages, temperature=temperature)
        message = response.choices[0].message
        content = getattr(message, "content", None) or ""
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content)

    def _mock_completion(self, messages: list[dict[str, str]]) -> str:
        last_user = next((message["content"] for message in reversed(messages) if message["role"] == "user"), "")
        lowered = last_user.lower()
        if any(token in lowered for token in ("refund", "reset", "password", "invoice")):
            payload = {
                "decision": "search_docs",
                "confidence": 0.82,
                "answer": "",
                "tool": "search_docs",
            }
        else:
            payload = {
                "decision": "create_ticket",
                "confidence": 0.43,
                "answer": "",
                "tool": "create_ticket",
            }
        return json.dumps(payload)


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
                parsed = self._parse(raw)
                parsed["raw"] = raw
                parsed["model"] = self.model
                return parsed
            except Exception as exc:  # pragma: no cover - safety fallback
                last_error = exc
                if attempt > self.retry_count:
                    break
                time.sleep(min(0.2 * attempt, 0.5))
        raise RuntimeError("LLM completion failed") from last_error

    def _parse(self, raw: str) -> dict[str, Any]:
        try:
            payload = self._extract_json(raw)
            data = json.loads(payload)
            return data if isinstance(data, dict) else {"answer": str(data)}
        except json.JSONDecodeError:
            return {"decision": "answer", "confidence": 0.5, "answer": raw, "tool": None}

    def _extract_json(self, raw: str) -> str:
        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL)
        if fenced:
            return fenced.group(1)
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return raw[start : end + 1]
        if raw.strip().startswith("("):
            return json.dumps(ast.literal_eval(raw))
        return raw
