from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MemorySnapshot:
    summary: str
    turns: list[dict[str, str]]


class ConversationMemory:
    """Conversation memory with a sliding window and rolling summaries."""

    def __init__(self, window_size: int = 6, summary_trigger: int = 12, max_summary_chars: int = 500) -> None:
        self.window_size = window_size
        self.summary_trigger = summary_trigger
        self.max_summary_chars = max_summary_chars
        self._turns: dict[str, list[dict[str, str]]] = {}
        self._summaries: dict[str, str] = {}

    def append(self, conversation_id: str, role: str, content: str) -> None:
        turns = self._turns.setdefault(conversation_id, [])
        turns.append({"role": role, "content": content})
        if len(turns) > self.summary_trigger:
            self._rollup(conversation_id)

    def get_snapshot(self, conversation_id: str) -> MemorySnapshot:
        turns = self._turns.get(conversation_id, [])
        summary = self._summaries.get(conversation_id, "")
        return MemorySnapshot(summary=summary, turns=turns[-self.window_size :])

    def export_all(self, conversation_id: str) -> list[dict[str, str]]:
        return list(self._turns.get(conversation_id, []))

    def _rollup(self, conversation_id: str) -> None:
        turns = self._turns.get(conversation_id, [])
        if not turns:
            return
        keep = turns[-self.window_size :]
        older = turns[: -self.window_size]
        old_summary = self._summaries.get(conversation_id, "")
        rolled = self._summarize(older)
        combined = (f"{old_summary}\n{rolled}" if old_summary else rolled).strip()
        self._summaries[conversation_id] = combined[: self.max_summary_chars]
        self._turns[conversation_id] = keep

    def _summarize(self, turns: list[dict[str, str]]) -> str:
        lines: list[str] = []
        for turn in turns:
            role = turn.get("role", "unknown")
            content = turn.get("content", "").replace("\n", " ").strip()
            if not content:
                continue
            lines.append(f"{role}: {content[:120]}")
        if not lines:
            return ""
        return "Summary: " + " | ".join(lines[:6])
