from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.models.schemas import ToolResult


class Tool(ABC):
    name: str

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def allow_list(self) -> list[str]:
        return sorted(self._tools)
