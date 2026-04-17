from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.schemas import RetrievalResult, ToolResult
from app.services.retriever import Retriever
from app.tools.base import Tool


@dataclass
class SearchDocsTool(Tool):
    retriever: Retriever
    name: str = "search_docs"

    def execute(self, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", ""))
        top_k = int(kwargs.get("top_k", 4))
        metadata_filter = kwargs.get("metadata_filter")
        results = self.retriever.search(query=query, top_k=top_k, metadata_filter=metadata_filter)
        payload = [
            {
                "chunk_id": result.chunk.chunk_id,
                "source": result.chunk.source,
                "section": result.chunk.section,
                "content": result.chunk.content,
                "score": result.score,
                "metadata": result.chunk.metadata,
            }
            for result in results
        ]
        return ToolResult(name=self.name, success=True, data={"results": payload})
