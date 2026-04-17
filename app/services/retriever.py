from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from app.models.schemas import DocumentChunk, RetrievalResult
from app.services.embedding import Embedder
from app.services.vector_store import SQLiteVectorStore


@dataclass
class Retriever:
    store: SQLiteVectorStore
    embedder: Embedder = field(default_factory=Embedder)

    def index(self, chunks: Sequence[DocumentChunk]) -> None:
        indexed: list[DocumentChunk] = []
        for chunk in chunks:
            if not chunk.embedding:
                chunk.embedding = self.embedder.embed(chunk.content)
            indexed.append(chunk)
        self.store.upsert(indexed)

    def search(
        self,
        query: str,
        top_k: int = 4,
        metadata_filter: dict[str, str] | None = None,
        rerank: bool = False,
    ) -> list[RetrievalResult]:
        query_embedding = self.embedder.embed(query)
        chunks = self.store.search(query_embedding, top_k=top_k, metadata_filter=metadata_filter)
        if not chunks:
            return []
        selected = [RetrievalResult(chunk=chunk, score=self._similarity(query_embedding, chunk.embedding)) for chunk in chunks]
        if rerank:
            selected = self._rerank(query, selected)
        return selected

    def _similarity(self, left: Sequence[float], right: Sequence[float]) -> float:
        left_values = list(left)
        right_values = list(right)
        return sum(a * b for a, b in zip(left_values, right_values))

    def _rerank(self, query: str, scored: list[RetrievalResult]) -> list[RetrievalResult]:
        import re

        query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        rescored = []
        for item in scored:
            content_tokens = set(re.findall(r"[a-z0-9]+", item.chunk.content.lower()))
            overlap = len(query_tokens & content_tokens)
            rescored.append(RetrievalResult(chunk=item.chunk, score=item.score + overlap * 0.05))
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored
