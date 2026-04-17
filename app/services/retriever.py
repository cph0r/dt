from __future__ import annotations

import math
import re
import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from app.models.schemas import DocumentChunk, RetrievalResult
from app.services.vector_store import SQLiteVectorStore


class Embedder:
    def embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vector = [0.0] * 64
        for token in tokens:
            bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % 64
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


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
        chunks = self.store.all_chunks()
        if not chunks:
            return []
        query_embedding = self.embedder.embed(query)
        candidates = [chunk for chunk in chunks if self._matches(chunk, metadata_filter)]
        scored = [RetrievalResult(chunk=chunk, score=self._similarity(query_embedding, chunk.embedding)) for chunk in candidates]
        scored.sort(key=lambda item: item.score, reverse=True)
        selected = scored[:top_k]
        if rerank:
            selected = self._rerank(query, selected)
        return selected

    def _matches(self, chunk: DocumentChunk, metadata_filter: dict[str, str] | None) -> bool:
        if not metadata_filter:
            return True
        return all(str(chunk.metadata.get(key, getattr(chunk, key, ""))) == value for key, value in metadata_filter.items())

    def _similarity(self, left: Iterable[float], right: Iterable[float]) -> float:
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
