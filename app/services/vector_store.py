from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Iterable, Protocol

from app.models.schemas import DocumentChunk
from app.services.embedding import Embedder


class VectorStore(Protocol):
    def upsert(self, chunks: Iterable[DocumentChunk]) -> None:
        ...

    def all_chunks(self) -> list[DocumentChunk]:
        ...

    def count(self) -> int:
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[DocumentChunk]:
        ...


class SQLiteVectorStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    section TEXT NOT NULL,
                    content TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def upsert(self, chunks: Iterable[DocumentChunk]) -> None:
        with self._lock, self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO chunks (chunk_id, source, section, content, keywords, metadata, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    source = excluded.source,
                    section = excluded.section,
                    content = excluded.content,
                    keywords = excluded.keywords,
                    metadata = excluded.metadata,
                    embedding = excluded.embedding
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.source,
                        chunk.section,
                        chunk.content,
                        json.dumps(chunk.keywords),
                        json.dumps(chunk.metadata),
                        json.dumps(chunk.embedding),
                    )
                    for chunk in chunks
                ],
            )
            connection.commit()

    def all_chunks(self) -> list[DocumentChunk]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id, source, section, content, keywords, metadata, embedding FROM chunks ORDER BY chunk_id"
            ).fetchall()
        return [
            DocumentChunk(
                chunk_id=row[0],
                source=row[1],
                section=row[2],
                content=row[3],
                keywords=json.loads(row[4]),
                metadata=json.loads(row[5]),
                embedding=json.loads(row[6]),
            )
            for row in rows
        ]

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0] if row else 0)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[DocumentChunk]:
        candidates = [chunk for chunk in self.all_chunks() if self._matches(chunk, metadata_filter)]
        candidates.sort(key=lambda chunk: self._score(query_embedding, chunk.embedding), reverse=True)
        return candidates[:top_k]

    def _matches(self, chunk: DocumentChunk, metadata_filter: dict[str, str] | None) -> bool:
        if not metadata_filter:
            return True
        return all(str(chunk.metadata.get(key, getattr(chunk, key, ""))) == value for key, value in metadata_filter.items())

    def _score(self, left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))


class PgVectorStore:
    def __init__(self, database_url: str, dimension: int = 64) -> None:
        self.database_url = database_url
        self.dimension = dimension
        self.embedder = Embedder()
        self._initialize()

    def _connect(self):
        import psycopg
        from pgvector.psycopg import register_vector

        connection = psycopg.connect(self.database_url)
        register_vector(connection)
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    section TEXT NOT NULL,
                    content TEXT NOT NULL,
                    keywords JSONB NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding VECTOR({self.dimension}) NOT NULL
                )
                """
            )
            connection.commit()

    def upsert(self, chunks: Iterable[DocumentChunk]) -> None:
        from pgvector import Vector

        with self._connect() as connection:
            with connection.cursor() as cursor:
                for chunk in chunks:
                    if not chunk.embedding:
                        chunk.embedding = self.embedder.embed(chunk.content)
                    cursor.execute(
                        """
                        INSERT INTO chunks (chunk_id, source, section, content, keywords, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            source = EXCLUDED.source,
                            section = EXCLUDED.section,
                            content = EXCLUDED.content,
                            keywords = EXCLUDED.keywords,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding
                        """,
                        (
                            chunk.chunk_id,
                            chunk.source,
                            chunk.section,
                            chunk.content,
                            json.dumps(chunk.keywords),
                            json.dumps(chunk.metadata),
                            Vector(chunk.embedding),
                        ),
                    )
            connection.commit()

    def all_chunks(self) -> list[DocumentChunk]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id, source, section, content, keywords, metadata, embedding FROM chunks ORDER BY chunk_id"
            ).fetchall()
        return [
            DocumentChunk(
                chunk_id=row[0],
                source=row[1],
                section=row[2],
                content=row[3],
                keywords=row[4] if isinstance(row[4], list) else json.loads(row[4]),
                metadata=row[5] if isinstance(row[5], dict) else json.loads(row[5]),
                embedding=list(row[6]) if not isinstance(row[6], list) else row[6],
            )
            for row in rows
        ]

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0] if row else 0)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[DocumentChunk]:
        from pgvector import Vector

        clauses = []
        params: list[object] = []
        if metadata_filter:
            clauses.append("metadata @> %s::jsonb")
            params.append(json.dumps(metadata_filter))
        params.extend([Vector(query_embedding), top_k])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT chunk_id, source, section, content, keywords, metadata, embedding
            FROM chunks
            {where}
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [
            DocumentChunk(
                chunk_id=row[0],
                source=row[1],
                section=row[2],
                content=row[3],
                keywords=row[4] if isinstance(row[4], list) else json.loads(row[4]),
                metadata=row[5] if isinstance(row[5], dict) else json.loads(row[5]),
                embedding=list(row[6]) if not isinstance(row[6], list) else row[6],
            )
            for row in rows
        ]


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._embedder = Embedder()

    def upsert(self, chunks: Iterable[DocumentChunk]) -> None:
        lookup = {chunk.chunk_id: chunk for chunk in self._chunks}
        for chunk in chunks:
            if not chunk.embedding:
                chunk.embedding = self._embedder.embed(chunk.content)
            lookup[chunk.chunk_id] = chunk
        self._chunks = list(lookup.values())

    def all_chunks(self) -> list[DocumentChunk]:
        return list(self._chunks)

    def count(self) -> int:
        return len(self._chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 4,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[DocumentChunk]:
        candidates = [chunk for chunk in self._chunks if self._matches(chunk, metadata_filter)]
        candidates.sort(key=lambda chunk: sum(a * b for a, b in zip(query_embedding, chunk.embedding)), reverse=True)
        return candidates[:top_k]

    def _matches(self, chunk: DocumentChunk, metadata_filter: dict[str, str] | None) -> bool:
        if not metadata_filter:
            return True
        return all(str(chunk.metadata.get(key, getattr(chunk, key, ""))) == value for key, value in metadata_filter.items())