from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Iterable

from app.models.schemas import DocumentChunk


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