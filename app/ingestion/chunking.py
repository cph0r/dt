from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.schemas import DocumentChunk


@dataclass
class ChunkingPipeline:
    max_chars: int = 1200

    def chunk(self, source: str, text: str) -> list[DocumentChunk]:
        blocks = self._split_blocks(text)
        chunks: list[DocumentChunk] = []
        section = "root"
        buffered_section = section
        buffer: list[str] = []
        chunk_index = 0
        for block in blocks:
            heading = self._extract_heading(block)
            if heading:
                section = heading
            if len("\n\n".join(buffer + [block])) > self.max_chars and buffer:
                chunks.append(self._build_chunk(source, buffered_section, buffer, chunk_index))
                chunk_index += 1
                buffer = []
                buffered_section = section
            if not buffer:
                buffered_section = section
            buffer.append(block)
        if buffer:
            chunks.append(self._build_chunk(source, buffered_section, buffer, chunk_index))
        return chunks

    def _split_blocks(self, text: str) -> list[str]:
        raw_blocks = re.split(r"\n\s*\n", text.strip())
        return [block.strip() for block in raw_blocks if block.strip()]

    def _extract_heading(self, block: str) -> str | None:
        lines = block.splitlines()
        if not lines:
            return None
        first = lines[0].strip()
        if first.startswith("#"):
            return first.lstrip("#").strip()
        if re.match(r"^[A-Z][A-Za-z0-9\s:-]{3,}$", first) and len(lines) > 1:
            return first
        return None

    def _build_chunk(self, source: str, section: str, blocks: list[str], index: int) -> DocumentChunk:
        content = "\n\n".join(blocks)
        keywords = sorted(set(re.findall(r"[a-z0-9]+", content.lower())))[:12]
        return DocumentChunk(
            chunk_id=f"{source}-{index}",
            source=source,
            section=section,
            content=content,
            keywords=keywords,
            metadata={"source": source, "section": section, "chunk_id": f"{source}-{index}"},
        )
