from __future__ import annotations

import hashlib
import math
import re


class Embedder:
    def embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vector = [0.0] * 64
        for token in tokens:
            bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % 64
            vector[bucket] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
