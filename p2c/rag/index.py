"""In-memory vector index for code chunks using numpy cosine similarity."""

from __future__ import annotations

import base64
import struct
from dataclasses import dataclass

import numpy as np

from p2c.rag.chunker import CodeChunk


@dataclass
class RetrievalResult:
    """A retrieved chunk with its similarity score."""

    chunk: CodeChunk
    score: float


class CodeIndex:
    """In-memory vector store for code chunks."""

    def __init__(self, chunks: list[CodeChunk], embeddings: np.ndarray) -> None:
        self.chunks = chunks
        # L2-normalize for cosine similarity via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.embeddings = embeddings / norms  # shape: (n_chunks, dim)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    def query(self, query_embedding: list[float], top_k: int = 10) -> list[RetrievalResult]:
        """Return top-k chunks by cosine similarity."""
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        scores = self.embeddings @ q  # (n_chunks,)
        k = min(top_k, len(self.chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [
            RetrievalResult(chunk=self.chunks[i], score=float(scores[i]))
            for i in top_indices
        ]

    def query_multi(
        self,
        query_embeddings: list[list[float]],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Query with multiple embeddings, deduplicate by file+lines, return top-k overall."""
        if not query_embeddings:
            return []

        # Gather per-query results with extra headroom
        per_query_k = max(top_k, top_k // len(query_embeddings) + 5)
        seen: set[tuple[str, int, int]] = set()
        all_results: list[RetrievalResult] = []

        for qe in query_embeddings:
            for r in self.query(qe, top_k=per_query_k):
                key = (r.chunk.file_path, r.chunk.start_line, r.chunk.end_line)
                if key in seen:
                    continue
                seen.add(key)
                all_results.append(r)

        # Sort by score descending, take top-k
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def serialize(self) -> dict:
        """Serialize to JSON-compatible dict for artifact persistence."""
        raw = self.embeddings.astype(np.float32).tobytes()
        return {
            "model": "text-embedding-3-small",
            "chunk_count": len(self.chunks),
            "dim": int(self.embeddings.shape[1]) if self.embeddings.ndim == 2 else 0,
            "chunks": [
                {
                    "file_path": c.file_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "chunk_type": c.chunk_type,
                    "name": c.name,
                    "content": c.content,
                }
                for c in self.chunks
            ],
            "embeddings_b64": base64.b64encode(raw).decode("ascii"),
        }

    @classmethod
    def deserialize(cls, data: dict) -> CodeIndex:
        """Reconstruct from serialized form."""
        chunks = [
            CodeChunk(
                file_path=c["file_path"],
                start_line=c["start_line"],
                end_line=c["end_line"],
                chunk_type=c["chunk_type"],
                name=c.get("name"),
                content=c["content"],
            )
            for c in data["chunks"]
        ]
        dim = data.get("dim", 1536)
        raw = base64.b64decode(data["embeddings_b64"])
        n_floats = len(raw) // 4
        flat = list(struct.unpack(f"{n_floats}f", raw))
        arr = np.array(flat, dtype=np.float32).reshape(len(chunks), dim)
        # Note: constructor will re-normalize
        return cls(chunks, arr)
