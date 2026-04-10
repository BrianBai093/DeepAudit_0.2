"""Build a CodeIndex from a target repository."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from p2c.rag.chunker import CodeChunk, chunk_repo
from p2c.rag.embeddings import EmbeddingClient, EmbeddingError
from p2c.rag.index import CodeIndex

if TYPE_CHECKING:
    from p2c.io_artifacts import ArtifactManager

logger = logging.getLogger(__name__)

_RAG_FILE_COUNT_THRESHOLD = 20


def build_code_index(
    repo_dir: Path,
    artifacts: ArtifactManager,
    *,
    min_files_threshold: int = _RAG_FILE_COUNT_THRESHOLD,
) -> CodeIndex | None:
    """Build a code embedding index for the target repo.

    Returns ``None`` when RAG is not worth it (small repo) or when
    the embedding API is unavailable — callers should fall back to
    the existing prompt-stuffing approach.

    Persists the index as ``task/code_index.json``.
    """
    chunks = chunk_repo(repo_dir)

    # Count unique files
    unique_files = {c.file_path for c in chunks}
    if len(unique_files) < min_files_threshold:
        logger.info(
            "RAG skipped: repo has %d indexable files (threshold=%d)",
            len(unique_files),
            min_files_threshold,
        )
        return None

    logger.info("RAG: chunked %d files into %d chunks", len(unique_files), len(chunks))

    # Embed all chunks
    try:
        client = EmbeddingClient()
        texts = [_chunk_to_embedding_text(c) for c in chunks]
        vectors = client.embed_batch(texts)
    except (EmbeddingError, Exception) as exc:  # noqa: BLE001
        logger.warning("RAG embedding failed, falling back to non-RAG: %s", exc)
        return None

    embeddings = np.array(vectors, dtype=np.float32)
    index = CodeIndex(chunks, embeddings)

    # Persist
    try:
        artifacts.write_json("task/code_index.json", index.serialize())
        logger.info("RAG index persisted to task/code_index.json (%d chunks)", len(chunks))
    except Exception:  # noqa: BLE001
        logger.warning("Failed to persist RAG index, continuing in-memory only")

    return index


def _chunk_to_embedding_text(chunk: CodeChunk) -> str:
    """Build the text to embed for a chunk — includes file path as context."""
    header = f"# {chunk.file_path}"
    if chunk.name:
        header += f" :: {chunk.name}"
    header += f" (lines {chunk.start_line}-{chunk.end_line})"
    return f"{header}\n{chunk.content}"
