"""Query the code index with paper claims and format results for prompt injection."""

from __future__ import annotations

import logging
from typing import Any

from p2c.rag.embeddings import EmbeddingClient, EmbeddingError
from p2c.rag.index import CodeIndex, RetrievalResult

logger = logging.getLogger(__name__)


def retrieve_for_claims(
    index: CodeIndex | None,
    claims: list[dict[str, Any]],
    *,
    top_k: int = 10,
    max_chars: int = 12000,
) -> str:
    """Retrieve code context relevant to a set of claims.

    Returns a formatted prompt section string.
    Returns empty string if *index* is ``None`` or retrieval fails
    (graceful degradation).
    """
    if index is None or not claims:
        return ""

    # Build query strings from claims
    queries = _claims_to_queries(claims)
    if not queries:
        return ""

    try:
        client = EmbeddingClient()
        query_vectors = client.embed_batch(queries)
    except (EmbeddingError, Exception):  # noqa: BLE001
        logger.warning("RAG query embedding failed, skipping retrieval")
        return ""

    results = index.query_multi(query_vectors, top_k=top_k)
    if not results:
        return ""

    return _format_retrieved_context(results, max_chars)


def _claims_to_queries(claims: list[dict[str, Any]]) -> list[str]:
    """Convert claim dicts into query strings for embedding."""
    queries: list[str] = []
    seen: set[str] = set()

    for claim in claims:
        parts: list[str] = []

        # Use predicate/fact as main query text
        pred = str(claim.get("predicate") or claim.get("fact") or "").strip()
        if pred:
            parts.append(pred)

        # Add metric name for specificity
        metric = str(claim.get("metric") or claim.get("metric_name") or "").strip()
        if metric:
            parts.append(f"metric: {metric}")

        # Add description if available
        desc = str(claim.get("description") or "").strip()
        if desc and desc != pred:
            parts.append(desc)

        query = " | ".join(parts)
        if query and query not in seen:
            seen.add(query)
            queries.append(query)

    return queries


def _format_retrieved_context(
    results: list[RetrievalResult],
    max_chars: int,
) -> str:
    """Format retrieval results as a prompt section."""
    lines: list[str] = ["## Retrieved Code Context (RAG)"]
    char_count = len(lines[0])

    for r in results:
        header = f"### {r.chunk.file_path} (lines {r.chunk.start_line}-{r.chunk.end_line}, score={r.score:.2f})"
        # Determine code fence language
        ext = r.chunk.file_path.rsplit(".", 1)[-1] if "." in r.chunk.file_path else ""
        lang = {"py": "python", "sh": "bash", "yaml": "yaml", "yml": "yaml",
                "toml": "toml", "r": "r", "jl": "julia"}.get(ext, "")

        block = f"{header}\n```{lang}\n{r.chunk.content.rstrip()}\n```\n"

        if char_count + len(block) > max_chars:
            break

        lines.append(block)
        char_count += len(block)

    if len(lines) <= 1:
        return ""

    return "\n".join(lines)
