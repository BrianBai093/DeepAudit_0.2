"""Embedding client — calls OpenAI embeddings API via urllib."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class EmbeddingError(RuntimeError):
    pass


class EmbeddingClient:
    """Calls OpenAI text-embedding-3-small via urllib, matching LLMClient patterns."""

    MODEL = "text-embedding-3-small"
    DIMENSIONS = 1536
    BATCH_SIZE = 64

    def __init__(self) -> None:
        key = os.getenv("OPENAI_API_KEY")
        self.api_key = key.strip() if key else None
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.base_url = (base_url.strip() if base_url else "https://api.openai.com/v1")
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url.rstrip("/") + "/v1"
        model = os.getenv("OPENAI_EMBEDDING_MODEL", self.MODEL)
        self.model = model.strip() if model else self.MODEL

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches. Returns list of float vectors."""
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            # Truncate very long texts to avoid token limits
            batch = [t[:8000] for t in batch]
            data = self._post_embeddings(batch)
            vectors = self._extract_vectors(data, expected=len(batch))
            all_embeddings.extend(vectors)
        return all_embeddings

    def embed_one(self, text: str) -> list[float]:
        """Convenience for a single text."""
        results = self.embed_batch([text])
        return results[0]

    def _post_embeddings(self, texts: list[str]) -> dict:
        if not self.api_key:
            raise EmbeddingError("OPENAI_API_KEY is not set")

        url = f"{self.base_url}/embeddings"
        payload = {
            "model": self.model,
            "input": texts,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            details = e.read().decode("utf-8", errors="ignore")
            raise EmbeddingError(f"OpenAI embedding HTTP error {e.code}: {details}") from e
        except urllib.error.URLError as e:
            raise EmbeddingError(f"OpenAI embedding request failed: {e}") from e
        except Exception as e:  # noqa: BLE001
            raise EmbeddingError(f"OpenAI embedding unexpected error: {e}") from e

    @staticmethod
    def _extract_vectors(data: dict, expected: int) -> list[list[float]]:
        """Extract embedding vectors from API response, sorted by index."""
        try:
            items = data["data"]
            items_sorted = sorted(items, key=lambda x: x["index"])
            vectors = [item["embedding"] for item in items_sorted]
        except (KeyError, TypeError, IndexError) as e:
            raise EmbeddingError(f"Unexpected embedding response shape: {data}") from e
        if len(vectors) != expected:
            raise EmbeddingError(
                f"Expected {expected} embeddings, got {len(vectors)}"
            )
        return vectors
