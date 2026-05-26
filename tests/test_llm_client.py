from __future__ import annotations

from p2c.llm.client import LLMClient


def test_llm_client_timeout_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TIMEOUT_SEC", "420")
    monkeypatch.setenv("OPENAI_VISION_TIMEOUT_SEC", "600")

    client = LLMClient()

    assert client.timeout_sec == 420
    assert client.vision_timeout_sec == 600


def test_llm_client_timeout_defaults(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("OPENAI_VISION_TIMEOUT_SEC", raising=False)

    client = LLMClient()

    assert client.timeout_sec == 300
    assert client.vision_timeout_sec == 360
