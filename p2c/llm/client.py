from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    def __init__(self) -> None:
        key = os.getenv("OPENAI_API_KEY")
        self.api_key = key.strip() if key else None
        model = os.getenv("OPENAI_MODEL", "gpt-5.4")
        self.model = model.strip() if model else "gpt-5.4"
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.base_url = base_url.strip() if base_url else "https://api.openai.com/v1"
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url.rstrip("/") + "/v1"

    @staticmethod
    def _validate_ascii_env(name: str, value: str | None) -> None:
        if value is None:
            return
        try:
            value.encode("ascii")
        except UnicodeEncodeError as e:
            raise LLMClientError(
                f"{name} contains non-ASCII characters. "
                "Please re-set it with plain ASCII text in your shell."
            ) from e

    def chat_text(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = self._post_json(payload)
        return self._extract_content(data)

    def chat_json(self, schema: dict[str, Any], system: str, user: str) -> dict[str, Any]:
        schema_text = json.dumps(schema, ensure_ascii=False)
        strict_user = (
            f"{user}\n\nReturn exactly one JSON object matching this schema guide: {schema_text}. "
            "Do not include markdown or commentary. Fill missing fields with null/[] and reason_codes."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": strict_user},
            ],
            "response_format": {"type": "json_object"},
        }
        data = self._post_json(payload)
        content = self._extract_content(data)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", content, flags=re.S)
            if not match:
                raise LLMClientError("LLM did not return JSON")
            return json.loads(match.group(0))

    def chat_vision(
        self,
        system: str,
        user_text: str,
        images: list[str],
        detail: str = "high",
    ) -> str:
        """Multimodal chat with text + base64 image data URIs."""
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": img, "detail": detail},
            })
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        }
        data = self._post_json(payload, timeout=180)
        return self._extract_content(data)

    def chat_vision_json(
        self,
        schema: dict[str, Any],
        system: str,
        user_text: str,
        images: list[str],
        detail: str = "high",
    ) -> dict[str, Any]:
        """Vision call that returns structured JSON."""
        schema_text = json.dumps(schema, ensure_ascii=False)
        strict_text = (
            f"{user_text}\n\nReturn exactly one JSON object matching this schema guide: "
            f"{schema_text}. Do not include markdown or commentary."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": strict_text}]
        for img in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": img, "detail": detail},
            })
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "response_format": {"type": "json_object"},
        }
        data = self._post_json(payload, timeout=180)
        text = self._extract_content(data)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                raise LLMClientError("Vision LLM did not return JSON")
            return json.loads(match.group(0))

    def _post_json(self, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
        if not self.api_key:
            raise LLMClientError("OPENAI_API_KEY is not set")
        # urllib/http headers are latin-1 encoded. Non-ASCII key/base_url values cause opaque
        # UnicodeEncodeError like: "latin-1 codec can't encode characters in position 7-8".
        self._validate_ascii_env("OPENAI_API_KEY", self.api_key)
        self._validate_ascii_env("OPENAI_BASE_URL", self.base_url)
        url = f"{self.base_url}/chat/completions"
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
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            details = e.read().decode("utf-8", errors="ignore")
            raise LLMClientError(f"OpenAI HTTP error {e.code}: {details}") from e
        except urllib.error.URLError as e:
            raise LLMClientError(f"OpenAI request failed: {e}") from e
        except UnicodeEncodeError as e:
            raise LLMClientError(
                "OpenAI request header encoding failed. "
                "Check OPENAI_API_KEY/OPENAI_BASE_URL for non-ASCII characters."
            ) from e
        except Exception as e:  # noqa: BLE001
            raise LLMClientError(f"OpenAI client unexpected error: {e}") from e

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            raise LLMClientError(f"Unexpected OpenAI response shape: {data}") from e
        if isinstance(content, list):
            joined = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    joined.append(part.get("text", ""))
                elif isinstance(part, str):
                    joined.append(part)
            return "\n".join(joined)
        return str(content)
