from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from p2c.io_artifacts import ArtifactManager
from p2c.llm.client import LLMClient, LLMClientError
from p2c.utils.console import format_log


class AgentError(RuntimeError):
    pass


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        llm: LLMClient,
        artifacts: ArtifactManager,
        step_index: int,
        step_total: int,
        max_retries: int = 2,
    ):
        self.name = name
        self.llm = llm
        self.artifacts = artifacts
        self.step_index = step_index
        self.step_total = step_total
        self.max_retries = max_retries

    def log(self, state: str, message: str) -> None:
        line = format_log(
            agent=self.name,
            state=state,
            step=f"{self.step_index}/{self.step_total}",
            message=message,
        )
        print(line, flush=True)
        self.artifacts.append_text("execution/run.log", line + "\n")

    def run(self, ctx: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        self.log("START", f"starting with ctx keys={sorted(ctx.keys())}")
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                result = self.execute(ctx)
                elapsed = time.time() - started
                self.log("DONE", f"completed in {elapsed:.2f}s")
                return result
            except Exception as e:  # noqa: BLE001
                last_error = e
                self.log("PROGRESS", f"attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(0.5 * attempt)
        raise AgentError(f"{self.name} failed after retries: {last_error}")

    def safe_chat_json(
        self,
        schema: dict[str, Any],
        system: str,
        user: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            data = self.llm.chat_json(schema=schema, system=system, user=user)
            return data, None
        except LLMClientError as e:
            self.log("PROGRESS", f"LLM unavailable: {e}")
            return None, str(e)

    def safe_chat_text(self, system: str, user: str) -> tuple[str | None, str | None]:
        try:
            text = self.llm.chat_text(system=system, user=user)
            return text, None
        except LLMClientError as e:
            self.log("PROGRESS", f"LLM unavailable: {e}")
            return None, str(e)

    @abstractmethod
    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
