"""Shared Claude Code Agent SDK runner for Phase 2 repair sessions."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

try:
    from claude_agent_sdk import (  # type: ignore[import-untyped]
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
        query,
    )

    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False
    AssistantMessage = type("AssistantMessage", (), {})  # type: ignore[misc,assignment]
    ClaudeAgentOptions = type("ClaudeAgentOptions", (), {})  # type: ignore[misc,assignment]
    ResultMessage = type("ResultMessage", (), {})  # type: ignore[misc,assignment]
    ToolResultBlock = type("ToolResultBlock", (), {})  # type: ignore[misc,assignment]
    ToolUseBlock = type("ToolUseBlock", (), {})  # type: ignore[misc,assignment]
    UserMessage = type("UserMessage", (), {})  # type: ignore[misc,assignment]

    async def query(**kwargs):  # type: ignore[misc]
        raise RuntimeError("claude-agent-sdk is not installed")
        yield


DEFAULT_REPAIR_CLAUDE_MODEL = "claude-haiku-4-5-20251001"
_FORWARD_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "CONDA_EXE",
    "CONDA_PREFIX",
    "HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "NO_PROXY",
    "OPENAI_API_KEY",
    "PATH",
    "SHELL",
    "USER",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


@dataclass(frozen=True)
class ClaudeCodeSessionResult:
    stdout: str
    stderr: str
    narrative: str
    returncode: int


def claude_code_sdk_available() -> bool:
    return _SDK_AVAILABLE


def _stream_prompt(prompt: str) -> AsyncIterator[dict[str, Any]]:
    async def _prompt_messages() -> AsyncIterator[dict[str, Any]]:
        yield {
            "type": "user",
            "session_id": "",
            "message": {
                "role": "user",
                "content": prompt,
            },
            "parent_tool_use_id": None,
        }

    return _prompt_messages()


def _extract_tool_text(block: Any) -> str:
    content = getattr(block, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content or "")


def run_claude_code_session(
    *,
    prompt: str,
    cwd: Path,
    system_prompt: str,
    artifacts: Any,
    log_prefix: str,
    timeout_sec: int,
    max_turns: int,
    allowed_tools: list[str],
    permission_mode: str = "bypassPermissions",
) -> ClaudeCodeSessionResult:
    if not _SDK_AVAILABLE:
        return ClaudeCodeSessionResult("", "claude-agent-sdk is not installed", "", 1)

    model = (os.getenv("P2C_CLAUDE_MODEL") or DEFAULT_REPAIR_CLAUDE_MODEL).strip()
    child_env = {key: value for key, value in os.environ.items() if key in _FORWARD_ENV_KEYS and value}
    stdout_rel = f"{log_prefix}/sdk_session_stdout.log"
    stderr_rel = f"{log_prefix}/sdk_session_stderr.log"
    narrative_rel = f"{log_prefix}/sdk_session_narrative.log"
    artifacts.write_text(stdout_rel, "")
    artifacts.write_text(stderr_rel, "")
    artifacts.write_text(narrative_rel, "")

    async def _execute() -> ClaudeCodeSessionResult:
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        narrative_parts: list[str] = []
        last_exit_code = 0

        options_kwargs: dict[str, Any] = {
            "cwd": str(cwd),
            "allowed_tools": allowed_tools,
            "permission_mode": permission_mode,
            "max_turns": max_turns,
            "model": model,
            "system_prompt": system_prompt,
            "env": child_env,
        }
        async for msg in query(
            prompt=_stream_prompt(prompt),
            options=ClaudeAgentOptions(**options_kwargs),
        ):
            if isinstance(msg, AssistantMessage):
                for block in getattr(msg, "content", []):
                    if isinstance(block, ToolUseBlock):
                        name = getattr(block, "name", "")
                        tool_input = getattr(block, "input", {}) or {}
                        command = tool_input.get("command") if isinstance(tool_input, dict) else ""
                        if command:
                            line = f"[tool_use:{name}] {command}"
                        else:
                            line = f"[tool_use:{name}]"
                        narrative_parts.append(line)
                        artifacts.append_text(narrative_rel, line + "\n")
                        continue
                    text = getattr(block, "text", None)
                    if isinstance(text, str) and text.strip():
                        narrative_parts.append(text)
                        artifacts.append_text(narrative_rel, f"[assistant]\n{text.rstrip()}\n")
            elif isinstance(msg, UserMessage):
                for block in getattr(msg, "content", []):
                    if not isinstance(block, ToolResultBlock):
                        continue
                    text = _extract_tool_text(block)
                    if text:
                        stdout_parts.append(text)
                        artifacts.append_text(stdout_rel, text.rstrip() + "\n")
                        artifacts.append_text(narrative_rel, f"[tool]\n{text.rstrip()}\n")
                    if getattr(block, "is_error", False):
                        last_exit_code = 1
                        stderr_parts.append(text)
                        artifacts.append_text(stderr_rel, text.rstrip() + "\n")
                        artifacts.append_text(narrative_rel, f"[tool_error]\n{text.rstrip()}\n")
            elif isinstance(msg, ResultMessage):
                result = getattr(msg, "result", None)
                if result:
                    narrative_parts.append(result)
                    artifacts.append_text(narrative_rel, f"[result]\n{str(result).rstrip()}\n")
                subtype = getattr(msg, "subtype", "")
                if subtype and subtype != "success":
                    last_exit_code = 1

        return ClaudeCodeSessionResult(
            stdout="\n".join(stdout_parts),
            stderr="\n".join(stderr_parts),
            narrative="\n".join(narrative_parts),
            returncode=last_exit_code,
        )

    try:
        return asyncio.run(asyncio.wait_for(_execute(), timeout=float(timeout_sec)))
    except asyncio.TimeoutError:
        message = f"Claude Code SDK session timed out after {timeout_sec}s"
        artifacts.append_text(stderr_rel, message + "\n")
        return ClaudeCodeSessionResult("", message, "", 1)
    except Exception as exc:  # noqa: BLE001
        message = f"Claude Code SDK session error: {exc}"
        artifacts.append_text(stderr_rel, message + "\n")
        return ClaudeCodeSessionResult("", message, "", 1)
