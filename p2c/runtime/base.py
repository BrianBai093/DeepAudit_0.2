from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class RuntimeCommandResult:
    command: str
    cwd: str
    rc: int
    stdout: str
    stderr: str


class ExecutionRuntime(Protocol):
    backend_name: str

    def ensure_started(self) -> None: ...

    def upload_dir(
        self,
        local_dir: Path,
        remote_dir: str,
        exclude_globs: list[str] | None = None,
    ) -> None: ...

    def upload_file(self, local_file: Path, remote_path: str) -> None: ...

    def download_file(self, remote_path: str, local_file: Path) -> None: ...

    def run_command(self, command: str, cwd: str, timeout_sec: int = 120) -> RuntimeCommandResult: ...

    def read_text(self, remote_path: str) -> str: ...

    def write_text(self, remote_path: str, content: str) -> None: ...

    def close(self) -> None: ...

    def metadata(self) -> dict[str, Any]: ...
