from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from p2c.runtime.base import RuntimeCommandResult


class LocalRuntime:
    backend_name = "local"

    def __init__(self) -> None:
        self._started = False

    def ensure_started(self) -> None:
        self._started = True

    def upload_dir(self, local_dir: Path, remote_dir: str) -> None:
        target = Path(remote_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(local_dir, target)

    def run_command(self, command: str, cwd: str, timeout_sec: int = 120) -> RuntimeCommandResult:
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        return RuntimeCommandResult(
            command=command,
            cwd=cwd,
            rc=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
        )

    def read_text(self, remote_path: str) -> str:
        return Path(remote_path).read_text(encoding="utf-8", errors="ignore")

    def write_text(self, remote_path: str, content: str) -> None:
        p = Path(remote_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def close(self) -> None:
        self._started = False

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "started": self._started,
        }
