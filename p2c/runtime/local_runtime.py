from __future__ import annotations

import shutil
import subprocess
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from p2c.runtime.base import RuntimeCommandResult


class LocalRuntime:
    backend_name = "local"

    def __init__(self) -> None:
        self._started = False

    def ensure_started(self) -> None:
        self._started = True

    @staticmethod
    def _is_excluded(rel_path: str, exclude_globs: list[str]) -> bool:
        if not rel_path:
            return False
        rel = rel_path.replace("\\", "/")
        for pattern in exclude_globs:
            pat = pattern.replace("\\", "/").strip()
            if not pat:
                continue
            if pat.endswith("/**"):
                prefix = pat[:-3].rstrip("/")
                if rel == prefix or rel.startswith(prefix + "/"):
                    return True
            if fnmatch(rel, pat):
                return True
        return False

    def upload_dir(
        self,
        local_dir: Path,
        remote_dir: str,
        exclude_globs: list[str] | None = None,
    ) -> None:
        target = Path(remote_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        exclude = list(exclude_globs or [])
        for src in sorted(local_dir.rglob("*")):
            rel = src.relative_to(local_dir)
            rel_posix = rel.as_posix()
            if self._is_excluded(rel_posix, exclude):
                continue
            dst = target / rel
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    def upload_file(self, local_file: Path, remote_path: str) -> None:
        target = Path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_file, target)

    def download_file(self, remote_path: str, local_file: Path) -> None:
        local_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(remote_path), local_file)

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
