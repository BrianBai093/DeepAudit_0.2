from __future__ import annotations

import base64
import io
import os
import shlex
import tarfile
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any

from p2c.runtime.base import RuntimeCommandResult


class E2BRuntimeError(RuntimeError):
    pass


class E2BRuntime:
    backend_name = "e2b"

    def __init__(self, timeout_sec: int = 3600) -> None:
        self._timeout_sec = timeout_sec
        self._sandbox = None
        self._sandbox_id: str | None = None

    def ensure_started(self) -> None:
        if self._sandbox is not None:
            return

        api_key = (os.getenv("E2B_API_KEY") or "").strip()
        if not api_key:
            raise E2BRuntimeError("E2B_API_KEY is required for e2b runtime")
        openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()

        sandbox_cls = None
        sandbox_cls_ref = ""
        last_err: Exception | None = None
        for mod, name in [
            ("e2b", "Sandbox"),
            ("e2b_code_interpreter", "Sandbox"),
        ]:
            try:
                m = __import__(mod, fromlist=[name])
                sandbox_cls = getattr(m, name)
                sandbox_cls_ref = f"{mod}.{name}"
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue

        if sandbox_cls is None:
            raise E2BRuntimeError(f"E2B SDK not installed or incompatible: {last_err}")

        # Some SDK versions read E2B_API_KEY from environment and do not accept an `api_key` ctor arg.
        os.environ.setdefault("E2B_API_KEY", api_key)
        envs = {"OPENAI_API_KEY": openai_key} if openai_key else {}
        attempts: list[dict[str, Any]] = [
            {"template": "openai-codex", "envs": envs, "timeout": self._timeout_sec},
            {"template": "openai-codex", "envs": envs},
            {"template": "openai-codex", "timeout": self._timeout_sec},
            {"template": "openai-codex"},
        ]
        type_errors: list[str] = []

        # Enforce template-based creation; do not fallback to raw constructor.
        for method_name in ("create", "spawn", "new", "start"):
            factory = getattr(sandbox_cls, method_name, None)
            if not callable(factory):
                continue
            for kwargs in attempts:
                try:
                    self._sandbox = factory(**kwargs)
                    break
                except TypeError as e:
                    type_errors.append(f"{sandbox_cls_ref}.{method_name}{sorted(kwargs.keys())}: {e}")
                    continue
                except Exception as e:  # noqa: BLE001
                    keys = ",".join(sorted(kwargs.keys())) or "<none>"
                    raise E2BRuntimeError(
                        f"Failed to initialize E2B via {sandbox_cls_ref}.{method_name} with args [{keys}]: {e}"
                    ) from e
            if self._sandbox is not None:
                break

        if self._sandbox is None:
            detail = "; ".join(type_errors[-6:]) if type_errors else "no constructor/factory attempts executed"
            raise E2BRuntimeError(
                "Unable to initialize E2B sandbox with required template 'openai-codex' "
                f"from {sandbox_cls_ref}: {detail}"
            )

        self._sandbox_id = str(getattr(self._sandbox, "sandbox_id", None) or getattr(self._sandbox, "id", "unknown"))

    def _commands_api(self):
        assert self._sandbox is not None
        cmds = getattr(self._sandbox, "commands", None)
        if cmds is None:
            raise E2BRuntimeError("E2B Sandbox.commands API not available")
        return cmds

    def _files_api(self):
        assert self._sandbox is not None
        files = getattr(self._sandbox, "files", None)
        if files is None:
            raise E2BRuntimeError("E2B Sandbox.files API not available")
        return files

    @staticmethod
    def _normalize_remote_path(path: str) -> str:
        raw = str(path or "").strip().replace("\\", "/")
        if not raw:
            raise E2BRuntimeError("remote path cannot be empty")
        if not raw.startswith("/"):
            raise E2BRuntimeError(f"remote path must be absolute POSIX path: {path!r}")
        if ".." in raw.split("/"):
            raise E2BRuntimeError(f"remote path must not contain '..': {path!r}")
        # Collapse duplicate separators while keeping POSIX semantics.
        normalized = str(PurePosixPath(raw))
        return normalized

    def _write_file_with_fallback(self, filename: str, content: str) -> str:
        files = self._files_api()
        safe_name = PurePosixPath(filename).name
        candidates = [
            f"/tmp/{safe_name}",
            f"/var/tmp/{safe_name}",
        ]
        errors: list[str] = []
        for path in candidates:
            path = self._normalize_remote_path(path)
            try:
                files.write(path, content)
                # Validate readability to fail fast on permission quirks.
                _ = files.read(path)
                return path
            except Exception as e:  # noqa: BLE001
                errors.append(f"{path}: {e}")
                continue

        detail = "; ".join(errors[-4:]) if errors else "no writable path candidates tried"
        raise E2BRuntimeError(f"Failed to write upload payload to sandbox filesystem: {detail}")

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
        self.ensure_started()
        if not local_dir.exists() or not local_dir.is_dir():
            raise E2BRuntimeError(f"upload_dir source missing: {local_dir}")
        remote_dir = self._normalize_remote_path(remote_dir)
        exclude = list(exclude_globs or [])

        # Package local dir to tar.gz and upload as base64, then extract in sandbox.
        def _tar_filter(tar_info: tarfile.TarInfo) -> tarfile.TarInfo | None:
            if not exclude:
                return tar_info
            parts = PurePosixPath(tar_info.name).parts
            rel = "/".join(parts[1:]) if len(parts) > 1 else ""
            if self._is_excluded(rel, exclude):
                return None
            return tar_info

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            tf.add(local_dir, arcname="repo", filter=_tar_filter)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")

        b64_path = self._write_file_with_fallback("p2c_repo.tgz.b64", encoded)
        scratch_root = str(PurePosixPath(b64_path).parent / "p2c_repo_extract")
        remote_dir_q = shlex.quote(remote_dir)

        extract_cmd = (
            f"mkdir -p {remote_dir_q} && "
            "python3 - <<'PY'\n"
            "import base64, tarfile, io, os, shutil\n"
            f"b64_path='{b64_path}'\n"
            f"target='{remote_dir}'\n"
            f"tmp='{scratch_root}'\n"
            "with open(b64_path,'r',encoding='utf-8') as f:\n"
            "    raw=base64.b64decode(f.read())\n"
            "bio=io.BytesIO(raw)\n"
            "if os.path.exists(tmp):\n"
            "    shutil.rmtree(tmp)\n"
            "os.makedirs(tmp, exist_ok=True)\n"
            "with tarfile.open(fileobj=bio, mode='r:gz') as tf:\n"
            "    tf.extractall(tmp)\n"
            "src=os.path.join(tmp,'repo')\n"
            "if os.path.exists(target):\n"
            "    shutil.rmtree(target)\n"
            "shutil.move(src, target)\n"
            "print('uploaded')\n"
            "PY"
        )
        result = self.run_command(extract_cmd, cwd="/", timeout_sec=300)
        if result.rc != 0:
            raise E2BRuntimeError(f"Failed to extract upload: {result.stderr[:500]}")

    def upload_file(self, local_file: Path, remote_path: str) -> None:
        self.ensure_started()
        if not local_file.exists() or not local_file.is_file():
            raise E2BRuntimeError(f"upload_file source missing: {local_file}")
        remote_path = self._normalize_remote_path(remote_path)
        raw = local_file.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        b64_path = self._write_file_with_fallback("p2c_file_upload.b64", encoded)
        cmd = (
            "python3 - <<'PY'\n"
            "import base64, os\n"
            f"src='{b64_path}'\n"
            f"dst='{remote_path}'\n"
            "os.makedirs(os.path.dirname(dst), exist_ok=True)\n"
            "with open(src,'r',encoding='utf-8') as f:\n"
            "    raw=base64.b64decode(f.read())\n"
            "with open(dst,'wb') as f:\n"
            "    f.write(raw)\n"
            "print('uploaded_file')\n"
            "PY"
        )
        result = self.run_command(cmd, cwd="/", timeout_sec=120)
        if result.rc != 0:
            raise E2BRuntimeError(f"Failed to upload file to {remote_path}: {result.stderr[:500]}")

    def download_file(self, remote_path: str, local_file: Path) -> None:
        self.ensure_started()
        remote_path = self._normalize_remote_path(remote_path)
        text = self.read_text(remote_path)
        local_file.parent.mkdir(parents=True, exist_ok=True)
        local_file.write_text(text, encoding="utf-8")

    def run_command(self, command: str, cwd: str, timeout_sec: int = 120) -> RuntimeCommandResult:
        self.ensure_started()
        cmds = self._commands_api()

        run_fn = getattr(cmds, "run", None)
        if run_fn is None:
            raise E2BRuntimeError("E2B commands.run not available")

        def _from_exception(exc: Exception) -> RuntimeCommandResult | None:
            # E2B SDK raises CommandExitException on non-zero exit code for some versions.
            exit_code = getattr(exc, "exit_code", None)
            if exit_code is None:
                return None
            stdout = getattr(exc, "stdout", "")
            stderr = getattr(exc, "stderr", "")
            err_msg = getattr(exc, "error", None)
            if err_msg and not stderr:
                stderr = str(err_msg)
            return RuntimeCommandResult(
                command=command,
                cwd=cwd,
                rc=int(exit_code),
                stdout=str(stdout or ""),
                stderr=str(stderr or ""),
            )

        kwargs_variants: list[dict[str, Any]] = []
        base_kwargs = {"cmd": command, "cwd": cwd, "timeout": timeout_sec}
        if timeout_sec == 0:
            # Some SDK/runtime combinations require request_timeout=0 too.
            kwargs_variants.append({**base_kwargs, "request_timeout": 0})
        kwargs_variants.append(base_kwargs)

        out = None
        last_type_err: Exception | None = None
        for run_kwargs in kwargs_variants:
            try:
                out = run_fn(**run_kwargs)
                break
            except TypeError as e:
                last_type_err = e
                continue
            except Exception as e:  # noqa: BLE001
                mapped = _from_exception(e)
                if mapped is not None:
                    return mapped
                raise
        if out is None:
            # Fallback API variants.
            try:
                out = run_fn(command)
            except TypeError as e:
                raise E2BRuntimeError(
                    f"E2B commands.run signature mismatch for command={command!r}: {last_type_err or e}"
                ) from e
            except Exception as e:  # noqa: BLE001
                mapped = _from_exception(e)
                if mapped is not None:
                    return mapped
                raise

        rc = getattr(out, "exit_code", None)
        if rc is None:
            rc = getattr(out, "code", None)
        if rc is None and isinstance(out, dict):
            rc = out.get("exit_code", out.get("code", 1))
        rc = int(rc if rc is not None else 1)

        stdout = getattr(out, "stdout", None)
        stderr = getattr(out, "stderr", None)
        if isinstance(out, dict):
            stdout = out.get("stdout", stdout)
            stderr = out.get("stderr", stderr)

        return RuntimeCommandResult(
            command=command,
            cwd=cwd,
            rc=rc,
            stdout=str(stdout or ""),
            stderr=str(stderr or ""),
        )

    def read_text(self, remote_path: str) -> str:
        self.ensure_started()
        remote_path = self._normalize_remote_path(remote_path)
        files = self._files_api()
        read_fn = getattr(files, "read", None)
        if read_fn is None:
            raise E2BRuntimeError("E2B files.read not available")
        out = read_fn(remote_path)
        if isinstance(out, bytes):
            return out.decode("utf-8", errors="ignore")
        return str(out)

    def write_text(self, remote_path: str, content: str) -> None:
        self.ensure_started()
        remote_path = self._normalize_remote_path(remote_path)
        files = self._files_api()
        write_fn = getattr(files, "write", None)
        if write_fn is None:
            raise E2BRuntimeError("E2B files.write not available")
        write_fn(remote_path, content)

    def close(self) -> None:
        if self._sandbox is None:
            return
        for fn_name in ["close", "kill", "stop"]:
            fn = getattr(self._sandbox, fn_name, None)
            if fn is None:
                continue
            try:
                fn()
                break
            except Exception:  # noqa: BLE001
                continue
        self._sandbox = None

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "sandbox_id": self._sandbox_id,
            "timeout_sec": self._timeout_sec,
        }
