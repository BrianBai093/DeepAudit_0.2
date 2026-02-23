from __future__ import annotations

import base64
import io
import os
import tarfile
from pathlib import Path
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

        attempts: list[dict[str, Any]] = [
            {"api_key": api_key, "timeout": self._timeout_sec},
            {"api_key": api_key},
            {"timeout": self._timeout_sec},
            {},
        ]
        type_errors: list[str] = []

        # Newer SDKs expose factory classmethods and do not allow direct __init__.
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

        # Fallback for SDKs that still allow direct constructor.
        if self._sandbox is None:
            for kwargs in attempts:
                try:
                    self._sandbox = sandbox_cls(**kwargs)
                    break
                except TypeError as e:
                    # Signature mismatch across SDK versions; try next constructor shape.
                    type_errors.append(f"{sandbox_cls_ref}.__init__{sorted(kwargs.keys())}: {e}")
                    continue
                except Exception as e:  # noqa: BLE001
                    keys = ",".join(sorted(kwargs.keys())) or "<none>"
                    raise E2BRuntimeError(
                        f"Failed to initialize E2B sandbox with ctor args [{keys}] from {sandbox_cls_ref}: {e}"
                    ) from e

        if self._sandbox is None:
            detail = "; ".join(type_errors[-6:]) if type_errors else "no constructor/factory attempts executed"
            raise E2BRuntimeError(
                f"Unable to initialize E2B sandbox from {sandbox_cls_ref} due to constructor/factory mismatch: {detail}"
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

    def _write_file_with_fallback(self, filename: str, content: str) -> str:
        files = self._files_api()
        candidates = [
            f"/home/user/{filename}",
            f"/home/sandbox/{filename}",
            f"/workspace/{filename}",
            f"/tmp/{filename}",
        ]
        errors: list[str] = []
        for path in candidates:
            parent = str(Path(path).parent)
            try:
                # Best effort parent creation; if mkdir fails we'll try next path.
                self.run_command(f"mkdir -p {parent}", cwd="/", timeout_sec=20)
                files.write(path, content)
                # Validate readability to fail fast on permission quirks.
                _ = files.read(path)
                return path
            except Exception as e:  # noqa: BLE001
                errors.append(f"{path}: {e}")
                continue

        detail = "; ".join(errors[-4:]) if errors else "no writable path candidates tried"
        raise E2BRuntimeError(f"Failed to write upload payload to sandbox filesystem: {detail}")

    def upload_dir(self, local_dir: Path, remote_dir: str) -> None:
        self.ensure_started()
        if not local_dir.exists() or not local_dir.is_dir():
            raise E2BRuntimeError(f"upload_dir source missing: {local_dir}")

        # Package local dir to tar.gz and upload as base64, then extract in sandbox.
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            tf.add(local_dir, arcname="repo")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")

        b64_path = self._write_file_with_fallback("p2c_repo.tgz.b64", encoded)
        scratch_root = str(Path(b64_path).parent / "p2c_repo_extract")

        extract_cmd = (
            f"mkdir -p {remote_dir} && "
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

    def run_command(self, command: str, cwd: str, timeout_sec: int = 120) -> RuntimeCommandResult:
        self.ensure_started()
        cmds = self._commands_api()

        run_kwargs = {
            "cmd": command,
            "cwd": cwd,
            "timeout": timeout_sec,
        }

        run_fn = getattr(cmds, "run", None)
        if run_fn is None:
            raise E2BRuntimeError("E2B commands.run not available")

        try:
            out = run_fn(**run_kwargs)
        except TypeError:
            # Fallback API variants.
            out = run_fn(command)

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
