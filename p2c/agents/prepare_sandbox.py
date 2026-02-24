from __future__ import annotations

import os
import platform
import shlex
from pathlib import Path

from p2c.agents.base import BaseAgent
from p2c.runtime.factory import ensure_runtime
from p2c.schemas import DataManifest, DataManifestEntry, SystemInfo

SYSTEM_PROMPT = (
    "You summarize runtime system info. Return strict JSON and avoid fabrication."
)

USER_PROMPT_TEMPLATE = "Input: local runtime info. Output: execution/system_info.json"

DATA_DIR_NAMES = {"data", "dataset", "datasets", "input", "inputs"}
DATA_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".npy", ".npz", ".txt", ".pt", ".pth"}


class PrepareSandboxAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="prepare_sandbox", *args, **kwargs)

    @staticmethod
    def _discover_data_entries(repo_dir: Path) -> list[Path]:
        picked: list[Path] = []
        seen: set[Path] = set()

        # Prefer canonical data directories.
        for p in sorted(repo_dir.rglob("*")):
            if p.is_dir() and p.name.lower() in DATA_DIR_NAMES:
                if p not in seen:
                    picked.append(p)
                    seen.add(p)

        # Add loose data files not already covered by picked directories.
        for p in sorted(repo_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in DATA_SUFFIXES:
                continue
            if any(parent in seen for parent in p.parents):
                continue
            if p not in seen:
                picked.append(p)
                seen.add(p)

        return picked

    @staticmethod
    def _pick_workspace_root(runtime) -> str:
        override = (os.getenv("P2C_WORKSPACE_ROOT") or "").strip()
        preferred = [override] if override else []
        preferred.extend(["/workspace", "/home/user/workspace", "/home/sandbox/workspace", "/tmp/workspace"])
        # Deduplicate while preserving order.
        uniq: list[str] = []
        for p in preferred:
            if p and p not in uniq:
                uniq.append(p)
        errors: list[str] = []
        for root in uniq:
            try:
                root_q = shlex.quote(root)
                probe = runtime.run_command(
                    f"mkdir -p {root_q} && test -w {root_q}",
                    cwd="/",
                    timeout_sec=20,
                )
                if probe.rc == 0:
                    return root
                errors.append(f"{root}: rc={probe.rc} stderr={probe.stderr[:120]}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{root}: {e}")
                continue
        detail = "; ".join(errors[-4:]) if errors else "no candidates checked"
        raise RuntimeError(f"no writable workspace root found; candidates={uniq}; detail={detail}")

    def execute(self, ctx: dict) -> dict:
        runtime = ensure_runtime(ctx, self.artifacts)
        if (getattr(runtime, "backend_name", "") or "").lower() != "e2b":
            raise RuntimeError("prepare_sandbox requires P2C_RUNTIME_BACKEND=e2b")
        llm_schema = {
            "type": "object",
            "properties": {
                "notes": {"type": "string"},
                "reason_codes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["notes", "reason_codes"],
        }
        llm_data, _ = self.safe_chat_json(llm_schema, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)

        repo_dir = Path(ctx.get("repo_dir", ""))
        if not repo_dir.exists():
            raise FileNotFoundError(f"repo_dir not found: {repo_dir}")

        task_spec_local = self.artifacts.path("task/task_spec.json")
        claims_ir_local = self.artifacts.path("fingerprint/claims_ir.json")
        if not task_spec_local.exists():
            raise FileNotFoundError(f"missing required artifact: {task_spec_local}")
        if not claims_ir_local.exists():
            raise FileNotFoundError(f"missing required artifact: {claims_ir_local}")

        workspace_root = self._pick_workspace_root(runtime)
        workspace_repo_dir = f"{workspace_root}/repo"
        workspace_data_dir = f"{workspace_root}/data"
        workspace_inputs_dir = f"{workspace_root}/inputs"
        workspace_outputs_dir = f"{workspace_root}/outputs"

        ctx["workspace_root"] = workspace_root
        ctx["workspace_repo_dir"] = workspace_repo_dir
        ctx["workspace_data_dir"] = workspace_data_dir
        ctx["workspace_inputs_dir"] = workspace_inputs_dir
        ctx["workspace_outputs_dir"] = workspace_outputs_dir
        # Keep backward-compatible keys for agents that still look for runtime_*.
        ctx["runtime_repo_dir"] = workspace_repo_dir

        repo_q = shlex.quote(workspace_repo_dir)
        data_q = shlex.quote(workspace_data_dir)
        inputs_q = shlex.quote(workspace_inputs_dir)
        outputs_q = shlex.quote(workspace_outputs_dir)
        mk = runtime.run_command(
            f"mkdir -p {repo_q} {data_q} {inputs_q} {outputs_q}",
            cwd=workspace_root,
            timeout_sec=30,
        )
        if mk.rc != 0:
            raise RuntimeError(f"failed to prepare workspace dirs: {mk.stderr[:300]}")

        include_git = (os.getenv("P2C_INCLUDE_GIT") or "").strip() == "1"
        repo_excludes = None if include_git else [".git", ".git/**"]
        self.artifacts.append_text(
            "execution/run.log",
            f"[prepare_sandbox] repo upload mode=local_dir include_git={1 if include_git else 0}\n",
        )
        runtime.upload_dir(local_dir=repo_dir, remote_dir=workspace_repo_dir, exclude_globs=repo_excludes)
        runtime.upload_file(local_file=task_spec_local, remote_path=f"{workspace_inputs_dir}/task_spec.json")
        runtime.upload_file(local_file=claims_ir_local, remote_path=f"{workspace_inputs_dir}/claims_ir.json")

        entries: list[DataManifestEntry] = []
        for src in self._discover_data_entries(repo_dir):
            rel = src.relative_to(repo_dir)
            remote = f"{workspace_data_dir}/{str(rel).replace('\\', '/')}"
            try:
                if src.is_dir():
                    runtime.upload_dir(local_dir=src, remote_dir=remote)
                    size = None
                else:
                    runtime.upload_file(local_file=src, remote_path=remote)
                    size = src.stat().st_size
                entries.append(
                    DataManifestEntry(
                        path=str(rel),
                        exists=True,
                        size_bytes=size,
                        sandbox_path=remote,
                    )
                )
            except Exception:  # noqa: BLE001
                entries.append(
                    DataManifestEntry(
                        path=str(rel),
                        exists=False,
                        size_bytes=None,
                        sandbox_path=remote,
                    )
                )

        manifest = DataManifest(
            entries=entries,
            unresolved=len(entries) == 0,
            reason_codes=["NO_DATA_FILES_DISCOVERED"] if len(entries) == 0 else [],
        )
        self.artifacts.write_json("execution/data_manifest.json", manifest.model_dump())

        mem_gb = None
        try:
            probe = runtime.run_command(
                "python3 - <<'PY'\nimport os,platform\nprint(platform.system())\nprint(platform.release())\nprint(platform.python_version())\ntry:\n p=os.sysconf('SC_PAGE_SIZE')\n n=os.sysconf('SC_PHYS_PAGES')\n print(round((p*n)/(1024**3),2))\nexcept Exception:\n print('')\nPY",
                cwd=workspace_repo_dir,
                timeout_sec=30,
            )
            lines = [x.strip() for x in (probe.stdout or "").splitlines() if x.strip()]
            platform_name = lines[0] if len(lines) > 0 else platform.system()
            release = lines[1] if len(lines) > 1 else platform.release()
            pyver = lines[2] if len(lines) > 2 else platform.python_version()
            if len(lines) > 3:
                try:
                    mem_gb = float(lines[3])
                except ValueError:
                    mem_gb = None
        except Exception:  # noqa: BLE001
            platform_name = platform.system()
            release = platform.release()
            pyver = platform.python_version()

        info = SystemInfo(
            platform=platform_name,
            platform_release=release,
            python_version=pyver,
            cpu_count=None,
            memory_gb=mem_gb,
            reason_codes=(llm_data or {}).get("reason_codes", []),
        )
        self.artifacts.write_json("execution/system_info.json", info.model_dump())

        return {
            "system_info": info.model_dump(),
            "data_manifest": manifest.model_dump(),
            "workspace": {
                "repo": workspace_repo_dir,
                "data": workspace_data_dir,
                "inputs": workspace_inputs_dir,
                "outputs": workspace_outputs_dir,
            },
        }
