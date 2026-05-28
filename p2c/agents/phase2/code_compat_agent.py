"""CodeCompatAgent — minimal source compatibility patches for repaired envs."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.claude_code_session import (
    claude_code_sdk_available,
    run_claude_code_session,
)
from p2c.agents.phase2.executor_agent import ExecutorAgent
from p2c.agents.phase2.tool_agent import ToolAgent
from p2c.schemas import CodeCompatResult, EnvRepairResult, ExecutorEnvSpec


class CodeCompatAgent(BaseAgent):
    """Apply LLM-generated compatibility patches after env repair."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="code_compat_agent", *args, **kwargs)

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        repo_dir = Path(str(ctx["repo_dir"])).resolve()
        env_mgr = ctx.get("_p2_env_mgr")
        if env_mgr is None:
            result = CodeCompatResult(
                status="failed",
                validation_passed=False,
                reason_codes=["CODE_COMPAT_ENV_MANAGER_MISSING"],
            )
            self._persist(result)
            return {"code_compat_result": result}

        env_repair_result = self._coerce_env_repair_result(ctx.get("_p2_env_repair_result"))
        if env_repair_result is None or env_repair_result.status != "success":
            result = CodeCompatResult(status="skipped", reason_codes=["CODE_COMPAT_SKIPPED_NO_REPAIRED_ENV"])
            self._persist(result)
            return {"code_compat_result": result}

        env_spec = self._load_repaired_env_spec(ctx)
        validation_command = self._build_validation_command(repo_dir, env_spec)
        baseline = self._capture_repo_state(repo_dir)
        baseline_text = self._capture_repo_text_snapshot(repo_dir)
        first_validation = env_mgr.run_in_env(validation_command, cwd=str(repo_dir), timeout_sec=120)
        if first_validation.returncode == 0:
            result = CodeCompatResult(
                status="success",
                changed_files=[],
                patch_path=None,
                validation_command=validation_command,
                validation_passed=True,
                stdout_tail=first_validation.stdout[-2000:],
                stderr_tail=first_validation.stderr[-2000:],
                reason_codes=["CODE_COMPAT_NO_PATCH_NEEDED"],
            )
            self._persist(result)
            return {"code_compat_result": result}

        sdk_result = self._run_claude_sdk_compat(
            repo_dir=repo_dir,
            env_mgr=env_mgr,
            env_spec=env_spec,
            validation_command=validation_command,
            stdout=first_validation.stdout,
            stderr=first_validation.stderr,
            baseline=baseline,
            baseline_text=baseline_text,
        )
        if sdk_result is not None:
            return {"code_compat_result": sdk_result}

        max_iters = max(1, int(os.getenv("P2C_CODE_COMPAT_MAX_PATCHES", "2")))
        current_stdout = first_validation.stdout
        current_stderr = first_validation.stderr
        patch_texts: list[str] = []
        for idx in range(1, max_iters + 1):
            patch_text, notes, reason_codes = self._request_patch(
                repo_dir=repo_dir,
                validation_command=validation_command,
                stdout=current_stdout,
                stderr=current_stderr,
                iteration=idx,
            )
            if not patch_text.strip():
                result = CodeCompatResult(
                    status="failed",
                    validation_command=validation_command,
                    validation_passed=False,
                    stdout_tail=current_stdout[-2000:],
                    stderr_tail=current_stderr[-2000:],
                    notes=notes or "LLM did not return a patch.",
                    reason_codes=["CODE_COMPAT_PATCH_EMPTY", *reason_codes],
                )
                self._persist(result)
                return {"code_compat_result": result}
            apply_out = self._apply_patch(repo_dir, patch_text)
            patch_texts.append(patch_text)
            if apply_out.returncode != 0:
                result = CodeCompatResult(
                    status="failed",
                    validation_command=validation_command,
                    validation_passed=False,
                    stdout_tail=apply_out.stdout[-2000:],
                    stderr_tail=apply_out.stderr[-2000:],
                    notes=notes or "Patch application failed.",
                    reason_codes=["CODE_COMPAT_PATCH_APPLY_FAILED", *reason_codes],
                )
                self._write_patch(patch_texts)
                self._persist(result)
                return {"code_compat_result": result}

            validation = env_mgr.run_in_env(validation_command, cwd=str(repo_dir), timeout_sec=120)
            current_stdout = validation.stdout
            current_stderr = validation.stderr
            if validation.returncode == 0:
                patch_rel = self._write_patch(patch_texts)
                changed_files = self._changed_files(repo_dir, baseline)
                result = CodeCompatResult(
                    status="success",
                    changed_files=changed_files,
                    patch_path=patch_rel,
                    validation_command=validation_command,
                    validation_passed=True,
                    stdout_tail=validation.stdout[-2000:],
                    stderr_tail=validation.stderr[-2000:],
                    notes=notes,
                    reason_codes=["CODE_COMPAT_PATCH_APPLIED", "PATCHED_REPRODUCTION", *reason_codes],
                )
                self._persist(result)
                return {"code_compat_result": result}

        patch_rel = self._write_patch(patch_texts)
        result = CodeCompatResult(
            status="failed",
            changed_files=self._changed_files(repo_dir, baseline),
            patch_path=patch_rel,
            validation_command=validation_command,
            validation_passed=False,
            stdout_tail=current_stdout[-2000:],
            stderr_tail=current_stderr[-2000:],
            reason_codes=["CODE_COMPAT_FAILED"],
        )
        self._persist(result)
        return {"code_compat_result": result}

    def _run_claude_sdk_compat(
        self,
        *,
        repo_dir: Path,
        env_mgr: Any,
        env_spec: ExecutorEnvSpec,
        validation_command: str,
        stdout: str,
        stderr: str,
        baseline: dict[str, str],
        baseline_text: dict[str, str],
    ) -> CodeCompatResult | None:
        if not self._should_use_claude_sdk():
            return None

        runtime_spec = ExecutorAgent._build_runtime_spec(env_mgr)
        managed_validation_command = self._managed_validation_command(validation_command, runtime_spec.python_command)
        prompt = self._build_sdk_compat_prompt(
            repo_dir=repo_dir,
            env_spec=env_spec,
            validation_command=validation_command,
            managed_validation_command=managed_validation_command,
            stdout=stdout,
            stderr=stderr,
        )
        self.artifacts.write_text("execution/code_compat/sdk_code_compat_prompt.txt", prompt)
        session = run_claude_code_session(
            prompt=prompt,
            cwd=repo_dir,
            system_prompt=self._sdk_compat_system_prompt(runtime_spec.python_command),
            artifacts=self.artifacts,
            log_prefix="execution/code_compat",
            timeout_sec=max(300, int(os.getenv("P2C_CODE_COMPAT_SDK_TIMEOUT_SEC", "900"))),
            max_turns=max(8, int(os.getenv("P2C_CODE_COMPAT_SDK_MAX_TURNS", "24"))),
            allowed_tools=["Bash", "Read", "Glob", "Grep", "Edit", "MultiEdit", "Write"],
        )
        self.artifacts.write_json(
            "execution/code_compat/sdk_session_result.json",
            {
                "returncode": session.returncode,
                "stdout_tail": session.stdout[-4000:],
                "stderr_tail": session.stderr[-4000:],
                "reason_codes": ["CODE_COMPAT_CLAUDE_SDK_SESSION"],
            },
        )

        validation = env_mgr.run_in_env(validation_command, cwd=str(repo_dir), timeout_sec=120)
        changed_files = self._changed_files(repo_dir, baseline)
        patch_rel = self._write_diff_from_snapshot(repo_dir, baseline_text, changed_files)
        payload = self.artifacts.read_json("execution/code_compat/code_compat_result.json")
        notes = payload.get("notes") if isinstance(payload, dict) else None
        sdk_codes = [
            str(code)
            for code in (payload.get("reason_codes", []) if isinstance(payload, dict) else [])
            if str(code).strip() and str(code) != "INITIALIZED_PLACEHOLDER"
        ]

        if validation.returncode == 0:
            reason_codes = ["CODE_COMPAT_CLAUDE_SDK"]
            if changed_files:
                reason_codes.extend(["CODE_COMPAT_PATCH_APPLIED", "PATCHED_REPRODUCTION"])
            else:
                reason_codes.append("CODE_COMPAT_NO_PATCH_NEEDED")
            result = CodeCompatResult(
                status="success",
                changed_files=changed_files,
                patch_path=patch_rel if changed_files else None,
                validation_command=validation_command,
                validation_passed=True,
                stdout_tail=validation.stdout[-2000:],
                stderr_tail=validation.stderr[-2000:],
                notes=notes,
                reason_codes=self._dedupe_reason_codes([*reason_codes, *sdk_codes]),
            )
            self._persist(result)
            return result

        result = CodeCompatResult(
            status="failed",
            changed_files=changed_files,
            patch_path=patch_rel if changed_files else None,
            validation_command=validation_command,
            validation_passed=False,
            stdout_tail=validation.stdout[-2000:],
            stderr_tail=validation.stderr[-2000:],
            notes=notes or "Claude Code SDK compatibility session did not make import validation pass.",
            reason_codes=self._dedupe_reason_codes([
                "CODE_COMPAT_CLAUDE_SDK",
                "CODE_COMPAT_FAILED",
                *sdk_codes,
            ]),
        )
        self._persist(result)
        return result

    @staticmethod
    def _coerce_env_repair_result(raw: Any) -> EnvRepairResult | None:
        if isinstance(raw, EnvRepairResult):
            return raw
        if isinstance(raw, dict):
            return EnvRepairResult(**raw)
        return None

    @staticmethod
    def _should_use_claude_sdk() -> bool:
        if os.getenv("P2C_DISABLE_CLAUDE_REPAIR_SDK", "0") == "1":
            return False
        return claude_code_sdk_available()

    @staticmethod
    def _dedupe_reason_codes(codes: list[str]) -> list[str]:
        return list(dict.fromkeys(str(code) for code in codes if str(code).strip()))

    def _load_repaired_env_spec(self, ctx: dict[str, Any]) -> ExecutorEnvSpec:
        payload = self.artifacts.read_json("execution/env_repair/repaired_environment_spec.json")
        if payload.get("env_name"):
            return ExecutorEnvSpec(**payload)
        raw = ctx.get("_p2_env_spec") or {}
        return raw if isinstance(raw, ExecutorEnvSpec) else ExecutorEnvSpec(**raw)

    def _build_validation_command(self, repo_dir: Path, env_spec: ExecutorEnvSpec) -> str:
        imports = ToolAgent._derive_key_imports(env_spec)
        for name in self._local_import_candidates(repo_dir):
            if name not in imports:
                imports.append(name)
        script_lines = [
            "import importlib",
            "mods = " + repr(imports[:12]),
            "for mod in mods:",
            "    importlib.import_module(mod)",
            "print('CODE_COMPAT_IMPORT_VALIDATION_OK')",
        ]
        script = "\n".join(script_lines)
        return f"python -c {shlex.quote(script)}"

    @staticmethod
    def _managed_validation_command(validation_command: str, python_command: str) -> str:
        if validation_command.startswith("python "):
            return f"{python_command} {validation_command[len('python '):]}"
        return validation_command

    def _build_sdk_compat_prompt(
        self,
        *,
        repo_dir: Path,
        env_spec: ExecutorEnvSpec,
        validation_command: str,
        managed_validation_command: str,
        stdout: str,
        stderr: str,
    ) -> str:
        result_path = self.artifacts.path("execution/code_compat/code_compat_result.json").resolve()
        patch_path = self.artifacts.path("execution/code_compat/code_compat_patch.diff").resolve()
        return (
            "Patch this repository for source compatibility with the repaired environment.\n"
            f"Repository root: {repo_dir}\n"
            f"Repaired env spec:\n```json\n{json.dumps(env_spec.model_dump(), ensure_ascii=False, indent=2)[:8000]}\n```\n\n"
            "## Import Validation\n"
            f"Host validation command form: `{validation_command}`\n"
            f"Run this exact managed command after every patch attempt: `{managed_validation_command}`\n"
            f"Initial stdout:\n```\n{stdout[-3000:]}\n```\n\n"
            f"Initial stderr:\n```\n{stderr[-6000:]}\n```\n\n"
            "## Task\n"
            "Use Claude Code tools to inspect and minimally edit repository source files until the managed import validation passes. "
            "You own the compatibility repair attempt.\n\n"
            "## Patch Policy\n"
            "1. Only make compatibility edits required for imports to pass under the repaired environment.\n"
            "2. Do not change tests, generated files, lock files, checkpoints, data, or audit artifacts.\n"
            "3. Do not add broad rewrites or behavior changes. Prefer small API compatibility fixes.\n"
            "4. Examples of acceptable edits: removed numpy aliases, sklearn module moves, torch CPU map_location, TensorFlow compat.v1 imports.\n"
            "5. After edits, rerun the managed validation command above.\n\n"
            "## Required Output Files\n"
            f"1. `{result_path}`: JSON matching CodeCompatResult fields.\n"
            f"2. `{patch_path}`: best-effort unified diff of compatibility edits if you can produce one. "
            "The host will also regenerate this diff from its pre-patch snapshot.\n\n"
            "For success, write status=`success`, validation_passed=true, changed_files, notes, "
            "and reason_codes containing `CODE_COMPAT_CLAUDE_SDK`. Include `CODE_COMPAT_PATCH_APPLIED` "
            "and `PATCHED_REPRODUCTION` when source files were changed."
        )

    @staticmethod
    def _sdk_compat_system_prompt(python_command: str) -> str:
        return (
            "You are CodeCompatAgent inside a reproducibility audit pipeline. "
            "You may modify the target repository source code only to make it compatible with the repaired environment. "
            f"All Python validation commands must use this managed Python command: `{python_command}`. "
            "Keep edits minimal and record the patch provenance."
        )

    @staticmethod
    def _local_import_candidates(repo_dir: Path) -> list[str]:
        candidates: list[str] = []
        for child in sorted(repo_dir.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if (child / "__init__.py").is_file() and re.match(r"^[A-Za-z_]\w*$", child.name):
                candidates.append(child.name)
        setup_py = repo_dir / "setup.py"
        if setup_py.is_file():
            text = setup_py.read_text(encoding="utf-8", errors="ignore")[:4000]
            match = re.search(r"name\s*=\s*['\"]([A-Za-z0-9_.-]+)['\"]", text)
            if match:
                candidate = match.group(1).replace("-", "_")
                if re.match(r"^[A-Za-z_]\w*$", candidate):
                    candidates.append(candidate)
        return list(dict.fromkeys(candidates))[:4]

    def _request_patch(
        self,
        *,
        repo_dir: Path,
        validation_command: str,
        stdout: str,
        stderr: str,
        iteration: int,
    ) -> tuple[str, str | None, list[str]]:
        if self.llm is None:
            return "", "LLM client unavailable.", ["CODE_COMPAT_LLM_UNAVAILABLE"]
        schema = {
            "patch": "unified diff patch to apply from repository root",
            "notes": "short explanation",
            "reason_codes": ["CODE_COMPAT_LLM_PATCH"],
        }
        context = self._repo_context(repo_dir)
        user = (
            "The repaired Python environment imports failed. Generate the smallest source compatibility "
            "unified diff needed to make import-only validation pass. Do not change tests, generated files, "
            "or unrelated code.\n\n"
            f"Iteration: {iteration}\n"
            f"Repository root: {repo_dir}\n"
            f"Validation command: {validation_command}\n"
            f"STDOUT:\n{stdout[-3000:]}\n\nSTDERR:\n{stderr[-5000:]}\n\n"
            f"Relevant repository context:\n{context}"
        )
        data, err = self.safe_chat_json(
            schema=schema,
            system=(
                "You patch old ML repositories for dependency/API compatibility. "
                "Return only a minimal unified diff plus notes and reason codes."
            ),
            user=user,
        )
        if err or not data:
            return "", err, ["CODE_COMPAT_LLM_UNAVAILABLE"]
        return str(data.get("patch") or ""), data.get("notes"), [str(code) for code in data.get("reason_codes", [])]

    @staticmethod
    def _repo_context(repo_dir: Path) -> str:
        snippets: list[str] = []
        for path in sorted(repo_dir.rglob("*.py"))[:30]:
            if any(part in {".git", "__pycache__", ".venv", "venv"} for part in path.parts):
                continue
            rel = path.relative_to(repo_dir).as_posix()
            text = path.read_text(encoding="utf-8", errors="ignore")
            snippets.append(f"### {rel}\n{text[:3000]}")
            if sum(len(item) for item in snippets) > 16000:
                break
        return "\n\n".join(snippets)

    @staticmethod
    def _apply_patch(repo_dir: Path, patch_text: str) -> subprocess.CompletedProcess[str]:
        last = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="patch was not attempted")
        for strip in ("-p0", "-p1"):
            last = subprocess.run(
                ["patch", strip, "--forward", "--batch"],
                input=patch_text,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if last.returncode == 0:
                return last
        return last

    def _write_patch(self, patches: list[str]) -> str:
        rel = "execution/code_compat/code_compat_patch.diff"
        self.artifacts.write_text(rel, "\n\n".join(patches).strip() + "\n")
        return rel

    def _persist(self, result: CodeCompatResult) -> None:
        self.artifacts.write_json("execution/code_compat/code_compat_result.json", result.model_dump())

    @staticmethod
    def _capture_repo_state(repo_dir: Path) -> dict[str, str]:
        state: dict[str, str] = {}
        for path in repo_dir.rglob("*"):
            if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            try:
                state[path.relative_to(repo_dir).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
        return state

    @staticmethod
    def _capture_repo_text_snapshot(repo_dir: Path) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        allowed_suffixes = {".cfg", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
        for path in repo_dir.rglob("*"):
            if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            if path.suffix.lower() not in allowed_suffixes:
                continue
            try:
                if path.stat().st_size > 2_000_000:
                    continue
                snapshot[path.relative_to(repo_dir).as_posix()] = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
        return snapshot

    def _write_diff_from_snapshot(
        self,
        repo_dir: Path,
        baseline_text: dict[str, str],
        changed_files: list[str],
    ) -> str | None:
        diff_parts: list[str] = []
        for rel in changed_files:
            before = baseline_text.get(rel)
            if before is None:
                continue
            path = repo_dir / rel
            try:
                after = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
            except OSError:
                continue
            if before == after:
                continue
            diff_parts.extend(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    after.splitlines(keepends=True),
                    fromfile=f"a/{rel}",
                    tofile=f"b/{rel}",
                )
            )
        if not diff_parts:
            return None
        rel_path = "execution/code_compat/code_compat_patch.diff"
        self.artifacts.write_text(rel_path, "".join(diff_parts))
        return rel_path

    @staticmethod
    def _changed_files(repo_dir: Path, baseline: dict[str, str]) -> list[str]:
        current = CodeCompatAgent._capture_repo_state(repo_dir)
        changed = [
            rel for rel, digest in current.items()
            if baseline.get(rel) != digest
        ]
        removed = [rel for rel in baseline if rel not in current]
        return sorted([*changed, *removed])
