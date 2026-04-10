from __future__ import annotations

import ast
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.schemas import DependencyProfile, Entrypoint, RepoAnalysis

try:  # pragma: no cover - Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except Exception:  # pragma: no cover - optional dependency
        tomllib = None


SYSTEM_PROMPT = "You analyze repository structure deterministically and emit strict JSON artifacts."
USER_PROMPT_TEMPLATE = "Input: repo_dir. Output: task/repo_analysis.json"

_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
}

_SCRIPT_PRIORITY = {
    "train": 0.98,
    "start": 0.96,
    "run": 0.94,
    "eval": 0.92,
    "test": 0.80,
    "dev": 0.75,
}

_SOURCE_PRIORITY = {
    "readme_workflow_primary": 600,
    "manifest_explicit": 500,
    "code_cli": 400,
    "notebook_explicit": 350,
    "readme_verified": 300,
    "wrapper_target": 200,
    "unspecified": 0,
}


def _safe_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_excluded(path: Path, root: Path) -> bool:
    rel_parts = path.relative_to(root).parts
    return any(part in _EXCLUDE_DIRS for part in rel_parts)


def _load_toml(path: Path) -> dict[str, Any]:
    if tomllib is None or not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def _dedupe_str(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = str(item or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


class SystemRepoAnalyzer:
    def __init__(self, repo_dir: Path) -> None:
        self.repo_dir = repo_dir.resolve()

    def analyze(self) -> RepoAnalysis:
        profiles = self._build_dependency_profiles()
        profile_map = {p.profile_id: p for p in profiles}
        candidates = self._build_entrypoint_candidates(profile_map)
        self._apply_readme_evidence(candidates, profile_map)
        candidates = sorted(candidates, key=lambda item: self._candidate_sort_key(item, profile_map), reverse=True)
        ecosystems = _dedupe_str([p.ecosystem for p in profiles] + [c.runtime for c in candidates])
        reason_codes: list[str] = []
        if len(ecosystems) > 1:
            reason_codes.append("REPO_ANALYSIS_MULTI_ECOSYSTEM")
        primary = self._pick_primary_candidate(candidates, profile_map)
        if primary is None:
            reason_codes.append("REPO_ANALYSIS_NO_EXECUTABLE_CANDIDATE")
        else:
            reason_codes.append("ENTRYPOINT_SELECTED_PRIMARY")
            if len(candidates) > 1:
                reason_codes.append("ENTRYPOINT_SELECTED_BACKUP")
        return RepoAnalysis(
            ecosystems=ecosystems,
            dependency_profiles=profiles,
            entrypoint_candidates=candidates,
            primary_entrypoint_id=primary.entrypoint_id if primary else None,
            reason_codes=_dedupe_str(reason_codes),
        )

    def _iter_files(self, pattern: str) -> list[Path]:
        files: list[Path] = []
        for path in sorted(self.repo_dir.rglob(pattern)):
            if not path.is_file() or _is_excluded(path, self.repo_dir):
                continue
            files.append(path)
        return files

    def _iter_roots(self, filename: str) -> list[Path]:
        files: list[Path] = []
        for path in sorted(self.repo_dir.rglob(filename)):
            if not path.is_file() or _is_excluded(path, self.repo_dir):
                continue
            files.append(path)
        return files

    def _rel_or_none(self, path: Path) -> str | None:
        try:
            return path.relative_to(self.repo_dir).as_posix()
        except Exception:  # noqa: BLE001
            return None

    def _cwd_rel(self, path: Path) -> str:
        rel = self._rel_or_none(path)
        return rel or "."

    def _resolve_from_virtual_cwd(self, raw_path: str, virtual_cwd: Path) -> Path | None:
        cleaned = str(raw_path or "").strip()
        if not cleaned:
            return None
        candidate = (virtual_cwd / cleaned).resolve()
        if self._rel_or_none(candidate) is None:
            return None
        return candidate

    @staticmethod
    def _shell_tokens(line: str) -> list[str]:
        try:
            return shlex.split(line)
        except ValueError:
            return []

    @staticmethod
    def _strip_shell_comment(line: str) -> str:
        if "#" not in line:
            return line.strip()
        if line.lstrip().startswith("#"):
            return ""
        return re.sub(r"\s+#.*$", "", line).strip()

    @staticmethod
    def _is_readme_workflow_primary(rel_path: str) -> bool:
        rel = str(rel_path or "").strip()
        parent = Path(rel).parent.as_posix()
        name = Path(rel).name.lower()
        return parent in {"", "."} and name.startswith(("run", "start", "launch"))

    def _shell_candidate_command(self, rel_path: str, *, cwd: str, runtime: str) -> str:
        if runtime == "python":
            target = Path(rel_path).name if cwd != "." else rel_path
            return f"python {target}"
        if runtime == "make":
            return "make"
        target = Path(rel_path).name if cwd != "." else rel_path
        return f"bash {target}"

    def _make_entrypoint(
        self,
        *,
        entrypoint_id: str,
        path: str,
        command: str,
        cwd: str,
        runtime: str,
        dependency_profile_id: str | None,
        confidence: float,
        evidence: str,
        reason_codes: list[str] | None = None,
        path_resolution_mode: str | None = None,
        derived_from_wrapper: str | None = None,
    ) -> Entrypoint:
        return Entrypoint(
            entrypoint_id=entrypoint_id,
            path=path,
            command=command,
            cwd=cwd,
            runtime=runtime,
            dependency_profile_id=dependency_profile_id,
            confidence=confidence,
            evidence=evidence,
            reason_codes=_dedupe_str(reason_codes or []),
            path_resolution_mode=path_resolution_mode,
            derived_from_wrapper=derived_from_wrapper,
        )

    def _derive_from_shell_wrapper(
        self,
        script_rel: str,
        *,
        invoke_cwd: str,
        profile_map: dict[str, DependencyProfile],
        root_wrapper: str | None = None,
        seen: set[tuple[str, str]] | None = None,
    ) -> list[Entrypoint]:
        script_path = self.repo_dir / script_rel
        if not script_path.is_file():
            return []

        if seen is None:
            seen = set()
        key = (script_rel, invoke_cwd)
        if key in seen:
            return []
        seen.add(key)

        try:
            text = script_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            return []

        wrapper_id = root_wrapper or script_rel
        virtual_cwd = (self.repo_dir / invoke_cwd).resolve()
        derived: list[Entrypoint] = []

        for raw_line in text.splitlines():
            line = self._strip_shell_comment(raw_line)
            if not line or line.startswith(("if ", "then", "fi", "for ", "while ", "case ", "do ", "done", "else")):
                continue
            if "$" in line:
                continue
            tokens = self._shell_tokens(line)
            if not tokens:
                continue

            while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
                name, _, value = tokens[0].partition("=")
                if name.isidentifier() and value:
                    tokens = tokens[1:]
                    continue
                break
            if not tokens:
                continue

            head = tokens[0]
            if head == "cd" and len(tokens) >= 2:
                next_cwd = self._resolve_from_virtual_cwd(tokens[1], virtual_cwd)
                if next_cwd is not None and next_cwd.is_dir():
                    virtual_cwd = next_cwd
                continue

            current_cwd = self._cwd_rel(virtual_cwd)
            profile = self._match_profile(virtual_cwd, "python", profile_map)
            reason_codes = ["ENTRYPOINT_DERIVED_FROM_WRAPPER"]
            if current_cwd != ".":
                reason_codes.append("ENTRYPOINT_CWD_FROM_WRAPPER")

            if head == "make":
                target_cwd = current_cwd
                make_tokens = list(tokens)
                if "-C" in make_tokens:
                    idx = make_tokens.index("-C")
                    if idx + 1 < len(make_tokens):
                        resolved_dir = self._resolve_from_virtual_cwd(make_tokens[idx + 1], virtual_cwd)
                        if resolved_dir is not None:
                            target_cwd = self._cwd_rel(resolved_dir)
                        del make_tokens[idx:idx + 2]
                makefile_rel = f"{target_cwd}/Makefile" if target_cwd != "." else "Makefile"
                if (self.repo_dir / makefile_rel).exists():
                    derived.append(
                        self._make_entrypoint(
                            entrypoint_id=f"shell-derived:{wrapper_id}:{makefile_rel}@{target_cwd}",
                            path=makefile_rel,
                            command=shlex.join(make_tokens),
                            cwd=target_cwd,
                            runtime="make",
                            dependency_profile_id=profile.profile_id if profile else None,
                            confidence=0.78,
                            evidence=f"derived from shell wrapper `{script_rel}`",
                            reason_codes=reason_codes,
                            path_resolution_mode="wrapper_virtual_cwd",
                            derived_from_wrapper=wrapper_id,
                        )
                    )
                continue

            runtime = None
            raw_target = None
            command_tokens = list(tokens)
            if head in {"python", "python3"} and len(tokens) >= 2 and tokens[1].endswith(".py"):
                runtime = "python"
                raw_target = tokens[1]
            elif head in {"bash", "sh"} and len(tokens) >= 2 and tokens[1].endswith(".sh"):
                runtime = "shell"
                raw_target = tokens[1]
                command_tokens[0] = "bash"
            elif (head.startswith("./") or head.startswith("../")) and head.endswith(".sh"):
                runtime = "shell"
                raw_target = head
                command_tokens = ["bash", *tokens]

            if runtime is None or raw_target is None:
                continue

            resolved_target = self._resolve_from_virtual_cwd(raw_target, virtual_cwd)
            if resolved_target is None:
                continue
            rel_target = self._rel_or_none(resolved_target)
            if not rel_target:
                continue

            target_profile = self._match_profile(resolved_target, "python" if runtime == "python" else "make", profile_map)
            derived.append(
                self._make_entrypoint(
                    entrypoint_id=f"shell-derived:{wrapper_id}:{rel_target}@{current_cwd}",
                    path=rel_target,
                    command=shlex.join(command_tokens),
                    cwd=current_cwd,
                    runtime=runtime,
                    dependency_profile_id=target_profile.profile_id if target_profile else None,
                    confidence=0.86 if runtime == "shell" else 0.84,
                    evidence=f"derived from shell wrapper `{script_rel}`",
                    reason_codes=reason_codes,
                    path_resolution_mode="wrapper_virtual_cwd",
                    derived_from_wrapper=wrapper_id,
                )
            )
            if runtime == "shell":
                derived.extend(
                    self._derive_from_shell_wrapper(
                        rel_target,
                        invoke_cwd=current_cwd,
                        profile_map=profile_map,
                        root_wrapper=wrapper_id,
                        seen=seen,
                    )
                )

        return derived

    def _build_dependency_profiles(self) -> list[DependencyProfile]:
        profiles: list[DependencyProfile] = []
        seen: set[str] = set()

        for pyproject in self._iter_roots("pyproject.toml"):
            rel = _safe_rel(pyproject, self.repo_dir)
            cwd = _safe_rel(pyproject.parent, self.repo_dir) or "."
            poetry_lock = pyproject.parent / "poetry.lock"
            data = _load_toml(pyproject)
            if poetry_lock.exists():
                profile = DependencyProfile(
                    profile_id=f"python-poetry:{cwd}",
                    ecosystem="python",
                    manager="poetry",
                    cwd=cwd,
                    manifest_paths=_dedupe_str([rel, _safe_rel(poetry_lock, self.repo_dir)]),
                    install_command="poetry install",
                )
                profiles.append(profile)
                seen.add(profile.profile_id)
            elif isinstance(data.get("build-system"), dict) or isinstance(data.get("project"), dict):
                profile = DependencyProfile(
                    profile_id=f"python-pyproject:{cwd}",
                    ecosystem="python",
                    manager="pip_editable",
                    cwd=cwd,
                    manifest_paths=[rel],
                    install_command="python -m pip install -e .",
                )
                if profile.profile_id not in seen:
                    profiles.append(profile)
                    seen.add(profile.profile_id)

        for req in sorted(self.repo_dir.rglob("requirements*.txt")):
            if not req.is_file() or _is_excluded(req, self.repo_dir):
                continue
            cwd = _safe_rel(req.parent, self.repo_dir) or "."
            profile = DependencyProfile(
                profile_id=f"python-requirements:{_safe_rel(req, self.repo_dir)}",
                ecosystem="python",
                manager="pip_requirements",
                cwd=cwd,
                manifest_paths=[_safe_rel(req, self.repo_dir)],
                install_command=f"python -m pip install -r {req.name}",
            )
            if profile.profile_id not in seen:
                profiles.append(profile)
                seen.add(profile.profile_id)

        for setup_name in ("setup.py", "setup.cfg"):
            for setup_file in self._iter_roots(setup_name):
                cwd = _safe_rel(setup_file.parent, self.repo_dir) or "."
                profile = DependencyProfile(
                    profile_id=f"python-setuptools:{cwd}",
                    ecosystem="python",
                    manager="pip_editable",
                    cwd=cwd,
                    manifest_paths=[_safe_rel(setup_file, self.repo_dir)],
                    install_command="python -m pip install -e .",
                )
                if profile.profile_id not in seen:
                    profiles.append(profile)
                    seen.add(profile.profile_id)

        for package_json in self._iter_roots("package.json"):
            cwd = _safe_rel(package_json.parent, self.repo_dir) or "."
            manifest_paths = [_safe_rel(package_json, self.repo_dir)]
            manager = "npm"
            for lock_name, manager_name in (
                ("pnpm-lock.yaml", "pnpm"),
                ("yarn.lock", "yarn"),
                ("package-lock.json", "npm"),
            ):
                lock_path = package_json.parent / lock_name
                if lock_path.exists():
                    manifest_paths.append(_safe_rel(lock_path, self.repo_dir))
                    manager = manager_name
                    break
            install_map = {
                "npm": "npm ci",
                "pnpm": "pnpm install --frozen-lockfile",
                "yarn": "yarn install --frozen-lockfile",
            }
            profile = DependencyProfile(
                profile_id=f"node-{manager}:{cwd}",
                ecosystem="node",
                manager=manager,
                cwd=cwd,
                manifest_paths=_dedupe_str(manifest_paths),
                install_command=install_map.get(manager),
            )
            if profile.profile_id not in seen:
                profiles.append(profile)
                seen.add(profile.profile_id)

        for env_file in self._iter_roots("environment.yml"):
            cwd = _safe_rel(env_file.parent, self.repo_dir) or "."
            profile = DependencyProfile(
                profile_id=f"conda:{cwd}",
                ecosystem="conda",
                manager="conda",
                cwd=cwd,
                manifest_paths=[_safe_rel(env_file, self.repo_dir)],
                install_command=None,
                auto_bootstrap_supported=False,
                reason_codes=["DEPENDENCY_PROFILE_UNSUPPORTED"],
            )
            if profile.profile_id not in seen:
                profiles.append(profile)
                seen.add(profile.profile_id)

        for dockerfile in self._iter_roots("Dockerfile"):
            cwd = _safe_rel(dockerfile.parent, self.repo_dir) or "."
            profile = DependencyProfile(
                profile_id=f"docker:{cwd}",
                ecosystem="docker",
                manager="docker",
                cwd=cwd,
                manifest_paths=[_safe_rel(dockerfile, self.repo_dir)],
                install_command=None,
                auto_bootstrap_supported=False,
                reason_codes=["DEPENDENCY_PROFILE_UNSUPPORTED"],
            )
            if profile.profile_id not in seen:
                profiles.append(profile)
                seen.add(profile.profile_id)

        for makefile in self._iter_roots("Makefile"):
            cwd = _safe_rel(makefile.parent, self.repo_dir) or "."
            profile = DependencyProfile(
                profile_id=f"make:{cwd}",
                ecosystem="make",
                manager="make",
                cwd=cwd,
                manifest_paths=[_safe_rel(makefile, self.repo_dir)],
                install_command=None,
                auto_bootstrap_supported=False,
                reason_codes=["DEPENDENCY_PROFILE_UNSUPPORTED"],
            )
            if profile.profile_id not in seen:
                profiles.append(profile)
                seen.add(profile.profile_id)

        return profiles

    def _match_profile(self, path: Path, ecosystem: str, profile_map: dict[str, DependencyProfile]) -> DependencyProfile | None:
        rel_dir = self._rel_or_none(path.parent) or "."
        best: DependencyProfile | None = None
        best_len = -1
        for profile in profile_map.values():
            if profile.ecosystem != ecosystem:
                continue
            prefix = profile.cwd or "."
            if prefix == ".":
                if best is None and best_len < 0:
                    best = profile
                    best_len = 0
                continue
            if rel_dir == prefix or rel_dir.startswith(prefix + "/"):
                plen = len(prefix.split("/"))
                if plen > best_len:
                    best = profile
                    best_len = plen
        return best

    def _parse_python_file(self, path: Path) -> dict[str, Any]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            return {}
        try:
            tree = ast.parse(text)
        except Exception:  # noqa: BLE001
            return {}
        imports: set[str] = set()
        has_main_guard = False
        has_main_fn = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split(".")[0])
            elif isinstance(node, ast.FunctionDef) and node.name == "main":
                has_main_fn = True
            elif isinstance(node, ast.If):
                try:
                    test_text = ast.unparse(node.test)
                except Exception:  # noqa: BLE001
                    test_text = ""
                if "__name__" in test_text and "__main__" in test_text:
                    has_main_guard = True
        return {
            "imports": imports,
            "has_main_guard": has_main_guard,
            "has_main_fn": has_main_fn,
        }

    def _build_python_candidates(self, profile_map: dict[str, DependencyProfile]) -> list[Entrypoint]:
        candidates: list[Entrypoint] = []
        seen: set[tuple[str, str]] = set()

        for pyproject in self._iter_roots("pyproject.toml"):
            data = _load_toml(pyproject)
            cwd = _safe_rel(pyproject.parent, self.repo_dir) or "."
            for section in (data.get("project", {}), data.get("tool", {}).get("poetry", {})):
                scripts = {}
                if isinstance(section, dict):
                    raw_scripts = section.get("scripts")
                    if isinstance(raw_scripts, dict):
                        scripts = raw_scripts
                for script_name in scripts.keys():
                    key = (cwd, f"python::{script_name}")
                    if key in seen:
                        continue
                    seen.add(key)
                    profile = self._match_profile(pyproject, "python", profile_map)
                    candidates.append(
                        Entrypoint(
                            entrypoint_id=f"python-script:{cwd}:{script_name}",
                            path=script_name,
                            command=str(script_name),
                            cwd=cwd,
                            runtime="python",
                            dependency_profile_id=profile.profile_id if profile else None,
                            confidence=0.99,
                            evidence=f"pyproject console script `{script_name}`",
                        )
                    )

        for path in self._iter_files("*.py"):
            rel = _safe_rel(path, self.repo_dir)
            if rel.startswith("tests/") or rel.endswith("_test.py") or "/tests/" in rel:
                continue
            parsed = self._parse_python_file(path)
            imports = parsed.get("imports", set())
            has_main_guard = bool(parsed.get("has_main_guard"))
            has_main_fn = bool(parsed.get("has_main_fn"))
            cli_import = any(x in imports for x in {"argparse", "click", "typer", "fire"})
            if not (has_main_guard or cli_import or has_main_fn):
                continue
            confidence = 0.72
            if has_main_guard:
                confidence = 0.93
            elif cli_import and has_main_fn:
                confidence = 0.90
            elif cli_import or has_main_fn:
                confidence = 0.86
            profile = self._match_profile(path, "python", profile_map)
            reason_codes: list[str] = []
            cwd = profile.cwd if profile else "."
            if cwd != ".":
                reason_codes.append("ENTRYPOINT_CWD_INFERRED")
            candidates.append(
                Entrypoint(
                    entrypoint_id=f"python-file:{rel}",
                    path=rel,
                    command=f"python {rel}",
                    cwd=cwd,
                    runtime="python",
                    dependency_profile_id=profile.profile_id if profile else None,
                    confidence=confidence,
                    evidence="python CLI entrypoint detected",
                    reason_codes=reason_codes,
                )
            )
        return candidates

    def _build_notebook_candidates(self, profile_map: dict[str, DependencyProfile]) -> list[Entrypoint]:
        candidates: list[Entrypoint] = []
        for path in self._iter_files("*.ipynb"):
            rel = _safe_rel(path, self.repo_dir)
            if rel.startswith("tests/") or "/tests/" in rel or ".ipynb_checkpoints/" in rel:
                continue

            rel_dir = _safe_rel(path.parent, self.repo_dir) or "."
            profile = self._match_profile(path, "python", profile_map)
            cwd = rel_dir
            if cwd == ".":
                notebook_ref = rel
            else:
                notebook_ref = path.name

            output_name = f"{path.stem}.executed.ipynb"
            command = (
                "python -m jupyter nbconvert "
                f"--to notebook --execute {shlex.quote(notebook_ref)} "
                f"--output {shlex.quote(output_name)}"
            )
            confidence = 0.83 if "code/" in rel or rel_dir == "code" else 0.78
            reason_codes: list[str] = []
            if cwd != ".":
                reason_codes.append("ENTRYPOINT_CWD_INFERRED")

            candidates.append(
                Entrypoint(
                    entrypoint_id=f"notebook:{rel}",
                    path=rel,
                    command=command,
                    cwd=cwd,
                    runtime="python",
                    dependency_profile_id=profile.profile_id if profile else None,
                    confidence=confidence,
                    evidence="jupyter notebook entrypoint detected",
                    reason_codes=reason_codes,
                )
            )
        return candidates

    def _build_node_candidates(self, profile_map: dict[str, DependencyProfile]) -> list[Entrypoint]:
        candidates: list[Entrypoint] = []
        for package_json in self._iter_roots("package.json"):
            rel = _safe_rel(package_json, self.repo_dir)
            cwd = _safe_rel(package_json.parent, self.repo_dir) or "."
            try:
                payload = json.loads(package_json.read_text(encoding="utf-8", errors="ignore"))
            except Exception:  # noqa: BLE001
                payload = {}
            profile = self._match_profile(package_json, "node", profile_map)
            manager = profile.manager if profile else "npm"
            scripts = payload.get("scripts")
            if isinstance(scripts, dict):
                for name in scripts.keys():
                    base = _SCRIPT_PRIORITY.get(str(name), 0.60)
                    if manager == "yarn":
                        cmd = f"yarn {name}"
                    else:
                        cmd = f"{manager} run {name}"
                    candidates.append(
                        Entrypoint(
                            entrypoint_id=f"node-script:{cwd}:{name}",
                            path=rel,
                            command=cmd,
                            cwd=cwd,
                            runtime="node",
                            dependency_profile_id=profile.profile_id if profile else None,
                            confidence=base,
                            evidence=f"package.json script `{name}`",
                            reason_codes=["ENTRYPOINT_CWD_INFERRED"] if cwd != "." else [],
                        )
                    )
            main_path = payload.get("main")
            if isinstance(main_path, str) and main_path.strip():
                main_file = (package_json.parent / main_path).resolve()
                try:
                    rel_main = _safe_rel(main_file, self.repo_dir)
                except Exception:  # noqa: BLE001
                    rel_main = ""
                if rel_main:
                    candidates.append(
                        Entrypoint(
                            entrypoint_id=f"node-main:{cwd}:{rel_main}",
                            path=rel_main,
                            command=f"node {rel_main}",
                            cwd=cwd,
                            runtime="node",
                            dependency_profile_id=profile.profile_id if profile else None,
                            confidence=0.88,
                            evidence="package.json main",
                            reason_codes=["ENTRYPOINT_CWD_INFERRED"] if cwd != "." else [],
                        )
                    )
        return candidates

    def _build_shell_candidates(self, profile_map: dict[str, DependencyProfile]) -> list[Entrypoint]:
        candidates: list[Entrypoint] = []
        for path in self._iter_files("*.sh"):
            rel = _safe_rel(path, self.repo_dir)
            filename = path.name.lower()
            if not (
                re.match(r"(run|train|eval|start|test|launch).*\.sh$", filename)
                or rel.count("/") == 0
            ):
                continue
            profile = self._match_profile(path, "make", profile_map)
            candidates.append(
                self._make_entrypoint(
                    entrypoint_id=f"shell:{rel}",
                    path=rel,
                    command=f"bash {rel}",
                    cwd=".",
                    runtime="shell",
                    dependency_profile_id=profile.profile_id if profile else None,
                    confidence=0.58,
                    evidence="shell script candidate",
                    path_resolution_mode="repo_root",
                )
            )
            candidates.extend(
                self._derive_from_shell_wrapper(
                    rel,
                    invoke_cwd=".",
                    profile_map=profile_map,
                )
            )
        return candidates

    def _build_make_candidates(self, profile_map: dict[str, DependencyProfile]) -> list[Entrypoint]:
        candidates: list[Entrypoint] = []
        for makefile in self._iter_roots("Makefile"):
            cwd = _safe_rel(makefile.parent, self.repo_dir) or "."
            profile = self._match_profile(makefile, "make", profile_map)
            try:
                text = makefile.read_text(encoding="utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                text = ""
            for line in text.splitlines():
                m = re.match(r"^([A-Za-z0-9_.-]+):(?:\s|$)", line)
                if not m:
                    continue
                target = m.group(1)
                if target.startswith(".") or "%" in target or target not in _SCRIPT_PRIORITY:
                    continue
                candidates.append(
                    Entrypoint(
                        entrypoint_id=f"make:{cwd}:{target}",
                        path="Makefile",
                        command=f"make {target}",
                        cwd=cwd,
                        runtime="make",
                        dependency_profile_id=profile.profile_id if profile else None,
                        confidence=0.57 + (_SCRIPT_PRIORITY[target] / 10.0),
                        evidence=f"Makefile target `{target}`",
                        reason_codes=["ENTRYPOINT_CWD_INFERRED"] if cwd != "." else [],
                    )
                )
        return candidates

    def _build_entrypoint_candidates(self, profile_map: dict[str, DependencyProfile]) -> list[Entrypoint]:
        merged = (
            self._build_python_candidates(profile_map)
            + self._build_notebook_candidates(profile_map)
            + self._build_node_candidates(profile_map)
            + self._build_shell_candidates(profile_map)
            + self._build_make_candidates(profile_map)
        )
        by_id: dict[str, Entrypoint] = {}
        for item in merged:
            entrypoint_id = str(item.entrypoint_id or item.path)
            if entrypoint_id not in by_id or item.confidence > by_id[entrypoint_id].confidence:
                by_id[entrypoint_id] = item
        ordered = sorted(by_id.values(), key=lambda item: self._candidate_sort_key(item, profile_map), reverse=True)
        return ordered

    def _readme_commands(self) -> list[str]:
        readmes = [self.repo_dir / "README.md", self.repo_dir / "readme.md"]
        commands: list[str] = []
        line_patterns = [
            r"^\s*(python(?:3)?\s+[^\n`]+)$",
            r"^\s*(node\s+[^\n`]+)$",
            r"^\s*((?:npm|pnpm)\s+run\s+[^\n`]+)$",
            r"^\s*(yarn\s+[^\n`]+)$",
            r"^\s*(bash\s+[^\n`]+\.sh(?:\s+[^\n`]+)*)$",
            r"^\s*(sh\s+[^\n`]+\.sh(?:\s+[^\n`]+)*)$",
            r"^\s*((?:\./|\.\./)[^\s`]+\.sh(?:\s+[^\n`]+)*)$",
            r"^\s*(make\s+[^\n`]+)$",
        ]
        fragment_patterns = [
            r"(?:^|&&\s*|;\s*|time\s+)((?:\./|\.\./)[^\s`]+\.sh(?:\s+[^\n`]+)*)",
            r"(?:^|&&\s*|;\s*|time\s+)(bash\s+[^\n`]+\.sh(?:\s+[^\n`]+)*)",
            r"(?:^|&&\s*|;\s*|time\s+)(sh\s+[^\n`]+\.sh(?:\s+[^\n`]+)*)",
            r"(?:^|&&\s*|;\s*|time\s+)(python(?:3)?\s+[^\n`]+)",
            r"(?:^|&&\s*|;\s*|time\s+)(make\s+[^\n`]+)",
        ]
        for readme in readmes:
            if not readme.exists():
                continue
            text = readme.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                stripped = line.strip()
                for pattern in line_patterns:
                    m = re.match(pattern, stripped)
                    if m:
                        commands.append(m.group(1).strip())
                for pattern in fragment_patterns:
                    m = re.search(pattern, stripped)
                    if m:
                        commands.append(m.group(1).strip())
        return _dedupe_str(commands)

    def _readme_shell_candidate(
        self,
        command: str,
        profile_map: dict[str, DependencyProfile],
    ) -> Entrypoint | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        if not tokens:
            return None
        runtime = "shell"
        command_tokens = list(tokens)
        if tokens[0] in {"bash", "sh"}:
            if len(tokens) < 2 or not tokens[1].endswith(".sh"):
                return None
            raw_target = tokens[1]
            command_tokens[0] = "bash"
        elif tokens[0].endswith(".sh") or tokens[0].startswith(("./", "../")):
            raw_target = tokens[0]
            command_tokens = ["bash", *tokens]
        else:
            return None
        target = self._resolve_from_virtual_cwd(raw_target, self.repo_dir)
        if target is None or not target.exists():
            return None
        rel = self._rel_or_none(target)
        if not rel:
            return None
        profile = self._match_profile(target, "make", profile_map)
        reason_codes: list[str] = []
        if self._is_readme_workflow_primary(rel):
            reason_codes.append("README_WORKFLOW_PRIMARY")
        return self._make_entrypoint(
            entrypoint_id=f"readme-shell:{rel}",
            path=rel,
            command=shlex.join(command_tokens),
            cwd=".",
            runtime=runtime,
            dependency_profile_id=profile.profile_id if profile else None,
            confidence=0.88 if reason_codes else 0.72,
            evidence="README verified shell command",
            reason_codes=reason_codes,
            path_resolution_mode="repo_root",
        )

    def _apply_readme_evidence(
        self,
        candidates: list[Entrypoint],
        profile_map: dict[str, DependencyProfile],
    ) -> None:
        commands = self._readme_commands()
        for command in commands:
            readme_shell = self._readme_shell_candidate(command, profile_map)
            if readme_shell is not None:
                matched = False
                for idx, candidate in enumerate(candidates):
                    if candidate.path == readme_shell.path and candidate.runtime == readme_shell.runtime:
                        merged_reason_codes = list(candidate.reason_codes) + list(readme_shell.reason_codes)
                        candidates[idx] = candidate.model_copy(
                            update={
                                "command": readme_shell.command,
                                "confidence": min(1.0, max(candidate.confidence, readme_shell.confidence) + 0.04),
                                "evidence": f"{candidate.evidence}; README verified",
                                "reason_codes": _dedupe_str(merged_reason_codes),
                                "path_resolution_mode": readme_shell.path_resolution_mode,
                            }
                        )
                        matched = True
                        break
                if not matched:
                    candidates.append(readme_shell)
                continue

            matched = False
            for idx, candidate in enumerate(candidates):
                if candidate.command == command:
                    merged_reason_codes = list(candidate.reason_codes)
                    candidates[idx] = candidate.model_copy(
                        update={
                            "confidence": min(1.0, candidate.confidence + 0.03),
                            "evidence": f"{candidate.evidence}; README verified",
                            "reason_codes": _dedupe_str(merged_reason_codes),
                        }
                    )
                    matched = True
                    break
            if matched:
                continue
            if command.startswith("python"):
                m = re.match(r"python(?:3)?\s+([^\s]+\.py)", command)
                if not m:
                    continue
                rel = m.group(1)
                path = self.repo_dir / rel
                if not path.exists():
                    continue
                profile = self._match_profile(path, "python", profile_map)
                candidates.append(
                    self._make_entrypoint(
                        entrypoint_id=f"readme-python:{rel}",
                        path=rel,
                        command=f"python {rel}",
                        cwd=profile.cwd if profile else ".",
                        runtime="python",
                        dependency_profile_id=profile.profile_id if profile else None,
                        confidence=0.68,
                        evidence="README verified python command",
                        reason_codes=["ENTRYPOINT_CWD_INFERRED"] if profile and profile.cwd != "." else [],
                        path_resolution_mode="repo_root",
                    )
                )

    def _candidate_sort_key(self, candidate: Entrypoint, profile_map: dict[str, DependencyProfile]) -> tuple[int, float, int, int, str]:
        profile = profile_map.get(str(candidate.dependency_profile_id or ""))
        source_kind = "unspecified"
        evidence = str(candidate.evidence or "").lower()
        path_hint = f"{candidate.path} {candidate.command}".lower()
        if "README_WORKFLOW_PRIMARY" in candidate.reason_codes:
            source_kind = "readme_workflow_primary"
        elif "console script" in evidence or "package.json script" in evidence or "main" in evidence:
            source_kind = "manifest_explicit"
        elif "notebook" in evidence:
            source_kind = "notebook_explicit"
        elif "readme verified" in evidence:
            source_kind = "readme_verified"
        elif "derived from shell wrapper" in evidence or "makefile target" in evidence or "shell script" in evidence:
            source_kind = "wrapper_target"
        elif "cli" in evidence:
            source_kind = "code_cli"
        source_score = _SOURCE_PRIORITY[source_kind]
        semantic_bonus = 0.0
        if any(token in path_hint for token in ("train", "fit", "trainer")):
            semantic_bonus += 75.0
        if any(token in path_hint for token in ("threshold", "tune", "predict", "app", "streamlit", "demo", "explain")):
            semantic_bonus -= 40.0
        bootstrap_score = 1 if profile is None or profile.auto_bootstrap_supported else 0
        profile_score = 1 if candidate.dependency_profile_id else 0
        return (
            bootstrap_score,
            float(source_score) + float(candidate.confidence) + semantic_bonus,
            profile_score,
            1 if candidate.cwd == "." else 0,
            str(candidate.entrypoint_id or candidate.path),
        )

    def _pick_primary_candidate(
        self,
        candidates: list[Entrypoint],
        profile_map: dict[str, DependencyProfile],
    ) -> Entrypoint | None:
        for candidate in candidates:
            profile = profile_map.get(str(candidate.dependency_profile_id or ""))
            if profile is None or profile.auto_bootstrap_supported:
                return candidate
        return candidates[0] if candidates else None


class RepoAnalysisAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="repo_analysis", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)
        repo_dir = Path(ctx["repo_dir"])
        analysis = SystemRepoAnalyzer(repo_dir).analyze()
        self.artifacts.write_json("task/repo_analysis.json", analysis.model_dump())
        return {"repo_analysis": analysis.model_dump()}
