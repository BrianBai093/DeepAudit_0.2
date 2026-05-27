"""EnvRepairAgent — bounded repair path for failed native conda environments."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import subprocess
from typing import Any

try:  # Optional; keep a text fallback so PyYAML is not a runtime requirement.
    import yaml  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.tool_agent import ToolAgent
from p2c.runtime.conda_env import CondaEnvManager
from p2c.schemas import (
    CondaDependency,
    EnvRepairResult,
    EnvSetupResult,
    ExecutorEnvSpec,
    RepoAnalysis,
)


_TORCH_FAMILY = {"pytorch", "torch", "torchvision", "torchaudio"}
_CUDA_PACKAGES = {"cudatoolkit", "cuda", "cuda-toolkit", "pytorch-cuda"}
_SKIP_PACKAGES = {"python", "pip", "setuptools", "wheel"}


@dataclass
class _RepairCandidate:
    name: str
    python_version: str
    conda_dependencies: list[CondaDependency] = field(default_factory=list)
    pip_dependencies: list[str] = field(default_factory=list)
    pre_install_commands: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)


class EnvRepairAgent(BaseAgent):
    """Repair native conda env failures by converting them to bounded installs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="env_repair_agent", *args, **kwargs)
        self._env_mgr: CondaEnvManager | None = None

    @property
    def env_manager(self) -> CondaEnvManager | None:
        return self._env_mgr

    def cleanup(self) -> None:
        if self._env_mgr:
            self.log("PROGRESS", "cleaning up repaired environment...")
            self._env_mgr.cleanup()

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        repo_dir = Path(str(ctx["repo_dir"])).resolve()
        env_spec_raw = ctx.get("_p2_env_spec") or {}
        env_spec = env_spec_raw if isinstance(env_spec_raw, ExecutorEnvSpec) else ExecutorEnvSpec(**env_spec_raw)
        native_env = self._resolve_native_environment_file(env_spec, repo_dir)
        if native_env is None:
            repo_analysis = self._load_repo_analysis()
            native_env = ToolAgent._find_native_environment_file(repo_dir, repo_analysis)
        if native_env is None:
            result = EnvRepairResult(
                status="failed",
                env_name=env_spec.env_name,
                python_version=env_spec.python_version,
                reason_codes=["ENV_REPAIR_NO_NATIVE_ENV_FILE"],
            )
            self._persist(result)
            return {"env_repair_result": result}

        original_failure = ctx.get("_p2_env_failure")
        if isinstance(original_failure, EnvSetupResult):
            failure_codes = original_failure.reason_codes
            failure_log = "\n".join(original_failure.conda_install_log + original_failure.pip_install_log)
        elif isinstance(original_failure, dict):
            failure_codes = [str(code) for code in original_failure.get("reason_codes", [])]
            failure_log = json.dumps(original_failure, ensure_ascii=False)[:4000]
        else:
            failure_codes = ["ENV_REPAIR_FORCE_MODE"] if ctx.get("phase2_force_env_repair") else []
            failure_log = ""

        parsed = self._parse_native_environment(native_env)
        self.artifacts.write_json(
            "execution/env_repair/native_env_diagnosis.json",
            {
                "native_environment_file": str(native_env),
                "failure_codes": failure_codes,
                "failure_class": self._classify_failure(failure_log, failure_codes),
                "reason_codes": ["ENV_REPAIR_NATIVE_ENV_DIAGNOSIS"],
            },
        )
        guidance = self._request_repair_guidance(
            native_env=native_env,
            parsed=parsed,
            failure_codes=failure_codes,
            failure_log=failure_log,
        )
        if guidance is not None:
            self.artifacts.write_json("execution/env_repair/llm_repair_guidance.json", guidance)

        candidates = self._build_candidates(
            env_spec=env_spec,
            parsed=parsed,
            original_python=parsed.get("python_version") or env_spec.python_version or "3.10",
        )
        result = EnvRepairResult(
            status="failed",
            env_name=env_spec.env_name,
            python_version=env_spec.python_version,
            reason_codes=["ENV_REPAIR_ATTEMPTED"],
        )
        for candidate in candidates:
            candidate_out = self._try_candidate(repo_dir, env_spec, candidate, native_env)
            if candidate_out["ok"]:
                result = candidate_out["result"]
                self._persist(result)
                return {"env_repair_result": result, "env_manager": self._env_mgr}
            result.failed_candidates.append(candidate_out["candidate"])

        result.reason_codes.append("ENV_REPAIR_FAILED")
        self._persist(result)
        return {"env_repair_result": result}

    def _try_candidate(
        self,
        repo_dir: Path,
        env_spec: ExecutorEnvSpec,
        candidate: _RepairCandidate,
        native_env: Path,
    ) -> dict[str, Any]:
        command_log: list[str] = []
        failed_record: dict[str, Any] = {
            "strategy": candidate.name,
            "python_version": candidate.python_version,
            "reason_codes": list(candidate.reason_codes),
        }
        self.log("PROGRESS", f"env repair candidate={candidate.name}, python={candidate.python_version}")
        manager = CondaEnvManager(env_name=env_spec.env_name, python_version=candidate.python_version)
        manager.cleanup()
        self._env_mgr = manager
        try:
            create_out = manager.create()
            command_log.append(f"create python={candidate.python_version} ok={create_out['ok']}")
            if not create_out["ok"]:
                failed_record["log_tail"] = str(create_out.get("log", ""))[-2000:]
                self._record_candidate(candidate, "failed", command_log, failed_record)
                manager.cleanup()
                return {"ok": False, "candidate": failed_record}

            repair_spec = ExecutorEnvSpec(
                env_name=env_spec.env_name,
                python_version=candidate.python_version,
                native_environment_file=None,
                conda_dependencies=candidate.conda_dependencies,
                pip_dependencies=candidate.pip_dependencies,
                pre_install_commands=candidate.pre_install_commands,
                reason_codes=["ENV_REPAIR_DERIVED_SPEC", *candidate.reason_codes],
            )
            repaired_rel = "execution/env_repair/repaired_environment_spec.json"
            self.artifacts.write_json(repaired_rel, repair_spec.model_dump())
            self.artifacts.write_text(
                "execution/env_repair/repaired_environment.yml",
                self._render_repaired_environment_yml(repair_spec, native_env),
            )

            layers = ToolAgent._build_layers(repair_spec)
            if layers:
                for layer_result in manager.install_layered(layers):
                    command_log.append(
                        f"layer={layer_result.layer_name} ok={layer_result.ok} elapsed={layer_result.elapsed_sec:.1f}s"
                    )
                    if not layer_result.ok and any(layer.name == layer_result.layer_name and layer.is_critical for layer in layers):
                        failed_record["log_tail"] = layer_result.log[-2000:]
                        failed_record["failed_packages"] = layer_result.failed_packages
                        self._record_candidate(candidate, "failed", command_log, failed_record)
                        manager.cleanup()
                        return {"ok": False, "candidate": failed_record}

            for cmd in candidate.pre_install_commands:
                proc = manager.run_in_env(cmd, cwd=str(repo_dir), timeout_sec=300)
                command_log.append(f"pre_install({cmd[:80]}) rc={proc.returncode}")
                if proc.returncode != 0:
                    failed_record["log_tail"] = (proc.stdout + proc.stderr)[-2000:]
                    self._record_candidate(candidate, "failed", command_log, failed_record)
                    manager.cleanup()
                    return {"ok": False, "candidate": failed_record}

            key_imports = ToolAgent._derive_key_imports(repair_spec)
            validation_passed = manager.validate(key_imports)
            command_log.append(f"validate imports={key_imports} ok={validation_passed}")
            if not validation_passed:
                failed_record["validation_imports"] = key_imports
                self._record_candidate(candidate, "failed", command_log, failed_record)
                manager.cleanup()
                return {"ok": False, "candidate": failed_record}

            pip_freeze = manager.freeze()
            pip_freeze_rel = "execution/env_repair/pip_freeze.txt"
            conda_list_rel = "execution/env_repair/conda_list.txt"
            self.artifacts.write_text(pip_freeze_rel, pip_freeze)
            self.artifacts.write_text(conda_list_rel, self._conda_list(manager))
            result = EnvRepairResult(
                status="success",
                selected_strategy=candidate.name,
                python_version=candidate.python_version,
                backend=manager.backend,
                env_name=env_spec.env_name,
                env_path=manager.env_path_actual(),
                commands=command_log,
                validation_passed=True,
                repaired_environment_file="execution/env_repair/repaired_environment.yml",
                pip_freeze_path=pip_freeze_rel,
                conda_list_path=conda_list_rel,
                reason_codes=[
                    "ENV_REPAIR_APPLIED",
                    *candidate.reason_codes,
                ],
            )
            self._record_candidate(candidate, "success", command_log, result.model_dump())
            return {"ok": True, "result": result}
        except subprocess.TimeoutExpired as exc:
            failed_record["reason_codes"].append("ENV_REPAIR_TIMEOUT")
            failed_record["log_tail"] = f"timed out after {exc.timeout}s"
        except Exception as exc:  # noqa: BLE001
            failed_record["reason_codes"].append("ENV_REPAIR_EXCEPTION")
            failed_record["log_tail"] = str(exc)[-2000:]

        self._record_candidate(candidate, "failed", command_log, failed_record)
        manager.cleanup()
        return {"ok": False, "candidate": failed_record}

    def _load_repo_analysis(self) -> RepoAnalysis:
        payload = self.artifacts.read_json("task/repo_analysis.json")
        return RepoAnalysis(**payload) if payload else RepoAnalysis()

    @staticmethod
    def _resolve_native_environment_file(env_spec: ExecutorEnvSpec, repo_dir: Path) -> Path | None:
        if not env_spec.native_environment_file:
            return None
        path = Path(env_spec.native_environment_file)
        if not path.is_absolute():
            path = repo_dir / path
        return path if path.is_file() else None

    @classmethod
    def _build_candidates(
        cls,
        *,
        env_spec: ExecutorEnvSpec,
        parsed: dict[str, Any],
        original_python: str,
    ) -> list[_RepairCandidate]:
        conda_deps = list(parsed.get("conda_dependencies", []))
        pip_deps = list(parsed.get("pip_dependencies", []))
        pre_install_commands = list(env_spec.pre_install_commands)
        candidates = [
            _RepairCandidate(
                name="relaxed_native",
                python_version=original_python,
                conda_dependencies=cls._relax_conda_dependencies(conda_deps, cpu_torch=False),
                pip_dependencies=pip_deps,
                pre_install_commands=pre_install_commands,
                reason_codes=["ENV_REPAIR_RELAXED_NATIVE"],
            ),
            _RepairCandidate(
                name="cpu_relaxed_py310",
                python_version="3.10",
                conda_dependencies=cls._relax_conda_dependencies(conda_deps, cpu_torch=True),
                pip_dependencies=pip_deps,
                pre_install_commands=pre_install_commands,
                reason_codes=["ENV_REPAIR_CPU_FALLBACK", "ENV_REPAIR_PY310_FALLBACK"],
            ),
        ]
        deduped: list[_RepairCandidate] = []
        seen: set[tuple[str, str]] = set()
        for candidate in candidates:
            key = (candidate.name, candidate.python_version)
            if key not in seen:
                deduped.append(candidate)
                seen.add(key)
        return deduped

    @staticmethod
    def _relax_conda_dependencies(deps: list[CondaDependency], *, cpu_torch: bool) -> list[CondaDependency]:
        relaxed: list[CondaDependency] = []
        has_torch = False
        for dep in deps:
            name = dep.package.lower()
            if name in _SKIP_PACKAGES:
                continue
            if cpu_torch and name in _CUDA_PACKAGES:
                continue
            if name in _TORCH_FAMILY:
                has_torch = True
                relaxed.append(CondaDependency(package=dep.package, channel="pytorch", pip_fallback=True))
                continue
            relaxed.append(
                CondaDependency(
                    package=dep.package,
                    version_constraint=dep.version_constraint,
                    channel=dep.channel or "conda-forge",
                    pip_fallback=True,
                )
            )
        if cpu_torch and has_torch and not any(dep.package.lower() == "cpuonly" for dep in relaxed):
            relaxed.append(CondaDependency(package="cpuonly", channel="pytorch"))
        return relaxed

    @classmethod
    def _parse_native_environment(cls, path: Path) -> dict[str, Any]:
        if yaml is not None:
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}
                return cls._parse_yaml_payload(data, path)
            except Exception:  # noqa: BLE001
                pass
        return cls._parse_environment_text(path)

    @classmethod
    def _parse_yaml_payload(cls, data: dict[str, Any], path: Path) -> dict[str, Any]:
        channels = [str(item) for item in data.get("channels", []) if str(item).strip()]
        default_channel = "conda-forge" if "conda-forge" in channels else (channels[0] if channels else "conda-forge")
        conda_deps: list[CondaDependency] = []
        pip_deps: list[str] = []
        python_version: str | None = None
        for item in data.get("dependencies", []) or []:
            if isinstance(item, str):
                package, version = cls._split_conda_spec(item)
                if not package:
                    continue
                if package.lower() == "python":
                    python_version = cls._extract_python_minor(version or item) or python_version
                    continue
                conda_deps.append(CondaDependency(package=package, version_constraint=version, channel=default_channel, pip_fallback=True))
            elif isinstance(item, dict) and isinstance(item.get("pip"), list):
                pip_deps.extend(str(dep) for dep in item["pip"] if str(dep).strip())
        return {
            "python_version": python_version or ToolAgent._python_version_from_environment(path) or "3.10",
            "conda_dependencies": conda_deps,
            "pip_dependencies": pip_deps,
        }

    @classmethod
    def _parse_environment_text(cls, path: Path) -> dict[str, Any]:
        conda_deps: list[CondaDependency] = []
        pip_deps: list[str] = []
        in_pip = False
        text = path.read_text(encoding="utf-8", errors="ignore")
        for raw in text.splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "- pip:" or stripped == "pip:":
                in_pip = True
                continue
            if not stripped.startswith("-"):
                in_pip = False
                continue
            item = stripped.lstrip("-").strip()
            if in_pip:
                pip_deps.append(item)
                continue
            package, version = cls._split_conda_spec(item)
            if package and package.lower() != "python":
                conda_deps.append(CondaDependency(package=package, version_constraint=version, channel="conda-forge", pip_fallback=True))
        return {
            "python_version": ToolAgent._python_version_from_environment(path) or "3.10",
            "conda_dependencies": conda_deps,
            "pip_dependencies": pip_deps,
        }

    @staticmethod
    def _split_conda_spec(spec: str) -> tuple[str, str | None]:
        token = str(spec or "").strip()
        if not token or token.startswith(("http://", "https://", "git+")):
            return "", None
        if "::" in token:
            token = token.split("::", 1)[1]
        parts = token.split("=")
        package = parts[0].strip()
        if not package:
            return "", None
        if len(parts) >= 2 and parts[1].strip():
            version = parts[1].strip()
            version = re.sub(r"(\.\*)$", "", version)
            return package, version or None
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*([<>=!~].+)?$", token)
        if match:
            return match.group(1), match.group(2).strip() if match.group(2) else None
        return package, None

    @staticmethod
    def _extract_python_minor(text: str) -> str | None:
        return ToolAgent._extract_python_minor(text)

    @staticmethod
    def _classify_failure(log_text: str, reason_codes: list[str]) -> str:
        lowered = (log_text or " ".join(reason_codes)).lower()
        if "timeout" in lowered or "timed out" in lowered:
            return "SOLVER_TIMEOUT"
        if "unsatisfiable" in lowered or "resolvepackagenotfound" in lowered:
            return "UNSAT_CONSTRAINT"
        if "package not found" in lowered or "packagesnotfounderror" in lowered:
            return "PACKAGE_NOT_FOUND"
        if "cuda" in lowered or "cudatoolkit" in lowered:
            return "CUDA_CONFLICT"
        return "UNKNOWN_ENV_FAILURE"

    def _record_candidate(self, candidate: _RepairCandidate, status: str, commands: list[str], payload: dict[str, Any]) -> None:
        self.artifacts.append_jsonl(
            "execution/env_repair/candidates.jsonl",
            {
                "strategy": candidate.name,
                "status": status,
                "commands": commands,
                "payload": payload,
            },
        )

    def _persist(self, result: EnvRepairResult) -> None:
        self.artifacts.write_json("execution/env_repair/env_repair_result.json", result.model_dump())

    @staticmethod
    def _render_repaired_environment_yml(env_spec: ExecutorEnvSpec, native_env: Path) -> str:
        lines = [
            f"name: {env_spec.env_name}",
            "channels:",
            "  - conda-forge",
            "  - pytorch",
            "dependencies:",
            f"  - python={env_spec.python_version}",
        ]
        for dep in env_spec.conda_dependencies:
            spec = dep.package
            if dep.version_constraint:
                spec += dep.version_constraint if dep.version_constraint[0] in "=<>!~" else f"={dep.version_constraint}"
            lines.append(f"  - {spec}")
        if env_spec.pip_dependencies:
            lines.append("  - pip:")
            for dep in env_spec.pip_dependencies:
                lines.append(f"      - {dep}")
        lines.append(f"# repaired_from: {native_env}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _conda_list(manager: CondaEnvManager) -> str:
        if getattr(manager, "_use_venv_fallback", False):
            return manager.freeze()
        conda_bin = getattr(manager, "_conda_bin", None)
        if not conda_bin:
            return ""
        proc = subprocess.run(
            [conda_bin, "list", "-n", manager.env_name],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return proc.stdout if proc.returncode == 0 else proc.stderr

    def _request_repair_guidance(
        self,
        *,
        native_env: Path,
        parsed: dict[str, Any],
        failure_codes: list[str],
        failure_log: str,
    ) -> dict[str, Any] | None:
        if self.llm is None:
            return None
        schema = {
            "failure_class": "SOLVER_TIMEOUT|UNSAT_CONSTRAINT|PACKAGE_NOT_FOUND|CUDA_CONFLICT|UNKNOWN_ENV_FAILURE",
            "recommended_strategies": ["short strategy names in priority order"],
            "python_version": "recommended Python minor or null",
            "dependency_notes": ["specific package/channel/version observations"],
            "reason_codes": ["ENV_REPAIR_LLM_GUIDANCE"],
        }
        user = (
            "Diagnose this failed native conda/mamba environment for an old ML repository. "
            "Recommend bounded repair strategies; do not recommend editing source code here.\n\n"
            f"Native env file: {native_env}\n"
            f"Failure codes: {failure_codes}\n"
            f"Parsed python: {parsed.get('python_version')}\n"
            f"Parsed conda deps: {[getattr(dep, 'package', str(dep)) for dep in parsed.get('conda_dependencies', [])][:40]}\n"
            f"Parsed pip deps: {parsed.get('pip_dependencies', [])[:40]}\n"
            f"Failure log tail:\n{failure_log[-4000:]}\n"
        )
        data, err = self.safe_chat_json(
            schema=schema,
            system=(
                "You are an environment repair planner for reproducibility audits. "
                "Prefer faithful, bounded, auditable environment repairs over broad upgrades."
            ),
            user=user,
        )
        if err or not data:
            return {
                "failure_class": "UNKNOWN_ENV_FAILURE",
                "recommended_strategies": [],
                "dependency_notes": [err or "LLM did not return guidance"],
                "reason_codes": ["ENV_REPAIR_LLM_GUIDANCE_UNAVAILABLE"],
            }
        data.setdefault("reason_codes", ["ENV_REPAIR_LLM_GUIDANCE"])
        return data
