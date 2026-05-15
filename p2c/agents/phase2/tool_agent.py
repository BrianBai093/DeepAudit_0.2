"""ToolAgent — creates conda/venv environments from repository-derived specs."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

try:  # pragma: no cover - Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except Exception:  # pragma: no cover - optional dependency
        tomllib = None

from p2c.agents.base import BaseAgent
from p2c.runtime.conda_env import CondaEnvManager, DepLayer
from p2c.schemas import (
    CondaDependency,
    DependencyProfile,
    EnvSetupResult,
    ExecutorEnvSpec,
    RepoAnalysis,
)


class ToolAgent(BaseAgent):
    """Pure-subprocess agent: deterministic environment setup from repo manifests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="tool_agent", *args, **kwargs)
        self._env_mgr: CondaEnvManager | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        env_spec_raw = ctx["_p2_env_spec"]
        env_spec = env_spec_raw if isinstance(env_spec_raw, ExecutorEnvSpec) else ExecutorEnvSpec(**env_spec_raw)
        repo_dir = Path(str(ctx["repo_dir"])).resolve()
        native_environment_file = self._resolve_native_environment_file(env_spec, repo_dir)
        if native_environment_file is None:
            self._augment_env_spec_dependencies(env_spec, repo_dir)
        self._env_mgr = CondaEnvManager(
            env_name=env_spec.env_name,
            python_version=env_spec.python_version,
        )
        self.log("PROGRESS", f"backend={self._env_mgr.backend}, env={env_spec.env_name}, python={env_spec.python_version}")

        result = EnvSetupResult(
            env_name=env_spec.env_name,
            python_version=env_spec.python_version,
        )

        self.log("PROGRESS", "creating environment...")
        if native_environment_file is not None:
            self.log("PROGRESS", f"using native conda environment file: {native_environment_file}")
            create_out = self._env_mgr.create_from_environment_file(native_environment_file)
            result.install_commands.append(f"conda env create -f {native_environment_file} (ok={create_out['ok']})")
            if create_out["ok"]:
                result.reason_codes.append("NATIVE_CONDA_ENV_CREATED")
        else:
            create_out = self._env_mgr.create()
            result.install_commands.append(f"create env (ok={create_out['ok']})")
        if not create_out["ok"]:
            self.log("PROGRESS", f"env creation failed: {create_out['log'][:500]}")
            if native_environment_file is not None:
                result.reason_codes.append("NATIVE_CONDA_ENV_CREATE_FAILED")
                self._persist(result)
                return {"env_result": result}
            if env_spec.python_version != "3.10":
                self.log("PROGRESS", "retrying with python=3.10")
                self._env_mgr = CondaEnvManager(env_name=env_spec.env_name, python_version="3.10")
                create_out = self._env_mgr.create()
                result.python_version = "3.10"
                if not create_out["ok"]:
                    result.reason_codes.append("ENV_CREATE_FAILED")
                    self._persist(result)
                    return {"env_result": result}

        result.env_path = self._env_mgr.env_path_actual()
        if native_environment_file is not None:
            actual_python = self._env_mgr.python_version_actual()
            if actual_python:
                result.python_version = actual_python.split()[0]

        if env_spec.system_packages:
            self.log("PROGRESS", f"installing {len(env_spec.system_packages)} system packages...")
            proc = self._env_mgr.install_system_packages(env_spec.system_packages)
            result.system_install_log.append(proc.stderr[:2000] if proc.stderr else "ok")
            result.install_commands.append(f"system packages rc={proc.returncode}")

        for cmd in env_spec.pre_install_commands:
            self.log("PROGRESS", f"pre-install: {cmd[:80]}")
            proc = self._env_mgr.run_in_env(cmd, cwd=str(repo_dir), timeout_sec=300)
            result.install_commands.append(f"pre-install({cmd[:60]}) rc={proc.returncode}")
            if proc.returncode != 0:
                result.reason_codes.append("PREINSTALL_FAILED")

        use_layered = os.getenv("P2C_LAYERED_INSTALL", "1") == "1"
        if native_environment_file is not None:
            self.log("PROGRESS", "native conda env created; skipping derived dependency install")
        elif use_layered and (env_spec.conda_dependencies or env_spec.pip_dependencies):
            self._install_layered(env_spec, result)
        else:
            self._install_flat(env_spec, result)

        self.log("PROGRESS", "validating environment...")
        key_imports = self._derive_key_imports(env_spec)
        result.validation_passed = self._env_mgr.validate(key_imports)

        result.installed_packages_snapshot = self._env_mgr.freeze()
        self.artifacts.write_text("execution/env_lock/pip_freeze.txt", result.installed_packages_snapshot)

        if result.failed_packages:
            result.reason_codes.append("SOME_PACKAGES_FAILED")
        if not result.validation_passed:
            result.reason_codes.append("VALIDATION_FAILED")

        self._persist(result)
        self.log("DONE", f"env ready: {len(result.failed_packages)} failures, valid={result.validation_passed}")
        return {"env_result": result}

    def build_env_spec(self, ctx: dict[str, Any]) -> ExecutorEnvSpec:
        repo_dir = Path(str(ctx["repo_dir"])).resolve()
        repo_analysis_raw = self.artifacts.read_json("task/repo_analysis.json")
        repo_analysis = RepoAnalysis(**repo_analysis_raw) if repo_analysis_raw else RepoAnalysis()
        env_name = f"{ctx.get('run_id', 'p2c')}_executor"
        env_spec = self._build_env_spec(repo_dir=repo_dir, repo_analysis=repo_analysis, env_name=env_name)
        self.artifacts.write_json("execution/executor_env_spec.json", env_spec.model_dump())
        return env_spec

    def cleanup(self) -> None:
        if self._env_mgr:
            self.log("PROGRESS", "cleaning up environment...")
            self._env_mgr.cleanup()

    @property
    def env_manager(self) -> CondaEnvManager | None:
        return self._env_mgr

    # ------------------------------------------------------------------
    # Deterministic env-spec builder
    # ------------------------------------------------------------------

    @classmethod
    def _build_env_spec(
        cls,
        *,
        repo_dir: Path,
        repo_analysis: RepoAnalysis,
        env_name: str,
    ) -> ExecutorEnvSpec:
        pip_dependencies: list[str] = []
        conda_dependencies: list[CondaDependency] = []
        pre_install_commands: list[str] = []
        reason_codes: list[str] = ["REPO_ENV_SPEC"]
        native_environment_file = cls._find_native_environment_file(repo_dir, repo_analysis)

        for profile in repo_analysis.dependency_profiles:
            if native_environment_file is not None and profile.manager == "conda":
                reason_codes.append("ENV_SPEC_FROM_NATIVE_CONDA_FILE")
                continue
            cls._merge_profile_dependencies(
                repo_dir=repo_dir,
                profile=profile,
                pip_dependencies=pip_dependencies,
                conda_dependencies=conda_dependencies,
                pre_install_commands=pre_install_commands,
                reason_codes=reason_codes,
            )

        python_version = cls._infer_python_version(repo_dir, repo_analysis)
        return ExecutorEnvSpec(
            env_name=env_name,
            python_version=python_version,
            native_environment_file=(
                cls._repo_relative_path(repo_dir, native_environment_file)
                if native_environment_file is not None
                else None
            ),
            conda_dependencies=conda_dependencies,
            pip_dependencies=pip_dependencies,
            system_packages=[],
            pre_install_commands=pre_install_commands,
            reason_codes=list(dict.fromkeys(reason_codes)),
        )

    @classmethod
    def _merge_profile_dependencies(
        cls,
        *,
        repo_dir: Path,
        profile: DependencyProfile,
        pip_dependencies: list[str],
        conda_dependencies: list[CondaDependency],
        pre_install_commands: list[str],
        reason_codes: list[str],
    ) -> None:
        if profile.manager == "pip_requirements":
            for manifest_path in profile.manifest_paths:
                manifest = repo_dir / manifest_path
                for requirement in cls._parse_requirements_file(manifest):
                    cls._append_unique_pip(pip_dependencies, requirement)
            reason_codes.append("ENV_SPEC_FROM_REQUIREMENTS")
            return

        if profile.manager == "pip_editable":
            cmd = "python -m pip install -e ."
            if profile.cwd and profile.cwd != ".":
                cmd = f"cd {profile.cwd} && {cmd}"
            cls._append_unique_command(pre_install_commands, cmd)
            reason_codes.append("ENV_SPEC_FROM_EDITABLE_INSTALL")
            return

        if profile.manager == "poetry":
            cmd = "python -m pip install poetry && poetry install"
            if profile.cwd and profile.cwd != ".":
                cmd = f"cd {profile.cwd} && {cmd}"
            cls._append_unique_command(pre_install_commands, cmd)
            reason_codes.append("ENV_SPEC_FROM_POETRY")
            return

        if profile.manager == "conda":
            for manifest_path in profile.manifest_paths:
                manifest = repo_dir / manifest_path
                for dep in cls._parse_environment_yml(manifest):
                    cls._append_unique_conda(conda_dependencies, dep)
            reason_codes.append("ENV_SPEC_FROM_CONDA_FILE")

    @classmethod
    def _infer_python_version(cls, repo_dir: Path, repo_analysis: RepoAnalysis) -> str:
        for profile in repo_analysis.dependency_profiles:
            for manifest_path in profile.manifest_paths:
                path = repo_dir / manifest_path
                if path.name == "pyproject.toml":
                    version = cls._python_version_from_pyproject(path)
                    if version:
                        return version
                if path.name in {"environment.yml", "environment.yaml"}:
                    version = cls._python_version_from_environment(path)
                    if version:
                        return version

        setup_py = repo_dir / "setup.py"
        if setup_py.is_file():
            text = setup_py.read_text(encoding="utf-8", errors="ignore")
            match = re.search(r"python_requires\s*=\s*['\"]([^'\"]+)['\"]", text)
            version = cls._extract_python_minor(match.group(1)) if match else None
            if version:
                return version

        return "3.10"

    @classmethod
    def _find_native_environment_file(cls, repo_dir: Path, repo_analysis: RepoAnalysis) -> Path | None:
        for profile in repo_analysis.dependency_profiles:
            if profile.manager != "conda":
                continue
            for manifest_path in profile.manifest_paths:
                path = repo_dir / manifest_path
                if path.name in {"environment.yml", "environment.yaml"} and path.is_file():
                    return path
        return None

    @staticmethod
    def _repo_relative_path(repo_dir: Path, path: Path) -> str:
        try:
            return str(path.relative_to(repo_dir))
        except ValueError:
            return str(path)

    @staticmethod
    def _resolve_native_environment_file(env_spec: ExecutorEnvSpec, repo_dir: Path) -> Path | None:
        if not env_spec.native_environment_file:
            return None
        path = Path(env_spec.native_environment_file)
        if not path.is_absolute():
            path = repo_dir / path
        return path if path.is_file() else None

    @classmethod
    def _python_version_from_pyproject(cls, path: Path) -> str | None:
        if tomllib is None or not path.is_file():
            return None
        try:
            with path.open("rb") as handle:
                data = tomllib.load(handle)
        except Exception:  # noqa: BLE001
            return None
        project = data.get("project") if isinstance(data, dict) else {}
        requires_python = project.get("requires-python") if isinstance(project, dict) else None
        if isinstance(requires_python, str):
            return cls._extract_python_minor(requires_python)
        return None

    @classmethod
    def _python_version_from_environment(cls, path: Path) -> str | None:
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"^\s*-\s*python\s*=\s*(3\.(?:8|9|10|11|12))", text, re.M)
        return match.group(1) if match else cls._extract_python_minor(text)

    @staticmethod
    def _extract_python_minor(text: str) -> str | None:
        match = re.search(r"3\.(8|9|10|11|12)", text)
        return f"3.{match.group(1)}" if match else None

    @classmethod
    def _parse_environment_yml(cls, path: Path) -> list[CondaDependency]:
        deps: list[CondaDependency] = []
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            return deps
        for raw_line in text.splitlines():
            stripped = raw_line.strip().lstrip("-").strip()
            if not stripped or stripped.startswith("#") or stripped == "dependencies:" or stripped == "pip:":
                continue
            if stripped.startswith(("git+", "http://", "https://")):
                continue
            if stripped.startswith("python="):
                continue
            package, version = cls._split_dep_spec(stripped)
            if package:
                deps.append(CondaDependency(package=package, version_constraint=version, channel="conda-forge"))
        return deps

    # ------------------------------------------------------------------
    # Layered vs flat install
    # ------------------------------------------------------------------

    def _install_layered(self, env_spec: ExecutorEnvSpec, result: EnvSetupResult) -> None:
        layers = self._build_layers(env_spec)
        self.log("PROGRESS", f"layered install: {len(layers)} tiers ({', '.join(l.name for l in layers)})")

        layer_results = self._env_mgr.install_layered(layers)
        for lr in layer_results:
            entry = f"layer={lr.layer_name} ok={lr.ok} elapsed={lr.elapsed_sec:.1f}s"
            if lr.ok:
                result.conda_install_log.append(entry)
            else:
                result.conda_install_log.append(f"{entry} FAILED: {lr.log}")
                result.failed_packages.extend(lr.failed_packages)
            result.install_commands.append(entry)
            self.log("PROGRESS", entry)

            if lr.layer_name == "core" and lr.ok and not self._env_mgr.validate_abi():
                self.log("PROGRESS", "numpy ABI mismatch detected — reinstalling numpy via pip")
                self._env_mgr.run_in_env("pip install --force-reinstall numpy", timeout_sec=120)
                result.install_commands.append("numpy ABI fix (pip reinstall)")

    def _install_flat(self, env_spec: ExecutorEnvSpec, result: EnvSetupResult) -> None:
        if env_spec.conda_dependencies:
            self.log("PROGRESS", f"installing {len(env_spec.conda_dependencies)} conda packages...")
            logs = self._env_mgr.install_conda_packages(env_spec.conda_dependencies)
            for entry in logs:
                result.conda_install_log.append(f"channel={entry['channel']} specs={entry['specs']} rc={entry['rc']}")
                if entry["rc"] != 0:
                    result.failed_packages.extend(entry["specs"])

        if env_spec.pip_dependencies:
            self.log("PROGRESS", f"installing {len(env_spec.pip_dependencies)} pip packages...")
            proc = self._env_mgr.install_pip_packages(env_spec.pip_dependencies)
            result.pip_install_log.append(proc.stderr[:3000] if proc.stderr else "ok")
            result.install_commands.append(f"pip install rc={proc.returncode}")
            if proc.returncode != 0:
                for pkg in env_spec.pip_dependencies:
                    single = self._env_mgr.install_pip_packages([pkg])
                    if single.returncode != 0:
                        result.failed_packages.append(pkg)

    @staticmethod
    def _build_layers(env_spec: ExecutorEnvSpec) -> list[DepLayer]:
        CORE_PKGS = {
            "torch", "torchvision", "torchaudio", "pytorch",
            "tensorflow", "tensorflow-gpu", "tf-nightly",
            "jax", "jaxlib", "numpy", "scipy",
        }
        ML_PKGS = {
            "scikit-learn", "sklearn", "pandas", "matplotlib",
            "seaborn", "xgboost", "lightgbm", "transformers",
            "datasets", "tokenizers", "accelerate", "huggingface-hub",
        }
        IMPORT_MAP = {
            "torch": "torch", "torchvision": "torchvision",
            "tensorflow": "tensorflow", "jax": "jax",
            "numpy": "numpy", "scipy": "scipy",
            "scikit-learn": "sklearn", "pandas": "pandas",
            "transformers": "transformers",
        }

        def pkg_name(spec: str) -> str:
            return spec.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip().lower()

        skip_pkgs = {"python", "pip", "setuptools", "wheel"}
        core_conda, ml_conda, paper_conda = [], [], []
        for dep in env_spec.conda_dependencies:
            name = pkg_name(dep.package)
            if name in skip_pkgs:
                continue
            if name in CORE_PKGS:
                core_conda.append(dep)
            elif name in ML_PKGS:
                ml_conda.append(dep)
            else:
                paper_conda.append(dep)

        core_pip, ml_pip, paper_pip = [], [], []
        for dep in env_spec.pip_dependencies:
            name = pkg_name(dep)
            if name in CORE_PKGS:
                core_pip.append(dep)
            elif name in ML_PKGS:
                ml_pip.append(dep)
            else:
                paper_pip.append(dep)

        frameworks = {"torch", "torchvision", "torchaudio", "pytorch", "tensorflow", "tensorflow-gpu", "tf-nightly", "jax", "jaxlib"}
        if any(pkg_name(dep) in frameworks for dep in core_pip):
            abi_sensitive = {"numpy", "scipy", "h5py"}
            for tier_conda, tier_pip in [(core_conda, core_pip), (ml_conda, ml_pip)]:
                to_remove: list[CondaDependency] = []
                for dep in tier_conda:
                    if pkg_name(dep.package) in abi_sensitive:
                        spec = dep.package if not dep.version_constraint else f"{dep.package}{dep.version_constraint if dep.version_constraint[0] in ('=', '>', '<', '!', '~') else '==' + dep.version_constraint}"
                        if spec not in tier_pip and spec not in core_pip:
                            core_pip.append(spec)
                        to_remove.append(dep)
                for dep in to_remove:
                    tier_conda.remove(dep)
            if not any("numpy" in dep.lower() for dep in core_pip):
                core_pip.append("numpy")

        def imports_for(conda_deps: list[CondaDependency], pip_deps: list[str]) -> list[str]:
            imports: list[str] = []
            for dep in conda_deps:
                name = pkg_name(dep.package)
                if name in IMPORT_MAP:
                    imports.append(IMPORT_MAP[name])
            for dep in pip_deps:
                name = pkg_name(dep)
                if name in IMPORT_MAP:
                    imports.append(IMPORT_MAP[name])
            return imports[:5]

        layers: list[DepLayer] = []
        if core_conda or core_pip:
            layers.append(DepLayer("core", core_conda, core_pip, imports_for(core_conda, core_pip), True))
        if ml_conda or ml_pip:
            layers.append(DepLayer("ml_libs", ml_conda, ml_pip, imports_for(ml_conda, ml_pip), False))
        if paper_conda or paper_pip:
            layers.append(DepLayer("paper_specific", paper_conda, paper_pip, [], False))
        return layers

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _persist(self, result: EnvSetupResult) -> None:
        self.artifacts.write_json("execution/env_setup_result.json", result.model_dump())

    @staticmethod
    def _base_pkg_name(spec: str) -> str:
        token = str(spec or "").strip()
        token = token.split(";")[0].strip()
        token = token.split("#")[0].strip()
        return re.split(r"[<>=!~\[\s]", token, maxsplit=1)[0].strip().lower()

    @classmethod
    def _parse_requirements_file(cls, path: Path) -> list[str]:
        requirements: list[str] = []
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            return requirements
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("-r", "--", "git+", "http://", "https://", "-e ")):
                continue
            requirements.append(line)
        return requirements

    @classmethod
    def _augment_env_spec_dependencies(cls, env_spec: ExecutorEnvSpec, repo_dir: Path) -> None:
        manifest_candidates = ["requirements.txt", "requirements-dev.txt", "requirements_dev.txt"]
        existing = {cls._base_pkg_name(dep) for dep in env_spec.pip_dependencies}
        aliases = {"theano": "theano-pymc", "sklearn": "scikit-learn"}
        for manifest_name in manifest_candidates:
            manifest = repo_dir / manifest_name
            if not manifest.is_file():
                continue
            for requirement in cls._parse_requirements_file(manifest):
                base = cls._base_pkg_name(requirement)
                if not base:
                    continue
                if base in existing or aliases.get(base) in existing:
                    continue
                env_spec.pip_dependencies.append(requirement)
                existing.add(base)

    @staticmethod
    def _derive_key_imports(env_spec: ExecutorEnvSpec) -> list[str]:
        imports: list[str] = []
        known_map = {
            "torch": "torch", "torchvision": "torchvision",
            "tensorflow": "tensorflow", "keras": "keras",
            "scikit-learn": "sklearn", "opencv-python": "cv2",
            "opencv-python-headless": "cv2", "pillow": "PIL",
            "pyyaml": "yaml", "beautifulsoup4": "bs4",
            "imbalanced-learn": "imblearn",
        }
        for dep in env_spec.pip_dependencies[:20]:
            name = dep.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip().lower()
            if name in known_map:
                imports.append(known_map[name])
            elif name.replace("-", "_").isidentifier():
                imports.append(name.replace("-", "_"))
        return imports[:10]

    @staticmethod
    def _append_unique_command(commands: list[str], command: str) -> None:
        if command and command not in commands:
            commands.append(command)

    @classmethod
    def _append_unique_pip(cls, dependencies: list[str], requirement: str) -> None:
        base = cls._base_pkg_name(requirement)
        if not base:
            return
        if base not in {cls._base_pkg_name(item) for item in dependencies}:
            dependencies.append(requirement)

    @staticmethod
    def _append_unique_conda(dependencies: list[CondaDependency], dep: CondaDependency) -> None:
        existing = {(item.package.lower(), item.version_constraint or "", item.channel) for item in dependencies}
        key = (dep.package.lower(), dep.version_constraint or "", dep.channel)
        if key not in existing:
            dependencies.append(dep)

    @staticmethod
    def _split_dep_spec(spec: str) -> tuple[str, str | None]:
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*([<>=!~].+)?$", spec.strip())
        if not match:
            return "", None
        package = match.group(1)
        version = match.group(2).strip() if match.group(2) else None
        return package, version
