"""ToolAgent — creates conda/venv environment and installs dependencies from the plan."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.runtime.conda_env import CondaEnvManager, DepLayer
from p2c.schemas import CondaDependency, EnvSetupResult, ExecutionPlan


class ToolAgent(BaseAgent):
    """Pure-subprocess agent: no LLM calls, just environment setup."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="tool_agent", *args, **kwargs)
        self._env_mgr: CondaEnvManager | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        plan: ExecutionPlan = ctx["_p2_plan"]
        repo_dir = Path(str(ctx["repo_dir"])).resolve()
        self._augment_plan_dependencies(plan, repo_dir)
        self._env_mgr = CondaEnvManager(
            env_name=plan.env_name,
            python_version=plan.python_version,
        )
        self.log("PROGRESS", f"backend={self._env_mgr.backend}, env={plan.env_name}, "
                              f"python={plan.python_version}")

        result = EnvSetupResult(
            env_name=plan.env_name,
            python_version=plan.python_version,
        )

        # 1. Create environment
        self.log("PROGRESS", "creating environment...")
        create_out = self._env_mgr.create()
        result.install_commands.append(f"create env (ok={create_out['ok']})")
        if not create_out["ok"]:
            self.log("PROGRESS", f"env creation failed: {create_out['log'][:500]}")
            # Try fallback python version
            if plan.python_version != "3.10":
                self.log("PROGRESS", "retrying with python=3.10")
                self._env_mgr = CondaEnvManager(env_name=plan.env_name, python_version="3.10")
                create_out = self._env_mgr.create()
                result.python_version = "3.10"
                if not create_out["ok"]:
                    result.reason_codes.append("ENV_CREATE_FAILED")
                    self._persist(result)
                    return {"env_result": result}

        result.env_path = self._env_mgr.env_path_actual()

        # 2. System packages (best-effort)
        if plan.system_packages:
            self.log("PROGRESS", f"installing {len(plan.system_packages)} system packages...")
            proc = self._env_mgr.install_system_packages(plan.system_packages)
            result.system_install_log.append(proc.stderr[:2000] if proc.stderr else "ok")
            result.install_commands.append(f"system packages rc={proc.returncode}")

        # 3. Pre-install commands
        for cmd in plan.pre_install_commands:
            self.log("PROGRESS", f"pre-install: {cmd[:80]}")
            proc = self._env_mgr.run_in_env(cmd, cwd=str(repo_dir), timeout_sec=120)
            result.install_commands.append(f"pre-install({cmd[:60]}) rc={proc.returncode}")
            if proc.returncode != 0:
                result.reason_codes.append("PREINSTALL_FAILED")

        # 4 & 5. Dependencies — layered or flat install
        use_layered = os.getenv("P2C_LAYERED_INSTALL", "1") == "1"
        if use_layered and (plan.conda_dependencies or plan.pip_dependencies):
            self._install_layered(plan, result)
        else:
            self._install_flat(plan, result)

        # 6. Validate
        self.log("PROGRESS", "validating environment...")
        key_imports = self._derive_key_imports(plan)
        result.validation_passed = self._env_mgr.validate(key_imports)

        # 7. Snapshot
        result.installed_packages_snapshot = self._env_mgr.freeze()
        self.artifacts.write_text("execution/env_lock/pip_freeze.txt", result.installed_packages_snapshot)

        if result.failed_packages:
            result.reason_codes.append("SOME_PACKAGES_FAILED")
        if not result.validation_passed:
            result.reason_codes.append("VALIDATION_FAILED")

        self._persist(result)
        self.log("DONE", f"env ready: {len(result.failed_packages)} failures, "
                          f"valid={result.validation_passed}")
        return {"env_result": result}

    # ------------------------------------------------------------------
    # Cleanup (called by orchestrator)
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        if self._env_mgr:
            self.log("PROGRESS", "cleaning up environment...")
            self._env_mgr.cleanup()

    @property
    def env_manager(self) -> CondaEnvManager | None:
        return self._env_mgr

    # ------------------------------------------------------------------
    # Layered vs flat install
    # ------------------------------------------------------------------

    def _install_layered(self, plan: ExecutionPlan, result: EnvSetupResult) -> None:
        """Install deps in priority tiers: core → ML frameworks → paper-specific."""
        layers = self._build_layers(plan)
        self.log("PROGRESS", f"layered install: {len(layers)} tiers "
                              f"({', '.join(l.name for l in layers)})")

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

            # After core layer, verify numpy ABI compatibility
            if lr.layer_name == "core" and lr.ok:
                if not self._env_mgr.validate_abi():
                    self.log("PROGRESS", "numpy ABI mismatch detected — reinstalling numpy via pip")
                    self._env_mgr.run_in_env(
                        "pip install --force-reinstall numpy", timeout_sec=120,
                    )
                    result.install_commands.append("numpy ABI fix (pip reinstall)")

    def _install_flat(self, plan: ExecutionPlan, result: EnvSetupResult) -> None:
        """Original flat install path (no layering)."""
        if plan.conda_dependencies:
            self.log("PROGRESS", f"installing {len(plan.conda_dependencies)} conda packages...")
            logs = self._env_mgr.install_conda_packages(plan.conda_dependencies)
            for entry in logs:
                result.conda_install_log.append(
                    f"channel={entry['channel']} specs={entry['specs']} rc={entry['rc']}"
                )
                if entry["rc"] != 0:
                    for spec in entry["specs"]:
                        result.failed_packages.append(spec)

        if plan.pip_dependencies:
            self.log("PROGRESS", f"installing {len(plan.pip_dependencies)} pip packages...")
            proc = self._env_mgr.install_pip_packages(plan.pip_dependencies)
            result.pip_install_log.append(proc.stderr[:3000] if proc.stderr else "ok")
            result.install_commands.append(f"pip install rc={proc.returncode}")
            if proc.returncode != 0:
                for pkg in plan.pip_dependencies:
                    single = self._env_mgr.install_pip_packages([pkg])
                    if single.returncode != 0:
                        result.failed_packages.append(pkg)

    @staticmethod
    def _build_layers(plan: ExecutionPlan) -> list[DepLayer]:
        """Partition plan dependencies into install tiers.

        Tier 1 (core):   python runtime deps, pytorch/tensorflow, numpy
        Tier 2 (ml_libs): common ML libraries (sklearn, scipy, pandas, etc.)
        Tier 3 (paper):   everything else (paper-specific packages)

        Each tier gets verify_imports so a failure is caught before the next
        tier is installed. Tier 1 is critical — if it fails, abort.
        """
        # Known packages for each tier
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
        # Import verification map
        IMPORT_MAP = {
            "torch": "torch", "torchvision": "torchvision",
            "tensorflow": "tensorflow", "jax": "jax",
            "numpy": "numpy", "scipy": "scipy",
            "scikit-learn": "sklearn", "pandas": "pandas",
            "transformers": "transformers",
        }

        def _pkg_name(spec: str) -> str:
            return spec.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip().lower()

        # Packages that are already handled by env creation — skip them
        _SKIP_PKGS = {"python", "pip", "setuptools", "wheel"}

        # Partition conda deps
        core_conda, ml_conda, paper_conda = [], [], []
        for dep in plan.conda_dependencies:
            name = _pkg_name(dep.package)
            if name in _SKIP_PKGS:
                continue  # already in env from conda create
            if name in CORE_PKGS:
                core_conda.append(dep)
            elif name in ML_PKGS:
                ml_conda.append(dep)
            else:
                paper_conda.append(dep)

        # Partition pip deps
        core_pip, ml_pip, paper_pip = [], [], []
        for dep in plan.pip_dependencies:
            name = _pkg_name(dep)
            if name in CORE_PKGS:
                core_pip.append(dep)
            elif name in ML_PKGS:
                ml_pip.append(dep)
            else:
                paper_pip.append(dep)

        # ------------------------------------------------------------------
        # ABI compatibility: when a DL framework (tensorflow/torch/jax) is
        # installed via pip, numpy MUST also come from pip.  Mixing a
        # conda-forge numpy with a pip tensorflow causes C-level ABI
        # mismatches ("numpy.dtype size changed").
        # ------------------------------------------------------------------
        DL_FRAMEWORKS = {
            "torch", "torchvision", "torchaudio", "pytorch",
            "tensorflow", "tensorflow-gpu", "tf-nightly",
            "jax", "jaxlib",
        }
        dl_in_pip = any(_pkg_name(dep) in DL_FRAMEWORKS for dep in core_pip)
        if dl_in_pip:
            # Move numpy (and scipy) from conda → pip
            _ABI_SENSITIVE = {"numpy", "scipy", "h5py"}
            moved_from_conda: list[str] = []
            for tier_conda, tier_pip in [
                (core_conda, core_pip),
                (ml_conda, ml_pip),
            ]:
                to_remove = []
                for d in tier_conda:
                    if _pkg_name(d.package) in _ABI_SENSITIVE:
                        vc = d.version_constraint
                        # Build pip-style spec: numpy==1.26 (bare version gets ==)
                        if vc and vc[0] not in ("=", ">", "<", "!", "~"):
                            spec = f"{d.package}=={vc}"
                        else:
                            spec = f"{d.package}{vc or ''}"
                        if spec not in tier_pip and spec not in core_pip:
                            core_pip.append(spec)
                        moved_from_conda.append(d.package)
                        to_remove.append(d)
                for d in to_remove:
                    tier_conda.remove(d)

            # Also ensure numpy is explicitly in pip if not already present
            # (it may only be a transitive dep of tensorflow)
            has_numpy_pip = any("numpy" in dep.lower() for dep in core_pip)
            if not has_numpy_pip:
                core_pip.append("numpy")

        # Build verify lists
        def _imports_for(conda_deps: list[CondaDependency], pip_deps: list[str]) -> list[str]:
            imports = []
            for d in conda_deps:
                name = _pkg_name(d.package)
                if name in IMPORT_MAP:
                    imports.append(IMPORT_MAP[name])
            for d in pip_deps:
                name = _pkg_name(d)
                if name in IMPORT_MAP:
                    imports.append(IMPORT_MAP[name])
            return imports[:5]

        layers = []
        if core_conda or core_pip:
            layers.append(DepLayer(
                name="core",
                conda_deps=core_conda,
                pip_deps=core_pip,
                verify_imports=_imports_for(core_conda, core_pip),
                is_critical=True,
            ))
        if ml_conda or ml_pip:
            layers.append(DepLayer(
                name="ml_libs",
                conda_deps=ml_conda,
                pip_deps=ml_pip,
                verify_imports=_imports_for(ml_conda, ml_pip),
                is_critical=False,
            ))
        if paper_conda or paper_pip:
            layers.append(DepLayer(
                name="paper_specific",
                conda_deps=paper_conda,
                pip_deps=paper_pip,
                verify_imports=[],  # too diverse to verify generically
                is_critical=False,
            ))

        # Edge case: if no layers were created (all deps are empty), return empty
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
    def _augment_plan_dependencies(cls, plan: ExecutionPlan, repo_dir: Path) -> None:
        manifest_candidates = ["requirements.txt", "requirements-dev.txt", "requirements_dev.txt"]
        existing = {cls._base_pkg_name(dep) for dep in plan.pip_dependencies}
        aliases = {
            "theano": "theano-pymc",
            "sklearn": "scikit-learn",
        }
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
                plan.pip_dependencies.append(requirement)
                existing.add(base)

    @staticmethod
    def _derive_key_imports(plan: ExecutionPlan) -> list[str]:
        """Guess top-level importable package names from pip deps."""
        imports: list[str] = []
        known_map = {
            "torch": "torch", "torchvision": "torchvision",
            "tensorflow": "tensorflow", "keras": "keras",
            "scikit-learn": "sklearn", "opencv-python": "cv2",
            "opencv-python-headless": "cv2", "pillow": "PIL",
            "pyyaml": "yaml", "beautifulsoup4": "bs4",
            "imbalanced-learn": "imblearn",
        }
        for dep in plan.pip_dependencies[:20]:  # only check first 20
            name = dep.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip().lower()
            if name in known_map:
                imports.append(known_map[name])
            elif name.replace("-", "_").isidentifier():
                imports.append(name.replace("-", "_"))
        return imports[:10]  # cap at 10
