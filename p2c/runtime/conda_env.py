"""Conda / venv environment lifecycle manager for Phase 2 local execution.

Features beyond basic create/install:
  - **Snapshot & restore**: ``conda create --clone`` based fast backup (hard-links,
    completes in seconds) so the env can be rolled back after a failed install.
  - **Layered install**: install dependencies in priority tiers (core → ML libs →
    paper-specific), verify each tier, rollback only the failing tier.
  - **Freeze-installed protection**: ``--freeze-installed`` flag on conda install
    prevents the solver from downgrading critical packages (torch, numpy, python).
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from p2c.schemas import CondaDependency

logger = logging.getLogger(__name__)


def _conda_spec(dep: CondaDependency) -> str:
    """Build a proper conda spec string like ``python=3.10`` or ``numpy>=1.24``.

    Handles the common case where ``version_constraint`` is a bare version
    (e.g. ``"3.10"``) without an operator prefix — conda requires ``=`` as
    the separator.
    """
    vc = dep.version_constraint
    if not vc:
        return dep.package
    # Already has an operator (=, ==, >=, <=, !=, ~=, etc.)
    if vc[0] in ("=", ">", "<", "!", "~"):
        return f"{dep.package}{vc}"
    # Bare version like "3.10" → conda uses single "="
    return f"{dep.package}={vc}"


def _pip_spec(dep: CondaDependency) -> str:
    """Build a pip spec string like ``numpy==1.26.4`` or ``tensorflow>=2.12``."""
    vc = dep.version_constraint
    if not vc:
        return dep.package
    if vc[0] in ("=", ">", "<", "!", "~"):
        return f"{dep.package}{vc}"
    # Bare version → pip uses "=="
    return f"{dep.package}=={vc}"


# ---------------------------------------------------------------------------
# Layered install data structures
# ---------------------------------------------------------------------------

@dataclass
class DepLayer:
    """One tier in the layered dependency installation strategy."""

    name: str
    conda_deps: list[CondaDependency] = field(default_factory=list)
    pip_deps: list[str] = field(default_factory=list)
    verify_imports: list[str] = field(default_factory=list)
    is_critical: bool = False  # If True, failure aborts entire install


@dataclass
class LayerResult:
    """Outcome of installing a single DepLayer."""

    layer_name: str
    ok: bool
    failed_packages: list[str] = field(default_factory=list)
    log: str = ""
    elapsed_sec: float = 0.0


class CondaEnvManager:
    """Create, install into, validate, and destroy a conda (or venv fallback) environment."""

    def __init__(self, env_name: str, python_version: str = "3.10") -> None:
        self.env_name = env_name
        self.python_version = python_version
        self._conda_bin = self._find_conda()
        self._use_venv_fallback = self._conda_bin is None
        # venv lives under /tmp so it doesn't pollute the repo
        self._venv_path = Path(os.environ.get("P2C_VENV_ROOT", "/tmp")) / f"p2c_venv_{env_name}"
        # Track active snapshots for cleanup
        self._snapshots: list[str] = []

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_conda() -> str | None:
        for cmd in ("mamba", "conda"):
            if shutil.which(cmd):
                return cmd
        return None

    @property
    def backend(self) -> str:
        return "venv" if self._use_venv_fallback else (self._conda_bin or "conda")

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create(self) -> dict[str, Any]:
        """Create the isolated environment.  Returns ``{"ok": bool, "log": str}``."""
        if self._use_venv_fallback:
            return self._create_venv()
        proc = subprocess.run(
            [self._conda_bin, "create", "-n", self.env_name,
             f"python={self.python_version}", "-y", "--quiet"],
            capture_output=True, text=True, timeout=300,
        )
        ok = proc.returncode == 0
        if ok:
            self._ensure_python3_symlink()
        return {"ok": ok, "log": (proc.stdout + proc.stderr).strip()}

    def _ensure_python3_symlink(self) -> None:
        """Ensure ``python3`` exists in the env's bin dir.

        Some conda builds only install ``python`` and not ``python3``.  Repo
        scripts, Makefiles, and shebangs commonly use ``python3``, so this
        symlink prevents a confusing fallback to ``/usr/bin/python3``.
        """
        env_path = self.env_path_actual()
        if not env_path:
            return
        bin_dir = Path(env_path) / "bin"
        python3 = bin_dir / "python3"
        python_ = bin_dir / "python"
        if python_.exists() and not python3.exists():
            try:
                python3.symlink_to(python_)
                logger.info("created python3 symlink in %s", bin_dir)
            except OSError:
                logger.warning("could not create python3 symlink in %s", bin_dir)

    def _create_venv(self) -> dict[str, Any]:
        self._venv_path.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["python3", "-m", "venv", str(self._venv_path)],
            capture_output=True, text=True, timeout=120,
        )
        return {"ok": proc.returncode == 0, "log": (proc.stdout + proc.stderr).strip()}

    # ------------------------------------------------------------------
    # Run commands inside the env
    # ------------------------------------------------------------------

    def run_in_env(
        self, command: str, cwd: str = ".", timeout_sec: int = 600,
    ) -> subprocess.CompletedProcess[str]:
        """Execute *command* inside the managed environment."""
        if self._use_venv_fallback:
            activate = f"source {shlex.quote(str(self._venv_path / 'bin' / 'activate'))} && {command}"
            return subprocess.run(
                ["bash", "-lc", activate],
                cwd=cwd, capture_output=True, text=True, timeout=timeout_sec,
            )
        return subprocess.run(
            [self._conda_bin, "run", "--no-capture-output", "-n", self.env_name,
             "bash", "-lc", command],
            cwd=cwd, capture_output=True, text=True, timeout=timeout_sec,
        )

    # ------------------------------------------------------------------
    # Dependency installation
    # ------------------------------------------------------------------

    def install_conda_packages(self, deps: list[CondaDependency]) -> list[dict[str, Any]]:
        """Install conda dependencies grouped by channel.  Falls back to pip when allowed."""
        if self._use_venv_fallback:
            # All go through pip
            specs = [_pip_spec(d) for d in deps]
            if specs:
                proc = self.run_in_env(f"pip install {' '.join(shlex.quote(s) for s in specs)}")
                return [{"channel": "pip", "specs": specs, "rc": proc.returncode, "log": proc.stderr[:2000]}]
            return []

        from collections import defaultdict
        by_channel: dict[str, list[tuple[str, CondaDependency]]] = defaultdict(list)
        for d in deps:
            spec = _conda_spec(d)
            by_channel[d.channel].append((spec, d))

        results: list[dict[str, Any]] = []
        for channel, items in by_channel.items():
            specs = [s for s, _ in items]
            proc = subprocess.run(
                [self._conda_bin, "install", "-n", self.env_name,
                 "-c", channel, *specs, "-y", "--quiet"],
                capture_output=True, text=True, timeout=600,
            )
            if proc.returncode == 0:
                results.append({"channel": channel, "specs": specs, "rc": 0, "log": ""})
                continue
            # Batch failed — try individually
            for spec, dep in items:
                single = subprocess.run(
                    [self._conda_bin, "install", "-n", self.env_name,
                     "-c", channel, spec, "-y", "--quiet"],
                    capture_output=True, text=True, timeout=300,
                )
                if single.returncode != 0 and dep.pip_fallback:
                    self.run_in_env(f"pip install {shlex.quote(spec)}", timeout_sec=120)
                results.append({
                    "channel": channel, "specs": [spec],
                    "rc": single.returncode, "log": single.stderr[:2000],
                })
        return results

    def install_pip_packages(self, packages: list[str]) -> subprocess.CompletedProcess[str]:
        """Install pip dependencies into the env."""
        if not packages:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        safe = " ".join(shlex.quote(p) for p in packages)
        return self.run_in_env(f"pip install {safe}", timeout_sec=600)

    def install_system_packages(self, packages: list[str]) -> subprocess.CompletedProcess[str]:
        """Best-effort system package install (apt-get on Linux, brew on macOS)."""
        if not packages:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        if shutil.which("apt-get"):
            cmd = f"sudo apt-get install -y {' '.join(shlex.quote(p) for p in packages)}"
        elif shutil.which("brew"):
            cmd = f"brew install {' '.join(shlex.quote(p) for p in packages)}"
        else:
            return subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr="No supported package manager found"
            )
        return subprocess.run(
            ["bash", "-lc", cmd], capture_output=True, text=True, timeout=300,
        )

    # ------------------------------------------------------------------
    # Snapshot & restore (fast clone-based backup)
    # ------------------------------------------------------------------

    def snapshot(self, tag: str = "") -> str | None:
        """Create a fast clone of the current env for rollback.

        Uses ``conda create --clone`` which hard-links packages (seconds, not
        minutes).  For venv fallback, uses ``pip freeze`` as a lightweight
        snapshot (restore is slower but still avoids full rebuild).

        Returns the snapshot identifier, or None on failure.
        """
        snap_name = f"{self.env_name}__snap_{tag or int(time.time())}"

        if self._use_venv_fallback:
            # venv: save a requirements.txt snapshot
            snap_file = self._venv_path.parent / f"{snap_name}.txt"
            freeze = self.freeze()
            if freeze:
                snap_file.write_text(freeze)
                self._snapshots.append(str(snap_file))
                return str(snap_file)
            return None

        proc = subprocess.run(
            [self._conda_bin, "create", "--clone", self.env_name,
             "-n", snap_name, "-y", "--quiet"],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode == 0:
            self._snapshots.append(snap_name)
            return snap_name
        logger.warning("snapshot failed: %s", proc.stderr[:500])
        return None

    def restore(self, snapshot_id: str) -> bool:
        """Restore the env from a previously created snapshot.

        Destroys the current env and recreates it from the snapshot.
        Returns True on success.
        """
        if self._use_venv_fallback:
            # venv: reinstall from requirements.txt
            snap_file = Path(snapshot_id)
            if not snap_file.exists():
                return False
            self.run_in_env(f"pip install -r {shlex.quote(str(snap_file))}", timeout_sec=600)
            return True

        # Remove current env
        subprocess.run(
            [self._conda_bin, "env", "remove", "-n", self.env_name, "-y"],
            capture_output=True, text=True, timeout=120,
        )
        # Clone snapshot back
        proc = subprocess.run(
            [self._conda_bin, "create", "--clone", snapshot_id,
             "-n", self.env_name, "-y", "--quiet"],
            capture_output=True, text=True, timeout=120,
        )
        return proc.returncode == 0

    def cleanup_snapshots(self) -> None:
        """Remove all snapshots created by this manager."""
        for snap in self._snapshots:
            if self._use_venv_fallback:
                Path(snap).unlink(missing_ok=True)
            else:
                subprocess.run(
                    [self._conda_bin, "env", "remove", "-n", snap, "-y"],
                    capture_output=True, text=True, timeout=60,
                )
        self._snapshots.clear()

    # ------------------------------------------------------------------
    # Layered dependency installation
    # ------------------------------------------------------------------

    def install_layered(self, layers: list[DepLayer]) -> list[LayerResult]:
        """Install dependencies in priority tiers with per-tier verify + rollback.

        For each layer:
          1. Snapshot the env
          2. Install conda deps (with ``--freeze-installed``) then pip deps
          3. Verify key imports
          4. On failure: restore from snapshot, record failure, continue to next layer

        Critical layers (``is_critical=True``) abort the entire install on failure.
        """
        results: list[LayerResult] = []

        for layer in layers:
            t0 = time.time()
            snap_id = self.snapshot(tag=layer.name)

            failed_pkgs: list[str] = []
            log_parts: list[str] = []

            # --- Conda deps (with freeze-installed protection) ---
            if layer.conda_deps:
                conda_results = self._install_conda_frozen(layer.conda_deps)
                for entry in conda_results:
                    log_parts.append(f"conda {entry['specs']} rc={entry['rc']}")
                    if entry["rc"] != 0:
                        failed_pkgs.extend(entry["specs"])

            # --- Pip deps ---
            if layer.pip_deps:
                proc = self.install_pip_packages(layer.pip_deps)
                log_parts.append(f"pip rc={proc.returncode}")
                if proc.returncode != 0:
                    # Try one-by-one to isolate failures
                    for pkg in layer.pip_deps:
                        single = self.install_pip_packages([pkg])
                        if single.returncode != 0:
                            failed_pkgs.append(pkg)

            # --- Verify ---
            verified = True
            if layer.verify_imports:
                verified = self.validate(key_imports=layer.verify_imports)
                log_parts.append(f"verify={'ok' if verified else 'FAIL'}")

            elapsed = time.time() - t0
            ok = verified and not failed_pkgs

            if not ok and snap_id:
                # Rollback this layer
                log_parts.append("ROLLBACK")
                self.restore(snap_id)

            results.append(LayerResult(
                layer_name=layer.name,
                ok=ok,
                failed_packages=failed_pkgs,
                log="; ".join(log_parts),
                elapsed_sec=elapsed,
            ))

            if not ok and layer.is_critical:
                logger.warning("critical layer '%s' failed, aborting install", layer.name)
                break

        # Cleanup snapshots after all layers are done
        self.cleanup_snapshots()
        return results

    def _install_conda_frozen(self, deps: list[CondaDependency]) -> list[dict[str, Any]]:
        """Install conda packages with ``--freeze-installed`` to protect core deps.

        Falls back to unconstrained install if frozen install fails.
        """
        if self._use_venv_fallback:
            return self.install_conda_packages(deps)

        from collections import defaultdict
        by_channel: dict[str, list[tuple[str, CondaDependency]]] = defaultdict(list)
        for d in deps:
            spec = _conda_spec(d)
            by_channel[d.channel].append((spec, d))

        results: list[dict[str, Any]] = []
        for channel, items in by_channel.items():
            specs = [s for s, _ in items]

            # Try with --freeze-installed first (protects torch, numpy, etc.)
            proc = subprocess.run(
                [self._conda_bin, "install", "-n", self.env_name,
                 "-c", channel, *specs, "--freeze-installed", "-y", "--quiet"],
                capture_output=True, text=True, timeout=600,
            )
            if proc.returncode == 0:
                results.append({"channel": channel, "specs": specs, "rc": 0, "log": ""})
                continue

            # Frozen failed — try without (allow solver to adjust)
            proc = subprocess.run(
                [self._conda_bin, "install", "-n", self.env_name,
                 "-c", channel, *specs, "-y", "--quiet"],
                capture_output=True, text=True, timeout=600,
            )
            if proc.returncode == 0:
                results.append({"channel": channel, "specs": specs, "rc": 0, "log": "unfrozen"})
                continue

            # Batch failed — try individually with pip fallback
            for spec, dep in items:
                single = subprocess.run(
                    [self._conda_bin, "install", "-n", self.env_name,
                     "-c", channel, spec, "-y", "--quiet"],
                    capture_output=True, text=True, timeout=300,
                )
                if single.returncode != 0 and dep.pip_fallback:
                    self.run_in_env(f"pip install {shlex.quote(spec)}", timeout_sec=120)
                results.append({
                    "channel": channel, "specs": [spec],
                    "rc": single.returncode, "log": single.stderr[:2000],
                })

        return results

    # ------------------------------------------------------------------
    # Validation & snapshot
    # ------------------------------------------------------------------

    def validate(self, key_imports: list[str] | None = None) -> bool:
        """Verify the env works and key packages are importable."""
        probe = self.run_in_env("python -c 'import sys; print(sys.version)'", timeout_sec=30)
        if probe.returncode != 0:
            return False
        for pkg in key_imports or []:
            result = self.run_in_env(f"python -c 'import {pkg}'", timeout_sec=30)
            if result.returncode != 0:
                return False
        return True

    def validate_abi(self) -> bool:
        """Check numpy / DL-framework ABI compatibility.

        A conda-forge numpy paired with a pip tensorflow/torch can cause
        ``numpy.dtype size changed`` crashes.  This runs a quick import
        probe that catches the error early.
        """
        check_script = (
            "import numpy; "
            "try:\n"
            "  import tensorflow\n"
            "except ImportError:\n"
            "  pass\n"
            "try:\n"
            "  import torch\n"
            "except ImportError:\n"
            "  pass\n"
            "print('ABI_OK')"
        )
        # Use a simpler one-liner that catches the typical ABI crash
        probe = self.run_in_env(
            "python -c '"
            "import numpy; "
            "ok=True; "
            "exec(\"try:\\n import tensorflow\\nexcept ImportError:\\n pass\\nexcept Exception as e:\\n print(e); ok=False\"); "
            "exec(\"try:\\n import torch\\nexcept ImportError:\\n pass\\nexcept Exception as e:\\n print(e); ok=False\"); "
            "print(\"ABI_OK\" if ok else \"ABI_FAIL\")"
            "'",
            timeout_sec=60,
        )
        if probe.returncode != 0:
            logger.warning("ABI validation probe failed: %s", probe.stderr[:500])
            return False
        if "ABI_FAIL" in probe.stdout or "binary incompatibility" in (probe.stderr or "").lower():
            logger.warning("NumPy ABI mismatch detected")
            return False
        return True

    def freeze(self) -> str:
        result = self.run_in_env("pip freeze", timeout_sec=30)
        return result.stdout if result.returncode == 0 else ""

    def python_version_actual(self) -> str:
        result = self.run_in_env("python -c 'import sys; print(sys.version)'", timeout_sec=15)
        return result.stdout.strip() if result.returncode == 0 else ""

    def env_path_actual(self) -> str:
        if self._use_venv_fallback:
            return str(self._venv_path)
        result = subprocess.run(
            [self._conda_bin, "info", "--envs"],
            capture_output=True, text=True, timeout=30,
        )
        for line in result.stdout.splitlines():
            if self.env_name in line and not line.strip().startswith("#"):
                parts = line.split()
                if len(parts) >= 1:
                    return parts[-1]
        return ""

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        self.cleanup_snapshots()
        if self._use_venv_fallback:
            shutil.rmtree(self._venv_path, ignore_errors=True)
            return
        subprocess.run(
            [self._conda_bin, "env", "remove", "-n", self.env_name, "-y"],
            capture_output=True, text=True, timeout=120,
        )
