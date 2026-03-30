"""Tests for Phase 2 stability fixes: conda spec, metric extraction, env forwarding."""

from __future__ import annotations


# ---- Fix 1: conda/pip spec construction ----

def test_conda_spec_bare_version():
    """version_constraint='3.10' should produce 'python=3.10', not 'python3.10'."""
    from p2c.runtime.conda_env import _conda_spec
    from p2c.schemas import CondaDependency

    dep = CondaDependency(package="python", version_constraint="3.10")
    assert _conda_spec(dep) == "python=3.10"


def test_conda_spec_with_operator():
    """version_constraint='>=2.12' should pass through as-is."""
    from p2c.runtime.conda_env import _conda_spec
    from p2c.schemas import CondaDependency

    dep = CondaDependency(package="tensorflow", version_constraint=">=2.12,<2.16")
    assert _conda_spec(dep) == "tensorflow>=2.12,<2.16"


def test_conda_spec_no_version():
    """No version constraint should return just the package name."""
    from p2c.runtime.conda_env import _conda_spec
    from p2c.schemas import CondaDependency

    dep = CondaDependency(package="numpy")
    assert _conda_spec(dep) == "numpy"


def test_pip_spec_bare_version():
    """Bare version '1.26.4' should become '==' for pip."""
    from p2c.runtime.conda_env import _pip_spec
    from p2c.schemas import CondaDependency

    dep = CondaDependency(package="numpy", version_constraint="1.26.4")
    assert _pip_spec(dep) == "numpy==1.26.4"


def test_pip_spec_with_operator():
    from p2c.runtime.conda_env import _pip_spec
    from p2c.schemas import CondaDependency

    dep = CondaDependency(package="keras", version_constraint=">=2.12")
    assert _pip_spec(dep) == "keras>=2.12"


# ---- Fix 2: skip python/pip from layer partitioning ----

def test_build_layers_skips_python_package():
    """python and pip should NOT appear in any install layer."""
    from p2c.agents.phase2.tool_agent import ToolAgent
    from p2c.schemas import CondaDependency, ExecutionPlan

    plan = ExecutionPlan(
        plan_id="test",
        env_name="test_env",
        execution_steps=[],
        conda_dependencies=[
            CondaDependency(package="python", version_constraint="3.10"),
            CondaDependency(package="pip"),
        ],
        pip_dependencies=["numpy", "pandas"],
    )
    layers = ToolAgent._build_layers(plan)

    all_conda_pkgs = []
    for layer in layers:
        for dep in layer.conda_deps:
            all_conda_pkgs.append(dep.package.lower())

    assert "python" not in all_conda_pkgs, "python should be skipped — already in env"
    assert "pip" not in all_conda_pkgs, "pip should be skipped — already in env"


# ---- Fix 3: val/train metric distinction ----

def test_metric_extraction_prefixed():
    """val_accuracy and train_accuracy should be extracted separately."""
    from p2c.agents.phase2.result_extraction import extract_metrics_from_stdout
    from p2c.schemas import MetricContract

    stdout = (
        "Epoch 27/27\n"
        "train accuracy: 0.9972\n"
        "val accuracy: 0.9882\n"
        "METRIC:accuracy=0.9882\n"
        "METRIC:val_accuracy=0.9882\n"
        "METRIC:train_accuracy=0.9972\n"
    )
    contract = MetricContract()
    metrics = extract_metrics_from_stdout(stdout, contract)

    assert metrics.get("val_accuracy") == 0.9882
    assert metrics.get("train_accuracy") == 0.9972
    # Unprefixed accuracy should be the val (last METRIC line or val extraction)
    assert metrics.get("accuracy") == 0.9882


def test_metric_extraction_unprefixed_prefers_val():
    """When only Layer 3 patterns exist, unprefixed should come from val/test, not train."""
    from p2c.agents.phase2.result_extraction import extract_metrics_from_stdout
    from p2c.schemas import MetricContract

    stdout = "train accuracy: 0.99\nval accuracy: 0.95\ntest accuracy: 0.93\n"
    contract = MetricContract()
    metrics = extract_metrics_from_stdout(stdout, contract)

    # val sets the unprefixed name first
    assert metrics.get("val_accuracy") == 0.95
    assert metrics.get("test_accuracy") == 0.93
    assert metrics.get("train_accuracy") == 0.99
    # unprefixed should NOT be 0.99 (train)
    assert metrics.get("accuracy") != 0.99


# ---- Fix 4: env variable forwarding ----

def test_build_child_env_includes_path():
    """_build_child_env should include host PATH."""
    import os
    from p2c.runtime.conda_env import CondaEnvManager

    env = CondaEnvManager._build_child_env()
    assert "PATH" in env
    assert env["PATH"] == os.environ.get("PATH", "")


def test_run_in_env_uses_non_login_bash_for_conda(monkeypatch):
    """run_in_env should avoid ``bash -lc`` and skip ``--no-capture-output`` for mamba."""
    import subprocess
    from p2c.runtime.conda_env import CondaEnvManager

    calls = {}

    def fake_run(args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    mgr = CondaEnvManager(env_name="dummy", python_version="3.10")
    mgr._conda_bin = "mamba"
    mgr._use_venv_fallback = False
    mgr.run_in_env("python -V", cwd="/tmp", timeout_sec=12)

    assert calls["args"][:3] == ["mamba", "run", "-n"]
    assert calls["args"][-2] == "-c"
    assert calls["args"][-1].endswith("python -V")
    assert "-lc" not in calls["args"]
    assert "--no-capture-output" not in calls["args"]


def test_run_in_env_keeps_no_capture_output_for_conda(monkeypatch):
    """conda run still benefits from ``--no-capture-output`` for better streaming behavior."""
    import subprocess
    from p2c.runtime.conda_env import CondaEnvManager

    calls = {}

    def fake_run(args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    mgr = CondaEnvManager(env_name="dummy", python_version="3.10")
    mgr._conda_bin = "conda"
    mgr._use_venv_fallback = False
    mgr.run_in_env("python -V", cwd="/tmp", timeout_sec=12)

    assert calls["args"][:4] == ["conda", "run", "--no-capture-output", "-n"]
    assert calls["args"][-2] == "-c"
    assert calls["args"][-1].endswith("python -V")


def test_run_in_env_uses_non_login_bash_for_venv(monkeypatch):
    """The venv fallback should also avoid relying on login-shell startup files."""
    import subprocess
    from pathlib import Path
    from p2c.runtime.conda_env import CondaEnvManager

    calls = {}

    def fake_run(args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    mgr = CondaEnvManager(env_name="dummy", python_version="3.10")
    mgr._use_venv_fallback = True
    mgr._venv_path = Path("/tmp/p2c_venv_dummy")
    mgr.run_in_env("python -V", cwd="/tmp", timeout_sec=12)

    assert calls["args"][0:2] == ["bash", "-c"]
    assert calls["args"][2].endswith("source /tmp/p2c_venv_dummy/bin/activate && python -V")
    assert "-lc" not in calls["args"]


def test_build_child_env_adds_resolved_codex_bin(monkeypatch):
    """Child env should expose resolved Codex and Node tool dirs even with a minimal PATH."""
    from p2c.runtime.conda_env import CondaEnvManager

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.delenv("P2C_CODEX_BIN", raising=False)
    monkeypatch.setattr(
        CondaEnvManager,
        "_resolve_binary",
        staticmethod(lambda binary, explicit_env=None: {
            "codex": "/tmp/agent/bin/codex",
            "node": "/tmp/base/bin/node",
            "npm": "/tmp/base/bin/npm",
        }.get(binary)),
    )

    env = CondaEnvManager._build_child_env()

    assert env["P2C_CODEX_BIN"] == "/tmp/agent/bin/codex"
    assert env["PATH"].split(":")[0] == "/tmp/agent/bin"
    assert env["PATH"].split(":")[1] == "/tmp/base/bin"


def test_shell_wrap_command_re_exports_forwarded_vars():
    """PATH and resolved Codex path should be exported in the final shell command."""
    from p2c.runtime.conda_env import CondaEnvManager

    wrapped = CondaEnvManager._shell_wrap_command(
        {"PATH": "/tmp/agent/bin:/usr/bin", "P2C_CODEX_BIN": "/tmp/agent/bin/codex"},
        "codex --version",
    )

    assert "export PATH=/tmp/agent/bin:/usr/bin" in wrapped
    assert "export P2C_CODEX_BIN=/tmp/agent/bin/codex" in wrapped
    assert wrapped.endswith("codex --version")


def test_run_codex_uses_resolved_absolute_binary(monkeypatch):
    """Codex executor should not rely on PATH-only lookup inside transient envs."""
    from p2c.agents.phase2.codex_executor import CodexExecutorAgent
    from p2c.runtime.conda_env import CondaEnvManager

    recorded = {}

    class DummyEnvMgr:
        def run_in_env(self, command, cwd=".", timeout_sec=600):
            recorded["command"] = command
            recorded["cwd"] = cwd
            recorded["timeout_sec"] = timeout_sec
            return None

    monkeypatch.delenv("P2C_CODEX_BIN", raising=False)
    monkeypatch.setattr(CondaEnvManager, "_resolve_codex_bin", staticmethod(lambda: "/tmp/agent/bin/codex"))

    CodexExecutorAgent._run_codex(DummyEnvMgr(), "hello world", "/tmp/repo", timeout_sec=42)

    assert recorded["command"].startswith("/tmp/agent/bin/codex exec --full-auto")
    assert recorded["cwd"] == "/tmp/repo"
    assert recorded["timeout_sec"] == 42
