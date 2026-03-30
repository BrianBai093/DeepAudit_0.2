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
