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


def test_metric_extraction_credit_fraud_outputs():
    """Fraud pipeline stdout should yield precision/recall/f1 plus richer derived metrics."""
    from p2c.agents.phase2.result_extraction import extract_metrics_from_stdout
    from p2c.schemas import MetricContract

    stdout = (
        "=== XGBoost ===\n"
        "ROC-AUC: 0.9806\n"
        "PR-AUC:  0.7296\n"
        "Precision: 0.0372\n"
        "Recall:    0.9184\n"
        "F1-score:  0.0714\n"
        "\nClassification Report:\n"
        "              precision    recall  f1-score   support\n"
        "\n"
        "           0     0.9999    0.9590    0.9790     56864\n"
        "           1     0.0372    0.9184    0.0714        98\n"
        "    accuracy                         0.9589     56962\n"
        "   macro avg     0.5185    0.9387    0.5252     56962\n"
        "weighted avg     0.9982    0.9589    0.9774     56962\n"
        "\n=== Threshold sweep ===\n"
        "thr\tprecision\trecall\tf1\tTP\tFP\tFN\tTN\n"
        "0.80\t0.0639\t\t0.9082\t0.1194\t89\t1304\t9\t55560\n"
        "0.90\t0.0882\t\t0.8980\t0.1606\t88\t910\t10\t55954\n"
        "\nBEST_F1_ROW:\n"
        " threshold        0.900000\n"
        "precision        0.088176\n"
        "recall           0.897959\n"
        "f1               0.160584\n"
        "\nRecommended threshold (recall >= 0.90): 0.80\n"
        "Precision: 0.0639, Recall: 0.9082, F1: 0.1194\n"
        "\n=== Best model based on PR-AUC ===\n"
        "{'name': 'XGBoost', 'roc_auc': 0.9806250933125078, 'pr_auc': 0.7296002366909252, "
        "'precision': 0.037159372419488024, 'recall': 0.9183673469387755, 'f1': 0.07142857142857142}\n"
    )
    contract = MetricContract()
    metrics = extract_metrics_from_stdout(stdout, contract)

    assert metrics["precision"] == 0.037159372419488024
    assert metrics["recall"] == 0.9183673469387755
    assert metrics["f1"] == 0.07142857142857142
    assert metrics["roc_auc"] == 0.9806250933125078
    assert metrics["pr_auc"] == 0.7296002366909252
    assert metrics["class_1_precision"] == 0.0372
    assert metrics["recommended_threshold"] == 0.8
    assert metrics["recommended_f1"] == 0.1194
    assert metrics["best_f1_f1"] == 0.160584


def test_metric_contract_pr_auc_regex_does_not_capture_roc_auc():
    """Contract regexes should not let PR-AUC consume a later roc_auc dict value."""
    from p2c.agents.phase1.compile_task_spec import CompileTaskSpecAgent
    from p2c.agents.phase2.result_extraction import extract_metrics_from_stdout
    from p2c.schemas import MetricContract

    stdout = (
        "=== Best model based on PR-AUC ===\n"
        "{'name': 'XGBoost', 'roc_auc': 0.9806250933125078, 'pr_auc': 0.7296002366909252, "
        "'precision': 0.037159372419488024, 'recall': 0.9183673469387755, 'f1': 0.07142857142857142}\n"
    )
    contract = MetricContract(
        required_metrics=["precision", "recall", "f1"],
        parsers=CompileTaskSpecAgent._metric_parsers_for(["precision", "recall", "f1"]),
        normalization={},
    )

    metrics = extract_metrics_from_stdout(stdout, contract)

    assert metrics["pr_auc"] == 0.7296002366909252
    assert 0.9806250933125078 not in metrics.get("pr_auc_all", [])
    assert metrics["roc_auc"] == 0.9806250933125078


def test_metric_extraction_skips_static_inspection_commands():
    """Static source inspection output should not be treated as executed metrics."""
    from p2c.agents.phase2.result_extraction import extract_metrics_from_stdout
    from p2c.schemas import MetricContract

    stdout = (
        "ROC-AUC: 0.4\n"
        "PR-AUC: 0.4\n"
        "Precision: 0.4\n"
        "Recall: 0.2\n"
        "F1-score: 0.4\n"
    )
    contract = MetricContract()

    metrics = extract_metrics_from_stdout(
        stdout,
        contract,
        command="sed -n '1,260p' src/train_fraud_model.py",
    )

    assert metrics == {}


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


def test_build_child_env_adds_node_tool_dirs(monkeypatch):
    """Child env should expose resolved Node tool dirs even with a minimal PATH."""
    from p2c.runtime.conda_env import CondaEnvManager

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(
        CondaEnvManager,
        "_resolve_binary",
        staticmethod(lambda binary, explicit_env=None: {
            "node": "/tmp/base/bin/node",
            "npm": "/tmp/base/bin/npm",
        }.get(binary)),
    )

    env = CondaEnvManager._build_child_env()

    assert env["P2C_HOST_TOOL_DIRS"] == "/tmp/base/bin"
    assert env["PATH"].split(":")[0] == "/tmp/base/bin"


def test_shell_wrap_command_preserves_env_path():
    """Shell wrapper should preserve the activated env PATH and append forwarded tool dirs."""
    from p2c.runtime.conda_env import CondaEnvManager

    wrapped = CondaEnvManager._shell_wrap_command(
        {
            "PATH": "/tmp/agent/bin:/usr/bin",
            "P2C_HOST_TOOL_DIRS": "/tmp/agent/bin:/tmp/base/bin",
        },
        "python --version",
    )

    assert 'export PATH="$PATH":' in wrapped
    assert "/tmp/agent/bin:/tmp/base/bin" in wrapped
    assert wrapped.endswith("python --version")


def test_run_claude_returns_claude_result(monkeypatch):
    """_run_claude should return a ClaudeResult with stdout/stderr/returncode."""
    from p2c.agents.phase2.codex_executor import ClaudeResult, CodexExecutorAgent
    import p2c.agents.phase2.codex_executor as executor_mod

    class DummyEnvMgr:
        env_name = "test_env"

    # Mock ClaudeAgentOptions to accept keyword arguments
    class FakeOptions:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(executor_mod, "ClaudeAgentOptions", FakeOptions)

    # Mock the SDK query() to return an empty async generator
    async def fake_query(*, prompt, options=None):
        if False:
            yield  # makes this an async generator

    monkeypatch.setattr(executor_mod, "query", fake_query)

    result = CodexExecutorAgent._run_claude(
        DummyEnvMgr(), "hello world", "/tmp/repo", timeout_sec=10
    )

    assert isinstance(result, ClaudeResult)
    assert isinstance(result.stdout, str)
    assert isinstance(result.stderr, str)
    assert isinstance(result.returncode, int)
    assert result.returncode == 0  # no errors from empty session


def test_execute_step_uses_claude_as_primary(tmp_path, monkeypatch):
    """Every step should go through Claude Code as the primary executor."""
    from p2c.agents.phase2.codex_executor import ClaudeResult, CodexExecutorAgent
    from p2c.io_artifacts import ArtifactManager
    from p2c.schemas import ExecutionStep, MetricContract

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    artifacts = ArtifactManager(tmp_path / "artifacts", "run123")
    artifacts.ensure_tree()
    agent = CodexExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)

    calls = []

    class DummyEnvMgr:
        env_name = "test_env"

    def fake_claude(env_mgr, prompt, cwd, timeout_sec=600):
        calls.append(("claude", cwd, timeout_sec))
        return ClaudeResult(
            stdout="METRIC:accuracy=0.91\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(CodexExecutorAgent, "_run_claude", staticmethod(fake_claude))

    result = agent._execute_step(
        step=ExecutionStep(
            step_id="train",
            description="run training",
            command="python train.py",
            expected_metrics=["accuracy"],
        ),
        env_mgr=DummyEnvMgr(),
        repo_dir=str(repo_dir),
        contract=MetricContract(),
        outputs_dir=str(artifacts.path("execution/codex_outputs")),
        timeout_sec=60,
    )

    assert len(calls) == 1
    assert calls[0][1] == str(repo_dir)
    assert result["execution_mode"] == "claude_primary"
    assert result["exit_code"] == 0
    assert result["metrics"]["accuracy"] == 0.91
    stored = artifacts.read_json("execution/codex_outputs/step_train_result.json")
    assert stored["command"] == "python train.py"
    assert stored["exit_code"] == 0


def test_execute_step_claude_writes_step_result_exit_code(tmp_path, monkeypatch):
    """If Claude Code writes a failed step_result.json, that exit code should win over agent's own rc."""
    from p2c.agents.phase2.codex_executor import ClaudeResult, CodexExecutorAgent
    from p2c.io_artifacts import ArtifactManager
    from p2c.schemas import ExecutionStep, MetricContract

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    artifacts = ArtifactManager(tmp_path / "artifacts", "run456")
    artifacts.ensure_tree()
    agent = CodexExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)

    class DummyEnvMgr:
        env_name = "test_env"

    def fake_claude(env_mgr, prompt, cwd, timeout_sec=600):
        artifacts.write_json(
            "execution/codex_outputs/step_train_result.json",
            {
                "command": "python train.py",
                "exit_code": 1,
                "metrics": {},
                "notes": "dependency still missing",
            },
        )
        return ClaudeResult(
            stdout="claude attempted fix but still failed\n",
            stderr="ModuleNotFoundError: No module named pandas\n",
            returncode=0,
        )

    monkeypatch.setattr(CodexExecutorAgent, "_run_claude", staticmethod(fake_claude))

    result = agent._execute_step(
        step=ExecutionStep(
            step_id="train",
            description="run training",
            command="python train.py",
            expected_metrics=["accuracy"],
        ),
        env_mgr=DummyEnvMgr(),
        repo_dir=str(repo_dir),
        contract=MetricContract(),
        outputs_dir=str(artifacts.path("execution/codex_outputs")),
        timeout_sec=60,
    )

    assert result["execution_mode"] == "claude_primary"
    assert result["command"] == "python train.py"
    assert result["exit_code"] == 1
    assert result["error_type"] in {"dependency", "import"}


def test_build_run_manifest_preserves_partial_status():
    """Manifest builder should preserve executor-provided partial statuses."""
    from p2c.agents.phase2.result_extraction import build_run_manifest

    manifest = build_run_manifest(
        [
            {
                "step_id": "predict",
                "command": "test -f models/best_model.joblib",
                "cwd": ".",
                "exit_code": 0,
                "status": "partial",
                "params": {"degraded_success": True},
                "metrics": {},
                "reason_codes": ["PRIMARY_FAILED_FALLBACK_SUCCEEDED"],
            }
        ]
    )

    assert manifest.runs[0].status == "partial"
    assert manifest.runs[0].params["degraded_success"] is True


def test_execute_step_claude_primary_produces_metrics(tmp_path, monkeypatch):
    """Claude Code primary execution should produce metrics in step result."""
    from p2c.agents.phase2.codex_executor import ClaudeResult, CodexExecutorAgent
    from p2c.io_artifacts import ArtifactManager
    from p2c.schemas import ExecutionStep, MetricContract

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    artifacts = ArtifactManager(tmp_path / "artifacts", "run_partial")
    artifacts.ensure_tree()
    agent = CodexExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)

    class DummyEnvMgr:
        env_name = "test_env"

    def fake_claude(env_mgr, prompt, cwd, timeout_sec=600):
        return ClaudeResult(
            stdout="METRIC:val_accuracy=0.95\nMETRIC:train_accuracy=0.99\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(CodexExecutorAgent, "_run_claude", staticmethod(fake_claude))

    result = agent._execute_step(
        step=ExecutionStep(
            step_id="predict",
            description="run prediction validation",
            command="python src/predict_fraud.py",
        ),
        env_mgr=DummyEnvMgr(),
        repo_dir=str(repo_dir),
        contract=MetricContract(),
        outputs_dir=str(artifacts.path("execution/codex_outputs")),
        timeout_sec=60,
    )

    assert result["exit_code"] == 0
    assert result["execution_mode"] == "claude_primary"
    assert result["metrics"]["val_accuracy"] == 0.95
    assert result["metrics"]["train_accuracy"] == 0.99


def test_execute_step_claude_failure_propagates_error(tmp_path, monkeypatch):
    """Claude Code failure should propagate the error message from its output."""
    from p2c.agents.phase2.codex_executor import ClaudeResult, CodexExecutorAgent
    from p2c.io_artifacts import ArtifactManager
    from p2c.schemas import ExecutionStep, MetricContract

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    artifacts = ArtifactManager(tmp_path / "artifacts", "run789")
    artifacts.ensure_tree()
    agent = CodexExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)

    class DummyEnvMgr:
        env_name = "test_env"

    def fake_claude(env_mgr, prompt, cwd, timeout_sec=600):
        return ClaudeResult(
            stdout="Loading input file: data/sample_new_transactions.csv\n",
            stderr=(
                "Traceback (most recent call last):\n"
                "FileNotFoundError: Could not find input CSV at: data/sample_new_transactions.csv\n"
            ),
            returncode=1,
        )

    monkeypatch.setattr(CodexExecutorAgent, "_run_claude", staticmethod(fake_claude))

    result = agent._execute_step(
        step=ExecutionStep(
            step_id="predict",
            description="run prediction smoke test",
            command="python src/predict_fraud.py",
        ),
        env_mgr=DummyEnvMgr(),
        repo_dir=str(repo_dir),
        contract=MetricContract(),
        outputs_dir=str(artifacts.path("execution/codex_outputs")),
        timeout_sec=60,
    )

    assert result["exit_code"] == 1
    assert result["execution_mode"] == "claude_primary"
    assert "sample_new_transactions.csv" in result["error_message"]


def test_planner_sanitizes_help_probe_and_passive_fallbacks(tmp_path):
    """Planner sanitization should replace unsafe --help probes and drop passive fallbacks."""
    from p2c.agents.phase2.planner import PlannerAgent
    from p2c.schemas import ExecutionPlan, ExecutionStep

    repo_dir = tmp_path / "repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "train_fraud_model.py").write_text("print('train')\n", encoding="utf-8")
    (src_dir / "predict_fraud.py").write_text("print('predict')\n", encoding="utf-8")

    plan = ExecutionPlan(
        plan_id="plan",
        env_name="env",
        execution_steps=[
            ExecutionStep(
                step_id="inspect",
                description="inspect training cli",
                command="python src/train_fraud_model.py --help",
                fallback_commands=["python -c \"print('noop')\""],
            ),
            ExecutionStep(
                step_id="predict",
                description="validate prediction",
                command="python src/predict_fraud.py",
                fallback_commands=[
                    "test -f models/best_model.joblib && ls -lah models",
                    "PYTHONUNBUFFERED=1 python src/predict_fraud.py",
                ],
            ),
        ],
    )

    PlannerAgent._sanitize_plan(plan, str(repo_dir))

    assert "--help" not in plan.execution_steps[0].command
    assert "read_text" in plan.execution_steps[0].command
    assert plan.execution_steps[1].fallback_commands == ["PYTHONUNBUFFERED=1 python src/predict_fraud.py"]


def test_planner_rewrites_wrapper_derived_shell_steps(tmp_path):
    """Wrapper-derived entrypoints should keep their inferred cwd and command."""
    from p2c.agents.phase2.planner import PlannerAgent
    from p2c.schemas import ExecutionPlan, ExecutionStep

    repo_dir = tmp_path / "repo"
    workdir = repo_dir / "workdir"
    scripts_dir = repo_dir / "scripts"
    workdir.mkdir(parents=True)
    scripts_dir.mkdir(parents=True)
    (repo_dir / "run.sh").write_text("#!/usr/bin/env bash\ncd workdir\n../scripts/do.sh\n", encoding="utf-8")
    (scripts_dir / "do.sh").write_text("#!/usr/bin/env bash\npython ../scripts/tool.py\n", encoding="utf-8")
    (scripts_dir / "tool.py").write_text("print('ok')\n", encoding="utf-8")

    repo_analysis = {
        "entrypoint_candidates": [
            {
                "entrypoint_id": "shell:run.sh",
                "path": "run.sh",
                "command": "bash run.sh",
                "cwd": ".",
                "runtime": "shell",
                "confidence": 0.99,
                "evidence": "README verified shell command",
                "reason_codes": ["README_WORKFLOW_PRIMARY"],
            },
            {
                "entrypoint_id": "shell-derived:run.sh:scripts/do.sh@workdir",
                "path": "scripts/do.sh",
                "command": "bash ../scripts/do.sh",
                "cwd": "workdir",
                "runtime": "shell",
                "confidence": 0.9,
                "evidence": "derived from shell wrapper `run.sh`",
                "reason_codes": ["ENTRYPOINT_DERIVED_FROM_WRAPPER", "ENTRYPOINT_CWD_FROM_WRAPPER"],
                "path_resolution_mode": "wrapper_virtual_cwd",
                "derived_from_wrapper": "run.sh",
            },
        ]
    }
    task_spec = {
        "tasks": [
            {
                "task_id": "task_01",
                "entrypoint": "scripts/do.sh",
                "command": "bash ../scripts/do.sh",
                "cwd": "workdir",
                "runtime": "shell",
                "confidence": 0.9,
                "evidence": "derived from shell wrapper `run.sh`",
                "reason_codes": ["ENTRYPOINT_DERIVED_FROM_WRAPPER", "ENTRYPOINT_CWD_FROM_WRAPPER"],
                "path_resolution_mode": "wrapper_virtual_cwd",
                "derived_from_wrapper": "run.sh",
            }
        ]
    }

    plan = ExecutionPlan(
        plan_id="plan",
        env_name="env",
        execution_steps=[
            ExecutionStep(
                step_id="run_wrapper_child",
                description="run wrapper-derived child",
                command="bash scripts/do.sh",
                cwd=".",
                required_artifacts=["../data/input.csv", "scores/out.txt"],
            )
        ],
    )

    PlannerAgent._sanitize_plan(plan, str(repo_dir), repo_analysis=repo_analysis, task_spec=task_spec)

    step = plan.execution_steps[0]
    assert step.command == "bash ../scripts/do.sh"
    assert step.cwd == "workdir"
    assert step.path_resolution_mode == "wrapper_virtual_cwd"
    assert step.derived_from_wrapper == "run.sh"
    assert step.required_artifacts == ["data/input.csv", "workdir/scores/out.txt"]


def test_derive_key_imports_maps_imblearn():
    """Environment validation should import imbalanced-learn via `imblearn`."""
    from p2c.agents.phase2.tool_agent import ToolAgent
    from p2c.schemas import ExecutionPlan

    plan = ExecutionPlan(
        plan_id="plan",
        env_name="env",
        execution_steps=[],
        pip_dependencies=["imbalanced-learn", "scikit-learn"],
    )

    imports = ToolAgent._derive_key_imports(plan)

    assert "imblearn" in imports
    assert "sklearn" in imports


def test_tool_agent_augments_missing_requirements_dependency(tmp_path):
    """Repo requirements should backfill planner omissions like passage."""
    from p2c.agents.phase2.tool_agent import ToolAgent
    from p2c.schemas import ExecutionPlan

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "requirements.txt").write_text(
        "numpy>=1.8.2\npandas>=0.16.0\nTheano>=0.7\npassage>=0.2.4\n",
        encoding="utf-8",
    )

    plan = ExecutionPlan(
        plan_id="plan",
        env_name="env",
        execution_steps=[],
        pip_dependencies=["numpy==1.23.5", "pandas==1.5.3", "Theano-PyMC==1.1.2"],
    )

    ToolAgent._augment_plan_dependencies(plan, repo_dir)

    assert "passage>=0.2.4" in plan.pip_dependencies
    assert "Theano>=0.7" not in plan.pip_dependencies


def test_tool_agent_runs_preinstall_in_repo_dir(tmp_path, monkeypatch):
    """Pre-install commands should resolve paths relative to repo_dir, not project cwd."""
    from p2c.agents.phase2.tool_agent import ToolAgent
    from p2c.schemas import ExecutionPlan

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    artifacts_dir = tmp_path / "artifacts"
    from p2c.io_artifacts import ArtifactManager

    artifacts = ArtifactManager(artifacts_dir, "run_tool")
    artifacts.ensure_tree()

    calls = {"pre": []}

    class DummyEnvMgr:
        backend = "dummy"
        def __init__(self, *args, **kwargs):
            self.env_name = kwargs.get("env_name", "env")
            self.python_version = kwargs.get("python_version", "3.10")
        def create(self):
            return {"ok": True, "log": ""}
        def env_path_actual(self):
            return "/tmp/env"
        def run_in_env(self, command, cwd=".", timeout_sec=600):
            import subprocess
            calls["pre"].append((command, cwd, timeout_sec))
            if command == "pip freeze":
                return subprocess.CompletedProcess(args=[command], returncode=0, stdout="", stderr="")
            return subprocess.CompletedProcess(args=[command], returncode=0, stdout="", stderr="")
        def install_layered(self, layers):
            return []
        def validate(self, key_imports=None):
            return True
        def freeze(self):
            return ""

    monkeypatch.setattr("p2c.agents.phase2.tool_agent.CondaEnvManager", DummyEnvMgr)

    plan = ExecutionPlan(
        plan_id="plan",
        env_name="env",
        execution_steps=[],
        pre_install_commands=["python - <<'PY'\nprint('ok')\nPY"],
    )

    agent = ToolAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    agent.execute({"_p2_plan": plan, "repo_dir": str(repo_dir)})

    assert calls["pre"][0][1] == str(repo_dir)


def test_observe_metrics_reads_step_stdout_logs(tmp_path, monkeypatch):
    """Phase 3 metrics observation should recover fraud metrics from full step stdout logs."""
    from p2c.agents.phase3.observe_metrics import ObserveMetricsAgent
    from p2c.io_artifacts import ArtifactManager

    artifacts = ArtifactManager(tmp_path / "artifacts", "run_metrics")
    artifacts.ensure_tree()
    artifacts.write_json(
        "task/metric_contract.json",
        {
            "required_metrics": ["precision", "recall", "f1"],
            "parsers": [],
            "normalization": {},
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "execution/codex_outputs/run_manifest.json",
        {
            "runs": [
                {
                    "run_id": "step_02_train_model",
                    "command": "python src/train_fraud_model.py",
                    "params": {},
                    "cwd": ".",
                    "exit_code": 0,
                    "status": "ok",
                    "runtime_sec": 1.0,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "metrics": {},
                    "reason_codes": [],
                }
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_text(
        "execution/codex_outputs/step_step_02_train_model_stdout.log",
        "Precision: 0.0372\nRecall: 0.9184\nF1-score: 0.0714\n",
    )

    agent = ObserveMetricsAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    monkeypatch.setattr(agent, "safe_chat_text", lambda system, user: (None, None))

    result = agent.execute({})
    records = result["metrics"]["records"]
    metric_names = {row["metric_name"] for row in records}

    assert "precision" in metric_names
    assert "recall" in metric_names
    assert "f1" in metric_names
    assert all(row["metric_name"] != "unknown" for row in records)


def test_observe_metrics_ignores_static_inspection_steps(tmp_path, monkeypatch):
    """Phase 3 should ignore polluted inspect-step metrics from static source reads."""
    from p2c.agents.phase3.observe_metrics import ObserveMetricsAgent
    from p2c.io_artifacts import ArtifactManager

    artifacts = ArtifactManager(tmp_path / "artifacts", "run_inspect")
    artifacts.ensure_tree()
    artifacts.write_json(
        "task/metric_contract.json",
        {
            "required_metrics": ["precision", "recall", "f1"],
            "parsers": [],
            "normalization": {},
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "execution/codex_outputs/run_manifest.json",
        {
            "runs": [
                {
                    "run_id": "step_01_inspect_train_script",
                    "command": "sed -n '1,260p' src/train_fraud_model.py",
                    "params": {},
                    "cwd": ".",
                    "exit_code": 0,
                    "status": "ok",
                    "runtime_sec": 0.1,
                    "stdout_tail": "Precision: 0.4\nRecall: 0.2\nF1-score: 0.4\n",
                    "stderr_tail": "",
                    "metrics": {"precision": 0.4, "recall": 0.2, "f1": 0.4},
                    "reason_codes": [],
                },
                {
                    "run_id": "step_02_train_model",
                    "command": "python src/train_fraud_model.py",
                    "params": {},
                    "cwd": ".",
                    "exit_code": 0,
                    "status": "ok",
                    "runtime_sec": 1.0,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "metrics": {"precision": 0.0372},
                    "reason_codes": [],
                },
            ],
            "reason_codes": [],
        },
    )
    artifacts.write_text(
        "execution/codex_outputs/step_step_01_inspect_train_script_stdout.log",
        "Precision: 0.4\nRecall: 0.2\nF1-score: 0.4\n",
    )
    artifacts.write_text(
        "execution/codex_outputs/step_step_02_train_model_stdout.log",
        "Precision: 0.0372\nRecall: 0.9184\nF1-score: 0.0714\n",
    )

    agent = ObserveMetricsAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    monkeypatch.setattr(agent, "safe_chat_text", lambda system, user: (None, None))

    result = agent.execute({})
    records = result["metrics"]["records"]
    values_by_metric = {}
    for row in records:
        values_by_metric.setdefault(row["metric_name"], set()).add(row["value"])

    assert values_by_metric["precision"] == {0.0372}
    assert values_by_metric["recall"] == {0.9184}
    assert values_by_metric["f1"] == {0.0714}


def test_execute_step_static_inspection_does_not_emit_metrics(tmp_path, monkeypatch):
    """Claude Code primary execution for inspect-like commands still succeeds."""
    from p2c.agents.phase2.codex_executor import CodexExecutorAgent
    from p2c.io_artifacts import ArtifactManager
    from p2c.schemas import ExecutionStep, MetricContract

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    artifacts = ArtifactManager(tmp_path / "artifacts", "run_inspect_exec")
    artifacts.ensure_tree()
    agent = CodexExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)

    class DummyEnvMgr:
        env_name = "test_env"

    from p2c.agents.phase2.codex_executor import ClaudeResult

    def fake_claude(env_mgr, prompt, cwd, timeout_sec=600):
        return ClaudeResult(
            stdout="Precision: 0.4\nRecall: 0.2\nF1-score: 0.4\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(CodexExecutorAgent, "_run_claude", staticmethod(fake_claude))

    result = agent._execute_step(
        step=ExecutionStep(
            step_id="inspect_train",
            description="inspect training script",
            command="sed -n '1,260p' src/train_fraud_model.py",
            retry_on_failure=False,
        ),
        env_mgr=DummyEnvMgr(),
        repo_dir=str(repo_dir),
        contract=MetricContract(),
        outputs_dir=str(artifacts.path("execution/codex_outputs")),
        timeout_sec=60,
    )

    assert result["exit_code"] == 0


def test_execute_step_shell_false_success_is_forced_failed(tmp_path, monkeypatch):
    """Shell wrappers with path errors should not be marked successful on rc=0 alone."""
    from p2c.agents.phase2.codex_executor import CodexExecutorAgent, ClaudeResult
    from p2c.io_artifacts import ArtifactManager
    from p2c.schemas import ExecutionStep, MetricContract

    repo_dir = tmp_path / "repo"
    scripts_dir = repo_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "run.sh").write_text("#!/usr/bin/env bash\ncd missing_dir\npython train.py\n", encoding="utf-8")

    artifacts = ArtifactManager(tmp_path / "artifacts", "run_shell_false_success")
    artifacts.ensure_tree()
    agent = CodexExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)

    class DummyEnvMgr:
        env_name = "test_env"

    def fake_claude(env_mgr, prompt, cwd, timeout_sec=600):
        return ClaudeResult(
            stdout="",
            stderr="scripts/run.sh: line 2: cd: missing_dir: No such file or directory\n",
            returncode=0,
        )

    monkeypatch.setattr(CodexExecutorAgent, "_run_claude", staticmethod(fake_claude))

    result = agent._execute_step(
        step=ExecutionStep(
            step_id="wrapper",
            description="run shell wrapper",
            command="bash scripts/run.sh",
            produced_artifacts=["models/best_model.joblib"],
            path_resolution_mode="repo_root",
        ),
        env_mgr=DummyEnvMgr(),
        repo_dir=str(repo_dir),
        contract=MetricContract(),
        outputs_dir=str(artifacts.path("execution/codex_outputs")),
        timeout_sec=60,
    )

    assert result["exit_code"] == 1
    assert result["status"] == "failed"
    assert result["params"]["effective_cwd"] == "."
    assert result["params"]["path_resolution_mode"] == "repo_root"
    assert "SHELL_WRAPPER_FALSE_SUCCESS" in result["reason_codes"]


def test_execute_step_claude_failure_returns_error(tmp_path, monkeypatch):
    """Claude Code failure (non-zero exit) propagates as failed step result."""
    from p2c.agents.phase2.codex_executor import CodexExecutorAgent, ClaudeResult
    from p2c.io_artifacts import ArtifactManager
    from p2c.schemas import ExecutionStep, MetricContract

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    artifacts = ArtifactManager(tmp_path / "artifacts", "run_claude_fail")
    artifacts.ensure_tree()
    agent = CodexExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)

    class DummyEnvMgr:
        env_name = "test_env"

    def fake_claude(env_mgr, prompt, cwd, timeout_sec=600):
        return ClaudeResult(
            stdout="",
            stderr="SyntaxError: Missing parentheses in call to 'print'\n",
            returncode=1,
        )

    monkeypatch.setattr(CodexExecutorAgent, "_run_claude", staticmethod(fake_claude))

    result = agent._execute_step(
        step=ExecutionStep(
            step_id="pipe",
            description="run pipeline with tee",
            command="PYTHONPATH=.. python nbsvm.py --ngram 1 --out ../scores/NBSVM-VALID-1GRAM | tee ../scores/NBSVM-VALID-1GRAM.log",
            retry_on_failure=False,
        ),
        env_mgr=DummyEnvMgr(),
        repo_dir=str(repo_dir),
        contract=MetricContract(),
        outputs_dir=str(artifacts.path("execution/codex_outputs")),
        timeout_sec=60,
    )

    assert result["exit_code"] == 1
    assert result["status"] == "failed"
