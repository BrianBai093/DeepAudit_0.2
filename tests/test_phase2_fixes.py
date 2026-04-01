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
    assert env["P2C_HOST_TOOL_DIRS"] == "/tmp/agent/bin:/tmp/base/bin"
    assert env["PATH"].split(":")[0] == "/tmp/agent/bin"
    assert env["PATH"].split(":")[1] == "/tmp/base/bin"


def test_shell_wrap_command_preserves_env_path_and_exports_codex():
    """Shell wrapper should preserve the activated env PATH and append forwarded tool dirs."""
    from p2c.runtime.conda_env import CondaEnvManager

    wrapped = CondaEnvManager._shell_wrap_command(
        {
            "PATH": "/tmp/agent/bin:/usr/bin",
            "P2C_HOST_TOOL_DIRS": "/tmp/agent/bin:/tmp/base/bin",
            "P2C_CODEX_BIN": "/tmp/agent/bin/codex",
        },
        "codex --version",
    )

    assert 'export PATH="$PATH":' in wrapped
    assert "/tmp/agent/bin:/tmp/base/bin" in wrapped
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


def test_execute_step_prefers_direct_env_execution(tmp_path, monkeypatch):
    """Planned commands should run directly inside env_mgr before any Codex recovery."""
    import subprocess
    from p2c.agents.phase2.codex_executor import CodexExecutorAgent
    from p2c.io_artifacts import ArtifactManager
    from p2c.schemas import ExecutionStep, MetricContract

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    artifacts = ArtifactManager(tmp_path / "artifacts", "run123")
    artifacts.ensure_tree()
    agent = CodexExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)

    calls = []

    class DummyEnvMgr:
        def run_in_env(self, command, cwd=".", timeout_sec=600):
            calls.append((command, cwd, timeout_sec))
            return subprocess.CompletedProcess(
                args=[command],
                returncode=0,
                stdout="METRIC:accuracy=0.91\n",
                stderr="",
            )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Codex recovery should not run when direct execution succeeds")

    monkeypatch.setattr(CodexExecutorAgent, "_run_codex", staticmethod(fail_if_called))

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

    assert calls == [("python train.py", str(repo_dir), 60)]
    assert result["execution_mode"] == "direct"
    assert result["exit_code"] == 0
    assert result["metrics"]["accuracy"] == 0.91
    stored = artifacts.read_json("execution/codex_outputs/step_train_result.json")
    assert stored["command"] == "python train.py"
    assert stored["exit_code"] == 0


def test_execute_step_codex_recovery_uses_step_result_exit_code(tmp_path, monkeypatch):
    """If Codex writes a failed step_result.json, that exit code should win over Codex's own rc."""
    import subprocess
    from p2c.agents.phase2.codex_executor import CodexExecutorAgent
    from p2c.io_artifacts import ArtifactManager
    from p2c.schemas import ExecutionStep, MetricContract

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    artifacts = ArtifactManager(tmp_path / "artifacts", "run456")
    artifacts.ensure_tree()
    agent = CodexExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)

    class DummyEnvMgr:
        def run_in_env(self, command, cwd=".", timeout_sec=600):
            return subprocess.CompletedProcess(
                args=[command],
                returncode=1,
                stdout="",
                stderr="ModuleNotFoundError: No module named pandas\n",
            )

    def fake_codex(env_mgr, prompt, cwd, timeout_sec=600):
        artifacts.write_json(
            "execution/codex_outputs/step_train_result.json",
            {
                "command": "python train.py",
                "exit_code": 1,
                "metrics": {},
                "notes": "dependency still missing",
            },
        )
        return subprocess.CompletedProcess(
            args=["codex"],
            returncode=0,
            stdout="codex attempted recovery\n",
            stderr="",
        )

    monkeypatch.setattr(CodexExecutorAgent, "_run_codex", staticmethod(fake_codex))

    result = agent._execute_step(
        step=ExecutionStep(
            step_id="train",
            description="run training",
            command="python train.py",
            expected_metrics=["accuracy"],
            retry_on_failure=True,
        ),
        env_mgr=DummyEnvMgr(),
        repo_dir=str(repo_dir),
        contract=MetricContract(),
        outputs_dir=str(artifacts.path("execution/codex_outputs")),
        timeout_sec=60,
    )

    assert result["execution_mode"] == "codex_recovery"
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


def test_execute_step_marks_non_equivalent_fallback_as_partial(tmp_path):
    """Fallback artifact checks should not erase a failed primary script execution."""
    import subprocess
    from p2c.agents.phase2.codex_executor import CodexExecutorAgent
    from p2c.io_artifacts import ArtifactManager
    from p2c.schemas import ExecutionStep, MetricContract

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    artifacts = ArtifactManager(tmp_path / "artifacts", "run_partial")
    artifacts.ensure_tree()
    agent = CodexExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)

    attempts = [
        subprocess.CompletedProcess(
            args=["python src/predict_fraud.py"],
            returncode=1,
            stdout="Loading input file: data/sample_new_transactions.csv\n",
            stderr="FileNotFoundError: Could not find input CSV at: data/sample_new_transactions.csv\n",
        ),
        subprocess.CompletedProcess(
            args=["test -f models/best_model.joblib && ls -lah models"],
            returncode=0,
            stdout="best_model.joblib\n",
            stderr="",
        ),
    ]

    class DummyEnvMgr:
        def run_in_env(self, command, cwd=".", timeout_sec=600):
            return attempts.pop(0)

    result = agent._execute_step(
        step=ExecutionStep(
            step_id="predict",
            description="run prediction validation",
            command="python src/predict_fraud.py",
            fallback_commands=["test -f models/best_model.joblib && ls -lah models"],
            retry_on_failure=False,
        ),
        env_mgr=DummyEnvMgr(),
        repo_dir=str(repo_dir),
        contract=MetricContract(),
        outputs_dir=str(artifacts.path("execution/codex_outputs")),
        timeout_sec=60,
    )

    assert result["exit_code"] == 0
    assert result["status"] == "partial"
    assert result["params"]["planned_command"] == "python src/predict_fraud.py"
    assert result["params"]["primary_exit_code"] == 1
    assert result["params"]["degraded_success"] is True
    assert "NON_EQUIVALENT_FALLBACK" in result["reason_codes"]


def test_execute_step_prefers_direct_business_failure_over_codex_infra(tmp_path, monkeypatch):
    """Final failure should reflect the original business error, not a later Codex infrastructure failure."""
    import subprocess
    from p2c.agents.phase2.codex_executor import CodexExecutorAgent
    from p2c.io_artifacts import ArtifactManager
    from p2c.schemas import ExecutionStep, MetricContract

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    artifacts = ArtifactManager(tmp_path / "artifacts", "run789")
    artifacts.ensure_tree()
    agent = CodexExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)

    attempts = [
        subprocess.CompletedProcess(
            args=["python src/predict_fraud.py"],
            returncode=1,
            stdout="Loading input file: data/sample_new_transactions.csv\n",
            stderr=(
                "Traceback (most recent call last):\n"
                "FileNotFoundError: Could not find input CSV at: data/sample_new_transactions.csv\n"
            ),
        ),
        subprocess.CompletedProcess(
            args=["PYTHONPATH=. python src/predict_fraud.py"],
            returncode=1,
            stdout="Loading input file: data/sample_new_transactions.csv\n",
            stderr=(
                "Traceback (most recent call last):\n"
                "FileNotFoundError: Could not find input CSV at: data/sample_new_transactions.csv\n"
            ),
        ),
    ]

    class DummyEnvMgr:
        def run_in_env(self, command, cwd=".", timeout_sec=600):
            return attempts.pop(0)

    def fake_codex(env_mgr, prompt, cwd, timeout_sec=600):
        return subprocess.CompletedProcess(
            args=["codex"],
            returncode=127,
            stdout="",
            stderr="env: 'node': No such file or directory\n",
        )

    monkeypatch.setattr(CodexExecutorAgent, "_run_codex", staticmethod(fake_codex))

    result = agent._execute_step(
        step=ExecutionStep(
            step_id="predict",
            description="run prediction smoke test",
            command="python src/predict_fraud.py",
            fallback_commands=["PYTHONPATH=. python src/predict_fraud.py"],
            retry_on_failure=True,
        ),
        env_mgr=DummyEnvMgr(),
        repo_dir=str(repo_dir),
        contract=MetricContract(),
        outputs_dir=str(artifacts.path("execution/codex_outputs")),
        timeout_sec=60,
    )

    assert result["exit_code"] == 1
    assert "sample_new_transactions.csv" in result["error_message"]
    assert "node" not in result["error_message"]


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
