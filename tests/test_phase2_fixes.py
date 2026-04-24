from __future__ import annotations

from collections.abc import AsyncIterable
from pathlib import Path

from p2c.agents.phase2.executor_agent import ExecutorAgent, ExecutorSessionResult
from p2c.agents.phase2.tool_agent import ToolAgent
from p2c.io_artifacts import ArtifactManager
from p2c.runtime.conda_env import CondaEnvManager
from p2c.schemas import MetricContract


class DummyEnvMgr:
    env_name = "test_env"
    backend = "venv"
    _use_venv_fallback = True
    _conda_bin = None

    @staticmethod
    def env_path_actual() -> str:
        return "/tmp/p2c_venv_test_env"


def make_artifacts(tmp_path: Path, run_id: str) -> ArtifactManager:
    artifacts = ArtifactManager(tmp_path / "artifacts", run_id)
    artifacts.ensure_tree()
    return artifacts


def test_tool_agent_builds_env_spec_from_repo_manifests_only(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "requirements.txt").write_text("numpy==1.26.4\npandas\n", encoding="utf-8")

    artifacts = make_artifacts(tmp_path, "env_spec")
    artifacts.write_json(
        "task/repo_analysis.json",
        {
            "dependency_profiles": [
                {
                    "profile_id": "python-requirements:requirements.txt",
                    "ecosystem": "python",
                    "manager": "pip_requirements",
                    "cwd": ".",
                    "manifest_paths": ["requirements.txt"],
                    "install_command": "python -m pip install -r requirements.txt",
                    "auto_bootstrap_supported": True,
                    "reason_codes": [],
                }
            ],
            "entrypoint_candidates": [],
            "primary_entrypoint_id": None,
            "reason_codes": [],
        },
    )
    artifacts.write_json(
        "fingerprint/claims_ir.json",
        {
            "experiments": [
                {
                    "experiment_id": "exp_01",
                    "name": "run",
                    "description": "",
                    "dataset": "CIFAR10",
                    "table_anchor": "Table 1",
                    "primary_metrics": ["accuracy"],
                    "is_primary": True,
                    "notes": None,
                }
            ],
            "claims": [
                {
                    "claim_id": "claim_01",
                    "type": "result",
                    "predicate": "accuracy = 0.9",
                    "metric": "accuracy",
                    "target": 0.9,
                    "conditions": {"experiment_id": "exp_01"},
                }
            ],
            "reason_codes": [],
        },
    )

    agent = ToolAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    env_spec = agent.build_env_spec({"repo_dir": str(repo_dir), "run_id": "env_spec"})

    assert env_spec.pip_dependencies == ["numpy==1.26.4", "pandas"]
    assert "accuracy" not in " ".join(env_spec.pip_dependencies)
    assert env_spec.env_name == "env_spec_executor"


def test_executor_agent_detects_source_mutation(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("python train.py\n", encoding="utf-8")
    tracked_file = repo_dir / "train.py"
    tracked_file.write_text("print('before')\n", encoding="utf-8")

    artifacts = make_artifacts(tmp_path, "source_guard")
    artifacts.write_json(
        "fingerprint/claims_ir.json",
        {
            "experiments": [
                {
                    "experiment_id": "exp_01",
                    "name": "mutating run",
                    "description": "",
                    "dataset": None,
                    "table_anchor": None,
                    "primary_metrics": ["accuracy"],
                    "is_primary": True,
                    "notes": None,
                }
            ],
            "claims": [],
            "reason_codes": [],
        },
    )
    artifacts.write_json("task/repo_analysis.json", {"dependency_profiles": [], "entrypoint_candidates": [], "reason_codes": []})
    artifacts.write_json("task/metric_contract.json", {"required_metrics": ["accuracy"], "parsers": [], "normalization": {}, "reason_codes": []})

    def fake_session(env_mgr, prompt, cwd, timeout_sec=600):
        tracked_file.write_text("print('after')\n", encoding="utf-8")
        artifacts.write_json("execution/executor_outputs/executor_results.json", {"runs": []})
        return ExecutorSessionResult(stdout="", stderr="", returncode=0, narrative="")

    monkeypatch.setattr(ExecutorAgent, "_run_executor_session", staticmethod(fake_session))

    agent = ExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    result = agent.execute(
        {
            "repo_dir": str(repo_dir),
            "_p2_env_mgr": DummyEnvMgr(),
            "_p2_remaining_sec": 60,
            "_p2_attempt": 1,
        }
    )

    assert result["success"] is False
    failure = result["failure"]
    assert failure.reason_codes == ["SOURCE_MUTATION_DETECTED"]


def test_executor_agent_writes_experiment_scoped_manifest_and_logs(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("python train.py\n", encoding="utf-8")
    (repo_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")

    artifacts = make_artifacts(tmp_path, "manifest_logs")
    artifacts.write_json(
        "fingerprint/claims_ir.json",
        {
            "experiments": [
                {
                    "experiment_id": "exp_01",
                    "name": "fc table 1",
                    "description": "run fc model",
                    "dataset": "MNIST",
                    "table_anchor": "Table 1",
                    "primary_metrics": ["accuracy"],
                    "is_primary": True,
                    "notes": None,
                }
            ],
            "claims": [],
            "reason_codes": [],
        },
    )
    artifacts.write_json("task/repo_analysis.json", {"dependency_profiles": [], "entrypoint_candidates": [], "reason_codes": []})
    artifacts.write_json("task/metric_contract.json", {"required_metrics": ["accuracy"], "parsers": [], "normalization": {}, "reason_codes": []})

    def fake_session(env_mgr, prompt, cwd, timeout_sec=600):
        artifacts.write_text("execution/executor_outputs/experiment_exp_01_stdout.log", "METRIC:accuracy=0.97\n")
        artifacts.write_text("execution/executor_outputs/experiment_exp_01_stderr.log", "")
        artifacts.write_text("execution/executor_outputs/experiment_exp_01_narrative.log", "ran train.py\n")
        artifacts.write_text(
            "execution/executor_outputs/executor_activity.jsonl",
            '{"ts":"2026-04-22T00:00:00Z","event":"session_start","experiment_id":null,"cwd":".","command":"executor","status":"started","exit_code":null,"duration_sec":0.0,"artifacts":[],"message":"start"}\n',
        )
        artifacts.write_json(
            "execution/executor_outputs/executor_results.json",
            {
                "runs": [
                    {
                        "experiment_id": "exp_01",
                        "experiment_name": "fc table 1",
                        "dataset": "MNIST",
                        "command": "python train.py",
                        "commands_attempted": ["python train.py", "python eval.py"],
                        "cwd": ".",
                        "exit_code": 0,
                        "status": "ok",
                        "runtime_sec": 12.5,
                        "artifacts": ["results/model.pt"],
                        "metrics": {"accuracy": 0.97},
                        "notes": "completed",
                        "logs": {
                            "stdout": "execution/executor_outputs/experiment_exp_01_stdout.log",
                            "stderr": "execution/executor_outputs/experiment_exp_01_stderr.log",
                            "narrative": "execution/executor_outputs/experiment_exp_01_narrative.log",
                            "activity": "execution/executor_outputs/executor_activity.jsonl",
                        },
                        "reason_codes": [],
                    }
                ]
            },
        )
        return ExecutorSessionResult(stdout="", stderr="", returncode=0, narrative="done")

    monkeypatch.setattr(ExecutorAgent, "_run_executor_session", staticmethod(fake_session))

    agent = ExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    result = agent.execute(
        {
            "repo_dir": str(repo_dir),
            "_p2_env_mgr": DummyEnvMgr(),
            "_p2_remaining_sec": 60,
            "_p2_attempt": 1,
        }
    )

    assert result["success"] is True
    manifest = artifacts.read_json("execution/executor_outputs/run_manifest.json")
    run = manifest["runs"][0]
    assert run["run_id"] == "exp_01"
    assert run["experiment_id"] == "exp_01"
    assert run["experiment_name"] == "fc table 1"
    assert "claim_ids" not in run
    assert run["commands_attempted"] == ["python train.py", "python eval.py"]
    assert run["logs"]["activity"] == "execution/executor_outputs/executor_activity.jsonl"


def test_executor_agent_recovers_results_written_under_target_repo_artifacts(tmp_path: Path, monkeypatch) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("python train.py --epochs 1\n", encoding="utf-8")
    (repo_dir / "train.py").write_text("print('ok')\n", encoding="utf-8")

    artifacts = make_artifacts(tmp_path, "misplaced_results")
    artifacts.write_json(
        "fingerprint/claims_ir.json",
        {
            "experiments": [
                {
                    "experiment_id": "exp_01",
                    "name": "misplaced run",
                    "description": "",
                    "dataset": "MNIST",
                    "table_anchor": "Table 1",
                    "primary_metrics": ["accuracy"],
                    "is_primary": True,
                    "notes": None,
                }
            ],
            "claims": [],
            "reason_codes": [],
        },
    )
    artifacts.write_json("task/repo_analysis.json", {"dependency_profiles": [], "entrypoint_candidates": [], "reason_codes": []})
    artifacts.write_json("task/metric_contract.json", {"required_metrics": ["accuracy"], "parsers": [], "normalization": {}, "reason_codes": []})

    def fake_session(env_mgr, prompt, cwd, timeout_sec=600):
        misplaced_dir = repo_dir / "artifacts" / "misplaced_results" / "execution" / "executor_outputs"
        misplaced_dir.mkdir(parents=True)
        (misplaced_dir / "experiment_exp_01_stdout.log").write_text("METRIC:accuracy=0.93\n", encoding="utf-8")
        (misplaced_dir / "experiment_exp_01_stderr.log").write_text("", encoding="utf-8")
        (misplaced_dir / "experiment_exp_01_narrative.log").write_text("ran from misplaced dir\n", encoding="utf-8")
        (misplaced_dir / "executor_results.json").write_text(
            """{
  "runs": [
    {
      "experiment_id": "exp_01",
      "experiment_name": "misplaced run",
      "dataset": "MNIST",
      "command": "python train.py --epochs 1",
      "commands_attempted": ["python train.py --epochs 1"],
      "cwd": ".",
      "exit_code": 0,
      "status": "ok",
      "fidelity": "smoke",
      "evidence_source": "fresh_run",
      "override_args": ["--epochs=1"],
      "metrics": {"accuracy": 0.93},
      "logs": {
        "stdout": "artifacts/misplaced_results/execution/executor_outputs/experiment_exp_01_stdout.log",
        "stderr": "artifacts/misplaced_results/execution/executor_outputs/experiment_exp_01_stderr.log",
        "narrative": "artifacts/misplaced_results/execution/executor_outputs/experiment_exp_01_narrative.log",
        "activity": "artifacts/misplaced_results/execution/executor_outputs/executor_activity.jsonl"
      },
      "reason_codes": []
    }
  ]
}
""",
            encoding="utf-8",
        )
        return ExecutorSessionResult(stdout="", stderr="", returncode=0, narrative="done")

    monkeypatch.setattr(ExecutorAgent, "_run_executor_session", staticmethod(fake_session))

    agent = ExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    result = agent.execute(
        {
            "repo_dir": str(repo_dir),
            "_p2_env_mgr": DummyEnvMgr(),
            "_p2_remaining_sec": 60,
            "_p2_attempt": 1,
        }
    )

    assert result["success"] is True
    assert artifacts.path("execution/executor_outputs/executor_results.json").is_file()
    manifest = artifacts.read_json("execution/executor_outputs/run_manifest.json")
    run = manifest["runs"][0]
    assert run["status"] == "ok"
    assert run["logs"]["stdout"] == "execution/executor_outputs/experiment_exp_01_stdout.log"
    assert "EXPERIMENT_RESULT_MISSING" not in run["reason_codes"]
    assert "DECLARED_LOG_MISSING" not in run["reason_codes"]


def test_executor_prompt_mentions_experiments_repo_and_readme_only(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "README.md").write_text("python main.py --epochs 3\n", encoding="utf-8")

    runtime_spec = ExecutorAgent._build_runtime_spec(DummyEnvMgr())
    prompt = ExecutorAgent._build_prompt(
        repo_dir=repo_dir,
        experiments=[
            {
                "experiment_id": "exp_01",
                "name": "main run",
                "description": "reproduce table 1",
                "dataset": "MNIST",
                "table_anchor": "Table 1",
                "primary_metrics": ["accuracy"],
                "is_primary": True,
                "notes": None,
            }
        ],
        repo_analysis={"entrypoint_candidates": [{"path": "main.py", "command": "python main.py"}]},
        readme_content="python main.py --epochs 3",
        dependency_files={"requirements.txt": "numpy\n"},
        runtime_spec=runtime_spec,
        outputs_dir=tmp_path / "artifacts" / "run" / "execution" / "executor_outputs",
        budget_sec=600,
        soft_budget_sec_per_experiment=300,
    )

    assert "main run" in prompt
    assert "README" in prompt
    assert "Repository Analysis" in prompt
    assert "artifact -> smoke -> trend -> full" in prompt
    assert "Soft budget per experiment: 300 seconds" in prompt
    assert "100+ epoch schedules" in prompt
    assert "estimated full runtime is <= 80% of the remaining global budget" in prompt
    assert runtime_spec.python_command in prompt
    assert "task_spec" not in prompt
    assert "execution_plan" not in prompt
    assert "claim_alignment" not in prompt


def test_executor_system_prompt_mentions_long_horizon_policy() -> None:
    runtime_spec = ExecutorAgent._build_runtime_spec(DummyEnvMgr())

    prompt = ExecutorAgent._build_system_prompt(runtime_spec)

    assert "100+ epoch schedules" in prompt
    assert "<= 5% of the declared schedule" in prompt
    assert "estimated full runtime is <= 80% of the remaining global budget" in prompt
    assert runtime_spec.python_command in prompt


def test_executor_runtime_spec_uses_absolute_venv_paths() -> None:
    runtime_spec = ExecutorAgent._build_runtime_spec(DummyEnvMgr())

    assert runtime_spec.backend == "venv"
    assert runtime_spec.env_name == "test_env"
    assert runtime_spec.env_path == "/tmp/p2c_venv_test_env"
    assert runtime_spec.python_command == "/tmp/p2c_venv_test_env/bin/python"
    assert runtime_spec.pip_command == "/tmp/p2c_venv_test_env/bin/pip"


def test_conda_env_manager_finds_absolute_conda_binary(monkeypatch) -> None:
    monkeypatch.setattr(
        CondaEnvManager,
        "_resolve_binary",
        staticmethod(lambda binary, explicit_env=None: "/home/test/miniconda3/bin/conda" if binary == "conda" else None),
    )

    manager = CondaEnvManager(env_name="phase2_env", python_version="3.10")

    assert manager._use_venv_fallback is False
    assert manager.backend == "/home/test/miniconda3/bin/conda"


def test_executor_guardrail_blocks_background_jobs_and_naked_python() -> None:
    allowed, code, _ = ExecutorAgent._evaluate_bash_guardrail("python train.py --epochs 1", "phase2_env")
    assert allowed is False
    assert code == "CONDA_PREFIX_REQUIRED"

    allowed, code, _ = ExecutorAgent._evaluate_bash_guardrail(
        "conda run --no-capture-output -n phase2_env python train.py &",
        "phase2_env",
    )
    assert allowed is False
    assert code == "BACKGROUND_PROCESS_BLOCKED"


def test_executor_guardrail_blocks_mutation_and_disallowed_overrides() -> None:
    allowed, code, _ = ExecutorAgent._evaluate_bash_guardrail(
        "conda run --no-capture-output -n phase2_env sed -i 's/1/2/' train.py",
        "phase2_env",
    )
    assert allowed is False
    assert code == "DESTRUCTIVE_COMMAND_BLOCKED"

    allowed, code, _ = ExecutorAgent._evaluate_bash_guardrail(
        "conda run --no-capture-output -n phase2_env python train.py --epoch-budget 4",
        "phase2_env",
    )
    assert allowed is True
    assert code is None

    allowed, code, _ = ExecutorAgent._evaluate_bash_guardrail(
        "conda run --no-capture-output -n phase2_env python train.py --iteration-mode cosine",
        "phase2_env",
    )
    assert allowed is False
    assert code == "OVERRIDE_FLAG_NOT_ALLOWED"


def test_executor_result_normalization_accepts_common_executor_aliases() -> None:
    assert (
        ExecutorAgent._normalize_evidence_source(
            "existing_artifact",
            stdout_text="",
            existing_artifacts=[],
            commands_attempted=[],
        )
        == "existing_results"
    )
    assert ExecutorAgent._command_was_observed(
        "python train.py --epochs 1",
        observed_command_set=set(),
        observed_commands=["/opt/mamba run -n phase2_env python train.py --epochs 1 > /tmp/run.log"],
    )


def test_executor_session_uses_streaming_prompt_for_claude_sdk(monkeypatch) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setenv("P2C_USE_CLAUDE_TOOL_GUARDRAILS", "1")

    class FakeClaudeAgentOptions:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    async def fake_query(*, prompt, options):
        seen["prompt"] = prompt
        seen["options"] = options
        seen["messages"] = [item async for item in prompt]
        if False:
            yield None

    monkeypatch.setattr("p2c.agents.phase2.executor_agent.ClaudeAgentOptions", FakeClaudeAgentOptions)
    monkeypatch.setattr("p2c.agents.phase2.executor_agent.query", fake_query)

    result = ExecutorAgent._run_executor_session(DummyEnvMgr(), "run experiment", cwd="/tmp", timeout_sec=30)

    assert result.returncode == 0
    assert isinstance(seen["prompt"], AsyncIterable)
    assert seen["messages"] == [
        {
            "type": "user",
            "session_id": "",
            "message": {"role": "user", "content": "run experiment"},
            "parent_tool_use_id": None,
        }
    ]
    options = seen["options"]
    assert getattr(options, "cwd") == "/tmp"
    assert "Bash" not in getattr(options, "allowed_tools")
    assert getattr(options, "can_use_tool") is not None


def test_executor_load_runs_normalizes_progressive_fidelity_fields(tmp_path: Path) -> None:
    artifacts = make_artifacts(tmp_path, "normalize_runs")
    outputs_dir = artifacts.path("execution/executor_outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    artifacts.write_text("execution/executor_outputs/experiment_exp_01_stdout.log", "METRIC:accuracy=0.88\n")
    artifacts.write_text("execution/executor_outputs/experiment_exp_01_stderr.log", "")
    artifacts.write_text("execution/executor_outputs/experiment_exp_01_narrative.log", "short run\n")
    artifacts.write_text("execution/executor_outputs/executor_activity.jsonl", "{}\n")
    artifacts.write_json(
        "execution/executor_outputs/executor_results.json",
        {
            "runs": [
                {
                    "experiment_id": "exp_01",
                    "experiment_name": "short run",
                    "command": "conda run --no-capture-output -n test_env python train.py --epochs 1",
                    "commands_attempted": [
                        "conda run --no-capture-output -n test_env python train.py --epochs 1"
                    ],
                    "cwd": ".",
                    "exit_code": 0,
                    "status": "ok",
                    "fidelity": "full",
                    "evidence_source": "fresh_run",
                    "override_args": ["--epochs=1"],
                    "runtime_sec": 1.0,
                    "metrics": {"accuracy": 0.88},
                    "logs": {
                        "stdout": "execution/executor_outputs/experiment_exp_01_stdout.log",
                        "stderr": "execution/executor_outputs/experiment_exp_01_stderr.log",
                        "narrative": "execution/executor_outputs/experiment_exp_01_narrative.log",
                        "activity": "execution/executor_outputs/executor_activity.jsonl",
                    },
                    "reason_codes": [],
                }
            ]
        },
    )

    agent = ExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    runs = agent._load_executor_runs(
        artifacts.path("execution/executor_outputs/executor_results.json"),
        contract=MetricContract(required_metrics=["accuracy"], parsers=[], normalization={}, reason_codes=[]),
        session_stdout="",
        experiments=[
            {
                "experiment_id": "exp_01",
                "name": "short run",
                "dataset": "MNIST",
                "table_anchor": "Table 1",
            }
        ],
        outputs_dir=outputs_dir,
        observed_commands=["conda run --no-capture-output -n test_env python train.py --epochs 1"],
    )

    run = runs[0]
    assert run["fidelity"] == "trend"
    assert run["execution_outcome"] == "TREND_SUPPORTED"
    assert "FULL_WITH_OVERRIDE_ARGS" in run["reason_codes"]
    assert run["override_args"] == ["--epochs=1"]


def test_executor_load_runs_synthesizes_missing_experiment_rows(tmp_path: Path) -> None:
    artifacts = make_artifacts(tmp_path, "missing_rows")
    outputs_dir = artifacts.path("execution/executor_outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    artifacts.write_json("execution/executor_outputs/executor_results.json", {"runs": []})
    artifacts.write_text("execution/executor_outputs/executor_activity.jsonl", "{}\n")

    agent = ExecutorAgent(llm=None, artifacts=artifacts, step_index=1, step_total=1)
    runs = agent._load_executor_runs(
        artifacts.path("execution/executor_outputs/executor_results.json"),
        contract=MetricContract(required_metrics=[], parsers=[], normalization={}, reason_codes=[]),
        session_stdout="",
        experiments=[{"experiment_id": "exp_missing", "name": "missing run"}],
        outputs_dir=outputs_dir,
        observed_commands=[],
    )

    run = runs[0]
    assert run["experiment_id"] == "exp_missing"
    assert run["status"] == "failed"
    assert run["stop_reason"] == "runtime_failure"
    assert "EXPERIMENT_RESULT_MISSING" in run["reason_codes"]
