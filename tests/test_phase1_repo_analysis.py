from __future__ import annotations

from pathlib import Path

from p2c.agents.phase1.compile_task_spec import CompileTaskSpecAgent
from p2c.agents.phase1.repo_analysis import SystemRepoAnalyzer
from p2c.io_artifacts import ArtifactManager


class DummyLLM:
    def chat_text(self, system: str, user: str) -> str:
        return ""

    def chat_json(self, schema, system: str, user: str):
        return {"notes": "", "reason_codes": []}


def make_artifacts(tmp_path: Path, run_id: str = "phase1_repo_analysis") -> ArtifactManager:
    artifacts = ArtifactManager(tmp_path, run_id)
    artifacts.ensure_tree()
    return artifacts


def test_repo_analysis_detects_notebook_entrypoint(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    code_dir = repo_dir / "code"
    code_dir.mkdir(parents=True)
    (repo_dir / "requirements.txt").write_text("notebook\njupyter\n", encoding="utf-8")
    (code_dir / "train.ipynb").write_text('{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}\n', encoding="utf-8")

    analysis = SystemRepoAnalyzer(repo_dir).analyze()

    assert analysis.entrypoint_candidates
    primary = analysis.entrypoint_candidates[0]
    assert primary.path == "code/train.ipynb"
    assert primary.cwd == "code"
    assert primary.runtime == "python"
    assert "python -m jupyter nbconvert" in primary.command
    assert "--execute" in primary.command
    assert "--output train.executed.ipynb" in primary.command


def test_compile_task_spec_emits_task_for_notebook_repo(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    code_dir = repo_dir / "code"
    code_dir.mkdir(parents=True)
    (repo_dir / "requirements.txt").write_text("notebook\njupyter\n", encoding="utf-8")
    (code_dir / "train.ipynb").write_text('{"cells":[],"metadata":{},"nbformat":4,"nbformat_minor":5}\n', encoding="utf-8")

    artifacts = make_artifacts(tmp_path)
    artifacts.write_json(
        "fingerprint/claims_ir.json",
        {
            "claims": [
                {
                    "claim_id": "C1",
                    "type": "result",
                    "predicate": "accuracy reaches reported value",
                    "metric": "accuracy",
                    "target": 0.9,
                    "baseline": None,
                    "conditions": {},
                    "aggregation": "best",
                    "evidence_set": ["paper_text"],
                    "tolerance_policy": {"abs_eps": 0.01, "rel_eps": 0.02},
                    "unverifiable_from_paper": False,
                    "code_verifiable": True,
                    "reason_codes": [],
                    "notes": None,
                }
            ],
            "reason_codes": [],
        },
    )

    agent = CompileTaskSpecAgent(llm=DummyLLM(), artifacts=artifacts, step_index=1, step_total=1)
    result = agent.execute({"repo_dir": str(repo_dir), "budget_minutes": 30, "max_self_heal_iters": 6})

    assert result["task_spec"]["tasks"]
    task = result["task_spec"]["tasks"][0]
    assert task["entrypoint"] == "code/train.ipynb"
    assert task["cwd"] == "code"
    assert "python -m jupyter nbconvert" in task["command"]


def test_repo_analysis_prefers_training_script_as_primary(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    src_dir = repo_dir / "src"
    src_dir.mkdir(parents=True)
    (repo_dir / "requirements.txt").write_text("pandas\nscikit-learn\n", encoding="utf-8")
    (src_dir / "train_model.py").write_text("if __name__ == '__main__':\n    print('train')\n", encoding="utf-8")
    (src_dir / "threshold_tuning.py").write_text("if __name__ == '__main__':\n    print('tune')\n", encoding="utf-8")

    analysis = SystemRepoAnalyzer(repo_dir).analyze()

    assert analysis.primary_entrypoint_id is not None
    primary = next(ep for ep in analysis.entrypoint_candidates if ep.entrypoint_id == analysis.primary_entrypoint_id)
    assert primary.path == "src/train_model.py"


def test_repo_analysis_detects_readme_wrapper_and_derived_cwd(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    workdir = repo_dir / "workdir"
    scripts_dir = repo_dir / "scripts"
    workdir.mkdir(parents=True)
    scripts_dir.mkdir(parents=True)
    (repo_dir / "README.md").write_text("./run.sh\n", encoding="utf-8")
    (repo_dir / "run.sh").write_text(
        "#!/usr/bin/env bash\n"
        "cd workdir\n"
        "../scripts/do.sh\n",
        encoding="utf-8",
    )
    (scripts_dir / "do.sh").write_text(
        "#!/usr/bin/env bash\n"
        "python ../scripts/tool.py\n",
        encoding="utf-8",
    )
    (scripts_dir / "tool.py").write_text(
        "if __name__ == '__main__':\n"
        "    print('ok')\n",
        encoding="utf-8",
    )

    analysis = SystemRepoAnalyzer(repo_dir).analyze()

    primary = next(ep for ep in analysis.entrypoint_candidates if ep.entrypoint_id == analysis.primary_entrypoint_id)
    assert primary.path == "run.sh"
    assert "README_WORKFLOW_PRIMARY" in primary.reason_codes

    derived_shell = next(
        ep for ep in analysis.entrypoint_candidates
        if ep.path == "scripts/do.sh" and ep.derived_from_wrapper == "run.sh"
    )
    assert derived_shell.cwd == "workdir"
    assert derived_shell.command == "bash ../scripts/do.sh"
    assert derived_shell.path_resolution_mode == "wrapper_virtual_cwd"

    derived_python = next(
        ep for ep in analysis.entrypoint_candidates
        if ep.path == "scripts/tool.py" and ep.derived_from_wrapper == "run.sh"
    )
    assert derived_python.cwd == "workdir"
    assert derived_python.command == "python ../scripts/tool.py"


def test_repo_analysis_prefers_exact_readme_python_command(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (repo_dir / "main.py").write_text(
        "if __name__ == '__main__':\n"
        "    print('ok')\n",
        encoding="utf-8",
    )
    (repo_dir / "README.md").write_text(
        "python main.py --task CIFAR10 --epochs 120 --data-aug\n",
        encoding="utf-8",
    )

    analysis = SystemRepoAnalyzer(repo_dir).analyze()

    assert analysis.primary_entrypoint_id is not None
    primary = next(ep for ep in analysis.entrypoint_candidates if ep.entrypoint_id == analysis.primary_entrypoint_id)
    assert primary.path == "main.py"
    assert primary.command == "python main.py --task CIFAR10 --epochs 120 --data-aug"
    assert "README_VERIFIED_COMMAND" in primary.reason_codes


def test_compile_task_spec_prioritizes_readme_command(tmp_path: Path) -> None:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    (repo_dir / "main.py").write_text(
        "if __name__ == '__main__':\n"
        "    print('ok')\n",
        encoding="utf-8",
    )
    (repo_dir / "README.md").write_text(
        "python main.py --task CIFAR10 --epochs 120 --data-aug\n",
        encoding="utf-8",
    )

    artifacts = make_artifacts(tmp_path, run_id="phase1_repo_analysis_readme")
    artifacts.write_json(
        "fingerprint/claims_ir.json",
        {
            "claims": [
                {
                    "claim_id": "C1",
                    "type": "result",
                    "predicate": "test error reaches reported value",
                    "metric": "accuracy",
                    "target": 0.9,
                    "baseline": None,
                    "conditions": {},
                    "aggregation": "best",
                    "evidence_set": ["paper_text"],
                    "tolerance_policy": {"abs_eps": 0.01, "rel_eps": 0.02},
                    "unverifiable_from_paper": False,
                    "code_verifiable": True,
                    "reason_codes": [],
                    "notes": None,
                }
            ],
            "reason_codes": [],
        },
    )

    agent = CompileTaskSpecAgent(llm=DummyLLM(), artifacts=artifacts, step_index=1, step_total=1)
    result = agent.execute({"repo_dir": str(repo_dir), "budget_minutes": 30, "max_self_heal_iters": 6})

    assert result["task_spec"]["tasks"]
    task = result["task_spec"]["tasks"][0]
    assert task["entrypoint"] == "main.py"
    assert task["command"] == "python main.py --task CIFAR10 --epochs 120 --data-aug"
