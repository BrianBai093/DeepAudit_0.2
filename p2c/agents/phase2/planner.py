"""PlannerAgent — reads Phase 1 artifacts + repo and produces an ExecutionPlan."""

from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.local_prompt_templates import (
    PLANNER_SYSTEM_PROMPT,
    build_planner_user_prompt,
)
from p2c.schemas import ExecutionFailure, ExecutionPlan

# Files we try to read from the repo to feed into the prompt
_DEP_FILE_NAMES = [
    "requirements.txt",
    "requirements-dev.txt",
    "requirements_dev.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "environment.yml",
    "environment.yaml",
    "Pipfile",
    "Pipfile.lock",
    "conda-requirements.txt",
    "Makefile",
]

_EXECUTION_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "plan_id": {"type": "string"},
        "plan_version": {"type": "integer"},
        "python_version": {"type": "string"},
        "conda_dependencies": {"type": "array"},
        "pip_dependencies": {"type": "array"},
        "system_packages": {"type": "array"},
        "pre_install_commands": {"type": "array"},
        "execution_steps": {"type": "array"},
        "expected_results": {"type": "array"},
        "compatibility_issues": {"type": "array"},
        "env_name": {"type": "string"},
        "codex_autonomous_fallback": {"type": "boolean"},
        "total_budget_sec": {"type": "integer"},
        "reason_codes": {"type": "array"},
        "notes": {"type": ["string", "null"]},
    },
    "required": ["plan_id", "execution_steps", "env_name"],
}


class PlannerAgent(BaseAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="planner", *args, **kwargs)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Produce an ``ExecutionPlan`` and persist it as an artifact."""
        failures: list[ExecutionFailure] = ctx.get("_p2_failures", [])
        plan_version = len(failures) + 1

        # Gather inputs
        repo_dir = str(ctx["repo_dir"])
        budget_sec = int(ctx.get("budget_minutes", 30)) * 60
        run_id = str(ctx.get("run_id", "run"))
        env_name = f"p2c_{run_id[:8]}_{uuid.uuid4().hex[:6]}"

        claims_ir = self.artifacts.read_json("fingerprint/claims_ir.json")
        task_spec = self.artifacts.read_json("task/task_spec.json")
        metric_contract = self.artifacts.read_json("task/metric_contract.json")
        repo_analysis = self.artifacts.read_json("task/repo_analysis.json")

        repo_tree = self._repo_tree(repo_dir)
        readme = self._read_readme(repo_dir)
        dep_files = self._read_dependency_files(repo_dir)

        failure_ctx = None
        if failures:
            failure_ctx = json.dumps(
                [f.model_dump() if hasattr(f, "model_dump") else f for f in failures],
                indent=2, ensure_ascii=False,
            )

        user_prompt = build_planner_user_prompt(
            claims_ir_json=json.dumps(claims_ir, indent=2, ensure_ascii=False),
            task_spec_json=json.dumps(task_spec, indent=2, ensure_ascii=False),
            metric_contract_json=json.dumps(metric_contract, indent=2, ensure_ascii=False),
            repo_analysis_json=json.dumps(repo_analysis, indent=2, ensure_ascii=False),
            repo_tree=repo_tree,
            readme_content=readme,
            dependency_files=dep_files,
            failure_context=failure_ctx,
            env_name=env_name,
            budget_sec=budget_sec,
        )

        data, err = self.safe_chat_json(
            schema=_EXECUTION_PLAN_SCHEMA,
            system=PLANNER_SYSTEM_PROMPT,
            user=user_prompt,
        )
        if data is None:
            raise RuntimeError(f"PlannerAgent LLM call failed: {err}")

        # Ensure required fields
        data.setdefault("plan_id", uuid.uuid4().hex[:12])
        data.setdefault("env_name", env_name)
        data["plan_version"] = plan_version
        data["total_budget_sec"] = budget_sec

        # Validate & persist
        plan = ExecutionPlan(**data)
        self._validate_plan(plan, repo_dir)
        self.artifacts.write_json("execution/execution_plan.json", plan.model_dump())
        self.log("DONE", f"plan v{plan.plan_version}: {len(plan.execution_steps)} steps, "
                         f"{len(plan.pip_dependencies)} pip deps, env={plan.env_name}")
        return {"plan": plan}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _repo_tree(repo_dir: str, max_entries: int = 500) -> str:
        try:
            proc = subprocess.run(
                ["find", ".", "-maxdepth", "4", "-not", "-path", "./.git/*",
                 "-not", "-path", "./__pycache__/*", "-not", "-path", "./node_modules/*"],
                cwd=repo_dir, capture_output=True, text=True, timeout=15,
            )
            lines = proc.stdout.strip().splitlines()[:max_entries]
            return "\n".join(sorted(lines))
        except Exception:  # noqa: BLE001
            return "(tree unavailable)"

    @staticmethod
    def _read_readme(repo_dir: str) -> str:
        for name in ("README.md", "readme.md", "README.rst", "README.txt", "README"):
            p = Path(repo_dir) / name
            if p.exists():
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    return text[:8000]
                except Exception:  # noqa: BLE001
                    pass
        return "(no README found)"

    @staticmethod
    def _read_dependency_files(repo_dir: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for name in _DEP_FILE_NAMES:
            p = Path(repo_dir) / name
            if p.exists():
                try:
                    result[name] = p.read_text(encoding="utf-8", errors="ignore")[:4000]
                except Exception:  # noqa: BLE001
                    pass
        return result

    @staticmethod
    def _validate_plan(plan: ExecutionPlan, repo_dir: str) -> None:
        """Light validation — log warnings but don't block."""
        step_ids = {s.step_id for s in plan.execution_steps}
        for step in plan.execution_steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    pass  # dangling dep — executor will ignore ordering

        # Check that referenced cwds exist
        for step in plan.execution_steps:
            cwd_path = Path(repo_dir) / step.cwd
            if not cwd_path.is_dir():
                step.cwd = "."  # auto-fix
