from __future__ import annotations

import os
import re
from pathlib import Path

from p2c.agents.base import BaseAgent
from p2c.schemas import Entrypoint, MetricContract, MetricObserver, MetricParser, RunConfig, TaskCompileOutput, TaskSpec

SYSTEM_PROMPT = (
    "You compile an executable TaskSpec from claims and repository clues. "
    "Only use existing file paths. Output strict JSON."
)

USER_PROMPT_TEMPLATE = (
    "Inputs: fingerprint/claims_ir.json + repo_dir scan\n"
    "Outputs: task/task_spec.json and task/metric_contract.json\n"
    "Constraints: keep entrypoints <= 5, include at least one metric observer."
)


class CompileTaskSpecAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="compile_task_spec", *args, **kwargs)

    @staticmethod
    def _scan_entrypoints(repo_dir: Path) -> list[Entrypoint]:
        candidates: list[Entrypoint] = []

        for path in sorted(repo_dir.glob("main*.py")):
            candidates.append(
                Entrypoint(
                    path=str(path.relative_to(repo_dir)),
                    command=f"python3 {path.relative_to(repo_dir)}",
                    confidence=0.8,
                    evidence="main*.py discovered",
                )
            )

        for path in sorted(repo_dir.rglob("train.py"))[:3]:
            candidates.append(
                Entrypoint(
                    path=str(path.relative_to(repo_dir)),
                    command=f"python3 {path.relative_to(repo_dir)}",
                    confidence=0.7,
                    evidence="train.py discovered",
                )
            )

        readme = repo_dir / "README.md"
        if readme.exists():
            text = readme.read_text(encoding="utf-8", errors="ignore")
            for cmd in re.findall(r"python\s+([\w./-]+\.py)", text):
                p = repo_dir / cmd
                if p.exists():
                    candidates.append(
                        Entrypoint(
                            path=str(Path(cmd)),
                            command=f"python3 {cmd}",
                            confidence=0.75,
                            evidence="README python command",
                        )
                    )

        # Deduplicate while preserving order.
        uniq: dict[str, Entrypoint] = {}
        for entry in candidates:
            if entry.path not in uniq:
                uniq[entry.path] = entry
        return list(uniq.values())[:5]

    def execute(self, ctx: dict) -> dict:
        repo_dir = Path(ctx["repo_dir"])
        claims_doc = self.artifacts.read_json("fingerprint/claims_ir.json")
        claim_ids = [c.get("claim_id", "") for c in claims_doc.get("claims", []) if c.get("claim_id")]
        candidates = self._scan_entrypoints(repo_dir)

        llm_schema = {
            "type": "object",
            "properties": {
                "selected_paths": {"type": "array", "items": {"type": "string"}},
                "reason_codes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["selected_paths", "reason_codes"],
        }
        llm_user = (
            USER_PROMPT_TEMPLATE
            + "\nCandidates:\n"
            + "\n".join(f"- {e.path}: {e.command}" for e in candidates)
            + "\nClaims:\n"
            + ", ".join(claim_ids)
        )
        llm_data, llm_err = self.safe_chat_json(llm_schema, SYSTEM_PROMPT, llm_user)

        if llm_data and llm_data.get("selected_paths"):
            selected_set = set(str(x) for x in llm_data["selected_paths"])
            selected = [e for e in candidates if e.path in selected_set][:5]
            if not selected:
                selected = candidates[:5]
                reason_codes = ["LLM_SELECTION_EMPTY", "HEURISTIC_FALLBACK"]
            else:
                reason_codes = list(llm_data.get("reason_codes", []))
        else:
            selected = candidates[:5]
            reason_codes = ["LLM_UNAVAILABLE", "HEURISTIC_FALLBACK"]

        observers = [
            MetricObserver(name="accuracy_percent", kind="stdout_regex", pattern=r"accuracy[^0-9]*(\d+(?:\.\d+)?)%"),
            MetricObserver(name="accuracy_decimal", kind="stdout_regex", pattern=r"accuracy[^0-9]*(0\.\d+|1\.0+)"),
        ]

        run_matrix = [
            RunConfig(
                seed=0,
                timeout_sec=min(1800, max(120, int(ctx.get("budget_minutes", 60)) * 60)),
                budget_minutes=int(ctx.get("budget_minutes", 60)),
            )
        ]

        if not selected:
            reason_codes.append("NO_ENTRYPOINT_FOUND")

        task_spec = TaskSpec(
            goal=claim_ids,
            constraints={
                "budget_minutes": int(ctx.get("budget_minutes", 60)),
                "network": "limited",
                "allowed_modification_scope": "Target/code",
                "max_self_heal_iters": int(ctx.get("max_self_heal_iters", 6)),
            },
            entrypoints=selected,
            metric_observers=observers,
            run_matrix=run_matrix,
            reason_codes=reason_codes,
        )

        required_metrics = []
        for claim in claims_doc.get("claims", []):
            metric = claim.get("metric")
            if metric and metric not in required_metrics:
                required_metrics.append(metric)
        if not required_metrics:
            required_metrics = ["accuracy"]

        metric_contract = MetricContract(
            required_metrics=required_metrics,
            parsers=[
                MetricParser(name="acc_percent", regex=r"accuracy[^0-9]*(\d+(?:\.\d+)?)%", metric_name="accuracy"),
                MetricParser(name="acc_decimal", regex=r"accuracy[^0-9]*(0\.\d+|1\.0+)", metric_name="accuracy"),
            ],
            normalization={
                "accuracy": {
                    "percent_to_decimal": True,
                    "clip": [0, 1],
                }
            },
            reason_codes=[] if selected else ["NO_ENTRYPOINT_FOUND"],
        )

        output = TaskCompileOutput(task_spec=task_spec, metric_contract=metric_contract)
        self.artifacts.write_json("task/task_spec.json", output.task_spec.model_dump())
        self.artifacts.write_json("task/metric_contract.json", output.metric_contract.model_dump())

        # Ensure top-5 and path validity invariants.
        for e in output.task_spec.entrypoints:
            if not (repo_dir / e.path).exists():
                raise ValueError(f"Entrypoint does not exist: {e.path}")
        if len(output.task_spec.entrypoints) > 5:
            raise ValueError("Too many entrypoints")

        return {
            "task_spec": output.task_spec.model_dump(),
            "metric_contract": output.metric_contract.model_dump(),
        }
