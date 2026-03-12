from __future__ import annotations

import re
from pathlib import Path

from p2c.agents.base import BaseAgent
from p2c.agents.phase1.repo_analysis import SystemRepoAnalyzer
from p2c.schemas import (
    Entrypoint,
    MetricContract,
    MetricObserver,
    MetricParser,
    RepoAnalysis,
    RunConfig,
    TaskCompileOutput,
    TaskItem,
    TaskSpec,
)

SYSTEM_PROMPT = (
    "You compile an executable TaskSpec from repository clues and code-verifiable claims. "
    "Only use existing file paths. Output strict JSON."
)

USER_PROMPT_TEMPLATE = (
    "Inputs: fingerprint/claims_ir.json + repo_dir scan\n"
    "Outputs: task/task_spec.json and task/metric_contract.json\n"
    "Constraints: keep tasks <= 5, include at least one metric observer."
)


class CompileTaskSpecAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="compile_task_spec", *args, **kwargs)

    @staticmethod
    def _load_repo_analysis(artifacts, repo_dir: Path) -> RepoAnalysis:
        payload = artifacts.read_json("task/repo_analysis.json")
        if payload.get("entrypoint_candidates") or payload.get("dependency_profiles"):
            return RepoAnalysis(**payload)
        analysis = SystemRepoAnalyzer(repo_dir).analyze()
        artifacts.write_json("task/repo_analysis.json", analysis.model_dump())
        return analysis

    @staticmethod
    def _extract_hyperparams(claims: list[dict]) -> dict[str, float | int | str]:
        out: dict[str, float | int | str] = {}
        patterns: list[tuple[str, str]] = [
            ("lr", r"(?:learning\s*rate|lr)\D*(\d+(?:\.\d+)?(?:e-?\d+)?)"),
            ("epochs", r"(?:epoch|epochs)\D*(\d+)"),
            ("batch_size", r"(?:batch(?:\s*size)?)\D*(\d+)"),
            ("seed", r"(?:seed|random\s*seed)\D*(\d+)"),
        ]
        for row in claims:
            predicate = str(row.get("predicate") or "")
            target = row.get("target")
            text = f"{predicate} {target if target is not None else ''}".lower()
            for key, pattern in patterns:
                if key in out:
                    continue
                m = re.search(pattern, text)
                if not m:
                    continue
                raw = m.group(1)
                try:
                    if key in {"epochs", "batch_size", "seed"}:
                        out[key] = int(float(raw))
                    else:
                        out[key] = float(raw)
                except ValueError:
                    out[key] = raw
        return out

    @staticmethod
    def _collect_required_metrics(claims: list[dict]) -> list[str]:
        metrics: list[str] = []
        for row in claims:
            if not bool(row.get("code_verifiable", not row.get("unverifiable_from_paper", False))):
                continue
            metric = str(row.get("metric") or "").strip().lower()
            if metric and metric not in metrics:
                metrics.append(metric)
        if not metrics:
            metrics = ["accuracy"]
        return metrics

    @staticmethod
    def _to_task_items(
        selected: list[Entrypoint],
        *,
        required_metrics: list[str],
        hyperparams: dict[str, float | int | str],
        budget_minutes: int,
    ) -> list[TaskItem]:
        items: list[TaskItem] = []
        for idx, ep in enumerate(selected, start=1):
            timeout_class = "medium"
            if budget_minutes <= 10:
                timeout_class = "short"
            elif budget_minutes >= 45:
                timeout_class = "long"
            items.append(
                TaskItem(
                    task_id=f"task_{idx:02d}",
                    entrypoint=ep.path,
                    command=ep.command,
                    cwd=ep.cwd,
                    runtime=ep.runtime,
                    dependency_profile_id=ep.dependency_profile_id,
                    timeout_class=timeout_class,
                    expected_metrics=required_metrics,
                    hyperparams=hyperparams,
                    confidence=ep.confidence,
                    evidence=ep.evidence,
                )
            )
        return items

    def execute(self, ctx: dict) -> dict:
        repo_dir = Path(ctx["repo_dir"])
        claims_doc = self.artifacts.read_json("fingerprint/claims_ir.json")
        all_claims = [c for c in claims_doc.get("claims", []) if isinstance(c, dict)]
        verifiable_claims = [
            c for c in all_claims if bool(c.get("code_verifiable", not c.get("unverifiable_from_paper", False)))
        ]
        repo_analysis = self._load_repo_analysis(self.artifacts, repo_dir)
        candidates = [Entrypoint(**row) if isinstance(row, dict) else row for row in repo_analysis.entrypoint_candidates]
        primary_id = str(repo_analysis.primary_entrypoint_id or "")
        selected: list[Entrypoint] = []
        if primary_id:
            selected.extend([e for e in candidates if str(e.entrypoint_id or "") == primary_id][:1])
        for candidate in candidates:
            if len(selected) >= 5:
                break
            if any(str(x.entrypoint_id or "") == str(candidate.entrypoint_id or "") for x in selected):
                continue
            selected.append(candidate)

        reason_codes = list(repo_analysis.reason_codes)
        if selected:
            reason_codes.append("ENTRYPOINT_SELECTED_PRIMARY")
            if len(selected) > 1:
                reason_codes.append("ENTRYPOINT_SELECTED_BACKUP")
        else:
            reason_codes.append("REPO_ANALYSIS_NO_EXECUTABLE_CANDIDATE")

        observers = [
            MetricObserver(name="accuracy_percent", kind="stdout_regex", pattern=r"accuracy[^0-9]*(\d+(?:\.\d+)?)%"),
            MetricObserver(name="accuracy_decimal", kind="stdout_regex", pattern=r"accuracy[^0-9]*(0\.\d+|1\.0+)"),
        ]

        budget_minutes = int(ctx.get("budget_minutes", 60))
        run_matrix = [
            RunConfig(
                seed=0,
                timeout_sec=min(1800, max(120, budget_minutes * 60)),
                budget_minutes=budget_minutes,
            )
        ]

        if not selected:
            reason_codes.append("NO_ENTRYPOINT_FOUND")

        required_metrics = self._collect_required_metrics(verifiable_claims)
        hyperparams = self._extract_hyperparams(verifiable_claims)
        tasks = self._to_task_items(
            selected,
            required_metrics=required_metrics,
            hyperparams=hyperparams,
            budget_minutes=budget_minutes,
        )

        task_spec = TaskSpec(
            constraints={
                "budget_minutes": budget_minutes,
                "network": "limited",
                "allowed_modification_scope": "Target/code",
                "max_self_heal_iters": int(ctx.get("max_self_heal_iters", 6)),
            },
            tasks=tasks,
            entrypoints=selected,
            metric_observers=observers,
            run_matrix=run_matrix,
            selection_notes=[
                "compiled_from_repo_analysis",
                f"verifiable_claims={len(verifiable_claims)}",
                f"filtered_non_verifiable={max(0, len(all_claims) - len(verifiable_claims))}",
                f"primary_entrypoint_id={primary_id or ''}",
                f"primary_entrypoint_path={selected[0].path if selected else ''}",
                f"primary_entrypoint_evidence={selected[0].evidence if selected else ''}",
            ],
            reason_codes=list(dict.fromkeys(reason_codes)),
        )

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
            reason_codes=[] if selected else ["REPO_ANALYSIS_NO_EXECUTABLE_CANDIDATE"],
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
