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
    "Constraints: prefer wrapper-first workflow coverage and include at least one metric observer."
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
                    path_resolution_mode=ep.path_resolution_mode,
                    derived_from_wrapper=ep.derived_from_wrapper,
                    reason_codes=ep.reason_codes,
                )
            )
        return items

    @staticmethod
    def _is_readme_candidate(candidate: Entrypoint) -> bool:
        evidence = str(candidate.evidence or "").lower()
        reason_codes = {str(code) for code in candidate.reason_codes}
        return (
            "README_WORKFLOW_PRIMARY" in reason_codes
            or "README_VERIFIED_COMMAND" in reason_codes
            or "readme verified" in evidence
        )

    @staticmethod
    def _is_compat_task_entrypoint(candidate: Entrypoint) -> bool:
        path = str(candidate.path or "").lower()
        command = str(candidate.command or "").lower()
        if path.endswith(".ipynb"):
            return False
        if "jupyter nbconvert" in command and "--execute" in command:
            return False
        return candidate.runtime in {"python", "shell", "make"}

    @classmethod
    def _select_entrypoints(cls, candidates: list[Entrypoint], primary_id: str) -> list[Entrypoint]:
        selected: list[Entrypoint] = []
        seen: set[str] = set()

        def add(candidate: Entrypoint) -> None:
            if not cls._is_compat_task_entrypoint(candidate):
                return
            candidate_id = str(candidate.entrypoint_id or candidate.path)
            if candidate_id in seen:
                return
            seen.add(candidate_id)
            selected.append(candidate)

        primary_candidate: Entrypoint | None = None
        if primary_id:
            for candidate in candidates:
                if str(candidate.entrypoint_id or "") == primary_id:
                    primary_candidate = candidate
                    break

        readme_candidates = [candidate for candidate in candidates if cls._is_readme_candidate(candidate)]
        if primary_candidate and cls._is_readme_candidate(primary_candidate):
            add(primary_candidate)
        for candidate in readme_candidates:
            add(candidate)
        if primary_candidate is not None:
            add(primary_candidate)

        primary_wrapper = None
        for candidate in selected:
            if "README_WORKFLOW_PRIMARY" in candidate.reason_codes:
                primary_wrapper = candidate.path
                break
        for candidate in candidates:
            if "README_WORKFLOW_PRIMARY" in candidate.reason_codes:
                add(candidate)
        if primary_wrapper:
            for candidate in candidates:
                if candidate.derived_from_wrapper == primary_wrapper:
                    add(candidate)

        for candidate in candidates:
            if cls._is_compat_task_entrypoint(candidate):
                add(candidate)
            if len(selected) >= 12:
                break
        return selected

    @staticmethod
    def _default_metric_patterns() -> dict[str, list[str]]:
        return {
            "accuracy": [
                r"accuracy[^0-9]*(\d+(?:\.\d+)?)%",
                r"accuracy[^0-9]*(0\.\d+|1\.0+)",
            ],
            "precision": [
                r"(?im)^\s*precision\s*[:=]\s*(0?\.\d+|1(?:\.0+)?)",
                r"['\"]precision['\"]\s*:\s*(0?\.\d+|1(?:\.0+)?)",
            ],
            "recall": [
                r"(?im)^\s*recall\s*[:=]\s*(0?\.\d+|1(?:\.0+)?)",
                r"['\"]recall['\"]\s*:\s*(0?\.\d+|1(?:\.0+)?)",
            ],
            "f1": [
                r"(?im)^\s*f1(?:-score)?\s*[:=]\s*(0?\.\d+|1(?:\.0+)?)",
                r"['\"]f1['\"]\s*:\s*(0?\.\d+|1(?:\.0+)?)",
            ],
            "roc_auc": [
                r"(?im)^\s*roc[-_ ]auc\s*[:=]\s*(0?\.\d+|1(?:\.0+)?)\s*$",
                r"['\"]roc_auc['\"]\s*:\s*(0?\.\d+|1(?:\.0+)?)",
            ],
            "pr_auc": [
                r"(?im)^\s*pr[-_ ]auc\s*[:=]\s*(0?\.\d+|1(?:\.0+)?)\s*$",
                r"['\"]pr_auc['\"]\s*:\s*(0?\.\d+|1(?:\.0+)?)",
            ],
        }

    @classmethod
    def _metric_observers_for(cls, required_metrics: list[str]) -> list[MetricObserver]:
        core_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
        patterns = cls._default_metric_patterns()
        observers: list[MetricObserver] = []
        for metric_name in sorted(core_metrics.union(required_metrics)):
            for idx, pattern in enumerate(patterns.get(metric_name, []), start=1):
                observers.append(
                    MetricObserver(
                        name=f"{metric_name}_pattern_{idx}",
                        kind="stdout_regex",
                        pattern=pattern,
                    )
                )
        return observers

    @classmethod
    def _metric_parsers_for(cls, required_metrics: list[str]) -> list[MetricParser]:
        core_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
        patterns = cls._default_metric_patterns()
        parsers: list[MetricParser] = []
        for metric_name in sorted(core_metrics.union(required_metrics)):
            for idx, pattern in enumerate(patterns.get(metric_name, []), start=1):
                parsers.append(
                    MetricParser(
                        name=f"{metric_name}_pattern_{idx}",
                        regex=pattern,
                        metric_name=metric_name,
                    )
                )
        return parsers

    @staticmethod
    def _metric_normalization(required_metrics: list[str]) -> dict[str, dict[str, object]]:
        normalized = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
        normalized.update(required_metrics)
        return {
            metric_name: {"percent_to_decimal": True, "clip": [0, 1]}
            for metric_name in sorted(normalized)
        }

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
        selected = self._select_entrypoints(candidates, primary_id)
        excluded_compat_candidates = [
            candidate for candidate in candidates if not self._is_compat_task_entrypoint(candidate)
        ]

        reason_codes = list(repo_analysis.reason_codes)
        if excluded_compat_candidates:
            reason_codes.append("TASK_SPEC_NON_SCRIPT_ENTRYPOINTS_EXCLUDED")
        if selected:
            reason_codes.append("ENTRYPOINT_SELECTED_PRIMARY")
            if len(selected) > 1:
                reason_codes.append("ENTRYPOINT_SELECTED_BACKUP")
        else:
            reason_codes.append("REPO_ANALYSIS_NO_EXECUTABLE_CANDIDATE")

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
        observers = self._metric_observers_for(required_metrics)
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
            parsers=self._metric_parsers_for(required_metrics),
            normalization=self._metric_normalization(required_metrics),
            reason_codes=[] if selected else ["REPO_ANALYSIS_NO_EXECUTABLE_CANDIDATE"],
        )

        output = TaskCompileOutput(task_spec=task_spec, metric_contract=metric_contract)
        self.artifacts.write_json("task/task_spec.json", output.task_spec.model_dump())
        self.artifacts.write_json("task/metric_contract.json", output.metric_contract.model_dump())

        # Ensure top-5 and path validity invariants.
        for e in output.task_spec.entrypoints:
            if not (repo_dir / e.path).exists():
                raise ValueError(f"Entrypoint does not exist: {e.path}")

        return {
            "task_spec": output.task_spec.model_dump(),
            "metric_contract": output.metric_contract.model_dump(),
        }
