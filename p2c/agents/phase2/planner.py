"""PlannerAgent — reads Phase 1 artifacts + repo and produces an ExecutionPlan."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.agents.phase2.local_prompt_templates import (
    PLANNER_SYSTEM_PROMPT,
    build_planner_user_prompt,
)
from p2c.schemas import Entrypoint, ExecutionFailure, ExecutionPlan

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

_MIN_EXECUTION_STEP_TIMEOUT_SEC = 7200
_DEFAULT_PHASE2_MAX_CLAIMS = 80
_DEFAULT_PHASE2_MAX_EXPECTED_RESULTS = 80
_DEFAULT_PHASE2_MAX_ENTRYPOINTS = 24
_DEFAULT_PHASE2_MAX_DEP_PROFILES = 16
_DEFAULT_PHASE2_MAX_TASKS = 16
_DEFAULT_PHASE2_MAX_METRIC_PARSERS = 32
_DEFAULT_PHASE2_REPO_TREE_ENTRIES = 300
_DEFAULT_PHASE2_README_CHARS = 5000
_DEFAULT_PHASE2_DEP_FILE_CHARS = 2000
_DEFAULT_PHASE2_RAG_TOP_K = 8
_DEFAULT_PHASE2_RAG_CHARS = 6000
_DEFAULT_PHASE2_FAILURE_CHARS = 6000

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


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, int(raw.strip()))
    except ValueError:
        return default


def _cap_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars].rstrip() + f"\n...[truncated {omitted} chars]"


def _prompt_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


class PlannerAgent(BaseAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="planner", *args, **kwargs)

    @staticmethod
    def _prompt_limits() -> dict[str, int]:
        return {
            "max_claims": _env_int("P2C_PHASE2_MAX_CLAIMS", _DEFAULT_PHASE2_MAX_CLAIMS),
            "max_expected_results": _env_int(
                "P2C_PHASE2_MAX_EXPECTED_RESULTS", _DEFAULT_PHASE2_MAX_EXPECTED_RESULTS,
            ),
            "max_entrypoints": _env_int("P2C_PHASE2_MAX_ENTRYPOINTS", _DEFAULT_PHASE2_MAX_ENTRYPOINTS),
            "max_dep_profiles": _env_int("P2C_PHASE2_MAX_DEP_PROFILES", _DEFAULT_PHASE2_MAX_DEP_PROFILES),
            "max_tasks": _env_int("P2C_PHASE2_MAX_TASKS", _DEFAULT_PHASE2_MAX_TASKS),
            "max_metric_parsers": _env_int(
                "P2C_PHASE2_MAX_METRIC_PARSERS", _DEFAULT_PHASE2_MAX_METRIC_PARSERS,
            ),
            "repo_tree_entries": _env_int(
                "P2C_PHASE2_REPO_TREE_ENTRIES", _DEFAULT_PHASE2_REPO_TREE_ENTRIES,
            ),
            "readme_chars": _env_int("P2C_PHASE2_README_CHARS", _DEFAULT_PHASE2_README_CHARS),
            "dep_file_chars": _env_int("P2C_PHASE2_DEP_FILE_CHARS", _DEFAULT_PHASE2_DEP_FILE_CHARS),
            "rag_top_k": _env_int("P2C_PHASE2_RAG_TOP_K", _DEFAULT_PHASE2_RAG_TOP_K),
            "rag_chars": _env_int("P2C_PHASE2_RAG_CHARS", _DEFAULT_PHASE2_RAG_CHARS),
            "failure_chars": _env_int("P2C_PHASE2_FAILURE_CHARS", _DEFAULT_PHASE2_FAILURE_CHARS),
        }

    @staticmethod
    def _truthy_claim_verifiable(row: dict[str, Any]) -> bool:
        if "code_verifiable" in row:
            return bool(row.get("code_verifiable"))
        return not bool(row.get("unverifiable_from_paper", False))

    @classmethod
    def _claim_priority(cls, row: dict[str, Any], index: int) -> tuple[int, int, int]:
        conditions = _as_dict(row.get("conditions"))
        is_primary = bool(conditions.get("is_primary"))
        claim_type = str(row.get("type") or "")
        has_metric_target = bool(row.get("metric")) and row.get("target") is not None
        code_verifiable = cls._truthy_claim_verifiable(row)
        if code_verifiable and claim_type == "result" and has_metric_target:
            bucket = 0
        elif code_verifiable and claim_type == "result":
            bucket = 1
        elif code_verifiable and claim_type == "config":
            bucket = 2
        else:
            bucket = 3
        return (bucket, 0 if is_primary else 1, index)

    @staticmethod
    def _compact_claim_row(row: dict[str, Any]) -> dict[str, Any]:
        conditions = _as_dict(row.get("conditions"))
        compact_conditions = {
            key: conditions[key]
            for key in (
                "experiment_id",
                "is_primary",
                "scope",
                "table_anchor",
                "dataset",
                "model",
                "split",
            )
            if key in conditions and conditions[key] not in (None, "", [], {})
        }
        compact: dict[str, Any] = {
            "claim_id": str(row.get("claim_id") or ""),
            "type": str(row.get("type") or "config"),
            "predicate": _cap_text(str(row.get("predicate") or ""), 500),
            "metric": row.get("metric"),
            "target": row.get("target"),
            "code_verifiable": bool(
                row.get("code_verifiable", not bool(row.get("unverifiable_from_paper", False)))
            ),
        }
        if compact_conditions:
            compact["conditions"] = compact_conditions
        evidence_set = [str(x) for x in _as_list(row.get("evidence_set")) if str(x).strip()]
        if evidence_set:
            compact["evidence_set"] = evidence_set[:2]
        return {k: v for k, v in compact.items() if v not in (None, "", [], {})}

    @classmethod
    def _compact_claims_ir(cls, claims_ir: dict[str, Any], *, max_claims: int) -> dict[str, Any]:
        claims = [row for row in _as_list(claims_ir.get("claims")) if isinstance(row, dict)]
        ranked = sorted(enumerate(claims), key=lambda item: cls._claim_priority(item[1], item[0]))
        selected_pairs = ranked[:max_claims]
        selected_indices = {idx for idx, _ in selected_pairs}
        omitted_claims = [row for idx, row in enumerate(claims) if idx not in selected_indices]
        omitted_metrics = sorted(
            {
                str(row.get("metric") or "").strip()
                for row in omitted_claims
                if str(row.get("metric") or "").strip()
            }
        )

        experiments: list[dict[str, Any]] = []
        for exp in _as_list(claims_ir.get("experiments")):
            if not isinstance(exp, dict):
                continue
            if len(experiments) >= 12:
                continue
            compact_exp = {
                "experiment_id": exp.get("experiment_id"),
                "name": _cap_text(str(exp.get("name") or ""), 160),
                "dataset": exp.get("dataset"),
                "table_anchor": exp.get("table_anchor"),
                "repo_coverage": exp.get("repo_coverage"),
                "repo_entrypoint": exp.get("repo_entrypoint"),
                "notes": _cap_text(str(exp.get("notes") or ""), 240) if exp.get("notes") else None,
            }
            experiments.append({k: v for k, v in compact_exp.items() if v not in (None, "", [], {})})

        selected_rows = [row for idx, row in enumerate(claims) if idx in selected_indices]
        return {
            "summary": {
                "total_claims": len(claims),
                "included_claims": len(selected_rows),
                "omitted_claims": max(0, len(claims) - len(selected_rows)),
                "selection": (
                    "planner context is capped; result claims with metrics/targets and primary claims are prioritized"
                ),
                "omitted_metrics": omitted_metrics[:40],
            },
            "experiments": experiments,
            "claims": [cls._compact_claim_row(row) for row in selected_rows],
            "reason_codes": _as_list(claims_ir.get("reason_codes")),
        }

    @staticmethod
    def _compact_entrypoint(row: dict[str, Any]) -> dict[str, Any]:
        compact = {
            "entrypoint_id": row.get("entrypoint_id"),
            "path": row.get("path"),
            "command": row.get("command"),
            "cwd": row.get("cwd", "."),
            "runtime": row.get("runtime"),
            "dependency_profile_id": row.get("dependency_profile_id"),
            "confidence": row.get("confidence"),
            "evidence": _cap_text(str(row.get("evidence") or ""), 220),
            "reason_codes": _as_list(row.get("reason_codes"))[:8],
            "path_resolution_mode": row.get("path_resolution_mode"),
            "derived_from_wrapper": row.get("derived_from_wrapper"),
        }
        return {k: v for k, v in compact.items() if v not in (None, "", [], {})}

    @staticmethod
    def _compact_task(row: dict[str, Any]) -> dict[str, Any]:
        compact = {
            "task_id": row.get("task_id"),
            "entrypoint": row.get("entrypoint"),
            "command": row.get("command"),
            "cwd": row.get("cwd", "."),
            "runtime": row.get("runtime"),
            "dependency_profile_id": row.get("dependency_profile_id"),
            "timeout_class": row.get("timeout_class"),
            "expected_metrics": _as_list(row.get("expected_metrics"))[:20],
            "hyperparams": _as_dict(row.get("hyperparams")),
            "confidence": row.get("confidence"),
            "evidence": _cap_text(str(row.get("evidence") or ""), 220),
            "reason_codes": _as_list(row.get("reason_codes"))[:8],
            "path_resolution_mode": row.get("path_resolution_mode"),
            "derived_from_wrapper": row.get("derived_from_wrapper"),
        }
        return {k: v for k, v in compact.items() if v not in (None, "", [], {})}

    @classmethod
    def _compact_task_spec(cls, task_spec: dict[str, Any], *, max_tasks: int) -> dict[str, Any]:
        tasks = [row for row in _as_list(task_spec.get("tasks")) if isinstance(row, dict)]
        entrypoints = [row for row in _as_list(task_spec.get("entrypoints")) if isinstance(row, dict)]
        return {
            "summary": {
                "total_tasks": len(tasks),
                "included_tasks": min(len(tasks), max_tasks),
                "omitted_tasks": max(0, len(tasks) - max_tasks),
            },
            "constraints": _as_dict(task_spec.get("constraints")),
            "tasks": [cls._compact_task(row) for row in tasks[:max_tasks]],
            "entrypoints": [cls._compact_entrypoint(row) for row in entrypoints[: min(max_tasks, 8)]],
            "run_matrix": _as_list(task_spec.get("run_matrix"))[:3],
            "selection_notes": [
                _cap_text(str(note), 240)
                for note in _as_list(task_spec.get("selection_notes"))[:8]
            ],
            "reason_codes": _as_list(task_spec.get("reason_codes")),
        }

    @staticmethod
    def _compact_dependency_profile(row: dict[str, Any]) -> dict[str, Any]:
        compact = {
            "profile_id": row.get("profile_id"),
            "ecosystem": row.get("ecosystem"),
            "manager": row.get("manager"),
            "cwd": row.get("cwd", "."),
            "manifest_paths": _as_list(row.get("manifest_paths"))[:6],
            "install_command": row.get("install_command"),
            "auto_bootstrap_supported": row.get("auto_bootstrap_supported"),
            "reason_codes": _as_list(row.get("reason_codes"))[:8],
        }
        return {k: v for k, v in compact.items() if v not in (None, "", [], {})}

    @classmethod
    def _compact_repo_analysis(
        cls,
        repo_analysis: dict[str, Any],
        *,
        max_entrypoints: int,
        max_dep_profiles: int,
    ) -> dict[str, Any]:
        entrypoints = [
            row for row in _as_list(repo_analysis.get("entrypoint_candidates")) if isinstance(row, dict)
        ]
        profiles = [
            row for row in _as_list(repo_analysis.get("dependency_profiles")) if isinstance(row, dict)
        ]
        primary_id = str(repo_analysis.get("primary_entrypoint_id") or "")
        selected_entrypoints = entrypoints[:max_entrypoints]
        if primary_id and all(str(row.get("entrypoint_id") or "") != primary_id for row in selected_entrypoints):
            for row in entrypoints:
                if str(row.get("entrypoint_id") or "") == primary_id:
                    selected_entrypoints = [row, *selected_entrypoints[: max(0, max_entrypoints - 1)]]
                    break

        referenced_profiles = {
            str(row.get("dependency_profile_id") or "")
            for row in selected_entrypoints
            if str(row.get("dependency_profile_id") or "").strip()
        }
        selected_profiles: list[dict[str, Any]] = []
        seen_profiles: set[str] = set()
        for row in profiles:
            profile_id = str(row.get("profile_id") or "")
            if profile_id in referenced_profiles and profile_id not in seen_profiles:
                selected_profiles.append(row)
                seen_profiles.add(profile_id)
        for row in profiles:
            if len(selected_profiles) >= max_dep_profiles:
                break
            profile_id = str(row.get("profile_id") or "")
            if profile_id not in seen_profiles:
                selected_profiles.append(row)
                seen_profiles.add(profile_id)

        return {
            "summary": {
                "total_entrypoint_candidates": len(entrypoints),
                "included_entrypoint_candidates": len(selected_entrypoints),
                "omitted_entrypoint_candidates": max(0, len(entrypoints) - len(selected_entrypoints)),
                "total_dependency_profiles": len(profiles),
                "included_dependency_profiles": len(selected_profiles),
                "omitted_dependency_profiles": max(0, len(profiles) - len(selected_profiles)),
            },
            "ecosystems": _as_list(repo_analysis.get("ecosystems")),
            "dependency_profiles": [cls._compact_dependency_profile(row) for row in selected_profiles],
            "entrypoint_candidates": [cls._compact_entrypoint(row) for row in selected_entrypoints],
            "primary_entrypoint_id": repo_analysis.get("primary_entrypoint_id"),
            "reason_codes": _as_list(repo_analysis.get("reason_codes")),
        }

    @staticmethod
    def _compact_metric_contract(metric_contract: dict[str, Any], *, max_parsers: int) -> dict[str, Any]:
        parsers = [row for row in _as_list(metric_contract.get("parsers")) if isinstance(row, dict)]
        selected: list[dict[str, Any]] = []
        per_metric_count: dict[str, int] = {}
        for row in parsers:
            metric_name = str(row.get("metric_name") or row.get("name") or "")
            if per_metric_count.get(metric_name, 0) >= 2:
                continue
            compact = {
                "name": row.get("name"),
                "regex": row.get("regex"),
                "metric_name": row.get("metric_name"),
                "transform": row.get("transform", "float"),
            }
            selected.append({k: v for k, v in compact.items() if v not in (None, "", [], {})})
            per_metric_count[metric_name] = per_metric_count.get(metric_name, 0) + 1
            if len(selected) >= max_parsers:
                break
        return {
            "summary": {
                "total_parsers": len(parsers),
                "included_parsers": len(selected),
                "omitted_parsers": max(0, len(parsers) - len(selected)),
            },
            "required_metrics": _as_list(metric_contract.get("required_metrics"))[:80],
            "parsers": selected,
            "normalization": _as_dict(metric_contract.get("normalization")),
            "reason_codes": _as_list(metric_contract.get("reason_codes")),
        }

    @classmethod
    def _compact_phase1_inputs(
        cls,
        *,
        claims_ir: dict[str, Any],
        task_spec: dict[str, Any],
        metric_contract: dict[str, Any],
        repo_analysis: dict[str, Any],
        limits: dict[str, int],
    ) -> dict[str, dict[str, Any]]:
        return {
            "claims_ir": cls._compact_claims_ir(claims_ir, max_claims=limits["max_claims"]),
            "task_spec": cls._compact_task_spec(task_spec, max_tasks=limits["max_tasks"]),
            "metric_contract": cls._compact_metric_contract(
                metric_contract, max_parsers=limits["max_metric_parsers"],
            ),
            "repo_analysis": cls._compact_repo_analysis(
                repo_analysis,
                max_entrypoints=limits["max_entrypoints"],
                max_dep_profiles=limits["max_dep_profiles"],
            ),
        }

    @classmethod
    def _backfill_expected_results(
        cls,
        data: dict[str, Any],
        claims_ir: dict[str, Any],
        *,
        max_expected_results: int,
    ) -> None:
        expected = data.get("expected_results")
        if not isinstance(expected, list):
            expected = []
            data["expected_results"] = expected
        existing_claim_ids = {
            str(row.get("claim_id") or "")
            for row in expected
            if isinstance(row, dict) and str(row.get("claim_id") or "").strip()
        }
        claims = [row for row in _as_list(claims_ir.get("claims")) if isinstance(row, dict)]
        ranked = sorted(enumerate(claims), key=lambda item: cls._claim_priority(item[1], item[0]))
        eligible_claim_ids = {
            str(row.get("claim_id") or "").strip()
            for _, row in ranked
            if (
                str(row.get("claim_id") or "").strip()
                and str(row.get("type") or "") == "result"
                and cls._truthy_claim_verifiable(row)
                and str(row.get("metric") or "").strip()
            )
        }
        for _, row in ranked:
            if len(expected) >= max_expected_results:
                break
            claim_id = str(row.get("claim_id") or "").strip()
            metric_name = str(row.get("metric") or "").strip()
            if (
                not claim_id
                or claim_id in existing_claim_ids
                or str(row.get("type") or "") != "result"
                or not cls._truthy_claim_verifiable(row)
                or not metric_name
            ):
                continue
            target = row.get("target")
            expected.append(
                {
                    "claim_id": claim_id,
                    "metric_name": metric_name,
                    "target_value": target if isinstance(target, (int, float)) else None,
                    "extraction_hint": "collect this metric from stdout or produced metric files",
                }
            )
            existing_claim_ids.add(claim_id)
        covered_eligible = existing_claim_ids.intersection(eligible_claim_ids)
        if len(eligible_claim_ids) > len(covered_eligible):
            reason_codes = data.setdefault("reason_codes", [])
            if isinstance(reason_codes, list) and "EXPECTED_RESULTS_CAPPED" not in reason_codes:
                reason_codes.append("EXPECTED_RESULTS_CAPPED")

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
        limits = self._prompt_limits()

        claims_ir = self.artifacts.read_json("fingerprint/claims_ir.json")
        task_spec = self.artifacts.read_json("task/task_spec.json")
        metric_contract = self.artifacts.read_json("task/metric_contract.json")
        repo_analysis = self.artifacts.read_json("task/repo_analysis.json")
        phase1_prompt_inputs = self._compact_phase1_inputs(
            claims_ir=claims_ir,
            task_spec=task_spec,
            metric_contract=metric_contract,
            repo_analysis=repo_analysis,
            limits=limits,
        )

        repo_tree = self._repo_tree(repo_dir, max_entries=limits["repo_tree_entries"])
        readme = self._read_readme(repo_dir, max_chars=limits["readme_chars"])
        dep_files = self._read_dependency_files(repo_dir, max_chars=limits["dep_file_chars"])

        failure_ctx = None
        if failures:
            failure_ctx = _cap_text(_prompt_json(
                [f.model_dump() if hasattr(f, "model_dump") else f for f in failures],
            ), limits["failure_chars"])

        # RAG: retrieve relevant code context for claims
        rag_context = ""
        code_index = ctx.get("_code_index")
        if code_index is None:
            try:
                from p2c.rag.index import CodeIndex
                index_data = self.artifacts.read_json("task/code_index.json")
                if index_data and "chunks" in index_data:
                    code_index = CodeIndex.deserialize(index_data)
            except Exception:  # noqa: BLE001
                pass
        if code_index is not None:
            try:
                from p2c.rag.query import retrieve_for_claims
                claims_list = phase1_prompt_inputs["claims_ir"].get("claims", [])
                rag_context = retrieve_for_claims(
                    code_index,
                    claims_list if isinstance(claims_list, list) else [],
                    top_k=limits["rag_top_k"],
                    max_chars=limits["rag_chars"],
                )
            except Exception:  # noqa: BLE001
                pass

        omitted_claims = phase1_prompt_inputs["claims_ir"].get("summary", {}).get("omitted_claims", 0)
        omitted_entrypoints = (
            phase1_prompt_inputs["repo_analysis"].get("summary", {}).get("omitted_entrypoint_candidates", 0)
        )
        if omitted_claims or omitted_entrypoints:
            self.log(
                "PROGRESS",
                "using compact phase 1 prompt context "
                f"(omitted_claims={omitted_claims}, omitted_entrypoints={omitted_entrypoints})",
            )

        user_prompt = build_planner_user_prompt(
            claims_ir_json=_prompt_json(phase1_prompt_inputs["claims_ir"]),
            task_spec_json=_prompt_json(phase1_prompt_inputs["task_spec"]),
            metric_contract_json=_prompt_json(phase1_prompt_inputs["metric_contract"]),
            repo_analysis_json=_prompt_json(phase1_prompt_inputs["repo_analysis"]),
            repo_tree=repo_tree,
            readme_content=readme,
            dependency_files=dep_files,
            failure_context=failure_ctx,
            env_name=env_name,
            budget_sec=budget_sec,
            rag_context=rag_context,
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
        raw_expected = data.get("expected_results") or []
        if isinstance(raw_expected, list):
            data["expected_results"] = [
                item for item in raw_expected
                if isinstance(item, dict) and isinstance(item.get("metric_name"), str) and item.get("metric_name").strip()
            ]
        else:
            data["expected_results"] = []
        self._backfill_expected_results(
            data,
            claims_ir,
            max_expected_results=limits["max_expected_results"],
        )

        # Drop expected_results entries the LLM failed to map to a metric
        # (metric_name is required by the ExpectedResult schema).
        raw_expected = data.get("expected_results") or []
        if isinstance(raw_expected, list):
            data["expected_results"] = [
                item for item in raw_expected
                if isinstance(item, dict) and isinstance(item.get("metric_name"), str) and item.get("metric_name").strip()
            ]

        # Validate & persist
        plan = ExecutionPlan(**data)
        self._sanitize_plan(plan, repo_dir, repo_analysis=repo_analysis, task_spec=task_spec)
        self._validate_plan(plan, repo_dir)
        self.artifacts.write_json("execution/execution_plan.json", plan.model_dump())
        self.log("DONE", f"plan v{plan.plan_version}: {len(plan.execution_steps)} steps, "
                         f"{len(plan.pip_dependencies)} pip deps, env={plan.env_name}")
        return {"plan": plan}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _repo_tree(repo_dir: str, max_entries: int = _DEFAULT_PHASE2_REPO_TREE_ENTRIES) -> str:
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
    def _read_readme(repo_dir: str, max_chars: int = _DEFAULT_PHASE2_README_CHARS) -> str:
        for name in ("README.md", "readme.md", "README.rst", "README.txt", "README"):
            p = Path(repo_dir) / name
            if p.exists():
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    return _cap_text(text, max_chars)
                except Exception:  # noqa: BLE001
                    pass
        return "(no README found)"

    @staticmethod
    def _read_dependency_files(repo_dir: str, max_chars: int = _DEFAULT_PHASE2_DEP_FILE_CHARS) -> dict[str, str]:
        result: dict[str, str] = {}
        for name in _DEP_FILE_NAMES:
            p = Path(repo_dir) / name
            if p.exists():
                try:
                    result[name] = _cap_text(p.read_text(encoding="utf-8", errors="ignore"), max_chars)
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

    @classmethod
    def _sanitize_plan(
        cls,
        plan: ExecutionPlan,
        repo_dir: str,
        *,
        repo_analysis: dict[str, Any] | None = None,
        task_spec: dict[str, Any] | None = None,
    ) -> None:
        """Deterministically fix risky LLM planner outputs before execution."""
        repo_root = Path(repo_dir)
        candidate_index = cls._candidate_index(repo_root, repo_analysis or {}, task_spec or {})
        for step in plan.execution_steps:
            rewritten = cls._rewrite_help_probe(step.command, repo_root)
            if rewritten:
                step.command = rewritten
            cls._normalize_step_timeout(step)
            step.fallback_commands = cls._sanitize_fallbacks(step.command, step.fallback_commands)
            cls._rewrite_step_from_candidates(step, repo_root, candidate_index)
            step.required_artifacts = cls._normalize_artifact_paths(step.required_artifacts, step.cwd, repo_root)
            step.produced_artifacts = cls._normalize_artifact_paths(step.produced_artifacts, step.cwd, repo_root)

    @staticmethod
    def _normalize_step_timeout(step: Any) -> None:
        """Keep executable/reproduction steps from being killed mid-training."""
        if getattr(step, "is_setup", False):
            return
        try:
            min_timeout = int(os.getenv("P2C_MIN_EXEC_TIMEOUT_SEC", str(_MIN_EXECUTION_STEP_TIMEOUT_SEC)))
        except ValueError:
            min_timeout = _MIN_EXECUTION_STEP_TIMEOUT_SEC
        if int(getattr(step, "timeout_sec", 0) or 0) < min_timeout:
            step.timeout_sec = min_timeout

    @staticmethod
    def _extract_python_script(command: str) -> str | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
            name, _, value = tokens[0].partition("=")
            if name.isidentifier() and value:
                tokens = tokens[1:]
                continue
            break
        if len(tokens) < 2 or tokens[0] != "python":
            return None
        script = tokens[1]
        if script.endswith(".py"):
            return script
        return None

    @classmethod
    def _rewrite_help_probe(cls, command: str, repo_root: Path) -> str | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None
        assignments: list[str] = []
        while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
            name, _, value = tokens[0].partition("=")
            if name.isidentifier() and value:
                assignments.append(tokens.pop(0))
                continue
            break
        if len(tokens) != 3 or tokens[0] != "python" or tokens[2] not in {"--help", "-h"}:
            return None
        script = tokens[1]
        if not script.endswith(".py") or not (repo_root / script).is_file():
            return None
        prefix = f"{' '.join(assignments)} " if assignments else ""
        return (
            f"{prefix}python -c 'from pathlib import Path; "
            f"p = Path({script!r}); "
            "print(p.read_text(encoding=\"utf-8\", errors=\"ignore\")[:6000])'"
        )

    @classmethod
    def _sanitize_fallbacks(cls, command: str, fallbacks: list[str]) -> list[str]:
        primary_script = cls._extract_python_script(command)
        if primary_script is None:
            return fallbacks
        sanitized: list[str] = []
        for fallback in fallbacks:
            fallback_script = cls._extract_python_script(fallback)
            if fallback_script == primary_script:
                sanitized.append(fallback)
        return sanitized

    @staticmethod
    def _iter_entrypoint_rows(repo_analysis: dict[str, Any], task_spec: dict[str, Any]) -> list[Entrypoint]:
        rows: list[Entrypoint] = []
        for row in repo_analysis.get("entrypoint_candidates", []):
            if isinstance(row, dict):
                rows.append(Entrypoint(**row))
        for task in task_spec.get("tasks", []):
            if not isinstance(task, dict):
                continue
            rows.append(
                Entrypoint(
                    entrypoint_id=task.get("task_id"),
                    path=str(task.get("entrypoint") or ""),
                    command=str(task.get("command") or ""),
                    cwd=str(task.get("cwd") or "."),
                    runtime=str(task.get("runtime") or "python"),
                    dependency_profile_id=task.get("dependency_profile_id"),
                    confidence=float(task.get("confidence") or 0.0),
                    evidence=str(task.get("evidence") or ""),
                    reason_codes=list(task.get("reason_codes") or []),
                    path_resolution_mode=task.get("path_resolution_mode"),
                    derived_from_wrapper=task.get("derived_from_wrapper"),
                )
            )
        return rows

    @staticmethod
    def _candidate_rewrite_score(candidate: Entrypoint) -> tuple[int, int, int, int, float]:
        reason_codes = {str(code) for code in candidate.reason_codes}
        evidence = str(candidate.evidence or "").lower()
        is_task_spec = 1 if str(candidate.entrypoint_id or "").startswith("task_") else 0
        if "README_WORKFLOW_PRIMARY" in reason_codes:
            readme_priority = 2
        elif "README_VERIFIED_COMMAND" in reason_codes or "readme verified" in evidence:
            readme_priority = 1
        else:
            readme_priority = 0
        return (
            is_task_spec,
            readme_priority,
            1 if candidate.derived_from_wrapper else 0,
            1 if candidate.cwd != "." else 0,
            float(candidate.confidence),
        )

    @staticmethod
    def _resolve_rel(path_value: str, *, cwd: str, repo_root: Path) -> str | None:
        raw = str(path_value or "").strip()
        if not raw:
            return None
        resolved = (repo_root / cwd / raw).resolve()
        try:
            return resolved.relative_to(repo_root.resolve()).as_posix()
        except Exception:  # noqa: BLE001
            return None

    @classmethod
    def _extract_command_target(cls, command: str) -> tuple[str | None, str | None]:
        try:
            tokens = shlex.split(command)
        except ValueError:
            return None, None
        while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
            name, _, value = tokens[0].partition("=")
            if name.isidentifier() and value:
                tokens = tokens[1:]
                continue
            break
        if not tokens:
            return None, None
        head = tokens[0]
        if head in {"python", "python3"} and len(tokens) >= 2 and tokens[1].endswith(".py"):
            return "python", tokens[1]
        if head in {"bash", "sh"} and len(tokens) >= 2 and tokens[1].endswith(".sh"):
            return "shell", tokens[1]
        if head == "make":
            return "make", "Makefile"
        if (head.startswith("./") or head.startswith("../")) and head.endswith(".sh"):
            return "shell", head
        return None, None

    @classmethod
    def _candidate_index(
        cls,
        repo_root: Path,
        repo_analysis: dict[str, Any],
        task_spec: dict[str, Any],
    ) -> dict[tuple[str, str], Entrypoint]:
        indexed: dict[tuple[str, str], Entrypoint] = {}
        for candidate in cls._iter_entrypoint_rows(repo_analysis, task_spec):
            if not candidate.path:
                continue
            key = (candidate.path, candidate.runtime)
            current = indexed.get(key)
            if current is None:
                indexed[key] = candidate
                continue
            current_score = cls._candidate_rewrite_score(current)
            candidate_score = cls._candidate_rewrite_score(candidate)
            if candidate_score > current_score:
                indexed[key] = candidate
        return indexed

    @classmethod
    def _rewrite_step_from_candidates(
        cls,
        step,
        repo_root: Path,
        candidate_index: dict[tuple[str, str], Entrypoint],
    ) -> None:
        runtime, target = cls._extract_command_target(step.command)
        if runtime is None or target is None:
            return
        resolved_target = cls._resolve_rel(target, cwd=step.cwd or ".", repo_root=repo_root)
        if not resolved_target:
            return
        candidate = candidate_index.get((resolved_target, runtime))
        if candidate is None:
            return
        step.command = candidate.command
        step.cwd = candidate.cwd
        step.path_resolution_mode = candidate.path_resolution_mode or "candidate_rewrite"
        step.derived_from_wrapper = candidate.derived_from_wrapper

    @classmethod
    def _normalize_artifact_paths(cls, paths: list[str], cwd: str, repo_root: Path) -> list[str]:
        normalized: list[str] = []
        for raw in paths:
            rel = cls._resolve_rel(raw, cwd=cwd or ".", repo_root=repo_root)
            normalized.append(rel or str(raw))
        return normalized
