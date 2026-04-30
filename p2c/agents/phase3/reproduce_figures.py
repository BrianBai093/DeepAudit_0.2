"""LLM-assisted reproduction of paper figures/tables from Phase 2 evidence."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.agents.phase3.execution_summary_evidence import (
    PHASE2_PACKAGE_PATH,
    PHASE2_RESULTS_PATH,
    load_phase2_execution_package,
)
from p2c.agents.phase3.execution_log_evidence import LOG_EVIDENCE_PATH
from p2c.schemas import (
    ReproducedFigure,
    ReproducedFiguresDoc,
    SkippedReproducedTarget,
)


FIGURE_REPRO_SYSTEM_PROMPT = """\
You are an ML reproducibility figure-planning agent.

Use only the provided Phase 1 visual description and canonical Phase 2 execution evidence.
Return strict JSON matching the requested plot_spec schema.

Rules:
- Use a graded match_level: EXACT, PARTIAL, RELATED, or NO_EVIDENCE.
- If algorithm, dataset, model family, or metric evidence is incomplete or only adjacent, you may still return
  decision="PLOT", but match_level MUST be PARTIAL or RELATED and the title/comparison_note MUST say so.
- Never claim BP evidence reproduces PEPITA, FA, DFA, DRTP, or any other algorithm. It may only be shown as
  available related/partial evidence when clearly labeled.
- If there are only skip/error/config logs and no numeric values, return either decision="SKIP" or a text-panel
  with match_level="NO_EVIDENCE".
- Prefer executable Phase 2 metrics and stdout curves over prose summaries.
- Produce a compact plot_spec for the reproduced side only; the host will compose it with the paper crop.
- Include every source file or attempt id used in evidence_sources.
"""


FIGURE_CODEGEN_SYSTEM_PROMPT = """\
You are a restricted matplotlib code generator for one reproduced figure panel.

Return strict JSON with a single field named code.
The code must:
- Use only the variables payload and output_path provided by the host.
- Use matplotlib with the Agg backend and optionally numpy/math/statistics/json.
- Save exactly one PNG to output_path.
- Avoid all filesystem reads, network access, subprocesses, shell calls, eval/exec, imports outside plotting/numeric modules.
"""


PLOT_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string"},
        "chart_type": {"type": "string"},
        "title": {"type": "string"},
        "x_label": {"type": "string"},
        "y_label": {"type": "string"},
        "series": {"type": "array"},
        "table": {"type": "object"},
        "unit": {"type": ["string", "null"]},
        "normalization": {"type": ["string", "null"]},
        "evidence_sources": {"type": "array"},
        "comparison_note": {"type": "string"},
        "match_level": {"type": "string"},
        "matched_scope": {"type": "object"},
        "coverage_note": {"type": "string"},
        "confidence": {"type": "number"},
        "reason_codes": {"type": "array"},
    },
    "required": [
        "decision",
        "chart_type",
        "title",
        "x_label",
        "y_label",
        "series",
        "table",
        "unit",
        "normalization",
        "evidence_sources",
        "comparison_note",
        "match_level",
        "matched_scope",
        "coverage_note",
        "confidence",
        "reason_codes",
    ],
}

CODEGEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
}

SUPPORTED_SPEC_CHARTS = {"line", "bar", "scatter", "table", "heatmap", "text-panel"}
RESULT_KEYWORDS = {
    "accuracy",
    "acc",
    "auc",
    "curve",
    "loss",
    "metric",
    "performance",
    "precision",
    "recall",
    "result",
    "score",
    "table",
    "test",
    "train",
    "training",
}
ALGORITHM_ALIASES = {
    "bp": {"bp", "backprop", "back-prop", "back propagation", "back-propagation"},
    "pepita": {"pepita", "erin"},
    "fa": {"fa", "feedback alignment", "feedback-alignment"},
    "dfa": {"dfa", "direct feedback alignment", "direct feedback-alignment"},
    "drtp": {"drtp"},
    "rp": {"rp", "random projection"},
}


class ReproduceFiguresAgent(BaseAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="reproduce_figures", *args, **kwargs)

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        visual_targets_doc = self._safe_read("fingerprint/visual_targets.json")
        visual_elements_doc = self._safe_read("fingerprint/visual_elements.json")
        claims_doc = self._safe_read("fingerprint/claims_ir.json")
        metrics_doc = self._safe_read("results/metrics.json")
        verdict_doc = self._safe_read("results/verdict.json")
        parsed_evidence_doc = self._safe_read("results/parsed_evidence.json")
        phase2_package = load_phase2_execution_package(self.artifacts)
        phase2_results_text = self._safe_read_text(PHASE2_RESULTS_PATH, limit=12000)
        execution_log_evidence_doc = self._safe_read(LOG_EVIDENCE_PATH)

        figures_dir = self.artifacts.path("results/figures").resolve()
        figures_dir.mkdir(parents=True, exist_ok=True)

        elements_by_id = _elements_by_id(visual_elements_doc)
        targets = _load_visual_targets(visual_targets_doc, visual_elements_doc)
        attempts = _merge_attempts_with_log_evidence(
            _collect_phase2_attempts(phase2_package, self.artifacts),
            execution_log_evidence_doc,
        )

        reproduced: list[ReproducedFigure] = []
        skipped: list[SkippedReproducedTarget] = []

        for target in targets:
            bundle = _build_evidence_bundle(
                target=target,
                element=elements_by_id.get(str(target.get("element_id") or ""), {}),
                claims_doc=claims_doc,
                verdict_doc=verdict_doc,
                parsed_evidence_doc=parsed_evidence_doc,
                metrics_doc=metrics_doc,
                phase2_package=phase2_package,
                phase2_results_text=phase2_results_text,
                attempts=attempts,
            )
            decision = _target_reproduction_decision(bundle)
            if not decision["eligible"]:
                skipped.append(_skipped_from_bundle(bundle, decision["reason"], decision["reason_codes"]))
                continue

            fig = self._reproduce_bundle(bundle, figures_dir)
            if fig.reproduction_status == "SKIPPED":
                skipped.append(
                    SkippedReproducedTarget(
                        element_id=fig.element_id,
                        visual_anchor=fig.visual_anchor,
                        skip_reason=fig.comparison_notes,
                        evidence_sources=fig.evidence_sources,
                        reason_codes=fig.reason_codes,
                    )
                )
            else:
                reproduced.append(fig)

        doc = ReproducedFiguresDoc(
            figures=reproduced,
            skipped_targets=skipped,
            reason_codes=["LLM_ASSISTED_FIGURE_REPRODUCTION"],
        )
        self.artifacts.write_json("results/reproduced_figures.json", doc.model_dump())
        self.log("DONE", f"Generated {len(reproduced)} comparison figures; skipped {len(skipped)} targets")
        return {"figures": doc.model_dump()}

    def _safe_read(self, path: str) -> dict[str, Any]:
        try:
            payload = self.artifacts.read_json(path)
            return payload if isinstance(payload, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def _safe_read_text(self, path: str, *, limit: int | None = None) -> str:
        try:
            file_path = self.artifacts.path(path)
            if not file_path.exists():
                return ""
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            return text[:limit] if limit else text
        except Exception:  # noqa: BLE001
            return ""

    def _reproduce_bundle(self, bundle: dict[str, Any], figures_dir: Path) -> ReproducedFigure:
        element_id = str(bundle.get("element_id") or "visual")
        reference_path = bundle.get("reference_image_path")
        right_rel = f"results/figures/{element_id}_reproduced.png"
        comparison_rel = f"results/figures/{element_id}_comparison.png"
        right_path = self.artifacts.path(right_rel).resolve()
        comparison_path = self.artifacts.path(comparison_rel).resolve()

        plot_spec, llm_note = self._plan_plot_spec(bundle)
        if not plot_spec:
            plot_spec = _deterministic_plot_spec(bundle)
            llm_note = "LLM unavailable; deterministic fallback used." if plot_spec else llm_note

        if not plot_spec:
            return _failed_figure(
                bundle,
                reason="No renderable Phase 2 evidence matched this visual target.",
                reason_codes=["NO_RENDERABLE_EVIDENCE"],
            )
        plot_spec = _ensure_plot_spec_match_fields(plot_spec, bundle)

        decision = str(plot_spec.get("decision") or "PLOT").upper()
        if decision == "SKIP":
            return ReproducedFigure(
                element_id=element_id,
                visual_anchor=str(bundle.get("visual_anchor") or ""),
                reference_image_path=reference_path,
                reproduced_image_path=None,
                image_path="",
                comparison_notes=str(plot_spec.get("comparison_note") or "LLM skipped this target."),
                evidence_sources=_string_list(plot_spec.get("evidence_sources")),
                reproduction_status="SKIPPED",
                plot_spec=plot_spec,
                llm_decision_summary=llm_note,
                match_level=str(plot_spec.get("match_level") or bundle.get("match_level") or "NO_EVIDENCE"),
                matched_scope=plot_spec.get("matched_scope") if isinstance(plot_spec.get("matched_scope"), dict) else bundle.get("matched_scope", {}),
                coverage_note=str(plot_spec.get("coverage_note") or bundle.get("coverage_note") or ""),
                reason_codes=["LLM_SKIPPED_TARGET", *_string_list(plot_spec.get("reason_codes"))],
            )

        rendered = _render_plot_spec(plot_spec, right_path)
        code_path: str | None = None
        matplotlib_code = ""
        reason_codes = ["LLM_PLOT_SPEC_RENDERED"] if rendered else []

        if not rendered:
            code_result = self._render_with_codegen(bundle, plot_spec, right_path)
            rendered = code_result["success"]
            code_path = code_result.get("code_path")
            matplotlib_code = code_result.get("code", "")
            reason_codes.append(code_result.get("reason_code", "CODEGEN_RENDER_FAILED"))

        if not rendered:
            return _failed_figure(
                bundle,
                reason="Figure planning succeeded but rendering failed.",
                reason_codes=reason_codes or ["RENDER_FAILED"],
                plot_spec=plot_spec,
                llm_note=llm_note,
                code_path=code_path,
                code=matplotlib_code,
            )

        composed = _compose_comparison_image(
            reference_path=_resolve_artifact_path(self.artifacts, reference_path),
            reproduced_path=right_path,
            output_path=comparison_path,
            title=str(plot_spec.get("title") or bundle.get("caption") or element_id),
            note=str(plot_spec.get("comparison_note") or ""),
        )
        if not composed:
            return _failed_figure(
                bundle,
                reason="Reproduced panel rendered but comparison composition failed.",
                reason_codes=["COMPARISON_COMPOSITION_FAILED"],
                plot_spec=plot_spec,
                llm_note=llm_note,
                code_path=code_path,
                code=matplotlib_code,
            )

        evidence_sources = _dedupe_strings(
            [*_string_list(plot_spec.get("evidence_sources")), *bundle.get("evidence_sources", [])]
        )
        return ReproducedFigure(
            element_id=element_id,
            visual_anchor=str(bundle.get("visual_anchor") or ""),
            reference_image_path=reference_path,
            reproduced_image_path=right_rel,
            image_path=comparison_rel,
            comparison_notes=str(plot_spec.get("comparison_note") or "Generated from Phase 2 execution evidence."),
            evidence_sources=evidence_sources,
            reproduction_status="REPRODUCED",
            plot_spec=plot_spec,
            matplotlib_code=matplotlib_code,
            code_path=code_path,
            llm_decision_summary=llm_note,
            match_level=str(plot_spec.get("match_level") or bundle.get("match_level") or "EXACT"),
            matched_scope=plot_spec.get("matched_scope") if isinstance(plot_spec.get("matched_scope"), dict) else bundle.get("matched_scope", {}),
            coverage_note=str(plot_spec.get("coverage_note") or bundle.get("coverage_note") or ""),
            reason_codes=_dedupe_strings([*reason_codes, *_string_list(plot_spec.get("reason_codes"))]),
        )

    def _plan_plot_spec(self, bundle: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        if self.llm is None:
            return None, "LLM client unavailable."
        prompt = _build_plot_spec_prompt(bundle)
        try:
            data, err = self.safe_chat_json(schema=PLOT_SPEC_SCHEMA, system=FIGURE_REPRO_SYSTEM_PROMPT, user=prompt)
        except Exception as exc:  # noqa: BLE001
            return None, f"LLM plot planning failed: {exc}"
        if not data:
            return None, f"LLM plot planning unavailable: {err}"
        return _normalize_plot_spec(data), "LLM selected Phase 2 evidence and plot specification."

    def _render_with_codegen(
        self,
        bundle: dict[str, Any],
        plot_spec: dict[str, Any],
        output_path: Path,
    ) -> dict[str, Any]:
        if self.llm is None:
            return {"success": False, "reason_code": "CODEGEN_LLM_UNAVAILABLE"}
        prompt = _build_codegen_prompt(bundle, plot_spec, str(output_path))
        try:
            data, err = self.safe_chat_json(schema=CODEGEN_SCHEMA, system=FIGURE_CODEGEN_SYSTEM_PROMPT, user=prompt)
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "reason_code": "CODEGEN_LLM_FAILED", "error": str(exc)}
        if not data or not str(data.get("code") or "").strip():
            return {"success": False, "reason_code": "CODEGEN_LLM_UNAVAILABLE", "error": err}

        code = str(data.get("code") or "")
        ok, reason = _validate_codegen_code(code)
        if not ok:
            return {"success": False, "reason_code": "CODEGEN_REJECTED", "code": code, "error": reason}

        code_rel = f"results/figures/{bundle.get('element_id')}_codegen.py"
        full_code = _wrap_codegen_code(code, bundle, plot_spec, str(output_path))
        self.artifacts.write_text(code_rel, full_code)
        success = self._run_python_code(full_code, output_path)
        return {
            "success": success,
            "reason_code": "CODEGEN_RENDERED" if success else "CODEGEN_RENDER_FAILED",
            "code": code,
            "code_path": code_rel,
        }

    def _run_python_code(self, code: str, output_path: Path) -> bool:
        try:
            if output_path.exists():
                output_path.unlink()
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.artifacts.path(".").resolve()),
            )
            if proc.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                return True
            diagnostic = (proc.stderr or proc.stdout or "")[-1000:]
            self.log("PROGRESS", f"figure codegen failed: {diagnostic}")
            return False
        except Exception as exc:  # noqa: BLE001
            self.log("PROGRESS", f"figure codegen execution error: {exc}")
            return False


def _elements_by_id(visual_elements_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("element_id")): row
        for row in visual_elements_doc.get("elements", [])
        if isinstance(row, dict) and row.get("element_id")
    }


def _load_visual_targets(
    visual_targets_doc: dict[str, Any],
    visual_elements_doc: dict[str, Any],
) -> list[dict[str, Any]]:
    targets = [
        dict(row)
        for row in visual_targets_doc.get("visual_targets", [])
        if isinstance(row, dict) and row.get("element_id")
    ]
    if targets:
        return targets
    out: list[dict[str, Any]] = []
    for elem in visual_elements_doc.get("elements", []):
        if not isinstance(elem, dict) or not elem.get("element_id"):
            continue
        out.append(
            {
                "element_id": elem.get("element_id"),
                "visual_anchor": elem.get("visual_anchor") or elem.get("element_id"),
                "element_type": elem.get("element_type") or "figure",
                "chart_type": elem.get("chart_type"),
                "caption": elem.get("caption") or "",
                "page": elem.get("page"),
                "reference_image_path": elem.get("crop_path") or elem.get("raw_page_image"),
                "axis_labels": elem.get("axis_labels", {}),
                "legend_entries": elem.get("legend_entries", []),
                "series_names": [s.get("name") for s in elem.get("data_series", []) if isinstance(s, dict)],
                "metric_names": [],
                "model_names": elem.get("model_names", []),
                "sampling_strategy": elem.get("sampling_strategy"),
                "semantic_summary": elem.get("caption") or "",
                "reconstruction_instructions": [],
                "associated_claim_ids": elem.get("associated_claim_ids", []),
                "reason_codes": ["VISUAL_ELEMENT_FALLBACK_TARGET"],
            }
        )
    return out


def _build_evidence_bundle(
    *,
    target: dict[str, Any],
    element: dict[str, Any],
    claims_doc: dict[str, Any],
    verdict_doc: dict[str, Any],
    parsed_evidence_doc: dict[str, Any],
    metrics_doc: dict[str, Any],
    phase2_package: dict[str, Any],
    phase2_results_text: str,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    element_id = str(target.get("element_id") or element.get("element_id") or "")
    visual_anchor = str(target.get("visual_anchor") or element.get("visual_anchor") or element_id)
    caption = str(target.get("caption") or element.get("caption") or "")
    reference_image_path = target.get("reference_image_path") or element.get("crop_path") or element.get("raw_page_image")
    target_algorithms = _algorithms_for_visual(target, element)
    target_claims = _claims_for_visual(target, element, claims_doc)
    verdicts_by_id = {
        str(row.get("claim_id")): row
        for row in verdict_doc.get("claim_verdicts", [])
        if isinstance(row, dict) and row.get("claim_id")
    }
    parsed_by_claim = {
        str(row.get("claim_id")): row
        for row in parsed_evidence_doc.get("claim_evidence", [])
        if isinstance(row, dict) and row.get("claim_id")
    }
    claim_rows = []
    source_attempt_ids: set[str] = set()
    source_experiment_ids: set[str] = set()
    for claim in target_claims:
        claim_id = str(claim.get("claim_id") or "")
        verdict = verdicts_by_id.get(claim_id, {})
        parsed = parsed_by_claim.get(claim_id, {})
        for record in parsed.get("matched_records", []) if isinstance(parsed, dict) else []:
            if not isinstance(record, dict):
                continue
            run_id = str(record.get("run_id") or "")
            exp_id = str(record.get("experiment_id") or "")
            if run_id:
                source_attempt_ids.add(run_id)
            if exp_id:
                source_experiment_ids.add(exp_id)
        conditions = claim.get("conditions", {}) if isinstance(claim, dict) else {}
        if isinstance(conditions, dict) and conditions.get("experiment_id"):
            source_experiment_ids.add(str(conditions.get("experiment_id")))
        claim_rows.append(
            {
                "claim_id": claim_id,
                "predicate": claim.get("predicate") or claim.get("metric") or claim_id,
                "metric": claim.get("metric"),
                "target": verdict.get("target_value", claim.get("target")),
                "reproduced": verdict.get("compared_value"),
                "status": verdict.get("status"),
                "detail": verdict.get("detail", ""),
                "matched_records": parsed.get("matched_records", []) if isinstance(parsed, dict) else [],
            }
        )

    for experiment in phase2_package.get("experiments", []) if isinstance(phase2_package, dict) else []:
        if not isinstance(experiment, dict):
            continue
        exp_anchor = str(experiment.get("table_anchor") or "")
        refs = " ".join(str(x) for x in experiment.get("paper_target_refs", []))
        if visual_anchor and (visual_anchor == exp_anchor or visual_anchor in refs):
            source_experiment_ids.add(str(experiment.get("experiment_id") or ""))

    visual_text = " ".join(
        [
            caption,
            str(target.get("semantic_summary") or ""),
            " ".join(str(x) for x in target.get("metric_names", [])),
            " ".join(str(x) for x in target.get("series_names", [])),
        ]
    )
    selected_attempts = _select_attempts_for_visual(
        attempts=attempts,
        source_attempt_ids=source_attempt_ids,
        source_experiment_ids=source_experiment_ids,
        target_algorithms=target_algorithms,
        visual_text=visual_text,
    )
    candidate_metrics = _filter_metrics_for_visual(
        _dedupe_metric_dicts(
            [metric for attempt in selected_attempts for metric in attempt.get("metrics", []) if isinstance(metric, dict)]
        ),
        visual_text,
    )
    curves = _dedupe_curves(
        [
            curve
            for attempt in selected_attempts
            for curve in attempt.get("curves", [])
            if isinstance(curve, dict) and curve.get("points") and _curve_relevant_to_visual(curve, visual_text)
        ]
    )
    skip_evidence = [
        {
            "attempt_id": attempt.get("attempt_id"),
            "experiment_id": attempt.get("experiment_id"),
            "scope": attempt.get("scope") if isinstance(attempt.get("scope"), dict) else {},
            "source": attempt.get("stdout_source") or attempt.get("evidence_source"),
            "skip_reason": attempt.get("skip_reason"),
            "error_summary": attempt.get("error_summary"),
            "reason_codes": attempt.get("reason_codes", []),
        }
        for attempt in selected_attempts
        if attempt.get("skip_reason") or attempt.get("error_summary")
    ]
    evidence_sources = _dedupe_strings(
        [
            PHASE2_PACKAGE_PATH,
            LOG_EVIDENCE_PATH if execution_log_evidence_present(selected_attempts) else "",
            *[str(metric.get("source") or metric.get("source_attempt_id") or "") for metric in candidate_metrics],
            *[str(curve.get("source") or "") for curve in curves],
            *[str(item.get("source") or "") for item in skip_evidence],
            *[str(attempt.get("attempt_id") or "") for attempt in selected_attempts],
        ]
    )

    bundle = {
        "element_id": element_id,
        "visual_anchor": visual_anchor,
        "element_type": target.get("element_type") or element.get("element_type") or "figure",
        "chart_type": target.get("chart_type") or element.get("chart_type"),
        "caption": caption,
        "reference_image_path": reference_image_path,
        "axis_labels": target.get("axis_labels") or element.get("axis_labels") or {},
        "legend_entries": target.get("legend_entries") or element.get("legend_entries") or [],
        "series_names": target.get("series_names") or [],
        "metric_names": target.get("metric_names") or [],
        "model_names": target.get("model_names") or element.get("model_names") or [],
        "semantic_summary": target.get("semantic_summary") or "",
        "reconstruction_instructions": target.get("reconstruction_instructions") or [],
        "paper_data_series": element.get("data_series") or [],
        "paper_matrix": element.get("matrix"),
        "paper_x_labels": element.get("x_labels") or [],
        "paper_y_labels": element.get("y_labels") or [],
        "target_algorithms": sorted(target_algorithms),
        "claim_rows": claim_rows,
        "selected_attempts": _summarize_attempts(selected_attempts),
        "candidate_metrics": candidate_metrics,
        "curves": curves,
        "skip_evidence": skip_evidence,
        "phase2_results_excerpt": phase2_results_text[:4000],
        "all_metrics_excerpt": metrics_doc.get("records", [])[:20],
        "evidence_sources": evidence_sources,
    }
    bundle.update(_match_metadata_for_bundle(bundle, visual_text))
    return bundle


def execution_log_evidence_present(attempts: list[dict[str, Any]]) -> bool:
    return any(str(attempt.get("evidence_source") or "") == LOG_EVIDENCE_PATH for attempt in attempts)


def _match_metadata_for_bundle(bundle: dict[str, Any], visual_text: str) -> dict[str, Any]:
    target_algorithms = set(bundle.get("target_algorithms") or [])
    target_datasets = _datasets_from_text(_norm(visual_text))
    target_families = _model_families_from_text(_norm(visual_text))
    evidence_algorithms = _scope_set(bundle, "algorithm", normalize_algorithm=True)
    evidence_datasets = _scope_set(bundle, "dataset")
    evidence_families = _scope_set(bundle, "model_family")
    fidelities = _scope_set(bundle, "fidelity")
    metric_names = _metric_names_for_bundle(bundle)
    has_numeric = _has_renderable_evidence(bundle)
    has_skip_only = bool(bundle.get("skip_evidence")) and not has_numeric

    if has_skip_only:
        level = "NO_EVIDENCE"
    elif not has_numeric:
        level = "NO_EVIDENCE"
    elif target_algorithms and evidence_algorithms and not (target_algorithms & evidence_algorithms):
        level = "RELATED"
    elif target_datasets and evidence_datasets and not (target_datasets & evidence_datasets):
        level = "RELATED"
    elif target_families and evidence_families and not (target_families & evidence_families):
        level = "RELATED"
    elif target_algorithms and not target_algorithms.issubset(evidence_algorithms):
        level = "PARTIAL"
    elif "smoke" in fidelities and not (fidelities & {"full", "trend", "artifact"}):
        level = "PARTIAL"
    else:
        level = "EXACT"

    matched_scope = {
        "target_algorithms": sorted(target_algorithms),
        "evidence_algorithms": sorted(evidence_algorithms),
        "target_datasets": sorted(target_datasets),
        "evidence_datasets": sorted(evidence_datasets),
        "target_model_families": sorted(target_families),
        "evidence_model_families": sorted(evidence_families),
        "fidelities": sorted(fidelities),
        "metrics": sorted(metric_names),
    }
    return {
        "match_level": level,
        "matched_scope": matched_scope,
        "coverage_note": _coverage_note(level, matched_scope, has_skip_only=has_skip_only),
    }


def _scope_set(bundle: dict[str, Any], key: str, *, normalize_algorithm: bool = False) -> set[str]:
    values: set[str] = set()
    for row in [*bundle.get("candidate_metrics", []), *bundle.get("curves", [])]:
        if not isinstance(row, dict):
            continue
        value = str(row.get(key) or "").strip()
        if value:
            values.add(_normalize_algorithm(value) if normalize_algorithm else _norm(value))
    for attempt in bundle.get("selected_attempts", []):
        if not isinstance(attempt, dict):
            continue
        if key == "fidelity":
            value = str(attempt.get("fidelity") or "").strip()
        else:
            scope = attempt.get("scope") if isinstance(attempt.get("scope"), dict) else {}
            value = str(scope.get(key) or "").strip()
        if value:
            values.add(_normalize_algorithm(value) if normalize_algorithm else _norm(value))
    return {value for value in values if value}


def _metric_names_for_bundle(bundle: dict[str, Any]) -> set[str]:
    names = {
        _norm(row.get("metric_name") or row.get("raw_metric_name") or "")
        for row in bundle.get("candidate_metrics", [])
        if isinstance(row, dict)
    }
    names.update(
        _norm(row.get("metric_name") or "")
        for row in bundle.get("curves", [])
        if isinstance(row, dict)
    )
    for row in bundle.get("claim_rows", []):
        if isinstance(row, dict):
            names.add(_norm(row.get("metric") or row.get("predicate") or ""))
    return {name for name in names if name}


def _coverage_note(level: str, matched_scope: dict[str, Any], *, has_skip_only: bool) -> str:
    if has_skip_only:
        return "Phase 2 logs contain skip/error/config evidence but no numeric metric for this visual."
    target_algorithms = set(matched_scope.get("target_algorithms", []))
    evidence_algorithms = set(matched_scope.get("evidence_algorithms", []))
    missing_algorithms = sorted(target_algorithms - evidence_algorithms)
    if level == "RELATED" and target_algorithms and evidence_algorithms and not (target_algorithms & evidence_algorithms):
        return (
            "Available algorithms "
            f"{', '.join(sorted(evidence_algorithms))} do not overlap the target algorithms "
            f"{', '.join(sorted(target_algorithms))}."
        )
    if missing_algorithms:
        return f"Missing target algorithm evidence for {', '.join(missing_algorithms)}."
    if level == "PARTIAL":
        return "Phase 2 evidence covers only a reduced fidelity or subset of the target scope."
    if level == "RELATED":
        return "Phase 2 evidence is adjacent to the target scope and should not be read as a full reproduction."
    return ""


def _claims_for_visual(target: dict[str, Any], element: dict[str, Any], claims_doc: dict[str, Any]) -> list[dict[str, Any]]:
    element_id = str(target.get("element_id") or element.get("element_id") or "")
    visual_anchor = str(target.get("visual_anchor") or element.get("visual_anchor") or "")
    associated = {str(x) for x in target.get("associated_claim_ids", []) if str(x).strip()}
    rows = []
    for claim in claims_doc.get("claims", []):
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("claim_id") or "")
        conditions = claim.get("conditions", {})
        visual_data = conditions.get("visual_data", {}) if isinstance(conditions, dict) else {}
        table_anchor = str(conditions.get("table_anchor") or "") if isinstance(conditions, dict) else ""
        visual_id = str(visual_data.get("element_id") or "") if isinstance(visual_data, dict) else ""
        if claim_id in associated or visual_id == element_id or table_anchor in {element_id, visual_anchor}:
            rows.append(claim)
    return rows


def _collect_phase2_attempts(phase2_package: dict[str, Any], artifacts) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for experiment in phase2_package.get("experiments", []) if isinstance(phase2_package, dict) else []:
        if not isinstance(experiment, dict):
            continue
        for attempt in experiment.get("attempts", []):
            if not isinstance(attempt, dict):
                continue
            row = dict(attempt)
            row["experiment_id"] = row.get("experiment_id") or experiment.get("experiment_id")
            row["experiment_name"] = row.get("experiment_name") or experiment.get("name")
            stdout_ref = _stdout_ref_for_attempt(row)
            stdout_text = _read_attempt_stdout(stdout_ref, artifacts)
            row["stdout_source"] = stdout_ref
            row["curves"] = _extract_phase2_curves(
                stdout_text,
                source=stdout_ref or f"{PHASE2_PACKAGE_PATH}:{row.get('attempt_id')}",
                attempt=row,
            )
            row["metrics"] = row.get("metrics", []) if isinstance(row.get("metrics"), list) else []
            attempts.append(row)
    return attempts


def _merge_attempts_with_log_evidence(
    attempts: list[dict[str, Any]],
    log_evidence_doc: dict[str, Any],
) -> list[dict[str, Any]]:
    """Add raw executor log evidence as first-class Phase 2 attempts."""
    merged = list(attempts)
    for row in log_evidence_doc.get("logs", []) if isinstance(log_evidence_doc, dict) else []:
        if not isinstance(row, dict):
            continue
        metrics = [dict(metric) for metric in row.get("metrics", []) if isinstance(metric, dict)]
        curves = [dict(curve) for curve in row.get("curves", []) if isinstance(curve, dict)]
        skip_reason = str(row.get("skip_reason") or "").strip()
        error_summary = str(row.get("error_summary") or "").strip()
        if not metrics and not curves and not skip_reason and not error_summary:
            continue
        path = str(row.get("path") or "")
        scope = {
            "algorithm": row.get("algorithm"),
            "dataset": row.get("dataset"),
            "model_family": row.get("model_family"),
        }
        status = "skipped" if skip_reason else "failed" if error_summary else "ok"
        merged.append(
            {
                "attempt_id": f"log:{path}" if path else f"log:{len(merged)}",
                "experiment_id": row.get("experiment_id"),
                "experiment_name": row.get("config_name") or row.get("experiment_id"),
                "scope": scope,
                "status": status,
                "fidelity": row.get("fidelity"),
                "execution_outcome": "SKIPPED" if skip_reason else "EXECUTED" if (metrics or curves) else "FAILED",
                "evidence_source": LOG_EVIDENCE_PATH,
                "stop_reason": skip_reason or error_summary,
                "stdout_source": path,
                "metrics": metrics,
                "curves": curves,
                "skip_reason": skip_reason,
                "error_summary": error_summary,
                "reason_codes": row.get("reason_codes", []),
                "logs": {str(row.get("log_kind") or "log"): path} if path else {},
            }
        )
    return merged


def _stdout_ref_for_attempt(attempt: dict[str, Any]) -> str:
    logs = attempt.get("logs") if isinstance(attempt.get("logs"), dict) else {}
    stdout_ref = str(logs.get("stdout") or "")
    if stdout_ref:
        return stdout_ref
    artifacts = attempt.get("artifacts") or []
    for ref in artifacts:
        ref_s = str(ref)
        if ref_s.endswith("_stdout.log") or ref_s.endswith(".log"):
            return ref_s
    return ""


def _read_attempt_stdout(ref: str, artifacts) -> str:
    path = _resolve_artifact_path(artifacts, ref)
    if path and path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


def _resolve_artifact_path(artifacts, ref: str | None) -> Path | None:
    if not ref:
        return None
    path = Path(str(ref))
    if path.is_absolute():
        return path
    candidate = artifacts.path(str(ref))
    if candidate.exists() or str(ref).startswith("execution/") or str(ref).startswith("fingerprint/"):
        return candidate
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return candidate


def _extract_phase2_curves(stdout: str, *, source: str, attempt: dict[str, Any]) -> list[dict[str, Any]]:
    if not stdout:
        return []
    epoch_rows: dict[int, dict[str, Any]] = {}
    current_epoch: int | None = None
    for line in stdout.splitlines():
        epoch_match = re.search(r"\[(\d+),\s*\d+\]\s+loss:\s*([\d.eE+-]+)", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epoch_rows.setdefault(current_epoch, {})["loss"] = float(epoch_match.group(2))
            continue
        test_match = re.search(r"Test accuracy:\s*([\d.eE+-]+)\s*%?", line, re.IGNORECASE)
        if test_match and current_epoch is not None:
            epoch_rows.setdefault(current_epoch, {})["test_accuracy"] = _percent_to_ratio(float(test_match.group(1)))
            continue
        train_match = re.search(r"Training accuracy\s*=\s*([\d.eE+-]+)", line, re.IGNORECASE)
        if train_match:
            epoch_rows.setdefault(current_epoch or 1, {})["train_accuracy"] = _percent_to_ratio(float(train_match.group(1)))
            continue
        val_match = re.search(r"Validation accuracy\s*=\s*([\d.eE+-]+)", line, re.IGNORECASE)
        if val_match:
            epoch_rows.setdefault(current_epoch or 1, {})["val_accuracy"] = _percent_to_ratio(float(val_match.group(1)))

    curves: list[dict[str, Any]] = []
    scope = attempt.get("scope") if isinstance(attempt.get("scope"), dict) else {}
    meta = {
        "attempt_id": attempt.get("attempt_id"),
        "experiment_id": attempt.get("experiment_id"),
        "fidelity": attempt.get("fidelity"),
        "algorithm": scope.get("algorithm"),
        "dataset": scope.get("dataset"),
        "model_family": scope.get("model_family"),
        "source": source,
    }
    for metric_name in ("test_accuracy", "train_accuracy", "val_accuracy", "loss"):
        points = [
            {"x": epoch, "y": values[metric_name]}
            for epoch, values in sorted(epoch_rows.items())
            if metric_name in values
        ]
        if points:
            curves.append({"metric_name": metric_name, "points": points, **meta})
    return curves


def _percent_to_ratio(value: float) -> float:
    return value / 100.0 if value > 1.0 else value


def _select_attempts_for_visual(
    *,
    attempts: list[dict[str, Any]],
    source_attempt_ids: set[str],
    source_experiment_ids: set[str],
    target_algorithms: set[str],
    visual_text: str,
) -> list[dict[str, Any]]:
    wanted_text = _norm(visual_text)
    wanted_datasets = _datasets_from_text(wanted_text)
    wanted_families = _model_families_from_text(wanted_text)
    metric_tokens = _visual_metric_tokens(wanted_text)
    scored: list[tuple[int, dict[str, Any]]] = []
    for attempt in attempts:
        attempt_id = str(attempt.get("attempt_id") or "")
        experiment_id = str(attempt.get("experiment_id") or "")
        scope = attempt.get("scope") if isinstance(attempt.get("scope"), dict) else {}
        algorithm = _normalize_algorithm(str(scope.get("algorithm") or ""))
        dataset = _norm(scope.get("dataset") or "")
        family = _norm(scope.get("model_family") or "")
        score = 0
        if source_attempt_ids:
            score += 8 if attempt_id in source_attempt_ids else 0
        if source_experiment_ids:
            score += 6 if experiment_id in source_experiment_ids else 0
        if wanted_datasets and dataset in wanted_datasets:
            score += 4
        if wanted_families and family in wanted_families:
            score += 3
        if target_algorithms and algorithm in target_algorithms:
            score += 3
        elif target_algorithms and algorithm:
            score += 1
        if _attempt_has_metric_token(attempt, metric_tokens):
            score += 3
        if attempt.get("metrics") or attempt.get("curves"):
            score += 1
        if attempt.get("skip_reason") or attempt.get("error_summary"):
            score += 1
        if score:
            scored.append((score, attempt))
    if source_attempt_ids or source_experiment_ids:
        selected = [attempt for score, attempt in scored if score >= 6]
        if selected:
            return selected
    selected = [attempt for score, attempt in scored if score >= 3]
    if selected:
        return selected
    if wanted_datasets or wanted_families or target_algorithms or metric_tokens:
        return []
    return [
        attempt
        for attempt in attempts
        if attempt.get("metrics") or attempt.get("curves") or attempt.get("skip_reason") or attempt.get("error_summary")
    ]


def _target_reproduction_decision(bundle: dict[str, Any]) -> dict[str, Any]:
    chart_type = str(bundle.get("chart_type") or "").lower()
    if chart_type == "diagram":
        return {"eligible": False, "reason": "Pure method diagram, not a result figure/table.", "reason_codes": ["SKIP_DIAGRAM"]}
    if not _is_result_related(bundle):
        return {
            "eligible": False,
            "reason": "Visual target is not result/metric oriented.",
            "reason_codes": ["SKIP_NON_RESULT_VISUAL"],
        }
    if not _has_any_phase2_evidence(bundle):
        return {
            "eligible": False,
            "reason": "No executable Phase 2 metric, curve, or claim evidence for this visual target.",
            "reason_codes": ["SKIP_NO_PHASE2_EVIDENCE"],
        }
    return {
        "eligible": True,
        "reason": "",
        "reason_codes": [
            "RESULT_VISUAL_WITH_PHASE2_EVIDENCE",
            f"MATCH_LEVEL_{str(bundle.get('match_level') or 'RELATED')}",
        ],
    }


def _filter_metrics_for_visual(metrics: list[dict[str, Any]], visual_text: str) -> list[dict[str, Any]]:
    text = _norm(visual_text)
    if not metrics:
        return []
    metric_tokens = _visual_metric_tokens(text)
    if not metric_tokens:
        return metrics[:30]
    filtered = []
    for metric in metrics:
        name = _norm(metric.get("metric_name") or metric.get("raw_metric_name") or "")
        if not name:
            continue
        if any(token in name for token in metric_tokens):
            filtered.append(metric)
    return filtered


def _curve_relevant_to_visual(curve: dict[str, Any], visual_text: str) -> bool:
    text = _norm(visual_text)
    metric_name = _norm(curve.get("metric_name") or "")
    if any(token in text for token in ("t sne", "tsne", "embedding", "weight distribution", "histogram", "frequency")):
        if not any(token in text for token in ("accuracy", "loss", "test curve", "training curve")):
            return False
    if "accuracy" in metric_name:
        return any(token in text for token in ("accuracy", "test curve", "training", "train"))
    if "loss" in metric_name:
        return "loss" in text
    metric_tokens = _visual_metric_tokens(text)
    if metric_tokens:
        return any(token in metric_name for token in metric_tokens)
    return any(token in text for token in RESULT_KEYWORDS)


def _attempt_has_metric_token(attempt: dict[str, Any], metric_tokens: set[str]) -> bool:
    if not metric_tokens:
        return False
    for metric in attempt.get("metrics", []):
        if not isinstance(metric, dict):
            continue
        name = _norm(metric.get("metric_name") or metric.get("raw_metric_name") or "")
        if any(token in name for token in metric_tokens):
            return True
    for curve in attempt.get("curves", []):
        if not isinstance(curve, dict):
            continue
        name = _norm(curve.get("metric_name") or "")
        if any(token in name for token in metric_tokens):
            return True
    return False


def _visual_metric_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    if any(token in text for token in ("accuracy", "acc", "test curve")):
        tokens.add("accuracy")
    if "loss" in text:
        tokens.add("loss")
    if "precision" in text:
        tokens.add("precision")
    if "recall" in text:
        tokens.add("recall")
    if "f1" in text:
        tokens.add("f1")
    if "auc" in text or "roc" in text:
        tokens.add("auc")
    return tokens


def _is_result_related(bundle: dict[str, Any]) -> bool:
    text = _norm(
        " ".join(
            [
                str(bundle.get("element_type") or ""),
                str(bundle.get("chart_type") or ""),
                str(bundle.get("caption") or ""),
                str(bundle.get("semantic_summary") or ""),
                " ".join(str(x) for x in bundle.get("metric_names", [])),
                " ".join(str(x) for x in bundle.get("series_names", [])),
            ]
        )
    )
    if str(bundle.get("element_type") or "").lower() == "table" and any(k in text for k in RESULT_KEYWORDS):
        return True
    if any(k in text for k in RESULT_KEYWORDS):
        return True
    return any(row.get("target") is not None for row in bundle.get("claim_rows", []))


def _algorithm_mismatch_without_partial_evidence(bundle: dict[str, Any]) -> bool:
    target_algorithms = set(bundle.get("target_algorithms") or [])
    if not target_algorithms:
        return False
    evidence_algorithms = {
        _normalize_algorithm(str(attempt.get("scope", {}).get("algorithm") or ""))
        for attempt in bundle.get("selected_attempts", [])
        if isinstance(attempt, dict)
    }
    evidence_algorithms.discard("")
    return bool(evidence_algorithms and not (target_algorithms & evidence_algorithms))


def _algorithm_coverage_skip_reason(bundle: dict[str, Any]) -> dict[str, Any] | None:
    target_algorithms = set(bundle.get("target_algorithms") or [])
    if not target_algorithms or str(bundle.get("element_type") or "").lower() == "table":
        return None
    evidence_algorithms = _renderable_evidence_algorithms(bundle)
    evidence_algorithms.discard("")
    missing = target_algorithms - evidence_algorithms
    if missing:
        return {
            "eligible": False,
            "reason": (
                "Phase 2 evidence covers only a subset of algorithms in this paper visual; "
                f"missing {', '.join(sorted(missing))}."
            ),
            "reason_codes": ["SKIP_INCOMPLETE_ALGORITHM_COVERAGE"],
        }
    return None


def _renderable_evidence_algorithms(bundle: dict[str, Any]) -> set[str]:
    algorithms = {
        _normalize_algorithm(str(curve.get("algorithm") or ""))
        for curve in bundle.get("curves", [])
        if isinstance(curve, dict)
    }
    for metric in bundle.get("candidate_metrics", []):
        if not isinstance(metric, dict):
            continue
        algorithms.add(_normalize_algorithm(str(metric.get("algorithm") or "")))
    return {algorithm for algorithm in algorithms if algorithm}


def _missing_target_algorithms(bundle: dict[str, Any]) -> list[str]:
    target_algorithms = set(bundle.get("target_algorithms") or [])
    evidence_algorithms = _renderable_evidence_algorithms(bundle)
    return sorted(target_algorithms - evidence_algorithms)


def _has_renderable_evidence(bundle: dict[str, Any]) -> bool:
    if bundle.get("curves"):
        return True
    if bundle.get("candidate_metrics"):
        return True
    return any(row.get("target") is not None and row.get("reproduced") is not None for row in bundle.get("claim_rows", []))


def _has_any_phase2_evidence(bundle: dict[str, Any]) -> bool:
    return _has_renderable_evidence(bundle) or bool(bundle.get("skip_evidence"))


def _with_match_fields(spec: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    spec = dict(spec)
    return _ensure_plot_spec_match_fields(spec, bundle)


def _ensure_plot_spec_match_fields(spec: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    spec = dict(spec)
    level = str(spec.get("match_level") or bundle.get("match_level") or "RELATED").upper()
    if level not in {"EXACT", "PARTIAL", "RELATED", "NO_EVIDENCE"}:
        level = str(bundle.get("match_level") or "RELATED").upper()
    spec["match_level"] = level if level in {"EXACT", "PARTIAL", "RELATED", "NO_EVIDENCE"} else "RELATED"
    if not isinstance(spec.get("matched_scope"), dict):
        spec["matched_scope"] = bundle.get("matched_scope", {})
    spec["coverage_note"] = str(spec.get("coverage_note") or bundle.get("coverage_note") or "")
    return spec


def _comparison_note(bundle: dict[str, Any], base: str) -> str:
    level = str(bundle.get("match_level") or "RELATED").upper()
    prefix = {
        "EXACT": "Exact evidence: ",
        "PARTIAL": "Partial evidence only: ",
        "RELATED": "Related evidence only: ",
        "NO_EVIDENCE": "No executable numeric evidence: ",
    }.get(level, "Related evidence only: ")
    note = prefix + base
    coverage = str(bundle.get("coverage_note") or "").strip()
    if coverage:
        note = f"{note} {coverage}"
    return note


def _skipped_from_bundle(bundle: dict[str, Any], reason: str, reason_codes: list[str]) -> SkippedReproducedTarget:
    return SkippedReproducedTarget(
        element_id=str(bundle.get("element_id") or ""),
        visual_anchor=str(bundle.get("visual_anchor") or ""),
        skip_reason=reason,
        evidence_sources=bundle.get("evidence_sources", []),
        reason_codes=reason_codes,
    )


def _deterministic_plot_spec(bundle: dict[str, Any]) -> dict[str, Any] | None:
    comparable = [
        row for row in bundle.get("claim_rows", [])
        if row.get("target") is not None and row.get("reproduced") is not None
    ]
    if comparable and (
        str(bundle.get("element_type") or "").lower() == "table"
        or str(bundle.get("chart_type") or "").lower() == "table"
    ):
        rows = [
            [
                str(row.get("predicate") or row.get("claim_id") or "")[:48],
                _format_number(row.get("target")),
                _format_number(row.get("reproduced")),
                str(row.get("status") or ""),
            ]
            for row in comparable[:12]
        ]
        return _with_match_fields({
            "decision": "PLOT",
            "chart_type": "table",
            "title": _short_title(bundle),
            "x_label": "",
            "y_label": "",
            "series": [],
            "table": {
                "columns": ["Claim", "Paper", "Reproduced", "Status"],
                "rows": rows,
                "source": PHASE2_PACKAGE_PATH,
            },
            "unit": None,
            "normalization": None,
            "evidence_sources": _dedupe_strings(bundle.get("evidence_sources", [])),
            "comparison_note": _comparison_note(bundle, "Table rebuilt from matched claim verdicts and Phase 2 metrics."),
            "confidence": 0.70,
            "reason_codes": ["DETERMINISTIC_VERDICT_TABLE_FALLBACK"],
        }, bundle)

    curves = bundle.get("curves", [])
    if curves:
        selected = _preferred_curves(curves)
        return _with_match_fields({
            "decision": "PLOT",
            "chart_type": "line",
            "title": _short_title(bundle),
            "x_label": "Epoch",
            "y_label": _metric_axis_label(selected),
            "series": [
                {
                    "name": _curve_name(curve),
                    "x": [point["x"] for point in curve.get("points", [])],
                    "y": [point["y"] for point in curve.get("points", [])],
                    "source": curve.get("source"),
                    "style": {"marker": "o"},
                }
                for curve in selected
            ],
            "table": {"columns": [], "rows": [], "source": None},
            "unit": "ratio" if any("accuracy" in str(c.get("metric_name")) for c in selected) else None,
            "normalization": "Percent accuracies normalized to ratio when needed.",
            "evidence_sources": _dedupe_strings(str(c.get("source") or "") for c in selected),
            "comparison_note": _comparison_note(bundle, "Reproduced from Phase 2 stdout curves."),
            "confidence": 0.75,
            "reason_codes": ["DETERMINISTIC_CURVE_FALLBACK"],
        }, bundle)

    if comparable:
        rows = [
            [
                str(row.get("predicate") or row.get("claim_id") or "")[:48],
                _format_number(row.get("target")),
                _format_number(row.get("reproduced")),
                str(row.get("status") or ""),
            ]
            for row in comparable[:12]
        ]
        return _with_match_fields({
            "decision": "PLOT",
            "chart_type": "table",
            "title": _short_title(bundle),
            "x_label": "",
            "y_label": "",
            "series": [],
            "table": {
                "columns": ["Claim", "Paper", "Reproduced", "Status"],
                "rows": rows,
                "source": PHASE2_PACKAGE_PATH,
            },
            "unit": None,
            "normalization": None,
            "evidence_sources": _dedupe_strings(bundle.get("evidence_sources", [])),
            "comparison_note": _comparison_note(bundle, "Table rebuilt from matched claim verdicts and Phase 2 metrics."),
            "confidence": 0.70,
            "reason_codes": ["DETERMINISTIC_VERDICT_TABLE_FALLBACK"],
        }, bundle)

    metrics = [m for m in bundle.get("candidate_metrics", []) if _numeric(m.get("value")) is not None]
    if metrics:
        rows = []
        for metric in metrics[:12]:
            label = str(metric.get("metric_name") or metric.get("raw_metric_name") or "metric")
            scope = " ".join(str(metric.get(key) or "") for key in ("algorithm", "dataset", "model_family")).strip()
            rows.append([label[:44], _format_number(metric.get("value")), scope or str(metric.get("fidelity") or "")])
        missing_algorithms = _missing_target_algorithms(bundle)
        for algorithm in missing_algorithms[:6]:
            rows.append([f"{algorithm} evidence", "missing", "not executed"])
        return _with_match_fields({
            "decision": "PLOT",
            "chart_type": "table",
            "title": _short_title(bundle),
            "x_label": "",
            "y_label": "",
            "series": [],
            "table": {"columns": ["Metric", "Value", "Scope"], "rows": rows, "source": PHASE2_PACKAGE_PATH},
            "unit": None,
            "normalization": "Bounded metrics are stored as ratios.",
            "evidence_sources": _dedupe_strings(bundle.get("evidence_sources", [])),
            "comparison_note": _comparison_note(bundle, "Reproduced as a compact Phase 2 metric table."),
            "confidence": 0.60,
            "reason_codes": ["DETERMINISTIC_METRIC_TABLE_FALLBACK"],
        }, bundle)
    skip_evidence = bundle.get("skip_evidence", [])
    if skip_evidence:
        lines = []
        for item in skip_evidence[:6]:
            source = item.get("source") or item.get("attempt_id") or "phase2 log"
            reason = item.get("skip_reason") or item.get("error_summary") or "no numeric metric emitted"
            lines.append(f"{source}: {_short(reason, 160)}")
        return _with_match_fields({
            "decision": "PLOT",
            "chart_type": "text-panel",
            "title": _short_title(bundle),
            "x_label": "",
            "y_label": "",
            "series": [],
            "table": {"columns": [], "rows": [], "source": LOG_EVIDENCE_PATH},
            "unit": None,
            "normalization": None,
            "evidence_sources": _dedupe_strings(bundle.get("evidence_sources", [])),
            "comparison_note": _comparison_note(
                bundle,
                "Phase 2 produced skip/error evidence but no numeric data to plot. " + " | ".join(lines),
            ),
            "confidence": 0.45,
            "reason_codes": ["DETERMINISTIC_NO_EVIDENCE_TEXT_PANEL"],
        }, bundle)
    return None


def _preferred_curves(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for metric in ("test_accuracy", "val_accuracy", "train_accuracy", "loss"):
        selected = [curve for curve in curves if curve.get("metric_name") == metric]
        if selected:
            return selected[:6]
    return curves[:6]


def _metric_axis_label(curves: list[dict[str, Any]]) -> str:
    names = {str(curve.get("metric_name") or "") for curve in curves}
    if any("accuracy" in name for name in names):
        return "Accuracy"
    if "loss" in names:
        return "Loss"
    return "Value"


def _curve_name(curve: dict[str, Any]) -> str:
    parts = [
        curve.get("algorithm"),
        curve.get("dataset"),
        curve.get("model_family"),
        curve.get("metric_name"),
    ]
    return " ".join(str(x) for x in parts if x)


def _render_plot_spec(spec: dict[str, Any], output_path: Path) -> bool:
    chart_type = str(spec.get("chart_type") or "").lower()
    if chart_type not in SUPPORTED_SPEC_CHARTS:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        if chart_type == "line":
            for series in spec.get("series", []):
                xs, ys = _series_xy(series)
                if not xs or not ys:
                    continue
                ax.plot(xs, ys, marker=str(series.get("style", {}).get("marker") or "o"), linewidth=2, label=_short(series.get("name"), 36))
            ax.set_xlabel(str(spec.get("x_label") or "x"))
            ax.set_ylabel(str(spec.get("y_label") or "value"))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        elif chart_type == "scatter":
            for series in spec.get("series", []):
                xs, ys = _series_xy(series)
                if not xs or not ys:
                    continue
                ax.scatter(xs, ys, label=_short(series.get("name"), 36), s=28)
            ax.set_xlabel(str(spec.get("x_label") or "x"))
            ax.set_ylabel(str(spec.get("y_label") or "value"))
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        elif chart_type == "bar":
            series = [s for s in spec.get("series", []) if isinstance(s, dict)]
            if not series:
                return False
            labels = [str(x) for x in series[0].get("x", [])]
            x = np.arange(len(labels))
            width = min(0.8 / max(1, len(series)), 0.35)
            for idx, row in enumerate(series):
                ys = [_numeric(v) or 0.0 for v in row.get("y", [])]
                offset = (idx - (len(series) - 1) / 2.0) * width
                ax.bar(x[: len(ys)] + offset, ys, width=width, label=_short(row.get("name"), 30))
            ax.set_xticks(x)
            ax.set_xticklabels([_short(label, 18) for label in labels], rotation=35, ha="right")
            ax.set_ylabel(str(spec.get("y_label") or "value"))
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(fontsize=8)
        elif chart_type == "heatmap":
            table = spec.get("table") or {}
            rows = table.get("rows") or []
            arr = np.array([[float(v) for v in row] for row in rows], dtype=float)
            im = ax.imshow(arr, cmap="viridis")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            columns = table.get("columns") or []
            if columns:
                ax.set_xticks(np.arange(len(columns)))
                ax.set_xticklabels([_short(c, 14) for c in columns], rotation=45, ha="right")
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    ax.text(j, i, f"{arr[i, j]:.3g}", ha="center", va="center", color="white", fontsize=8)
        elif chart_type == "table":
            _render_table_spec(ax, spec)
        elif chart_type == "text-panel":
            ax.axis("off")
            text = str(spec.get("comparison_note") or spec.get("title") or "No reproduced evidence.")
            ax.text(0.03, 0.97, text, ha="left", va="top", transform=ax.transAxes, wrap=True, fontsize=10)
        ax.set_title(_short(spec.get("title"), 78))
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def _render_table_spec(ax, spec: dict[str, Any]) -> None:
    ax.axis("off")
    table_spec = spec.get("table") or {}
    columns = [str(c) for c in table_spec.get("columns", [])][:8]
    rows = table_spec.get("rows", [])[:16]
    if not columns or not rows:
        ax.text(0.5, 0.5, "No table rows available", ha="center", va="center")
        return
    cell_text = []
    for row in rows:
        if isinstance(row, dict):
            values = [row.get(column, "") for column in columns]
        else:
            values = list(row) if isinstance(row, (list, tuple)) else [row]
        cell_text.append([_short(cell, 28) for cell in values[: len(columns)]])
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)


def _compose_comparison_image(
    *,
    reference_path: Path | None,
    reproduced_path: Path,
    output_path: Path,
    title: str,
    note: str,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
        if reference_path and reference_path.exists():
            axes[0].imshow(mpimg.imread(reference_path))
            axes[0].set_axis_off()
        else:
            axes[0].axis("off")
            axes[0].text(0.5, 0.5, "reference crop unavailable", ha="center", va="center", fontsize=11)
        axes[0].set_title("Original paper crop")
        axes[1].imshow(mpimg.imread(reproduced_path))
        axes[1].set_axis_off()
        axes[1].set_title("Phase 2 reproduced evidence")
        fig.suptitle(_short(title, 110), fontsize=13)
        if note:
            fig.text(0.5, 0.015, _short(note, 180), ha="center", va="bottom", fontsize=9, wrap=True)
        fig.tight_layout(rect=(0, 0.04, 1, 0.94))
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def _series_xy(series: dict[str, Any]) -> tuple[list[Any], list[float]]:
    xs = list(series.get("x", []))
    ys = [_numeric(v) for v in series.get("y", [])]
    clean_x: list[Any] = []
    clean_y: list[float] = []
    for x, y in zip(xs, ys):
        if y is None:
            continue
        clean_x.append(x)
        clean_y.append(y)
    return clean_x, clean_y


def _validate_codegen_code(code: str) -> tuple[bool, str]:
    lowered = code.lower()
    forbidden_fragments = [
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "shutil",
        "pathlib",
        "os.",
        "sys.",
        "open(",
        "eval(",
        "exec(",
        "__import__",
        "compile(",
        "input(",
    ]
    for fragment in forbidden_fragments:
        if fragment in lowered:
            return False, f"forbidden fragment: {fragment}"
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"
    allowed_imports = {"json", "math", "statistics", "matplotlib", "matplotlib.pyplot", "numpy"}
    forbidden_calls = {"eval", "exec", "open", "compile", "input", "__import__", "globals", "locals"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in allowed_imports:
                    return False, f"import not allowed: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module not in allowed_imports:
                return False, f"import not allowed: {node.module}"
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in forbidden_calls:
                return False, f"call not allowed: {func.id}"
            if isinstance(func, ast.Attribute) and func.attr in {"system", "popen", "run", "remove", "unlink", "rmtree"}:
                return False, f"attribute call not allowed: {func.attr}"
    return True, ""


def _wrap_codegen_code(code: str, bundle: dict[str, Any], plot_spec: dict[str, Any], output_path: str) -> str:
    payload = {"bundle": _compact_bundle_for_llm(bundle), "plot_spec": plot_spec}
    return "\n".join(
        [
            "import json",
            "import matplotlib",
            'matplotlib.use("Agg")',
            f"payload = json.loads({json.dumps(json.dumps(payload, ensure_ascii=True))})",
            f"output_path = {output_path!r}",
            code,
        ]
    )


def _build_plot_spec_prompt(bundle: dict[str, Any]) -> str:
    compact = _compact_bundle_for_llm(bundle)
    return (
        "Plan the reproduced side of an original-vs-reproduced comparison figure.\n"
        "Return decision='PLOT' or decision='SKIP'.\n\n"
        f"Evidence bundle:\n{json.dumps(compact, ensure_ascii=False, indent=2)[:18000]}"
    )


def _build_codegen_prompt(bundle: dict[str, Any], plot_spec: dict[str, Any], output_path: str) -> str:
    payload = {"bundle": _compact_bundle_for_llm(bundle), "plot_spec": plot_spec, "output_path": output_path}
    return (
        "The deterministic renderer could not render this plot_spec. Generate restricted matplotlib code "
        "for the reproduced panel only.\n\n"
        f"Payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)[:18000]}"
    )


def _normalize_plot_spec(data: dict[str, Any]) -> dict[str, Any]:
    spec = dict(data)
    spec["decision"] = str(spec.get("decision") or "PLOT").upper()
    spec["chart_type"] = str(spec.get("chart_type") or "text-panel").lower()
    spec["series"] = spec.get("series") if isinstance(spec.get("series"), list) else []
    spec["table"] = spec.get("table") if isinstance(spec.get("table"), dict) else {"columns": [], "rows": [], "source": None}
    spec["evidence_sources"] = _string_list(spec.get("evidence_sources"))
    level = str(spec.get("match_level") or "RELATED").upper()
    spec["match_level"] = level if level in {"EXACT", "PARTIAL", "RELATED", "NO_EVIDENCE"} else "RELATED"
    spec["matched_scope"] = spec.get("matched_scope") if isinstance(spec.get("matched_scope"), dict) else {}
    spec["coverage_note"] = str(spec.get("coverage_note") or "")
    spec["reason_codes"] = _string_list(spec.get("reason_codes"))
    return spec


def _compact_bundle_for_llm(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "element_id": bundle.get("element_id"),
        "visual_anchor": bundle.get("visual_anchor"),
        "element_type": bundle.get("element_type"),
        "chart_type": bundle.get("chart_type"),
        "caption": bundle.get("caption"),
        "axis_labels": bundle.get("axis_labels"),
        "legend_entries": bundle.get("legend_entries"),
        "series_names": bundle.get("series_names"),
        "metric_names": bundle.get("metric_names"),
        "model_names": bundle.get("model_names"),
        "target_algorithms": bundle.get("target_algorithms"),
        "semantic_summary": bundle.get("semantic_summary"),
        "paper_data_series": bundle.get("paper_data_series", [])[:8],
        "claim_rows": bundle.get("claim_rows", [])[:20],
        "selected_attempts": bundle.get("selected_attempts", [])[:12],
        "candidate_metrics": bundle.get("candidate_metrics", [])[:30],
        "curves": bundle.get("curves", [])[:10],
        "skip_evidence": bundle.get("skip_evidence", [])[:10],
        "match_level": bundle.get("match_level"),
        "matched_scope": bundle.get("matched_scope"),
        "coverage_note": bundle.get("coverage_note"),
        "phase2_results_excerpt": bundle.get("phase2_results_excerpt", "")[:3000],
        "evidence_sources": bundle.get("evidence_sources", []),
    }


def _failed_figure(
    bundle: dict[str, Any],
    *,
    reason: str,
    reason_codes: list[str],
    plot_spec: dict[str, Any] | None = None,
    llm_note: str = "",
    code_path: str | None = None,
    code: str = "",
) -> ReproducedFigure:
    return ReproducedFigure(
        element_id=str(bundle.get("element_id") or ""),
        visual_anchor=str(bundle.get("visual_anchor") or ""),
        reference_image_path=bundle.get("reference_image_path"),
        reproduced_image_path=None,
        image_path="",
        comparison_notes=reason,
        evidence_sources=bundle.get("evidence_sources", []),
        reproduction_status="FAILED",
        plot_spec=plot_spec or {},
        matplotlib_code=code,
        code_path=code_path,
        llm_decision_summary=llm_note,
        match_level=str(bundle.get("match_level") or "NO_EVIDENCE"),
        matched_scope=bundle.get("matched_scope", {}),
        coverage_note=str(bundle.get("coverage_note") or ""),
        reason_codes=reason_codes,
    )


def _summarize_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for attempt in attempts:
        scope = attempt.get("scope") if isinstance(attempt.get("scope"), dict) else {}
        rows.append(
            {
                "attempt_id": attempt.get("attempt_id"),
                "experiment_id": attempt.get("experiment_id"),
                "experiment_name": attempt.get("experiment_name"),
                "scope": scope,
                "status": attempt.get("status"),
                "fidelity": attempt.get("fidelity"),
                "execution_outcome": attempt.get("execution_outcome"),
                "evidence_source": attempt.get("evidence_source"),
                "stop_reason": attempt.get("stop_reason"),
                "stdout_source": attempt.get("stdout_source"),
                "notes": attempt.get("notes"),
                "skip_reason": attempt.get("skip_reason"),
                "error_summary": attempt.get("error_summary"),
                "reason_codes": attempt.get("reason_codes", []),
                "metric_names": [m.get("metric_name") for m in attempt.get("metrics", []) if isinstance(m, dict)][:12],
                "curve_names": [c.get("metric_name") for c in attempt.get("curves", []) if isinstance(c, dict)][:8],
            }
        )
    return rows


def _algorithms_for_visual(target: dict[str, Any], element: dict[str, Any]) -> set[str]:
    text = " ".join(
        [
            str(target.get("caption") or ""),
            str(target.get("semantic_summary") or ""),
            " ".join(str(x) for x in target.get("model_names", [])),
            " ".join(str(x) for x in target.get("series_names", [])),
            " ".join(str(x) for x in target.get("legend_entries", [])),
            str(element.get("caption") or ""),
            " ".join(str(x) for x in element.get("model_names", [])),
            " ".join(str(x) for x in element.get("legend_entries", [])),
        ]
    )
    normalized = _norm(text)
    out = set()
    for canonical, aliases in ALGORITHM_ALIASES.items():
        if any(_has_phrase(normalized, alias) for alias in aliases):
            out.add(canonical)
    return out


def _normalize_algorithm(raw: str) -> str:
    normalized = _norm(raw)
    for canonical, aliases in ALGORITHM_ALIASES.items():
        if normalized == canonical or any(_has_phrase(normalized, alias) for alias in aliases):
            return canonical
    return normalized


def _datasets_from_text(text: str) -> set[str]:
    datasets = set()
    for name in ("mnist", "cifar10", "cifar100"):
        if name in text.replace("-", ""):
            datasets.add(name)
    if "cifar 10" in text:
        datasets.add("cifar10")
    if "cifar 100" in text:
        datasets.add("cifar100")
    return datasets


def _model_families_from_text(text: str) -> set[str]:
    families = set()
    if "fully connected" in text or "fc" in text.split():
        families.add("fc")
    if "convolution" in text or "conv" in text.split():
        families.add("conv")
    return families


def _dedupe_metric_dicts(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for metric in metrics:
        key = (
            str(metric.get("metric_name") or metric.get("raw_metric_name") or ""),
            str(metric.get("value") if metric.get("value") is not None else metric.get("value_ratio")),
            str(metric.get("source_attempt_id") or metric.get("source") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(metric)
    return out


def _dedupe_curves(curves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, int]] = set()
    out: list[dict[str, Any]] = []
    for curve in curves:
        points = curve.get("points", [])
        key = (
            str(curve.get("metric_name") or ""),
            str(curve.get("source") or ""),
            str(curve.get("algorithm") or ""),
            len(points) if isinstance(points, list) else 0,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(curve)
    return out


def _dedupe_strings(values) -> list[str]:
    seen = set()
    out = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _string_list(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return _dedupe_strings(raw)
    if raw in (None, ""):
        return []
    return [str(raw)]


def _numeric(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: Any) -> str:
    numeric = _numeric(value)
    if numeric is None:
        return str(value)
    if abs(numeric) <= 1.0:
        return f"{numeric:.4f}"
    return f"{numeric:.3g}"


def _short(text: Any, max_len: int = 60) -> str:
    value = str(text or "")
    return value if len(value) <= max_len else value[: max_len - 3] + "..."


def _short_title(bundle: dict[str, Any]) -> str:
    return _short(bundle.get("caption") or bundle.get("visual_anchor") or bundle.get("element_id"), 90)


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _has_phrase(text: str, phrase: str) -> bool:
    norm_phrase = _norm(phrase)
    return bool(norm_phrase and re.search(rf"(^| )({re.escape(norm_phrase)})( |$)", text))
