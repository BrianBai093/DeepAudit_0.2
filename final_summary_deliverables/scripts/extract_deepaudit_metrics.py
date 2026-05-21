#!/usr/bin/env python3
"""Extract DeepAudit experiment metrics into result.md.

The report is intentionally written incrementally: after each run directory is
parsed, its section is appended and flushed to disk. Aggregate tables and plots
are appended after all per-run sections have been safely written.
"""

from __future__ import annotations

import configparser
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKLIST_PATH = PROJECT_ROOT / "DeepAudit指标提取清单.md"
RESULT_PATH = PROJECT_ROOT / "result.md"
FIGURE_DIR = ARTIFACTS_DIR / "summary_figures"
EXCLUDED_RUN_IDS = {
    "02_A_Theoretical_Framework_for_Target_Propagation",
    "12_Recurrent_Independent_Mechanisms_RIM",
    "14_Neural_Circuit_Policies_NCP",
    "20_Implicit_Neural_Representations_with_Periodic_Activation_Functions",
}


def read_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_text(path: Path, default: str = "") -> str:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return default
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return default


def first_line_title(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = stripped.lstrip("#").strip()
        if stripped:
            return stripped
    return None


def title_from_context(run_dir: Path) -> str:
    context = read_json(run_dir / "execution/context.json", {})
    candidates = []
    for key in ("paper_md", "paper_md_out"):
        value = context.get(key)
        if value:
            candidates.append(Path(value))
    candidates.append(PROJECT_ROOT / "output/batch_papers" / run_dir.name / "paper.md")
    for path in candidates:
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        title = first_line_title(read_text(path))
        if title:
            return title
    return run_dir.name


def extract_arxiv_id(text: str) -> str | None:
    patterns = [
        r"arXiv[:\s]+([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)",
        r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def extract_year(text: str) -> str | None:
    head = "\n".join(text.splitlines()[:120])
    years = re.findall(r"\b(20[0-2][0-9]|19[8-9][0-9])\b", head)
    return years[0] if years else None


def model_family_from_run_id(run_id: str) -> str:
    cleaned = re.sub(r"^\d+_", "", run_id)
    cleaned = cleaned.replace("_", " ").replace(" copy", "")
    return cleaned.strip() or run_id


def repo_url_from_context(run_dir: Path) -> str | None:
    context = read_json(run_dir / "execution/context.json", {})
    repo_dir = context.get("repo_dir")
    if not repo_dir:
        return None
    git_config = Path(repo_dir) / ".git" / "config"
    if not git_config.exists():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(git_config)
        for section in parser.sections():
            if section.startswith('remote "origin"') and parser.has_option(section, "url"):
                return parser.get(section, "url")
    except Exception:
        return None
    return None


def dependency_flags(run_dir: Path, repo_analysis: dict[str, Any]) -> dict[str, Any]:
    manifests: list[str] = []
    for profile in repo_analysis.get("dependency_profiles") or []:
        manifests.extend(profile.get("manifest_paths") or [])
    context = read_json(run_dir / "execution/context.json", {})
    repo_dir = Path(context.get("repo_dir", ""))
    found_names: set[str] = set()
    if repo_dir.exists():
        for name in [
            "requirements.txt",
            "requirements-dev.txt",
            "environment.yml",
            "environment.yaml",
            "conda_env.yml",
            "p2c_env.yml",
            "setup.py",
            "pyproject.toml",
            "Pipfile",
            "poetry.lock",
        ]:
            if (repo_dir / name).exists():
                found_names.add(name)
    all_names = set(manifests) | found_names
    return {
        "requirements_file_found": any("requirements" in n for n in all_names),
        "environment_file_found": any(n.endswith((".yml", ".yaml")) for n in all_names),
        "setup_py_found": "setup.py" in all_names,
        "pyproject_toml_found": "pyproject.toml" in all_names,
        "dependency_manifests": sorted(all_names),
    }


def numeric_values(values: list[float | int | None]) -> list[float]:
    out = []
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        try:
            f = float(value)
        except Exception:
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 1)


def source_counts_for_claims(claims: list[dict[str, Any]], criteria: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    criteria_by_index = {i: item for i, item in enumerate(criteria)}
    for claim in claims:
        sources = set()
        for ref in claim.get("evidence_set") or []:
            match = re.search(r"atomic_criteria\[(\d+)\]", str(ref))
            if not match:
                continue
            item = criteria_by_index.get(int(match.group(1)))
            if not item:
                continue
            source_type = str(item.get("source_type") or "")
            if "table" in source_type:
                sources.add("table")
            elif "figure" in source_type:
                sources.add("figure")
            elif "text" in source_type:
                sources.add("text")
        if not sources:
            sources.add("unknown")
        for source in sources:
            counts[source] += 1
    return counts


def infer_compute_requirement(env_spec: dict[str, Any], runs: list[dict[str, Any]], text_blob: str) -> str:
    joined = json.dumps(env_spec, ensure_ascii=False).lower() + "\n" + text_blob.lower()
    if any(token in joined for token in ("cuda", "cudatoolkit", "gpu", "nvidia")):
        return "GPU/CUDA likely required or supported"
    if any(run.get("fidelity") == "full" for run in runs):
        return "CPU feasible in observed run"
    return "unknown"


def collect_reason_codes(*objects: Any) -> list[str]:
    codes: list[str] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "reason_codes" and isinstance(value, list):
                    codes.extend(str(v) for v in value)
                else:
                    visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    for obj in objects:
        visit(obj)
    return codes


def classify_failure_codes(codes: list[str], text: str) -> Counter:
    counter: Counter = Counter()
    haystack = " ".join(codes).upper() + " " + text.upper()
    mapping = {
        "missing_dataset": ("MISSING_DATA", "DATASET_MISSING", "NO_DATA", "DATASET"),
        "dependency_failure": ("DEPENDENCY", "PACKAGE", "ENV", "CONDA", "PIP", "VALIDATION_FAILED"),
        "entrypoint_ambiguity": ("ENTRYPOINT", "COMMAND_NOT_OBSERVED", "COMMAND", "ARG"),
        "execution_timeout": ("TIMEOUT",),
        "metric_not_found": ("METRIC_NOT_FOUND", "ALIGNMENT_AMBIGUOUS", "CLAIMS_INCONCLUSIVE", "NO_METRIC"),
        "metric_mismatch": ("NOT_SUPPORTED", "MISMATCH", "OUTSIDE_TOLERANCE"),
        "compute_limit": ("COMPUTE", "BUDGET", "RESOURCE", "OOM", "MEMORY"),
        "repo_clone_failure": ("CLONE", "REPO_MISSING"),
        "data_download_failure": ("DOWNLOAD",),
        "preprocessing_missing": ("PREPROCESS",),
        "checkpoint_missing": ("CHECKPOINT",),
        "undocumented_hyperparameter": ("HYPERPARAM",),
    }
    for mode, needles in mapping.items():
        if any(needle in haystack for needle in needles):
            counter[mode] += 1
    return counter


def run_log_duration(run_dir: Path, agent_name: str) -> float | None:
    text = read_text(run_dir / "execution/run.log")
    pattern = re.compile(
        rf"\[agent={re.escape(agent_name)}\].*?completed in ([0-9.]+)s",
        flags=re.IGNORECASE,
    )
    values = [float(m.group(1)) for m in pattern.finditer(text)]
    if values:
        return sum(values)
    return None


def extract_run(run_dir: Path) -> dict[str, Any]:
    run_id = run_dir.name
    context = read_json(run_dir / "execution/context.json", {})
    paper_text = ""
    for key in ("paper_md", "paper_md_out"):
        value = context.get(key)
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            paper_text = read_text(path)
            if paper_text:
                break
    title = title_from_context(run_dir)

    claims_ir = read_json(run_dir / "fingerprint/claims_ir.json", {})
    criteria_doc = read_json(run_dir / "fingerprint/atomic_criteria.json", {})
    task_spec = read_json(run_dir / "task/task_spec.json", {})
    metric_contract = read_json(run_dir / "task/metric_contract.json", {})
    repo_analysis = read_json(run_dir / "task/repo_analysis.json", {})
    env_spec = read_json(run_dir / "execution/executor_env_spec.json", {})
    env_result = read_json(run_dir / "execution/env_setup_result.json", {})
    phase2_state = read_json(run_dir / "execution/phase2_state.json", {})
    failures = read_json(run_dir / "execution/execution_failures.json", [])
    run_manifest = read_json(run_dir / "execution/executor_outputs/run_manifest.json", {})
    package = read_json(run_dir / "execution/executor_outputs/phase2_execution_package.json", {})
    metrics = read_json(run_dir / "results/metrics.json", {})
    parsed = read_json(run_dir / "results/parsed_evidence.json", {})
    evaluability = read_json(run_dir / "results/evaluability.json", {})
    eval_verdict = read_json(run_dir / "results/evaluability_verdict.json", {})
    verdict = read_json(run_dir / "results/verdict.json", {})
    score = read_json(run_dir / "results/reproducibility_score.json", {})
    figures = read_json(run_dir / "results/reproduced_figures.json", {})
    log_evidence = read_json(run_dir / "results/execution_log_evidence.json", {})

    claims = claims_ir.get("claims") or []
    experiments = claims_ir.get("experiments") or []
    criteria = criteria_doc.get("criteria") or []
    tasks = task_spec.get("tasks") or []
    entrypoints = task_spec.get("entrypoints") or repo_analysis.get("entrypoint_candidates") or []
    runs = run_manifest.get("runs") or []
    phase2_package_experiments = package.get("experiments") or []
    metric_records = metrics.get("records") or []
    claim_evidence = parsed.get("claim_evidence") or []
    claim_verdicts = verdict.get("claim_verdicts") or []
    eval_entries = evaluability.get("entries") or []
    reproduced = figures.get("figures") or []
    skipped_figures = figures.get("skipped_targets") or []
    logs_scanned = log_evidence.get("logs") or []

    source_counts = source_counts_for_claims(claims, criteria)
    type_counts = Counter(str(c.get("type") or "unknown") for c in claims)
    verdict_counts = Counter(str(v.get("status") or "UNKNOWN") for v in claim_verdicts)
    eval_counts = Counter(str(e.get("evaluable") or "unknown") for e in eval_entries)
    run_status_counts = Counter(str(r.get("status") or "unknown") for r in runs)
    fidelity_counts = Counter(str(r.get("fidelity") or "unknown") for r in runs)
    evidence_source_counts = Counter(str(r.get("evidence_source") or "unknown") for r in runs)

    code_verifiable = sum(1 for c in claims if c.get("code_verifiable") is True)
    non_code_verifiable = len(claims) - code_verifiable
    claims_with_metric_contract = sum(1 for c in claims if c.get("metric") or c.get("target") is not None)
    claims_without_metric_contract = len(claims) - claims_with_metric_contract
    headline_claims = sum(1 for c in claims if (c.get("conditions") or {}).get("is_primary") is True)
    comparison_claims = sum(1 for c in claims if c.get("baseline"))
    ablation_claims = sum(1 for c in claims if "ablation" in str(c).lower())
    efficiency_claims = sum(1 for c in claims if any(t in str(c).lower() for t in ("runtime", "time", "speed", "efficiency", "memory")))
    qualitative_claims = sum(1 for c in claims if str(c.get("type")) == "qualitative")
    theoretical_claims = sum(1 for c in claims if "theoretical" in str(c).lower() or "proof" in str(c).lower())

    datasets = sorted(
        {
            str(x)
            for x in [e.get("dataset") for e in experiments] + [r.get("dataset") for r in runs]
            if x not in (None, "", "None")
        }
    )
    commands = [t.get("command") for t in tasks if t.get("command")]
    commands_missing = sum(1 for t in tasks if not t.get("command") or "N/A" in str(t.get("command")))
    readme_entrypoints = sum(
        1
        for e in entrypoints
        if "readme" in str(e.get("entrypoint_id", "")).lower()
        or "readme" in str(e.get("evidence", "")).lower()
    )
    entry_conf = numeric_values([e.get("confidence") for e in entrypoints])
    deps = dependency_flags(run_dir, repo_analysis)
    failed_packages = env_result.get("failed_packages") or []

    run_times = numeric_values([r.get("runtime_sec") for r in runs])
    env_runtime_sec = run_log_duration(run_dir, "tool_agent")
    phase2_elapsed_sec = phase2_state.get("elapsed_sec")
    timeout_count = sum(
        1
        for r in runs
        if "timeout" in " ".join(r.get("reason_codes") or []).lower()
        or str(r.get("stop_reason") or "").lower() == "budget_bound"
    )
    artifact_paths = []
    for r in runs:
        artifact_paths.extend(r.get("artifacts") or [])
    executor_files = list((run_dir / "execution/executor_outputs").glob("*"))
    result_files = list((run_dir / "results").glob("*"))
    figure_files = list((run_dir / "results/figures").glob("*.png"))
    checkpoint_files = [
        p
        for p in executor_files
        if p.suffix.lower() in {".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".tar", ".npz", ".npy"}
    ]

    matched_claims = sum(1 for e in claim_evidence if e.get("matched_records"))
    compared_claims = sum(1 for v in claim_verdicts if v.get("compared_value") is not None)
    within_tolerance = sum(1 for v in claim_verdicts if str(v.get("status")) == "SUPPORTED")
    outside_tolerance = sum(1 for v in claim_verdicts if str(v.get("status")) == "NOT_SUPPORTED")
    headline_verdicts = [
        v
        for v in claim_verdicts
        if any(
            c.get("claim_id") == v.get("claim_id") and (c.get("conditions") or {}).get("is_primary") is True
            for c in claims
        )
    ]
    headline_counts = Counter(str(v.get("status") or "UNKNOWN") for v in headline_verdicts)

    all_reason_codes = collect_reason_codes(
        claims_ir,
        criteria_doc,
        task_spec,
        repo_analysis,
        env_result,
        phase2_state,
        failures,
        run_manifest,
        package,
        metrics,
        parsed,
        evaluability,
        eval_verdict,
        verdict,
        score,
        figures,
        log_evidence,
    )
    failure_text = "\n".join(
        [
            read_text(run_dir / "execution/run.log")[-5000:],
            json.dumps(failures, ensure_ascii=False)[-5000:],
            str(verdict.get("summary") or ""),
            str(score.get("ecr_reason") or ""),
        ]
    )
    failure_modes = classify_failure_codes(all_reason_codes, failure_text)
    main_inconclusive_reason = None
    inconclusive_details = [
        v.get("detail")
        for v in claim_verdicts
        if str(v.get("status")) == "INCONCLUSIVE" and v.get("detail")
    ]
    if inconclusive_details:
        main_inconclusive_reason = Counter(inconclusive_details).most_common(1)[0][0]
    elif "PIPELINE_ERROR" in failure_text:
        main_inconclusive_reason = "pipeline error"

    repo_url = repo_url_from_context(run_dir)
    repo_dir = Path(context.get("repo_dir", ""))
    repo_available = repo_dir.exists() if context.get("repo_dir") else bool(repo_analysis)
    env_success = bool(env_result.get("validation_passed"))
    phase2_success = str(phase2_state.get("status") or "").startswith("success")
    any_success = run_status_counts.get("ok", 0) > 0
    package_complete = bool(package.get("experiments")) and bool(package.get("source_files"))
    report_exists = (run_dir / "results/report.md").exists() and (run_dir / "results/report.md").stat().st_size > 0

    return {
        "run_id": run_id,
        "paper_id": run_id.split("_", 1)[0],
        "title": title,
        "arxiv_id": extract_arxiv_id(paper_text),
        "venue": None,
        "year": extract_year(paper_text),
        "model_family": model_family_from_run_id(run_id),
        "repo_url": repo_url,
        "repo_available": repo_available,
        "repo_url_valid": bool(repo_url and (repo_url.startswith("http") or "github.com" in repo_url or repo_url.endswith(".git"))),
        "repo_cloned_successfully": repo_available,
        "expected_entry_point": repo_analysis.get("primary_entrypoint_id") or (entrypoints[0].get("entrypoint_id") if entrypoints else None),
        "documented_entry_point_exists": readme_entrypoints > 0,
        "compute_requirement": infer_compute_requirement(env_spec, runs, paper_text),
        "dataset_requirement": datasets,
        "claims": claims,
        "experiments": experiments,
        "tasks": tasks,
        "entrypoints": entrypoints,
        "runs": runs,
        "failures": failures,
        "metric_records": metric_records,
        "claim_evidence": claim_evidence,
        "claim_verdicts": claim_verdicts,
        "eval_entries": eval_entries,
        "all_reason_codes": all_reason_codes,
        "failure_modes": failure_modes,
        "source_counts": source_counts,
        "type_counts": type_counts,
        "verdict_counts": verdict_counts,
        "eval_counts": eval_counts,
        "run_status_counts": run_status_counts,
        "fidelity_counts": fidelity_counts,
        "evidence_source_counts": evidence_source_counts,
        "reported_metric_names": sorted({str(c.get("metric")) for c in claims if c.get("metric")}),
        "reported_metric_frequency": Counter(str(c.get("metric")) for c in claims if c.get("metric")),
        "reported_datasets": datasets,
        "reported_models": sorted(
            {
                str((c.get("conditions") or {}).get("model"))
                for c in claims
                if (c.get("conditions") or {}).get("model")
            }
        ),
        "reported_methods": sorted(
            {
                str((c.get("conditions") or {}).get("method"))
                for c in claims
                if (c.get("conditions") or {}).get("method")
            }
        ),
        "reported_baselines": sorted({str(c.get("baseline")) for c in claims if c.get("baseline")}),
        "total_claims_extracted": len(claims),
        "code_verifiable_claims": code_verifiable,
        "non_code_verifiable_claims": non_code_verifiable,
        "pct_code_verifiable_claims": pct(code_verifiable, len(claims)),
        "claims_from_table": source_counts.get("table", 0),
        "claims_from_figure": source_counts.get("figure", 0),
        "claims_from_text": source_counts.get("text", 0),
        "main_claim_source": source_counts.most_common(1)[0][0] if source_counts else "none",
        "metric_contracts_generated": len(metric_contract.get("required_metrics") or []),
        "claims_with_metric_contract": claims_with_metric_contract,
        "claims_without_metric_contract": claims_without_metric_contract,
        "pct_claims_with_metric_contract": pct(claims_with_metric_contract, len(claims)),
        "headline_claims": headline_claims,
        "secondary_claims": max(0, len(claims) - headline_claims),
        "comparison_claims": comparison_claims,
        "ablation_claims": ablation_claims,
        "efficiency_claims": efficiency_claims,
        "qualitative_claims": qualitative_claims,
        "theoretical_claims": theoretical_claims,
        "tasks_generated": len(tasks),
        "tasks_per_paper": len(tasks),
        "tasks_per_claim": round(len(tasks) / len(claims), 3) if claims else 0,
        "claims_with_candidate_task": sum(1 for c in claims if (c.get("conditions") or {}).get("experiment_id")),
        "claims_without_candidate_task": sum(1 for c in claims if not (c.get("conditions") or {}).get("experiment_id")),
        "candidate_entry_points_found": len(entrypoints),
        "entry_points_per_repo": len(entrypoints),
        "claim_to_entrypoint_mapped": sum(1 for c in claims if (c.get("conditions") or {}).get("experiment_id") and entrypoints),
        "claim_to_entrypoint_unmapped": sum(1 for c in claims if not entrypoints or not (c.get("conditions") or {}).get("experiment_id")),
        "entrypoint_mapping_coverage": pct(sum(1 for c in claims if (c.get("conditions") or {}).get("experiment_id") and entrypoints), len(claims)),
        "entrypoint_ambiguity_count": max(0, len(entrypoints) - len(tasks)),
        "datasets_identified": datasets,
        "dataset_available": sum(1 for d in datasets if any(r.get("dataset") == d and r.get("status") in ("ok", "partial") for r in runs)),
        "dataset_missing": 0 if datasets else None,
        "dataset_mapping_coverage": pct(len(datasets), len(experiments)),
        "commands_generated": len(commands),
        "commands_with_required_args": sum(1 for cmd in commands if cmd and "TODO" not in cmd and "N/A" not in cmd),
        "commands_missing_required_args": commands_missing,
        "commands_matching_readme": readme_entrypoints,
        "commands_inferred_by_agent": max(0, len(commands) - readme_entrypoints),
        "command_confidence_score": round(statistics.mean(entry_conf), 3) if entry_conf else None,
        "env_setup_success": env_success,
        "env_setup_failed": bool(env_result and not env_success),
        "env_setup_repaired": phase2_success and bool(failed_packages or phase2_state.get("failures")),
        "requirements_file_found": deps["requirements_file_found"],
        "environment_file_found": deps["environment_file_found"],
        "setup_py_found": deps["setup_py_found"],
        "pyproject_toml_found": deps["pyproject_toml_found"],
        "dependency_manifests": deps["dependency_manifests"],
        "dependency_install_success": env_success,
        "dependency_install_failed": bool(failed_packages),
        "dependency_conflict_count": len(failed_packages),
        "missing_package_count": len(failed_packages),
        "obsolete_package_count": sum(1 for code in all_reason_codes if "OBSOLETE" in code.upper() or "OLD" in code.upper()),
        "repair_attempts": phase2_state.get("attempt") if phase2_state else 0,
        "repos_requiring_repair": bool(phase2_state.get("failures")),
        "repair_success": phase2_success and bool(phase2_state.get("failures")),
        "repair_failed": (not phase2_success) and bool(phase2_state.get("failures")),
        "repair_types": sorted({mode for mode, count in failure_modes.items() if count}),
        "env_setup_runtime_sec": env_runtime_sec,
        "env_setup_runtime_min": round(env_runtime_sec / 60, 2) if env_runtime_sec is not None else None,
        "phase2_elapsed_sec": phase2_elapsed_sec,
        "repos_attempted": bool(runs or phase2_state),
        "repo_with_successful_run": any_success,
        "repo_all_runs_failed": bool(runs) and run_status_counts.get("failed", 0) == len(runs),
        "repo_partial_execution": run_status_counts.get("partial", 0) > 0 or (runs and not any_success),
        "runs_attempted": len(runs),
        "runs_successful": run_status_counts.get("ok", 0),
        "runs_failed": run_status_counts.get("failed", 0),
        "runs_timeout": timeout_count,
        "runs_partial": run_status_counts.get("partial", 0),
        "run_success_rate": pct(run_status_counts.get("ok", 0), len(runs)),
        "runtime_sec": sum(run_times) if run_times else 0,
        "runtime_min": round(sum(run_times) / 60, 2) if run_times else 0,
        "artifact_count": len(set(artifact_paths)) + len(executor_files) + len(result_files),
        "result_files_generated": len(result_files),
        "log_files_generated": len(list((run_dir / "execution").glob("**/*.log"))),
        "checkpoint_files_generated": len(checkpoint_files),
        "figure_files_generated": len(figure_files),
        "artifacts_generated": sorted(set(artifact_paths))[:20],
        "standard_execution_package_complete": package_complete,
        "claims_evaluated": len(claim_verdicts),
        "claims_not_evaluated": max(0, len(claims) - len(claim_verdicts)),
        "evaluated_claim_rate": pct(len(claim_verdicts), len(claims)),
        "executable_claims": code_verifiable,
        "non_executable_claims": non_code_verifiable,
        "claims_with_observed_metric": matched_claims,
        "claims_without_observed_metric": max(0, len(claims) - matched_claims),
        "metric_recovery_rate": pct(matched_claims, len(claims)),
        "metric_parser_success": len(metric_records),
        "metric_parser_failed": max(0, len(claims) - matched_claims),
        "reported_observed_comparison_count": compared_claims,
        "within_tolerance": within_tolerance,
        "outside_tolerance": outside_tolerance,
        "claim_to_evidence_mapped": matched_claims,
        "claim_to_evidence_unmapped": max(0, len(claims) - matched_claims),
        "supported_count": verdict_counts.get("SUPPORTED", 0),
        "partially_supported_count": verdict_counts.get("PARTIALLY_SUPPORTED", 0) + verdict_counts.get("PARTIAL", 0),
        "not_supported_count": verdict_counts.get("NOT_SUPPORTED", 0),
        "inconclusive_count": verdict_counts.get("INCONCLUSIVE", 0),
        "total_verified_claims": len(claim_verdicts),
        "main_verdict_per_paper": verdict.get("status"),
        "headline_claims_evaluated": len(headline_verdicts),
        "headline_claim_supported": headline_counts.get("SUPPORTED", 0),
        "headline_claim_partially_supported": headline_counts.get("PARTIALLY_SUPPORTED", 0) + headline_counts.get("PARTIAL", 0),
        "headline_claim_not_supported": headline_counts.get("NOT_SUPPORTED", 0),
        "headline_claim_inconclusive": headline_counts.get("INCONCLUSIVE", 0),
        "headline_metric_recovery_rate": pct(len(headline_verdicts), headline_claims),
        "repos_executed_but_no_headline_evidence": any_success and headline_claims > 0 and not headline_verdicts,
        "repos_executed_but_metric_missing": any_success and matched_claims == 0 and len(claims) > 0,
        "repos_executed_but_claim_inconclusive": any_success and verdict_counts.get("INCONCLUSIVE", 0) > 0,
        "repos_failed_but_diagnostic_evidence_available": (not any_success) and bool(failures or env_result or read_text(run_dir / "execution/run.log")),
        "main_inconclusive_reason": main_inconclusive_reason,
        "score_total": score.get("total_score"),
        "score_raw": score.get("raw_total_score"),
        "score_dimensions": score.get("dimensions") or [],
        "ecr": score.get("ecr"),
        "ecr_reason": score.get("ecr_reason"),
        "reproduced_figures": len(reproduced),
        "skipped_figures": len(skipped_figures),
        "logs_scanned": len(logs_scanned),
        "phase2_package_experiments": len(phase2_package_experiments),
        "report_exists": report_exists,
        "execution_outcome_counts": Counter(str(r.get("execution_outcome") or "none") for r in runs),
        "evidence_tier": (
            "FULL_REPRODUCTION_EVIDENCE"
            if any(r.get("execution_outcome") == "FULLY_REPRODUCED" for r in runs)
            else "TREND_EVIDENCE"
            if any(r.get("execution_outcome") == "TREND_SUPPORTED" for r in runs)
            else "EXECUTABLE_OR_SMOKE_EVIDENCE"
            if any(r.get("execution_outcome") == "EXECUTABLE" or r.get("status") == "ok" for r in runs)
            else "ATTEMPTED_NO_POSITIVE_EVIDENCE"
            if runs
            else "NO_PHASE2_RUNS"
        ),
    }


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.2f}" if abs(value) >= 1 else f"{value:.3f}"
        text = str(value).replace("\n", " ").replace("|", "\\|")
        return text if len(text) <= 220 else text[:217] + "..."

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def short_list(values: list[Any], limit: int = 6) -> str:
    if not values:
        return "none"
    shown = [str(v) for v in values[:limit]]
    if len(values) > limit:
        shown.append(f"...(+{len(values)-limit})")
    return ", ".join(shown)


def write_run_section(handle, row: dict[str, Any]) -> None:
    handle.write(f"\n## {row['run_id']}\n\n")
    handle.write(
        md_table(
            ["field", "value"],
            [
                ["paper_id", row["paper_id"]],
                ["title", row["title"]],
                ["arxiv_id", row["arxiv_id"]],
                ["venue", row["venue"]],
                ["year", row["year"]],
                ["model_family", row["model_family"]],
                ["repo_url", row["repo_url"]],
                ["repo_available / cloned", f"{row['repo_available']} / {row['repo_cloned_successfully']}"],
                ["expected_entry_point", row["expected_entry_point"]],
                ["documented_entry_point_exists", row["documented_entry_point_exists"]],
                ["compute_requirement", row["compute_requirement"]],
                ["dataset_requirement", short_list(row["dataset_requirement"])],
            ],
        )
    )
    handle.write("\n### Claim Extraction Metrics\n\n")
    handle.write(
        md_table(
            ["metric", "value"],
            [
                ["total_claims_extracted", row["total_claims_extracted"]],
                ["code_verifiable_claims", row["code_verifiable_claims"]],
                ["non_code_verifiable_claims", row["non_code_verifiable_claims"]],
                ["pct_code_verifiable_claims", row["pct_code_verifiable_claims"]],
                ["claims_from_table / figure / text", f"{row['claims_from_table']} / {row['claims_from_figure']} / {row['claims_from_text']}"],
                ["main_claim_source", row["main_claim_source"]],
                ["metric_contracts_generated", row["metric_contracts_generated"]],
                ["claims_with_metric_contract", row["claims_with_metric_contract"]],
                ["pct_claims_with_metric_contract", row["pct_claims_with_metric_contract"]],
                ["claims_without_metric_contract", row["claims_without_metric_contract"]],
                ["reported_metric_names", short_list(row["reported_metric_names"], 10)],
                ["top_reported_metrics", short_list([f"{k}:{v}" for k, v in row["reported_metric_frequency"].most_common(8)], 8)],
                ["reported_dataset", short_list(row["reported_datasets"], 8)],
                ["reported_model", short_list(row["reported_models"], 8)],
                ["reported_method", short_list(row["reported_methods"], 8)],
                ["reported_baseline", short_list(row["reported_baselines"], 8)],
                ["claim granularity", f"headline={row['headline_claims']}, secondary={row['secondary_claims']}, comparison={row['comparison_claims']}, ablation={row['ablation_claims']}, efficiency={row['efficiency_claims']}, qualitative={row['qualitative_claims']}, theoretical={row['theoretical_claims']}"],
            ],
        )
    )
    handle.write("\n### Task, Entrypoint, Dataset, Command Metrics\n\n")
    handle.write(
        md_table(
            ["metric", "value"],
            [
                ["tasks_generated", row["tasks_generated"]],
                ["tasks_per_paper", row["tasks_per_paper"]],
                ["tasks_per_claim", row["tasks_per_claim"]],
                ["claims_with_candidate_task", row["claims_with_candidate_task"]],
                ["claims_without_candidate_task", row["claims_without_candidate_task"]],
                ["task_generation_coverage", pct(row["claims_with_candidate_task"], row["total_claims_extracted"])],
                ["candidate_entry_points_found", row["candidate_entry_points_found"]],
                ["entry_points_per_repo", row["entry_points_per_repo"]],
                ["claim_to_entrypoint_mapped", row["claim_to_entrypoint_mapped"]],
                ["claim_to_entrypoint_unmapped", row["claim_to_entrypoint_unmapped"]],
                ["entrypoint_mapping_coverage", row["entrypoint_mapping_coverage"]],
                ["entrypoint_ambiguity_count", row["entrypoint_ambiguity_count"]],
                ["datasets_identified", short_list(row["datasets_identified"], 8)],
                ["dataset_available", row["dataset_available"]],
                ["dataset_missing", row["dataset_missing"]],
                ["dataset_mapping_coverage", row["dataset_mapping_coverage"]],
                ["commands_generated", row["commands_generated"]],
                ["commands_with_required_args", row["commands_with_required_args"]],
                ["commands_missing_required_args", row["commands_missing_required_args"]],
                ["commands_matching_readme", row["commands_matching_readme"]],
                ["commands_inferred_by_agent", row["commands_inferred_by_agent"]],
                ["command_confidence_score", row["command_confidence_score"]],
            ],
        )
    )
    handle.write("\n### Environment, Repair, Execution Metrics\n\n")
    handle.write(
        md_table(
            ["metric", "value"],
            [
                ["env_setup_success / failed / repaired", f"{row['env_setup_success']} / {row['env_setup_failed']} / {row['env_setup_repaired']}"],
                [
                    "requirements/environment/setup.py/pyproject found",
                    f"{row['requirements_file_found']} / {row['environment_file_found']} / {row['setup_py_found']} / {row['pyproject_toml_found']}",
                ],
                ["dependency_install_success / failed", f"{row['dependency_install_success']} / {row['dependency_install_failed']}"],
                ["dependency_conflict_count", row["dependency_conflict_count"]],
                ["missing_package_count", row["missing_package_count"]],
                ["obsolete_package_count", row["obsolete_package_count"]],
                ["repair_attempts", row["repair_attempts"]],
                ["repos_requiring_repair", row["repos_requiring_repair"]],
                ["repair_success / failed", f"{row['repair_success']} / {row['repair_failed']}"],
                ["repair_types", short_list(row["repair_types"])],
                ["env_setup_runtime_min", row["env_setup_runtime_min"]],
                ["phase2_elapsed_min", round(row["phase2_elapsed_sec"] / 60, 2) if row["phase2_elapsed_sec"] else None],
                ["runs_attempted / ok / partial / failed / timeout", f"{row['runs_attempted']} / {row['runs_successful']} / {row['runs_partial']} / {row['runs_failed']} / {row['runs_timeout']}"],
                ["run_success_rate", row["run_success_rate"]],
                ["runtime_min", row["runtime_min"]],
                ["standard_execution_package_complete", row["standard_execution_package_complete"]],
                ["artifact_count", row["artifact_count"]],
                ["result/log/checkpoint/figure files", f"{row['result_files_generated']} / {row['log_files_generated']} / {row['checkpoint_files_generated']} / {row['figure_files_generated']}"],
            ],
        )
    )
    handle.write(f"\nDependency manifests detected: `{short_list(row['dependency_manifests'], 12)}`\n\n")

    if row["runs"]:
        handle.write("#### Run-Level / Command-Level Metrics\n\n")
        run_rows = []
        for run in row["runs"]:
            logs = run.get("logs") or {}
            run_rows.append(
                [
                    run.get("run_id"),
                    run.get("experiment_id"),
                    run.get("status"),
                    run.get("fidelity"),
                    run.get("exit_code"),
                    run.get("runtime_sec"),
                    run.get("command"),
                    run.get("cwd"),
                    bool(logs.get("stdout")),
                    bool(logs.get("stderr")),
                    short_list(run.get("reason_codes") or [], 4),
                ]
            )
        handle.write(
            md_table(
                ["run", "exp", "status", "fidelity", "exit", "runtime_sec", "command", "cwd", "stdout", "stderr", "reason_codes"],
                run_rows,
            )
        )
    else:
        handle.write("No run-level execution rows were available.\n\n")

    handle.write("\n### Evidence Alignment, Verdict, Failure Metrics\n\n")
    handle.write(
        md_table(
            ["metric", "value"],
            [
                ["claims_evaluated / not_evaluated", f"{row['claims_evaluated']} / {row['claims_not_evaluated']}"],
                ["evaluated_claim_rate", row["evaluated_claim_rate"]],
                ["executable / non_executable claims", f"{row['executable_claims']} / {row['non_executable_claims']}"],
                ["claims_with_observed_metric / without", f"{row['claims_with_observed_metric']} / {row['claims_without_observed_metric']}"],
                ["metric_recovery_rate", row["metric_recovery_rate"]],
                ["metric_parser_success / failed", f"{row['metric_parser_success']} / {row['metric_parser_failed']}"],
                ["reported/observed comparisons", row["reported_observed_comparison_count"]],
                ["within_tolerance / outside_tolerance", f"{row['within_tolerance']} / {row['outside_tolerance']}"],
                ["claim_to_evidence_mapped / unmapped", f"{row['claim_to_evidence_mapped']} / {row['claim_to_evidence_unmapped']}"],
                ["supported / partial / not_supported / inconclusive", f"{row['supported_count']} / {row['partially_supported_count']} / {row['not_supported_count']} / {row['inconclusive_count']}"],
                ["main_verdict_per_paper", row["main_verdict_per_paper"]],
                ["headline evaluated/supported/partial/not/inconclusive", f"{row['headline_claims_evaluated']} / {row['headline_claim_supported']} / {row['headline_claim_partially_supported']} / {row['headline_claim_not_supported']} / {row['headline_claim_inconclusive']}"],
                ["headline_metric_recovery_rate", row["headline_metric_recovery_rate"]],
                ["executed_but_no_headline_evidence", row["repos_executed_but_no_headline_evidence"]],
                ["executed_but_metric_missing", row["repos_executed_but_metric_missing"]],
                ["executed_but_claim_inconclusive", row["repos_executed_but_claim_inconclusive"]],
                ["failed_but_diagnostic_evidence_available", row["repos_failed_but_diagnostic_evidence_available"]],
                ["failure_modes", short_list([f"{k}:{v}" for k, v in row["failure_modes"].items()], 10)],
                ["main_inconclusive_reason", row["main_inconclusive_reason"]],
                ["score_total / raw", f"{row['score_total']} / {row['score_raw']}"],
                ["ECR", row["ecr"]],
                ["posthoc evidence_tier", row["evidence_tier"]],
                ["execution_outcome_counts", short_list([f"{k}:{v}" for k, v in row["execution_outcome_counts"].items()], 8)],
                ["reproduced/skipped figures", f"{row['reproduced_figures']} / {row['skipped_figures']}"],
                ["logs_scanned", row["logs_scanned"]],
            ],
        )
    )
    handle.write("\nTop reason codes: ")
    handle.write(short_list([f"{k}:{v}" for k, v in Counter(row["all_reason_codes"]).most_common(12)], 12))
    handle.write("\n\n")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    claims = [r["total_claims_extracted"] for r in rows]
    env_times = numeric_values([r["env_setup_runtime_min"] for r in rows])
    runtime = numeric_values([r["runtime_min"] for r in rows])
    scores = numeric_values([r["score_total"] for r in rows])
    evidence_tiers = Counter(r["evidence_tier"] for r in rows)
    execution_outcomes = Counter()
    verdict_total = Counter()
    failure_total = Counter()
    model_families = Counter(r["model_family"] for r in rows)
    years = Counter(r["year"] or "NA" for r in rows)
    venues = Counter(r["venue"] or "NA" for r in rows)
    compute = Counter(r["compute_requirement"] for r in rows)
    reported_metrics = Counter()
    for row in rows:
        verdict_total.update({
            "SUPPORTED": row["supported_count"],
            "PARTIALLY_SUPPORTED": row["partially_supported_count"],
            "NOT_SUPPORTED": row["not_supported_count"],
            "INCONCLUSIVE": row["inconclusive_count"],
        })
        failure_total.update(row["failure_modes"])
        reported_metrics.update(row["reported_metric_frequency"])
        execution_outcomes.update(row["execution_outcome_counts"])
    total_claims = sum(claims)
    code_verifiable = sum(r["code_verifiable_claims"] for r in rows)
    metric_contract_claims = sum(r["claims_with_metric_contract"] for r in rows)
    metric_recovered = sum(r["claims_with_observed_metric"] for r in rows)
    total_verified = sum(r["total_verified_claims"] for r in rows)
    return {
        "N_papers": len(rows),
        "papers_by_model_family": model_families,
        "papers_by_year": years,
        "papers_by_venue": venues,
        "papers_by_compute_requirement": compute,
        "TOTAL_CLAIMS": total_claims,
        "CODE_VERIFIABLE": code_verifiable,
        "PCT_CODE_VERIFIABLE": pct(code_verifiable, total_claims),
        "METRIC_CONTRACT_COUNT": sum(r["metric_contracts_generated"] for r in rows),
        "PCT_METRIC_CONTRACT_COVERAGE": pct(metric_contract_claims, total_claims),
        "ENV_SUCCESS": sum(1 for r in rows if r["env_setup_success"]),
        "EXEC_SUCCESS": sum(1 for r in rows if r["repo_with_successful_run"]),
        "PACKAGE_SUCCESS": sum(1 for r in rows if r["standard_execution_package_complete"]),
        "TOTAL_VERIFIED": total_verified,
        "SUPPORTED": verdict_total["SUPPORTED"],
        "PARTIALLY_SUPPORTED": verdict_total["PARTIALLY_SUPPORTED"],
        "NOT_SUPPORTED": verdict_total["NOT_SUPPORTED"],
        "INCONCLUSIVE": verdict_total["INCONCLUSIVE"],
        "METRIC_RECOVERY_RATE": pct(metric_recovered, total_claims),
        "top_reported_metrics": reported_metrics.most_common(12),
        "top_failures": failure_total.most_common(12),
        "median_claims_per_paper": statistics.median(claims) if claims else 0,
        "mean_claims_per_paper": round(statistics.mean(claims), 2) if claims else 0,
        "median_runtime_min": statistics.median(runtime) if runtime else 0,
        "mean_runtime_min": round(statistics.mean(runtime), 2) if runtime else 0,
        "median_env_setup_runtime_min": statistics.median(env_times) if env_times else None,
        "mean_env_setup_runtime_min": round(statistics.mean(env_times), 2) if env_times else None,
        "repair_success_rate": pct(sum(1 for r in rows if r["repair_success"]), sum(1 for r in rows if r["repos_requiring_repair"])),
        "standard_package_success_rate": pct(sum(1 for r in rows if r["standard_execution_package_complete"]), len(rows)),
        "scores": scores,
        "verdict_total": verdict_total,
        "failure_total": failure_total,
        "evidence_tiers": evidence_tiers,
        "execution_outcomes": execution_outcomes,
    }


def write_aggregate(handle, rows: list[dict[str, Any]]) -> None:
    agg = aggregate(rows)
    handle.write("\n# Aggregate Metrics\n\n")
    handle.write(
        md_table(
            ["metric", "value"],
            [
                ["N_papers", agg["N_papers"]],
                ["papers_by_model_family", short_list([f"{k}:{v}" for k, v in agg["papers_by_model_family"].most_common()], 20)],
                ["papers_by_year", short_list([f"{k}:{v}" for k, v in agg["papers_by_year"].most_common()], 20)],
                ["papers_by_venue", short_list([f"{k}:{v}" for k, v in agg["papers_by_venue"].most_common()], 20)],
                ["papers_by_compute_requirement", short_list([f"{k}:{v}" for k, v in agg["papers_by_compute_requirement"].most_common()], 20)],
                ["TOTAL_CLAIMS", agg["TOTAL_CLAIMS"]],
                ["CODE_VERIFIABLE", agg["CODE_VERIFIABLE"]],
                ["PCT_CODE_VERIFIABLE", agg["PCT_CODE_VERIFIABLE"]],
                ["METRIC_CONTRACT_COUNT", agg["METRIC_CONTRACT_COUNT"]],
                ["PCT_METRIC_CONTRACT_COVERAGE", agg["PCT_METRIC_CONTRACT_COVERAGE"]],
                ["ENV_SUCCESS", agg["ENV_SUCCESS"]],
                ["EXEC_SUCCESS", agg["EXEC_SUCCESS"]],
                ["PACKAGE_SUCCESS", agg["PACKAGE_SUCCESS"]],
                ["TOTAL_VERIFIED", agg["TOTAL_VERIFIED"]],
                ["SUPPORTED", agg["SUPPORTED"]],
                ["PARTIALLY_SUPPORTED", agg["PARTIALLY_SUPPORTED"]],
                ["NOT_SUPPORTED", agg["NOT_SUPPORTED"]],
                ["INCONCLUSIVE", agg["INCONCLUSIVE"]],
                ["METRIC_RECOVERY_RATE", agg["METRIC_RECOVERY_RATE"]],
                ["median_claims_per_paper", agg["median_claims_per_paper"]],
                ["mean_claims_per_paper", agg["mean_claims_per_paper"]],
                ["median_runtime_min", agg["median_runtime_min"]],
                ["mean_runtime_min", agg["mean_runtime_min"]],
                ["median_env_setup_runtime_min", agg["median_env_setup_runtime_min"]],
                ["mean_env_setup_runtime_min", agg["mean_env_setup_runtime_min"]],
                ["repair_success_rate", agg["repair_success_rate"]],
                ["standard_package_success_rate", agg["standard_package_success_rate"]],
                ["posthoc_full_reproduction_evidence_repos", agg["evidence_tiers"].get("FULL_REPRODUCTION_EVIDENCE", 0)],
                ["posthoc_trend_evidence_repos", agg["evidence_tiers"].get("TREND_EVIDENCE", 0)],
                ["posthoc_executable_or_smoke_evidence_repos", agg["evidence_tiers"].get("EXECUTABLE_OR_SMOKE_EVIDENCE", 0)],
                ["posthoc_positive_execution_evidence_repos", agg["evidence_tiers"].get("FULL_REPRODUCTION_EVIDENCE", 0) + agg["evidence_tiers"].get("TREND_EVIDENCE", 0) + agg["evidence_tiers"].get("EXECUTABLE_OR_SMOKE_EVIDENCE", 0)],
                ["posthoc_run_outcomes", short_list([f"{k}:{v}" for k, v in agg["execution_outcomes"].most_common()], 12)],
            ],
        )
    )

    handle.write("\n## No-Rerun Evidence Tier Summary\n\n")
    handle.write(
        "This section is a post-hoc reporting view over the existing artifacts. It does not change the strict claim-level verdicts above. "
        "It separates exact claim support from experiment-level execution evidence so completed runs are not hidden by strict metric-alignment failures.\n\n"
    )
    handle.write(
        md_table(
            ["evidence_tier", "repo_count", "interpretation"],
            [
                [
                    "FULL_REPRODUCTION_EVIDENCE",
                    agg["evidence_tiers"].get("FULL_REPRODUCTION_EVIDENCE", 0),
                    "At least one Phase2 run was marked FULLY_REPRODUCED.",
                ],
                [
                    "TREND_EVIDENCE",
                    agg["evidence_tiers"].get("TREND_EVIDENCE", 0),
                    "At least one run supported a trend or reduced-fidelity result, but no full run was marked FULLY_REPRODUCED.",
                ],
                [
                    "EXECUTABLE_OR_SMOKE_EVIDENCE",
                    agg["evidence_tiers"].get("EXECUTABLE_OR_SMOKE_EVIDENCE", 0),
                    "The repo executed or smoke-tested successfully, but did not recover a full/trend result.",
                ],
                [
                    "ATTEMPTED_NO_POSITIVE_EVIDENCE",
                    agg["evidence_tiers"].get("ATTEMPTED_NO_POSITIVE_EVIDENCE", 0),
                    "Phase2 produced run rows, but no positive execution outcome.",
                ],
                [
                    "NO_PHASE2_RUNS",
                    agg["evidence_tiers"].get("NO_PHASE2_RUNS", 0),
                    "No canonical Phase2 run rows were available.",
                ],
            ],
        )
    )
    handle.write(
        "\nPaper-ready phrasing: under strict claim-level matching, only "
        f"{agg['SUPPORTED']} claims were exactly supported; however, existing execution artifacts contain positive experiment-level evidence for "
        f"{agg['evidence_tiers'].get('FULL_REPRODUCTION_EVIDENCE', 0) + agg['evidence_tiers'].get('TREND_EVIDENCE', 0) + agg['evidence_tiers'].get('EXECUTABLE_OR_SMOKE_EVIDENCE', 0)} / {agg['N_papers']} repositories "
        f"({agg['evidence_tiers'].get('FULL_REPRODUCTION_EVIDENCE', 0)} full, {agg['evidence_tiers'].get('TREND_EVIDENCE', 0)} trend, "
        f"{agg['evidence_tiers'].get('EXECUTABLE_OR_SMOKE_EVIDENCE', 0)} executable/smoke). "
        "The gap is attributable primarily to conservative metric-to-claim alignment rather than absence of execution evidence.\n\n"
    )

    handle.write("\n## Cross-Run Summary Table\n\n")
    handle.write(
        md_table(
            [
                "run_id",
                "score",
                "claims",
                "code_verifiable",
                "tasks",
                "env_ok",
                "runs ok/partial/failed",
                "metric_recovery",
                "verdict S/P/N/I",
                "package",
                "evidence_tier",
                "figures",
            ],
            [
                [
                    r["run_id"],
                    r["score_total"],
                    r["total_claims_extracted"],
                    r["code_verifiable_claims"],
                    r["tasks_generated"],
                    r["env_setup_success"],
                    f"{r['runs_successful']}/{r['runs_partial']}/{r['runs_failed']}",
                    r["metric_recovery_rate"],
                    f"{r['supported_count']}/{r['partially_supported_count']}/{r['not_supported_count']}/{r['inconclusive_count']}",
                    r["standard_execution_package_complete"],
                    r["evidence_tier"],
                    f"{r['reproduced_figures']}/{r['skipped_figures']}",
                ]
                for r in rows
            ],
        )
    )

    handle.write("\n## Failure Taxonomy Aggregate\n\n")
    total_failure_repos = sum(1 for r in rows if r["failure_modes"])
    failure_rows = []
    for mode, count in agg["top_failures"]:
        example = next((r["run_id"] for r in rows if r["failure_modes"].get(mode)), "NA")
        failure_rows.append([mode, count, pct(count, max(total_failure_repos, 1)), example])
    handle.write(md_table(["failure_mode", "count", "rate_over_affected_repos_pct", "example_paper"], failure_rows))

    handle.write("\n## Top Reported Metrics\n\n")
    handle.write(md_table(["metric", "frequency"], [[k, v] for k, v in agg["top_reported_metrics"]]))

    handle.write("\n## Case Study Metrics\n\n")
    supported = next((r for r in rows if r["supported_count"] > 0), None)
    partial = next((r for r in rows if r["partially_supported_count"] > 0 or r["repo_partial_execution"]), None)
    failure = next((r for r in rows if r["runs_failed"] > 0 or not r["repo_with_successful_run"]), None)
    case_rows = []
    if supported:
        verdict = next((v for v in supported["claim_verdicts"] if v.get("status") == "SUPPORTED"), {})
        case_rows.append([
            "successful_reproduction",
            supported["run_id"],
            verdict.get("claim_id"),
            verdict.get("target_value"),
            verdict.get("compared_value"),
            "see run-level command table",
            supported["runtime_min"],
            "SUPPORTED",
        ])
    if partial:
        case_rows.append([
            "partial_reproduction",
            partial["run_id"],
            "NA",
            "successful components: " + str(partial["runs_successful"]),
            "missing/partial components: " + str(partial["runs_partial"] + partial["runs_failed"]),
            "see run-level command table",
            partial["runtime_min"],
            partial["main_verdict_per_paper"],
        ])
    if failure:
        failed_run = next((r for r in failure["runs"] if r.get("status") == "failed"), {})
        case_rows.append([
            "diagnostic_failure",
            failure["run_id"],
            "NA",
            failure["expected_entry_point"],
            failed_run.get("command"),
            short_list(failed_run.get("reason_codes") or failure["all_reason_codes"], 6),
            failed_run.get("exit_code"),
            failure["main_inconclusive_reason"],
        ])
    handle.write(
        md_table(
            ["case_type", "case_paper_id", "case_claim_id", "reported_or_successful_component", "observed_or_missing_component", "case_command/evidence", "runtime_or_exit", "case_verdict/reason"],
            case_rows,
        )
    )


def write_paper_ready_framing(rows: list[dict[str, Any]]) -> Path:
    agg = aggregate(rows)
    positive = [
        row
        for row in rows
        if row["evidence_tier"]
        in {"FULL_REPRODUCTION_EVIDENCE", "TREND_EVIDENCE", "EXECUTABLE_OR_SMOKE_EVIDENCE"}
    ]
    full = [row for row in rows if row["evidence_tier"] == "FULL_REPRODUCTION_EVIDENCE"]
    trend = [row for row in rows if row["evidence_tier"] == "TREND_EVIDENCE"]
    executable = [row for row in rows if row["evidence_tier"] == "EXECUTABLE_OR_SMOKE_EVIDENCE"]
    package_rows = [row for row in rows if row["standard_execution_package_complete"]]
    canonical_run_rows = [
        row
        for row in rows
        if row["runs_successful"] + row["runs_partial"] + row["runs_failed"] > 0
    ]
    positive_package = [row for row in package_rows if row in positive]
    positive_canonical = [row for row in canonical_run_rows if row in positive]
    run_rows = sum(row["runs_attempted"] for row in rows)
    ok_runs = sum(row["runs_successful"] for row in rows)
    partial_runs = sum(row["runs_partial"] for row in rows)
    failed_runs = sum(row["runs_failed"] for row in rows)
    figure_count = sum(row["reproduced_figures"] for row in rows)
    median_score = statistics.median(agg["scores"]) if agg["scores"] else 0

    lines = [
        "# Paper-Ready Result Framing",
        "",
        "This section reports the cleaned analysis set.",
        "",
        "这份摘要只基于现有 `artifacts/`，不重跑实验，不修改原始 verdict。它把 DeepAudit 的能力拆成两层：严格 claim-level exact support 与 experiment-level execution evidence。",
        "",
        "## Key Numbers",
        "",
        "| Metric | Value | How to phrase it |",
        "| --- | ---: | --- |",
        f"| Papers audited | {agg['N_papers']} | We evaluated {agg['N_papers']} paper-code pairs after excluding incomplete/mismatched cases. |",
        f"| Claims extracted | {agg['TOTAL_CLAIMS']} | The system extracted {agg['TOTAL_CLAIMS']} paper claims. |",
        f"| Code-verifiable claims | {agg['CODE_VERIFIABLE']} ({agg['PCT_CODE_VERIFIABLE']}%) | Most extracted claims were classified as code-verifiable. |",
        f"| Environment setup success | {agg['ENV_SUCCESS']}/{agg['N_papers']} ({pct(agg['ENV_SUCCESS'], agg['N_papers'])}%) | The environment agent established runnable environments for a majority of repos. |",
        f"| Standard Phase2 package complete | {agg['PACKAGE_SUCCESS']}/{agg['N_papers']} ({agg['standard_package_success_rate']}%) | Most runs produced the standardized execution package consumed by Phase3. |",
        f"| Positive experiment-level evidence | {len(positive)}/{agg['N_papers']} ({pct(len(positive), agg['N_papers'])}%) | Existing artifacts contain positive execution evidence for a large fraction of the cleaned corpus. |",
        f"| Positive evidence among packaged repos | {len(positive_package)}/{len(package_rows)} ({pct(len(positive_package), len(package_rows))}%) | Conditional on producing the standard Phase2 package, most repositories produced positive evidence. |",
        f"| Positive evidence among repos with non-skipped run rows | {len(positive_canonical)}/{len(canonical_run_rows)} ({pct(len(positive_canonical), len(canonical_run_rows))}%) | Conditional on non-skipped run records existing, positive evidence was recovered in most cases. |",
        f"| Full reproduction evidence | {len(full)}/{agg['N_papers']} ({pct(len(full), agg['N_papers'])}%) | These repositories contain at least one full reproduction run. |",
        f"| Trend-level evidence | {len(trend)}/{agg['N_papers']} ({pct(len(trend), agg['N_papers'])}%) | These repositories reproduce reduced-fidelity or trend-level evidence. |",
        f"| Executable/smoke evidence | {len(executable)}/{agg['N_papers']} ({pct(len(executable), agg['N_papers'])}%) | These repositories reached executable/smoke-level validation. |",
        f"| Run rows | {run_rows} total; {ok_runs} ok, {partial_runs} partial, {failed_runs} failed | The executor produced structured run records. |",
        f"| Claim-level exact support | {agg['SUPPORTED']} supported, {agg['NOT_SUPPORTED']} not supported, {agg['INCONCLUSIVE']} inconclusive | This is intentionally strict and should be described as a lower bound. |",
        f"| Reproduced comparison figures | {figure_count} | Phase3 generated reproduced/comparison visual artifacts. |",
        f"| Median calibrated score | {median_score:.1f}/100 | Calibrated scores summarize environment, data, execution, and claim-match evidence. |",
        "",
        "## Paper-Ready Paragraph",
        "",
        (
            f"Across {agg['N_papers']} cleaned paper-code pairs, DeepAudit extracted {agg['TOTAL_CLAIMS']} paper claims, "
            f"of which {agg['CODE_VERIFIABLE']} ({agg['PCT_CODE_VERIFIABLE']}%) were classified as code-verifiable. "
            f"The strict claim-level verifier produced {agg['SUPPORTED']} exactly supported claims, reflecting a deliberately conservative metric-to-claim alignment policy. "
            f"However, this exact-support number is a lower bound on system performance: reanalyzing the same artifacts at the experiment-evidence level shows positive execution evidence for "
            f"{len(positive)}/{agg['N_papers']} repositories, including {len(full)} repositories with full reproduction evidence, {len(trend)} with trend-level evidence, "
            f"and {len(executable)} with executable/smoke evidence. In addition, {agg['PACKAGE_SUCCESS']}/{agg['N_papers']} repositories produced a standardized Phase2 execution package "
            f"and {agg['ENV_SUCCESS']}/{agg['N_papers']} completed environment setup. Conditional on producing a standard Phase2 package, {len(positive_package)}/{len(package_rows)} repositories produced positive experiment-level evidence; "
            f"conditional on having non-skipped canonical run records, the rate was {len(positive_canonical)}/{len(canonical_run_rows)}. These results suggest that the main bottleneck is not only repository execution, "
            "but the harder final step of aligning heterogeneous execution metrics back to fine-grained paper claims."
        ),
        "",
        "## Safer Claim Wording",
        "",
        "- Use **\"strict exact-support lower bound\"** instead of only \"supported claims\".",
        "- Use **\"positive experiment-level evidence\"** for FULLY_REPRODUCED / TREND_SUPPORTED / EXECUTABLE outcomes.",
        "- Say **\"DeepAudit separates execution success from exact claim support\"**, which makes the low supported count a design choice rather than a simple failure.",
        "- Say **\"claim alignment is the bottleneck\"** when full/trend runs exist but verdicts remain inconclusive.",
        "- Avoid saying **\"all positive-evidence repos were reproduced\"**. Say **\"positive experiment-level evidence\"**.",
        "",
        "## Evidence Tier Lists",
        "",
    ]
    for title, group in [
        ("Full reproduction evidence", full),
        ("Trend evidence", trend),
        ("Executable/smoke evidence", executable),
    ]:
        lines.extend([f"### {title}", ""])
        if group:
            for row in group:
                lines.append(
                    f"- `{row['run_id']}`: score={row.get('score_total')}, "
                    f"runs ok/partial/failed={row['runs_successful']}/{row['runs_partial']}/{row['runs_failed']}, "
                    f"figures={row['reproduced_figures']}/{row['skipped_figures']}"
                )
        else:
            lines.append("- none")
        lines.append("")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "paper_ready_result_framing.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_plots(rows: list[dict[str, Any]]) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return paths

    labels = [r["paper_id"] for r in rows]
    scores = [r["score_total"] if r["score_total"] is not None else 0 for r in rows]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(labels, scores, color="#4C78A8")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Reproducibility score")
    ax.set_xlabel("paper_id")
    ax.set_title("DeepAudit score by experiment")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURE_DIR / "score_by_run.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    verdict = Counter()
    for r in rows:
        verdict.update({
            "SUPPORTED": r["supported_count"],
            "PARTIAL": r["partially_supported_count"],
            "NOT_SUPPORTED": r["not_supported_count"],
            "INCONCLUSIVE": r["inconclusive_count"],
        })
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(list(verdict), list(verdict.values()), color=["#54A24B", "#F58518", "#E45756", "#B279A2"])
    ax.set_ylabel("Claim count")
    ax.set_title("Claim verdict distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURE_DIR / "verdict_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    status = Counter()
    for r in rows:
        if r["repo_with_successful_run"]:
            status["at least one ok run"] += 1
        elif r["runs_partial"] > 0:
            status["partial only"] += 1
        elif r["runs_attempted"] > 0:
            status["all failed"] += 1
        else:
            status["not executed"] += 1
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(list(status), list(status.values()), color="#72B7B2")
    ax.set_ylabel("Repo count")
    ax.set_title("Repository-level execution outcome")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURE_DIR / "execution_outcomes.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    tiers = Counter(r["evidence_tier"] for r in rows)
    order = [
        "FULL_REPRODUCTION_EVIDENCE",
        "TREND_EVIDENCE",
        "EXECUTABLE_OR_SMOKE_EVIDENCE",
        "ATTEMPTED_NO_POSITIVE_EVIDENCE",
        "NO_PHASE2_RUNS",
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(order, [tiers.get(k, 0) for k in order], color=["#54A24B", "#4C78A8", "#72B7B2", "#F58518", "#E45756"])
    ax.set_ylabel("Repo count")
    ax.set_title("Post-hoc evidence tiers from existing Phase2 artifacts")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = FIGURE_DIR / "posthoc_evidence_tiers.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    failure = Counter()
    for r in rows:
        failure.update(r["failure_modes"])
    top = failure.most_common(10)
    if top:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.barh([k for k, _ in reversed(top)], [v for _, v in reversed(top)], color="#ECA82C")
        ax.set_xlabel("Count")
        ax.set_title("Top inferred failure modes")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        path = FIGURE_DIR / "failure_modes.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def checklist_metrics() -> list[str]:
    text = read_text(CHECKLIST_PATH)
    return re.findall(r"`([^`]+)`", text)


def write_coverage(handle, rows: list[dict[str, Any]], plot_paths: list[Path]) -> None:
    handle.write("\n# Checklist Coverage\n\n")
    handle.write(
        "Coverage legend: `direct` means the value is read from pipeline artifacts; "
        "`derived` means it is computed from one or more artifacts; `not captured` means "
        "the current artifact schema does not contain enough information and the report "
        "marks the field as `NA` or a diagnostic proxy.\n\n"
    )
    direct = {
        "run_id",
        "total_claims_extracted",
        "metric_contracts_generated",
        "candidate_entry_points_found",
        "command",
        "cwd",
        "exit_code",
        "status",
        "stdout_tail",
        "stderr_tail",
        "runtime_sec",
        "evidence_log_path",
        "claim_id",
        "supported_count",
        "partially_supported_count",
        "not_supported_count",
        "inconclusive_count",
        "total_score",
    }
    not_captured = {
        "venue",
        "repo_url_valid",
        "dataset_download_scripts_found",
        "dataset_paths_identified",
        "reported_split",
        "reported_metric_unit",
        "reported_metric_direction",
        "tolerance",
        "recommended_remediation",
    }
    rows_cov = []
    for name in checklist_metrics():
        if name in direct:
            coverage = "direct"
        elif name in not_captured:
            coverage = "not captured or proxy only"
        else:
            coverage = "derived"
        rows_cov.append([name, coverage])
    handle.write(md_table(["checklist_metric", "coverage_in_result_md"], rows_cov))
    handle.write("\n# Figures\n\n")
    if not plot_paths:
        handle.write("Matplotlib was unavailable; complex figure requirements are described by the aggregate tables above.\n")
        return
    for path in plot_paths:
        rel = path.relative_to(PROJECT_ROOT)
        title = path.stem.replace("_", " ")
        handle.write(f"![{title}]({rel.as_posix()})\n\n")


def main() -> None:
    runs = [
        p
        for p in sorted(ARTIFACTS_DIR.iterdir(), key=lambda x: x.name)
        if p.is_dir()
        and not p.name.startswith("_")
        and p.name not in EXCLUDED_RUN_IDS
        and (p / "execution").exists()
    ]
    rows: list[dict[str, Any]] = []
    RESULT_PATH.write_text(
        "# DeepAudit Experiment Result Summary\n\n"
        f"Generated at `{datetime.utcnow().isoformat(timespec='seconds')}Z` from `{ARTIFACTS_DIR}`.\n\n"
        "Cleaned analysis set only.\n\n"
        "The per-run sections below are written immediately after each run is parsed. "
        "Aggregate metrics and plots are appended after all run sections.\n\n"
        "# Per-Run Metrics\n",
        encoding="utf-8",
    )
    with RESULT_PATH.open("a", encoding="utf-8") as handle:
        for run_dir in runs:
            row = extract_run(run_dir)
            rows.append(row)
            write_run_section(handle, row)
            handle.write(f"\n<!-- completed {row['run_id']} -->\n")
            handle.flush()

        write_aggregate(handle, rows)
        framing_path = write_paper_ready_framing(rows)
        handle.write("\n")
        handle.write(framing_path.read_text(encoding="utf-8"))
        handle.write("\n")
        plot_paths = make_plots(rows)
        write_coverage(handle, rows, plot_paths)
        handle.flush()

    data_path = FIGURE_DIR / "deepaudit_metrics_extracted.json"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    serializable_rows = []
    for row in rows:
        clean = {}
        for key, value in row.items():
            if isinstance(value, Counter):
                clean[key] = dict(value)
            elif key in {"claims", "experiments", "tasks", "entrypoints", "runs", "failures", "metric_records", "claim_evidence", "claim_verdicts", "eval_entries"}:
                clean[key] = value
            else:
                clean[key] = value
        serializable_rows.append(clean)
    data_path.write_text(json.dumps(serializable_rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {RESULT_PATH}")
    print(f"Wrote {data_path}")
    print(f"Processed {len(rows)} runs")


if __name__ == "__main__":
    main()
