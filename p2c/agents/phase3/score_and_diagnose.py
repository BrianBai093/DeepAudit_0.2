"""ScoreAndDiagnoseAgent — computes 0-100 reproducibility score and classifies gaps."""

from __future__ import annotations

import re
from typing import Any

from p2c.agents.base import BaseAgent
from p2c.schemas import (
    DimensionScore,
    GapDiagnosis,
    ReproducibilityScore,
)

# ---------------------------------------------------------------------------
# Gap taxonomy patterns
# ---------------------------------------------------------------------------

_DATA_MISSING_PATTERNS = [
    r"FileNotFoundError",
    r"No such file or directory",
    r"data.*not found",
    r"dataset.*missing",
    r"cannot find.*data",
    r"download.*fail",
]

_PREPROCESS_PATTERNS = [
    r"preprocess",
    r"transform.*fail",
    r"feature.*extraction.*error",
    r"normalization.*error",
    r"tokeniz.*error",
]

_CHECKPOINT_PATTERNS = [
    r"\.ckpt",
    r"\.pth",
    r"\.pt\b",
    r"\.h5\b",
    r"pretrained",
    r"checkpoint.*not found",
    r"weights.*missing",
    r"model.*load.*fail",
]

_ENV_PATTERNS = [
    r"ModuleNotFoundError",
    r"ImportError",
    r"No module named",
    r"version.*mismatch",
    r"incompatible",
]

_ENTRYPOINT_PATTERNS = [
    r"entry.*point.*not found",
    r"No such file.*\.py",
    r"command not found",
    r"FileNotFoundError.*\.py",
]

_COMPUTE_PATTERNS = [
    r"CUDA.*out of memory",
    r"OOM",
    r"MemoryError",
    r"RuntimeError.*CUDA",
    r"timeout",
    r"Killed",
    r"signal 9",
]


class ScoreAndDiagnoseAgent(BaseAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(name="score_and_diagnose", *args, **kwargs)

    def execute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        # Load all required artifacts
        verdict = self._safe_read("results/verdict.json")
        manifest = self._safe_read("execution/codex_outputs/run_manifest.json")
        env_result = self._safe_read("execution/env_setup_result.json")
        plan = self._safe_read("execution/execution_plan.json")
        repo_analysis = self._safe_read("task/repo_analysis.json")
        claims_ir = self._safe_read("fingerprint/claims_ir.json")
        failures = self._safe_read("execution/execution_failures.json")

        # Compute dimension scores
        env_score = self._score_environment(env_result, plan, repo_analysis)
        data_score = self._score_data_availability(manifest, claims_ir, repo_analysis)
        exec_score = self._score_execution_success(manifest)
        claim_score = self._score_claim_match(verdict)

        dimensions = [env_score, data_score, exec_score, claim_score]

        total = sum(d.weighted_score for d in dimensions)

        # Classify gaps
        gaps = self._classify_gaps(
            verdict=verdict,
            manifest=manifest,
            env_result=env_result,
            failures=failures,
        )

        # Determine ECR
        ecr, ecr_reason = self._compute_ecr(verdict, manifest, exec_score.score)

        score = ReproducibilityScore(
            total_score=round(total, 1),
            dimensions=dimensions,
            ecr=ecr,
            ecr_reason=ecr_reason,
            gaps=gaps,
            reason_codes=["SCORE_COMPUTED"],
        )

        self.artifacts.write_json("results/reproducibility_score.json", score.model_dump())
        self.log("DONE", f"Score: {score.total_score}/100, ECR: {score.ecr}, Gaps: {len(gaps)}")
        return {"score": score.model_dump()}

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------

    def _safe_read(self, path: str) -> Any:
        try:
            return self.artifacts.read_json(path)
        except Exception:  # noqa: BLE001
            return {}

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    def _score_environment(
        self, env_result: dict, plan: dict, repo_analysis: dict,
    ) -> DimensionScore:
        """Environment dimension (25%): deterministic env build quality."""
        score = 100
        evidence: list[str] = []
        reason_codes: list[str] = []

        # Check validation
        if not env_result.get("validation_passed", False):
            failed_pkgs = env_result.get("failed_packages", [])
            if failed_pkgs:
                penalty = min(len(failed_pkgs) * 10, 50)
                score -= penalty
                evidence.append(f"{len(failed_pkgs)} failed packages: {', '.join(failed_pkgs[:5])}")
                reason_codes.append("ENV_FAILED_PACKAGES")
            else:
                score -= 10
                evidence.append("Validation failed but no specific package failures")
                reason_codes.append("ENV_VALIDATION_WARNING")
        else:
            evidence.append("Environment validation passed")

        # Check for reproducibility-enabling files
        entrypoints = repo_analysis.get("entrypoint_candidates", [])
        profiles = repo_analysis.get("dependency_profiles", [])

        has_lockfile = False
        has_dockerfile = False
        for p in profiles:
            manifests = p.get("manifest_paths", [])
            for m in manifests:
                ml = m.lower()
                if "lock" in ml or "environment.yml" in ml or "environment.yaml" in ml:
                    has_lockfile = True
                if "dockerfile" in ml:
                    has_dockerfile = True

        if has_dockerfile:
            evidence.append("Dockerfile present")
        elif has_lockfile:
            evidence.append("Lock file / environment.yml present")
        else:
            score -= 15
            evidence.append("No lock file or Dockerfile for deterministic env")
            reason_codes.append("ENV_NO_LOCKFILE")

        if not profiles:
            score -= 20
            evidence.append("No dependency manifest detected")
            reason_codes.append("ENV_NO_MANIFEST")

        score = max(0, min(100, score))
        return DimensionScore(
            dimension="environment",
            score=score,
            weight=0.25,
            weighted_score=round(score * 0.25, 1),
            evidence=evidence,
            reason_codes=reason_codes,
        )

    def _score_data_availability(
        self, manifest: dict, claims_ir: dict, repo_analysis: dict,
    ) -> DimensionScore:
        """Data availability dimension (25%): datasets, checkpoints, preprocessing."""
        score = 100
        evidence: list[str] = []
        reason_codes: list[str] = []

        runs = manifest.get("runs", [])

        # Check data-related steps
        data_steps = [
            r for r in runs
            if any(kw in r.get("run_id", "").lower() for kw in ("data", "download", "prepar", "verify"))
        ]
        failed_data = [r for r in data_steps if r.get("status") == "failed"]
        if failed_data:
            score -= min(len(failed_data) * 20, 60)
            evidence.append(f"{len(failed_data)}/{len(data_steps)} data steps failed")
            reason_codes.append("DATA_STEPS_FAILED")
        elif data_steps:
            evidence.append(f"All {len(data_steps)} data steps succeeded")

        # Check experiments for data coverage
        experiments = claims_ir.get("experiments", [])
        not_found = [e for e in experiments if e.get("repo_coverage") == "not_found"]
        if not_found:
            penalty = min(len(not_found) * 15, 40)
            score -= penalty
            evidence.append(f"{len(not_found)} experiments not found in repo")
            reason_codes.append("EXPERIMENTS_NOT_FOUND")

        # Check for stderr data-related errors across all runs
        for run in runs:
            stderr = run.get("stderr_tail", "") or ""
            if re.search(r"(?i)data.*not found|FileNotFoundError.*data", stderr):
                score -= 15
                evidence.append(f"Data error in {run.get('run_id')}")
                reason_codes.append("DATA_ERROR_IN_STDERR")
                break

        score = max(0, min(100, score))
        return DimensionScore(
            dimension="data_availability",
            score=score,
            weight=0.25,
            weighted_score=round(score * 0.25, 1),
            evidence=evidence,
            reason_codes=reason_codes,
        )

    def _score_execution_success(self, manifest: dict) -> DimensionScore:
        """Execution success dimension (20%): code runs end-to-end."""
        score = 100
        evidence: list[str] = []
        reason_codes: list[str] = []

        runs = manifest.get("runs", [])
        if not runs:
            return DimensionScore(
                dimension="execution_success",
                score=0,
                weight=0.20,
                weighted_score=0.0,
                evidence=["No execution runs found"],
                reason_codes=["NO_RUNS"],
            )

        # Exclude setup steps for scoring
        non_setup = [r for r in runs if not r.get("run_id", "").startswith("step_00")]
        if not non_setup:
            non_setup = runs

        ok_count = sum(1 for r in non_setup if r.get("status") == "ok")
        partial_count = sum(1 for r in non_setup if r.get("status") == "partial")
        failed_count = sum(1 for r in non_setup if r.get("status") == "failed")
        total = len(non_setup)

        if total > 0:
            per_step = 100.0 / total
            score = round(ok_count * per_step + partial_count * per_step * 0.5)

        evidence.append(f"{ok_count} ok, {partial_count} partial, {failed_count} failed out of {total} steps")

        if failed_count > 0:
            reason_codes.append("STEPS_FAILED")
        if partial_count > 0:
            reason_codes.append("STEPS_PARTIAL")

        score = max(0, min(100, score))
        return DimensionScore(
            dimension="execution_success",
            score=score,
            weight=0.20,
            weighted_score=round(score * 0.20, 1),
            evidence=evidence,
            reason_codes=reason_codes,
        )

    def _score_claim_match(self, verdict: dict) -> DimensionScore:
        """Claim match dimension (30%): reproduced results match paper."""
        evidence: list[str] = []
        reason_codes: list[str] = []

        claim_verdicts = verdict.get("claim_verdicts", [])
        # Only score result claims (not config)
        result_verdicts = [
            cv for cv in claim_verdicts
            if cv.get("status") != "INCONCLUSIVE" or cv.get("compared_value") is not None
        ]

        if not result_verdicts:
            # Use all verdicts if filtering is too aggressive
            result_verdicts = claim_verdicts

        if not result_verdicts:
            return DimensionScore(
                dimension="claim_match",
                score=0,
                weight=0.30,
                weighted_score=0.0,
                evidence=["No claim verdicts available"],
                reason_codes=["NO_VERDICTS"],
            )

        n = len(result_verdicts)
        supported = sum(1 for cv in result_verdicts if cv.get("status") == "SUPPORTED")
        partial = sum(1 for cv in result_verdicts if cv.get("status") == "PARTIALLY_SUPPORTED")
        not_supported = sum(1 for cv in result_verdicts if cv.get("status") == "NOT_SUPPORTED")
        inconclusive = sum(1 for cv in result_verdicts if cv.get("status") == "INCONCLUSIVE")

        # Score: each SUPPORTED = full points, PARTIAL = 50%, INCONCLUSIVE = 25%, NOT_SUPPORTED = 0
        if n > 0:
            per_claim = 100.0 / n
            score = round(
                supported * per_claim
                + partial * per_claim * 0.5
                + inconclusive * per_claim * 0.25
            )
        else:
            score = 0

        evidence.append(
            f"{supported} supported, {partial} partial, {not_supported} not supported, "
            f"{inconclusive} inconclusive out of {n} claims"
        )

        if not_supported > 0:
            reason_codes.append("CLAIMS_NOT_SUPPORTED")
        if inconclusive > 0:
            reason_codes.append("CLAIMS_INCONCLUSIVE")

        score = max(0, min(100, score))
        return DimensionScore(
            dimension="claim_match",
            score=score,
            weight=0.30,
            weighted_score=round(score * 0.30, 1),
            evidence=evidence,
            reason_codes=reason_codes,
        )

    # ------------------------------------------------------------------
    # ECR computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ecr(verdict: dict, manifest: dict, exec_score: int) -> tuple[bool, str]:
        """Compute binary ECR (Executable-Claim Reproducible) label."""
        claim_verdicts = verdict.get("claim_verdicts", [])
        if not claim_verdicts:
            return False, "No claim verdicts available"

        # Check all primary result claims are SUPPORTED
        non_config = [
            cv for cv in claim_verdicts
            if cv.get("status") != "INCONCLUSIVE" or cv.get("compared_value") is not None
        ]
        if not non_config:
            non_config = claim_verdicts

        all_supported = all(cv.get("status") == "SUPPORTED" for cv in non_config)
        if not all_supported:
            unsupported = [cv["claim_id"] for cv in non_config if cv.get("status") != "SUPPORTED"]
            return False, f"Claims not supported: {', '.join(unsupported[:5])}"

        # Check execution success
        if exec_score < 80:
            return False, f"Execution score {exec_score}/100 < 80 threshold"

        # Check no manual fixes needed
        runs = manifest.get("runs", [])
        manual_codes = {"MANUAL_FIX", "MANUAL_INTERVENTION"}
        for run in runs:
            if manual_codes & set(run.get("reason_codes", [])):
                return False, "Manual fixes were needed during execution"

        return True, "All primary claims supported, execution score >= 80, no manual fixes"

    # ------------------------------------------------------------------
    # Gap taxonomy classifier
    # ------------------------------------------------------------------

    def _classify_gaps(
        self,
        verdict: dict,
        manifest: dict,
        env_result: dict,
        failures: Any,
    ) -> list[GapDiagnosis]:
        """Classify reproduction failures into gap categories."""
        gaps: list[GapDiagnosis] = []
        gap_counter = 0

        # Collect all error text from manifest runs + failures
        error_texts: list[tuple[str, str]] = []  # (run_id, error_text)
        for run in manifest.get("runs", []):
            if run.get("status") in ("failed", "partial"):
                stderr = run.get("stderr_tail", "") or ""
                stdout = run.get("stdout_tail", "") or ""
                error_texts.append((run.get("run_id", "unknown"), f"{stderr}\n{stdout}"))

        for fail in _failure_entries(failures):
            for sf in fail.get("step_failures", []):
                error_texts.append((
                    sf.get("step_id", "unknown"),
                    f"{sf.get('stderr_tail', '')}\n{sf.get('error_message', '')}",
                ))

        # Classify each error
        for run_id, text in error_texts:
            category = self._match_gap_category(text)
            if category:
                gap_counter += 1
                gaps.append(GapDiagnosis(
                    gap_id=f"gap_{gap_counter:02d}",
                    category=category,
                    description=f"Detected in {run_id}: {text[:200].strip()}",
                    severity=_severity_for_category(category),
                    reason_codes=[f"PATTERN_MATCH_{category}"],
                ))

        # Check env-specific gaps
        if not env_result.get("validation_passed", False):
            failed_pkgs = env_result.get("failed_packages", [])
            if failed_pkgs:
                gap_counter += 1
                gaps.append(GapDiagnosis(
                    gap_id=f"gap_{gap_counter:02d}",
                    category="ENVIRONMENT_UNDERDEFINED",
                    description=f"Environment validation failed with packages: {', '.join(failed_pkgs[:10])}",
                    severity="major",
                    reason_codes=["ENV_VALIDATION_FAILED"],
                ))

        # Check verdict for RESULT_MISMATCH gaps
        for cv in verdict.get("claim_verdicts", []):
            if cv.get("status") == "NOT_SUPPORTED":
                gap_counter += 1
                target = cv.get("target_value")
                reproduced = cv.get("compared_value")
                desc = f"Claim {cv['claim_id']}: paper={target}, reproduced={reproduced}"
                gaps.append(GapDiagnosis(
                    gap_id=f"gap_{gap_counter:02d}",
                    category="RESULT_MISMATCH",
                    claim_ids=[cv["claim_id"]],
                    description=desc,
                    severity="major",
                    reason_codes=["VERDICT_NOT_SUPPORTED"],
                ))

        return gaps

    @staticmethod
    def _match_gap_category(text: str) -> str | None:
        """Match error text against gap category patterns."""
        for pattern_list, category in [
            (_CHECKPOINT_PATTERNS, "CHECKPOINT_MISSING"),
            (_DATA_MISSING_PATTERNS, "DATA_MISSING"),
            (_PREPROCESS_PATTERNS, "PREPROCESS_UNSPECIFIED"),
            (_COMPUTE_PATTERNS, "COMPUTE_INFEASIBLE"),
            (_ENTRYPOINT_PATTERNS, "ENTRYPOINT_UNCLEAR"),
            (_ENV_PATTERNS, "ENVIRONMENT_UNDERDEFINED"),
        ]:
            for pattern in pattern_list:
                if re.search(pattern, text, re.IGNORECASE):
                    return category
        return None


def _severity_for_category(category: str) -> str:
    """Map gap category to default severity."""
    critical = {"DATA_MISSING", "CHECKPOINT_MISSING", "COMPUTE_INFEASIBLE"}
    minor = {"NONDETERMINISM"}
    if category in critical:
        return "critical"
    if category in minor:
        return "minor"
    return "major"


def _failure_entries(failures: Any) -> list[dict]:
    """Normalize execution_failures.json across legacy/current shapes."""
    if isinstance(failures, list):
        return [f for f in failures if isinstance(f, dict)]
    if isinstance(failures, dict):
        rows = failures.get("failures", [])
        if isinstance(rows, list):
            return [f for f in rows if isinstance(f, dict)]
        if failures.get("step_failures"):
            return [failures]
    return []
