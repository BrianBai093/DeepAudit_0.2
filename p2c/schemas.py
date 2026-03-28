from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


VerdictStatus = Literal[
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "NOT_SUPPORTED",
    "INCONCLUSIVE",
]


class ReasonedModel(BaseModel):
    reason_codes: list[str] = Field(default_factory=list)
    notes: str | None = None


class Section(BaseModel):
    heading: str
    level: int = Field(ge=1, le=6)
    content: str


class PaperText(BaseModel):
    sections: list[Section] = Field(default_factory=list)
    raw_text: str
    figure_descriptions: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class FingerprintConfigurations(BaseModel):
    dataset_specs: list[dict[str, Any]] = Field(default_factory=list)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    evaluation_metrics: list[str] = Field(default_factory=list)


class FingerprintTolerance(BaseModel):
    abs: float | None = None
    rel: float | None = None
    text: str | None = None


class FingerprintEvidenceAnchors(BaseModel):
    text_anchor: str | None = None
    visual_anchor: str | None = None
    visual_data: dict[str, Any] = Field(default_factory=dict)


class FingerprintClaim(BaseModel):
    id: str
    claim_type: Literal["result", "config"] = "config"
    fact: str
    scope: str
    comparator: str | None = None
    verification_logic: Literal["exact_match", "greater_than_margin", "trend_match", "unknown"] = "unknown"
    tolerance: FingerprintTolerance = Field(default_factory=FingerprintTolerance)
    evidence_anchors: FingerprintEvidenceAnchors = Field(default_factory=FingerprintEvidenceAnchors)
    reason_codes: list[str] = Field(default_factory=list)


class Fingerprint(BaseModel):
    fingerprint_id: str | None = None
    configurations: FingerprintConfigurations = Field(default_factory=FingerprintConfigurations)
    claims: list[FingerprintClaim] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    notes: str | None = None


class GuideUnit(BaseModel):
    unit_id: str
    type: Literal["sentence", "table_block"]
    text: str
    origin_indices: list[int] = Field(default_factory=list)


class GuideSentencesDoc(BaseModel):
    sentence_count: int = 0
    sentences: list[str] = Field(default_factory=list)
    unit_count: int = 0
    units: list[GuideUnit] = Field(default_factory=list)
    selected_unit_ids: list[str] = Field(default_factory=list)
    selected_sentence_indices: list[int] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class AtomicCriterion(BaseModel):
    criterion: str
    fact: str
    scope: str
    facet: Literal["metric_result", "execution_param"] = "execution_param"
    source_type: Literal["table_metric", "text_metric", "text_statement"] = "text_statement"
    metric_name: str | None = None
    metric_value: float | None = None
    metric_unit: str | None = None
    entity: str | None = None
    comparator: str | None = None
    dataset_scope: str | None = None
    table_anchor: str | None = None
    input_unit_id: str | None = None
    reason_codes: list[str] = Field(default_factory=list)


class AtomicCriteriaDoc(BaseModel):
    criteria: list[AtomicCriterion] = Field(default_factory=list)
    selected_unit_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class AtomicRejectedItem(BaseModel):
    unit_id: str | None = None
    raw: Any = None
    reason_codes: list[str] = Field(default_factory=list)


class AtomicRejectedDoc(BaseModel):
    rejected: list[AtomicRejectedItem] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class ClaimItem(BaseModel):
    claim_id: str
    type: Literal["result", "config"] = "config"
    predicate: str
    metric: str | None = None
    target: float | None = None
    baseline: float | None = None
    conditions: dict[str, Any] = Field(default_factory=dict)
    aggregation: str | None = None
    evidence_set: list[str] = Field(default_factory=list)
    tolerance_policy: dict[str, float] = Field(
        default_factory=lambda: {"abs_eps": 0.01, "rel_eps": 0.02}
    )
    unverifiable_from_paper: bool = False
    code_verifiable: bool = True
    reason_codes: list[str] = Field(default_factory=list)
    notes: str | None = None


class ClaimsIR(BaseModel):
    claims: list[ClaimItem] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class Entrypoint(BaseModel):
    entrypoint_id: str | None = None
    path: str
    command: str
    cwd: str = "."
    runtime: str = "python"
    dependency_profile_id: str | None = None
    confidence: float = Field(ge=0, le=1)
    evidence: str
    reason_codes: list[str] = Field(default_factory=list)


class MetricObserver(BaseModel):
    name: str
    kind: Literal["stdout_regex", "json_file", "csv_file"] = "stdout_regex"
    pattern: str


class RunConfig(BaseModel):
    seed: int
    timeout_sec: int = 900
    budget_minutes: int = 60


class TaskItem(BaseModel):
    task_id: str
    entrypoint: str
    command: str
    cwd: str = "."
    runtime: str = "python"
    dependency_profile_id: str | None = None
    timeout_class: Literal["short", "medium", "long"] = "medium"
    expected_metrics: list[str] = Field(default_factory=list)
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1, default=0.7)
    evidence: str = ""


class TaskSpec(BaseModel):
    tasks: list[TaskItem] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    # Deprecated compatibility fields; runner uses `tasks` as source of truth.
    entrypoints: list[Entrypoint] = Field(default_factory=list)
    metric_observers: list[MetricObserver] = Field(default_factory=list)
    run_matrix: list[RunConfig] = Field(default_factory=list)
    selection_notes: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class DependencyProfile(BaseModel):
    profile_id: str
    ecosystem: str
    manager: str
    cwd: str = "."
    manifest_paths: list[str] = Field(default_factory=list)
    install_command: str | None = None
    auto_bootstrap_supported: bool = True
    reason_codes: list[str] = Field(default_factory=list)


class RepoAnalysis(BaseModel):
    ecosystems: list[str] = Field(default_factory=list)
    dependency_profiles: list[DependencyProfile] = Field(default_factory=list)
    entrypoint_candidates: list[Entrypoint] = Field(default_factory=list)
    primary_entrypoint_id: str | None = None
    reason_codes: list[str] = Field(default_factory=list)


class MetricParser(BaseModel):
    name: str
    regex: str
    metric_name: str
    transform: str = "float"


class MetricContract(BaseModel):
    required_metrics: list[str] = Field(default_factory=list)
    parsers: list[MetricParser] = Field(default_factory=list)
    normalization: dict[str, Any] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)


class SystemInfo(BaseModel):
    platform: str
    platform_release: str
    python_version: str
    cpu_count: int | None = None
    memory_gb: float | None = None
    reason_codes: list[str] = Field(default_factory=list)


class DataManifestEntry(BaseModel):
    path: str
    exists: bool
    size_bytes: int | None = None
    sandbox_path: str | None = None


class DataManifest(BaseModel):
    entries: list[DataManifestEntry] = Field(default_factory=list)
    unresolved: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class CommandRecord(BaseModel):
    ts: str
    cwd: str
    cmd: str
    rc: int
    stdout_summary: str | None = None
    stderr_summary: str | None = None
    resource_usage: dict[str, Any] | None = None


class RepoState(BaseModel):
    head: str | None = None
    branch: str | None = None
    diff_summary: str | None = None
    submodules: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class CodexRun(BaseModel):
    run_id: str
    command: str
    params: dict[str, Any] = Field(default_factory=dict)
    cwd: str
    exit_code: int
    status: str
    runtime_sec: float | None = None
    stdout_tail: str | None = None
    stderr_tail: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)


class RunManifestDoc(BaseModel):
    runs: list[CodexRun] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class ClaimAlignmentItem(BaseModel):
    claim_id: str
    required_metrics: list[str] = Field(default_factory=list)
    source: list[str] = Field(default_factory=list)
    evaluable: Literal["yes", "no", "partial"] = "partial"
    reason: str | None = None


class ClaimAlignmentDoc(BaseModel):
    claims: list[ClaimAlignmentItem] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class ExecutionSummaryTaskResult(BaseModel):
    task_id: str
    planned_command: str
    final_command: str
    status: Literal["ok", "failed", "skipped"] = "failed"
    notes: str = ""


class ExecutionSummaryDoc(BaseModel):
    project_type: str = "unknown"
    dependency_steps: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    success_basis: Literal["run", "test", "build", "none"] = "none"
    execution_succeeded: bool = False
    attempt_count: int = 0
    task_results: list[ExecutionSummaryTaskResult] = Field(default_factory=list)
    remaining_blockers: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class CodexFailureDoc(BaseModel):
    stage: Literal["precheck", "main", "repair", "postcheck"] = "postcheck"
    last_command: str
    exit_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    codex_exec_log_tail: str = ""
    pip_log_tail: str = ""
    capability_snapshot: dict[str, Any] = Field(default_factory=dict)
    dependency_bootstrap_trace: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class MetricRecord(BaseModel):
    metric_name: str
    value: float | None = None
    unit: str | None = None
    source: str
    claim_id: str | None = None
    parsed: bool = True
    reason_codes: list[str] = Field(default_factory=list)


class MetricsDoc(BaseModel):
    records: list[MetricRecord] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class ClaimEvidence(BaseModel):
    claim_id: str
    matched_records: list[MetricRecord] = Field(default_factory=list)
    missing_reason: str | None = None


class ParsedEvidence(BaseModel):
    claim_evidence: list[ClaimEvidence] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class EvaluabilityEntry(BaseModel):
    claim_id: str
    evaluable: Literal["yes", "no", "partial"] = "partial"
    source: list[str] = Field(default_factory=list)
    reason: str | None = None


class EvaluabilityDoc(BaseModel):
    entries: list[EvaluabilityEntry] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class ClaimVerdict(BaseModel):
    claim_id: str
    status: VerdictStatus
    detail: str
    compared_value: float | None = None
    target_value: float | None = None
    reason_codes: list[str] = Field(default_factory=list)


class VerdictDoc(BaseModel):
    status: VerdictStatus
    claim_verdicts: list[ClaimVerdict] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    summary: str | None = None


class EvaluabilityVerdictRow(BaseModel):
    claim_id: str
    status: Literal["EVALUABLE", "PARTIAL", "NOT_EVALUABLE"]
    detail: str
    reason_codes: list[str] = Field(default_factory=list)


class EvaluabilityVerdictDoc(BaseModel):
    status: Literal["EVALUABLE", "PARTIAL", "NOT_EVALUABLE"]
    claim_rows: list[EvaluabilityVerdictRow] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    summary: str | None = None


class ClaimExtractionOutput(BaseModel):
    fingerprint: Fingerprint
    claims_ir: ClaimsIR


class TaskCompileOutput(BaseModel):
    task_spec: TaskSpec
    metric_contract: MetricContract


# ---------------------------------------------------------------------------
# Phase 2 local execution schemas
# ---------------------------------------------------------------------------


class CondaDependency(BaseModel):
    """A single conda or pip-fallback dependency."""

    package: str
    version_constraint: str | None = None
    channel: str = "defaults"
    pip_fallback: bool = False


class ExecutionStep(BaseModel):
    """One atomic step in the execution plan."""

    step_id: str
    description: str
    command: str
    cwd: str = "."
    timeout_sec: int = 600
    depends_on: list[str] = Field(default_factory=list)
    expected_metrics: list[str] = Field(default_factory=list)
    is_setup: bool = False
    retry_on_failure: bool = True
    fallback_commands: list[str] = Field(default_factory=list)


class CompatibilityIssue(BaseModel):
    issue_type: Literal[
        "python_version", "cuda_version", "package_conflict", "os_dependency", "other"
    ]
    description: str
    resolution: str


class ExpectedResult(BaseModel):
    """Maps a paper claim to the metric the executor should capture."""

    claim_id: str
    metric_name: str
    target_value: float | None = None
    extraction_hint: str | None = None


class ExecutionPlan(BaseModel):
    """Output of the PlannerAgent — drives ToolAgent + CodexExecutor."""

    plan_id: str
    plan_version: int = 1
    python_version: str = "3.10"
    conda_dependencies: list[CondaDependency] = Field(default_factory=list)
    pip_dependencies: list[str] = Field(default_factory=list)
    system_packages: list[str] = Field(default_factory=list)
    pre_install_commands: list[str] = Field(default_factory=list)
    execution_steps: list[ExecutionStep] = Field(default_factory=list)
    expected_results: list[ExpectedResult] = Field(default_factory=list)
    compatibility_issues: list[CompatibilityIssue] = Field(default_factory=list)
    env_name: str
    codex_autonomous_fallback: bool = True
    total_budget_sec: int = 1800
    reason_codes: list[str] = Field(default_factory=list)
    notes: str | None = None


class EnvSetupResult(BaseModel):
    """Output of the ToolAgent — conda/venv environment readiness."""

    env_name: str
    env_path: str = ""
    python_version: str = ""
    install_commands: list[str] = Field(default_factory=list)
    conda_install_log: list[str] = Field(default_factory=list)
    pip_install_log: list[str] = Field(default_factory=list)
    system_install_log: list[str] = Field(default_factory=list)
    validation_passed: bool = False
    failed_packages: list[str] = Field(default_factory=list)
    installed_packages_snapshot: str = ""
    reason_codes: list[str] = Field(default_factory=list)


class StepFailure(BaseModel):
    """Failure record for a single execution step."""

    step_id: str
    command: str
    exit_code: int
    error_type: Literal[
        "dependency", "import", "runtime", "timeout", "data_missing", "permission", "unknown"
    ]
    error_message: str
    stdout_tail: str = ""
    stderr_tail: str = ""
    traceback: str | None = None
    suggested_fix: str | None = None
    # --- v2 taxonomy fields (populated when classify_error_v2 is used) ---
    failure_code: str | None = None  # e.g. "DEP_MISSING_PACKAGE"
    failure_layer: str | None = None  # e.g. "dependency"
    repair_strategy: str | None = None  # e.g. "inline_fix"
    repair_action: str | None = None  # human-readable suggested action
    auto_repair_confidence: float | None = None  # 0.0-1.0


class ExecutionFailure(BaseModel):
    """Aggregated failure for one planner→execute cycle."""

    attempt: int
    plan_version: int = 1
    stage: Literal["planning", "env_setup", "execution", "autonomous"]
    step_failures: list[StepFailure] = Field(default_factory=list)
    overall_error: str = ""
    is_dependency_issue: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class Phase2State(BaseModel):
    """Orchestrator bookkeeping — persisted as phase2_state.json."""

    status: Literal[
        "planning", "env_setup", "executing", "replanning", "autonomous", "success", "failed"
    ] = "planning"
    attempt: int = 0
    max_attempts: int = 3
    total_budget_sec: int = 1800
    elapsed_sec: float = 0.0
    plan: ExecutionPlan | None = None
    env_result: EnvSetupResult | None = None
    failures: list[ExecutionFailure] = Field(default_factory=list)
    final_manifest: RunManifestDoc | None = None
    final_alignment: ClaimAlignmentDoc | None = None
