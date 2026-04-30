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
    source_type: Literal[
        "table_metric",
        "text_metric",
        "text_statement",
        "visual_metric",
        "llm_table_metric",
        "llm_table_param",
        "visual_table_metric",
        "visual_table_param",
    ] = "text_statement"
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


class Experiment(BaseModel):
    """A distinct experiment described in the paper."""

    experiment_id: str
    name: str
    description: str = ""
    dataset: str | None = None
    table_anchor: str | None = None
    primary_metrics: list[str] = Field(default_factory=list)
    is_primary: bool = False
    notes: str | None = None


class ClaimsIR(BaseModel):
    experiments: list[Experiment] = Field(default_factory=list)
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
    path_resolution_mode: str | None = None
    derived_from_wrapper: str | None = None
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
    path_resolution_mode: str | None = None
    derived_from_wrapper: str | None = None
    reason_codes: list[str] = Field(default_factory=list)


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


class ExecutionLogRefs(BaseModel):
    stdout: str | None = None
    stderr: str | None = None
    narrative: str | None = None
    activity: str | None = None


ExecutionRunStatus = Literal["ok", "partial", "failed", "skipped"]
ExecutionFidelity = Literal["artifact", "smoke", "trend", "full"]
ExecutionOutcome = Literal["EXECUTABLE", "TREND_SUPPORTED", "FULLY_REPRODUCED"]
ExecutionEvidenceSource = Literal["fresh_run", "checkpoint_eval", "existing_logs", "existing_results", "mixed"]
ExecutionStopReason = Literal[
    "checkpoint_eval",
    "existing_artifact",
    "budget_bound",
    "early_stop_evidence",
    "full_run_complete",
    "repo_missing_path",
    "runtime_failure",
    "guardrail_blocked",
    "skipped_nonessential",
]


class ExecutionRun(BaseModel):
    run_id: str
    experiment_id: str | None = None
    experiment_name: str | None = None
    dataset: str | None = None
    command: str
    commands_attempted: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    cwd: str
    exit_code: int
    status: ExecutionRunStatus
    fidelity: ExecutionFidelity | None = None
    execution_outcome: ExecutionOutcome | None = None
    evidence_source: ExecutionEvidenceSource | None = None
    override_args: list[str] = Field(default_factory=list)
    observed_signals: list[str] = Field(default_factory=list)
    stop_reason: ExecutionStopReason | None = None
    notes: str | None = None
    runtime_sec: float | None = None
    stdout_tail: str | None = None
    stderr_tail: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    logs: ExecutionLogRefs = Field(default_factory=ExecutionLogRefs)
    reason_codes: list[str] = Field(default_factory=list)


class RunManifestDoc(BaseModel):
    runs: list[ExecutionRun] = Field(default_factory=list)
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


class ExecutorFailureDoc(BaseModel):
    stage: Literal["precheck", "main", "repair", "postcheck"] = "postcheck"
    last_command: str
    exit_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    executor_exec_log_tail: str = ""
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
    run_id: str | None = None
    experiment_id: str | None = None
    fidelity: ExecutionFidelity | None = None
    execution_outcome: ExecutionOutcome | None = None
    evidence_source: ExecutionEvidenceSource | None = None
    parsed: bool = True
    reason_codes: list[str] = Field(default_factory=list)


class ExecutorResultRun(BaseModel):
    experiment_id: str
    experiment_name: str | None = None
    dataset: str | None = None
    command: str = ""
    commands_attempted: list[str] = Field(default_factory=list)
    cwd: str = "."
    exit_code: int = 1
    status: ExecutionRunStatus = "failed"
    fidelity: ExecutionFidelity | None = None
    execution_outcome: ExecutionOutcome | None = None
    evidence_source: ExecutionEvidenceSource | None = None
    override_args: list[str] = Field(default_factory=list)
    observed_signals: list[str] = Field(default_factory=list)
    stop_reason: ExecutionStopReason | None = None
    notes: str | None = None
    runtime_sec: float | None = None
    artifacts: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    logs: ExecutionLogRefs = Field(default_factory=ExecutionLogRefs)
    reason_codes: list[str] = Field(default_factory=list)


class ExecutorResultsDoc(BaseModel):
    runs: list[ExecutorResultRun] = Field(default_factory=list)


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
# Visual extraction schemas (Phase 1 PDF → figures/tables)
# ---------------------------------------------------------------------------


class VisualElement(BaseModel):
    """A figure or table extracted from the paper PDF via vision API."""

    element_id: str                                # e.g. "fig_1", "table_3"
    element_type: Literal["figure", "table"]
    page: int
    caption: str = ""
    chart_type: str | None = None                  # "bar", "line", "scatter", "heatmap", "table", "diagram"
    axis_labels: dict[str, str] = Field(default_factory=dict)
    legend_entries: list[str] = Field(default_factory=list)
    data_series: list[dict[str, Any]] = Field(default_factory=list)
    visual_anchor: str = ""                        # "Figure 1", "Table 3"
    bbox: dict[str, float] = Field(default_factory=dict)  # normalized x0/y0/x1/y1 page coordinates
    raw_page_image: str | None = None               # run-root-relative page image path
    crop_path: str | None = None                    # run-root-relative cropped element image path
    x_axis_range: list[float] | None = None
    y_axis_range: list[float] | None = None
    series_semantics: list[dict[str, Any]] = Field(default_factory=list)
    model_names: list[str] = Field(default_factory=list)
    sampling_strategy: str | None = None
    numeric_confidence: float | None = None
    associated_claim_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class VisualElementsDoc(BaseModel):
    """All visual elements extracted from the paper PDF."""

    elements: list[VisualElement] = Field(default_factory=list)
    page_count: int = 0
    reason_codes: list[str] = Field(default_factory=list)


class VisualTarget(BaseModel):
    """Object-level reconstruction target derived from a paper figure or table."""

    element_id: str
    visual_anchor: str = ""
    element_type: Literal["figure", "table"]
    chart_type: str | None = None
    caption: str = ""
    page: int | None = None
    reference_image_path: str | None = None
    axis_labels: dict[str, str] = Field(default_factory=dict)
    legend_entries: list[str] = Field(default_factory=list)
    series_names: list[str] = Field(default_factory=list)
    metric_names: list[str] = Field(default_factory=list)
    model_names: list[str] = Field(default_factory=list)
    sampling_strategy: str | None = None
    semantic_summary: str = ""
    reconstruction_instructions: list[str] = Field(default_factory=list)
    associated_claim_ids: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class VisualTargetsDoc(BaseModel):
    visual_targets: list[VisualTarget] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class ReproducedFigure(BaseModel):
    """A figure reproduced from execution results in Phase 3."""

    element_id: str
    matplotlib_code: str = ""
    image_path: str = ""                           # relative path to saved PNG
    comparison_notes: str = ""
    visual_anchor: str = ""
    reference_image_path: str | None = None
    reproduced_image_path: str | None = None
    evidence_sources: list[str] = Field(default_factory=list)
    reproduction_status: Literal["REPRODUCED", "SKIPPED", "FAILED"] = "REPRODUCED"
    plot_spec: dict[str, Any] = Field(default_factory=dict)
    code_path: str | None = None
    llm_decision_summary: str = ""
    match_level: Literal["EXACT", "PARTIAL", "RELATED", "NO_EVIDENCE"] = "EXACT"
    matched_scope: dict[str, Any] = Field(default_factory=dict)
    coverage_note: str = ""
    reason_codes: list[str] = Field(default_factory=list)


class SkippedReproducedTarget(BaseModel):
    """A paper visual target that was intentionally not reproduced."""

    element_id: str
    visual_anchor: str = ""
    skip_reason: str = ""
    evidence_sources: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class ReproducedFiguresDoc(BaseModel):
    figures: list[ReproducedFigure] = Field(default_factory=list)
    skipped_targets: list[SkippedReproducedTarget] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class VisualRepoAlignmentItem(BaseModel):
    """Strict mapping from a paper visual element to a repo artifact or NO_MATCH."""

    element_id: str
    status: Literal["MATCH", "NO_MATCH"] = "NO_MATCH"
    repo_artifact_path: str | None = None
    artifact_type: str | None = None
    confidence: float = 0.0
    matched_model_names: list[str] = Field(default_factory=list)
    matched_sampling_strategy: str | None = None
    matched_metric_names: list[str] = Field(default_factory=list)
    mismatch_reasons: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class VisualRepoAlignmentDoc(BaseModel):
    alignments: list[VisualRepoAlignmentItem] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Reproducibility scoring schemas (Phase 3)
# ---------------------------------------------------------------------------


class DimensionScore(BaseModel):
    """Single dimension score (0-100)."""

    dimension: Literal["environment", "data_availability", "execution_success", "claim_match"]
    score: int = Field(ge=0, le=100, default=0)
    weight: float = 0.25
    weighted_score: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class GapDiagnosis(BaseModel):
    """Classified reproduction failure."""

    gap_id: str
    category: Literal[
        "DATA_MISSING",
        "PREPROCESS_UNSPECIFIED",
        "CHECKPOINT_MISSING",
        "ENVIRONMENT_UNDERDEFINED",
        "ENTRYPOINT_UNCLEAR",
        "NONDETERMINISM",
        "COMPUTE_INFEASIBLE",
        "RESULT_MISMATCH",
    ]
    claim_ids: list[str] = Field(default_factory=list)
    description: str = ""
    severity: Literal["critical", "major", "minor"] = "major"
    reason_codes: list[str] = Field(default_factory=list)


class ReproducibilityScore(BaseModel):
    """Complete reproducibility assessment with 0-100 weighted score."""

    total_score: float = 0.0
    raw_total_score: float | None = None
    calibration_notes: list[str] = Field(default_factory=list)
    dimensions: list[DimensionScore] = Field(default_factory=list)
    ecr: bool = False                              # Executable-Claim Reproducible
    ecr_reason: str = ""
    gaps: list[GapDiagnosis] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


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
    timeout_sec: int = 7200  # default 2h; long ML training steps may override higher
    depends_on: list[str] = Field(default_factory=list)
    expected_metrics: list[str] = Field(default_factory=list)
    is_setup: bool = False
    retry_on_failure: bool = True
    fallback_commands: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    produced_artifacts: list[str] = Field(default_factory=list)
    path_resolution_mode: str | None = None
    derived_from_wrapper: str | None = None


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
    """Deprecated plan schema retained only for artifact compatibility."""

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
    executor_autonomous_fallback: bool = True
    total_budget_sec: int = 10800  # default 3h; override via --budget_minutes
    reason_codes: list[str] = Field(default_factory=list)
    notes: str | None = None


class ExecutorEnvSpec(BaseModel):
    """Deterministic environment spec derived only from repository artifacts."""

    env_name: str
    python_version: str = "3.10"
    conda_dependencies: list[CondaDependency] = Field(default_factory=list)
    pip_dependencies: list[str] = Field(default_factory=list)
    system_packages: list[str] = Field(default_factory=list)
    pre_install_commands: list[str] = Field(default_factory=list)
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
        "env_setup", "executing", "repairing", "success", "failed"
    ] = "env_setup"
    attempt: int = 0
    max_attempts: int = 3
    total_budget_sec: int = 10800  # default 3h
    elapsed_sec: float = 0.0
    env_spec: ExecutorEnvSpec | None = None
    env_result: EnvSetupResult | None = None
    failures: list[ExecutionFailure] = Field(default_factory=list)
    final_manifest: RunManifestDoc | None = None
