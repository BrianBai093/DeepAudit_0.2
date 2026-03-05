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


class Citation(BaseModel):
    marker: str
    context: str


class CitationsDoc(BaseModel):
    citations: list[Citation] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class FingerprintMetadata(BaseModel):
    paper_id: str | None = None
    repository_url: str | None = None
    venue: str | None = None
    year: int | None = None


class FingerprintConfigurations(BaseModel):
    dataset_specs: list[dict[str, Any]] = Field(default_factory=list)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    model_arch: list[str] = Field(default_factory=list)
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
    claim_type: Literal["Empirical", "Methodological", "Comparative", "Unknown"] = "Unknown"
    fact: str
    scope: str
    comparator: str | None = None
    verification_logic: Literal["exact_match", "greater_than_margin", "trend_match", "unknown"] = "unknown"
    tolerance: FingerprintTolerance = Field(default_factory=FingerprintTolerance)
    evidence_anchors: FingerprintEvidenceAnchors = Field(default_factory=FingerprintEvidenceAnchors)
    reason_codes: list[str] = Field(default_factory=list)


class Fingerprint(BaseModel):
    fingerprint_id: str | None = None
    metadata: FingerprintMetadata = Field(default_factory=FingerprintMetadata)
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
    facet: Literal[
        "metric",
        "hyperparameter",
        "architecture",
        "algorithm",
        "dataset_task",
        "preprocess",
        "environment",
        "other",
    ] = "other"
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
    type: Literal["absolute", "relative", "ranking", "other"] = "other"
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
    path: str
    command: str
    confidence: float = Field(ge=0, le=1)
    evidence: str


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


class PaperIngestOutput(BaseModel):
    paper_text: PaperText
    citations: CitationsDoc


class ClaimExtractionOutput(BaseModel):
    fingerprint: Fingerprint
    claims_ir: ClaimsIR


class TaskCompileOutput(BaseModel):
    task_spec: TaskSpec
    metric_contract: MetricContract
