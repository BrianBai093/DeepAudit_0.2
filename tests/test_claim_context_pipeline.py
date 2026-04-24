from __future__ import annotations

from p2c.schemas import ClaimsIR, Experiment, MetricRecord


def test_run_phase_1_keeps_repo_analysis_before_build_claims_ir():
    from p2c.graph import run_phase_1

    order = []

    class DummyAgent:
        def __init__(self, name: str):
            self.name = name

        def run(self, ctx):
            order.append(self.name)

    agents = {
        "ingest_paper": DummyAgent("ingest_paper"),
        "extract_fingerprint_guide": DummyAgent("extract_fingerprint_guide"),
        "extract_fingerprint_atomic": DummyAgent("extract_fingerprint_atomic"),
        "extract_fingerprint_filter": DummyAgent("extract_fingerprint_filter"),
        "repo_analysis": DummyAgent("repo_analysis"),
        "build_claims_ir": DummyAgent("build_claims_ir"),
        "compile_task_spec": DummyAgent("compile_task_spec"),
    }

    run_phase_1({}, agents)

    assert order == [
        "ingest_paper",
        "extract_fingerprint_guide",
        "extract_fingerprint_atomic",
        "extract_fingerprint_filter",
        "repo_analysis",
        "build_claims_ir",
        "compile_task_spec",
    ]


def test_build_claims_ir_preserves_table_anchor_from_fingerprint():
    from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent

    fingerprint = {
        "claims": [
            {
                "id": "claim_09",
                "claim_type": "result",
                "fact": "accuracy = 0.9685",
                "scope": "from classification report in paper",
                "evidence_anchors": {
                    "text_anchor": "atomic_criteria[8]",
                    "visual_anchor": "Table 1",
                },
                "reason_codes": [],
            }
        ],
        "reason_codes": [],
    }

    agent = BuildClaimsIRAgent.__new__(BuildClaimsIRAgent)
    claims, _ = agent._claims_from_fingerprint(fingerprint)

    assert claims[0].conditions["table_anchor"] == "Table 1"
    assert claims[0].conditions["scope"] == "from classification report in paper"


def test_build_experiment_prompt_is_paper_only():
    from p2c.agents.phase1.build_claims_ir import _build_experiment_user_prompt

    prompt = _build_experiment_user_prompt(
        {
            "claims": [
                {
                    "id": "claim_01",
                    "claim_type": "result",
                    "fact": "precision = 0.72",
                    "scope": "fraud evaluation",
                    "evidence_anchors": {"visual_anchor": "Table 1"},
                    "reason_codes": [],
                }
            ]
        }
    )

    assert "precision = 0.72" in prompt
    assert "Repository Analysis" not in prompt
    assert "src/train_fraud_model.py" not in prompt


def test_build_claims_ir_attaches_experiment_rollups_without_repo_fields():
    from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent

    fingerprint = {
        "claims": [
            {
                "id": "claim_01",
                "claim_type": "result",
                "fact": "accuracy = 0.97",
                "scope": "paper table",
                "evidence_anchors": {"text_anchor": "a1", "visual_anchor": "Table 1"},
                "reason_codes": [],
            },
            {
                "id": "claim_02",
                "claim_type": "config",
                "fact": "epochs = 3",
                "scope": "paper setup",
                "evidence_anchors": {"text_anchor": "a2", "visual_anchor": "Table 1"},
                "reason_codes": [],
            },
        ],
        "reason_codes": [],
    }

    agent = BuildClaimsIRAgent.__new__(BuildClaimsIRAgent)
    base_claims, _ = agent._claims_from_fingerprint(fingerprint)
    agent.safe_chat_json = lambda schema, system, user: (
        {
            "experiments": [
                {
                    "experiment_id": "exp_01",
                    "name": "table 1 fc",
                    "description": "fully connected run",
                    "dataset": "MNIST",
                    "table_anchor": "Table 1",
                    "notes": "paper-side experiment",
                }
            ],
            "claims": [
                {
                    "claim_id": "claim_01",
                    "type": "result",
                    "predicate": "ignored rewrite",
                    "metric": "accuracy",
                    "target": 0.97,
                    "experiment_id": "exp_01",
                    "table_anchor": "Table 1",
                    "scope": "evaluation",
                    "is_primary": True,
                    "reason": "primary claim",
                },
                {
                    "claim_id": "claim_02",
                    "type": "config",
                    "predicate": "ignored rewrite",
                    "metric": None,
                    "target": None,
                    "experiment_id": "exp_01",
                    "table_anchor": "Table 1",
                    "scope": "setup",
                    "is_primary": False,
                    "reason": "config claim",
                },
            ],
        },
        None,
    )

    claims_ir = agent._build_claims_ir_via_llm(fingerprint=fingerprint, base_claims=base_claims)
    exp = claims_ir.experiments[0]

    assert claims_ir.claims[0].predicate == "accuracy = 0.97"
    assert exp.primary_metrics == ["accuracy"]
    assert exp.is_primary is True
    assert not hasattr(exp, "claim_ids")
    assert not hasattr(exp, "repo_coverage")
    assert not hasattr(exp, "repo_entrypoint")


def test_build_claims_ir_drops_empty_experiments_after_rollup():
    from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent

    fingerprint = {
        "claims": [
            {
                "id": "claim_01",
                "claim_type": "result",
                "fact": "accuracy = 0.97",
                "scope": "paper table",
                "evidence_anchors": {"text_anchor": "a1", "visual_anchor": "Table 1"},
                "reason_codes": [],
            }
        ],
        "reason_codes": [],
    }

    agent = BuildClaimsIRAgent.__new__(BuildClaimsIRAgent)
    base_claims, _ = agent._claims_from_fingerprint(fingerprint)
    agent.safe_chat_json = lambda schema, system, user: (
        {
            "experiments": [
                {
                    "experiment_id": "exp_01",
                    "name": "table 1 run",
                    "description": "main run",
                    "dataset": "MNIST",
                    "table_anchor": "Table 1",
                    "notes": "paper-side experiment",
                },
                {
                    "experiment_id": "exp_99",
                    "name": "dangling run",
                    "description": "should be dropped",
                    "dataset": None,
                    "table_anchor": None,
                    "notes": "no linked claims",
                },
            ],
            "claims": [
                {
                    "claim_id": "claim_01",
                    "type": "result",
                    "predicate": "ignored rewrite",
                    "metric": "accuracy",
                    "target": 0.97,
                    "experiment_id": "exp_01",
                    "table_anchor": "Table 1",
                    "scope": "evaluation",
                    "is_primary": True,
                    "reason": "primary claim",
                }
            ],
        },
        None,
    )

    claims_ir = agent._build_claims_ir_via_llm(fingerprint=fingerprint, base_claims=base_claims)

    assert [exp.experiment_id for exp in claims_ir.experiments] == ["exp_01"]
    assert "EXPERIMENTS_WITHOUT_CLAIMS_DROPPED" in claims_ir.reason_codes


def test_align_evidence_prefers_run_scoped_sources():
    from p2c.agents.phase3.align_evidence import AlignEvidenceAgent

    claim = {
        "claim_id": "claim_01",
        "type": "result",
        "metric": "accuracy",
        "target": 0.97,
        "predicate": "accuracy = 0.97",
        "conditions": {"experiment_id": "exp_01"},
    }
    records = [
        MetricRecord(metric_name="accuracy", value=0.97, source="execution/executor_outputs/run_manifest.json:exp_01"),
        MetricRecord(metric_name="accuracy", value=0.81, source="execution/executor_outputs/run_manifest.json:exp_02"),
    ]
    candidate_runs = [
        type("Run", (), {"run_id": "exp_01", "experiment_id": "exp_01", "logs": None})()
    ]

    matched = AlignEvidenceAgent._match_records(
        claim=claim,
        candidate_runs=candidate_runs,
        records=records,
        ambiguous_metric=True,
    )

    assert [row.value for row in matched] == [0.97]


def test_verify_inconclusive_for_missing_experiment_run():
    from p2c.agents.phase3.verify_claims import _fallback_evaluate as evaluate_claim

    claim = {
        "claim_id": "claim_10",
        "type": "result",
        "metric": "accuracy",
        "target": 0.7617,
        "tolerance_policy": {"abs_eps": 0.005, "rel_eps": 0.02},
    }

    verdict = evaluate_claim(
        claim,
        matched_records=[],
        missing_reason="No recorded run for experiment `Table 2 conv run`.",
    )

    assert verdict.status == "INCONCLUSIVE"
    assert "MISSING_RECORDS" in verdict.reason_codes


def test_verify_supported_when_metric_is_within_tolerance():
    from p2c.agents.phase3.verify_claims import _fallback_evaluate as evaluate_claim

    claim = {
        "claim_id": "claim_12",
        "type": "result",
        "metric": "f1",
        "target": 1.0,
        "tolerance_policy": {"abs_eps": 0.02, "rel_eps": 0.02},
    }

    verdict = evaluate_claim(
        claim,
        matched_records=[MetricRecord(metric_name="class_0_f1", value=0.9817, source="run_manifest")],
    )

    assert verdict.status == "SUPPORTED"
    assert "WITHIN_TOLERANCE" in verdict.reason_codes
