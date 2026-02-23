from __future__ import annotations

from typing import Any

from p2c.agents.align_evidence import AlignEvidenceAgent
from p2c.agents.audit_report import AuditReportAgent
from p2c.agents.build_claims_ir import BuildClaimsIRAgent
from p2c.agents.compile_task_spec import CompileTaskSpecAgent
from p2c.agents.execute_and_heal import ExecuteAndHealAgent
from p2c.agents.extract_fingerprint_atomic import ExtractFingerprintAtomicAgent
from p2c.agents.extract_fingerprint_filter import ExtractFingerprintFilterAgent
from p2c.agents.extract_fingerprint_guide import ExtractFingerprintGuideAgent
from p2c.agents.ingest_paper import IngestPaperAgent
from p2c.agents.observe_metrics import ObserveMetricsAgent
from p2c.agents.prepare_sandbox import PrepareSandboxAgent
from p2c.agents.resolve_data import ResolveDataAgent
from p2c.agents.setup_env import SetupEnvAgent
from p2c.agents.verify_claims import VerifyClaimsAgent


def run_phase_1(ctx: dict[str, Any], agents: dict[str, Any]) -> None:
    agents["ingest_paper"].run(ctx)
    agents["extract_fingerprint_guide"].run(ctx)
    agents["extract_fingerprint_atomic"].run(ctx)
    agents["extract_fingerprint_filter"].run(ctx)
    agents["build_claims_ir"].run(ctx)
    agents["compile_task_spec"].run(ctx)


def run_phase_2(ctx: dict[str, Any], agents: dict[str, Any]) -> None:
    agents["prepare_sandbox"].run(ctx)
    agents["setup_env"].run(ctx)
    agents["resolve_data"].run(ctx)
    agents["execute_and_heal"].run(ctx)


def run_phase_3(ctx: dict[str, Any], agents: dict[str, Any]) -> None:
    agents["observe_metrics"].run(ctx)
    agents["align_evidence"].run(ctx)
    agents["verify_claims"].run(ctx)
    agents["audit_report"].run(ctx)


def build_agents(llm, artifacts) -> dict[str, Any]:
    return {
        "ingest_paper": IngestPaperAgent(llm=llm, artifacts=artifacts, step_index=1, step_total=14),
        "extract_fingerprint_guide": ExtractFingerprintGuideAgent(
            llm=llm, artifacts=artifacts, step_index=2, step_total=14
        ),
        "extract_fingerprint_atomic": ExtractFingerprintAtomicAgent(
            llm=llm, artifacts=artifacts, step_index=3, step_total=14
        ),
        "extract_fingerprint_filter": ExtractFingerprintFilterAgent(
            llm=llm, artifacts=artifacts, step_index=4, step_total=14
        ),
        "build_claims_ir": BuildClaimsIRAgent(llm=llm, artifacts=artifacts, step_index=5, step_total=14),
        "compile_task_spec": CompileTaskSpecAgent(llm=llm, artifacts=artifacts, step_index=6, step_total=14),
        "prepare_sandbox": PrepareSandboxAgent(llm=llm, artifacts=artifacts, step_index=7, step_total=14),
        "setup_env": SetupEnvAgent(llm=llm, artifacts=artifacts, step_index=8, step_total=14),
        "resolve_data": ResolveDataAgent(llm=llm, artifacts=artifacts, step_index=9, step_total=14),
        "execute_and_heal": ExecuteAndHealAgent(llm=llm, artifacts=artifacts, step_index=10, step_total=14),
        "observe_metrics": ObserveMetricsAgent(llm=llm, artifacts=artifacts, step_index=11, step_total=14),
        "align_evidence": AlignEvidenceAgent(llm=llm, artifacts=artifacts, step_index=12, step_total=14),
        "verify_claims": VerifyClaimsAgent(llm=llm, artifacts=artifacts, step_index=13, step_total=14),
        "audit_report": AuditReportAgent(llm=llm, artifacts=artifacts, step_index=14, step_total=14),
    }


def build_langgraph_if_available():
    try:
        from langgraph.graph import END, START, StateGraph
    except Exception:  # noqa: BLE001
        return None

    class State(dict):
        pass

    graph = StateGraph(State)
    graph.add_node("phase1", lambda state: state)
    graph.add_node("phase2", lambda state: state)
    graph.add_node("phase3", lambda state: state)
    graph.add_edge(START, "phase1")
    graph.add_edge("phase1", "phase2")
    graph.add_edge("phase2", "phase3")
    graph.add_edge("phase3", END)
    return graph.compile()
