from __future__ import annotations

from typing import Any
import os

from p2c.agents.phase3.align_evidence import AlignEvidenceAgent
from p2c.agents.phase3.audit_report import AuditReportAgent
from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent
from p2c.agents.phase2.collect_codex_outputs import CollectCodexOutputsAgent
from p2c.agents.phase2.collect_codex_outputs_newstyle import CollectCodexOutputsNewstyleAgent
from p2c.agents.phase1.compile_task_spec import CompileTaskSpecAgent
from p2c.agents.phase1.extract_fingerprint_atomic import ExtractFingerprintAtomicAgent
from p2c.agents.phase1.extract_fingerprint_filter import ExtractFingerprintFilterAgent
from p2c.agents.phase1.extract_fingerprint_guide import ExtractFingerprintGuideAgent
from p2c.agents.phase1.ingest_paper import IngestPaperAgent
from p2c.agents.phase1.repo_analysis import RepoAnalysisAgent
from p2c.agents.phase3.observe_metrics import ObserveMetricsAgent
from p2c.agents.phase2.prepare_sandbox import PrepareSandboxAgent
from p2c.agents.phase2.prepare_sandbox_newstyle import PrepareSandboxNewstyleAgent
from p2c.agents.phase2.run_codex_exec import RunCodexExecAgent
from p2c.agents.phase2.run_codex_exec_newstyle import RunCodexExecNewstyleAgent
from p2c.agents.phase3.verify_claims import VerifyClaimsAgent


def _phase2_style(ctx: dict[str, Any]) -> str:
    return str(ctx.get("phase2_style") or os.getenv("P2C_PHASE2_STYLE") or "legacy").strip().lower()


def run_phase_1(ctx: dict[str, Any], agents: dict[str, Any]) -> None:
    agents["ingest_paper"].run(ctx)
    agents["extract_fingerprint_guide"].run(ctx)
    agents["extract_fingerprint_atomic"].run(ctx)
    agents["extract_fingerprint_filter"].run(ctx)
    agents["build_claims_ir"].run(ctx)
    agents["repo_analysis"].run(ctx)
    agents["compile_task_spec"].run(ctx)


def run_phase_2(ctx: dict[str, Any], agents: dict[str, Any]) -> None:
    if _phase2_style(ctx) == "new":
        agents["prepare_sandbox_newstyle"].run(ctx)
        agents["run_codex_exec_newstyle"].run(ctx)
        agents["collect_codex_outputs_newstyle"].run(ctx)
        return
    agents["prepare_sandbox"].run(ctx)
    agents["run_codex_exec"].run(ctx)
    agents["collect_codex_outputs"].run(ctx)


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
        "repo_analysis": RepoAnalysisAgent(llm=llm, artifacts=artifacts, step_index=6, step_total=15),
        "compile_task_spec": CompileTaskSpecAgent(llm=llm, artifacts=artifacts, step_index=7, step_total=15),
        "prepare_sandbox": PrepareSandboxAgent(llm=llm, artifacts=artifacts, step_index=8, step_total=15),
        "run_codex_exec": RunCodexExecAgent(llm=llm, artifacts=artifacts, step_index=9, step_total=15),
        "collect_codex_outputs": CollectCodexOutputsAgent(llm=llm, artifacts=artifacts, step_index=10, step_total=15),
        "prepare_sandbox_newstyle": PrepareSandboxNewstyleAgent(
            llm=llm, artifacts=artifacts, step_index=8, step_total=15
        ),
        "run_codex_exec_newstyle": RunCodexExecNewstyleAgent(
            llm=llm, artifacts=artifacts, step_index=9, step_total=15
        ),
        "collect_codex_outputs_newstyle": CollectCodexOutputsNewstyleAgent(
            llm=llm, artifacts=artifacts, step_index=10, step_total=15
        ),
        "observe_metrics": ObserveMetricsAgent(llm=llm, artifacts=artifacts, step_index=12, step_total=15),
        "align_evidence": AlignEvidenceAgent(llm=llm, artifacts=artifacts, step_index=13, step_total=15),
        "verify_claims": VerifyClaimsAgent(llm=llm, artifacts=artifacts, step_index=14, step_total=15),
        "audit_report": AuditReportAgent(llm=llm, artifacts=artifacts, step_index=15, step_total=15),
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
