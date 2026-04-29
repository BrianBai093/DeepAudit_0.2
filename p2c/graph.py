from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from p2c.agents.phase1.build_claims_ir import BuildClaimsIRAgent
from p2c.agents.phase1.compile_task_spec import CompileTaskSpecAgent
from p2c.agents.phase1.enrich_claims_visual import EnrichClaimsVisualAgent
from p2c.agents.phase1.extract_fingerprint_atomic import ExtractFingerprintAtomicAgent
from p2c.agents.phase1.extract_fingerprint_filter import ExtractFingerprintFilterAgent
from p2c.agents.phase1.extract_fingerprint_guide import ExtractFingerprintGuideAgent
from p2c.agents.phase1.extract_visual_elements import ExtractVisualElementsAgent
from p2c.agents.phase1.ingest_paper import IngestPaperAgent
from p2c.agents.phase1.repo_analysis import RepoAnalysisAgent
from p2c.agents.phase2.executor_agent import ExecutorAgent
from p2c.agents.phase2.orchestrator import Phase2Orchestrator
from p2c.agents.phase2.tool_agent import ToolAgent
from p2c.agents.phase3.align_evidence import AlignEvidenceAgent
from p2c.agents.phase3.audit_report import AuditReportAgent
from p2c.agents.phase3.execution_summary_evidence import ExecutionSummaryEvidenceAgent
from p2c.agents.phase3.observe_metrics import ObserveMetricsAgent
from p2c.agents.phase3.reproduce_figures import ReproduceFiguresAgent
from p2c.agents.phase3.score_and_diagnose import ScoreAndDiagnoseAgent
from p2c.agents.phase3.verify_claims import VerifyClaimsAgent
from p2c.agents.phase3.visual_to_repo_alignment import VisualToRepoAlignmentAgent

_STEP_TOTAL = 20


def run_phase_1(ctx: dict[str, Any], agents: dict[str, Any]) -> None:
    agents["ingest_paper"].run(ctx)

    # Visual extraction from PDF (skipped if no --paper_pdf)
    if ctx.get("paper_pdf"):
        agents["extract_visual_elements"].run(ctx)

    agents["extract_fingerprint_guide"].run(ctx)
    agents["extract_fingerprint_atomic"].run(ctx)

    # Enrich claims with visual data (skipped if no PDF)
    if ctx.get("paper_pdf"):
        agents["enrich_claims_visual"].run(ctx)

    agents["extract_fingerprint_filter"].run(ctx)
    agents["repo_analysis"].run(ctx)

    # Build RAG code index (graceful degradation on failure)
    try:
        from p2c.rag.builder import build_code_index
        _log = logging.getLogger(__name__)
        _log.info("RAG: building code index for %s", ctx["repo_dir"])
        ctx["_code_index"] = build_code_index(
            Path(ctx["repo_dir"]),
            agents["repo_analysis"].artifacts,
        )
        if ctx["_code_index"] is None:
            _log.info("RAG: index not built (small repo or embedding unavailable)")
        else:
            _log.info("RAG: index built successfully")
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).warning("RAG index build failed: %s", exc)
        ctx["_code_index"] = None

    agents["build_claims_ir"].run(ctx)
    agents["compile_task_spec"].run(ctx)


def run_phase_2(ctx: dict[str, Any], agents: dict[str, Any]) -> None:
    agents["phase2_orchestrator"].run(ctx)


def run_phase_3(ctx: dict[str, Any], agents: dict[str, Any]) -> None:
    agents["execution_summary_evidence"].run(ctx)
    agents["observe_metrics"].run(ctx)
    agents["align_evidence"].run(ctx)
    agents["verify_claims"].run(ctx)
    agents["score_and_diagnose"].run(ctx)
    agents["visual_to_repo_alignment"].run(ctx)
    agents["reproduce_figures"].run(ctx)
    agents["audit_report"].run(ctx)


def build_agents(llm, artifacts) -> dict[str, Any]:
    # Phase 1
    phase1 = {
        "ingest_paper": IngestPaperAgent(llm=llm, artifacts=artifacts, step_index=1, step_total=_STEP_TOTAL),
        "extract_visual_elements": ExtractVisualElementsAgent(
            llm=llm, artifacts=artifacts, step_index=2, step_total=_STEP_TOTAL,
        ),
        "extract_fingerprint_guide": ExtractFingerprintGuideAgent(
            llm=llm, artifacts=artifacts, step_index=3, step_total=_STEP_TOTAL,
        ),
        "extract_fingerprint_atomic": ExtractFingerprintAtomicAgent(
            llm=llm, artifacts=artifacts, step_index=4, step_total=_STEP_TOTAL,
        ),
        "enrich_claims_visual": EnrichClaimsVisualAgent(
            llm=llm, artifacts=artifacts, step_index=5, step_total=_STEP_TOTAL,
        ),
        "extract_fingerprint_filter": ExtractFingerprintFilterAgent(
            llm=llm, artifacts=artifacts, step_index=6, step_total=_STEP_TOTAL,
        ),
        "build_claims_ir": BuildClaimsIRAgent(llm=llm, artifacts=artifacts, step_index=7, step_total=_STEP_TOTAL),
        "repo_analysis": RepoAnalysisAgent(llm=llm, artifacts=artifacts, step_index=8, step_total=_STEP_TOTAL),
        "compile_task_spec": CompileTaskSpecAgent(llm=llm, artifacts=artifacts, step_index=9, step_total=_STEP_TOTAL),
    }

    # Phase 2 — environment setup + autonomous executor
    tool_agent = ToolAgent(llm=llm, artifacts=artifacts, step_index=10, step_total=_STEP_TOTAL)
    executor_agent = ExecutorAgent(llm=llm, artifacts=artifacts, step_index=11, step_total=_STEP_TOTAL)
    orchestrator = Phase2Orchestrator(
        tool_agent=tool_agent,
        executor_agent=executor_agent,
        llm=llm,
        artifacts=artifacts,
        step_index=12,
        step_total=_STEP_TOTAL,
    )
    phase2 = {
        "tool_agent": tool_agent,
        "executor_agent": executor_agent,
        "phase2_orchestrator": orchestrator,
    }

    # Phase 3
    phase3 = {
        "execution_summary_evidence": ExecutionSummaryEvidenceAgent(
            llm=llm, artifacts=artifacts, step_index=13, step_total=_STEP_TOTAL,
        ),
        "observe_metrics": ObserveMetricsAgent(llm=llm, artifacts=artifacts, step_index=14, step_total=_STEP_TOTAL),
        "align_evidence": AlignEvidenceAgent(llm=llm, artifacts=artifacts, step_index=15, step_total=_STEP_TOTAL),
        "verify_claims": VerifyClaimsAgent(llm=llm, artifacts=artifacts, step_index=16, step_total=_STEP_TOTAL),
        "score_and_diagnose": ScoreAndDiagnoseAgent(llm=llm, artifacts=artifacts, step_index=17, step_total=_STEP_TOTAL),
        "visual_to_repo_alignment": VisualToRepoAlignmentAgent(
            llm=llm, artifacts=artifacts, step_index=18, step_total=_STEP_TOTAL,
        ),
        "reproduce_figures": ReproduceFiguresAgent(llm=llm, artifacts=artifacts, step_index=19, step_total=_STEP_TOTAL),
        "audit_report": AuditReportAgent(llm=llm, artifacts=artifacts, step_index=20, step_total=_STEP_TOTAL),
    }

    return {**phase1, **phase2, **phase3}


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
