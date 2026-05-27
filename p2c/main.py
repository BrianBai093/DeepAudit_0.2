from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from p2c.graph import build_agents, run_phase_1, run_phase_2, run_phase_3
from p2c.io_artifacts import ArtifactManager
from p2c.llm.client import LLMClient
from p2c.schemas import VerdictDoc
from p2c.utils.console import format_log
from p2c.utils.mineru_client import default_paper_md_from_pdf


def log_global(artifacts: ArtifactManager, state: str, step: str, message: str) -> None:
    line = format_log("orchestrator", state, step, message)
    print(line, flush=True)
    artifacts.append_text("execution/run.log", line + "\n")


def ensure_phase_prereq(phase: int, artifacts: ArtifactManager) -> None:
    if phase == 2:
        claims_ir = artifacts.path("fingerprint/claims_ir.json")
        if not claims_ir.exists() or claims_ir.stat().st_size == 0:
            raise RuntimeError(
                "Phase 2 requires phase 1 artifacts. Run: python -m p2c.main --phase 1 ..."
            )
        try:
            payload = json.loads(claims_ir.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Phase 2 requires a valid phase 1 claims_ir.json. "
                "Run: python -m p2c.main --phase 1 ..."
            ) from e
        experiments = payload.get("experiments") if isinstance(payload.get("experiments"), list) else []
        if not experiments:
            raise RuntimeError(
                "Phase 2 requires phase 1 outputs with non-empty experiments. "
                "Run: python -m p2c.main --phase 1 ..."
            )
    if phase == 3:
        phase2_package = artifacts.path("execution/executor_outputs/phase2_execution_package.json")
        if phase2_package.exists() and phase2_package.stat().st_size > 0:
            try:
                package_payload = json.loads(phase2_package.read_text(encoding="utf-8"))
                if package_payload.get("experiments"):
                    return
            except Exception:
                pass
        run_manifest = artifacts.path("execution/executor_outputs/run_manifest.json")
        executor_results = artifacts.path("execution/executor_outputs/executor_results.json")
        if not run_manifest.exists() or run_manifest.stat().st_size == 0:
            raise RuntimeError(
                "Phase 3 requires phase 2 artifacts. Run: python -m p2c.main --phase 2 ..."
            )
        manifest_has_runs = False
        try:
            manifest_payload = json.loads(run_manifest.read_text(encoding="utf-8"))
            manifest_has_runs = bool(manifest_payload.get("runs"))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Phase 3 requires a valid execution/executor_outputs/run_manifest.json from phase 2. "
                "Run: python -m p2c.main --phase 2 ..."
            ) from e
        if manifest_has_runs:
            return
        executor_results_has_runs = False
        if executor_results.exists() and executor_results.stat().st_size > 0:
            try:
                executor_payload = json.loads(executor_results.read_text(encoding="utf-8"))
                executor_results_has_runs = bool(executor_payload.get("runs"))
            except Exception:
                executor_results_has_runs = False
        if not executor_results_has_runs:
            raise RuntimeError(
                "Phase 3 requires phase 2 run_manifest runs or executor_results runs. "
                "Run: python -m p2c.main --phase 2 ..."
            )


def write_inconclusive_verdict(artifacts: ArtifactManager, reason: str) -> None:
    verdict = VerdictDoc(
        status="INCONCLUSIVE",
        claim_verdicts=[],
        reason_codes=[reason],
        summary=f"Pipeline finished before full verification: {reason}",
    )
    artifacts.write_json("results/verdict.json", verdict.model_dump())


def serializable_context(ctx: dict) -> dict:
    """Return the public, JSON-serializable context fields for context.json."""
    return {k: v for k, v in ctx.items() if not str(k).startswith("_")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper2Code MVP runner")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--paper_md", default=None)
    parser.add_argument("--paper_md_out", required=True)
    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--artifacts_dir", default="./artifacts")
    parser.add_argument("--budget_minutes", type=int, default=30)
    parser.add_argument("--max_self_heal_iters", type=int, default=6)
    parser.add_argument("--paper_pdf", default=None, help="Paper PDF for visual extraction (optional)")
    parser.add_argument(
        "--phase2_force_env_repair",
        action="store_true",
        help="For phase 2, skip native conda env creation and enter the env repair branch directly.",
    )
    args = parser.parse_args()
    if not args.paper_md:
        if args.phase == 1 and args.paper_pdf:
            args.paper_md = str(default_paper_md_from_pdf(args.paper_pdf))
        else:
            parser.error("--paper_md is required unless phase 1 is given --paper_pdf")
    return args


def main() -> None:
    args = parse_args()
    artifacts = ArtifactManager(args.artifacts_dir, args.run_id)
    artifacts.ensure_tree()

    ctx = {
        "phase": args.phase,
        "paper_md": str(Path(args.paper_md)),
        "paper_md_out": str(Path(args.paper_md_out)),
        "repo_dir": str(Path(args.repo_dir)),
        "run_id": args.run_id,
        "artifacts_dir": str(Path(args.artifacts_dir)),
        "budget_minutes": args.budget_minutes,
        "max_self_heal_iters": args.max_self_heal_iters,
        "paper_pdf": str(Path(args.paper_pdf)) if args.paper_pdf else None,
        "phase2_force_env_repair": bool(args.phase2_force_env_repair),
    }

    log_global(artifacts, "START", "0/3", f"phase={args.phase} run_id={args.run_id}")
    llm = LLMClient()
    agents = build_agents(llm=llm, artifacts=artifacts)

    try:
        ensure_phase_prereq(args.phase, artifacts)

        if args.phase == 1:
            log_global(artifacts, "PROGRESS", "1/3", "running phase 1 ingestion + fingerprint extraction")
            run_phase_1(ctx, agents)
            write_inconclusive_verdict(artifacts, "PHASE_1_ONLY")

        elif args.phase == 2:
            log_global(artifacts, "PROGRESS", "2/3", "running phase 2 local execution")
            run_phase_2(ctx, agents)
            write_inconclusive_verdict(artifacts, "PHASE_2_ONLY")

        elif args.phase == 3:
            run_phase_3(ctx, agents)

        log_global(artifacts, "DONE", "3/3", "pipeline completed")
    except Exception as e:  # noqa: BLE001
        log_global(artifacts, "ERROR", "3/3", str(e))
        write_inconclusive_verdict(artifacts, f"PIPELINE_ERROR:{e}")
        raise

    # Strip internal runtime objects before serializing context.
    artifacts.write_text("execution/context.json", json.dumps(serializable_context(ctx), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
