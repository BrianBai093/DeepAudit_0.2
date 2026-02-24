from __future__ import annotations

import argparse
import json
from pathlib import Path

from p2c.graph import build_agents, run_phase_1, run_phase_2, run_phase_3
from p2c.io_artifacts import ArtifactManager
from p2c.llm.client import LLMClient
from p2c.runtime.factory import close_runtime
from p2c.schemas import VerdictDoc
from p2c.utils.console import format_log


def log_global(artifacts: ArtifactManager, state: str, step: str, message: str) -> None:
    line = format_log("orchestrator", state, step, message)
    print(line, flush=True)
    artifacts.append_text("execution/run.log", line + "\n")


def ensure_phase_prereq(phase: int, artifacts: ArtifactManager) -> None:
    if phase == 2:
        task_spec = artifacts.path("task/task_spec.json")
        if not task_spec.exists() or task_spec.stat().st_size == 0:
            raise RuntimeError(
                "Phase 2 requires phase 1 artifacts. Run: python -m p2c.main --phase 1 ..."
            )
        try:
            payload = json.loads(task_spec.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Phase 2 requires a valid phase 1 task_spec.json. "
                "Run: python -m p2c.main --phase 1 ..."
            ) from e
        if not payload.get("entrypoints"):
            raise RuntimeError(
                "Phase 2 requires phase 1 outputs with non-empty entrypoints. "
                "Run: python -m p2c.main --phase 1 ..."
            )
    if phase == 3:
        run_manifest = artifacts.path("execution/codex_outputs/run_manifest.json")
        claim_alignment = artifacts.path("execution/codex_outputs/claim_alignment.json")
        if not run_manifest.exists() or run_manifest.stat().st_size == 0:
            raise RuntimeError(
                "Phase 3 requires phase 2 artifacts. Run: python -m p2c.main --phase 2 ..."
            )
        try:
            manifest_payload = json.loads(run_manifest.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Phase 3 requires a valid execution/codex_outputs/run_manifest.json from phase 2. "
                "Run: python -m p2c.main --phase 2 ..."
            ) from e
        if not manifest_payload.get("runs"):
            raise RuntimeError(
                "Phase 3 requires phase 2 run_manifest runs. Run: python -m p2c.main --phase 2 ..."
            )
        if not claim_alignment.exists() or claim_alignment.stat().st_size == 0:
            raise RuntimeError(
                "Phase 3 requires phase 2 claim alignment output. Run: python -m p2c.main --phase 2 ..."
            )


def write_inconclusive_verdict(artifacts: ArtifactManager, reason: str) -> None:
    verdict = VerdictDoc(
        status="INCONCLUSIVE",
        claim_verdicts=[],
        reason_codes=[reason],
        summary=f"Pipeline finished before full verification: {reason}",
    )
    artifacts.write_json("results/verdict.json", verdict.model_dump())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper2Code MVP runner")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--paper_md", required=True)
    parser.add_argument("--paper_md_out", required=True)
    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--artifacts_dir", default="./artifacts")
    parser.add_argument("--budget_minutes", type=int, default=60)
    parser.add_argument("--max_self_heal_iters", type=int, default=6)
    return parser.parse_args()


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
            try:
                run_phase_2(ctx, agents)
            finally:
                close_runtime(ctx, artifacts)
            write_inconclusive_verdict(artifacts, "PHASE_2_ONLY")

        elif args.phase == 3:
            run_phase_3(ctx, agents)

        log_global(artifacts, "DONE", "3/3", "pipeline completed")
    except Exception as e:  # noqa: BLE001
        log_global(artifacts, "ERROR", "3/3", str(e))
        write_inconclusive_verdict(artifacts, f"PIPELINE_ERROR:{e}")
        raise

    artifacts.write_text("execution/context.json", json.dumps(ctx, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
