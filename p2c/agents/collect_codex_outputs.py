from __future__ import annotations

import json
from pathlib import Path

from p2c.agents.base import BaseAgent
from p2c.runtime.factory import ensure_runtime
from p2c.schemas import ClaimAlignmentDoc, RepoState, RunManifestDoc

SYSTEM_PROMPT = "You collect and validate Codex output artifacts."
USER_PROMPT_TEMPLATE = "Input: workspace outputs directory. Output: execution/codex_outputs/*"


class CollectCodexOutputsAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(name="collect_codex_outputs", *args, **kwargs)

    def execute(self, ctx: dict) -> dict:
        self.safe_chat_text(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE)
        runtime = ensure_runtime(ctx, self.artifacts)
        if (getattr(runtime, "backend_name", "") or "").lower() != "e2b":
            raise RuntimeError("collect_codex_outputs requires P2C_RUNTIME_BACKEND=e2b")

        required_ctx = ["workspace_root", "workspace_outputs_dir", "workspace_repo_dir"]
        missing = [k for k in required_ctx if not ctx.get(k)]
        if missing:
            raise RuntimeError(f"collect_codex_outputs missing workspace context keys: {missing}")

        outputs_dir = str(ctx["workspace_outputs_dir"])

        files = {
            f"{outputs_dir}/run_manifest.json": "execution/codex_outputs/run_manifest.json",
            f"{outputs_dir}/claim_alignment.json": "execution/codex_outputs/claim_alignment.json",
            f"{outputs_dir}/codex_worklog.jsonl": "execution/codex_outputs/codex_worklog.jsonl",
            f"{outputs_dir}/patches.diff": "execution/codex_outputs/patches.diff",
            f"{outputs_dir}/codex_exec.log": "execution/codex_outputs/codex_exec.log",
        }
        optional_files = {
            f"{outputs_dir}/dependency_solver.json": "execution/codex_outputs/dependency_solver.json",
            f"{outputs_dir}/pip_install.log": "execution/codex_outputs/pip_install.log",
            f"{outputs_dir}/codex_failure.json": "execution/codex_outputs/codex_failure.json",
            f"{outputs_dir}/codex_main.log": "execution/codex_outputs/codex_main.log",
            f"{outputs_dir}/codex_repair.log": "execution/codex_outputs/codex_repair.log",
        }

        reason_codes: list[str] = []
        for remote, rel in files.items():
            try:
                runtime.download_file(remote, self.artifacts.path(rel))
            except Exception as e:  # noqa: BLE001
                if remote.endswith("/patches.diff"):
                    self.artifacts.write_text(rel, "")
                    reason_codes.append("PATCH_DIFF_UNAVAILABLE")
                    continue
                raise RuntimeError(f"failed to download {remote}: {e}") from e
        for remote, rel in optional_files.items():
            try:
                runtime.download_file(remote, self.artifacts.path(rel))
            except Exception:  # noqa: BLE001
                continue

        run_manifest_raw = self.artifacts.path("execution/codex_outputs/run_manifest.json").read_text(
            encoding="utf-8", errors="ignore"
        )
        claim_alignment_raw = self.artifacts.path("execution/codex_outputs/claim_alignment.json").read_text(
            encoding="utf-8", errors="ignore"
        )

        try:
            run_manifest = RunManifestDoc(**json.loads(run_manifest_raw))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"invalid run_manifest.json schema: {e}") from e

        try:
            claim_alignment = ClaimAlignmentDoc(**json.loads(claim_alignment_raw))
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"invalid claim_alignment.json schema: {e}") from e

        reason_codes.append("NO_GIT_METADATA")

        diff_text = self.artifacts.path("execution/codex_outputs/patches.diff").read_text(encoding="utf-8", errors="ignore")
        repo_state = RepoState(
            head=None,
            branch=None,
            diff_summary=diff_text[:1200],
            submodules=[],
            reason_codes=reason_codes,
        )
        self.artifacts.write_json("execution/repo_state.json", repo_state.model_dump())

        # Mirror validated payloads (normalized) back to disk.
        self.artifacts.write_json("execution/codex_outputs/run_manifest.json", run_manifest.model_dump())
        self.artifacts.write_json("execution/codex_outputs/claim_alignment.json", claim_alignment.model_dump())

        return {
            "codex_outputs": {
                "run_manifest_runs": len(run_manifest.runs),
                "aligned_claims": len(claim_alignment.claims),
                "reason_codes": reason_codes,
            }
        }
